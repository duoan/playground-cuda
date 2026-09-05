#import "../template.typ": *

= TorchDynamo：从 Python 字节码到 FX Graph

面试里问 `torch.compile` 通常只有两个方向：*它是怎么把 Python 变成 kernel 的*，和*它为什么在我的模型上不快*。两个问题的答案都从 Dynamo 开始——它是唯一直接面对你写的 Python 代码的那一层，也是 graph break 和 recompile 这两个最常见性能杀手的发源地。这一章讲 Dynamo 的机制、guard 与 graph break 的成因与观测方法；AOTAutograd 见第 13 章，Inductor 见第 14 章。

本章所有输出都在 torch 2.10.0+cu128 / A100-SXM4-80GB 上实跑得到，做了裁剪但没有改写。

== 四层流水线

`torch.compile(model)` 不是一个编译器，是四层组件串起来的一条流水线。

#figure(
  align(center, flow-boxes(
    boxes: ("Python bytecode", "FX Graph + guards", "joint fwd/bwd graph", "Triton / C++ kernel"),
    box-w: 3.0,
  )),
  caption: [`torch.compile` 的四层：Dynamo 抓图，AOTAutograd 补反向，Inductor 生成 kernel。],
)

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*层*], [*输入 → 输出*], [*核心职责*],
  [TorchDynamo],
    [CPython frame（字节码）→ FX Graph + guards + 残余字节码],
    [符号执行字节码，抓出 tensor 计算，其余留在 Python],
  [AOTAutograd],
    [FX Graph（只有前向）→ 前向图 + 反向图],
    [提前 trace 出反向、functionalize、decompose、切分],
  [PrimTorch],
    [2000+ 个 ATen op → 一小组 prim op],
    [收窄编译器要支持的 op 集合],
  [TorchInductor],
    [切分后的 FX Graph → Triton (GPU) / C++ OpenMP (CPU)],
    [lowering、fusion、autotune、codegen、缓存],
)

只有 Dynamo 这一层是"必须理解 Python 语义"的。后面三层拿到的都已经是纯粹的张量计算图，不再有 Python 对象、控制流和副作用。这个分工决定了后面所有的坑都长什么样。

== Dynamo 怎么接管 Python 执行

CPython 3.6 引入的 PEP 523 允许把解释器求值一个 code object 的入口函数替换掉。Dynamo 就是注册了这样一个自定义 frame evaluation 回调：每当 Python 要执行一个被 `torch.compile` 包住的函数的 frame，控制权先交给 Dynamo。

Dynamo 拿到 frame 后做*符号执行*（symbolic evaluation）：逐条解释字节码，但栈上放的不是真实值，而是 `VariableTracker` 这类符号对象。

- 遇到 `TensorVariable` 上的 torch 操作 → 记一个节点到 FX Graph，返回一个新的 `TensorVariable`（带 fake 的 shape/dtype/device 元信息，不算真实数据）。
- 遇到纯 Python 逻辑（`for` 展开、常量折叠、属性访问、`nn.Module` 的 `__call__` 派发）→ 直接在符号执行里*消化掉*，不进图。这就是为什么编译后的图里看不到 `for layer in self.layers` 这个循环，只看到被展开的一串 op。
- 遇到看不懂的东西 → graph break（下面细讲）。

符号执行结束后，Dynamo 产出三样东西：一张 FX Graph、一组 guard、一段*重写过的字节码*。重写后的字节码把原来那段计算替换成"调用编译好的 callable"，其余部分照原样保留。

```python
import torch

def f(x, w):
    return (x @ w).relu().sum()

gm = []
torch.compile(f, backend=lambda g, ex: (gm.append(g), g.forward)[1])(
    torch.randn(8, 8), torch.randn(8, 8))
print(gm[0].code)
```

实跑输出（Dynamo 抓到的图，注意这时还是 `torch` 层的 method call，不是 ATen）：

```text
def forward(self, L_x_ : torch.Tensor, L_w_ : torch.Tensor):
    l_x_ = L_x_
    l_w_ = L_w_
    matmul = l_x_ @ l_w_;  l_x_ = l_w_ = None
    relu = matmul.relu();  matmul = None
    sum_1 = relu.sum();  relu = None
    return (sum_1,)
```

#note[
  自定义 backend 就是一个 `(gm, example_inputs) -> callable` 的函数。这是最省事的"我到底抓到了什么图"调试手段，也是 `backend="eager"` / `"aot_eager"` 这些调试后端的本质：把图原样跑掉，不做 codegen，用来判断问题出在 Dynamo 还是 Inductor。
]

== guard：编译产物是特化的

Dynamo 抓的图是在*一组具体假设*下才正确的。这些假设被记录成 guard，挂在编译产物上；每次调用先跑一遍 guard 检查，全通过才复用编译结果。

guard 覆盖的东西比大多数人以为的多：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*guard 类型*], [*记录了什么*],
  [`TENSOR_MATCH`], [输入 tensor 的 dtype / device / 维数 / 各维 size / `requires_grad` / layout],
  [`EQUALS_MATCH`, `CONSTANT_MATCH`], [被当成常量参与 trace 的 Python 标量的具体值],
  [`TYPE_MATCH`, `ID_MATCH`], [Python 对象的类型，或某个 `nn.Module` 实例的 `id`],
  [`GRAD_MODE`], [当前是否在 `no_grad` / `inference_mode` 下],
  [`DEFAULT_DEVICE`, `DETERMINISTIC_ALGORITHMS`], [全局状态],
  [`SHAPE_ENV`], [动态 shape 的符号约束（`s0 >= 2` 这类）],
)

`torch._dynamo.explain(f)(args)` 会把 guard 全部打印出来。下面是实跑里一条 guard 的原样输出：

```text
  Guard 29:
    Name: "L['b']"
    Source: local
    Create Function: TENSOR_MATCH
    Guard Types: ['TENSOR_MATCH']
    Code List: ["hasattr(L['b'], '_dynamo_dynamic_indices') == False"]
```

guard 失败就 recompile。用 `TORCH_LOGS="recompiles"` 能直接看到原因，实跑输出：

```text
[0/1] [__recompiles] Recompiling function f in /tmp/v3_logs.py:3
[0/1] [__recompiles]     triggered by the following guard failure(s):
[0/1] [__recompiles]     - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 5
```

`[0/1]` 这个编号要会读：斜杠前是 *frame 序号*（同一个函数第几段图），斜杠后是 *该段图的第几份编译产物*。看到 `[0/7]` 就说明第 0 段图已经编了 8 份。

=== recompile 上限与静默 fallback

同一段代码的编译产物数量有上限。torch 2.6 起这个配置的正式名字是 `torch._dynamo.config.recompile_limit`（旧名 `cache_size_limit` 仍是别名），默认 *8*；另有 `accumulated_recompile_limit`，默认 256，管一个进程内的总量。

超限后 Dynamo *不报错*，只打一条 warning，然后把这个函数标记成"以后别再编了"，永久 fallback 到 eager：

```text
W convert_frame.py:1676] [0/8] torch._dynamo hit config.recompile_limit (8)
W convert_frame.py:1676] [0/8]    function: 'f' (/tmp/v4_cachelimit.py:3)
W convert_frame.py:1676] [0/8]    last reason: 0/7: tensor 'x' size mismatch
                                  at index 0. expected 11, actual 12
W convert_frame.py:1676] [0/8] To log all recompilation reasons, use
                               TORCH_LOGS="recompiles"
```

#warn[
  这是生产里最阴的一类"compile 没效果"：日志里只有一行 warning，训练照样跑，速度悄悄退回 eager。上线前务必确认 stderr 里没有 `hit config.recompile_limit`。正确修法不是把 limit 调大（那只是把编译时间浪费掉），而是找到 recompile 的源头——通常是变长 shape，见第 15 章的动态 shape 一节。
]

#insight[
  guard 是 `torch.compile` 能"既激进特化又保证正确"的全部原因。TorchScript 没有 guard，所以它必须在编译期把所有情况都表达出来（于是要求你写受限的静态类型代码）；Dynamo 选择"先假设，再运行时校验"，代价是 recompile，收益是你完全不用改代码。
]

== graph break：图被切开

Dynamo 符号执行时遇到无法表达成图的东西，就地把图切断：把已经攒到的 op 编译成一段图，然后回到 Python 逐字节码执行那段不认识的代码，再从下一条指令开始新的一段图。结果是 `[编译段] → [Python 段] → [编译段]` 交替。

常见成因，按遇到频率排：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*成因*], [*为什么*],
  [`print` / `logging` 调用], [有副作用且返回值不是张量，无法进图],
  [`.item()` / `.tolist()` / `float(t)`], [要求真实数据；trace 时只有 fake tensor，没有值],
  [`.cpu()` / `.numpy()`], [同上，跨设备取值意味着同步],
  [依赖张量值的 `if` / `while`], [分支走哪边取决于运行时数据，图里表达不了],
  [不支持的第三方库调用（`numpy` 之外的 C 扩展等）], [Dynamo 没有对应的 `VariableTracker`],
  [`try` / `except` 里抛出并被捕获的路径], [异常控制流的支持有限],
  [部分生成器 / 闭包 / `**kwargs` 的动态用法], [符号执行无法静态解析],
  [数据相关的 shape（`x[mask]`、`nonzero`）], [输出 shape 依赖数据],
)

#note[
  `.item()` 与数据相关 shape 这两类不是绝对不行：`torch._dynamo.config.capture_scalar_outputs = True` 和 `capture_dynamic_output_shape_ops = True` 可以让它们进图，代价是引入 unbacked symbol，下游要么加 `torch._check` 约束，要么在 Inductor 里退化。默认关掉是因为它们经常把编译搞得更糟。
]

=== 三种观测手段

*一、`torch._dynamo.explain`* —— 一次拿到图数量、break 数量、每段的 op、以及每个 break 的原因。它返回 `ExplainOutput` 对象，字段是 `graph_count` / `graph_break_count` / `op_count` / `break_reasons` / `ops_per_graph` / `out_guards` / `graphs` / `compile_times`。

```python
import torch
import torch._dynamo as dynamo

def f(x, y):
    a = torch.sin(x) + y
    print("hello")            # graph break 1
    b = a * 2
    if b.sum().item() > 0:    # graph break 2
        b = b + 1
    return b.relu()

exp = dynamo.explain(f)(torch.randn(8, 8), torch.randn(8, 8))
print(exp.graph_count, exp.graph_break_count, exp.op_count)
```

实跑输出（裁掉了 guard 部分，那是几十行）：

```text
Graph Count: 3
Graph Break Count: 2
Op Count: 4
Break Reasons:
  Break Reason 1:
    Reason: Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print` with
               argument types ['str'] (has_kwargs False)
  Hint: Avoid calling builtin `print` with argument types ['str']. ...
  Hint: If you are attempting to call a logging function (e.g. `print`), you can
        try adding it to torch._dynamo.config.reorderable_logging_functions.
    User Stack:
      <FrameSummary file /tmp/v1_explain.py, line 7 in f>
  Break Reason 2:
    Reason: Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with
               config.capture_scalar_outputs=False.
Ops per Graph:
  Ops 1:
    <built-in method sin of type object at 0x747b04929b40>
    <built-in function add>
  Ops 2:
    <built-in function mul>
  Ops 3:
    <built-in function add>
```

两个 break 把代码切成 3 段图，一共只有 4 个 op——每段图都小到 Inductor 几乎没得优化。

*二、`TORCH_LOGS="graph_breaks"`* —— 训练跑起来之后被动观测，不用改代码。它按发生顺序打，还带 user code traceback，能直接定位到行。和 `recompiles` 一起开最有用：

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```

*三、`fullgraph=True`* —— 开发期首选。它把第一个 graph break 变成异常抛出，逼你当场处理：

```python
torch.compile(f, fullgraph=True)(torch.randn(4))
```

实跑得到的异常（类型是 `torch._dynamo.exc.Unsupported`）：

```text
Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print` with
               argument types ['str'] (has_kwargs False)
  ...
from user code:
   File "/tmp/v2_fullgraph.py", line 5, in f
    print("hi")

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace ...
```

#insight[
  工作流应该是：开发时 `fullgraph=True` 把图打通，确认能整图编译；上生产再决定要不要放开成 `fullgraph=False`（换取对偶发不可 trace 代码的容忍）。反过来先上 `fullgraph=False` 的人，通常到最后都不知道自己的模型被切成了几十段。
]

=== graph break 的代价

+ *fusion 机会消失。* Inductor 只能在一段图内融合。切开之后每段图边界上的中间张量必须落到 HBM，第 14 章会算这笔账。
+ *每段之间要回 Python。* 除了字节码执行本身的开销，进出编译产物还要跑一遍 guard 检查。
+ *CUDA graph 用不了。* `mode="reduce-overhead"` 需要一段连续的、地址固定的 kernel 序列；中间插了 Python 就断了。
+ *编译产物数量翻倍。* 每段图独立编译、独立 guard、独立计入 `recompile_limit`。上面那个例子里 `torch_dynamo_resume_in_f_at_6` 就是 break 之后那段图的名字，它有自己的 `[1/0]`、`[1/1]` 编号。

不是所有 break 都要消灭。在 `forward` 之外、每个 step 只发生一次的 break（比如 dataloader 逻辑、metric 打印）代价可以忽略。要消灭的是*在热点 forward 内部、每层都发生一次*的那种。

== 手动控制编译边界

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*API*], [*用途*],
  [`@torch.compiler.disable`],
    [标记"这个函数别 trace"。Dynamo 直接在此 break 并原样调用它，比让它 trace 失败干净],
  [`torch.compiler.allow_in_graph(fn)`],
    [反过来：让 Dynamo *不要*看函数内部，把它当成一个不透明的 op 塞进图。前提是它输入输出都是张量且无副作用],
  [`torch._dynamo.mark_dynamic(t, dim)`],
    [声明 `t` 的第 `dim` 维是动态的，别为它生成 size guard。见第 15 章],
  [`torch.compiler.nested_compile_region`],
    [标记一个可复用的子区域，让重复结构（比如 N 个同构 layer）少编几次],
  [`torch.compiler.set_stance("...")`],
    [运行时切换编译策略，比如 `"force_eager"`、`"eager_on_recompile"`。用来快速做 A/B 对比],
)

实测 `@torch.compiler.disable` 的效果：把带 `print` 和 `.item()` 的日志函数标上之后，`explain` 报 `graph_count 2, breaks 1`——一次干净的 break，而不是函数内部每个不支持的操作各来一次。

`allow_in_graph` 的效果同样可验证，被标记的函数在图里变成一个单节点：

```text
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sin = l_x_.sin();  l_x_ = None
    custom = __main___custom(sin);  sin = None
    return (custom,)
```

#warn[
  `allow_in_graph` 是"我保证"级别的 API。如果被标记的函数里有副作用、有 in-place 修改、或者对同样的输入返回不同结果，编译后的行为会静默地和 eager 不一致——因为 AOTAutograd 之后的每一层都相信这个节点是纯函数。自定义 CUDA op 的正确做法是注册成 custom op（`torch.library`），不是 `allow_in_graph`。
]

== 为什么 Dynamo 成了，TorchScript 没成

这是高频对比题。答案不是"Dynamo 更快"，而是*采纳成本*。

#ladder(
  ("torch.jit.trace", "跑一遍记录 op 序列", "控制流被固化成一条路径，静默出错"),
  ("torch.jit.script", "解析源码到 TorchScript 子集", "要求你改代码，第三方库全部不可用"),
  ("torch.compile", "字节码符号执行 + guard + graph break", "不改代码，能 fallback，可能 recompile"),
)

三条关键差别：

+ *不要求改代码。* `jit.script` 只支持 Python 的一个静态类型子集：不能用 `**kwargs`、不能用大多数第三方库、`Optional` 要显式标注、类要写 `@torch.jit.export`。一个真实模型迁过去经常要重写几百行。Dynamo 直接吃你现在的代码。
+ *能优雅降级。* `jit.script` 遇到不支持的语法就是编译失败，你要么改代码要么放弃。Dynamo 遇到就 graph break：不支持的部分回 Python 跑，其余照样编译。这让"部分收益"成为可能，而部分收益是能推广的前提。
+ *不需要静态类型推断。* Dynamo 在符号执行时手上有*真实的第一批输入*，shape、dtype、device 全都是已知的具体值，特化后由 guard 兜住。`jit.script` 必须在没有输入的情况下静态推断类型，这在 Python 里是打不赢的仗。

`torch.jit.trace` 的坑另说：它记录的是一次执行的 op 序列，条件分支和循环次数被固化，换一条路径的输入结果就是错的，而且*不报错*。第 16 章给了实测复现。

#note[
  TorchScript 目前处于维护模式（不再加新特性）。面试里问到"你们模型怎么部署"，答 `jit.script` 会被追问为什么不用 `torch.export`；反过来能说清"trace 丢控制流、script 采纳成本高、export 是现在的正确答案"就是加分项。
]

== 面试考点

#interview[
  *Q1*：`torch.compile` 内部有哪几层？各层负责什么？

  A：四层。Dynamo 用 PEP 523 接管 CPython frame，符号执行字节码，产出 FX Graph + guard + 重写的字节码；AOTAutograd 提前 trace 出反向，做 functionalization 和 decomposition，切成前向图和反向图；PrimTorch 把 2000+ ATen op 收窄成一小组 prim op；Inductor 把图 lowering 成 loop-level IR，做 fusion 和 codegen，GPU 出 Triton、CPU 出 C++/OpenMP。
]

#interview[
  *Q2*：什么是 guard？为什么必须有？

  A：guard 是编译时假设的运行时校验条件，覆盖输入 tensor 的 dtype/device/维数/各维 size/`requires_grad`、被折成常量的 Python 值、`nn.Module` 的 `id`、`grad_mode` 等全局状态。因为 Dynamo 抓的图是对第一批输入特化过的，guard 是"特化"与"正确"之间的唯一保险。guard 检查失败就 recompile。
]

#interview[
  *Q3*：什么是 graph break？列 4 个常见成因。

  A：Dynamo 遇到无法表达成图的代码，就把图切断成"编译段 - Python 段 - 编译段"。常见成因：`print`/`logging` 这类有副作用的调用；`.item()`/`.tolist()`/`.cpu()` 这类需要真实数据的操作；依赖张量值的 `if`/`while`；Dynamo 不认识的第三方库调用。数据相关 shape（`x[mask]`、`nonzero`）和部分 `try/except` 也会。
]

#interview[
  *Q4*：graph break 有什么代价？为什么不能只看"反正还能跑"？

  A：四条。fusion 只能发生在一段图内，切开后段边界的中间张量必须写回 HBM；每段之间要回 Python 执行字节码并重跑 guard；CUDA graph（`mode="reduce-overhead"`）需要连续 kernel 序列，插了 Python 就用不了；每段图独立编译独立 guard，各自计入 `recompile_limit`。极端情况下切成几十段，编译时间全付了、收益接近零。
]

#interview[
  *Q5*：怎么定位 graph break？

  A：开发期用 `fullgraph=True`，第一个 break 直接抛 `Unsupported` 异常并指到源码行；要看全部 break 用 `torch._dynamo.explain(fn)(args)`，它给 `graph_count` / `graph_break_count` / `break_reasons` 和每段的 op；线上被动观测用 `TORCH_LOGS="graph_breaks,recompiles"`，带 user traceback 且不用改代码。
]

#interview[
  *Q6*：模型每个 step 的 shape 都不一样，会发生什么？

  A：每种新 shape 触发一次 guard 失败 → recompile。Dynamo 有 automatic dynamic shapes：同一维度出现第二个值时会自动把它符号化，所以通常第 3 种 shape 起就不再重编。但如果 shape 变化模式复杂（多维同时变、或 `dynamic=False` 强制静态），编译份数会撞到 `recompile_limit`（默认 8），之后 Dynamo 打一条 warning 就永久 fallback 到 eager——训练照跑，加速全没了。修法是 `mark_dynamic` 标注动态维，或把输入 pad 到固定的几个桶。
]

#interview[
  *Q7*：为什么 `torch.compile` 比 `torch.jit.script` 成功？

  A：采纳成本。`jit.script` 要求代码落在 TorchScript 的静态类型子集里，第三方库基本不可用，迁移经常要重写；遇到不支持就是编译失败，没有中间状态。Dynamo 不要求改代码，遇到不认识的就 graph break 回 Python，收益可以是部分的；而且它在符号执行时手上有真实输入，shape/dtype 都是已知值，靠 guard 兜住特化，不需要静态类型推断。
]

#interview[
  *Q8*：`torch.compiler.disable` 和 `torch.compiler.allow_in_graph` 分别什么时候用？

  A：`disable` 是"别 trace 这段"，用在日志、metric、调试代码上，让 Dynamo 干净地 break 一次而不是在函数内部反复失败。`allow_in_graph` 是反向：让 Dynamo 不看函数内部，把它当不透明 op 塞进图，用在 Dynamo 追不进去但语义是纯函数的调用上。后者危险——下游所有层都假设该节点无副作用，有 in-place 或随机性就会和 eager 静默不一致；自定义 kernel 的正确做法是用 `torch.library` 注册 custom op。
]
