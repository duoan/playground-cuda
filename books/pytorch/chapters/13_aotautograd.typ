#import "../template.typ": *

= AOTAutograd：提前把反向图也编译出来

Dynamo 交出来的图只有前向。如果流水线到这里就结束，反向仍然由 eager 的 autograd engine 一个 op 一个 kernel 地跑——而训练里反向占约 2/3 的计算量，收益直接砍掉一大半。AOTAutograd 这一层的任务就是*提前*（ahead-of-time）把前向和反向一起 trace 成一张 joint graph，做完 functionalization 和 decomposition，再切成前向图和反向图交给 Inductor。

这一层还顺手解释了两件生产里绕不开的事：为什么 compile 之后显存占用会变、以及为什么 `torch.compile` 和 activation checkpointing 会打架。

== 先算清楚"只编前向"值多少

以最基本的 `y = x @ W`（`x` 是 `(M, K)`，`W` 是 `(K, N)`）为例：

#formula[$ "fwd": y = x W arrow.r 2 M N K "FLOPs" $]

反向要算两个梯度，各是一次同规模 GEMM：

#formula[$ "bwd": d x = d y W^T, quad d W = x^T d y arrow.r 4 M N K "FLOPs" $]

#insight[
  反向的计算量是前向的 2 倍，占训练总量的 2/3。只编译前向，理论上限就是 1/3 的加速空间；而 elementwise/norm 的反向同样是 memory-bound，正是 fusion 收益最大的地方。这就是"为什么反向也要编译"的标准答案。
]

== 怎么在 dispatcher 层 trace 出反向

AOTAutograd 不解析 Python，它站在 PyTorch 的 dispatcher 上工作（dispatcher 见第 7 章）：

+ 用 `FakeTensor` 按 Dynamo 给的输入元信息造出一批"只有 shape/dtype/device 没有数据"的输入。FakeTensor 走真实的 meta kernel 做 shape 推导，所以不用分配一字节 HBM 就能算出每一步的 shape。
+ 用 `__torch_dispatch__` 挂在 dispatcher 上，把每个落到 dispatcher 的 ATen op 记录成 FX 节点。关键是它挂的位置在 *Autograd key 之下*：autograd 已经把反向展开成了具体的 ATen 调用，所以记录下来的图里反向是*显式的 op*，不是一个黑盒的 `backward()`。
+ 前向 trace 完之后，用一个 dummy 的输出梯度（图里叫 `tangents`）调 `.backward()`，让反向也被同一个 `__torch_dispatch__` 记下来。前向 + 反向记在一张图里，就是 joint graph。

看真实的 joint graph（`TORCH_LOGS="aot_joint_graph"`，`f(x, w) = gelu(x @ w).sum()`，A100 上实跑）：

```text
===== Joint graph 0 =====
class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[8, 8][8, 1]cuda:0"; primals_2: "f32[8, 8][8, 1]cuda:0";
        tangents_1: "f32[][]cuda:0";

        mm: "f32[8, 8]" = torch.ops.aten.mm.default(primals_1, primals_2)
        gelu: "f32[8, 8]" = torch.ops.aten.gelu.default(mm)
        sum_1: "f32[]" = torch.ops.aten.sum.default(gelu);  gelu = None
        expand: "f32[8, 8][0, 0]" = torch.ops.aten.expand.default(tangents_1, [8, 8])
        gelu_backward: "f32[8, 8]" = torch.ops.aten.gelu_backward.default(expand, mm)
        t: "f32[8, 8][1, 8]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
        mm_1: "f32[8, 8]" = torch.ops.aten.mm.default(t, gelu_backward);  t = None
        t_1: "f32[8, 8][1, 8]" = torch.ops.aten.t.default(primals_2)
        mm_2: "f32[8, 8]" = torch.ops.aten.mm.default(gelu_backward, t_1)
        return pytree.tree_unflatten([sum_1, mm_2, mm_1], self._out_spec)
```

读法：`primals_*` 是前向输入（含参数），`tangents_*` 是输出梯度。`gelu_backward`、`t`、两个 `mm` 都是反向——它们在图里和前向的 op 平级，Inductor 完全不知道也不需要知道哪些是反向。`f32[8, 8][1, 8]` 这个标注里第二个方括号是 stride，可以看出 `t` 是零拷贝的 stride 变换。

#note[
  `aten.t` 出现在图里而不是被折进 `mm`，是因为 AOTAutograd 只负责如实记录；把 `t + mm` 变成一次 `mm` 的 transposed 读取是 Inductor 的活（第 14 章里能看到它生成 `reinterpret_tensor` 而不是真的转置拷贝）。
]

== functionalization：把副作用消掉

Dynamo 抓到的图里 in-place 操作原样保留。用一个自定义 backend 打印 Dynamo 图，实跑输出：

```text
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_.clone();  l_x_ = None
    add_ = y.add_(1.0);  add_ = None
    mul_ = y.mul_(2.0);  mul_ = None
    sum_1 = y.sum();  y = None
    return (sum_1,)
```

同一段代码经过 AOTAutograd（`TORCH_LOGS="aot_graphs"`），in-place 全部被改写：

```text
===== Forward graph 0 =====
    def forward(self, primals_1: "f32[4][1]cpu"):
        clone: "f32[4][1]cpu" = torch.ops.aten.clone.default(primals_1)
        add:   "f32[4][1]cpu" = torch.ops.aten.add.Tensor(clone, 1.0);  clone = None
        mul:   "f32[4][1]cpu" = torch.ops.aten.mul.Tensor(add, 2.0);    add = None
        sum_1: "f32[][]cpu"   = torch.ops.aten.sum.default(mul);        mul = None
        return (sum_1,)
```

`add_` → `add`，`mul_` → `mul`。这个过程叫 functionalization：所有 in-place（`add_`、`copy_`、`index_put_`）和 view 上的写入（`x[0] = v`、`x.transpose(0,1).mul_(2)`）都被重写成"读旧值 → 算新值 → 产生新张量"的纯函数形式；如果被修改的张量是图的输入，最后再补一个显式的 `copy_` 把结果写回去。

为什么必须做这一步：

- *下游做的每个变换都假设无副作用。* fusion、重排、CSE（公共子表达式消除）、死代码消除，都建立在"节点之间只有数据依赖"这个前提上。有 in-place 就多了隐式的顺序依赖，而 FX 图里并不表达这种依赖。
- *min-cut partitioner 要重算某些中间结果。* 只有纯函数才能被安全地重算（见下节）。
- *别名（aliasing）分析是不可判定的噩梦。* 与其在每个 pass 里处理"这两个张量是否共享 storage"，不如一次性把别名消掉。

#warn[
  functionalization 不等于"in-place 白写"。它只是让编译器*内部*看到纯函数；如果你在 `forward` 里 in-place 修改了一个外部张量（比如往一个全局 buffer 里写），AOTAutograd 会在图末尾补 `copy_` 保证语义，但这个 `copy_` 是真实的额外 HBM 写。而 in-place 修改 *图的输入且该输入 requires grad* 常常直接触发 graph break 或报错。
]

== decomposition 与 PrimTorch

ATen 有 2000 多个 op（算上 overload 更多）。让每个后端都实现 2000 个 kernel 是不现实的。decomposition 就是一组 Python 写的规则，把复合 op 拆成更基础的 op：

```python
# 概念示意：真实实现在 torch/_decomp/decompositions.py
def gelu_decomp(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
```

第 14 章里贴的真实 Triton kernel 中就能看到 `0.5`、`0.7071067811865476`、`libdevice.erf` 这几个常量——那正是 `aten.gelu` 被分解后的样子。

规模上，torch 2.10 里实测：`torch.ops.prims` 下有 *125* 个 prim op，而 `torch._decomp.core_aten_decompositions()` 返回 *1004* 条分解规则。也就是说一千多个 ATen op 的语义被一百多个 prim 表达出来了。

```python
import torch
print(len([n for n in dir(torch.ops.prims) if not n.startswith("_")]))  # 125
print(len(torch._decomp.core_aten_decompositions()))                     # 1004
```

有了这层收窄，一个新后端只要实现这一百多个 prim 就能跑全部 PyTorch 模型——这也是 PrimTorch 存在的商业理由（硬件厂商接入成本）。

#note[
  decomposition 是可选的、可配置的。`torch.export` 默认保留较高层的 ATen op（导出的图里能看到 `aten.linear.default` 而不是 `mm + add`，见第 16 章），而给 Inductor 的图分解得更彻底。同一个模型在不同下游会看到不同粒度的图，这不是 bug。
]

== min-cut partitioner：显存和重算的取舍

joint graph 要切成前向图和反向图。切在哪里等价于回答一个问题：*哪些中间结果在前向算完后保存下来给反向用，哪些在反向里重算？*

- 保存 → 占显存，从前向一直活到反向。
- 重算 → 省显存，多花计算。

把 joint graph 看成一个有向图，"前向侧"和"反向侧"之间必须传递的张量就是一个割（cut）。割的容量是这些张量的总字节数。要最小化保存的显存，就是求*最小割*——这就是 `min_cut_rematerialization_partition` 的名字来源。

它就是 Inductor 路径上的默认切法：`torch/_inductor/compile_fx.py` 里的 `partition_fn` 在 `config.custom_partitioner_fn is None` 时直接调它。`torch._functorch.partitioners` 里另有一个 `default_partition`，是朴素切法（只保存 autograd 本来就要 `save_for_backward` 的那些），一般只在调试对比时用。

实际的策略比纯最小割精细：便宜的 elementwise op（`add`、`mul`、`gelu`、`sigmoid`）倾向于重算，因为它们 memory-bound，重算的成本远低于多存一份中间张量的代价；GEMM 这类贵的 op 一定保存。上面那张 joint graph 切完之后的前向图返回值就体现了这一点：

```text
===== Forward graph 0 =====
        mm    = aten.mm.default(primals_1, primals_2)
        gelu  = aten.gelu.default(mm)
        sum_1 = aten.sum.default(gelu);  gelu = None
        t     = aten.t.default(primals_1)
        t_1   = aten.t.default(primals_2)
        return (sum_1, mm, t, t_1)
```

它把 `mm` 的结果存了下来（因为 `gelu_backward` 需要它，而重算 `mm` 太贵），但*没有*存 `gelu` 的输出（反向里由 `gelu_backward(expand, mm)` 直接重算）。

=== 实测：显存确实变少了

A100-SXM4-80GB，bf16，`x`、`w` 都是 `(4096, 4096)`，函数体是一次 GEMM 后接 4 轮 `sigmoid(tanh(y) * 1.5)`，做完整的 fwd + bwd，测 `torch.cuda.max_memory_allocated()`：

#figure(
  align(center, hbar-chart(
    (("eager", 368), ("torch.compile", 176)),
    unit: "MiB", width: 7,
  )),
  caption: [峰值显存实测。eager 每个 elementwise 的中间结果都被 autograd 存下来；partitioner 判断它们该重算。],
)

eager 里 8 个 elementwise op 各存一份 `(4096, 4096)` bf16 中间结果（32 MiB 一份）；编译后 partitioner 只保存 GEMM 的输出，其余在反向重算。

#warn[
  "compile 之后显存一定变少"是错的。partitioner *这一层*的目标是"在不慢于 default 策略的前提下不比 eager 用更多 activation 显存"，但整个 `torch.compile` 的峰值显存可能上升，原因在别处：`mode="reduce-overhead"` 的 CUDA graph 要独占一个显存池；`max-autotune` 在 autotune 期间要为每个候选实现分配 workspace；Inductor 的 buffer 复用决策与 caching allocator 的实际行为不总是一致。所以换配置之后必须重测峰值，别靠推理。
]

=== 手动控制取舍

`torch._functorch.config.activation_memory_budget` 是这一层最有用的旋钮，取值 0.0 到 1.0：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*值*], [*含义*],
  [`1.0`（默认）], [纯按运行时最优切，只做那些"既省显存又不明显变慢"的重算],
  [`0.4`], [保存的 activation 压到默认策略的 40%，用 0-1 背包求"为达标所需的最小重算量"],
  [`0.0`], [相当于对整个被编译区域做 activation checkpointing],
)

它比手写 `checkpoint()` 好的地方是*粒度*：`checkpoint()` 是"整个 block 的中间结果全丢，反向从入口重算全部"，budget 是"按每个 op 的实际字节数和实际耗时求解，只丢该丢的"。

== 与 activation checkpointing 的关系

两者做的是同一件事——用重算换显存——所以叠加时要小心。

#warn[
  `torch.utils.checkpoint.checkpoint` 必须用 `use_reentrant=False`（non-reentrant 实现，基于 saved-tensor hooks）才能和 `torch.compile` 正常配合。老的 reentrant 实现会在反向里再进一次 autograd engine，Dynamo 追不进去，典型症状是 graph break 或者编译报错。这个参数在较新版本已经开始强制显式传值。
]

叠加时的实际后果：`checkpoint` 区域内部的 activation 已经被丢弃了，AOTAutograd 的 partitioner 在这个区域里能腾挪的空间就很小——你实际上是用一个粗粒度的手工决策覆盖了一个细粒度的自动决策。所以对*已经能整图编译*的模型，先试 `activation_memory_budget` 调到需要的显存水位，往往比手写 `checkpoint` 更快；`checkpoint` 的价值在于它跨 graph break 也有效、并且显存上限可预测。

超大模型上两者常常必须共存（`checkpoint` 保证显存能装下，compile 负责 fusion）。这时把 `checkpoint` 的粒度放大到整个 transformer block，让 compile 在 block 内部自由发挥。FSDP / 分布式场景下的组合见第 15 章和第四部分。

== 反向不总能提前编出来：compiled autograd

AOTAutograd 的前提是"前向能整图抓下来"。前向一旦有 graph break，反向就被切成对应的若干段，段与段之间仍然由 eager 的 autograd engine 串起来。更麻烦的是有些反向逻辑*根本不在前向的 trace 范围内*：注册在张量上的 backward hook、DDP 的 bucket AllReduce hook、跨编译段的 `AccumulateGrad`——这些是 autograd engine 在运行时才知道的。

`torch._dynamo.config.compiled_autograd = True` 换一个思路：不在前向时提前 trace 反向，而是在 `backward()` *真正开始时*，把 autograd engine 即将执行的整张反向图抓下来，当场编译。代价是每次 backward 前多一次 trace/guard 检查，收益是拿到一张真正完整的反向图（含 hook、含跨段部分）。

按 `torch/_dynamo/config.py` 里的注释，它还会放松前向 trace 的一些限制，比如允许在编译区域内的张量上注册 backward hook。它也是 DDP 的 `optimize_ddp="python_reducer"` 模式的前提——那个模式让 DDP 关掉 C++ reducer 改用 Python reducer，好让 compiled autograd 能把通信 trace 进反向图，从而不靠 graph break 就实现 comm/compute overlap。

#note[
  compiled autograd 目前默认关闭，属于"知道它存在、知道它解决什么问题"的层次。面试里被问"前向有 graph break 时反向怎么办"，能说出"AOTAutograd 只能按段编，完整反向图要靠 compiled autograd 在 backward 时现抓"就足够了。分布式场景下的取舍见第四部分。
]

== 怎么看 AOTAutograd 产出的图

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*开关*], [*看到什么*],
  [`TORCH_LOGS="graph_code"`], [Dynamo 抓到的图（AOTAutograd 之前），Python 代码形式],
  [`TORCH_LOGS="aot_joint_graph"`], [切分*之前*的 joint graph。判断"反向到底有没有被 trace 到"],
  [`TORCH_LOGS="aot_graphs"`], [切分*之后*的前向图和反向图。判断"哪些中间结果被保存了"],
  [`TORCH_LOGS="post_grad_graphs"`], [post-grad pass 之后、真正交给 Inductor lowering 的图],
  [`TORCH_LOGS="graph_sizes"`], [Dynamo 图里每个节点的 shape，动态 shape 调试用],
  [`TORCH_LOGS_OUT=/tmp/log.txt`], [同时把日志写到文件。这些图动辄几千行，必须重定向],
)

配套的调试 backend（都是 `torch.compile(f, backend=...)`）：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*backend*], [*行为*], 
  [`"eager"`], [Dynamo 抓图后原样用 eager 跑。验证"问题是不是 Dynamo 引起的"],
  [`"aot_eager"`], [走完 AOTAutograd（含 functionalization、partition）但不 codegen。定位"是 AOTAutograd 还是 Inductor"],
  [`"inductor"`], [默认，全流程],
)

#insight[
  三分法排查 `torch.compile` 引起的数值错误或崩溃：`backend="eager"` 通过说明 Dynamo 没问题；`backend="aot_eager"` 也通过说明 AOTAutograd 没问题，锅在 Inductor 的 codegen；`aot_eager` 就挂了说明是 functionalization 或 partition 的问题（通常源头是你代码里的 in-place / 别名操作）。这套方法在面试里被问"你怎么 debug compile"时是标准答案。
]

== 面试考点

#interview[
  *Q1*：AOTAutograd 解决什么问题？为什么必须有它？

  A：Dynamo 只抓到前向，反向仍由 eager 的 autograd engine 逐 op 跑。而反向的计算量是前向的 2 倍（`dx = dy W^T` 和 `dW = x^T dy` 各一次同规模 GEMM），占训练总量 2/3。AOTAutograd 提前把前向和反向 trace 成一张 joint graph 再切开，让 Inductor 也能对反向做 fusion。
]

#interview[
  *Q2*：AOTAutograd 是怎么 trace 出反向图的？

  A：用 `FakeTensor` 造出只有元信息没有数据的输入（走 meta kernel 做 shape 推导，不分配显存），用 `__torch_dispatch__` 挂在 dispatcher 上记录每个 ATen op。关键是挂的位置在 Autograd key 之下，autograd 已经把反向展开成具体的 ATen 调用，所以图里的 `mm`、`t`、`gelu_backward` 都是显式节点。前向 trace 完用一个 dummy 的输出梯度调 backward，把反向也记进同一张图。
]

#interview[
  *Q3*：什么是 functionalization？为什么必须做？

  A：把 in-place 和 view 写入重写成纯函数形式（`y.add_(1)` 变成 `add(y, 1)` 产生新张量；如果改的是图输入，末尾补一个显式 `copy_`）。必须做是因为下游的 fusion、重排、CSE、死代码消除都假设节点间只有数据依赖；in-place 引入的隐式顺序依赖在 FX 图里没法表达。另外 min-cut partitioner 要重算中间结果，只有纯函数才能安全重算。
]

#interview[
  *Q4*：decomposition 和 PrimTorch 是干什么的？

  A：ATen 有 2000+ 个 op，让每个后端都实现是不现实的。decomposition 是一组 Python 规则把复合 op 拆成基础 op（`gelu` 拆成 `mul + erf + add`），PrimTorch 定义了目标的那一小组 prim op（两百个上下量级，覆盖 elementwise / reduction / view / 数据搬运）。新后端只实现这一小组就能跑全部模型。注意分解粒度是可配的：`torch.export` 默认保留较高层的 ATen op，给 Inductor 的图分解得更彻底。
]

#interview[
  *Q5*：min-cut partitioner 在决定什么？为什么叫 min-cut？

  A：决定 joint graph 里哪些中间结果在前向保存、哪些在反向重算。把"前向侧到反向侧必须传递的张量集合"看成图的一个割，割的容量是这些张量的总字节数，最小化保存显存就是求最小割。实际策略还带成本模型：便宜的 memory-bound elementwise 倾向重算，GEMM 输出一定保存。实测一个 GEMM + 8 个 elementwise 的 bf16 例子，A100 上峰值显存从 eager 的 368 MiB 降到 176 MiB。
]

#interview[
  *Q6*：`torch.compile` 之后显存一定变少吗？

  A：不一定。partitioner 这一层的目标是不比 eager 用更多 activation 显存，但整体峰值可能上升：`mode="reduce-overhead"` 的 CUDA graph 要独占显存池，`max-autotune` 期间每个候选实现要 workspace，Inductor 的 buffer 复用与 caching allocator 的实际行为也不总一致。换配置必须重测 `max_memory_allocated()`。想主动压显存用 `torch._functorch.config.activation_memory_budget`（1.0 是默认的运行时最优，0.0 相当于对整个编译区域做 checkpointing）。
]

#interview[
  *Q7*：activation checkpointing 和 `torch.compile` 一起用要注意什么？

  A：`checkpoint` 必须传 `use_reentrant=False`；老的 reentrant 实现在反向里会再进一次 autograd engine，Dynamo 追不进去，会 graph break 或直接报错。语义上两者做同一件事（重算换显存），叠加时 `checkpoint` 会用粗粒度的手工决策覆盖 partitioner 的细粒度自动决策，所以能整图编译的模型优先调 `activation_memory_budget`。超大模型两者共存时，把 `checkpoint` 的粒度放到整个 transformer block，让 compile 在 block 内自由发挥。
]
