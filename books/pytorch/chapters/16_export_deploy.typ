#import "../template.typ": *

= 导出与部署

`torch.compile` 解决的是"在 Python 里跑得更快"。部署常常有另一个要求：*把模型变成一个不依赖 Python 的产物*——因为推理服务是 C++ 写的、因为要跑在没有 Python 的设备上、因为几十秒的 JIT 冷启动放在请求路径上不可接受。这是 `torch.export` 和 AOTInductor 的地盘。

面试里这块的高频题就两类：三代方案的关系与取舍，以及 `trace` 和 `script` 的区别。

== 三代方案

#ladder(
  ("TorchScript (2019)", "jit.trace 记录执行 / jit.script 解析源码", "维护模式，不再加新特性"),
  ("torch.compile (2023)", "Dynamo + Inductor，JIT，可 graph break", "仍需要 Python runtime"),
  ("torch.export + AOTInductor", "AOT 严格导出成图，再编成 .so", "无 Python 依赖，但要求整图"),
)

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left),
  [*方案*], [*时机*], [*graph break*], [*产物 / 用途*],
  [`torch.jit.trace`], [AOT], [不存在（直接丢控制流）], [`ScriptModule`，历史遗留],
  [`torch.jit.script`], [AOT], [不允许（语法不支持就失败）], [`ScriptModule`，历史遗留],
  [`torch.compile`], [JIT], [允许，自动 fallback], [进程内的编译产物，训练与推理加速],
  [`torch.export`], [AOT], [*不允许*，遇到就报错], [`ExportedProgram`（图 + 签名 + 权重）],
  [AOTInductor], [AOT], [不允许], [`.so` / `.pt2` 包，C++ runtime 加载],
  [ONNX], [AOT], [不允许], [`.onnx`，交给 ORT / TensorRT 等],
)

#insight[
  一句话总结取舍：`torch.compile` 是 JIT 且宽容（有 fallback，所以能上任何模型，但要 Python）；`torch.export` 是 AOT 且严格（必须整图，所以要改代码，但换来一个可序列化、可脱离 Python 的产物）。两者共享 Dynamo 和 AOTAutograd 的前半段，所以*把模型改到 `fullgraph=True` 能过，就是把它改到能 export 的九成工作量*。
]

== jit.trace vs jit.script

这题几乎必问，因为它检验你是否理解"记录一次执行"和"理解程序"的区别。

```python
def g(x):
    if x.sum() > 0:
        return x * 2
    return x * 3

t = torch.jit.trace(g, torch.ones(3))
print(t.code)
```

实跑输出——`if` 整个消失了：

```text
def g(x: Tensor) -> Tensor:
  return torch.mul(x, CONSTANTS.c0)
```

于是换一条分支的输入结果就是错的，而且*不抛任何异常*：

```text
trace(ones)  -> [2.0, 2.0, 2.0]     eager: [2.0, 2.0, 2.0]    ✓
trace(-ones) -> [-2.0, -2.0, -2.0]  eager: [-3.0, -3.0, -3.0] ✗
```

`torch.jit.script(g)` 解析源码，保留了 `if`，`script(-ones)` 实跑得到 `[-3.0, -3.0, -3.0]`，正确。

#warn[
  `jit.trace` 会固化的东西不止 `if`：`for` 的循环次数被展开成固定值（变长序列直接错）、`.item()` 的结果变成常量、`tensor.shape[0]` 变成一个整数字面量、`if self.training` 被定死在 trace 时的模式。它只在 trace 时对 shape 变化和不同分支发出 `TracerWarning`——而这个 warning 在真实项目的日志里几乎注定被淹掉。这是 PyTorch 历史上制造过最多线上事故的 API 之一。
]

`jit.script` 正确但采纳成本高：只支持 Python 的一个静态类型子集，第三方库基本不可用，`Optional` 要显式标注。这是它没能普及的原因，也是 Dynamo 设计时刻意避开的坑（第 12 章）。

== torch.export

```python
torch.export.export(mod, args, kwargs=None, *, dynamic_shapes=None,
                    strict=False, preserve_module_call_signature=(),
                    prefer_deferred_runtime_asserts_over_guards=False)
```

产出的 `ExportedProgram` 有三个组成部分：*图*、*graph signature*、*state dict*。

```python
import torch, torch.nn as nn
from torch.export import export, Dim

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return torch.relu(self.fc(x))

m = M().eval()
B = Dim("B", min=1, max=1024)
ep = export(m, (torch.randn(2, 8),), dynamic_shapes={"x": {0: B}})
print(ep.graph_signature)
print(ep.graph_module.code)
print(list(ep.state_dict.keys()))
print(ep.module()(torch.randn(5, 8)).shape)     # 用别的 batch 跑
```

实跑输出：

```text
# inputs
p_fc_weight: PARAMETER target='fc.weight'
p_fc_bias:   PARAMETER target='fc.bias'
x:           USER_INPUT

# outputs
relu: USER_OUTPUT

def forward(self, p_fc_weight, p_fc_bias, x):
    linear = torch.ops.aten.linear.default(x, p_fc_weight, p_fc_bias)
    relu = torch.ops.aten.relu.default(linear)
    return (relu,)

['fc.weight', 'fc.bias']
torch.Size([5, 4])
```

三件事：

+ *参数被提升成图的输入。* graph signature 记录了"第 1 个输入其实是 `fc.weight` 这个 parameter"。图本身是纯函数，权重通过 `state_dict` 单独携带——这是能序列化、能换权重、能量化的前提。
+ *op 层级比给 Inductor 的图高。* 这里是 `aten.linear.default`，不是 `mm + add`。想彻底分解调 `ep.run_decompositions()`，实跑变成 `permute + addmm + relu`。
+ *`ep.module()` 能直接当普通 module 跑*，方便做数值对比。

序列化用 `torch.export.save` / `load`（实测上面这个模型存成 `.pt2` 是 8356 字节）：

```python
torch.export.save(ep, "/tmp/ep.pt2")
ep2 = torch.export.load("/tmp/ep.pt2")
```

=== 用 `dynamic_shapes` 声明动态维

用 `Dim` 对象声明哪一维动态。三种写法：

```python
B = Dim("B", min=1, max=1024)      # 具名，给范围；同名 Dim 表示"这几处必须相等"
Dim.AUTO                            # 让 export 自己推断这一维是否动态
Dim.STATIC                          # 显式声明静态
dynamic_shapes={"x": {0: B}, "mask": {0: B}}   # 两个输入的第 0 维绑定成同一个符号
```

导出后的约束可以直接看出来：

```python
print(ep.range_constraints)     # {s77: VR[2, int_oo]}
```

`VR[2, int_oo]` 的下界是 *2* 而不是 1——这就是第 15 章讲的 0/1 specialization 在 export 里的体现：符号维度隐含 `>= 2`，要支持 batch=1 得单独导一份。

=== 必须整图

export 不允许 graph break。碰到数据相关的控制流就直接报错。实跑 `if x.sum() > 0:`：

```text
GuardOnDataDependentSymNode: Could not guard on data-dependent expression
Eq(u0, 1) (unhinted: Eq(u0, 1)).  (Size-like symbols: none)

consider using data-dependent friendly APIs such as guard_or_false,
guard_or_true and statically_known_true.
...
The following call raised this error:
  File "/tmp/v11_export.py", line 27, in forward
    if x.sum() > 0:
```

`u0` 是 unbacked symbol——一个"运行时才知道值"的符号。export 拿它没法决定走哪条分支，于是失败。修法：把数据相关控制流改写成张量运算（`torch.where`）、或用 `torch.cond` 这类显式的控制流算子、或把这段逻辑挪到模型外面。

#note[
  torch 2.10 里 `strict` 的默认值是 `False`。非严格模式直接在 Python runtime 下 trace，仍然校验 shape 安全这类关键假设，但不验证全部隐式假设；`strict=True` 走 TorchDynamo，保证图的 soundness，代价是 Python 特性覆盖更窄、更容易报错。文档明确说*两者产出的 IR 规格相同*，序列化方式也一样。调试导出失败时可以用 `torch.export.draft_export` 先拿到一份带诊断信息的结果。
]

== AOTInductor

把 `ExportedProgram` 编成一个自包含的包，用 C++ runtime 加载，完全不需要 Python：

```python
ep = torch.export.export(m, (x,), dynamic_shapes={"x": {0: Dim("B", max=64)}})
path = torch._inductor.aoti_compile_and_package(ep, package_path="/tmp/m.pt2")
runner = torch._inductor.aoti_load_package(path)     # Python 侧也能加载，方便验证
runner(torch.randn(7, 64, device="cuda"))            # batch 7，动态维生效
```

实测这个小模型的 `.pt2` 包是 434824 字节，`aoti_load_package` 加载后用 batch=7 调用正常。C++ 侧用 `torch::inductor::AOTIModelPackageLoader` 加载同一个包。

适用场景：

- *C++ 推理服务。* 已有的 C++ 服务框架，不想引入 Python 解释器和 GIL。
- *冷启动敏感。* 编译在离线做完，上线加载 `.so` 是毫秒级，而 `torch.compile` 的首次调用是几十秒。
- *交付一个二进制。* 不想把模型代码和一堆 Python 依赖发给客户。

#warn[
  AOTInductor 的产物是*绑定 GPU 架构和 torch 版本*的。为 A100（sm80）编的包不能拿去 H100（sm90）跑，torch 小版本升级也可能不兼容。CI 里要按"torch 版本 × GPU 架构"的矩阵产出多份包，并且把这两个维度写进产物的文件名。
]

== ONNX

`torch.onnx.export` 在较新版本里已经切到基于 `torch.export` 的新路径，`dynamo` 参数在 torch 2.10 上*默认就是 `True`*：

```python
torch.onnx.export(model, (x,), "m.onnx", dynamo=True,
                  dynamic_shapes={"x": {0: Dim("B", max=128)}},
                  opset_version=18, optimize=True)
```

新旧两条路径的区别：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*路径*], [*行为*],
  [旧 tracer（`dynamo=False`）],
    [基于 `jit.trace`，继承了它丢控制流的全部问题；动态轴用 `dynamic_axes` 声明（只是给轴起名，不做符号推理）],
  [新路径（`dynamo=True`，默认）],
    [先 `torch.export` 拿到 `ExportedProgram` 再转 ONNX；动态维用 `dynamic_shapes` + `Dim`，有真正的符号 shape 约束],
)

常见导出失败原因：

+ *op 在目标 opset 里没有对应实现。* 报"unsupported operator"。对策：升 `opset_version`、换等价写法、或写 custom translation。
+ *data-dependent shape。* `nonzero`、`x[mask]`、`unique` 的输出 shape 依赖数据，ONNX 的 shape 推理表达不了。
+ *data-dependent 控制流。* 和 export 同一个问题，先在 PyTorch 侧改掉。
+ *动态轴声明不全。* 导出时用一个具体 batch，忘了声明动态，结果 ONNX 图里 batch 被写死成常量。这类问题不报错，只在换 batch 时才暴露。

#note[
  `dynamo=True` 这条路径依赖 `onnxscript` 包。这套环境里没装，实跑直接得到 `ModuleNotFoundError: No module named 'onnxscript'`——所以本节的 ONNX 代码没有在本机跑通，签名是从 `inspect.signature(torch.onnx.export)` 实测读出来的，行为描述来自文档。部署链路里 ONNX 通常只是中转格式，真正跑的是 ORT / TensorRT，它们各自还有一层图优化和 op 支持范围。
]

== 推理侧的常规动作

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*动作*], [*作用*],
  [`model.eval()`],
    [切 dropout 到恒等、BN 用 running stats。忘了这句是最经典的"推理结果抖动"故障],
  [`with torch.inference_mode():`],
    [比 `no_grad` 更强：不建 autograd 图，还关掉 version counter 和 view 追踪，省一点开销。代价是产出的张量不能再参与 autograd],
  [fuse conv + bn],
    [推理时 BN 是仿射变换，可以吸进前面卷积的权重和 bias，省一个 kernel 和一次往返。`torch.ao.nn.intrinsic` 和 `torch.fx.experimental.optimization.fuse` 有现成实现；`torch.compile` 也会做类似的常量折叠（`freezing` 配置）],
  [KV cache],
    [自回归解码里避免重算历史 token 的 K/V。这是 LLM 推理的第一优化，见第 26 章],
  [量化],
    [PTQ（训练后量化，校准几百条数据就能上）与 QAT（训练中插伪量化，精度更好但要重训）。现在的推荐入口是 `torchao`，老的 `torch.quantization` / `torch.ao.quantization` eager 模式路径已不推荐],
  [batching],
    [静态 batching 攒够就发；continuous batching 让每个请求独立进出，GPU 利用率高得多。这是推理引擎（vLLM / SGLang）的职责，不是 PyTorch 层要解决的问题],
)

#insight[
  面试里问"你怎么部署 LLM"，正确的答案层次是：单模型的图优化归 `torch.compile` / `torch.export`；调度、KV cache 管理、continuous batching、prefix caching 归推理引擎（vLLM / SGLang）。自己拿 `torch.export` 从零搭一个 LLM 服务是重复造轮子，说得出这个边界比说得出 API 更重要。
]

== `state_dict` vs 整模型保存

```python
torch.save(model.state_dict(), "sd.pt")     # 推荐：只存张量
torch.save(model, "whole.pt")               # 不推荐：pickle 整个对象
```

`torch.save(model)` 用 pickle 序列化整个对象图，文件里只存了*类的引用路径*（`mypkg.models.MyModel`），不存类的定义。后果：

- 加载时必须能 import 到同名同路径的类。重构包结构、改类名、换代码分支，checkpoint 就打不开了。
- pickle 能执行任意代码，加载不可信文件等于执行不可信代码。

`torch.load` 在 torch 2.6 起把 `weights_only` 的默认行为改成了只允许张量和少数安全类型。实测直接 `torch.load("whole.pt")` 现在会失败：

```text
UnpicklingError: Weights only load failed. This file can still be loaded, to do so
you have two options, do those steps only if you trust the source of the checkpoint.
  (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument
      in `torch.load` from `False` to `True`. Re-running `torch.load` with
      `weights_only=False` ...
```

#warn[
  看到这个报错的正确反应*不是*马上加 `weights_only=False`。先问：这个文件是谁产的？如果是自己训练产出的 `state_dict`，它本来就该能以 `weights_only=True` 加载，报错说明存的是整模型或者混了自定义对象——改存 `state_dict`。只有确认来源可信、且真的需要反序列化自定义类时才关掉，或者用 `torch.serialization.add_safe_globals` 把需要的类逐个加白名单。
]

一个完整的 checkpoint 应该是可自解释的 `dict`，全部由张量和基础类型组成：

```python
torch.save({
    "model": model.state_dict(),
    "optim": opt.state_dict(),
    "step": step,
    "config": asdict(cfg),      # dataclass → dict，不要存 dataclass 对象本身
}, path)
```

分布式场景下的 sharded checkpoint（`torch.distributed.checkpoint`）见第 22 章。

== 面试考点

#interview[
  *Q1*：`torch.jit.trace` 和 `torch.jit.script` 的区别？

  A：`trace` 跑一遍记录 op 序列，控制流被固化——`if` 只保留 trace 时走的那条分支、`for` 的次数被展开成常量、`.item()` 和 `shape[0]` 变成字面量。实测 `if x.sum() > 0` 的函数 trace 出来只剩 `torch.mul(x, c0)`，喂负输入结果错且不报错。`script` 解析源码保留控制流，但只支持 TorchScript 的静态类型子集，第三方库基本不可用，采纳成本高。两者都已进入维护模式。
]

#interview[
  *Q2*：`torch.export` 和 `torch.compile` 的区别？

  A：`compile` 是 JIT 且宽容——运行时编译，遇到不能 trace 的就 graph break 回 Python，产物在进程内、依赖 Python runtime。`export` 是 AOT 且严格——不允许 graph break，遇到数据相关控制流直接报 `GuardOnDataDependentSymNode`，产出可序列化的 `ExportedProgram`（图 + graph signature + state dict），能脱离 Python。两者共享 Dynamo/AOTAutograd 的前半段，所以把模型改到 `fullgraph=True` 能过基本就等于改到能 export。
]

#interview[
  *Q3*：`ExportedProgram` 里有什么？

  A：三部分。图（ATen 层级的 FX graph，默认保留 `aten.linear.default` 这样较高层的 op，要彻底分解得调 `run_decompositions()`）；graph signature（记录每个图输入是 user input 还是 parameter/buffer，以及对应的 `state_dict` key——参数被提升成图的输入，图本身是纯函数）；state dict（权重）。用 `torch.export.save` / `load` 序列化成 `.pt2`。
]

#interview[
  *Q4*：`torch.export` 怎么声明动态 shape？

  A：`dynamic_shapes` 参数配 `Dim` 对象，比如 `{"x": {0: Dim("B", min=1, max=1024)}}`。同一个 `Dim` 用在多处表示这几维必须相等。也可以用 `Dim.AUTO` 让 export 自己推断、`Dim.STATIC` 显式声明静态。导出后 `ep.range_constraints` 能看到实际约束，实测是 `{s77: VR[2, int_oo]}`——下界是 2 不是 1，因为 0/1 specialization 让符号维度隐含 `>= 2`，要支持 batch=1 得单独导一份。
]

#interview[
  *Q5*：AOTInductor 是什么？什么时候用？

  A：把 `ExportedProgram` 编译成自包含的 `.so` / `.pt2` 包，用 C++ runtime（`AOTIModelPackageLoader`）加载，完全不需要 Python。API 是 `torch._inductor.aoti_compile_and_package(ep, package_path=...)` 和 `aoti_load_package`。用在三种场景：已有 C++ 推理服务不想引入 Python 和 GIL；冷启动敏感（编译离线做完，加载是毫秒级，而 `torch.compile` 首次调用几十秒）；要交付一个二进制而不是一堆 Python 依赖。注意产物绑定 GPU 架构和 torch 版本，sm80 编的包不能拿去 sm90 跑。
]

#interview[
  *Q6*：ONNX 导出常见失败原因？新旧路径的区别？

  A：失败原因：目标 opset 里没有对应 op；data-dependent shape（`nonzero`、`x[mask]`）ONNX 的 shape 推理表达不了；data-dependent 控制流；动态轴声明不全导致 batch 被写死（这类不报错，换 batch 才暴露）。路径上，旧 tracer 基于 `jit.trace`，继承丢控制流的问题，动态轴用 `dynamic_axes` 只是给轴起名；新路径（`torch.onnx.export(dynamo=True)`，torch 2.10 已是默认）先走 `torch.export` 再转 ONNX，动态维用 `dynamic_shapes` + `Dim`，有真正的符号 shape 约束。
]

#interview[
  *Q7*：`torch.save(model.state_dict())` 和 `torch.save(model)` 有什么区别？

  A：`state_dict` 只存张量，与代码解耦。`torch.save(model)` 用 pickle 序列化整个对象图，但只存类的引用路径不存定义——加载时必须能 import 到同名同路径的类，重构包结构或改类名 checkpoint 就废了；而且 pickle 能执行任意代码，加载不可信文件等于执行不可信代码。torch 2.6 起 `torch.load` 默认只允许张量和少数安全类型，加载整模型会直接抛 `UnpicklingError`。正确做法是存一个全由张量和基础类型组成的 dict（model / optim / step / config）。
]

#interview[
  *Q8*：推理侧除了图优化还会做什么？

  A：`model.eval()`（切 dropout 和 BN）、`inference_mode()`（比 `no_grad` 更强，连 version counter 和 view 追踪都关掉）、fuse conv+bn（BN 在推理时是仿射变换，可以吸进卷积权重）、KV cache（自回归解码避免重算历史 K/V，第 26 章）、量化（PTQ 校准几百条就能上，QAT 精度更好但要重训，现在的入口是 `torchao`）。调度层面的 continuous batching、prefix caching 归推理引擎（vLLM / SGLang），不是 PyTorch 这一层的事——说清这个边界比背 API 重要。
]
