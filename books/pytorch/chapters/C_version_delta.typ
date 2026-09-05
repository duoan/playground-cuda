#import "../template.typ": *

= 附录 C：PyTorch 2.x 版本差异

"你用的什么版本？2.x 有什么变化？"——这题几乎每场面试都会问，而且是个陷阱：背版本号没用，面试官想听的是*你知不知道 PyTorch 这几年往哪个方向走*。答案只有两条主线：*编译栈*（`torch.compile` 那一整套）和 *DTensor*（分布式的统一底座）。其余都是这两条线上的枝节。

本书的环境是 torch 2.10.0+cu128 / CUDA 12.8。

#warn[
  下面表格里的 minor 版本号，只有两条有硬证据：`torch.compile` 是 2.0 的发布主题，`torch.load` 的 `weights_only` 默认值变更由报错文本自己写明"In PyTorch 2.6, we changed the default value"。*其余各行是按发布节奏归置的，可能差一个小版本。* 面试里报"2.x 中期"这种粒度就够，不要为了显得精确去报一个记不准的号——报错了反而扣分。
]

== 1.x $arrow.r$ 2.0：唯一的分水岭

PyTorch 1.x 时代的图捕获方案是 *TorchScript*（`torch.jit.trace` / `torch.jit.script`）。它的思路是让 Python 子集变成静态语言：`script` 直接解析 Python AST，遇到动态特性就报错；`trace` 只记录一次执行的算子序列，控制流和 shape 全被固化。结果是"能跑通的模型要重写一遍"，用户体验差到大部分人宁可不用。

2.0 换了思路：*不去理解 Python，而是在字节码层面拦截*。Dynamo 从 CPython 的 frame evaluation 钩子进去，把能捕获的部分抽成 FX graph，捕获不了的地方直接 *graph break*、回退 Python 执行。代价是图可能被切碎，好处是*任何 Python 代码都能跑*，只是收益不同。这是 `torch.compile` 成功而 TorchScript 失败的根本原因（第 12 章）。

同一时期 TorchScript 进入维护模式，到本书环境的 2.10，`torch.jit.script` 已经会打出显式弃用警告：

```text
UserWarning: `torch.jit.script` is deprecated.
Please switch to `torch.compile` or `torch.export`.
```

== 逐版本要点

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray, inset: 5pt, align: (left, left, left),
  [*版本*], [*关键特性*], [*对你写代码的影响*],
  [2.0],
  [`torch.compile` 发布（Dynamo + AOTAutograd + Inductor 三层）；`F.scaled_dot_product_attention` 进入 core；TorchScript 转维护模式],
  [训练脚本加一行 `model = torch.compile(model)` 就能拿收益；手写 attention 一律改调 SDPA，它会自动选 FlashAttention / memory-efficient / math 后端],
  [2.1],
  [`torch.export` 预览；DTensor 原型；automatic dynamic shapes（第一次遇到 shape 变化就自动重编成动态版本）],
  [导出路线开始从 TorchScript 迁到 export；变长输入的重编译问题被缓解，但仍需手动 `dynamic=True` 兜底],
  [2.2],
  [SDPA 加入 FlashAttention-2 后端；`DeviceMesh` 进入 beta；functional collectives（可被 compile 追踪的集合通信）],
  [长序列 attention 明显变快且不用自己装 flash-attn 包；分布式代码可以开始用 mesh 描述拓扑，不再手工 `new_group`],
  [2.3],
  [Tensor Parallel 高层 API（`torch.distributed.tensor.parallel` 的 `ColwiseParallel` / `RowwiseParallel`）；用户自定义 Triton kernel 可被 `torch.compile` 追踪],
  [TP 从"照抄 Megatron 的手工切分"变成"给每层挂一个 parallel style"；自己写的 Triton kernel 不再是 graph break],
  [2.4],
  [FSDP2（`fully_shard`，基于 DTensor 重写）；`torch.distributed.pipelining`；TCPStore 换 libuv 后端],
  [FSDP 的 `FlatParameter` 黑盒变成一等公民的 DTensor 参数，能逐参数控制切分、和 TP 组合更干净（第 19、21 章）],
  [2.5],
  [FlexAttention（用 Python 写 score modifier，编译成融合 kernel）；SDPA 的 cuDNN 后端；regional compilation（只编译重复的子模块以省编译时间）],
  [ALiBi / sliding window / soft-cap 这类 attention 变体不用再手写 kernel；大模型的编译时间可以从分钟级压下来],
  [2.6],
  [*`torch.load` 的 `weights_only` 默认值从 `False` 改为 `True`*；`torch.compiler.set_stance`；AOTInductor 打包能力],
  [这是最容易踩的一次 break change，见下一节。`set_stance` 让你能在运行时切换"强制编译 / 强制 eager"，调试和 A/B 很方便],
  [2.7--2.8],
  [Blackwell 与 CUDA 12.8 支持；编译缓存与编译时间的持续优化；量化生态（torchao）成熟],
  [新卡上不用再自己编 torch；分布式训练的编译缓存能跨 rank 复用，冷启动变快],
  [2.9--2.10\ （本书环境）],
  [DTensor 与 FSDP2 的公开路径稳定为 `torch.distributed.tensor` 和 `torch.distributed.fsdp.fully_shard`；`torch.jit.script` 打出显式弃用警告；引入统一的 `fp32_precision` 精度控制 API],
  [新项目直接写公开路径，不要再 import `_composable` / `_tensor` 这类私有模块；精度开关从一堆 `allow_tf32` 布尔量往 `fp32_precision` 字符串收敛],
)

#insight[
  把这张表压成一句话：*2.0--2.2 是编译栈从能用到好用，2.3--2.5 是分布式从手工切分转到 DTensor 抽象，2.6 之后是工程收尾（默认值、缓存、新硬件）。* 面试里这一句比背十个版本号有用。
]

== 那些"默认值变了"的坑

这类最容易踩，因为代码没改、行为变了。

=== `torch.load` 的 `weights_only`

2.6 起默认从 `False` 变成 `True`。反序列化时不再执行任意 pickle 代码，只允许张量和一小撮白名单类型。加载老 checkpoint（里面存了 `argparse.Namespace`、自定义 config 对象、`numpy` 标量）会直接失败：

```text
_pickle.UnpicklingError: Weights only load failed. ... (1) In PyTorch 2.6, we
changed the default value of the `weights_only` argument in `torch.load` from
`False` to `True`. ... WeightsUnpickler error: Unsupported global: GLOBAL
__main__.Foo was not an allowed global by default.
```

处理方式按信任程度分两档：

```python
torch.load(path, weights_only=False)                       # 只对自己产出的文件
torch.serialization.add_safe_globals([MyConfig])           # 白名单指定类型
```

真正的根治是*别往 checkpoint 里塞对象*：`state_dict` 存张量，超参存 JSON。

=== `zero_grad(set_to_none)`

2.0 起默认 `True`（此前默认把梯度填 0）。省一次 memset、省一份显存，代价是行为差异：没参与 forward 的参数梯度是 `None` 而不是 `0`。自己遍历 `p.grad` 做统计的代码要加 `if p.grad is not None`。

=== `checkpoint` 的 `use_reentrant`

`torch.utils.checkpoint.checkpoint` 有两套实现。老的 reentrant 版本靠 `autograd.Function` 重入实现，限制多（不支持 `grad(inputs=)`、输出必须有 requires_grad、不能有多次 backward）；新的非重入版本用 saved-tensor hook，没这些限制。现在不传这个参数会警告：

```text
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be
passed explicitly. Starting in PyTorch 2.9, calling checkpoint without
use_reentrant will raise an exception. use_reentrant=False is recommended ...
```

#note[
  警告文本里说 2.9 起会抛异常，但在 2.10 上实测仍然只是警告——官方的弃用节奏比警告文本慢。不管哪个版本，*一律显式传 `use_reentrant=False`*。
]

=== TF32 开关

TF32 是 Ampere 引入的 19 位格式（8 位指数 + 10 位尾数），fp32 的 matmul 可以走 tensor core，快很多、精度降到约 fp16 的尾数水平。历史上这个默认值反复变过：早期 Ampere 上 matmul 默认开 TF32，后来因为"用户不知情地损失精度"改成默认关；cuDNN 那一路一直是开的。

torch 2.10 上实测的默认值：

```python
torch.backends.cuda.matmul.allow_tf32    # False
torch.backends.cudnn.allow_tf32          # True
```

*两个开关默认值不一样*，这是个爱考的细节。想让 fp32 matmul 提速就得显式打开 `torch.backends.cuda.matmul.allow_tf32 = True`；反过来，追求数值可复现（比如比对两个实现是否等价）就得把 cudnn 那一路也关掉。较新版本引入了统一的 `fp32_precision` 字符串 API，同一台机器上 `torch.backends.cuda.matmul.fp32_precision` 是 `"none"`、`torch.backends.cudnn.conv.fp32_precision` 是 `"tf32"`，和上面两个布尔量是一致的两种表达。详见第 10 章。

=== 怎么自己确认一个默认值

不要凭记忆，也不要问模型，当场查最快：

```python
import inspect, torch
inspect.signature(torch.optim.SGD.zero_grad)       # (self, set_to_none: bool = True)
inspect.signature(torch.load).parameters["weights_only"]
torch.backends.cuda.matmul.allow_tf32              # 布尔量直接打印
```

签名里写死的默认值一目了然；签名是 `None` 的（比如 `torch.load` 的 `weights_only` 和 `checkpoint` 的 `use_reentrant`）说明真实默认值藏在函数体里，得看行为——最省事的办法是故意触发一次，读报错或警告文本，PyTorch 的这类消息通常会把版本和迁移方式一起写清楚。本章的几条硬证据就是这么来的。

== 已废弃 / 不推荐清单

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray, inset: 5pt, align: (left, left, left),
  [*东西*], [*状态*], [*用什么替代*],
  [`nn.DataParallel`],
  [没有运行时警告，但文档明确推荐改用 DDP。单进程多线程，被 GIL 卡住、主卡显存不均],
  [`DistributedDataParallel` + `torchrun`（第 18 章）],
  [`torch.jit.script` / `torch.jit.trace`],
  [2.10 上打出显式弃用警告，TorchScript 已是维护模式],
  [训练加速用 `torch.compile`，导出部署用 `torch.export` + AOTInductor（第 16 章）],
  [`torch.autograd.Variable`],
  [1.0 起就是 `Tensor` 的别名，纯历史遗留],
  [直接用 `Tensor` 和 `requires_grad=`],
  [`.data`],
  [能用但绕过 version counter，会让"原地修改"类 bug 变成静默错误],
  [读值用 `.detach()`，改值用 `with torch.no_grad():`（第 1、6 章）],
  [`torch.cuda.amp.autocast`\ `torch.cuda.amp.GradScaler`],
  [有明确的弃用警告：`torch.cuda.amp.autocast(args...) is deprecated`],
  [`torch.autocast("cuda", dtype=...)` / `torch.amp.GradScaler("cuda")`（第 5 章）],
  [FSDP1（`FullyShardedDataParallel`）],
  [仍可用，但新功能都在 FSDP2 上做],
  [`torch.distributed.fsdp.fully_shard`（第 19 章）],
  [手工 `dist.new_group` 拼多维并行],
  [不算废弃，但容易把 rank 算错],
  [`DeviceMesh` + DTensor placement（第 21 章）],
)

== 生态：各个 `torch*` 包是干什么的

面试官问"你了解 PyTorch 生态吗"，想听的是你能一句话说清每个包的定位：

- *torchao*：量化与稀疏。训练后量化、QAT、int8/fp8 训练、低比特优化器，是官方推的量化路线。
- *torchtitan*：大模型预训练的参考实现。用来看 FSDP2 + TP + PP + CP 怎么正确组合，而不是当框架用。
- *torchtune*：微调库。LoRA / QLoRA / 全参微调的配方式实现。
- *DTensor*：不是一个包，是 `torch.distributed` 里的核心抽象。FSDP2、TP、分布式 checkpoint 现在都建在它上面——*这是趋势本身*，其他几个包只是它的应用。

一句话概括方向：*PyTorch 正在把"并行"从一堆互不相通的 wrapper（DDP / FSDP / Megatron 式 TP）收敛成"在 DeviceMesh 上给张量标 placement"这一个抽象。*

== 面试怎么答版本问题

不要背版本号。三句话的模板：

+ *说环境*："我平时用 2.x，具体到项目上是 2.10 + CUDA 12.8。"
+ *说主线*："2.x 相比 1.x 的关键变化就两条：一条是编译栈，`torch.compile` 把 Dynamo 抓图 + Inductor codegen 这套做成了默认加速路径，TorchScript 退出历史舞台；另一条是分布式，从 DDP/FSDP1 这种整体 wrapper，转向 DeviceMesh + DTensor 这个统一抽象，FSDP2 和 TP 现在都建在上面。"
+ *展开你真用过的*："我实际用得比较深的是 `torch.compile` 这条线，踩过重编译超限静默 fallback 的坑，后来是靠 `TORCH_LOGS=recompiles` 定位到输入 shape 一直变，加 `drop_last` 和 `dynamic=True` 解决的。"

第三句是分水岭。前两句谁都能背，第三句有没有具体的坑和具体的定位手段，面试官一听就知道你是不是真用过。

#warn[
  被追问"这个特性是哪个版本引入的"而你记不准时，说"记不清具体是 2.3 还是 2.4，我印象里是 TP 高层 API 那一批一起进来的"——这比报一个错误的号强得多。报错一个版本号，面试官会开始怀疑你前面说的其他数字。
]
