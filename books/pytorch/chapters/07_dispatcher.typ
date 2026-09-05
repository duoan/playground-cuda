#import "../template.typ": *

= Dispatcher：一次 `a + b` 到底走了什么路

"你了解 PyTorch 的内部实现吗"是中高级岗位的标准问题。绝大多数候选人的回答停在"Python 调 C++，C++ 调 CUDA"，这个答案换不来任何加分。真正的答案是：PyTorch 的核心是一个 *多分派（multiple dispatch）表*，op 是行、DispatchKey 是列，autograd、AMP、量化、`torch.compile` 的 tracing 全都是这张表上的一层，而不是散落在各处的 `if`。把这个讲清楚，面试官会立刻知道你读过源码。

这一章跟着 `a + b` 走完全程。CUDA 层面的 async launch 细节见第 9 章，Inductor 怎么把这些 op 融成一个 kernel 见第 14 章。

== 分层总览

#figure(
  align(center, flow-boxes(
    boxes: ("Python", "torch._C", "Dispatcher", "ATen kernel", "cuBLAS / cuDNN"),
    box-w: 2.5, gap-x: 0.45,
  )),
  caption: [`a + b` 的调用栈。前两层是薄的胶水，Dispatcher 是决策中心，ATen kernel 是真正干活的 C++/CUDA 代码，重型线性代数再转手给厂商库。],
) <fig-torch-layers>

一层层说：

+ *Python 前端*。`a + b` 触发 `Tensor.__add__`，它是 `torch._C._TensorBase` 上的一个 C 函数槽位，不是 Python 代码。`torch.add(a, b)` 走的是同一条路。这一层的开销主要是 Python 解释器本身和参数解析（`PythonArgParser` 把 Python 对象翻译成 C++ 类型），单次约几微秒 —— 小 kernel 场景下这就是"Python overhead"的来源，也是 CUDA Graph 和 `torch.compile` 要消灭的东西。
+ *`torch._C`（pybind + 代码生成）*。`torch/_C/_VariableFunctions.pyi` 里那几千个函数签名是从 `native_functions.yaml` 自动生成的。这一层负责把 Python 参数转成 `at::Tensor`，然后调用 `at::add(a, b)`。
+ *Dispatcher*。查表：根据这个 op 的 schema 和参数的 DispatchKey，决定调哪个 kernel。这是本章的主题。
+ *ATen kernel*。真正的实现。`aten/src/ATen/native/` 下按后端分目录：`native/cpu/`、`native/cuda/`、`native/mkldnn/`。
+ *厂商库*。matmul 转给 cuBLAS/cuBLASLt，卷积转给 cuDNN，CPU 上转给 oneDNN/MKL。PyTorch 自己不写 GEMM。

#insight[
  这五层里，*只有 Dispatcher 是 PyTorch 的独特设计*，其他四层任何框架都有。所以面试问"PyTorch 架构"，答案的重心必须放在 dispatcher 上。
]

== Dispatcher 的核心：op $times$ DispatchKey $arrow.r$ kernel

=== op 与 schema

每个 op 由 `native_functions.yaml` 里的一条 *schema* 定义。schema 是这个 op 的类型签名，包括参数名、类型、默认值、哪些参数会被原地修改（`(a!)` 标记）、返回值。运行时可以直接查：

```python
import torch
print(torch.ops.aten.add.Tensor._schema)
# aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
```

同一个"逻辑 op"可以有多个 *overload*：`add.Tensor`、`add.Scalar`、`add.out`、`add_`（in-place）。它们是 dispatcher 里独立的条目。

=== DispatchKey 与 key set

每个 tensor 携带一个 *DispatchKeySet*（一个位集合），描述"我需要哪些层来处理"。可以直接打出来：

```python
>>> torch._C._dispatch_key_set(torch.randn(3, device="cuda"))
DispatchKeySet(CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA)
```

常见的 key 及其职责：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*DispatchKey*], [*这一层干什么*],
  [`Autocast{CUDA,CPU}`], [AMP：按 op 的白/黑名单把输入 cast 成 `bfloat16` / `float16`，然后 redispatch],
  [`Autograd{CPU,CUDA}`], [建反向图：挂 `grad_fn`、记录 `next_functions`、`save_for_backward`，然后 redispatch],
  [`ADInplaceOrView`], [维护 in-place 与 view 的记账：bump version counter、记录 view 关系],
  [`Functionalize`], [把 in-place / view 改写成纯函数形式，`torch.compile` 与 `export` 的前置],
  [`Python` / `PythonTLSSnapshot`], [把分派交回 Python（`__torch_dispatch__`，FakeTensor / 各种 tensor subclass 靠它）],
  [`BackendSelect`], [factory 函数（`torch.empty` 等）没有输入 tensor，靠 `device` 参数决定后端],
  [`Meta`], [只推 shape/dtype 不算数值，`FakeTensor` 与 `torch.compile` 的 shape 推断],
  [`Sparse{CPU,CUDA}` / `NestedTensor`], [稀疏、变长布局的独立实现],
  [`CPU` / `CUDA` / `MPS` / `XPU`], [真正算数的后端 kernel，分派链的终点],
)

=== 关键机制：按优先级逐层剥离

DispatchKey 有一个 *全局固定的优先级*（在 `c10/core/DispatchKey.h` 里按枚举顺序排）。dispatcher 每次只取 key set 里 *优先级最高的那个 key*，找到对应 kernel 执行；那个 kernel 干完自己的活以后，把这个 key 从 set 里"剥掉"（`c10::impl::ExcludeDispatchKeyGuard`），再 *redispatch* —— 于是控制权落到下一个 key。

所以一次 `a + b`（`a`、`b` 在 CUDA 上、`requires_grad=True`、处在 `autocast` 上下文里）的完整路径是：

```text
at::add(a, b)
 └─ Autocast: 查 add 的策略 → 按 promote 规则统一 dtype，cast 输入
     └─ redispatch（Autocast 已排除）
         └─ Autograd: 记录 AddBackward0，把输入的 grad_fn 连进 next_functions
             └─ redispatch（Autograd 已排除）
                 └─ ADInplaceOrView: add 不是 in-place，这层基本 fallthrough
                     └─ redispatch
                         └─ CUDA: at::native::add → TensorIterator → 启动 elementwise kernel
```

#insight[
  这段路径就是本章最值钱的一句话：*autograd 不是 `add` 实现里的一个 `if (requires_grad)`，而是分派链上的一整层*。同理，AMP 不是在 `nn.Linear` 里写了个 cast，量化不是重写了一遍 op，`FakeTensor` 不是 mock。它们都是 *在同一张分派表上注册的一层 kernel*，彼此正交、可任意组合。这就是为什么 PyTorch 能把 autograd + AMP + 量化 + compile 叠在一起而不互相污染，而这也是面试里能拉开差距的点。
]

#note[
  想亲眼看这张表，用 `torch._C._dispatch_dump("aten::add.Tensor")`，它会把这个 op 上注册的所有 key 和对应 kernel 的源文件位置全部打出来。跑一遍比读十页文档管用。另外 `TORCH_SHOW_DISPATCH_TRACE=1` 环境变量（debug build 才有）能打印运行时的每次 redispatch。
]

=== fallthrough 与 boxed fallback

不是每一层都要为每个 op 写 kernel。两个机制补足：

- *fallthrough*：这一层对这个 op 没事可做，直接透传给下一个 key。`ADInplaceOrView` 对绝大多数非 in-place op 就是 fallthrough，零开销。
- *boxed fallback*：给一整个 key 注册一个"通用兜底 kernel"，它把所有参数打包成 `IValue` 列表统一处理。autograd 的 "not implemented" 报错、`Functionalize` 的通用改写、`Python` key 的回调都用这个。代价是 boxing/unboxing 的开销，好处是几千个 op 只写一份代码。

== `TensorImpl` / `Storage` / `TensorOptions`

分派看的是 tensor 上的元数据，所以要知道 tensor 到底存了什么。

- *`Storage`*：一块连续的字节 + 一个 `Allocator`。它不知道 shape、不知道 dtype，就是一段内存。多个 tensor 可以共享同一个 `Storage`（这就是 view）。显存实际是 caching allocator 管的，见第 8 章。
- *`TensorImpl`*：`sizes`、`strides`、`storage_offset`、`dtype`、`device`、`DispatchKeySet`、`version_counter`、`autograd_meta`（`grad_fn` / `grad` / `requires_grad` 挂在这里）。`Tensor` 本身只是一个指向 `TensorImpl` 的智能指针，所以 `Tensor` 传参是廉价的。
- *`TensorOptions`*：`(dtype, device, layout, requires_grad, memory_format)` 的打包。`torch.empty(..., dtype=, device=)` 传下去的就是它，`BackendSelect` key 靠它决定分派到哪个后端。

*`DeviceGuard`* 是一个 RAII 对象：构造时把当前 CUDA device 切到目标 device，析构时切回去。每个 CUDA kernel 的入口都有一个，这就是为什么你可以在 device 0 的上下文里直接对一个 device 1 的 tensor 做运算而不用手动 `set_device`。相应的 `OptionalDeviceGuard` 用于"可能不需要切"的场合，省掉一次 `cudaSetDevice`。

#warn[
  `DeviceGuard` 只管 *当前 device*，不管 *当前 stream*。自己起 side stream 做 overlap 时（见第 9 章），跨 stream 用到的 tensor 必须自己 `record_stream()` 或用 event 同步，caching allocator 不会替你推断依赖。这是手写 overlap 最常见的一类内存复用 bug：tensor 被提前回收给了别的 kernel，数据被覆盖，而且不报错。
]

== TensorIterator：elementwise op 的通用引擎

`add`、`mul`、`relu`、`where`、`copy_`、各种归约 —— 这几百个 op 的 CUDA kernel 不是一个个手写的，它们共用 `TensorIterator`。它负责：

+ *广播（broadcasting）*：把所有输入的 shape 对齐到公共 shape，对被广播的维度把 stride 设成 0。
+ *类型提升*：按下一节的规则算出 common dtype，必要时插入 cast。
+ *维度合并（coalescing）*：如果相邻维度在内存里是连续的，就把它们合并成一维。`(1024, 1024)` 的 contiguous tensor 会被压成 `(1048576,)`，于是能用最简单的一维 grid-stride loop。
+ *维度重排*：按 stride 大小重排循环顺序，让最内层循环走内存里最连续的方向。
+ *并行化与向量化*：CPU 上切分给 `at::parallel_for`；CUDA 上决定 grid/block、以及能不能用 `float4` 这类向量化访存（要求指针对齐且 stride 为 1）。

#insight[
  理解 TensorIterator 就能直接回答"为什么 elementwise op 是 memory-bound"：一次 `y = a + b` 要从 HBM 读 2 份数据、写 1 份，中间只做一次浮点加法。算术强度 $approx 1 "FLOP" / 12 "byte"$，而 A100 的 FP32 算力与 HBM 带宽之比在 $10^2$ 量级 —— 差了两个数量级，所以时间 100% 花在搬数据上。

  推论：*把 $n$ 个连续的 elementwise op 融成一个 kernel，就能把 HBM 往返从 $n$ 次降到 1 次*，理论加速接近 $n$ 倍。这正是 `torch.compile` / Inductor 的第一大收益来源（第 14 章），也是为什么"给 elementwise op 换更快的算法"毫无意义 —— 它根本不在算。
]

== 类型提升（type promotion）

规则简单但有一个高频坑。PyTorch 把参与运算的操作数分成三档 *participating category*：张量（dim $>= 1$）、0 维张量（scalar tensor）、Python 标量。提升按类别的优先级来：*有 dim $>= 1$ 的张量参与时，0 维张量和 Python 标量都不参与 dtype 决策，只跟着走*。

```python
import torch
h = torch.ones(3, dtype=torch.float16)

torch.result_type(h, 2.0)                                   # float16 — Python 标量不提升
torch.result_type(h, torch.tensor(2.0))                     # float16 — 0 维张量也不提升
torch.result_type(h, torch.tensor([2.0]))                   # float32 — 1 维张量参与，提升了
```

第三行就是坑：混合精度训练里，你以为在做一个 fp16 的逐元素运算，因为某个常量被写成 `torch.tensor([eps])`（多了一对方括号）就悄悄提升成了 fp32，多出一次 cast、多一份显存、还可能在 `torch.compile` 里造成额外的 kernel。

同一类别内部按"能不能无损容纳"提升：`bool < 整数 < 浮点 < 复数`，同类内按位宽。跨类别时结果取更宽的那类。

```python
torch.result_type(torch.ones(3, dtype=torch.int64),
                  torch.ones(3, dtype=torch.float16))       # float16 —— 浮点类别赢，不看位宽
```

#warn[
  整数除法与 in-place 是两个额外的坑。

  - `a / b` 对整数 tensor 返回 *浮点*（true division）。想要整除用 `torch.div(a, b, rounding_mode="floor")`，`//` 在旧版本有过 deprecation 反复，现在等价于 floor 模式。
  - *in-place op 不做类型提升*：输出 dtype 永远是左操作数的 dtype。同类别内会 *静默降精度* —— `x_fp16.add_(y_fp32)` 不报错，但结果被截回 fp16；跨类别则直接报错 —— `x_int64.add_(y_fp32)` 得到 `result type Float can't be cast to the desired output type Long`。所以 AMP 代码里的 `+=` 既可能悄悄丢精度也可能突然报错，两种都要留意。写成 out-of-place 就按正常提升规则走。
]

== 注册自定义 op：`torch.ops` 与 `torch.library`

`torch.ops.<namespace>.<op>.<overload>` 是访问 dispatcher 里任何 op 的统一入口，包括 PyTorch 自带的（`torch.ops.aten.add.Tensor`）和你自己注册的。

现代做法是 `torch.library.custom_op`（torch 2.4+ 稳定）。它比早年的 `torch.library.Library("myns", "DEF")` + 手写 schema 字符串好用得多：schema 从 Python 类型注解自动推断，也不需要写 C++。

```python
import torch
from torch import Tensor

@torch.library.custom_op("mylib::rms_norm", mutates_args=())
def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    # 这里可以调你自己的 CUDA / Triton kernel
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight

@rms_norm.register_fake
def _(x, weight, eps=1e-6):
    # 只描述 shape / dtype / device，不碰数值
    return torch.empty_like(x)

def _backward(ctx, grad):
    x, weight = ctx.saved_tensors
    ...                     # 返回与 forward 输入一一对应的梯度
    return grad_x, grad_w, None

def _setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)

torch.library.register_autograd(
    "mylib::rms_norm", _backward, setup_context=_setup_context)

y = torch.ops.mylib.rms_norm(x, w)          # 走 dispatcher，和内置 op 平权
```

三个要点：

- *`mutates_args`* 必须如实填。写了哪些参数会被原地改，`Functionalize` 层才能正确改写，`torch.compile` 才敢重排。填错会得到静默的错误结果。
- *`register_fake`（旧名 `impl_abstract`）是给编译器用的*。`torch.compile` 在 trace 时用 `FakeTensor` 跑一遍图来推断每个中间量的 shape/dtype，它不能真的执行你的 CUDA kernel（没有真实数据、也不该花那个时间）。不注册 fake，遇到你的 op 就 graph break，编译收益直接归零。
- *`register_autograd`* 单独注册反向，语义上等价于 `autograd.Function`，但它注册在 dispatcher 的 Autograd key 上，因此对 `torch.compile`、`vmap`、`export` 都可见。库代码用它，训练脚本里的一次性需求用 `autograd.Function`（第 6 章）就够。

用 `torch.library.opcheck(op, args)` 可以一次性检查 schema 一致性、fake kernel 与真实 kernel 的 shape 是否吻合、autograd 是否正确 —— 写完自定义 op 应当跑一遍。

#note[
  `torch.library` 这套 API 从 torch 2.1 到 2.4 经历过多次改名（`impl_abstract` $arrow.r$ `register_fake`、`define` $arrow.r$ `custom_op`），且仍在演进。这里给的是 2.4+ 的稳定形态，*具体签名以你使用版本的文档为准*。
]

=== C++ 扩展 vs `torch.compile` / Triton

写自定义 kernel 有三条路，面试里问"你会怎么优化这个算子"时要能说清取舍：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*方案*], [*适合*], [*代价*],
  [`torch.compile`], [连续的 elementwise/归约链，只是想省 HBM 往返], [几乎零成本，先试这个],
  [Triton kernel + `custom_op`], [有特殊访存模式（flash-attention 类）、需要手控 tile 和 shared memory], [要懂 GPU 内存层级，调 `num_warps` / `BLOCK`；可与 compile 共存],
  [C++/CUDA 扩展], [要用 CUDA 特有能力（warp intrinsic、TMA、自定义 PTX）、或要接第三方库], [编译工具链、跨平台构建、要自己写 fake kernel 和 autograd],
)

顺序是明确的：*先 `torch.compile`，不够再 Triton，最后才 C++*。绝大多数"我要写个 CUDA kernel"的需求，`torch.compile` 已经能拿到 80% 的收益且零维护成本。

== CPU 与 CUDA kernel 的差异

同一个 op 落到 `CPU` key 和 `CUDA` key 上是两份完全不同的代码：

- *CUDA 是异步 launch*。`at::native::add` 里的 `cudaLaunchKernel` 立刻返回，kernel 排进当前 stream 排队执行。所以 Python 侧测到的 "op 耗时" 常常只是 launch 的几微秒，真正的执行时间要靠 `torch.cuda.synchronize()` 或 CUDA event 才能测准。这也意味着 *CPU 端的 dispatch 开销可以被 GPU 执行掩盖* —— 只要 kernel 够大。kernel 太小时 CPU 反而成了瓶颈（launch bound），这是 CUDA Graph 和 `torch.compile` 的用武之地。详见第 9 章。
- *CPU 是同步的*，并且用 `at::parallel_for` 做线程级并行。底层后端由构建时决定（OpenMP 或 TBB），线程数用 `torch.set_num_threads(n)` 控制。DataLoader 的 worker 进程里通常要把它设成 1，否则每个 worker 都开满线程会互相抢核，反而更慢。

```python
torch.set_num_threads(8)        # op 内部并行（intra-op）
torch.set_num_interop_threads(2)  # op 之间并行（inter-op），必须在任何 op 之前调
```

== 后端开关：cuDNN benchmark 与 TF32

这几个全局开关面试常问，因为它们直接影响"你的模型为什么这次快下次慢/为什么精度对不上"。

```python
torch.backends.cudnn.benchmark = True          # 卷积算法自动选优
torch.backends.cuda.matmul.allow_tf32 = True   # matmul 走 TF32
torch.backends.cudnn.allow_tf32 = True         # 卷积走 TF32（历史上默认就是 True）
torch.set_float32_matmul_precision("high")     # 上面那个 matmul 开关的高层封装
```

*`cudnn.benchmark`* 的语义：第一次遇到某个 `(input shape, dtype, conv 参数)` 组合时，实测所有候选算法各跑一遍，选最快的缓存起来；之后同样的 shape 直接用。收益在固定 shape 的 CNN 上很明显，代价是每个新 shape 都要付一次 benchmark 的时间。所以 *shape 频繁变化的场景（变长序列、动态 batch）要关掉它*，否则会不停地重新 benchmark，反而更慢。它还是不确定性的来源之一（不同次运行可能选到不同算法），见第 10 章。

*TF32 是什么*：Ampere（A100）引入的 tensor core 数据格式，19 位 —— 1 符号 + 8 指数 + 10 尾数。指数位跟 FP32 一样（所以动态范围相同，不会溢出），尾数只有 10 位（所以精度约等于 FP16 的尾数）。用法是：输入输出仍是 FP32，只在 tensor core 内部把尾数截断到 10 位做乘法，累加仍用 FP32。

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left),
  [*格式*], [*指数位*], [*尾数位*], [*说明*],
  [FP32], [8], [23], [基准],
  [TF32], [8], [10], [动态范围同 FP32，精度约同 FP16；只用于 tensor core 的乘法],
  [FP16], [5], [10], [动态范围小，训练需要 loss scaling],
  [BF16], [8], [7], [动态范围同 FP32，精度更低，训练不需要 loss scaling],
)

`set_float32_matmul_precision` 的三档与 `allow_tf32` 的关系：`"highest"` = 纯 FP32（等价 `allow_tf32 = False`）；`"high"` 和 `"medium"` 都允许用 TF32（等价 `allow_tf32 = True`），区别在于 `"medium"` 还允许用 bf16 做更激进的近似。torch 2.x 里 *matmul 的 TF32 默认是关的*，而 *cuDNN 卷积的 TF32 历史上默认是开的* —— 这个不对称经常让人困惑。

#warn[
  典型翻车：训练脚本里开了 `set_float32_matmul_precision("high")`，然后拿这个模型的输出跟一份 FP32 参考结果对比数值，发现相对误差到了 $10^(-3)$ 量级，怀疑代码写错了。其实是 TF32 的尾数只有 10 位，这个误差完全正常。

  规则：*训练开 TF32（收益大、对收敛无影响）；写单元测试和做数值对拍时关掉它*。
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False   # 对拍前先关
  ```
]

`torch.__config__.show()` 打印这个 build 的全部信息：编译器版本、MKL/MKL-DNN 版本、CUDA / cuDNN / NCCL 版本、是否开了各种后端。排查"为什么这台机器上慢/为什么这个 op 不可用"时第一件事就是跑它。`torch.backends.cudnn.version()`、`torch.version.cuda`、`torch.cuda.get_device_capability()` 是更精确的单点查询。

== 面试怎么讲：60 秒的架构答案

被问"讲讲 PyTorch 的架构"，按这个结构说，不要展开成源码导读：

#quote(block: true)[
  PyTorch 分五层：Python 前端、`torch._C` 的 pybind 胶水、dispatcher、ATen kernel、厂商库。真正有设计含量的是中间的 dispatcher。

  它本质是一张二维表：行是 op（由 `native_functions.yaml` 的 schema 定义），列是 DispatchKey。每个 tensor 带一个 DispatchKeySet，dispatcher 按固定优先级取出最高的那个 key、调对应 kernel，那个 kernel 干完自己的事就把这个 key 排除掉再 redispatch，控制权落到下一层。

  所以一次 CUDA 上的 `a + b`，如果在 autocast 里而且要梯度，会依次经过 Autocast（统一 dtype）、Autograd（挂 `AddBackward0`）、`ADInplaceOrView`（记账，这里是 fallthrough），最后落到 CUDA kernel，由 TensorIterator 处理广播、类型提升、维度合并，再启动一个 elementwise kernel。

  这个设计的价值在于：autograd、AMP、量化、`FakeTensor` 的 shape 推断、`torch.compile` 的 functionalization，全都是这张表上正交的一层，可以任意组合而不互相污染。换成"在每个 op 里写 if"的实现，这些功能就没法叠加了。
]

想再深一层，就接一句"最下面的 elementwise op 全是 memory-bound，所以 Inductor 的主要收益是把它们融成一个 kernel 省 HBM 往返" —— 顺势把话题引到你更熟的第 14 章。

== 面试考点

#interview[
  *Q1*：一次 `a + b` 在 PyTorch 内部发生了什么？

  A：Python 的 `__add__` 是 C 函数槽位，经参数解析进 `at::add`，交给 dispatcher。dispatcher 按 DispatchKeySet 的优先级逐层走：Autocast 统一 dtype、Autograd 挂 `AddBackward0` 并连 `next_functions`、`ADInplaceOrView` 记账（非 in-place 时 fallthrough），每层做完自己的事就排除该 key 再 redispatch，最后落到 CUDA kernel。CUDA kernel 里由 TensorIterator 处理广播、类型提升、维度合并，然后异步 launch 一个 elementwise kernel。
]

#interview[
  *Q2*：DispatchKey 是什么？为什么说 autograd 是"一层"而不是一个 `if`？

  A：DispatchKey 标识"这个 tensor 需要哪些层参与处理"，每个 tensor 带一个 key set。dispatcher 按全局固定的优先级取最高的 key 找 kernel，kernel 执行完把该 key 排除后 redispatch，形成一条剥洋葱式的链。autograd 就是链上优先级高于后端的一层 kernel：它建完图就把 Autograd key 排除掉再往下调真正的计算。这样 AMP、量化、`FakeTensor`、functionalization 也各占一层，彼此正交、可任意组合。
]

#interview[
  *Q3*：TensorIterator 干什么？为什么 elementwise op 是 memory-bound？

  A：它是所有 elementwise 和归约 op 共用的引擎，负责广播（把被广播维的 stride 设 0）、类型提升、连续维度合并、循环重排、并行化与向量化。elementwise op 的算术强度极低 —— `y = a + b` 读 2 份写 1 份只做 1 次加法，约 1 FLOP / 12 byte，而 GPU 的算力带宽比差两个数量级，所以时间全花在 HBM 往返上。推论是把 $n$ 个连续 elementwise op 融成一个 kernel 能接近 $n$ 倍加速，这就是 Inductor 的主要收益来源。
]

#interview[
  *Q4*：`torch.tensor(2.0)` 和 `torch.tensor([2.0])` 跟一个 fp16 张量相乘，结果 dtype 一样吗？

  A：不一样。类型提升按"参与类别"分档：只要有 dim $>= 1$ 的张量参与，0 维张量和 Python 标量就不参与 dtype 决策。所以 `fp16_tensor * torch.tensor(2.0)` 是 fp16，`fp16_tensor * torch.tensor([2.0])` 是 fp32。混合精度代码里多写一对方括号就会引入意外的 fp32 提升。用 `torch.result_type(a, b)` 可以提前查。
]

#interview[
  *Q5*：注册自定义 op 时为什么必须写 `register_fake`？

  A：`torch.compile` 在 trace 时用 `FakeTensor` 走一遍图来推断每个中间量的 shape/dtype，它没有真实数据、也不该真的执行你的 kernel。`register_fake` 就是告诉编译器"给我这样的输入，输出是什么 shape/dtype/device"。不注册的话，编译器碰到你的 op 只能 graph break，编译收益直接没了。同理 `mutates_args` 要如实填，否则 functionalization 会改写错、编译器会做不安全的重排。
]

#interview[
  *Q6*：TF32 是什么？`allow_tf32` 和 `set_float32_matmul_precision` 什么关系？

  A：TF32 是 Ampere 起 tensor core 支持的格式，8 位指数（动态范围同 FP32）+ 10 位尾数（精度约同 FP16），累加仍用 FP32。输入输出都还是 FP32，只是乘法时截断尾数。`set_float32_matmul_precision("high"/"medium")` 是 `torch.backends.cuda.matmul.allow_tf32 = True` 的高层封装，`"highest"` 对应关闭。torch 2.x 里 matmul 的 TF32 默认关、cuDNN 卷积的默认开。训练建议开（收益明显、不影响收敛），做数值对拍和写单测时必须关。
]

#interview[
  *Q7*：`torch.backends.cudnn.benchmark = True` 什么时候该开，什么时候是坑？

  A：它让 cuDNN 在第一次遇到某组 shape 时实测所有候选卷积算法、选最快的缓存下来。固定 shape 的 CNN 训练该开，收益明显。变长输入 / 动态 batch 场景是坑：每个新 shape 都要重新 benchmark 一遍，总开销可能超过收益。它还引入不确定性 —— 不同次运行可能选到不同算法导致数值不完全复现，要严格复现就得关掉。
]

#interview[
  *Q8*：想加速一个自定义算子，`torch.compile`、Triton、C++ 扩展怎么选？

  A：按成本从低到高试。先 `torch.compile` —— 如果瓶颈是一串 elementwise/归约的 HBM 往返，Inductor 融合就能拿到大部分收益，零维护成本。不够再写 Triton kernel 并用 `torch.library.custom_op` 注册进 dispatcher，适合需要手控 tile 和 shared memory 的访存模式（flash-attention 那一类）。只有当你要用 CUDA 特有能力（warp intrinsic、自定义 PTX）或接第三方库时才上 C++/CUDA 扩展，代价是构建工具链和手写 fake kernel、autograd 的维护成本。
]
