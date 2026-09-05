#import "../template.typ": *

= Tensor：storage、stride、dtype、device

面试问 Tensor，问的不是 `torch.zeros` 怎么调用，而是"这个张量占多少内存"、"哪个操作偷偷 copy 了"、"为什么改了一个视图另一个也变了"。这一章把 Tensor 拆成两半：一块一维连续的 `Storage`，加一组描述"怎么走这块内存"的元数据。看懂这个模型，`view` / `reshape` / `expand` / `contiguous` 这一串题就变成同一道题的不同问法。autograd 的图结构见第 6 章，caching allocator 与显存峰值见第 8 章。

== Tensor = Storage + 元数据

一个 `torch.Tensor` 只有两部分：

- *Storage*：一块一维、连续、无类型的裸内存（CPU 上的 RAM 或 GPU 上的 HBM）。多个 Tensor 可以指向同一个 Storage。
- *元数据*：`shape`、`stride`、`storage_offset`、`dtype`、`device`、`layout`、`requires_grad`。这些都是 CPU 侧的小整数，改它们不碰 GPU。

逻辑下标 $(i_0, i_1, ..., i_(n-1))$ 映射到 storage 线性下标的公式只有一条：

#formula[$ "idx" = "offset" + sum_(k=0)^(n-1) i_k dot "stride"_k $]

`stride[k]` 的含义是"第 $k$ 维下标 +1，storage 下标前进多少个元素"（单位是元素，不是字节）。

```python
x = torch.arange(12).reshape(3, 4)
x.shape                  # torch.Size([3, 4])
x.stride()               # (4, 1)
x.storage_offset()       # 0
x.untyped_storage().nbytes()  # 96 = 12 个 int64 × 8 字节（注意返回的是字节数）
x.data_ptr()             # storage 基址 + offset*itemsize
```

#figure(
  align(center, stride-view(shape: (3, 4), stride: (4, 1), n-storage: 12,
                            title: "x = torch.arange(12).reshape(3, 4)")),
  caption: [contiguous 张量：`stride=(4,1)`，逐行扫 storage，逻辑顺序 = 物理顺序。],
) <fig-contig>

`reshape(3,4)` 之所以零成本，是因为它只写了一组新的 `shape` / `stride`——storage 一个字节都没动。

#figure(
  align(center, stride-view(shape: (4, 3), stride: (1, 4), n-storage: 12,
                            title: "x.T — 同一 storage，stride 交换")),
  caption: [`transpose` 只交换 stride。`stride=(1,4)` 表示"往右走一格跳 4 个元素"，
    逻辑上的一行在 storage 里是跳跃的 → non-contiguous。],
) <fig-transpose>

#insight[
  `view`、`transpose`、`permute`、`slice`、`expand`、`squeeze`、`unsqueeze` 全都是*纯元数据操作*，
  时间复杂度与张量大小无关。真正花钱的只有"必须重排内存"的那几个：`contiguous`、`clone`、
  `repeat`、以及 fallback 到 copy 的 `reshape`。
]

== is_contiguous 的判定规则

"contiguous"（准确说是 `torch.contiguous_format`）的定义：从最后一维往前推，stride 必须等于其后所有维度 shape 的乘积。

#formula[$ "stride"_k = product_(j > k) "shape"_j $]

外加两条实现细节，面试爱追问：

- *size 为 1 的维度被跳过*：它的 stride 取任何值都不影响遍历，所以不参与判定。
  `torch.randn(4, 1, 5).transpose(1, 2)` 依然 `is_contiguous() == True`。
- *空张量恒为 contiguous*：`numel() == 0` 直接返回 `True`。

```python
x = torch.arange(12).reshape(3, 4)
x.is_contiguous()                    # True   stride=(4,1) == (4,1)
x.T.is_contiguous()                  # False  stride=(1,4) != (3,1)
x[:, 1:3].is_contiguous()            # False  stride=(4,1)，但行内只取 2 个
x[1:3, :].is_contiguous()            # True   切最外维不破坏连续性
torch.randn(4, 1, 5).transpose(1, 2).is_contiguous()   # True（size-1 维被忽略）
```

规律记这一句：*切最外层维度、或转置 size 为 1 的维度，不破连续性；其它 stride 重排基本都破。*

== view vs reshape vs permute vs transpose

#table(
  columns: (auto, 1.1fr, auto, 1.3fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, center, left),
  [*操作*], [*做什么*], [*可能 copy*], [*要求*],
  [`view(...)`], [改 shape/stride], [否], [新 shape 必须能用纯 stride 表达，否则报错],
  [`reshape(...)`], [先试 `view`，失败则 `contiguous().view()`], [是], [无],
  [`transpose(d0, d1)`], [交换两维的 shape 与 stride], [否], [无],
  [`permute(*dims)`], [任意重排维度], [否], [必须给出全部维度],
  [`movedim(src, dst)`], [把某维搬到某位置], [否], [比 `permute` 好读],
  [`contiguous()`], [按当前逻辑顺序重排内存], [是（已连续则 no-op）], [无],
  [`flatten(s, e)`], [合并连续维度], [是（必要时）], [无],
)

`x.T.view(...)` 报错的原因不是"PyTorch 保守"，而是 stride 系统表达不出来。`x.T` 的 shape 是 `(4,3)`、stride 是 `(1,4)`；要把它看成 `(12,)`，需要一个走位序列 `0,4,8,1,5,9,...`，这不是任何单一 stride 能生成的等差数列。所以 `view` 只能拒绝：

```python
x = torch.arange(12).reshape(3, 4)
x.T.view(12)      # RuntimeError: view size is not compatible with input tensor's
                  # size and stride ... Use .reshape(...) instead.
x.T.reshape(12)   # OK —— 内部做了 contiguous()，发生了一次真实 copy
x.T.contiguous().view(12)   # 等价，但 copy 是你写出来的，读代码的人看得见
```

#warn[
  `reshape` 会不会 copy *取决于输入的 stride*，也就是取决于运行时数据。这意味着显存峰值不确定：
  一个 12 GB 的 activation 走到 `reshape` 那行，可能白涨 12 GB。
  在显存吃紧的路径上（比如 attention 里 merge head 之后）请显式写 `.contiguous().view(...)`，
  让 copy 出现在代码里而不是藏在语义里。
]

一个高频追问：*`permute` 之后想 flatten 怎么办？* 用 `reshape`，或者显式 `contiguous().view(...)`。分布式代码里（NCCL 要求连续 buffer）一律显式，见第 17 章。

== expand vs repeat：stride=0 的广播视图

`expand` 把 size 为 1 的维度"拉长"，实现方式是把该维 stride 设成 0——下标怎么变，storage 位置都不动。所以它是零 copy 的视图。`repeat` 是真的把数据复制 $n$ 份。

#figure(
  align(center, stride-view(shape: (3, 4), stride: (0, 1), n-storage: 4,
                            title: "torch.arange(4).expand(3, 4) — stride[0] = 0")),
  caption: [`expand` 出来的 3 行都映射到 storage 的同一 4 个格子。
    storage 只有 4 个元素，逻辑上却有 12 个位置。],
) <fig-expand>

```python
r = torch.arange(4)                # shape (4,)
e = r.expand(3, 4)                 # shape (3,4) stride (0,1)
e.untyped_storage().nbytes()       # 32 —— 还是那 4 个 int64，storage 没变大
p = r.repeat(3, 1)                 # shape (3,4)，storage 变成 12 个元素，真 copy

e.expand(3, 5)                     # RuntimeError: 只能 expand size 为 1 的维度
r.unsqueeze(0).expand(3, 4)        # 想 expand 先 unsqueeze 造出 size-1 维
```

#warn[
  *往 `expand` 的结果里写数据是错的。* 多个逻辑位置指向同一块内存，写入语义未定义：

  ```python
  e = torch.zeros(1, 4).expand(3, 4)
  e.add_(1)     # RuntimeError: unsupported operation: more than one element of the
                # written-to tensor refers to a single memory location.
  ```

  PyTorch 的 `TensorIterator` 会做 internal-overlap 检查并报错，但*不是所有 op 都走这条检查*
  （尤其是自定义 CUDA kernel 和某些 `index_put_` 路径）。规则：`expand` 的结果只读；要写就先
  `.clone()` 或改用 `repeat`。
]

选择原则：只是要参与广播运算（`a.expand(...) + b`）就用 `expand`，省一次 HBM 往返；
需要一块独立可写的、或者要交给 NCCL / 外部 kernel 的连续内存，才用 `repeat`。
实际上大多数时候连 `expand` 都不用写——算子的广播会自动做等价的事。

== 共享内存 vs 复制：一张清单

#table(
  columns: (1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*共享 storage（视图）*], [*新分配 storage（copy）*],
  [`view` / `reshape`（能走 view 时）], [`clone()`],
  [`transpose` / `permute` / `movedim` / `.T`], [`contiguous()`（原本不连续时）],
  [basic indexing：`x[1]`、`x[:, 2:5]`、`x[..., ::2]`], [advanced indexing：`x[[0,2]]`、`x[mask]`],
  [`squeeze` / `unsqueeze` / `expand` / `broadcast_to`], [`repeat` / `repeat_interleave` / `tile`],
  [`detach()` / `requires_grad_()`], [`to(dtype)`、`float()`、`half()`（dtype 变了必 copy）],
  [`split` / `chunk` / `unbind` / `narrow`], [`cat` / `stack` / `pad`],
  [`t.numpy()`（CPU）、`torch.from_numpy(a)`], [`.cpu()` / `.cuda()`（跨 device 必 copy）],
  [`as_strided`（危险，可越界）], [`torch.tensor(data)`（永远 copy）],
)

`.to()` 的语义是 *"需要才 copy"*：`x.to(torch.float32)` 在 `x` 已是 fp32 时返回 `x` 本身（同一对象），
dtype 或 device 不同时才分配新内存。想强制拿到副本用 `x.to(dtype, copy=True)`。

NumPy 侧记三条：`torch.from_numpy(a)` 共享内存（改一个另一个变）；`t.numpy()` 也共享，但 `t` 必须在 CPU 且 `requires_grad=False`；`torch.tensor(a)` 永远 copy，`torch.as_tensor(a)` 尽量共享。

#note[
  验证是否共享，`data_ptr()` 不够——切片会有 offset。可靠做法是比较
  `x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()`。
]

== clone / detach / .data：谁保留 autograd 连接

这四个写法看着差不多，autograd 行为完全不同：

#table(
  columns: (auto, auto, auto, 1.3fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left),
  [*写法*], [*新内存*], [*连着 autograd*], [*用途*],
  [`x.clone()`], [是], [是（`CloneBackward0`，梯度回传给 `x`）], [要副本又要可微],
  [`x.detach()`], [否（共享 storage）], [否], [只想切断梯度，不想花内存],
  [`x.detach().clone()`], [是], [否], [存快照 / 传给日志或 EMA，最安全],
  [`x.data`], [否], [否，且*不更新 version counter*], [不要用],
)

```python
x = torch.randn(3, requires_grad=True)
y = (x * 2)

y.clone().sum().backward()      # x.grad 有值 —— clone 是可微的
y.detach().sum().backward()     # RuntimeError: does not require grad
```

#warn[
  *不要用 `.data`。* 它绕过了 autograd 的 version counter，导致 in-place 修改不被察觉，
  backward 拿着已被改坏的值算梯度，*不报错、结果错*：

  ```python
  x = torch.randn(4, requires_grad=True)
  y = x.sigmoid()          # backward 需要 y 本身
  y.data.mul_(0)           # 静默改坏 y，version counter 不变
  y.sum().backward()       # 不报错，x.grad 全是 0，纯错
  ```

  换成 `y.detach().mul_(0)` 就会正常抛
  "one of the variables needed for gradient computation has been modified"。
  推荐写法：切梯度用 `detach()`，要存副本用 `detach().clone()`，改参数用 `with torch.no_grad():`。
]

顺序上 `detach().clone()` 比 `clone().detach()` 好一点点：前者 clone 的是一个不需要梯度的张量，
不会在图上挂一个立刻被丢弃的 `CloneBackward0` 节点。功能上两者等价。

== dtype：位宽、动态范围与训练偏好

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, center, right, right),
  [*dtype*], [*位宽*], [*指数位*], [*尾数位*], [*max*], [*eps*],
  [`float32`], [32], [8], [23], [3.4e38], [1.19e-07],
  [TF32（计算模式）], [32 存 / 19 算], [8], [10], [3.4e38], [9.77e-04],
  [`float16`], [16], [5], [10], [65504], [9.77e-04],
  [`bfloat16`], [16], [8], [7], [3.39e38], [7.81e-03],
  [`float8_e4m3fn`], [8], [4], [3], [448], [0.125],
  [`float8_e5m2`], [8], [5], [2], [57344], [0.25],
)

关键对比就一句：*bf16 和 fp32 的指数位都是 8 位，动态范围一样；fp16 只有 5 位指数。*
所以 fp16 的上限是 65504、最小正规数约 6.1e-5，训练中很容易出现两类问题：大 logit / 大梯度上溢成 `inf`，
小梯度下溢成 0。这就是 fp16 训练必须配 `GradScaler`（把 loss 放大再缩回来）的原因。
bf16 的范围与 fp32 相同，通常不需要 scaler；代价是尾数只剩 7 位，`eps` 差了约 8 倍，
所以累加类操作（reduction、optimizer state、norm 统计量）仍然要在 fp32 里做。混精细节见第 5 章。

TF32 不是存储 dtype，而是 Ampere+ 上 tensor core 的一种 fp32 matmul 计算模式：
输入按 10 位尾数截断，累加仍是 fp32。它只影响 matmul / conv 的精度，不改变张量的 dtype 和显存占用。

```python
torch.set_float32_matmul_precision("high")    # 允许 matmul 走 TF32（推荐写法）
torch.backends.cuda.matmul.allow_tf32 = True  # 旧开关，等价；默认 False
torch.backends.cudnn.allow_tf32 = True        # conv 的开关，默认已是 True
```

查一个 dtype 的边界不要背，用 `torch.finfo`（浮点）/ `torch.iinfo`（整数）：

```python
fi = torch.finfo(torch.bfloat16)
fi.max, fi.min, fi.eps, fi.tiny, fi.bits    # tiny = 最小正规数
torch.finfo(torch.float16).max              # 65504.0
torch.iinfo(torch.int8).max                 # 127

# 实用场景：给 attention mask 填一个"安全的负无穷"
scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
```

#note[
  上面用 `finfo(dtype).min` 而不是 `float("-inf")`：`-inf` 在整行全被 mask 时会让 softmax 产出 `nan`
  （分子分母都是 0），用有限的最小值则得到均匀分布，训练不会炸。这是个常考的小陷阱。
]

类型提升遵循 NumPy 风格的规则：`fp16 + fp32 → fp32`，`int + float → float`，
但 *Python 标量不参与提升*——`torch.ones(3, dtype=torch.float16) * 2.0` 仍是 fp16。

== device 与 .to()：什么时候真的异步

`.to(device)` 跨 device 一定 copy。真正的考点是它同步不同步。

```python
x = torch.randn(1024, 1024, pin_memory=True)   # pinned（page-locked）host 内存
g = x.to("cuda", non_blocking=True)            # 真异步：立刻返回，DMA 在后台跑

y = torch.randn(1024, 1024)                    # 普通 pageable 内存
g2 = y.to("cuda", non_blocking=True)           # 参数被忽略，实际是同步 copy
```

原因：DMA 引擎只能安全地读物理地址固定的内存。pageable 内存可能被 OS 换页，
所以 CUDA 必须先拷到一块内部 staging buffer 再传，这一步是同步的。
`DataLoader(pin_memory=True)` 就是为了让 batch 落在 pinned 内存里，配上 `non_blocking=True`
才能把 H2D 藏进上一个 step 的计算里。注意 `pin_memory()` 自身的分配是昂贵且同步的，
不要在训练循环里现场 pin。

#warn[
  *D2H 方向的 `non_blocking=True` 是个陷阱。* `gpu_t.to("cpu", non_blocking=True)` 会立刻返回，
  但数据还没到；此时读 CPU 张量拿到的是垃圾。必须自己插同步点：

  ```python
  cpu_t = gpu_t.to("cpu", non_blocking=True)
  torch.cuda.current_stream().synchronize()   # 或 event.synchronize()
  print(cpu_t)                                # 现在才是对的
  ```

  H2D 方向不用担心：后续 kernel 排在同一个 stream 上，顺序天然被保证。
]

会隐式同步（阻塞 CPU 直到 GPU 追上）的常见操作，写训练循环时要盯：
`.item()`、`.tolist()`、`float(t)` / `bool(t)`、`print(t)`、`.cpu()` / `.numpy()`、
`torch.nonzero`、以及所有依赖 `nonzero` 的布尔掩码索引 `x[mask]`。
所以 `loss.item()` 每 step 都调是在拖 pipeline——攒到 log 间隔再调，或者把 loss 累加在 GPU 上。

```python
torch.cuda.set_sync_debug_mode("warn")   # 或 "error"：任何隐式同步都告警
```

== in-place 操作与 autograd

in-place op（带下划线后缀：`add_`、`relu_`、`clamp_`，以及 `x[0] = 1`、`x += 1`）省一次内存分配，
但和 autograd 有两处硬冲突：

+ *叶子张量不能原地改*：`w.add_(1)` 在 `w.requires_grad=True` 且是 leaf 时报
  "a leaf Variable that requires grad is being used in an in-place operation"。
  这正是 optimizer 必须在 `torch.no_grad()` 下更新参数的原因。
+ *backward 需要的中间值不能被改*：每个张量有 version counter，in-place 会 +1。
  backward 时发现存下来的版本号变了就报
  "one of the variables needed for gradient computation has been modified by an inplace operation"。

```python
x = torch.randn(4, requires_grad=True)
y = x.sigmoid()
y.mul_(2)                # sigmoid 的 backward 需要 y 本身 → 这里就埋雷
y.sum().backward()       # RuntimeError: ... modified by an inplace operation

# 而 relu 的 backward 只需要 mask，所以 inplace 是安全的：
h = torch.relu_(x * 2)   # OK，nn.ReLU(inplace=True) 同理
```

规则：*backward 用到 output 的 op（sigmoid、tanh、exp、softmax）后面不能 in-place；
只用到 input 或 mask 的 op（relu、conv、matmul）通常安全。* 不确定就别 in-place，
省的那点内存不值得调这个 bug。

`torch.no_grad()` 下 in-place 一律放行，因为不建图、不存中间值。手写 optimizer 的模板：

```python
with torch.no_grad():
    for p in model.parameters():
        p.add_(p.grad, alpha=-lr)     # 原地更新，不产生新张量
```

== 内存格式：channels_last

`memory_format` 是 stride 的另一种排布约定，不改 shape。NCHW 张量转 `channels_last` 后
shape 仍是 `(N,C,H,W)`，但 stride 变成 `(H*W*C, 1, W*C, C)`——物理上按 NHWC 存。

```python
x = torch.randn(8, 64, 56, 56)
xl = x.to(memory_format=torch.channels_last)
xl.shape                                          # 不变 (8, 64, 56, 56)
xl.stride()                                       # (200704, 1, 3584, 64)
xl.is_contiguous()                                # False（按 NCHW 判定）
xl.is_contiguous(memory_format=torch.channels_last)   # True
model = model.to(memory_format=torch.channels_last)   # 权重也要转
```

好处是 cuDNN / cutlass 的卷积 tensor core kernel 原生吃 NHWC，用 NCHW 时框架要插隐式转置。
CNN + AMP 场景值得试。Transformer 用不到——那边全是 matmul，没有 channel 维。

== 面试考点

#interview[
  *Q1*：`view` 和 `reshape` 的区别？什么时候必须用 `reshape`？

  A：`view` 只改元数据、绝不 copy，要求目标 shape 能被单一 stride 表达，否则报错。
  `reshape` 先试 `view`，失败就 `contiguous().view()`，可能悄悄 copy。
  `permute` / `transpose` / 跨维切片之后想合并维度，就只能用 `reshape`（或显式 `contiguous().view()`）。
  显存敏感的路径推荐显式写 `contiguous()`，让 copy 可见。
]

#interview[
  *Q2*：`is_contiguous()` 到底怎么判定？为什么 `randn(4,1,5).transpose(1,2)` 还是 contiguous？

  A：判定是"从最后一维往前，`stride[k]` 等于其后所有 `shape` 的乘积"。
  实现里 size 为 1 的维度被跳过——它的 stride 取什么值都不影响遍历顺序，
  所以转置一个 size-1 维度不破坏连续性。空张量也恒为 contiguous。
]

#interview[
  *Q3*：`expand` 和 `repeat` 的区别？为什么不能往 `expand` 的结果里写？

  A：`expand` 把 size-1 维的 stride 设成 0，零 copy 的视图；`repeat` 真复制数据、显存翻 $n$ 倍。
  stride=0 意味着多个逻辑位置映射到同一块内存，写入语义未定义，`TensorIterator` 的
  internal-overlap 检查会报错。要写先 `clone()`，或直接用 `repeat`。
]

#interview[
  *Q4*：`clone()` / `detach()` / `.data` 分别做什么？推荐怎么写？

  A：`clone()` 新内存且可微，梯度经 `CloneBackward0` 回传。`detach()` 共享内存但切断图。
  `.data` 也切断图，*但不更新 version counter*，会让 autograd 察觉不到 in-place 修改，
  产生静默的错误梯度。推荐：切梯度 `detach()`，存快照 `detach().clone()`，改参数 `with torch.no_grad():`。
]

#interview[
  *Q5*：为什么大模型训练偏好 bf16 而不是 fp16？

  A：bf16 的指数位是 8 位，动态范围和 fp32 一致；fp16 只有 5 位指数，max 是 65504、
  最小正规数约 6.1e-5，大 logit 会上溢成 `inf`、小梯度会下溢成 0，必须配 `GradScaler`。
  bf16 通常不需要 scaler，代价是尾数只有 7 位、`eps` 约 7.8e-3，
  所以 reduction、optimizer state、norm 统计量仍要放在 fp32。
]

#interview[
  *Q6*：`non_blocking=True` 什么时候真的异步？

  A：H2D 方向必须源张量在 pinned（page-locked）内存里；pageable 内存会退化成同步 copy，
  因为 DMA 需要物理地址固定。所以要配 `DataLoader(pin_memory=True)`。
  D2H 方向 `non_blocking=True` 会立刻返回但数据未就绪，读之前必须自己 `synchronize()`，
  否则拿到垃圾数据。
]

#interview[
  *Q7*：训练循环里哪些操作会隐式同步 GPU？怎么查？

  A：`.item()`、`.cpu()`、`.numpy()`、`.tolist()`、`print(tensor)`、`bool(tensor)`，
  以及 `torch.nonzero` 和依赖它的布尔掩码索引 `x[mask]`。
  每 step 调 `loss.item()` 就是在打断 CPU 的 launch pipeline。
  用 `torch.cuda.set_sync_debug_mode("warn")` 可以把所有隐式同步点打出来。
]

#interview[
  *Q8*：什么样的 in-place 操作会破 autograd？为什么 `nn.ReLU(inplace=True)` 却没问题？

  A：backward 需要 *output* 的 op 不能被 in-place 覆盖（sigmoid、tanh、exp、softmax），
  version counter 一变 backward 就报错。relu 的 backward 只需要"哪些位置大于 0"这个 mask，
  可以从改写后的 output 直接推出来，所以 in-place 安全。另外叶子参数在 `requires_grad=True`
  时不允许 in-place，必须包在 `torch.no_grad()` 里——这就是 optimizer 的写法。
]
