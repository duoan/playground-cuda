#import "../template.typ": *

= 随机性、确定性与可复现

"我设了 seed，为什么两次跑出来的 loss 不一样？"——这是最能区分"背过八股"和"真训过模型"的一道题。标准答案不是"再多设几个 seed"，而是：*浮点加法不满足结合律，而 GPU 上的规约顺序不是固定的*。这一章先把随机数的来源理清楚，给一份完整的复现清单，然后解释为什么清单做全了结果仍可能有差异，最后给数值调试和 NaN 排查的方法。核心结论提前说：工程上追求的是"可复现的实验结论"，不是"逐 bit 相同的数值"。

== 随机数从哪来

PyTorch 里不止一个随机数生成器（generator），这是第一个坑：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*Generator*], [*说明*],
  [CPU generator], [`torch.default_generator`，管所有 CPU 上的 `randn` / `dropout` 等],
  [每个 CUDA device 一个], [device 0 和 device 1 的 generator 是*独立*的，状态互不影响],
  [`DataLoader` worker], [每个 worker 进程的 seed 由主进程派生，见下],
  [用户显式创建的], [`g = torch.Generator(device="cuda"); g.manual_seed(0)`，传给需要的 op],
  [Python `random`], [标准库，`random.shuffle` 之类用它],
  [NumPy], [`np.random`，很多 `Dataset` 的数据增强用它],
)

`torch.manual_seed(n)` *会*同时设置 CPU generator 和*所有* CUDA device 的 generator（等价于额外调了 `torch.cuda.manual_seed_all(n)`）。所以日常不需要单独写 `torch.cuda.manual_seed`——这也是一道爱问的细节题。但它*不会*管 Python 的 `random` 和 NumPy。

`DataLoader` 的 worker seed 是派生的：主进程从 DataLoader 的 `generator` 里抽一个 `base_seed`，worker $i$ 拿到 `base_seed + i`，然后在 worker 里用它设置 `torch.manual_seed`、`random.seed` 和 `np.random.seed`。

#note[
  "必须写 `worker_init_fn` 去 seed NumPy" 是一条过时的建议——较新版本的 PyTorch 已经在 worker 启动时帮你 seed 了 `random` 和 `numpy`。仍然需要 `worker_init_fn` 的场合是：你用了*其他*带全局 RNG 状态的库（OpenCV、albumentations、某些 tokenizer），或者想按自己的规则派生 seed。

  但要注意 `base_seed` 是*每次创建 iterator 时*重新抽的。所以第 2 个 epoch 的数据增强随机性和第 1 个不同（这是我们想要的），而如果你想让整个 job 从头到尾可复现，必须给 DataLoader 传一个显式的 `generator`。
]

== 完整的复现清单

```python
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> torch.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)   # 只对新起的子进程生效
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                    # 含所有 CUDA device
    g = torch.Generator()
    g.manual_seed(seed)
    return g

g = seed_everything(42)

def worker_init_fn(worker_id: int) -> None:
    # 只在用了 torch 不负责 seed 的第三方库时才需要
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    generator=g,                  # 决定 shuffle 顺序与 worker 的 base_seed
    worker_init_fn=worker_init_fn,
    persistent_workers=True,
)
```

`PYTHONHASHSEED` 影响的是 `str`/`bytes` 的哈希值，从而影响 `set` 和 `dict` 的迭代顺序。在 Python 里设 `os.environ` 对*当前*进程无效（哈希种子在解释器启动时就定了），要真正生效必须在 shell 里 `export PYTHONHASHSEED=42` 或者用 `torchrun` 传环境变量。大多数训练代码不依赖 set 顺序，所以这条经常可以跳过——但如果你的词表构建、数据分片用了 `set`，它就是必要的。

== 为什么设了 seed 结果还不一样

seed 只控制*随机数序列*。结果的差异还有另外一整类来源，它们和 seed 无关：

- *非确定性 kernel*。任何用 `atomicAdd` 做规约的 kernel，多个线程往同一个地址累加的*到达顺序由硬件调度决定*，每次都不同。浮点加法不满足结合律（下一节），所以顺序不同结果就不同。典型 op：`scatter_add_`、`index_add_`、`index_select` 的反向、`bincount`、`embedding` 的反向（稠密梯度累加）、`nll_loss` 的反向、各种上采样（`interpolate`）的反向、`histc`。
- *cudnn benchmark*。`torch.backends.cudnn.benchmark = True` 会在第一次遇到某个 shape 时实测几种卷积算法挑最快的。机器负载不同 → 挑中不同算法 → 数值不同。
- *TF32 / reduced precision*。见后面单独一节。
- *多线程规约顺序*。CPU 上的 `sum` 用 OpenMP 分块并行，块数取决于 `torch.set_num_threads()` 和实际线程数，块数变了归约树就变了。
- *NCCL 的规约顺序*。AllReduce 在不同拓扑（ring vs tree）、不同 message size 下走不同算法，规约顺序不同。同一个 job 换机器、换 `NCCL_ALGO` 就会得到不同的 bit。这是"分布式训练两次跑 loss 逐 bit 不一致"的直接原因。
- *`torch.compile` / autotune*。Inductor 的 kernel 选择、tile 大小 autotune 结果与机器负载相关，而不同的 tile 划分意味着不同的规约顺序。
- *未初始化内存*。极少见但存在：某些 op 会读到 `torch.empty` 的未初始化内容（通常是 bug）。`torch.utils.deterministic.fill_uninitialized_memory = True`（默认已开）可以让这种情况暴露出来。

== 浮点不结合律：所有问题的根

这是根因，也是面试里最该讲清楚的一点。fp32 只有 24 bit 有效位（尾数 23 bit + 隐含的 1），每次加法都要把结果*舍入*到最近的可表示值。舍入发生在哪、发生几次，取决于加法的顺序。

#formula[
  $ (a + b) + c != a + (b + c) quad "在浮点下" $
]

一个能直接跑的最小例子（fp32，`eps` $= 2^(-23) approx 1.19 times 10^(-7)$）：

```python
import torch

a = torch.tensor(1.0,  dtype=torch.float32)
b = torch.tensor(6e-8, dtype=torch.float32)   # 比 eps 小一点

print(repr(((a + b) + b).item()))   # 1.000000238418579   → 1 + 2*eps
print(repr(( a + (b + b)).item()))  # 1.0000001192092896  → 1 + 1*eps
```

左边先算 `1 + 6e-8`，`6e-8` 大于 `eps/2` 所以向上进一位得到 `1 + eps`，再加一次又进一位得到 `1 + 2*eps`。右边先算 `6e-8 + 6e-8 = 1.2e-7`，这个数刚好略大于 `eps`，加到 1 上只进一位。*同样三个数、同样两次加法，差了一个 `eps`。*

放大到真实规约的规模（A100，fp32，$10^6$ 个 `randn(seed=0)` 元素，只是换个求和顺序）：

```python
x = torch.randn(1_000_000)            # torch.manual_seed(0)
x.sum()          # -1561.4530029296875
x.flip(0).sum()  # -1561.4528808593750   ← 同样的数，反着加
x.double().sum() # -1561.4529594577189   ← fp64 参考值
```

相对误差量级 $10^(-7)$，正好是 fp32 的精度极限。GPU 上的规约是分块的树形归约，块的划分和 kernel 的 grid 配置有关，所以"顺序"在你眼里是不可控的。

再看一个 GPU 上真实的非确定性 op（A100，`index_add_` 把 $2^20$ 个 fp32 累加进 64 个 bin，*同一个进程内连续跑 5 次*）：

```python
idx = torch.randint(0, 64, (1 << 20,), device="cuda")
src = torch.randn(1 << 20, device="cuda")
outs = []
for _ in range(5):
    o = torch.zeros(64, device="cuda")
    o.index_add_(0, idx, src)
    outs.append(o.clone())
# 5 次结果互不相同，两两最大绝对差约 1e-3
```

注意这里 seed 完全固定、输入完全一样，差异纯粹来自 `atomicAdd` 的到达顺序。$2^20 / 64 = 16384$ 次累加进一个 bin，误差就累积到 $10^(-3)$ 这个量级了。

#insight[
  面试标准答案：*seed 保证的是"随机数序列相同"，不是"数值结果相同"*。GPU kernel 的并行规约顺序不固定，而浮点加法不满足结合律，所以逐 bit 相同需要额外强制（`use_deterministic_algorithms`），且要付性能代价。分布式场景下 NCCL 的规约顺序还会随拓扑和 message size 变，更难保证。
]

== `use_deterministic_algorithms` 与它的代价

```python
import torch

torch.use_deterministic_algorithms(True)        # 核心开关
torch.backends.cudnn.deterministic = True       # 卷积走确定性算法
torch.backends.cudnn.benchmark = False          # 关掉算法自动选择
```

`use_deterministic_algorithms(True)` 的行为分三种：*有确定性实现的* op 切换到确定性版本（通常更慢，因为要放弃 `atomicAdd` 改用排序或分段规约）；*没有确定性实现的* op 直接抛 `RuntimeError`，错误信息里会告诉你是哪个 op；少数 op 需要额外的环境变量配合。

最常见的那个额外要求是 cuBLAS：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8    # 或 :16:8
```

cuBLAS 在多个 stream 上复用 workspace 时，split-k 的分块数会随可用 workspace 变化，导致规约顺序不确定。固定 workspace 配置就固定了分块。不设这个变量而开了确定性模式，涉及 matmul 的地方会直接报错。

`torch.use_deterministic_algorithms(True, warn_only=True)` 是一个实用的中间档：不支持的 op 只 warning 不报错，让你先跑起来、同时知道哪些地方仍然不确定。

三个档位的取舍：

#ladder(
  ([不做任何事], [只设 seed], [零代价；loss 曲线形状一致，逐点数值不同]),
  ([`warn_only=True`], [+ `cudnn.deterministic`、关 `benchmark`], [轻微变慢；知道哪些 op 还不确定]),
  ([完全确定性], [+ `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`],
   [部分 op 报错需改代码；卷积和规约类 op 明显变慢]),
)

== TF32：同样的代码在 V100 和 A100 上结果不同

A100 起的 Tensor Core 支持 TF32：和 fp32 一样的 8 bit 指数范围，但尾数只有 10 bit。用它做 matmul 快得多，代价是精度从约 7 位有效十进制数掉到约 3 位。

*默认值经常被记错，这是个好的考点*。在当前的 PyTorch 里：

```python
torch.backends.cuda.matmul.allow_tf32   # False —— matmul 默认不用 TF32
torch.backends.cudnn.allow_tf32         # True  —— 卷积默认用 TF32
torch.get_float32_matmul_precision()    # "highest"
```

也就是说 *fp32 matmul 默认是不走 TF32 的*（1.12 版本改过一次默认值，从 True 改成 False，因为太多人被静默的精度损失坑到）；而 cudnn 卷积默认*是*走 TF32 的。要打开 matmul 的 TF32：

```python
torch.set_float32_matmul_precision("high")   # 允许 TF32
# "highest" = 严格 fp32；"high" = TF32；"medium" = 允许 bf16 级别
# 等价的底层开关：torch.backends.cuda.matmul.allow_tf32 = True
```

较新版本还提供了更细的 `fp32_precision` 接口（`torch.backends.cuda.matmul.fp32_precision`，取值 `"none"` / `"tf32"` / `"ieee"`），语义更明确，逐步替代 `allow_tf32` 这个布尔量。

#warn[
  TF32 是"同样的代码在不同 GPU 上结果不同"的头号原因：V100 没有 TF32，A100 有；同一份代码在两台机器上跑，卷积部分的精度就不一样。做跨机器的数值对比时先把 `torch.backends.cudnn.allow_tf32 = False`、`torch.set_float32_matmul_precision("highest")` 关掉，再比。

  反过来，*正常训练时你应该开 TF32*——bf16/TF32 级别的精度对 SGD 类优化过程完全够用，收益是实打实的速度。
]

== 什么时候需要 bit-wise 确定性

需要的场景其实很少：

- *调试*。定位"哪一步开始出 NaN"、二分查找引入 bug 的 commit，必须先消除噪声，否则你分不清是自己改坏了还是浮点抖动。
- *回归测试*。CI 里断言"这个 kernel 的输出和黄金值一致"。
- *复现别人的 bug*，尤其是那种"跑到第 3721 步崩"的。
- *某些合规/审计要求*。

不需要的场景是大多数：*正常训练只需要统计等价*。两次跑出来 loss 曲线逐点不同但形状一致、最终指标在同一个方差范围内，这就是合格的可复现性。强行追求逐 bit 相同要付性能代价，而且在多机场景下（NCCL 拓扑、机器数变化）根本做不到。

#insight[
  面试怎么答：*"我追求的是可复现的实验，不是逐 bit 相同的数值。"* 具体做法是固定 seed 和数据顺序，把超参和 commit hash 记进 log，然后用*多个 seed 跑同一配置*来估计指标的方差——只有当两个配置的差异大于种子间方差时，才说这个改动有效。这个回答同时展示了工程能力和实验素养，比背 API 强得多。
]

== 数值调试：怎么比较两个结果

比较不能用 `==`。`torch.allclose` 的判据是 $|a - b| <= "atol" + "rtol" dot |b|$：

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*场景*], [*rtol / atol*], [*理由*],
  [fp32 单个 op], [`1e-5` / `1e-8`], [接近 fp32 的 eps 量级],
  [fp32 长规约（大 matmul）], [`1e-4` / `1e-6`], [误差随规约长度累积],
  [bf16], [`1.6e-2` / `1e-5`], [bf16 只有 8 bit 尾数],
  [fp16], [`1e-3` / `1e-5`], [10 bit 尾数],
)

优先用 `torch.testing.assert_close`，它比 `allclose` 好在三点：按 dtype 自动选默认容差、失败时打印*最大差值出现在哪个位置*、还会检查 dtype 和 shape 是否一致。

```python
import torch

torch.testing.assert_close(mine, reference)                      # 用 dtype 默认容差
torch.testing.assert_close(mine, reference, rtol=1e-4, atol=1e-6)
```

两个实用原则：

+ *低精度实现要和高精度参考比，不是和另一个低精度实现比*。验证 bf16 kernel 时，参考值用 fp64 算，然后看 bf16 结果的相对误差是否在 bf16 的理论精度内。两个 bf16 实现互相比，误差会互相掩盖。
+ *比 loss 曲线，不比单点 loss*。单个 step 的 loss 差 $10^(-4)$ 什么都说明不了。要看的是几百步的曲线是否重合、以及最终指标是否落在多种子的方差范围内。

#figure(
  align(center, line-plot(
    series: (
      ("run A (同 seed)", ((0, 6.9), (50, 4.1), (100, 3.2), (150, 2.8), (200, 2.55), (250, 2.41), (300, 2.33))),
      ("run B (同 seed)", ((0, 6.9), (50, 4.13), (100, 3.17), (150, 2.82), (200, 2.53), (250, 2.43), (300, 2.31))),
      ("run C (有 bug)", ((0, 6.9), (50, 4.12), (100, 3.35), (150, 3.20), (200, 3.15), (250, 3.12), (300, 3.10))),
    ),
    x-label: "step", y-label: "loss", width: 8.5, height: 4,
  )),
  caption: [示意图，非实测数据。A 和 B 是同一配置的两次运行：逐点数值不同（浮点噪声），但曲线重合，*这就是合格的可复现性*。C 从第 100 步开始系统性偏离——这才是真 bug 的信号。判据是"偏离是否超出多种子的方差范围"，不是"数值是否相同"。],
) <fig-loss-repro>

== NaN / Inf 排查

NaN 一出现就会通过梯度污染整个模型，所以要*尽早*发现。

*最便宜的检查点*，开销可以忽略（每 step 一次同步，本来 `clip_grad_norm_` 之后往往就有）：

```python
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

if not torch.isfinite(total_norm):
    # 梯度炸了：跳过这一步，别让 NaN 进参数
    opt.zero_grad(set_to_none=True)
    n_skipped += 1
    continue
opt.step()
```

`clip_grad_norm_` 反正要算全局 norm，顺手检查它是否有限，是零额外成本的守卫。注意判断要放在 `opt.step()` *之前*——NaN 一旦进了参数就再也回不来了。

*NaN 最常来自这几个地方*：

- `log(x)` 当 $x = 0$，`log_softmax` 之外手写的 softmax 忘了减 max 导致 `exp` 溢出
- 除零：normalize 时 `x / x.norm()` 而 `x` 全零；`var` 为 0 时的标准化忘了 `eps`
- `sqrt(0)` 的*反向*：$d/(d x) sqrt(x) = 1/(2 sqrt(x))$ 在 0 点是 `inf`。`x.norm()`、`x.pow(0.5)` 在 0 点反向都会炸。写成 `torch.sqrt(x + 1e-12)` 或 `torch.clamp_min(x, 1e-12).sqrt()`
- attention mask 用 `-inf` 填充，而某一行*全部*被 mask 掉 → `softmax` 得到 $0/0$ → NaN。用 `-1e9`（fp32）或 `torch.finfo(dtype).min` 代替 `-inf`，或者保证每行至少一个位置可见
- fp16 溢出：fp16 最大值 65504，激活或梯度超过就变 `inf`。这是 `GradScaler` 存在的理由；bf16 指数范围和 fp32 一样，基本免疫这个问题——这也是现在训练一律用 bf16 而不是 fp16 的主因
- 学习率过大导致的正常发散，先炸 grad norm 再炸 loss

*定位到具体 op* 用 anomaly detection：

```python
with torch.autograd.detect_anomaly():        # 只在调试时用
    loss = model(x).sum()
    loss.backward()
# 一旦某个 backward 产生 NaN，抛异常并打印该 op 的 forward 调用栈
```

它的价值是把"backward 里出的 NaN"映射回*forward 里那一行代码*，这个信息用别的办法很难拿到。代价是每个 op 都要存 forward 的 Python 栈并检查输出，慢好几倍，绝对不能留在生产脚本里。

#warn[
  别在训练 loop 里逐层插 `if torch.isnan(t).any():`。每一次都是一个 `.item()` 级别的隐式同步（第 9 章），几十个检查点能让 step time 翻倍。正确做法是：平时只在 loss 和 grad norm 上各检查一次（用得上现成的同步点），确认有问题之后再开 `detect_anomaly` 或用 forward hook 逐层抓一次。
]

== 面试考点

#interview[
  *Q1*：`torch.manual_seed` 会设置 CUDA 的 generator 吗？

  A：会，它等价于同时设置 CPU generator 和*所有* CUDA device 的 generator，不需要额外调 `torch.cuda.manual_seed_all`。但它管不到 Python 的 `random` 和 NumPy 的 `np.random`，这两个要单独设。注意每个 CUDA device 有独立的 generator。
]

#interview[
  *Q2*：为什么固定了所有 seed，两次训练的 loss 还是不逐 bit 相同？

  A：seed 只固定随机数序列。差异来自：用 `atomicAdd` 做规约的非确定性 kernel（`index_add_`、`scatter_add_`、embedding 反向等），累加顺序由硬件调度决定；`cudnn.benchmark` 可能选到不同算法；TF32 等低精度路径；CPU 多线程的分块规约；分布式下 NCCL 的规约顺序随拓扑和 message size 变。根因是浮点加法不满足结合律，所以顺序不同则结果不同。
]

#interview[
  *Q3*：用一个例子说明浮点加法不满足结合律，以及它为什么和分布式训练有关。

  A：fp32 下取 `a=1.0`、`b=6e-8`（略小于 `eps=1.19e-7`）：`(a+b)+b` 得 `1+2*eps`，`a+(b+b)` 得 `1+eps`，因为舍入发生的次数和位置不同。AllReduce 要把 N 张卡的梯度加起来，ring 和 tree 算法的加法顺序不同、message 分块大小不同，得到的和就差在最后几个 bit。梯度差几个 bit 经过几千步优化会放大成可见的 loss 差异——但这属于正常的数值噪声，不是 bug。
]

#interview[
  *Q4*：`torch.use_deterministic_algorithms(True)` 做什么？代价是什么？

  A：让有确定性实现的 op 切到确定性版本（放弃 `atomicAdd`，改用排序或分段规约，通常更慢），没有确定性实现的 op 直接抛 `RuntimeError`。还需要配合 `torch.backends.cudnn.deterministic = True`、`benchmark = False`，以及给 cuBLAS 设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（否则 matmul 会报错）。`warn_only=True` 是实用的中间档：不支持的 op 只警告，让你先知道哪些地方还不确定。
]

#interview[
  *Q5*：TF32 是什么？默认开不开？

  A：A100 起 Tensor Core 支持的格式，指数范围和 fp32 一样，尾数只有 10 bit。当前 PyTorch 里 `torch.backends.cuda.matmul.allow_tf32` 默认是 *False*（1.12 改过默认值），而 `torch.backends.cudnn.allow_tf32` 默认是 *True*——matmul 不用、卷积用。要开 matmul 的 TF32 写 `torch.set_float32_matmul_precision("high")`。它也是"同样代码在 V100 和 A100 上结果不同"的主要原因。
]

#interview[
  *Q6*：你怎么保证实验可复现？

  A：追求的是*可复现的实验结论*而不是逐 bit 相同。做法：固定 `torch.manual_seed` / `random.seed` / `np.random.seed`，给 DataLoader 显式传 `generator`，记录 commit hash、完整超参、依赖版本和硬件型号；然后*用多个 seed 跑同一配置*来估计指标方差，只有当配置间差异大于种子间方差时才认为改动有效。只有调试和回归测试才需要打开完全确定性模式。
]

#interview[
  *Q7*：怎么排查 NaN？

  A：分两层。廉价守卫：`clip_grad_norm_` 的返回值本来就要算，顺手 `torch.isfinite(total_norm)` 检查，不通过就跳过这一步、别让 NaN 进参数；再在 loss 上查一次。定位阶段：`torch.autograd.detect_anomaly()` 能把 backward 里的 NaN 映射回 forward 的那行代码，但慢好几倍，只调试用。高发位置：`log(0)`、除零、`sqrt(0)` 的反向（导数是 `1/(2*sqrt(x))`）、attention 某行被完全 mask 后 softmax 的 $0/0$、以及 fp16 溢出（这就是要用 bf16 的原因）。
]

#interview[
  *Q8*：怎么验证你手写的 kernel 和 `torch.nn` 的实现等价？

  A：用 `torch.testing.assert_close`，它按 dtype 给默认容差、失败时报出最大差异的位置。关键是*参考值要用更高精度算*：验证 bf16 实现就拿 fp64 结果做参考，看相对误差是否在 bf16 的理论精度（尾数 8 bit，约 $10^(-2)$ 相对）之内；两个低精度实现互比会互相掩盖误差。绝不能用 `==`，也不要在长规约上用 fp32 的默认容差——误差随规约长度累积。
]
