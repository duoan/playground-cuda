#import "../template.typ": *

= LayerNorm

LayerNorm 是 transformer 里每个 sub-layer 前后的标配算子——和 softmax 一样，*按行归约*，但归约的是 mean / variance 而不是 max / sum-exp。这一章我们要把它讲透：

- LayerNorm 的数学定义，以及训练框架为什么存 `rstd` 而不是 `var`。
- naive 三 pass（one thread per row）——小 batch 时够快，大 hidden 时为什么崩。
- Welford 算法——单 pass 数值稳定的 mean / var，以及和 naive 单 pass 公式的对比。
- block-per-row + shared memory reduction——"两个 reduction + 一个逐元素变换"的标准模板。
- warp shuffle reduction + sum/sumsq 融合——一次扫描同时得到 mean 和 var。
- `float4` 向量化 load/store，以及 fp16/bf16 输入时 fp32 累加的陷阱。
- RMSNorm 与 fused LayerNorm + residual add 的生产意义。

对应源码：`src/cuda/05_layernorm.cu`。

本章 optimization ladder：

#ladder(
  ("naive",       "1 thread / row, 3 pass",              "~0.9%"),
  ("welford",     "1 thread / row, 1 pass stable",       "—"),
  ("block",       "1 block / row, 2× smem reduce",       "~39%"),
  ("warp",        "1 block / row, shuffle + sum/sumsq",  "~34%"),
  ("vectorized",  "float4 load/store in block kernel",   "—"),
)

ladder 里百分比是 A100 上 `sm__cycles_active`（SM %），不是相对 naive 的加速比——实测 naive 101 μs、block 5.7 μs、warp 4.6 μs（$64 times 256$ shape）。welford / vectorized 未单独 benchmark。

与第 2 章 reduce sum、第 3 章 softmax 的关系：LayerNorm 的 mean-reduction 和 var-reduction *复用同一套 block / warp shuffle 模板*——区别是 var 需要先 broadcast mean，且 warp 版可以用 $sum(x^2) - (sum x)^2 / H$ 把两次扫描合成一次。

== 问题定义

给定形状 $[ "rows", H ]$ 的输入 $X$（在 LLM 里 `rows = B times S`，$H$ 是 hidden size），可学习参数 $gamma, beta in RR^H$，对每一行 $x in RR^H$ 独立做：

$ mu = frac(1, H) sum_(j=0)^(H-1) x_j $

$ sigma^2 = frac(1, H) sum_(j=0)^(H-1) (x_j - mu)^2 $

$ hat(x)_j = frac(x_j - mu, sqrt(sigma^2 + epsilon)) $

$ y_j = gamma_j dot hat(x)_j + beta_j $

$epsilon$ 是防止除零的小常数（典型 $10^(-5)$）。PyTorch 的 `LayerNorm` 默认 `normalized_shape = (H,)`，即对最后一维归一化。

=== 存 rstd，不存 var

训练 backward 需要 $hat(x)$ 和 $1 / sqrt(sigma^2 + epsilon)$。定义：

$ "rstd" = frac(1, sqrt(sigma^2 + epsilon)) = (sigma^2 + epsilon)^(-1/2) $

forward 保存 `(mean, rstd)` 或只存 `rstd`（mean 可从输入重算，但通常一起存）。*不存 $sigma^2$* 的原因：

1. *backward 直接用 rstd*：$partial y / partial x$ 的公式里出现的是 $1/sqrt(sigma^2 + epsilon)$，不是 $sigma^2$ 本身。
2. *少一次 sqrt + div*：forward 已经算过 `inv_std`，backward 复用，避免重复开方。
3. *数值范围更友好*：$sigma^2$ 可能极小（接近 0 时 rstd 很大），存 rstd 和存 var 在 fp16 下精度特性不同——框架统一存 rstd。

#insight[
  面试里问 "forward 该存什么"：答 *rstd（reciprocal std）+ mean*（或 input 以便重算）。这和 cuDNN / PyTorch 的 saved tensors 一致。
]

=== Roofline：mostly memory-bound，但 pass 数决定上限

对一行长度 $H$ 的元素，naive 三 pass 版本每元素大致：

- Pass 1 (mean)：读 1×，1 FADD
- Pass 2 (var)：读 1×，1 FSUB + 1 FMUL + 1 FADD
- Pass 3 (normalize + affine)：读 $x, gamma, beta$ 各 1×，写 1×

*每元素约 5 次读 + 1 次写*（affine 多读 $gamma, beta$）。算术强度：

$ "AI" approx frac(4 "FLOP", 24 "B") approx 0.17 "FLOP/B" $

A100 ridge point 约 13 FLOP/B——*memory-bound*。但和 vector add 不同，naive 版本*同一行读 3 遍*，理论 traffic 是单 pass 的 ~2.5×。$H = 4096$ 时每行 16 KB，三 pass 读 $x$  alone 就是 48 KB。

LLM 典型 shape：$B = 32, S = 2048, H = 4096$ → $"rows" = 65536$。naive 的 grid 有 65536 个 thread，足够喂饱 GPU；但 $B = 1, S = 1, H = 4096$ 时只有 1 行——*必须 block-per-row*。

#warn[
  LayerNorm 的性能诊断要*分开看 rows 和 H*：rows 小 → 并行度不够，block-per-row 也不够（grid = rows）；H 大 → 三 pass 访存浪费是主因。Transformer inference 的 decode 阶段（$S = 1$）是经典踩坑场景。
]

== v1: naive 三 pass（one thread per row）

```cpp
__global__ void layernorm_naive_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int rows, int cols, float eps) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float mean = 0.0f;
  for (int col = 0; col < cols; ++col) {
    mean += row_ptr[col];
  }
  mean /= static_cast<float>(cols);

  float var = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const float diff = row_ptr[col] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(cols);

  const float inv_std = 1.0f / sqrtf(var + eps);

  for (int col = 0; col < cols; ++col) {
    const float normalized = (row_ptr[col] - mean) * inv_std;
    out_ptr[col] = normalized * gamma[col] + beta[col];
  }
}
```

Launch：`blocks = ceil(rows / 256)`，256 threads/block——每个 thread 独占一行，顺序扫 $H$ 三遍。

=== 为什么小 batch 时它"够用"

*1. 零同步开销*。一个 thread 干完一行，不需要 `__syncthreads`、shared memory、shuffle——代码路径最短，debug 最容易。

*2. rows 大时并行度够*。训练时 $B times S$ 通常是几千到几万，grid 能填满 SM。每 thread 的工作量 = $O(H)$，$H = 4096$ 时 ~12K FLOP + 20 KB 访存，不算太小。

*3. 和 CPU reference 一一对应*。源码 `layernorm_cpu` 用 `double` 累加 mean/var，GPU naive 用 float——足够建立正确性 baseline。

=== 什么时候崩

*1. rows 极小*。Decode 阶段 $B times S = 1$，整个 GPU 只有 1 个 thread 在算 LayerNorm——其余 SM 空转。这是*并行模型错误*，不是带宽问题。

*2. H 很大，三 pass 读带宽*。$H = 8192$ 时每行 32 KB，读 $x$ 三遍 = 96 KB/row。block 版本协作读一遍，traffic 立刻少 2/3。

*3. fp32 累加 var 的数值风险*。两 pass 公式（先 mean 再 $sum (x-mu)^2$）在极端数据分布下，$x - mu$ 可能 catastrophic cancellation——见下一节 Welford。对一般神经网络激活这不是主因，但面试会考。

*4. 没有向量化*。顺序 scalar load 无法触发 `LDG.E.128`，指令数和 MSHR 利用率都不如 `float4` 版。

#note[
  naive 不是"写错了"——是*刻意保留的教学 baseline*。生产环境（Megatron、FlashAttention 配套 kernel）不会用 one-thread-per-row，但理解三 pass 逻辑是读 block/warp 版的前提。
]

=== ncu 实测

#ncu-snapshot(
  version: "naive (one thread per row)",
  size: [$"rows" = 256$, $"cols" = 4096$],
  rows: (
    ("Duration",            "4 270 µs", ""),
    ("Memory SOL",          "0.8 %",    "跟 softmax naive 完全同构——单 SM 空转"),
    ("Compute SOL",         "0.0 %",    ""),
    ("Achieved Occupancy",  "12.5 %",   ""),
    ("Grid Size",           "1",        "一个 block！108 SM 中 107 个 idle"),
  ),
)

跟 03_softmax naive 一模一样的病：*rows 太少 + one-thread-per-row = GPU 完全空转*。LayerNorm 里更糟：三 pass 而不是两 pass。

#verdict(
  problem: [one-thread-per-row 让 4096 元素被单 thread 串行处理，$"rows" = 256$ 又太少不能撑满 grid],
  evidence: [Grid Size 1, Occupancy 12.5%, memSOL 0.8%],
  next: [v3 (block-per-row) 让一个 256-thread block 协作处理一行——256 rows 就是 256 blocks，能填满 108 SM 大约 2.4 waves]
)

== v2: Welford 在线算法

两 pass 公式（先算 mean，再算 $sum(x-mu)^2$）数学上精确，但*单 pass 朴素公式*：

$ sigma^2 = frac(sum x_i^2, H) - mu^2 $

在浮点运算下可能出问题：当 $x_i$ 都很大且方差很小时，$sum x_i^2 / H approx mu^2$，两项接近相等，相减*丢失有效位数*（catastrophic cancellation）。

=== Welford 递推

维护运行统计量 $(n, M_n, S_n)$，每来一个新样本 $x_(n+1)$：

$ M_(n+1) = M_n + (x_(n+1) - M_n) / (n+1) $

$ S_(n+1) = S_n + (x_(n+1) - M_n)(x_(n+1) - M_(n+1)) $

最终 $sigma^2 = S_H / H$。关键性质：$S_n$ 始终是*非负*的平方和累积，不做"两个大数相减"。

```cpp
// device 或 host 单 thread 版
float mean = 0.0f;
float m2 = 0.0f;  // Welford's sum of squared deviations
for (int i = 0; i < H; ++i) {
  const float x = row_ptr[i];
  const float delta = x - mean;
  mean += delta / static_cast<float>(i + 1);
  const float delta2 = x - mean;
  m2 += delta * delta2;
}
const float var = m2 / static_cast<float>(H);
const float inv_std = 1.0f / sqrtf(var + eps);
```

=== 三种方法对比

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*方法*], [*Pass 数*], [*数值稳定性*],
    [两 pass：mean 再 $sum(x-mu)^2$], [2 读 $x$], [mean 误差会传入第二 pass；一般够用],
    [单 pass：$E[x^2] - E[x]^2$], [1 读 $x$], [大均值 + 小方差时 cancellation],
    [Welford 单 pass], [1 读 $x$], [避免大数相减；$S_n >= 0$ 不变量],
  ),
  caption: [*Table:* LayerNorm mean/var 三种单-thread 算法的 pass 数与数值稳定性对比。*Pass 数* 指每行读输入 $x$ 的遍数；*数值稳定性* 描述浮点下 catastrophic cancellation 风险。],
  kind: table,
)

*Observation*：pass 数与稳定性呈清晰 trade-off——两 pass 数学上最直观但读 $x$ 两遍；单 pass $E[x^2]-E[x]^2$ 省访存但在大均值 + 小方差时两项相减丢精度；Welford 单 pass 用递推避免大数相减且 $S_n >= 0$ 不变量，是 CPU reference 和 naive 单-thread 改进的首选。GPU block/warp 版通常不走 Welford 递推，而用 sum/sumsq + shuffle——LLM 激活范围有限，cancellation 不构成主风险。

#insight[
  GPU block/warp 版通常用*并行 reduction 求 sum 和 sumsq*，再用 $"var" = "sumsq" / H - mu^2$——因为 shuffle reduction 天然维护两个累加器，且 LLM 激活值范围有限，cancellation 不严重。Welford 的*单 thread 递推*更适合 CPU 或 warp 内串行聚合；面试推导 Welford 是独立考点，不要求你在 warp shuffle 里实现完整 Welford tree。
]

=== Welford 第二项的推导直觉

展开 $S_(n+1) = S_n + (x - M_n)(x - M_(n+1))$：每次更新把"新点离旧均值的偏差"乘上"新点离新均值的偏差"累加。可以证明 $S_H = sum_(i=1)^H (x_i - mu)^2$（对真实 mean $mu$），而递推过程中只用运行 mean $M_n$，不需要回溯。

=== 完整推导（面试级）

*Step 1：证明 $M_H = mu$。* 归纳法：$M_1 = x_1$。假设 $M_n = frac(1,n) sum_(i=1)^n x_i$，则

$ M_(n+1) = M_n + frac(x_(n+1) - M_n, n+1) = frac(n M_n + x_(n+1), n+1) = frac(sum_(i=1)^(n+1) x_i, n+1) $

*Step 2：证明 $S_H = sum (x_i - mu)^2$。* 关键恒等式：每次加入 $x_(n+1)$ 时，

$ S_(n+1) - S_n = (x_(n+1) - M_n)(x_(n+1) - M_(n+1)) $

可以验证这与"新增点对旧均值的平方偏差"在代数上等价于增量方差贡献。归纳可得 $S_H$ 等于对最终 mean $M_H = mu$ 的离差平方和。

*Step 3：和两 pass 的关系。* 若 mean 精确，$S_H / H = sigma^2$。Welford 的优势是 $M_n$ 和 $S_n$ 在*同一次扫描*中更新，$M_n$ 的舍入误差不会先被放大再传入 var 计算——两 pass 里第一 pass 的 mean 误差会进入 $(x - mu)^2$。

#note[
  数值实验（面试口述即可）：设 $x_i = 10^6 + "noise"_i$，noise 标准差 1。单 pass $E[x^2] - E[x]^2$ 在 float32 下 var 可能变成 0 或 NaN；Welford 仍给出 $approx 1$。
]

=== naive 单 pass vs. Welford vs. GPU sum/sumsq

| 场景 | 推荐 |
| CPU reference / 双精度 | 两 pass + double 累加（源码 `layernorm_cpu`） |
| GPU 单 thread（naive v1 改进） | Welford 单 pass，省 2/3 读 $x$ |
| GPU block/warp 并行 | sum + sumsq + shuffle reduction |
| fp16 输入 | 以上所有路径：*累加器 fp32* |

Welford 版本在 ladder 里和 naive 并行度相同（one thread per row），主要收益是*数值*和*单 pass 读带宽*——不是并行度。要获得数量级加速，必须上 block/warp 版。

== v3: block-per-row + shared memory reduction

```cpp
__global__ void layernorm_block_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int rows, int cols, float eps) {
  __shared__ float shared_sum[kThreadsPerBlock];
  __shared__ float shared_var[kThreadsPerBlock];

  const int row = blockIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  // Pass 1: mean reduction
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += row_ptr[col];
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float mean = shared_sum[0] / static_cast<float>(cols);

  // Pass 2: variance reduction
  float local_var = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float diff = row_ptr[col] - mean;
    local_var += diff * diff;
  }
  shared_var[threadIdx.x] = local_var;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_var[threadIdx.x] += shared_var[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inv_std = 1.0f / sqrtf(shared_var[0] / static_cast<float>(cols) + eps);

  // Pass 3: normalize + affine
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized = (row_ptr[col] - mean) * inv_std;
    out_ptr[col] = normalized * gamma[col] + beta[col];
  }
}
```

Launch：`<<<rows, 256>>>`——*一个 block 负责一行*，block 内 256 thread 协作。

=== 结构分解

LayerNorm = *两个 reduction + 一个 elementwise transform*。和 softmax block 版一模一样，只是：

- 第一次规约算 `sum` → `mean`
- 第二次规约算 `sum((x-mean)^2)` → `var`
- 第三次逐元素：`(x - mean) * inv_std * gamma + beta`

列方向 stride loop `col += blockDim.x` 保证合并访问：warp 内 lane 0..31 读连续 32 个 float。

=== 代价：4 次 `__syncthreads` + 2 遍读 $x$

两次 tree reduction 各需 $log_2(256) = 8$ 步 sync——共 4 次 barrier（mean 树 + var 树）。加上 mean 算完后*所有 thread 必须看到同一个 mean* 才能算 diff——这是 var pass 无法和 mean pass 完全 fuse 的原因（除非改用 sum/sumsq 技巧，见 v4）。

#note[
  第 2 章讲的 sequential addressing、bank conflict（第一轮 offset = blockSize/2 最严重）、unroll last warp——全部适用于这里的 `shared_sum` / `shared_var` tree。LayerNorm 只是 reduction 的*消费者*。本章 $H = 256$、block 256 时 `smem conf. = 0`——tree 结构上*可能*有 bank conflict，但规模太小、规约步数太少，ncu 累积不到可观测的 conflict 次数；不能从源码 alone 断言"一定有 conflict"。
]

=== block size 与 $H$ 的关系

256-thread block 处理 $H = 4096$：每个 thread 的列循环 $4096 / 256 = 16$ 次——足够摊销 sync 开销。$H = 128$ 时每个 thread 只处理 0–1 个元素，256 个 thread 里一半 idle，*block 过大浪费*。

实践中的调参：

- $H <= 256$：考虑 *warp-per-row*（32 thread）甚至单 warp shuffle 规约整行（第 3 章 softmax warp 版同一思路）。
- $H in [256, 4096]$：256-thread block 是甜点。
- $H >= 8192$：block 内每个 thread 持有多列（multi-items-per-thread），步长仍用 `blockDim.x`，与 reduce 章 chunked 版相同——先 register 累加再进 tree。

Launch `<<<rows, 256>>>` 的 grid 维 = batch 行数。训练时 rows 大，occupancy 高；推理 decode 时 rows = batch，若 batch=1 则整个 LayerNorm 只有一个 block 在跑——*此时瓶颈不是 kernel 内部，而是上层 batching*。

=== 和 softmax block 版的逐项对照

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*步骤*], [*Softmax*], [*LayerNorm*],
    [Reduction 1], [$max_j x_j$], [$sum x_j / H = mu$],
    [Reduction 2], [$sum exp(x_j - m)$], [$sum (x_j - mu)^2 / H = sigma^2$],
    [Elementwise], [$exp(x-m)/Z$], [$(x-mu) dot "rstd" dot gamma + beta$],
    [Saved for backward], [$m$, $Z$ 或 log-sum-exp], [$mu$, rstd, $hat(x)$],
  ),
  caption: [*Table:* Softmax 与 LayerNorm block-per-row 版的三阶段结构对照——两次 row-wise reduction + 一次 elementwise 变换的同一模板，区别在归约对象、saved backward tensor 和 affine 步骤。],
  kind: table,
)

*Observation*：两算子共享第 2/3 章 block reduction 骨架——Softmax 归 max 与 $sum exp$（subtract-max 保数值稳定），LayerNorm 归 sum 与 $sum(x-mu)^2$（mean broadcast 后才能算 diff）。LayerNorm 多一步 affine（$gamma, beta$）且 backward 存 rstd 而非 var——模板可复用，优化技巧正交。

softmax 的 subtract-max 是*数值*技巧；LayerNorm 的 sum/sumsq 是*访存*技巧——两者正交，FlashAttention 后的 post-norm block 里两个算子都会出现。

=== ncu 实测

#ncu-snapshot(
  version: "block (one block per row)",
  size: [$"rows" = 256$, $"cols" = 4096$],
  rows: (
    ("Duration",            "21.7 µs",  "*比 naive 快 197×*"),
    ("Memory SOL",          "12.5 %",   "跟 softmax block 版一样，被 shared memory reduction 阻塞"),
    ("Compute SOL",         "16.7 %",   ""),
    ("Achieved Occupancy",  "29.9 %",   ""),
    ("Grid Size",           "256",      "256 blocks / 108 SM = 2.4 waves"),
  ),
)

- *197× 提速*——绝大部分来自并行度改善（naive Grid=1 → block Grid=256）。algorithm 上是"三 pass → 三 pass 但块内协作"。
- *memSOL 12.5%*：还是很低。原因是 kernel 内有 4 次 `__syncthreads`（load / mean / var / done）加上两次 shared memory tree reduction，这些成本被摊平在访存时间里。

#verdict(
  problem: [block 版依然是三 pass（load x → 算 mean、load x → 算 var、load x → 算 output），read $x$ 3 次；shared memory tree reduction 占用不少 cycle],
  evidence: [Memory SOL 12.5% 远低于 vector add 的 84%；smem tree 内 4 次 `__syncthreads`；两次独立 reduction 各一次 barrier],
  next: [v4 (warp_shuffle) 做两件事：(a) warp shuffle 替代 smem tree —— 消除 bank conflict + 减少 sync；(b) 一次扫描同时累积 sum 和 sumsq，减少 read $x$ 次数]
)

== v4: warp shuffle + sum/sumsq 融合

源码 `layernorm_warp_kernel` 把 reduction 收到 warp 级，并用*一次扫描*同时累积 `sum` 和 `sumsq`：

```cpp
__device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}
```

```cpp
  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = row_ptr[col];
    local_sum += value;
    local_sumsq += value * value;
  }

  local_sum = warp_reduce_sum(local_sum);
  local_sumsq = warp_reduce_sum(local_sumsq);

  const int lane = threadIdx.x % kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;

  if (lane == 0) {
    warp_sums[warp_id] = local_sum;
    warp_sumsq[warp_id] = local_sumsq;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int num_warps = blockDim.x / kWarpSize;
    float sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    float sumsq = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);
    if (lane == 0) {
      const float mean = sum / static_cast<float>(cols);
      const float variance = sumsq / static_cast<float>(cols) - mean * mean;
      stats[0] = mean;
      stats[1] = 1.0f / sqrtf(variance + eps);  // rstd
    }
  }
  __syncthreads();
```

=== sum/sumsq 公式

$ sigma^2 = frac(sum x_i^2, H) - (frac(sum x_i, H))^2 = "sumsq" / H - mu^2 $

*只读 $x$ 一遍* 就得到 mean 和 var 所需的两个矩。对比 block 版：mean 树 + var 树 = 读 $x$ 两遍。

=== warp 两级规约

和 reduce sum 的 warp 版相同（第 2 章 v6）：

1. *Warp 内*：`__shfl_down_sync` 规约每个 thread 的 local partial sum/sumsq。
2. *Warp 间*：lane 0 写 `warp_sums[warp_id]`，warp 0 再 shuffle 规约 8 个 warp partial。
3. *Broadcast*：`stats[0..1]` 写入 shared memory，`__syncthreads` 后所有 thread 读 `mean` 和 `inv_std`。

sync 次数：2 次 `__syncthreads`（对比 block 版 4 次）。shuffle 绕过 smem tree 的读写——本章规模下两版 `smem conf.` 都是 0，差别在 sync 次数和 smem 流量，不在可测的 bank conflict。

=== 读 $x$ 遍数：block vs warp

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*版本*], [*读 $x$*], [*`__syncthreads`*],
    [naive 3-pass], [3], [0],
    [block 2×reduce], [2], [4],
    [warp sum/sumsq], [1], [2],
    [warp + vectorized], [1], [2],
  ),
  caption: [*Table:* LayerNorm ladder 各版本在 mean/var 阶段的读 $x$ 遍数与 `__syncthreads` 次数。读 $x$ 仅计 reduction 阶段；normalize pass 各版均再读 $x$ 一遍（表中未计入）。],
  kind: table,
)

读 $x$ 遍数减半（block 2 遍 vs warp 1 遍）在理论上能省 traffic，但 normalize pass 仍要读 $x$ 一遍，整体不可能 2×。本章 micro-benchmark（$H = 256$, rows = 64，并行度已对齐）上 block → warp 只快 *~23%*（5.7 μs vs 4.6 μs）——shuffle 替代 smem reduction 的实际收益远小于 naive → block 的 *~18×* 并行化收益。$H$ 更大时读遍数权重会上升，但仍不要直接把"少读一遍 $x$"翻译成"快 2×"。

#insight[
  生产级 LayerNorm（Apex、fused kernel in Megatron-LM）在 $H$ 是 128 倍数时用向量化 load 填充 local_sum/sumsq，block size 对齐 warp 倍数，并针对 $H in {1024, 4096, 8192}$ 模板特化——骨架就是本章 v4。
]

=== broadcast mean / rstd

`stats[0]` 和 `stats[1]` 写入 shared memory 后，*所有 thread* 通过 `__syncthreads` 看到相同的 mean 和 rstd。这和 softmax 里把 max 和 sum 广播给全 block 写 normalize 是同一模式——区别是 LayerNorm 的 affine 步还要读 $gamma, beta$，访存从 $1 + 2/H$ 倍（相对纯 normalize）变成 $1 + 2/H + 2$ 倍（$gamma, beta$ 各读一次）。

=== ncu 实测

#ncu-snapshot(
  version: "warp_shuffle (fused sum+sumsq)",
  size: [$"rows" = 256$, $"cols" = 4096$],
  rows: (
    ("Duration",            "20.4 µs",  "比 block 快 6%（不是量级性的）"),
    ("Memory SOL",          "11.1 %",   ""),
    ("Compute SOL",         "11.5 %",   ""),
    ("Achieved Occupancy",  "29.5 %",   ""),
  ),
)

*收益比想象中小*——只快 6%。为什么？

- warp shuffle *确实* 消除了 shared memory tree 的 bank conflict 和 barrier 数（从 4 次减到 1 次）。
- 但 LayerNorm 是 *2-pass memory bound*：pass 1 (load x → 算 mean + variance)，pass 2 (load x, γ, β → 写 y)。真正的瓶颈是 $x$ 被读两次——warp shuffle 优化的是行内规约，*不减少访存*。
- 如果继续追求量级性能，唯一的路径是*减少 pass 数*（如 online 版本，用增量更新的方式一遍搞定 mean + var），或者*和上下游融合*（把 pre-LN 和 attention 融合成一个 kernel）。

#final-verdict(
  status: [warp shuffle 版已经达到 standalone LayerNorm 的高效实现。],
  note: [如果 rows 更大（batch × seq = 1024+），memSOL 会自然拉高到 20-30%（HBM 有时间稳态）。生产系统里 LayerNorm 更多是*被融合到别的 kernel 里*：pre-LN 融进 attention 的 Q/K/V 投影；post-LN 融进 residual add + activation；training 里融进 backward 的 chain rule。standalone LayerNorm 到这里够用。]
)

== v5: 向量化 load/store

block / warp 版的列循环可以换成 `float4`——和 vector add 第 4 章同一原则：

```cpp
  const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
  const int vec_cols = cols / 4;
  float local_sum = 0.0f;
  float local_sumsq = 0.0f;

  for (int v = threadIdx.x; v < vec_cols; v += blockDim.x) {
    const float4 val = row4[v];
    local_sum += val.x + val.y + val.z + val.w;
    local_sumsq += val.x * val.x + val.y * val.y
                 + val.z * val.z + val.w * val.w;
  }
  // tail: scalar loop for cols % 4 elements
```

写回阶段同理：`float4` 存 `normalized * gamma + beta`，$gamma, beta$ 也按 `float4` 读。

=== 收益与约束

- *16B 对齐*：`cols` 和 row 起始地址必须 4 的倍数（`cudaMalloc` 保证基址对齐；`H` 不是 4 倍数要 scalar tail，见 vector add v5 fuse 写法）。
- *收益来源*：减少 load 指令数、更大 outstanding request——对 $H >= 1024$ 的 memory-bound 行内扫描，实测常比 scalar 版快 10–20%。
- *不能破坏 stride loop 的合并访问*：vector index `v` 的 stride 仍是 `blockDim.x`，不是 1。

#warn[
  向量化读 $x$ 算 sum/sumsq 时，四个元素在 lane 内串行累加——不会破坏 warp 合并。但*写回*若用 `float4`，确保 $gamma, beta$ 和 output 对齐一致。
]

== fp16 / bf16 输入：累加必须在 fp32

推理和训练常用 `half` / `__nv_bfloat16` 存激活，LayerNorm 的 reduction 仍应在 *fp32* 完成：

```cpp
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float x = __half2float(row_ptr_half[col]);  // 或 __bfloat162float
    local_sum += x;
  }
```

=== 为什么 fp16 累加会错

$H = 4096$，元素 $O(1)$，真实 sum $O(4096)$。fp16 尾数只有 10 bit（有效精度 ~3 位十进制），*每加一次都可能舍入*——4096 次累加后相对误差可达 $10^(-2)$ 量级。mean 错 → var 错 → 整个 hidden 层 drift。

#insight[
  这是*所有 row-wise reduction 算子*的通用规则：softmax sum、LayerNorm mean/var、RMSNorm sumsq——*load 窄、累加宽、store 窄*。NVIDIA 的 `half` LayerNorm fused kernel 内部用 `float` accumulator；面试说 "fp16 输入 fp16 累加" 是红旗。
]

bf16 有 8 bit 尾数，比 fp16 更差——同样必须 fp32 累加。输出可以 cast 回 fp16 存。

=== 混合精度训练的特殊情况

AMP（automatic mixed precision）里 LayerNorm 常*强制 fp32*（PyTorch `LayerNorm` 的 `dtype` 与 input 相同，但 master weights 和 reduction 在 fp32）。原因：

1. var 是*二阶统计量*，对舍入误差比 mean 敏感一个数量级。
2. rstd 涉及 `sqrt`，输入 var 的 ULP 误差在 rstd 上被放大（导数正比于 $sigma^(-3/2)$）。
3. 训练 backward 对 $hat(x)$ 和 rstd 的依赖链更长——forward 误差会累积进 gradient。

#warn[
  面试陷阱："fp16 Tensor Core matmul 后面接 fp16 LayerNorm 全程 fp16"——*错误*。正确做法：matmul 输出 fp16，LayerNorm kernel 内 promote 到 fp32 做 reduction，再 cast 回 fp16 写 output。
]

== RMSNorm：LayerNorm 的简化兄弟

LLaMA、GPT-NeoX 等主流 LLM 用 *RMSNorm* 代替 LayerNorm：

$ y_j = frac(x_j, "RMS"(x)) dot gamma_j, quad "RMS"(x) = sqrt(frac(1, H) sum x_j^2 + epsilon) $

*没有 mean centering*（不做 $x - mu$），*没有 beta*。少一次 mean reduction，公式更简单。

```cpp
// 概念代码：只需 sumsq → rms → scale
const float sumsq = /* warp/block reduce x*x */;
const float rms_inv = rsqrtf(sumsq / H + eps);
y[j] = x[j] * rms_inv * gamma[j];
```

#note[
  面试常问 "LLM 为什么用 RMSNorm"：答 (1) 少算 mean，省一次 reduction 和一次 pass；(2) 实践中效果与 LayerNorm 相当（Noam Shazeer 2020）；(3) 和 weight decay / residual 配合训练稳定。优化 ladder 和 LayerNorm 几乎相同，只是去掉 mean 相关步骤。
]

== fused ops：LayerNorm + Residual Add

Transformer block 里典型模式：

$ x = x + "Attention"(x); quad x = "LayerNorm"(x) $

PyTorch 分开写是两次 kernel launch + 中间结果写 HBM。Fused kernel：

```cpp
// 伪代码：一个 block 处理一行
// 1. 读 x 和 residual，算 x = x + residual（可只在 smem / reg）
// 2. 对 x 做 LayerNorm in-place
// 3. 写回
```

收益：

1. *省一次 HBM 读写*：residual add 的结果不必落盘再读。
2. *省一次 launch*：~5 μs × 每层 × 每 token，decode 时显著。
3. *backward 也需要 fuse*：Megatron 的 `fused_layer_norm_affine` 配套 fused backward。

FlashAttention 之后的 block 往往 fuse 更多：Attention + bias + dropout + residual + LayerNorm → 一个 mega-kernel 不现实，但 *LN + add* 是最常见的 fuse 单元。

=== Pre-LN vs Post-LN 对 kernel 的影响

*Post-LN*（原始 Transformer）：$x = "LN"(x + "Sublayer"(x))$——fuse 顺序是 sublayer output → add residual → LayerNorm。

*Pre-LN*（现代 LLM 默认）：$x = x + "Sublayer"("LN"(x))$——LayerNorm 在 sublayer *之前*，residual 分支不经过 LN。Kernel 层面：Pre-LN 的 LN 输入是 sublayer 输入（未加 residual），fuse 模式变成 `"LN → matmul/attention"` 而非 `"add → LN"`。

两者对 LayerNorm kernel 本身*没有算法差异*——shape 仍是 $(B times S, H)$ row-wise——但 fuse 的上下游不同，决定 epilogue 能不能和相邻 GEMM 合并。

== backward 简述

Forward 存 $hat(x)$, $mu$, $"rstd"$。Backward 给定 $partial L / partial y$：

$ partial L / partial hat(x)_j = partial L / partial y_j dot gamma_j $

$ partial L / partial x_i = "rstd" / H dot (H dot partial L / partial hat(x)_i - sum_j partial L / partial hat(x)_j - hat(x)_i sum_j partial L / partial hat(x)_j dot hat(x)_j) $

（完整公式见 PyTorch 文档或 *Layer Normalization* 原论文 Appendix。）

=== backward 的结构

和 forward 对称——同样是 *两个 row-wise reduction + elementwise*：

1. 算 $partial L / partial hat(x)_j = partial L / partial y_j dot gamma_j$（elementwise）。
2. $S_1 = sum_j partial L / partial hat(x)_j$，$S_2 = sum_j partial L / partial hat(x)_j dot hat(x)_j$（两个 reduction）。
3. 逐元素组合出 $partial L / partial x_i$（公式里 $S_1, S_2$ 和 rstd 代入）。

$partial gamma_j = sum_"rows" partial L / partial y_j dot hat(x)_j$，$partial beta_j = sum_"rows" partial L / partial y_j$——这两个是*跨 batch 的 reduction*（在 $(B, S, H)$ 里对 $B times S$ 个 row 求和），和 forward 的 row-wise 规约正交。

#insight[
  backward kernel 的优化 ladder 和 forward 同构：block-per-row shuffle reduction 算 $S_1, S_2$。fuse backward + 上游 grad 的 elementwise 可以省一次 global write。Megatron 的 `fused_layer_norm_affine` backward 就是这么干的。
]

面试不要求背完整公式，但要能说出 "backward 需要两个 reduction + 复用 rstd"，并解释 $partial L / partial hat(x)$ 是连接 $y$ 和 $x$ 的桥梁。

== 实测

$"rows" = 64, "cols" = 256$（与源码一致；input + output 各 64 KB，$gamma, beta$ 共 2 KB，整 tensor 约 130 KB），A100 80GB SXM4，`ncu` 抓取。GB/s 列写作 *HBM 实测 / 逻辑*：前者 `dram__bytes.sum / time`，后者按各版本 pass 数估算的理论搬运量。

`ncu` 对 binary 各 kernel 只 profile 一次 launch。$64 "rows"$ 时 $ceil(64 / 256) = 1$——naive 的 grid 是 $(1, 1, 1)$，block 内 256 thread 只有 64 个有行可算；block / warp 版 `#raw("<<<rows, 256>>>")` 才是 grid $(64, 1, 1)$。

#include "../bench/05_layernorm.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*先看 perf 表：*

- naive *100.9 μs* vs block *5.66 μs* vs warp *4.61 μs*——naive → block 是 *~18×*，block → warp 只有 *~23%*。和 ladder 预告一致：第一步是并行模型，第二步是算法细节。
- SM %：naive *0.9%* → block *39.0%* → warp *34.3%*。naive 只占 1 个 block（grid = 1），A100 108 个 SM 里 107 个空转；block / warp 同时跑 64 个 block，SM 利用率跳一个数量级。warp 版 SM % 略低于 block——不是"更慢所以 SM 更闲"，而是 kernel 更短、issue 密度不同；*比 time 列，不比 SM % 排序*。
- HBM % 全部 < 1%，HBM 实测 GB/s 只有 1–17 GB/s，逻辑 GB/s 却写到 1481–1820——130 KB 工作集*全在 L2*，`dram__bytes` 几乎为零。版本间优劣看 `time (μs)`，不要用 `% peak` 或逻辑 GB/s。

*并行模型：两种 launch 的几何差异*

#figure(
  warp-lanes(active: (0,), cell: 0.34,
             title: "naive（decode rows=1）：1 thread 独占整行，warp 里只有 1 个 lane 在扫 H 列"),
  caption: [绿色 = 干活的 lane。$B times S = 1$ 时整个 LayerNorm 只有一个 thread 在工作——和 `pred_on/32` 高无关，是*grid 只有 1 个 active thread*。]
)

#figure(
  warp-lanes(active: range(32), cell: 0.34,
             title: "block/warp 版：warp 内 32 lane 协作 stride loop（col += blockDim.x）"),
  caption: [同一 warp 的 lane 0..31 读连续 32 个 float——合并访问 + 协作规约。$64 "rows"$ 时 grid = 64 block，每 block 256 thread 处理一行。]
)

*再看 diag 表：*

*a) naive `pred_on/32 = 31.9`——lane 利用率高，但并行度为零。* `issued/32 = 32.0`，没有 predication 浪费 issue slot 的问题。31.9 接近 32 是因为*有活干的 thread* 在 scalar loop 里几乎每条指令都 pred_on——但这 64 个 thread 分散在 256-thread block 里，192 个 thread 从 launch 起就 idle。`mem stall = 27.46` 反映单 thread 扫 $H = 256$ 列时 load 延迟无法被同 warp 其他 lane 掩盖（其他 lane 根本没在 load）。

*b) block `pred_on/32 = 22.5`——tree reduction 每轮浪费一半 lane。* `issued/32 = 32.0`：`if (threadIdx.x < offset)` 编译成 predicated add，*不是* warp divergence（没有 lane 走不同 basic block）。但每轮只有前 $s$ 个 thread 的 pred_on 为真——8 步 tree 平均下来约 $1/2 + 1/4 + ... approx 1$ 有效 lane 每轮，累积 `pred_on/32` 掉到 22.5。这是*规约算法的 lane 利用率*，不是访存合并问题。

#figure(
  tree-reduction(mode: "sequential", n: 16, cell: 0.36),
  caption: [block 版 mean/var tree 同一模式：每轮活跃 lane 连续但数量减半。$H = 256$ 时 `smem conf. = 0`——源码里第一轮 offset = 128 *可能* 触发 32-way bank conflict，但规约步数少、block 仅 64 个，ncu 累积不到可观测 conflict；*不能从代码结构 alone 断言 bench 里一定有 conflict*。]
)

*c) warp `pred_on/32 = 27.4`——shuffle 比 tree 保更多 lane 干活。* `issued/32 = 30.0`（略低于 32：warp 间规约阶段 `if (warp_id == 0)` 和 `if (lane == 0)` 让部分 lane predicated-off）。规约主体走 `__shfl_down_sync`，32 lane 全程参与 shuffle，所以 pred_on 比 block 的 22.5 高 5 个点——*这才是 block → warp 23% 加速的定量来源之一*。

*d) warp `barrier stall = 4.98` > block `1.81`——shuffle 不是白赚的。* warp 版只有 2 次 `__syncthreads`，但每次 sync 前要做两轮 warp 内 shuffle + warp 间写 `warp_sums`/`warp_sumsq`——warp 0 等其余 warp 写完才能继续。`barrier stall` 衡量每 issue-active cycle 有多少 warp 卡在 `__syncthreads`；warp 版更高说明*省 smem tree 的代价是把等待转移到 warp sync 路径上*。block 版 4 次 barrier 但每步 tree 更简单，stall 反而更低。

*e) 两版 `smem conf. = 0`——本章规模下 bank conflict 不可见。* block 有 2048 B smem + 两次 8 步 tree，warp 只有 72 B——smem 用量差 28×，但 conflict 计数都是 0。和第 2 章 $N = 2^27$ 看到 314k conflicts 不同：这里 $H$ 小、规约轮次少、同时 active 的 block 只有 64 个。

#insight[
  LayerNorm ladder 的定量故事：`pred_on/32` 解释 block → warp（22.5 → 27.4，lane 利用率提升）；`barrier stall` 解释为什么提升只有 23% 而不是 2×（4.98 vs 1.81，sync 开销换 smem tree）。naive → block 的 18× 几乎全来自 grid 从 1 变 64——看 SM % 0.9% → 39%，不是 Welford 或 sum/sumsq 公式。
]

#warn[
  不要把"读 $x$ 遍数减半"直接翻译成"block → warp 快 2×"。normalize pass 仍要读 $x$、$gamma$、$beta$；micro-benchmark 实测 5.66 → 4.61 μs（~23%）。$H = 4096$ 时绝对差会放大，但相对加速通常仍在 1.1–1.3×，不是 2×。
]

#warn[
  不要跨 shape 比较 ladder 百分比或绝对时间。面试给 numbers 前先确认 $(B, S, H)$ 和是训练还是 decode。rows = 1 时 block / warp 也只有一个 block——和 naive decode 踩坑是同一类并行度问题。
]

== ncu 该看什么

```
ncu --set full --section SpeedOfLight ./build/05_layernorm
```

关键 metric（对照本章 $64 times 256$ 实测）：

- `gpu__time_duration.sum`：naive 101 μs vs block 5.7 μs vs warp 4.6 μs——端到端对比用这个，不是逻辑 GB/s。
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`（SM %）：naive 0.9% vs block 39.0% vs warp 34.3%——*并行度的一眼指标*。naive 单 block 时 107 个 SM 空闲。
- `smsp__thread_inst_executed_per_inst_executed.ratio` (issued/32) + `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio` (pred_on/32)：block 22.5 vs warp 27.4——规约 lane 利用率；issued = 32 说明是 predication 不是 divergence。
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` (barrier stall)：warp 4.98 vs block 1.81——shuffle 版的 sync 等待。
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld+st.sum` (smem conf.)：本章三版都是 0——$H = 256$ 规模太小；增大 $H$ / rows 后再看 block tree 是否冒出 conflict。
- `launch__registers_per_thread` / `launch__shared_mem_per_block_static`：block 21 regs + 2048 B vs warp 20 regs + 72 B。
- `dram__bytes.sum`：全部 < 1% HBM——L2 全命中，增大 shape 后再比 block vs warp 的 DRAM 差。

== 面试白板 code

面试官说"手写一个 layernorm forward"——不要写三 pass（会被追问怎么合成一 pass）。直接写 Welford 单 pass 版：

```cpp
// x: [B, H], y: [B, H], gamma/beta: [H]. 每 block 一 row.
__global__ void layernorm_forward(const float* x, const float* gamma, const float* beta,
                                  float* y, float* mean_out, float* rstd_out,
                                  int H, float eps) {
  int row = blockIdx.x;
  const float* xr = x + row * H;
  float*       yr = y + row * H;

  // Pass 1: Welford online 求 (mean, M2)——每 thread 一份 local，然后 block combine.
  float mean = 0.f, M2 = 0.f;
  int   n    = 0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    ++n;
    float v = xr[j];
    float delta = v - mean;
    mean += delta / n;
    M2   += delta * (v - mean);
  }
  // block reduce (mean, M2, n)：warp shuffle + smem，combine 公式：
  //   n* = n_a + n_b
  //   d  = mean_b - mean_a
  //   mean* = mean_a + d * n_b / n*
  //   M2*   = M2_a + M2_b + d*d * n_a * n_b / n*
  block_reduce_welford(mean, M2, n);  // 见 ch2 warp reduce 骨架 + 上面公式

  __shared__ float mu, rstd;
  if (threadIdx.x == 0) {
    mu   = mean;
    rstd = rsqrtf(M2 / H + eps);        // 1 / sqrt(var + eps)
    if (mean_out) mean_out[row] = mu;   // 存给 backward 用
    if (rstd_out) rstd_out[row] = rstd;
  }
  __syncthreads();

  // Pass 2: 归一化 + affine.
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    yr[j] = (xr[j] - mu) * rstd * gamma[j] + beta[j];
  }
}

// ==== Launch config ====
// gridDim  = B (总行数, batch * seq_len)：每 block 一 row.
// blockDim: 依 H 选，跟 softmax 类似——保证每 lane 处理 4-32 元素:
//   * H <= 512:  block = 128 (4 warp)
//   * H <= 4096: block = 256 (8 warp)   ← LLaMA hidden_size 4096 走这条
//   * H > 4096:  block = 512 或 1024
// 必须是 32 的倍数（Welford block reduce 靠 warp 对齐）。
int block = (H <= 512) ? 128 : (H <= 4096 ? 256 : 512);
layernorm_forward<<<B, block>>>(x, gamma, beta, y, mean_out, rstd_out, H, eps);
```

*核心考点*（追问顺序）：

- *"为什么 Welford 而不是先算 mean 再算 var？"* → 两 pass 各要一次 gmem read $x$——$H$ 大时 $x$ 未必在 L1 里。Welford 一次 pass 拿到 (mean, var)，只读 $x$ 一次。数值稳定性也更好（避免 $sum x^2 - (sum x)^2$ 的灾难性抵消）。
- *"Welford combine 公式记不住怎么办？"* → 写出定义 $M_2 = sum (x_i - "mean")^2$，两块合并时 $d = "mean"_b - "mean"_a$、$"mean"^* = "mean"_a + d dot n_b / n^*$、$M_2^* = M_2^a + M_2^b + d^2 dot n_a n_b / n^*$——推一遍就有了。
- *"forward 存什么给 backward？"* → 存 $mu$ 和 `rstd = 1/sqrt(var+eps)`——backward 直接乘，省一次 sqrt。不要存 var。
- *"RMSNorm 呢？"* → 去掉 mean 相关的所有东西，只留 `rstd = rsqrtf(sum_x2 / H + eps)`，reduction 从两个（mean, M2）降到一个（sum_x2），forward 快约 10%。
- *"$partial L / partial gamma, partial L / partial beta$ 怎么算？"* → 跨 batch 的 reduction：$partial L / partial gamma_j = sum_"row" (partial L / partial y_"row,j") dot hat(x)_"row,j"$。生产实现里用 grid-level partial sum + 第二 stage kernel（见附录 D 第 6 题）。
- *"grid = B 会不会太大？B = 32K 时 grid 也 32K？"* → 没关系。A100 grid.x 上限 $2^31 - 1$，32K 完全放得下；SM 只有 108 个，会依次调度 300 个 block/SM。可能的问题是 kernel launch latency（10μs 级）在小 batch 场景下相对显著——生产实现会把多个 layernorm launch 合并进 CUDA graph 或 fused kernel。

== 面试考点

#interview[
  *Q1*: LayerNorm forward 的数学步骤？存 var 还是 rstd？

  A: (1) 求 mean $mu = sum x / H$；(2) 求 var $sigma^2 = sum(x-mu)^2 / H$；(3) $hat(x) = (x-mu) / sqrt(sigma^2 + epsilon)$；(4) $y = gamma dot hat(x) + beta$。存 *rstd = 1/sqrt(sigma^2 + epsilon)* 和 mean，backward 直接用，少一次 sqrt。
]

#interview[
  *Q2*: naive one-thread-per-row 什么时候够用，什么时候不够？

  A: 够用：rows 大（训练 batch）、先验证正确性。不够：decode rows=1 并行度为零；H 大三 pass 读带宽浪费；需要 fp16 向量化或 fuse 时。应换 block-per-row。
]

#interview[
  *Q3*: Welford 算法解决什么问题？递推公式？

  A: 单 pass 数值稳定求 mean 和 var，避免 $E[x^2] - E[x]^2$ 的大数相减。$M_(n+1) = M_n + (x - M_n)/(n+1)$，$S_(n+1) = S_n + (x - M_n)(x - M_(n+1))$，$sigma^2 = S_H / H$。
]

#interview[
  *Q4*: GPU 上为什么常用 sum/sumsq 而不是 Welford tree？

  A: 并行 reduction 天然维护多个累加器；LLM 激活范围有限，$"sumsq"/H - mu^2$ 足够稳定；Welford 递推是串行依赖，不适合 warp shuffle 的并行模式。Welford 是面试推导题，sum/sumsq 是工程实现。
]

#interview[
  *Q5*: block 版和 warp 版 LayerNorm 的核心区别？

  A: block 版两次 smem tree reduction，读 $x$ 两遍（mean pass + var pass），4 次 `__syncthreads`，`pred_on/32` 约 22.5（tree 每轮半数 lane predicated-off）。warp 版 shuffle + sum/sumsq 融合，读 $x$ 一遍，2 次 sync，`pred_on/32` 约 27.4，但 `barrier stall` 更高（4.98 vs 1.81）。实测 block → warp ~23%，不是 2×。
]

#interview[
  *Q6*: fp16 输入为什么必须用 fp32 累加 mean/var？

  A: fp16 尾数 10 bit，$H = 4096$ 次累加误差累积到 $10^(-2)$ 量级，mean/var 漂移。规则：load 窄、accumulate 宽、store 窄。bf16 更差（8 bit 尾数）。
]

#interview[
  *Q7*: RMSNorm 和 LayerNorm 差什么？LLM 为何偏好 RMSNorm？

  A: RMSNorm 去掉 mean centering 和 beta，只做 $x / "RMS"(x) dot gamma$。少一次 reduction 和一次 pass，效果相当，实现更简单更快。
]

#interview[
  *Q8*: fused LayerNorm + residual add 的意义？

  A: 省一次 HBM 读写（add 结果不必落盘）和一次 kernel launch。Transformer 每层都调用，decode 时 launch 开销占比高。是生产 fused kernel 的基本单元。
]

#interview[
  *Q9*: LayerNorm backward 需要哪些 saved tensor 和 reduction？

  A: 存 $hat(x)$, $mu$, rstd（或足够重算的量）。backward 对 $partial L / partial hat(x)$ 做两次 row-wise reduction（sum 和 dot with $hat(x)$），复用 forward 的 rstd，不重新开方。
]

#interview[
  *Q10*: $H$ 不是 4 的倍数时 float4 向量化怎么处理？

  A: 主体处理 `cols/4` 个 float4，尾巴 0–3 个元素用 scalar loop（tid 0..n_tail），可和 vector add v5 一样在同一个 kernel 里 fuse，避免二次 launch。
]
