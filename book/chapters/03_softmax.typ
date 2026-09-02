#import "../template.typ": *

= Softmax

softmax 是 transformer 里 attention score 归一化的核心，也是面试里*数值稳定性*和*online 算法*的高频考点。这一章我们要把它讲透：

- 为什么 naive 三遍扫行在 GPU 上是灾难（访存 3×、并行度差）。
- subtract-max trick 的数学推导——不是"经验技巧"，而是严格等价变换。
- online softmax（Milakov & Gimelshein, 2018）——*单遍扫描*维护 running max + running sum，FlashAttention 的数值基础。
- block-per-row + shared memory 两次 reduction 的标准并行模板。
- warp-per-row + shuffle 两次 reduction——行宽 ≤ 32 时的极致路径。
- 向量化 load/store 与 fused softmax（mask、causal、scale）。

对应源码：`src/cuda/03_softmax.cu`。

本章 optimization ladder：

#ladder(
  ("naive",        "1 thread / row, 3 pass",           "~5%"),
  ("online",       "1 thread / row, running max+sum",  "~5%"),
  ("block",        "1 block / row, 2× smem reduce",    "~40%"),
  ("warp",         "1 warp / row, 2× shuffle reduce",  "~55%"),
  ("vectorized",   "float4 load in block kernel",      "~50%"),
  ("fused",        "mask / causal / scale inline",     "—"),
)

前两个版本并行度相同（一行一 thread），差异在 pass 数和数值技巧；block / warp 解决行内并行；向量化与 fusion 是带宽与 launch 层面的锦上添花。ladder 里百分比是 A100 上 `sm__cycles_active`（SM %），不是相对 naive 的加速比——实测 naive / online 的 SM % 都约 0.9%（单 block），block 约 35%。

与第 2 章 reduce sum 的关系：softmax 的 max-reduction 和 sum-reduction *复用同一套 block/warp shuffle 模板*——区别只是归约算子（`fmaxf` vs `+`）和 sum 阶段需要先 broadcast max。能把 reduce 章的 ladder 平移过来，是面试里展示"kernel 模式可组合"的好例子。

== 问题定义

给定形状 $[ "rows", "cols" ]$ 的矩阵 $X$，对每一行独立做 softmax：

$ y_i = frac(exp(x_i), sum_(j=0)^("cols"-1) exp(x_j)), quad i = 0, 1, ..., "cols"-1 $

输出每行元素非负且和为 1，可解释为概率分布。在 attention 里，$X$ 是 $Q K^T / sqrt(d)$ 的 logits，$Y$ 是 attention weights。

=== 数值稳定性：overflow 与 subtract-max trick

直接算 `exp(x[i])` 有两个问题：

*1. Overflow*。FP32 的 `expf` 在 $x > 88$ 时溢出为 `inf`，后续 sum 变成 `inf`，归一化出 NaN。

*2. Underflow*。$x << 0$ 时 `expf(x)` 下溢为 0，虽然单个 0 无害，但若所有项都下溢，sum = 0，除法出 NaN。

标准做法：令 $m = max_j x_j$，计算

$ tilde(y)_i = frac(exp(x_i - m), sum_j exp(x_j - m)) $

*等价性证明*：分子分母同乘 $exp(m)$：

$ frac(exp(x_i - m), sum_j exp(x_j - m)) = frac(exp(x_i - m) dot exp(m), sum_j exp(x_j - m) dot exp(m)) = frac(exp(x_i), sum_j exp(x_j)) = y_i $

减去 max 后，至少有一项 $x_i - m = 0$，对应 $exp(0) = 1$，sum $>= 1$，既避免 overflow（最大项指数为 0），又保证分母非零。

=== log-sum-exp 形式

定义 $ "LSE"(x) = log(sum_j exp(x_j)) = m + log(sum_j exp(x_j - m)) $。则：

$ y_i = exp(x_i - m - log d) = exp(x_i - "LSE"(x)) $

log-softmax 直接算 $x_i - "LSE"(x)$，不经过 $y_i$ 中间态——数值上更稳。online 算法维护的 $d$ 就是 $exp("LSE" - m)$ 里的 sum 部分；merge 公式等价于 log-sum-exp 的 associative 版本。

#insight[
  subtract-max 不改变数学结果，只改变*计算的数值路径*。面试里必须能独立推导等价性——这是 safe softmax 和 log-softmax 的共同基础。
]

#warn[
  全行都是 $-oo$（例如 attention 里整行被 mask 掉）时，$m = -oo$，sum 仍为 0。生产代码要单独处理：输出全零或 uniform，不能盲目除法。
]

=== Roofline：softmax 是什么 bound？

对一行长度 $C = "cols"$ 的元素，三 pass naive 版本每元素大致：

- Pass 1 (max)：读 1×，比较 $O(C)$
- Pass 2 (sum exp)：读 1×，1 exp + 1 add
- Pass 3 (normalize)：读 1×，1 exp + 1 div + 1 写

*每元素约 3 次读 + 1 次写 + 2 exp + 1 div*。算术强度：

$ "AI" approx frac(2 "exp" + 1 "div", 16 "B") $

`expf` / `div` 在 GPU 上 latency 高（数十 cycle），但每元素内存访问 16B（1 float 读 × 3 + 1 float 写）。$C$ 较小时 compute 占比上升；$C$ 较大（如 4096、8192）时仍偏 memory-bound，但*重复读同一行 3 遍*是首要浪费——这是和 vector add 的本质区别。

Attention 典型 shape：$B times H times S times S$ 的 logits，reshape 成 $N_"rows" = B dot H dot S$ 行、每行 $C = S$ 列。$S = 4096$ 时每行 16 KB，batch 32 × heads 32 → $N_"rows" = 32768$ 行——足够喂饱 GPU；但 $S = 128$ 时 rows 可能只有几千，block-per-row 的 grid 维不够大，需要 fuse 多行 per block 或增大 batch。

#insight[
  softmax 的性能诊断要*分开看*：$C$ 小 → 看 MUFU / div 吞吐和 warp 利用率；$C$ 大 → 看 dram bytes 和 pass 数。给 numbers 之前先问序列长度和 rows 数。
]

单 pass 理论 traffic（读 1 + 写 1）：$2 N_"rows" C times 4$ B。naive 三 pass 读 3 遍：$4 N_"rows" C times 4$ B（3 读 + 1 写）。*访存优化上限约 2×*——还没算重复 exp 的 compute 浪费。

== v1: naive 三 pass（one thread per row）

```cpp
__global__ void softmax_naive_kernel(
    const float* input, float* output, int rows, int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float max_value = row_ptr[0];
  for (int col = 1; col < cols; ++col) {
    max_value = fmaxf(max_value, row_ptr[col]);
  }

  double sum = 0.0;
  for (int col = 0; col < cols; ++col) {
    sum += exp(static_cast<double>(row_ptr[col] - max_value));
  }

  for (int col = 0; col < cols; ++col) {
    out_ptr[col] = static_cast<float>(
        exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
  }
}
```

一个 thread 负责一整行，逻辑和 CPU reference 一模一样。

=== 三个结构性问题

*1. 并行度差*。$"rows" = 64$ 时只有 64 个 thread 在干活，GPU 有上万个 core。行数小于 SM 数时，大量 SM 空转。

*2. 访存 3×*。同一行数据从 HBM 读了 3 遍（max、sum、normalize）。对 $C = 4096$ 的一行，16 KB 数据读 3 次 = 48 KB traffic，而理论最小只需读 1 次 + 写 1 次。

*3. Pass 2 和 Pass 3 重复算 exp*。`exp(x - max)` 算了两次。可以缓存中间结果到 shared memory / register，但 naive 版本没做。

*4. exp 是特殊函数*。SASS 里 `exp` 走 MUFU（multi-function unit），不是普通 FFMA pipeline。一个 SM 上 MUFU 吞吐有限——当 rows 少、$C$ 小，整个 kernel 可能变成*transcendental-bound* 而非 memory-bound。`ncu` 里看 `smsp__sass_thread_inst_executed_op_mufu` 的占比。

#note[
  sum 用 `double` 累加是刻意的：$C$ 很大时 float 累加 $exp(x - m) in [0, 1]$ 会丢精度。面试追问"为什么不用 float 累加"时，答*大 C 下 ULP 误差*。
]

并行度问题是本章实测里最大的杀手——$64 "rows"$ 时 naive / online / masked / causal 的 grid 只有 $(1, 1, 1)$，整颗 GPU 几乎空转。详见本章「实测」节。

=== 和 vector add 的对比

vector add 里每个元素独立，thread 之间零通信；softmax 里同一行的所有元素通过 max 和 sum *强耦合*——必须先知道整行的 max 才能安全算 exp，必须先知道整行的 sum 才能归一化。这个依赖链决定了优化 ladder 的主线：*先把行内规约并行化*（block / warp reduction），再考虑*减少 pass 数*（online），最后才是 load/store 宽度与 fusion。

== Online Softmax：单遍维护 max 与 sum

Milakov & Gimelshein (2018) 在论文 *Online normalizer calculation for softmax* 中提出：可以在*一次扫描*中同时维护 running max 和 running sum，最终与三 pass subtract-max softmax 在浮点语义下一致。这个技巧后来被 FlashAttention (Dao et al., 2022) 直接采用，成为 attention kernel 的数值基石。

=== 动机

FlashAttention 不能把整行 logits 放进 SRAM——它按 block 流式处理 $Q K^T$ 的片段。每来一个新 block，必须*合并*已有的 $(m, s)$ 和新 block 的统计量，而不能回头重读全局 memory。online softmax 就是这个合并规则的抽象。

具体场景：算 attention 时 $S = Q K^T$，shape $[N, N]$。标准实现 materialize 整个 $S$，做 softmax，再乘 $V$——$S$ 的 HBM 读写是 $O(N^2)$。FlashAttention 每次只算 $S$ 的一个 $B_r times B_c$ tile，tile 的 partial softmax 必须用 online 公式 merge 到 running state，否则无法保证与完整 softmax 等价。

#note[
  论文原文用 "safe softmax" 指 subtract-max 版本。本书沿用源码命名：online softmax = 单遍维护 $(m, d)$ 的算法；block 版 = 两遍（先规约 max/sum，再写回）。
]

=== 推导

假设已处理元素 $x_1, ..., x_(t-1)$，维护：

- $m_(t-1) = max(x_1, ..., x_(t-1))$
- $d_(t-1) = sum_(i=1)^(t-1) exp(x_i - m_(t-1))$

来了新元素 $x_t$，令 $m_t = max(m_(t-1), x_t)$。

关键：旧 sum 的基准 max 是 $m_(t-1)$，新 sum 的基准 max 是 $m_t$。当 $m_t > m_(t-1)$ 时，旧项要*重标定*：

$ d_t = sum_(i=1)^t exp(x_i - m_t) = underbrace(sum_(i=1)^(t-1) exp(x_i - m_t))_("旧元素重标定") + exp(x_t - m_t) $

对 $i < t$：$exp(x_i - m_t) = exp(x_i - m_(t-1)) dot exp(m_(t-1) - m_t)$

因此：

$ d_t = d_(t-1) dot exp(m_(t-1) - m_t) + exp(x_t - m_t) $

当 $m_t = m_(t-1)$（新元素不是更大）时，$exp(m_(t-1) - m_t) = 1$，退化为 $d_t = d_(t-1) + exp(x_t - m_t)$。

*更新公式（代码直接对应）*：

$ m_t = max(m_(t-1), x_t) $
$ d_t = d_(t-1) dot exp(m_(t-1) - m_t) + exp(x_t - m_t) $

扫描结束后 $(m_T, d_T)$ 就是 subtract-max softmax 里的 $(m, sum)$。第二遍写输出：

$ y_i = exp(x_i - m_T) / d_T $

=== 数值走查（手算验证）

取 $x = [1, 2, 3]$：

*Step 0*：$m_0 = -oo$，$d_0 = 0$。

*Step 1*（$x_1 = 1$）：$m_1 = 1$，$d_1 = exp(1 - 1) = 1$。

*Step 2*（$x_2 = 2$）：$m_2 = 2$，$d_2 = d_1 dot exp(1 - 2) + exp(2 - 2) = 1 dot e^(-1) + 1 approx 1.368$。

*Step 3*（$x_3 = 3$）：$m_3 = 3$，$d_3 = 1.368 dot exp(2 - 3) + exp(0) approx 1.368 dot 0.368 + 1 approx 1.503$。

最终：$y_1 = exp(-2)/1.503 approx 0.090$，$y_2 = exp(-1)/1.503 approx 0.245$，$y_3 = exp(0)/1.503 approx 0.665$。三者之和 $approx 1.000$——与 `softmax([1,2,3])` 一致。

注意 Step 2→3 时 $m$ 从 2 升到 3，旧 sum 被乘以 $exp(2 - 3) = e^(-1)$——这就是"重标定"的直觉：max 变大后，旧 exp 值相对新基准都变小了，必须统一缩放。

=== 正确性（合并两个 chunk）

设行分为 $A = {x_1,...,x_k}$ 和 $B = {x_(k+1),...,x_n}$。chunk $A$ 的统计 $(m_A, d_A)$，chunk $B$ 的 $(m_B, d_B)$（各自内部 relative to 局部 max）。

全局 $m = max(m_A, m_B)$，全局 sum：

$ d = d_A dot exp(m_A - m) + d_B dot exp(m_B - m) $

*证明思路*：对 chunk $A$ 中任意 $x_i$，它在全局 sum 里的贡献是 $exp(x_i - m) = exp(x_i - m_A) dot exp(m_A - m)$；对 $A$ 内所有项求和即 $d_A dot exp(m_A - m)$。$B$ 同理。加在一起就是全局 $d$。

这个 merge 是*结合律*的：先 merge $A$ 和 $B$，再 merge $C$，与一次性 merge $A union B union C$ 结果相同——因此可以并行算各 chunk 的 $(m, d)$，再用 tree reduction 合并。FlashAttention 在 SRAM tile 之间正是这么做的。

=== 两 chunk merge 数值走查

仍用 $x = [1, 2, 3]$，拆成 $A = [1, 2]$ 和 $B = [3]$：

*Chunk A*：online 扫描得 $m_A = 2$，$d_A = exp(1-2) + exp(2-2) = e^(-1) + 1 approx 1.368$。

*Chunk B*：$m_B = 3$，$d_B = exp(3-3) = 1$。

*Merge*：$m = max(2, 3) = 3$。

$ d = d_A dot exp(2 - 3) + d_B dot exp(3 - 3) = 1.368 dot e^(-1) + 1 approx 1.503 $

与单遍 online 得到的 $d_3 = 1.503$ 一致。注意 $m_A < m$ 时 $d_A$ 必须乘 $exp(m_A - m)$——这就是 FlashAttention 在 tile 边界做的事。

#insight[
  online softmax 的价值不只是"少一遍循环"。它把 softmax 从*全量依赖*变成*流式可合并*——这是 FlashAttention 不需要 materialize $N times N$ attention matrix 的数值前提。
]

=== 实现细节：为什么 sum 用 double

更新公式里 `row_sum * exp(row_max - new_row_max)` 涉及两个 exp 结果的乘加。当 $C$ 很大（8192、32768）且 logits 分布极端（某些 $x_i - m approx 0$，其余 $<< 0$）时，float 累加会丢 ULP。源码里 `row_sum` 用 `double`，merge 时也 cast 到 double——这是 attention 长序列下的标准做法。block 版 local_sum 仍用 float 是为了 smem 大小和速度；$C > 4096$ 的生产 kernel 往往全程 double 或 Kahan summation。

=== v2: online kernel（one thread per row）

```cpp
__global__ void softmax_online_kernel(
    const float* input, float* output, int rows, int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float row_max = -INFINITY;
  double row_sum = 0.0;
  for (int col = 0; col < cols; ++col) {
    const float x = row_ptr[col];
    const float new_row_max = fmaxf(row_max, x);
    row_sum = row_sum * exp(static_cast<double>(row_max - new_row_max))
              + exp(static_cast<double>(x - new_row_max));
    row_max = new_row_max;
  }

  for (int col = 0; col < cols; ++col) {
    out_ptr[col] = static_cast<float>(
        exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
  }
}
```

Pass 1 合并了 max + sum（online），Pass 2 写输出。访存从 3× 降到 2×，且 Pass 1 的 exp 只算一次（更新 sum 时），Pass 2 的 exp 无法避免（除非缓存 $exp(x - m)$ 到 scratch，用空间换时间）。

并行度问题和 naive 一样——仍是一 thread 一行。下一节解决*行内*并行。

#warn[
  *单跑 online kernel 不会比 naive 快*——实测 `time` 252.29 μs vs 120.80 μs（慢 2.1×）。Pass 1 里每步要算 `exp(row_max - new_row_max)` 做重标定，MUFU 指令比 naive 三 pass 的"先 max 再 bulk exp"更碎；且仍是单 block launch，`SM %` ~0.9%。online 的价值是*流式可合并*（FlashAttention tile merge），不是 standalone softmax 的加速手段。
]

== v3: block-per-row + shared memory reduction

```cpp
constexpr int kThreadsPerBlock = 256;

__global__ void softmax_block_kernel(
    const float* input, float* output, int rows, int cols) {
  __shared__ float shared_max[kThreadsPerBlock];
  __shared__ float shared_sum[kThreadsPerBlock];

  const int row = blockIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float local_max = -INFINITY;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_ptr[col]);
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_max[threadIdx.x] = fmaxf(
          shared_max[threadIdx.x], shared_max[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float row_max = shared_max[0];

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += expf(row_ptr[col] - row_max);
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float row_sum = shared_sum[0];

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
  }
}
```

=== 结构：两个 reduction + 一个 map

softmax 的并行骨架和第 2 章 reduce sum 同构：

1. *grid-stride 局部归约*：每个 thread 扫 `col += blockDim.x` 的列，算 local max / local sum。
2. *block 级 tree reduction*：shared memory 上折半规约。
3. *broadcast + normalize*：`row_max` / `row_sum` 在 shared[0]，所有 thread 读同一个值写回。

Launch：`<<<rows, 256>>>`——一行一个 block，block 内 256 thread 协作。

=== reduction 树的两轮结构

以 $C = 1024$、256 threads 为例：

*Max pass*：
- 每个 thread 处理 4 列（stride 256），得 local_max。
- Tree reduction：128→64→32→16→8→4→2→1，共 8 步 `__syncthreads`。
- thread 0 的 shared_max[0] = 整行 max。

*Sum pass*：
- 已知 row_max，每个 thread 算 4 个 `expf(x - row_max)` 的 local_sum。
- 同样的 tree reduction 得 row_sum。

*Normalize pass*：
- 无 sync，每个 thread 写回自己负责的列。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*阶段*], [*每 thread 工作量*], [*同步次数*],
    [local max], [$ceil(C / 256)$ 次 fmaxf], [0],
    [max reduction], [8 步 tree], [8 × syncthreads],
    [local sum exp], [$ceil(C / 256)$ 次 expf], [0],
    [sum reduction], [8 步 tree], [8 × syncthreads],
    [normalize], [$ceil(C / 256)$ 次 expf + div + store], [0],
  ),
  caption: [*Table:* block softmax kernel 的五阶段工作分解（$C = 1024$、256 threads 示例）。*每 thread 工作量* 以 fmaxf / expf 调用次数计；*同步次数* 为 `__syncthreads` 调用数。两轮 tree reduction 各需 $log_2(256) = 8$ 步，local 与 normalize 阶段无 block 内同步。],
  kind: table,
)

*Observation*：五阶段里只有两轮 tree reduction 需要 `__syncthreads`——local max / local sum exp / normalize 都是 embarrassingly parallel 的 grid-stride 扫描。max reduction 与 sum reduction 之间*必须*插入 barrier：所有 thread 必须看到同一个 `row_max` 才能算 `exp(x - m)`，这正是"先 max 再 sum"两趟 pass 的并行代价，也是 block 内不能 online 化的根因。

两次 reduction 之间必须 `__syncthreads`——所有 thread 必须看到同一个 row_max 才能算 exp。normalize 阶段 row_max 和 row_sum 已在 shared[0]，broadcast 无需额外 sync（只要前面 reduction 的 syncthreads 已完成）。

=== 为什么先 max 再 sum，不能 online 化 block 内？

block 内各 thread 看到的数据是*交错的列*（stride = blockDim），不是连续前缀。online 更新要求*顺序*语义——thread 0 看 col 0, 256, 512...，不能代表"先看到 col 0 再看到 col 1"。

block 版本用两次独立 reduction 是正确的并行分解。online 公式用在*跨 block / 跨 wave* merge（FlashAttention），不是 block 内 thread 之间。

===  occupancy 与 smem

`shared_max[256] + shared_sum[256]` = 2 KB smem——远低于 SM 48 KB 上限。寄存器方面，每 thread 几个 float + 循环变量，通常 < 32 regs，occupancy 接近 100%。瓶颈不在资源，在*每行只用一个 block*——rows 不够多时 SM 填不满。

改进：一个 block 处理*多行*——例如 256 threads = 8 warps，每 warp 一行（即 v4 warp-per-row 的扩展）。grid 维从 rows 降到 $ceil("rows" / 8)$，同时提高 SM 利用率。

=== block 大小的选择

`kThreadsPerBlock = 256` 是教学默认。$C = 4096$ 时每个 thread 处理 16 个元素——足够摊销 reduction 开销。$C < 256$ 时有 thread 空转，但 reduction 仍正确（local_max 初始 $-oo$，local_sum 初始 0）。

#warn[
  `blockIdx.x = row` 意味着 grid 维 = rows。batch × heads × seq 很大时 rows 足够；若 rows 很小（如 1），整个 GPU 只跑 1 个 block——此时应合并多行到一个 block，或走 warp-per-row。
]

== v4: warp-per-row + shuffle reduction

当 $C <= 32$（或 padding 到 32）时，一行恰好在一个 warp 内——可以用 shuffle 做 reduction，*零 shared memory*。

```cpp
__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void softmax_warp_kernel(
    const float* input, float* output, int rows, int cols) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int r = row * warps_per_block + warp_id;
  if (r >= rows) return;

  const float* row_ptr = input + r * cols;
  float* out_ptr = output + r * cols;

  float x = (lane < cols) ? row_ptr[lane] : -INFINITY;
  const float row_max = warp_reduce_max(x);

  float exp_val = (lane < cols) ? expf(x - row_max) : 0.0f;
  const float row_sum = warp_reduce_sum(exp_val);

  if (lane < cols) {
    out_ptr[lane] = exp_val / row_sum;
  }
}
```

=== 机制

- lane $i$ 持有 $x_i$（$i >= C$ 的 lane 喂 $-oo$ / 0，不参与）。
- `warp_reduce_max`：5 步 shuffle down，lane 0 得到 row max。
- 每 lane 算 $exp(x_i - m)$，再 `warp_reduce_sum`。
- 每 lane 写回自己的输出——*无 shared memory sync*，latency 最低。

Launch 示例：`<<<ceil(rows/8), 256>>>`，一个 block 8 个 warp = 8 行。cuDNN / PyTorch 对小 $C$ 的 softmax 走类似路径。

=== shuffle reduction 逐步拆解

以 warp 内 max reduction 为例（5 步，offset = 16, 8, 4, 2, 1）：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    [ *step* ], [ *lane 0 持有* ],
    [初始], [$x_0$],
    [offset=16], [$max(x_0, x_16)$],
    [offset=8], [$max(x_0, x_8, x_16, x_24)$],
    [offset=4], [前 16 个 lane 的 max],
    [offset=2], [前 8 个 lane 的 max],
    [offset=1], [前 4 个 lane 的 max],
    [结束], [lane 0 = 全 warp 32 列的 max],
  ),
  caption: [*Table:* warp 内 max reduction 的 5 步 `__shfl_down_sync` 过程（offset = 16, 8, 4, 2, 1）。每步 lane 0 持有的值覆盖 lane 数翻倍：1 → 2 → 4 → 8 → 16 → 32。sum reduction 结构相同，仅 `fmaxf` 换为 `+`。],
  kind: table,
)

*Observation*：5 步 shuffle 与 block 版 8 步 tree reduction 同构——都是 $O(log N)$ 深度，但 warp shuffle *零 shared memory、零 barrier*，latency 最低。代价是覆盖范围锁死在 32 lane：$C > 32$ 时必须 multi-warp 或退回 block 版，这正是 v4 warp-per-row 的适用边界。

`__shfl_down_sync` 把高 lane 的值拉到低 lane；5 步后 lane 0 持有全局 max。sum reduction 结构相同，只是 `fmaxf` 换成 `+`。

#warn[
  $C > 32$ 时单个 warp 放不下整行——需要 multi-warp reduction（各 warp 先 local max，再 cross-warp reduce），或退回 block 版。head_dim=64/128 的小向量 softmax 常用 2~4 warp per row。
]

#note[
  `__shfl_down_sync(0xffffffff, ...)` 的 mask 必须是参与 warp 的全部 active lane。如果有 divergence，mask 要按实际 ballot 结果设置——面试常考。
]

== v5: 向量化 load/store

block kernel 的 grid-stride 循环可以向量化：每个 thread 一次读 `float4`（4 列），local max / local sum 在 4 个元素上展开。

```cpp
for (int col = threadIdx.x * 4; col < cols; col += blockDim.x * 4) {
  if (col + 3 < cols) {
    float4 v = reinterpret_cast<const float4*>(row_ptr)[col / 4];
    local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    // exp sum 同理，对 v.x..v.w 各算 expf(v.* - row_max)  — 在 row_max 已知后
  } else {
    // scalar 尾巴
  }
}
```

*注意顺序*：向量化 load 用在*第一轮 max reduction 之前*有效；sum 阶段需要已知 `row_max`，不能和 max 融合到同一次 vector load（除非缓存 4 个 float 到 register，等 max 出来再算 exp——寄存器压力上升）。

收益：max pass 和 normalize pass 的 load 宽度 ×4，指令数下降。对 $C = 4096$ 的 memory-bound 场景，*理论上*可再提 10~20%——本章 benchmark 未覆盖 vectorized 版，需单独 profile。

=== 寄存器缓存 exp 的 trade-off

进阶做法：normalize pass 不重新读 global input，而在 max pass 时把每列 $x_i$ 缓存在 register / shared memory，max 出来后一次算 $exp(x_i - m)$，同时累加 sum 和写回。这样整行只读 global *一遍*——但 register 用量 $O(C / "threads")$，$C = 8192$ 时可能 occupancy 暴跌。CUTLASS / cuDNN 根据 $C$ 和 arch 在"2 pass + 低寄存器"和"1 pass + 高寄存器"之间 auto-tune。

#insight[
  softmax 的向量化比 vector add 更 tricky：中间有*跨全部元素的依赖*（max → sum → div）。只能向量化*无依赖的 load/store 阶段*，不能盲目 float4 整个 kernel。
]

== v6: fused softmax — mask、causal、scale

训练框架里 softmax 很少单独出现——前面有 scale（除以 $sqrt(d)$），后面接 matmul $V$，中间有 padding mask / causal mask。能 fuse 的尽量 fuse，少写 global memory。

=== masked softmax

```cpp
for (int col = 0; col < cols; ++col) {
  if (mask_ptr[col] == 0) continue;
  const float x = row_ptr[col];
  const float new_row_max = fmaxf(row_max, x);
  row_sum = row_sum * exp(static_cast<double>(row_max - new_row_max))
            + exp(static_cast<double>(x - new_row_max));
  row_max = new_row_max;
}
// 写回：mask=0 → 0，否则正常 normalize
```

mask=0 的位置*不参与* max/sum——等价于 $x = -oo$。输出置 0（不是 uniform），和 PyTorch `masked_fill(..., -inf)` 再 softmax 一致。

源码 `softmax_masked_kernel` 完整结构：第一遍 online 循环 skip mask=0；第二遍写回时 mask=0 置 0。与 `softmax_online_kernel` 相比只多了 branch——*没有额外 global memory pass*。

变长序列（NLP padding）时，不同 row 的有效长度不同，但 tensor 仍是矩形——mask 保证 padding 位不影响有效位的 softmax 归一化。若忘记 mask，padding 的 0 会参与 max（当有效位都是负数时 max 可能变成 0），attention 权重错误。

=== scale 融合

Attention logits：$x = (Q K^T) / sqrt(d)$。scale 可以在读入时乘，也可以 fuse 进 online 循环：

```cpp
const float x = row_ptr[col] * inv_sqrt_d;
```

fuse scale + softmax 成一个 kernel，避免写 scaled logits 到 global。FlashAttention 把 scale 合在 $Q K^T$ matmul 的 epilogue 里——更彻底的 fusion。

=== causal kernel（源码对应）

```cpp
__global__ void softmax_causal_kernel(
    const float* input, float* output, int rows, int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float row_max = -INFINITY;
  double row_sum = 0.0;
  for (int col = 0; col <= row && col < cols; ++col) {
    const float x = row_ptr[col];
    const float new_row_max = fmaxf(row_max, x);
    row_sum = row_sum * exp(static_cast<double>(row_max - new_row_max))
              + exp(static_cast<double>(x - new_row_max));
    row_max = new_row_max;
  }

  for (int col = 0; col < cols; ++col) {
    out_ptr[col] = (col > row) ? 0.0f
        : static_cast<float>(
              exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
  }
}
```

这里 `row` 既是 block 内的行索引，也是 causal 边界：query position $i$ 只能 attend 到 key position $j <= i$。future 位置直接写 0（不是 $-oo$ 再 exp——已经 out of softmax support）。

#insight[
  mask / causal / scale 三种 fusion 的共性：在 online 循环里*跳过*或*变换*输入，而不是先写中间 tensor 再读。每多一次 global round-trip，attention 端到端 latency 就多 ~5 μs + 带宽浪费。
]

== log-softmax 的数值考量

loss 计算常用 $log "softmax"(x)_i = x_i - m - log(d)$，*直接算 log 空间*，避免先 exp 再 log 的精度损失：

$ log(y_i) = x_i - m - log(sum_j exp(x_j - m)) $

PyTorch `log_softmax` 就是这个。面试追问：为什么不在 softmax 输出上取 `log`？答：$y_i$ 很小时 $log(y_i)$ 损失大量有效位；log-softmax 全程在 log-sum-exp 框架里，和 online 公式一样稳定。

Cross-entropy loss：$cal(L) = -sum_i t_i log(y_i)$。若 $t$ 是 one-hot，只需 $-log(y_k) = -x_k + m + log(d)$——*全程不需要 materialize $y$*。这是 fused softmax+crossentropy kernel 的基础（PyTorch `cross_entropy` 内部路径）。

== safe softmax vs approx softmax

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*方法*], [*做法*], [*场景*],
    [safe (subtract-max)], [$exp(x - m) / sum$], [训练/推理默认],
    [approx (piecewise linear exp)], [查表 / 多项式近似 exp], [极端 latency 敏感、可接受误差],
    [online merge], [分块 merge $(m, d)$], [FlashAttention、长序列],
  ),
  caption: [*Table:* softmax 数值策略三路对比。*safe (subtract-max)* 是本章 block / warp kernel 的实现路径；*approx* 用查表或多项式替代 `expf`；*online merge* 维护 running $(m, d)$ 跨分块合并，是 FlashAttention 的核心。],
  kind: table,
)

*Observation*：生产路径几乎总是 subtract-max——数值稳定、实现简单、与 autograd 兼容。online merge 不是 block 内的替代，而是*跨 tile / 跨 wave* 的 merge（第 8 章 FA）；approx 只在 exp 成为绝对瓶颈且误差可接受时出现（INT8/FP8 量化是另一层近似，与 calibration 绑定）。

推理量化（INT8/FP8）里 exp 表 + scale 是另一层近似——数值范围和校准 (calibration) 绑定。

== 源码里的五个版本如何对应

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*函数*], [*Launch*], [*用途*],
    [`softmax_naive_kernel`], [`<<<ceil(rows/256), 256>>>`], [教学：CPU 逻辑直译],
    [`softmax_block_kernel`], [`<<<rows, 256>>>`], [行内并行的标准模板],
    [`softmax_online_kernel`], [`<<<ceil(rows/256), 256>>>`], [online 公式 + 少 pass],
    [`softmax_masked_kernel`], [`<<<ceil(rows/256), 256>>>`], [padding / 变长序列],
    [`softmax_causal_kernel`], [`<<<ceil(rows/256), 256>>>`], [decoder self-attention],
  ),
  caption: [*Table:* 本章五个 kernel 函数及其 CUDA launch 配置与用途对照。*Launch* 列直接对应 `<<<grid, block>>>` 参数（`launch__grid_size` / `launch__block_size`）。`softmax_block_kernel` 唯一采用 `<<<rows, 256>>>`（一行一 block）；其余四版用 `<<<ceil(rows/256), 256>>>` grid-stride 覆盖多行。],
  kind: table,
)

*Observation*：launch 配置揭示了并行粒度差异——block 版 grid 维 = rows，每 block 256 thread 协作归约*一行*；其余版本 grid-stride 让每 block 处理多行，grid 更小但每 thread 串行更多行。masked / causal 在 compute 路径上与 naive 相同，差异在*哪些元素参与 max/sum*（mask / 三角约束），不是 reduction 结构本身。

运行：`make build/03_softmax && ./build/03_softmax`。默认 $64 times 257$，causal 用 $64 times 64$。所有版本与 CPU reference 对齐，容差 $10^(-4)$。默认 $64 times 257$，causal 用 $64 times 64$。所有版本与 CPU reference 对齐，容差 $10^(-4)$。

== 从本章到 FlashAttention 的衔接

第 8 章 FlashAttention 会把本章的 online merge 嵌入 matmul 循环。这里先建立*符号对应*，避免到那章时公式对不上：

- 本章 $(m, d)$ = FlashAttention 的 $(m, ell)$（running max 和 running sum of exp）。
- 本章"一行 logits"= FA 里一个 query 对*所有 key* 的 score 向量。
- FA 按 $K/V$ 的 column block 切分：每算完一个 $B_c times d$ 的 $K$ tile，得到局部 $(m_j, d_j)$，用 merge 公式更新全局 $(m, d)$。
- FA 的 output $O$ 也是 online 维护的——不只是 softmax 分母，分子 $sum exp(s) V$ 也要随 max 变化重标定。第 8 章会推导 $O$ 的 merge。

#warn[
  本章 softmax 的输入 $X$ 已经 materialize 在 global memory。FA 的 $S = Q K^T$ *从不完整写出*——这是 IO 复杂度的区别（$O(N^2)$ → $O(N)$ HBM traffic），不是 softmax 公式本身的区别。面试问"FlashAttention 快在哪"，*第一答案永远是 IO*，第二答案才是 online softmax。
]

=== 常见实现错误

*错误 1*：不做 subtract-max，直接 `expf(x[i])`。大 logits 立刻 inf/NaN。

*错误 2*：mask 位置参与 max/sum。padding 的 0 会污染 max（除非故意用 0 mask）；应 skip 或置 $-oo$。

*错误 3*：block 内 reduction 后忘记 `__syncthreads`，thread 读到 stale shared memory。

*错误 4*：shuffle reduction 用 `__shfl_xor_sync` 和 `__shfl_down_sync` 混用却不理解 lane 布局——max/sum 结果错一位。

*错误 5*：online merge 时 max 相等分支漏乘 $exp(m_"old" - m_"new")$。当 $m_"new" > m_"old"$ 时必须缩放旧 sum；当相等时因子为 1——代码里统一写乘法形式最稳。

== 实测

$"rows" = 64, "cols" = 257$（input + output 各约 66 KB，整 tensor < 132 KB；causal 用 $64 times 64$），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。GB/s 列写作 *HBM 实测 / 逻辑*：前者 `dram__bytes.sum / time`，后者按各版本 pass 数估算的理论搬运量。

$ceil(64 / 256) = 1$——naive / online / masked / causal 的 grid 都是 $(1, 1, 1)$，256 thread 里只有 64 个处理行；block 版 `#raw("<<<rows, 256>>>")` 的 grid 是 $(64, 1, 1)$。

#include "../bench/03_softmax.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *block 比 naive 快 21×——表里最大的 time 比值*。`time` 120.80 μs → 5.73 μs，伴随 `SM %` 0.9% → 36.6%：一行一 block 把 grid 从 1 扩到 64，*并行度*是数量级差距的主因，不是 subtract-max 或 online 公式。

+ *online 比 naive 慢 2.1×*。252.29 μs vs 120.80 μs——两者 `SM %` 都是 0.9%（仍是单 block）；Pass 1 每步可能 `exp(m_"old" - m_"new")` 重标定，MUFU 比 naive 三 pass 的"先 max 再 bulk exp"更碎。

+ *masked 最慢：436.35 μs*。比 naive 慢 3.6×——在 online 循环上叠加 mask tensor 读和 `if (mask==0)` predication；causal 46.50 μs 不能和 masked 直接比绝对时间（shape $64 times 64$ vs $64 times 257$，且内层只扫 `col <= row`）。

*HBM % 全表 ≈ 0，`% peak` 无诊断价值*——132 KB 工作集在 L2 内，block 版逻辑 GB/s 2197 超过 HBM peak 2039 是*分母过小*的假象，不是真的打满带宽。

#figure(
  hbar-chart(
    (
      ("block", 5.73),
      ("causal", 46.50),
      ("naive", 120.80),
      ("online", 252.29),
      ("masked", 436.35),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)` 排序：block 一行一 block 并行 vs 其余单 block 路径，差距是数量级。],
)

*diag 表读关键教学点：*

*a) naive / online：`issued/32 = pred_on/32 = 32.0`*。单 thread 扫整行，无 predicated branch；`barrier stall = 0`（无 `__syncthreads`）。慢在 grid = 1，不是 lane 浪费。

*b) block：`pred_on/32 = 21.4`，`issued/32 = 31.1`*。issued − pred_on = 9.7 来自 grid-stride 尾部和 reduction 树里 `if (tid < offset)` 的 predicated-off lane——*不是* warp divergence（不同 basic block），因为 `issued/32` 仍近 32。`barrier stall = 2.43` 是两次 smem tree 的 `__syncthreads` 成本；`smem conf. = 0`，*不能*从源码推断 bank conflict，metric 说没有。

*c) masked：`issued/32 = 26.3`，`pred_on/32 = 25.9`*。issued − pred_on 仅 0.4——`if (mask==0) continue` 是 predication，predicated-off lane 占 issue slot 但不做 exp 更新；`mem stall = 9.71` 是全表最高（额外 mask 读 + MUFU），kernel 太短不足以解读为 memory-bound。

#figure(
  warp-lanes(active: range(26), cell: 0.32,
             title: "masked：平均 pred_on/32 = 25.9，约 26 条 lane 在做有效 work"),
  caption: [绿色 = predicated-on lane；灰色 = predicated-off 或 idle。gap 在 issued − pred_on，不是分支 divergence。],
)

*d) causal：`issued/32 = 22.7`，`pred_on/32 = 22.4`*。低于 32 因为 `col <= row` 和 `col > row` 写 0 都是 predicated 路径；`mem stall = 1.70` 低于 masked，和更小矩阵 + 约半量扫描一致。

*无信息或为零的 metric：*

- `smem conf.`：除 block 外均为 0（无 smem）；block 也是 0——本规模下 sequential tree 未累积到可测 bank conflict。
- `barrier stall`：naive / online / masked / causal 均为 0.00——单 thread 路径无 sync，*符合预期*。
- `HBM %`、`% peak`：全表 ≈ 0——小规模 L2 resident，*不要用来论输赢*。

#insight[
  softmax 优化 ladder 的第一步永远是*把行内规约并行化*（block / warp reduction）。在 grid 能喂饱 GPU 之前，讨论"少一遍 pass"几乎没有意义——online 252 μs 比 naive 120 μs 还慢就是证据。
]

#insight[
  online softmax 是 FlashAttention 的*数值前提*，不是 standalone 加速技巧。面试若被问"online 为什么比 naive 快"，*正确答案是它通常不快*；快的是 FA 把 merge 嵌进分块 matmul，省掉 $O(N^2)$ HBM round-trip。
]

== ncu 该看什么

```
ncu --set full --section SpeedOfLight ./build/03_softmax
```

关键 metric：

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：exp/div 多时有 compute 占比。
- `dram__bytes.sum.per_second`：对比 naive 3 pass vs block 1.5 pass（读 2 遍 + 写 1 遍）的 traffic 下降。
- `smsp__sass_thread_inst_executed_op_dadd_pred_on` vs `_fmul` / `_mufu`（MUFU = exp sin cos 单元）：确认 bottleneck 在 transcendental 还是 memory。
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`：行是否被 cache 命中（同一行重复读时 L2 命中率高，naive 仍浪费 bandwidth）。

本章实测（$64 times 257$，L2 resident）的直接读数：

- *naive vs block*：`time` 120.80 μs → 5.73 μs（21×）；`SM %` 0.9% → 36.6%。HBM 实测 GB/s 1 vs 13——L2 吃掉 3 pass 重复读，不能从这里得出"访存减半"。
- *online vs naive*：`time` 252.29 μs vs 120.80 μs（online 更慢）；Pass 少了，但 per-step exp 重标定让 MUFU 更忙——看 `smsp__sass_thread_inst_executed_op_mufu` 占比，online 往往*更高*而非更低。
- *grid 列*：`launch__grid_size` naive/online/masked/causal = 1，block = 64。若 `SM %` < 5%，先查 grid 是不是只有 1 block，再查算法。

放大规模后重跑：`ncu -k regex:softmax --launch-count 3 ./build/03_softmax`，把 rows 改到 $8192+$、cols 改到 $4096+$，HBM % 才会上来，dram 对比才有意义。

== 面试白板 code

面试官说"手写一个 softmax"——不要写三 pass naive 版（会被追问怎么合并）。直接写 online softmax（单 pass，也是 flash-attention 的核心 primitive）：

```cpp
// 每 block 处理一 row (长度 N).
__global__ void softmax_online(const float* x, float* y, int N) {
  int row = blockIdx.x;
  const float* xr = x + row * N;
  float*       yr = y + row * N;

  // Pass 1: online 求 (m, s)，m = max(x)，s = sum exp(x - m).
  // 每 thread 先本地维护 (m, s)，然后 block reduce 合并。
  float m = -FLT_MAX, s = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float xj = xr[j];
    float new_m = fmaxf(m, xj);
    // 关键更新公式：换 m 时 s 需要 rescale.
    s = s * expf(m - new_m) + expf(xj - new_m);
    m = new_m;
  }
  // block-reduce (m, s)：两 warp 合并时用 combine 公式.
  //   m* = max(m_a, m_b);  s* = s_a * exp(m_a - m*) + s_b * exp(m_b - m*).
  __shared__ float sh_m[32], sh_s[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float om = __shfl_down_sync(0xffffffff, m, off);
    float os = __shfl_down_sync(0xffffffff, s, off);
    float nm = fmaxf(m, om);
    s = s * expf(m - nm) + os * expf(om - nm);
    m = nm;
  }
  if (lane == 0) { sh_m[wid] = m; sh_s[wid] = s; }
  __syncthreads();
  if (wid == 0) {
    int nw = (blockDim.x + 31) >> 5;
    m = (lane < nw) ? sh_m[lane] : -FLT_MAX;
    s = (lane < nw) ? sh_s[lane] : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float om = __shfl_down_sync(0xffffffff, m, off);
      float os = __shfl_down_sync(0xffffffff, s, off);
      float nm = fmaxf(m, om);
      s = s * expf(m - nm) + os * expf(om - nm);
      m = nm;
    }
    if (lane == 0) { sh_m[0] = m; sh_s[0] = s; }
  }
  __syncthreads();
  float M = sh_m[0], S = sh_s[0];

  // Pass 2: 写归一化结果. 只用读 M, S——不用读回中间 exp.
  float inv_S = 1.f / S;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    yr[j] = expf(xr[j] - M) * inv_S;
  }
}

// ==== Launch config ====
// gridDim  = B (batch × head 或 num rows)：每 block 一整 row.
// blockDim: 根据 N 选，让每 lane 处理 4-16 个元素——不要一 thread 一元素.
//   * N <= 1024: block = min(N, 128); N 小时用 warp shuffle 就够, 128 (4 warp) 减少 barrier;
//   * 1024 < N <= 8192: block = 256;  (每 lane 处理 4-32 元素)
//   * N > 8192: block = 512 或 1024，视寄存器压力；再大就要 tile over N (flash-attention 场景).
// 用几个 warp 影响 smem 大小 (sh_m[warps], sh_s[warps])——所以 warp 数最好 <= 32.
int block = (N <= 1024) ? 128 : (N <= 8192 ? 256 : 512);
softmax_online<<<B, block>>>(x, y, N);
```

*核心考点*（追问顺序）：

- *"为什么 online 只要一 pass？三 pass naive 哪三 pass？"* → naive: (1) max, (2) sum exp, (3) 归一化。online 把 max 和 sum 合成一个 pass，靠 rescale 公式 $s' = s dot e^(m - m') + e^(x - m')$。仍需 pass 2 写结果（可省 exp 中间存储）。
- *"为什么要减 max？"* → `expf(x)` 在 x > 88 溢出 FP32、x > 11 溢出 FP16。减 max 保证 exponent $<= 0$，`exp` 值域 $(0, 1]$。
- *"(m, s) 怎么合并？"* → 白板写清楚 `combine((m_a, s_a), (m_b, s_b)) = (max(m_a, m_b), s_a * exp(m_a - m*) + s_b * exp(m_b - m*))`——这是 flash-attention 的核心 primitive。
- *"$N$ 极大装不下一 row 怎么办？"* → tile over $N$，两个 tile 之间用 combine 公式合并。这就是 flash-attention 的 K/V tile 循环。
- *"backward？"* → $partial L / partial x_i = y_i (partial L / partial y_i - sum_j y_j partial L / partial y_j)$。fused forward + backward 时把 $y$ 重新算（recomputation）比存下来更省内存。
- *"为什么每 block 一 row 而不是几 block 一 row？"* → row 之间独立、无需通信，用 gridDim 天然并行。一 row 一 block 让 max/sum reduction 走 warp shuffle + smem，无跨 block barrier；如果一 row 跨多 block、就得走 grid-level 同步或多 kernel stage、复杂度骤增。只有 $N$ 极大（$>=$ 32K）到 register/smem 装不下时，才考虑 tile over $N$——那就是 flash-attention 路数了。

== 面试考点

#interview[
  *Q1*: 为什么 softmax 要 subtract max？证明等价性。

  A: 令 $m = max(x)$，分子分母同乘 $exp(m)$，$exp(x_i - m)/sum exp(x_j - m) = exp(x_i)/sum exp(x_j)$。数值上最大项 exp 为 1，防 overflow，分母 $>= 1$ 防除零。
]

#interview[
  *Q2*: online softmax 的更新公式是什么？为什么正确？

  A: $m' = max(m, x_n)$，$d' = d dot exp(m - m') + exp(x_n - m')$。正确性：把旧 sum 从基准 $m$ 重标定到新基准 $m'$，再加新项。两个 chunk $(m_A,d_A)$ 和 $(m_B,d_B)$ 可 merge：$m = max(m_A,m_B)$，$d = d_A exp(m_A - m) + d_B exp(m_B - m)$——FlashAttention 核心。
]

#interview[
  *Q3*: naive softmax GPU 版为什么慢？

  A: (a) 一行一 thread，并行度低；(b) 同一行读 3 遍 HBM；(c) exp 算 2 遍。不是算法错，是并行模型和访存模式错。
]

#interview[
  *Q4*: block softmax 里为什么两次 reduction 不能合成一次？

  A: sum 依赖 max 的结果（要先 $x - m$ 再 exp）。max reduction 必须完成后才能算 exp sum。block 内 thread 处理的是 stride 列，不能用 online 前缀语义。
]

#interview[
  *Q5*: warp-per-row softmax 怎么做两次 reduction？

  A: 每 lane 持一列，shuffle down 做 max（5 步）；各 lane 算 exp，再 shuffle down 做 sum；lane 写回。零 smem，$C <= 32$ 最优。
]

#interview[
  *Q6*: log-softmax 为什么比 `log(softmax(x))` 好？

  A: log-softmax 用 log-sum-exp：$x_i - m - log d$，避免 $y_i arrow.r 0$ 时 log 下溢、有效位损失。训练 cross-entropy 直接用这个形式。
]

#interview[
  *Q7*: safe softmax 和 approx softmax 区别？masked 全 -inf 行怎么办？

  A: safe 用 subtract-max 精确 FP32；approx 用表/多项式换速度。全 mask 行 $m = -oo, d = 0$，要特判输出全 0 或 skip，不能除零。
]

#interview[
  *Q8*: online softmax 和 FlashAttention 的关系？

  A: FlashAttention 分 tile 算 $Q K^T$，每 tile 产生局部 $(m, d)$，用 online merge 公式合并，Never materialize 完整 $N times N$ 矩阵。第 8 章会在此基础上加 SRAM tiling。
]
