#import "../template.typ": *

= 面试高频算子优化题目

本书正文覆盖了 vector add / reduce / softmax / matmul / layernorm / mlp / attention / flash-attention 八类。面试中还有一批高频题目正文没展开，本附录列出其中 *15 道最常问的*，每道给出：

- *问题定义*（形状、约束）
- *ladder*（naive → 最优的关键步骤）
- *核心考点*（面试官会追问什么）
- *参考实测*（可预期的 HBM% / 加速比数量级——写代码验证再上机）

规则：面试官问一道题，你的第一句话不应该是 "写代码"，应该是 *"这是 memory-bound 还是 compute-bound？"* + *"访问模式是不是 coalesced？"* 然后再看要不要 tile / share / shuffle / async。

== 1. transpose (matrix transpose)

*问题*：给定 $M times N$ 矩阵 $A$，计算 $B = A^T$。

*为什么难*：读 $A$ 是 coalesced（沿行），写 $B$ 就是 strided（跨列），或反之——*读写不能同时 coalesced*。

*Ladder*：

+ *Naive*：`B[j, i] = A[i, j]`。读 coalesced、写 strided → 写变 32 独立 transaction，掉到 $1/32$ 带宽。
+ *Shared memory tile*：block 拷贝 32×32 tile 到 smem，改成 `smem[threadIdx.y, threadIdx.x] = A[i, j]`（读 coalesced），`__syncthreads`，然后 `B[j', i'] = smem[threadIdx.x, threadIdx.y]`（写 coalesced）。
+ *Padded smem*（关键）：`__shared__ float smem[32][33]`（第二维 +1）。原因：`smem[threadIdx.x, threadIdx.y]`——warp 内 32 个 lane 沿列访问，如果第二维 = 32 会全 32-way bank conflict。加 1 让每行错开 1 个 bank。

*参考代码*（tiled + padded，面试白板版）：

```cpp
constexpr int TILE = 32;

__global__ void transpose_tiled(const float* A, float* B, int M, int N) {
  // +1 padding 打散 bank conflict——smem 每行 33 个 float，
  // lane 沿列访问时 addr % 32 = (row * 33 + col) % 32 全不同。
  __shared__ float smem[TILE][TILE + 1];

  int row_in = blockIdx.y * TILE + threadIdx.y;  // A 行
  int col_in = blockIdx.x * TILE + threadIdx.x;  // A 列
  if (row_in < M && col_in < N) {
    // 读 A：warp 内 lane 沿 col_in（threadIdx.x）连续 → coalesced。
    smem[threadIdx.y][threadIdx.x] = A[row_in * N + col_in];
  }
  __syncthreads();

  int row_out = blockIdx.x * TILE + threadIdx.y;  // B 行 = A 列所在 tile
  int col_out = blockIdx.y * TILE + threadIdx.x;  // B 列
  if (row_out < N && col_out < M) {
    // 写 B：threadIdx.x 沿 col_out 连续 → coalesced。
    // 读 smem 时 threadIdx.x 变第一维 → 沿列，靠 +1 padding 避冲突。
    B[row_out * M + col_out] = smem[threadIdx.x][threadIdx.y];
  }
}

// launch: dim3 grid((N+31)/32, (M+31)/32); dim3 block(32, 32);
```

*核心考点*：

- *"Naive transpose 为什么慢？"* → 写侧 strided，$1/32$ 带宽。ncu 证据：`l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` 会比 coalesced 高 ~32×。
- *"Padded smem 里的 +1 是什么？"* → bank conflict 消除。metric 证据：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 从数千降到 0。
- *"能不能用 swizzle 代替 padding？"* → 可以。padding 浪费 3% smem，swizzle 是异或索引（`col ^= (row & 0x1f)`），无空间开销，Hopper 上有硬件 swizzle 模式。

*参考实测*（A100，$M = N = 4096$）：naive ~$1 / 30$ peak；tiled+padded ~$0.75$ peak。

== 2. reduce max / argmax

*问题*：给数组 $x$，求 $max_i x_i$，或返回 $arg max_i x_i$。

*和 reduce sum 的关键差别*：max 无逆运算，*不能用 atomicMax + subtract-max trick*；argmax 需要*同时*传递 index 和 value——warp shuffle 一次 shuffle 只搬 32-bit，argmax 要么用 int64 打包（value + index），要么两次 shuffle。

*Ladder*：sum 版的 ladder 全套复用（atomic → shared tree → warp shuffle → chunked hierarchical），加两个 argmax 特殊技巧：

- *打包 index 到低位*：`packed = (value_bits << 32) | index`（value 是 int/float bit pattern，index 是 uint32）→ `atomicMax` 一次搞定。float 需要处理负数 bit pattern 单调性（把符号位翻转再比较）。
- *warp shuffle 里传两个值*：两次 `__shfl_down_sync`，先 shuffle value 再 shuffle 对应 index；或用 CUB `WarpReduce<KeyValuePair>`。

*参考代码*（warp-level argmax，同时 shuffle value 和 index）：

```cpp
struct MaxIdx { float v; int i; };

__device__ MaxIdx warp_argmax(MaxIdx x) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(0xffffffff, x.v, offset);
    int   other_i = __shfl_down_sync(0xffffffff, x.i, offset);
    // Tie break: 取更小 index（reproducibility）。
    if (other_v > x.v || (other_v == x.v && other_i < x.i)) {
      x.v = other_v;
      x.i = other_i;
    }
  }
  return x;  // lane 0 持有 warp 内 argmax
}

__global__ void argmax_kernel(const float* x, int N, MaxIdx* out) {
  MaxIdx v{-FLT_MAX, -1};
  // grid-stride load，每 thread 先本地累计一个 argmax
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N; i += gridDim.x * blockDim.x) {
    if (x[i] > v.v) v = {x[i], i};
  }
  v = warp_argmax(v);

  __shared__ MaxIdx warp_maxes[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  if (lane == 0) warp_maxes[wid] = v;
  __syncthreads();

  if (wid == 0) {
    v = (lane < blockDim.x / 32) ? warp_maxes[lane] : MaxIdx{-FLT_MAX, -1};
    v = warp_argmax(v);
    if (lane == 0) out[blockIdx.x] = v;  // 每 block 一个 partial argmax，第二 stage 再 reduce
  }
}
```

*核心考点*：

- *"Softmax 里的 reduce max 为什么可以不用 atomic？"* → 单 row 内 reduction，用 block 级 tree + warp shuffle 就够，不需要跨 block 通信。
- *"argmax tie breaking？"* → 定义决定：取最小 index 或最大 index。打包时 index 放高位或低位，配合 max 顺序。
- *"FP16 argmax 有什么坑？"* → 相同 value 的 bit pattern 唯一，但两个不同 float 可能相等（例如 subnormal），tie breaking 必须显式。

== 3. cumulative sum / prefix scan

*问题*：给数组 $x[0..N)$，计算 $y[i] = sum_(j <= i) x[j]$（inclusive）或 $y[i] = sum_(j < i) x[j]$（exclusive）。

*为什么难*：*天然串行*。每个输出依赖前一个 → 不能像 reduce 那样"随便配对"。

*Ladder*：

+ *串行*：单 thread 循环，$O(N)$ 但零并行度。
+ *Hillis-Steele scan*（single-pass tree）：每一步 `y[i] += y[i - stride]`，$O(N log N)$ FLOP、$O(log N)$ step。适合 warp 内（`__shfl_up_sync` 直接实现）。
+ *Blelloch scan*（up-sweep + down-sweep）：$O(N)$ FLOP、$O(log N)$ step。适合 block/grid 级。
+ *Grid 级 scan + look-back*（CUB `DeviceScan`）：block 间无需 barrier，通过 partial value 广播 + look-back 完成——现代最优。

*参考代码*（block-level inclusive scan：warp scan + inter-warp scan）：

```cpp
__device__ float warp_inclusive_scan(float v) {
  int lane = threadIdx.x & 31;
  #pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    float t = __shfl_up_sync(0xffffffff, v, s);
    if (lane >= s) v += t;
  }
  return v;
}

__global__ void block_scan_kernel(const float* in, float* out, int N) {
  __shared__ float warp_sums[32];  // 每 warp 的 tail

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (tid < N) ? in[tid] : 0.f;

  // Step 1: warp 内 inclusive scan.
  v = warp_inclusive_scan(v);

  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  if (lane == 31) warp_sums[wid] = v;  // 把每 warp 的 tail 存起来
  __syncthreads();

  // Step 2: warp 0 对 warp_sums 再做 scan（exclusive：不加自身）。
  if (wid == 0) {
    float w = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.f;
    w = warp_inclusive_scan(w);
    warp_sums[lane] = w;
  }
  __syncthreads();

  // Step 3: 每 warp 加上前面所有 warp 的 tail（exclusive prefix）。
  if (wid > 0) v += warp_sums[wid - 1];
  if (tid < N) out[tid] = v;
}
// grid 级 scan 需要再加 block partial + 第二 kernel look-back / device scan。
```

*核心考点*：

- *"exclusive vs inclusive？"* → inclusive 的结果右 shift 一位、第 0 位置 0 就是 exclusive。
- *"grid 级 scan 怎么并行 block 之间？"* → 每 block 输出 tail，用 CUB `DeviceScan` 的 decoupled look-back：每 block 发布自己的 aggregate、等前一 block 的 inclusive prefix 到达。
- *"数值稳定性？"* → 大数组用 Kahan compensated summation 或 pairwise reduction；scan 里较少见。

== 4. histogram

*问题*：给数组 $x[i] in [0, K)$，统计 `hist[k] = |{i : x[i] = k}|`。

*为什么难*：多 thread 更新同一个 bin → *atomic contention*。

*Ladder*：

+ *全局 atomic*：每 thread `atomicAdd(&hist[x[i]], 1)`。如果 distribution skewed（比如 90% 的 x 集中在少数 bin），contention 灾难，慢 100× 以上。
+ *Shared memory privatized*：每 block 一份 `__shared__ int local_hist[K]`，block 内 atomic（smem atomic 比 global atomic 快 20-50×），结束时把 `local_hist` merge 到 global `hist`（一次 atomic per bin per block，contention 大幅降低）。
+ *K 大时只能 sample subset 用 smem*：K = 1M 时 smem 装不下，退回 subset caching + spill。
+ *Match+CTA reduction*（更高级）：warp 内 `__ballot_sync` / `__match_any_sync` 判断哪些 lane 打同一个 bin，先在 warp 内 reduce，减少 atomic 次数。

*参考代码*（privatized shared memory histogram）：

```cpp
constexpr int K = 256;  // bin 数

__global__ void histogram_privatized(const int* x, int N, int* hist) {
  __shared__ int local_hist[K];

  // Step 1: 清空 local_hist（block 内 stride 清）。
  for (int k = threadIdx.x; k < K; k += blockDim.x) local_hist[k] = 0;
  __syncthreads();

  // Step 2: block 内 grid-stride 打 smem atomic（smem atomic 快得多）。
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N; i += gridDim.x * blockDim.x) {
    atomicAdd(&local_hist[x[i]], 1);
  }
  __syncthreads();

  // Step 3: 把 local_hist merge 到 global（每 bin 一次 global atomic）。
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    if (local_hist[k] > 0) atomicAdd(&hist[k], local_hist[k]);
  }
}
```

*核心考点*：

- *"为什么 smem atomic 快 20-50×？"* → smem atomic 走片上 lock，$\sim$4 cycle；global atomic 走 L2，延迟 200+ cycle。
- *"如果输入是 sorted 的呢？"* → 每 block 处理连续段，几乎无 contention。sort-then-histogram 有时比直接 histogram 快。
- *"K 很大（比如 float 分桶）呢？"* → 先 quantize / hash 到 fixed K，再走标准 pipeline。
- *"contention 怎么定量看？"* → ncu `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` 会非常高（atomic 反复等 L2）。

== 5. top-k

*问题*：给数组 $x[0..N)$，返回最大的 $k$ 个（value 和/或 index）。$k$ 通常 << $N$，比如 $N = 10^6, k = 32$。

*为什么难*：既不能全排序（$O(N log N)$）也不能纯 reduce（无满足要求的关联运算）。

*Ladder*：

+ *全排序 + 取前 k*：$O(N log N)$，可用 CUB `DeviceRadixSort`。$k << N$ 时浪费严重。
+ *Bitonic top-k*：warp/block 内 bitonic sort 前 $k$ + tournament，$O(N log^2 k)$。$k <= 32$ 时非常适合 warp 内 fully-in-register。
+ *Heap-based*（threshold-based）：每 thread 维护 size-$k$ heap；输入元素比 heap 顶大才插入。适合 stream 场景。
+ *Approximate top-k*：sample + threshold estimation → 单 pass 过滤 → 精确 top-k。生产用于 LLM logit filtering。

*参考代码*（warp-level top-32 via bitonic + shuffle，$k = 32$ 常见 case）：

```cpp
// warp 内每 lane 持有一个 (value, index)，做 5 层 bitonic sort。
__device__ void bitonic_compare_swap(float& v, int& i, int xor_lane, bool up) {
  float other_v = __shfl_xor_sync(0xffffffff, v, xor_lane);
  int   other_i = __shfl_xor_sync(0xffffffff, i, xor_lane);
  int   me      = threadIdx.x & 31;
  bool  me_low  = (me & xor_lane) == 0;  // 我在配对中的低位？
  // up=true 时低位保留小者、高位保留大者，反之反过来。
  bool keep_other = (me_low != up) ? (other_v < v) : (other_v > v);
  if (keep_other) { v = other_v; i = other_i; }
}

__device__ void warp_bitonic_sort_desc(float& v, int& i) {
  // 标准 5 层 bitonic：size = 2, 4, 8, 16, 32.
  for (int size = 2; size <= 32; size <<= 1) {
    bool up = ((threadIdx.x & size) == 0);  // 交替升/降构造 bitonic
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      bitonic_compare_swap(v, i, stride, up);
    }
  }
  // 最终做一次全 descending 的 merge:
  for (int stride = 16; stride > 0; stride >>= 1) {
    bitonic_compare_swap(v, i, stride, false);
  }
}

__global__ void warp_top32(const float* x, int N, float* out_v, int* out_i) {
  int lane = threadIdx.x & 31;
  float v = -FLT_MAX; int i = -1;

  // 每 lane 扫自己那一路，本地维护单个 max（k=1 版），最后 warp 排 32 个。
  // 更严谨版：每 lane 维护一个 size-k/32 heap。这里简化到 k=32、每 lane 拿 1 个。
  for (int t = blockIdx.x * 32 + lane; t < N; t += gridDim.x * 32) {
    if (x[t] > v) { v = x[t]; i = t; }
  }
  warp_bitonic_sort_desc(v, i);
  out_v[blockIdx.x * 32 + lane] = v;
  out_i[blockIdx.x * 32 + lane] = i;
}
```

*核心考点*：

- *"attention softmax 前的 top-k logit filtering 怎么做？"* → 逐 row，$N$ = vocab_size (几万)，$k$ = 32-512；bitonic 或 approx。
- *"bitonic 为什么要交替 up/down？"* → 构造 bitonic sequence（先增后减）才能一次 merge 全排。
- *"top-k 和 sort 的区别？"* → top-k 只保证前 $k$ 正确，不 care 内部顺序，可以更快。

== 6. layernorm backward

*问题*：给定 $y = gamma dot hat(x) + beta$（forward），backward 输入 $partial L / partial y$，输出 $partial L / partial x$、$partial L / partial gamma$、$partial L / partial beta$。

*核心公式*：

$ frac(partial L, partial x_i) = frac(1, sqrt(sigma^2 + epsilon)) [frac(partial L, partial hat(x)_i) - frac(1, H) sum_j frac(partial L, partial hat(x)_j) - hat(x)_i dot frac(1, H) sum_j frac(partial L, partial hat(x)_j) dot hat(x)_j] $

其中 $partial L / partial hat(x)_j = partial L / partial y_j dot gamma_j$。

*为什么难*：
- 每 row 有 *两个* 全 row 的 reduction（sum of $partial L / partial hat(x)$、sum of $partial L / partial hat(x) dot hat(x)$）。
- $partial L / partial gamma$、$partial L / partial beta$ 是*跨 row* 的 reduction（sum over batch）。

*Ladder*：

+ *Naive*：三 pass（先两 reduction，再逐元素计算），每 pass 一次 kernel launch。
+ *Fused row backward*：block-per-row，一次 kernel 完成两 reduction + 逐元素，用 shared memory tree 或 warp shuffle。
+ *Fused param grad*：$partial L / partial gamma$、$partial L / partial beta$ 需要 sum over batch → 用 grid-level partial sum + 第二 stage reduce（类似 reduce sum ladder）。
+ *保存 $mu$、$sigma$ 到 forward*：forward pass 存 $"rstd" = 1/sqrt(sigma^2 + epsilon)$ 到 buffer，backward 直接读，省一次 reduction。

*参考代码*（fused row backward for $partial L / partial x$，block-per-row）：

```cpp
// forward 时已存好 mean[row], rstd[row] = 1 / sqrt(var + eps)
__global__ void layernorm_backward_dx(
    const float* dy, const float* x, const float* gamma,
    const float* mean, const float* rstd,
    float* dx, int H) {
  int row = blockIdx.x;
  const float* xr  = x  + row * H;
  const float* dyr = dy + row * H;
  float*       dxr = dx + row * H;
  float mu   = mean[row];
  float rstd_r = rstd[row];

  // Step 1: 两个 row-level reduction:
  //   s1 = sum_j (dy_j * gamma_j)
  //   s2 = sum_j (dy_j * gamma_j) * xhat_j    其中 xhat_j = (x_j - mu) * rstd
  float s1 = 0.f, s2 = 0.f;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float dyhat = dyr[j] * gamma[j];
    float xhat  = (xr[j] - mu) * rstd_r;
    s1 += dyhat;
    s2 += dyhat * xhat;
  }
  s1 = block_reduce_sum(s1);  // 用前面章节的 block reduce
  s2 = block_reduce_sum(s2);
  __shared__ float ss1, ss2;
  if (threadIdx.x == 0) { ss1 = s1 / H; ss2 = s2 / H; }
  __syncthreads();

  // Step 2: 逐元素套公式.
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float dyhat = dyr[j] * gamma[j];
    float xhat  = (xr[j] - mu) * rstd_r;
    dxr[j] = rstd_r * (dyhat - ss1 - xhat * ss2);
  }
}
```

*核心考点*：

- *"forward 存 $mu, "rstd"$ 还是 $mu, sigma^2$？"* → 存 $"rstd"$，backward 直接乘，省一次 sqrt。
- *"$partial L / partial gamma$ 的 batch reduction 怎么并行？"* → 每 block 处理一段 batch × H → 输出 partial gamma[H]，第二 stage kernel reduce partials。
- *"能不能一次 kernel 全做完？"* → 数值上可以（每 block 处理一个 batch × H tile，atomic 到 gamma/beta grad），实践上 atomic 慢，分 stage 更快。

== 7. cross-entropy loss (fused softmax + NLL)

*问题*：给 logits $ell in RR^(B times V)$、labels $y in {0, ..., V-1}^B$，计算 $L = -1/B sum_b log ("softmax"(ell_b))_(y_b)$，同时输出 $partial L / partial ell$。

*为什么难*：
- 数值稳定：`log(softmax(x)) = x - logsumexp(x)`，避免 `log(exp(...))`。
- forward + backward 常一起做（rematerialize activation）。
- $V$ 大到 32K-100K（LLM vocab）→ 每 row 就是大 reduction。

*Ladder*：

+ *Naive 三 kernel*：softmax → gather label → NLL → backward exchange chunks。
+ *Fused forward*：单 kernel，per-row: `max` + `sum_exp` + `log_sum_exp` → $L$ + $partial L / partial ell_i = "softmax"(ell)_i - [i = y]$（就是 softmax 输出减 one-hot）。
+ *Online logsumexp*（和 online softmax 同结构）：单 pass 完成，避免存 softmax 中间结果。
+ *Chunk over vocab*：$V$ 极大时，一 row 的 logits 装不下寄存器，需要 tile over vocab dim。

*参考代码*（fused forward + grad，block-per-row，online logsumexp）：

```cpp
__global__ void cross_entropy_fused(
    const float* logits,   // [B, V]
    const int*   labels,   // [B]
    float*       loss,     // [B]
    float*       grad,     // [B, V]  = softmax - onehot(label)
    int V, float inv_B) {
  int b = blockIdx.x;
  const float* row = logits + b * V;
  float*       gr  = grad   + b * V;
  int label = labels[b];

  // Pass 1: online reduce over V for (m, s) where m = max, s = sum exp(x - m).
  float m = -FLT_MAX, s = 0.f;
  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    float xj = row[j];
    float new_m = fmaxf(m, xj);
    s = s * expf(m - new_m) + expf(xj - new_m);
    m = new_m;
  }
  // block reduce (m, s) 合并成一个 (m*, s*)——见 softmax 章节。
  block_reduce_online_softmax(m, s);

  float lse = m + logf(s);  // logsumexp
  if (threadIdx.x == 0) {
    loss[b] = (lse - row[label]) * inv_B;
  }

  // Pass 2: 写 grad = softmax - onehot(label)，并且乘 1/B（loss 平均）。
  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    float p = expf(row[j] - lse);
    float g = p - (j == label ? 1.f : 0.f);
    gr[j] = g * inv_B;
  }
}
```

*核心考点*：

- *"backward 为什么就是 softmax − onehot？"* → 直接对 $L = -log("softmax")_y$ 求导，交叉熵和 softmax 的耦合让 grad 变得极简。
- *"$partial L / partial ell$ 需要保留完整 softmax 吗？"* → 不需要。fused backward 里重新计算 softmax（recomputation），只花 $O(V)$ 内存开销。
- *"label smoothing 怎么改？"* → target 从 one-hot 变为 smoothed distribution，backward 变为 `softmax − smoothed_target`。

== 8. GELU / SwiGLU activation + backward

*问题*：GELU 有两个近似公式：

$ "GELU"(x) = x dot Phi(x) approx 0.5 x [1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))] $

SwiGLU 用于 LLaMA/Gemini：

$ "SwiGLU"(x, y) = "silu"(x) dot y = frac(x, 1 + e^(-x)) dot y $

*Ladder*：

+ *Vanilla element-wise*：一 kernel，就是标准 vector-like，naive 就近 peak（vector-add 定律）。
+ *Fused epilogue*：把 GELU / SwiGLU 融合到 Linear 的 epilogue（cuBLASLt / CUTLASS），省一次 read/write（output 4B 保存到 smem 后 in-place 应用）。
+ *Fused forward + backward stash*：训练时 backward 需要 $x$ 或 $"silu"(x)$ → forward 存最合适的中间量（存 $"silu"(x)$，backward 用 `silu * (1 - silu) * x + silu` 之类）。

*参考代码*（GELU tanh 近似 + SwiGLU forward，向量化 4 元素）：

```cpp
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
  constexpr float k1 = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void gelu_kernel(const float* in, float* out, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // float4 向量化——element-wise 天然 memory-bound，v = 4 拉满带宽.
  for (int i = tid; i * 4 < N; i += gridDim.x * blockDim.x) {
    float4 x = reinterpret_cast<const float4*>(in)[i];
    float4 y;
    y.x = gelu_tanh(x.x); y.y = gelu_tanh(x.y);
    y.z = gelu_tanh(x.z); y.w = gelu_tanh(x.w);
    reinterpret_cast<float4*>(out)[i] = y;
  }
}

// SwiGLU: LLaMA FFN 用 gate 和 up 两路投影，然后 silu(gate) * up.
__global__ void swiglu_kernel(const float* gate, const float* up,
                              float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float g = gate[i];
  float sig_g = 1.f / (1.f + expf(-g));  // sigmoid
  out[i] = (g * sig_g) * up[i];          // silu(g) * up
}
```

*核心考点*：

- *"tanh 版 GELU 和 erf 版 GELU 精度差多少？"* → tanh 版误差 $< 10^(-4)$，训练时无差别；tanh + polynomial 比 erf 快 3×。
- *"SwiGLU 的 backward？"* → 两个输入分别求导：$partial L / partial x = partial L / partial "out" dot y dot "silu"'(x)$，$partial L / partial y = partial L / partial "out" dot "silu"(x)$；`silu'(x) = sig(x) + x*sig(x)*(1-sig(x))`。
- *"activation function 的性能瓶颈？"* → 永远是 memory-bound。fuse 到 matmul epilogue 是唯一有意义的优化。

== 9. RMSNorm

*问题*：LayerNorm 的简化版（LLaMA 用）：

$ "RMS"(x) = sqrt(1/H sum_i x_i^2), quad y_i = frac(x_i, "RMS"(x)) dot gamma_i $

*和 LayerNorm 的差别*：*不减 mean*、*不加 beta*，只做 rescale。

*Ladder*：几乎与 LayerNorm 相同，简化：

+ 只需要 *一次* row reduction（sum of $x^2$），而 LayerNorm 需要两次（mean, var）。
+ 反向可以更简洁：$partial L / partial x_i = frac(1, "RMS") [partial L / partial y_i - x_i dot frac(1, H) frac(1, "RMS"^2) sum_j x_j dot partial L / partial y_j]$。

*参考代码*（block-per-row，一次 reduction）：

```cpp
__global__ void rmsnorm_kernel(const float* x, const float* gamma,
                               float* y, int H, float eps) {
  int row = blockIdx.x;
  const float* xr = x + row * H;
  float*       yr = y + row * H;

  // Pass 1: sum of squares.
  float sq = 0.f;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float v = xr[j];
    sq += v * v;
  }
  sq = block_reduce_sum(sq);

  __shared__ float rrms;
  if (threadIdx.x == 0) rrms = rsqrtf(sq / H + eps);  // 1 / RMS
  __syncthreads();

  // Pass 2: normalize + scale by gamma（无 beta！）
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    yr[j] = xr[j] * rrms * gamma[j];
  }
}
```

*核心考点*：

- *"RMSNorm 为什么比 LayerNorm 快？"* → 少一次 reduction（不算 mean）、少一个可学习参数 $beta$。实测 forward 快 ~10%，训练稳定性差不多（LLaMA 论文验证过）。
- *"为什么 LLaMA 用 RMSNorm？"* → 训练稳定性 + 推理速度。
- *"`rsqrtf` 和 `1/sqrtf` 有区别吗？"* → `rsqrtf` 是硬件 SFU 指令，1 cycle；`1.f/sqrtf` 会被 nvcc 融合成 `rsqrtf`（`-use_fast_math` 下），但显式写更清楚。

== 10. rotary position embedding (RoPE)

*问题*：给 query/key $Q, K in RR^(N times d)$，把每 pair 相邻通道 $(Q_(2j), Q_(2j+1))$ 视为复数，旋转角度 $theta_(i, j) = i dot 10000^(-2j/d)$。

$ Q'_(i, 2j) = Q_(i, 2j) cos theta_(i,j) - Q_(i, 2j+1) sin theta_(i,j) $
$ Q'_(i, 2j+1) = Q_(i, 2j) sin theta_(i,j) + Q_(i, 2j+1) cos theta_(i,j) $

*Ladder*：

+ *Naive 独立 kernel*：读 Q, 计算 cos/sin, 写 Q'.
+ *Fused into attention*：RoPE 通常*不单独*做——直接在 attention kernel 里，load Q/K 时应用 rotation，省一次 read/write。Flash-Attention 3 就这么做。
+ *cos/sin 预计算 vs on-the-fly*：cos/sin 表大小 $N times d$，预计算需要 cache；on-the-fly `__sincosf` 单周期 4 cycle，通常 fuse 更好。

*参考代码*（on-the-fly RoPE，处理 $Q$，pair-wise 旋转）：

```cpp
// Q shape [N, d]，每 pair (2j, 2j+1) 视为复数、旋转角 theta = pos * base^(-2j/d).
__global__ void rope_kernel(float* Q, int N, int d, float base) {
  int pos = blockIdx.x;                // token 位置 i
  int tid = threadIdx.x;               // 每 thread 处理一对 (2j, 2j+1)
  if (tid * 2 >= d) return;

  int j = tid;
  float* q = Q + pos * d;

  // 频率：base^(-2j/d) = exp(-2j/d * log(base))
  float freq = expf(-(2.f * j / (float)d) * logf(base));
  float angle = pos * freq;
  float c, s;
  __sincosf(angle, &s, &c);  // 硬件 SFU 一次算 sin+cos

  float q0 = q[2 * j];
  float q1 = q[2 * j + 1];
  q[2 * j]     = q0 * c - q1 * s;
  q[2 * j + 1] = q0 * s + q1 * c;
}
// launch: rope_kernel<<<N, d/2>>>(...) —— 每 token 一个 block，每 pair 一个 thread。
// 生产版会 fuse 到 attention 的 Q/K load 中，永不 materialize 中间 Q'.
```

*核心考点*：

- *"RoPE 为什么不用绝对位置？"* → 相对位置 = $theta_(i, j) - theta_(k, j)$，attention $q dot k$ 自动携带相对位置。
- *"long context extrapolation 怎么调？"* → NTK-aware scaling / YaRN，本质是调 base（默认 10000）或对高频通道特殊处理。
- *"RoPE 是 memory-bound 还是 compute-bound？"* → memory-bound（每元素几 flop）。不 fuse 单独 kernel 是浪费。

== 11. paged attention (vLLM)

*问题*：在 LLM inference 时，KV cache 巨大（每 token 数 KB），batch 里多请求的 cache 长度不同 → 传统 contiguous KV 浪费严重。paged attention 把 KV cache 分块，每 block 16-32 token，用 *block table* 索引到物理页——像 OS 的虚拟内存。

*Ladder*：

+ *Naive*：contiguous KV, padding 到 max_len → GPU 内存爆炸。
+ *Continuous batching*：请求动态加入/退出，重排 KV，内存碎片严重。
+ *Paged*：block-based 分配，block table 描述 logical seq → physical block。

*参考代码*（paged attention 的 K/V load，简化到 single-head 单 request）：

```cpp
// KV cache 布局: [num_blocks, block_size, head_dim]
// block_table[seq, logical_block_idx] = physical_block_idx
constexpr int BLOCK = 16;   // 每 physical block 16 个 token

__global__ void paged_attn_kernel(
    const float* Q,              // [head_dim]
    const float* KV_cache,       // [num_blocks, BLOCK, head_dim]  K 和 V 分别一份
    const int*   block_table,    // [max_logical_blocks]
    int seq_len, int head_dim,
    float* out) {
  int t = blockIdx.x * BLOCK + threadIdx.y;   // logical token id
  if (t >= seq_len) return;

  // Step 1: 通过 block table 查物理位置——这是 paged attention 的核心。
  int logical_block = t / BLOCK;
  int within_block  = t % BLOCK;
  int phys_block    = block_table[logical_block];
  const float* K_ptr = KV_cache + (phys_block * BLOCK + within_block) * head_dim;

  // Step 2: block 内每个 token 由 blockDim.x 个 lane 协作算 dot(Q, K_t).
  // 因为 within_block 连续、head_dim 是最内维 → warp 内 lane 沿 head_dim 走 coalesced.
  float partial = 0.f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    partial += Q[d] * K_ptr[d];
  }
  partial = warp_reduce_sum(partial);
  if (threadIdx.x == 0) out[t] = partial;  // 后续再做 softmax * V
}
```

*核心考点*：

- *"paged attention kernel 里怎么读 K/V？"* → 每 token 通过 block_table 间接查找。global memory access 变成 *gather*：block index → physical KV pointer → load.
- *"gather 会破坏 coalesced？"* → 会。paged attention 通过让 block size $= 16$ tokens、每 block 内 contiguous，保证一个 block 内的读是 coalesced；跨 block 才是 gather。
- *"share prefix caching？"* → 多个请求共享 system prompt 前缀，block table 里共享同一批 physical block。

== 12. quantized matmul (INT8 / INT4)

*问题*：$A in "int"8^(M times K), B in "int"8^(K times N)$，$s_A in RR^M, s_B in RR^N$ 是 per-row / per-col scale。计算 $C_(i j) = s_A[i] dot s_B[j] dot sum_k A[i, k] dot B[k, j]$。

*为什么难*：
- 数据类型混合：INT8 mul → INT32 accum → FP16 rescale → FP16 output。
- Tensor Core 有 INT8 mode（A100: 624 TFLOPS，2× FP16）。
- Scale broadcast 和 dequant 可能出现 bank conflict / lane divergence。

*Ladder*：

+ *Naive INT8 mul + FP32 accum*：手写 FMA，慢。
+ *Tensor Core INT8*：用 `wmma::mma_sync` 的 `int8_t` 模式，或 CUTLASS `Gemm<int8_t, int8_t, int32_t>`。
+ *Fused dequant epilogue*：INT32 accum 后立即乘 scale 转 FP16，避免存 INT32 中间结果。
+ *INT4 with LUT (GPTQ, AWQ)*：weight INT4 pack 到 INT8/32，运行时 dequant via lookup。

*参考代码*（Tensor Core INT8 GEMM 的核心指令 + fused dequant epilogue）：

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 16x16x16 INT8 → INT32 accum tile.
// A, B are int8_t global memory; scale_A[M], scale_B[N] fp32.
__global__ void int8_gemm_wmma(
    const int8_t* A, const int8_t* B, __half* C,
    const float* scale_A, const float* scale_B,
    int M, int N, int K) {
  int warp_m = blockIdx.y * 16;
  int warp_n = blockIdx.x * 16;

  fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
  fragment<accumulator, 16, 16, 16, int> c_frag;
  fill_fragment(c_frag, 0);

  // K 维度 tile 循环
  for (int k = 0; k < K; k += 16) {
    load_matrix_sync(a_frag, A + warp_m * K + k, K);
    load_matrix_sync(b_frag, B + k * N + warp_n, N);
    mma_sync(c_frag, a_frag, b_frag, c_frag);  // INT8 × INT8, INT32 accum
  }

  // Fused dequant epilogue: INT32 accum → FP16 output, 乘 per-row/col scale.
  int lane = threadIdx.x & 31;
  #pragma unroll
  for (int e = 0; e < c_frag.num_elements; ++e) {
    int i = warp_m + (lane / 4) + (e / 4) * 8;   // wmma 元素布局（简化）
    int j = warp_n + (lane % 4) * 2 + (e % 2);
    float f = c_frag.x[e] * scale_A[i] * scale_B[j];
    // 立即转 FP16，避免存 INT32 中间
    // ...store_matrix_sync 或手动写...
  }
}
```

*核心考点*：

- *"per-tensor / per-channel / per-token scale 有什么区别？"* → per-tensor 一个 scalar，最粗；per-channel（B 的每列一个 scale）是 GEMM 标准；per-token（A 的每行一个 scale）适合 activation quant。
- *"INT8 GEMM 的 accum 为什么用 INT32 而不 FP32？"* → INT8 × INT8 = INT16，$K$ 累加 → INT16 × 1024 会溢出，需要 INT32。硬件直接支持 INT8 dot4 INT32 accum。
- *"outlier channel 怎么办？"* → SmoothQuant / GPTQ 把 outlier 挪到 weight 或用高精度 subchannel。

== 13. layer fusion (Linear + GELU + Dropout + Add)

*问题*：Transformer FFN 是 `y = Add(x, Dropout(Linear2(GELU(Linear1(x)))))`。分离 5 个 kernel 时，中间结果反复读写 → memory bound。

*Ladder*：

+ *Split*：5 个 kernel，读写 $x, "h1", "h2", "h3", "h4", y$ 6 份数据。
+ *Epilogue fusion（Linear1 + GELU）*：GEMM 结束时在 smem 上直接 GELU，输出到 h2 位置。cuBLASLt / CUTLASS 支持。
+ *Fully fused*（除 Linear1 的 GEMM 外）：Linear2 之后 + Dropout + Add + LayerNorm 全部 fuse 到一 kernel。Flash-Attention 系列的做法。
+ *Auto tuning*（CUTLASS / Triton）：不同 shape 选不同 tile / warp partition。

*参考代码*（Linear2 后的 Add + Dropout + LayerNorm 融合成 1 个 kernel）：

```cpp
// h = Linear2(...) 已经写到 h; 这里 fuse: y = LayerNorm(Dropout(h) + x_residual)
// 每 block 处理一 row (H 维).
__global__ void fused_dropout_add_ln(
    const float* h, const float* x_residual, const float* gamma, const float* beta,
    float* y, uint64_t rng_seed, uint64_t rng_offset,
    float dropout_p, float scale,  // scale = 1/(1-p)
    int H, float eps) {
  int row = blockIdx.x;
  const float* hr = h          + row * H;
  const float* xr = x_residual + row * H;
  float*       yr = y          + row * H;

  // Pass 1: dropout + add + accumulate (mean, mean_of_sq) via online algorithm.
  // 用 Philox 从 (seed, offset + row*H + j) 拿到确定性随机数——不用存 mask.
  float mean = 0.f, m2 = 0.f;
  int n = 0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float rand01 = philox_uniform(rng_seed, rng_offset + row * H + j);
    float mask = (rand01 >= dropout_p) ? scale : 0.f;
    float v = hr[j] * mask + xr[j];
    // Welford online update
    ++n;
    float delta = v - mean;
    mean += delta / n;
    m2 += delta * (v - mean);
  }
  // block reduce (mean, m2, n) 合并——Welford combine 公式.
  block_reduce_welford(mean, m2, n);
  float rstd = rsqrtf(m2 / H + eps);

  // Pass 2: 再算一次 dropout+add（重复 Philox，同 seed+offset 拿同 random），
  // 归一化 + affine 写出。省 O(H) 中间存储.
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float rand01 = philox_uniform(rng_seed, rng_offset + row * H + j);
    float mask = (rand01 >= dropout_p) ? scale : 0.f;
    float v = hr[j] * mask + xr[j];
    yr[j] = (v - mean) * rstd * gamma[j] + beta[j];
  }
}
```

*核心考点*：

- *"epilogue fusion 的边界在哪？"* → 只能 fuse *element-wise* + *broadcast*；不能 fuse 需要 reduce 的（比如 softmax）。
- *"Dropout 训练时的随机数生成？"* → Philox counter-based PRNG，每 element 独立 stateless，可 fuse——backward 用同一 (seed, offset) 重放，不用存 mask。
- *"fuse 太多有什么坏处？"* → 寄存器压力上升 → occupancy 下降；kernel 变复杂难调优；一个坏 kernel 拖后腿。

== 14. group / channel-wise conv (im2col vs implicit)

*问题*：给 input $"NCHW"$、weight $"KCRS"$，计算 conv output。

*Ladder*：

+ *Direct conv*：naive 7 层循环。
+ *im2col + GEMM*：input 展开成 matrix，转 matmul。$M = N times "H_out" times "W_out"$，$K = C times R times S$，$N = K_"out"$。缺点：im2col buffer 大（约 $R times S$ 倍 input）。
+ *Implicit GEMM*：不显式 im2col，在 GEMM kernel 里用 index 计算读原 input 的位置。cuDNN 默认。
+ *Winograd*（3×3 kernel）：Winograd F(2,3) 把 4-output-per-block 从 36 mul 减到 16 mul，加法多但 mul 少。
+ *FFT conv*（大 kernel）：kernel size $> 7$ 时 FFT 更快。

*参考代码*（implicit GEMM conv，NCHW，简化：stride=1、padding=1、3×3 kernel）：

```cpp
// input:  [N, C, H, W]
// weight: [K, C, R, S]  (R=S=3)
// output: [N, K, H, W]
// implicit GEMM: 把 output[n, k, oh, ow] 视为 GEMM 的 M×N 上一点，
//   M = N * H * W, N = K, GEMM K = C * R * S.
__global__ void implicit_gemm_conv3x3(
    const float* input, const float* weight, float* output,
    int N, int C, int H, int W, int K) {
  int n_ohow = blockIdx.y * 16 + threadIdx.y;   // GEMM 的 M 维
  int k_out  = blockIdx.x * 16 + threadIdx.x;   // GEMM 的 N 维
  int n      = n_ohow / (H * W);
  int oh     = (n_ohow / W) % H;
  int ow     = n_ohow % W;
  if (n >= N || k_out >= K) return;

  float acc = 0.f;
  for (int c = 0; c < C; ++c) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
      int ih = oh + r - 1;      // padding=1
      if (ih < 0 || ih >= H) continue;
      #pragma unroll
      for (int s = 0; s < 3; ++s) {
        int iw = ow + s - 1;
        if (iw < 0 || iw >= W) continue;
        float x = input [((n * C + c) * H + ih) * W + iw];
        float w = weight[((k_out * C + c) * 3 + r) * 3 + s];
        acc += x * w;
      }
    }
  }
  output[((n * K + k_out) * H + oh) * W + ow] = acc;
}
// 生产版会用 cuDNN / CUTLASS，把 (r, s, c) 循环转成 tensor-core mma.sync。
```

*核心考点*：

- *"Winograd 为什么只对 3×3 好？"* → transform 引入额外加法，kernel 大时不划算。
- *"Depthwise conv 为什么慢？"* → 每 channel 独立，无跨 channel 的 K → matmul 变短胖，tensor core 利用率低。
- *"group conv 怎么优化？"* → 每 group 独立 GEMM，batch-GEMM 或 grouped GEMM (CUTLASS 3.x)。

== 15. all-reduce (multi-GPU)

*问题*：多 GPU 上每 GPU 有 partial gradient $g_i$，计算 $sum_i g_i$ 广播回每 GPU。

*为什么难*：
- 跨节点 NVLink / IB 带宽有限。
- naive scatter-gather 有 $O(P^2)$ 通信。

*Ladder*：

+ *Naive*：GPU 0 收所有 partial，reduce，再 broadcast → $O(P)$ 通信量、$O(P)$ 时延。
+ *Ring all-reduce*：$P$ 个 GPU 连成环，$2(P-1)$ step，每 step 传 $"size"/P$ → $O("size")$ 通信量、$O(P)$ 时延。NCCL 默认。
+ *Tree all-reduce*：$O("size")$ 通信量、$O(log P)$ 时延。适合大 $P$ 少 size。
+ *Double binary tree / SHARP*（InfiniBand hardware reduce）：hardware-accelerated in-network reduction。

*参考代码*（ring all-reduce 骨架——面试白板画出 chunk 流转即可，NCCL 会替你实现）：

```cpp
// 每 GPU 上有 buffer[size]，rank 是 0..P-1，环上 next = (rank+1)%P.
// 分成 P 个 chunk，chunk[i] 大小 = size / P.
//
// Phase 1: reduce-scatter，P-1 步.
//   step k: rank r 把 chunk[(r - k) mod P] 发给 next；
//           收到时加到本地 chunk[(r - k - 1) mod P] 上.
//   结束时: rank r 持有 fully-reduced 的 chunk[(r + 1) mod P].
//
// Phase 2: all-gather，P-1 步.
//   step k: rank r 把已经 reduced 的 chunk 发给 next；收到就直接覆盖.
//   结束时: 每 rank 持有所有完整 chunk.
//
// 总通信量 per GPU: 2 * (P-1)/P * size ≈ 2 * size，bandwidth-optimal.
// 步数: 2(P-1)，latency 随 P 线性增长——大 P 时改用 tree.

// 实际调用（NCCL）：
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
```

*核心考点*：

- *"ring 和 tree 什么时候用？"* → NCCL 会根据 size 和 P 自动选。size 大时 ring（bandwidth optimal），size 小时 tree（latency optimal）。
- *"gradient overlap with backward？"* → PyTorch DDP 在 backward 里每算完一层立即 all-reduce 那层的 gradient，通信和计算重叠。
- *"低比特 all-reduce？"* → FP16 grad 直接 all-reduce；BF16 mixed precision；1-bit / signSGD / PowerSGD 通信压缩。

== 常见 combo 问题

面试官经常把上面的题组合起来问：

- *"attention forward+backward 一起写"* → 主体是本书 ch 7-10 + 上面 Q6 (layernorm bw 的思路搬到 softmax + gemm bw)。
- *"实现 fused RMSNorm + Linear + SwiGLU（LLaMA FFN）"* → Q9 + Q13 + Q8。
- *"KV cache 的量化和 paged storage"* → Q11 + Q12。
- *"给我写一个训练循环 forward 的 kernel timeline"* → 需要能画出：Embedding → LayerNorm → Attention → LayerNorm → FFN → LayerNorm，每一步说 fuse 边界、哪些是 memory-bound、哪些是 compute-bound。

#insight[
  面试通用心法：*任何优化的第一步都是"这是什么 bound"*。看 AI (arithmetic intensity)，对比硬件 ridge point：
  - AI < 1 FLOP/B：memory-bound，优化目标是压满 HBM。
  - AI > 100 FLOP/B：compute-bound，优化目标是压满 Tensor Core。
  - 中间地带：设计能 overlap（async copy）、tile 让 AI 提升到 compute-bound 侧。

  面试官问 "怎么优化 X" 时你先说 "先看 X 是 memory-bound 还是 compute-bound"——这一句话就把你和"直接开始 tile"的候选人分开了。
]

#warn[
  这份 list 不是"背下来就够"。真正能 pass 的候选人：
  - 能*当场画*出一道题的 kernel skeleton（block / thread / smem 分配）。
  - 能*算*出 AI 和 roofline，判断上界。
  - 能*说*出哪些 ncu metric 会验证 / 反驳自己的猜测。
  这三点全书正文都在训练——所以先把正文里 8 个 kernel 每个都手动 code + ncu 一遍，再回来看这份 list。
]
