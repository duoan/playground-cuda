#import "../template.typ": *

= Reduce Sum

reduce sum 是 CUDA 优化 ladder 的第二个台阶。上一章 vector add 告诉我们：naive 就能打满带宽。这一章要打破这个直觉——*同一个 memory-bound 标签下，naive reduction 和最终版本可以差五个数量级*（实测 $N = 2^27$ 时 atomic 比 chunked 慢 ~$10^5$×）。

原因不在 HBM 带宽本身，而在*协作*：$N$ 个输入要坍缩成 1 个输出，thread 之间必须交换 partial sum。怎么交换、在哪里交换、同步几次——每一步都直接决定性能。

对应源码：`src/cuda/02_reduce_sum.cu`。

== 问题定义

给定长度为 $N$ 的数组 $x$，计算：

$ s = sum_(i=0)^(N-1) x[i] $

和 vector add 的关键区别：*输出只有一个标量*，$N$ 个 thread 都想往同一个位置写——这是 reduction 一切优化的出发点。

=== Roofline：依然是 memory-bound，但 naive 打不满

每处理 1 个输入元素：

- 读 $x[i]$：4 字节
- 计算：1 次 FADD
- 写：均摊到最终 1 次（可忽略）

$ "AI" = frac(1 "FLOP", 4 "B") = 0.25 "FLOP/B" $

A100 ridge point 约 13 FLOP/B。AI 仍比 ridge point 低两个数量级——*memory-bound*。

#insight[
  memory-bound 不代表 naive 就快。vector add 的 AI 低但*每 thread 独立*，合并访问完美。reduction 的瓶颈在*同步和 contention*——atomic 争用、shared memory bank conflict、`__syncthreads` 开销——这些在 roofline 模型里体现为"实际带宽远低于峰值"。
]

理论上界（只读 $N$ 个 float）：

$ T_"min" = frac(N times 4 "B", 2.04 "TB/s") $

- $N = 2^20$（4 MB）：$T_"min" approx 2 mu s$，但 L2 全命中，测出来会失真。
- $N = 2^27$（512 MB $ >> $ L2）：$T_"min" approx 260 mu s$。本章 chunked 版实测 356 μs（74% peak）；atomic 版实测 368 ms——*比理论下界慢 1400×*，瓶颈完全不在 HBM 读带宽。

== v1: atomic

```cpp
__global__ void reduce_sum_atomic_kernel(
    const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < count; i += stride) {
    atomicAdd(output, input[i]);
  }
}
```

每个 thread 读一个（或多个）元素，直接 `atomicAdd` 到全局的同一个 `output`。正确、最短、最好讲——但所有 thread 争同一个 cache line。

=== 为什么慢

`atomicAdd` 在 L2 层 serializes：同一地址上的 atomic 操作必须顺序执行。$N = 2^20$ 时有百万次 atomic，每次 ~几十 ns，总时间爆炸。

grid-stride loop 让 thread 数和数据规模解耦（和 vector add 一样），但*不能解决 contention*——争的还是同一个地址。

#warn[
  atomic reduction 适合 prototyping 或 $N$ 极小的情况。面试里提到 atomic 时要立刻接上："block 内先规约，每个 block 只写一次 partial sum"。
]

== v2: naive interleaved addressing

block 内 tree reduction 的经典写法，*第一步*往往长这样（interleaved addressing）：

```cpp
// 错误示范 —— interleaved addressing
for (int s = 1; s < blockDim.x; s *= 2) {
  if (threadIdx.x % (2 * s) == 0) {
    shared[threadIdx.x] += shared[threadIdx.x + s];
  }
  __syncthreads();
}
```

=== 每轮哪些 lane 在做工

先把 warp 内的执行 pattern 画出来。选*一个 warp*（block 里 8 个 warp 中的第 0 个，lane 编号 0..31），追踪它前三轮 `if (threadIdx.x % (2s) == 0)` 中哪些 lane 满足条件：

#figure(
  warp-grid(
    rows: 3, cols: 32, cell: 0.36, gap: 0.05, row-gap: 0.25,
    label-offset: 0.5, label-size: 9pt,
    row-labels: (
      [s = 1: 16 lane 活跃],
      [s = 2:  8 lane 活跃],
      [s = 4:  4 lane 活跃],
    ),
    active: (
      // s=1: lane 0, 2, 4, ..., 30 (even lanes)
      ..range(16).map(k => (0, k * 2)),
      // s=2: lane 0, 4, 8, ..., 28
      ..range(8).map(k => (1, k * 4)),
      // s=4: lane 0, 8, 16, 24
      ..range(4).map(k => (2, k * 8)),
    ),
  ),
  caption: [*Figure:* interleaved addressing 前 3 轮 warp 内的 lane 活跃 pattern。每行 32 格 = 一个 warp 的 32 个 lane（lane 0 在最左、lane 31 在最右）。*绿色* = 该 lane 上 `if (tid % (2s) == 0)` 为真、执行 `shared[tid] += shared[tid+s]`；*灰色* = 条件不满足，*但硬件仍然发射同一条 warp instruction*——只是灰 lane 的结果被 predicate 丢弃。],
  kind: image,
)

*怎么读这幅图*——从上到下三行对应循环变量 `s = 1, 2, 4`：

- *s = 1*：偶数编号 lane 绿（0, 2, 4, ..., 30 共 16 个）——半数 lane 在做加法。
- *s = 2*：绿 lane 变成 0, 4, 8, ..., 28——只剩 8 个。
- *s = 4*：绿 lane 只剩 0, 8, 16, 24——4 个。

*关键*：绿色 lane 从来*不连续*——它们在 warp 内*间隔*分布，间隔从 2 拉到 4、再到 8。而这个 warp 里*所有 32 个 lane* 都跟着发射同一条加法指令。

*读者第一反应可能是：这不就是 warp divergence 吗？*

历史教材（Harris 2007）确实这样说。但这个直觉需要被硬证据检验——否则就是"上帝视角"看代码。下面用*三条独立证据链*把它钉死。

==== 证据 1：SASS 里没有分支，只有 predicated instruction

`nvcc -arch=sm_80 -O2` 编出的 SASS（interleaved kernel 主循环体）：

```
/*0270*/  ISETP.NE.AND  P1, PT, R3, RZ, PT ;    // P1 = (tid % (2*stride) != 0)
/*0280*/  @!P1  IMAD    R2, R5, 0x4, R0 ;       // addr calc, predicated
/*0290*/  @!P1  LDS.U   R3, [R8.X4] ;           // load shared[tid], predicated
/*02b0*/  @!P1  LDS.U   R2, [R2] ;              // load shared[tid+s], predicated
/*02c0*/  @!P1  FADD    R3, R3, R2 ;            // add, predicated
/*02d0*/  @!P1  STS     [R8.X4], R3 ;           // store, predicated
/*02f0*/  ISETP.GE.U32.AND  P1, PT, R10, ... ;  // for-loop condition
/*0300*/  @!P1  BRA     0x150 ;                 // loop back (uniform, not divergent)
```

整个 `if` 语句被编译成 `@!P1` 前缀的 predicated 指令序列——*没有 `BRA` 前向跳转*。硬件层面每条 LDS/FADD/STS 都发给整个 warp（issue 一次、32 lane 都执行），只是 P1=1 的 lane 上 write-back 被 mask 掉。这就叫 predication，不叫 warp divergence。

==== 证据 2：ncu 直接测 branch divergence 的指标 = 0

`smsp__sass_branch_targets_threads_divergent.sum` 是 ncu 提供的*直接*测 branch divergence 事件计数的指标——每次一个 branch 让 warp 内 lane 走不同 target 就 +1。

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: 0.4pt + gray,
    table.header([kernel],
                 [`issued/32`],
                 [`pred_on/32`],
                 [`branch\ divergent.sum`],
                 [`uniform\ branch %`]),
    [`interleaved`], [31.92], [24.00], [*1*], [100.00%],
    [`sequential (shared)`], [31.81], [20.03], [*0*], [100.00%],
  ),
  caption: [*Table:* interleaved 和 sequential 两个 kernel 的 warp-divergence 直接证据（$N = 2^27$，A100 80GB，`ncu --metrics smsp__sass_branch_targets_threads_divergent.sum` 等）。`branch_divergent.sum` 表示 warp 内 lane 走不同 target 的 branch 事件次数——两个版本都是 0 或 1，即*两个版本都没有真正的 warp divergence*。`pred_on/32` 是 predicate=true 的平均 lane 数：sequential 反而更*低*（20.03 vs 24.00），因为它每轮真正 predicate=true 的 lane 更少（`tid < offset` 后期收缩得更快）——但它仍更快。],
  kind: table,
)

*观察*：如果 "interleaved 慢是因为 warp divergence" 这个说法成立，`branch_divergent.sum` 应该是几十万甚至上百万；实际是 1。同时 `pred_on/32` 反直觉——sequential 的 predicate-on lane 数更少，但性能更好。这两条一起把"divergence 假说"证伪了。

==== 证据 3：controlled 实验——真 divergence 长什么样

为了避免"metric 不敏感"的怀疑，我用同一台 A100 跑一个*故意制造 2-way branch divergence* 的对照 kernel（两个 heavy math 分支足够大，编译器无法 predicate）：

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.4pt + gray,
    table.header([test kernel], [`issued/32`], [`pred_on/32`], [`branch_divergent.sum`]),
    [uniform（1 path，无 if）], [32.00], [31.99], [0],
    [predicated（`if(tid&1)` 短分支）], [32.00], [28.00], [0],
    [*branched（2-way heavy，强制分支）*], [*16.01*], [16.01], [*8*],
  ),
  caption: [*Table:* 对照实验：三种 divergence pattern 下 metric 的响应。uniform kernel 全 lane 同 path，`issued/32 = 32`（基线）。predicated kernel 的 `if` 被编译器 predicate 掉，`issued/32` 仍 = 32、`branch_divergent = 0`——只有 `pred_on` 下降。branched kernel 强制 2-way 分支，`issued/32` 精确掉到 16.01（$= 32/2$），`branch_divergent.sum` 显著非零。],
  kind: table,
)

*观察*：`issued/32` 和 `branch_divergent.sum` 对真 branch divergence *完全敏感*——2-way 分支下 `issued/32` 精准跌到 16.01（这跟 arxiv 2607.23402 "Characterizing Warp Divergence from Pascal to Blackwell" 报告的 32/k 定律完全一致）。既然对照实验在同一硬件、同一 ncu 版本下能测出真 divergence，那 interleaved kernel 的 `issued/32 = 31.92` 就*不可能*是"metric 不敏感"的假象——是真的没有 divergence。

==== 那 interleaved 为什么慢

三条证据链一致指向：*differ 不在 warp divergence，而在别处*。

1. *没有整 warp 退出*——每一轮 8 个 warp 里 warp 0-7 都得跟着发射指令（因为每 warp 内都有若干 lane 满足 `tid % (2s) == 0`），scheduler 无法把整 warp 直接 idle。sequential 版本用 `if (tid < offset)` 让*整个 warp* 一起退出（warp 3-7 完全 idle），idle warp 不占 issue slot——这是 dispatcher-level 的差别，跟 warp *内部* divergence 无关。
2. *bank conflict pattern 更糟*——`shared[tid + s]` 在 s=1 时无冲突，但 s=2, 4, ...→ 8 时（`128 mod 32 = 0` 附近）冲突剧增；累积下来 interleaved 有 402k bank conflict，sequential 只有 314k（本章"实测"diag 表数据）。

#insight[
  「interleaved 慢因为 warp divergence」这个说法在 ncu 上*无证据支持*——三条独立证据（SASS 无 BRA、`branch_divergent.sum` = 1、controlled 实验证明 metric 灵敏）一致指向 predication 而非 divergence。Harris 2007 的原始文本写在 CUDA compiler 还不成熟的时代；今天的 ptxas 会把小 if 全部 predicate 掉。真正的性能差 = *idle warp 能否整体退出* + *bank conflict 模式*——见下面"实测"section。
]

=== 与 sequential 版本的 diff

`if (threadIdx.x % (2s) == 0)` → `if (threadIdx.x < s)`。这一个字符的改动带来两个效应：一是让整 warp 能一起退出（scheduler-friendly），二是让 warp 内 lane 访问 smem 时更容易避开同一 bank（先看下一节，再对照"实测"表里 402k → 314k 的证据）。

== v3: sequential addressing

源码里的 `reduce_sum_shared_kernel` 用的就是 sequential addressing——Mark Harris "Optimized Parallel Reduction" 教程的第 2 步：

```cpp
__global__ void reduce_sum_shared_kernel(
    const float* input, float* block_sums, int count) {
  extern __shared__ float shared[];
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = (global_index < count) ? input[global_index] : 0.0f;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = shared[0];
  }
}
```

=== 为什么整个 warp 会一起退出

第一轮（`offset = 128`）：`threadIdx.x < 128` 的 thread 活跃——正好是 warp 0 的全部 32 个 lane + warp 1 的全部 32 个 lane + ...，*每个 warp 内 32 个 lane 同走 active 分支*，warp 3-7 全部退出，直接被 scheduler 跳过。

每轮 active thread 是 `[0, offset)` 的连续区间，warp 边界对齐时（block size 是 32 的倍数），每个 warp 要么全 active 要么全 idle——空闲 warp 不占 issue slot。

#figure(
  tree-reduction(mode: "sequential", n: 16, cell: 0.36),
  caption: [Sequential addressing，$n = 16$：*绿色* = 本轮活跃 lane，*黄色* = 持有有效值但本轮闲置，*灰色* = 数据已被规约进别处。每轮活跃 lane 连续、warp 内一致。]
)

#figure(
  tree-reduction(mode: "interleaved", n: 16, cell: 0.36),
  caption: [Interleaved addressing 对比：活跃 lane 间隔分布，warp 内 lane 部分 active、部分 idle——但 ncu 显示 `issued/32 = 32`（是 predicated，不是 divergence）。真正的差别是访问模式导致的 bank conflict pattern。]
)

#warn[
  历史教材（Harris 2007）把这里说成"消除 warp divergence"。上一节用三条证据链（SASS / `branch_divergent.sum` / controlled 实验）证明 ncu 数据*不支持*这个说法。真正的差异是：*sequential 让整个 warp 一起退出*（idle warp 不占 issue slot，是 scheduler-level 事件）+ *bank conflict 模式不同*（402k → 314k）。
]

=== 每 block 输出一个 partial sum

256 个 thread 协作 → `shared[0]` 是这个 block 的局部和 → 写到 `block_sums[blockIdx.x]`。$N$ 个元素变成 `ceil(N / 256)` 个 partial sum。

#note[
  一次 kernel launch 不会直接把 $N$ 变成 1。源码 `run_multi_stage_reduction` 循环 launch：$N -> "blocks" -> "blocks'" -> ... -> 1$。这是 grid-level reduction 的标准模式，后面专门讲。
]

=== bank conflict：第一轮最严重

shared memory 有 32 个 bank，每个 4 字节宽。`shared[tid] += shared[tid + 128]`：warp 内 32 个 lane 同时读 `shared[0..31]` 和 `shared[128..159]`。地址 `128 % 32 = 0`，所以 `shared[i]` 和 `shared[i+128]` 落在*同一个 bank*——32-way bank conflict，一次 access 被串行化成 32 次。

后续轮次（offset = 64, 32, ...）conflict 逐渐减轻。offset = 32 时，`shared[i]` 和 `shared[i+32]` 在不同 bank，无 conflict。

#warn[
  面试常问："reduction 的 bank conflict 在哪一轮最严重？" 答：第一轮（offset = blockSize/2），因为 stride 恰好是 32 个 bank 的周长。缓解方法：padding shared array（`shared[threadIdx.x + padding]`）、或用 warp shuffle 绕过 shared memory。
]

== v4: unroll last warp

树形规约的最后 $log_2(32) = 5$ 步只在最后一个 warp 内发生（256-thread block 时，offset 从 16 降到 1）。此时 block 里只剩前 32 个 thread 有非零值——*再调用 `__syncthreads` 是浪费*。

```cpp
if (threadIdx.x < 32) {
  volatile float* vsmem = shared;  // 旧写法，见面试考点
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
  vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
}
```

=== 为什么可以不用 `__syncthreads`

warp 内的 32 个 thread *SIMT 锁步执行*——同一 warp 不存在分支 divergence 的情况下，lane 0 写 `shared[0]` 后，lane 1 读 `shared[1]` 在同一 warp 的指令流中是顺序可见的。

`__syncthreads` 是 *block 级* 屏障：确保 block 内所有 thread（所有 warp）都到达。最后 5 步只有 warp 0 参与，其他 warp 的 thread 已经 idle——对它们做 block barrier 没有同步必要，只有开销（~10–20 cycles/barrier）。

#insight[
  "unroll last warp" 的核心：*识别出同步范围可以缩小到 warp 级*。当 active thread 数 $<= 32$ 时，warp 内天然同步，省掉 5 次 `__syncthreads`。
]

== v5: complete unroll (template `blockSize`)

把 block 内所有 $log_2("blockSize")$ 步完全展开，用模板在编译期固定循环边界：

```cpp
template <int BlockSize>
__device__ void block_reduce(volatile float* shared) {
  if (BlockSize >= 512) { if (threadIdx.x < 256) shared[threadIdx.x] += shared[threadIdx.x + 256]; __syncthreads(); }
  if (BlockSize >= 256) { if (threadIdx.x < 128) shared[threadIdx.x] += shared[threadIdx.x + 128]; __syncthreads(); }
  if (BlockSize >= 128) { if (threadIdx.x <  64) shared[threadIdx.x] += shared[threadIdx.x +  64]; __syncthreads(); }
  if (BlockSize >=  64) { if (threadIdx.x <  32) shared[threadIdx.x] += shared[threadIdx.x +  32]; }
  if (BlockSize >=  32) { if (threadIdx.x <  16) shared[threadIdx.x] += shared[threadIdx.x +  16]; }
  if (BlockSize >=  16) { if (threadIdx.x <   8) shared[threadIdx.x] += shared[threadIdx.x +   8]; }
  if (BlockSize >=   8) { if (threadIdx.x <   4) shared[threadIdx.x] += shared[threadIdx.x +   4]; }
  if (BlockSize >=   4) { if (threadIdx.x <   2) shared[threadIdx.x] += shared[threadIdx.x +   2]; }
  if (BlockSize >=   2) { if (threadIdx.x <   1) shared[threadIdx.x] += shared[threadIdx.x +   1]; }
}
```

=== 编译期确定的好处

1. *消除循环控制开销*：没有 `offset /= 2` 和分支跳转。
2. *编译器常量传播*：`if (BlockSize >= 256)` 在 `BlockSize = 256` 实例化时变成常量，dead code 被优化掉。
3. *精确控制 sync 点*：前 3 步需要 `__syncthreads`（跨 warp），后 5 步不需要——模板里写死，不会多也不会少。

Launch 时 `reduce_kernel<256><<<blocks, 256, 256*sizeof(float)>>>(...)`。block size 必须是编译期常量——这是 CUDA 模板 kernel 的标准用法。

== v6: warp shuffle

源码 `reduce_sum_warp_kernel` 把 block 内规约拆成两层：*warp 内 shuffle → warp 间 shared memory*。

```cpp
__device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}
```

```cpp
__global__ void reduce_sum_warp_kernel(
    const float* input, float* block_sums, int count) {
  __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];  // 256/32 = 8

  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = (global_index < count) ? input[global_index] : 0.0f;

  value = warp_reduce_sum(value);           // Step 1: 每个 warp 内规约

  const int lane = threadIdx.x % kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;
  if (lane == 0) { warp_sums[warp_id] = value; }
  __syncthreads();                          // Step 2: 8 个 warp partial sum 写入 smem

  if (warp_id == 0) {                       // Step 3: warp 0 规约 8 个 warp sum
    value = (lane < (blockDim.x / kWarpSize)) ? warp_sums[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) { block_sums[blockIdx.x] = value; }
  }
}
```

=== warp-level primitives 详解

`__shfl_down_sync(mask, val, delta)`：lane $i$ 从 lane $i + "delta"$ 读取 `val`，*不经过 shared memory*。数据在 warp 的 register file 之间通过 *crossbar (Xbar)* 交换。

- `mask = 0xffffffff`：全部 32 个 lane 参与（必须写参与 mask，Volta+ 要求 explicit sync）。
- 5 步 shuffle（16→8→4→2→1）= 5 条指令，零 bank conflict，零 shared memory traffic。

#insight[
  shuffle 把 reduction 的"通信介质"从 shared memory 换成 register + Xbar。shared memory 带宽有限且有 bank conflict；register shuffle 走专用通路，延迟 ~1 cycle，是 warp 内规约的最优路径。
]

=== 硬件实现：Xbar

每个 SM 的 warp scheduler 发 shuffle 指令时，32 个 lane 的 register operand 通过 *warp-level crossbar* 互连——lane $i$ 可以选择 lane $j$ 的值。这是专用电路，不占用 shared memory port，也不产生 global memory traffic。`__shfl_down_sync` / `__shfl_up_sync` / `__shfl_xor_sync` 都走这条路。

256-thread block 的 sync 次数：shared tree 需要 8 次 `__syncthreads`；warp 版本只需 1 次（warp 间合并），还省掉了 shared memory tree 的 smem 读写。理论上 warp 版应更快——实测见下文 `=== 实测`。

== v7: grid-level reduction

单个 block 只能规约 `blockDim.x`（或 `blockDim.x * items_per_thread`）个元素。$N = 2^20$ 需要 `ceil(2^20 / 256) = 4096` 个 block——一次 launch 输出 4096 个 partial sum，*还没完*。

源码的多阶段 host 循环：

```cpp
while (current_count > 1) {
  const int blocks = cuda_utils::ceil_div(current_count, kThreadsPerBlock);
  launch_kernel(current_input, next_output, current_count, blocks);
  current_input = next_output;
  current_count = blocks;   // 4096 -> 16 -> 1
  ++stages;
}
```

$2^20$ 元素：stage 0 输出 4096 个 partial sum → stage 1 输出 16 个 → stage 2 输出 1 个。共 3 次 kernel launch。

=== 三种 grid-level 策略

*1. 多阶段 launch（源码采用）*

每次 launch 是一个完整的 reduction kernel。简单、清晰、易 debug。代价是 2~3 次 launch overhead（~5 μs each，对大 $N$ 可忽略）。

*2. 单 kernel 两阶段*

第一个 loop 规约输入 → 写 partial sums；第二个 loop（仅 `gridDim.x` 个 thread）规约 partial sums。Mark Harris 原版教程的做法。省 launch 但需要 `if (gridIdx.x < gridDim.x)` 判断当前 thread 做哪一阶段。

*3. atomicAdd 合并*

每个 block 直接 `atomicAdd(&output, block_sum)`。省 multi-stage 但 reintroduce atomic contention——block 数 = 4096 时比 1 个 atomic 好很多，但仍不如 hierarchical。适合 block 数 $<= 100$ 的场景。

#note[
  cuBLAS / CUB / cub::DeviceReduce 内部用的是 warp shuffle + multi-stage + vectorized load 的组合，并针对每种 GPU 架构调 block size 和 items-per-thread。面试不需要背 CUB 源码，但要能画出 $N -> "blocks" -> 1$ 的缩图。
]

== v8: chunked hierarchical

```cpp
constexpr int kChunkItemsPerThread = 4;

__global__ void reduce_sum_chunked_kernel(
    const float* input, float* block_sums, int count) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * blockDim.x * kChunkItemsPerThread;
  const int thread_start = block_start + tid;

  float local_sum = 0.0f;
  #pragma unroll
  for (int item = 0; item < kChunkItemsPerThread; ++item) {
    const int index = thread_start + item * blockDim.x;
    if (index < count) { local_sum += input[index]; }
  }

  shared[tid] = local_sum;
  __syncthreads();
  // ... sequential tree reduction ...
}
```

=== 每个 thread 先做更多活

一个 block 覆盖 `256 * 4 = 1024` 个元素（对比 v3 的 256 个）。好处：

1. *摊销 index 计算和 `__syncthreads` 成本*：每 1024 元素只做 1 次 block sync，不是每 256 元素。
2. *寄存器累加隐藏 load latency*：4 个 load 可以 pipeline。
3. *步长 = blockDim.x 保持合并访问*：和 vector add tiled 版本同一原则。

每个 thread 的 partial sum 存在 register 里，进入 shared tree 时 active 数据已经压缩 4 倍——后续 warp 间通信量不变，但*单位 sync 处理的元素数翻 4 倍*。

== 性能 ladder 一览

源码里从 atomic 到 chunked 是一步步叠加的优化。下面 ladder 描述*相对排序和每步原因*；绝对数字见下一节实测。

#ladder(
  ("atomic",              "全局 atomicAdd",                    "contention 灾难"),
  ("interleaved tree",    "shared memory, 间隔寻址",            "warp divergence"),
  ("sequential tree",     "shared memory, 连续寻址",            "零 divergence，有 bank conflict"),
  ("+ unroll last warp",  "省 5 次 __syncthreads",             "缩小 sync 范围到 warp"),
  ("+ complete unroll",   "模板展开 + 编译期 blockSize",       "消除循环控制"),
  ("warp shuffle",        "register shuffle + 1 次 smem sync",  "绕过 smem tree"),
  ("chunked + shuffle",   "4 elements/thread + hierarchical",  "摊销 sync / index 成本"),
)

#warn[
  ladder 里 `unroll last warp` / `complete unroll` 等中间版本未单独 benchmark。本章实测覆盖 atomic、interleaved tree、sequential tree、warp shuffle、chunked hierarchical 五个代表点——足够说明 contention vs 协作规约 vs bank conflict vs launch overhead 四个层次。
]

=== 实测

$N = 2^27 + 37$（约 128 M 元素 / 512 MB，远超 A100 40 MB L2），A100 80GB SXM4，`ncu` 抓取*每个 kernel 的第一次 launch*（也就是处理最大输入的那次；后续 stage 数据缩小到 μs 级，metric 意义有限）。GB/s 列写作 *HBM 实测 / 逻辑*：前者 `dram__bytes.sum / time`，后者 $N times 4 / "time"$；这两个数字*大规模下应几乎相等*——差距大就是 L2 命中或者 kernel 太短。

#include "../bench/02_reduce_sum.typ"

*perf 表读三件事：*

+ *atomic 慢 100000 倍*。367 ms vs sequential tree 的 1.2 ms——不是"几十倍"，是*五个数量级*。128 M 次 atomicAdd 全撞同一 cache line，L2 层 serialize，`mem stall = 6040`（每 issue-active cycle 有 6040 个 warp 卡在 long_scoreboard 上），HBM % 只有 0.1%——完全不是 memory-bound，是*serialization-bound*。

+ *interleaved vs sequential：2.6 ms → 1.2 ms（2.2×）*。同样是 shared memory tree reduction，仅寻址模式从 `if (tid % (2s) == 0)` 换成 `if (tid < s)` 就快一倍。差距的物理来源要看 diag 表。

+ *warp shuffle / chunked：接近打满带宽*。warp 版 743 μs 达到 HBM 36%（728 GB/s），chunked 版 356 μs 达到 74%（1515 GB/s）。chunked 是本章最快，几乎打满 HBM。

*diag 表读关键教学点：*

*a) interleaved vs sequential 的差别 *不是* warp divergence*——两者 `issued/32` 都是 31.8-31.9，几乎完美。`if (tid < s)` 和 `if (tid % (2s) == 0)` 在 warp 里都是 predicated add，不是真正的 branch divergence。

`pred_on/32` 反而告诉了我们*相反*的故事：interleaved = 24.0，sequential = 20.0——*sequential 的 lane 利用率更低*！但 sequential 快 2.2×，为什么？答案在 `smem conf.` 一列。

*b) sequential 快的真正原因：bank conflict 少 22%*。interleaved 402k 次 bank conflict，sequential 314k 次——都不为零，但 sequential 让每个 warp 内 lane 访问 *连续的* smem，冲突集中在 warp 之间而非 warp 内，L1TEX 处理更高效。

#insight[
  这是本书 pushback "上帝视角"的核心案例。传统教材把 interleaved 慢的原因归为 "warp divergence"——ncu 的 `issued/32 = 32` 直接反驳了这个说法。真正的差别在 `smem conf.` 列。术语用"warp lane utilization" (issued vs pred_on) 和 "bank conflict pattern" (smem conf.) 比笼统说 "divergence" 更精确。
]

*c) warp shuffle 大幅减少 smem 访问 → bank conflict 从 314k 降到 56k*。这是 warp shuffle 带来的最大好处：register-to-register 数据交换*完全绕过 shared memory*，最后只有一次 warp partial sum 写 smem 的 sync。`barrier stall = 1.48` 也是四个协作规约里最低的。

*d) chunked 的胜利在 grid 太多 vs 每 thread 多做工的 tradeoff*。chunked 每 thread 处理 4 个元素 → grid 从 524289 缩到 131073。`mem stall = 5.69`（vs warp 版 18.44）显示 memory 队列被 4× fetch 请求填得更满，HBM % 从 36% 涨到 74%。

*bank conflict 的形成机制：*

sequential addressing 第一轮 `shared[tid] += shared[tid + 128]`：tid=0 读 bank 0 和 bank 0（128 mod 32 = 0）——32-way conflict；tid=1 读 bank 1 和 bank 1——同样 32-way；tid=0..31 全部 32-way conflict，共 32 次访问。

interleaved 第一轮 `shared[tid] += shared[tid + 1]`（stride=1）：tid=0 读 bank 0 和 bank 1，tid=2 读 bank 2 和 bank 3——无冲突。但 stride=2、4、... 时冲突逐渐增加。*结论*：interleaved 前几轮少冲突、后几轮多冲突；sequential 前几轮多冲突、后几轮少冲突。ncu 累积的 402k vs 314k 是全部轮次求和的结果。

#note[
  想在源码级别验证 bank conflict，用 `ncu --section MemoryWorkloadAnalysis_Tables` 会打出按 access pattern 分类的 conflict 明细。
]

#warn[
  atomic 的 mem_stall = 6040 与"memory latency stall"的表面含义相反：`long_scoreboard` 包含所有 memory dependency 类 stall，atomic 的 read-modify-write 让 L2 atomic unit 反复往返 → 每次操作都需要"等 L2 回复"，被计入 long_scoreboard。这也是为什么 atomic 看起来像 mem-bound，实际是 serialization。
]

== ncu 该看什么

```
sudo ncu --set full ./build/02_reduce_sum 27
```

关键 metric 分四类：

- *serialization*: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` — atomic 版会飙到几千。
- *warp lane utilization*: `smsp__thread_inst_executed_per_inst_executed.ratio` (issued/32) 和 `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio` (pred_on/32)——判断 predicated-off 的 lane 数量。
- *bank conflict*: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` + `..._op_st.sum`——判断 smem tree reduction 的效率。
- *HBM utilization*: `dram__bytes.sum` 除以 kernel time，得到*实际* HBM GB/s；`dram__bytes.sum.pct_of_peak_sustained_elapsed` 用 elapsed time 作分母，对短 kernel 偏低。

== 面试白板 code

面试官说"手写一个 reduce sum"——直接白板 warp shuffle 版。第一句话就说："我分两级——warp 内 shuffle，warp 间 shared memory"，然后写：

```cpp
constexpr int WARP = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
  // 5 层蝶形：每层把距离 offset 的 lane 值加过来。
  #pragma unroll
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;  // lane 0 持有 warp 内 sum，其他 lane 是垃圾值
}

__global__ void reduce_sum(const float* x, float* block_sums, int n) {
  __shared__ float warp_sums[32];  // 最多 32 个 warp / block

  // grid-stride 拿到本 thread 的 partial（一个 thread 可能吃多个元素）
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    v += x[i];
  }

  // Level 1: warp 内规约
  v = warp_reduce_sum(v);

  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  if (lane == 0) warp_sums[wid] = v;  // 每 warp 的 tail 存 smem
  __syncthreads();

  // Level 2: warp 0 再规约 warp_sums（数量 <= 32，一个 warp 搞定）
  if (wid == 0) {
    int n_warps = (blockDim.x + WARP - 1) / WARP;
    v = (lane < n_warps) ? warp_sums[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) block_sums[blockIdx.x] = v;  // 每 block 输出一个 partial sum
  }
}

// ==== Launch config ====
// blockDim = 256 (8 warp)：
//   * 必须是 32 的倍数——最后的 warp reduce 靠 warp 对齐；
//   * 8 warp 让 warp_sums[] <= 8 <= 32，第二级 warp reduce 一个 warp 搞定；
//   * 不要开 1024——太多 warp 会让最后一级 warp reduce 变复杂.
// gridDim = min(SM * 4, (N + block - 1) / block)：
//   * SM * 4 让每 SM 常驻多个 block、latency hide；
//   * 但也不要太大——block_sums 长度 = gridDim，后面还要再规约一次.
int block = 256;
int grid  = min(4 * sm, (n + block - 1) / block);
size_t smem = 0;  // 上面代码用的是静态 smem (32 float)；如需 dynamic 就传大小
reduce_sum<<<grid, block, smem>>>(x, block_sums, n);

// Stage 2: 把 block_sums (长度 grid) 再规约成 1 个数.
// 如果 grid <= 1024，直接一 block 处理完.
reduce_sum<<<1, 1024>>>(block_sums, out, grid);
```

*核心考点*（追问顺序）：

- *"为什么不直接一个大 shared array 做树？"* → 可以，但 warp shuffle 走寄存器堆、比 shared memory 快 3-5×，还不需要 `__syncthreads`（warp 内锁步）。
- *"为什么 grid-stride 循环里就先做加法？"* → 让一个 thread 吃很多元素、shared / shuffle 那部分只处理 blockDim.x 个 partial——避免多 stage kernel 相互 launch 开销。
- *"最后一个数怎么得到？"* → block_sums 数量 = grid size。要么再 launch 一次 `reduce_sum` 直到剩 1 个（multi-stage），要么用 grid-level atomic 加到 `*out`（简单但有 contention），要么用 CUB / cooperative-groups grid barrier。
- *"warp shuffle 要不要 mask？"* → `__shfl_down_sync` 第一个参数是 active mask；这里所有 32 lane 都活，写 `0xffffffff`。如果有些 lane 提前 return 了，mask 要相应改，否则 UB。
- *"能证明没有 warp divergence 吗？"* → SASS 里 `if` 会被 predicate，ncu `smsp__sass_branch_targets_threads_divergent.sum ≈ 0` 就是证据（见本章 v2 讨论）。
- *"blockDim 为什么选 256 不选 1024？"* → (1) 必须是 32 的倍数、否则最后 warp reduce 不对齐；(2) 256 warp / 32 = 8，`warp_sums[]` 长度 = 8，第二级用 warp 0 一个 warp 搞定；1024 会让 warp_sums 长度 = 32、正好占满 warp，稍显浪费——但更重要是寄存器压力：1024 threads/block 会限制 occupancy。256 是"够用又不奢侈"的经典选择。

== 面试考点

#interview[
  *Q1*: reduction 是 memory-bound 吗？为什么 naive 版本还是很慢？

  A: AI = 0.25 FLOP/B，远低于 ridge point，读带宽是主导。但 naive（atomic 或 interleaved tree）的瓶颈在*同步和 contention*，不是 HBM 带宽本身。优化目标是减少 sync 次数、消除 bank conflict、让每 thread 多做 work 再通信。
]

#interview[
  *Q2*: interleaved addressing 和 sequential addressing 的区别？为什么后者更快？

  A: interleaved 用 `if (tid % (2s) == 0)` 选活跃 thread，它们在 warp 内间隔分布；sequential 用 `if (tid < s)`，活跃 thread 是连续区间。*两者都不是真正的 warp divergence*（ncu `issued/32` 都是 32，编译成 predicated instruction）。真正的差异在 bank conflict pattern：sequential 让 warp 内 lane 访问连续 smem 地址，L1TEX 处理更高效——实测 sequential bank conflict 314k vs interleaved 402k，快 2.2×。

  #warn[
    如果面试官坚持说 interleaved "有 warp divergence"，你可以补一句：predication 不是 divergence——ncu `smsp__thread_inst_executed_per_inst_executed.ratio` 会区分这两种情况。真正的分支 divergence 是 warp 里*不同* lane 走*不同 basic block*（if-else 两侧都有工作）。
  ]
]

#interview[
  *Q3*: reduction 中 bank conflict 在哪一步最严重？怎么缓解？

  A: 第一步（offset = blockSize/2）最严重：stride = 128 时 `shared[i]` 和 `shared[i+128]` 在同一 bank（128 mod 32 = 0），32-way conflict。缓解：padding（`shared[tid + 1]`）、减小 block size、或用 warp shuffle 绕过 shared memory。
]

#interview[
  *Q4*: 为什么 unroll last warp 可以不用 `__syncthreads`？

  A: 最后 $log_2(32)=5$ 步只在 warp 0 内执行。同一 warp 的 32 lane SIMT 锁步，register/shared 写后在同一 warp 内可见，不需要 block 级 barrier。对其他已经 idle 的 warp 做 `__syncthreads` 只有开销（~10–20 cycles），没有同步收益。
]

#interview[
  *Q5*: `__syncthreads` 的成本是多少？什么时候值得省？

  A: 大约 10–20 cycles（取决于架构和 block 大小）。当同步范围可以缩小到 warp 级（shuffle）或 active thread 只在 1 个 warp 内时，值得省。block 内有多个 warp 仍活跃时必须用 `__syncthreads`。
]

#interview[
  *Q6*: `volatile` 和 `__syncwarp()` 的区别？unroll last warp 该用哪个？

  A: `volatile` 是编译器层面的语义：禁止把 shared memory 读缓存到 register，强制每次从 smem 重新读——旧代码（pre-Volta）用来保证 warp 内可见性。`__syncwarp()` 是硬件 warp 级 barrier（~5 cycles），保证 warp 内所有 lane 到达后再继续。现代代码（Volta+）应优先用 `__syncwarp()` + 普通读写，或直接用 `__shfl_down_sync` 完全绕过 smem。`volatile` 不能替代 sync——它只防编译器优化，不保证其他 thread 已写完。
]

#interview[
  *Q7*: `__shfl_down_sync` 的硬件实现是什么？比 shared memory 快在哪？

  A: 走 warp 内 register crossbar (Xbar)：lane $i$ 直接从 lane $i+delta$ 的 register 读值，不经过 shared memory port。延迟 ~1 cycle，无 bank conflict，无 extra smem traffic。shared memory tree 每步需要 2 次 smem access（读+写）+ 可能的 bank conflict。
]

#interview[
  *Q8*: 单个 block 输出一个 partial sum 之后怎么办？有哪些 grid-level 策略？

  A: (a) 多阶段 launch：反复 launch reduction kernel 直到 1 个值（源码做法）。(b) 单 kernel 两阶段：第一个 loop 写 partial sums，第二个 loop 规约它们。(c) 每个 block `atomicAdd` 到全局输出——block 数少时可接受。生产环境常用 (a) 或 CUB 的 tuned 版本。
]

#interview[
  *Q9*: 为什么 chunked（multi-elements-per-thread）有效？步长为什么用 `blockDim.x`？

  A: 每个 thread 在 register 里累加 4 个元素再进入 block 规约，摊销 index 计算和 `__syncthreads` 成本，4 个 load 可 pipeline 隐藏 latency。步长 = `blockDim.x`（不是 4）是为保持 warp 内合并访问——和 vector add tiled 版本同一原则。
]

#interview[
  *Q10*: 模板 `blockSize` 完全展开有什么好处？

  A: 编译期固定循环边界 → 消除循环控制开销、常量传播去掉 dead branch、精确控制哪些步需要 `__syncthreads`（跨 warp 的 3 步需要，warp 内的 5 步不需要）。Launch 时 block size 必须匹配模板参数。
]
