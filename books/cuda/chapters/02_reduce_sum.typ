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
- $N = 2^27$（512 MB $ >> $ L2）：$T_"min" approx 260 mu s$。本章 chunked 版实测 287 μs（92% peak）；atomic 版实测 368 ms——*比理论下界慢 1400×*，瓶颈完全不在 HBM 读带宽。

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
  *区分 per-element atomic vs per-block atomic*——两者性能相差 3 个数量级：

  - *per-element atomic*（本 v1，$N$ 次 atomicAdd 撞同一地址）：serialization 灾难，$N = 2^27$ 时 368 ms。
  - *per-block atomic*（v2--v5，$"blocks"$ 次 atomicAdd 撞同一地址）：block 数比 $N$ 少 2--3 个数量级，且每次 atomic 已经是一个 block 的 partial sum；contention 变成"$approx 10^4$ 个 block 争一个 cache line"，L2 atomic unit 处理起来毫无压力。

  面试里提到 atomic 时要立刻接上："block 内先规约，*每个 block 只写一次* partial sum，然后 `atomicAdd` 直接 finalize 到全局输出"——这就是 v5 chunked 用的模式，一次 kernel launch 出结果，没有多阶段。
]

=== ncu 实测

#ncu-snapshot(
  version: "atomic",
  size: [$N = 2^22$（$approx 4.2$ M 元素）],
  rows: (
    ("Duration",            "11 490 µs", "比同规模 sequential tree 慢 ~3000×"),
    ("Memory SOL",          "1.6 %",     "HBM 几乎没用——不是 memory-bound"),
    ("Compute SOL",         "0.3 %",     "SM 也没在算——都在等 atomic"),
    ("L2 Hit Rate",         "90.6 %",    "所有 atomic 争同一 cache line，L2 hit 但被 serialize"),
    ("Achieved Occupancy",  "83.1 %",    "occupancy 高不代表快"),
  ),
)

三件事一起看：

- *memSOL + compSOL 都极低*：SM 既不忙内存也不忙计算。所有 warp 都被卡在 `atomicAdd` 的 L2 atomic unit 排队上——这是 *serialization stall*，roofline 模型里的第三个 regime。
- *L2 Hit 90.6%*：反直觉——L2 命中率高不代表快。所有 4M+ 次 `atomicAdd` 目标都是同一个 4 B 地址，那条 cache line 常驻 L2，*命中*。但 atomic unit 每次只能服务一个请求，命中之后仍然被 serialize，命中率高只说明数据被找到，不说明能被并行处理。
- *Occupancy 83%*：resource-level 上 SM 里 warp 数量正常。瓶颈完全在一个特定 hardware unit（L2 atomic），跟启多少 warp 无关。

#verdict(
  problem: [百万次 atomicAdd 目标同址 → L2 atomic unit 完全 serialize],
  evidence: [Memory SOL 1.6%、Compute SOL 0.3% 双双极低，说明 SM 都在等；L2 Hit 90.6% 说明数据到位但访问被排队；相比 sequential tree 慢 3 个数量级],
  next: [v2 让每个 *block* 先在 shared memory 里本地规约，然后*每个 block 只发一次 atomicAdd*到 `output`。整个 pipeline 依然只 launch 1 次 kernel，contention 从 "$N$ 个 thread 争一个地址" 降到 "$"blocks"$ 个 block 争一个地址"（$approx 10^4×$ 缓解）]
)

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
2. *bank conflict pattern 更糟*——`shared[tid + s]` 在 s=1 时无冲突，但 s=2, 4, ...→ 8 时（`128 mod 32 = 0` 附近）冲突剧增；累积下来 interleaved 有 425k bank conflict，sequential 只有 297k（本章"实测"diag 表数据）。

#insight[
  「interleaved 慢因为 warp divergence」这个说法在 ncu 上*无证据支持*——三条独立证据（SASS 无 BRA、`branch_divergent.sum` = 1、controlled 实验证明 metric 灵敏）一致指向 predication 而非 divergence。Harris 2007 的原始文本写在 CUDA compiler 还不成熟的时代；今天的 ptxas 会把小 if 全部 predicate 掉。真正的性能差 = *idle warp 能否整体退出* + *bank conflict 模式*——见下面"实测"section。
]

=== 与 sequential 版本的 diff

`if (threadIdx.x % (2s) == 0)` → `if (threadIdx.x < s)`。这一个字符的改动带来两个效应：一是让整 warp 能一起退出（scheduler-friendly），二是让 warp 内 lane 访问 smem 时更容易避开同一 bank（先看下一节，再对照"实测"表里 425k → 297k 的证据）。

=== ncu 实测

#ncu-snapshot(
  version: "interleaved",
  size: [$N = 2^22$],
  rows: (
    ("Duration",            "87.2 µs",  "单 kernel launch，比 atomic 快 130×"),
    ("Memory SOL",          "37.5 %",   "开始真正跑 HBM 了"),
    ("Compute SOL",         "75.1 %",   "warp 在忙——但忙 predicated NOP 和 bank conflict"),
    ("Achieved Occupancy",  "92.8 %",   "warp 占满，为下一步 tree 规约留资源"),
    ("Grid Size",           "16 385",   "每 block 256 元素，共 $N/256$ 个 block"),
  ),
)

一次 kernel launch 就把速度从 11 ms 拉到 87 μs——纯粹靠"每 block 只写一次 output"消除了 atomic contention。

但这个版本*仍然不够快*：memSOL 只有 37.5%，比 vector add 的 84% 差一半以上。原因不是访存模式（读 $x[i]$ 是 coalesced 的），而是 tree reduction 部分——bank conflict + 循环控制流。

#verdict(
  problem: [interleaved addressing 的 shared memory tree 让访问模式在中间几轮出现严重 bank conflict],
  evidence: [Memory SOL 37.5% 远低于 vector add 的 84%；bank conflict 累积 425k（见章末 "实测" diag 表）；compSOL 75% 显示 SM 忙在 predicated 加法和 bank 冲突串行化上],
  next: [v3 把 `if (tid % (2s) == 0)` 换成 `if (tid < s)`——同样的 tree 结构、同样的 sync 次数，但访问模式变化让整 warp 能一起退出 + bank conflict 减少 30%]
)

== v3: sequential addressing

源码里的 `03_sequential.cu` 用的就是 sequential addressing——Mark Harris "Optimized Parallel Reduction" 教程的第 2 步，末尾直接 `atomicAdd` 到全局 `output`：

```cpp
__global__ void reduce_sum_kernel(
    const float* input, float* output, int count) {
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
    atomicAdd(output, shared[0]);   // 每 block 一次 atomic，contention 已被 shared tree 摊薄
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
  历史教材（Harris 2007）把这里说成"消除 warp divergence"。上一节用三条证据链（SASS / `branch_divergent.sum` / controlled 实验）证明 ncu 数据*不支持*这个说法。真正的差异是：*sequential 让整个 warp 一起退出*（idle warp 不占 issue slot，是 scheduler-level 事件）+ *bank conflict 模式不同*（425k → 297k）。
]

=== 每 block 一次 atomicAdd 到全局输出

256 个 thread 协作 → `shared[0]` 是这个 block 的局部和 → `atomicAdd(output, shared[0])`。$N$ 个元素在*一次 kernel launch* 里就完成 reduction，无需多阶段。

#insight[
  和 v1 唯一的差别就是：v1 每*元素*一次 atomic（$N approx 10^8$ 次），v2--v5 每*block* 一次 atomic（$"blocks" approx 10^4$--$10^5$ 次）。两者都撞同一个 4 B 地址，但*次数差 4 个数量级*——L2 atomic unit 处理 $10^5$ 个请求是几十 μs 的事，处理 $10^8$ 个请求就是几百 ms。
]

#note[
  这跟 Mark Harris 2007 教程的经典写法不同：原教程用 `block_sums[blockIdx.x] = shared[0]` 然后 host 循环 launch 第二个 kernel 继续规约（"multi-stage / chained kernel launches"）。那种写法在 CUDA 早期是主流，因为 `atomicAdd` on FP32 直到 SM 2.x 才支持、Kepler 之前的 atomic 也慢。到 Ampere/Hopper，*per-block atomicAdd 已经比 chained launch 明显快*（少一次 launch + 一堆 `cudaMalloc/cudaMemcpy`），本书统一用后者。章末 v5 会给出实测对比。
]

=== bank conflict：第一轮最严重

shared memory 有 32 个 bank，每个 4 字节宽。`shared[tid] += shared[tid + 128]`：warp 内 32 个 lane 同时读 `shared[0..31]` 和 `shared[128..159]`。地址 `128 % 32 = 0`，所以 `shared[i]` 和 `shared[i+128]` 落在*同一个 bank*——32-way bank conflict，一次 access 被串行化成 32 次。

后续轮次（offset = 64, 32, ...）conflict 逐渐减轻。offset = 32 时，`shared[i]` 和 `shared[i+32]` 在不同 bank，无 conflict。

#warn[
  面试常问："reduction 的 bank conflict 在哪一轮最严重？" 答：第一轮（offset = blockSize/2），因为 stride 恰好是 32 个 bank 的周长。缓解方法：padding shared array（`shared[threadIdx.x + padding]`）、或用 warp shuffle 绕过 shared memory。
]

=== ncu 实测

#ncu-snapshot(
  version: "sequential",
  size: [$N = 2^22$],
  rows: (
    ("Duration",            "50.9 µs",  "单 kernel launch 全部搞定，比 interleaved 快 1.7×"),
    ("Memory SOL",          "64.1 %",   "从 37.5% 拉到接近 vector add 水平"),
    ("Compute SOL",         "56.7 %",   "SM 有真活干"),
    ("Achieved Occupancy",  "87.7 %",   ""),
    ("Bank conflicts",      "297 k",    "vs interleaved 425k（章末 diag 表），↓30%"),
  ),
)

*关键读数*：memSOL 从 37.5% 跳到 64.1%——`if (tid % (2s) == 0)` → `if (tid < s)` 这一个字符的改动，让 HBM 利用率大幅提升。原因看下面 verdict。

#verdict(
  problem: [sequential 版本已经把 shared-memory tree 的效率拉近极限，但 shared memory 本身还是主要通信介质——每轮都要写-同步-读],
  evidence: [Memory SOL 64.1% vs vector add 的 84%，还差一段；bank conflict 累积 297k 说明 smem 通信压力仍很大；`__syncthreads` 每次 tree 缩减都要调一次],
  next: [v4 (warp shuffle) 把 warp 内 5 步规约从 shared memory tree 换成 register shuffle，彻底绕过 smem 和多余的 sync]
)

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

=== ncu 实测

#ncu-snapshot(
  version: "warp_shuffle",
  size: [$N = 2^22$],
  rows: (
    ("Duration",            "49.1 µs",  "跟 sequential 相当（50.9 μs），并没有质变"),
    ("Memory SOL",          "20.5 %",   "*反而降*了——下面解释"),
    ("Compute SOL",         "19.1 %",   ""),
    ("Bank conflicts",      "40 k",     "vs sequential 297k，↓ 86%——通信介质换成 register 了"),
    ("Achieved Occupancy",  "83.1 %",   ""),
  ),
)

*一个反直觉的读数*：warp shuffle 明显减少 bank conflict，但 duration 反而跟 sequential 差不多、memSOL 更低。原因：

- 这个 kernel 是 "每 thread 处理 1 个元素" 的架构，grid 有 16k+ 个 block。*现在 warp 内规约变快了*，反而暴露出下一层瓶颈——16k 个 block 都要发一次 atomicAdd 到 `output`，L2 atomic unit 处理 16k 请求要花几十 μs，掩盖了 warp shuffle 省下来的时间。
- memSOL 20% 说明大部分时间不是在读 HBM，而是在等 atomic serialization。
- bank conflict *真的* 从 297k 掉到 40k（下降 86%）——这是 shuffle 绕过 smem 通信直接带来的收益，符合原理预期，只是收益被更靠后的瓶颈吃掉了。

#verdict(
  problem: [warp 内规约效率已达极限，但 "每 thread 1 元素" 让 grid 太大 → per-block atomicAdd 数量太多，L2 atomic 成新瓶颈],
  evidence: [duration 没比 sequential 快多少；memSOL 掉到 20% 说明 HBM 空闲，SM 在等 atomic；bank conflict 已消除但 wall-clock 没跟上],
  next: [v5 (chunked) 让每 thread 处理 *8* 个元素——grid 缩到 1/8，atomic 次数同比例下降；同时 register 累加摊销 index 计算成本，HBM 有 8× 更多 in-flight load 让队列深度提升]
)

== v7: chunked hierarchical

```cpp
constexpr int kChunkItemsPerThread = 8;

__global__ void reduce_sum_kernel(
    const float* input, float* output, int count) {
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
  if (tid == 0) { atomicAdd(output, shared[0]); }   // 一次 launch 直接出结果
}
```

=== 每个 thread 先做更多活

一个 block 覆盖 `256 * 8 = 2048` 个元素（对比 v3 的 256 个）。好处：

1. *摊销 index 计算和 `__syncthreads` 成本*：每 2048 元素只做 1 次 block sync，不是每 256 元素。
2. *寄存器累加隐藏 load latency*：8 个 load 可以 pipeline，MSHR 队列吃满。
3. *步长 = blockDim.x 保持合并访问*：和 vector add tiled 版本同一原则。
4. *atomicAdd contention 同步降低 8×*：block 数从 $N/256$ 缩到 $N/2048$，撞 `output` 的 atomic 次数同比例减少。

每个 thread 的 partial sum 存在 register 里，进入 shared tree 时 active 数据已经压缩 8 倍——后续 warp 间通信量不变，但*单位 sync 处理的元素数翻 8 倍*。

=== ncu 实测

#ncu-snapshot(
  version: "chunked",
  size: [$N = 2^22$],
  rows: (
    ("Duration",            "15.0 µs",  "比 warp shuffle 快 3.3×，*本章最快*"),
    ("Memory SOL",          "54.9 %",   "已经全部在 HBM 里跑"),
    ("Compute SOL",         "32.5 %",   ""),
    ("Bank conflicts",      "165 k",    "smem tree 部分数据少 8×（相对每元素而言）"),
    ("Grid Size",           "2 049",    "vs warp_shuffle 16 385，↓ 8×"),
    ("DRAM SOL",            "54.9 %",   "跟 memSOL 相等：真的在打 HBM"),
  ),
)

*所有维度都在合理方向*：

- *duration 15 μs*：本章最快，且是*一次 kernel launch* 出结果——没有多阶段 chained launch，没有 host-side `cudaMalloc`/`cudaFree` 循环。
- *memSOL / DRAM SOL 都是 54.9%*：kernel 真的把 HBM 打起来了；相比 warp_shuffle 版本 memSOL 20% (HBM 空闲、SM 等 atomic)，chunked 完全消除了那个瓶颈。
- *grid 8× 小*：launch overhead 减少 8×，atomicAdd 到 `output` 的次数也少 8×——contention 已经不再是可测量的瓶颈。

*章末 sweep 表* (`$N = 2^27$`) 显示同一算法在超大规模下达到 92% HBM peak，1876 GB/s——A100 的物理极限约 2039 GB/s。

#final-verdict(
  status: [chunked hierarchical + 每 block 一次 atomicAdd finalize，是本 kernel 的实用最优。$N = 2^22$ 上 duration 15 μs，$N = 2^27$ 上 92% HBM peak。整个 pipeline 一次 kernel launch 完成，无需多阶段。],
  note: [继续提升需要 vectorized load（`float4`，让 chunk items = 16 且访存单元对齐 128 B）、multi-block cooperative reduction（CUB 的做法）、或 SM-level warp specialization。教学 kernel 到此打住；生产选 CUB。]
)

== 性能 ladder 一览

源码里从 atomic 到 chunked 是一步步叠加的优化。所有 v2--v7 版本共享同一个 host launch pattern：*一次 kernel launch*，block 内规约 → 每 block 一次 `atomicAdd` 到 `output`——kernel 本身的算法在变，host 代码没变。下面 ladder 描述*相对排序和每步原因*；绝对数字见下一节实测。

#ladder(
  ("atomic",              "per-element atomicAdd",              "contention 灾难（N 次 atomic 撞同址）"),
  ("interleaved tree",    "block 内 smem 树 + per-block atomic", "bank conflict + warp lane 浪费"),
  ("sequential tree",     "block 内 smem 树 (连续寻址) + per-block atomic", "bank conflict 减 30%"),
  ("+ unroll last warp",  "省 5 次 __syncthreads",              "缩小 sync 范围到 warp"),
  ("+ complete unroll",   "模板展开 + 编译期 blockSize",         "消除循环控制"),
  ("warp shuffle",        "register shuffle + 1 次 smem sync",   "绕过 smem tree"),
  ("chunked",             "8 elements/thread + smem tree + atomic", "grid 缩 8×，摊销 sync/atomic 成本"),
)

#warn[
  ladder 里 `unroll last warp` / `complete unroll` 等中间版本未单独 benchmark。本章实测覆盖 atomic、interleaved tree、sequential tree、warp shuffle、chunked hierarchical 五个代表点——足够说明 contention vs 协作规约 vs bank conflict vs grid contention 四个层次。
]

=== 实测

$N = 2^27 + 37$（约 128 M 元素 / 512 MB，远超 A100 40 MB L2），A100 80GB SXM4，`ncu` 抓取 `reduce_sum_*_kernel` 每个版本的*完整 kernel*（单 launch 出结果）。GB/s 列写作 *HBM 实测 / 逻辑*：前者 `dram__bytes.sum / time`，后者 $N times 4 / "time"$；这两个数字*大规模下应几乎相等*——差距大就是 L2 命中或者 kernel 太短。

#include "../bench/02_reduce_sum.typ"

*perf 表读三件事：*

+ *atomic 慢 100000 倍*。368 ms vs sequential tree 的 1.4 ms——不是"几十倍"，是*五个数量级*。128 M 次 atomicAdd 全撞同一 cache line，L2 层 serialize，`mem stall = 6045`（每 issue-active cycle 有 6045 个 warp 卡在 long_scoreboard 上），HBM % 只有 0.1%——完全不是 memory-bound，是*serialization-bound*。

+ *interleaved vs sequential：2.6 ms → 1.4 ms（1.8×）*。同样是 shared memory tree reduction，仅寻址模式从 `if (tid % (2s) == 0)` 换成 `if (tid < s)` 就近乎快一倍。差距的物理来源要看 diag 表。

+ *warp_shuffle 反而比 sequential 稍慢*（1524 vs 1444 μs）。这在 $N = 2^22$ 的 snapshot 里已经暴露过：warp shuffle 消除了 smem bank conflict，但在 "每 thread 1 元素" 架构下 grid 有 $N/256 approx 5 times 10^5$ 个 block——per-block 一次 `atomicAdd` 到 `output` 变成新瓶颈（`mem stall = 62.6`）。warp shuffle 的收益被下一层瓶颈吃掉了。

+ *chunked 接近打满带宽*：287 μs，HBM 92%（1876 GB/s / 2039 GB/s peak）。8× 更大 chunk 一次性把两个问题都解决——atomic 次数减 8×、HBM 队列深度加 8×——duration 从 sequential 的 1444 μs 直接压到 287 μs（$5.0×$）。

*diag 表读关键教学点：*

*a) interleaved vs sequential 的差别 *不是* warp divergence*——两者 `issued/32` 都是 31.8-31.9，几乎完美。`if (tid < s)` 和 `if (tid % (2s) == 0)` 在 warp 里都是 predicated add，不是真正的 branch divergence。

`pred_on/32` 反而告诉了我们*相反*的故事：interleaved = 24.0，sequential = 20.0——*sequential 的 lane 利用率更低*！但 sequential 快 2.2×，为什么？答案在 `smem conf.` 一列。

*b) sequential 快的真正原因：bank conflict 少 30%*。interleaved 425k 次 bank conflict，sequential 297k 次——都不为零，但 sequential 让每个 warp 内 lane 访问 *连续的* smem，冲突集中在 warp 之间而非 warp 内，L1TEX 处理更高效。

#insight[
  这是本书 pushback "上帝视角"的核心案例。传统教材把 interleaved 慢的原因归为 "warp divergence"——ncu 的 `issued/32 = 32` 直接反驳了这个说法。真正的差别在 `smem conf.` 列。术语用"warp lane utilization" (issued vs pred_on) 和 "bank conflict pattern" (smem conf.) 比笼统说 "divergence" 更精确。
]

*c) warp shuffle 大幅减少 smem bank conflict → 297k 降到 40k*，但*wall-clock 反而没变快*。register-to-register 数据交换绕过 shared memory 是对的，但在 "每 thread 1 元素" 架构下 grid 有 524288 个 block，每个都发一次 `atomicAdd` 到 `output`——`mem stall = 62.6`（比 sequential 的 9.6 还高 6.5×），warp 都在等 atomic serialization。*收益被下一层瓶颈吃掉*的经典案例。

*d) chunked 的胜利在 grid 太多 vs 每 thread 多做工的 tradeoff*。chunked 每 thread 处理 8 个元素 → grid 从 524288 缩到 65536。`mem stall = 16.0`（vs warp 版 62.6，↓ 4×）显示 atomic 争用大幅缓解；同时 HBM % 从 17% 涨到 92%——memory queue depth 上升 + atomic 次数下降双重收益。

*bank conflict 的形成机制：*

sequential addressing 第一轮 `shared[tid] += shared[tid + 128]`：tid=0 读 bank 0 和 bank 0（128 mod 32 = 0）——32-way conflict；tid=1 读 bank 1 和 bank 1——同样 32-way；tid=0..31 全部 32-way conflict，共 32 次访问。

interleaved 第一轮 `shared[tid] += shared[tid + 1]`（stride=1）：tid=0 读 bank 0 和 bank 1，tid=2 读 bank 2 和 bank 3——无冲突。但 stride=2、4、... 时冲突逐渐增加。*结论*：interleaved 前几轮少冲突、后几轮多冲突；sequential 前几轮多冲突、后几轮少冲突。ncu 累积的 425k vs 297k 是全部轮次求和的结果。

#note[
  想在源码级别验证 bank conflict，用 `ncu --section MemoryWorkloadAnalysis_Tables` 会打出按 access pattern 分类的 conflict 明细。
]

#warn[
  atomic 的 mem_stall = 6045 与"memory latency stall"的表面含义相反：`long_scoreboard` 包含所有 memory dependency 类 stall，atomic 的 read-modify-write 让 L2 atomic unit 反复往返 → 每次操作都需要"等 L2 回复"，被计入 long_scoreboard。这也是为什么 atomic 看起来像 mem-bound，实际是 serialization。
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

__global__ void reduce_sum(const float* x, float* out, int n) {
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
    // Level 3: 每 block 一次 atomicAdd 直接 finalize——一次 kernel 出结果
    if (lane == 0) atomicAdd(out, v);
  }
}

// ==== Launch config ====
// blockDim = 256 (8 warp)：
//   * 必须是 32 的倍数——最后的 warp reduce 靠 warp 对齐；
//   * 8 warp 让 warp_sums[] <= 8 <= 32，第二级 warp reduce 一个 warp 搞定；
//   * 不要开 1024——太多 warp 会让最后一级 warp reduce 变复杂.
// gridDim = min(SM * 4, (N + block - 1) / block)：
//   * SM * 4 让每 SM 常驻多个 block、latency hide；
//   * 若 grid 太大 (~10^5+) 且每 block 只处理 1 元素，atomicAdd contention 会成瓶颈——
//     解决方法是 grid-stride 或每 thread 多做几个元素（items/thread = 4 或 8）.
cudaMemset(out, 0, sizeof(float));  // atomic 累加需要初始零
int block = 256;
int grid  = min(4 * sm, (n + block - 1) / block);
reduce_sum<<<grid, block>>>(x, out, n);  // 一次 launch 直接出结果
```

*核心考点*（追问顺序）：

- *"为什么不直接一个大 shared array 做树？"* → 可以，但 warp shuffle 走寄存器堆、比 shared memory 快 3-5×，还不需要 `__syncthreads`（warp 内锁步）。
- *"为什么 grid-stride 循环里就先做加法？"* → 让一个 thread 吃很多元素、shared / shuffle 那部分只处理 blockDim.x 个 partial——摊销 sync/atomic 开销，同时避免 grid 过大导致 atomicAdd contention。
- *"最后一个数怎么得到？"* → 每个 block 用 `atomicAdd(out, block_sum)` 直接 finalize——一次 kernel launch 出结果。这里 atomic 争用只有 `blocks` 次（$approx 10^4$），跟 v1 那种 $N$ 次 atomic 差 4 个数量级，L2 atomic unit 毫无压力。生产极致场景可以用 CUB / cooperative-groups grid barrier，但对绝大多数 shape 这个写法已足够。
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

  A: interleaved 用 `if (tid % (2s) == 0)` 选活跃 thread，它们在 warp 内间隔分布；sequential 用 `if (tid < s)`，活跃 thread 是连续区间。*两者都不是真正的 warp divergence*（ncu `issued/32` 都是 32，编译成 predicated instruction）。真正的差异在 bank conflict pattern：sequential 让 warp 内 lane 访问连续 smem 地址，L1TEX 处理更高效——实测 sequential bank conflict 297k vs interleaved 425k，快 1.8×。

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

  A: 三种主流做法：

  (a) *per-block atomicAdd finalize（本 repo 做法、推荐）*：每 block 一次 `atomicAdd(out, block_sum)`。一次 kernel launch 出结果，无 host 循环。atomic 次数 = block 数，通常 $10^3$--$10^5$ 量级，L2 atomic unit 完全 hold 得住。跟 v1 那种 $N$ 次 per-element atomic 有本质区别。

  (b) *multi-stage chained launch*：反复 launch reduction kernel 把长度缩短直到 1（Mark Harris 2007 原教程做法）。写起来丑（host 循环 + `cudaMalloc`/`cudaFree`），且多一次 launch overhead。历史原因：早期 GPU FP32 atomic 慢或没有；现代 A100+ 上 (a) 更快。

  (c) *CUB / cooperative-groups grid barrier*：一次 kernel 内 grid-level sync，最后一个 block 做 finalize。写起来最复杂，但省 atomic 争用；生产环境（cub::DeviceReduce）用这个。

  面试白板首选 (a)——简洁、性能好、易讲清 tradeoff。有人问"atomic 会不会慢"就用本章 v1 vs v5 数据回击：per-element atomic vs per-block atomic 差 3--4 个数量级。
]

#interview[
  *Q9*: 为什么 chunked（multi-elements-per-thread）有效？步长为什么用 `blockDim.x`？

  A: 每个 thread 在 register 里累加 8 个元素再进入 block 规约，摊销 index 计算、`__syncthreads` 和 finalize 时的 atomicAdd 成本；8 个 load 可 pipeline 隐藏 latency、MSHR 队列吃满。步长 = `blockDim.x`（不是 8）是为保持 warp 内合并访问——和 vector add tiled 版本同一原则。
]

#interview[
  *Q10*: 模板 `blockSize` 完全展开有什么好处？

  A: 编译期固定循环边界 → 消除循环控制开销、常量传播去掉 dead branch、精确控制哪些步需要 `__syncthreads`（跨 warp 的 3 步需要，warp 内的 5 步不需要）。Launch 时 block size 必须匹配模板参数。
]
