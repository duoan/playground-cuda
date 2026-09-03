#import "../template.typ": *

= Vector Add

vector add 是 CUDA 的 "hello world"，但它的正确用法不是当例子随便过一遍——而是作为*一整套 GPU 思维方式的入门载体*。这一章我们要把它讲透：

- 为什么 naive 版本已经能跑到接近峰值带宽（下一章 reduction 就不能）。
- grid-stride 这个模式在生产代码里为什么无处不在，它的 *性能收益是 0*，但仍然值得写。
- `float4` 宽访问为什么有效，什么时候它反而变成陷阱。
- kernel launch 的开销结构，为什么把主体和尾巴 fuse 成一个 kernel 是有意义的。

对应源码：`src/cuda/01_vector_add.cu`。

== 问题定义

给定长度为 $N$ 的三个数组 $a, b, c$，计算：

$ c[i] = a[i] + b[i], quad i = 0, 1, ..., N-1 $

关键性质：*完全无依赖*、*每元素 1 次加法*。这决定了它的性能特征。

=== Roofline：这是个 memory-bound kernel

算一下算术强度 (arithmetic intensity, AI)：每处理 1 个输出元素，需要：

- 读 $a[i]$：4 字节
- 读 $b[i]$：4 字节
- 写 $c[i]$：4 字节
- 计算：1 次 FADD

$ "AI" = frac(1 "FLOP", 12 "B") approx 0.083 "FLOP/B" $

A100 80GB SXM4 HBM 带宽约 2.04 TB/s，FP32 峰值算力约 19.5 TFLOPS。它的 balance point（Ridge Point）在：

$ frac(19.5 "TFLOPS", 2.04 "TB/s") approx 9.6 "FLOP/B" $

我们的 AI 是 0.083，比 ridge point 低两个数量级——*妥妥的 memory-bound*。

#insight[
  memory-bound 的直接后果：*所有优化的目标只有一个——尽可能压满 HBM 带宽*。任何"减少计算量"的优化在这里都没用；任何"多做几次冗余读"的错误都会立即体现在性能上。
]

理论上界：

$ T_"min" = frac(3 N times 4 "B", 2.04 "TB/s") $

- $N = 2^20$（4 MB per array，12 MB 总量）：$T_"min" approx 6 mu s$。但 A100 L2 = 40 MB，$3 N = 12 "MB"$ *完全装得下 L2*——第二次读几乎不到 HBM。所以我们不能拿这个规模来判定 HBM 利用率。
- $N = 2^27$（512 MB per array，1.5 GB 总量）：$T_"min" approx 750 mu s$。这个规模才真正打满 HBM，也是本书用来做 vector add benchmark 的规模。

任何比对应规模下 $T_"min"$ 慢 2 倍以上的实现，都存在明确的可优化点。

== v1: naive

```cpp
__global__ void vector_add_naive_kernel(
    const float* a, const float* b, float* c, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    c[i] = a[i] + b[i];
  }
}
```

Launch：

```cpp
const int blocks = (count + 255) / 256;
vector_add_naive_kernel<<<blocks, 256>>>(a, b, c, count);
```

一个 thread 处理一个元素。就这么简单。

=== 为什么它已经很好

三件事都做对了：

*1. 合并访问 (coalesced access)*

warp 里 32 个 lane 的 `i` 是 `..., 32k, 32k+1, ..., 32k+31`——连续 32 个 int。访问 `a[i]` 就是访问 128 字节连续内存。GPU 的 memory controller 把这 32 个请求合并成 *一个* 128B 的 memory transaction。

#figure(
  mem-access(mapping: range(32), n-words: 32,
             title: "coalesced：lane i 访问 word (base + i)",
             caption: "红色实线框 = 触发的 128B transaction；32 个 lane 落在 1 个 transaction 里。"),
  caption: [绿色 = 参与访存的 lane；黄色 = 被访问的 4B word；蓝色箭头 = lane→word 的映射。所有 32 个 lane 的地址连续，memory controller 用 1 次 128B transaction 全部搬完。]
)

如果 stride 不对（比如让 lane 0 访问 `a[0]`、lane 1 访问 `a[32]`、lane 2 访问 `a[64]`...），32 个 lane 的地址跨越 32 个不同的 128B 段——就要拆成 *32 个独立 transaction*，带宽利用率立刻掉到 $1/32$。

#figure(
  mem-access-scattered(
    lane-words: range(8).map(i => i * 32),
    n-lanes: 8,
    title: "uncoalesced：lane i 访问 word (i × 32)"),
  caption: [段间的 `...` 表示省略的 28 个未访问 word（每 lane 之间跨度 128 B）。8 个 lane 触发 *8 个独立的* 128B transaction（红框），每个 transaction 只用 1 个 4B word——其余 124 B 被浪费。32 个 lane 会需要 32 个 transaction，浪费 31/32 的带宽。]
)

#insight[
  合并访问的物理机制：*memory controller 一次最少搬 32B 或 128B*（sector / transaction 粒度）。coalesced 情况下 32 个 lane 的 128B 需求正好等于 1 次 transaction，*每字节都被真需要*；uncoalesced 情况下每次搬 128B 但只用其中 4B，浪费 96.9% 的传输量。
]

#note[
  合并访问是 CUDA 的第一性能法则。写完 kernel 后第一件要问自己的事：*同一个 warp 里的 32 个 lane，访问的地址是不是尽可能连续？*
]

*2. 没有 warp divergence*

`if (i < count)` 只有最后一个 block 里的部分 warp 会分裂。绝大多数 warp 里 32 个 lane 走同一条路径，SIMT 执行没有浪费。

*3. 占用率 (occupancy) 足够*

`vector_add_naive_kernel` 用 16 个寄存器（`ncu` 报告 `launch__registers_per_thread`）、0 shared memory。A100 一个 SM 支持 *2048 threads / 64 warps*，寄存器（64 K per SM）和 smem 都不是瓶颈，occupancy 接近 100%。

=== 实测

$N = 2^27$（每个数组 512 MB，总量 1.5 GB $ ≫ $ L2 = 40 MB），A100 80GB SXM4，`ncu` 抓取，`--binary-args "27"`。表中「HBM GB/s」列写作 *实测 / 逻辑*：前者是 `dram__bytes.sum / time`（ncu 实测的 HBM 传输量），后者是 $3 N dot 4 / "time"$（如果每字节都真去 HBM 拿一次的理论带宽）。这两个数字在 L2 装得下的小规模里差别巨大；在大规模下应当相互吻合。

#include "../bench/01_vector_add.typ"

*先看第一张 perf 表：*

- 「实测 / 逻辑」两列*几乎完全一致*（naive: 1706/1715, vectorized: 1825/1837）——L2 命中率极低，*我们真的在打 HBM*。
- naive 达到 *83.7% HBM peak*（1706 GB/s vs 2039 GB/s）——vector add 的 naive 已经是"接近打满带宽"的实现。这是本章第一个反直觉的结论：加优化 ≠ 显著提升。
- `vectorized` / `vectorized-fused` 达到 *89.4-89.5%*，比 naive 高 5.8 个百分点。收益来自 `LDG.E.128` 一次搬 16 字节而非 4 字节，让 memory pipeline 上每 warp 的 outstanding request 携带更多数据（同 warp 的 32 lane × 128 bit = 512 B per LDG.E.128 vs. 32 × 32 bit = 128 B per LDG.E）。
- `grid-stride` 反而*比 naive 慢 3%*，`tiled` 与 naive 打平。这直接印证我们之前说的：grid-stride 的价值在*通用性*（可以处理任意大 $N$，不受 grid 上限约束），不在性能。

*再看第二张 diag 表：*

- `issued/32 = 32.0`：warp 里 32 个 lane *全都* 参与每条指令。所有 vector add 变体都是这样——没有 `if`，没有分支，硬件无需 predication mask。
- `pred_on/32 = 27.8-30.9`：有 1-4 个 lane 是 predicated-off。这些是 grid-stride 的 loop tail、以及 vectorized-fused 的 tail-scalar 部分。看起来"效率下降"，其实*完全不影响性能*——因为这一切的瓶颈是 HBM，不是 SM 发射带宽。
- `smem conf. = 0`：没有 shared memory 使用，理所当然。
- `barrier stall = 0`：没有 `__syncthreads`。
- `mem stall = 82-296`：*非常大*。这是关键——`smsp__average_warps_issue_stalled_long_scoreboard`，意思是每 issue-active cycle 有 82-296 个 warp *正在等 global memory*。这就是 memory-bound 的定量证据。

#insight[
  `mem stall` 是判断 memory-bound 的第一手证据。如果这个数字大（几十到几百），说明 warp 大量时间在等 HBM 回来。搭配 `HBM %` 一起看：mem_stall 高 + HBM % 高 → 已经打满 HBM，别再想减少访存以外的优化了；mem_stall 高 + HBM % 低 → 访存模式有问题（未合并、bank conflict、非对齐），有优化空间。
]

*为什么 vectorized `mem stall = 296` 比 naive `82` 高，但速度更快？*

一个 warp 用 LDG.E.128 拿 512 B，需要*一次*长延迟等待；用 LDG.E 只拿 128 B，也需要一次长延迟等待。以每条 issued 指令的 "warps waiting" 归一化，vectorized 每个 issue slot 上等 memory 的 warp 密度*更高*——因为总 issue 数量少了（4 倍），但等 memory 的绝对时长没变。*这个 metric 不能孤立解读*，得配合 HBM % 才能得出"vectorized 更快"的结论。

#warn[
  A100 datasheet 的 HBM peak 2039 GB/s 是理论上界，实测受限于 memory scheduler、地址冲突、read/write 比例。工程上把 *~85-90% peak* 视作 "已经打满"。想过 95% 需要单纯读或单纯写（不像我们这里 2 读 + 1 写）。
]

#warn[
  以上是 vector add 的特殊性——它的访存模式对硬件最友好。下一章 reduction 我们会看到 naive 到最终版本几十倍差距，不要把 "vector add naive 就够快" 推广到别的 kernel。
]

== v2: tiled (multi-items-per-thread)

```cpp
constexpr int kTiledThreadsPerBlock = 128;
constexpr int kItemsPerThread = 4;
constexpr int kTileSize = kTiledThreadsPerBlock * kItemsPerThread;  // 512

__global__ void vector_add_tiled_kernel(
    const float* a, const float* b, float* c, int count) {
  const int tile_start = blockIdx.x * kTileSize;
  const int thread_base = tile_start + threadIdx.x;

  #pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int i = thread_base + item * kTiledThreadsPerBlock;
    if (i < count) {
      c[i] = a[i] + b[i];
    }
  }
}
```

*核心变化*：一个 thread 处理 4 个元素，一个 block（128 threads）处理一个 512 元素的 tile。

=== 步长为什么是 `kTiledThreadsPerBlock`，不是 4

关键在于*保持 warp 内的合并访问*。看两种写法：

*错误写法*：每 thread 处理 4 个*连续*元素。
```
thread 0: 0, 1, 2, 3
thread 1: 4, 5, 6, 7
thread 2: 8, 9, 10, 11
...
```

第 0 轮迭代（`item=0`），warp 内 32 个 lane 访问 `[0, 4, 8, ..., 124]`——跨 32×4=128 个元素但只用 32 个，*破坏合并*。

*正确写法*（本 kernel）：每 thread 步长 `blockDim.x`。
```
thread 0: 0, 128, 256, 384
thread 1: 1, 129, 257, 385
thread 2: 2, 130, 258, 386
...
```

第 0 轮，warp 访问 `[0, 1, ..., 31]`——连续，合并访问。第 1 轮 `[128, 129, ..., 159]`——还是连续。

#insight[
  多元素-per-thread 的核心原则：*让 warp 内 32 个 lane 在每一轮迭代里都保持连续访问*。step 应该等于 `blockDim.x`（对单维 block），不是 `kItemsPerThread`。
]

=== 为什么它对 vector add 也不显著变快

理论上，多元素-per-thread 能：

1. *摊销 index 计算*：`blockIdx.x * blockDim.x + threadIdx.x` 只算一次。
2. *让编译器有机会用更宽的 load*：`#pragma unroll` 展开后，nvcc 有时能合并成 128-bit load。
3. *隐藏访存延迟*：多个 outstanding load 同时飞在管线里。

但 vector add naive 已经打满带宽了，这些优化空间为 0。tiled 版本这里主要作为*思维训练*——学习"一个 block 系统性处理一块数据"的分工方式。matmul 里 shared memory tiling 用的是同一个骨架。

== v3: grid-stride loop

```cpp
__global__ void vector_add_grid_stride_kernel(
    const float* a, const float* b, float* c, int count) {
  const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = i0; i < count; i += stride) {
    c[i] = a[i] + b[i];
  }
}
```

看起来只是加了个循环。实际上这是 CUDA 里*最重要的一个模式*。

#note[
  三行代码里的 `blockIdx.x / blockDim.x / gridDim.x / threadIdx.x` 是什么？为什么 `stride = blockDim.x * gridDim.x`？如果不熟，翻回上一章「CUDA 基本概念速查」——那里有一张图专门画 grid-stride 的 stride 从哪来。]

=== 它解耦了 grid 大小和数据大小

naive 版本：`blocks = ceil(count / 256)`。如果 `count = 1e9`，就是 4M 个 block。

grid-stride：*你想开几个 block 就开几个*。常见做法：

```cpp
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
const int blocks_per_sm = 8;  // occupancy 计算或试出来的
const int blocks = num_sms * blocks_per_sm;
vector_add_grid_stride_kernel<<<blocks, 256>>>(a, b, c, count);
```

四个直接好处：

*1. Launch 开销固定*。kernel launch 有 ~5 μs 的固定成本。数据规模翻 100 倍，naive 版本 block 数也翻 100 倍，但 launch 本身还是那几微秒——不影响。真正被影响的是*runtime queue 长度*和*grid 遍历开销*，不过对 A100 这种 GPU 数百万 block 都能吞。

*2. 绕过 `gridDim.x` 上限*。`gridDim.x` 硬件上限是 $2^31 - 1$，看似很大，但如果你在 y、z 两个维度上还有 block（比如 batch 维），单维就容易超。grid-stride 天生不受这个限制。

*3. Occupancy 精确对齐硬件*。你开 `num_sms * blocks_per_sm` 个 block，恰好把 SM 填满，不多不少。naive 版本 block 数由数据决定，可能是 SM 数的几万倍——多出来的 block 只是排队等前面的做完，没有并行度收益。

*4. 数据形状无关*。库代码里必须这么写——你不知道用户会传多大数组。

=== stride 方向的讲究

`stride = blockDim.x * gridDim.x` 保证 warp 内每一轮迭代都连续访问：

- 第 0 轮：thread 0..31 访问 `a[0..31]`
- 第 1 轮：thread 0..31 访问 `a[stride..stride+31]`

如果写反了（`for (int j = 0; j < K; ++j) { c[i0*K + j] = ... }`），warp 就会跨越大的地址范围，破坏合并。

#warn[
  grid-stride 这个模式*本身不会让 kernel 变快*。vector add 上实测 naive ≈ grid-stride（甚至因为循环开销略慢）。它的价值是*通用性*：一份代码搞定任意数据规模，任意 GPU 型号。生产代码基本都用这个骨架。
]

== v4: 向量化 load/store

```cpp
__global__ void vector_add_vectorized_kernel(
    const float4* a4, const float4* b4, float4* c4, int vector_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vector_count) {
    const float4 lhs = a4[i];
    const float4 rhs = b4[i];
    c4[i] = make_float4(
        lhs.x + rhs.x, lhs.y + rhs.y,
        lhs.z + rhs.z, lhs.w + rhs.w);
  }
}
```

把 `float*` 重解释成 `float4*`，一次读写 128 bit。这一步在有些 memory-bound kernel 上能提 30~50% 性能。vector add 本身已经打满，提升有限，但机制值得讲清楚。

=== 编译器视角：LDG.E.128

`float4` 是编译器认识的 built-in type，字段对齐到 16 字节。上面的 kernel 编译出来的 SASS 里，`a4[i]` 变成一条 `LDG.E.128` 指令——单条指令加载 128 bit。

对比 naive 版本的三次单独 `LDG.E.32`（如果编译器没合并），指令数是 1/4。指令 fetch/decode 是有代价的，虽然对 memory-bound kernel 不显著，对 compute-bound 或 instruction-bound kernel（比如 softmax 里的大量除法）就有影响。

=== 硬件视角：memory transaction 数

一个 warp（32 lane）用 `float4` 每 lane 加载 16B，总共 32 × 16 = 512B。这被拆成 4 个 128B 的 memory transaction（HBM 的自然粒度）。

一个 warp 用 `float` 每 lane 加载 4B，总共 32 × 4 = 128B。1 个 128B transaction。

*两种情况处理相同字节数需要的 transaction 数是一样的*（都是 4 transactions / 128B = 1 transaction）。所以 memory-bound 上限不变。收益来自：

1. 减少的指令数 → 稍微少一点 instruction issue 压力。
2. 每个 outstanding request 携带更多数据 → 更容易饱和 MSHR（Miss Status Holding Register）等硬件队列。

#insight[
  向量化的收益在 MSHR/L2 miss handling 队列容量有限时最明显。对纯 HBM-bound 且 warp 数量足以掩盖延迟的 kernel（vector add 就是），收益可能只有 5% 甚至 0。对访存模式复杂、latency 难以掩盖的 kernel（stride 大、cache miss 高），提升可能到 50%。
]

=== 对齐要求

`float4` 需要 16 字节对齐的地址。`cudaMalloc` 保证至少 256 字节对齐，从数组开头 reinterpret 到 `float4*` 安全。

但如果你要从数组中间某个 offset 开始向量化：

```cpp
auto* p4 = reinterpret_cast<float4*>(a + offset);  // 危险
```

只有 `offset % 4 == 0` 时才对齐。否则会触发 `misaligned address` 错误（或者更糟：在某些老架构上静默降级到多次窄访问）。

=== 尾巴问题

`count` 不是 4 的倍数怎么办？例如 `count = 10`：主体处理 `count / 4 = 2` 个 `float4`（下标 0..7），剩下 `10 - 8 = 2` 个 float（下标 8, 9）不能用 `float4` 读。

本节 kernel 只处理完整 `float4` 前缀，尾巴由 host 侧另起一个 scalar kernel 处理。这个思路清晰，但要两次 launch。下一节 fuse 起来。

== v5: fused vectorized + tail

```cpp
__global__ void vector_add_vectorized_fused_kernel(
    const float* a, const float* b, float* c, int count) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int n_vec = count / 4;
  const int n_tail = count - n_vec * 4;
  const int tail_offset = n_vec * 4;

  if (tid < n_vec) {
    const auto* a4 = reinterpret_cast<const float4*>(a);
    const auto* b4 = reinterpret_cast<const float4*>(b);
    auto* c4 = reinterpret_cast<float4*>(c);
    const float4 lhs = a4[tid];
    const float4 rhs = b4[tid];
    c4[tid] = make_float4(
        lhs.x + rhs.x, lhs.y + rhs.y,
        lhs.z + rhs.z, lhs.w + rhs.w);
  }

  if (tid < n_tail) {
    const int i = tail_offset + tid;
    c[i] = a[i] + b[i];
  }
}
```

=== 关键观察

`n_tail` 永远 $in {0, 1, 2, 3}$。所以*前 4 个 thread* 就足以覆盖尾巴。

grid 只需按 `n_vec` 开——尾巴由已存在的 tid=0..3 顺手处理。tid=0 在这个 kernel 里干了两件事：一次 float4 加法 + 一次 scalar 加法。

例：`count = 10`，`n_vec = 2`，`n_tail = 2`，`tail_offset = 8`。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (center, center, center),
    [*tid*], [*主体做啥 (tid < 2)*], [*尾巴做啥 (tid < 2)*],
    [0], [处理 float4[0] = a[0..3]+b[0..3]], [处理 a[8]+b[8]],
    [1], [处理 float4[1] = a[4..7]+b[4..7]], [处理 a[9]+b[9]],
    [2], [—], [—],
    [3+], [—], [—],
  ),
  caption: [*Table:* v5 fused vectorized + tail kernel 在 $"count" = 10$（$n_"vec" = 2$、$n_"tail" = 2$、$"tail"_"offset" = 8$）时各 thread 的分工。*tid* 为 block 内 thread 下标；*主体* / *尾巴* 列分别对应 `if (tid < n_vec)` 与 `if (tid < n_tail)` 两个独立 guard 的激活范围与写入目标。],
  kind: table,
)

*Observation*：表里只有 tid 0、1 在两个分支上都「有活干」——主体用 tid 0–1 覆盖 8 个 float4 元素，尾巴用同一对 tid 覆盖下标 8–9；tid 2 起两个 `if` 都不进。这正是 $n_"tail" <= 3$ 时 grid 按 $n_"vec"$ 开仍够用的原因：前几个 thread 天然空闲，顺手吃掉尾巴，无需为 tail 单独扩 grid 或再 launch 一次 scalar kernel。

=== 为什么两个 `if` 不能合成 `if-else`

新手常犯错：

```cpp
if (tid < n_vec) {
  // body
} else {
  // tail  ← 错！
}
```

`else` 会让所有 `tid >= n_vec` 的 thread 进 tail，但 tail 只需要 `n_tail`（≤ 3）个 thread。多余的 thread 会越界写。

要么两个独立 `if`（当前写法），要么 `else if (tid - n_vec < n_tail)` + 相应下标调整。前者更简洁。

=== fuse 的实际收益

省一次 kernel launch = 省 ~5 μs。对 $N = 2^20$ 的 vector add（本身也就几微秒），这个省法能让端到端时间 -30%。对 $N = 10^9$ 的大数组（几百毫秒级别），launch overhead 可以忽略。

*fuse 的普适规则*：kernel 越小、launch 越频繁，fuse 收益越大。这是为什么框架里有 `torch.jit.script` / `torch.compile` 的 elementwise fuser。

== 关于 kernel launch 开销

vector add 是理解 launch overhead 的最好例子，因为它自身太快，launch 的 5 μs 相对显著。

launch 一次 kernel 涉及：

1. Host: 检查 arg、构造 launch descriptor。
2. Driver: enqueue 到 CUDA context 的 command buffer。
3. GPU: 从 command buffer 取出，分配 SM 资源，启动 block。
4. Kernel epilogue: block 释放资源、warp scheduler 空转。

典型开销：~5 μs（现代 driver，warm path）。冷启动首次可能 100 μs 以上。

减小 launch 开销的方法：

- *CUDA Graph*：把一系列 launch 录制成 graph，一次 replay。framework 里 `torch.compile` 会用到。
- *Fused kernel*：多个逻辑步骤在一个 kernel 内完成（本章 v5，或者更复杂的 attention 融合）。
- *Persistent kernel*：一个 kernel 长时间跑，通过 device-side 同步接受任务。RNN inference 用得比较多。

== ncu 该看什么

`ncu` (Nsight Compute) 对 vector add 的关键 metric：

- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`：SM 有多忙。
- `dram__bytes.sum.per_second` / `dram__bytes.sum.pct_of_peak_sustained_elapsed`：HBM 用了多少 / 打满没有。
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` 等：L1 texture cache 的 global load 字节数。

对健康的 vector add，你应该看到：*effective GB/s 接近硬件 peak*（A100 上 > 1700 GB/s = 83% peak）、`sm__cycles_active` 接近 100%（memory-bound kernel 里 SM 一直在发起 load/store，SM active 高但每个 cycle 大部分 warp 在等 memory）、`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` 数十到数百（大量 warp 在等 HBM）。`dram__bytes.pct_of_peak_sustained_elapsed` 对短 kernel（μs 级）通常低估，因为分母是 elapsed time；用 effective GB/s 更可靠。

```
ncu --set full --section SpeedOfLight ./build/01_vector_add
```

== 面试白板 code

面试官说"手写一个 vector add"——不要写 naive 1-thread-1-element 版本（会被追问怎么处理 N > grid size、怎么打满带宽）。直接给这份：

```cpp
// c[i] = a[i] + b[i]，任意 N，无对齐要求。
// 白板要点：grid-stride 循环 + float4 主体 + 标量 tail.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int stride = gridDim.x * blockDim.x;
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;

  // 主体：向量化 4 元素/thread. n_vec 是完整 float4 的数量。
  int n_vec = n / 4;
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4*       c4 = reinterpret_cast<float4*>(c);
  for (int i = tid; i < n_vec; i += stride) {
    float4 av = a4[i], bv = b4[i];
    float4 cv = {av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w};
    c4[i] = cv;
  }

  // Tail: n 不是 4 的倍数时，剩下 0-3 个元素. tail_start 之后由前几个 lane 补齐.
  int tail_start = n_vec * 4;
  int tail       = n - tail_start;
  if (tid < tail) c[tail_start + tid] = a[tail_start + tid] + b[tail_start + tid];
}

// ==== Launch config ====
// blockDim = 256 (8 warp)：能塞满 SM 的 warp scheduler、寄存器压力低.
// gridDim  = min(2 * SM 数, (N/4 + block - 1) / block)：
//   * 上界 2×SM 是 "grid-stride 惯用值"——每 SM 常驻 2 个 block 就足够
//     隐藏 memory latency，再多的 block 只带来 launch/schedule 开销；
//   * 下界保证 N 很小时不启多余 block.
int block = 256;
int sm    = /* cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) */ 108;
int grid  = min(2 * sm, ((n / 4) + block - 1) / block);
vector_add<<<grid, block>>>(a, b, c, n);
```

*核心考点*（追问顺序）：

- *"为什么用 grid-stride 而不是 1-thread-1-element？"* → 解耦 kernel launch 参数和输入大小，可以按硬件 SM 数调 grid（launch 开销 fixed），一份 kernel 处理任意 $N$。
- *"float4 有什么好处？"* → 每条 `LDG.128` 指令搬 16 字节，指令数减 4×、latency 更好被掩盖。前提是 `a`, `b`, `c` 都 16-byte 对齐（`cudaMalloc` 保证）。
- *"tail 为什么要单独处理？"* → `n % 4 != 0` 时最后 0-3 个元素凑不齐 float4，随便读会越界。前 `tail` 个 thread（不是最后几个）来做，避免另开 launch。
- *"你怎么验证这是 memory-bound？"* → 算 AI = 12B in + 4B out per FMA-add ≈ 0.083 FLOP/B，远小于 A100 ridge point 13 FLOP/B，纯 memory-bound；ncu 上 `dram__throughput.pct > 85%` 就到 peak 了。
- *"为什么 grid 不直接开 `N/block`？"* → memory-bound kernel launch 开销约 5 μs、每 block 的 fixed cost 也不小；grid 开到 2×SM 就够常驻满 warp 隐藏 latency，再多是浪费。这也是 grid-stride 存在的意义——解耦"处理多少数据"和"启多少 block"。

== 面试考点

#interview[
  *Q1*: vector add 的 naive 版本 A100 上大约能打到多少的峰值带宽？为什么？

  A: 实测 *~84% peak*（$N = 2^27$，1706 GB/s vs peak 2039 GB/s）。原因：(a) 完美的合并访问，warp 里 32 lane 连续；(b) `issued/32 = 32.0` 无分支、无 predication；(c) occupancy 高，warp 数足够掩盖 HBM 延迟；(d) 计算量微不足道，纯 memory-bound。想拿到 > 90% 用 `float4` 让 LDG.E.128 一次拉 16 B（实测 89.5%）；剩下 10% 是 memory controller 无法完全消除的地址冲突和 read/write 切换开销。

  #warn[
    小规模（$N = 2^20$，12 MB）测出来会看到假的"逻辑 GB/s 1700-1800"，那是 L2 缓存艺术。要判断真实性能，用 $N >= 2^27$ 或直接看 `dram__bytes.sum`。
  ]
]

#interview[
  *Q2*: 什么是合并访问 (coalesced access)？破坏它会怎样？

  A: 同一个 warp 内 32 个 lane 访问连续的 128 字节，被 memory controller 合并成 1 个 transaction。破坏了会拆成多个（最坏 32 个）transaction，实际带宽掉到 1/32。
]

#interview[
  *Q3*: grid-stride loop 有什么优势？会更快吗？

  A: 主要优势是解耦 grid 和数据规模——launch 开销固定、绕过 gridDim 上限、occupancy 可控、代码通用。对 vector add 这种已经打满带宽的 kernel*不会更快*，甚至因为循环开销略慢。它的价值是通用模板。
]

#interview[
  *Q4*: `float4` 向量化的收益来自哪？什么时候没用？

  A: 收益来自 (a) 减少指令数、(b) 单个 outstanding request 携带更多数据，更容易饱和 MSHR。对已经打满 HBM 且 warp 足够多的 kernel（vector add）收益很小；对访存模式复杂、延迟难掩盖的 kernel 提升明显。要求 16B 对齐。
]

#interview[
  *Q5*: 为什么 tiled 版本里 thread 的 stride 是 `blockDim.x` 而不是 `items_per_thread`？

  A: 保持每一轮迭代 warp 内合并访问。stride = items_per_thread 会让 warp 内 lane 跨大范围访问，破坏合并，性能骤降。
]

#interview[
  *Q6*: 一次 kernel launch 的开销大约多少？怎么减小？

  A: 现代 driver 上大约 5 μs 每次 launch。减小办法：CUDA Graph（录制并 replay 一批 launch）、fused kernel（多逻辑步骤合并）、persistent kernel（长驻 kernel 接受任务）。
]

#interview[
  *Q7*: 如何估计一个 kernel 是 memory-bound 还是 compute-bound？

  A: 算 arithmetic intensity（每字节内存访问对应多少次 FLOP），和硬件的 ridge point（peak FLOPS / peak BW）比较。低于 ridge point 是 memory-bound，高于则 compute-bound。A100 FP32 大约 13 FLOP/B。
]
