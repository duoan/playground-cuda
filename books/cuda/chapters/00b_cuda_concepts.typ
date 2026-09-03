#import "../template.typ": *

= CUDA 基本概念速查

这一章不是完整的 CUDA 入门教程——是本书正文里*会反复出现的几个术语和内置变量*的精确定义。写 GPU 代码时你会不断遇到 `blockIdx.x * blockDim.x + threadIdx.x` 这类表达式，如果对每个变量到底指什么*没有强直觉*，后面 grid-stride、tile 索引、warp id 计算就会一直卡壳。

读完本章你应该能：

- 心里有一张 "grid → block → thread" 三层图，能在白板上画出来。
- 看到 `blockIdx.x * blockDim.x + threadIdx.x` 立刻知道它算的是什么。
- 手推 grid-stride loop 里 `stride = blockDim.x * gridDim.x` 的由来。

== 三层结构：grid、block、thread

CUDA 用*三层嵌套*来组织"要启动多少个 thread"：

+ *thread*（线程）：最小执行单位。一个 thread 就是一份 kernel 代码在自己上下文（自己的寄存器、自己的 threadIdx）里跑一遍。
+ *block*（线程块）：一组 thread 的集合。同一 block 的所有 thread 一起被调度到*同一个 SM*（streaming multiprocessor）上，*可以*通过 shared memory 通信、可以用 `__syncthreads()` 同步。
+ *grid*（网格）：一次 kernel launch 启动的所有 block 的集合。不同 block 之间*独立*，*不能*直接通信（要通过 global memory + atomic 或再来一次 kernel）。

三层各自都可以是 *1D / 2D / 3D*——用 `.x / .y / .z` 分量索引。

#figure(
  cetz.canvas({
    import cetz.draw: *
    let block-w = 1.6
    let block-h = 1.0
    let gap = 0.15

    // Grid: 3×2 的 block 网格
    content((3.0, 3.9), text(weight: "bold", size: 10pt, [Grid（一次 kernel launch）]))
    for by in range(2) {
      for bx in range(3) {
        let x = bx * (block-w + gap)
        let y = (1 - by) * (block-h + gap) + 1.0
        rect((x, y), (x + block-w, y + block-h),
             fill: rgb("#dbeafe"), stroke: 0.6pt + rgb("#1e40af"))
        content((x + block-w/2, y + block-h/2),
                text(size: 7.5pt, [Block\ ($bx$, $by$)]))
      }
    }
    // Grid dim label
    content((-0.9, 2.5), align(right, text(size: 9pt, [gridDim\ = (3, 2)])))

    // Zoom-in of one block
    line((3 * (block-w + gap) - 0.1, 1.5),
         (7.2, 2.2), stroke: (paint: gray, dash: "dashed", thickness: 0.4pt))
    line((3 * (block-w + gap) - 0.1, 1.0),
         (7.2, -1.2), stroke: (paint: gray, dash: "dashed", thickness: 0.4pt))

    content((9.0, 3.9), text(weight: "bold", size: 10pt, [一个 block 内部]))
    // 4×2 threads within one block
    let cell = 0.55
    let cgap = 0.05
    for ty in range(2) {
      for tx in range(4) {
        let x = 7.4 + tx * (cell + cgap)
        let y = (1 - ty) * (cell + cgap) - 0.3
        rect((x, y), (x + cell, y + cell),
             fill: rgb("#dcfce7"), stroke: 0.4pt + rgb("#166534"))
        content((x + cell/2, y + cell/2),
                text(size: 6.5pt, [t($tx$,$ty$)]))
      }
    }
    content((7.4 + 2 * (cell + cgap), 1.1),
            text(size: 8.5pt, [8 个 thread]))
    content((7.4 + 2 * (cell + cgap), -0.85),
            text(size: 8.5pt, [blockDim = (4, 2)]))
  }),
  caption: [*Figure:* CUDA 三层组织。左侧 grid 由 $3 times 2 = 6$ 个 block 组成 (`gridDim = (3, 2)`)；右侧放大一个 block 的内部，包含 $4 times 2 = 8$ 个 thread (`blockDim = (4, 2)`)。总共启动 $6 times 8 = 48$ 个 thread。每个 thread 通过 `(blockIdx, threadIdx)` 唯一标识自己。],
  kind: image,
)

*总 thread 数 = 所有 block 数 × 每 block 的 thread 数*：

$ #text[总 thread 数] = "gridDim.x" times "gridDim.y" times "gridDim.z" times "blockDim.x" times "blockDim.y" times "blockDim.z" $

一维简化版（本书 90% 场景）：

$ #text[总 thread 数] = "gridDim.x" times "blockDim.x" $

== 六个内置变量

在 `__global__` kernel 里，CUDA 为每个 thread *自动*提供这六个内置变量。它们是 `uint3`（三分量整数），可以用 `.x / .y / .z` 访问。

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.4pt + gray,
    table.header([变量], [含义], [值范围（在 kernel 内看）]),
    [`gridDim`], [整个 grid 里有多少 block（每维）], [跟 launch 时 `<<<gridDim, ...>>>` 一致，*所有 thread 看到同一值*],
    [`blockDim`], [每个 block 里有多少 thread（每维）], [跟 launch 时 `<<<..., blockDim>>>` 一致，*所有 thread 看到同一值*],
    [`blockIdx`], [*当前 thread 所在的 block* 在 grid 里的坐标], [`0..gridDim.x-1` 等，每 block 内 32 lane 看到同值],
    [`threadIdx`], [*当前 thread* 在自己 block 里的坐标], [`0..blockDim.x-1` 等，每 thread 看到不同值],
    [`warpSize`], [warp 宽度，NVIDIA GPU 一直是 32], [永远 = 32],
    [`__CUDA_ARCH__`], [编译期宏，SM 版本 × 10（如 sm_80 → 800）], [编译期常量],
  ),
  caption: [*Table:* CUDA kernel 里可用的六个内置变量。前两个（`gridDim`, `blockDim`）描述整个 launch 的形状，*所有 thread 看到相同值*；中间两个（`blockIdx`, `threadIdx`）标识每个 thread 的身份，*每 thread 不同*。区分好"grid 级常量"和"thread 级身份"是读懂 CUDA 代码的第一步。],
  kind: table,
)

*观察*：`gridDim` 和 `blockDim` 是*形状*（launch 时决定，运行时不变），`blockIdx` 和 `threadIdx` 是*身份*（每 thread 各不同）。这两组区分记牢，后面所有索引推导都是它们的组合。

== 从内置变量算全局线程 ID

最经典的表达式：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

*怎么理解*：想象 grid 就是一条长队伍，队伍被切成 `gridDim.x` 段（每段是一个 block），每段有 `blockDim.x` 个人（thread）。

- `blockIdx.x` = 我在第几段
- `blockDim.x` = 每段多少人
- `blockIdx.x * blockDim.x` = 我这段的第一个人在整条队伍里的编号
- `+ threadIdx.x` = 加上我在段内的偏移

结果就是"我在整条队伍里第几个"。

#figure(
  cetz.canvas({
    import cetz.draw: *
    let cell = 0.42
    let gap = 0.03
    let n_per_block = 4
    let n_blocks = 3

    // Draw all threads as a linear array
    for b in range(n_blocks) {
      for t in range(n_per_block) {
        let x = (b * n_per_block + t) * (cell + gap) + b * 0.25
        rect((x, 0), (x + cell, cell),
             fill: rgb("#dcfce7"), stroke: 0.4pt + rgb("#166534"))
        content((x + cell/2, cell/2),
                text(size: 7pt, str(b * n_per_block + t)))
      }
      // block bracket
      let x0 = b * n_per_block * (cell + gap) + b * 0.25
      let x1 = x0 + n_per_block * (cell + gap) - gap
      line((x0, -0.25), (x0, -0.15), stroke: 0.6pt + rgb("#1e40af"))
      line((x0, -0.15), (x1, -0.15), stroke: 0.6pt + rgb("#1e40af"))
      line((x1, -0.15), (x1, -0.25), stroke: 0.6pt + rgb("#1e40af"))
      content(((x0 + x1)/2, -0.5),
              text(size: 8pt, [blockIdx.x = #b]))
    }

    // threadIdx labels on second block only
    for t in range(n_per_block) {
      let x = (1 * n_per_block + t) * (cell + gap) + 1 * 0.25
      content((x + cell/2, cell + 0.25),
              text(size: 6.5pt, [tx=#t]))
    }
    content((-1.5, cell/2), align(right, text(size: 9pt, [tid = ])))
    content((-1.5, -0.5), align(right, text(size: 9pt, [block ])))

    // formula highlight
    let hx = 1 * n_per_block * (cell + gap) + 1 * 0.25 + 2 * (cell + gap)
    line((hx + cell/2, cell + 0.55), (hx + cell/2, cell + 0.9),
         stroke: (paint: rgb("#dc2626"), thickness: 0.8pt), mark: (end: ">"))
    content((hx + cell/2, cell + 1.15),
            text(size: 8pt, [tid = 1 × 4 + 2 = *6*]))
  }),
  caption: [*Figure:* `blockIdx.x * blockDim.x + threadIdx.x` 的几何含义。gridDim = 3、blockDim = 4，共 12 个 thread。中间那个高亮的 thread 位于 block 1、块内 threadIdx.x = 2，所以全局 tid = $1 times 4 + 2 = 6$。],
  kind: image,
)

*同样的模式扩展到多维*：2D 索引一个 tile 的行/列，就是分别对 `.x` 和 `.y` 做同样的展开：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;   // 沿输出矩阵的行
int col = blockIdx.x * blockDim.x + threadIdx.x;   // 沿输出矩阵的列
```

这就是本书 matmul、attention、layernorm 里反复出现的坐标计算。

== warp：SIMT 执行的最小单位

*warp = 32 个连续 thread*，硬件层面 *它们锁步执行同一条指令*——不管你 kernel 代码里怎么写，SM 每周期发射一条指令，对整个 warp 生效。

- lane id = `threadIdx.x % 32`（1D block 时）
- warp id in block = `threadIdx.x / 32`

*为什么 32 这个数字*：NVIDIA 硬件设计选择，从 Tesla 到 Blackwell 一直不变。AMD GPU 是 64（叫 wavefront）。

*warp 的三条硬性约束* 是本书后面反复要用的：

+ 同一 warp 内 32 lane 走同一条 branch 时，走的是"predicated instruction"（不是分支），无成本。走*不同* branch 时才是 warp divergence，硬件要 serialize $k$ 条 path。
+ 同一 warp 内 32 lane *不需要 `__syncthreads`* 就能看到彼此写入 shared memory 的结果（SIMT 锁步）。跨 warp 就必须 `__syncthreads`。
+ warp shuffle 指令（`__shfl_*_sync`）只在 warp 内工作，跨 warp 只能走 shared memory。

== Launch grammar

kernel 调用语法：

```cpp
kernel<<<grid, block, smem, stream>>>(args...);
```

四个 launch 参数：

+ `grid`：`int`（1D 简写）或 `dim3(x, y, z)`——决定 `gridDim`。
+ `block`：`int` 或 `dim3(x, y, z)`——决定 `blockDim`。*每 block thread 总数上限 1024*（A100/H100）。
+ `smem`（可选）：dynamic shared memory 字节数。kernel 里用 `extern __shared__ float shared[]` 声明。默认 0。
+ `stream`（可选）：CUDA stream，默认 stream 0。

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.4pt + gray,
    table.header([限制项], [A100 (sm_80)], [H100 (sm_90)]),
    [每 block 最多 threads], [1024], [1024],
    [每 SM 最多 threads], [2048], [2048],
    [每 SM 最多 warps], [64], [64],
    [每 SM 最多 blocks], [32], [32],
    [每 SM shared memory], [164 KB], [228 KB],
    [每 SM registers], [65536 (fp32)], [65536],
    [gridDim.x 上限], [$2^31 - 1$], [$2^31 - 1$],
    [gridDim.y/.z 上限], [65535], [65535],
  ),
  caption: [*Table:* A100 和 H100 的关键硬件上限。这些数字决定 kernel launch 时哪些配置合法、以及 occupancy 上限。本书里"每 block 用几个 warp"、"能不能开 4M 个 block"这类问题都要回来查这张表。],
  kind: table,
)

*观察*：每 SM 最多 2048 threads = 64 warps。这就是 A100 的"满 occupancy"目标——kernel 每 SM 常驻 64 warp 时 latency hiding 达到最好。

== 手推 grid-stride 的 stride

现在回到读者最常卡的一段代码（vector add ch1 v3）：

```cpp
__global__ void vector_add_grid_stride_kernel(
    const float* a, const float* b, float* c, int count) {
  const int i0     = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = i0; i < count; i += stride) {
    c[i] = a[i] + b[i];
  }
}
```

*逐行读*：

+ `i0 = blockIdx.x * blockDim.x + threadIdx.x`——上面已经推导过：这是"我"作为一个 thread 在整个 grid 里的全局 id。
+ `stride = blockDim.x * gridDim.x`——总 thread 数（1D 时）。
+ `for (int i = i0; i < count; i += stride)`——从我的起点 `i0` 出发，每次跳 `stride`（= 总 thread 数）。

*为什么 stride 是"总 thread 数"*：

假设一共 launch 了 $T$ 个 thread（$T = "blockDim.x" times "gridDim.x"$），要处理 $N$ 个元素，$N gt.eq T$。

- 第 1 轮：thread 0 做 `c[0]`, thread 1 做 `c[1]`, ..., thread $T-1$ 做 `c[T-1]`——正好覆盖 $[0, T)$。
- 第 2 轮：thread 0 做 `c[T]`, thread 1 做 `c[T+1]`, ..., thread $T-1$ 做 `c[2T-1]`——正好覆盖 $[T, 2T)$。
- ...
- 第 $k$ 轮：thread $t$ 做 `c[k T + t]`。

*每一轮每个 thread 向前跳一整个"grid 步长" $T$*。这样所有 $N$ 个元素都会被覆盖，且每个元素只被访问一次。

#figure(
  cetz.canvas({
    import cetz.draw: *
    let cell = 0.5
    let gap = 0.04
    let T = 4   // 4 threads total (for drawing)
    let rounds = 3

    // Draw N=12 elements
    for i in range(T * rounds) {
      let x = i * (cell + gap)
      rect((x, 0), (x + cell, cell),
           fill: rgb("#f3f4f6"), stroke: 0.3pt + gray)
      content((x + cell/2, cell/2), text(size: 7pt, [c[#i]]))
    }

    // Show one thread's trajectory (thread 1)
    let tid = 1
    let colors = (rgb("#22c55e"), rgb("#3b82f6"), rgb("#a855f7"))
    for r in range(rounds) {
      let i = r * T + tid
      let x = i * (cell + gap)
      rect((x - 0.03, -0.03), (x + cell + 0.03, cell + 0.03),
           stroke: (paint: colors.at(r), thickness: 1.4pt), fill: none)
      content((x + cell/2, cell + 0.35),
              text(size: 7pt, fill: colors.at(r), [round #(r+1)]))
    }
    // Arrows between them
    for r in range(rounds - 1) {
      let i0 = r * T + tid
      let i1 = (r + 1) * T + tid
      let x0 = i0 * (cell + gap) + cell/2
      let x1 = i1 * (cell + gap) + cell/2
      line((x0, -0.25), (x1, -0.25),
           stroke: (paint: rgb("#dc2626"), thickness: 0.7pt),
           mark: (end: ">", size: 0.15))
    }
    content(((rounds - 1) * T * (cell + gap) / 2, -0.6),
            text(size: 8pt, fill: rgb("#dc2626"), [+ stride = + #T]))
    content((-1.4, cell/2),
            align(right, text(size: 9pt, [thread 1\ 的轨迹:])))
  }),
  caption: [*Figure:* grid-stride loop 的可视化，假设总共 $T = 4$ 个 thread、$N = 12$ 个元素。thread 1 的轨迹是 `c[1] → c[5] → c[9]`，每次跳 stride = 4（= 总 thread 数）。所有 $T$ 个 thread 并行地各走自己的一条轨迹，最终每个元素都恰好被覆盖一次。],
  kind: image,
)

*观察*：如果 stride 写错——比如写成 `blockDim.x`（一个 block 内的 thread 数）——那 thread 0 走 `c[0], c[256], c[512], ...`、thread 256 也走 `c[256], c[512], ...`，就冲突了。stride *必须* = 总 thread 数才能保证 disjoint 覆盖。

#insight[
  grid-stride 的核心一句话：*launch 多少 thread 由 SM 数决定（比如 2 × SM_count × block_size），每个 thread 循环处理 $N / T$ 个元素*。这把"数据规模 $N$"和"launch 配置 $T$"解耦——一份 kernel 处理任意 $N$。
]

== 常见坑

三个初学者最常犯的错：

+ *忘了 boundary check*。naive 版 `blocks = (N + 255) / 256` 会向上取整，最后一个 block 里有些 thread 的 `i >= N`，如果不加 `if (i < N)` 就会越界访问。grid-stride 里 `for (int i = i0; i < count; ...)` 的 `i < count` 就是同一个检查，天然内嵌在循环条件里。
+ *把 blockIdx 和 threadIdx 搞混*。记住："*Idx* 是身份、*Dim* 是形状"。blockIdx = 我在哪个 block、threadIdx = 我在 block 内哪个位置。
+ *2D block 时 x/y 弄反*。CUDA 约定 `threadIdx.x` 是*最内维*（stride-1，coalesced 方向）。写 `A[row * N + col]` 时 col 应该走 `threadIdx.x`，row 走 `threadIdx.y`。反过来会让 gmem 全变成 uncoalesced。

后面章节遇到这些概念时不会再重复解释——如果卡住，翻回来查这张速查表。
