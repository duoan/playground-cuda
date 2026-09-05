#import "../template.typ": *

= Flash-Attention v3

Flash-Attention v3（Shah et al., 2024）是全书*最后一章正文*，也是 ladder 的终点：从第 8 章 v1 的 IO-aware 分块，到第 9 章 v2 的 sequence-parallel 与 warp 协作，再到 v3 把整条 pipeline *搬到 Hopper (H100, SM90)* 上——用 WGMMA、TMA、FP8 tensor core 把 FA2 在 H100 上仅 ~35% 的利用率推到 FP16 75%、FP8 接近 1.2 PFLOPS。

这一章我们要把它讲透：

- Hopper 相对 Ampere 的三大硬件增量：WGMMA、TMA、FP8 tensor core，以及 Thread Block Cluster / DSMEM。
- WGMMA 与 `mma.sync` 的 async 语义、fragment 布局、accumulator 生命周期差异。
- TMA 相对 `cp.async`：独立硬件单元、多维 tensor copy、boundary 自动处理。
- v3 三大创新：producer-consumer warp specialization、`mbarrier` pipeline、2-stage GEMM-softmax overlap。
- FP8 attention：量化位置、block scaling、incoherent processing、精度陷阱。
- v1 / v2 / v3 性能数字与局限；面试里 Hopper 特化 attention 怎么答。

对应源码：`src/cuda/10_flash_attention_v3.cu`。教学规模 $N_q = 4$, $N_k = 16$, $d = 8$——*刻意缩小*，因为 Hopper 指令无法在本书 scalar teaching kernel 里完整复现；源码展示 warp 分工与 pipeline *骨架*，生产实现见 CUTLASS / 官方 FlashAttention-3 repo。

本章 optimization ladder：

#ladder(
  ("CPU reference",       "double-precision online softmax",        "—"),
  ("v3 warp-specialized", "1 warp / query, lane roles + shuffle",   "—"),
  ("v3 pipeline",         "2-stage K/V smem ping-pong (conceptual)", "—"),
  ("production FA3",      "TMA + WGMMA + mbarrier + FP8（H100）",   "~75% H100 peak"),
)

前两个 GPU 版本*数学与 v1 相同*（online merge）；第三行是论文/CUTLASS 级实现的目标。

与前面章节的依赖：第 3 章 online softmax；第 4 章 matmul pipeline / tensor core / `cp.async`；第 8 章 FA v1 分块算法；第 9 章 FA v2 的 warpgroup 内循环与 parallel 策略（v3 在其算法骨架上换 Hopper 指令）。

=== FA v2 回顾：v3 的算法起点

第 9 章 FA v2 相对 v1 的三条主线，v3 *全部保留*：

1. *Parallel 策略*：在 batch × head × query-block 三维并行；内层沿 key sequence 分 $B_c$ 大小的 tile 循环（$T_c = ceil(N_k / B_c)$）。
2. *Warpgroup 内 GEMM*：$Q K^T$ 与 $P V$ 用 tensor core tile 完成，不再 scalar dot product。
3. *Online softmax*：每个 key tile 产出局部 $(m_j, ell_j)$，merge 到 running state，同步 rescale $O$ 分子。

v2 在 A100 上已经能把 IO 降到 $O(N d^2)$ 量级；搬到 H100 后，*瓶颈从 HBM 转向「如何用满 Tensor Core + 隐藏 SFU latency」*。论文原话：FA2 on H100 仅 ~35% peak，而 tuned GEMM 可达 80–90%——说明 *算法正确但 execution model 过时*。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*版本*], [*核心贡献*], [*硬件假设*],
    [FA v1], [IO-aware tiling，不写 $S$ 到 HBM], [SMEM 够用即可，scalar/warp GEMM],
    [FA v2], [Sequence parallel，warpgroup 内 matmul], [Ampere `mma.sync` + `cp.async`],
    [FA v3], [Async pipeline，FP8，warp specialization], [Hopper WGMMA + TMA + mbarrier],
  ),
  caption: [*Table:* Flash-Attention 三个版本的核心贡献与硬件假设对照。版本列对应全书 attention ladder；硬件假设说明每版依赖的 GEMM 指令与 memory 路径（scalar/warp、`mma.sync`/`cp.async` vs WGMMA/TMA/mbarrier）。],
  kind: table,
)

*Observation*：v3 相对 v1/v2 *换 execution model 不换公式*——三行硬件假设从「SMEM 够用即可」递进到「Hopper WGMMA + TMA」，对应全书从 IO-aware 分块到 Hopper async pipeline 的主线。

== 问题定义

=== 数学（不变）

单 head，$Q in RR^(N_q times d)$, $K, V in RR^(N_k times d)$：

$ O = "softmax"(frac(Q K^T, sqrt(d))) V $

v3 *不改变* attention 的数学——仍是 exact attention（无近似）。变化的是 *如何在 Hopper 上执行* FA v2 已经证明正确的分块 online 算法。

=== FLOPs 与 Roofline（H100 视角）

Forward FLOPs（单 head，忽略 mask）：

$ "FLOPs" approx 4 N_q N_k d quad "(两个 matmul，各 2 N_q N_k d)" $

LLM 典型 $N = 8192, d = 128$：FLOPs $approx 2^{36}$，H100 FP16 peak 740 TFLOPS 实现下 *理论下限* $approx 0.07$ ms——实际还受 launch、occupancy、causal 影响。

HBM 流量（FA3 目标）：$Q, K, V$ 各读一次量级 $O(N d)$ per layer pass——*无* $S, P$ 的 $O(N^2)$ 项。Standard attention 在同样 FLOPs 下多 $O(N^2)$ 读写——*memory wall* 先于 compute wall 触发。v3 在 H100 上进一步把 *已省下的 HBM 带宽* 换成「填满 Tensor Core」的问题。

=== 从 v2 到 v3：换的是 execution model，不是公式

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*维度*], [*FA v2 (Ampere)*], [*FA v3 (Hopper)*],
    [GEMM 指令], [`mma.sync` / WMMA，warp 同步], [WGMMA，warpgroup 异步],
    [Global→Smem], [thread `cp.async`], [TMA 硬件单元],
    [Warp 角色], [所有 warp 又搬又算], [Producer warp 只 TMA；Consumer 只 WGMMA+softmax],
    [Softmax 与 GEMM], [串行：`wait` 后算 softmax], [2-stage pipeline 重叠],
    [低精度], [FP16/BF16 为主], [FP8 E4M3/E5M2 + block scale],
    [H100 利用率], [~35%（论文）], [FP16 ~75%，FP8 ~1.2 PFLOPS],
  ),
  caption: [*Table:* FA v2 (Ampere) 与 FA v3 (Hopper) 的 execution model 对照。维度涵盖 GEMM 指令、Global→Smem 搬运、warp 角色、softmax 与 GEMM 调度、低精度路径及 H100 论文利用率（FP16 ~75%，FP8 ~1.2 PFLOPS）。],
  kind: table,
)

*Observation*：*每一行都是 async 深化*——`mma.sync`→WGMMA、`cp.async`→TMA、全员 load→producer-consumer、串行 softmax→2-stage overlap。论文 H100 利用率从 ~35% 到 ~75% 不是算法改动，而是五维度 execution 对齐 Hopper。

#insight[
  面试先说清楚：v3 不是新 attention 算法，是 *Hopper-aware 的 FA2 实现*。三大创新全部围绕 *asynchrony*——让 TMA、Tensor Core、SFU（exp）三类硬件单元同时忙。
]

== Hopper 硬件回顾

=== SM90 相对 SM80 的结构变化

Hopper H100 每个 SM 的关键增量（面试必背）：

1. *第四代 Tensor Core*：支持 warpgroup 级 WGMMA，FP8 吞吐约为 FP16 的 2×。
2. *TMA (Tensor Memory Accelerator)*：CTA 内一条 descriptor 发起 multidimensional async copy（GMEM ↔ SMEM），不占 CUDA core 做 load/store。
3. *更大的 SMEM*：228 KiB/SM（可配置），配合 TMA 搬更大的 $K/V$ tile。
4. *Thread Block Cluster*：最多 16 个 CTA 组成 cluster，可访问 *Distributed Shared Memory (DSMEM)*——跨 CTA 的 smem 映射，用于超大 tile 或 epilogue 通信。
5. *mbarrier*：硬件 barrier，配合 TMA/WGMMA 的 async completion token（`expect_tx` / `complete`）。

#note[
  本书 preface 的性能数字默认 A100 (SM80)。*本章及 v3 数字均在 H100 SXM5 上*——峰值 FP16 tensor ~989 TFLOPS，HBM ~3.35 TB/s（表 1，FlashAttention-3 论文）。
]

=== 内存层次（Thread–Memory Hierarchy）

Hopper 上 programmer 必须同时考虑 *五级* agent/locale 配对（论文 Table 1）：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, right),
    [*Hardware*], [*Parallel agent*], [*Data locale*], [*Capacity @ BW*],
    [Chip], [Grid], [GMEM/HBM], [80 GiB @ 3.35 TB/s],
    [GPC], [Threadblock Cluster], [L2], [50 MiB @ 12 TB/s],
    [SM], [Threadblock (CTA)], [SMEM], [228 KiB/SM @ ~31 TB/s aggregate],
    [Thread], [Thread], [RMEM], [256 regs max，~256 KiB/SM pool],
  ),
  caption: [*Table:* Hopper (H100) Thread–Memory Hierarchy 五级对照（论文 Table 1）。*Hardware* / *Parallel agent* / *Data locale* 列对应 programmer 模型；*Capacity @ BW* 给出 GMEM、L2、SMEM、RMEM 容量与峰值带宽（GiB / TB/s）。],
  kind: table,
)

*Observation*：FA3 设计目标是让 $Q,K,V$ tile 在 *SMEM* locale（~228 KiB @ ~31 TB/s aggregate）完成两次 GEMM + softmax，*Chip/GPC 级 HBM/L2 只进活一次*——层级表把「为何 TMA 搬 bulk tile、WGMMA 在 smem 算」quantify 成带宽差 100×+。

FA3 的设计目标：让 $Q, K, V$ tile 在 SMEM 停留期间完成两次 GEMM + softmax，*HBM 只进不出*（对 $S, P$ 而言）。TMA 负责 GMEM→SMEM 的 bulk 搬运；WGMMA 在 SMEM/RMEM 上算；softmax 的 exp 在 RMEM 上算完写回 RMEM 的 $tilde(P)$，*不触 HBM*。

=== 手算：为什么 softmax 会吃掉 50% cycle

H100 SXM5 峰值（论文 §3.1）：

- FP16 matmul（Tensor Core）：~989 TFLOPS
- Special function（exp 等，SFU）：~3.9 TFLOPS → *约为 matmul 的 1/256*

Attention forward（FP16，head dim $d = 128$，忽略 mask）每个 query block 对一个 key tile：

- GEMM0 $Q K^T$：$O(B_r B_c d)$ FLOPs，以 matmul 吞吐算 cycle
- Softmax：每个 score 一次 exp + 若干 FMA——$O(B_r B_c)$ 次 exp
- GEMM1 $P V$：同量级 matmul FLOPs

FLOPs 比：matmul : exp $approx 512 : 1$（论文），但吞吐比 $approx 256 : 1$ → exp 的 *cycle 占比* $approx 512/256 = 2$ 倍于「按 FLOPs 比例」——即 softmax 与 matmul *几乎抢同样多的 wall-clock*。FP8 matmul 再快 2×，exp 吞吐不变 → softmax 瓶颈*更严重*。这就是 v3 必须 overlap GEMM 与 softmax 的 quantitative 理由。

#insight[
  面试画 Roofline 不够——attention 是 *multi-pipeline* kernel：TC、SFU、TMA 三套单元。FA2 在 H100 上慢，often 不是因为 HBM，而是因为 SFU 与 TC *串行互斥*。
]

=== WGMMA：Warpgroup Matrix Multiply-Accumulate

*Warpgroup* = 4 个连续 warp（128 threads）。一条 `wgmma.mma_async` 由整个 warpgroup 发起，硬件在 background 执行，thread 可继续发别的指令（直到 `wgmma.wait_group`）。

对比 Ampere 的 `mma.sync`：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*特性*], [mma.sync (SM80)], [WGMMA (SM90)],
    [协作粒度], [1 warp (32 threads)], [1 warpgroup (128 threads)],
    [执行语义], [同步——指令内隐含 wait], [异步——`commit` + `wait_group`],
    [操作数来源], [smem 或 rmem fragment], [smem 为主（SS/RS 变体）],
    [典型 shape], [m16n8k16], [m64n128k16 等更大 tile],
    [FP8], [不支持], [支持，仅 k-major operand layout],
  ),
  caption: [*Table:* Ampere `mma.sync` (SM80) 与 Hopper WGMMA (SM90) 特性对照。涵盖协作粒度（1 warp vs 1 warpgroup）、执行语义（同步 vs `commit`+`wait_group`）、operand 来源、典型 MMA shape 及 FP8 支持。],
  kind: table,
)

*Observation*：WGMMA 把协作粒度从 32 扩到 128 thread，并把 *issue 与 complete 拆开*——这是第 9 章 FA2 无法 overlap GEMM 与 softmax 的硬件根因；FP8 仅 WGMMA + k-major 是 Innovation 3 工程成本来源。

=== TMA：Tensor Memory Accelerator

TMA 用 *tensor map descriptor*（host 侧构造）描述一次 multidimensional transfer：

```cpp
// 概念性 CUTLASS/CUTE 风格（非本书源码）
// 一个 thread（通常 warp 0 的 lane 0）发起：
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
  [smem_dst], [tensor_map, offset0, offset1], [mbarrier];
```

特点：

- *独立硬件单元*：copy 进行时，128 个 thread 不必逐元素 load——与 `cp.async` 仍需每个 thread 算地址并发起 copy 不同。
- *Multidimensional*：一次搬 2D/3D/4D tile，stride/boundary 在 descriptor 里——越界自动 zero-fill 或 clamp。
- *与 mbarrier 集成*：TMA 完成后 hardware 更新 barrier，consumer warpgroup 在 `mbarrier.try_wait` 后继续。

=== FP8 Tensor Core

Hopper FP8 两种格式：

- *E4M3*：4 bit exponent，3 bit mantissa——动态范围小、精度稍高；常用于 forward activations。
- *E5M2*：5 bit exponent，2 bit mantissa——动态范围大；常用于 gradient。

FP8 WGMMA 峰值约为 FP16 的 2×（同 power/area）。*约束*：FP8 operand 在 smem 中必须是 *k-major*（内维连续），且 FP32 accumulator layout 与下一条 GEMM 的 FP8 operand A layout *不兼容*——v3 需要 register permute + in-kernel $V$ transpose（论文 §3.3）。

=== Thread Block Cluster 与 DSMEM

Cluster 让同一 GPC 内多个 CTA 的 smem 可互相寻址——用于：

- 超大 $Q/K/V$ tile 跨 CTA 协作；
- Epilogue 里跨 block reduce（某些 split 策略）。

FlashAttention-3 主路径仍以单 CTA 处理一个 $Q$ tile 为主；cluster 是 Hopper 通用能力，面试知道定义即可。

=== DSMEM 使用场景（扩展）

DSMEM 允许 cluster 内 CTA $A$ 读取 CTA $B$ 的 smem 地址——映射由 `mapa.shared::cluster` 完成。潜在用途：

- *Split-KV*：一个 CTA 算 $Q K_1^T$，邻居 CTA 算 $Q K_2^T$，partial softmax merge——类似 GEMM split-K，但 merge 公式是 online softmax（第 3 章）。
- *大 tile epilogue*：多个 CTA 协作写 $O$ 的 overlapping region（需 atomic 或 deterministic reduction）。

FA3 论文主 forward kernel *未依赖* DSMEM——面试答「知道 capability，FA3 核心在 TMA+WGMMA」即可。

#warn[
  v3 *无法在 Ampere / Ada 上运行*——没有 TMA/WGMMA/FP8 TC。PyTorch 集成通常 `sm_90` 专用编译。这是 v3 最大的部署局限。
]

== WGMMA vs `mma.sync`：深入对比

=== Async 语义与 pipeline

Ampere matmul 主循环（第 4 章）：

```
cp.async.load tile → wait → ldmatrix → mma.sync → ...
         ↑__________________|  同步点：mma 完成才能下一轮
```

Hopper WGMMA 主循环：

```
TMA.load K_tile → mbarrier.wait
wgmma QK^T, commit          ← 不 wait，立即发下一条
TMA.load V_tile
wgmma PV, commit
wgmma.wait_group 0          ← 按需 wait
softmax on S                ← 可与下一 warpgroup 的 wgmma 重叠
```

#insight[
  `mma.sync` 的 "sync" 把 *issue* 和 *complete* 绑在一起；WGMMA 拆开二者，才给 GEMM-softmax overlap 留出指令级空隙。面试画 timeline 时，WGMMA 的 `commit`/`wait_group` 相当于 `cp.async` 的 `commit_group`/`wait_group`，但操作对象是 tensor core 而非 load unit。
]

=== Fragment 布局与 accumulator 生命周期

Ampere：每个 thread 持 `a_frag[4], b_frag[2], c_frag[4]`（m16n8k16），`ldmatrix.sync` 从 swizzled smem 加载，`mma.sync` 原地累加 `c_frag`。

Hopper WGMMA：

- Accumulator 在 *warpgroup* 的 register 里分布，shape 更大（如 64×128 tile 的部分和）。
- *SS-GEMM*：两操作数来自 smem；*RS-GEMM*：A 来自 register（用于 $P V$，$P$ 在 softmax 后驻留 rmem）。
- FP8 路径：第一个 WGMMA 的 FP32 acc layout ≠ 第二个 WGMMA 的 FP8 A layout——需 `byte_perm` 重排（论文 Fig. 3–4）。

*Accumulator 生命周期*：在 2-stage pipeline 里，$S_"cur"$ 和 $S_"next"$ 同时驻留 register——register pressure 上升，可能迫使减小 tile size（论文 §3.2 register pressure 讨论）。

=== WGMMA 指令形态（PTX 级直觉）

```cpp
// 概念性：warpgroup 内协作，非单 thread 可调用
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16.f32
  {acc...}, {a...}, {b...}, scale_d;
wgmma.commit_group.sync.aligned;
// ... 其他 warpgroup 或 SFU 工作 ...
wgmma.wait_group.sync.aligned 0;
```

与 `mma.sync` 对比：`mma_async` + `commit_group` + `wait_group` 三级把 *launch* 与 *retire* 拆开。FlashAttention 在 `commit` 之后、`wait` 之前插入 softmax 的 exp/max——前提是 softmax 读的是*已 retire* 的 $S_"cur"$，而 $S_"next"$ 的 WGMMA 仍在飞。

=== `setmaxnreg`：Producer/Consumer 寄存器预算

Hopper 允许 warpgroup 动态调整 register 上限：

- *Producer warpgroup*：`setmaxnreg` 降到 ~32 regs——只发 TMA，不需要大 accumulator。
- *Consumer warpgroup*：`setmaxnreg` 提到 ~240 regs——持 WGMMA accumulator + pipeline 缓冲。

这是 warp specialization 的硬件基础：*同一 CTA 内不同 warpgroup 可以有不同 register footprint*，提高整体 occupancy 效率。

=== mn-major 与 k-major（FP8 面试追问）

WGMMA 对 operand layout 的约定（论文 §2.2）：对 $A times B^T$ 的 GEMM，$A$ 为 $M times K$，$B$ 为 $N times K$。

- *mn-major*：operand 在*外维*（$M$ 或 $N$）连续。
- *k-major*：operand 在*内维* $K$ 连续。

FP16 WGMMA：smem 中 mn-major 与 k-major *均可*。FP8 WGMMA：*仅 k-major*。Attention 中 $Q, K$ 通常 head-dim contiguous（k-major 友好），但 $V$ 默认 layout 是 `[seq, head]` 即 mn-major——第二个 GEMM 必须 transpose tile。这是 FP8 FA3 比 FP16 多出来的主要 engineering 成本。

=== SS-GEMM 与 RS-GEMM 在 attention 中的分工

- *GEMM0* $S = Q K^T$：$Q, K$ 均在 smem → *SS-GEMM*（Shared–Shared）。
- *GEMM1* $O += P V$：$P$ 在 softmax 后驻留 register（未写 smem）→ *RS-GEMM*（Register–Shared）。

Ampere FA2 同样区分 WMMA 的 operand 来源，但 WGMMA 的 RS 路径与 async wait 组合更复杂——Algorithm 2 里两条 WGMMA 可以 *in-flight*，靠 `wait_group` 精细控制。

== TMA 与 `cp.async` 对比

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*维度*], [cp.async (SM80)], [TMA (SM90)],
    [发起者], [每个 thread 一条 copy], [通常 1 thread 发 bulk transfer],
    [地址计算], [thread 算 smem/gmem 地址], [descriptor 预编码 stride/shape],
    [Boundary], [kernel 内 if/ padding], [hardware 处理 OOB],
    [Completion], [`cp.async.wait_group`], [`mbarrier` transaction count],
    [Multidim], [需手写 nested loop], [原生 2D/3D/4D tensor copy],
    [Occupancy 影响], [所有 thread 参与 load], [producer 仅少量 thread，其余可 sleep/yield],
  ),
  caption: [*Table:* SM80 `cp.async` 与 SM90 TMA 的多维对比。维度包括发起者、地址计算、boundary 处理、completion 机制（`cp.async.wait_group` vs `mbarrier` transaction count）、多维 copy 能力及对 occupancy 的影响。],
  kind: table,
)

*Observation*：TMA 把 load 从 *256 thread 全员参与* 降到 *1 thread 发 bulk*——consumer warpgroup 的 issue slot 全留给 WGMMA/SFU；与 WGMMA async 配对才是 Hopper attention 相对 FA2 的完整升级，单换 WGMMA 不够。

第 4 章 matmul 的 `cp.async` 双缓冲是 *thread-participatory* pipeline——256 个 thread 都要写 copy 指令。FA3 的 TMA 把 load 从 consumer 的 instruction stream 里*剥离*，consumer warpgroup 的 issue slot 全留给 WGMMA 和 softmax。

#note[
  Triton 版 FA2 在 H100 上也能用 Hopper 指令，但仍未做 producer-consumer 分离——论文显示 FA3 仍比 Triton FA2 快 ~1.5×。差距主要来自 warp specialization + intra-warpgroup overlap，不只是 "用了 WGMMA" 这一点。
]

=== TMA descriptor 与 boundary 处理

Host 侧用 `cuTensorMapEncode` 或 CUTLASS `make_tma_copy` 构造 descriptor，编码：

- Base pointer、shape、stride（最多 5D）
- Box size（每次 copy 的 tile 形状）
- Swizzle mode（与 smem bank layout 对齐）

Kernel 内 producer 只需 `tensor_map + offset`——*boundary check 在 hardware*：若 tile 超出 tensor 边界，多余元素填 0（对 attention padding/mask 友好）。对比 `cp.async`：每个 thread 自己算 `if (idx < bound)`，256 threads 重复同一逻辑，issue 效率低。

=== Circular SMEM buffer 与 stage 数

FA3 forward 典型 $S = 2$ 或 $3$ stage 的 ring buffer：

```
smem[K/V][stage][B_c × d]   stage ∈ {0, 1, ..., S-1}
```

Producer 写 stage $(j mod S)$；Consumer 读 stage $(j mod S)$ 并在算完后 release。Stage 数 $S$ 与 Algorithm 2 的 register pipeline depth *不必相等*——smem stage 管 TMA 与 compute 的 data race；register stage 管 GEMM-softmax overlap。面试别混：*smem ping-pong ≠ GEMM-softmax 2-stage*，但可以协同设计。

=== `cp.async` 在 FA2 中的位置（对照）

第 9 章 / FA2 Ampere 实现典型 pattern：

```cpp
// 每个 thread 参与
cp.async.cg.shared.global [smem + off], [gmem + off], 16;
cp.async.commit_group();
cp.async.wait_group(stages - 1);
__syncthreads();
ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...;
```

256 threads 每轮 K-slab 各发若干 `cp.async`——instruction issue 与 `ldmatrix`/`mma` 竞争。Hopper 上即使用 `cp.async` 替换 TMA（不推荐），仍缺 warpgroup async mma 的 overlap 能力——*TMA + WGMMA 是配对升级*。

== v1: CPU reference

源码 `flash_attention_cpu_reference` 与第 8 章逻辑一致：对每个 query 行，沿 key 维流式维护 $(m, ell)$ 和 accum，最后除以 $ell$。

```cpp
void flash_attention_cpu_reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& out) {
  for (int row = 0; row < kQueryCount; ++row) {
    double scores[kKeyCount];
    double row_max = -1.0e30;
    for (int key = 0; key < kKeyCount; ++key) {
      scores[key] = dot_product(&q[row * kHeadDim], &k[key * kHeadDim], kHeadDim);
      if (scores[key] > row_max) row_max = scores[key];
    }
    double row_sum = 0.0;
    double accum[kHeadDim] = {0.0};
    for (int key = 0; key < kKeyCount; ++key) {
      const double weight = std::exp(scores[key] - row_max);
      row_sum += weight;
      for (int d = 0; d < kHeadDim; ++d) {
        accum[d] += weight * static_cast<double>(v[key * kHeadDim + d]);
      }
    }
    for (int d = 0; d < kHeadDim; ++d) {
      out[row * kHeadDim + d] = static_cast<float>(accum[d] / (row_sum + kEps));
    }
  }
}
```

Teaching 版用 double 累加——作为 GPU kernel 的 golden reference。生产 FA3 在 FP16/FP8 路径仍用 FP32 做 softmax rescale（与 v2 相同），保证数值与 FP32 reference 对齐。

Teaching CPU 版*故意*两遍 scan key（先算全部分数再 softmax）——与 GPU online 单遍不等价于代码结构，但 double 精度下数值一致。验证 GPU kernel 时以 online 版 GPU 对 GPU 为准。

== v2: warp-specialized teaching kernel

```cpp
__global__ void flash_attention_v3_kernel(
    const float* q, const float* k, const float* v, float* out,
    int query_count, int key_count, int head_dim) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= query_count) return;

  float accum[kHeadDim] = {0.0f};
  float running_max = -1.0e30f;
  float running_sum = 0.0f;

  for (int key = 0; key < key_count; ++key) {
    float partial = 0.0f;
    if (lane < head_dim) {
      partial = q[row * head_dim + lane] * k[key * head_dim + lane];
    }
    const float score = __shfl_sync(0xffffffffu, warp_reduce_sum(partial), 0);

    float old_scale = 0.0f, weight = 0.0f;
    if (lane == 0) { /* update running_max, running_sum */ }

    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    weight = __shfl_sync(0xffffffffu, weight, 0);

    if (lane < head_dim) {
      accum[lane] = accum[lane] * old_scale + weight * v[key * head_dim + lane];
    }
  }
  if (lane < head_dim) {
    out[row * head_dim + lane] = accum[lane] / (final_sum + kEps);
  }
}
```

Launch：`<<<kQueryCount, kWarpSize>>>`——*一个 warp 处理一个 query 行*。

=== 逐行解读 teaching kernel

*Block 映射*：`blockIdx.x = row`——每个 block 一条 query 行，与 FA v1 register kernel 相同。Production FA3 是一个 block 处理 $B_r$ 行 query tile，内有多 warpgroup。

*Score 计算*：`lane < head_dim` 的 thread 各算 `Q[row,lane] * K[key,lane]`，再 `warp_reduce_sum`——这是 $d$ 很小时的 teaching 版 $Q K^T$；生产版用 WGMMA 一次算 $B_r times B_c$ 的 score tile。

*Online merge*：`lane == 0` 维护 `running_max`, `running_sum`，算 `old_scale = exp(m_old - m_new)` 与 `weight = exp(score - m_new)`，再 shuffle broadcast——与第 8 章公式一致。

*Output 累加*：每个 lane 持 `accum[lane]` = $O[i, ell]$ 的未归一化分子（$i$ 为 query row，$ell$ 为 lane）；每次 key 更新 `accum = accum * old_scale + weight * V[key, lane]`。

*Epilogue*：`accum / running_sum` 得最终 $O$。

#warn[
  Teaching kernel 每 key 从 global 读 $K, V$——复杂度 $O(N_q N_k d)$ HBM，*没有* FA 的 IO 优势。它只教 warp 内角色分工；IO 优化看 v1 shared / production FA3。
]

=== 与 production FA3 的 warp 角色对应

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*Teaching kernel*], [*Production FA3*], [*职责*],
    [`lane < d` 算 partial dot], [Consumer WGMMA $Q K^T$], [GEMM0：score tile],
    [`lane == 0` 维护 $(m,ell)$], [Consumer SFU：exp / max / sum], [Online softmax],
    [`lane < d` 更新 accum], [Consumer WGMMA $P V$ 或 FMA], [GEMM1：累加 $O$],
    [(无)], [Producer warpgroup TMA], [搬 $Q, K, V$ tile],
    [warp 内 shuffle], [mbarrier + TMA complete], [同步],
  ),
  caption: [*Table:* 本书 teaching kernel 与 production FA3 的 warp 角色对应。*Teaching kernel* 列指 `flash_attention_v3_kernel` 的 lane 分工；*Production FA3* 列指 producer/consumer warpgroup 职责。],
  kind: table,
)

*Observation*：Teaching kernel 在 *单 warp 内* 压缩 GEMM0/SFU/GEMM1 三类职责，但省略 producer TMA——仍是 global 每 key 读 $K,V$，故不能测 IO 优势；读 SASS 时官方 FA3 应看到 producer/consumer 两条几乎不交叉的 instruction stream。

Teaching kernel 把 producer 省略——$K,V$ 仍从 global 每 key 读。Production FA3 把 load *完全*交给 producer warpgroup。

#insight[
  Lane 角色分工是 FA3 的缩影：dot product 需要 warp 归约；softmax 标量状态只需 lane 0 更新再 broadcast；$O$ 的 $d$ 维分量各 lane 独立更新——*同一 warp 内混合 SIMD + scalar control*。Warpgroup  specialization 是把这套分工放大到 128 threads 的两条 pipeline。
]

=== Online merge 公式（与第 8 章对照）

对 query 行 $i$，处理第 $j$ 个 key 时 teaching kernel 执行：

$ m_"new" = max(m, S[i,j]), quad ell arrow.l ell dot exp(m - m_"new") + exp(S[i,j] - m_"new") $

$ tilde(O)[i,:] arrow.l tilde(O)[i,:] dot exp(m - m_"new") + exp(S[i,j] - m_"new") dot V[j,:] $

最后 $O[i,:] = tilde(O)[i,:] / ell$。Production FA3 在 *tile 粒度* 做同样 merge：一个 WGMMA 产出 $B_r times B_c$ 的 $S$ tile，warp 协作 reduce max / sum，再 RS-GEMM 更新 $B_r times d$ 的 $O$ tile——*公式不变，并行粒度变大*。

=== ncu 实测

#ncu-snapshot(
  version: "warp_streaming (single-buffered)",
  size: [$N_q = 128$, $N_k = 512$, head_dim = 32],
  rows: (
    ("Duration",            "483.9 µs", ""),
    ("Memory SOL",          "10.2 %",   "*较 FA v2 反而低*——教学 kernel 简化，未 chase peak"),
    ("Compute SOL",         "1.8 %",    ""),
    ("Achieved Occupancy",  "1.9 %",    "同样 warp-per-row 设计"),
  ),
)

*比 FA v2 慢*——这是本章一开始就 disclaim 的：*A100 上跑 FA v3 teaching kernel 只能观察结构，看不到 SM90 特性带来的加速*。原因：

- FA v3 的核心 innovation（TMA async load、WGMMA、setmaxnreg producer/consumer 寄存器分区）*只在 SM90+（Hopper H100）架构上有硬件支持*。A100 (SM80) 上跑 v3 teaching kernel = 拿到"结构正确、性能不如 FA v2"的 baseline。
- 本 kernel 用 scalar dot + 单 warp per query row + 单 buffer $K/V$——比 FA v2 warp_specialised 还少了几个 optimization（v2 的 warp specialization 在 A100 上还有意义，v3 的 producer-consumer 分离只在 H100 上有 TMA 支持）。

#verdict(
  problem: [单 buffered $K/V$：加载下一个 tile 时 compute 完全 stall；SM80 上没有 TMA 硬件加速 async load],
  evidence: [duration 484 μs vs FA v2 warp 330 μs（慢 47%）；memSOL 10% 说明 memory pipeline 完全串行],
  next: [v3 pipeline 加入 $"kPipelineStages"$-buffered $K/V$ smem——load stage $s$ 和 compute stage $s-1$ 尝试用 `__syncthreads` 结构性 overlap（不是真正 async，因为 A100 没 TMA）]
)

== v3: pipeline teaching kernel

```cpp
__global__ void flash_attention_v3_pipeline_kernel(...) {
  __shared__ float k_stage[kPipelineStages][kHeadDim];
  __shared__ float v_stage[kPipelineStages][kHeadDim];
  // prologue: load key=0 into stage 0
  for (int key = 0; key < key_count; ++key) {
    const int stage = key % kPipelineStages;
    const int next_stage = (key + 1) % kPipelineStages;
    // prefetch key+1 into next_stage  ← 对应 TMA 预取
    __syncthreads();
    // compute score/softmax/accum using k_stage[stage], v_stage[stage]
    __syncthreads();
  }
}
```

这是第 4 章 matmul ping-pong 的 attention 版——`kPipelineStages = 2`。Production FA3 在同一 smem ring buffer 上挂 `mbarrier`：producer 在 stage $i mod S$ 写 TMA，consumer 在 stage 算完后 `arrive` 释放。

=== Pipeline kernel 与 matmul v5 的对应

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*Matmul v5 (ch.4)*], [*FA3 pipeline teaching*], [*Production FA3*],
    [`a_tiles[2][BM][BK]`], [`k_stage[2][d]`], [TMA ring $K_j, V_j$],
    [prologue load tile 0], [preload key 0], [TMA load $Q$, $K_0$],
    [compute stage *i*], [softmax on stage *i*], [WGMMA + softmax],
    [prefetch stage *i+1*], [prefetch key *i+1*], [TMA async prefetch],
    [`__syncthreads`], [`__syncthreads`], [`mbarrier` + release],
  ),
  caption: [*Table:* 第 4 章 matmul v5 pipeline、FA3 teaching pipeline kernel 与 production FA3 的控制流对应。三列对照 smem ring buffer、prologue、compute/prefetch 阶段及同步原语（`__syncthreads` vs `mbarrier`）。],
  kind: table,
)

*Observation*：*控制流形状同构*——`k_stage[2][d]` 对应 TMA ring $K_j,V_j$，但 teaching 版用 `__syncthreads` 无 true async overlap；这正是 Step 2「把 sync 换成 mbarrier」的 hop 点。

Teaching 版用 `__syncthreads` 保证 smem stage 不被覆写——与 matmul teaching kernel *同样*没有真正 async overlap，但 *控制流形状* 与 Algorithm 1–2 一致。读 SASS 时 production FA3 会把 prefetch 与 WGMMA 交错。

=== 为什么 v2 里 softmax 是「串行插入」

FA v2（Ampere）consumer 主循环（Algorithm 1 风格）：

```
wait K_j loaded
S = Q K_j^T          (mma.sync, 同步完成)
softmax(S) → P̃       (必须等 S 完整)
wait V_j loaded
O += P̃ V_j           (mma.sync)
```

`mma.sync` 在 issue 点阻塞直到完成——softmax 无法在 $Q K^T$ 执行期间插入。且 load/compute 常由同一 warp 交错，instruction issue 竞争严重。

H100 上 exp 吞吐 ~3.9 TFLOPS vs matmul ~989 TFLOPS（FP16）——head dim 128 时 exp 可占 ~50% cycle（论文 §3.1）。*不 overlap 则一半 SM 时间在等 SFU 或 TC 空闲*。

=== Causal mask 在 FA3 中的处理

与第 8–9 章相同：在 $S$ tile 算完、softmax 前应用 $-infinity$ mask。Hopper 路径的额外考量：

- *Skip tile*：若整个 $K_j$ tile 的 key index 均 $>$ 当前 query block 的最大 index，producer 可 skip TMA（省 bandwidth）。
- *Partial tile*：tile 跨 causal 对角线时，在 register 上对 $S$ 做 mask 再 softmax——不能 skip WGMMA，但可 skip 无效行的 exp。
- *FP8 + causal*：mask 后 tile 内有效元素稀疏，block quant dynamic range 改善，但 warp 利用率下降——论文 FP8 causal 略慢于 cuDNN 的原因之一。

=== ncu 实测

#ncu-snapshot(
  version: "pipeline (multi-buffered smem)",
  size: [$N_q = 128$, $N_k = 512$, head_dim = 32],
  rows: (
    ("Duration",            "438.9 µs", "vs warp_streaming 484 μs，*快 10%*"),
    ("Memory SOL",          "11.3 %",   "略高于单 buffer 版"),
    ("Compute SOL",         "2.4 %",    "*↑ 从 1.8%*——compute 相对忙一点"),
    ("Achieved Occupancy",  "1.9 %",    ""),
  ),
)

*10% 提升* —— 结构性 overlap 通过多 buffer 拿到了小额收益：当 compute stage $s-1$ 在算 $S/P$ 时，load stage $s$ 已经在往 smem 里搬 $K_j, V_j$。但 A100 上没有 TMA，load 走的还是普通的 `LDG.E`，需要 warp 主动 issue —— *这不是真的 async*，只是 shared-memory pre-fetch，`__syncthreads` 依然阻塞。

在 H100 上，这个 kernel 骨架换成 producer/consumer warp specialization + TMA + `mbarrier` + WGMMA 之后，overlap 会做到真正的 async，性能可以 2-4× 提升。SM80 上跑这个 kernel 主要是*结构演示*。

#final-verdict(
  status: [FA v3 教学 kernel 展示了 producer/consumer + multi-buffered pipeline 的骨架。],
  note: [A100 上跑这套 kernel 只能拿到 A100 上的性能上限（比 FA v2 的 warp specialization 略慢，因为 v3 引入的复杂度在 SM80 上没有对应硬件支持）。真正的 FA v3 加速：需要 H100 (SM90+)，用 TMA async load 让 producer warp group 独立发 memory 请求，用 WGMMA 让 consumer warp group 独立发 tensor core 请求，两者互不阻塞。这一节到此完成教学目的——把 FA v1 → v2 → v3 的三步演进讲清楚。生产实现请参考 flash-attn 官方库。]
)

== Innovation 1：Producer-Consumer Warp Specialization

=== 角色划分

一个 CTA（处理 $Q$ 的一个 row-block）内：

*Producer warpgroup*（通常 1 个 warpgroup = 128 threads，实际只需少量 thread 发 TMA）：

1. `setmaxnreg` 降低 register 预算。
2. TMA load $Q$（一次）→ commit mbarrier。
3. Loop $j = 0..T_c-1$：等 stage $(j mod S)$ 空 → TMA load $K_j, V_j$ → commit。

*Consumer warpgroup*（1–2 个 warpgroup）：

1. `setmaxnreg` 提高 register 预算。
2. 等 $Q$ ready → WGMMA $S^{(j)} = Q K_j^T$。
3. Softmax + rescale $O$。
4. WGMMA $O += tilde(P)^{(j)} V_j$。
5. `arrive` 释放 smem stage 给 producer。

=== mbarrier pipeline

Hopper `mbarrier` 替代 FA2 的 `__syncthreads` + `cp.async.wait` 组合：

```cpp
// 概念性伪代码
__shared__ uint64_t tma_barrier[kStages];
__shared__ uint64_t wgmma_barrier;

// Producer:
mbarrier.init(&tma_barrier[s], expected_tx_count);
cp.async.bulk.tensor... [&tma_barrier[s]];  // TMA 完成 → barrier arrive
mbarrier.arrive(&tma_barrier[s]);           // producer 侧 commit

// Consumer:
mbarrier.try_wait(&tma_barrier[s], phase);  // 等 K,V tile ready
// ... WGMMA compute ...
mbarrier.arrive(&release_barrier[s]);       // 通知 producer 可覆写 stage
```

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [*阶段*], [*同步对象*],
    [TMA $K_j, V_j$ 完成], [`mbarrier` transaction complete → consumer wait],
    [Consumer 算完 stage $j$], [release → producer 可写 $(j mod S)$],
    [WGMMA $Q K^T$ 完成], [wgmma.wait\_group],
    [CTA 内 warpgroup 间 pingpong], [`bar.sync` / cluster barrier],
  ),
  caption: [*Table:* FA3 CTA 内 producer-consumer pipeline 各阶段的同步对象。阶段列对应 TMA load、consumer compute、WGMMA retire 与 warpgroup pingpong；同步对象列说明 `mbarrier`、`wgmma.wait_group`、`bar.sync` 各管什么。],
  kind: table,
)

*Observation*：*三类 barrier 不可混*——`mbarrier` 管 TMA transaction complete + smem stage release，`wgmma.wait_group` 管 tensor core retire，`bar.sync` 管 warpgroup 间 issue 协调；CUTLASS pipeline 类把这套协议封装，手写时 phase flip 错一位即 deadlock。

#warn[
  `mbarrier` phase 是 1-bit 翻转计数——producer/consumer 必须严格配对 `expect_tx` 与 `complete`，否则 deadlock 或 data race。CUTLASS pipeline 类封装了这套协议；手写 kernel 时先看 generated SASS 验证 barrier 顺序。
]

=== 动机（面试版）

*为什么*要 producer-consumer？

1. TMA 指令由少数 thread 发起即可——若所有 warp 都参与 load，consumer 的 WGMMA issue 被挤占。
2. Producer 低 register → 更多 CTA 可驻 SM；Consumer 高 register → 大 tile accumulator 不 spill。
3. Load latency 藏在 consumer 计算后面——*software pipeline depth* 增加。

=== Algorithm 1 骨架（论文，CTA 视角）

Production FP16 FA3 在不 overlap 的 consumer 路径上等价于：

```
Producer WG:
  TMA load Q → commit
  for j in 0..T_c-1:
    wait stage (j % S) empty
    TMA load K_j, V_j → stage (j % S) → commit

Consumer WG:
  init O=0, l=0, m=-inf
  wait Q ready
  for j in 0..T_c-1:
    wait K_j in smem
    S = WGMMA(Q, K_j^T)      // SS-GEMM
    update m, l, P̃ = exp(S-m), rescale O
    wait V_j in smem
    O += WGMMA_RS(P̃, V_j)   // RS-GEMM
    release stage (j % S)
  write O, logsumexp to HBM
```

Algorithm 2 把 Consumer 的 inner loop 换成 2-stage register pipeline（前文 §Innovation 2）。*Producer 不变*——这是 warp specialization 的 clean 接口。

=== MQA / GQA

Multi-Query / Grouped-Query Attention：多个 query head 共享同一 $K, V$ head。FA3 沿 FA2 做法调整 tensor indexing——*不复制* $K, V$ in HBM；TMA descriptor 的 head stride 指向 shared KV head。Producer 每个 KV head 的 tile 可被多个 Q head 的 CTA 复用（L2 cache 友好），但 smem 仍 per-CTA 私有。

== Innovation 2：2-Stage GEMM-Softmax Overlap

=== v2 串行 vs v3 并行

论文 Algorithm 2 的核心：在 register 里同时持 $S_"cur"$ 和 $S_"next"$，主循环：

```
// iteration j:
WGMMA S_next = Q K_{j+1}     ; commit, NO wait
WGMMA O += P̃_cur V_{j-1}    ; commit, NO wait
wait S_next ready
softmax(S_next) → P̃_next    ← 与上面 WGMMA 并行
wait PV complete; rescale O
```

*Pingpong warpgroup*（§3.1）：两个 consumer warpgroup 交替——WG1 做 softmax 时 WG2 做 GEMM（`bar.sync` 协调 issue 顺序），进一步把 SFU 和 TC 填满。

=== Timeline 图示

*FA v2（串行）*——同一 warpgroup，一个 key tile：

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 4pt,
    align: (center, center),
    [*周期*], [1], [2], [3], [4], [5], [6],
    [TC], [GEMM $Q K^T$], [—], [—], [GEMM $P V$], [—], [—],
    [SFU], [—], [softmax], [—], [—], [—], [—],
    [TMA], [load $K,V$], [—], [—], [—], [load next], [—],
  ),
  caption: [*Table:* FA v2（Ampere，`mma.sync` 同步 GEMM）对单个 key tile 的理想化周期 timeline。行 TC/SFU/TMA 表示各硬件单元在各周期 Busy（— 表示空闲）；6 个周期覆盖 load $K,V$、GEMM $Q K^T$、softmax、GEMM $P V$。],
  kind: table,
)

*Observation*：*TC 与 SFU 互斥*——softmax 必须等 GEMM0 完整完成（`mma.sync`），周期 2–3 TC 全空而 SFU 忙；TMA load next 与 compute 也串行。这是 H100 上 exp 占 ~50% cycle 且 FA2 仅 ~35% peak 的 timeline 解释。

*FA v3（2-stage + async）*——理想化重叠：

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 4pt,
    align: (center, center),
    [*周期*], [1], [2], [3], [4], [5], [6],
    [TC/WG2], [GEMM $Q K_0$], [GEMM $Q K_1$], [GEMM $P V_0$], [GEMM $Q K_2$], [GEMM $P V_1$], [GEMM $Q K_3$],
    [SFU/WG1], [—], [softmax 0], [softmax 1], [softmax 2], [softmax 3], [softmax 4],
    [TMA/prod], [load 0], [load 1], [load 2], [load 3], [load 4], [load 5],
  ),
  caption: [*Table:* FA v3（2-stage + async WGMMA）理想化周期 timeline。TC/WG2、SFU/WG1、TMA/prod 三行展示 warpgroup specialization 下 GEMM、softmax、TMA load 的重叠；6 周期内 TC 几乎无空档。],
  kind: table,
)

*Observation*：与上一表对比：*TC 周期 1–6 几乎连续 GEMM*，SFU 在 WG1 并行 softmax *上一 tile* 已 wait 的 $S_"cur"$——async WGMMA 把 issue 与 complete 拆开才留出这条空隙；论文 ablation 570→661 TFLOPS 主要来自这张 timeline 的实现。

#insight[
  重叠能成立，因为 WGMMA 是 *async*：softmax 读的是 *上一轮* 已 wait 完成的 $S_"cur"$，而 TC 在算 *下一轮* $Q K_{j+1}^T$。FA2 的 `mma.sync` 把这两步锁在同一时间点。论文 ablation：head dim 128, seqlen 8448 上，2-stage pipeline + warp specialization 从 570 → 661 TFLOPS；pingpong 可到 620–640 TFLOPS。
]

=== 编译器与 register pressure

NVCC 可能 reorder 指令——论文 §B.2 用 SASS 验证 overlap 确实发生。2-stage 额外需要 $S_"next"$ register buffer（大小 $O(B_r times B_c)$ per block）——与更大 tile 的优化冲突，需 profile 权衡。3-stage 变体（Appendix B.3）进一步 overlap 第二个 WGMMA 与 softmax，但 register 压力更大。

=== Ablation 数字（论文 Table 2）

固定 config：batch=4, seqlen=8448, nheads=16, hdim=128，非 causal FP16 forward：

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right),
    [*配置*], [*TFLOPS*],
    [Baseline（无 specialization / 无 overlap）], [~570],
    [+ warp specialization], [~620–640],
    [+ 2-stage GEMM-softmax pipeline], [~661],
  ),
  caption: [*Table:* FlashAttention-3 论文 Table 2 ablation（H100 FP16 forward）。固定 batch=4、seqlen=8448、nheads=16、hdim=128、非 causal；*配置* 列逐项叠加 warp specialization 与 2-stage GEMM-softmax pipeline；*TFLOPS* 单位为 TFLOPS（算力吞吐）。],
  kind: table,
)

*Observation*：三项创新 *可叠加*——baseline 570 → +specialization ~620–640 → +overlap ~661 TFLOPS，说明 load/compute 争用与 SFU/TC 互斥是 *两个独立瓶颈*；specialization 管前者，2-stage overlap 管后者。

说明三项创新*可叠加*：specialization 解决 load/compute 争用；overlap 解决 SFU/TC 互斥。

== Innovation 3：FP8 Attention

=== 量化位置

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*张量*], [*典型量化点*], [*说明*],
    [$Q$], [GEMM 前，block-wise], [与 rotary 融合；k-major for WGMMA],
    [$K$], [同 $Q$], [block scale $s_Q, s_K$ 吸收进 $S$ scale],
    [$V$], [GEMM 前 + layout 变换], [需 sequence-major tile → in-kernel transpose],
    [$P$], [通常*不*存 FP8 到 HBM], [softmax 后在 rmem，permute 后作 GEMM1 的 A],
    [$O$ accumulator], [FP32], [与 v2 相同，最终 cast 到 FP16/BF16 输出],
  ),
  caption: [*Table:* FA3 FP8 attention 各张量的量化位置与说明。张量 $Q,K,V,P,O$ 列给出 block-wise quant、layout 变换、FP32 accumulator 等策略；说明列强调 k-major、in-kernel transpose、$P$ 不写 HBM 等 Hopper 约束。],
  kind: table,
)

*Observation*：*量化位置跟着 data flow 走*——$Q,K$ 在 GEMM 前 block quant 且 k-major；$V$ 多一步 transpose；$P$ 驻 rmem 作 GEMM1 operand；$O$ 仍 FP32 acc——FP8 工程成本集中在 layout（Innovation 3 §3.3），不是多一个 `__float2fp8`。

=== Block scaling

Per-tensor scaling（一个标量 $s_Q, s_K, s_V$）在 outlier 特征下误差大。FA3 用 *block quantization*：

$ tilde(Q)_"block" = "quantize"(Q_"block" / s_"block"), quad s_"block" = max(abs(Q_"block")) / "FP8_max" $

每个 $B_r times d$ 或 $B_c times d$ block 一个 scale。Attention 天然按 block 遍历 $K/V$——scale 在 kernel 内*零额外开销*吸收到 $S$ 的缩放里。

=== Incoherent processing

Outlier 使 FP8 动态范围不够。FA3 在量化前对 $Q, K$ 乘随机正交矩阵 $M$：

$Q_M K_M^T = Q K^T$

选 $M = D_1 H D_2$（对角 ±1 × Hadamard）——$O(d log d)$ 乘法，可与 rotary 融合。*Spread out* outliers，降低 block 内 dynamic range。

=== 数值例子：block quant vs per-tensor

设某 block 内 $Q$ 元素 $=[0.01, 0.02, 8.0]$（outlier 8.0）：

- *Per-tensor* scale $s = 8.0 / 448 approx 0.018$：小元素量化后全进同一 bin，信息丢失。
- *Per-block* scale 仅对该 block：小元素相对精度提升；outlier 8.0 仍饱和，但 *不影响其他 block*。

Incoherent processing：$Q' = Q M$，若 $M$ 为 Hadamard，每个 $Q'_i$ 是 $Q$ 的均匀混合——outlier 能量扩散到多个 channel，block max 下降 → FP8 量化 SNR 提升。论文：相对 baseline FP8 attention（per-tensor），FA3 FP8 误差 *2.6× 更低*。

=== 精度实验数据

论文 §4.3（H100，LLM 典型 config）：

- FP16 FA3 与 FP16 FA2 *数值误差同级*（中间 softmax rescale 仍 FP32）。
- FP8 FA3 + block quant + incoherent processing vs *baseline FP8 attention（per-tensor quant）*：误差降低 *2.6×*。
- Head dim 256 forward：FP8 达 ~1.2 PFLOPS；causal mask 下 FP8 与 cuDNN 互有胜负（head dim 64 FP8 领先，128/256 非 causal 持平）。

#warn[
  FP8 *训练*稳定性仍不如 BF16——梯度 E5M2、loss scaling、accumulator 精度需全流程验证。FP8 FA3 当前主要价值在 *inference throughput*；训练领域更多用 FP8 GEMM + BF16 attention 混合。
]

=== FP8 的 layout 陷阱（面试高频）

1. $V$ 在 HBM 是 head-contiguous，FP8 WGMMA 要 sequence-contiguous → in-kernel LDSM/STSM transpose。
2. GEMM0 的 FP32 acc layout ≠ GEMM1 的 FP8 A layout → `byte_perm` 重排 $tilde(P)$。
3. Transpose 后的 $V$ 列序需与 permute 后的 $P$ 行序匹配——否则 $P V$ 数学错误。

=== In-kernel $V$ transpose 流程（概念）

Producer TMA 把 $V_j$ 以 mn-major 搬进 smem → Consumer 或 Producer 用 `ldmatrix`/`stmatrix` warp 协作转置到 k-major staging buffer → 第二个 WGMMA 读转置后的 tile。论文：转置可在 *上一轮* 两个 WGMMA 的 shadow 里执行，摊销 latency。

=== E4M3 vs E5M2 怎么选

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*格式*], [*动态范围*], [*典型用途*],
    [E4M3], [较小，精度稍好], [Forward $Q,K,V$ activations],
    [E5M2], [较大，精度更粗], [Backward gradients],
  ),
  caption: [*Table:* Hopper FP8 两种格式 E4M3 与 E5M2 的动态范围与典型用途。动态范围列对比 exponent/mantissa 分配；用途列区分 forward activation (E4M3) 与 backward gradient (E5M2)。],
  kind: table,
)

*Observation*：Inference forward 首选 E4M3（精度稍好）+ block scale；training 梯度用 E5M2 保 dynamic range——混用是 FA3 FP8 *推理* 与 *训练* 路径分叉的第一道选择，不能假设一种格式通吃。

Inference forward 首选 E4M3 + block scale；training 需 A/B test loss curve，不能假设 FP8 attention 与 BF16 可互换。

== 性能对比：v1 vs v2 vs v3

H100 80GB SXM5，论文 Fig. 5–7 量级（FP16 forward，无 causal，head dim 64/128）：

#ladder(
  ("Standard attention",  "materialize $S/P$",                    "~200 TFLOPS"),
  ("FA v1/v2",            "Ampere sync pipeline",                 "~280–340 TFLOPS"),
  ("FA2 Triton (Hopper)", "H100 instr, 无 full specialization",    "~380–400 TFLOPS"),
  ("cuDNN FA2",           "vendor tuned",                         "~400–460 TFLOPS"),
  ("FA v3 FP16",          "TMA+WGMMA+overlap",                    "up to ~740 TFLOPS"),
  ("FA v3 FP8",           "block quant + FP8 TC",                 "~1200 TFLOPS"),
)

相对峰值：H100 FP16 tensor ~989 TFLOPS → FA3 ~75% utilization；FA2 ~35%。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*对比*], [*Forward*], [*Backward*],
    [FA3 vs FA2], [1.5–2.0×], [1.5–1.75×],
    [FA3 vs Standard], [3–16×], [—],
    [FA3 FP16 vs cuDNN], [中长 seq 更快], [—],
  ),
  caption: [*Table:* FA3 相对 FA2、Standard attention 及 cuDNN 的性能倍数（FlashAttention-3 论文 Fig. 5–7 量级）。*对比* 列说明 baseline；*Forward* / *Backward* 列为倍数（无单位）；H100 FP16 tensor peak ~989 TFLOPS 作参照。],
  kind: table,
)

*Observation*：Forward 1.5–2× vs FA2、3–16× vs Standard 的主因是 *IO + execution* 双优化；backward 1.5–1.75× 略低因 recomputation FLOPs 占比高、overlap 更难——与 §Backward 简述一致。

#note[
  Teaching kernel（本书源码）*无任何上述 TFLOPS*——规模 $N_q=4, N_k=16, d=8$，launch 开销主导。性能数字一律指 CUTLASS/CUDA 版 FA3。跑 benchmark 用官方 repo + $N >= 8192$。
]

=== 论文 benchmark 设置（复现参考）

- GPU：H100 80GB SXM5
- Sequence length：512, 1k, …, 16k；总 token 数固定 16k（batch × seqlen）
- Hidden 2048；head dim 64 / 128 / 256
- Forward FLOPs：$4 times "seqlen"^2 times d times "nheads"$（causal 约 ÷2）
- Backward FLOPs $approx 2.5 times$ forward（5 vs 2 matmul + recomputation）

Fig. 5（head dim 64, 无 causal）：FA3 FP16 在 8k–16k seq 达 700+ TFLOPS；Standard attention 324 TFLOPS 后 OOM。Fig. 7（head dim 256, FP8 forward）：峰值 ~1200 TFLOPS。

== 局限

1. *硬件*：仅 Hopper (sm_90) 完整支持；Ada/Ampere 无 TMA/WGMMA。
2. *FP8 训练*：attention 量化误差在 deep stack 累积；outlier 处理增加工程复杂度。
3. *Register / tile tradeoff*：2-stage/3-stage pipeline 与大 tile 互斥——autotune 依赖 shape。
4. *Backward*：仍比 forward 慢 ~2.5× FLOPs（5 matmul + recomputation）；FP8 backward 更不成熟。
5. *Causal + FP8 + head dim 256*：论文显示 FP8 在 causal 下可能落后 cuDNN——mask 破坏规律 tile 时 overlap 收益下降。

=== Backward 简述

FA3 backward 沿用 FA2 的 recomputation：不存 $P$，按 tile 重算 $S = Q K^T$，用 forward 存的 log-sum-exp $(m, ell)$ 求 `dP`，链式得到 `dQ, dK, dV`。Hopper 路径同样 split producer/consumer，WGMMA 算五个 matmul 型步骤（论文 Appendix B.1）。Forward 1.5–2× 加速；backward 1.5–1.75×——略低因为 recomputation FLOPs 占比更高、overlap 更难。

=== CUTLASS / 官方实现

FlashAttention-3 用 CUTLASS 3.x 的 CUTE abstractions 封装 WGMMA/TMA/mbarrier。面试不要求手写 PTX，但应能读 `GemmUniversal` 风格的 *mainloop*（producer）与 *mma*（consumer）分离。Repo：`github.com/Dao-AILab/flash-attention`（Hopper 分支）。

== ncu 该看什么

H100 上 profile FA3（或 CUTLASS GEMM proxy）：

```bash
ncu --set full --section SpeedOfLight ./flash_attn_h100_benchmark
```

关键 metric：

- `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`：WGMMA/TC 利用率（目标 >70%）。
- `sm__inst_executed_pipe_sfu.avg.pct_of_peak_sustained_active`：exp 等 SFU——overlap 成功时应与 TC 同时非零。
- `gpu__tma_throughput.avg.pct_of_peak_sustained_elapsed`：TMA 是否饱和。
- `smsp__sass_thread_inst_executed_op_wgmma`：WGMMA 指令计数。
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`：smem swizzle 是否正确。

Teaching kernel 在 ncu 里只会看到 shuffle + FMA——用于验证 correctness，不用于 chase FA3 性能。

=== 诊断 checklist

若 `pipe_tensor` 低但 `dram` 高：TMA 或 tile 太小，compute 未主导——查 grid 大小与 $B_r, B_c$。

若 `pipe_sfu` 高但 `pipe_tensor` 低：softmax 仍串行——检查 SASS 是否有 WGMMA 与 exp 重叠。

若 `bank_conflicts` 非零：smem swizzle 与 TMA box layout 不匹配——对照 CUTLASS layout。

若 TMA throughput 低：producer warpgroup 可能被 `__syncthreads` 误伤——应改用 mbarrier。

== 实测

*读表前先定性*：正文讲 WGMMA、TMA、producer-consumer warp specialization、mbarrier——这些都是 *Hopper (H100, SM90) 生产 FA3* 的设计语言。本书 `10_flash_attention_v3.cu` 在 *A100 (SM80)* 上跑：scalar FFMA + warp shuffle + smem ping-pong，*没有* 任何一条 WGMMA/TMA/mbarrier 指令。ncu 数字反映的是*教学 kernel 结构*（lane 分工、双 stage smem），*不是* CUTLASS FA3 在 H100 上的 TC/TMA 利用率。论文里的 740 TFLOPS / 75% peak 只能引用官方 benchmark；本节只验证骨架正确性与 ping-pong 控制流。

$N_q = 4$, $N_k = 16$, $d = 8$（$Q,K,V,O$ 各 512 B，整 problem $< 4$ KB），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。TC % = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；warp % = `sm__warps_active.avg.pct_of_peak_sustained_elapsed`；HBM % = `dram__bytes.sum.pct_of_peak_sustained_elapsed`。

Launch 配置（两版均为 1 warp / query 行）：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right, left),
    [*version*], [*grid*], [*block*], [*active threads*], [*演示点*],
    [v3], [(4, 1, 1)], [(32, 1, 1)], [128（4 block × 32）], [warp 内 lane 角色分工；无 smem],
    [v3-pipeline], [(4, 1, 1)], [(32, 1, 1)], [128], [2-stage K/V smem ping-pong],
  ),
  caption: [*Table:* ch10 FA3 teaching kernel 的 launch 配置。*version* / *grid* / *block* 对应 CUDA `<<<grid, block>>>`（`launch__grid_size` / `launch__block_size`）；*active threads* = grid×block 活跃 thread 数；*演示点* 说明 warp 分工 vs smem ping-pong。规模 $N_q=4, N_k=16, d=8$，A100 SXM4。],
  kind: table,
)

*Observation*：`<<<4, 32>>>` 仅 128 active threads、4 block——*远填不满 108 SM*，与 ch4 matmul 同理；ncu 的 warp %/HBM % 在此规模无诊断价值，只能读 `issued/32`、`mem stall` 等结构性指标。

#include "../bench/10_flash_attention_v3.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *v3-pipeline 15.49 μs vs v3 17.66 μs——快约 12%*（$(17.66 - 15.49) / 17.66 approx 12.3%$）。pipeline 版用 2-stage smem ring buffer（`k_stage[2][d]`, `v_stage[2][d]`，136 B smem）在算 key $j$ 时预取 key $j+1$——*控制流形状*对齐 FA3 的 TMA circular buffer，但 teaching 版仍用 `__syncthreads` 而非 `mbarrier`，没有真正 async overlap；12% 主要来自 smem 预取减少 global $K,V$ 读等待，*不是* Hopper pipeline 的全部收益。生产 FA3 在 H100 上 TMA + WGMMA 双缓冲相对 FA2 的 1.5–2×——*与这 12% 不可比*。

+ *TC % 两版均为 0.0%——预期，且是诚实结论。* 教学 kernel 走 CUDA core FFMA + shuffle，tensor pipe 从未激活。在 H100 上 profile 官方 FA3 才应看到 TC % >70%、`gpu__tma_throughput` 非零、`smsp__sass_thread_inst_executed_op_wgmma` 计数——*本表 TC % = 0 不能用来否定 FA3 论文，只能说明我们没在测真 FA3*。

+ *warp % / HBM % 均为 0.0–0.1%——micro 规模无并行度与带宽诊断价值。* `<<<4, 32>>>` 仅 4 个 warp 活跃，108 SM 几乎全空；工作集 $< 4$ KB 全 L2 resident。上文 ladder 里 production FA3 的 ~75% H100 peak、FP8 ~1.2 PFLOPS 均来自*论文/CUTLASS*，不是本表。

#figure(
  hbar-chart(
    (
      ("v3", 17.66),
      ("v3-pipeline", 15.49),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)`：pipeline 版比 warp-specialized 版快 ~12%——smem ping-pong 骨架有效，但绝对时间仍在 15–18 μs 带内（launch + 极短 kernel 主导）。],
)

#figure(
  warp-grid(
    rows: 2, cols: 4,
    cell: 0.35,
    active: ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3)),
    row-labels: ("Producer WG（概念）", "Consumer WG（概念）"),
    title: "Production FA3 warp specialization：Producer 发 TMA，Consumer 跑 WGMMA + softmax",
  ),
  caption: [
    一行 = 一个 warpgroup（4 warp 简化为 4 格）。Producer 仅少量 thread/lane 活跃（TMA 发起）；Consumer 全 warpgroup 算 WGMMA 与 SFU。
    *本书 teaching kernel 在单 warp 内混合这些角色*——没有真正的 producer/consumer 分离；读 SASS 时官方 FA3 应看到两条 instruction stream 几乎不交叉。
  ],
)

*diag 表读关键教学点：*

*a) v3 `issued/32 = 14.0`，`pred_on/32 = 11.5`——一 warp 内 lane 角色分工，不是 warp divergence*

#raw("<<<4, 32>>>") 每 block 恰好一 warp：`lane < d`（$d=8$）算 partial dot 与 accum 更新，`lane == 0` 维护 $(m, ell)$，`__shfl_sync` 广播——并行 phase 拉高 `issued/32` 到 14.0。issued − pred_on $approx 2.5$ 来自 `if (lane < head_dim)`、`if (lane == 0)` 等 guard 的 *predicated-off lane*——*predication*，不是不同 basic block 的真 divergence（SPEC §1：单边 `if` 空 `else` 编译为 predicated 指令）。

#figure(
  warp-lanes(active: range(8), cell: 0.34,
             title: [v3 score/accum 阶段：lane 0–7 活跃；softmax merge 仅 lane 0]),
  caption: [Teaching kernel 把 production FA3 的 Consumer 职责压缩进*一个 warp*——lane 0–7 对应 WGMMA 的 dot/accum，lane 0 对应 SFU 标量控制。],
)

*b) v3-pipeline `issued/32 = 16.5`，`pred_on/32 = 12.3`——ping-pong 略抬 lane 利用率*

prefetch + compute 双 phase 让更多 lane 同时参与 smem/global 协作，`issued/32` 高于 v3（16.5 vs 14.0），time 快 ~12%。issued − pred_on $approx 4.2$ 略大于 v3——pipeline 里更多 `__syncthreads` 边界上的 guard predication，仍*不是* branch divergence。

*c) `smem conf. = 0`（两版）——metric 证实无 bank conflict*

不能从 `k_stage[stage][d]` 的 `[stage][d]` 布局单独推断 conflict；ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld+st.sum = 0` 才作数。v3-pipeline 136 B smem、v3 无 smem——conflict 为 0 也可能表示「有 smem 但访问模式干净」。

*d) `barrier stall`：v3 = 0.00，v3-pipeline = 0.32；`mem stall`：v3 = 5.53，v3-pipeline = 3.04。* pipeline 版 `__syncthreads` 更多（stage 切换）→ barrier stall 从 0 升到 0.32，仍很小。mem stall 降（5.53 → 3.04）与 smem 预取隐藏 global latency 的*方向一致*——kernel $< 20$ μs，*不能*用来断言 memory-bound 或 TMA 饱和。

*regs / smem fingerprint：* v3：32 regs，0 B smem（每 key 直读 global $K,V$）。v3-pipeline：28 regs，136 B smem（`k_stage[2][8]` + `v_stage[2][8]` ping-pong）——用 smem 换 latency hide；对应 production FA3 里 TMA ring buffer 的*资源账本*，但 teaching 版无 TMA 硬件单元。

*无信息或为零的 metric：*

- `TC %`：全表 0.0——scalar FFMA；*不能*用本表讨论 WGMMA 利用率。
- `HBM %`：全表 0.0——L2 resident；*不能*验证 FA 的 IO 优势。
- `warp %`：0.1%——4 block 填不满 GPU。

#insight[
  本章 A100 实测的*唯一可靠结论*：两 kernel PASS CPU reference；pipeline 版更快且 `mem stall` 更低——ping-pong 控制流正确。WGMMA/TMA/warp specialization 只能在*设计层*理解；要验证 FA3 论文数字，必须在 H100 上 ncu 官方实现，看 TC、TMA、SFU 三套 pipe 是否同时非零。
]

#warn[
  *误区*：「跑了 `flash_attention_v3_kernel` 就等于理解 FA3 性能。」——错。A100 teaching kernel 与 H100 CUTLASS FA3 共享 online merge 公式，*不共享* execution model。也不要把 `issued/32` 低于 32 说成「严重 warp divergence」——用 `issued/32` vs `pred_on/32` 区分 predication 与真 divergence；本规模首先是 lane 角色分工（`lane == 0` 串行 softmax）造成的*结构性 lane 利用模式*。面试报 740 TFLOPS / 75% peak 必须注明 *H100 + 官方实现 + $N >= 8192$*。
]

Launch（正确性对照）：

```bash
make build/10_flash_attention_v3 && ./build/10_flash_attention_v3
```

输出 `Check: PASS` 表示两 GPU kernel 与 CPU reference 对齐（容差 $10^(-4)$）。

=== 全书 ladder 收尾

从第 1 章 vector add 到本章 FA3，optimization 的主线始终是：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [Memory-bound →], [合并访问、向量化、grid-stride（ch.1–2）],
    [Reduce / scan →], [warp shuffle、online 公式（ch.2–3）],
    [Compute-bound GEMM →], [smem tile、tensor core、cp.async（ch.4）],
    [Fusion →], [LayerNorm/MLP epilogue（ch.5–6）],
    [Attention IO →], [FA v1 分块、v2 parallel（ch.7–9）],
    [Hopper async →], [TMA + WGMMA + FP8（本章）],
  ),
  caption: [*Table:* 全书 optimization ladder 主线收尾对照。左列 problem 类型（Memory-bound、Reduce、GEMM、Fusion、Attention IO、Hopper async）；右列对应章节与关键技术路径。],
  kind: table,
)

*Observation*：*最后一里路是 async 对齐*——从 ch.1 合并访存到本章 TMA+WGMMA+FP8，算法 cleverness 让位于与 hardware agent/locale 配对；Blackwell 延续同一趋势（FP4、更大 TMA）。

*最后一里路*不是更 clever 的算法，而是 *与硬件 async 模型对齐*——producer-consumer、multi-stage pipeline、低精度 tensor core。Blackwell 延续同一趋势。

== 常见误区

#warn[
  *误区 1*：「FA3 是新的 attention 近似算法。」——错。数学与 FA2 相同，是 *Hopper 实现*。
]

#warn[
  *误区 2*：「在 H100 上 recompile FA2 就等于 FA3。」——错。Triton FA2 用了部分 Hopper 指令仍慢 ~1.5×；缺 warp specialization 与 GEMM-softmax overlap。
]

#warn[
  *误区 3*：「FP8 attention 可以直接替换 FP16 training。」——错。需 block quant + incoherent processing + 全流程 loss 验证；论文主要验证 forward 精度。
]

#warn[
  *误区 4*：「Teaching pipeline kernel 慢是因为没优化。」——错。它故意 scalar + 小数据；结构对齐 FA3，不 chase TFLOPS。
]

#insight[
  读 FA3 论文时把 Algorithm 1（producer-consumer + 串行 softmax）与 Algorithm 2（2-stage overlap）*分开画 timeline*——前者对应 warp specialization 收益，后者对应 SFU/TC 重叠收益。Ablation 表显示两者叠加。
]

== v3 之后：Blackwell 与 FA4 方向

面试常问 "下一步是什么"——简要路线图：

- *Blackwell (SM100)*：FP4/MXFP4 tensor core、更大 TMA throughput、进一步 warp specialization。
- *Split-K / Stream-K* 与 attention 结合——超长 $N$ 时 grid utilization。
- *Paged KV cache* inference 路径与 TMA 的 descriptor 更新（vLLM 集成）。
- *跨 node attention*（DeepSeek MLA 等）——Hopper NVLink + cluster 不够，需算法层压缩 $K/V$。

#note[
  截至 2024 论文，FlashAttention-3 已开源并计划集成 PyTorch。Blackwell 专用 fork 可能在 CUTLASS 3.x 主线——关注 `wgmma` → 下一代 MMA 指令名变更。
]

== 面试考点

#interview[
  *Q1*: Hopper 三大特性（WGMMA / TMA / FP8）分别解决什么？

  A: WGMMA——warpgroup 异步 tensor core GEMM，更大 tile + 与 softmax 重叠；TMA——硬件 async bulk copy，把 load 从 consumer 剥离、减轻 issue 压力；FP8——tensor core 吞吐 2×，配合 block quant 保精度。
]

#interview[
  *Q2*: 为什么 FA3 需要 producer-consumer，FA2 不需要？

  A: FA2 用 `cp.async` 全员参与 load + `mma.sync` 同步 GEMM，load 与 compute 争用 issue slot，且无法低 register 专职 TMA。FA3 的 TMA 只需少数 thread，producer 低 reg / consumer 高 reg，pipeline 更深。
]

#interview[
  *Q3*: GEMM-softmax overlap 的具体机制？为什么 FA2 做不到？

  A: 2-stage register pipeline：async WGMMA 发起 $Q K_{j+1}^T$ 不 wait，同时 softmax 处理已完成的 $S_j$；/pingpong 两 warpgroup 交替 GEMM 与 softmax。FA2 的 `mma.sync` 和 wait 语句强制串行。
]

#interview[
  *Q4*: `mbarrier` 与 `__syncthreads` 在 FA3 pipeline 里各管什么？

  A: `mbarrier` 管 TMA async complete（transaction count + phase flip）和 smem stage 释放；`__syncthreads` 管 CTA 内所有 thread 视图一致。WGMMA 用 `wait_group` 单独 wait。三者不可混用语义。
]

#interview[
  *Q5*: TMA 比 `cp.async` 强在哪？

  A: 独立硬件单元；1 thread 发 multidim bulk copy；descriptor 处理 boundary；与 mbarrier 原生集成；consumer warpgroup 完全不参与地址计算。
]

#interview[
  *Q6*: FP8 attention 的精度陷阱？

  A: Per-tensor scale 对 outlier 失效；$V$ layout 与 WGMMA k-major 不匹配；$P$ 的 acc layout 与 GEMM1 operand layout 冲突；causal + FP8 可能 slower/less accurate。Mitigation：block quant + incoherent processing + FP32 softmax state。
]

#interview[
  *Q7*: FA v1 / v2 / v3 性能数量级？

  A: v1/v2 Ampere ~280–340 TFLOPS on H100（~35% peak）；v3 FP16 ~740 TFLOPS（~75%）；v3 FP8 ~1.2 PFLOPS。v3 vs v2 forward 1.5–2×。
]

#interview[
  *Q8*: WGMMA 与 `mma.sync` 的核心语义差异？

  A: WGMMA 异步（commit/wait_group），warpgroup 128 threads；mma.sync 同步完成，warp 32 threads。WGMMA 支持 FP8 与更大 mma shape。
]

#interview[
  *Q9*: 本书 `flash_attention_v3_pipeline_kernel` 在教什么？

  A: K/V 双 stage smem ping-pong + 预取下一 key——对应 FA3 的 TMA circular buffer 与 release/acquire 协议；scalar 版无 TMA/mbarrier，但 pipeline *形状* 与 Algorithm 2 同构。
]

#interview[
  *Q10*: v3 之后可能方向？

  A: Blackwell FP4、attention+GEMM 更深 fuse、paged KV TMA、MLA/压缩 KV 算法层、跨 GPU sequence parallel。硬件迭代继续放大 asynchrony + low-precision 趋势。
]

#interview[
  *Q11*: Thread Block Cluster 和 DSMEM 是什么？FA3 用了吗？

  A: Cluster = 最多 16 个 CTA 组，DSMEM = 跨 CTA 访问 smem 的能力。FA3 主 forward 不依赖 DSMEM；知道定义用于 Hopper 综合题。
]

#interview[
  *Q12*: 为什么 FP8 WGMMA 要求 k-major？对 $V$ 有什么影响？

  A: Hopper FP8 tensor core 硬件约束；$V$ 默认 head-contiguous 是 mn-major，需 in-kernel transpose 或 GMEM 预转置。FP16 两种 layout 均可，故 FP8 工程更复杂。
]

#interview[
  *Q13*: pingpong warpgroup 和 2-stage register pipeline 区别？

  A: 2-stage 在*同一* warpgroup 内用 async WGMMA 与 softmax 重叠；pingpong 用*两个* warpgroup 交替 GEMM 与 softmax（bar.sync 协调）。可叠加。
]

#interview[
  *Q14*: Standard attention 3–16× 慢于 FA3 的主要原因？

  A: Materialize $S, P$ 到 HBM——$O(N^2)$ 读写；FA 系列 $O(N)$ HBM for activations。FLOPs 相同，IO 不同。
]

== 从 teaching kernel 到 production FA3

若你要在 H100 上追 FA3 性能，recommended 路径：

*Step 1*：读 CUTLASS 3 `examples/57_hopper_gemm` 与 FlashAttention-3 repo 的 `hopper` 目录——先看 TMA descriptor 构造与 WGMMA mainloop 分离。

*Step 2*：用 `flash_attention_v3_pipeline_kernel` 理解 smem ring buffer——把 `__syncthreads` 换成 mbarrier 是第一步 hop。

*Step 3*：profile 标量 teaching kernel 确认 online merge 正确——再换 WGMMA，*不要同时 debug 算法与 Hopper 指令*。

*Step 4*：ncu 对比 FA2 vs FA3 官方 benchmark——看 `pipe_tensor`、`pipe_sfu`、`tma_throughput` 三角是否同时高。

*Step 5*：FP8 最后上——先 FP16 路径 overlap 调满，再加 block quant / transpose / byte_perm。

#note[
  手写完整 FA3 不现实（CUTLASS + 数万行 tuning）。面试目标：*讲清三大创新 + 画 timeline + 报出 740/1200 TFLOPS 数量级 + 知道 FP8 layout 坑*。实现细节指向官方 repo 即可。
]

全书正文在此结束——附录含环境配置与论文链接。祝你在 kernel 面试里，从 vector add 的合并访问一直讲到 Hopper 的 async warpgroup，*一条 ladder 走到底*。
