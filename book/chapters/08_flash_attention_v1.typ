#import "../template.typ": *

= Flash-Attention v1

Flash-Attention（Dao et al., 2022）是 transformer 长序列时代最重要的 kernel 之一。面试里它几乎和 naive attention 成对出现——追问的不是「会不会背公式」，而是 *IO 复杂度、online softmax 正确性、block size 怎么选、为什么 FLOPs 没变却快很多*。

这一章只讲 *v1*：论文 Algorithm 1 的分块 fused forward，把 $S = Q K^T$ *从不完整 materialize 到 HBM*。v2 的 sequence-parallel 与 warp 分工、v3 的 Hopper FP8 路径留到后续章节。

对应源码：`src/cuda/08_flash_attention_v1.cu`。教学规模 $N_q = 4$, $N_k = 16$, $d = 8$；正文推导用 LLM 典型 $N = 1024$, $d = 64$。

本章 optimization ladder：

#ladder(
  ("naive (ch.7)",     "3 pass, materialize $S/P$",           "—"),
  ("v1 register",      "1 thread / query, key tile stream",  "—"),
  ("v1 shared-mem",    "1 block / query, $Q,K,V$ tile smem", "—"),
)

前两个 GPU 版本*数学相同*——都是 online softmax + 分块累加 $O$；差别在 $Q/K/V$ 从 HBM 读几次、是否在 shared memory 复用。性能数字 intentionally 不 chase peak——先把算法讲透。

与前面章节的依赖：第 3 章 online softmax 的 $(m, ell)$ merge；第 4 章 matmul tile 的 $Q K^T$；第 7 章 naive attention 的 $O(N^2)$ 中间矩阵痛点。

== 问题定义

=== 数学（单 head，与第 7 章一致）

给定 $Q in RR^(N_q times d)$, $K, V in RR^(N_k times d)$（self-attention 时 $N_q = N_k = N$），scaled dot-product attention：

$ O = "softmax"(frac(Q K^T, sqrt(d))) V $

令 $S = Q K^T / sqrt(d) in RR^(N_q times N_k)$，对*每个 query 行* $i$ 做 softmax 得 $P[i, :]$, 再 $O[i, :] = P[i, :] V$。

#note[
  源码 `flash_attention_cpu_reference` 为可读性*未乘* $1/sqrt(d)$——与第 7 章 `kScale` 相比差一个常数因子，不影响 online merge 结构。生产 kernel 把 scale 融进 $Q K^T$ dot product（见下文 kernel 节）。
]

=== Naive 的致命伤（复习）

第 7 章三 pass：写 $S$ → 读 $S$ 写 $P$ → 读 $P$ 乘 $V$。单 head FP32 中间矩阵 $N^2 times 4 "B"$；$N = 4096$ 时 64 MB/head，32 heads 即 2 GB——*还没算 backward*。

#insight[
  Flash-Attention 快在 *IO*，不是 FLOPs。Forward FLOPs 仍是 $O(N^2 d)$（两个 matmul），但 *HBM 上 $S/P$ 的 $O(N^2)$ 流量*被消掉。面试第一答案永远是 memory / bandwidth，第二才是 fusion 与 SRAM 复用。
]

== 核心思想：IO-aware 分块融合

=== 观察：$S$ 是带宽黑洞，不是算力黑洞

Naive attention 的瓶颈分解：

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*步骤*], [*HBM 主导项*], [*能否省*],
    [$S = Q K^T$], [写 $N^2$ + 读 $Q,K$], [不在 HBM 存 $S$],
    [softmax], [读+写 $N^2$], [online merge，tile 内完成],
    [$O = P V$], [读 $N^2$ 的 $P$], [与 softmax 同步累加分子],
  ),
  caption: [*Table:* naive attention 三 pass 的 HBM 瓶颈分解。列 *HBM 主导项* 给出每步与序列长度 $N$ 相关的 square 流量量级；*能否省* 说明 Flash-Attention 的 fusion 策略——$S$ 和 $P$ 均可在 on-chip SRAM 内完成、不写 global。],
  kind: table,
)

*Observation*：三行中 $S$、softmax、$P V$ 的 HBM 项全是 $O(N^2)$，而 $Q,K,V$ 仅 $O(N d)$——当 $N >> d$ 时 square 项碾压 linear 项。FA v1 的核心不是减少 matmul FLOPs，而是让 square 矩阵*从不触 HBM*。

$Q, K, V$ 各 $O(N d)$；$S$ 和 $P$ 各 $O(N^2)$——当 $N >> d$ 时，*square 项碾压*。

=== Flash-Attention v1 做了什么

把 attention 拆成 *sequence 维上的 tile*，在 on-chip SRAM（shared memory / register）里完成：

1. 算 $S_"tile" = Q_i K_j^T$（small tile，不写 global）。
2. 对 tile 做局部 softmax 统计 $(m_j, ell_j)$，用 online 公式 merge 到 running $(m, ell)$。
3. *同时*用局部权重累加 $O$ 的分子——max 变化时对旧 $O$ *重标定*。
4. 处理完所有 $K/V$ tile 后，$O_i = "accum" / ell$。

#insight[
  *Key insight.* $S$ 的每个元素在 naive 路径里至少经历「写 HBM → 读 HBM → exp → 写 HBM → 再读」——4+ 次 square 流量。FA v1 让 $S_"tile"$ 生在 smem/reg，用完即弃，*square 项从不触 HBM*。
]

=== 与第 3 章 online softmax 的对应

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    [ *第 3 章* ], [ *Flash-Attention* ],
    [一行 logits $x_1,...,x_N$], [query $i$ 对所有 key 的 score 向量],
    [running $(m, d)$], [running $(m_i, ell_i)$ per query row],
    [chunk merge], [每个 $K_j$ tile 的局部统计 merge],
    [输出 $y_i = exp(x_i-m)/d$], [$O_i = ("weighted sum of" V) / ell_i$],
  ),
  caption: [*Table:* 第 3 章 online softmax 与 Flash-Attention 的符号对照。左列为标量 softmax 概念，右列为 attention 中 per-query-row 的向量推广；merge 公式结构相同，FA 额外维护输出分子 $O_i$。],
  kind: table,
)

*Observation*：$(m, ell)$ 的 merge 公式与第 3 章 chunk 结合律一一对应；FA 多出来的结构是分子 $O_i = sum_j P[i,j] V[j,:]$——max 升高时旧分子必须乘 $exp(m_"old" - m_"new")$，这是 v1 相对 plain online softmax 的额外推导。

第 3 章只维护 softmax *分母*；FA 还要维护 *分子* $O_i = sum_j P[i,j] V[j,:]$——max 升高时，旧分子必须乘 $exp(m_"old" - m_"new")$。这是 v1 比「单纯 online softmax」多出来的推导。

== IO 复杂度分析

=== 标准 attention 的 HBM 访问

论文分析（单 head，忽略 cache 复用）forward HBM access 量级：

$ "HBM"_"naive" = Theta(N d + N^2) approx Theta(N^2) quad (N >> d) $

更细地按 *元素流量* 计（与第 7 章手算一致）：$S/P$ 相关读写主导 $approx 4 N^2$ floats（写 $S$、读 $S$ 写 $P$、读 $P$），加上 $Q,K,V$ 的 $O(N d)$ 项。合并成面试常用表述：

$ "HBM"_"naive" = O(N^2 d) quad "(matmul 复读 Q,K 与 square 矩阵多次往返)" $

=== Flash-Attention v1 的 HBM 访问

设 on-chip SRAM 容量为 $M$ *字节*，block tile 大小 $B_r times B_c$（query × key），head dim $d$。论文 Theorem 1：

$ "HBM"_"FA" = O(N^2 d^2 / M) $

直觉：每个 SRAM tile 能装 $Theta(M/d)$ 量级的 $Q/K/V/S$ 数据；要完成 $N times N$ 的 attention map 语义，需 $Theta(N^2 / (M/d)) = Theta(N^2 d / M)$ 次 tile 换入——再乘每 tile 与 $d$ 相关的 matmul 因子，得 $O(N^2 d^2 / M)$。

#note[
  常数依赖 loop order（外 $K/V$ 内 $Q$ 时每个 $K_j$ 只读一次 HBM）。面试写 Big-O 即可；手算用下面具体数字建立量级感。
]

=== 手算：$N = 1024$, $d = 64$, $M = 100 "KB"$（A100 单 block smem 预算）

取 A100 单 block 可用 shared memory $approx 100 "KB"$（48 KB 默认 × 动态扩到 164 KB 的上限内，留余量给寄存器 spill 与 padding）。

*Naive*（$S/P$ 主导，按 $O(N^2 d)$ 量级估 HBM 元素访问）：

$ N^2 d = 1024^2 times 64 = 67108864 quad "element-access 量级" $

换算流量：$67 times 10^6 times 4 "B" approx 268 "MB"$（单 head、单次 forward 与 square 矩阵相关的同阶量）。

*Flash-Attention v1*：

$ frac(N^2 d^2, M) = frac(1024^2 times 64^2, 100 times 1024) = frac(4294967296, 102400) approx 41943 $

$41943 times 4 "B" approx 168 "KB"$ *同阶 HBM 流量*。

*比值*：

$ frac(67108864, 41943) approx 1600 times $

——不是 FLOPs 差 1600 倍（FLOPs 同为 $Theta(N^2 d)$），而是 *square 矩阵相关的 HBM 往返*差三个数量级。生产 FlashAttention（Tri Dao 的 `flash-attn` 库，$N >= 1024$、tensor core GEMM）相对 naive 的端到端加速通常是 tens of ×——*本章 teaching kernel 规模 $N_q=4$ 测不到这个量级*，见下文「实测」节。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*方法*], [*HBM 量级（单 head）*], [$N=1024,d=64$ 估算],
    [Naive], [$O(N^2 d)$], [$approx 268 "MB"$ 同阶],
    [FA v1], [$O(N^2 d^2 / M)$], [$approx 168 "KB"$ 同阶],
    [比值], [$—$], [$approx 1600 times$],
  ),
  caption: [*Table:* naive vs FA v1 的 HBM 流量手算对比（$N=1024$, $d=64$, on-chip SRAM 预算 $M=100 "KB"$）。第三列为 element-access 量级换算成字节后的估算值；FLOPs 两方法同为 $Theta(N^2 d)$，差异仅在 square 矩阵相关的 HBM 往返。],
  kind: table,
)

*Observation*：比值 $approx 1600 times$ 来自 square 项被 SRAM tiling 消掉——不是 FLOPs 差 1600 倍。面试答「FA 快在哪」应先说 HBM $O(N^2 d) arrow.r O(N^2 d^2/M)$，再提 fusion 与 $K/V$ tile 复用。

#warn[
  手算时 *FLOPs 与 HBM 分开写*。面试官给 $N=8192$ 先算 $N^2$ 是否 OOM，再算 $N^2 d^2/M$ 是否可接受——把 FA 说成「减少计算量」会直接扣分。
]

=== Roofline 视角

FA 不改变 attention 的 FLOPs ridge 位置，但把 *有效 AI*（FLOPs / 实际 HBM byte）抬高——同样 $O(N^2 d)$ 次 FMA，实际从 HBM 读写的 byte 从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$。长序列下 kernel 从「memory-bound on square tensor」变成「compute-bound on matmul tile + exp」——这才使得 tensor core / 更大 tile 有意义（v2 方向）。

== Algorithm 1：论文伪代码详解

FlashAttention 论文 Forward 核心（符号与论文一致：$N$ 为序列长，$B_r, B_c$ 为 block 大小，$T_r = ceil(N/B_r)$, $T_c = ceil(N/B_c)$）：

```
// Input: Q, K, V ∈ R^{N×d} in HBM
// Output: O ∈ R^{N×d} in HBM
// On-chip: SRAM of size M

Partition Q into Tr blocks Q1..QTr of size Br×d each
Partition K, V into Tc blocks K1..Kc, V1..Vc of size Bc×d each

for j = 1 to Tc do
    Load Kj, Vj from HBM to on-chip SRAM                    // 外循环：K/V block

    for i = 1 to Tr do
        Load Qi from HBM to on-chip SRAM                      // 内循环：Q block

        On chip:
            Sij = Qi Kj^T ∈ R^{Br×Bc}                         // tile matmul
            mij = rowmax(Sij)                                   // 每 query 行一个 max
            Pij = exp(Sij - mij)                                // 局部 exp（未归一化）
            lij = rowsum(Pij)                                   // 局部 sum

            // Online softmax merge + output rescale
            mi_new = rowmax(mi, mij)                            // 逐行
            li_new = exp(mi - mi_new) ⊙ li + exp(mij - mi_new) ⊙ lij
            Oi = diag(exp(mi - mi_new)) Oi + Pij Vj             // 重标定 O 再累加

        Write Oi to HBM (或最后统一写)
    end for
end for

Finalize：$O_i = O_i / li$                                          // 逐行除 running sum
```

=== Algorithm 1 与源码变量对照

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*论文符号*], [*源码（register kernel）*], [*含义*],
    [$Q_i$], [`q[row * head_dim + d]`], [单 query 行，一行一 thread],
    [$K_j, V_j$], [`k[key * head_dim + d]`], [key tile，`key_start` 步进],
    [$S_"ij"$], [`scores[t]`], [tile 内局部 logits，寄存器数组],
    [$m_i$], [`running_max`], [该行 running max],
    [$ell_i$], [`running_sum`], [running sum of exp],
    [$O_i$], [`accum[d]`], [未归一化输出分子],
    [$B_c$], [`kTileKeys` (=4)], [key 方向 tile 宽],
    [$B_r$], [1（每 thread 一行）], [query 方向 block 高——生产为 $B_r > 1$],
  ),
  caption: [*Table:* 论文 Algorithm 1 符号与 `flash_attention_v1_kernel` 源码变量对照。$B_r=1$、$B_c=4$ 是教学最小配置；生产 FA 用 $B_r, B_c approx 64–128$ 并在 warp 级做 tile matmul。],
  kind: table,
)

*Observation*：`scores[t]` 活在寄存器、用完即弃——对应论文「$S_"tile"$ 不写 HBM」。读源码时按 state machine 看：每个 query 行只有 $(m, ell, "accum")$ 三个状态，每个 key tile 是一次状态转移。

论文外层 $T_c = ceil(N / B_c)$ 次迭代对应源码 `for (key_start = 0; ...; key_start += kTileKeys)`。内层 $T_r$ 在 register 版由 *grid 上的 row index* 并行展开——每个 block 的 32 个 thread 各处理不同 query 行，等价于 Algorithm 1 内层 $i$ 的并行化。

=== 数据流（单 query 行）

```text
HBM: Q[row,:]  ──读一次──► 寄存器 q[row*head_dim+d]
                              │
         ┌────────────────────┘
         ▼
for each key tile [key_start, key_start+Bc):
    HBM: K[key,:], V[key,:]  ──读──► scores[t], tile_max
                              │
                              ▼
                    new_max, old_scale = exp(m_old - m_new)
                    accum *= old_scale          ← O rescale
                    weight = exp(score - new_max)
                    accum += weight * V[key,:]  ← P@V 融合
                    running_sum = running_sum * old_scale + tile_sum
                              │
         └────────────────────┘
         ▼
HBM: out[row,:] = accum / running_sum
```

*全程无* `cudaMalloc(N×N)` 的 `scores` buffer——这是与第 7 章 `device_scores` 的本质分界。

#insight[
  读源码时按 *state machine* 看：每个 query 行只有 $(m, ell, "accum")$ 三个状态；每个 key tile 是一次 *状态转移*。Debug 时打印每个 tile 后的 triple，与 CPU reference 逐步对比，比端到端 diff 更容易定位 merge bug。
]


*外循环 $j$*：每次把一个 key/value *列块*（长度 $B_c$）搬进 SRAM。$K_j, V_j$ 在 inner 所有 $Q_i$ 上*复用*——这是 IO 最优 loop order：每个 $K/V$ tile 只从 HBM 读 *一次*（对 fixed $j$）。

*内循环 $i$*：对每个 query 块 $Q_i$，与当前 $K_j$ 算 $B_r times B_c$ 的 $S_"tile"$，更新该 query 块内每行的 $(m, ell, O)$。

#insight[
  Loop order 反过来（外 $Q$ 内 $K$）则每个 $Q_i$ 只读一次，但 $K_j$ 会被重复读 $T_r$ 次——总 HBM 仍同阶，但常数更大。v1 论文与 cuDNN/FA2 均采用 *外 K/V、内 Q* 或等价的 swap-GEMM 布局。
]

=== 每次处理 $Q_i$ 时更新的状态

对 query block $i$ 内的*每一行*（query token）维护：

- $m_i in RR$：running max of scores seen so far。
- $ell_i in RR$：running sum $sum exp(s - m_i)$（relative to current $m_i$）。
- $O_i in RR^d$：running numerator $sum exp(s - m_i) V$（*尚未*除以 $ell_i$）。

每来一个 $K_j$ tile，用 tile 内统计 $(m_(i j), ell_(i j))$ merge 进 $(m_i, ell_i, O_i)$。全部 $j$ 处理完后 $O_i / ell_i$ 即为最终 attention 输出。

== Online Softmax Rescaling：$m$, $ell$, $O$ 推导

=== 符号

固定 query 行 $i$，已处理 key 集合 $cal(K)_"old"$，来了新 key 集合 $cal(K)_"new"$（一个 tile）。

- 旧状态：$(m, ell)$，其中 $ell = sum_(k in cal(K)_"old") exp(S[i,k] - m)$。
- 新 tile logits：$x_k = S[i,k]$ for $k in cal(K)_"new"$。
- 局部：$m' = max_(k in cal(K)_"new") x_k$，$ell' = sum_(k in cal(K)_"new") exp(x_k - m')$。

全局新 max：$m_"new" = max(m, m')$。

=== $ell$ 的 merge（第 3 章公式的 block 版）

旧 sum 相对 $m_"new"$ 需重标定：

$ ell_"new" = underbrace(sum_(k in cal(K)_"old") exp(x_k - m_"new"))_("旧 keys") + underbrace(sum_(k in cal(K)_"new") exp(x_k - m_"new"))_("新 tile") $

第一项：$ell dot exp(m - m_"new")$（与第 3 章 $d' = d dot exp(m - m') + exp(x - m')$ 相同）。

第二项：$ell' dot exp(m' - m_"new")$（tile 内先减 $m'$ 算 exp，再统一基准到 $m_"new"$）。

合并：

$ ell_"new" = ell dot exp(m - m_"new") + ell' dot exp(m' - m_"new") $

#note[
  源码里 `tile_max` 对应 $m'$，`new_max` 对应 $m_"new"$，`tile_sum` 对应 $ell'$（relative to $m_"new"$ 时已用 `exp(scores[t] - new_max)`）。
]

=== $O$ 的 merge（Flash-Attention 特有）

目标：$O = sum_k P[i,k] V[k,:]$，$P[i,k] = exp(S[i,k]-m_"final") / ell_"final"$。

维护未归一化分子 $tilde(O) = sum_(k in "processed") exp(S[i,k] - m) V[k,:]$（relative to *current* $m$）。

当 $m arrow.r m_"new"$ 时，旧贡献统一缩放：

$ tilde(O)_"new" = exp(m - m_"new") dot tilde(O)_"old" + sum_(k in "new tile") exp(x_k - m_"new") V[k,:] $

与 $ell$ 同步：先 `accum *= exp(m - m_new)`，再加 `weight * V`。

Finalize：$O = tilde(O) / ell$。

=== 正确性：等价于全局 softmax

设完整 logits $x_1,...,x_N$，全局 $m^* = max_j x_j$，$ell^* = sum_j exp(x_j - m^*)$。

归纳：处理完任意 prefix 后，$(m, ell, tilde(O))$ 满足

$ tilde(O) = sum_(j in "prefix") exp(x_j - m) V[j,:], quad ell = sum_(j in "prefix") exp(x_j - m) $

且 $m = max_(j in "prefix") x_j$。merge 公式即第 3 章 *chunk 结合律* 对 $(m, ell)$ 的推广；$O$ 的缩放因子与 $ell$ 相同，保证

$ frac(tilde(O), ell) = frac(sum_j exp(x_j - m^*) V[j,:], ell^*) $

与 naive 三 pass 一致。

=== 数值走查

Query 行 $i$，$d=1$，$V[k]=1$（标量）。Scores 分两 tile：$[1, 2]$ 与 $[3]$。

*Tile 1*：$m'=2$, $ell' = e^(-1)+1$, $tilde(O) = 1 dot e^(-1) + 2 dot 1 = e^(-1)+2$。设 $m=2, ell = e^(-1)+1, tilde(O)=e^(-1)+2$。

*Tile 2*：$x=3$, $m_"new"=3$, scale $= exp(2-3)=e^(-1)$。

$ ell_"new" = (e^(-1)+1) dot e^(-1) + 1 = e^(-2)+e^(-1)+1 $
$ tilde(O)_"new" = (e^(-1)+2) dot e^(-1) + 3 = e^(-2)+2e^(-1)+3 $

全局 softmax 权重 $approx [0.09, 0.24, 0.67]$，$O = 0.09+0.48+2.01 = 2.58$。用 $tilde(O)_"new"/ell_"new"$ 算得相同——与 `flash_attention_cpu_reference` 一致。

#insight[
  面试手推：先写 $ell$ merge（第 3 章），再写 $O$ 乘同一个 $exp(m - m_"new")$。漏掉 $O$ rescale 是 FA 实现最常见 bug。
]

== Block Size 选择：$B_r times B_c$

=== Shared memory 预算

单 block 典型 resident 数据（FP32）：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*Buffer*], [*大小*], [*说明*],
    [$Q_i$ tile], [$B_r times d times 4$ B], [query 块],
    [$K_j$ tile], [$B_c times d times 4$ B], [key 块],
    [$V_j$ tile], [$B_c times d times 4$ B], [value 块],
    [$S_"ij"$ tile], [$B_r times B_c times 4$ B], [score 矩阵],
  ),
  caption: [*Table:* 单 CTA 典型 shared memory 驻留 buffer（FP32，4 B/float）。大小公式 $approx 4(B_r + B_c)d + B_r B_c$ floats；$B_r B_c$ 项是 square——$B_r, B_c$ 各翻倍 smem 近似四倍。A100 单 block 上限 164 KB。],
  kind: table,
)

*Observation*：$d=64$, $B_r=B_c=64$ 时合计 $approx 65 "KB"$，fit $M=100 "KB"$ 预算；但 $B_r=B_c=128$, $d=128$ 可超 164 KB 直接 launch 失败——block size 选型首要约束是 smem，不是 FLOPs。

合计 $approx 4(B_r + B_c)d + B_r B_c$ floats。$d=64$, $B_r=B_c=64$：$3 times 64 times 64 times 4 + 64^2 times 4 approx 49 "KB" + 16 "KB" = 65 "KB"$——fit $M=100 "KB"$。

#warn[
  $B_r B_c$ 项是 square——$B_r, B_c$ 各翻倍，smem *近似四倍*。$B_r=B_c=128$, $d=128$ 时单 block 可超 164 KB *直接 launch 失败*。
]

=== 寄存器压力

Teaching kernel `flash_attention_v1_kernel` 每 thread 持有 `accum[kHeadDim]` + `scores[kTileKeys]`——$d$ 或 tile 过大时 *register spill* → local memory → 性能雪崩。生产 FA 用 warp 分工算 $S_"tile"$，每 thread 只持 partial row。

=== Warp 数与 occupancy

`flash_attention_v1_shared_kernel` 用 8 threads/block（教学）——真实 kernel 需 $>= 1$ warp（32 threads）避免 underutilization。FA2 典型 128–256 threads/block，4–8 warps，在 smem 与 occupancy 间 trade-off：

- smem 大 → max active blocks/SM 降 → latency 难 hide。
- smem 小 → tile 小 → HBM 读次数增。

#note[
  选型流程：先算 smem 公式 → 查 occupancy calculator → profile 几个 $(B_r, B_c)$ 候选。面试答「$B_r, B_c approx sqrt(M/d)$ 量级」即可，不必背具体 magic number。
]

=== $B_r$ vs $B_c$ 不对称

Causal mask 下 query row $i$ 只 attend $j <= i$——许多 $(i,j)$ tile 全 mask，可 *skip entire block*（见下节）。常取 $B_c$ 略大以提高 $K/V$ 复用；$B_r$ 对齐 warp 行数以利 $S_"tile"$ 行并行。

== v1: Register Streaming Kernel

源码 `flash_attention_v1_kernel`：*一个 thread 负责一个 query 行*，沿 key 维以 `kTileKeys` 为步长流式处理。

```cpp
__global__ void flash_attention_v1_kernel(
    const float* q, const float* k, const float* v,
    float* out, int query_count, int key_count, int head_dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= query_count) return;

  float running_max = -1.0e30f;
  float running_sum = 0.0f;
  float accum[kHeadDim];
  // ... init accum ...

  float scores[kTileKeys];

  for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
    // 1. 算 tile 内 scores + tile_max
    // 2. new_max, old_scale = exp(running_max - new_max)
    // 3. accum *= old_scale
    // 4. tile_sum, accum += weight * V
    // 5. running_sum = running_sum * old_scale + tile_sum
    // 6. running_max = new_max
  }

  out[row * head_dim + d] = accum[d] / (running_sum + kEps);
}
```

=== 逐步对应算法

*Step 1 — $S$ 计算（tile matmul 的极简版）*

```cpp
for (int t = 0; t < tile_count; ++t) {
  const int key = key_start + t;
  const float score = dot_row_device(
      &q[row * head_dim], &k[key * head_dim], head_dim);
  scores[t] = score;
  tile_max = fmaxf(tile_max, score);
}
```

每个 thread 独立 dot product——无 smem 复用，*$K$ 每 tile 从 HBM 重读*。教学清晰；生产应用 tiled GEMM。

*Step 2 — Softmax rescale*

```cpp
const float new_max = fmaxf(running_max, tile_max);
const float old_scale = (running_sum == 0.0f) ? 0.0f
                        : expf(running_max - new_max);
for (int d = 0; d < kHeadDim; ++d) {
  accum[d] *= old_scale;
}
```

`running_sum == 0` 分支：首 tile 时 `old_scale=0`，避免 `exp(-inf)` 噪声——与「尚无 prefix」语义一致。

*Step 3 — $P @ V$ 累加（$P$ 不 materialize）*

```cpp
const float weight = expf(scores[t] - new_max);
tile_sum += weight;
for (int d = 0; d < kHeadDim; ++d) {
  accum[d] += weight * v[key * head_dim + d];
}
running_sum = running_sum * old_scale + tile_sum;
running_max = new_max;
```

`weight` 即 $P[i,k]$ 的未归一化分子 relative to $m_"new"$——*从不写 global $P$*。

Launch：`<<<ceil(query_count / 32), 32>>>`。并行轴 = query 行数 $N_q$；$N_q$ 小则 SM 填不满——这是 v1 局限。

=== 完整 inner loop（对照源码）

```cpp
for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
  const int tile_count = min(kTileKeys, key_count - key_start);

  float tile_max = -1.0e30f;
  for (int t = 0; t < tile_count; ++t) {
    const int key = key_start + t;
    scores[t] = dot_row_device(&q[row * head_dim],
                               &k[key * head_dim], head_dim);
    tile_max = fmaxf(tile_max, scores[t]);
  }

  const float new_max = fmaxf(running_max, tile_max);
  const float old_scale = (running_sum == 0.0f) ? 0.0f
                        : expf(running_max - new_max);

  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] *= old_scale;
  }

  float tile_sum = 0.0f;
  for (int t = 0; t < tile_count; ++t) {
    const float weight = expf(scores[t] - new_max);
    tile_sum += weight;
    const int key = key_start + t;
    for (int d = 0; d < kHeadDim; ++d) {
      accum[d] += weight * v[key * head_dim + d];
    }
  }

  running_sum = running_sum * old_scale + tile_sum;
  running_max = new_max;
}
```

*读代码 checklist*：

1. `scores[t]` 在 merge 前算完——mask（若启用）应插在 `scores[t] = ...` 之后、`tile_max` 更新之前。
2. `old_scale` 同时缩放 `accum` 和 `running_sum`——*必须一致*，否则 $O$ 与 $ell$ 基准错位。
3. `weight` 用 `new_max` 而非 `running_max`——与第 3 章 online 公式中 $exp(x - m_"new")$ 一致。
4. `kEps` 防除零：全 mask 行时 $ell = 0$，生产应 skip 该行而非依赖 epsilon。

=== Scale $1/sqrt(d)$ 应加在哪

教学源码省略 scale；接入第 7 章约定只需一行：

```cpp
const float score = dot_row_device(...) * inv_sqrt_head_dim;
```

或在 `dot_row_device` 内乘。数学上 scale 在 softmax 前等价；IO 上 fuse 进 dot 避免额外 global pass。cuBLAS / CUTLASS 路径用 epilogue alpha；FA 手写 kernel 在 score 写回 register 时乘最自然。

#note[
  FP16/BF16 训练时 scale 常与 layer norm 输出尺度耦合——kernel 接口常暴露 `softmax_scale` 指针（含 $1/sqrt(d)$ 与可选 temperature），便于 inference KV cache 场景调 logits 温度而不重编译 kernel。
]


`flash_attention_v1_shared_kernel`：*一个 block 处理一个 query 行*，显式把 $Q$, $K$, $V$ tile 搬进 shared memory。

```cpp
__global__ void flash_attention_v1_shared_kernel(...) {
  const int row = blockIdx.x;
  __shared__ float q_shared[kHeadDim];
  __shared__ float k_tile[kTileKeys][kHeadDim];
  __shared__ float v_tile[kTileKeys][kHeadDim];
  __shared__ float scores[kTileKeys];
  // ...
}
```

=== Q 加载

```cpp
if (threadIdx.x < kHeadDim) {
  q_shared[threadIdx.x] = q[row * head_dim + threadIdx.x];
}
__syncthreads();
```

一行 $Q[i,:]$ 读入 smem 一次，*所有 key tile 复用*——对应 Algorithm 1 的 `Load Qi` 后在内层 $j$ 循环复用（此处 outer 是 key tile，$Q$ 已 resident）。

=== K / V 加载

```cpp
const int token = threadIdx.x;
const int key = key_start + token;
if (token < kTileKeys && key < key_count) {
  for (int d = 0; d < kHeadDim; ++d) {
    k_tile[token][d] = k[key * head_dim + d];
    v_tile[token][d] = v[key * head_dim + d];
  }
}
__syncthreads();
```

8 threads 协作搬 $4 times 8 = 32$ 维——教学用 `kSharedThreadsPerBlock=8`；生产每 thread 搬更多元素或用 `cp.async`。

=== $S$ 计算、softmax、$O$ 更新

Thread 0 串行执行 tile 内 matmul + online merge（与 register 版数学相同），但 $K,V$ 从 `k_tile`/`v_tile` 读——*illustrate SRAM reuse*。

Launch：`<<<query_count, kSharedThreadsPerBlock>>>` = `<<<4, 8>>>`。

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*版本*], [*优点*], [*缺点*],
    [`flash_attention_v1_kernel`], [代码短；多 query 行并行], [$K,V$ 每 tile 每 thread 重读 HBM],
    [`flash_attention_v1_shared_kernel`], [$Q$ 与 $K,V$ tile 在 smem 复用], [单 block 单 row；thread 0 串行算],
  ),
  caption: [*Table:* 本章两个 teaching kernel 的 launch 与 IO 权衡。register 版 `<<<ceil(N_q/32), 32>>>`；shared 版 `<<<N_q, 8>>>`。数学相同，差别在 $Q/K/V$ 从 HBM 读几次、是否在 smem 复用。],
  kind: table,
)

*Observation*：shared 版用 smem 换 HBM 重读——实测约 1.9× 快于 register 版，但 thread 0 串行 merge 拉低 lane 利用率。生产 FA = staged kernel + warp 级 tile matmul + 更大 $B_r, B_c$，二者是 Algorithm 1 的最小正确子集。

#insight[
  生产 FlashAttention = staged kernel + warp 级 tile matmul + 外 $K/V$ 内 $Q$ 双重循环 + 更大 $B_r, B_c$。本章两个 kernel 是 Algorithm 1 的 *最小正确子集*。
]

=== Shared memory 布局与 bank conflict

`k_tile[token][d]` 按 *token 主序*：固定 `token` 时相邻 `d` 连续——thread `token` 写一行 $K[j,:]$，与 row-major $K$ global layout 一致。若改成 `k_tile[d][token]`，thread 协作 load 时 stride 为 `kTileKeys`，易 bank conflict。

`kHeadDim=8`, `kTileKeys=4` 时 smem 用量：

- `q_shared`: 8 × 4 = 32 B
- `k_tile + v_tile`: 2 × 4 × 8 × 4 = 256 B
- `scores`: 16 B

合计 $< 1$ KB——远低于 A100 48 KB 默认限额。换成 $d=128, B_c=64$ 时仅 $K/V$ tile 就 $2 times 64 times 128 times 4 = 64$ KB，*必须* 与 $S_"tile"$ 复用 buffer 或 double-buffer 流水线——那是 v2 工程细节。

=== 生产 kernel 还差什么

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [*组件*], [*本章 teaching vs 生产 FA*],
    [$Q K^T$ tile], [单 thread dot → warp mma.sync / WGMMA],
    [并行轴], [row-parallel → $(B_r, B_c)$ block 2D tile],
    [$Q$ 复用], [shared 版仅单行 → 多行 $B_r$ 共用 $K_j$],
    [GQA/MQA], [未涉及 → $K,V$ head 数 $H_"kv" < H$],
    [Dropout], [未涉及 → forward 存 random sign / backward 重放],
    [Split-KV], [未涉及 → 超长 $N$ 时 sequence parallel],
  ),
  caption: [*Table:* 本章 teaching kernel 与生产 FlashAttention 的功能差距清单。算法核心（online merge 循环）已覆盖；性能差距在 tile GEMM（tensor core）、occupancy 调参、GQA/dropout 等工程特性。],
  kind: table,
)

*Observation*：面试说「读过 FA 源码」时，*online merge 循环*是算法核心，*性能*在 tile GEMM 与 occupancy 调参——二者不可混为一谈。本章刻意 scalar dot 以便读 merge 循环，不是 chase peak 的实现。

面试说「我读过 FA 源码」时，至少应能指出：*online merge 循环*是算法核心；*性能*在 tile GEMM 与 occupancy 调参——二者不可混为一谈。

== Multi-Head 与 Batch 扩展

单 head kernel 推广到 $(B, H)$ 只需在 index 里加 batch/head offset，*算法不变*：

```cpp
__global__ void flash_attention_mha_kernel(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int seq_len, int head_dim) {
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size || h >= num_heads || row >= seq_len) return;

  const int head_elems = seq_len * head_dim;
  const int batch_stride = num_heads * head_elems;
  const int head_stride = head_elems;
  const float* q_head = q + b * batch_stride + h * head_stride;
  const float* k_head = k + b * batch_stride + h * head_stride;
  const float* v_head = v + b * batch_stride + h * head_stride;
  float* out_head = out + b * batch_stride + h * head_stride;

  // ... 与单 head 相同的 online loop，指针换 q_head/k_head/v_head ...
}
```

Launch：`dim3 grid(ceil(seq_len/32), num_heads, batch_size)`。Head 间零同步——与第 7 章 multi-head naive 相同，但 *不再有 $B times H times N^2$ 的 $S$ tensor*。

#warn[
  Layout 必须确认：PyTorch $(B, N, H, d)$ vs 本书 $(B, H, N, d)$ offset 不同。面试写 kernel 前 *先问 layout*——错 stride 比错 online 公式更常见。
]

Cross-attention：$N_q != N_k$ 时 loop 上界改为 `key_count`，$Q$ 来自 decoder、$K/V$ 来自 encoder——merge 公式不变，causal 通常关闭。

== 常见实现错误

*错误 1*：只 rescale $ell$，忘记 `accum *= old_scale`。输出偏大/偏小，max abs error 随序列长度累积。

*错误 2*：用 `running_max` 算 `weight` 而非 `new_max`。当 tile 引入更大 max 时，新 tile 权重基准错误。

*错误 3*：mask 在 softmax 之后 zero。分母含 masked 项，有效权重和 $< 1$。

*错误 4*：causal skip 条件写成 `key_start > row` 而非 `key_start > row` 且考虑 block 边界——partial tile 仍需 per-element mask。

*错误 5*：`running_sum` 用 float 累加长序列 $N > 8192$ 且 logits 极端时丢精度——应学第 3 章用 `double` 或 block-wide Kahan。

*错误 6*：以为 FA 不需要 subtract-max。Online merge *就是* subtract-max 的分块形式；去掉 merge 重标定等价于数值爆炸。

#insight[
  单元测试策略：固定 $N_q=1$ 的小 $N_k$（如 3、5、7），手算 $(m, ell, O)$ 逐步对比 GPU 每个 tile 后的状态——比直接 `assert_allclose` 整表更容易教新人。
]


=== 原则：在 $S$ 算完、softmax 前应用

Mask 改变 logits 支持集，必须在 $m$, $ell$ 更新*之前*写入 $S[i,j]$：

- *Padding / 无效 key*：$S[i,j] = -oo$（或足够小的 `-1e30`）。
- *Causal*：对 query $i$，key $j > i$ 置 $-oo$。

被 mask 的项 $exp(-oo)=0$，不贡献 $ell$ 与 $O$——与第 3 章 `softmax_masked_kernel` skip 等价。

=== Causal：跳过整个 block

若 causal 且 key block $j$ 满足 $j_"min" > i_"max"$（整个 $B_c times B_c$ tile 全在 future）——*无需 Load $K_j$, $V_j$、无需算 matmul*。

```cpp
if (causal && key_start > row) {
  continue;  // 整个 tile 对 query row 不可见
}
```

Partial tile（block 跨越对角线）：只对 $j > i$ 的元素 mask，或只对 $"col" <= i$ 算 dot product。

#warn[
  Causal 优化漏判 partial tile 会错；全 skip 条件必须是「tile 内所有 key index $>$ 当前 query index」。Decoder 长序列下 skip 贡献显著常数因子。
]

=== 与 naive 第 7 章对比

Naive 在 scores kernel 写 `-1e30` 再整表 softmax——仍占 $N^2$ HBM。FA 在 tile 内 branch，*masked 位置甚至不进 smem 累加*，square 流量为 0。

== Backward 简述

Forward 不存 $S/P$，backward 需要 `dS`——FA *recompute*：反向再按 tile 算 $S = Q K^T$，用 forward 存的 log-sum-exp 统计（或 $m, ell$）求 `dP`，再链式传 `dQ`, `dK`, `dV`。

#insight[
  Recomputation trade-off：*多算 FLOPs，少占 HBM*。训练时 activation checkpoint 同一哲学——FA backward 是「强制 checkpoint attention map」。
]

Backward 也可 fused 成单 kernel（论文 Algorithm 4）：与前向相同的 tile 遍历，在 smem 重算 $S_"tile"$，即时累加 `dQ`, `dK`, `dV`，*不写 `dS` 到 global*。数值上需保存 forward 的 $(m, ell)$ per row（$O(N)$ 存储，非 $O(N^2)$）。

面试答「backward 为什么能 fuse」：*recompute + 同 tile 遍历顺序 + online softmax 反向公式*；显存从 $O(N^2)$ 降到 $O(N)$。

=== Backward 链式法则（概要）

设 forward 已得 $O$, 保存 per-row $(m, ell)$。Backward 接收 $d O$：

1. *重算* $S_"tile" = Q K^T$（同 forward tile 边界）。
2. 由 $O = tilde(O) / ell$ 得 $d tilde(O) = d O / ell$（逐行）。
3. Softmax backward：$d S[i,j] = P[i,j] (d tilde(O)[i,:] dot V[j,:] - sum_k P[i,k] d tilde(O)[i,:] dot V[k,:])$——可在 tile 内流式算，不写 full $d S$。
4. $d Q = d S K$, $d K = d S^T Q$, $d V = P^T d tilde(O)$——仍是 tile GEMM，累加到 global $d Q, d K, d V$。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*Tensor*], [*Naive backward 存储*], [*FA backward*],
    [$P$ 或 $S$], [$O(N^2)$ checkpoint], [recompute per tile],
    [$(m, ell)$], [可选], [$O(N)$ 必须存],
    [$d S$], [$O(N^2)$ materialize], [tile 即时消费，不写出],
  ),
  caption: [*Table:* naive vs FA backward 的 activation 存储对比。FA 用 recompute 换 HBM：反向再算 $S = Q K^T$，只存 per-row $(m, ell)$（$O(N)$），`dS` 在 tile 内即时消费。],
  kind: table,
)

*Observation*：recompute 的 FLOPs 约为 forward 的 1.5–2×，但 HBM 省下的 $N^2$ 读写在长序列下永远划算——与 activation checkpointing 的 trade-off 曲线同形。面试追问「存什么 activation」：典型是 $(m, ell)$ 或 LSE，*不是* full attention matrix。

Recompute 的 FLOPs 约为 forward 的 1.5–2×（多一次 $Q K^T$），但 HBM 省下的 $N^2$ 读写在长序列下永远划算——与 activation checkpointing 的 trade-off 曲线同形。

#note[
  PyTorch `flash_attn` 包 backward 与 forward 共用 tile scheduler；面试追问「存什么 activation」：典型是 $(m, ell)$ 或 LSE，*不是* full attention matrix。
]


=== Block size 受 shared memory 限制

$d$ 或 $N$ 增大时，$B_r, B_c$ 不能无限涨——smem 爆 → occupancy 降 → 即使 IO 理论优也跑不满 GPU。

=== Long context 并行度

`flash_attention_v1_kernel` 并行轴 primarily $N_q$（query 行）。Prefill 时 $N_q = N$ 大尚可；decode 时 $N_q = 1$ *单 thread 串整条 key 序列*——必须换 parallel 策略（v2 的 sequence parallel split $K/V$、warps 协作一行）。

=== v1 → v2 主要改动（面试常考）

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*设计点*], [*FA v1*], [*FA v2*],
    [Parallelism], [主要按 $Q$ block / row], [warps 分工 + sequence parallel],
    [Loop 调度], [外 $K/V$ 内 $Q$（教学）], [优化 work partition 减 idle],
    [Causal 边界], [block skip 手动], [更细 tile 调度 + 预编译],
    [Head dim 大], [smem 压力], [split-K / 更小 tile + recombine],
    [Inference decode], [未优化], [KV cache aware kernel],
  ),
  caption: [*Table:* FA v1 → v2 主要工程改动（面试常考）。v1 论文已含 IO 最优算法；v2 是*工程实现*把 SM 利用率做上去——算法/online 公式不变，变的是 loop order、accumulator 表示与 warp 切分。],
  kind: table,
)

*Observation*：v1 解决「要不要 $N times N$ 矩阵」；v2 解决「既然不要了，怎么把 fused kernel 跑满 tensor core」。decode 阶段 $N_q=1$ 时 v1 并行度差——这是 v2 sequence parallel 的动机之一。

#note[
  v1 论文已含 IO 最优算法；v2 是 *工程实现* 把 SM 利用率做上去——算法思想不变，*不是*新的 online 公式。
]

== 源码运行与对照

```bash
make build/08_flash_attention_v1 && ./build/08_flash_attention_v1
```

默认 $N_q=4$, $N_k=16$, $d=8$, tile size $= 4$。`flash_attention_cpu_reference` 与两个 GPU kernel 对齐，容差 $10^(-4)$。

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*函数*], [*Launch*], [*角色*],
    [`flash_attention_v1_kernel`], [`<<<ceil(4/32), 32>>>`], [register streaming baseline],
    [`flash_attention_v1_shared_kernel`], [`<<<4, 8>>>`], [smem staging 演示],
  ),
  caption: [*Table:* 本章两个 GPU kernel 的 launch 配置与角色。*grid* / *block* 对应 `<<<grid, block>>>` CUDA launch 参数；默认 $N_q=4$, $N_k=16$, $d=8$, tile size $= 4$。],
  kind: table,
)

*Observation*：register 版 1 block × 32 thread 仅 4 行 $Q$ 有活干；shared 版 4 block × 8 thread 每 block 一行但 blockDim 仍凑不满 warp——两者都是 teaching 规模下的结构性 underutilization，不是算法 bug。

== 实测

$N_q = 4$, $N_k = 16$, $d = 8$（源码 `kQueryCount` / `kKeyCount` / `kHeadDim`；$Q,K,V,O$ 各 512 B，整 problem $< 4$ KB），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。表中 TC % = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；warp % = `sm__warps_active.avg.pct_of_peak_sustained_elapsed`；HBM % = `dram__bytes.sum.pct_of_peak_sustained_elapsed`。

Launch 配置决定并行轴与 warp lane 利用率——$N_q = 4$ 远填不满 GPU：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right, left),
    [*version*], [*grid*], [*block*], [*active threads*], [*并行轴*],
    [v1 register], [(1, 1, 1)], [(32, 1, 1)], [4], [`threadIdx.x` = query 行；每 thread 流式扫 $N_k$],
    [v1-shared], [(4, 1, 1)], [(8, 1, 1)], [32（4 block × 8）], [`blockIdx.x` = query 行；8 thread 协作 load，thread 0 串行 merge],
  ),
  caption: [*Table:* ch8 FA v1 teaching kernel 的 launch 配置（`launch__grid_size` / `launch__block_size`）。*active threads* 为实际参与计算的 thread 数；$N_q = 4$ 远填不满 A100 108 SM。],
  kind: table,
)

*Observation*：register 版 `<<<1, 32>>>` 一个 warp 里只有 4 个 thread 对应 4 行 $Q$；shared 版 4 block 各处理一行但 blockDim = 8 仍凑不满 warp——`issued/32` 低是配置问题，不是 online merge 里的 branch divergence。

register 版 `<<<ceil(4/32), 32>>>` = `<<<1, 32>>>`——一个 warp 里只有 4 个 thread 对应 4 行 $Q$。shared 版 `<<<4, 8>>>`——4 个 block 各处理一行，但 blockDim = 8 仍凑不满 warp，且 tile 内 matmul 由 thread 0 独占。

#include "../bench/08_flash_attention_v1.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *v1-shared 9.28 μs vs v1 register 17.86 μs——约 1.9×，smem staging 的收益在本规模上*可测*。* 绝对时间仍在 9–18 μs noise 带内，但 2× 差距稳定——说明 $Q$ 一次进 smem、$K/V$ tile 协作 load 后从 smem 读，确实比 register 版每 tile 每 thread 重读 global $K,V$ 省流量。*不能*用 HBM %（两版均 0.1%）验证这一点——$< 4$ KB 工作集全在 L2，dram 几乎不动。

+ *TC % / warp % / HBM % 全表 0.0–0.1%*。TC % = 0：scalar `dot_row_device`，无 WMMA。warp % = 0.0–0.1%：1–4 个 block 填不满 108 SM——和第 7 章 attention micro-benchmark 同一类「GPU 空转」。*两版都没有触 HBM 带宽墙*；上文 $N = 1024$ 手算的 $approx 1600 times$ IO 差在本 shape 上测不出来。

+ *regs / smem fingerprint*：两版均 40 regs；v1 0 smem，v1-shared 312 B（`q_shared` + `k_tile` + `v_tile` + `scores`）。smem 312 B 换 1.9× 是*方向正确的教学演示*；生产 FA 在此基础上把 smem 拉到 48–164 KB、换 warp 级 TC matmul。

#figure(
  hbar-chart(
    (
      ("v1-shared", 9.28),
      ("v1 register", 17.86),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)`：shared 版约 1.9× 于 register 版——smem 复用 $Q$ 与 $K/V$ tile 的真实收益；绝对时间仍在 teaching 规模的 noise 带内。],
)

#figure(
  warp-grid(
    rows: 4, cols: 16,
    cell: 0.22,
    active: (
      (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
      (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15),
      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
      (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15),
      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
      (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15),
      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
      (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15),
    ),
    row-labels: ("Q0", "Q1", "Q2", "Q3"),
    col-labels: ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13", "K14", "K15"),
    title: "FA v1：$4 times 16$ attention map，外循环 $T_c = 4$ 个 $B_c = 4$ 的 key tile",
  ),
  caption: [
    绿色 cell $(i, j)$ = query $i$ 对 key $j$ 的 score（$S[i,j]$），*从不写 HBM*。
    外循环 `key_start += kTileKeys` 把列方向切成 4 段（K0–K3、K4–K7、…）——每段 load 一次 $K_j, V_j$，在内层对所有 $Q_i$ 复用（register 版是 per-thread 复用逻辑，shared 版是 smem resident）。
  ],
)

*diag 表读关键教学点：*

*a) v1 register `issued/32 = 4.1`，`pred_on/32 = 3.8`——blockDim = 32 但只有 4 行 $Q$*

#raw("<<<1, 32>>>") 里只有 thread 0–3 有输出；其余 28 lane 占 issue slot 但无对应 query 行——*结构性 idle*，不是 warp divergence（不同 lane 走不同 basic block）。`issued/32` $approx 4.1 approx 4/32 times 32$；issued − pred_on $approx 0.3$ 来自 `if (row >= query_count) return` 等 guard 的 predicated-off lane。

#figure(
  warp-lanes(active: range(4), cell: 0.34,
             title: [v1 register：#raw("<<<1, 32>>>")，仅 lane 0–3 各负责一行 Q]),
  caption: [绿色 = 4 个 query thread。灰色 = warp 内无对应行的 lane——*凑不满 warp 的配置问题*，不是 online merge 里的 branch divergence。],
)

*b) v1-shared `issued/32 = 2.2`——比 register 版更低，仍是 teaching 规模 artifact*

shared 版 blockDim = 8：协作 load $K/V$ tile 时最多 8 lane 同时写 smem；tile 内 dot product + online merge 由 thread 0 *串行*执行——多数 phase 只有 1–8 lane 参与 issued 指令，平均 `issued/32` 降到 2.2。*这不是 smem 把 kernel 变慢*（time 9.28 < 17.86），而是*并行度更差但 IO 更好*——4 个 block 分摊 4 行 $Q$，每 block 用 smem 省 global $K,V$ 重读。生产 FA 用多 warp 算 $S_"tile"$ 后 `issued/32` 会回到 28–32。

*c) `smem conf. = 0`；shared 版 `barrier stall = 0.13`*。312 B smem + 每 key tile 一次 `__syncthreads`——metric 证实*无 bank conflict*（不能从 `k_tile[token][d]` 行主序布局单独推断）。`barrier stall` 略高于 register 版（0.00），符合协作 load 的 sync 成本；绝对值仍很小，kernel 太短时不具生产诊断价值。

*d) `mem stall`：v1 = 4.43，v1-shared = 1.79*。shared 版 global load 次数少 → long_scoreboard 略降——*定性方向对*，但两版 kernel 均 $< 20$ μs，stall 比率*不能*用来断言 memory-bound vs compute-bound。

*无信息或为零的 metric：*

- `TC %`：全表 0.0——scalar FFMA，无 tensor pipe；生产 FA 的 $Q K^T$ / $P V$ 走 `mma.sync` / WGMMA。
- `HBM %`：全表 0.1%——$< 4$ KB 工作集 L2 resident；*不能*验证上文 $approx 1600 times$ HBM 流量差。
- `warp %`：0.0–0.1%——1–4 block 填不满 GPU。

#insight[
  1.9× 加速*是真实的 smem staging 收益*，不是 TC 或 occupancy 的胜利——HBM / TC / warp % 全部 $approx 0$。它对应 Algorithm 1 里「$Q_i$ load 一次、$K_j,V_j$ tile 复用」的 IO 结构；放大到 $N >= 1024, d >= 64$ 且工作集超出 L2 后，同一结构才体现为 `dram__bytes` 相对 naive 的 1–2 个数量级降幅。
]

#insight[
  shared 版 `issued/32`（2.2）*低于* register 版（4.1）——反直觉但可解释：更快的那版并行度更差（thread 0 串行 + blockDim = 8）。*lane 利用率低不等于 kernel 慢*；Flash-Attention 生产实现要在 smem 复用与多 warp tile matmul 之间同时拉满 `issued/32` 和带宽。
]

#warn[
  面试别把「FlashAttention 快」和「本章 kernel 快」混为一谈。前者是 IO 最优 + TC GEMM + 大 tile；后者是*最小正确子集*，刻意 scalar 以便读 online merge 循环。也不要把低 `issued/32` 说成「严重 warp divergence」——本规模首先是 blockDim $< 32$ 与 thread-0 串行的*结构性 lane 浪费*；用 `issued/32` vs `pred_on/32` 区分 predication 与真 divergence。
]

== ncu 该看什么

```
ncu --set full --section SpeedOfLight ./build/08_flash_attention_v1
```

教学规模太小，metric 仅作方法练习；换 $N=1024, d=64$ 的 production FA 时期望：

- *dram\_\_bytes.sum*：相对 naive attention 同 shape 降 1–2 个数量级。
- *smsp\_\_sass\_thread\_inst\_executed\_op\_mufu*：exp 仍在 hot path——FA 不是「去掉 exp」。
- *l1tex\_\_t\_bytes\_pipe\_lsu\_mem\_global\_op\_ld.sum* vs smem：`shared_kernel` 应看到更多 smem load、更少重复 global K/V read。
- *gpu\_\_compute\_memory\_throughput.avg.pct\_of\_peak\_sustained\_elapsed*：IO 优化后 compute 占比应上升。

== 面试考点

#interview[
  *Q1*: FlashAttention 快在哪？FLOPs 变了吗？

  A: FLOPs 同阶 $O(N^2 d)$；快在 HBM——不 materialize $S/P$，IO 从 $O(N^2 d)$ 量级降到 $O(N^2 d^2/M)$。fusion 减 launch；SRAM 复用 $K/V$ tile。
]

#interview[
  *Q2*: 推导 online merge 的 $ell_"new"$ 和 $O_"new"$。

  A: $m_"new"=max(m,m')$；$ell_"new" = ell dot exp(m-m_"new") + ell' dot exp(m'-m_"new")$；$tilde(O)_"new" = exp(m-m_"new") tilde(O)_"old" + sum_(k in "tile") exp(x_k-m_"new") V_k$。Finalize $O=tilde(O)/ell$。与第 3 章 chunk merge 一致。
]

#interview[
  *Q3*: IO 复杂度 naive vs FA？$N=1024$, $d=64$, $M=100 "KB"$ 估比。

  A: Naive $O(N^2 d) approx 67 times 10^6$ element-access 量级；FA $O(N^2 d^2 / M) approx 4.2 times 10^4$；比 $approx 1600 times$。强调 FLOPs 不变。
]

#interview[
  *Q4*: 为什么外循环 $K/V$、内循环 $Q$？

  A: 每个 $K_j,V_j$ tile 只从 HBM 读一次，在内层所有 $Q_i$ 上复用——最小化 $K,V$ 的 HBM 流量。反之则 $Q$ 复读次数增。
]

#interview[
  *Q5*: $B_r, B_c$ 怎么选？受什么约束？

  A: smem $approx 4(B_r+B_c)d + B_r B_c$ floats；不能超过 per-block 限额（A100 常用 48–164 KB）。寄存器/warp 数限制 occupancy。$B_r,B_c approx sqrt(M/d)$ 量级；causal 下 $B_c$ 略大有利 $K/V$ 复用。
]

#interview[
  *Q6*: Causal mask 在 FA 里怎么实现？能否 skip block？

  A: 在 $S$ 算完、softmax 前对 $j>i$ 置 $-oo$。若整个 key tile 的 index 均 $>$ 当前 query row block 的最大 index，可 skip 整个 tile（不 load $K,V$，不 matmul）。
]

#interview[
  *Q7*: Backward 为什么不存 $S$ 还能算？

  A: Recompute——反向用相同 tile 遍历再算 $S = Q K^T$；存 forward 的 $(m, ell)$ per row（$O(N)$）。trade FLOPs for HBM。
]

#interview[
  *Q8*: v1 局限？v2 改了什么？

  A: v1：smem 限 block size；长序列 decode 并行度差；单 SM 内串行多。v2：warp 分工、sequence parallel、更好 occupancy——*算法/online 公式不变*，工程并行度改进。
]

#interview[
  *Q9*: `old_scale = exp(running_max - new_max)` 首 tile 为何可为 0？

  A: 尚无 prefix 时 $ell=0$，不应缩放「旧」累加器；设 scale=0 使 `accum*=0` 等价于 fresh start，再加重当前 tile 贡献。
]

#interview[
  *Q10*: Register 版 vs shared 版 teaching kernel 差别？

  A: 数学相同；register 版每 thread 每 tile 从 global 读 $K,V$；shared 版 $Q$ 进 smem 一次、$K,V$ 按 tile 进 smem 再算——演示 IO-aware reuse，后者接近生产结构。
]
