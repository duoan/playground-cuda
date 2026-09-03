#import "../template.typ": *

= Flash-Attention v2

第 8 章 FlashAttention v1 解决了*要不要 materialize $N times N$ attention map* 的问题——online softmax + SRAM tiling 把 HBM traffic 从 $O(N^2)$ 压到 $O(N)$。但 v1 的生产 kernel 在 A100 上仍只用到约 25–40% 的理论 matmul 吞吐。v2 论文（Dao, 2023）*没有改 attention 的数学*，而是从三个工程维度把执行计划重排：

- *(a) 延迟归一化*：中间步不再每 tile 做 $O \/ l_"new"$，只维护未归一化的 $tilde(O)$ 和分母 $l$，最后一步才除。
- *(b) 循环调换*：v1 外循环 K/V tile、内循环 Q row；v2 外循环 Q block、内循环 K/V tile——每个 Q block 独立成一个 CTA，*sequence parallelism* 大幅提升。
- *(c) warp 分工*：v1 的 split-K（4 warp 各算 K 的一部分，warp 间 merge）→ v2 的 split-Q（4 warp 各拿 Q 的一个 slice，算完直接写，无 warp 间通信）。

对应源码：`src/cuda/09_flash_attention_v2.cu`。本章 ladder：

#ladder(
  ("CPU reference",     "逐行 materialize softmax",           "—"),
  ("v2 smem staging",   "1 CTA / Q row, K/V tile 进 smem",  "—"),
  ("v2 warp-specialized", "lane 分组：load / score / accum", "—"),
)

教学 kernel 规模 $N_q = 4$, $N_k = 16$, $d = 8$；正文推导用 LLM 典型 shape（$N = 4096$, $d = 64$, $H = 32$）。

本章在全书中的位置：第 8 章 v1 讲 *IO-aware tiling + online softmax*；本章 v2 讲 *如何把已经 IO 最优的算法跑满 SM*；第 10 章 v3 讲 *Hopper 异步 pipeline*。读代码顺序：`./build/08_flash_attention_v1` 确认 online merge → `./build/09_flash_attention_v2` 对照 Q-outer 与 warp 分工。

#insight[
  v2 面试最高频的一句话：*算法没变，变的是 loop order、accumulator 表示、和 warp 切分轴*。能独立写出 v1/v2 伪代码 diff 和 $O_t$ vs $tilde(O)_t$ 公式，就已经超过大多数候选人。
]

=== 三项改进如何协同（一图胜千言）

把一次 attention 看成许多*独立工人*（CTA），每人负责若干 query 行，按 key 顺序流式读 $K,V$：

```
v1 生产布局（K/V-outer）:
  CTA-0: 持 K tile 0 → 扫 Q row 0..127, 128..255, ...  (Q 内循环)
  CTA-1: 持 K tile 1 → 扫所有 Q ...
  ... CTA 数 ~ N_k/B_c；每个 CTA 内 split-K merge

v2 布局（Q-outer）:
  CTA-0: 持 Q row 0..127 → 扫 K tile 0, 1, 2, ...  (K/V 内循环)
  CTA-1: 持 Q row 128..255 → 扫所有 K ...
  ... CTA 数 ~ N_q/B_r；每个 CTA 内 split-Q 独立
```

*(a) 延迟归一化*：每个工人在 inner loop 里少做 $d$ 次除法，生命周期内 CPU core 更闲、tensor core 更忙。

*(b) Q-outer*：工人数量从「K tile 数」变成「Q block 数」——prefill 时长序列下工人多一个数量级。

*(c) split-Q*：每个工人内部再分 4 个小组（warp），各组独立干活不互相等——消除 v1 split-K 的 merge 排队。

三者叠加才是论文 ~2×；只做其中一项（例如只改 accum 表示但不改 loop order）收益远小于完整 v2。

=== 与第 3 章 softmax 的符号对照

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*第 3 章*], [*FA v1/v2*], [*含义*],
    [$(m, d)$], [$(m, l)$], [running max + exp sum],
    [一行 logits $x_j$], [query $i$ 对 key $j$ 的 $S[i,j]$], [softmax 输入],
    [chunk merge], [K/V tile merge], [分块合并统计量],
    [—], [$tilde(O)$ / $O$], [FA 额外维护分子],
    [输出 $y_j$], [输出 $O[i,:]$], [归一化结果],
  ),
  caption: [*Table:* 第 3 章 online softmax 与 FA v1/v2 的三列符号对照。$d$ 对应 $l$（softmax 分母）；FA 额外维护向量 $tilde(O)$（未归一化 output accumulator），v2 延迟归一化直到循环结束才除 $l$。],
  kind: table,
)

*Observation*：符号不变、工程变——v1 每 tile 可隐式维护 $l dot O$；v2 显式分离 $tilde(O)$ 与最终除法，inner loop 去掉 $d$ 次 FDIV。读本章时重点转为*三件套如何更省、如何按 Q block 并行、如何按 warp 切 Q 行*。

第 3 章只维护标量 $d$（分母）；FA 还要维护向量 $tilde(O) in RR^d$（分子）。v1 每 tile 把 $tilde(O)/l$ 算出来再当 $O$ 用；v2 始终维护 $tilde(O)$ 直到循环结束——这是 FA 相对 plain online softmax 的*额外*结构，也是延迟归一化的切入点。

读第 8 章时若已熟悉 $(m, ell, tilde(O))$ 三件套，本章重点转为*这三件套在 v2 里如何更省、如何按 Q block 并行、如何按 warp 切 Q 行*——符号不变，工程变。

== 问题定义

=== 数学（与 v1 相同）

单 head attention（省略 batch / scale）：

$ O = "softmax"(Q K^T) V, quad Q in RR^(N_q times d), K,V in RR^(N_k times d) $

FlashAttention 的核心约束不变：*$S = Q K^T$ 永不完整写入 HBM*，按 tile 流式处理，用 online state 合并。

=== IO 复杂度：v2 与 v1 相同

第 8 章已证：FA v1 的 HBM traffic 是 $O(N^2 d^2 / M)$ 量级，相对 naive 的 $O(N^2 d)$ 降 $O(d/M)$。v2 *不改变 IO 渐近阶*——$Q,K,V$ 仍各读 $O(N d)$，中间不 materialize $S/P$。v2 的所有收益来自 *单 CTA 内算得更快* 和 *同时跑更多 CTA*，不是新的 IO 算法。

#warn[
  面试别混淆：v1 解决「要不要 $N times N$ 矩阵」；v2 解决「既然不要了，怎么把 fused kernel 跑满 tensor core」。IO 题答 v1；利用率题答 v2。
]

=== v1 在线状态（回顾）

对 query 行 $i$，处理第 $j$ 个 K/V tile 后，v1 维护：

- $m$：running row max（数值稳定）
- $l$：running sum of exp（softmax 分母）
- $O in RR^d$：*已归一化*的 running output（每步 rescale 后仍是 $O = sum w_k V_k \/ l$）

第 3 章 softmax 的 $(m, d)$ 对应 FA 的 $(m, l)$；v1 额外维护分子向量 $O$。

#note[
  符号：本章用 $l$ 表示 softmax 分母（论文里的 $ell$），$tilde(O)$ 表示*未除 $l$* 的 output accumulator。与源码 `running_sum` / `accum` 一一对应。
]

=== CPU reference（对照基准）

源码 `flash_attention_cpu_reference` 按 query 行 materialize 全部 score，再 two-pass softmax——与第 7 章 naive attention 同构，只是合成一个函数便于对拍：

```cpp
for (int row = 0; row < query_count; ++row) {
  // Pass 1: scores[key] = dot(q[row], k[key]); track row_max
  // Pass 2: weight = exp(scores[key] - row_max); accum += weight * v[key]
  // out[row] = accum / row_sum
}
```

GPU kernel 永远不应回到这种 $O(N_k)$ 临时数组——但 CPU 版是验证 online 公式正确性的 golden。`09_flash_attention_v2.cu` 的 `main` 对 `flash_attention_v2_kernel` 和 `flash_attention_v2_warp_kernel` 都做 `check_output`，容差 $10^(-4)$。

=== 从 CPU 到 GPU：v2 改了什么、没改什么

*没改*：attention 的数学定义；online softmax merge 对 $(m, l)$ 的更新式；$S = Q K^T$ 和 $O = P V$ 两个 matmul 的 FLOPs 渐近阶；HBM 上不 materialize $N times N$ 矩阵的 IO 目标。

*改了*：$O$ 的*表示*（归一化 vs $tilde(O)$）；double loop 的*外层绑定 grid*；warp 在 matmul 上的*切分轴*；causal 下*能否 tile 级 skip*；单 CTA 内 CUDA core 的*指令 mix*。

这是典型的「算法正确、实现不饱和」→「实现饱和」二阶段优化。第 7 章 naive attention 是 IO 灾难；第 8 章 v1 解决 IO；本章 v2 解决 compute utilization——面试叙事应沿这条线展开，不要跳步。

== v1 的三个瓶颈

FlashAttention v1 论文在 A100 上测得：即使 IO 已经最优，kernel 仍远未打满 tensor core。根因不是 online softmax 公式错，而是*执行计划*有三处硬伤：

=== 瓶颈 1：非 matmul FLOP 占比过高

v1 每个 inner step 在 rescale $O$ 时做完整归一化：

$ O_"new" = frac(l_"old" dot e^(m_"old" - m_"new") dot O_"old" + tilde(P) V_j, l_"new") $

其中 $tilde(P) V_j$ 是 matmul（tensor core 友好），但分子分母的 rescale 涉及：

- 向量乘标量 $O_"old" times exp(m_"old" - m_"new")$：$d$ 次 FMA
- 向量除标量 $\/ l_"new"$：$d$ 次 FDIV

每个 K/V tile 都做一遍*除法*。$N_k \/ B_c$ 个 tile → $O(N_k \/ B_c dot d)$ 次 FDIV。A100 上 tensor core FP16/BF16 matmul 峰值 ~312 TFLOPS，但 FP32 FDIV 走 CUDA core，吞吐差一个数量级——*非 matmul FLOP 是 v1 的头号敌人*。

#insight[
  Roofline 视角：FlashAttention 的 matmul 强度 $approx 2 d "FLOP/B"$（读 Q/K/V tile），已在 ridge 附近。额外 FDIV/FMA rescale 把有效 AI 拉低，且这些 ops *无法进 tensor core*。v2 改进 (a) 的目标就是砍掉 inner loop 里的 $d$ 次除法。
]

=== 手算：非 matmul FLOP 在 hot loop 里占多少

固定 $d = 128$, $B_c = 64$, 一个 Q block 对一个 K tile：

- *Matmul* $Q K^T$：$2 B_r B_c d approx 2 times 128 times 64 times 128 = 2.1"M"$ FMA → tensor core
- *Softmax*：$B_r B_c$ 次 exp + reduce → CUDA core，但 $O(B_r B_c)$，通常小于 matmul 主项
- *Rescale v1*：$B_r times d$ 次 FMA + $B_r times d$ 次 FDIV *每 tile* → CUDA core

整条序列 $N_k / B_c = 64$ 个 tile：v1 rescale  alone 产生 $64 times 128 times 128 approx 1"M"$ FDIV。Matmul 总量 $approx 64 times 2.1"M" approx 134"M"$ FMA——看起来 FDIV 占比小，但 *FDIV 吞吐只有 FMA 的 ~1/20*，且与 exp 竞争同一套 SFU/CUDA core。profiler 上常见现象：tensor core 30% 忙，而 SFU/`fdiv` 把 issue slot 占满——*v2 砍 FDIV 是在给 matmul 腾 issue 带宽*。

=== 瓶颈 2：并行度 = batch × head

v1 伪代码（论文原版，*外循环 K/V*）：

```cpp
// v1: outer K/V, inner Q
for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
  load K_tile, V_tile into SRAM;
  for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
    load Q_tile into SRAM;
    S_tile = Q_tile @ K_tile^T;       // matmul
    online_softmax_update(S_tile);     // update m, l, O for each Q row
    O_tile += P_tile @ V_tile;         // matmul
    write O_tile back;                 // 或留 smem 等下一 kv_tile
  }
}
```

一个 CTA 负责*一个 K/V tile 对所有 Q tile 的扫描*。Grid 维度 $approx "num_kv_tiles" times B times H$。当 $N_k$ 不大时（推理 decode 阶段 $N_k$ 只有几百），`num_kv_tiles` 很小——*整个 GPU 填不满*。

v1 当时把 Q 放在内循环，是为了让 $O$ 留在寄存器/smem 里、避免每个 kv_tile 都从 HBM reload $O$。这是正确的 SRAM 约束，但牺牲了 Q 维并行。

=== 瓶颈 3：split-K warp 分工 + warp 间通信

v1 在一个 CTA 内算 $S = Q K^T$ 时，把 K 维切给 4 个 warp（split-K matmul 风格）：

```
warp 0: Q @ K[:, 0:16]   ─┐
warp 1: Q @ K[:, 16:32]  ─┼─→ smem reduce → merge m, l, O
warp 2: Q @ K[:, 32:48]  ─┤
warp 3: Q @ K[:, 48:64]  ─┘
```

每个 warp 只算 score tile 的一部分，必须 `__syncthreads()` + smem atomic/reduction 合并 partial max、partial sum、partial $O$——*warp 间强同步*，latency 高。

=== 瓶颈 3 的 profiler 指纹

在 nsight-compute 里打开 FA v1 生产 kernel，常见：

- `smsp__warp_issue_stalled_barrier` 高——split-K merge 的 `__syncthreads`
- `smsp__sass_thread_inst_executed_op_ddiv` 高——每 tile 的 $O \/ l$
- `sm__inst_executed_pipe_tensor` 仅 25–40%——matmul 被 non-matmul 和 barrier 饿死

v2 的三项改进*精准对应*这三个 metric。面试若问「你怎么知道 bottleneck 在哪」，答：*先看 tensor core 利用率，再看 ddiv/barrier*——attention fused kernel 专用诊断路径。

#warn[
  本书第 8 章教学源码 `08_flash_attention_v1.cu` 用的是简化版（1 thread / Q row 或 1 block / Q row），便于理解 online softmax。论文生产 v1 kernel 才是上述 split-K + K/V outer 布局。面试追问 v1 vs v2 时，*以论文布局为准*，再对照源码理解数值部分。
]

== 改进 (a)：延迟归一化

=== v1 rescale：每步都除

设第 $t$ 步处理 tile $T_t$，更新后：

$ m_t = max(m_(t-1), max_(j in T_t) S[i,j]) $

$ l_t = l_(t-1) dot e^(m_(t-1) - m_t) + sum_(j in T_t) e^(S[i,j] - m_t) $

$ O_t = frac(l_(t-1) dot e^(m_(t-1) - m_t) dot O_(t-1) + sum_(j in T_t) e^(S[i,j] - m_t) V_j, l_t) $

最后一步 $O_T$ 已经是正确归一化输出。但*每步*都对 $d$ 维向量做除法。

=== v2 rescale：维护 $tilde(O)$，最后才除

关键观察：分子 $l_(t-1) e^(m_(t-1)-m_t) O_(t-1) + tilde(P) V_t$ 和分母 $l_t$ 同步缩放。定义*未归一化 accumulator*：

$ tilde(O)_t = l_t dot O_t = l_(t-1) dot e^(m_(t-1) - m_t) dot tilde(O)_(t-1) + tilde(P) V_t $

更新规则（*无除法*）：

$ tilde(O)_t = e^(m_(t-1) - m_t) dot tilde(O)_(t-1) + tilde(P) V_t quad "(当" l_(t-1) > 0 ")") $

$ l_t = l_(t-1) dot e^(m_(t-1) - m_t) + sum_(j in T_t) e^(S[i,j] - m_t) $

全部 tile 处理完后：

$ O = tilde(O)_T \/ l_T $

=== 等价性证明（面试可白板）

*命题*：设 v1 和 v2 从相同初始 state 出发，处理相同 key tile 序列，则 v2 最终 $tilde(O)_T \/ l_T$ 等于 v1 最终 $O_T$。

*归纳*：假设处理完 $t-1$ 个 tile 后 v1 得 $(m_(t-1), l_(t-1), O_(t-1))$，v2 得 $(m_(t-1), l_(t-1), tilde(O)_(t-1))$ 且不变式 $tilde(O)_(t-1) = l_(t-1) O_(t-1)$ 成立。

第 $t$ 个 tile 贡献的新增未归一化权重为 $Delta = sum_(j in T_t) e^(S[i,j]-m_t) V_j$（relative to 统一 $m_t$）。v1 更新：

$ O_t = frac(l_(t-1) e^(m_(t-1)-m_t) O_(t-1) + Delta, l_t) $

v2 更新 $tilde(O)_t = l_(t-1) e^(m_(t-1)-m_t) tilde(O)_(t-1) + Delta = l_t O_t$。故 $tilde(O)_t \/ l_t = O_t$。归纳成立。

*要点*：v2 只是把「先合并分子再除」改成「先合并未归一化分子最后除一次」——softmax 线性性保证等价，与第 3 章 online merge 证明同构。

=== FLOP 节省

每个 inner step 省 $d$ 次 FDIV。总节省：

$ "saved" = (N_k \/ B_c) times N_q times d quad "FDIV per head" $

例：$N_k = 4096$, $B_c = 64$, $N_q = 4096$, $d = 64$ → 每个 head 省 $64 times 4096 times 64 approx 1.7 times 10^7$ 次除法。32 heads × batch 8 → *数十亿次 FDIV* 从 hot loop 消失。

同时 rescale $tilde(O)$ 的 FMA 从 $2d$（乘 $l_"old"\/l_"new"$ 等价形式）降到 $d$（只乘 $e^(m_"old"-m_"new")$）——省一半非 matmul FMA。

#insight[
  延迟归一化不改变数学结果——只是交换了除法和乘法的顺序。与 softmax 章 online merge 同一思想：*把 expensive op 推迟到最后*。生产 v2 kernel 里 $tilde(O)$ 通常 FP32 累加，最后一步除 $l_T$ 才写 BF16/FP16 输出。
]

=== 数值例子：两 tile merge

$d = 1$（标量 $V$），query 行 $i$，两 key tile 各 1 个元素。Tile 1：$S = 1$, $V = 10$；Tile 2：$S = 3$, $V = 100$。

*标准 softmax*：$m = 3$, $l = e^(-2)+1 approx 1.135$, $O = (e^(-2) dot 10 + 100) / l approx 89.4$。

*v1 逐步归一化*（每 tile 后除 $l$）：

- Tile 1 后：$m=1$, $l=e^0=1$, $O = 10/1 = 10$
- Tile 2：$m_"new"=3$, scale $= e^(1-3)=e^(-2)$
  - $l = 1 dot e^(-2) + e^0 = e^(-2)+1$
  - $O = (10 dot e^(-2) + 100) / (e^(-2)+1) approx 89.4$ ✓

*v2 延迟归一化*（维护 $tilde(O) = l dot O$）：

- Tile 1：$tilde(O) = 10$, $l = 1$
- Tile 2：$tilde(O) = e^(-2) dot 10 + 100$, $l = e^(-2)+1$
- Final：$O = tilde(O)/l approx 89.4$ ✓——*中间从未做除法*

=== 源码对应

`flash_attention_v2_kernel` 里 `accum` 存的就是 $tilde(O)$（未除 $l$），`running_sum` 是 $l$。inner loop 只做 `accum[d] *= old_scale` + FMA，*没有* inner 除法：

```cpp
const float old_scale = (running_sum == 0.0f) ? 0.0f
    : expf(running_max - new_max);
for (int d = 0; d < head_dim; ++d) {
  accum[d] *= old_scale;
}
// ... tile_sum, accum += weight * V ...
running_sum = running_sum * old_scale + tile_sum;
// 循环结束后才除：
out[row * head_dim + d] = accum[d] / (running_sum + kEps);
```

== 改进 (b)：循环调换与 Sequence Parallelism

=== v1 vs v2 伪代码

*v1（K/V outer, Q inner）*：

```cpp
for kv_tile in 0..num_kv_tiles {          // grid.x
  load K_tile, V_tile;
  for q_tile in 0..num_q_tiles {          // 串行在 CTA 内
    load Q_tile;
    S = Q_tile @ K_tile^T;
    update(m, l, O);
    O += P @ V_tile;
  }
}
```

*v2（Q outer, K/V inner）*：

```cpp
for q_tile in 0..num_q_tiles {           // grid.x — 每个 Q block 一个 CTA
  load Q_tile into SRAM/reg;
  init m, l, tilde(O);
  for kv_tile in 0..num_kv_tiles {        // 串行在 CTA 内
    load K_tile, V_tile;
    S = Q_tile @ K_tile^T;
    update(m, l, tilde(O));               // 无 inner 除法
    tilde(O) += P @ V_tile;
  }
  O = tilde(O) / l;                       // 最后归一化，写 HBM
}
```

#note[
  伪代码里 `update(m, l, Õ)` 在 v2 实现中*不包含*除法——这是与 v1 伪代码的唯一数值差异。Loop 结构差异是 `grid` 绑在 `q_tile` 还是 `kv_tile`。
]

=== 为什么 v2 的 Q-outer 天然并行

每个 Q block 的 $(m, l, tilde(O))$ *完全独立*——不同 query 行之间无数据依赖。v2 直接：

```cpp
flash_attention_v2_kernel<<<num_q_tiles * B * H, threads_per_block>>>(...);
```

Grid 从 $O(N_k \/ B_c)$ 变成 $O(N_q \/ B_r times B times H)$。LLM prefill 时 $N_q = N_k = 4096$ 量级，CTA 数从 $approx 64$ 涨到 $approx 4096 \/ B_r times 32 approx 10^4$（$B_r = 128$）——*SM 利用率大幅提升*。

这就是 v2 引入的 *sequence parallelism*：在 sequence（Q）维上切 CTA，而非只在 batch × head 维并行。

=== v1 为什么当时 Q-inner

v1 把 Q 放内循环不是随便写的——*$O$ 必须跨 K/V tile 累积*。若 Q-outer 但每个 CTA 只处理一个 Q block，$tilde(O)$ 可以留在寄存器/smem 里走完所有 kv_tile，*不需要*每 tile 从 HBM reload $O$。v1 时代担心 smem/reg 不够放整个 Q block 的 state，所以让 K/V tile 「驻留」CTA、Q 行在内循环流式扫——用 Q-inner 换 $O$ 的 SRAM residency。

=== SRAM 账本：为什么 v2 敢 Q-outer

估算一个 CTA 的 SRAM 需求（FP32，$B_r=128$, $B_c=64$, $d=128$）：

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, left),
    [*Buffer*], [*Size*], [*说明*],
    [Q tile], [$128 times 128 times 4 = 64 "KB"$], [驻留整个 inner loop],
    [K tile], [$64 times 128 times 4 = 32 "KB"$], [每 kv 迭代覆盖],
    [V tile], [$32 "KB"$], [与 K 同形],
    [S tile], [$128 times 64 times 4 = 32 "KB"$], [score 暂存],
    [$tilde(O), m, l$], [$128 times 128 times 4 + 2 times 128 times 4 approx 65 "KB"$], [Q-outer state],
    [*合计*], [$approx 225 "KB"$], [超 A100 默认 48KB；需 dynamic smem 或减 tile],
  ),
  caption: [*Table:* v2 Q-outer CTA 的 SRAM 账本（FP32，$B_r=128$, $B_c=64$, $d=128$）。Q tile 驻留整个 inner kv loop；$tilde(O), m, l$ 是 Q-outer 相对 v1 K/V-outer 的新增 resident state。A100 单 block smem 上限 164 KB。],
  kind: table,
)

*Observation*：合计 $approx 225 "KB"$ 超默认 48 KB——v2 敢 Q-outer 的前提是硬件 smem 增大 + 延迟归一化减 reg 压力；v1 时代 K/V-outer 是 smem 不够大时的*正确妥协*。

论文通过 *减小 $B_r$ 或 $d$ 分块*、*寄存器 spill 到 smem*、*dynamic shared memory* 控制在 164 KB 上限内。v1 时代 K/V-outer 让 $K,V$ tile 在 outer CTA 驻留、$O$ 随 Q 行流式更新——是 smem 不够大时的*正确妥协*。硬件 smem 增大 + 延迟归一化减 reg 压力后，v2 翻转 loop order。

=== Occupancy 与 $B_r$ 的权衡实例

A100 单 SM：最多 2048 threads、64 warps；每 block 128 threads → 理论最多 16 block/SM。若 smem = 128 KB/block，164 KB 上限 → 每 SM 最多 1 block——*occupancy 从 16 降到 1*。

FA-2 论文常用 $B_r=128$, $B_c=64$, $d=128$ 时 smem $approx 128–160$ KB，occupancy 常只有 1–2 block/SM——*靠 grid 总 CTA 数填满载*，不靠单 SM 高 occupancy。这是 GEMM-like kernel 的常见模式：大 tile 低 occupancy × 多 SM 并行。

面试 contrast：vector add（第 1 章）靠高 occupancy 掩盖 latency；FA-2 靠大 matmul tile 提高 instruction intensity，occupancy 反而不是第一优化目标。

v2 证明：Q block 的 $tilde(O)$（$B_r times d$ 个 float）+ $m, l$（$B_r$ 个 float）完全可以放进 smem/reg。配合改进 (a) 不再 inner 除法，Q-outer 的 reload 代价消失。

#note[
  改进 (a) 和 (b) 互相依赖：Q-outer 让每个 CTA 持有完整 $tilde(O)$ 直到最后；延迟归一化保证 $tilde(O)$ 在 inner loop 里只做 FMA、不做 FDIV，寄存器压力可控。
]

=== 并行度公式

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*版本*], [*CTA 数（单 head）*], [*瓶颈场景*],
    [v1], [$ceil(N_k / B_c) times B times H$], [decode：$N_k$ 小 → grid 小],
    [v2], [$ceil(N_q / B_r) times B times H$], [prefill：$N_q$ 大 → grid 大],
  ),
  caption: [*Table:* v1 vs v2 单 head 的 CTA 数公式。v1 grid 绑 K/V tile 轴（$N_k$）；v2 grid 绑 Q block 轴（$N_q$）。decode 时 $N_q=1$ 则 v2 无 sequence parallel 优势。],
  kind: table,
)

*Observation*：当 $N_q = N_k$ 时两公式同阶；差异在 decode（$N_q=1$）或 cross-attention（$N_q != N_k$）——prefill 长 prompt 时 v2 CTA 数随 $N_q$ 线性增长，这是 sequence parallelism 的核心收益。

Decode 阶段 $N_q = 1$ 时 v2 也只有 1 个 Q CTA——此时 v2 无 sequence parallel 优势，靠 split-Q warp 分工和延迟归一化提速。Prefill 阶段 v2 相对 v1 *论文*报告 *~2×* 端到端加速（tensor core + 大 shape），主要来自并行度 + 非 matmul FLOP 削减——*本书 teaching kernel（$N_q=4$）只能看到 1–4% 量级差异*，见下文「实测」节。

=== 手算：prefill 时 CTA 数差多少

$B = 8$, $H = 32$, $N = 4096$, $B_c = B_r = 128$：

- v1：$ceil(4096/128) times 8 times 32 = 32 times 256 = 8192$ CTA
- v2：$ceil(4096/128) times 8 times 32 = 8192$ CTA——*同阶*？

看起来一样，是因为此例 $N_q = N_k$。差异在 *decode* 或 *cross-attention*（$N_q != N_k$）：

- Decode 一步：$N_q = 1$, $N_k = 8192$（KV cache 长度）
  - v1 grid：$ceil(8192/128) times 8 times 32 = 64 times 256 = 16384$ CTA——仍不少
  - v2 grid：$ceil(1/128) times 8 times 32 = 1 times 256 = 256$ CTA——*骤降*

- Cross-attention（encoder-decoder）：$N_q = 512$（decoder），$N_k = 4096$（encoder）
  - v1：$32 times 256 = 8192$
  - v2：$4 times 256 = 1024$——v1 在 K 维切更多 CTA，但每个 CTA 内要串行扫 512 行 Q

*Sequence parallelism* 的收益在 $N_q$ 大时最显著（prefill、长 prompt）；$N_q$ 小时靠 (a)(c) 优化单 CTA 效率。工业界因此 prefill 用 FA-2，decode 常配合 paging / split-KV 等专用 kernel（FlashDecoding 等）——超出本章，面试提一句即可。

=== Grid 映射（生产）

典型 launch：

```cpp
dim3 grid(ceil_div(N_q, kBlockM), H, B);  // x=Q block, y=head, z=batch
dim3 block(kThreadsPerBlock);               // 128 threads = 4 warps
```

`blockIdx.x` 决定 Q tile 起点，`blockIdx.y/z` 选 head 与 batch。v1 生产 kernel 常把 `grid.x` 绑 K tile——这是 v1/v2 *launch 签名* 最直观的差别。

== 改进 (c)：Split-K → Split-Q

=== v1 split-K 布局

一个 CTA 处理 $B_r times B_c$ 的 $S$ tile，K 维 $B_c$ 切给 $W$ 个 warp：

```
         K dim →
       [ 0..15 | 16..31 | 32..47 | 48..63 ]
Q  w0  [ score fragment 0 ] ──┐
   w1  [ score fragment 1 ] ──┼── smem: merge max, sum, O
   w2  [ score fragment 2 ] ──┤   __syncthreads × 2+
   w3  [ score fragment 3 ] ──┘
```

4 warp 各算 1/4 的 dot product，partial result 必须合并——*warp 间 barrier + smem 通信* 是 v1 CTA 内的常态。

=== v2 split-Q 布局

v2 把 Q 维 $B_r$ 切给 $W$ 个 warp，每个 warp 拿 $B_r \/ W$ 行 Q：

```
         Q rows →
       [ row 0-31 | row 32-63 | row 64-95 | row 96-127 ]
K  w0  [ 完整 m,l,Õ for rows 0-31  ] → 独立写 O[0:32]
   w1  [ 完整 m,l,Õ for rows 32-63 ] → 独立写 O[32:64]
   w2  [ 完整 m,l,Õ for rows 64-95 ] → 独立写 O[64:96]
   w3  [ 完整 m,l,Õ for rows 96-127] → 独立写 O[96:128]
```

每个 warp 拥有独立的 $(m, l, tilde(O))$，算完整条 Q slice 对所有 K/V tile 的 attention——*warp 间零通信*，不需要 `__syncthreads()` 做 reduction。

#insight[
  split-Q 能 work 的前提是改进 (b)：Q-outer 让每个 CTA 只服务一个 Q block，warp 切 Q 行自然独立。若仍是 v1 的 K/V-outer，Q 行在不同 kv_tile 间共享 state，就无法按 Q slice 切 warp。
]

K/V tile 仍由 block 内 thread 协作 load 到 smem（producer），各 warp consumer 读同一份 smem——这是 *smem broadcast*，不是 warp 间 reduction。

=== 教学 kernel 的 warp 分工

`flash_attention_v2_warp_kernel` 把 32 lane 分成三组（规模 $d=8$, $B_c=4$ 的玩具版）：

- lane 0–7：持有 `q_shared`，最后写 `out[row, d]`
- lane 0–3：协作 load K/V tile + 算 dot product score
- lane 0：online softmax state update（$m, l$, weight）

```cpp
if (lane < kTileKeys) {
  // load K/V + compute score for this lane's key
  shared_scores[lane] = dot(q_shared, shared_k[lane]);
}
__syncthreads();
if (lane == 0) {
  // merge max, compute weights, update running_sum
}
__syncthreads();
if (lane < kHeadDim) {
  accum[lane] *= old_scale_shared;
  for (int t = 0; t < kTileKeys; ++t)
    accum[lane] += shared_weights[t] * shared_v[t][lane];
}
```

生产 v2 kernel 是 4 warp × 32 lane = 128 threads/block，每 warp 32 行 Q，tensor core 做 $S = Q K^T$ 和 $P V$。教学版用 lane 分组*示意*角色分离——读源码时抓住「谁 load、谁 matmul、谁维护 state」。

=== split-K 与 split-Q 的 reduction 对比

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*操作*], [split-K (v1)], [split-Q (v2)],
    [切分轴], [K 维 $B_c$], [Q 维 $B_r$],
    [Partial 产物], [score fragment], [完整 row 的 $(m,l,tilde(O))$],
    [Merge 对象], [max, sum, $tilde(O)$ across K], [无——每 warp 完整],
    [Sync 类型], [`__syncthreads` + smem reduce], [仅 K/V load 后 broadcast],
    [写回], [merge 后统一写], [各 warp 写各自 Q slice],
    [Failure mode], [barrier 不平衡 → 慢], [smem bank conflict on broadcast],
  ),
  caption: [*Table:* v1 split-K 与 v2 split-Q 的 reduction 对比。split-K 每 tile 需 cross-warp merge $(m,l,tilde(O))$；split-Q 每 warp 独立维护完整 row state，仅 K/V load 后 smem broadcast。],
  kind: table,
)

*Observation*：split-Q 能 work 的前提是 Q-outer（改进 b）——每个 CTA 只服务一个 Q block，warp 切 Q 行自然独立。v1 split-K 的 `__syncthreads` barrier 是 profiler 上 `warp_issue_stalled_barrier` 高的根源。

生产 FA-2 用 swizzle layout 解决 broadcast 的 bank conflict；v1 用 warp shuffle 加速 split-K reduce——两种分工对应不同 bottleneck。

#warn[
  教学 kernel *不是*生产性能。它用 scalar dot product 代替 tensor core，单 thread 做 softmax merge。价值在于*看清 v2 的数据流*：Q-outer → K/V inner → 延迟归一化 → warp 角色分离。
]

=== 生产 kernel 的 warp 时间线（概念）

一个 128-thread block（4 warp）处理 $B_r = 128$ 行 Q、$B_c = 64$ 列 K 的理想分工：

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, left, left),
    [*Warp*], [*Q 行*], [*寄存器 state*], [*Matmul*], [*写回*],
    [w0], [0–31], [$(m,l,tilde(O))$ × 32 行], [S tile, P V], [独立],
    [w1], [32–63], [同上], [同上], [独立],
    [w2], [64–95], [同上], [同上], [独立],
    [w3], [96–127], [同上], [同上], [独立],
  ),
  caption: [*Table:* 生产 FA-2 128-thread block（4 warp）的理想 split-Q 分工（$B_r=128$, $B_c=64$）。每 warp 32 行 Q、独立 $(m,l,tilde(O))$；K/V tile 由 block 协作 load 后 broadcast 给 4 个 consumer warp。],
  kind: table,
)

*Observation*：K/V tile *一次 load，四次 consumer*——softmax rescale 在各 warp 私有寄存器完成，*无 cross-warp reduce*。这与 v1 split-K 每 tile 必须 4 warp merge 形成鲜明对比。

K/V tile load 由 block 内所有 thread 协作（或 dedicated producer warp）——*一次 load，四次 consumer*。Softmax rescale 在各 warp 私有寄存器完成，*无 cross-warp reduce*。这与 v1 split-K 每 tile 必须 4 warp merge 形成鲜明对比。

== v1: smem staging kernel

`flash_attention_v2_kernel` 是 v2 算法骨架的最简 GPU 实现：1 CTA = 1 Q row，K/V tile 进 smem，thread 0 做全部数值。

```cpp
__global__ void flash_attention_v2_kernel(
    const float* q, const float* k, const float* v, float* out,
    int query_count, int key_count, int head_dim) {
  const int row = blockIdx.x;   // Q-outer：每个 block 一行 Q
  // ...
  for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
    // threads 协作 load K/V tile → shared_k, shared_v
    __syncthreads();
    if (threadIdx.x == 0) {
      // score → online update accum (Õ), running_sum (l)
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[row * head_dim + d] = accum[d] / (running_sum + kEps);
}
```

Launch：`<<<kQueryCount, kTileKeys>>>`——grid = $N_q$，block = $B_c$ threads 协作 load。

=== 对照 `flash_attention_v2_kernel` 完整结构

教学 kernel 的 host 侧与 device 侧接口（摘自 `09_flash_attention_v2.cu`）：

```cpp
// Host
flash_attention_v2_kernel<<<kQueryCount, kThreadsPerBlock>>>(
    device_q, device_k, device_v, device_out,
    kQueryCount, kKeyCount, kHeadDim);

// Device 常量
constexpr int kQueryCount = 4;
constexpr int kKeyCount = 16;
constexpr int kHeadDim = 8;
constexpr int kTileKeys = 4;  // B_c
```

Device 侧 state 变量语义：

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*变量*], [*类型*], [*数学*],
    [`running_max`], [float], [$m$ — row max],
    [`running_sum`], [float], [$l$ — softmax 分母],
    [`accum[d]`], [float[kHeadDim]], [$tilde(O)$ — 未归一化 output],
    [`shared_k/v`], [smem], [当前 K/V tile],
    [`old_scale`], [float], [$exp(m_"old" - m_"new")$],
  ),
  caption: [*Table:* `flash_attention_v2_kernel` device 侧 state 变量与数学符号对照。`accum` 存 $tilde(O)$（未除 $l$）；inner loop 只做 `accum *= old_scale` + FMA，循环结束后才 `accum / running_sum`。],
  kind: table,
)

*Observation*：`old_scale` $= exp(m_"old" - m_"new")$ 同时缩放 $tilde(O)$ 和 $l$——与第 8 章 v1 相同 trick，但 v2 明确分离 $tilde(O)$ 与最终除法。`flash_attention_v2_warp_kernel` 把 state 放 smem 是教学简化，生产用 warp 私有寄存器。

`flash_attention_v2_warp_kernel` 把 `running_*` 放到 `*_shared`，让 lane 0 写、全 block 读——教学简化；生产用 warp 私有寄存器。

对比 `08_flash_attention_v1.cu` 的 `flash_attention_v1_shared_kernel`：v1 教学版也是 1 block / Q row，但 inner loop 的 rescale 逻辑等价于 v1 论文的*每步归一化*写法（`accum` 在每步后语义上等于 $l dot O$ 但未显式区分）。v2 kernel 明确分离 $tilde(O)$ 与最终除法，且为 warp 分工版铺路。

=== 完整 inner loop 解读

对照 `09_flash_attention_v2.cu` 第 136–189 行，一个 K/V tile 的生命周期：

*1. 协作 load（threadIdx.x = local key index）*

每个 thread 搬一个 key 的 `K[key, :]` 和 `V[key, :]` 到 `shared_k[local_key][d]` / `shared_v`。`kTileKeys = 4` threads 即可 cover tile——教学版 block 大小 = tile 宽。

*2. Score + online update（thread 0）*

- 对 tile 内每个 key 算 `score = dot(q[row], shared_k[t])`
- 更新 `running_max` → `new_max`，算 `old_scale = exp(running_max - new_max)`
- `accum[d] *= old_scale`——对应 $tilde(O)$ rescale，*无除法*
- 算 `weight = exp(score - new_max)`，累加 `tile_sum` 和 `accum += weight * V`
- `running_sum = running_sum * old_scale + tile_sum`——更新 $l$

*3. 循环结束*

`out[row,d] = accum[d] / running_sum`——*全函数唯一一次 FDIV 向量*。

#note[
  `old_scale == 0` 当 `running_sum == 0`（首个 tile）：`accum *= 0` 等价 fresh start，与第 8 章 v1 教学 kernel 相同 trick。
]

== v2: warp-specialized kernel

`flash_attention_v2_warp_kernel` 在 smem staging 之上加入 lane 角色分离：

```cpp
__global__ void flash_attention_v2_warp_kernel(...) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  // lane < d: load q_shared
  // lane < B_c: load K/V + score
  // lane == 0: softmax state
  // lane < d: update accum (Õ), final write
}
```

Launch：`<<<kQueryCount, 32>>>`——一个 warp 处理一行 Q。

关键差异：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*组件*], [*smem kernel*], [*warp kernel*],
    [Q load], [thread 0 隐式读 global], [lane 0–7 协作 load `q_shared`],
    [K/V load], [每 thread 1 key], [lane 0–3 各 load 1 key],
    [score], [thread 0 串行 dot], [lane 0–3 并行 dot],
    [softmax], [thread 0], [lane 0],
    [accum update], [thread 0 串行 $d$ 维], [lane 0–7 各更新 1 维],
    [写回], [thread 0], [lane 0–7 并行写],
  ),
  caption: [*Table:* `flash_attention_v2_kernel`（smem）与 `flash_attention_v2_warp_kernel` 的 lane 角色分工。warp 版把 score 与 $d$ 维 accum 更新从 thread 0 串行拆出——split-Q 在 $d$ 维的微型演示。],
  kind: table,
)

*Observation*：warp 版 `issued/32`（6.6）高于 smem 版（1.6），但 time 仅快约 1.1%——本规模瓶颈是 launch 固定开销，不是 SM issue 带宽。生产 split-Q 是 4 warp × 32 行 Q，而非 8 lane 分 $d$ 维。

warp 版把 $d$ 维 accum 更新*并行化*——生产 kernel 里这对应 split-Q：每 warp 独立维护 $tilde(O)$ 的若干维/行。

=== warp kernel 相对 smem 版的增量

`flash_attention_v2_warp_kernel` 新增：

1. *`q_shared`*：lane 0–7 预加载整行 Q，score 阶段不再读 global Q——减少 HBM 流量，也为 split-Q 预热。
2. *`shared_weights`*：lane 0 算完 softmax weight 后 broadcast，lane 0–7 并行做 `accum[d] += weight * V[t,d]`——$d$ 维 FMA 从串行变并行。
3. *state 放 smem*：`running_max_shared`, `old_scale_shared` 等——lane 0 写、其他 lane 读，教学用 smem 代替 warp shuffle。

生产 FA-2 会把 state entirely 放在 warp 私有寄存器，weight broadcast 用 `__shfl_sync`——但*数据流*与教学 kernel 一致。

=== warp kernel inner loop 逐步对照

以下映射 `flash_attention_v2_warp_kernel` 第 327–397 行：

*Step A — Q 驻留 smem*（只做一次）

```cpp
if (lane < kHeadDim)
  q_shared[lane] = q[row * head_dim + lane];
if (lane == 0) { running_max_shared = -1e30f; running_sum_shared = 0.0f; }
__syncthreads();
```

一行 Q 被 8 个 lane 载入 smem，后续所有 key tile 复用——对应生产里 Q tile 驻留寄存器/smem 直到 kv 循环结束。

*Step B — K/V tile load*（每 kv tile）

`lane < kTileKeys` 的 thread 各负责一个 key index，搬 `K[key,:]` / `V[key,:]` 到 `shared_k/v[lane][d]`。与 smem kernel 相同，但*不再*由 thread 0 独占 load。

*Step C — 并行 score*

每个 `lane < kTileKeys` 算 `dot(q_shared, shared_k[lane])` 写入 `shared_scores[lane]`——4 个 score 并行，对比 smem kernel 里 thread 0 串行 4 次 dot。

*Step D — lane 0 softmax merge*

与 smem kernel 相同：求 tile max → `new_max` → `old_scale` → `shared_weights[t] = exp(score - new_max)` → 更新 `running_sum_shared`。这是整个 block 唯一的*串行 softmax 瓶颈*——生产版用 warp 级 parallel max/sum 或 dedicated softmax warpgroup。

*Step E — 并行 accum 更新*

`lane < kHeadDim`：`accum[lane] *= old_scale_shared`，再对 tile 内每个 key 做 FMA。8 维 output 并行更新——split-Q 在 $d$ 维的微型演示；生产版是 32 行 Q × $d$ 维。

*Step F — 写回*

`out[row * head_dim + lane] = accum[lane] / running_sum_shared`——每 lane 写一维，无 thread 0 独占 epilogue。

=== 寄存器压力：为什么 split-Q 需要延迟归一化

若 v2 仍用 v1 的「每 tile 归一化 $O$」*且* split-Q，每个 warp 要在寄存器里存*已除 $l$* 的 $O$——每次 $m$ 更新还要对 $O$ 做 rescale + 再除新 $l$，寄存器读写次数翻倍。

延迟归一化让 warp 只维护 $tilde(O)$ 和标量 $l,m$——rescale 只做 `tilde(O) *= exp(m_old - m_new)`（FMA），无 FDIV。这对 split-Q 至关重要：4 warp × 32 行 × $d$ 维 state 全部在寄存器，任何多余 ops 都会 spill 到 local memory。

#insight[
  warp kernel 相对 smem kernel 的核心增量是 *Step C 和 Step E 的并行度*——这正是 split-Q 的缩影：把原本 thread 0 串行的 $O(d)$ 和 $O(B_c)$ 工作拆给多个 lane/warp。生产 FA-2 进一步把 Step C/E 换成 WGMMA，Step D 留在 CUDA core/SFU。
]

== Causal Mask 优化

Decoder self-attention 用 causal mask：query $i$ 只能 attend 到 key $j <= i$。v1 在每个 score 上 `if (i < j) score = -inf`，*仍加载整个 K tile*。

v2 利用 Q-outer + K/V inner 顺序：处理 Q block 行 $[i_0, i_0+B_r)$ 时，若整个 K tile 的 key index 范围 $[j_0, j_0+B_c)$ 满足 $j_0 > i_0 + B_r - 1$（整个 tile 在 causal 上三角），*直接 skip*——不算 matmul、不做 softmax。

#insight[
  对长序列 prefill，越靠后的 Q block 跳过的 K tile 越多。Q block 在序列尾部时，大约跳过 50% 的 K/V tile work。v1 因 K/V-outer 无法做 tile 级 skip（一个 kv_tile 要服务所有 Q row，部分 Q 行需要、部分不需要）。v2 Q-outer 让每个 CTA 知道自己的 Q 行范围，tile skip 成为*结构化优化*。
]

量化：$N = 4096$, $B_c = 64$ → 64 个 K tile。对最后一个 Q block（行 4032–4095），只需 attend 到 key 4095，前 63 个 K tile 中约 62 个完全在上三角——*skip $approx 97%$ tile*。平均 over all Q blocks，causal skip 省 $approx N_k \/ (2 B_c) \/ (N_k / B_c) = 50%$ 的 K/V tile 计算。

Mask 实现：生产 kernel 在 loop 开头比较 `kv_tile_idx * B_c > q_row_end`，成立则 `continue`——零 exp、零 matmul。

=== 手算：causal skip 省多少 work

$N = 4096$, $B_r = B_c = 64$ → 64 个 Q tile、64 个 K tile。Causal 下 Q tile $q$ 只需 K tile $0..q$（共 $q+1$ 个），不需要 $q+1..63$。

- Q tile 0：1 个 K tile
- Q tile 1：2 个
- ...
- Q tile 63：64 个

总 matmul tile 数：$sum_(q=0)^63 (q+1) = 64 times 65 / 2 = 2080$。无 causal 时需 $64 times 64 = 4096$。比值 $2080/4096 approx 50.8%$——*正好一半*。

最后一个 Q tile（$q=63$）仍要 64 个 K tile——*无法 skip*；第一个 Q tile（$q=0$）可 skip 63 个——*skip 98.4%*。v2 的收益在 prompt 前段最大（大量 skip），后段接近全量 matmul。这与 GPT prefill「越写越慢」的直觉一致——不是 bug，是 causal 结构。

v1 K/V-outer 即使用元素级 mask，每个 kv CTA 仍要对所有 Q tile 做 matmul（mask 在 softmax 前），无法获得上述 tile 级 50% 节省。

=== Causal tile skip 图示

Self-attention causal 矩阵（× = 需计算，空白 = 可 skip 的 block）：

```
      K tile →  0    1    2   ...  63
Q tile 0      [×] [ ] [ ] ... [ ]
      1      [×] [×] [ ] ... [ ]
      ...
     63      [×] [×] [×] ... [×]
```

Q tile $i$ 只需列 index $<= i$ 的 K tile。v2 每个 CTA 固定 Q tile index，内循环 `for kv_tile = 0..i` 即可——*连 load 都省略*。v1 K/V-outer 的 CTA 固定 kv tile，必须对所有 Q tile 计算，无法省略——最多在元素级 mask。

=== Partial tile（对角线穿过 block）

当 Q tile 与 K tile 索引区间相交但不完全包含时（block 跨对角线），不能整 tile skip，只对 $j > i$ 的元素置 $-infinity$ 或跳过 dot product 的 partial 累加。实现复杂度高，但 v2 Q-outer 至少能 skip *大量* 纯上三角 block——这是 GPT 类模型 prefill 的重要加速来源。

== 非 matmul FLOP 为什么重要

A100 FP16 tensor core：$approx 312$ TFLOPS。CUDA core FP32 FMA：$approx 19.5$ TFLOPS。FDIV 更慢。

=== 为什么 tensor core 时代 non-matmul 更「贵」

Roofline 的 ridge point 随 tensor core 峰值右移——matmul 有效算力涨 10×+，但 exp/div/add 仍走 legacy pipeline。Attention 的 fused kernel 里：

- *Matmul*（$Q K^T$, $P V$）：$approx 90%$ FLOPs，*应该*占 wall time 的 70%+
- *Softmax*（exp, max, sum）：$approx 5%$ FLOPs，但 latency 与 matmul *串行*（v1/v2 无 pipeline 时）
- *Rescale*（FMA, FDIV）：v1 占 $O("tiles" times d)$ CUDA core ops

当 matmul 从 20 TFLOPS（CUDA core GEMM）涨到 300 TFLOPS（TC）时，*同样次数的 exp/div 从「可忽略」变成「瓶颈」*——这就是 v2 在 v1 IO 最优之后仍要砍 FDIV 的硬件原因。Hopper v3 进一步用 async pipeline 让 softmax 与 matmul *重叠*，但 (a) 延迟归一化仍是 v2/v3 共有基础。

FlashAttention 的主要 FLOP 是两个 matmul（$Q K^T$ 和 $P V$），占 $approx 90%+$ 总 FLOP——*应该*走 tensor core。但 rescale、softmax、mask 走 CUDA core。v1 每 tile 的 $d$ 次 FDIV 让 CUDA core 成为 bottleneck；profiler 里看到 `mufu`/`fdiv` 占比高、tensor core 利用率低。

v2 改进 (a) 把 FDIV 从 $O("tiles" times d)$ 降到 $O(d)$（每个 Q row 一次）。改进 (c) 让 matmul warp 不被 softmax warp 的 barrier 拖住。两者叠加，论文报告 A100 上 FA-2 达 50–73% tensor core 理论峰值（取决于 shape / dtype）。

=== v1 vs v2 总览

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*维度*], [*FA v1（生产）*], [*FA v2*],
    [Outer loop], [K/V tile], [Q block],
    [Grid 并行轴], [$N_k$ / batch / head], [$N_q$ / batch / head],
    [Output state], [每步归一化 $O$], [未归一化 $tilde(O)$，最后除 $l$],
    [Inner rescale FDIV], [每 tile × $d$], [0（仅 final）],
    [Warp 切分], [split-K（merge）], [split-Q（独立）],
    [Warp barrier], [每 tile reduce], [仅 load sync],
    [Causal tile skip], [困难], [结构化 skip],
    [A100 TC 利用率], [~25–40%], [~50–73%],
  ),
  caption: [*Table:* FA v1 vs v2 生产 kernel 总览。IO 渐近阶相同（$O(N^2 d^2/M)$）；v2 收益来自 SM 利用率、非 matmul FLOP 削减（延迟归一化）、causal tile skip。TC 利用率数字来自 FA-2 论文 A100 benchmark。],
  kind: table,
)

*Observation*：三列对比清晰划分「算法层」（IO 最优，v1 已有）与「执行层」（loop order、FDIV、warp 切分、causal skip——v2 三项改进）。面试画这张表 + 一个 $tilde(O)$ 数值例子，基本覆盖 FA-2 核心考点。

== Block size 与 occupancy（v2 视角）

v2 Q-outer 后，每个 CTA 驻留整个 Q block 的 state：$B_r times d$ floats 的 $tilde(O)$ + $B_r$ 的 $m, l$ + matmul smem（$Q$, $K$, $V$ tile）。典型 $B_r = 128$, $d = 128$, FP32 $tilde(O)$：$128 times 128 times 4 = 64 "KB"$——*仅 output state 就占满旧版 smem 预算的一半*。

选型 trade-off：

- *增大 $B_r$*：摊销 K/V load、提高 matmul 效率；但 smem/reg 涨 → occupancy 降
- *增大 $B_c$*：每个 K tile 更宽，Q-outer 下每个 Q CTA 仍只顺序扫 kv tile——$B_c$ 影响 inner 次数 $N_k / B_c$，不影响 grid 大小
- *split-Q warp 数 $W$*：$B_r$ 必须被 $W times "rows_per_warp"$ 整除；常见 128 = 4 × 32

面试答「$B_r, B_c$ 怎么选」：*先算 smem 公式 fit 48/96/164 KB → 查 occupancy → profile 2–3 组候选*。v2 比 v1 更吃 $B_r times d$ 的 reg/smem——head dim 从 64 涨到 128 时，v2 往往要减 $B_r$ 或 enable dynamic smem。

== Backward pass：v2 的 state 存什么

Forward v2 最后输出 $O = tilde(O) \/ l$，但 backward 需要 per-row 的 log-sum-exp 统计来 recompute $P$。FA v2 backward 仍存：

- $m_i$：final row max（$N_q$ floats per head）
- $l_i$ 或 $"LSE"_i = m_i + log(l_i)$（$N_q$ floats）

*不存* $tilde(O)$ 到 global——backward 在 smem 重算 $S = Q K^T$，用 $(m, l)$ 恢复 $P = exp(S-m)/l$，再链式求 `dQ, dK, dV`。存储仍是 $O(N)$ per head，不是 $O(N^2)$。

延迟归一化对 backward 的影响：forward 少做 inner FDIV，backward 的 rescale 公式*结构不变*——仍是对 `dO` 乘 `1/l` 等操作。v2 backward 同样受益于 Q-outer（并行轴在 $N_q$）和 split-Q。

#note[
  本书 backward 不展开实现——面试知道「存 $(m,l)$、recompute $S$、Q-outer 遍历顺序与 forward 同构」即可。完整 Algorithm 4 见 v1 论文，v2 论文 Appendix 有 backward 吞吐对比。
]

== 论文 benchmark 数字（建立量级感）

Dao et al. FA-2 论文 A100 40GB 典型结果（FP16/BF16，head dim 64–128）：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*场景*], [*FA v1*], [*FA v2*],
    [Training forward (2048 seq)], [~115 TFLOPS], [~225 TFLOPS],
    [Training forward (4096 seq)], [~120 TFLOPS], [~230 TFLOPS],
    [Inference causal (4096 seq)], [~110 TFLOPS], [~210 TFLOPS],
    [TC 理论峰值占比], [~35–40%], [~50–73%],
  ),
  caption: [*Table:* Dao et al. FA-2 论文 A100 40GB 典型吞吐（FP16/BF16，head dim 64–128，单位 TFLOPS）。端到端 attention forward 吞吐，非单 kernel microbenchmark；与 PyTorch `sdpa` 对比需对齐 dtype、causal、batch/seq shape。],
  kind: table,
)

*Observation*：v2 相对 v1 约 2× TFLOPS 来自三项改进叠加——sequence parallel 提高 occupancy（30–40%）、延迟归一化减 SFU/div（20–30%）、split-Q + causal skip（10–20%）。*方向*比精确 TFLOPS 更重要；本书 teaching kernel 测不到这个量级。

加速来源分解（面试答「2× 从哪来」）：~30–40% 来自 sequence parallel 提高 occupancy；~20–30% 来自延迟归一化减 SFU/div；~10–20% 来自 split-Q 减 barrier + causal tile skip（causal 场景）。数字随 shape、dtype、GPU 型号变化——*方向*比精确 TFLOPS 更重要。

#note[
  上表 TFLOPS 是论文端到端 attention forward 吞吐，不是单 kernel microbenchmark。与 PyTorch `sdpa` 或 xformers 对比时，需对齐 dtype、causal、batch/seq shape——否则数字不可比。
]

== 源码运行与对照

```bash
make build/09_flash_attention_v2 && ./build/09_flash_attention_v2
```

预期输出：`Check: PASS`，三行 row 0 sample（reference / GPU / warp GPU）一致。

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*函数*], [*Launch*], [*演示点*],
    [`flash_attention_v2_kernel`], [`<<<4, 4>>>`], [Q-outer + smem K/V + 延迟归一化],
    [`flash_attention_v2_warp_kernel`], [`<<<4, 32>>>`], [lane 角色分离 + 并行 accum],
  ),
  caption: [*Table:* 本章两个 GPU kernel 的 launch 配置与演示重点。grid = $N_q = 4$（Q-outer）；smem 版 block = 4 thread，warp 版 block = 32 thread（一 warp）。默认 $N_k=16$, $d=8$, tile size $= 4$。],
  kind: table,
)

*Observation*：两版 grid 相同（Q-outer），差别在 accum 语义（$tilde(O)$ 延迟归一化）与 warp lane 分工——与第 8 章 v1-shared 的 grid 布局一致，micro-benchmark 只能比算法骨架，不能验证 sequence parallelism。

== 实测

$N_q = 4$, $N_k = 16$, $d = 8$（源码 `kQueryCount` / `kKeyCount` / `kHeadDim`；$Q,K,V,O$ 各 512 B，整 problem $< 4$ KB），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。表中 TC % = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；warp % = `sm__warps_active.avg.pct_of_peak_sustained_elapsed`；HBM % = `dram__bytes.sum.pct_of_peak_sustained_elapsed`。

Launch 配置与第 8 章 v1-shared 同为 Q-outer（1 block / query 行），差别在 accum 语义与 warp 分工：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right, left),
    [*version*], [*grid*], [*block*], [*active threads*], [*演示点*],
    [v2], [(4, 1, 1)], [(4, 1, 1)], [16（4 block × 4）], [延迟归一化；thread 0 串行 merge],
    [v2-warp], [(4, 1, 1)], [(32, 1, 1)], [128（4 block × 32）], [lane 分组 load / score / accum],
  ),
  caption: [*Table:* ch9 FA v2 teaching kernel 的 launch 配置（`launch__grid_size` / `launch__block_size`）。与第 8 章 v1-shared 同为 Q-outer（1 block / query 行）；TC % = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`。],
  kind: table,
)

*Observation*：v2 blockDim = 4 凑不满 warp（`issued/32` = 1.6）；v2-warp 一 block 一 warp 但 softmax merge 仍仅 lane 0——*结构性 lane 浪费*，不是 branch divergence。Teaching 规模下 v2 相对 v1-shared 仅 ~2.4%，在 noise 带内。

#include "../bench/09_flash_attention_v2.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *v2 9.06 μs vs 第 8 章 v1-shared 9.28 μs——仅快约 2.4%，在 teaching 规模的 noise 带内。* 两版都是 1 block / Q row、scalar dot、thread 0 串行 softmax merge；v2 的延迟归一化（inner loop 无 FDIV）在本 shape 上*几乎测不出来*——$d = 8$、$N_k / B_c = 4$ 个 tile，省下的 FDIV 次数 $< 32$，淹没在 launch 与 global load 噪声里。*诚实结论：micro-benchmark 不能验证 v2 论文的 ~2×；只能确认算法骨架正确。*

+ *v2-warp 8.96 μs vs v2 9.06 μs——约 1.1% 加速。* warp 版把 score（lane 0–3 并行 dot）与 $d$ 维 accum 更新（lane 0–7 并行 FMA）从 thread 0 串行里拆出——split-Q 的*微型演示*，不是生产 split-Q（4 warp × 32 行 Q）的量级。

+ *TC % / warp % / HBM % 全表 0.0–0.1%*。TC % = 0：scalar FFMA，无 tensor pipe。warp % = 0.1%：4 个 block 填不满 108 SM。HBM % = 0.1%：$< 4$ KB 工作集 L2 resident——*测不到* v2 相对 naive 的 IO 优势，也测不到 sequence parallelism（本书 v1/v2 教学代码 grid 都是 $N_q = 4$）。

#figure(
  hbar-chart(
    (
      ("v2-warp", 8.96),
      ("v2", 9.06),
      ("v1-shared (ch.8)", 9.28),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)`：三版 teaching kernel 落在同一 9 μs 带内——v2 相对 v1-shared 仅 ~2.4%，v2-warp 相对 v2 仅 ~1.1%；*不是*论文 ~2× 的缩影。],
)

#figure(
  warp-grid(
    rows: 4, cols: 4,
    cell: 0.28,
    active: (
      (0, 0), (0, 1), (0, 2), (0, 3),
      (1, 0), (1, 1), (1, 2), (1, 3),
      (2, 0), (2, 1), (2, 2), (2, 3),
      (3, 0), (3, 1), (3, 2), (3, 3),
    ),
    row-labels: ("CTA K0", "CTA K1", "CTA K2", "CTA K3"),
    col-labels: ("Q0", "Q1", "Q2", "Q3"),
    title: "论文 v1（K/V-outer）：grid ≈ $T_c = 4$ 个 K tile CTA，内循环串行扫 Q",
  ),
  caption: [
    每行 = 一个 CTA 持固定 $K_j$ tile，内循环 load 各 $Q_i$ 并 merge $(m, l, O)$。
    并行度 $approx T_c = N_k / B_c$；decode 时 $N_k$ 小 → grid 小。
  ],
)

#figure(
  warp-grid(
    rows: 4, cols: 4,
    cell: 0.28,
    active: (
      (0, 0), (0, 1), (0, 2), (0, 3),
      (1, 0), (1, 1), (1, 2), (1, 3),
      (2, 0), (2, 1), (2, 2), (2, 3),
      (3, 0), (3, 1), (3, 2), (3, 3),
    ),
    row-labels: ("CTA Q0", "CTA Q1", "CTA Q2", "CTA Q3"),
    col-labels: ("K0", "K1", "K2", "K3"),
    title: "论文 v2（Q-outer）：grid ≈ $T_r = 4$ 个 Q block CTA，内循环串行扫 K/V",
  ),
  caption: [
    每行 = 一个 CTA 持固定 $Q_i$ block，$(m, l, tilde(O))$ 驻留寄存器/smem，内循环流式读 $K_j, V_j$。
    prefill 时 $N_q$ 大 → grid 大；*本书教学代码已用此布局*，与 v1-shared 的差别主要在 accum 语义与 warp 分工，不在 grid 维数。
  ],
)

*diag 表读关键教学点：*

*a) v2 `issued/32 = 1.6`，`pred_on/32 = 1.6`——极低，blockDim = 4 凑不满 warp*

#raw("<<<4, 4>>>") 每 block 仅 4 thread：协作 load $K/V$ tile 时 4 lane 写 smem；score + online merge 由 thread 0 *串行*执行——多数 phase 只有 1–4 lane 参与 issued 指令，平均 `issued/32` 仅 1.6。issued − pred_on $approx 0$：几乎没有 predicated-off lane——*不是* branch divergence，而是*结构性 lane 浪费*（block 太小 + thread-0 串行）。

#figure(
  warp-lanes(active: range(4), cell: 0.34,
             title: [v2 smem：#raw("<<<4, 4>>>")，仅 lane 0–3 活跃；merge 阶段常仅 lane 0]),
  caption: [绿色 = 4 个协作 thread。灰色 = warp 内无对应工作的 lane——*配置问题*，不是 online merge 里的 branch divergence。],
)

*b) v2-warp `issued/32 = 6.6`，`pred_on/32 = 6.2`——仍远低于 32，但高于 v2*

#raw("<<<4, 32>>>") 每 block 一 warp：lane 0–7 load $Q$、lane 0–3 并行 dot、lane 0–7 并行 accum 更新——并行 phase 拉高 `issued/32` 到 6.6。issued − pred_on $approx 0.4$ 来自 #raw("if (lane < kTileKeys)") 等 guard 的 predicated-off lane——*predication*，不是不同 basic block 的 warp divergence。

#figure(
  warp-lanes(active: range(8), cell: 0.34,
             title: [v2-warp accum 阶段：lane 0–7 各更新 $tilde(O)$ 一维；softmax merge 仍仅 lane 0]),
  caption: [split-Q 微型演示——并行度集中在 $d$ 维 FMA；score/softmax 瓶颈仍在 lane 0，故 `issued/32` 远低于 32。],
)

*c) `smem conf. = 0`；`barrier stall`：v2 = 0.11，v2-warp = 0.31。* metric 证实*无 bank conflict*（不能从 `shared_k[token][d]` 布局单独推断）。v2-warp 的 `__syncthreads` 更多（lane 分组 + broadcast `shared_weights`）→ barrier stall 略升——*定性方向对*，绝对值仍很小。

*d) `mem stall`：v2 = 1.40，v2-warp = 2.19。* v2-warp 的 `q_shared` 预加载略增 smem 流量，long_scoreboard 略升——kernel $< 10$ μs，*不能*用来断言 memory-bound。

*regs / smem fingerprint：* v2：47 regs，272 B smem（`shared_k/v` + thread 0 串行 state）。v2-warp：32 regs，340 B smem（额外 `q_shared`、`shared_weights`；lane 并行使 per-thread reg 压力降、smem 略增）。

*无信息或为零的 metric：*

- `TC %`：全表 0.0——scalar FFMA；生产 FA-2 的 $Q K^T$ / $P V$ 才走 tensor pipe。
- `HBM %`：全表 0.1%——L2 resident；*不能*验证 IO 渐近阶差异。
- `warp %`：0.1%——4 block 填不满 GPU。

#insight[
  v2 相对 v1-shared 的 ~2.4% 差距*在 noise 带内*——延迟归一化、Q-outer、split-Q 三项改进在本规模上*都*测不出论文量级收益。它们的价值是*数据流正确*（`accum` = $tilde(O)$、inner 无除法、lane 角色分离），放大到 $N >= 4096, d >= 64$ + tensor core 后才体现为 TC 利用率与 CTA 并行度。
]

#insight[
  v2-warp 的 `issued/32`（6.6）*高于* v2（1.6）但 time 仅快 1.1%——*lane 利用率高不等于 kernel 快*。本规模瓶颈是 launch + 极短 kernel 的固定开销，不是 SM issue 带宽；生产 FA-2 要在 smem 复用、多 warp tile matmul、sequence parallel 之间同时拉满 `issued/32` 和 TC %。
]

#warn[
  面试别把「v2 论文 2×」映射到「v2 kernel 比 v1-shared 快 2×」。本书 v1/v2 教学代码*都已* Q-outer（1 block / Q row）——grid 并行度相同，micro-benchmark 只能比 accum 语义与 warp 分工。也不要把低 `issued/32` 说成「严重 warp divergence」——用 `issued/32` vs `pred_on/32` 区分 predication 与真 divergence（不同 lane 走不同 basic block）；本规模首先是 blockDim $< 32$ 与 thread-0 串行的*结构性 lane 浪费*。
]

== ncu 该看什么

```bash
ncu --set full --section SpeedOfLight ./build/09_flash_attention_v2
```

- `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`：tensor core 利用率（教学 kernel 接近 0%，预期如此）。
- `smsp__sass_thread_inst_executed_op_ddiv_pred_on.sum`：FDIV 次数——对比 v1 风格 inner 除法应显著下降。
- `sm__warps_active.avg.pct_of_peak_sustained_active`：split-Q 后 warp 独立度更高，active warps 应更均匀。
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`：K/V smem broadcast 的 bank conflict——生产 kernel 用 swizzle 消除。

教学 kernel 规模太小（$N_q=4$），ncu 只能验证*方法论*。若要看到 v2 特征，需把 shape 放大到 $N=1024, d=64$ 并换官方 `flash-attn` 库对比——期望 FA-2 相对 FA-1：`smsp__sass_thread_inst_executed_op_ddiv` 降一个数量级；`sm__maximum_warps_per_scheduler` 利用率升；`gpu__compute_memory_throughput` 中 compute 占比升（matmul 终于吃饱）。

=== 常见实现错误

*错误 1*：延迟归一化写错成 `accum += weight * V` 但忘记 `accum *= old_scale`——输出偏大，与 reference 在第二个 tile 起 diverge。

*错误 2*：Q-outer 但每个 kv tile 把 $O$ 写回 HBM 再读——退化成 naive 三 pass 的 IO，只是少了 $S$ matrix。v2 要求 $tilde(O)$ *全程 resident* 到 inner loop 结束。

*错误 3*：split-Q 时在 warp 间共享 $(m,l,tilde(O))$——破坏独立性，又回到 split-K 的 barrier 地狱。

*错误 4*：causal skip 条件写反（`kv_tile < q_tile` vs `kv_tile > q_tile`）——silent wrong answer，比 crash 更危险。

*错误 5*：最后归一化用 `running_max` 而非 `running_sum` 做除数——混淆 $m$ 与 $l$ 的角色。

== Multi-head 与 batch：并行轴叠加

单 head 算法清楚后，生产 launch 只是在 grid 上加维：

```cpp
// 逻辑 index
const int batch = blockIdx.z;
const int head  = blockIdx.y;
const int q_tile = blockIdx.x;
const int q_start = q_tile * kBlockM;

const float* q_ptr = q + batch * (H * N_q * d) + head * (N_q * d);
// K/V 同理；每个 (batch, head, q_tile) CTA 完全独立
```

总 CTA 数：

$ "grid" = ceil(N_q / B_r) times H times B $

A100 108 SM，每 SM 可同时驻留若干 block——$8192$ CTA 的 prefill 足够填满。Decode 时 $ceil(N_q/B_r)=1$，grid 只有 $H times B$（例如 $32 times 8 = 256$），此时 *occupancy 靠 head×batch*，不够就 merge batch 维或用小 batch inference。

#note[
  GQA/MQA（Grouped Query Attention）下 $K,V$ 的 head 数 $H_"kv" < H$，K/V 读取量减少，但 Q-outer 结构不变——只是 `head` index 映射到不同的 KV head。FlashAttention-2 官方实现对此有专门 stride 处理。
]

== 与官方 flash-attn 库的关系

PyTorch `flash_attn` 包（Dao 维护）是 FA-2 的生产实现，比本书教学 kernel 多：

- CUTLASS/CUTE 生成的 tensor core MMA 主循环
- 针对 SM80/SM86/SM89 的 tuned tile 参数表
- 变长序列（cu_seqlens）、dropout、alibi、softcap 等 epilogue
- Backward kernel（同样 Q-outer + recomputation）

本书 `09_flash_attention_v2.cu` 是*算法骨架*——面试讲清 v1→v2 三项改进后，可补一句「官方库在此基础上加 tensor core tiling 和 tuned block size」。读 CUTLASS 前先能在白板上画 split-Q 和 $tilde(O)$ 更新，比背 API 重要。

== 铺垫：v3 改了什么

FlashAttention-3（Dao et al., 2024，Hopper/H100）在 v2 执行计划之上进一步：

- *异步 pipeline*：`cp.async` / TMA 双缓冲 K/V tile，compute 与 load 重叠。
- *Warp-group matrix multiply*（WGMMA）：Hopper 新指令，比 Ampere `mma.sync` 更高吞吐。
- *FP8 / FP16 混合精度*：$S$ tile FP8 算、accum FP32。
- *Interleaving block scheduling*：SM 间动态分配 Q block，减少 tail effect。

=== v2 → v3 面试一句话

*v2*：「我把 attention 的 loop 顺序、accumulator 和 warp 切分改对了，Ampere 上 TC 利用率从 35% 拉到 70%。」

*v3*：「公式不变，我在 Hopper 上让 TMA 搬货、WGMMA 算 matmul、SFU 算 softmax *同时进行*，并上了 FP8。」

v2 → v3 的 shift：*少改公式，多改 pipeline*——第 10 章展开。

== 本章小结

FlashAttention v2 在 v1 的 IO 最优算法之上，用三个*正交*改进把 SM 利用率做上去：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [(a) 延迟归一化], [inner loop 去掉 $d$ 次 FDIV；维护 $tilde(O)$ 而非 $O$],
    [(b) Q-outer], [sequence parallelism；CTA 数约 $N_q$；causal tile skip],
    [(c) split-Q], [warp 独立 $(m,l,tilde(O))$；无 cross-warp softmax reduce],
  ),
  caption: [*Table:* FlashAttention v2 三项正交改进摘要。(a) 减 CUDA core FDIV；(b) 增 CTA 并行度；(c) 减 warp 间 barrier。三者缺一不可：(b) 让 (c) 有意义；(a) 让 (b) 的 smem/reg 压力可控。],
  kind: table,
)

*Observation*：面试最高频的一句话——*算法没变，变的是 loop order、accumulator 表示、和 warp 切分轴*。画 v1/v2 伪代码 diff + 一个 $tilde(O)$ 数值例子，基本覆盖 FA-2 核心考点。

三者缺一不可：(b) 让 (c) 有意义；(a) 让 (b) 的 smem/reg 压力可控。面试画一张 v1/v2 伪代码 diff + 一个 $tilde(O)$ 数值例子，基本覆盖 FA-2 的核心考点。

== 从 v1 教学代码到 v2 生产代码的映射

本书两个 FA 教学文件的关系：

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*概念*], [`08_flash_attention_v1.cu`], [`09_flash_attention_v2.cu`],
    [Grid 映射], [1 thread 或 1 block / Q row], [1 block / Q row（已是 Q-outer）],
    [Outer loop], [K/V tile（thread 内）], [K/V tile（block 内）],
    [Accum 语义], [每步后 `accum` ≈ 未显式命名的 $l dot O$], [显式 $tilde(O)$，最后除 $l$],
    [并行], [thread 0 串行], [warp lane 分组],
    [对应论文], [Algorithm 1 数值核心], [Algorithm 1 + v2 三项改进],
  ),
  caption: [*Table:* 本书 `08_flash_attention_v1.cu` 与 `09_flash_attention_v2.cu` 的概念映射。两文件 grid 布局都已 Q-outer（1 block / Q row）；v2 数值改进在 accum 语义与 warp 分工，*不是* grid 维数变化。],
  kind: table,
)

*Observation*：本书 v1 教学代码*已经* Q-outer——读 v2 时*不要*被教学代码误导：论文 v1 生产 = K/V outer + split-K；论文 v2 = Q outer + split-Q + 延迟归一化。`09` 对齐论文 v2 布局，主要差 accum 更新与 warp 分工。

注意：本书 v1 教学代码*已经*是 Q-outer（1 block 一行 Q）——因为它优先教 online softmax，而不是 v1 论文的 K/V-outer 生产布局。读 v2 章时*不要*被教学代码误导：论文 v1 生产 = K/V outer + split-K；论文 v2 = Q outer + split-Q + 延迟归一化。`09_flash_attention_v2.cu` 的 grid 布局对齐*论文 v2*，数值改进主要体现在 accum 更新与 warp 分工。

=== 读 `09_flash_attention_v2.cu` 的检查清单

跑通 `./build/09_flash_attention_v2` 后，打开源码逐项确认：

1. `blockIdx.x == row`：Q-outer，grid = `kQueryCount`
2. `for (key_start = 0; ...)`：K/V inner loop
3. `accum[d] *= old_scale` 在 inner loop，`/ running_sum` 只在循环外——延迟归一化
4. `flash_attention_v2_warp_kernel` 里 `lane < kHeadDim` 并行更新 accum——split-Q 雏形
5. `main` 里两个 kernel 输出都与 CPU reference 对拍——算法正确优先于性能

#insight[
  若只能记住 v2 一件事：*CTA 的生命周期 = 一个 Q block 从 init $(m,l,tilde(O))$ 到扫完所有 K/V tile 再写 $O$*。所有三项改进都服务这一生命周期——减少生命周期内的 non-matmul ops、增加生命周期 CTA 个数、减少生命周期内 warp 同步。
]

=== Prefill vs Decode 工程选型

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*阶段*], [$N_q$], [v2 主要收益来源],
    [Prefill], [$N$（整段 prompt）], [sequence parallel + causal skip],
    [Decode 单 token], [$1$], [延迟归一化 + split-Q（单 CTA 内）],
    [Decode + KV cache], [$1$, $N_k$ 增长], [inner kv loop 变长；(a) 省 FDIV 更明显],
    [Cross-attn], [$N_q != N_k$], [grid 随 $N_q$；encoder K 只读一次 per CTA],
  ),
  caption: [*Table:* LLM 推理各阶段下 FA v2 的主要收益来源。Prefill 时 $N_q$ 大 → sequence parallel + causal tile skip；Decode 时 $N_q=1$ → 靠 (a)(c) 优化单 CTA 效率，工业界 decode 常配合 paged KV 专用 kernel。],
  kind: table,
)

*Observation*：FA-2 不是 decode 唯一答案——prefill 几乎是标配，decode 阶段 vLLM/TensorRT-LLM 可能切换专用 kernel。面试应区分 prefill（v2 主战场）与 decode（(a) 延迟归一化随 KV cache 变长更明显）。

LLM 推理框架（vLLM、TensorRT-LLM）在 prefill 阶段调用 FA-2 varlen kernel；decode 阶段可能切换到 paged KV + 专用 decode kernel——FA-2 不是 decode 唯一答案，但 prefill 几乎是标配。

=== 自测：能否在 5 分钟内讲清 v2

闭卷回答下面 5 问，全部流畅即达标：

1. 写出 v1 与 v2 的 double-loop 伪代码，标出 grid 绑在哪一维。
2. 写出 v1 的 $O_t$ 更新式与 v2 的 $tilde(O)_t$ 更新式，说明为何等价。
3. 画 split-K 与 split-Q 的 ASCII 图，指出哪条需要 `__syncthreads` reduce。
4. 给定 causal self-attention，Q block 63 能 skip 多少比例的 K tile？
5. v3 相对 v2 改的是公式还是 pipeline？

参考答案要点：Q1 grid 绑 Q vs K；Q2 $tilde(O)$ 无 inner 除；Q3 split-K 要 reduce、split-Q 不要；Q4 平均 skip 50%；Q5 pipeline（TMA/WGMMA/FP8）。

=== 延伸阅读顺序

1. 第 3 章 online softmax merge 推导
2. 第 8 章 FA v1 IO 分析与 $tilde(O)$ 引入
3. 本章 v2 三项改进
4. 官方论文 FA-2 Appendix（backward、benchmark 细节）
5. 第 10 章 FA-3 Hopper pipeline
6. `dao-ailab/flash-attention` 源码 `csrc/flash_attn/src/flash_fwd_kernel.h`（可选，生产级）

== 面试考点

#interview[
  *Q1*: FlashAttention v2 相比 v1 的三个主要改进是什么？

  A: (a) 延迟归一化——维护 $tilde(O)$ 和 $l$，inner loop 不做 $\/l$，最后一步才除，砍掉每 tile $d$ 次 FDIV；(b) 循环调换——Q-outer / K/V-inner，每个 Q block 一个 CTA，引入 sequence parallelism；(c) warp 分工从 split-K 改 split-Q，每 warp 独立算 Q slice，warp 间无 reduction barrier。
]

#interview[
  *Q2*: 延迟归一化的公式对比？为什么数学等价？

  A: v1 每步 $O_t = (l_"old" e^(m_"old"-m_t) O_(t-1) + tilde(P)V_t) \/ l_t$。v2 维护 $tilde(O)_t = l_t O_t$，更新 $tilde(O)_t = e^(m_"old"-m_t) tilde(O)_(t-1) + tilde(P)V_t$，最后 $O = tilde(O)\/l$。等价因为分子分母同步缩放，$tilde(O)\/l$ 始终是正确的 weighted sum of $V$。
]

#interview[
  *Q3*: 非 matmul FLOP 为什么重要？

  A: Attention 主体 matmul 应走 tensor core（$approx 300+$ TFLOPS），但 rescale/softmax/mask 走 CUDA core（$approx 20$ TFLOPS）。v1 每 tile 对 $d$ 维向量做 FDIV，CUDA core 成为瓶颈。Tensor core 越快，非 matmul ops 的相对代价越大——v2 专门优化这部分。
]

#interview[
  *Q4*: v2 的并行度公式？什么时候 v2 比 v1 并行度好？

  A: v1 grid $approx ceil(N_k\/B_c) times B times H$；v2 grid $approx ceil(N_q\/B_r) times B times H$。Prefill（$N_q$ 大）时 v2 CTA 数远超 v1；Decode（$N_q=1$）时 v2 无 sequence parallel 优势，靠延迟归一化和 split-Q 提速。
]

#interview[
  *Q5*: v1 为什么用 Q-inner？v2 怎么解决 $O$ 的 SRAM residency？

  A: v1 K/V-outer 时 $O$ 跨 kv_tile 累积，Q-inner 让 $O$ 留在 CTA 寄存器/smem 避免 HBM reload。v2 Q-outer 时每个 CTA 只服务一个 Q block，$tilde(O)$ 大小 $B_r times d$ 可放进 smem/reg，走完所有 kv_tile 才写回——不需要 reload，故可安全外循环 Q。
]

#interview[
  *Q6*: split-K vs split-Q 图示区别？v2 warp 间需要 barrier 吗？

  A: split-K：4 warp 各算 K 的一部分 score，smem merge max/sum/$O$，需要 barrier。split-Q：4 warp 各算 Q 的 1/4 行，独立 $(m,l,tilde(O))$，算完直接写，*无 warp 间 reduction barrier*。K/V load 到 smem 后 broadcast 给各 warp 只需 producer sync，不是 split-K 的 merge。
]

#interview[
  *Q7*: v2 causal mask 的 tile skip 为什么 v1 做不到？量化收益？

  A: v2 Q-outer 每个 CTA 知 Q 行范围，若整个 K tile 在上三角（$j_0 > i_"end"$）直接 skip。v1 K/V-outer 同一 kv_tile 要服务所有 Q 行，无法 tile 级 skip。平均 skip $approx 50%$ K/V tile；序列末尾 Q block skip 可达 $90%+$。
]

#interview[
  *Q8*: v3 相对 v2 改了什么？

  A: 公式不变，pipeline 升级——Hopper `cp.async`/TMA 双缓冲、`wgmma` warp-group matmul、FP8 计算、interleaved block scheduling。v2 优化 execution plan；v3 优化 memory pipeline 和新一代 tensor core 利用率。详见第 10 章。
]

#interview[
  *Q9*: 为什么 v2 教学 kernel 不用 tensor core 仍有学习价值？

  A: 性能不是目标，*数据流*才是——Q-outer grid、`accum`/`running_sum` 对应 $tilde(O)$/ $l$、warp lane 角色、causal skip 条件，与生产 kernel 一一对应。先理解再读 CUTLASS 生成代码，否则只能背 API。
]

#interview[
  *Q10*: Decode（$N_q=1$）时 FA-2 还有优势吗？

  A: sequence parallel 优势消失（grid 只有 $H times B$ 个 Q CTA），但 (a) 延迟归一化、(c) split-Q 仍减少单 CTA 内 FDIV 和 warp barrier；更长 KV cache 时 inner kv loop 变长，(a) 节省更明显。极致 decode 需 FlashDecoding 等专门切 KV 维的 kernel。
]

#interview[
  *Q11*: v2 改变了 IO 复杂度吗？和 v1 的 HBM 访问量差多少？

  A: 渐近阶相同，都是 $O(N^2 d^2/M)$ 相对 naive 的 IO 优化。v2 不减少 HBM 字节数，减少的是*完成相同 IO pattern 时的 wall time*——靠更高 SM 利用率、更少 CUDA core ops、causal 场景下 tile skip 减实际 matmul 次数。
]

#interview[
  *Q12*: 若只有 128 个 Q row 但 $N_k = 65536$，v2 并行度够吗？

  A: grid $approx ceil(128/B_r) times B times H$ 可能只有几百 CTA——prefill 并行不足。inner loop 有 $65536/B_c$ 次 kv 迭代，单 CTA 很重；需增大 batch/head 并行，或用 FlashDecoding 切 KV 维。FA-2 不是万能，shape 决定切哪条轴。
]

#interview[
  *Q13*: FlashAttention v2 论文标题强调 "Better Parallelism and Work Partitioning"——三个改进里哪个算 parallelism、哪个算 work partitioning？

  A: (b) sequence parallelism 是 parallel 维度的扩展；(a) 延迟归一化 + (c) split-Q 是 work partitioning（单 CTA 内指令与 warp 分工）。三者正交：parallelism 填 SM，partitioning 让每个 SM 算得更快。
]

#interview[
  *Q14*: 写 v2 kernel 时先实现哪一项改进收益最大？

  A: 通常 (a) 延迟归一化——改动小（只改 accum 更新和 final epilogue）、收益确定（减 FDIV）。其次 (b) Q-outer 若当前是 v1 K/V-outer 布局。 (c) split-Q 依赖 (b) 且实现复杂，放最后。教学路径与 `09_flash_attention_v2.cu` ladder 一致：先 smem staging 正确，再加 warp 分工。
]

#interview[
  *Q15*: $1/sqrt(d)$ scale 在 FA v2 里放哪？

  A: 融进 $Q K^T$ matmul——对 $Q$ 预乘 $1/sqrt(d)$ 或在 dot product 累加时乘。不改变 online merge 结构；教学源码 `09` 省略 scale 是为简化对拍，与第 8 章 `08` 一致。
]

#interview[
  *Q16*: backward 需要存 $S$ 矩阵吗？v2 forward 的延迟归一化影响 backward 吗？

  A: 不存 $S$——backward recomputation，只存 per-row $(m,l)$ 共 $O(N)$。延迟归一化只改 forward 累加路径；backward 仍用 $P=exp(S-m)/l$ 重算，对 `dO` 的 rescale 仍含 $1/l$，结构不变。
]

#interview[
  *Q17*: 为何 v2 论文称 "2× faster" 但 IO 与 v1 相同？

  A: 2× 是*端到端 wall time*（含 SM 利用率、非 matmul ops、causal skip），不是 HBM 字节数减半。IO 在 v1 已最优；v2 让*同样字节数*算得更快、同时开更多 CTA——是 compute/parallel 优化，不是第二轮 IO 优化。
]

#interview[
  *Q18*: 生产 FA-2 用 BF16 算、FP32 累加——与延迟归一化有关吗？

  A: 有关。$tilde(O)$ 在 FP32 累加防溢出/精度损失；$m,l$ 也常用 FP32。最后 `out = tilde(O)/l` 再 cast 到 BF16——延迟归一化让 FP32 accum 全程不除，只在 epilogue 除一次，数值更稳。混合精度是 v2 生产实现的标配，并非 Hopper 架构 v3 独有特性。这一点在面试中须注意，常被误判为 v3 专有。
]
