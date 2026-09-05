#import "../template.typ": *

= 面试数学：从模型配置到 step time 的完整推导

面试常见的一类问题：*"给你 70B 模型、8k 序列、8 张 H100、TP=4 DP=2，估算一次 step 多长？显存够用吗？"* 这类问题不需要你把公式背下来，但需要你能*当场推导*。本章把推导过程拆成三个 estimator——参数/显存、计算量、通信量/时间，然后组合出 step-time 模型。每个 estimator 都给出 Python 参考实现（见 `src/distributed_training/`）与一到两个 worked example。

面试时不用完全按公式讲，*先分层*：`step = compute + comm + bubble + overhead`；再逐项估算；最后回归判断瓶颈（compute-bound / bandwidth-bound / memory-bound）。这套流程本身就是加分项，比记住 6.6 × 参数 × token 数这个公式更有效。

== 记号约定

#let sym(a, b) = ($#a$, [#b])

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 6pt,
  [*符号*], [*含义*],
  [$L$], [层数（transformer block 数）],
  [$H$], [hidden size (model dim)],
  [$A$], [attention heads],
  [$d_h = H / A$], [head dim],
  [$I$], [FFN inter size（GLU 类记 $I_"eff" = 2/3 · I_"expand"$）],
  [$V$], [词表大小],
  [$S$], [序列长度],
  [$B$], [global batch size (tokens = $B · S$)],
  [$m$], [micro batches per PP step],
  [$P, T, D, C, E$], [PP / TP / DP / CP / EP 度],
  [$W = P · T · D · C · E$], [总 GPU 数],
  [$b$], [每卡 bytes per element (BF16 = 2)],
  [$B W_"link"$], [单向链路带宽 (GB/s), NVLink ≈ 400, IB 400G ≈ 50],
)

== Estimator 1：参数量、显存

=== 参数量分解（不含 embedding）

单个 transformer block 的参数：
$
p_"attn" &= 4 H^2 quad ("Q, K, V, O each " H times H) \
p_"ffn"  &= 2 H I quad ("classic MLP, no GLU") \
p_"ffn-glu" &= 3 H I_"eff" quad ("SwiGLU: w1, w2, w3") \
p_"block" &= p_"attn" + p_"ffn" + O(H) quad ("LN weight+bias 可忽略")
$

嵌入 + LM head：$p_"emb" = 2 V H$（tied 时算一份）。总参数：$P_"tot" = L · p_"block" + 2 V H$。

#insight[
  常用近似：$P_"tot" ≈ L(4 H^2 + 3 H I) ≈ 12 L H^2$（若 $I = 4H$，且忽略 embedding），或 $≈ 12 L H^2 · (I / (4H))$ 更精细。这就是"model 12 系数"的由来。
]

=== 显存四大项

一次前向+反向+optimizer step 需要驻留的 GPU 显存：

$
"mem" = "params" + "grads" + "opt-state" + "activations" + "attn-workspace" + "overhead"
$

其中：
- `params`：$P · b$（BF16 → 2 bytes/元素）
- `grads`：$P · b$（同上；FP32 grad 累加则 $4P$）
- `opt-state` (Adam)：master weight $4P$ + m/v 各 $4P$ = *12 P* bytes（FP32）
- `activations`：随 $B, S, L$ 线性变化，见下

$"bytes"_"opt-total (BF16 + AdamW)" = 2P + 2P + 12P = 16 P thin "bytes"$

#formula[
  经验数字：$16 · P$ bytes ≈ *16 GB / B 参数*（FP32 opt）。7 B 模型只算 opt-related 就 ~112 GB。
]

#interview[
  面试题：*"为什么 7B 模型用 A100-80G 训练依然吃紧？"* 答：$16 P = 112 "GB"$ 已经超单卡；即使 ZeRO-1 把 opt-state 8 卡切分 (112/8 ≈ 14 GB)，加上 activation 与 attention workspace 仍需 40-50 GB。
]

=== Activation memory 的两种口径

大部分 activation 是 $O(B S H L)$ 级。最坏情况（不 recompute）：
$
A_"full" ≈ 34 · B S H L · b quad ("Megatron paper 公式")
$

`selective activation checkpointing`（只 recompute FFN 和 attention 里的大块）：
$
A_"selective" ≈ 12 · B S H L · b
$

`full checkpointing`：$A_"ckpt" ≈ 2 · B S H L · b$（只存每层输入）。

#insight[
  面试记 3 个魔法数：*34 / 12 / 2*。它们对应 Megatron paper 的 "no ckpt / selective / full"。SP 再除以 TP。
]

=== Attention workspace（FlashAttention 之外）

朴素 attention scores tensor：$O(B A S^2)$，是 $S$ 的平方。$S = 8k, B = 1, A = 32$ → $32 · 8192^2 · 2 = 4.3 "GB"$，且没算前反向都要 2 份。FlashAttention 用 tiling 把它降到 $O(S)$——这就是长序列必上 FA 的原因。

#raw(block: true, "```
# src/distributed_training/estimators.py 里的 mem_estimate()
def mem_estimate(cfg, tp=1, cp=1, dp=1, zero_stage=0,
                 ckpt='selective', bytes_per_elem=2):
    L, H, I, A, V, S, B = (cfg.L, cfg.H, cfg.I, cfg.A,
                            cfg.V, cfg.S, cfg.B_per_dp)
    P = L * (4*H*H + 3*H*I) + 2*V*H
    # ZeRO/TP shards
    p_shard = P / tp
    if zero_stage >= 1: opt = 12 * P / (tp * dp)
    else:               opt = 12 * p_shard
    if zero_stage >= 2: g = 2 * P / (tp * dp)
    else:               g = 2 * p_shard
    if zero_stage >= 3: params = 2 * P / (tp * dp)
    else:               params = 2 * p_shard
    factor = {'none': 34, 'selective': 12, 'full': 2}[ckpt]
    act = factor * B * S * H * L * bytes_per_elem / (tp * cp)
    return dict(params=params/1e9, grads=g/1e9, opt=opt/1e9,
                act=act/1e9)
```")

=== Worked example：7B / 8k / 8×A100 / TP=1 DP=8 ZeRO-2

- $L = 32, H = 4096, I = 11008, V = 32000$。$P ≈ 6.7 · 10^9$。
- opt (ZeRO-2, 8 卡)：$16 · 6.7 "e"9 / 8 ≈ 13.4 "GB"$
- params BF16：$2 · 6.7 "e"9 = 13.4 "GB"$（不切）
- grad shard：$13.4 / 8 = 1.7 "GB"$
- activation（selective, $B = 1$）：$12 · 1 · 8192 · 4096 · 32 · 2 / "e"9 ≈ 25.7 "GB"$

合计 ≈ 54 GB per GPU → A100-80G 尚有余裕。用 FSDP full-shard (ZeRO-3) 可再把 params 也切成 13.4/8 = 1.7 GB，activation 反而成主项。

== Estimator 2：FLOPs 和 step time (compute 部分)

=== 前向 FLOPs / token

一个 token 通过一个 transformer block 的 FLOPs：
$
"F"_"attn"^"per-tok" &= 2 · 4 H^2 + 2 · 2 S H quad ("QKV proj + attn matmul") \
"F"_"ffn"^"per-tok"  &= 2 · 2 H I quad ("2 大矩阵乘")
$

近似（忽略 attn 里的 $S H$ 项，长序列时不能忽略）：
$
"F"_"block"^"per-tok" ≈ 8 H^2 + 4 H I ≈ 24 H^2 quad ("if " I = 4H)
$

前向总量：$"F"_"fwd" ≈ P_"tot" · 2$（每个参数 1 次 mul + 1 次 add）。

反向约 2×前向。加上前向：$"F"_"train" ≈ 6 P · N_"tok"$。若使用 activation recompute（一次额外前向），则：
$
"F"_"train"^"recomp" ≈ 6 P · N_"tok" + 2 P · N_"tok" = 8 P · N_"tok"
$

#formula[
  面试万能公式：
  $"FLOPs/step" ≈ 6 · P · B · S quad ("no recompute")$
  $"FLOPs/step" ≈ 8 · P · B · S quad ("full recompute")$
]

（Attention 的 $O(B S^2)$ 项在长上下文时也不可忽略：额外 $≈ 4 · A · S^2 · d_h · B · L$；$S = 32k$ 时可占总量 30%。）

=== compute 时间 = FLOPs / (峰值 × MFU)

H100 BF16 peak = 989 TFLOPS（不算稀疏）。实测 MFU 40-55% 是好数。A100-80G peak = 312 TFLOPS。

$
t_"compute" = "FLOPs" / ("peak" · "MFU" · W)
$

$W$ 是所有参与 compute 的 GPU；TP、CP 会*分摊*一个 token 的计算量，DP、PP 不分摊 per-token（DP 分摊 batch）。

=== Worked example：70B / 8k / 8×H100 / TP=4 DP=2 no recompute

- $P = 70 · 10^9$，1 step = 每 DP shard 处理 $B$ tokens。设 global batch = $2 M$ tokens，DP=2 → per-DP $= 1 M$ tokens。
- FLOPs = $6 · 70 · 10^9 · 1 · 10^6 = 4.2 · 10^17$ per step
- 每个 GPU 的 compute（TP=4 分摊，DP=2 不分摊 per-token）：$4.2 · 10^17 / (4 · 2) = 5.25 · 10^16$
- 时间 = $5.25 · 10^16 / (989 · 10^12 · 0.45) = 118 "s"$ ← 单卡视角
- 但 8 卡同时算，实际 wall-clock 就是 $118 "s"$（因为 118s 已经是 per-GPU 的 wall time）

反算合理性：8 卡 · 989 TFLOPS · 0.45 = 3560 TFLOPS aggregate；$4.2 · 10^17 / 3.56 · 10^15 = 118 "s"$ ✓

#insight[
  快算捷径：*7 B 参数、1 M tokens、1 张 H100、MFU 50%* ≈ $6 · 7 · 10^9 · 10^6 / (989 · 10^12 · 0.5) = 84.9 "s"$。这是"$P$ 亿参数每 $10^6$ tokens 约 12 秒/卡"的口径。
]

== Estimator 3：通信量 & 通信时间

=== Ring AllReduce 的 per-GPU 通信量

对一个大小为 $V$ 的向量：
$
"vol"_"AR" = 2 · (W-1)/W · V ≈ 2 V quad "(W ≫ 1)"
$

*每方向* 各 $(W-1)/W · V$（reduce-scatter + all-gather 两阶段）。同理：
$
"vol"_"RS" = "vol"_"AG" = (W-1)/W · V ≈ V
$

#formula[
  记忆卡片：AR = 2V, AG = V, RS = V, A2A = V (per rank, single direction)
]

=== 各并行策略的通信量表（每 block、每 direction）

#cost-table(
  header: ([策略], [每层通信量 (bytes)], [同步次数], [备注]),
  ([TP (Megatron)], [$2 · 2 · B S H · b$], [4 (2×AR/block)], [FFN 后 AR + attn 后 AR]),
  ([TP + SP], [$2 · 2 · B S H · b$], [4 (2×AG + 2×RS)], [总量同 TP，但激活省 $1 / T$]),
  ([DP/DDP], [$2 · P · b / L$], [1 (grad AR，可 overlap)], [每 step 一次；bucket]),
  ([FSDP (ZeRO-3)], [$3 · P · b / L$ per block], [3 (AG param fwd + AG bwd + RS grad)], [param 每层 AG，grad RS]),
  ([PP], [$2 · B S H · b$], [$2(P-1)$ P2P], [仅 boundary，量小]),
  ([CP-Ring], [$4 · B S H · b · (C-1)/C$], [$2(C-1)$ P2P], [K/V + dK/dV rotate]),
  ([CP-Ulysses], [$4 · B S H · b$], [4 (2 a2a fwd + 2 a2a bwd)], [受 $C ≤ A$ 限制]),
  ([EP], [$2 · B S H · b · (E-1)/E$], [2 (dispatch + combine a2a)], [top-k 时 ×k]),
)

推导例：*TP FFN 通信量*。FFN 是 $W_1 (H → I)$ + $W_2 (I → H)$。Column-parallel $W_1$ 不需通信；Row-parallel $W_2$ 出口 AllReduce 一个 $(B, S, H)$ 张量，量 = $B S H · b$，AR per-GPU 通信 = $2 · B S H · b$。前后向各一次 → 每 block 每层 $4 B S H · b$。

推导例：*Ring Attention*。每步 rotate K + V，两个 $(B, A/C, S/C, d_h)$ 张量，per-step 通信 = $2 · B A S d_h · b / C = 2 · B S H · b / C$，共 $C - 1$ 步 → 前向 $2 (C-1)/C · B S H · b$；反向另一份 dK/dV rotate → ×2。

=== 通信时间 = volume / bandwidth

$
t_"comm" = "vol" / "BW"_"eff"
$

有效带宽考虑：小 message 会被 latency 主导（NCCL 默认 chunk ~4-8 MB，small msg 效率 < 30%）；大 message 稳定在 peak 的 70-85%。

#raw(block: true, "```
# src/distributed_training/estimators.py 里 comm_time()
def comm_time(volume_bytes, bw_GBps, alpha_us=5.0):
    # NCCL: t = alpha + volume/bw
    return alpha_us * 1e-6 + volume_bytes / (bw_GBps * 1e9)
```")

=== Worked example：TP=4 一次 forward AR 时间

- $B=1, S=8192, H=8192$, BF16, 单张 AR 的 volume: $2 · (T-1)/T · B S H · b = 2 · 0.75 · 1 · 8192 · 8192 · 2 = 201 "MB"$
- NVLink 4 卡 SXM (H100): 有效 300 GB/s → $t = 0.67 "ms"$
- 一个 block 前反向共 4 次 AR → $2.7 "ms"$
- 32 层 → 86 ms per step 花在 TP AR 上

#warn[
  跨节点 (IB) 场景 AR 时间会 ×5-10。所以 TP 必须留在节点内（NVLink domain）。
]

== 组合：完整 step time 模型

$
t_"step" = t_"compute" + t_"comm-exposed" + t_"bubble" + t_"overhead"
$

其中：
- `t_compute`：如上；DP 不减少 per-GPU compute，TP/CP 会
- `t_comm-exposed`：$= max(0, t_"comm" - t_"compute-overlap")$；DP grad AR 通常可 100% overlap 反向
- `t_bubble`：$(P-1)/m$ × ideal_iter (GPipe/1F1B)
- `t_overhead`：dataloader、evictions、checkpoint save 等，通常 \<5%

=== PP bubble 深潜

GPipe bubble ratio：$(P-1)/(m + P - 1)$。$P = 8, m = 32 → "bubble" = 7/39 = 17.9%$。1F1B 相同 bubble ratio 但 activation memory 只需 $O(P)$ 而非 $O(m)$。Interleaved-1F1B (Megatron)：将 bubble 降至 $(P-1)/(v · m + P - 1)$（$v$ 是每卡 chunks 数）。

#figure(
  align(center, pipeline-schedule(
    stages: 4, cell: 0.32,
    schedule: (
      (("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1)),
      (("_", 1), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1)),
      (("_", 2), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1)),
      (("_", 3), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("_", 3), ("F", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1), ("B", 1), ("W", 1)),
    ),
    title: "GPipe schedule (P=4, m=4)   ← 大块 idle 就是 bubble",
  )),
  caption: [GPipe 时序：warm-up 阶段 (F 阶梯) 和 cool-down 阶段每个 stage 空转。],
) <fig-gpipe-timeline>

#figure(
  align(center, pipeline-schedule(
    stages: 4, cell: 0.32,
    schedule: (
      (("F", 1), ("F", 1), ("F", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("B", 1), ("B", 1), ("B", 1)),
      (("_", 1), ("F", 1), ("F", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("B", 1), ("B", 1), ("B", 1)),
      (("_", 2), ("F", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("B", 1), ("_", 1)),
      (("_", 3), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("_", 2)),
    ),
    title: "1F1B schedule (P=4, m=8)   ← 稳态阶段 F/B 交替，活跃 activation 数只 P",
  )),
  caption: [1F1B 时序：稳态每 stage 交替 F/B，活跃 activation ≤ P 份。相同 bubble ratio，activation 显存低。],
) <fig-1f1b-timeline>

=== Worked example：70B / TP=4 / PP=2 / DP=2 / m=16 / 8k / 8×H100

先按 stage 拆：每 stage 40 B params。single stage single-token compute：$6 · 40 · 10^9 · 8192 = 2 · 10^15$ FLOPs（前反向 6P·tok）。per micro (assume $B_"micro" = 8$ per DP → 8 · 8192 tokens per micro): $2 · 10^15 · 8 = 1.6 · 10^16$

- compute one micro (TP=4): $1.6 · 10^16 / (989 · 10^12 · 0.45 · 4) = 9 "ms"$
- comm per micro: TP FFN AR + attn AR × 2 (fwd+bwd) × 40 layers ≈ 4 · 40 · $t_"AR"$。$t_"AR"$ per micro: $B_"micro" S H · b = 8 · 8192 · 8192 · 2 / (0.75 · 300 "GB/s") ≈ 4.8 "ms"$；每层 $t_"AR" · 4 ≈ 19 "ms"$；40 层 780 ms per micro。啊，通信主导。

这就是"TP 无脑开大 = 通信爆炸"的直觉——面试可以说：*"通信 quadratic 增长于 hidden size 而 compute 也是 quadratic，但常数比不同：TP 通信/compute 比 ≈ $2 / (S · "MFU" · "peak" / "BW")$，H 越大越不利。"*

真实场景往往用 TP=2 + more DP 减小 AR frequency。或用 SP 减 activation → 允许更大 micro batch → 摊薄 comm。

#interview[
  面试题：*"通信占 30% 你怎么优化？"* 答的层次：(1) 换 TP 结构？减 TP 度；(2) SP 减 activation → 增大 B_micro → comm/compute 比降低；(3) enable async TP overlap；(4) 上 FP8 —— 通信可用 FP8 或 BF16 with FP8 compute，量减半；(5) 若跨节点，检查是否误把 TP 拉到节点外。
]

== Roofline 判定

给定 op 的 arithmetic intensity（AI，FLOPs/byte），对比机器的 balance ridge ($"peak-FLOPS" / "peak-BW"$)：
- $"AI" > "ridge"$：compute-bound → 优化算子 (fusion, kernel)
- $"AI" < "ridge"$：memory-bound → 优化 memory (bigger batch, better layout)

H100 SXM: peak 989 TFLOPS / HBM 3.35 TB/s = *295 FLOPs/byte* balance ridge。GEMM $M · N · K$ 的 AI ≈ $M K + N K + M N$ 分之 $2 M N K$ ≈ $O(min(M, N, K))$：只有维度都够大时才 compute-bound。

$K = 8192, M = 4096, N = 4096$: AI = $2 · 4096 · 4096 · 8192 / (4096 · 8192 + 4096 · 8192 + 4096 · 4096) = 2731$ FLOPs/byte → compute-bound ✓

$K = 64, M = 4096, N = 4096$: AI ≈ $2 · 64 / 3 = 42$ FLOPs/byte → memory-bound（小 K 的 GEMM 效率差）。所以 attention 中 $q · k^T$ 里 $K = d_h = 128$ 时就已经开始离开 compute-bound zone。

#raw(block: true, "```
# src/distributed_training/estimators.py: roofline()
def roofline(ai, peak_flops=989e12, peak_bw=3.35e12):
    ridge = peak_flops / peak_bw
    if ai > ridge:
        return 'compute-bound', peak_flops
    return 'memory-bound', ai * peak_bw
```")

== 端到端：从 model config 到 iter 时长

组合上面 3 个 estimator，得到 `end_to_end_step_time(cfg, parallel)`。核心逻辑：

#raw(block: true, "```
def step_time(cfg, para, hw):
    # 1) FLOPs
    F_step = 6 * cfg.P * cfg.tokens_per_step
    if para.recompute == 'full': F_step *= 8/6
    # per GPU
    F_per_gpu = F_step / (para.T * para.C * para.P * para.D)
    t_comp = F_per_gpu / (hw.peak * hw.mfu)
    # 2) Comm
    t_comm = 0
    for coll in per_step_collectives(cfg, para):
        vol = coll.per_gpu_volume
        bw = hw.bw_for(coll.scope)  # intra vs inter node
        t_coll = vol / (bw * hw.eff)
        if coll.overlap_frac > 0:
            t_coll *= (1 - coll.overlap_frac)
        t_comm += t_coll
    # 3) Bubble (PP)
    t_bubble = (para.P - 1) / (para.m + para.P - 1) * (t_comp + t_comm)
    # 4) Total
    return t_comp + t_comm + t_bubble
```")

完整实现见 `src/distributed_training/estimators.py`；下一节给出 8 个常见配置的算例。

== 8 个 worked examples 一览

用 $B_"global" = 4 M$ tokens，H100 SXM，MFU 45%，BF16。

#table(
  columns: (2fr, auto, auto, 1.6fr, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left, center, center),
  [*配置*], [*P/T/D/C/E*], [*compute*], [*comm*], [*bubble*], [*step*],
  [7B · 8k · 8×H100],              [1/1/8/1/1],   [15 s], [1.2 s (DP overlap)],    [0],           [~15 s],
  [13B · 8k · 8×H100],             [1/2/4/1/1],   [28 s], [3.5 s (TP)],            [0],           [~31 s],
  [70B · 8k · 32×H100],            [1/8/4/1/1],   [26 s], [8 s (TP heavy)],        [0],           [~34 s],
  [70B · 8k · 64×H100],            [4/4/4/1/1],   [13 s], [2.5 s],                 [0.7 s (m=32)],[~16 s],
  [70B · 32k · 64×H100 CP=4],      [1/4/4/4/1],   [22 s ($S^2$ attn)], [4 s (ring)],[0],          [~26 s],
  [Llama3-405B · 8k · 512×H100],   [16/8/4/1/1],  [15 s], [3 s],                   [0.9 s (m=64)],[~19 s],
  [DeepSeek-V3 671B · 128×H100],   [2/1/8/1/8],   [17 s], [7 s (EP a2a)],          [0],           [~24 s],
  [Mixtral 47B MoE · 32×H100],     [1/2/8/1/2],   [11 s], [3 s],                   [0],           [~14 s],
)

*用法：* 面试拿到题，先按这个表找最近配置，再按目标模型 scale。90% 的题这样就能给出 ±20% 的答案。

== 面试脚本模板

*Q：估算 70B / 8k / 8×H100 一次 step。*

A：好，先分层：
1. *参数*：70B → BF16 opt-state 需要 $16 · 70 = 1120$ GB。8 卡 ZeRO-3 平摊 → 140 GB / 卡，H100-80G 装不下——所以要么 TP，要么 ZeRO+offload。假设 TP=4 DP=2，params/opt shard 后每卡 ~35 GB，可装。
2. *compute*：$6 · 70 · 10^9 · N_"tok"$。设 batch 4M tokens：$F = 1.7 · 10^18$。8 卡 45% MFU → $1.7 · 10^18 / (8 · 989 · 10^12 · 0.45) = 480 "s"$。等等，这是 4M tokens；per step 通常 1M tokens 更合理，即 ~120s。
3. *comm*：TP=4 每层 4 次 AR，$B_"micro" S H b ≈ 8 · 8192 · 8192 · 2 = 1 "GB"$ per AR × 0.75 × 4 层 · 80 = ~30s（NVLink 300 GB/s）。DP grad AR 一次 20 GB → 100ms，overlap 掉。
4. *bubble*：无 PP → 0。
5. *合*：120 + 30 = 150 s per step ≈ 2.5 min。若 comm 占 20%+ 要考虑 async TP overlap 降到 10%。

*注意：*面试时 (a) 报单位记得除；(b) 每步都留 sanity check ("这数字合理吗，H100 应该 3-5s per iter?"); (c) 一旦发现 comm 主导要主动 pivot 优化建议。这套流程比记住答案值钱多了。

== 附：estimator 完整代码位置

```
src/distributed_training/estimators.py
├── ModelConfig(L, H, I, A, V, S)
├── ParallelConfig(P, T, D, C, E, m, recompute)
├── HWConfig(peak, hbm_bw, nvlink_bw, ib_bw, mfu)
├── mem_estimate(cfg, para)
├── flops_step(cfg, para)
├── comm_volumes(cfg, para) -> dict of collective→bytes
├── comm_time(vol, bw, alpha_us=5)
├── roofline(ai, hw)
└── step_time(cfg, para, hw) -> dict of components
```

跑法：
```bash
cd src/distributed_training
python3 estimators.py --preset llama3-70b-tp4-dp2-h100x8
```

会打印下表：

```
Compute:  118.2 s  (78%)
Comm:      30.1 s  (20%)
Bubble:     0.0 s  ( 0%)
Overhead:   2.1 s  ( 1%)
Total:    150.4 s
Memory/GPU: 62 GB  (params 8 + grads 4 + opt 12 + act 38)
```
