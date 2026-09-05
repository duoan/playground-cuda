#import "../template.typ": *

= 主流 recipe 速查：Llama 3 / DeepSeek-V3 / Qwen 3 / Kimi K2

面试问："你训过 Llama？他们的 recipe 你能背几个？" 这章把 2024-2025 主流开放技术报告里的具体训练配方汇总成一张速查表，涵盖 optimizer、LR、init、batch scaling、precision、schedule、并行策略。目的是让你能*当场对比*不同模型的选择差异，而非死背某一个。

== Llama 3 (Meta, 2024)

*Model*: 8B / 70B / 405B. Dense (无 MoE). RMSNorm + SwiGLU + GQA-8 + RoPE 500K → YaRN 128K.

*Data*: 15T tokens (405B 版), 数据配比 heavy web + code + math.

*Training*:
- Optimizer: AdamW, `betas=(0.9, 0.95), eps=1e-8, wd=0.1`
- LR schedule: warmup 8000 步 → cosine to `lr_min = 0.1 × peak`
- Peak LR: 3e-4 (8B), 1.5e-4 (70B), 8e-5 (405B) —— sqrt scale by width
- Batch size: 8B → 4M tokens; 405B → 4M~16M ramp
- Init: Wang-style (residual output projection $"std" = 0.02 / sqrt(2L)$)
- Precision: BF16 mixed, master FP32
- Grad clip: 1.0

*Parallelism (405B on 16K H100)*:
- TP=8 (intra-node NVLink)
- PP=16 (across nodes)  
- CP=8 (long ctx 128K stage)
- DP=16 (with FSDP HYBRID)
- SP on
- 40 days per full run

*Long context stage*:
- 主 pretrain 完成在 8K → 追加 800B tokens 在 128K
- RoPE base 从 500K rewarm 到 8M (YaRN)

*RLHF stage*:
- SFT → DPO (chosen: rejected pair) 而不是 PPO
- 迭代式 5 rounds

*报告独特之处*: 
- "no significant loss spikes" - 主要靠 grad clip 1.0 + 保守 LR
- 论文明确列 hardware failure 每 3 hours 一次
- Checkpoint 每 10 min，可 recover

== DeepSeek-V3 (2024)

*Model*: 671B total / 37B active. MLA + fine-grained MoE (256 routed + 1 shared, top-8). RoPE + YaRN.

*Data*: 14.8T tokens.

*Training*:
- Optimizer: AdamW, `betas=(0.9, 0.95), wd=0.1`
- LR schedule: WSD variant
  - Warmup: 2000 steps
  - Stable: 大部分 (至 90% training)
  - Decay: 分两段 cosine-like, 最终 lr_min = 0.1 × peak
- Peak LR: 4.2e-4 (small tuning)
- Batch: 3072 seq × 4096 → ~12.5M tokens
- Init: MuP-inspired scaling
- Precision: *FP8 mixed*, weight BF16, forward FP8 E4M3, backward FP8 E5M2
- No aux loss (bias-based load balance)

*Parallelism (2048 × H800)*:
- TP=1 (显式不用！)
- PP=16 (DualPipe)
- EP=64 (MoE dispatch, DeepEP for a2a)
- FSDP on non-expert params
- ZeRO-1 on expert params

*Stability tricks*:
- Sub-Layer Normalization (类似 sandwich Norm)
- MLA 内 up-project matmul upcast FP32
- Router logit z-loss (虽然 no-aux)
- 训练 2.788M H800 hours (79 days on 2048 GPUs) with "remarkably stable" claim

*报告独特之处*:
- FP8 训练：H800 上 forward compute 1.5×
- DualPipe: 让 PP bubble 逼近 0（通过 EP a2a 与 PP compute overlap）
- Multi-token prediction (MTP): 加 speculative decoding target 头，Loss 略优

== Qwen 3 (Alibaba, 2025)

*Model*: 4B / 32B dense; 30A3B / 235A22B MoE. RMSNorm + SwiGLU + GQA-8. RoPE 1M + YaRN.

*Data*: 36T tokens (235B MoE 版).

*Training*:
- Optimizer: AdamW `betas=(0.9, 0.95), wd=0.1`
- LR schedule: WSD-like with 3-stage curriculum
  - Warmup 2000 步
  - Stage 1: general web + code (peak LR)
  - Stage 2: math heavy (LR × 0.7)  
  - Stage 3: high quality + reasoning (LR × 0.3)
  - Final decay 5%
- Peak LR: 1e-4 (MoE 235B)
- Batch: ~8M tokens
- Precision: BF16 mixed
- Aux loss: 保留 (small $alpha$)

*Parallelism (235B on ~4096 H800)*:
- TP=1 (MoE 版; dense 32B 用 TP=2)
- PP=8 (interleaved 1F1B)
- EP=8 or 16
- DP with FSDP2

*Notes*:
- Thinking mode training: RL 阶段特殊 loss，加上思考 token 的 mask
- Long ctx stage: 128K → 256K → 1M rewarm

== Kimi K2 (Moonshot, 2025)

*Model*: ~1T total (架构类似 DeepSeek). MLA + 384 experts + shared. YaRN long ctx.

*Data*: ~15T tokens.

*Training (最独特!)*:
- *Optimizer: Muon* (2D 权重), AdamW (1D)
  - Muon: `beta=0.95`, Newton-Schulz 5 iterations
  - AdamW: `betas=(0.9, 0.95), wd=0.1`
- LR schedule: WSD + mini-restart at 60%
  - Warmup 2000 → stable ~78% → mini rewarm (LR × 0.5 → warmup 500 → stable) → final decay
- Peak LR: 2e-3 (Muon 允许比 AdamW 大 3-5×)
- Init: near-orthogonal（配合 Muon）
- Batch: 8M tokens
- Precision: BF16 (Muon 与 FP8 集成尚未稳定)

*Parallelism*:
- ZeRO-1 (must! Muon needs full matrix per rank)
- EP=32
- PP=8
- No TP (like DeepSeek)

*报告独特之处*:
- Muon vs AdamW ablation: Muon loss curve 明显更平滑，无 spike
- Mini restart at 60%: 防 stagnation
- MuP scaling: 300M model 调 LR 迁移到 1T

== 老一代 recipe 对比

*GPT-3 (2020)*:
- Adam betas=(0.9, 0.95, 1e-8), wd=0.1
- Warmup 375M tokens → cosine to 10% peak → const
- 8×64 batch → 3.2M tokens

*OPT-175B (2022, Meta)*:
- AdamW, warmup 2K → linear decay
- 45+ manual restarts due to spike (recorded in paper)
- 教训: WSD/rewarming 尚未普及, cosine 硬训

*GLM-130B (2022, THUDM)*:
- Adafactor + Adam mixed
- Sandwich Norm (fix Pre-LN activation drift)
- Embedding layer norm gradient shrink

*Chinchilla (2022, DeepMind)*:
- 定义了 20 × params tokens = compute-optimal
- AdamW cosine, wd=0.1

*PaLM (2022, Google)*:
- Adafactor (TPU friendly)
- Loss "step function" spike; skip-and-retry restart

== 快速对比表

// Split into two narrower tables so nothing overflows.
*算法侧配方 (optim / LR / batch / precision)*

#table(
  columns: (1.4fr, auto, 1.3fr, auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, left, center, center, center),
  [*model*], [*optim*], [*LR sched*], [*peak LR*], [*batch*], [*prec*],
  [Llama-3 8B],     [AdamW],    [warmup+cos],       [3e-4],   [4M],    [BF16],
  [Llama-3 70B],    [AdamW],    [warmup+cos],       [1.5e-4], [4M],    [BF16],
  [Llama-3 405B],   [AdamW],    [warmup+cos],       [8e-5],   [4-16M], [BF16],
  [DeepSeek-V3],    [AdamW],    [WSD],              [4.2e-4], [12.5M], [FP8],
  [Qwen3 235B],     [AdamW],    [WSD 3-stage],      [1e-4],   [8M],    [BF16],
  [Kimi K2],        [Muon+AdW], [WSD+restart],      [2e-3],   [8M],    [BF16],
  [Mixtral 8×22B],  [AdamW],    [cosine],           [2e-4],   [4M],    [BF16],
  [GPT-3 175B],     [Adam],     [warmup+cos+const], [6e-5],   [3.2M],  [FP16],
  [OPT-175B],       [AdamW],    [warmup+linear],    [1.2e-4], [2M],    [FP16],
)

#v(0.6em)
*系统侧配方 (parallelism)*

#table(
  columns: (1.4fr, 2.5fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*model*], [*parallelism*],
  [Llama-3 8B],    [FSDP2 only],
  [Llama-3 70B],   [TP=4, PP=4, FSDP over DP],
  [Llama-3 405B],  [TP=8, PP=16, CP=8, FSDP over DP (16K H100)],
  [DeepSeek-V3],   [no TP, PP=16 (DualPipe), EP=64, FSDP non-expert + ZeRO-1 expert],
  [Qwen3 235B],    [no TP, PP=8, EP=8, FSDP2],
  [Kimi K2],       [ZeRO-1 only (Muon needs full matrix), PP=8, EP=32],
  [Mixtral 8×22B], [PP, EP=4, FSDP],
  [GPT-3 175B],    [Megatron TP + PP],
  [OPT-175B],      [Megatron TP + PP],
)

== 你要背的 5 个数字

面试常问"你们训 X 用什么超参"，报这 5 个至少 hit 一个：

+ *AdamW betas*: (0.9, 0.95) — 不是 0.999
+ *Weight decay*: 0.1
+ *Warmup steps*: 2000
+ *Grad clip*: 1.0
+ *lr_min / peak_lr*: 0.1

其他随 model 变，但这 5 个是全行业主流。

== 面试考点

#interview[
  *Q1*：DeepSeek 和 Llama 训练最大的三个差异？

  A：
  + *Precision*: DeepSeek 用 FP8 (H800 上 forward 1.5×)，Llama 保守 BF16
  + *TP*: DeepSeek 明确 *不用* TP（MLA + fine-grained MoE 让 TP 收益差），Llama 405B 用 TP=8
  + *LR schedule*: DeepSeek WSD, Llama cosine
  背后逻辑：DeepSeek 追极致 training efficiency (H800 受限)，Llama 保守求稳定。
]

#interview[
  *Q2*：Kimi K2 用 Muon 相对 AdamW 什么优势？

  A：
  + Loss curve 更平滑（Kimi 报告 0 vs 3 次可见 spike）
  + LR 可 3-5× 大 (2e-3 vs 4e-4)
  + Optim state 少一半 (无 second moment $v$)
  + 与 μP scaling 集成好
  代价：与 FSDP 不兼容（Newton-Schulz 要 full matrix），只能 ZeRO-1；1D 权重仍 AdamW；FP8 集成不成熟。
]

#interview[
  *Q3*：为什么 GPT-3 用 (0.9, 0.95) 而不是默认的 (0.9, 0.999)？

  A：LLM 数据分布快速变化（curriculum, packing 组合不断变化），$beta_2 = 0.999$ 让 $v$ 半衰期 700 步——反映了远古的 gradient scale，不 adaptive。$0.95$ 半衰期 14 步，紧跟当前 scale。这个默认改动被 GPT-3 论文推广，之后所有 LLM 遵循。
]

#interview[
  *Q4*：LLM 训练里 wd=0.1 是不是有点太大？

  A：不大，*正好*。Decoupled weight decay 独立于 gradient，`theta -= lr * wd * theta` 每 step 缩 `lr * wd = 3e-4 * 0.1 = 3e-5`——很小。1M steps 才 shrink 3%。目的是防止 weight 无限膨胀（activation 也跟着涨→BF16 drift）。CV 时代 wd=1e-4 是因为 L2 (被 grad scale 缩)，AdamW decoupled 后 wd 才敢开大到 0.1。
]

#interview[
  *Q5*：4M tokens 的 batch size 你会开吗？为什么不再大？

  A：4M 是 2024 主流 (Llama-3, Qwen-3)。DeepSeek-V3 到 12.5M, Llama-3 405B 涨到 16M。为什么不无限大：
  + Critical batch size (Kaplan)：超过后 loss 不再随 batch 增大同比降低
  + LR 只能 sqrt-scale，实际收益递减
  + 单 iter 太慢 → 反馈周期长，问题发现晚 (data corruption, spike 判断)
  经验：越大 model 越大 batch，8B → 4M, 70B → 4-8M, 400B+ → 8-16M。
]

#interview[
  *Q6*：训练中间 checkpoint，continual pretrain 应该从哪个 ckpt 起？

  A：*WSD 的 stable 阶段 ckpt*，不是 decay ckpt。因为 decay 完 LR 已很低，loss landscape 已"sharpen"到局部极值，rewarm 破坏这个。Stable ckpt 是 fully-trained-at-peak，loss landscape 平坦，rewarm 到新 peak 平滑过渡。这是 WSD 相对 cosine 的一大优势——cosine 无"稳定态 ckpt"可用。
]

#interview[
  *Q7*：你从 Kimi K2 recipe 里学到什么可以用到自己项目？

  A：可迁移的：
  + WSD + mini restart at 60% (防 stagnation)
  + Grad clip 1.0 + monitor grad_norm
  + BF16 mixed precision，LN var FP32
  + MuP scaling 让 hyper search 在 small model 上做
  不宜盲抄：
  + Muon (需 ZeRO-1，与现有 FSDP 生产不兼容)
  + 384 experts (需自研 dispatch lib)
  + 1T model scale (硬件不够)
]

#interview[
  *Q8*：面试官说"我们最近 loss spike 挺多"，你怎么问下一层？

  A：结构化提问：
  + 频率：每 100 步、每 1000 步、还是随机偶发？
  + 复发规律：与 LR schedule 阶段有关吗？与 curriculum 换阶段有关吗？
  + 阶段：warmup 阶段 vs stable 阶段 vs decay 阶段？
  + 严重度：能自愈还是需要 restart？
  + Monitor：`grad_norm` 是否也 spike？`weight_norm` 是否累积增长？
  这些答案能锁定问题：
  - 偶发 + 自愈 → data outlier（改 grad clip + skip step）
  - warmup 阶段 + 系统性 → LR 太大 or init 差（改 warmup step 或 init）
  - decay 阶段 + curriculum 相关 → 数据阶段切换 (rewarming)
  - 已 checkpoint 但 restart 后立即 spike → ckpt 损坏或 precision 变化
]
