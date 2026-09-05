#import "../template.typ": *

= 现代 LLM 架构：从 GPT-2 到 DeepSeek-V3

分布式训练面试常见问法："你训过 Llama，那 DeepSeek 的 MLA 有什么不同？"或"为什么现在都不用 Multi-Head Attention，改 GQA / MLA 了？" 答不上这些，就会被认为"只会调库"。这一章*不*教你怎么从零推导 attention（那是 Andrej 的视频），而是快速梳理业界主流 LLM 的关键选择、演进原因、对分布式训练的影响。

== Transformer 骨架的 8 个模块

现代 dense LLM (Llama, Qwen, Mistral) 的一层长这样：

#figure(
  align(center, op-stack(steps: (
    ("Input x",                    "residual stream (B, S, H)", "full"),
    ("RMSNorm 1 (Pre-LN)",         "归一化",                    "full"),
    ("QKV proj + RoPE(Q, K)",      "attention 入口",            "full"),
    ("Attention (GQA / MLA)",      "Flash-attn kernel",         "full"),
    ("Output projection",          "回到 H 维",                 "full"),
    ("+ residual",                 "跳连接",                    "comm"),
    ("RMSNorm 2",                  "归一化",                    "full"),
    ("FFN (SwiGLU)",               "gate / up / down 3 linear", "full"),
    ("+ residual",                 "跳连接",                    "comm"),
  ), width: 8.0, cell-h: 0.55)),
  caption: [现代 dense LLM (Llama-3, Qwen-3, Mistral) 一层 transformer 的完整数据流。Pre-LN 结构：LN 在每个子层入口，残差在末尾 (add)。SwiGLU 是标准 FFN，MHA 现在少见 —— 都换成了 GQA 或 MLA。],
) <fig-transformer-layer>

面试的每个模块都有可挑的点。逐个过。

== Pre-LN vs Post-LN：为什么现在都是 Pre-LN

*Post-LN (原 Transformer, BERT)*：`x → attn → +x → LN`。信号先加残差再归一，训练早期梯度大，需要 learning-rate warmup，深层难训。

*Pre-LN (GPT-2 起)*：`x → LN → attn → +x`。归一化在残差之前，梯度更稳。GPT-2 之后所有 decoder-only LLM 全用 Pre-LN。

*为什么 Pre-LN 好训*：Xiong et al. 2020 证明 Pre-LN 的梯度 norm 在 layer 深度上*保持 $O(1)$*，Post-LN 是 $O(sqrt(L))$——300 层 Post-LN 训不起来。

*Sandwich Normalization (GLM-130B, Cogview)*：Pre + Post 都加。缓解 Pre-LN 的 activation drift 问题（deep Pre-LN 中间层输出会累加变大）。DeepSeek-V3 用 Sub-Layer Normalization (类似 sandwich 的变体) 抑制 MoE spike。

== LayerNorm vs RMSNorm

*LayerNorm* (Ba 2016)：$y = "gain" · (x - mu) / sigma + "bias"$。要计算 mean 和 var，两次 reduction。

*RMSNorm* (Zhang 2019)：$y = "gain" · x / sqrt("mean"(x^2) + epsilon)$。只算 $x^2$ 的均值，一次 reduction。

*为什么 LLM 全换 RMSNorm*：
+ 速度快 ~10% (少一次 reduction + no bias)
+ Llama 论文实测 loss 曲线一致——没损失
+ 分布式训练里，SP 场景下 LN 需要跨 seq 组通信 mean/var，RMSNorm 只 mean(x²)，通信更少

Llama-1 起全用 RMSNorm。Qwen, DeepSeek, Kimi K2, Mistral 全用。GPT-4 未公开但推测也是 RMSNorm。

*eps 陷阱*：`eps=1e-5` 是默认，BF16 下会出问题——因为 `sqrt(var + 1e-5)` 在 var 特别小时精度不够。修法：`eps=1e-6` 甚至 `1e-8`，或 var 计算走 FP32（`upcast_var=True`）。DeepSeek-V3 训练报告明确提到 Norm 计算强制 FP32。

== SwiGLU 与其他 FFN 变体

传统 FFN：`W2 · GELU(W1 · x)`。两个矩阵乘。

*GLU 家族 (Shazeer 2020)*：`W3 · (act(W1 · x) ⊙ W2 · x)`。三个矩阵乘。用 activation 选：
- ReGLU: ReLU
- GEGLU: GELU
- *SwiGLU: SiLU (= x · sigmoid(x))* ← 目前主流

*参数量*：SwiGLU 用 3 个 linear。为等参数量，Llama 把 inter_size 设为 $8H/3$ 而不是 $4H$（乘 3/4 抵消）。

*为什么 SwiGLU 好*：Shazeer 论文 empirical：同参数量下 SwiGLU 优于 ReLU FFN 约 1% perplexity。没有理论解释，就是"work"。

Llama, Qwen, Mistral, DeepSeek 全 SwiGLU。GPT-3 是 GELU。

== 位置编码：从绝对到 RoPE

*绝对 (原 Transformer, GPT-2)*：learned 或 sinusoidal position embedding，加到 token embedding 上。缺点：长度不能外推。

*相对 (T5, DeBERTa)*：attention bias。可以外推但慢。

*RoPE (Su 2021, Llama)*：对 Q, K 应用旋转矩阵，让 attention score 天然带相对位置。

$
Q'_m = R_m · Q_m, quad K'_n = R_n · K_n, quad Q'_m · K'_n = Q_m · R_(n-m) · K_n
$

其中 $R_theta$ 是 2D 旋转矩阵。实际把 dim 拆成 pair，每对 (2i, 2i+1) 用不同 base 频率 $theta_i = 10000^(-2i/d_h)$。

*为什么 RoPE 赢：*
+ 无参数（不占 gradient / memory）
+ 天然相对位置——外推可以（虽然要调 base）
+ 与 KV cache 兼容——每个位置 rotate 一次，与 attention 内积交换律友好
+ 与 FlashAttention 完美兼容

*RoPE base scaling (长上下文关键)*：
- Llama-1: base = 10000, S = 2K
- Llama-2: base = 10000, S = 4K
- Llama-3: base = 500000, S = 8K
- Llama-3.1: base = 500000 + YaRN scaling, S = 128K
- Qwen 2.5: base = 1000000
- Kimi K2: NTK-aware + YaRN

*长上下文 RoPE 三大方案*：
+ Position Interpolation (Chen 2023)：把长 seq 的 position 缩放到 train 时的范围
+ NTK-aware / dynamic-NTK (bloc97)：让高频衰减，低频保留
+ *YaRN (Peng 2023)*：结合 NTK + temperature scaling + attention scaling，目前主流

*Rope overflow*：BF16 下 sin/cos 在 base=1e6 时精度差，需要用 FP32 计算 R 后 downcast。这是长上下文训练的隐蔽 NaN 源之一，见 P4。

== Attention 变体：MHA → MQA → GQA → MLA

*Multi-Head Attention (MHA)*：$A$ heads，每 head 独立 KV。参数量 $= 4 H^2$。KV cache size = $2 · A · d_h · S · b · L = 2 H S · b · L$。

*Multi-Query Attention (Shazeer 2019)*：KV 只 1 组，共 $A$ 个 Q head 用。KV cache 缩 $A$ 倍。快但质量下降。

*Grouped Query Attention (GQA, Ainslie 2023)*：折中——$G$ 组 KV，$A/G$ 个 Q 共用一组 KV。$G in [4, 8]$ 主流。KV cache 缩 $A/G$ 倍。质量与 MHA 接近。

- Llama-2 34B/70B: G=8
- Llama-3 8B/70B: G=8
- Llama-3.1 405B: G=8
- Qwen 2.5 72B: G=8
- Mistral 7B: G=8

*Multi-Head Latent Attention (MLA, DeepSeek-V2)*：把 KV 压缩到低秩隐空间 $c_"kv" in RR^(d_c)$，$d_c ≪ H$。attention 时先 up-project 回 $K, V$。参数减少、KV cache 极小（只存 $c_"kv"$）。

$
c_"kv" = W_"DKV" · x   quad "(down-project, cached)" \
K = W_"UK" · c_"kv" + "RoPE"   quad "(up-project on the fly)" \
V = W_"UV" · c_"kv"
$

*为什么 DeepSeek 选 MLA 而不是 GQA*：
+ KV cache 更小（DeepSeek-V3 每 token KV = 3.4 KB, vs GQA-8 的 12 KB）
+ 训练侧无损（DeepSeek 论文对比 GQA-8 相同 KV budget，MLA 略好）
+ RoPE 需要特殊设计（K 分成 rotated 部分 + non-rotated 部分）——工程复杂

*对分布式训练的影响*：MLA 的 QKV projection 结构比 MHA 复杂——列并行需要特别处理 (up-proj 权重也要沿 head 切)。DeepSeek 训练主要用 EP+FSDP，*不用 TP*（也无需处理 MLA 的 TP 切分）。

*Sliding Window Attention (SWA, Mistral)*：每 token 只看前 $W$ 个 token（$W = 4096$）。KV cache 常数化。Mistral 7B 用；Llama/DeepSeek 不用（认为 quality drop）。

*Attention Sink / Register (StreamingLLM, Xiao 2024)*：观察 attention 会强烈聚焦在前几个 token（sink）。Streaming 场景保留 sink 位置 + 滑窗，实现无限长 inference。Kimi Chat 用类似机制。训练时若有 sink token（如 BOS 特殊 token）可显式建模。

== MoE 层结构

Dense FFN → *MoE FFN*：每个 token 路由到 $K$ 个 expert（$K$ 通常 1-2），其余 expert 不激活。

*模块*：
+ *Gate / Router*：`gate = softmax(W_g · x)`，选 top-$K$。Loss 项：load balance loss (鼓励均匀路由)。
+ *Expert*：普通 FFN。
+ *Dispatch / Combine*：跨 EP rank 的 all-to-all（Ch8）。

*Fine-grained expert*：DeepSeek-V3 把 expert 数从常见 8 变成 256（每个更小）。经验：*expert 更多 + 更小*收益优于*少而大*。同参数量下前者 loss 更低。Kimi K2 用 384 experts。

*Shared expert (DeepSeek-V2/V3)*：每层除了 routed experts 还有 1-2 个"共享 expert"，每 token 都过。承担通用能力，减轻 routed expert 的负担。

*Auxiliary loss*：
- Load balance loss：$"lb-loss" = alpha · sum_i (f_i · P_i)$，$f_i$ = expert i 被路由到的 token 频率，$P_i$ = 平均概率
- Router z-loss (ST-MoE)：`log(sum_i exp(logit_i))^2`，抑制 logit 幅度爆炸
- Bias-based load balance (DeepSeek-V3, no aux)：给每 expert 加动态 bias，无需 aux loss，收敛更快

== Embedding tying / Untied

*Tied*：`lm_head.weight = embedding.weight.T`。省 $V · H$ 参数。Llama 系列 (7B/13B) tied；Llama-3 起 untied（35B extra 参数，quality gain 值得）。

*分布式的坑*：tied weight 意味着 `.parameters()` 出来只有一份，但 grad 会累加两遍——DDP 里没问题（都进同一个 bucket），但 FSDP 会 assert（同一 param 不能 shard 两次）。FSDP 用 `ignored_modules=[lm_head]` 或提前 untie。

== 各模型的架构 diff 表

#table(
  columns: (1.4fr, 1.2fr, auto, 1.1fr, 1.4fr, 1.6fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, center, center, left),
  [*模型*], [*LN*], [*FFN*], [*Attn*], [*PE (base) / Vocab*], [*MoE*],
  [GPT-3 175B],       [Pre-LN],       [GELU],   [MHA],       [learned / 50K],       [—],
  [Llama-2 70B],      [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE 10K / 32K],      [—],
  [Llama-3 70B],      [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE 500K / 128K],    [—],
  [Llama-3.1 405B],   [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE+YaRN / 128K],    [—],
  [Mistral 7B],       [RMSNorm],      [SwiGLU], [GQA-8+SWA], [RoPE / 32K],          [—],
  [Mixtral 8×22B],    [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE 1M / 32K],       [E=8, top-2],
  [Qwen 2.5 72B],     [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE 1M / 152K],      [—],
  [Qwen 3 235B],      [RMSNorm],      [SwiGLU], [GQA-8],     [RoPE+YaRN / 152K],    [E=128, top-8],
  [DeepSeek-V3 671B], [RMS + Sub-LN], [SwiGLU], [MLA],       [RoPE+YaRN / 129K],    [E=256 + 1 shared, top-8],
  [Kimi K2],          [RMSNorm],      [SwiGLU], [MLA],       [RoPE+YaRN+NTK / 163K],[E=384, MuP scaled],
)

#insight[
  面试快速识别 model family：*"用了 MLA 且 shared expert = DeepSeek 系；用 SWA = Mistral 系；GQA-8 + RoPE-500K = Llama 3 系"*。这三点能覆盖 80% 模型。
]

== 初始化 (init)

*Xavier / He*：早期 CNN 时代。LLM 不用。

*Fixed std (GPT-2)*：`init_std = 0.02`。所有 linear 都用此。

*Scaled residual init (GPT-2, Megatron)*：残差路径上的 `W_o`, `W_2` 用 $"std" = 0.02 / sqrt(2 L)$ 缩放，防止 activation norm 随层数爆炸。

*Wang init (Llama)*：`W_o` init std = $0.02 / sqrt(2 L)$；其他 linear std = $0.02$。

*Muon init (Kimi K2, 2025)*：与 Muon optimizer 配套。所有权重 near-orthogonal (SVD s ≈ 1)。

*Embedding init*：`init_std = 1.0` (Llama-2 论文)，比其他小；也有用 $sqrt(1/H)$ 的。

*位置 init (learned PE)*：uniform 或 sinusoidal。RoPE 无 param 免疑。

*RMSNorm gain init*：`1.0`。Post-init 后每层 gain 会自然增长到 1.5-2.0。

*为什么 init 重要*：GLM-130B 训练最初 loss spike 就是 init std 太大 + Pre-LN 累加 → 深层 activation 幅度爆炸。改小 init std 后稳定。

== 面试考点

#interview[
  *Q1*：Pre-LN vs Post-LN，为什么现在都是 Pre-LN？

  A：Xiong 2020 证明 Post-LN 梯度 norm 随深度 $O(sqrt(L))$ 增长，Pre-LN $O(1)$。所以 Post-LN 必须 warmup + 精细 LR，Pre-LN 更 robust。GPT-2 起全部 Pre-LN。缺点：activation 会累加（sandwich LN 解决）。
]

#interview[
  *Q2*：GQA 和 MLA 都省 KV cache，什么时候选谁？

  A：GQA 简单 (只是 replicate K/V head)，与 MHA 训练几乎无差别，工程零成本；缺点：KV cache 只能缩到 $A/G$ 倍。MLA 用 low-rank cache，缩到极小；缺点：需要 RoPE 特殊设计 (rotated + non-rotated 部分)，训练/推理代码复杂。想 push KV cache 极限（比如 128K 长上下文 inference）就上 MLA。
]

#interview[
  *Q3*：为什么 RMSNorm 能替换 LayerNorm？RMSNorm 的 eps 为什么不能太小？

  A：LN 减 mean 只是为了对齐 activation 分布，实测在 residual 网络里"减 mean 效果不明显"（因为 residual 保持零均值）——所以 RMSNorm 只除 std 就够。速度快 10%，quality 不降。eps 太小会让 BF16 下 `sqrt(var + eps)` 精度差，出 NaN。推荐 eps=1e-5 时把 var 计算 upcast 到 FP32。
]

#interview[
  *Q4*：RoPE 怎么外推到长上下文？三种方案？

  A：
  + Position Interpolation (Chen 2023): 直接把 pos 除以 factor
  + NTK-aware: 只缩小高频，保低频
  + YaRN (主流): NTK + temperature scaling on attention logits + length-scale
  外加改 RoPE base（10K → 500K → 1M）从 pretrain 阶段就支持长上下文。
]

#interview[
  *Q5*：DeepSeek-V3 为什么不用 TP？

  A：MLA + fine-grained expert 结构下，TP 收益递减。DeepSeek 采用 EP=64 + FSDP + DualPipe：EP 切 expert 权重，FSDP 切 attention/embedding 权重，DualPipe 切层。避免了 TP 内的 AllReduce（TP AR 是所有并行策略里最频繁的通信）。同时全部 comm 走 NVLink+DeepEP（自研 all-to-all lib），效率高。
]

#interview[
  *Q6*：Shared expert 是什么？为什么有效？

  A：MoE 层里*除了* $K$ 个 routed expert，还有 1-2 个"永远激活"的 shared expert，每个 token 都走。作用：承担通用能力（通用语法、常识），让 routed expert 专注于"细分能力"。DeepSeek 论文数据：加 1 个 shared expert 减少 routed expert 的 collapse，同参数量下 loss 更低。
]

#interview[
  *Q7*：MoE 里 auxiliary loss 是什么？DeepSeek-V3 的"no aux loss"怎么实现的？

  A：aux loss = load balance loss，鼓励 token 均匀分到各 expert。$"lb-loss" = alpha sum f_i P_i$。缺点：与 language loss 冲突，超参 $alpha$ 敏感。DeepSeek-V3 用 *bias-based balancing*：每 expert 给一个动态 bias（EMA 更新），bias 让"最近少被选的 expert"更容易被选，无需 aux loss。收敛更稳。
]

#interview[
  *Q8*：tied embedding 和 untied embedding 各有什么好处？

  A：Tied：省 $V · H$ 参数（Llama-7B 省 260M ≈ 3.7%）。Untied：embedding 与 lm_head 各自优化，quality 略高（Llama-3 起改 untied）。分布式：tied 在 FSDP 里麻烦（同 param 不能 shard 两次），需 `ignored_modules` 或 untie。
]

#interview[
  *Q9*：SwiGLU 相比 GELU 好在哪？为什么参数量看似多但要缩 inter_size？

  A：SwiGLU 三个 linear（gate, up, down）vs GELU 两个（up, down），参数量 1.5×。为等参数量，inter_size 从 $4H$ 缩到 $8H/3$。同参数量下 SwiGLU quality 略好（+1% perplexity）。工程上多一个 linear 意味着多一次 gemm——但 fused kernel（如 xformers 里的 `swiglu`）能压平差距。
]
