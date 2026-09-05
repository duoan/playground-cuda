#import "../template.typ": *

= 模型层与低精度：Fine-grained、Shared、MLA、FP8

前两章讲*怎么把训练跑快*，这一章讲*怎么把模型改得更适合训练与推理*。都是"改模型结构"或"改数值精度"层面的优化——与前面的 kernel / 通信优化正交，可以叠加。

主线：

+ *专家结构*：细粒度 (fine-grained) 与共享 (shared) 专家，2024 年之后的 SOTA MoE 都走这条
+ *低秩化*：LatentMoE / MoLAE / MLA，把 hidden 维压掉一部分省通信和参数
+ *低精度训练*：FP8 recipe，DeepSeek-V3 是当前唯一大规模验证过的
+ *生命周期*：Upcycling（dense → MoE）、Pruning / Merging（MoE → 小 MoE）

== 细粒度专家 + Shared Expert：DeepSeekMoE 范式

=== 动机：从 Mixtral 到 DeepSeekMoE

Mixtral 8×7B (2023) 用 $E = 8$，top-2。DeepSeekMoE (2024)#footnote[Dai et al., _DeepSeekMoE_, arXiv:2401.06066.] 观察到：$E$ 少而每个 expert 大，实际上限制了"专家专精"——每个 expert 被迫承担多种技能，router 学不好精细划分。

*两条改动*：

+ *Fine-grained segmentation*：把每个 expert 的 FFN intermediate dim 缩到 $1/m$，然后 expert 数变 $m$ 倍、top-K 变 $m$ 倍，*iso-FLOP*
+ *Shared expert isolation*：留出 $N_s$ 个 shared expert，*每个 token 都必经过*，吸收"公共知识"

组合数爆炸得很直观：Mixtral top-2 / 16 experts 只有 $binom(16, 2) = 120$ 种组合；DeepSeek-V3 top-8 / 256 experts 有 $binom(256, 8) approx 4.09 times 10^14$——路由空间大了 12 个数量级。

=== 结构公式

$
y = underbrace(sum_(i=1)^(N_s) "FFN"_i^"shared"(x), "always-on")
  + underbrace(sum_(k=1)^(K_r) tilde(w)_k dot "FFN"_(i_k)^"routed"(x), "top-K routed")
$

DeepSeek-V3 配置：$N_s = 1, N_r = 256, K_r = 8$，expert intermediate dim = 2048，sigmoid gating。

=== 训练代码骨架

```python
class DeepSeekMoELayer(nn.Module):
    def __init__(self, H, I_r, num_routed, num_shared, top_k):
        super().__init__()
        self.router  = DeepSeekV3Router(H, num_routed, top_k)   # 见第 6 章
        self.routed  = nn.ModuleList([Expert(H, I_r) for _ in range(num_routed)])
        self.shared  = nn.ModuleList([Expert(H, I_r * num_shared) for _ in range(1)])
        # 注: shared expert intermediate 通常合并 (单个更大的 FFN),
        # 因为它没有稀疏性可利用

    def forward(self, x):
        # x: (N, H)
        # 1. shared: 每 token 都算
        y_shared = self.shared[0](x)                             # (N, H)
        # 2. routed: 常规 top-K MoE
        weights, indices, logits = self.router(x)
        y_routed = dispatch_and_combine(x, weights, indices, self.routed)
        # 3. 相加
        return y_shared + y_routed, logits
```

=== 收益与代价

*收益*：
- Specialization 显著↑，同 FLOP 下 downstream loss 更低（2B 规模 DeepSeekMoE 匹配 1.5× 大的 GShard）
- Router 更容易学（选择空间大但不需要 aux loss 强约束）
- Fine-grained 让 aux-loss-free bias tuning 更平滑（$b_i$ 微调影响小）

*代价*：
- Kernel 碎片化——256 个小 expert 的 grouped GEMM 效率比 8 个大 expert 差，通信 / launch overhead 占比↑。这正是 §8 讲的 compute wall——需要 Sync-Free + CUDA Graph + DeepEP V2 才能压回来
- Shared expert 是"额外 dense 计算"，每 token 都算——直接让 activation params 从 $K_r dot P_"routed"$ 增加 $1 dot P_"shared"$

#insight[
  从系统角度看，*fine-grained + shared* 不是纯粹的模型改进——它是"用 kernel 挑战换算法收益"。如果你没有 DeepEP / HybridEP 级别的 a2a 库，256 experts 的通信会拖垮训练。生产选择：$E in [16, 64]$、$K in [2, 4]$ 是 kernel 挑战与算法收益的甜蜜点，$E >= 128$ 需要专门的 infra 团队。
]

== LatentMoE：把 hidden 压掉一部分

=== 核心思路

LatentMoE#footnote[arXiv:2601.18089] 在 routed expert 路径外套两层*共享投影*：

$
x_"lat" = D_"down"(x) in RR^ell,    quad ell = d / alpha, quad "常用" alpha = 4
$
$
y_"lat" = sum_k tilde(w)_k dot "FFN"_(i_k)^"routed"(x_"lat")
$
$
y = D_"up"(y_"lat"),  quad y in RR^d
$

Router 和 shared expert 依然在 $d$ 维上算。变化：

- Dispatch/combine 通信量：$times 1 / alpha$
- Expert 权重读取带宽：$times 1 / alpha$  
- 参数分布：expert weights 从 $O(d^2)$ 变 $O(d dot ell) = O(d^2 / alpha)$——但 $D_"up", D_"down"$ 是全 $d$ 的共享大矩阵

=== 两种变体

+ *ℓ-MoE_eff*：省下的带宽*不复用*，直接减少 GPU 使用（inference-friendly）
+ *ℓ-MoE_acc*：把省下的带宽*换算力*——增加 $N_r' = alpha N_r$ 或 $K_r' = alpha K_r$，*iso-compute* 但精度更高。95BT pretrain (8B activated) 报告 MMLU Pro *+5.65 pp*

=== 与 DeepSeekMoE 的关系

*不是同一件事*。DeepSeekMoE 是 "拆细 + shared"；LatentMoE 是 "投影到低维再做 routed"。可以叠加：LatentMoE + shared expert + fine-grained routed 是 Nemotron-3 的做法。

== Multi-head Latent Attention (MLA)

严格说 MLA 不是 MoE 技术，是 attention 侧的 KV cache 压缩——但 DeepSeek-V2/V3 把它与 MoE 一起用，训练/推理系统必须一起处理，所以放这里。

=== 机制

标准 MHA 每 token cache $2 n_h dot d_h$ 个数值 (K + V)；MLA 把 KV 联合压缩到 latent $c_t in RR^(d_c)$，只 cache $c_t$ 和一个 decoupled RoPE key $k_t^R$：

$
c_t^"KV" = W^"DKV" x_t,  quad c_t in RR^(d_c),  quad d_c = 512  "(V3)"
$
$
K_t, V_t = W^"UK" c_t, W^"UV" c_t  quad "推理时不 materialize, 融合到 attn 里"
$

每 token cache 大小从 $2 n_h d_h L$ (V3 是 128 × 128 × 2 = 32K) 减少到 $(d_c + d_h^R) L = 576$——*~57× 更小*。DeepSeek-V2 相对 DeepSeek 67B (MHA) KV cache 减少 *93.3%*，推理吞吐提升 *5.76×*。

=== 训练精度

MLA 的 down/up projection 参与训练，但*保持 BF16*（不进 FP8 通道）。DeepSeek-V3 §3.3.1 明确列出 attention 和 output head 不做 FP8。

== FP8 训练：DeepSeek-V3 recipe

FP8 训练是 2024-2026 的 MoE 训练主战场。DeepSeek-V3 是第一个大规模 (671B, 14.8T tokens) 端到端 FP8 pretrain 成功的公开工作。

=== 各 tensor 精度选择

DeepSeek-V3 §3.3 的完整精度表：

#figure(
  table(
    columns: (auto, auto, 1.5fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, center, left),
    [*组件*], [*精度*], [*说明*],
    [Linear GEMM (Fprop/Dgrad/Wgrad)], [*FP8 E4M3*], [三路 GEMM 全 FP8，输出 BF16/FP32],
    [Master weight],                    [FP32], [optimizer 持有],
    [Weight gradient],                  [FP32], [batch 内累积],
    [Optimizer state (Adam moment)],    [BF16], [master weight 仍 FP32],
    [Activation cache (Wgrad 用)],      [FP8],  [SwiGLU 输入也 FP8 + recompute],
    [Attention 后 Linear 输入],          [*E5M6* 自定义], [敏感路径，scale 2 的幂],
    [MoE gate / router],                [BF16 / FP32], [高精度保留],
    [Embedding, output head, LN, Attention], [BF16 / FP32], [稳定训练],
    [*MoE dispatch activation*],        [*FP8*], [up-projection 前量化，通信量 *-50%*],
    [*MoE combine (fwd + bwd)*],        [*BF16*], [关键路径保精度],
    [MoE activation grad (down-proj 前)], [FP8], [减 backward 通信],
  ),
  kind: table,
  caption: [DeepSeek-V3 FP8 训练精度分配。原则：*GEMM 用 FP8，reduce/accumulate/router 用高精度*。],
)

=== Fine-grained scaling

FP8 动态范围有限 (E4M3: $[-448, 448]$)，outlier 一多就 clip。DeepSeek 用 *fine-grained tile/block scaling*：

- *Activation*: 1 × 128 tile-wise (per token per 128 channels)
- *Weight*: 128 × 128 block-wise
- *Scale*: 2 的整数幂 (硬件友好)
- *Format*: 全链路 E4M3 (不用混合 E4M3/E5M2)，靠 scaling 扩动态范围

=== TensorCore 累加精度

H100 TensorCore FP8 累加实际只有 ~14 bit，当 K 维大 (K=4096) 时误差可达 *2%*。解决：每 $N_C = 128$ MMA 提升到 CUDA Core *FP32 累加*，再继续。

=== 框架支持

#figure(
  table(
    columns: (auto, 1.5fr, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*框架*], [*机制*], [*Flag*],
    [TransformerEngine], [`Float8BlockScaling` (Hopper), inspired by DS-V3], [`fp8_recipe=blockwise`],
    [TransformerEngine], [`MXFP8BlockScaling`, 32-element scaling (Blackwell)], [`fp8_recipe=mxfp8`],
    [TorchAO],           [MXFP8 EP a2a dispatch/combine 专门 kernel], [`a2a_dispatch_mxfp8_fwd_hp_bwd` 等],
    [Megatron-Bridge],   [--fp8-recipe blockwise + DeepEP dispatcher],  [见 §8 config B],
    [TensorRT-LLM],      [推理侧 EP a2a FP8],                              [`fp8_combine=True`],
    [FP8-Flow-MoE#footnote[arXiv:2511.02302]], [端到端 FP8 dataflow (研究代码)],   [671B 上 *+21% throughput, -16.5 GB peak*],
  ),
  kind: table,
  caption: [FP8 MoE 训练的生产框架。TransformerEngine 的 blockwise recipe 明确写"inspired by DeepSeek-V3"。],
)

=== 精度损失

DeepSeek-V3 报告：FP8 vs BF16 pretrain 相对 val loss error *< 0.25%*（论文 Appendix B.1），在训练随机性范围内。全 671B 未做端到端 ablation（成本原因），只在 16B / DS-V2-Lite 规模验证。

#warn[
  FP8 训练*不是 free lunch*。数值故障常见：Dgrad 用 block-wise quantization 曾导致 DeepSeek 16B 模型在 300B tokens 处发散 (Appendix B.2)。生产建议：先跑一个 BF16 baseline，再用 FP8 做等比训练验证 loss 曲线 diverge < 0.5%，再放大。
]

== 生命周期：Upcycling 与 Pruning / Merging

=== Upcycling: Dense → MoE

Sparse Upcycling#footnote[Komatsuzaki et al., _Sparse Upcycling_, arXiv:2212.05055.] 提出：从预训练 dense checkpoint 出发，*复制* FFN 权重成 $E$ 份 expert，attention/norm 直接拷贝，router 随机初始化。

*收益*：T5/ViT 上 *~50% dense 预训练成本*即可超过原 dense；也超过 from-scratch MoE。

*Megatron flag*：
```bash
--moe-use-upcycling
--moe-upcycling-granularity 1        # 每 dense FFN 拆几份
```

生产应用：Qwen1.5-MoE、GLM-4-MoE 都用了 upcycling。

=== Pruning / Merging: MoE → 小 MoE

推理侧压缩。有两类方法：

+ *Pruning* (去掉少用 expert)：REAP (arXiv:2510.13999) 证明 *pruning >> merging* 在生成任务上；50% 压缩 near-lossless
+ *Merging* (合并相似 expert)：MoE-SVD (ICML 2025)、Sub-MoE (arXiv:2506.23266)——25%/50% 削减保留 96%/86% 精度；小 MoE model routing 更简单

=== Expert 量化 (PTQ, 推理侧)

MoE 特有 PTQ 挑战：expert 激活极度不均衡 → calibration 数据分布 skewed。

- *Router 必须高精度* (W8A8+)：logit 微扰 → 路由翻转 → 灾难性误差传播
- *Shared expert 应分配更高 bit*：每 token 都激活，误差累积快
- *EAQuant* (arXiv:2506.13329)：expert-aware smoothing + router logit alignment，W4A4 avg *+1.15-1.37%* vs DuQuant

== 面试考点

#interview[
  *Q1*: 为什么 DeepSeek 从 Mixtral 的 $E=8, K=2$ 改成 $E=256, K=8$？系统上代价是什么？

  A: 算法上组合数 $10^12 times$，router 学到更精细专业化；系统代价：a2a 通信量不变（只与 $K H$ 有关，与 $E$ 无关，Node-Limited Routing 保证），但 grouped GEMM 的 $M_e$ 更小、更 skewed → kernel 效率下降。所以 DeepSeek 同时需要 DeepEP、Sync-Free、ECHO 才让 fine-grained 跑得动。
]

#interview[
  *Q2*: Shared expert 与 residual (skip) connection 有什么区别？

  A: Residual 是 $y = x + f(x)$，无参数、恒等映射；shared expert 是 $y = "FFN"_s (x) + "MoE"(x)$，有独立参数、每 token 必算。作用是承担"公共知识"（语言基本规律），让 routed expert 专注专业化。DeepSeek-MoE 的实验显示去掉 shared expert 后多数下游 benchmark 显著下降。
]

#interview[
  *Q3*: MLA 和 Grouped Query Attention (GQA) 的对比？

  A: 都是 KV cache 压缩。GQA 是简单的"多个 query head 共享一份 K/V"（cache 减 $n_h / n_g$ 倍）；MLA 是"KV 投影到低秩 latent，推理时只 cache latent"（cache 减 ~57×）。MLA 更激进、精度更高、需要 decoupled RoPE 处理位置，实现更复杂。V3/V2 用 MLA，Llama-3、Mistral、Gemma 用 GQA。
]

#interview[
  *Q4*: DeepSeek-V3 的 FP8 训练里，为什么 MoE combine 保持 BF16 而 dispatch 用 FP8？

  A: Dispatch 是 sender-side operation：token 从本卡出发，能容忍量化误差 (下游还有 GEMM 会"稀释"这个误差)。Combine 是 receiver-side 的加权求和：多份来自不同 expert 的结果相加，*量化误差会累积*（K 项相加）。且 combine 的输出直接送 residual + next-layer attention，误差被放大。所以 combine 必须 BF16。
]

#interview[
  *Q5*: Sparse Upcycling 相比 from-scratch MoE 训练的核心优势？

  A: 复用 dense 已经学到的"通用知识"，MoE 只需要额外学"路由 + 专业化"。等价于给 MoE 一个非常好的 warm start。代价：expert 初始都是 dense FFN 的 copy → 严重相似 → 早期 collapse 风险高，需要 warmup router 或加大 aux loss 系数一段时间。
]

#interview[
  *Q6*: 如果我要压缩一个 8×22B MoE 到能在 8×A100 上推理，选 pruning 还是 merging？

  A: 生成任务优先 *pruning* (REAP 论文结论：merging 会导致 functional subspace collapse，生成任务掉分明显)；判别任务两者接近。实操：先按 router 频次统计各 expert 使用率 → 剪掉最冷 25-50% → 短暂 finetune (~10B tokens) 恢复精度。加上 W4A16 量化能进一步 2× 内存节省。
]
