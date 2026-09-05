#import "../template.typ": *

= Optimizer：从 SGD 到 Muon

面试常问："为什么 LLM 不用 SGD？"、"AdamW 里 weight decay 和 L2 有什么区别？"、"你听说过 Muon 吗？" —— 这章把工业界见过的 optimizer 都过一遍，重点在*选择理由*和*与分布式的相互作用*，代码实现见 `src/distributed_training/12_optimizers.py`（AdamW / Lion / Muon 手写并与 torch.optim 对拍）。

== 更新规则通用形式

所有 first-order optimizer 长这样：

$
theta_(t+1) = theta_t - eta_t · "update"(g_t, "state"_t)
$

差异在于：
+ 用不用 momentum？
+ 用不用 second moment (Adam-style adaptive LR)？
+ update 是"raw gradient" 还是"归一化后的 direction"？
+ weight decay 加在哪？（loss 里 → L2；update 里 → decoupled = "W" 后缀）

== SGD + momentum

```
v_t = beta * v_{t-1} + g_t
theta_{t+1} = theta_t - lr * v_t
```

*状态*：$v$ 1 份，与参数同 shape。总显存 = $2 P b$（param + momentum）。

*为什么 LLM 不用*：
+ 深 transformer 的 loss landscape 非凸/病态 (ill-conditioned)——不同参数的 gradient scale 差 10^3 级，SGD 无法自适应
+ 无 second moment 意味着 lr 需要精调每层每参数——不现实
+ CNN 时代 SGD+momentum + LR schedule 精调可以打 Adam，但 transformer 已经明确"Adam 家族赢"

*仍在用 SGD 的场景*：（很少）
- 精调 CV 模型
- Segment Anything (Meta) 用 AdamW pretrain + SGD finetune
- 无 second moment 时显存少一半，工业界不太在意（AdamW 显存也扛得住）

== Adam / AdamW

```
m_t = beta1 * m_{t-1} + (1-beta1) * g_t              # 一阶
v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2            # 二阶
m_hat = m_t / (1 - beta1^t)                          # bias correction
v_hat = v_t / (1 - beta2^t)
theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)   # Adam
# AdamW 追加：decoupled weight decay
theta_{t+1} = theta_{t+1} - lr * wd * theta_t
```

*状态*：$m, v$ 各 1 份 FP32 = $8 P$ bytes。加 master weight FP32 ($4P$) = *12 P*。这就是 Ch4 里"opt state = 12 P"的来源。

*bias correction 的意义*：$m_0 = 0$，所以 $m_1 = 0.1 g_1$——第一步 update 幅度太小。除以 $(1 - beta_1^t)$ 消除 warmup。生产代码里若开 LR warmup，bias correction 可省（Adam-w-no-bias-correction 有人试过）。

*AdamW 关键：decoupled weight decay*：
- L2 regularization：`loss = loss + wd * ||theta||^2` → grad + `2 wd * theta` → 进 m, v，被 lr/sqrt(v) 调节
- decoupled wd (AdamW, Loshchilov 2019)：`theta -= lr * wd * theta` 直接减，不经过 m, v

*为什么 decoupled 更好*：L2 时 wd 会被 v 缩放——大 gradient 参数被少 decay（错的）；AdamW 让 wd 独立生效，公平地 shrink 所有参数。GPT-3 起 LLM 全用 AdamW。

*hyper 建议 (LLM 默认)*：
- `betas = (0.9, 0.95)` （不是 0.999！LLM 数据非 iid，$beta_2 = 0.999$ 收敛慢）
- `eps = 1e-8` (BF16 下用 1e-6 也可)
- `wd = 0.1` (Llama), 0.01-0.05 (Qwen), 0.001 (small model)

*为什么 $beta_2 = 0.95$ 不是 0.999*：LLM batch 内数据分布快速变化 (curriculum、packing)，$beta_2 = 0.999$ 会让 $v$ 反映远古的 gradient scale——不 adaptive。$0.95$ 意味着 $v$ 半衰期 ≈ 14 步，足够 track 局部 scale。

== LAMB

You et al. 2020。为解决"大 batch 训练时 Adam 崩溃"设计。

```
m_t, v_t = Adam(...)  # 同 Adam
r_t = m_t / (sqrt(v_t) + eps) + wd * theta_t   # Adam-style direction
# LAMB 特色：layer-wise scaling
phi = ||theta_t|| / ||r_t||       # trust ratio
theta_{t+1} = theta_t - lr * phi * r_t
```

*核心思想*：每一层的 update 大小 = weight norm × direction。避免大 batch 时"小 layer 的 update 相对 weight 太大"。

*适用*：BERT 8K batch → LAMB 让 lr 大 10× 而不崩；Llama-1 报告曾试 LAMB，最终选 AdamW（他们没用极大 batch）。

*现在的实际使用*：Google TPU pod BERT 训练；Anthropic Claude 训练报告提到 LAMB-like；Meta Llama 系列都用 AdamW。工业界 LLM 主流仍 AdamW，LAMB 是备选。

== Adafactor

Shazeer & Stern 2018。为省 optimizer memory 设计——Adam 需要 $O(P)$ 的 $v$，Adafactor 用 rank-1 factorization 把 $v$ 变成 $O(sqrt(P))$。

```
V_t 存为 outer(r_t, c_t)   # r_t: (rows,), c_t: (cols,)
```

*显存节省*：Adam 的 $v$ 从 $4P$ 降到 ~$4 sqrt(P)$，对大 model 显著。T5 (11B on 4×TPU) 因此可行。

*代价*：quality 稍降 (~0.5% perplexity)，一些场景需要 external LR schedule。

*现状*：LLM 时代 ZeRO/FSDP 分片了 $v$，Adafactor 显存优势不再突出。Google T5, PaLM 用；Meta/OpenAI 不用。

== Lion (EvoLved Sign Momentum)

Chen et al. 2023 (Google Brain, symbolic search 找出来的)。极简。

```
# 只用一阶 momentum，且只用 sign
c_t = beta1 * m_{t-1} + (1-beta1) * g_t
theta_{t+1} = theta_t - lr * (sign(c_t) + wd * theta_t)
m_t = beta2 * m_{t-1} + (1-beta2) * g_t     # 更慢的 momentum 用于下步
```

*特点*：
+ 只存一阶 $m$，显存 = $8 P$（省一半 vs AdamW 的 12P；无 master weight 就只 $4P$）
+ `sign()` 让每次 update 的每维幅度都是 `lr`，天然稳
+ 无 $v$ → 不 adaptive，但 sign 起了类似作用

*超参*：`(beta1, beta2) = (0.9, 0.99)`，lr 需 *比 AdamW 小 3-10 倍*（因为 sign 输出 unit norm，AdamW 的 direction 是 $O(1/sqrt(v))$，量级不同）。

*实际效果*：Google 报告 Lion 在 BERT / ViT / language modeling 上打平或略优 AdamW。工业界目前 few use，因为已有 AdamW 且 ZeRO 分片解决了显存问题。

== Muon (Kimi K2)

Bernstein et al. 2024, Jordan et al. 2024 (Muon)。$mu$-P 家族里的 optimizer，Kimi K2 (2025) 全量 pre-train 用。

```
# 只对"矩阵"权重（linear.weight, embedding），维度 ≥ 2
m_t = beta * m_{t-1} + g_t         # heavy-ball momentum
U = orthogonalize(m_t)              # Newton-Schulz iteration on m_t
theta -= lr * U

# 对"向量"权重（norm gain, bias, 1D）：仍用 AdamW
```

*核心*：orthogonalize 的 update。用 Newton-Schulz 5 次迭代把 momentum matrix $m$ 归一化到 SVD 值全 1（近正交）。这让 update 每一层每一维的 scale 一致，不需 second moment。

$
X_(k+1) = 1.5 X_k - 0.5 X_k X_k^T X_k
$

（大约 5 次迭代收敛到 orthogonal）。计算成本：$O(H^3)$ per layer per step——用 BF16 GEMM 在 GPU 上飞快，Kimi K2 报告开销 \<5%。

*为什么"orthogonalize"work*：直觉是 spectral norm bound——把 update 的 spectral norm 限制到 1，避免任何方向的 update 太大 (Adam 的 second moment 本质也在做类似事，但更粗糙)。

*显存*：只需一阶 $m$ + BF16 param = $4P + 2P$ = *6 P*，比 AdamW 少一半。

*适用*：仅 2D 及以上权重（linear/embedding）。1D (LN gain, bias) 仍 AdamW。
- 参数 90%+ 是 linear → Muon 覆盖了大部分
- Muon 的效率增益随 model size 增长

*Kimi K2 报告*：Muon vs AdamW 同 flops 达到相同 loss，*且 loss curve 更平滑*（fewer spikes）。这是 2025 年 optimizer 领域的重要突破。

*开源实现*：https://github.com/KellerJordan/Muon —— Meta/Anthropic 未公开使用，DeepSeek/Qwen 还在 AdamW。

== Sophia

Liu et al. 2023 (Stanford)。用 Hessian diagonal 估计做 second moment，取代 Adam 的 $v = g^2$。

*核心*：Adam 的 $sqrt(v)$ 逼近 Hessian 对角线的一个不精确估计——为什么不直接估 Hessian？Sophia 用 Hutchinson trick 每 $k$ 步（$k=10$）估一次 Hessian diag，其他步复用。

```
h_t = hutchinson_hessian_diag(loss)   # every k steps
theta -= lr * clip(m_t / (h_t + eps), rho)
```

*收益*：Sophia 报告 2× faster than AdamW（reach same loss with half tokens）。但*未复现*—— multiple labs 尝试都没能复现 2× 增益，Meta Llama 团队试后放弃。

*现状*：Sophia 是学术上很有吸引力的方向，但生产未用。面试遇到"你听说过 Sophia 吗"回答"了解，但未复现，业界仍用 AdamW"即可。

== Distributed Shampoo

Anil 2020 (Google)。全 Hessian 二阶方法。为参数矩阵 $W in RR^(m times n)$ 分别维护 $L in RR^(m times m)$ 和 $R in RR^(n times n)$ 两个"factor"，update 时用 $L^(-1/4) g R^(-1/4)$。

*显存*：$O(m^2 + n^2)$ per matrix，可跨 DP shard。
*成本*：矩阵求逆开销大——只更新 factor 每 $k=100$ 步一次。

*使用*：Google 内部用（PaLM 报告提到），未开源标准实现。DeepMind 有类似。

*现状*：与 Sophia 类似——理论优雅，实际 few 使用。

== Optimizer 与 ZeRO/FSDP 的相互作用

*ZeRO-1*：shard optim state ($m, v$，master weight)。每 rank 只更新自己那份 shard 的 param。AllGather 更新后的 param 回来（每 step）。

*ZeRO-2*：还 shard grad。每 rank 只 ReduceScatter 自己那份 grad → 更新自己那份 param → AllGather param。

*ZeRO-3 / FSDP FULL_SHARD*：还 shard param。每层 forward/backward 前 AllGather，用完 free。optim step 仍在 shard 上做。

*关键点*：optimizer.step() *总是在 shard 上跑*。这意味着：
+ Lion/Muon 等新 optimizer 若无 shard-aware 实现，会 assert
+ Muon 的 Newton-Schulz 需要*整个矩阵*——不能只在 shard 上跑！所以 Muon 与 ZeRO-3/FSDP 不兼容。Kimi K2 用 ZeRO-1 + Muon on full matrix per rank
+ 一些老 optimizer (Shampoo, K-FAC) 也有同样问题

*Muon + FSDP 的 workaround*：
+ Muon-DP：在 FSDP shard 内做 Newton-Schulz，效果打折
+ 定期全参 AllGather → Muon on rank 0 → broadcast 回来（通信爆炸）
+ ZeRO-1 only（Kimi K2 选择）——放弃 param/grad shard，只 shard optim state

== Optimizer 状态 offload

Adam/AdamW 12P bytes 是训练最大显存消耗。生产手段：
+ *ZeRO-Offload* (DeepSpeed)：optim state 放 CPU，反向后 stream 上来
+ *CPU-Adam*：在 CPU 上做 Adam update（因为 update 逻辑 memory-bound，CPU 也能扛）
+ *NVMe-Offload*：NVMe SSD 存 optim state (DeepSpeed ZeRO-Infinity)
+ *8-bit optimizer* (bnb, Dettmers)：把 $m, v$ 量化到 8-bit，quality 无损，显存缩 4×

生产选择：AdamW 8-bit + ZeRO-1 + gradient checkpointing 已经能训 70B on 8×80G。

== 心算 optimizer 显存

$16 P$ bytes = AdamW BF16 mixed precision 常见总数：
- 2P param BF16
- 2P grad BF16 (或 4P FP32 grad accum)
- 4P master weight FP32
- 4P m FP32
- 4P v FP32

若上 8-bit optim (`m, v` int8)：$= 2P + 2P + 4P + P + P = 10 P$，省 37.5%。
若上 Muon：$= 2P + 2P + 4P + 4P = 12 P$（无 $v$）。
若上 Lion：$= 2P + 2P + 4P + 4P = 12 P$（无 $v$，同 Muon）。

== 面试考点

#interview[
  *Q1*：AdamW 和 Adam+L2 有什么区别？为什么 LLM 用 AdamW？

  A：L2 把 `wd * ||theta||^2` 加进 loss，wd 项进 grad 后被 $sqrt(v)$ 归一化——大 gradient 的参数被少 decay（错的）。AdamW 把 `-lr * wd * theta` 独立减，让 decay 公平作用于所有参数。LLM 训练 wd = 0.1 时二者相差显著。GPT-3 起全部 AdamW。
]

#interview[
  *Q2*：为什么 LLM 用 $beta_2 = 0.95$ 而不是 0.999？

  A：LLM 数据分布快速变化（curriculum, packing），$beta_2 = 0.999$ 半衰期约 700 步，让 $v$ 太"老"，不 adaptive。$0.95$ 半衰期 ~14 步，track 局部 gradient scale。GPT-3, Llama, DeepSeek 全用 0.95。
]

#interview[
  *Q3*：Muon 是什么？为什么能打 AdamW？

  A：Muon 只用一阶 momentum + Newton-Schulz orthogonalize。update matrix 的 spectral norm 被限制到 1，等价于 per-direction 幅度均匀——比 Adam 的 diagonal second moment 更精细。显存少一半（无 $v$），loss curve 更平滑（Kimi K2 报告）。缺点：只对 2D+ 权重生效，1D 仍需 AdamW；与 FSDP 不兼容（Newton-Schulz 要整矩阵）。
]

#interview[
  *Q4*：LAMB 相比 AdamW 好在哪？现在还用吗？

  A：LAMB = AdamW + layer-wise trust ratio。让每层 update 与 weight norm 比例一致，*适合大 batch* (BERT 8K batch 时 LAMB 让 lr 大 10× 而不崩)。现在 Llama/GPT 用适中 batch (1-4M tokens) + AdamW 已够；LAMB 更多在 TPU pod 训 BERT 场景。
]

#interview[
  *Q5*：8-bit optimizer 怎么做到显存 4× 缩减而 quality 无损？

  A：`m` 和 `v` 是低精度容忍的（本身就是 exponential average）。bnb 用 block-wise quantization：每 2048 元素一 block，用一个 FP32 scale。Adam update 时先 dequantize、update、requantize。Dettmers 2021 论文实测 GPT-2 / GLUE 无损。工业界 (HF `transformers`, torchtitan) 主流选项。
]

#interview[
  *Q6*：ZeRO 分片 optim state 时，如果我用 Lion / Muon 需要注意什么？

  A：Lion 只需一阶 $m$，shard 完全 OK，与 ZeRO-1/2/3 兼容。Muon 的 Newton-Schulz 需*整个 matrix*——ZeRO-3/FSDP FULL_SHARD 下 param 分散在多卡，无法做 orthogonalize。Kimi K2 用 ZeRO-1 (只 shard optim state，param 复制)。若必须 FSDP，只能用 Muon-DP 变体（shard 内 approximate）。
]

#interview[
  *Q7*：为什么 Sophia 论文说 2× 更快但没人用？

  A：Sophia 论文报告 2× faster to reach same loss。Meta / Anthropic / DeepSeek 内部尝试都未复现 2×（只 1.1-1.3×）。可能原因：(1) 论文的 baseline AdamW 未调优；(2) Hutchinson 估 Hessian 噪声大，需要非常大 batch 才稳。目前无 open source 生产实例。面试遇到答"了解，未复现"即可。
]

#interview[
  *Q8*：算算 70B 模型 AdamW BF16 optim state 显存？8-bit 呢？Muon 呢？

  A：
  - AdamW BF16: 16 × 70B = 1120 GB (aggregate; ZeRO-3 分 100 卡后 11 GB / GPU)
  - 8-bit AdamW: 2P + 2P + 4P + 1P + 1P = 10P = 700 GB
  - Muon (BF16 param + FP32 master + FP32 m, no v): 12P = 840 GB
  Muon 比 AdamW 少 25%，比 8-bit AdamW 多 20%。8-bit + FSDP 是最激进的省显存组合。
]
