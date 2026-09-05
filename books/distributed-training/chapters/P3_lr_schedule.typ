#import "../template.typ": *

= 学习率调度：从 Cosine 到 WSD

学习率是"训练里唯一的超参"——夸张但接近事实。数据、模型、optimizer 定了之后，schedule 的选择直接决定 loss 曲线形状。这一章从经典 warmup+cosine 讲到 2024 出的 WSD，配 `src/distributed_training/13_lr_schedules.py` 的可视化。

== 为什么需要 LR schedule

*从零开始*：初始化后，网络处于混乱状态，大 LR 会让 gradient 爆炸 → loss NaN。需要 *warmup* 慢慢提到目标 LR。

*收敛后期*：LR 太大让模型在最优点周围振荡，无法收敛到 sharp minimum。需要 *decay* 让 LR 逐步降低。

所以 schedule 的骨架 = `warmup → main phase → decay`。差异在于 main phase 与 decay 的形状。

== warmup: linear 上升到 target LR

最简单：从 0 或小 LR 线性上升到 `peak_lr`，用时 `warmup_steps`。

```python
def linear_warmup(step, warmup_steps, peak_lr):
    if step >= warmup_steps:
        return peak_lr
    return peak_lr * step / warmup_steps
```

*长度*：LLM 常用 500-2000 步 warmup。相当于第一个 `warmup_tokens = warmup_steps × batch_tokens`，Llama 是 2000 步。

*为什么*：pre-LN 网络的 gradient variance 在最初几步爆炸性大（未 update 过的参数 activation 分布不 stable）。warmup 让 optimizer state ($m, v$) 有时间"稳定"到合理值。若开 AdamW `beta_1 = 0.9`，$m$ 半衰期约 10 步——warmup 少于此就没意义。

*Warmup 长度对 loss 的影响*：太短 → loss spike（见 P4）；太长 → 浪费 compute。经验：500 步够小 model (< 7B)，2000 步稳大 model。

== Cosine Annealing

Loshchilov & Hutter 2017。GPT-2/GPT-3 起主流 schedule。

$
"lr"(t) = "lr"_"min" + 0.5 · ("lr"_"peak" - "lr"_"min") · (1 + cos(pi · (t - t_"warm")/(T - t_"warm")))
$

*形状*：warmup 上到 peak → 沿 cosine 曲线降到 `lr_min` (通常 = 0.1 × peak_lr 或 0)。

*为什么用 cosine 不用 linear decay*：Cosine 早期降得慢（保留学习强度），后期降得快（精细化）。Empirical 上 cosine 比 linear 稍好 (~0.5% loss)。原因不清，可能与 loss landscape 曲率相关。

*总步数 `T`*：必须提前定！这是 cosine 的最大缺点——若训练中途想"再多训 20%"，需要重跑 schedule 或接 rewarming。

#figure(
  align(center, line-plot(
    series: (
      ("warmup+cosine (T=10k)", (
        (0, 0.0), (500, 0.5), (1000, 1.0),
        (2000, 0.994), (3000, 0.976), (4000, 0.941), (5000, 0.883),
        (6000, 0.794), (7000, 0.673), (8000, 0.528), (9000, 0.369),
        (10000, 0.1),
      )),
      ("linear decay", (
        (0, 0.0), (500, 0.5), (1000, 1.0),
        (5000, 0.5), (10000, 0.0),
      )),
      ("WSD (D=20%)", (
        (0, 0.0), (500, 0.5), (1000, 1.0),
        (8000, 1.0), (9000, 0.5), (10000, 0.0),
      )),
    ),
    width: 10, height: 4,
    x-label: "step", y-label: "LR / peak_lr",
    title: "三种典型 LR schedule (warmup 1000 步, T=10000)",
  )),
  caption: [warmup+cosine 是 2020-2023 主流；WSD 是 2024 后新宠（保持 peak 更久，最后短 decay）。生成脚本 `src/distributed_training/13_lr_schedules.py`。],
) <fig-lr-schedules>

== Warmup-Stable-Decay (WSD)

Hu et al. 2024 (MiniCPM), 后被 Kimi K2 / Qwen 3 / DeepSeek-V3 采用。

三段式：
+ *Warmup*：与 cosine 相同，比如 2000 步
+ *Stable*：保持 `peak_lr` 不变，占总步数的 80-90%
+ *Decay*：在最后 10-20% 步内 rapid decay 到 0（linear 或 sqrt）

*为什么 WSD 赢 cosine*：
+ *不需提前定 T*：stable 期间随时可决定"再训 100B token"——continual pre-training 友好
+ *中间 checkpoint 可用*：cosine 的中间 ckpt LR 已很低，不适合做 base model；WSD 的 stable ckpt 是 fully-trained-at-peak 状态，可作 branch pretrain 起点
+ *loss 略优*：MiniCPM 论文 empirical: WSD 比 warmup+cosine 低 0.005-0.01 perplexity

*decay 形状*：Kimi K2 用 linear，MiniCPM 试过 `1 - sqrt(x)`，二者接近。核心是"快速降到接近 0"，不像 cosine 一路缓降。

*WSD-Rewarm (阶段续训)*：
- 训了 100B tokens WSD (warmup 2K + stable 80B + decay 20B) → 得 model_1
- 继续训 100B tokens：从 stable 阶段 checkpoint 起 → warmup 短一点 (500 步) → 新一段 stable + decay → model_2
- 关键：*从 stable ckpt 恢复*，不是 decay ckpt——decay 完的 loss landscape 已 "sharpen"，rewarming 效果差

*现状*：
- Kimi K2 (2025): WSD, 20% decay
- Qwen 3: WSD-like，加 curriculum
- DeepSeek-V3: WSD, 分阶段 decay
- Llama 3: 仍 cosine（保守）
- Mistral: cosine
- OLMo 2: WSD 

WSD 是 2024-2025 事实标准替代 cosine。

== Rewarming

若已训模型 M，想 continual pretrain 或 fine-tune，从 M 的最终 LR (通常很低) 直接继续 → loss 不动或退化。需要 *rewarming*：短 warmup 上到新的 peak（比原 peak 低 3-10×）→ decay。

*为什么需要*：原训练结束时 optim state ($m, v$) 已适应"低 LR 小 update"。突然拉高 LR → gradient scale 陡增 → v 需要重新 adapt → loss spike。Warmup 500 步让 $m, v$ 平滑过渡。

*Rewarming 的 LR 选择*：
- Continual pretrain (~10% 原 token 量)：new_peak = 0.3 × original_peak
- Fine-tune (SFT, 1B tokens)：new_peak = 0.1 × original_peak
- RLHF：new_peak = 0.01 × original_peak（RL 目标不同，LR 极小）

== Cyclical LR / SGDR

Smith 2017。周期性重启 cosine：一段 cosine annealing → 突然回到 peak_lr → 再 cosine annealing...

*用途*：CV 领域证明有效（周期性重启帮助跳出 local min）。LLM 不常用（loss landscape 平坦，重启效果差）。

*Kimi K2 的 mini-restart*：训到 60% 时做一次小 rewarming，实测有效防止 loss 停滞——这是 SGDR 的变体。

== 与 Batch Size 的关系：linear scaling / sqrt scaling

*Linear scaling* (Goyal et al. 2017, "Accurate large minibatch SGD")：batch 增大 $k$ 倍，lr 也要增大 $k$ 倍，才保持训练动态一致。这在 CNN (ImageNet) 上被证实。

*Sqrt scaling* (LLM 领域)：batch $k$ 倍 → lr 增大 $sqrt(k)$。为什么不是 linear？
- LLM optimizer 是 Adam-style adaptive，本身对 batch size 变化有 buffer
- Linear scaling 在 LLM 大 batch (>4M tokens) 下会崩

实际 empirical: LLM lr ∝ $"batch"^0.5-0.7$。GPT-3 / Chinchilla 都用类似经验规则。

*Critical batch size* (Kaplan et al. 2020)：超过某个 batch $B_"crit"$，linear scaling 完全失效。$B_"crit"$ 随 model 大小、训练进度增大。Llama-3 405B 训练 batch 从 4M 逐步涨到 16M tokens。

== 与 Model Size 的关系：μP (Muon-friendly)

Yang & Hu 2022 (μP)。传统训练 LR 与 model size 有关（大 model 通常小 LR）。μP：一套 parametrization 让 LR 与 model size *完全无关*——可以在 tiny model (300M) 上调 LR，直接迁移到 huge model (100B) 用相同 LR。

*μP 需要的修改*：
+ Init std scaling：$std ∝ 1/sqrt("fan-in")$
+ Attention scale：$"logits" = Q K^T / d_h$ (不是 $sqrt(d_h)$)
+ Output projection 有额外 $1 / "width_mult"$

*现状*：Cerebras, Kimi K2 报告用 μP。Llama/DeepSeek 未明确用。

== 面试考点

#interview[
  *Q1*：warmup 的作用是什么？没有会怎样？

  A：AdamW 的 $m, v$ 初始为 0，前几步 update 幅度小 + $v$ 估计不准。若直接用 peak LR，gradient scale 与 $v$ 不匹配 → update 过大 → loss spike。Warmup 让 optim state 平滑升到"合理估计"。500-2000 步是 LLM 常规。
]

#interview[
  *Q2*：为什么 WSD 逐渐取代 cosine？

  A：
  + Cosine 需提前定总步数 T——continual pretrain / 中途扩训不友好
  + WSD stable 阶段 checkpoint 可作 base model；cosine 中间 ckpt LR 太低
  + Empirical WSD loss 略优
  + Decay 阶段只 10-20% 步，快
  MiniCPM 2024 first, Kimi K2, DeepSeek-V3, Qwen 3 全用 WSD 或变体。
]

#interview[
  *Q3*：Rewarming 是什么？为什么直接从 M 的低 LR 继续训不行？

  A：Rewarming = 短 warmup 到新 peak LR + decay。直接用 M 最终 LR 继续训：optim state 已适应"小 update"，新数据到来 gradient 会重新变大——$v$ 需要重新 adapt，期间 loss spike / drop。Rewarming 短 warmup (200-500 步) 让 $v$ 平滑升级。
]

#interview[
  *Q4*：batch size 从 1M 涨到 4M，LR 怎么调？

  A：sqrt scaling：$"new_lr" = "old_lr" × sqrt(4) = 2×$。LLM 实测更接近 $"batch"^0.6$，比 linear 保守。若 batch 超过 critical batch (Kaplan 2020)，则完全失效——不再增益。Llama-3 405B batch 从 4M 涨到 16M 就是接近 critical。
]

#interview[
  *Q5*：LR 太大会怎样？太小呢？如何 monitor？

  A：太大 → loss spike / NaN；具体表现：`grad_norm` 突然 10-100×，`weight_norm` 也涨。太小 → loss 缓降但停滞，`grad_norm` 稳定但小。Monitor：
  + `grad_norm` (每 step 记录)：spike > 10× median 即 alarm
  + `weight_norm` per layer：应该缓慢单调增长
  + `loss` 5-step moving average 与 baseline 对比
  见 `src/distributed_training/15_training_monitor.py`。
]

#interview[
  *Q6*：cosine 里 `lr_min` 通常设多少？为什么不是 0？

  A：`lr_min = 0.1 × peak_lr` 是主流。设成 0 会让最后几步 update 为 0——但如果训练还需继续（continual 或 SFT），LR 为 0 意味着彻底停止学习。留 10% 让后续可以温柔续训。DeepSeek 用 `lr_min = 0.1 peak`, Llama-3 也是。
]

#interview[
  *Q7*：μP 是什么？为什么大厂开始关注？

  A：μP (Yang & Hu 2022) 让最优 LR 与 model width 无关。传统训练里每次改 model size 都要重新扫 LR (grid search $O(N)$ runs)，成本极高。μP 允许在 300M model 扫好 LR，直接迁移到 100B。省 10^3× 的 hyper search compute。Cerebras 首推，Kimi K2 采用。Llama/DeepSeek 未公开使用，可能内部已在用但没写论文。
]

#interview[
  *Q8*：训到 80% 时 loss 突然平了，你怎么办？

  A：分层排查：
  + 是否 data distribution shift？（curriculum 到新阶段）
  + 是否 LR 已太小？（cosine tail 会 stuck）— 尝试 mini rewarming
  + 是否 optimizer state 已 saturate？— 尝试重置 $v$ 到 EMA of recent grad^2
  + Batch 是否太小已达 Chinchilla convergence？
  + Model 是否已"记住"数据，进入 overfit？— 加 wd, dropout, 或增数据
  典型 rewarming 方案：LR × 0.3 → warmup 200 步 → 继续 stable，一般 loss 立即恢复下降。
]
