#import "../template.typ": *

= 数值稳定性 & Loss 病症学

面试真实场景："上次训练 loss 突然 fly up，跳到 20，然后过一会又回来了，你觉得是什么原因？" 这一章系统整理 loss 病症（spike / drop / NaN / plateau），每种病症都给出：*成因* → *如何诊断* → *如何修复* → *经典案例*。配 `src/distributed_training/14_nan_reproducer.py`（可复现 5 种 NaN）和 `15_training_monitor.py`（监控工具）。

== 训练三种典型故障 & 曲线特征

#figure(
  align(center, line-plot(
    series: (
      ("healthy",   ((0, 10), (100, 5), (200, 3.5), (300, 2.8), (400, 2.4), (500, 2.1))),
      ("spike",     ((0, 10), (100, 5), (200, 3.5), (250, 12), (300, 4), (400, 2.6), (500, 2.2))),
      ("fly up",    ((0, 10), (100, 5), (200, 3.5), (250, 8), (300, 15), (400, 40), (500, 200))),
      ("NaN",       ((0, 10), (100, 5), (200, 3.5), (250, 30), (300, 999))),
    ),
    width: 10, height: 4.2,
    x-label: "step", y-label: "loss",
    y-log: true,
    y-max: 1000, y-min: 1,
    title: "四种 loss 曲线的典型模式",
  )),
  caption: [Healthy: 单调下降。Spike: 尖峰后回落 → gradient outlier 或 data corruption，通常自愈。Fly up: 持续上涨 → LR 太大 / init 差 / 优化 diverge，不可自愈。NaN: 突然 999+ → BF16 overflow / softmax overflow / grad explode / div-by-zero。],
) <fig-loss-patterns>

== NaN 的 7 个主要来源

=== 1. Softmax overflow

`softmax(x)` 计算 `exp(x - max(x)) / sum(...)`。若 `max(x)` 特别大（比如 attention 里 $Q K^T / sqrt(d_h)$ 出 500），BF16 里 `exp(500)` 早已溢出到 inf。

*触发场景*：
- Attention logit 出 outlier
- LM head logit outlier（未 pre-norm）
- Router logit outlier (MoE)

*诊断*：hook 到 softmax 输入，看 max value 是否 > 60 (BF16 exp 上限 ≈ 88, 但 sum 之后除法可能 inf/inf = NaN)。

*修复*：
- Attention: FlashAttention 内部就有 online softmax，天然稳。若用 SDPA 也有 safe softmax
- QK-norm (Chameleon, Cogview): 对 Q, K 分别 LayerNorm，把 QK^T 幅度控在 [-10, 10]
- Router: z-loss (`ST-MoE`): 抑制 logit 幅度
- LM head: 训练时 upcast to FP32 (`with torch.autocast(enabled=False)`)

=== 2. RMSNorm/LN 的 sqrt 域

`sqrt(mean(x^2) + eps)`。若 `mean(x^2)` 特别小 (<< eps) 但也不算 0 → BF16 下 `sqrt(1e-8)` 精度问题；若 `mean(x^2)` 特别大 → BF16 sum overflow → inf → sqrt(inf) = inf。

*诊断*：训练开始时 hook LN 输入，看每层 `mean(x^2)` 分布。

*修复*：
- var 计算 upcast FP32：`var = x.float().pow(2).mean()`
- Sub-Layer Normalization (DeepSeek-V3)：activation 累加后再 LN，抑制 activation drift
- Weight decay 稍大点 (0.1)：防止 weight 无限膨胀让 activation 变大

=== 3. Attention Q · K^T 的 BF16 accumulation

Q, K 是 BF16。$Q K^T$ 里内积 = sum 上千项 BF16 乘积。BF16 mantissa 7 位，累加时后加项被 truncate。真值 100 + 0.01 → 存 100 (0.01 丢失)。累积几百步就漂移。

*触发*：QK-norm 未开，且序列长 (S > 8k)。

*诊断*：nsys 里看 `flash_fwd_kernel` 是否用 FP32 accumulator。

*修复*：
- FlashAttention v2/v3 内部 accumulator = FP32（默认已开）
- 手写 attention: `Q @ K.T` 前 `Q = Q.float(), K = K.float()` 
- 或用 `torch.matmul(Q, K.T, out_dtype=torch.float32)` (Torch 2.4+)

=== 4. Gradient explosion / vanishing

反向路径 gradient 每层缩放 $g_(l-1) = J_l · g_l$。若 spectral norm > 1 → 逐层爆炸 → grad_norm 数千 → NaN。

*触发*：
- init 太大（Xavier 应用错误）
- 深度 Pre-LN 但无 residual scale（用 Wang init 修）
- Activation function 有 exponential region

*诊断*：
- `torch.nn.utils.clip_grad_norm_` 返回值 → 每 step 记录，画曲线
- 应该在 median 0.5-2, spike 时 20-100
- grad_norm > 1000 → NaN 边缘

*修复*：
- Grad clip: `clip_grad_norm_(model.parameters(), max_norm=1.0)`——LLM 标配
- Warmup 拉长
- Init std 缩小
- Wang init on residual output projection

=== 5. `sqrt(v) + eps` division

Adam 里 `m / (sqrt(v) + eps)`. 若 $v = 0$ 且 $m ≠ 0$（第一步 update，且 grad 巧合非零但为 rare 值），$m / "eps"$ 会极大。

*触发*：
- 训练第一步（bias correction 只能补救不完全）
- Resume 时 optim state 损坏

*修复*：
- 用 `eps = 1e-8` (default) 通常够
- Warmup 前几步不做 param update（可选）
- Resume 时验证 optim state hash

=== 6. Loss function 的 log(0)

Cross-entropy：`-log(p)`。若 model 对 target 极度 unconfident → `p ≈ 0` → `log(0) = -inf`。

*触发*：
- Label smoothing 未开且 model 出错
- Vocab 太大且 logit distribution 无 mass 在 target 上

*修复*：
- PyTorch `nn.CrossEntropyLoss` 内部用 log_softmax（safe），不会真的 log(0)
- 若手写 loss，一定用 `F.log_softmax` 而不是 `torch.log(F.softmax(x))`

=== 7. RoPE 里的 sin/cos overflow

RoPE base = 1e6 时，`theta = pos * base ^ (-2i/d)` 在 d=128 中 `theta` 覆盖 [1e-6, 1]。sin/cos 在 BF16 下精度 3 位有效数字——position 大时相位精度不够。

*触发*：长上下文训练 (S > 32k) + base 大 (1M+)

*诊断*：hook RoPE 输出，看 Q_rot 是否有异常大值

*修复*：
- RoPE 部分 FP32: `Q_rot = (Q.float() * cos.float() + rotate_half(Q).float() * sin.float()).to(Q.dtype)`
- YaRN scale 会自动缓解（本来就要 upcast）

== Loss 病症对照表

#table(
  columns: (auto, auto, 1.2fr, 1.8fr, 1.8fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left, left),
  [*病症*], [*何时出现*], [*grad_norm 特征*], [*可能原因 (按概率)*], [*快速修复*],
  [Spike (自愈)],   [任意],           [1 step 20-100×],
    [data outlier; 数值 rare event],
    [skip update; grad clip 1.0],
  [Fly up],          [warmup 后],      [持续 10× median],
    [LR 太大; init 差; 优化 diverge],
    [重启 + LR ×0.5 + warmup ×2],
  [NaN 短时],        [任意],           [NaN 前一步 100×],
    [softmax overflow; grad explode],
    [grad clip + upcast attn + skip step],
  [NaN 持续],        [restart 之后],   [重启即 NaN],
    [ckpt 损坏; optim state 坏; precision 变],
    [从更早 ckpt 恢复],
  [Loss 平坦不降],   [训练后期],       [稳定但很小],
    [LR 太小; 数据重复; capacity 满],
    [rewarming; 换数据; 扩 model],
  [Loss 慢升],       [训练中期],       [median 缓涨],
    [BF16 累积 drift; weight explode],
    [wd 拉大; norm eps ↑; upcast LN],
  [Loss 阶段跳变],   [curriculum 换],  [瞬间 2-3×],
    [数据阶段切换; SFT 起步],
    [rewarming 500 步; 查 data mix],
)

== 经典 loss spike 故事

*GLM-130B (2022)*：训练前期频繁 spike。原因 pre-LN 深层 activation drift + init std 偏大。Fix: Sandwich Norm + emb layer normalization + init std 缩到 0.02 → stable。

*OPT-175B (Meta 2022)*：训练日志里 45+ 次手动 restart。原因不同：hardware failure, loss spike, dataloader stall。Meta 论文 "Training the OPT 175B" 完整记录。教训：*checkpoint 频繁 (每 500 step) + 自动 restart 机制*。

*PaLM (Google 2022)*：出现 loss "step function" 上跳。原因未公开细节，Google 用"skip and retry"策略：spike 前 200 步的 ckpt 恢复 + skip 出问题的 batch。

*Llama 2 (Meta 2023)*：报告全程 loss curve smooth，但内部承认前几次尝试有 spike。修复 = data cleaning (删损坏 batch) + grad clip 1.0.

*DeepSeek-V3 (2024)*：宣称 "training was remarkably stable"。作者归因：Sub-Layer Norm + no-aux-loss balancing + Muon-style init scaling + BF16 with FP8 for weight only.

*Kimi K2 (2025)*：宣称 Muon optimizer 让 loss curve *显著更平*。对比图里 AdamW 有 3 次可见 spike，Muon 0 次。

== BF16 vs FP16 vs FP8

#table(
  columns: (auto, auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, center, left),
  [*format*], [*exp*], [*mantissa*], [*range*], [*LLM 用途*],
  [FP32], [8], [23], [1e-38..1e38], [master weight, m, v, LN var, loss],
  [FP16], [5], [10], [6e-5..65504], [训练主 dtype (V100 时代)],
  [BF16], [8], [7], [1e-38..1e38], [训练主 dtype (A100+)],
  [FP8 E4M3], [4], [3], [~1e-9..448], [forward activation (H100+)],
  [FP8 E5M2], [5], [2], [~1e-15..57344], [backward gradient (H100+)],
  [MXFP4], [—], [—], [block-scaled 4-bit], [推理为主 (2025)],
)

*BF16 vs FP16*：BF16 range 同 FP32 (exponent 8 bit)，精度差 (mantissa 7 bit)。FP16 range 小 (5 bit exp) 但精度稍好。LLM 全用 BF16：range 广不需要 loss scaling。

*FP16 陷阱*：
- Range 只到 65504——activation 大就 overflow
- 需要 loss scaling (`GradScaler`)：`loss.backward()` 前先乘 2^16，让 grad 落在 FP16 表示范围内

*BF16 陷阱*：
- Mantissa 只 7 bit——大数累加时小加项被 truncate
- LN 里 `sum(x^2)` 累加 4k 项时会 drift（用 FP32 upcast 修）
- Adam 里 `sqrt(v)` 精度差

*FP8 陷阱*（H100+ 才有）：
- range 更小；activation 需要 dynamic scaling
- Transformer Engine 用 "delayed scaling" 或 "current scaling" 追踪
- Backward gradient 常用 E5M2 (range 大)，forward 用 E4M3 (精度略高)
- 长训练需要 scale statistics stability，容易出 hidden NaN

== 具体调试脚本

*Step 1: 定位到 layer*

```python
def register_nan_hook(model):
    for name, module in model.named_modules():
        def hook(module, input, output, name=name):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"NaN/Inf at {name}, input stats:")
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            print(f"  input[{i}]: min={inp.min()}, max={inp.max()}, "
                                  f"has_nan={torch.isnan(inp).any()}")
                    raise RuntimeError(f"NaN at {name}")
        module.register_forward_hook(hook)
```

跑一步就能定位到哪个 module 先出 NaN。

*Step 2: 决定是 forward 还是 backward*

- forward hook 抓不到 → NaN 在 backward。
- 加 `torch.autograd.set_detect_anomaly(True)`（慢 5-10×，只 debug 时开），能定位反向哪个 op 出问题。

*Step 3: 决定是 数值 vs 数据*

- 用同 seed 重跑相同 batch → 若 reproduce → 数值问题
- 若不 reproduce → data corruption / race condition

*Step 4: 保存 poisoned batch*

```python
try:
    loss = model(batch)
    loss.backward()
except RuntimeError as e:
    torch.save(batch, "poisoned_batch.pt")
    torch.save(model.state_dict(), "poisoned_model.pt")
    raise
```

后续在小 machine 上 reproduce + 修。

== Loss 病症学：面试答题模板

面试常见问法："上次训练 loss fly up，你怎么办？" 结构化答法：

*Step 1: 观察阶段* — 描述当时看到的现象
- 具体数值 (loss 从 3 跳到 20，5 steps 内)
- grad_norm 表现 (是否也跳)
- 是否 1 次事件还是持续
- 训到了哪 step，什么阶段

*Step 2: 假设分层* — 按概率列可能原因
- LR/optimizer (最可能：warmup 不够 / LR 太大 / batch 变了没调 LR)
- 数据 (batch corrupt / curriculum 切换)
- 数值 (softmax overflow / BF16 drift / RoPE overflow)
- 硬件 (ECC error / NCCL 通信错乱)

*Step 3: 诊断动作*
- 立即 grep 日志：`grep "grad_norm" log`
- 查 checkpoint 之前的 loss 曲线
- 保存 poisoned batch
- 试 reproduce (同 seed + 同 batch)

*Step 4: 修复动作*
- 短期：从前 200 step 的 ckpt 恢复 + skip 出问题的 batch + grad clip
- 中期：如果 fly up 复发 → LR × 0.5 + warmup × 2 + upcast attention
- 长期：加 monitor (grad_norm, act stats)，改用 WSD 更 robust

面试官关心的是*你的思维结构*——不是背下所有可能原因，而是给出可复用的分层排查框架。

== Grad clip 详解

```python
# 标准 idiom
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm.isnan() or total_norm.isinf() or total_norm > 100:
    # skip this update
    optim.zero_grad()
    logger.warning(f"skip step: grad_norm={total_norm}")
    continue
optim.step()
```

*max_norm 选值*：
- LLM pretrain：1.0 (Llama)
- Fine-tune：1.0 或 0.5
- RLHF：0.5 (梯度更 noisy)

*clip 的实现*：`clip_grad_norm_` 计算所有 param 的 grad L2 全局 norm，若 > max_norm，等比例缩到 max_norm。*不是* per-param clip。

*clip on unscaled grad*：AMP + GradScaler 时，clip 必须在 `scaler.unscale_(optim)` 之后：

```python
scaler.scale(loss).backward()
scaler.unscale_(optim)                    # ← 先 unscale
clip_grad_norm_(model.parameters(), 1.0)  # ← 再 clip
scaler.step(optim)                        # scaler 会 skip 若有 inf
scaler.update()
```

== 面试考点

#interview[
  *Q1*：Loss NaN，你从哪里开始查？

  A：分四步：(1) 定位 layer：forward hook 抓第一个出 NaN 的 module；(2) forward vs backward: forward hook 不触发就 `set_detect_anomaly(True)`；(3) 数值 vs 数据：同 seed 重跑相同 batch reproduce？(4) 保存 poisoned batch 到小机器上 debug。
]

#interview[
  *Q2*：BF16 与 FP16 有什么区别？为什么 LLM 都用 BF16？

  A：BF16 exp=8 bit（同 FP32 range），mantissa 7 bit；FP16 exp=5 bit（小 range），mantissa 10 bit。LLM activation 会很大 (attention logit 100+)，FP16 溢出需要 loss scaling；BF16 天然覆盖 → 不需要 GradScaler。代价：BF16 精度低，长累加会 drift（LN var 用 FP32 fix）。A100+ 有硬件 BF16 支持后成主流。
]

#interview[
  *Q3*：Loss 突然 spike 然后回落，正常吗？

  A：*偶发 spike 正常*——data 里有 outlier batch 或数值 rare event。表现：1 step 内 grad_norm 100×，loss 5-10×，下一步就恢复。修：加 grad clip 1.0，或加"若 grad_norm > 100 skip step"逻辑。*持续 spike (每 100 step 一次)* → 系统性问题 (LR 太大 / init 差 / data 有 systematic corruption)。
]

#interview[
  *Q4*：LR 突然 fly up（不 NaN，但持续涨），怎么办？

  A：立即 rollback 到 spike 前 200 step 的 ckpt，然后：
  + Diagnose：grad_norm 是否也持续涨？（是→优化 diverge；否→数据/loss function 问题）
  + LR × 0.5, warmup 步数 × 2 重训
  + 若仍 fly up: 换 init（Wang init on residual）
  + 若仍 fly up: 换 optim (AdamW → Muon or Lion)
  + 若仍 fly up: 换 model 结构（Sub-LN 抑制 activation drift）
]

#interview[
  *Q5*：为什么 attention 里要用 QK-norm？

  A：$Q K^T / sqrt(d_h)$ 的每 entry = sum $d_h$ 项 dot product。BF16 下 sum 会 drift → 出现异常大 entry (500+) → softmax overflow → NaN。QK-norm (Chameleon)：对 Q, K 分别 LayerNorm，把 QK^T 幅度控在 [-10, 10]。Meta / Chameleon / Grok 都用。DeepSeek-V3 不用（改 upcast MLA 内部矩阵）。
]

#interview[
  *Q6*：训了 500B tokens 后 loss 突然开始缓慢上涨（不 spike），怎么办？

  A：典型"BF16 累积 drift"：weight norm 慢慢涨→activation 慢慢涨→LN var 慢慢涨→BF16 sqrt 精度慢慢差。诊断：monitor per-layer `weight.norm()`，若持续单调涨且已 > init 3× → drift。
  修：(1) 加 weight decay (0.1)；(2) LN var 计算 upcast FP32；(3) 若已上 FP8, 检查 scaling factor 是否 stale。
]

#interview[
  *Q7*：FP8 训练稳定吗？你会开吗？

  A：H100+ FP8 训练是 2024-2025 生产成熟技术，DeepSeek-V3、Kimi K2 都用（weight 存 BF16，compute 用 FP8）。关键：
  + `E4M3` for forward activation, `E5M2` for backward grad
  + Delayed scaling (每 step 追一次 max) 或 current scaling
  + Transformer Engine (`TE.Linear`) 已封装
  + Loss curve 与 BF16 几乎一致（差 < 0.5% perplexity）
  开 FP8 的收益：H100 上 forward 快 1.5-1.8×。风险：hidden NaN 需 monitor scale statistics 稳定性。
]

#interview[
  *Q8*：`clip_grad_norm_` 应该在 `scaler.unscale_` 之前还是之后？

  A：*之后*。因为 GradScaler 把 loss × 2^16，grad 也 × 2^16。若 clip 在 unscale 之前，你比的是 scaled grad norm——完全错误。正确顺序：`scale.backward → scaler.unscale_(optim) → clip_grad_norm_ → scaler.step → scaler.update`。BF16 场景不需要 scaler，直接 clip 即可。
]

#interview[
  *Q9*：checkpoint 恢复后 loss NaN，checkpoint 前一步 loss 正常，什么原因？

  A：几个可能：(1) 保存 checkpoint 时 optim state 未 sync；(2) 恢复时未 restore RNG state → dataloader 出不同的 batch；(3) precision 变了（BF16 存的但恢复时用 FP16）；(4) sharding 变了（原来 DP=8 恢复到 DP=4，optim state 无法重组）。
  修：从更早 ckpt 恢复；验证 ckpt integrity（每保存后 loss forward reproduce）；用 `torch.distributed.checkpoint` 而不是 `torch.save` （支持 rank change）。
]

#interview[
  *Q10*：loss stuck 不动，grad_norm 也稳定，什么问题？

  A：几种：
  + LR 已太小（cosine tail）：mini rewarming
  + optim state saturate ($v$ 太大，effective step size 微小)：reset $v$ 到 EMA of recent
  + 数据 curriculum 已"记住"：换 batch mixing / 加新 domain
  + Capacity 用尽：加 model size / expert 数
  + Numerical: BF16 update 太小被 round 掉（每 step $"lr" × m / sqrt(v) < 1e-4$ 就危险）
  Kimi K2 报告在 60% training 时 mini restart，就是防这种 stagnation。
]
