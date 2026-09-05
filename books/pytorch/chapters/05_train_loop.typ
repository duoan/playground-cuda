#import "../template.typ": *

= 训练循环的每一行

训练循环是面试里最经典的白板题：面试官让你"写一个训练循环"，然后从你写的每一行往下挖。`zero_grad` 为什么放这儿、`set_to_none` 是什么、AMP 和 grad clip 谁先谁后、scheduler 一个 epoch 调一次还是一个 step 调一次——这些问题的答案都藏在这十几行代码里。

这一章把训练循环拆开：梯度的生命周期、混合精度的数值机制、优化器的更新公式、checkpoint 要存什么。autograd 的图机制见第 6 章，DDP 的通信细节见第 18 章，显存分析见第 8 章。

== 骨架：七行代码，每行都有考点

```python
model.train()                                        # 1
for epoch in range(num_epochs):
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)        # 2
        logits = model(x)                            # 3
        loss = criterion(logits, y)                  # 4
        loss.backward()                              # 5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 6
        optimizer.step()                             # 7
        scheduler.step()                             # 8
```

#figure(
  align(center, flow-boxes(boxes: (
    "zero_grad", "forward", "backward",
    "clip_grad", "opt.step", "sched.step",
  ), box-w: 2.0, box-h: 0.8)),
  caption: [一个 step 的固定顺序。`clip_grad` 必须夹在 `backward` 和 `step` 之间——梯度已算好、参数还没更新的那个窗口。],
) <fig-step-order>

/ 1 `model.train()`: 只把所有 module 的 `self.training` 置 `True`。它*不*影响 autograd，只影响 Dropout（是否丢）和 BatchNorm（用 batch 统计还是 running 统计）。
/ 2 `zero_grad`: 清掉上一个 step 的梯度。必须在 `backward()` 之前，因为 autograd 是*累加*到 `.grad` 上的。
/ 3-4 forward + loss: 建 autograd 图，中间激活被挂住（训练显存的主要来源）。loss 必须是标量，否则 `backward()` 要你显式给 `gradient=`。
/ 5 `backward()`: 反向遍历图，把梯度累加进各叶子的 `.grad`，同时*释放*图（除非 `retain_graph=True`）。DDP 的 AllReduce 就挂在这一步的 hook 上。
/ 6 `clip_grad_norm_`: 原地缩放 `.grad`。位置唯一：`backward()` 之后、`step()` 之前。
/ 7 `optimizer.step()`: 读 `.grad`，按更新公式改 `.data`。它*不会*清梯度，这是 `zero_grad` 存在的原因。
/ 8 `scheduler.step()`: 改 `optimizer.param_groups[i]["lr"]`，供下一个 `step()` 使用。必须在 `optimizer.step()` 之后。

#warn[
  `scheduler.step()` 写在 `optimizer.step()` 前面，会让整条 lr 曲线前移一格，torch 会打 `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`——别忽略它。warmup 阶段尤其致命：第一个 step 本该用接近 0 的 lr，颠倒后直接用上第二步的值，配合大 batch 常表现为"训练前 100 步 loss 尖刺"。
]

== `zero_grad(set_to_none=True)` 为什么更好

`set_to_none=True` 是 torch 2.0+ 的默认值。两种做法的区别：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [`set_to_none=False`], [`set_to_none=True`],
  [做了什么], [对每个 `.grad` 跑一次 `zero_()`], [直接 `p.grad = None`],
  [kernel 数], [每个参数一个 memset], [零个 kernel],
  [显存], [grad buffer 全程常驻], [buffer 被释放，forward 峰值期少占一份],
  [`.grad` 累加], [autograd 原地 `add_` 进旧 buffer], [autograd 直接接管新 tensor],
  [没收到梯度的参数], [`.grad` 是 0，*仍会被 `step()` 更新*], [`.grad` 是 `None`，被 `step()` 跳过],
)

最后一行是真正的语义差别，也是面试能答出深度的点：Adam / SGD-with-momentum 在梯度为 0 时*更新量并不为 0*——momentum buffer 还有残值，weight decay 还在作用。所以 `set_to_none=False` 下，本 step 没参与计算的参数（MoE 里没被路由到的 expert、多任务里没用到的 head）依然会被 momentum 推着走、被 weight decay 缩小。`set_to_none=True` 让它们真正不动。

#warn[
  代价是 `.grad` 可能是 `None`，`sum(p.grad.norm() ** 2 for p in model.parameters())` 这类梯度监控代码会 `AttributeError`。自己写的统计要先 `if p.grad is not None` 过滤；`clip_grad_norm_` 内部已经过滤了，不用操心。
]

== 梯度累积

显存装不下大 batch 时，把它切成 `accum_steps` 个 micro-batch，各自 `backward()` 累积梯度，攒满后 `step()` 一次。数学上等价于一个大 batch——*前提是 loss 除以 `accum_steps`*。完整代码见下面的 AMP 组合写法，这里先说四个要点：

+ *`zero_grad` 只在累积窗口边界调*，不能每个 micro-step 都调，否则累积失效。
+ *loss 要除以 `accum_steps`*。`CrossEntropyLoss` 默认 `reduction="mean"`，每个 micro-batch 的 loss 已经是自己那份的均值；直接累加得到 4 倍的均值梯度，等于偷偷把 lr 放大了 4 倍。
+ *`scheduler.step()` 跟着 `optimizer.step()` 走*，不跟 micro-step。所以"总步数"要按 `len(loader) // accum_steps` 算，算错会让 cosine 曲线走不完或提前触底。
+ *clip 在累积完成后做*，clip 的是完整的累积梯度。

#warn[
  DDP 下*每次* `backward()` 都会触发一轮梯度 AllReduce，而前 `accum_steps - 1` 次的通信纯属浪费——反正还要继续累加。用 `ddp_model.no_sync()` 包住非边界的 micro-step（完整写法见下面的组合代码），通信量直接降到 `1/accum_steps`。

  注意最后一个 micro-step *必须*在 `no_sync()` 之外，否则梯度永远不同步、各 rank 参数会 drift。FSDP 也有 `no_sync()`，但它还要控制 unshard/reshard 时机，代价是多存一份完整梯度。
]

#note[
  除以 `accum_steps` 严格等价的前提是各 micro-batch 大小相同。变长序列 + token 级 loss 下有效 token 数不同，严格做法是 `reduction="sum"` 累加、最后除以窗口内总 token 数。这个偏差在长短句混排的数据上是实打实的，但很多代码库直接忽略了。
]

== 混合精度 AMP

AMP 的核心思路：*参数和优化器状态保持 fp32，只把 matmul / conv 这类计算密集的 op 的输入降到 fp16 或 bf16*。`torch.autocast` 是一个 dispatch 层的开关，它按一张内置的 op 列表决定每个 op 用什么精度。

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(x)                # matmul / conv / linear → bf16
    loss = criterion(logits, y)      # cross_entropy → 仍走 fp32
loss.backward()                      # autocast 区间外！
```

=== 哪些 op 在 autocast 下仍走 fp32

不是所有 op 都降精度。低精度*不安全*的两类 op 被 autocast 强制留在 fp32：

- *数值范围敏感的*：`exp`、`pow`、`log`、`softmax`、`log_softmax`、`cross_entropy`、`nll_loss`、`binary_cross_entropy_with_logits`
- *归约类*：`sum`、`norm`、`layer_norm`、`batch_norm`、`group_norm`——归约会把误差累积放大，fp16 累加几千个数很快就失精度

所以你不需要手动给 LayerNorm 或 loss 加 `.float()`，autocast 已经处理了。反过来，*不要*把 `backward()` 放进 autocast 块：反向的 dtype 是前向记录下来的，autocast 只需要包住 forward 和 loss。

#warn[
  `nn.BCELoss` 在 autocast 下会直接报错（"unsafe to autocast"），因为 `sigmoid` 输出在 fp16 下会饱和到精确的 0 或 1，`log(0)` 得到 `-inf`。改用 `BCEWithLogitsLoss`。
]

=== bf16 vs fp16：为什么只有 fp16 需要 GradScaler

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left),
  [], [指数位], [尾数位], [后果],
  [fp32], [8], [23], [基准],
  [fp16], [5], [10], [动态范围只到约 $6 times 10^(-5)$，小梯度直接 underflow 成 0],
  [bf16], [8], [7], [动态范围与 fp32 相同，精度低但不会 underflow],
)

激活的梯度天然很小（尤其深网络的浅层）。fp16 下大量梯度落到 `6e-5` 以下就变成 0，梯度信息直接丢掉——这不是精度问题，是*完全丢失*。bf16 牺牲尾数换来和 fp32 一样的指数范围，所以不会有这个问题，也就不需要 GradScaler。

*选择规则：* Ampere 及更新（A100 / H100）一律用 bf16，代码更简单、不会有 scale 抖动；只有 V100 / T4 这类不支持 bf16 的卡才用 fp16 + GradScaler。

=== GradScaler 的工作原理

思路是"把梯度整体搬到 fp16 能表示的区间里再搬回来"：

#figure(
  align(center, flow-boxes(boxes: (
    "loss × S", "backward", "grad ÷ S",
    "查 inf/nan", "step 或跳过", "更新 S",
  ), box-w: 2.0, box-h: 0.8)),
  caption: [GradScaler 的一个 step。查到 inf/nan 就整步跳过并把 $S$ 减半，连续 2000 步正常则把 $S$ 翻倍。],
) <fig-gradscaler>

+ `scaler.scale(loss)` 把 loss 乘一个缩放因子 $S$（默认初值 65536）。链式法则下所有梯度都被同倍放大，小梯度被抬进 fp16 的可表示区间。
+ `scaler.step(optimizer)` 先把梯度除回 $S$（`unscale_`），再检查有没有 `inf` / `nan`。
+ *有* → 说明 $S$ 太大导致溢出，*整个 step 被跳过*（不调 `optimizer.step()`），并在 `update()` 里把 $S$ 乘 `backoff_factor`（默认 0.5）。
+ *没有* → 正常 `optimizer.step()`；连续 `growth_interval`（默认 2000）步都没溢出，就把 $S$ 乘 `growth_factor`（默认 2.0），试探更大的范围。

所以训练刚开始会看到前几十步被跳过、`scaler.get_scale()` 从 65536 一路降下来，这是正常的自适应过程，不是 bug。但如果*一直*在跳步（`get_scale()` 掉到个位数还在降），说明梯度里有真的 `inf`/`nan`，去查数据和 loss，不是 scaler 的问题。

=== `unscale_` 与 grad clipping 的顺序

#warn[
  *必须先 `unscale_` 再 clip。* 这是 AMP 里最高频的面试题，也是最常见的静默错误。

  梯度在 `backward()` 之后是被放大 $S$ 倍的。直接对放大后的梯度做 `clip_grad_norm_(params, 1.0)`，实际上是把梯度裁到了"真实范数 $1 slash S$"——$S = 65536$ 时相当于 clip 到 `1.5e-5`。每一步的更新量都被压到几乎为零，loss 曲线看起来"在收敛但极慢"，没有任何报错。

  ```python
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)                                   # grad ÷= S
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # 现在是真实尺度
  scaler.step(optimizer)                                       # 不会重复 unscale
  scaler.update()
  ```

  `scaler.step()` 内部会检查该 optimizer 是否已经 unscale 过，所以不会除两次。但 `unscale_(optimizer)` 在一个 step 里对同一个 optimizer *只能调一次*，调两次抛 `RuntimeError`。
]

=== 完整组合：AMP + 梯度累积 + clip + scheduler

白板高频题，这段值得背下来：

```python
import contextlib, torch

USE_FP16 = False                                     # A100/H100 上用 bf16
amp_dtype = torch.float16 if USE_FP16 else torch.bfloat16
scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)   # bf16 时 enabled=False
accum_steps = 4

model.train()
optimizer.zero_grad(set_to_none=True)

for step, (x, y) in enumerate(loader):
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    is_last = (step + 1) % accum_steps == 0

    # DDP：非边界 micro-step 关掉梯度同步
    sync_ctx = contextlib.nullcontext() if is_last else model.no_sync()

    with sync_ctx:
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = criterion(model(x), y) / accum_steps
        scaler.scale(loss).backward()                # bf16 时 scale 是 no-op

    if is_last:
        scaler.unscale_(optimizer)                   # 必须在 clip 之前
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                       # 有 inf/nan 则跳过
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()                             # 与 optimizer.step() 同频
```

`enabled=False` 的 `GradScaler` 所有方法都退化成 no-op，所以这段代码切 bf16 / fp16 只改一个 flag，不用写两条分支。这是面试里可以主动提的工程细节。要监控就打 `grad_norm`、`scheduler.get_last_lr()[0]`、`scaler.get_scale()` 这三个值。

#note[
  `scaler.step()` 跳过某步时 `scheduler.step()` 照样会走，等于"消耗了一步 lr 预算但没更新参数"。几万步的量级下可以忽略。
]

== 梯度裁剪

#formula[
  $ g <- g dot min(1, "max_norm" / (||g||_2 + epsilon)), quad ||g||_2 = sqrt(sum_p ||g_p||_2^2) $
]

`clip_grad_norm_` 的语义是*全局*范数：把所有参数的梯度当成一个拼起来的长向量，算它的 L2 范数，超过 `max_norm` 就整体等比缩放。*方向不变，只改长度*。

`clip_grad_value_(params, v)` 则是逐元素 `clamp(-v, v)`，会改变梯度方向（大分量被砍、小分量不动），几乎只在 RNN 里还偶尔见到。现代训练一律用 `clip_grad_norm_`。

它的返回值是*裁剪前*的总范数（一个 GPU 上的标量 tensor，别在 hot loop 里 `.item()`），是免费的监控信号：稳定贴在 `max_norm` 上说明一直在被裁、clip 值设小了；突然尖刺到几百通常是脏数据或 lr 过大；掉到接近 0 说明梯度消失，或者你在 clip 之前忘了 `unscale_`。

#note[
  FSDP 下*不要*用 `torch.nn.utils.clip_grad_norm_`——每个 rank 只持有梯度的一个 shard，各自算局部范数得到的全局范数是错的。要用 FSDP 实例自己的 `model.clip_grad_norm_(max_norm)`，它内部会 AllReduce 汇总范数。DDP 不受影响，因为 `backward()` 结束时每个 rank 的梯度已经完整且同步过了。
]

== 优化器

=== 更新公式

*SGD + momentum*（PyTorch 的形式，`dampening=0`）：

#formula[
  $ b_t &= mu b_(t-1) + g_t \
    p_t &= p_(t-1) - eta b_t $
]

注意 PyTorch *不*对 $g_t$ 乘 $(1 - mu)$（那是另一种常见写法）。所以换 $mu$ 时有效步长也跟着变，`momentum=0.9` 的有效 lr 大约是 `momentum=0` 的 10 倍。

*Adam*：

#formula[
  $ m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t quad &&"(一阶动量)" \
    v_t &= beta_2 v_(t-1) + (1 - beta_2) g_t^2 quad &&"(二阶动量)" \
    hat(m)_t &= m_t / (1 - beta_1^t), quad hat(v)_t = v_t / (1 - beta_2^t) quad &&"(偏差校正)" \
    p_t &= p_(t-1) - eta hat(m)_t / (sqrt(hat(v)_t) + epsilon) $
]

偏差校正是必要的：$m_0 = v_0 = 0$，前几步被严重低估（$t = 1$ 时 $m_1 = 0.1 g_1$），不除 $1 - beta_1^t$ 就等于自带一个隐式 warmup，且 $beta_2 = 0.999$ 下要几千步才恢复正常。

*AdamW*：只改最后一行，weight decay 不进梯度，直接作用在参数上。

#formula[
  $ p_t = p_(t-1) - eta lambda p_(t-1) - eta hat(m)_t / (sqrt(hat(v)_t) + epsilon) $
]

=== AdamW 与 Adam + L2 的区别

这是必问题。`Adam(weight_decay=λ)` 做的是 *L2 正则*：把 $lambda p$ 加进梯度，

#formula[ $ g_t <- g_t + lambda p_(t-1) $ ]

然后这一项和真实梯度一起被 $sqrt(hat(v)_t)$ 除。后果是*衰减强度被自适应缩放污染了*：梯度历史大的参数（$hat(v)$ 大）被除得多，实际 decay 变弱；梯度小的参数反而被 decay 得狠。这与"weight decay 应该均匀地把所有权重往 0 拉"的初衷相反。

AdamW 的 *decoupled weight decay* 把这一项挪到自适应缩放之外，每个参数每步都被乘上固定的 $(1 - eta lambda)$，与它的梯度历史无关。

实践结论：Transformer 训练一律用 AdamW。注意两者的 `weight_decay` *默认值不同*（`Adam` 是 0，`AdamW` 是 0.01），而且*同一个 λ 在两者下强度完全不同*——从 Adam 迁到 AdamW 时不能照搬这个超参。

=== `param_groups`：谁不该被 weight decay

bias 和归一化层的权重不该被 weight decay：它们不是"连接强度"。把 LayerNorm 的 `weight` 往 0 拉等于削弱这一层的表达能力，而 bias 只有一个自由度、正则它没有防过拟合的意义。判据很简单——*所有 1-D 参数都不 decay*。

```python
decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]

optimizer = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True,
)
```

`param_groups` 里每个 dict 可以覆盖任何超参（`lr` / `betas` / `weight_decay`），分层 lr（backbone 小 lr、head 大 lr）是同一个机制。scheduler 会*按组*分别调 lr，所以 `get_last_lr()` 返回的是一个 list。

=== `foreach` 与 `fused`

一个 7B 模型有几百个参数张量。朴素实现下 `optimizer.step()` 要为每个张量跑好几个 kernel（乘、加、除……），几千次 kernel launch 的 CPU 开销可能比实际计算还长。

#ladder(
  ("for-loop", "逐参数逐 op 起 kernel", "kernel 数 = O(参数数 × op 数)"),
  ("foreach（默认）", "multi-tensor apply，一个 kernel 处理一批张量", "kernel 数降到 O(op 数)，峰值显存略高"),
  ("fused", "整条更新公式融进一个 kernel，读写各一次", "kernel 数最少，且省 HBM 往返"),
)

`foreach=True` 是 CUDA 上的默认行为（参数 dtype/device 一致时）。优化器 step 是纯 memory-bound 的，所以 `fused=True` 省掉的中间量 HBM 往返是实打实的收益，但要显式打开，且限制更多：只支持 CUDA 上的 float 参数，不支持 sparse 梯度，不能配 `differentiable=True`。不确定的话先用默认的 `foreach`，再试 `fused` 并对比 step time。

== 学习率调度

=== warmup 为什么必要

三个独立的原因，能说清任意两个就够：

+ *Adam 的二阶动量在开头不可靠。* $hat(v)_t$ 只由前几个 batch 估出，方差极大；分母不准则更新量不准，此时 lr 若已是峰值，一步就能破坏初始化。这也是 RAdam 的出发点。
+ *大 batch 要配大 lr，而大 lr 在初期最危险。* 随机初始化的网络梯度方向也接近随机，沿随机方向走一大步没有任何好处。
+ *Transformer 的结构性原因。* 残差 + LayerNorm 在初期对 lr 极敏感，Post-LN 不加 warmup 基本训不起来（Pre-LN 好一些但也建议加）。

典型配置：总步数的 1%–5% 做线性 warmup，之后 cosine 衰减到峰值的 10%。

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup, total = 2000, 100_000
sched = SequentialLR(optimizer, milestones=[warmup], schedulers=[
    LinearLR(optimizer, start_factor=1e-3, total_iters=warmup),
    CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=3e-5),
])
```

=== per-step 还是 per-epoch

#warn[
  *这是最容易写错、又最难发现的 scheduler bug：scheduler 的"单位"是 `scheduler.step()` 的调用次数，不是 epoch 也不是 batch。*

  ```python
  # 错：T_max 按 step 数给，却在 epoch 循环里调 step()
  sched = CosineAnnealingLR(optimizer, T_max=100_000)     # 100k steps
  for epoch in range(50):
      for batch in loader:
          ...; optimizer.step()
      sched.step()                    # 只调了 50 次 → cosine 才走了 0.05%
  ```

  结果是 lr 基本没衰减，训练末期还在用峰值 lr，loss 平台期怎么也下不去。反过来把 epoch 级的 `T_max=50` 配 per-step 调用，则 lr 在头 50 步就跌到底，之后全程接近 0。

  规则：*`T_max` / `total_iters` 的单位必须和 `step()` 的调用频率一致*。`OneCycleLR` 和所有 warmup 方案都必须 per-step（`OneCycleLR` 在超出 `total_steps` 时会直接抛异常，算比较友好）。`ReduceLROnPlateau` 是唯一的例外，它 per-epoch 调用且要传指标：`sched.step(val_loss)`。

  自检方法：把 `scheduler.get_last_lr()[0]` 打进日志，看第一步和最后一步的值是否符合预期。这一行日志能挡住 90% 的 scheduler bug。
]

常用调度的选择：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*调度*], [*特点与场景*],
  [linear decay], [从峰值线性降到 0 或某个下限。LLM 预训练常用，简单、可预测],
  [cosine], [先慢后快再慢，末期 lr 很小有利于收敛。视觉和 LLM 都是默认选择],
  [`OneCycleLR`], [warmup 到峰值再退火到极低，配合 momentum 反向摆动。小数据集快速收敛（super-convergence），必须 per-step],
  [constant + warmup], [不衰减。continual pretrain、或者你打算随时从中间 resume 时用],
  [`ReduceLROnPlateau`], [指标不降就砍 lr。适合不知道总步数的场景，但对大规模训练太被动],
)

#insight[
  cosine 的一个实际约束：*曲线依赖总步数*。如果你训到一半想延长训练，原来的 cosine 已经衰减到很低，续训效果会很差；而且从 checkpoint resume 时必须把 `scheduler.state_dict()` 一起恢复，否则 lr 会跳回峰值。想保留"随时延长"的自由度，用 constant + warmup 或 WSD（warmup-stable-decay）这类可以事后决定衰减点的方案。
]

== loss 的几个坑

=== `CrossEntropyLoss` 吃 logits，不吃 probs

`nn.CrossEntropyLoss` 内部就是 `log_softmax` + `nll_loss`，所以它要的是*原始 logits*：`criterion(model(x), y)`，其中 `model(x)` 是 `(N, C)` 的原始分数、`y` 是 `(N,)` 的 int64 类别下标。

#warn[
  最经典的初学者 bug：模型末尾加了 `nn.Softmax(dim=-1)`，又用 `CrossEntropyLoss`，等于做了*两次* softmax。它*不报错*，只是把已经在 $[0, 1]$ 的概率再压一次，logits 差距被极度压缩、梯度变平——训练能跑，但收敛慢得离谱、准确率上不去。同理 `nn.LogSoftmax` + `CrossEntropyLoss` 也是错的（该配 `NLLLoss`）。判断方法：分类模型的 `forward` 末尾*不该有任何激活函数*。

  为什么要融在一起？`log_softmax` 用的是 log-sum-exp 稳定形式（先减 max），而单独 `log(softmax(x))` 在 logits 大时溢出、小时 `log(0)` 得 `-inf`。融合还省掉一次中间张量的 HBM 往返——词表 128k 时那个中间张量是 `(B, T, 128k)`，非常可观。
]

其他几个参数：

/ `ignore_index`: 默认 `-100`。target 等于这个值的位置*既不算 loss 也不进 `reduction="mean"` 的分母*。padding 位的 label 设成 `-100` 就自动被跳过，这是变长序列训练的标准做法（见第 4 章的 collate）。
/ `reduction`: `"mean"`（默认，按有效元素数平均）/ `"sum"` / `"none"`（逐元素 loss，做 per-sample 加权或 token 级分析时用）。
/ `label_smoothing`: 把 one-hot target 软化成 $1 - alpha$ 与 $alpha slash (C - 1)$，抑制过度自信，分类任务常设 0.1。
/ shape 约定: `logits` 要是 `(N, C, d_1, ...)`，*类别维在第 1 维*。LM 训练里通常直接 flatten：`criterion(logits.reshape(-1, V), labels.reshape(-1))`。

=== `BCEWithLogitsLoss` 比 `sigmoid + BCELoss` 稳定

同样的融合逻辑。`BCEWithLogitsLoss` 用的是数值稳定形式：

#formula[
  $ ell = max(x, 0) - x y + log(1 + e^(-|x|)) $
]

这个式子对任意大的 $|x|$ 都不溢出。而 fp32 下 `sigmoid(x)` 在 $x$ 超过 17 左右就返回精确的 `1.0`，接着 `BCELoss` 算 `log(1 - 1.0) = log(0) = -inf`，loss 变 `nan`、梯度全丢；fp16 下这个阈值更低。它还额外带 `pos_weight` 参数处理正负样本不均衡，这是相比手写组合的另一个好处。

== checkpoint：存什么、怎么恢复

#warn[
  只存 `model.state_dict()` 的 checkpoint 是*不能续训*的。恢复训练至少需要下面这些，缺任何一项都会让 resume 后的曲线出现可见的跳变。
]

```python
torch.save({
    "model":     model.state_dict(),        # 不是 model 本身！
    "optimizer": optimizer.state_dict(),    # Adam 的 m/v，缺了等于重新 warmup
    "scheduler": scheduler.state_dict(),    # 缺了 lr 会跳回峰值
    "scaler":    scaler.state_dict(),       # fp16 的 scale 值
    "epoch": epoch, "step": global_step,    # DistributedSampler.set_epoch 要用
    "rng": {"torch": torch.get_rng_state(),           # dropout / shuffle 可复现
            "cuda":  torch.cuda.get_rng_state_all()},
    "config": vars(args),                   # 没有 config 的 ckpt 半年后就是废铁
}, path)
```

*为什么只存 `state_dict()` 而不是整个 `model`：* `torch.save(model)` 会 pickle 模型的*类引用*，加载时必须能 import 到完全同名同路径的类——文件挪个目录、类改个名、重构一下模块结构，ckpt 就废了。而且 pickle 一个对象等于允许执行任意代码，`weights_only=True` 加载不了它。`state_dict()` 只是纯粹的 `dict[str, Tensor]`，与代码解耦。

```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])       # 先建好 optimizer 再 load
scheduler.load_state_dict(ckpt["scheduler"])
start_step = ckpt["step"] + 1
```

/ `map_location="cpu"`: 不写的话 tensor 会被恢复到*保存时所在的设备*。8 卡训练里每个 rank 都 `torch.load` 一份，全部落到 `cuda:0` → 第一张卡直接 OOM。习惯性写 `map_location="cpu"`，再靠 `load_state_dict` 搬到各自的卡上。
/ `weights_only=True`: torch 2.6+ 的默认值，用受限 unpickler 只允许 tensor 和基础类型，杜绝"加载别人的 ckpt 等于执行别人的代码"。副作用是自定义类（包括某些 `argparse.Namespace`）会加载失败，此时把 config 单独存成 json，别为图省事关掉这个开关。

#note[
  两个 key 前缀问题：DDP 包装后 key 带 `module.` 前缀，`torch.compile` 后带 `_orig_mod.` 前缀，跨形态加载会报一堆 missing/unexpected key。干净的做法是保存时就取原始 module。分布式场景更推荐 `torch.distributed.checkpoint`，见第 22 章。
]

== 验证循环

```python
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()                                  # Dropout 关闭，BN 用 running stats
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(x), y)
        total_loss += loss.item() * y.size(0)     # 按样本数加权，别直接平均 batch 均值
        total_n += y.size(0)
    model.train()                                 # ← 切回来！
    return total_loss / total_n
```

`model.eval()` 和 `torch.no_grad()` 是*两件独立的事*，必须都做：

- `model.eval()` 改的是 *module 行为*：Dropout 变恒等，BatchNorm 用 running mean/var 而不是 batch 统计。少了它，验证结果会随 batch 组成波动，而且 BN 的 running stats 会被验证数据污染。
- `torch.no_grad()` 改的是 *autograd 行为*：不建图、不存激活。少了它，结果数值上完全正确，但会白存一整轮激活，验证集稍大就 OOM。

=== `inference_mode` 与 `no_grad`

`torch.inference_mode()` 比 `no_grad()` 更激进：除了不建图，它还关掉了 *version counter* 和 *view tracking*——这两个机制是 autograd 用来检测"你原地改了一个 backward 需要的张量"的。关掉它们省下每个 op 的一点簿记开销，小 kernel 密集的推理里有可测量的收益。

代价是它产出的张量被标记为 *inference tensor*，一旦进入需要 autograd 的计算就报 `RuntimeError: Inference tensors cannot be saved for backward`。

```python
with torch.inference_mode():                    # 纯推理 / 独立验证循环，更快
    preds = model(x)

with torch.no_grad():                           # 蒸馏 teacher：输出要回到训练图
    teacher_logits = teacher(x)
loss = kd_loss(student(x), teacher_logits)      # 换成 inference_mode 这行就报错
```

规则：*张量会不会流回训练图？* 会（伪标签、蒸馏 teacher、RL rollout）→ `no_grad`；不会 → `inference_mode`。

== 常见训练 bug 清单

#warn[
  *忘记 `zero_grad()`。* 梯度会跨 step 累加，等价于一个不断变大的 batch + 不断放大的 lr，loss 先降后炸。症状是"前几十步正常，然后 loss 变 nan"。
]

#warn[
  *在循环里累加 loss 张量。* `total_loss += loss` 累加的是*带 autograd 图的张量*，整个 epoch 的计算图都被 hold 住不释放，显存线性增长直到 OOM。

  正确写法是 `total_loss += loss.item()`（取出 python float，会触发一次 GPU 同步）或 `total_loss += loss.detach()`（留在 GPU 上不同步，最后再 `.item()`）。注意 `.item()` 强制同步 CPU 和 GPU，hot loop 里每步都调会打断 CPU 的提前 launch、让 GPU 出气泡；高频日志用 `detach()` 攒着，隔 N 步再 `.item()`。`clip_grad_norm_` 的返回值同理。
]

#warn[
  *`eval()` 之后忘记切回 `train()`。* 验证完直接继续训练，Dropout 全程关闭、BN 一直用 running stats。训练 loss 会*看起来更低*（Dropout 关了嘛），但正则化没了，泛化变差。这个 bug 极难发现，因为没有任何报错，指标也只是"稍微不对"。养成习惯：`evaluate()` 函数自己在返回前 `model.train()`，或者用 try/finally 保证。
]

#warn[
  *`scheduler.step()` 与 `optimizer.step()` 顺序颠倒*，以及 *scheduler 的单位和调用频率不一致*（见前面 scheduler 一节）。两个都不报错，只让 lr 曲线悄悄地错。

  *把 `retain_graph=True` 当创可贴用。* 遇到 "Trying to backward through the graph a second time" 就加 `retain_graph=True`，等于把显存泄漏当成修复。真正的原因通常是张量跨 step 复用了（RNN 的 hidden state 忘了 `detach()`），或者对同一个 loss 调了两次 `backward()`。先找根因。
]

== 面试考点

#interview[
  *Q1*：`optimizer.zero_grad(set_to_none=True)` 和 `=False` 有什么区别？为什么前者是默认？

  A：`False` 是给每个 `.grad` 跑一次 `zero_()`；`True` 直接置 `None`。三个好处：省掉每个参数一个 memset kernel；grad buffer 被释放，forward 峰值期少占一份显存；语义更对——梯度为 0 时 Adam / momentum 的更新量*不是* 0（momentum 有残值、weight decay 仍在作用），所以 `=False` 会让本 step 没参与计算的参数被继续推着走，`=True` 则让 `step()` 直接跳过它们。代价是 `.grad` 可能是 `None`，自己写的梯度监控要过滤。
]

#interview[
  *Q2*：梯度累积怎么写才对？为什么 loss 要除以 `accum_steps`？

  A：`zero_grad` 移到累积窗口边界，每个 micro-step 只做 `(loss / accum_steps).backward()`，攒满后 clip → `optimizer.step()` → `zero_grad` → `scheduler.step()`。除以 `accum_steps` 是因为 `reduction="mean"` 下每个 micro-batch 的 loss 已经是自己的均值，直接累加会得到 `accum_steps` 倍的梯度，等于偷偷放大了 lr。严格来说这要求各 micro-batch 大小相同；token 级 loss 要改用 `reduction="sum"` 除总 token 数。

  DDP 下还要加一条：每次 `backward()` 都触发一轮梯度 AllReduce，前 `accum_steps - 1` 次纯属浪费。用 `ddp_model.no_sync()` 包住非边界的 micro-step，通信量降到 `1/accum_steps`；最后一个 micro-step 必须在 `no_sync()` 之外，否则梯度永远不同步、各 rank 参数 drift。
]

#interview[
  *Q3*：bf16 和 fp16 训练有什么区别？为什么 bf16 不需要 GradScaler？

  A：fp16 是 5 位指数 + 10 位尾数，动态范围只到约 `6e-5`；bf16 是 8 位指数 + 7 位尾数，动态范围和 fp32 一样。激活梯度天然很小，fp16 下大量梯度直接 underflow 成 0，必须靠 GradScaler 把 loss 放大后再反向。bf16 不会 underflow，所以不需要 scaler；代价是尾数只有 7 位，精度更差，但训练对精度的容忍度远高于对范围的容忍度。A100 / H100 上一律用 bf16。
]

#interview[
  *Q4*：`GradScaler` 是怎么工作的？

  A：`scale(loss)` 把 loss 乘一个因子 $S$（初值 65536），链式法则下所有梯度同倍放大，小梯度被抬进 fp16 可表示区间。`step(optimizer)` 先 `unscale_` 把梯度除回来，再检查 inf/nan：有就*整步跳过*并把 $S$ 乘 0.5；没有就正常 step，连续 2000 步无溢出则把 $S$ 翻倍。所以训练开头看到几十步被跳过、scale 一路下降是正常的自适应；如果 scale 掉到个位数还在降，是数据或 loss 里真有 nan。
]

#interview[
  *Q5*：AMP 下做梯度裁剪，`unscale_` 和 `clip` 的顺序是什么？搞错会怎样？

  A：必须先 `scaler.unscale_(optimizer)` 再 `clip_grad_norm_`。`backward()` 之后梯度是被放大 $S$ 倍的，直接 clip 到 `max_norm=1.0` 相当于把真实梯度裁到 `1/S`（$S = 65536$ 时是 `1.5e-5`），每一步的更新量都被压到接近 0。它*不报任何错*，只表现为"loss 在降但慢得离谱"。`scaler.step()` 会检测到已经 unscale 过，不会重复除；但 `unscale_` 对同一 optimizer 一个 step 只能调一次。
]

#interview[
  *Q6*：AdamW 和 Adam + `weight_decay` 的区别？

  A：`Adam(weight_decay=λ)` 是 L2 正则——把 $lambda p$ 加进梯度，于是这一项也被 $sqrt(hat(v))$ 除，衰减强度被自适应缩放污染：梯度历史大的参数实际 decay 反而弱。AdamW 是 *decoupled weight decay*，把 $-eta lambda p$ 直接作用在参数上，与梯度历史无关，每个参数每步都乘固定的 $(1 - eta lambda)$。Transformer 训练一律用 AdamW，且从 Adam 迁过来时不能照搬 λ（两者默认值也不同：0 vs 0.01）。
]

#interview[
  *Q7*：warmup 为什么必要？

  A：三个原因。一是 Adam 的 $hat(v)_t$ 在前几步只由极少样本估出，方差极大，分母不准时大 lr 一步就能破坏初始化（RAdam 就是针对这点）。二是大 batch 要配大 lr，而初期梯度方向接近随机，沿随机方向走大步没有收益只有风险。三是 Transformer 的残差 + LayerNorm 结构在初期对 lr 极敏感，Post-LN 不加 warmup 基本训不起来。典型配置是总步数的 1%–5%。
]

#interview[
  *Q8*：`CrossEntropyLoss` 前面能不能加 softmax？

  A：不能。它内部就是 `log_softmax` + `nll_loss`，要的是原始 logits。加了 softmax 等于做两次，logits 差距被极度压缩、梯度变平，*不报错*但收敛极慢。融合的另外两个理由：`log_softmax` 用 log-sum-exp 稳定形式，避免 `log(0)` 和溢出；省掉 `(B, T, V)` 这个巨大中间张量的 HBM 往返。同理 `BCEWithLogitsLoss` 优于 `sigmoid + BCELoss`——后者在 logit 大于 17 左右时 `sigmoid` 饱和成精确的 1.0，`log(1-1)` 得 `-inf`。
]

#interview[
  *Q9*：checkpoint 要存哪些东西？为什么不直接 `torch.save(model)`？

  A：至少要 model / optimizer / scheduler / scaler 的 `state_dict()`，加上 epoch 和 global step，再加 RNG state 和 config。少了 optimizer 就丢了 Adam 的 m/v，resume 后相当于重新 warmup；少了 scheduler 则 lr 跳回峰值。不存整个 model 是因为 `torch.save(model)` pickle 的是类引用，改个类名或挪个文件 ckpt 就废了，而且 pickle 等于允许执行任意代码，`weights_only=True` 加载不了。加载时一定写 `map_location="cpu"`，否则 8 个 rank 的 tensor 全落到 `cuda:0` 上 OOM。
]

#interview[
  *Q10*：`no_grad` 和 `inference_mode` 有什么区别？

  A：`no_grad` 只是不建图。`inference_mode` 还额外关掉 version counter 和 view tracking（autograd 用来检测非法原地修改的簿记），所以更快，但产出的张量是 inference tensor，一旦进入需要 autograd 的计算就报 `Inference tensors cannot be saved for backward`。判断标准：张量会不会流回训练图？蒸馏 teacher 输出、伪标签、RL rollout 都会 → 用 `no_grad`；纯推理和独立的验证循环 → 用 `inference_mode`。另外别忘了 `model.eval()` 是另一件独立的事，管的是 Dropout 和 BN 的行为，两者都得做。
]
