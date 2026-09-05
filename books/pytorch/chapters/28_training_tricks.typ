#import "../template.typ": *

= 训练技巧类手写题

这一类题的共同特点是：*公式不难，细节全是坑*。label smoothing 的分母是 $C$ 还是 $C-1$、SGD 第一步的 momentum buffer 到底等于什么、AdamW 的 `eps` 加在开方前还是开方后 —— 每一个都能让你的实现和 `torch.optim` 差出可观的数值，而且不报错。

好消息是这一类题几乎每一道都能*和官方实现跑数值对齐*。面试时能说出"我写完会用 `assert_close` 和 `F.cross_entropy(label_smoothing=eps)` 对一遍"，比解释十分钟公式都有用。所以下面每题都给对齐方式。

第 5 章讲过训练循环的整体骨架（AMP、`zero_grad`、accum 的写法），这一章只讲*怎么把这些组件自己写出来*。下面的 loss 实现都省略了末尾那段相同的 `reduction` 收尾（`mean` / `sum` / `none` 三分支），只留逐样本的 `(N,)` 张量。配套实现与 33 个断言在 `python/pytorch/interview/test_training_tricks.py`。

== Label Smoothing Cross Entropy

*题目.* 手写 label smoothing 交叉熵，要求与 `F.cross_entropy(label_smoothing=eps)` 数值对齐。

#formula[$ q_c = (1 - epsilon) dot bb(1)[c = y] + epsilon / C, quad C = "类别总数" $]

注意两件事：*真实类的概率是 $1 - epsilon + epsilon\/C$，不是 $1 - epsilon$*；*分母是类别总数 $C$，不是 $C - 1$*。展开后有一个等价且好记的形式 —— *"真标签 CE"与"均匀分布 CE"的凸组合*：

#formula[$ cal(L) = (1 - epsilon) dot "NLL" + epsilon dot 1/C sum_c (-log "softmax"(x)_c) = (1-epsilon) H(p_"true", p) + epsilon H(p_"uniform", p) $]

按这个形式写实现最短：

```python
def label_smoothing_cross_entropy(logits, target, smoothing=0.0):
    logp = F.log_softmax(logits, dim=-1)
    nll    = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N,)
    smooth = -logp.mean(dim=-1)               # (N,) 对均匀分布的交叉熵
    return (1.0 - smoothing) * nll + smoothing * smooth
```

#warn[
  流传极广的那个版本是*另一个 loss*：

  ```python
  q = torch.full_like(logp, eps / (c - 1))     # ✗ 其余类 eps/(C-1)
  q.scatter_(1, target[:, None], 1.0 - eps)    # ✗ 真实类 1-eps
  ```

  它也是合法的平滑分布（不少论文用的就是它），但数值上和 `F.cross_entropy(label_smoothing=eps)` *对不上* —— $epsilon = 0.1$、$C = 5$ 时差值已到 $10^(-2)$ 量级。被追问"你确定分母是 $C$ 吗"，答案是：PyTorch 里就是 $C$，理由是它等价于上面那个凸组合（均匀分布覆盖全部 $C$ 类，包括真实类自己）。要对齐官方就必须用 $C$。
]

*它解决什么问题.* one-hot 目标要求正确类的 logit 与其他类拉开*无穷大*的间距，模型会不断增大权重范数去逼近这个不可达的目标，结果是过度自信（预测概率 0.99 但实际准确率只有 0.8）、校准变差、泛化下降。平滑后最优 logit 间距变成一个*有限值*（让 softmax 输出恰好等于 $q$ 所需的间距）：$Delta = log(q_y \/ q_"other") = log(1 + C(1-epsilon)\/epsilon)$。$C = 1000$、$epsilon = 0.1$ 时 $Delta approx 9.1$ —— 模型有了可达的最优解，正则效果就来自这里。*什么时候不该用*：知识蒸馏时（teacher 的软标签已经在做同样的事，叠加互相干扰）；需要精确概率的检索/排序任务（平滑会系统性压低 top-1 置信度）；标签本身有噪声时（平滑假设标签可信）。

*怎么自证.* 对 `eps` 和 `reduction` 做叉乘对齐，`eps=0` 必须精确退化成普通 CE。`reduction="none"` 一定要覆盖 —— 很多实现只在 `mean` 下对得上，因为把归约写进了公式里。

```python
for eps in (0.0, 0.05, 0.1, 0.3, 0.9):
    for red in ("mean", "sum", "none"):
        torch.testing.assert_close(
            label_smoothing_cross_entropy(logits, target, eps, reduction=red),
            F.cross_entropy(logits, target, label_smoothing=eps, reduction=red))
```

== Focal Loss

*题目.* 手写 focal loss，说明 $gamma$ 和 $alpha$ 各解决什么问题。

#formula[$ p_t = "softmax"("logits")_y, quad "FL" = -alpha_t (1 - p_t)^gamma log p_t $]

```python
def focal_loss(logits, target, alpha=None, gamma=2.0):
    logp  = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N,)
    pt    = logpt.exp()
    loss  = -((1.0 - pt) ** gamma) * logpt
    if alpha is not None:                     # 标量 或 (C,) 逐类别权重
        at = alpha.to(logits.device)[target] if isinstance(alpha, Tensor) \
             else torch.full_like(loss, float(alpha))
        loss = at * loss
    return loss
```

*它解决什么问题.* 一阶段检测器里负样本（背景框）有 $10^5$ 个、正样本只有几十个。问题*不只是数量不均衡*，更致命的是：那些"容易分对的负样本"单个 loss 很小，但*数量太多，求和之后压倒了少数难样本的梯度*，训练被简单样本主导。$(1 - p_t)^gamma$ 这个调制因子把权重从易分样本转移到难分样本：$p_t = 0.9$ 的易分样本在 $gamma = 2$ 时 loss 被压到 $1\/100$，$p_t = 0.1$ 的难样本几乎不受影响。$gamma = 0$ 且 `alpha=None` 时精确退化为（加权）CE。

*$alpha$ 和 $gamma$ 分工不同，别混为一谈.* $alpha$ 是按*类别*的静态权重，处理"数量"不均衡；$gamma$ 是按*样本难度*的动态权重，处理"难度"不均衡。两者正交。RetinaNet 论文里 $alpha = 0.25$、$gamma = 2$ —— 有意思的是最优 $alpha$ 反而*给正样本更低的权重*，因为 $gamma$ 已经大幅压低了负样本，再用大 $alpha$ 就矫枉过正了。

#warn[
  *验证退化时最容易踩的坑：`F.cross_entropy(weight=w)` 的 `reduction="mean"` 是加权平均，不是简单平均。* 它除的是*权重之和* $sum_i w_(y_i)$ 而不是样本数 $N$，所以必须用 `'none'` 或 `'sum'` 才能直接比：

  ```python
  # ✗ 对不上：左边除以 N，右边除以 sum(w[target])
  focal_loss(logits, target, alpha=w, gamma=0.0, reduction="mean")
  F.cross_entropy(logits, target, weight=w, reduction="mean")
  # ✓
  torch.testing.assert_close(
      focal_loss(logits, target, alpha=w, gamma=0.0, reduction="sum"),
      F.cross_entropy(logits, target, weight=w, reduction="sum"))
  ```

  纯 API 细节，但对齐测试写不出来就会以为自己公式错了，白查半小时。
]

*怎么自证.* 除了 $gamma=0$ 的退化，还要测那个核心性质 —— 相对权重发生*两个数量级*的转移，这就是 focal loss 全部的作用：

```python
logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]]); target = torch.tensor([0, 0])
ce = focal_loss(logits, target, gamma=0.0, reduction="none")   # 样本 0 易分
fl = focal_loss(logits, target, gamma=2.0, reduction="none")   # 样本 1 难分
assert (fl[0] / ce[0]) < 0.01      # 易分样本压到 1% 以下
assert (fl[1] / ce[1]) > 0.85      # 难样本基本保留
```

== Mixup 与 CutMix

*题目.* 实现 `mixup` 和 `cutmix`，并解释 `mixup_criterion` 为什么写成两个 CE 的线性组合。

```python
def mixup_data(x, y, alpha=1.0, generator=None):
    lam = 1.0 if alpha <= 0 else float(
        torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(x.shape[0], device=x.device, generator=generator)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam            # (x', y_a, y_b, lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
```

$lambda tilde "Beta"(alpha, alpha)$ 是对称分布，$alpha = 1$ 时就是 $"Uniform"(0,1)$；$alpha$ 越小越倾向抽到接近 0 或 1 的值（"几乎不混合"）。配对用 `randperm(B)` *在 batch 内部*完成，不需要额外取数据 —— 这是这个 trick 极其廉价的原因。原论文每个 batch 采一个标量 $lambda$（逐样本采也能跑、方差更小，但和原实现不一致，说清自己用哪种即可）。

*为什么不直接混合 label.* 因为 CE 对目标分布是*线性*的：

#formula[$ lambda dot "CE"(hat(y), y_a) + (1-lambda) "CE"(hat(y), y_b) = "CE"(hat(y), lambda "onehot"(y_a) + (1-lambda) "onehot"(y_b)) $]

两者*数学上完全相等*。返回 `(y_a, y_b, lam)` 的写法更好：不用把 `y` 展开成 `(B, C)` 的 one-hot（$C$ 大时省不少显存）、对任何 criterion 都适用、天然兼容 `label_smoothing`。*为什么有效*：它强制模型在样本之间做*线性插值*的行为，大幅收缩决策边界附近的置信度，等价于一种数据依赖的正则。代价是目标变难了，训练要更多 epoch 才收敛。

```python
def cutmix_data(x, y, alpha=1.0, generator=None):
    lam = 1.0 if alpha <= 0 else float(
        torch.distributions.Beta(alpha, alpha).sample())
    b, _, h, w = x.shape
    index = torch.randperm(b, device=x.device, generator=generator)

    ratio = math.sqrt(1.0 - lam)              # ← sqrt：控制的是面积比例
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))     # ← 用实际面积修正
    return out, y, y[index], lam, (y1, y2, x1, x2)
```

mixup 是全图*加权叠加*，产生"半透明重影"，不自然；cutmix 是*区域替换*，每个像素都来自某张真实图片，局部统计量正常，对卷积网络更友好，还天然带 Cutout 的遮挡正则。实践中两者常随机二选一交替用（timm 的 `mixup_prob` / `switch_prob`）。

#warn[
  CutMix 的两个必踩点。第一，*`cut_w = W * sqrt(1 - lam)` 里的开方不能忘* —— 要控制的是*面积*比例而不是边长比例，忘了开方替换面积就变成 $(1-lambda)^2$。第二，*$lambda$ 必须用实际面积重算一遍*：框中心均匀采样、边界会被 clip 到图像内，实际替换面积比目标小。不修正的话标签权重和图像内容对不上 —— 这是一种*静默的标签噪声*，不报错、不崩溃，只是精度上不去。自证方法：用"每个样本一个常数值"的假图，断言框内逐位等于配对样本、框外保持原样，且 `lam` 精确等于未替换面积比例。
]

== ModelEMA

*题目.* 实现权重的指数滑动平均，讨论 `decay` 怎么选、buffer 要不要 EMA。

#formula[$ "shadow" arrow.l d dot "shadow" + (1 - d) dot theta $]

```python
class ModelEMA:
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.decay, self.warmup_steps, self.num_updates = decay, warmup_steps, 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def current_decay(self):                  # timm 的 warmup 修正
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.num_updates / self.warmup_steps))

    @torch.no_grad()
    def update(self, model):
        d = self.current_decay()
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - d)      # s = (1-w)*s + w*end，w = 1-d
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)                        # buffer 直接拷贝，不做 EMA
        self.num_updates += 1
```

*为什么有用.* 训练末期参数在最优点附近震荡，EMA 相当于对参数轨迹做低通滤波，近似取"平均权重"，在 ViT / 检测 / 扩散上通常能白捡 0.2--1 个点。和 SWA 的区别是 EMA 给近期权重更高权重（指数衰减），SWA 是等权平均。

#warn[
  *$1\/(1-d)$ 就是有效平均窗口长度* —— `decay=0.9999` 相当于平均最近 1 万步。所以*如果总共只训 5000 步，影子权重还停在初始化附近，评测会惨不忍睹*。这是 EMA 最常见的翻车方式，现象非常迷惑：主模型好好的，EMA 模型像没训过。

  修法就是 warmup 修正：让有效 decay 从 0 慢慢升到目标值，早期几乎完全跟随当前权重。上面用的是 timm 的 $d_t = d(1 - e^(-t\/tau))$；另一种思路是像 Adam 那样做偏差校正 $"shadow"\/(1 - d^t)$。选 `decay` 的经验法则：*让 $1\/(1-d)$ 明显小于总步数*，典型 0.999（窗口 1000 步）到 0.9999（窗口 1 万步）。
]

*buffer 要不要 EMA.* 这是个真实的分歧点。BN 的 `running_mean` / `running_var` 本身已经是滑动平均了，再 EMA 一次相当于二重平滑，会让统计量严重滞后于影子权重对应的激活分布 —— 所以 timm 的默认（也是上面的实现）是*直接拷贝*。也有实现对 buffer 也做 EMA，差别通常很小。但 `num_batches_tracked` 这种*整型* buffer 绝对不能 EMA（会被转成浮点或截断出错），必须直接 copy。无论选哪种，*评测前一定要确认 buffer 被同步过* —— 忘了同步 BN buffer 会导致 EMA 模型精度离奇地差，非常难查。

*怎么自证.* EMA 有一个可解析验证的性质：目标权重固定不动时 shadow 以 $d^t$ 的速度指数逼近它。再加三个端点：`decay=0` 时 shadow 完全等于当前权重；`decay=1` 时永不移动；带 warmup 时第一次 `update` 的有效 decay 是 0，shadow 直接对齐当前权重。

```python
ema = ModelEMA(model, decay=0.9)
for s in ema.shadow.parameters():
    s.zero_()                                 # shadow 从 0 出发，target 固定
for _ in range(20):
    ema.update(model)
for s, t in zip(ema.shadow.parameters(), target):
    torch.testing.assert_close(s, t * (1 - 0.9 ** 20))    # 解析解
```

== Warmup + Cosine Scheduler

*题目.* 实现 linear warmup + cosine 退火到 `min_lr`，给出 `LambdaLR` 版和继承 `LRScheduler` 版。

#formula[$ "lr"(s) = cases(
  "lr"_"base" dot s \/ W & s < W,
  "lr"_min + ("lr"_"base" - "lr"_min) dot 1/2 (1 + cos(pi p)) quad & s >= W\, p = (s - W) / (T - W)
) $]

#figure(
  align(center, line-plot(
    series: (("lr", ((0, 0.0), (5, 0.05), (10, 0.1), (20, 0.097), (30, 0.088),
                     (40, 0.075), (50, 0.059), (60, 0.042), (70, 0.026),
                     (80, 0.013), (90, 0.004), (100, 0.001))),),
    x-label: "step", y-label: "lr",
    title: "base_lr=0.1, warmup=10, total=100, min_lr=0.001",
  )),
  caption: [三个必须对得上的关键点：`step=0` 时 lr 为 0、`step=W` 时正好是 `base_lr`（两段在此处连续）、`step=T` 时正好是 `min_lr`。中点约为 $("lr"_"base" + "lr"_min)\/2$，来自 cosine 的对称性。],
) <fig-warmup-cosine>

`LambdaLR` 版三行就能写完，是实战首选：

```python
def warmup_cosine_lambda(warmup, total, min_ratio=0.0):
    def fn(step):                       # 返回的是 base_lr 的【倍率】
        if warmup > 0 and step < warmup:
            return step / warmup
        p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return fn

sched = torch.optim.lr_scheduler.LambdaLR(opt,
            warmup_cosine_lambda(warmup=2000, total=100_000, min_ratio=0.01))
```

#note[
  `LambdaLR` 的 lambda 返回的是*相对 `base_lr` 的倍率*，所以只能表达"按比例的 `min_lr`"。多个 param group 的 `base_lr` 不同时，各组的绝对 `min_lr` 也会不同。要一个所有组共享的绝对 `min_lr`，就得用下面的继承版。
]

```python
class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0,
                 last_epoch=-1):
        assert 0 <= warmup_steps < total_steps
        self.warmup_steps, self.total_steps, self.min_lr = (
            warmup_steps, total_steps, min_lr)
        super().__init__(optimizer, last_epoch)   # ← 属性必须先设好

    def _lr_at(self, step, base_lr):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * step / self.warmup_steps
        p = (step - self.warmup_steps) / max(1, self.total_steps
                                                - self.warmup_steps)
        p = min(1.0, max(0.0, p))
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * p))

    def get_lr(self):
        return [self._lr_at(self.last_epoch, b) for b in self.base_lrs]
```

#warn[
  三个实现细节，写错了很难发现：

  + *属性必须在 `super().__init__()` 之前赋值*。父类构造函数内部会立刻调一次 `step()` $arrow.r$ `get_lr()`，此时 `self.warmup_steps` 还不存在就是 `AttributeError`。
  + *构造完成后 optimizer 里就已经是 step 0 的 lr 了*，不需要（也不应该）在训练循环开始前额外调一次 `scheduler.step()` —— 多调一次会把整条曲线整体前移一格。
  + `lr(0) = 0` 是 HF transformers 的约定，意味着第一步完全不更新。不想要就改成 `(step + 1) / W` 或设一个 `warmup_start_lr`。
]

*warmup 为什么必要.* 训练初期参数随机、梯度方向噪声极大，大 lr 会直接把参数推飞。对 *Adam 系*更关键：二阶矩 $v$ 在最初几步只有极少样本，估计方差巨大，$1\/sqrt(hat(v))$ 可能是个离谱的大数，导致巨大的更新步长（RAdam 正是为了从理论上修掉这一点提出的，warmup 是它的经验近似）。对 *post-norm Transformer* 更是刚需，没有 warmup 直接发散。大 batch 训练时 lr 按线性法则放大，warmup 长度也要相应加长。

*为什么 cosine 而不是 step decay.* cosine 中段下降平缓、末段快速趋近 0，没有 step decay 那种突变造成的 loss 抖动，而且*只有一个超参*（总步数），不用调 milestone。代价是它*依赖准确的总步数* —— 中途改变训练长度会让曲线整体错位，续训时必须把 `total_steps` 和已完成步数都对上。

== 梯度累积的等价性证明

*题目.* 证明"累积 $K$ 步、每步 `loss/K`"与"一次大 batch"的梯度*数值相同*，并说出例外。

设大 batch 有 $N$ 个样本、切成 $K$ 个*等大*的 micro-batch（每个 $m = N\/K$ 个）。大 batch 的平均 loss 等于各 micro-batch 平均 loss 的算术平均：

#formula[$ cal(L) = 1/N sum_(i=1)^N ell_i = 1/K sum_(k=1)^K underbrace(1/m sum_(i in "batch" k) ell_i, cal(L)_k) $]

梯度是*线性*算子，而 `backward()` 是把梯度*累加*到 `.grad`（见第 6 章），所以对每个 $cal(L)_k \/ K$ 各 backward 一次，累加结果就是 $nabla cal(L)$：

#formula[$ sum_(k=1)^K nabla (cal(L)_k / K) = nabla (1/K sum_k cal(L)_k) = nabla cal(L) $]

两个前提必须说清：*必须除以 $K$*（忘了就相当于把学习率放大 $K$ 倍，梯度恰好差 $K$ 倍）；*micro-batch 必须等大*（最后一个不满的 batch 会让加权变形，正确做法是按样本数加权而不是均匀除 $K$）。

#warn[
  等价性的三类例外：

  + *BatchNorm*。BN 用的是*当前 batch 内*的均值/方差：大 batch 用 $N$ 个样本统计，micro-batch 只用 $N\/K$ 个，归一化后的激活本身就不同，梯度自然对不上。这*不是实现 bug，是数学上就不等价*。工程解法是换 GroupNorm / LayerNorm / RMSNorm（不依赖 batch 统计，等价性立刻恢复），或者用 SyncBN。
  + *任何跨样本的 op*。in-batch 对比损失最典型：CLIP 的负样本数就是 $B - 1$（见第 27 章），把 batch 切成 $K$ 份，每份的负样本只剩 $N\/K - 1$ 个，loss 的定义都变了。三元组挖掘、batch-level 归一化同理。
  + *DDP*。默认每次 `backward()` 都触发一次梯度 AllReduce，累积 $K$ 步就白白通信 $K$ 次。必须用 `no_sync()` 包住前 $K-1$ 步：

  ```python
  for i, batch in enumerate(loader):
      ctx = model.no_sync() if (i + 1) % K != 0 else contextlib.nullcontext()
      with ctx:
          (loss_fn(model(batch)) / K).backward()
      if (i + 1) % K == 0:
          opt.step(); opt.zero_grad(set_to_none=True)
  ```
]

*怎么自证.* 一次大 batch backward 对比 $K$ 次 `loss/K` backward，逐位比 `.grad`。两个配套的*反面*测试同样值得写：忘了除 $K$ 时断言 `p_acc.grad == p_big.grad * K`；插一个 `BatchNorm1d` 时断言 `max_diff > 1e-4`，换成 `LayerNorm` 后等价性立刻恢复。能把"不等价"也测出来，说明你真的理解了边界。

== 手写优化器

*题目.* 继承 `torch.optim.Optimizer` 实现 `step()`，与 `torch.optim.SGD` / `AdamW` 逐位对齐。

=== SGD with momentum

PyTorch 官方文档的伪代码（*顺序很重要*，$lambda$ 是 weight decay、$tau$ 是 dampening）：

#formula[$ g_t = g + lambda theta_(t-1); quad b_t = cases(g_t & t = 1, mu b_(t-1) + (1 - tau) g_t & t > 1); quad theta_t = theta_(t-1) - eta dot cases(g_t + mu b_t & "nesterov", b_t & "else") $]

```python
class MySGDMomentum(torch.optim.Optimizer):
    # __init__ 只是把 lr/momentum/dampening/weight_decay/nesterov 塞进 defaults
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            damp, wd = group["dampening"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)         # 耦合的 L2：加进梯度
                if mu != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = g.clone().detach()   # 第一步：buffer = 梯度本身
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(mu).add_(g, alpha=1 - damp)
                    g = g.add(buf, alpha=mu) if group["nesterov"] else buf
                p.add_(g, alpha=-lr)
```

#warn[
  *三个几乎人人写错的细节。*

  + *第一步的动量 buffer 直接等于梯度本身*，不是 $(1 - tau) g$。PyTorch 就是这么实现的，写成 $(1-tau)g$ 时 `dampening != 0` 的配置第一步就对不上。附带的好性质：`momentum=0.9` 的第一步和纯 SGD 完全一样，可以直接断言。
  + *PyTorch 的动量不是教科书形式*。教科书写 `v = mu*v + lr*g; θ -= v`，PyTorch 是 `v = mu*v + g; θ -= lr*v`。两者*只在 lr 恒定时等价*：PyTorch 版本改 lr 会立刻用新 lr 缩放整个历史动量，教科书版本里旧动量还带着当时的 lr。这解释了 scheduler 与 momentum 的交互 —— cosine 退火末期，PyTorch 版本的有效步长下降得更快。从别的框架搬公式时要留意。
  + *`weight_decay` 是加到梯度上的（耦合的 L2）*，所以它会被动量累积、也会被 lr 缩放。这正是 AdamW 要"解耦"的东西。
]

*momentum 为什么能加速.* 它对梯度做指数滑动平均，在震荡方向上正负相消、在一致方向上累加，有效步长约放大 $1\/(1-mu)$ 倍（$mu = 0.9$ 时约 10 倍）。Nesterov 则是在"预估的下一个位置"求梯度，多一阶修正项。

=== AdamW

#formula[$ theta arrow.l theta (1 - eta lambda); quad m_t = beta_1 m_(t-1) + (1-beta_1) g_t; quad v_t = beta_2 v_(t-1) + (1-beta_2) g_t^2 $]

#formula[$ hat(m) = m_t / (1 - beta_1^t), quad hat(v) = v_t / (1 - beta_2^t), quad theta arrow.l theta - eta dot hat(m) / (sqrt(hat(v)) + epsilon) $]

```python
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]   # 都初始化为 0

                if wd != 0:
                    p.mul_(1.0 - lr * wd)          # 解耦：直接作用在参数上
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)  # eps 在开方之后
                p.addcdiv_(m, denom, value=-lr / bc1)
```

#warn[
  *AdamW 的三个对齐陷阱。*

  + *`eps` 加在开方之后*：PyTorch 是 $hat(m)\/(sqrt(hat(v)) + epsilon)$。有些实现（TF 的 `epsilon_hat`）写成 $hat(m)\/sqrt(hat(v) + epsilon)$，在 $v$ 很小时行为差别明显。对不齐官方实现时先查这里。
  + *解耦衰减量是 `lr * wd * θ`，和 lr 相乘*。所以*调 lr 会连带改变实际衰减强度* —— 做 lr 扫描时这是一个隐藏的耦合变量。（AdamW 原论文的形式其实不乘 lr；PyTorch 选了乘 lr 的版本，因为这样和 scheduler 配合更自然。）
  + *bias correction 不能省*。$m_t$ 是 $t$ 个梯度的加权和、权重之和是 $1 - beta_1^t$，除掉它才是无偏估计。$beta_2 = 0.999$ 时 $1 - beta_2^t$ 在 $t=1$ 只有 0.001，少了这一步第一步的 $v$ 会小 1000 倍、步长直接爆炸。
]

*AdamW vs Adam + L2，再强调一次.* Adam + L2 把 $lambda theta$ *加进梯度*，于是它会被 $1\/sqrt(hat(v))$ 自适应地缩放 —— 后果是*梯度大的参数（$v$ 大）实际受到的衰减反而更小*，梯度小的被衰减得更狠，与"weight decay 应该均匀地把权重拉向 0"的初衷完全相反。AdamW 把衰减从梯度里拿出来直接作用在参数上，与 $v$ 无关、对所有参数一视同仁。等价衰减强度差了 $sqrt(hat(v))$ 倍，所以*从 Adam 换到 AdamW 必须重调 `weight_decay`*（典型值从 $10^(-4)$ 量级变到 $10^(-2)$ 量级）。

=== 怎么自证

对齐测试是这一题的全部价值 —— 在固定数据和初始化上跑 8 步，逐位比参数：

```python
for cfg in [dict(lr=0.1, momentum=0.9),
            dict(lr=0.05, momentum=0.9, weight_decay=0.01),
            dict(lr=0.05, momentum=0.9, dampening=0.2),
            dict(lr=0.1, momentum=0.9, nesterov=True)]:
    got = run(m1, MySGDMomentum(m1.parameters(), **cfg))
    ref = run(m2, torch.optim.SGD(m2.parameters(), **cfg))
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-7)
```

三个更精巧的单点断言，能直接验证公式细节：

- *SGD 第一步 == 纯 SGD*：`MySGDMomentum(lr=0.1, momentum=0.9)` 跑 1 步 $equiv$ `SGD(lr=0.1)` 跑 1 步，证明 buffer 初值就是梯度。
- *AdamW `weight_decay=0` == Adam*：最快的"是否真的解耦了"检查。反过来 `MyAdamW` 与 `Adam(weight_decay=wd)` 必须*不同*（`max_diff > 1e-4`），这正是 AdamW 论文的全部意义。另外*无梯度时参数每步恰好乘 $(1 - eta lambda)$*，解耦衰减是精确可预测的。
- *AdamW 第一步的更新量恰好是 `lr`*：$t=1$ 时 $hat(m) = g$、$hat(v) = g^2$，比值是 $"sign"(g)$，*与梯度大小完全无关*。所以 `lr=0.1` 时参数正好走 0.1，把梯度放大 1000 倍结果不变。这是 bias correction 正确的直接证据（少了它第一步会小 $(1-beta_1)\/sqrt(1-beta_2) approx 3.2$ 倍）。

== 手写 `clip_grad_norm_`

*题目.* 手写全局梯度范数裁剪，与 `torch.nn.utils.clip_grad_norm_` 对齐。

#formula[$ "total" = norm("concat"[g_1, dots, g_n])_p, quad c = min(1, "max_norm" / ("total" + 10^(-6))), quad g_i arrow.l c g_i $]

```python
def my_clip_grad_norm_(params, max_norm, norm_type=2.0):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    norms = torch.stack([g.detach().norm(norm_type).to(device) for g in grads])
    total_norm = norms.norm(norm_type)         # 对"各张量范数"再取一次范数

    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef)
    return total_norm                          # 返回裁剪【前】的范数
```

*"全局"两个字是重点.* 必须先把*所有*参数的梯度当成一个大向量算总范数，再统一缩放。对每个 tensor 单独调 `clip_grad_norm_` 会改变各层梯度之间的*相对比例*，等于扭曲了梯度方向 —— 这是最常见的错误实现。代码里的技巧：$norm_p$ 的可加性让"先算每个张量的范数、再对这些范数取范数"等于"拼起来算一个范数"，不用真的 `torch.cat`（对 70B 模型省一次巨大的内存分配）。

*clip by norm vs clip by value.* by norm 保持梯度*方向*不变、只缩短长度，是默认选择；by value 逐元素截断会改变方向，一般只在特殊场景用。FSDP 下不能直接用这个函数，因为每个 rank 只有梯度的分片，必须用 `FSDP.clip_grad_norm_`（内部对局部范数的平方做 AllReduce，见第 19 章）。

#warn[
  四个必须记住的语义。

  + *只在 `clip_coef < 1` 时缩放*，范数小于阈值时*不放大*（所以要 `clamp(max=1.0)`）。写成无条件乘 `clip_coef` 就变成了"归一化"而不是"裁剪"，小梯度会被放大，训练行为完全不同。
  + *返回的是裁剪前的范数*，不是裁剪后的。所以日志里看到 `grad_norm` 超过 `max_norm` 是正常的 —— 这个返回值是最好用的训练健康度指标（突然的尖峰通常紧跟着 loss spike）。
  + `+ 1e-6` 是防止 `total_norm` 为 0 时除零。
  + *必须在 `backward()` 之后、`optimizer.step()` 之前调用；用 AMP 时还必须先 `scaler.unscale_(optimizer)`* —— 否则裁的是被 loss scale 放大过的梯度，阈值完全失效（见第 5 章）。这是混合精度训练里最经典的坑。
]

*怎么自证.*

```python
for max_norm in (0.01, 0.5, 1e6):                     # 覆盖"要裁"和"不用裁"
    for norm_type in (2.0, 1.0, float("inf")):
        got = my_clip_grad_norm_(list(m1.parameters()), max_norm, norm_type)
        ref = torch.nn.utils.clip_grad_norm_(m2.parameters(), max_norm, norm_type)
        torch.testing.assert_close(got, ref)               # 返回值
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad)   # 裁剪后的梯度

# 全局性：两个张量梯度范数 10 : 0.1，裁剪后比例必须不变、总范数正好等于阈值
my_clip_grad_norm_([a, b], max_norm=1.0)
assert abs((a.grad.norm() / b.grad.norm()) - 100.0) < 1e-4
assert abs(torch.cat([a.grad, b.grad]).norm() - 1.0) < 1e-5
# 对比：分别裁剪，比例被彻底破坏（10:0.1 变成 1:0.1）
```

== 面试考点

#interview[
  *Q1*：写出 PyTorch label smoothing 的确切公式。真实类的概率是多少？为什么能防过拟合？

  A：$q_c = (1-epsilon) bb(1)[c=y] + epsilon\/C$，*分母是类别总数 $C$*，所以真实类的概率是 $1 - epsilon + epsilon\/C$ 而*不是* $1-epsilon$。等价的好记形式是 $(1-epsilon) dot "NLL" + epsilon dot "mean"_c(-log "softmax"_c)$，即"真标签 CE"与"均匀分布 CE"的凸组合。流传很广的"真实类 $1-epsilon$、其余 $epsilon\/(C-1)$"是另一个 loss，也合法但和 `F.cross_entropy(label_smoothing=)` 数值对不上。防过拟合的机理：one-hot 要求正确类 logit 与其他类拉开无穷大间距，模型会不停增大权重范数去追一个不可达的目标；平滑后最优间距变成有限值 $log(1 + C(1-epsilon)\/epsilon)$，有了可达的最优解。不该用的场景是知识蒸馏（teacher 软标签已在做同样的事）、需要精确概率的检索/排序、以及标签本身有噪声时。
]

#interview[
  *Q2*：focal loss 解决什么问题？$gamma$ 和 $alpha$ 分别管什么？

  A：一阶段检测器里负样本 $10^5$ 个、正样本几十个，致命的*不只是数量*，而是"容易分对的负样本"单个 loss 很小但数量太多，求和后压倒了难样本的梯度、训练被简单样本主导。$(1-p_t)^gamma$ 把权重从易分样本转移到难分样本：$p_t=0.9$ 在 $gamma=2$ 时 loss 压到 $1\/100$，$p_t=0.1$ 几乎不动。$alpha$ 是按*类别*的静态权重（管数量），$gamma$ 是按*样本难度*的动态权重（管难度），两者正交。RetinaNet 用 $alpha=0.25,gamma=2$ —— $alpha$ 反而给正样本更低权重，因为 $gamma$ 已经压狠了负样本。
]

#interview[
  *Q3*：怎么验证 focal loss 在 $gamma=0$ 时退化为 CE？有什么陷阱？

  A：`gamma=0` 且 `alpha=None` 时应精确等于 `F.cross_entropy`，三种 reduction 都要过。陷阱在带 `alpha` 的版本：`F.cross_entropy(weight=w)` 的 `reduction="mean"` 是*加权平均*，除的是权重之和 $sum_i w_(y_i)$ 而不是样本数 $N$。所以要验证等价必须用 `reduction='none'` 或 `'sum'`，用 `'mean'` 直接比会以为自己公式错了。
]

#interview[
  *Q4*：`mixup_criterion` 为什么写成 $lambda "CE"(y_a) + (1-lambda)"CE"(y_b)$ 而不是混合 one-hot 标签？CutMix 有哪两个必须做对的细节？

  A：两者*数学上完全相等*，因为 CE 对目标分布是线性的。写成两项之和的好处是不用把 `y` 展开成 `(B, C)` 的 one-hot（$C$ 大时省显存）、对任何 criterion 都适用、天然兼容 `label_smoothing`。CutMix：(1) `cut_w = W * sqrt(1 - lam)` 里*必须开方*，控制的是面积比例不是边长比例；(2) *$lambda$ 必须用实际面积重算* —— 框被图像边界 clip 后实际替换面积更小，不修正就是静默的标签噪声，不报错只是精度上不去。
]

#interview[
  *Q5*：EMA 的 `decay` 怎么选？为什么需要 warmup？buffer 要不要 EMA？

  A：$1\/(1-d)$ 就是有效平均窗口长度，`decay=0.9999` 相当于平均最近 1 万步。所以*总步数只有 5000 时影子权重还停在初始化附近，评测会惨不忍睹* —— 这是 EMA 最常见的翻车，现象是主模型正常、EMA 模型像没训过。warmup 就是修这个冷启动：让有效 decay 从 0 升到目标值（timm 用 $d(1-e^(-t\/tau))$），早期几乎完全跟随当前权重。buffer 通常*直接拷贝*不做 EMA，因为 BN 的 running stats 本身已经是滑动平均、再平滑一次会严重滞后于影子权重对应的激活分布；`num_batches_tracked` 这种整型 buffer 更是绝对不能 EMA。
]

#interview[
  *Q6*：手写 warmup + cosine scheduler 有哪些实现细节？

  A：继承 `LRScheduler` 时*三个属性必须在 `super().__init__()` 之前赋值*，因为父类构造函数会立刻调 `get_lr()`。构造完成后 optimizer 里已经是 step 0 的 lr，*不要*在训练循环前再多调一次 `scheduler.step()`（会把曲线整体前移一格）。三个必须对上的关键点：`step=0` 为 0（HF 的约定）、`step=W` 正好是 `base_lr`（两段连续）、`step=T` 正好是 `min_lr`。`LambdaLR` 版更短但返回的是 `base_lr` 的*倍率*，只能表达按比例的 `min_lr`；要跨 param group 共享绝对 `min_lr` 就得用继承版。
]

#interview[
  *Q7*：证明梯度累积与大 batch 等价。有哪些例外？

  A：大 batch 的平均 loss 等于各等大 micro-batch 平均 loss 的算术平均，$cal(L) = (1\/K)sum_k cal(L)_k$；梯度是线性算子而 `backward()` 是累加到 `.grad`，所以对每个 $cal(L)_k\/K$ 各 backward 一次，累加就是 $nabla cal(L)$。前提是必须除 $K$、micro-batch 必须等大。三类例外：*BatchNorm*（用的是当前 batch 内统计量，数学上就不等价，换 LayerNorm/GroupNorm 立刻恢复）；*任何跨样本的 op*（in-batch 对比损失的负样本数从 $N-1$ 变成 $N\/K-1$，loss 定义都变了）；*DDP*（每次 backward 都触发 AllReduce，必须用 `no_sync()` 包住前 $K-1$ 步，否则通信量白涨 $K$ 倍）。
]

#interview[
  *Q8*：手写优化器时有哪三个细节最容易和 `torch.optim` 对不上？

  A：(1) *SGD 第一步的 momentum buffer 直接等于梯度本身*，不是 $(1-tau)g$ —— 所以 `momentum=0.9` 的第一步和纯 SGD 完全一样。(2) *PyTorch 的动量形式是 `v = mu*v + g; θ -= lr*v`*，不是教科书的 `v = mu*v + lr*g; θ -= v`；两者只在 lr 恒定时等价，lr 变化时 PyTorch 版本会立刻用新 lr 缩放整个历史动量，这解释了 scheduler 与 momentum 的交互。(3) *AdamW 的 `eps` 加在开方之后* $hat(m)\/(sqrt(hat(v))+epsilon)$，而 TF 的 `epsilon_hat` 是 $hat(m)\/sqrt(hat(v)+epsilon)$；另外 PyTorch 的解耦衰减量是 `lr*wd*θ`，*和 lr 相乘*，所以调 lr 会连带改变实际衰减强度。验证手段是在固定数据和初始化上跑 8 步逐位比参数；单点断言用"AdamW 第一步的更新量恰好是 lr、与梯度大小无关"。
]

#interview[
  *Q9*：手写 `clip_grad_norm_` 要注意什么？

  A：四点。*全局*范数 —— 所有参数的梯度当成一个大向量算一个 norm 再统一缩放，逐张量分别裁会改变各层梯度的相对比例、扭曲梯度方向。*只在 `clip_coef < 1` 时缩放*，范数小于阈值不放大，否则就变成"归一化"而不是"裁剪"。*返回裁剪前的范数*，所以日志里看到超过 `max_norm` 是正常的，而且这个值是最好用的健康度指标（尖峰常常紧跟 loss spike）。*顺序*：backward 之后、`step()` 之前；AMP 下必须先 `scaler.unscale_(optimizer)`，否则裁的是被放大过的梯度、阈值完全失效。FSDP 下要用 `FSDP.clip_grad_norm_`，因为每个 rank 只有梯度分片。
]
