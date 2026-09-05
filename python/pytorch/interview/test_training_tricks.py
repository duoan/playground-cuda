"""PyTorch 面试手写题：训练技巧类。

这一类题的共同特点是"公式不难，但细节全是坑"，而且几乎每一题都能
**和官方实现跑数值对齐**——把对齐测试写出来，比口头解释有说服力得多。

  1. label_smoothing_cross_entropy  -> 对齐 F.cross_entropy(label_smoothing=)
  2. focal_loss                     -> gamma=0 时退化为 CE
  3. mixup / cutmix
  4. ModelEMA
  5. WarmupCosineScheduler
  6. 梯度累积等价性（附 BatchNorm 例外）
  7. MySGDMomentum / MyAdamW        -> 对齐 torch.optim.SGD / AdamW
  8. clip_grad_norm                 -> 对齐 torch.nn.utils.clip_grad_norm_
"""

import copy
import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler

# =============================================================================
# 1. Label Smoothing Cross Entropy
# =============================================================================


def label_smoothing_cross_entropy(
    logits: Float[Tensor, "N C"],
    target: Int[Tensor, " N"],
    smoothing: float = 0.0,
    reduction: str = "mean",
) -> Float[Tensor, "..."]:
    r"""手写 label smoothing 交叉熵，与 ``F.cross_entropy(label_smoothing=eps)`` 对齐。

    **PyTorch 用的到底是哪个公式？**（这是最容易答错的一点）
    把 one-hot 目标 ``y`` 换成平滑分布 ``q``::

        q_c = (1 - eps) * [c == y]  +  eps / C          # C 是**类别总数**

    注意真实类的概率是 ``1 - eps + eps/C``，**不是 ``1 - eps``**。
    很多人（包括不少博客）写成"真实类 1-eps，其余类 eps/(C-1)"，
    那是另一个变体，数值上和 PyTorch **不一样**。
    展开后 PyTorch 的形式等价于::

        loss = (1 - eps) * NLL  +  eps * mean_c( -log_softmax(x)_c )
             = (1 - eps) * H(p_true, p) + eps * H(uniform, p)

    也就是"真标签的 CE"和"均匀分布的 CE"的凸组合，这个写法最好记
    （见 `test_label_smoothing_two_equivalent_forms`，两种写法数值完全相同）。

    面试要点：**它解决什么问题？**
        one-hot 目标要求正确类的 logit 与其他类拉开**无穷大**的间距，
        模型会不断增大权重范数去逼近这个不可达的目标，导致过度自信、
        校准变差（预测概率 0.99 但实际准确率只有 0.8）、泛化下降。
        平滑后最优 logit 间距是有限值 ``log((C-1)(1-eps)/eps)``，
        模型有了可达的最优解，正则效果来自于此。

    面试要点：**什么时候不该用？**
        - 知识蒸馏时（teacher 的软标签已经提供了同样的作用，叠加会互相干扰）；
        - 需要精确概率的任务（检索/排序），平滑会系统性压低 top-1 置信度；
        - 有噪声标签时反而可能有害（平滑本身就假设标签是可信的）。
    """
    c = logits.shape[-1]
    logp = F.log_softmax(logits, dim=-1)

    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N,)
    smooth = -logp.mean(dim=-1)  # (N,)，即对均匀分布的交叉熵
    loss = (1.0 - smoothing) * nll + smoothing * smooth

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# =============================================================================
# 2. Focal Loss
# =============================================================================


def focal_loss(
    logits: Float[Tensor, "N C"],
    target: Int[Tensor, " N"],
    alpha: float | Float[Tensor, " C"] | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Float[Tensor, "..."]:
    r"""Focal Loss（RetinaNet）。

    公式::

        p_t  = softmax(logits)[target]
        FL   = -alpha_t * (1 - p_t)^gamma * log(p_t)

    面试要点：**它解决什么问题？**
        一阶段检测器里负样本（背景框）能有 10^5 个、正样本只有几十个。
        问题**不只是数量不均衡**，更致命的是：那些"容易分对的负样本"
        单个 loss 很小，但**数量太多，求和之后压倒了少数难样本的梯度**，
        训练被简单样本主导。
        ``(1 - p_t)^gamma`` 这个调制因子的作用是：
        p_t=0.9 的易分样本，gamma=2 时 loss 被压到 1/100；
        p_t=0.1 的难样本几乎不受影响。
        **相当于把 loss 权重从易分样本转移到难分样本上**。

    面试要点：**alpha 和 gamma 分工不同，别混为一谈**
        - ``alpha``：按**类别**的静态权重，处理"数量"不均衡。
        - ``gamma``：按**样本难度**的动态权重，处理"难度"不均衡。
        两者正交，RetinaNet 论文里 ``alpha=0.25, gamma=2``。
        有意思的是最优 alpha 是 0.25（**给正样本更低的权重**），
        因为 gamma 已经大幅压低了负样本，再用大 alpha 会矫枉过正。

    面试要点：``gamma = 0`` 且 ``alpha = None`` 时精确退化为交叉熵——
        这是验证实现正确性最快的方法（见 `test_focal_loss_gamma_zero_is_ce`）。
    """
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N,)
    pt = logpt.exp()

    loss = -((1.0 - pt) ** gamma) * logpt

    if alpha is not None:
        if isinstance(alpha, Tensor):
            at = alpha.to(logits.device)[target]  # 逐类别权重
        else:
            at = torch.full_like(loss, float(alpha))
        loss = at * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# =============================================================================
# 3. Mixup / CutMix
# =============================================================================


def mixup_data(
    x: Float[Tensor, "B ..."],
    y: Int[Tensor, " B"],
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "B ..."], Int[Tensor, " B"], Int[Tensor, " B"], float]:
    r"""Mixup：``x' = lam * x_i + (1 - lam) * x_j``，标签同样混合。

    面试要点：**lam 为什么整个 batch 共用一个标量？**
        原论文就是这么做的（每个 batch 采一个 lam）。
        逐样本采 lam 也能跑且方差更小，但和原实现不一致；
        面试时说清楚自己用的是哪种即可。
        配对方式用 ``randperm(B)`` **在 batch 内部配对**，
        不需要额外取数据，是这个 trick 极其廉价的原因。

    面试要点：**为什么不直接混合标签成一个软标签？**
        因为 ``lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)``
        与 ``CE(pred, lam*onehot_a + (1-lam)*onehot_b)`` **数学上完全相等**
        （CE 对目标分布是线性的）。返回 (y_a, y_b, lam) 的写法更通用：
        它对任何 criterion 都适用，而且不用把 y 展开成 (B, C) 的 one-hot，
        省显存也兼容 label_smoothing。（见 `test_mixup_criterion_equals_soft_label_ce`。）

    面试要点：**为什么有效？**
        它强制模型在样本之间做**线性插值**的行为，
        大幅收缩了决策边界附近的置信度，等价于一种数据依赖的正则。
        代价是训练要更多 epoch 才收敛（目标变难了）。

    Returns:
        (混合后的 x, y_a, y_b, lam)。lam=1 时退化为原样本。
    """
    if alpha <= 0:
        lam = 1.0
    else:
        # Beta(alpha, alpha) 对称；alpha=1 时就是 Uniform(0,1)
        lam = float(torch.distributions.Beta(alpha, alpha).sample())

    b = x.shape[0]
    index = torch.randperm(b, device=x.device, generator=generator)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam


def mixup_criterion(
    criterion,
    pred: Float[Tensor, "B C"],
    y_a: Int[Tensor, " B"],
    y_b: Int[Tensor, " B"],
    lam: float,
) -> Float[Tensor, ""]:
    """对 mixup / cutmix 通用的 loss：两个标签的 loss 按 lam 线性组合。"""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def rand_bbox(h: int, w: int, lam: float, generator: torch.Generator | None = None):
    """按 ``1 - lam`` 的面积比例随机取一个矩形框，返回 (y1, y2, x1, x2)。

    面试要点：``cut_w = W * sqrt(1 - lam)``，用 **sqrt** 是因为
    要控制的是**面积**比例而不是边长比例。忘了开方是常见错误。
    框中心均匀采样、边界 clip 到图像内，所以实际面积会比目标略小——
    因此 lam 必须**用实际面积重新算一遍**（见 `cutmix_data`）。
    """
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy = int(torch.randint(h, (1,), generator=generator))
    cx = int(torch.randint(w, (1,), generator=generator))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def cutmix_data(
    x: Float[Tensor, "B C H W"],
    y: Int[Tensor, " B"],
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
):
    """CutMix：把一块矩形区域整个换成另一个样本的对应区域。

    面试要点：**和 mixup 的区别**
        mixup 是全图**加权叠加**，产生的图像是"半透明重影"，不自然；
        cutmix 是**区域替换**，每个像素都来自某张真实图片，局部统计量正常，
        对卷积网络更友好，且天然带有 Cutout 的遮挡正则效果。
        实践中两者常随机二选一交替用（timm 的 ``mixup_prob`` / ``switch_prob``）。

    面试要点：**lam 一定要用实际面积修正**
        框被图像边界裁掉后实际替换面积小于 ``1 - lam``，
        不修正的话标签权重和图像内容对不上，是静默的标签噪声。

    Returns:
        (混合后的 x, y_a, y_b, 修正后的 lam, bbox)
    """
    lam = 1.0 if alpha <= 0 else float(torch.distributions.Beta(alpha, alpha).sample())
    b, _, h, w = x.shape
    index = torch.randperm(b, device=x.device, generator=generator)

    y1, y2, x1, x2 = rand_bbox(h, w, lam, generator)
    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))  # 用实际面积修正
    return out, y, y[index], lam, (y1, y2, x1, x2)


# =============================================================================
# 4. Model EMA
# =============================================================================


class ModelEMA:
    r"""影子权重的指数滑动平均::

        shadow = decay * shadow + (1 - decay) * current

    面试要点：**为什么有用？**
        SGD 末期参数在最优点附近震荡，EMA 相当于对轨迹做低通滤波，
        近似取"平均权重"，通常能白捡 0.2~1 个点（尤其是 ViT / 检测 / 扩散）。
        它和 SWA 的区别是 EMA 给近期权重更高权重（指数衰减），SWA 是等权平均。
        decay 通常 0.999~0.9999：``1/(1-decay)`` 就是有效平均窗口长度，
        0.9999 相当于平均最近 1 万步——**如果总共只训 5000 步，
        影子权重还停在初始化附近，评测会惨不忍睹**，这是最常见的踩坑。

    面试要点：**warmup 修正是干嘛的？**
        就是为了解决上面那个冷启动问题：让 decay 从 0 慢慢升到目标值，
        早期几乎完全跟随当前权重，后期才真正开始平均。
        本实现用 ``decay_t = decay * (1 - exp(-t / tau))``（timm 的做法）。
        另一种思路是像 Adam 那样做偏差校正 ``shadow / (1 - decay^t)``。

    面试要点：**buffer（BN 的 running_mean/var）要不要 EMA？**
        这是个真实的分歧点：
        - **直接拷贝**（本实现默认，也是 timm 的默认）：
          running stats 本身已经是滑动平均了，再 EMA 一次相当于二重平滑，
          会让统计量严重滞后于影子权重对应的激活分布。
        - **也做 EMA**：某些实现这么做，差别通常很小。
        - 但 ``num_batches_tracked`` 这种**整型** buffer 绝对不能 EMA
          （会被转成浮点或截断出错），必须直接 copy。
        无论选哪种，评测前一定要确认 buffer 被同步过——
        忘了同步 BN buffer 会导致 EMA 模型精度离奇地差，非常难查。
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0
    ) -> None:
        assert 0.0 <= decay <= 1.0
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.num_updates / self.warmup_steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            # lerp_(end, w) 等价于 s = s + w * (end - s) = (1-w)*s + w*end
            s.lerp_(p.detach(), 1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)  # buffer 直接拷贝，不做 EMA（见 docstring）
        self.num_updates += 1


# =============================================================================
# 5. Warmup + Cosine Scheduler
# =============================================================================


class WarmupCosineScheduler(LRScheduler):
    r"""线性 warmup + cosine 退火到 ``min_lr``。

    公式::

        step < W:   lr = base_lr * step / W
        step >= W:  progress = (step - W) / (T - W)
                    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))

    面试要点：**warmup 为什么必要？**
        - 训练初期参数随机，梯度方向噪声极大，大学习率会直接把参数推飞。
        - 对 **Adam 系**更关键：二阶矩 ``v`` 在最初几步样本极少、估计方差巨大，
          ``1/sqrt(v)`` 可能是个离谱的大数，导致巨大的更新步长
          （RAdam 正是为了从理论上修掉这一点而提出的，warmup 是它的经验近似）。
        - 对 **post-norm Transformer** 更是刚需，没有 warmup 直接发散。
        - 大 batch 训练时 lr 按线性法则放大，warmup 长度也要相应加长。

    面试要点：**为什么 cosine 而不是 step decay？**
        cosine 在中段下降平缓、末段快速趋近 0，没有 step decay 那种
        突变造成的 loss 抖动，且**只有一个超参**（总步数），
        不用调 milestone。注意它**依赖准确的总步数**——
        中途改变训练长度会让曲线整体错位。

    实现细节：
      - ``lr(0) = 0``（HF transformers 的约定）。若不希望第一步完全不更新，
        可以改成 ``(step + 1) / W``，或设一个 ``warmup_start_lr``。
      - ``step = W`` 时两段的值都等于 ``base_lr``，曲线连续。
      - LRScheduler 的 ``__init__`` 内部会调一次 ``step()`` 把
        ``last_epoch`` 置 0 并立刻把 lr 写进 optimizer，
        所以**构造完成后 optimizer 里就已经是 step 0 的 lr 了**，
        不需要（也不应该）在训练循环开始前额外调一次 ``scheduler.step()``。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        assert 0 <= warmup_steps < total_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        # 这三个属性必须在 super().__init__ 之前设好：父类会立刻调 get_lr()
        super().__init__(optimizer, last_epoch)

    def _lr_at(self, step: int, base_lr: float) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> list[float]:
        return [self._lr_at(self.last_epoch, base) for base in self.base_lrs]


# =============================================================================
# 6. 手写优化器
# =============================================================================


class MySGDMomentum(torch.optim.Optimizer):
    r"""手写 SGD with momentum，与 ``torch.optim.SGD`` 逐位对齐。

    PyTorch 的更新公式（照抄官方文档的伪代码，顺序很重要）::

        g_t = grad + weight_decay * theta_{t-1}          # L2 正则融进梯度
        if t == 1:  b_t = g_t                            # 第一步直接用梯度
        else:       b_t = mu * b_{t-1} + (1 - dampening) * g_t
        if nesterov: g_t = g_t + mu * b_t
        else:        g_t = b_t
        theta_t = theta_{t-1} - lr * g_t

    面试要点（**三个几乎人人写错的细节**）：
        1. **第一步的动量 buffer 直接等于梯度**，而不是 ``(1-dampening) * g``。
           PyTorch 是这么实现的，写错的话第一步就对不上。
        2. **PyTorch 的动量不是"标准"形式**。教科书写
           ``v = mu*v + lr*g; theta -= v``，而 PyTorch 是
           ``v = mu*v + g; theta -= lr*v``。
           两者在 lr 恒定时等价，但**在 lr 变化时不等价**：
           PyTorch 版本改 lr 会立刻影响整个历史动量的步长。
           所以从别的框架搬公式时要留意。
        3. **weight_decay 是加到梯度上的（耦合的 L2）**，
           因此会被动量累积、也会被 lr 缩放。这正是 AdamW 要"解耦"的东西。

    面试要点：**momentum 为什么能加速？**
        它对梯度做指数滑动平均，在震荡方向上正负相消、在一致方向上累加，
        有效步长约放大 ``1/(1-mu)`` 倍（mu=0.9 时约 10 倍）。
        Nesterov 则是在"预估的下一个位置"求梯度，多一阶修正项。
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        assert not nesterov or (momentum > 0 and dampening == 0), (
            "Nesterov 要求 momentum > 0 且 dampening == 0"
        )
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, dampening=dampening,
                 weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            damp, wd, nesterov = group["dampening"], group["weight_decay"], group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                if mu != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = g.clone().detach()  # 第一步：buffer = 梯度本身
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(mu).add_(g, alpha=1 - damp)
                    g = g.add(buf, alpha=mu) if nesterov else buf

                p.add_(g, alpha=-lr)
        return loss


class MyAdamW(torch.optim.Optimizer):
    r"""手写 AdamW，与 ``torch.optim.AdamW`` 逐位对齐。

    更新公式::

        theta_t <- theta_{t-1} - lr * wd * theta_{t-1}     # 解耦的 weight decay
        m_t = b1 * m_{t-1} + (1 - b1) * g_t
        v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
        m_hat = m_t / (1 - b1^t)
        v_hat = v_t / (1 - b2^t)
        theta_t <- theta_t - lr * m_hat / (sqrt(v_hat) + eps)

    面试要点：**AdamW 的 decoupled weight decay 与 Adam + L2 有什么不同？**
        - Adam + L2：把 ``wd * theta`` **加进梯度**，于是它会被
          ``1/sqrt(v_hat)`` 自适应地缩放。后果是：
          **梯度大的参数（v 大）实际受到的衰减反而更小**，
          梯度小的参数被衰减得更狠——这与"weight decay 应该均匀地
          把权重拉向 0"的初衷完全相反，正则效果被自适应机制破坏了。
        - AdamW：把衰减**从梯度里拿出来**，直接作用在参数上
          ``theta *= (1 - lr * wd)``，与 v 无关，对所有参数一视同仁。
        - 数值上的直观差别：等价的衰减强度差了 ``sqrt(v_hat)`` 倍，
          所以从 Adam 换到 AdamW **必须重调 weight_decay**
          （典型值从 1e-4 量级变到 1e-2 量级）。
        - 注意 PyTorch 的 AdamW 衰减量是 ``lr * wd * theta``，
          **和 lr 相乘**，所以调 lr 会连带改变实际衰减强度。
          （见 `test_myadamw_matches_torch_adamw` 和
          `test_adamw_decoupled_differs_from_adam_l2`。）

    面试要点：**bias correction 是干嘛的？**
        m 和 v 初始化为 0，前几步的估计被严重拉向 0（有偏）。
        ``m_t`` 是 ``t`` 个梯度的加权和，权重之和是 ``1 - b1^t``，
        除掉它就得到无偏估计。b2=0.999 时 ``1-b2^t`` 在 t=1 只有 0.001，
        少了这一步，第一步的 ``v`` 会小 1000 倍，步长直接爆炸。

    面试要点：**eps 加在哪？**
        PyTorch 是 ``m_hat / (sqrt(v_hat) + eps)``，
        eps 加在**开方之后**。有些实现写成 ``m_hat / sqrt(v_hat + eps)``
        （TF 的 ``epsilon_hat``），两者在 v 很小时行为差别明显，
        对不齐官方实现时先查这里。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # 1) 解耦的 weight decay：直接作用在参数上，不进梯度
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # 2) 一阶/二阶矩
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # 3) bias correction
                bc1 = 1.0 - b1**t
                bc2 = 1.0 - b2**t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)  # eps 加在开方之后
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


# =============================================================================
# 7. 梯度裁剪
# =============================================================================


def my_clip_grad_norm_(params, max_norm: float, norm_type: float = 2.0) -> Tensor:
    r"""手写全局梯度范数裁剪，与 ``torch.nn.utils.clip_grad_norm_`` 对齐。

    算法::

        total_norm = ||concat([g.flatten() for g in grads])||_p
        clip_coef  = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:  每个 g *= clip_coef

    面试要点：**"全局"两个字是重点。**
        必须先把**所有**参数的梯度拼成一个大向量算总范数，再统一缩放。
        逐个张量分别裁剪（``clip_grad_norm_`` 对每个 tensor 单独调用）
        会改变各层梯度之间的**相对比例**，等于扭曲了梯度方向。
        这是最常见的错误实现。

    面试要点：**几个必须记住的实现细节**
        1. ``+ 1e-6`` 是防止 total_norm 为 0 时除零。
        2. 只在 ``clip_coef < 1`` 时缩放，**范数小于阈值时不放大**
           （clamp 到 1.0）。写成无条件乘 clip_coef 就变成了"归一化"
           而不是"裁剪"，小梯度会被放大，训练行为完全不同。
        3. **返回的是裁剪前的范数**，不是裁剪后的。
           所以日志里看到 grad_norm 超过 max_norm 是正常的。
        4. 必须在 ``backward()`` 之后、``optimizer.step()`` 之前调用。
           **用 AMP 时还必须先 ``scaler.unscale_(optimizer)``**，
           否则裁的是被 loss scale 放大过的梯度，阈值完全失效——
           这是混合精度训练里最经典的坑。

    面试要点：**clip by norm vs clip by value**
        by norm 保持梯度**方向**不变，只缩短长度，是默认选择；
        by value 逐元素截断，会改变方向，一般只在特殊场景用。
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    device = grads[0].device
    norms = torch.stack([g.detach().norm(norm_type).to(device) for g in grads])
    total_norm = norms.norm(norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = clip_coef.clamp(max=1.0)  # 小于阈值时不放大
    for g in grads:
        g.detach().mul_(clip_coef)
    return total_norm  # 返回裁剪**前**的范数


# =============================================================================
#                                  TESTS
# =============================================================================


def _tiny_mlp(in_f: int = 8, out_f: int = 4, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_f, 16), nn.Tanh(), nn.Linear(16, out_f))


# ---------------------------- label smoothing ----------------------------


def test_label_smoothing_matches_torch():
    """与 F.cross_entropy(label_smoothing=eps) 对齐，覆盖多个 eps 和 reduction。"""
    torch.manual_seed(0)
    logits = torch.randn(32, 7)
    target = torch.randint(0, 7, (32,))

    for eps in (0.0, 0.05, 0.1, 0.3, 0.9):
        for red in ("mean", "sum", "none"):
            mine = label_smoothing_cross_entropy(logits, target, eps, reduction=red)
            ref = F.cross_entropy(logits, target, label_smoothing=eps, reduction=red)
            torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-6, msg=f"{eps} {red}")


def test_label_smoothing_two_equivalent_forms():
    """PyTorch 用的是"真实类 1-eps+eps/C，其余 eps/C"，而不是 eps/(C-1) 那个变体。

    这里同时验证：
      (a) 凸组合形式 == 显式软标签形式（数学等价，都对）；
      (b) 常见的 eps/(C-1) 变体与官方**不相等**（是另一个 loss）。
    """
    torch.manual_seed(0)
    logits, target, eps, c = torch.randn(16, 5), torch.randint(0, 5, (16,)), 0.1, 5
    logp = F.log_softmax(logits, -1)
    ref = F.cross_entropy(logits, target, label_smoothing=eps)

    # (a) 显式构造 q_c = (1-eps)*onehot + eps/C
    q = torch.full_like(logp, eps / c)
    q.scatter_(1, target[:, None], 1.0 - eps + eps / c)
    torch.testing.assert_close((-(q * logp).sum(-1)).mean(), ref, rtol=1e-5, atol=1e-6)

    # (b) 流传很广的变体：真实类 1-eps、其余 eps/(C-1) —— 与官方不同
    q2 = torch.full_like(logp, eps / (c - 1))
    q2.scatter_(1, target[:, None], 1.0 - eps)
    variant = (-(q2 * logp).sum(-1)).mean()
    assert abs(variant.item() - ref.item()) > 1e-3, "这两个公式本就不该相等"


def test_label_smoothing_reduces_confidence():
    """平滑后模型的最优解是有限 logit 间距，训练出的置信度更低（校准更好）。"""
    torch.manual_seed(0)
    target = torch.randint(0, 5, (64,))
    x = torch.randn(64, 8)

    confs = []
    for eps in (0.0, 0.2):
        torch.manual_seed(1)
        model = nn.Linear(8, 5)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            label_smoothing_cross_entropy(model(x), target, eps).backward()
            opt.step()
        confs.append(model(x).softmax(-1).max(-1).values.mean().item())

    assert confs[0] > confs[1], confs  # 无平滑时更"自信"


# ---------------------------- focal loss ----------------------------


def test_focal_loss_gamma_zero_is_ce():
    """gamma=0 且无 alpha 时精确退化为交叉熵。"""
    torch.manual_seed(0)
    logits, target = torch.randn(32, 6), torch.randint(0, 6, (32,))
    for red in ("mean", "sum", "none"):
        torch.testing.assert_close(
            focal_loss(logits, target, alpha=None, gamma=0.0, reduction=red),
            F.cross_entropy(logits, target, reduction=red),
            rtol=1e-5, atol=1e-6,
        )


def test_focal_loss_gamma_zero_with_alpha_is_weighted_ce():
    """gamma=0 + 标量/逐类 alpha 时退化为（加权）CE。

    注意与 ``F.cross_entropy(weight=w)`` 的差别：后者 ``reduction='mean'``
    是**加权平均**（除以权重之和）而不是简单平均，所以要用 reduction='none'
    或 'sum' 才能直接对上。这是一个很容易踩的 API 细节。
    """
    torch.manual_seed(0)
    logits, target = torch.randn(32, 4), torch.randint(0, 4, (32,))
    ce_none = F.cross_entropy(logits, target, reduction="none")

    torch.testing.assert_close(
        focal_loss(logits, target, alpha=0.25, gamma=0.0, reduction="none"),
        0.25 * ce_none, rtol=1e-5, atol=1e-6,
    )

    w = torch.tensor([0.1, 0.5, 1.0, 2.0])
    torch.testing.assert_close(
        focal_loss(logits, target, alpha=w, gamma=0.0, reduction="sum"),
        F.cross_entropy(logits, target, weight=w, reduction="sum"),
        rtol=1e-5, atol=1e-5,
    )


def test_focal_loss_downweights_easy_samples():
    """核心性质：gamma 越大，易分样本的 loss 被压得越狠，难样本几乎不受影响。"""
    # 两个样本：第 0 个易分（p_t≈0.95），第 1 个难分（p_t≈0.05）
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 0])
    ce = focal_loss(logits, target, gamma=0.0, reduction="none")
    fl = focal_loss(logits, target, gamma=2.0, reduction="none")

    easy_ratio = (fl[0] / ce[0]).item()
    hard_ratio = (fl[1] / ce[1]).item()
    assert easy_ratio < 0.01, easy_ratio  # 易分样本被压到 1% 以下
    assert hard_ratio > 0.85, hard_ratio  # 难样本基本保留
    # 相对权重发生了两个数量级的转移
    assert hard_ratio / easy_ratio > 100


# ---------------------------- mixup / cutmix ----------------------------


def test_mixup_shapes_and_lam_range():
    torch.manual_seed(0)
    x, y = torch.randn(8, 3, 4, 4), torch.randint(0, 5, (8,))
    for _ in range(20):
        mixed, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
        assert mixed.shape == x.shape
        assert y_a.shape == y_b.shape == y.shape
        assert 0.0 <= lam <= 1.0
        assert torch.equal(y_a, y)  # y_a 永远是原标签


def test_mixup_lam_one_is_identity():
    """alpha<=0 => lam=1 => 输出就是原样本。"""
    torch.manual_seed(0)
    x, y = torch.randn(8, 6), torch.randint(0, 5, (8,))
    mixed, y_a, _, lam = mixup_data(x, y, alpha=0.0)
    assert lam == 1.0
    torch.testing.assert_close(mixed, x)
    assert torch.equal(y_a, y)


def test_mixup_is_exact_convex_combination():
    """手动复现配对关系，验证 mixed == lam*x + (1-lam)*x[perm]。"""
    torch.manual_seed(0)
    x, y = torch.randn(16, 5), torch.arange(16)
    g = torch.Generator().manual_seed(123)
    mixed, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, generator=g)
    # y 是 arange，所以 y_b 本身就是那个置换
    torch.testing.assert_close(mixed, lam * x + (1 - lam) * x[y_b], rtol=1e-5, atol=1e-6)


def test_mixup_criterion_equals_soft_label_ce():
    """lam*CE(p,y_a) + (1-lam)*CE(p,y_b) == CE(p, 混合后的软标签)。

    CE 对目标分布是线性的，所以这两种写法数学上完全等价——
    这解释了为什么返回 (y_a, y_b, lam) 而不是直接构造软标签。
    """
    torch.manual_seed(0)
    pred = torch.randn(16, 5)
    y_a, y_b, lam = torch.randint(0, 5, (16,)), torch.randint(0, 5, (16,)), 0.3

    via_two_terms = mixup_criterion(F.cross_entropy, pred, y_a, y_b, lam)
    soft = lam * F.one_hot(y_a, 5).float() + (1 - lam) * F.one_hot(y_b, 5).float()
    via_soft = (-(soft * F.log_softmax(pred, -1)).sum(-1)).mean()
    torch.testing.assert_close(via_two_terms, via_soft, rtol=1e-5, atol=1e-6)


def test_cutmix_region_comes_from_other_sample():
    """被替换区域的像素必须逐位来自配对样本，框外区域必须保持原样。"""
    torch.manual_seed(0)
    # 用常数图，每个样本一个独特值，替换关系一眼可查
    x = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1).expand(6, 3, 8, 8).contiguous()
    y = torch.arange(6)

    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        out, y_a, y_b, lam, (y1, y2, x1, x2) = cutmix_data(x, y, alpha=1.0, generator=g)
        assert out.shape == x.shape and torch.equal(y_a, y)
        if y2 <= y1 or x2 <= x1:
            continue  # 空框（lam 接近 1），跳过
        # 框内 = 配对样本
        torch.testing.assert_close(out[:, :, y1:y2, x1:x2], x[y_b][:, :, y1:y2, x1:x2])
        # 框外 = 原样本
        mask = torch.ones(8, 8, dtype=torch.bool)
        mask[y1:y2, x1:x2] = False
        torch.testing.assert_close(out[:, :, mask], x[:, :, mask])


def test_cutmix_lam_matches_actual_area():
    """修正后的 lam 必须精确等于"未被替换的面积比例"。"""
    torch.manual_seed(0)
    x, y = torch.randn(4, 3, 16, 16), torch.arange(4)
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        _, _, _, lam, (y1, y2, x1, x2) = cutmix_data(x, y, alpha=1.0, generator=g)
        area = (y2 - y1) * (x2 - x1)
        assert abs(lam - (1.0 - area / (16 * 16))) < 1e-6
        assert 0.0 <= lam <= 1.0


# ---------------------------- ModelEMA ----------------------------


def test_ema_decay_zero_copies_current_weights():
    """decay=0 => shadow 完全等于当前权重。"""
    torch.manual_seed(0)
    model = _tiny_mlp()
    ema = ModelEMA(model, decay=0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    for s, p in zip(ema.shadow.parameters(), model.parameters()):
        torch.testing.assert_close(s, p, rtol=0, atol=1e-7)


def test_ema_decay_one_never_moves():
    """decay=1 => shadow 永远停在初始权重（也是"decay 设太大"翻车的极端版）。"""
    torch.manual_seed(0)
    model = _tiny_mlp()
    ema = ModelEMA(model, decay=1.0)
    init = [p.clone() for p in ema.shadow.parameters()]
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)
    for s, i in zip(ema.shadow.parameters(), init):
        torch.testing.assert_close(s, i, rtol=0, atol=1e-7)


def test_ema_converges_toward_target():
    """权重固定不动时，shadow 以 decay^t 的速度指数逼近它（可解析验证）。"""
    torch.manual_seed(0)
    model = _tiny_mlp()
    target = [p.clone() for p in model.parameters()]
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():  # 让 shadow 与 target 拉开一个已知的差距
        for s in ema.shadow.parameters():
            s.zero_()

    n = 20
    for _ in range(n):
        ema.update(model)

    # shadow_n = target * (1 - decay^n)，因为初始 shadow 是 0
    for s, t in zip(ema.shadow.parameters(), target):
        torch.testing.assert_close(s, t * (1 - 0.9**n), rtol=1e-5, atol=1e-6)


def test_ema_warmup_starts_at_current_weights():
    """带 warmup 时第一次 update 的有效 decay 是 0，shadow 直接对齐当前权重。

    这就是 warmup 要解决的冷启动问题：没有它，decay=0.999 的影子
    在前几千步基本还停在随机初始化上。
    """
    torch.manual_seed(0)
    model = _tiny_mlp()
    ema = ModelEMA(model, decay=0.999, warmup_steps=100)
    assert ema.current_decay() == 0.0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for s, p in zip(ema.shadow.parameters(), model.parameters()):
        torch.testing.assert_close(s, p, rtol=0, atol=1e-7)

    # decay 随步数单调升向目标值，但永远不超过它
    decays = []
    for _ in range(300):
        ema.update(model)
        decays.append(ema.current_decay())
    assert all(a <= b + 1e-12 for a, b in zip(decays, decays[1:]))
    assert decays[-1] < 0.999 and decays[-1] > 0.9


def test_ema_buffers_are_copied_not_averaged():
    """BN 的 running stats 走直接拷贝，且整型 buffer 不会被破坏。"""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6))
    ema = ModelEMA(model, decay=0.9)

    model.train()
    model(torch.randn(16, 4))  # 更新 running_mean / var / num_batches_tracked
    ema.update(model)

    bn, bn_s = model[1], ema.shadow[1]
    torch.testing.assert_close(bn_s.running_mean, bn.running_mean, rtol=0, atol=0)
    torch.testing.assert_close(bn_s.running_var, bn.running_var, rtol=0, atol=0)
    assert bn_s.num_batches_tracked.dtype == torch.long
    assert int(bn_s.num_batches_tracked) == int(bn.num_batches_tracked) == 1
    # 而 weight（是 parameter 不是 buffer）确实做了 EMA，没有被直接拷贝
    assert not torch.allclose(bn_s.weight, bn.weight) or torch.allclose(
        bn_s.weight, torch.ones_like(bn_s.weight)
    )


# ---------------------------- scheduler ----------------------------


def test_warmup_cosine_key_points():
    """三个关键点：step 0 为 0、warmup 结束正好是 peak、终点正好是 min_lr。"""
    base_lr, warmup, total, min_lr = 0.1, 10, 100, 0.001
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base_lr)
    sch = WarmupCosineScheduler(opt, warmup, total, min_lr)

    # 构造完成后 optimizer 里已经是 step 0 的 lr
    assert sch.get_last_lr()[0] == 0.0
    assert opt.param_groups[0]["lr"] == 0.0

    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(total):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert abs(lrs[warmup] - base_lr) < 1e-12, lrs[warmup]  # peak
    assert abs(lrs[total] - min_lr) < 1e-12, lrs[total]  # 终点
    # 中点大约是 (base + min) / 2（cosine 的对称性）
    mid = warmup + (total - warmup) // 2
    assert abs(lrs[mid] - (base_lr + min_lr) / 2) < 1e-3


def test_warmup_cosine_monotonicity():
    """warmup 段严格单调增，cosine 段严格单调减，且全程落在 [min_lr, base_lr]。"""
    base_lr, warmup, total, min_lr = 0.5, 20, 200, 0.01
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base_lr)
    sch = WarmupCosineScheduler(opt, warmup, total, min_lr)

    lrs = []
    for _ in range(total + 1):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()

    up, down = lrs[: warmup + 1], lrs[warmup:]
    assert all(a < b for a, b in zip(up, up[1:])), "warmup 段必须递增"
    assert all(a > b for a, b in zip(down, down[1:])), "cosine 段必须递减"
    assert max(lrs) == lrs[warmup] and abs(max(lrs) - base_lr) < 1e-12
    assert min(lrs) >= 0.0 and abs(lrs[-1] - min_lr) < 1e-12


def test_warmup_cosine_handles_multiple_param_groups():
    """多个 param group 各自按自己的 base_lr 缩放（大模型分组 wd 时常见）。"""
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}])
    sch = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50, min_lr=0.0)
    for _ in range(5):
        opt.step()
        sch.step()
    lrs = sch.get_last_lr()
    assert abs(lrs[0] - 0.1) < 1e-12 and abs(lrs[1] - 0.01) < 1e-12


# ---------------------------- 梯度累积 ----------------------------


def test_grad_accumulation_equivalence():
    """**高频面试题**：梯度累积 K 步（每步 loss/K）与一次大 batch 的梯度**完全相同**。

    为什么成立：``backward()`` 是**累加**到 ``.grad`` 而不是覆盖，而
        mean_over_N(l) = (1/K) * sum_{k=1..K} mean_over_{N/K}(l_k)
    只要每个 micro-batch **大小相等**。所以每个 micro loss 除以 K 再 backward，
    累加出来就是大 batch 的平均梯度。

    三个必须说清的前提：
      1. **必须除以 K**（或者用 sum reduction）。忘了除 K 相当于把学习率放大 K 倍。
      2. **micro-batch 必须等大**。最后一个不满的 batch 会让加权变形，
         正确做法是按样本数加权而不是均匀除 K。
      3. **模型里不能有 batch 统计量**（BatchNorm）——见下一个测试。
    另外分布式下要用 ``no_sync()`` 包住前 K-1 步，否则每步都会 all-reduce，
    通信量白白放大 K 倍（这是梯度累积在 DDP 下的标准配套写法）。
    """
    torch.manual_seed(0)
    n, k = 32, 4
    x, y = torch.randn(n, 8), torch.randint(0, 4, (n,))

    big = _tiny_mlp(seed=1)
    acc = copy.deepcopy(big)

    # 一次大 batch
    F.cross_entropy(big(x), y).backward()

    # 累积 K 步
    micro = n // k
    for i in range(k):
        xi, yi = x[i * micro : (i + 1) * micro], y[i * micro : (i + 1) * micro]
        (F.cross_entropy(acc(xi), yi) / k).backward()

    for (n1, p1), (_, p2) in zip(big.named_parameters(), acc.named_parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6, msg=n1)


def test_grad_accumulation_forgetting_divide_scales_by_k():
    """反例：忘了除以 K，梯度正好被放大 K 倍（等价于偷偷把 lr 乘了 K）。"""
    torch.manual_seed(0)
    n, k = 32, 4
    x, y = torch.randn(n, 8), torch.randint(0, 4, (n,))
    big, acc = _tiny_mlp(seed=1), _tiny_mlp(seed=1)

    F.cross_entropy(big(x), y).backward()
    micro = n // k
    for i in range(k):
        F.cross_entropy(
            acc(x[i * micro : (i + 1) * micro]), y[i * micro : (i + 1) * micro]
        ).backward()

    for p1, p2 in zip(big.parameters(), acc.parameters()):
        torch.testing.assert_close(p2.grad, p1.grad * k, rtol=1e-5, atol=1e-5)


def test_grad_accumulation_breaks_with_batchnorm():
    """**BatchNorm 是那个例外**：等价性不成立。

    BN 用的是 **当前 batch 内**的均值/方差。大 batch 用 N 个样本统计，
    micro-batch 只用 N/K 个，归一化后的激活值本身就不同，
    梯度自然对不上——这不是实现 bug，是数学上就不等价。
    工程上的解法：
      - 换成 **GroupNorm / LayerNorm / RMSNorm**（不依赖 batch 统计，天然等价）；
      - 或者用 SyncBN / 把 BN 的 batch 维单独处理。
    同理，DDP 下不同卡的 BN 统计也是各算各的，也需要 SyncBN。
    """
    torch.manual_seed(0)
    n, k = 32, 4
    x, y = torch.randn(n, 8), torch.randint(0, 4, (n,))

    def make():
        torch.manual_seed(1)
        return nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Tanh(), nn.Linear(16, 4))

    big, acc = make(), make()
    big.train()
    acc.train()

    F.cross_entropy(big(x), y).backward()
    micro = n // k
    for i in range(k):
        (F.cross_entropy(
            acc(x[i * micro : (i + 1) * micro]), y[i * micro : (i + 1) * micro]
        ) / k).backward()

    diffs = [
        (p1.grad - p2.grad).abs().max().item()
        for p1, p2 in zip(big.parameters(), acc.parameters())
    ]
    assert max(diffs) > 1e-4, "有 BN 时梯度本就不该相等"

    # 换成 LayerNorm（不依赖 batch 统计）等价性立刻恢复
    def make_ln():
        torch.manual_seed(1)
        return nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Tanh(), nn.Linear(16, 4))

    big2, acc2 = make_ln(), make_ln()
    F.cross_entropy(big2(x), y).backward()
    for i in range(k):
        (F.cross_entropy(
            acc2(x[i * micro : (i + 1) * micro]), y[i * micro : (i + 1) * micro]
        ) / k).backward()
    for p1, p2 in zip(big2.parameters(), acc2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


# ---------------------------- 手写优化器 ----------------------------


def _run_optimizer(model: nn.Module, opt: torch.optim.Optimizer, steps: int = 8):
    """在固定数据上跑若干步，返回最终参数。数据/初始化都固定以保证可比。"""
    torch.manual_seed(7)
    x, y = torch.randn(24, 8), torch.randint(0, 4, (24,))
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def test_mysgd_matches_torch_sgd():
    """与 torch.optim.SGD 跑同样 8 步后参数逐位对齐（覆盖 4 种配置）。"""
    cfgs = [
        dict(lr=0.1, momentum=0.0),
        dict(lr=0.1, momentum=0.9),
        dict(lr=0.05, momentum=0.9, weight_decay=0.01),
        dict(lr=0.05, momentum=0.9, dampening=0.2),
    ]
    for cfg in cfgs:
        m1, m2 = _tiny_mlp(seed=3), _tiny_mlp(seed=3)
        got = _run_optimizer(m1, MySGDMomentum(m1.parameters(), **cfg))
        ref = _run_optimizer(m2, torch.optim.SGD(m2.parameters(), **cfg))
        for a, b in zip(got, ref):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7, msg=str(cfg))


def test_mysgd_nesterov_matches_torch():
    cfg = dict(lr=0.1, momentum=0.9, nesterov=True)
    m1, m2 = _tiny_mlp(seed=3), _tiny_mlp(seed=3)
    got = _run_optimizer(m1, MySGDMomentum(m1.parameters(), **cfg))
    ref = _run_optimizer(m2, torch.optim.SGD(m2.parameters(), **cfg))
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_mysgd_first_step_buffer_equals_grad():
    """验证细节 #1：第一步的动量 buffer 就是梯度本身，所以首步等价于纯 SGD。"""
    m1, m2 = _tiny_mlp(seed=3), _tiny_mlp(seed=3)
    got = _run_optimizer(m1, MySGDMomentum(m1.parameters(), lr=0.1, momentum=0.9), steps=1)
    ref = _run_optimizer(m2, torch.optim.SGD(m2.parameters(), lr=0.1), steps=1)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_myadamw_matches_torch_adamw():
    """与 torch.optim.AdamW 跑同样 8 步后参数逐位对齐。"""
    cfgs = [
        dict(lr=1e-2, weight_decay=0.0),
        dict(lr=1e-2, weight_decay=0.1),
        dict(lr=1e-3, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.05),
    ]
    for cfg in cfgs:
        m1, m2 = _tiny_mlp(seed=5), _tiny_mlp(seed=5)
        got = _run_optimizer(m1, MyAdamW(m1.parameters(), **cfg))
        ref = _run_optimizer(m2, torch.optim.AdamW(m2.parameters(), **cfg))
        for a, b in zip(got, ref):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=str(cfg))


def test_myadamw_zero_wd_matches_adam():
    """weight_decay=0 时 AdamW 就是 Adam（这是判断实现是否解耦的最快检查）。"""
    m1, m2 = _tiny_mlp(seed=5), _tiny_mlp(seed=5)
    got = _run_optimizer(m1, MyAdamW(m1.parameters(), lr=1e-2, weight_decay=0.0))
    ref = _run_optimizer(m2, torch.optim.Adam(m2.parameters(), lr=1e-2))
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_adamw_decoupled_differs_from_adam_l2():
    """AdamW 与 "Adam + L2 加进梯度" 结果**不同**，这正是 AdamW 论文的全部意义。

    差别的来源：L2 版本的衰减项被 ``1/sqrt(v_hat)`` 自适应缩放了，
    对不同参数的实际衰减强度不一致；AdamW 版本对所有参数一视同仁。
    """
    wd = 0.1
    m1, m2 = _tiny_mlp(seed=5), _tiny_mlp(seed=5)
    adamw = _run_optimizer(m1, MyAdamW(m1.parameters(), lr=1e-2, weight_decay=wd))
    # Adam + L2：把 wd 塞进 Adam 的 weight_decay（PyTorch 的 Adam 是耦合的）
    adam_l2 = _run_optimizer(
        m2, torch.optim.Adam(m2.parameters(), lr=1e-2, weight_decay=wd)
    )
    diffs = [(a - b).abs().max().item() for a, b in zip(adamw, adam_l2)]
    assert max(diffs) > 1e-4, diffs

    # 且 AdamW 的解耦衰减是精确可预测的：无梯度时参数每步恰好乘 (1 - lr*wd)
    p = nn.Parameter(torch.ones(3))
    opt = MyAdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(3)
    opt.step()
    torch.testing.assert_close(p.detach(), torch.full((3,), 1 - 0.1 * 0.5),
                               rtol=1e-6, atol=1e-7)


def test_myadamw_bias_correction_first_step():
    """第一步的更新量恰好是 lr（因为 m_hat / sqrt(v_hat) = g / |g| = sign(g)）。

    这是 bias correction 正确的直接证据：
    t=1 时 m_hat = g、v_hat = g^2，比值是 sign(g)，与梯度大小完全无关。
    少了 bias correction 的话第一步会小 (1-b1)/sqrt(1-b2) ≈ 3.2 倍。
    """
    for scale in (1e-3, 1.0, 1e3):
        p = nn.Parameter(torch.ones(4))
        opt = MyAdamW([p], lr=0.1, weight_decay=0.0, eps=0.0)
        p.grad = torch.tensor([1.0, -1.0, 2.0, -0.5]) * scale
        opt.step()
        torch.testing.assert_close(
            p.detach(), torch.tensor([0.9, 1.1, 0.9, 1.1]), rtol=1e-5, atol=1e-6
        )


# ---------------------------- 梯度裁剪 ----------------------------


def test_clip_grad_norm_matches_torch():
    """与 torch.nn.utils.clip_grad_norm_ 对齐：返回值和裁剪后的梯度都要一致。"""
    for max_norm in (0.01, 0.5, 1e6):  # 分别覆盖"要裁"和"不用裁"
        for norm_type in (2.0, 1.0, float("inf")):
            m1, m2 = _tiny_mlp(seed=9), _tiny_mlp(seed=9)
            torch.manual_seed(0)
            x, y = torch.randn(16, 8), torch.randint(0, 4, (16,))
            F.cross_entropy(m1(x), y).backward()
            F.cross_entropy(m2(x), y).backward()

            got = my_clip_grad_norm_(list(m1.parameters()), max_norm, norm_type)
            ref = torch.nn.utils.clip_grad_norm_(m2.parameters(), max_norm, norm_type)

            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-7)


def test_clip_grad_norm_returns_pre_clip_norm_and_does_not_upscale():
    """两个关键语义：返回裁剪前的范数；范数小于阈值时不放大。"""
    p = nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # L2 范数正好 5

    total = my_clip_grad_norm_([p], max_norm=1.0)
    assert abs(total.item() - 5.0) < 1e-6, "返回的必须是裁剪前的范数"
    torch.testing.assert_close(p.grad.norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)

    # 范数远小于阈值时不动
    p.grad = torch.tensor([0.03, 0.04, 0.0, 0.0])
    before = p.grad.clone()
    total = my_clip_grad_norm_([p], max_norm=10.0)
    assert abs(total.item() - 0.05) < 1e-6
    torch.testing.assert_close(p.grad, before, rtol=0, atol=0)


def test_clip_grad_norm_is_global_not_per_tensor():
    """全局裁剪保持各层梯度的相对比例；逐张量裁剪会破坏方向。"""
    a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([10.0, 0.0])
    b.grad = torch.tensor([0.1, 0.0])
    ratio_before = (a.grad.norm() / b.grad.norm()).item()

    my_clip_grad_norm_([a, b], max_norm=1.0)
    assert abs((a.grad.norm() / b.grad.norm()).item() - ratio_before) < 1e-4
    # 裁剪后的全局范数正好等于阈值
    total = torch.cat([a.grad, b.grad]).norm()
    torch.testing.assert_close(total, torch.tensor(1.0), rtol=1e-5, atol=1e-5)

    # 对比：逐张量分别裁剪，比例被彻底破坏（10:0.1 变成 1:0.1）
    a.grad, b.grad = torch.tensor([10.0, 0.0]), torch.tensor([0.1, 0.0])
    my_clip_grad_norm_([a], 1.0)
    my_clip_grad_norm_([b], 1.0)
    assert abs((a.grad.norm() / b.grad.norm()).item() - ratio_before) > 1.0


if __name__ == "__main__":
    import sys

    for name, fn in dict(globals()).items():
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all training-trick tests passed", file=sys.stderr)
