"""PyTorch 面试手写题（二）：自定义 torch.autograd.Function。

考察点是「你知不知道 autograd 到底在做什么」：forward 存什么、backward 返回几个值、
梯度可以不是真导数（STE / GRL）、以及怎么用 gradcheck 自证。

验证手段有三层，从弱到强：
  1. 前向和官方实现对齐；
  2. 反向和官方实现的梯度对齐；
  3. ``torch.autograd.gradcheck``：用 double 精度的**数值差分**逐元素比对解析梯度。
     这是唯一能抓出「backward 公式推错但形状正确」的方法，也是面试里最能体现
     工程素养的一句话：「我写完 backward 一定跑 gradcheck」。

为什么 gradcheck 必须用 float64？中心差分 (f(x+h)-f(x-h))/(2h) 的截断误差是 O(h^2)、
舍入误差是 O(eps/h)。fp32 的 eps≈1.2e-7，最优 h 下相对误差也有 1e-4 量级，
根本达不到 gradcheck 默认的 atol=1e-5；fp64 的 eps≈2.2e-16 才够用。
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function


# ---------------------------------------------------------------------------
# 1. ReLU —— 最简模板
# ---------------------------------------------------------------------------


class MyReLU(Function):
    """relu(x) = max(x, 0)。用来讲清楚 Function 的三件套。

    面试要点：
      1. **forward / backward 都是 staticmethod**，靠 ``ctx`` 这个上下文对象传递
         状态。调用时用 ``MyReLU.apply(x)`` 而不是 ``MyReLU()(x)``。
      2. **``ctx.save_for_backward`` vs ``ctx.xxx = ...``**：
         - 张量（尤其是输入/输出）必须用 save_for_backward。它会做引用计数和
           **版本号检查**：如果这个张量在 forward 之后被原地修改过，backward 时
           会直接报 "one of the variables needed for gradient computation has been
           modified by an inplace operation"，而不是静默算错。
         - 非张量（int/float/bool）直接挂在 ctx 上，用 save_for_backward 反而报错。
      3. **存什么最省显存**：这里存 mask（bool，1 byte/元素）比存 x（fp32，4 byte）
         省 4 倍。官方 relu 更进一步——它存的是**输出** y，因为 y>0 ⟺ x>0，
         而 y 本来就要留给下一层，等于零额外开销。这就是 ReLU 能 inplace 的原因。
      4. **``ctx.needs_input_grad``** 是一个 bool 元组，长度等于 forward 的输入个数。
         用它跳过不需要的梯度计算是标准优化；对应位置直接返回 None。
      5. **x=0 处不可导**，PyTorch 约定次梯度取 0（``x > 0`` 而不是 ``x >= 0``）。
         gradcheck 在 0 附近会失败，所以测试输入要避开 0。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        mask = x > 0
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        (mask,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        return grad_out * mask


# ---------------------------------------------------------------------------
# 2. GELU（tanh 近似）—— 手推导数
# ---------------------------------------------------------------------------

_C = math.sqrt(2.0 / math.pi)  # ≈ 0.7978845608
_A = 0.044715


class MyGELU(Function):
    r"""GELU 的 tanh 近似版（BERT/GPT-2 用的就是这个）。

    前向：
        s(x) = c * (x + a * x^3),  c = sqrt(2/pi), a = 0.044715
        gelu(x) = 0.5 * x * (1 + tanh(s))

    **反向手推**（面试要求当场写出来）：
        令 t = tanh(s)，则 dt/ds = 1 - t^2（sech^2）。
        ds/dx = c * (1 + 3a * x^2)
        d(gelu)/dx = 0.5 * (1 + t)                      # 对前面那个 x 求导
                   + 0.5 * x * (1 - t^2) * c * (1 + 3a x^2)   # 对 tanh 求导
        两项分别对应「线性因子」和「门控因子」的贡献，漏掉第二项是最常见的错误。

    面试要点：
      1. **为什么用 tanh 近似而不是精确的 0.5x(1+erf(x/sqrt2))**：2018 年 erf
         在部分硬件上没有快速实现，tanh 有。今天 ``F.gelu()`` 默认走的是精确版，
         想对齐 HuggingFace 老模型必须传 ``approximate='tanh'``。两者最大差值
         约 1e-3，落在激活值上不影响收敛，但做数值对齐时会被 assert_close 抓到。
      2. **GELU 不是单调函数**：导数在 x < -0.75 的整个区间上都是负的，最小值
         约 -0.129 出现在 x ≈ -1.42。这和 ReLU 的直觉完全不同，也是它能表达
         「适度的负输入被轻微保留、更负的反而被压回 0」的原因。
      3. 这里存 x（而不是 y）：因为反向公式里 x 和 t 都要用，而 t 可以由 x 重算。
         用重算换显存是 activation checkpointing 的微观版本。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        ctx.save_for_backward(x)
        s = _C * (x + _A * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(s))

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        (x,) = ctx.saved_tensors
        s = _C * (x + _A * x.pow(3))
        t = torch.tanh(s)
        dsdx = _C * (1.0 + 3.0 * _A * x.pow(2))
        local = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dsdx
        return grad_out * local


# ---------------------------------------------------------------------------
# 3. Softmax —— 最高频的反向公式
# ---------------------------------------------------------------------------


class MySoftmaxFunction(Function):
    r"""沿最后一维的 softmax，反向用那条经典恒等式。

    **反向推导**（必背）：
        y_i = e^{x_i} / sum_k e^{x_k}
        雅可比：dy_i/dx_j = y_i * (delta_ij - y_j)
        于是
            dL/dx_j = sum_i (dL/dy_i) * y_i * (delta_ij - y_j)
                    = dy_j * y_j - y_j * sum_i dy_i * y_i
                    = y_j * ( dy_j - sum_i dy_i y_i )
        向量化就是
            dx = (dy - (dy * y).sum(-1, keepdim=True)) * y

    面试要点：
      1. **不要显式构造雅可比**。D 维输入的雅可比是 (D, D)，对 (B, S, D) 就是
         (B, S, D, D)，序列一长直接 OOM。上面那条公式把它压成两次 elementwise +
         一次 reduction，O(D) 显存。
      2. **backward 只需要 y，不需要 x**。所以这里 save 的是输出。这也是为什么
         attention 的反向能只保存 softmax 结果（FlashAttention 更狠，连 y 都不存，
         用保存的 logsumexp 重算）。
      3. ``(dy * y).sum(-1)`` 这一项的物理含义是「加权平均的上游梯度」，减掉它
         保证了 ``dx.sum(-1) == 0`` —— 因为 softmax 的输出被约束在单纯形上，
         沿全 1 方向的扰动不改变输出。这个性质可以当作免费的自检。
      4. 和 cross-entropy 融合时，dy 恰好让整条式子塌缩成 ``y - onehot``，
         这就是 softmax_cross_entropy 反向如此简洁的原因。

    常见错法：写成 ``dx = dy * y * (1 - y)``（把雅可比当成对角阵），
    这是 sigmoid 的导数，不是 softmax 的。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        y = torch.exp(x - x.amax(dim=-1, keepdim=True))
        y = y / y.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "... D"]):
        (y,) = ctx.saved_tensors
        return (grad_out - (grad_out * y).sum(dim=-1, keepdim=True)) * y


# ---------------------------------------------------------------------------
# 4. LayerNorm —— 难度最高的手推题
# ---------------------------------------------------------------------------


class MyLayerNormFunction(Function):
    r"""沿最后一维的 LayerNorm，backward 全手推。

    **完整推导**（面试里能写到这一步基本就过了）

    设一行输入 x ∈ R^D：
        mu   = (1/D) * sum_i x_i
        var  = (1/D) * sum_i (x_i - mu)^2          # 有偏
        rstd = 1 / sqrt(var + eps)
        xhat = (x - mu) * rstd
        y    = w ⊙ xhat + b

    参数梯度（在 batch 维求和）：
        dL/dw = sum_batch (dy ⊙ xhat)
        dL/db = sum_batch dy

    输入梯度：先令 g = dy ⊙ w = dL/dxhat，然后走三条链路（x 影响 xhat 有三条路径：
    直接、经过 mu、经过 var）：

        dL/dvar = sum_i g_i * (x_i - mu) * (-1/2) * (var+eps)^{-3/2}
        dL/dmu  = sum_i g_i * (-rstd)      [+ dL/dvar * (-2/D) * sum_i (x_i-mu) = 0]
        dL/dx_i = g_i * rstd
                + dL/dvar * (2/D)(x_i - mu)
                + dL/dmu  * (1/D)

    把 xhat = (x-mu)*rstd 代回去合并同类项，得到那条著名的紧凑形式：

        dx = rstd * ( g - mean(g) - xhat * mean(g ⊙ xhat) )
                              ↑            ↑
                        去掉均值分量   去掉沿 xhat 方向的分量

    面试要点：
      1. **几何直观**：归一化把 x 投影到「均值为 0、模长为 sqrt(D)」的球面上，
         所以梯度里必须减掉两个方向的分量 —— 全 1 方向（改变均值）和 xhat 方向
         （改变模长），因为沿这两个方向移动 x 根本不改变 y。这也解释了为什么
         ``dx.sum(-1) == 0`` 且 ``(dx * xhat).sum(-1) == 0``，可以当自检。
      2. **要保存 rstd 而不是 var**：省一次 sqrt/除法；Triton/CUDA 的 LN kernel
         无一例外都存 (mean, rstd) 这两个 (B*S,) 的小张量。
      3. **dw/db 是跨 batch 的 reduction**，在 kernel 里要做两阶段规约
         （每个 block 先局部累加到 partial buffer，再第二个 kernel 汇总），
         直接 atomicAdd 到 (D,) 上会被严重的写冲突拖垮。
      4. 注意 mean 这里是**沿最后一维**的 mean，不是全局 mean。

    常见错法：只写 ``dx = g * rstd``（漏掉 mu 和 var 两条路径）。这个错误在
    小 batch 上 loss 也会下降，非常隐蔽，只有 gradcheck 能立刻抓出来。
    """

    @staticmethod
    def forward(
        ctx,
        x: Float[Tensor, "... D"],
        weight: Float[Tensor, " D"],
        bias: Float[Tensor, " D"],
        eps: float = 1e-5,
    ) -> Float[Tensor, "... D"]:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        xhat = (x - mu) * rstd
        # 保存 xhat 和 rstd（而不是 x/mu/var）：反向公式直接就用这两个
        ctx.save_for_backward(xhat, rstd, weight)
        return xhat * weight + bias

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "... D"]):
        xhat, rstd, weight = ctx.saved_tensors
        g = grad_out * weight  # dL/dxhat

        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = rstd * (
                g
                - g.mean(dim=-1, keepdim=True)
                - xhat * (g * xhat).mean(dim=-1, keepdim=True)
            )
        # 参数梯度：把除最后一维外的所有维度加掉
        reduce_dims = tuple(range(grad_out.dim() - 1))
        if ctx.needs_input_grad[1]:
            dw = (grad_out * xhat).sum(dim=reduce_dims)
        if ctx.needs_input_grad[2]:
            db = grad_out.sum(dim=reduce_dims)
        # 第 4 个输入 eps 是 float，返回 None 占位
        return dx, dw, db, None


# ---------------------------------------------------------------------------
# 5. Straight-Through Estimator
# ---------------------------------------------------------------------------


class RoundSTE(Function):
    """forward 做 round（阶梯函数，导数处处为 0），backward 直接把梯度透传。

    面试要点：
      1. **为什么需要 STE**：量化感知训练（QAT）里 round/sign/argmax 这类算子的
         真导数几乎处处为 0，梯度一到这里就断了，前面的层永远学不到东西。
         STE 的做法是「前向用离散值保证和推理一致，反向假装这一步是恒等函数」。
      2. **这是一个有偏估计器**，backward 返回的不是任何函数的真导数。它能 work
         的直觉是：round(x) ≈ x + noise，把 round 看成加了个零均值扰动的恒等映射，
         那么恒等的导数 1 就是个合理的代理。
      3. **实践里通常要 clip**：``grad * (|x| <= 1)``（clipped STE / hardtanh STE）。
         因为落在量化区间外的输入，再怎么推也不会改变输出，透传梯度纯属噪声。
      4. **一行实现的写法**：``y = x + (x.round() - x).detach()``。前向等于
         x.round()，反向因为括号里被 detach 了，梯度直接从 y 流到 x。面试里
         能说出这个技巧是加分项 —— 但它多一次加法，且没法定制 clip，所以
         生产代码还是写 Function。
      5. **必然通不过 gradcheck**，见 test_ste_fails_gradcheck_by_design。
         这不是 bug，是设计。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        (x,) = ctx.saved_tensors
        # clipped STE：量化区间外不透传
        return grad_out * (x.abs() <= 1.0)


class SignSTE(Function):
    """sign 的 STE 版本，二值化网络（BNN / XNOR-Net）的基本构件。"""

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        return grad_out


# ---------------------------------------------------------------------------
# 6. Gradient Reversal Layer
# ---------------------------------------------------------------------------


class GradientReversal(Function):
    """forward 恒等，backward 把梯度乘 -lambda。DANN（domain adaptation）的核心。

    面试要点：
      1. **用途**：训练一个 domain classifier 去分辨样本来自源域还是目标域，
         同时希望特征提取器**学不出**这个区分。把 GRL 插在特征提取器和 domain
         classifier 之间，classifier 正常最小化自己的 loss，而回传到特征提取器
         的梯度被翻转，等价于让特征提取器最大化 domain loss。一次前向、一个
         optimizer 就实现了 min-max 对抗，不用像 GAN 那样交替更新两个网络。
      2. **这是最好的教学例子：backward 完全可以不是 forward 的真导数**。
         autograd 只是忠实地按你写的 backward 做链式传播，它没有任何机制去校验
         你返回的东西是不是真的雅可比向量积。理解这一点，就理解了 STE、
         梯度裁剪、gradient checkpointing 里各种"作弊"手法的合法性来源。
      3. **lambda 通常带 schedule**：训练初期设 0（先让 domain classifier 学好），
         再按 2/(1+exp(-10p)) - 1 逐渐升到 1，否则早期噪声梯度会毁掉特征。
      4. 常见错法：写成 ``forward 返回 -x``。那样前向就变了，domain classifier
         看到的是取反后的特征，效果完全不同。forward 必须是恒等。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."], lambd: float = 1.0) -> Float[Tensor, "..."]:
        ctx.lambd = lambd  # float 不能 save_for_backward，直接挂 ctx
        return x.view_as(x)  # 恒等，但返回新 view 避免 autograd 的 inplace 告警

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        return -ctx.lambd * grad_out, None  # 第二个 None 对应 lambd


# ---------------------------------------------------------------------------
# 7. 教学用：非张量参数 & 二阶导
# ---------------------------------------------------------------------------


class ScaleBy(Function):
    """y = x * scale，其中 scale 是 python float（非张量）。

    面试要点：**backward 返回值的个数必须等于 forward 的输入个数**，
    对非张量参数（float/int/bool/str）返回 None 占位。少返回一个会报
    "expected to return 2 gradients but got 1"，多返回则报相反的错。
    这是自定义 Function 最常见的运行时错误。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."], scale: float) -> Float[Tensor, "..."]:
        ctx.scale = scale
        return x * scale

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        return grad_out * ctx.scale, None


class BuggyScaleBy(Function):
    """反面教材：backward 少返回了 scale 对应的 None 占位。"""

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."], scale: float) -> Float[Tensor, "..."]:
        ctx.scale = scale
        return x * scale

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        return grad_out * ctx.scale  # 缺 None


class MyCube(Function):
    r"""y = x^3，用来演示二阶导。

    面试要点 —— **自定义 Function 什么时候支持 double backward**？
      当你用 ``create_graph=True`` 反向时，autograd 会把 ``backward()`` 里的运算
      **本身也录进计算图**，再对这张图求一次导。所以只要满足两个条件就白拿二阶导：
        1. backward 全部用可微的 torch 算子写（不要用 ``.data`` / ``.detach()`` /
           numpy / 原地写 buffer）；
        2. 没有加 ``@torch.autograd.function.once_differentiable`` 装饰器
           —— 加了它就是显式声明「我只支持一阶导」，二阶时会直接报错。
      这里 backward 是 ``3 * x^2 * g``，对 x 再求导得 ``6 * x * g``，
      对 g 再求导得 ``3 * x^2``，都是可微算子，所以二阶导自动可得：d2y/dx2 = 6x。

    什么时候需要二阶导？WGAN-GP 的梯度惩罚、MAML 等元学习、Hessian-vector product。
    """

    @staticmethod
    def forward(ctx, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        ctx.save_for_backward(x)
        return x.pow(3)

    @staticmethod
    def backward(ctx, grad_out: Float[Tensor, "..."]):
        (x,) = ctx.saved_tensors
        return 3.0 * x.pow(2) * grad_out


# ===========================================================================
#                                  tests
# ===========================================================================


def _double_input(*shape: int, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.double, requires_grad=True)


# ---- MyReLU ---------------------------------------------------------------


def test_relu_forward_matches_official():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    torch.testing.assert_close(MyReLU.apply(x), F.relu(x))


def test_relu_backward_matches_official():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    MyReLU.apply(a).pow(2).sum().backward()
    F.relu(b).pow(2).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)


def test_relu_gradcheck():
    # 把输入推离 0：relu 在 0 处不可导，中心差分会跨过折点而失败
    x = _double_input(3, 4, seed=0)
    with torch.no_grad():
        x += 0.5 * torch.sign(x)
    assert torch.autograd.gradcheck(MyReLU.apply, (x,))


def test_relu_subgradient_at_zero_is_zero():
    """x=0 时 PyTorch 约定次梯度取 0（用 x > 0 而非 x >= 0）。"""
    x = torch.zeros(3, requires_grad=True)
    MyReLU.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros(3))


# ---- MyGELU ---------------------------------------------------------------


def test_gelu_forward_matches_official_tanh_approx():
    torch.manual_seed(0)
    x = torch.randn(5, 7) * 2
    torch.testing.assert_close(
        MyGELU.apply(x), F.gelu(x, approximate="tanh"), rtol=1e-5, atol=1e-6
    )


def test_gelu_tanh_approx_differs_from_exact():
    """考点：F.gelu 默认是 erf 精确版，和 tanh 近似有 ~1e-3 的差。"""
    torch.manual_seed(0)
    x = torch.randn(1000) * 3
    diff = (MyGELU.apply(x) - F.gelu(x)).abs().max().item()
    assert 1e-5 < diff < 1e-2, f"两种 GELU 应有小但非零的差异，实测 {diff}"


def test_gelu_backward_matches_official():
    torch.manual_seed(0)
    x = torch.randn(5, 7) * 2
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    MyGELU.apply(a).sum().backward()
    F.gelu(b, approximate="tanh").sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


def test_gelu_gradcheck():
    x = _double_input(3, 5, seed=1)
    assert torch.autograd.gradcheck(MyGELU.apply, (x,))


def test_gelu_derivative_is_negative_somewhere():
    """GELU 非单调：x < -0.75 时导数为负，最小值 ≈ -0.129 在 x ≈ -1.42。"""
    x = torch.tensor([-1.42, -1.0, 0.0, 1.0], requires_grad=True)
    MyGELU.apply(x).sum().backward()
    assert x.grad[0].item() < 0 and x.grad[1].item() < 0
    assert x.grad[2].item() > 0 and x.grad[3].item() > 0
    torch.testing.assert_close(x.grad[0], torch.tensor(-0.1290), rtol=0, atol=1e-3)


# ---- MySoftmaxFunction ----------------------------------------------------


def test_softmax_fn_forward_matches_official():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 6)
    torch.testing.assert_close(MySoftmaxFunction.apply(x), torch.softmax(x, dim=-1))


def test_softmax_fn_backward_matches_official():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 6)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    w = torch.randn(3, 4, 6)
    (MySoftmaxFunction.apply(a) * w).sum().backward()
    (torch.softmax(b, dim=-1) * w).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)


def test_softmax_fn_gradcheck():
    x = _double_input(4, 5, seed=2)
    assert torch.autograd.gradcheck(MySoftmaxFunction.apply, (x,))


def test_softmax_fn_grad_sums_to_zero():
    """自检性质：softmax 输入梯度沿归一化维求和恒为 0。"""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    (MySoftmaxFunction.apply(x) * torch.randn(4, 8)).sum().backward()
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(4), atol=1e-6, rtol=0)


def test_softmax_fn_wrong_diagonal_formula_is_detected():
    """反面教材：dx = dy*y*(1-y) 是 sigmoid 的导数，不是 softmax 的。"""
    torch.manual_seed(0)
    x = torch.randn(2, 5, requires_grad=True)
    dy = torch.randn(2, 5)
    (MySoftmaxFunction.apply(x) * dy).sum().backward()
    y = torch.softmax(x.detach(), dim=-1)
    wrong = dy * y * (1 - y)
    assert not torch.allclose(x.grad, wrong, atol=1e-4)


# ---- MyLayerNormFunction --------------------------------------------------


def test_layernorm_fn_forward_matches_official():
    torch.manual_seed(0)
    d = 8
    x = torch.randn(3, 4, d) * 3 + 1
    w, b = torch.randn(d), torch.randn(d)
    torch.testing.assert_close(
        MyLayerNormFunction.apply(x, w, b, 1e-5),
        F.layer_norm(x, (d,), w, b, 1e-5),
        rtol=1e-5,
        atol=1e-6,
    )


def test_layernorm_fn_backward_matches_official():
    torch.manual_seed(0)
    d = 8
    x0, w0, b0 = torch.randn(3, 4, d) * 3, torch.randn(d), torch.randn(d)
    mine = [t.clone().requires_grad_(True) for t in (x0, w0, b0)]
    ref = [t.clone().requires_grad_(True) for t in (x0, w0, b0)]
    up = torch.randn(3, 4, d)

    (MyLayerNormFunction.apply(*mine, 1e-5) * up).sum().backward()
    (F.layer_norm(ref[0], (d,), ref[1], ref[2], 1e-5) * up).sum().backward()
    for m, r, name in zip(mine, ref, ["dx", "dw", "db"]):
        torch.testing.assert_close(m.grad, r.grad, rtol=1e-4, atol=1e-5, msg=name)


def test_layernorm_fn_gradcheck():
    torch.manual_seed(3)
    d = 5
    x = torch.randn(4, d, dtype=torch.double, requires_grad=True)
    w = torch.randn(d, dtype=torch.double, requires_grad=True)
    b = torch.randn(d, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(MyLayerNormFunction.apply, (x, w, b, 1e-5))


def test_layernorm_fn_grad_orthogonality():
    """自检：dx 同时正交于全 1 方向和 xhat 方向（affine 为恒等时）。"""
    torch.manual_seed(0)
    d = 16
    x = torch.randn(4, d, requires_grad=True)
    w, b = torch.ones(d), torch.zeros(d)
    y = MyLayerNormFunction.apply(x, w, b, 1e-5)
    (y * torch.randn(4, d)).sum().backward()
    xhat = y.detach()
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        (x.grad * xhat).sum(-1), torch.zeros(4), atol=1e-4, rtol=0
    )


def test_layernorm_fn_naive_wrong_backward_is_detected():
    """反面教材：只写 dx = g * rstd（漏掉 mu / var 两条路径）会被 gradcheck 抓住。"""

    class NaiveLN(Function):
        @staticmethod
        def forward(ctx, x):
            mu = x.mean(-1, keepdim=True)
            rstd = torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
            ctx.save_for_backward(rstd)
            return (x - mu) * rstd

        @staticmethod
        def backward(ctx, g):
            (rstd,) = ctx.saved_tensors
            return g * rstd  # 错误：漏了两项

    x = _double_input(3, 6, seed=7)
    assert not torch.autograd.gradcheck(NaiveLN.apply, (x,), raise_exception=False)


# ---- STE ------------------------------------------------------------------


def test_ste_forward_is_round():
    x = torch.tensor([-1.6, -0.4, 0.4, 1.5, 2.5])
    torch.testing.assert_close(RoundSTE.apply(x), torch.round(x))


def test_ste_backward_passes_gradient_through():
    x = torch.tensor([-0.4, 0.3, 0.9], requires_grad=True)
    (RoundSTE.apply(x) * torch.tensor([2.0, 3.0, 4.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([2.0, 3.0, 4.0]))


def test_ste_clips_outside_quantization_range():
    x = torch.tensor([-3.0, 0.5, 5.0], requires_grad=True)
    RoundSTE.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 0.0]))


def test_ste_fails_gradcheck_by_design():
    """STE 的 backward 不是 forward 的真导数，gradcheck 必然失败 —— 这是设计不是 bug。

    round 是分段常数函数，数值差分给出的雅可比处处为 0；解析梯度是 1。
    """
    x = _double_input(3, 3, seed=0) * 0.3  # 避开 .5 边界
    x = x.detach().requires_grad_(True)
    assert not torch.autograd.gradcheck(RoundSTE.apply, (x,), raise_exception=False)


def test_ste_equivalent_one_liner():
    """x + (x.round() - x).detach() 是 STE 的一行写法；差别只在 clip 上。

    考点：一行写法**不带 clip**，量化区间外的梯度照样透传。
    """
    x0 = torch.tensor([-0.9, -0.2, 0.3, 0.8])  # 全部落在 |x| <= 1 之内
    a = x0.clone().requires_grad_(True)
    b = x0.clone().requires_grad_(True)
    (a + (a.round() - a).detach()).sum().backward()
    RoundSTE.apply(b).sum().backward()
    torch.testing.assert_close(a + (a.round() - a).detach(), RoundSTE.apply(b))
    torch.testing.assert_close(a.grad, b.grad)

    # 区间外：一行写法透传 1，clipped STE 给 0
    c = torch.tensor([5.0], requires_grad=True)
    d = torch.tensor([5.0], requires_grad=True)
    (c + (c.round() - c).detach()).sum().backward()
    RoundSTE.apply(d).sum().backward()
    torch.testing.assert_close(c.grad, torch.tensor([1.0]))
    torch.testing.assert_close(d.grad, torch.tensor([0.0]))


def test_sign_ste_binarization():
    torch.manual_seed(0)
    x = torch.randn(6, requires_grad=True)
    y = SignSTE.apply(x)
    assert set(y.unique().tolist()) <= {-1.0, 0.0, 1.0}
    (y * 2).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((6,), 2.0))


# ---- GradientReversal -----------------------------------------------------


def test_gradient_reversal_forward_is_identity():
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    torch.testing.assert_close(GradientReversal.apply(x, 0.7), x)


def test_gradient_reversal_flips_gradient():
    torch.manual_seed(0)
    x0 = torch.randn(3, 4)
    lambd = 0.7
    a = x0.clone().requires_grad_(True)
    b = x0.clone().requires_grad_(True)
    GradientReversal.apply(a, lambd).pow(2).sum().backward()
    b.pow(2).sum().backward()
    torch.testing.assert_close(a.grad, -lambd * b.grad)


def test_gradient_reversal_lambda_zero_blocks_gradient():
    """lambda=0 等价于 detach，DANN 的 warmup 阶段就用这个。"""
    x = torch.randn(3, requires_grad=True)
    GradientReversal.apply(x, 0.0).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros(3))


def test_gradient_reversal_gradcheck_behavior():
    """lambda=1 时 gradcheck 失败（符号反了），lambda=-1 退化成恒等则通过。"""
    x = _double_input(3, 4, seed=5)
    assert not torch.autograd.gradcheck(
        GradientReversal.apply, (x, 1.0), raise_exception=False
    )
    assert torch.autograd.gradcheck(GradientReversal.apply, (x, -1.0))


# ---- 非张量参数 / 二阶导 ---------------------------------------------------


def test_backward_returns_none_for_non_tensor_arg():
    """forward 有非 tensor 参数时，backward 必须返回等长的元组，用 None 占位。"""
    torch.manual_seed(0)
    x = torch.randn(4, requires_grad=True)
    y = ScaleBy.apply(x, 3.0)
    torch.testing.assert_close(y, x.detach() * 3.0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full((4,), 3.0))

    # 少返回一个占位符 -> 运行时报错
    xb = torch.randn(4, requires_grad=True)
    try:
        BuggyScaleBy.apply(xb, 3.0).sum().backward()
    except RuntimeError as e:
        assert "gradient" in str(e).lower(), f"应报梯度个数不匹配，实际: {e}"
    else:
        raise AssertionError("backward 少返回 None 占位符时应当报错")


def test_double_backward():
    """用 create_graph=True 对自定义 Function 求二阶导：y=x^3 => d2y/dx2 = 6x。"""
    x = torch.tensor([2.0, -3.0, 0.5], requires_grad=True)
    y = MyCube.apply(x)

    (grad1,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    torch.testing.assert_close(grad1, 3.0 * x.detach() ** 2)

    (grad2,) = torch.autograd.grad(grad1.sum(), x)
    torch.testing.assert_close(grad2, 6.0 * x.detach())


def test_double_backward_gradgradcheck():
    """gradgradcheck 一次性验证一阶和二阶导，比手算参考值更严格。"""
    x = _double_input(3, 4, seed=9)
    assert torch.autograd.gradgradcheck(MyCube.apply, (x,))
    assert torch.autograd.gradgradcheck(MyGELU.apply, (_double_input(3, 4, seed=10),))


def test_once_differentiable_blocks_double_backward():
    """加了 @once_differentiable 就是声明只支持一阶导，二阶时会直接报错。"""
    from torch.autograd.function import once_differentiable

    class OnceCube(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.pow(3)

        @staticmethod
        @once_differentiable
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return 3.0 * x.pow(2) * g

    x = torch.tensor([2.0], requires_grad=True)
    (grad1,) = torch.autograd.grad(OnceCube.apply(x).sum(), x, create_graph=True)
    try:
        torch.autograd.grad(grad1.sum(), x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("once_differentiable 之后二阶导应当报错")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
