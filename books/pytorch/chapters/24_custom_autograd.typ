#import "../template.typ": *

= 手写 autograd Function：backward 全靠自己推

上一章手写的是 forward，autograd 帮你把 backward 免费算了。这一章反过来：*forward 和 backward 都你写*。面试官问这块，考的不是 API，是"你知不知道 autograd 到底在做什么"——backward 收到的是什么、要返回几个东西、返回的东西必须是真导数吗、你怎么证明你推对了。

前三个问题有标准答案，第四个问题的标准答案是一句话：*跑 `gradcheck`*。能主动说出这句话的候选人，和只会写 `return grad_out * something` 的候选人，是两个段位。本章代码的可运行版本 + pytest 在 `python/pytorch/interview/test_custom_autograd.py`（33 个测试，`pytest python/pytorch/interview/test_custom_autograd.py -q` 直接跑）。章节实现去掉了 jaxtyping 标注，逻辑一致。autograd 引擎本身的机制（VJP、`grad_fn`、拓扑序）见第 6 章，这里只讲怎么往里插自己的节点。

== 标准模板与五条规则

```python
import torch
from torch.autograd import Function

class MyOp(Function):
    @staticmethod
    def forward(ctx, x, scale):          # scale 是 python float，非 tensor
        ctx.save_for_backward(x)         # tensor 走 save_for_backward
        ctx.scale = scale                # 非 tensor 直接挂 ctx
        return x * scale

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        dx = grad_out * ctx.scale if ctx.needs_input_grad[0] else None
        return dx, None                  # 两个输入 -> 两个返回值

y = MyOp.apply(x, 3.0)                   # 用 apply，不是 MyOp()(x, 3.0)
```

#table(
  columns: (auto, 1.5fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*规则*], [*说明*],
  [`forward` / `backward` 都是 `staticmethod`],
    [状态全靠 `ctx` 传递。入口是 `MyOp.apply(...)`；`MyOp()(...)` 在 torch 2.x 上直接报 "Legacy autograd function with non-static forward method is deprecated"。],
  [backward 返回值个数 = forward 输入个数],
    [严格相等、位置一一对应，包括非 tensor 参数。少返回就报 "returned an incorrect number of gradients (expected 2, got 1)"，多返回同样报错。*这是最高频的运行时错误*。],
  [不需要梯度的位置返回 `None`],
    [非 tensor 参数（`float` / `int` / `bool` / `str`）永远返回 `None`。],
  [`ctx.save_for_backward` 只存 tensor],
    [存非 tensor 会报错；反过来把 tensor 挂 `ctx.xxx` 会绕过版本检查和引用计数。],
  [`ctx.needs_input_grad` 是 bool 元组],
    [长度等于 forward 输入个数。用它跳过不必要的计算，对应位置返回 `None`。],
)

*为什么 tensor 必须走 `save_for_backward`*，两个理由，第二个是面试的落点。一是*引用计数*：`save_for_backward` 让引擎知道这个 tensor 被反向图持有，`backward()` 跑完自动释放（这也是第二次 backward 会报 "Trying to backward through the graph a second time" 的原因）；挂在 `ctx.x` 上则一直活着，等于内存泄漏。二是*版本号检查*：`saved_tensors` 读取时会比对 version counter，如果这个 tensor 在 forward 之后被原地改过，直接报 "modified by an inplace operation"，而不是*静默算错*。挂 `ctx.x` 完全绕过这层保护。

另外两个 `ctx` 方法不常用，但一被问到就能拉开差距。`ctx.mark_dirty(x)` 用在 forward 里*原地修改了输入*的时候：它给 version counter +1 并更新图里的引用，否则任何 `saved_tensors` 持有它的节点都会拿到被改过的数据；调了之后 `apply` 的返回值和输入*是同一个 Python 对象*，这是实现 `relu_` / 原地量化这类算子唯一的正确姿势。`ctx.mark_non_differentiable(i)` 用在某个输出天然没有梯度（整数索引、掩码、计数）的时候：标记后它的 `requires_grad` 恒为 `False`，用户对它 backward 会直接报错而不是拿到一堆 0——`torch.max(x, -1).indices.requires_grad` 永远是 `False` 就是这么实现的。注意*即使标记了，`backward` 的形参个数仍然等于 forward 的输出个数*，那个用不到的 `grad` 也要写在签名里。

#figure(
  align(center, autograd-graph(
    tensors: (("x", "leaf, requires_grad"), ("y", "MyOp.apply(x, 3.0)"), ("loss", "() scalar")),
    ops: (("y = MyOp.apply(x, s)", "MyOpBackward"), ("loss = y.sum()", "SumBackward0")),
  )),
  caption: [自定义 Function 在图里就是一个普通节点。`apply` 时引擎自动生成一个名为 `<类名>Backward` 的 `grad_fn` 并把 `ctx` 塞进去；`backward()` 走到它时调用你写的 `backward`。],
) <fig-custom-fn>

#note[
  torch 2.0+ 推荐把"存 context"从 `forward` 拆到独立的 `setup_context(ctx, inputs, output)`，让 `forward` 成为纯函数，从而能被 `torch.func`（`vmap`）正确处理。两种写法都还支持：纯训练脚本用旧写法没问题，写库就用新写法。签名细节见第 6 章。
]

== MyReLU：最简模板

```python
class MyReLU(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0                     # 存 bool（1 byte）比存 x（4 byte）省 4 倍
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        return grad_out * mask
```

*存输入还是存输出，是设计 activation kernel 的第一个决定。* 官方 `relu` 比上面更省：它存*输出* `y`，因为 $y > 0 <==> x > 0$，而 `y` 本来就要传给下一层，等于*零额外显存*。这也正是 ReLU 能安全 inplace（`relu_`）的原因——反向不需要原始输入。同理 `sigmoid` / `tanh` 也存输出（导数能用输出表达：$y(1-y)$、$1-y^2$）；而 `GELU` / `SiLU` 只能存输入，它们的导数无法只用输出写出来。

$x = 0$ 处 ReLU 不可导，PyTorch 约定次梯度取 0，实现上就是 `x > 0` 而不是 `x >= 0`。所以 `gradcheck` 的输入要避开 0 附近——中心差分会跨过折点，两边斜率不一致，必然失败；测试里用 `x += 0.5 * x.sign()` 把输入推离折点。

== MyGELU：手推导数

BERT / GPT-2 用的是 tanh 近似版：

#formula[$ s(x) = c (x + a x^3), quad c = sqrt(2/pi) approx 0.7979, quad a = 0.044715 \
"gelu"(x) = 1/2 x (1 + tanh s(x)) $]

*反向手推*（面试要求当场写）。令 $t = tanh s$，用 $(d t)/(d s) = 1 - t^2$ 和 $(d s)/(d x) = c (1 + 3 a x^2)$，对 $1/2 x (1 + t)$ 用乘法法则：

#formula[$ (d "gelu")/(d x) = underbrace(1/2 (1 + t), "对前面的 " x " 求导") + underbrace(1/2 x (1 - t^2) c (1 + 3 a x^2), "对 tanh 求导") $]

漏掉第二项是最常见的错误——它是"门控因子"的贡献，$x$ 大的时候不可忽略。

```python
import math
_C = math.sqrt(2.0 / math.pi)
_A = 0.044715

class MyGELU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)         # 存 x，backward 里重算 s 和 t
        s = _C * (x + _A * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(s))

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        s = _C * (x + _A * x.pow(3))
        t = torch.tanh(s)
        dsdx = _C * (1.0 + 3.0 * _A * x.pow(2))
        local = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dsdx
        return grad_out * local
```

这里存 `x` 然后在 backward 里重算 `s` 和 `t`，是 activation checkpointing 的微观版本：多两次 elementwise（在 memory-bound 的层上几乎免费），省掉一个和输入等大的张量。

=== 考点一：`F.gelu` 默认不是 tanh 近似

精确版是 $0.5 x (1 + "erf"(x\/sqrt(2)))$。2018 年选 tanh 近似是因为当时部分硬件上 `erf` 没有快速实现；今天 `erf` 已经不慢，所以 PyTorch 把精确版设成了默认。

```python
F.gelu(x)                       # 默认 approximate='none'，走精确的 erf 版本
F.gelu(x, approximate='tanh')   # BERT / GPT-2 用的是这个
```

#warn[
  两者的最大绝对差约 *4.7e-4*（出现在 $x approx plus.minus 2.70$）。落在激活值上完全不影响收敛，但做数值对齐时一定会被 `assert_close` 抓到。所以 `assert_close(MyGELU.apply(x), F.gelu(x))` 会失败，*不是因为你推错了，是因为默认走 erf* ——对齐必须写 `F.gelu(x, approximate="tanh")`。复现 HuggingFace 老模型时发现输出对不上小数点后三位，先查这个。
]

=== 考点二：GELU 不单调

这和 ReLU 的直觉完全不同，是个很好的追问。GELU 的导数在 $x < -0.752$ 的*整个区间*上都是负的（之后单调回升趋向 0），最小值约 *−0.129*，出现在 $x approx -1.419$。

```python
x = torch.tensor([-1.42, -1.0, 0.0, 1.0], requires_grad=True)
MyGELU.apply(x).sum().backward()
x.grad     # [-0.1290, -0.0830,  0.5000,  1.0830]  前两个是负的
```

#warn[
  网上常见的说法是"GELU 的导数在 $x approx -0.75$ 附近为负"——这是把*导数的零点*（$x approx -0.752$，导数由正转负的地方，也正是 GELU 函数自身的极小值点）当成了*导数的极小点*（$x approx -1.419$）。极小点在更左边。被问到时把这两个数分清楚，比记住一个模糊的"大约 −0.75"强得多。

  非单调的意义是"轻度的负输入被小幅保留，更负的输入反而被推回 0"——一种平滑的软门控，比 ReLU 的硬截断多一点表达力。
]

== Softmax 的 backward：最高频的手推题

设 $y_i = e^(x_i) \/ sum_k e^(x_k)$。$i = j$ 时商法则给出 $y_i - y_i^2$，$i != j$ 时给出 $-y_i y_j$，两种情况合并成一条：

#formula[$ (partial y_i)/(partial x_j) = y_i (delta_(i j) - y_j), quad quad J = "diag"(y) - y y^T $]

现在做 VJP。记上游梯度 $g_i = (partial L)\/(partial y_i)$：

#formula[$ (partial L)/(partial x_j) = sum_i g_i y_i (delta_(i j) - y_j) = g_j y_j - y_j sum_i g_i y_i = y_j (g_j - sum_i g_i y_i) $]

向量化就是那条要背下来的式子：

#formula[$ d x = (d y - (d y ⊙ y) dot.op "sum"_(-1)) ⊙ y $]

```python
class MySoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.exp(x - x.amax(dim=-1, keepdim=True))
        y = y / y.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(y)          # 只需要 y，不需要 x
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        return (grad_out - (grad_out * y).sum(dim=-1, keepdim=True)) * y
```

四个要点：

+ *绝不显式构造雅可比*。$D$ 维输入的 $J$ 是 $(D, D)$；对 $(B, S, D)$ 的输入就是 $(B, S, D, D)$，$S = D = 4096$ 时单是这一个张量就上百 GB。上面那条式子把它压成两次 elementwise + 一次 reduction，$O(D)$ 显存。这是"VJP 而不是 Jacobian"最具体的一次体现。
+ *backward 只需要 `y`，不需要 `x`*，所以存输出。attention 的反向能只保存 softmax 结果就是这个道理；FlashAttention 更狠，连 `y` 都不存，只存 $(B, H, S)$ 的 logsumexp 在反向里重算。
+ $(d y ⊙ y)$ 求和这一项的物理含义是*"上游梯度的加权平均"*（权重就是概率 $y$）。减掉它保证了 $sum_j d x_j = 0$：softmax 的输出被约束在单纯形上，沿全 1 方向扰动输入不改变输出，所以那个方向上的梯度必然为 0。这是一条*免费的自检*。
+ 和 cross-entropy 融合时，$g = -"onehot" \/ y$ 代入整条式子恰好塌缩成 $d x = y - "onehot"$。这就是 `softmax_cross_entropy` 的反向如此简洁、也如此适合融成一个 kernel 的原因。

#warn[
  最经典的错法：`dx = dy * y * (1 - y)`。这只保留了雅可比的对角线，是 *sigmoid* 的导数，不是 softmax 的。它丢掉了 $-y_j sum_i g_i y_i$ 这一项，也就丢掉了"各个类别之间此消彼长"的耦合。这个错误在 shape 上完全正确，loss 也会下降（方向大致对），$D = 2$ 时数值上还很接近——只有 `gradcheck` 能立刻抓出来。它还有个特征：不满足 $sum_j d x_j = 0$。
]

== LayerNorm 的 backward：难度最高的手推题

面试里能完整推到最后那条紧凑形式，这题基本就过了。设一行输入 $x in RR^D$：

#formula[$ mu = 1/D sum_i x_i, quad sigma^2 = 1/D sum_i (x_i - mu)^2, quad r = 1/sqrt(sigma^2 + epsilon) \
hat(x) = (x - mu) r, quad y = w ⊙ hat(x) + b $]

*参数梯度*是简单的部分，在除最后一维外的所有维度上求和（`w` / `b` 是 $(D,)$，而 `dy` 是 $(B, S, D)$）：

#formula[$ (partial L)/(partial w) = sum_"batch" (d y ⊙ hat(x)), quad quad (partial L)/(partial b) = sum_"batch" d y $]

*输入梯度*要走三条路径。先把 affine 剥掉：令 $g = d y ⊙ w = (partial L)\/(partial hat(x))$。关键在于 $x_i$ 影响 $hat(x)$ 有*直接、经 $mu$、经 $sigma^2$* 三条路，三条都要算：

#formula[$ (partial L)/(partial sigma^2) = sum_i g_i (x_i - mu) dot (-1/2) (sigma^2 + epsilon)^(-3\/2) $]
#formula[$ (partial L)/(partial mu) = -r sum_i g_i quad quad #text[（经 $sigma^2$ 的那一项含 $sum_i (x_i - mu) = 0$，自动消失）] $]
#formula[$ (partial L)/(partial x_i) = underbrace(g_i r, "直通") + underbrace((partial L)/(partial sigma^2) dot 2/D (x_i - mu), "经方差") + underbrace((partial L)/(partial mu) dot 1/D, "经均值") $]

*化简*：把 $x_i - mu = hat(x)_i \/ r$ 代回去，方差那一项变成

#formula[$ (partial L)/(partial sigma^2) dot 2/D (x_i - mu) = -1/D r^3 (x_i - mu) sum_j g_j (x_j - mu) = -r hat(x)_i dot 1/D sum_j g_j hat(x)_j $]

均值那一项直接是 $-r dot (1\/D) sum_j g_j$。三项合并，得到那条著名的紧凑形式（两个 `mean` 都是*沿归一化维*，不是全局均值）：

#formula[$ d x = r ( g - "mean"(g) - hat(x) ⊙ "mean"(g ⊙ hat(x)) ) $]

```python
class MyLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        xhat = (x - mu) * rstd
        ctx.save_for_backward(xhat, rstd, weight)   # 存 xhat/rstd，不是 x/mu/var
        return xhat * weight + bias

    @staticmethod
    def backward(ctx, grad_out):
        xhat, rstd, weight = ctx.saved_tensors
        g = grad_out * weight                        # dL/dxhat

        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = rstd * (
                g
                - g.mean(dim=-1, keepdim=True)
                - xhat * (g * xhat).mean(dim=-1, keepdim=True)
            )
        reduce_dims = tuple(range(grad_out.dim() - 1))
        if ctx.needs_input_grad[1]:
            dw = (grad_out * xhat).sum(dim=reduce_dims)
        if ctx.needs_input_grad[2]:
            db = grad_out.sum(dim=reduce_dims)
        return dx, dw, db, None      # 第 4 个输入 eps 是 float
```

*几何直观（能说出来就是加分项）.* 归一化把 $x$ 投影到"均值为 0、模长为 $sqrt(D)$"的球面上。沿两个方向移动 $x$ 根本不改变 $y$：*全 1 方向*（整体平移，$mu$ 跟着变，$x - mu$ 不变）和 *$hat(x)$ 方向*（整体缩放，$sigma$ 跟着变，$(x-mu)\/sigma$ 不变）。既然输出不变，梯度在这两个方向上的分量就必须为 0——公式里的 $-"mean"(g)$ 正是在减掉全 1 方向的分量，$-hat(x) dot "mean"(g ⊙ hat(x))$ 正是在减掉 $hat(x)$ 方向的分量。*这条式子就是一个正交投影。* 推论是两条免费自检：

#formula[$ sum_j d x_j = 0 quad quad #text[且] quad quad sum_j d x_j hat(x)_j = 0 $]

#warn[
  最常见的错法：只写 `dx = g * rstd`，漏掉 `mu` 和 `var` 两条路径。这个错误极其隐蔽——漏掉的两项是投影分量，量级通常比主项小，方向大体是对的，小 batch 上 *loss 照样会下降*，小模型甚至能训到收敛，只是慢一截、上限低一截。没有报错、没有 NaN，只有"我的实现好像比官方差一点"。`gradcheck` 一秒抓出来（`test_layernorm_fn_naive_wrong_backward_is_detected` 就是专门演示这个的反面教材）。
]

两个工程细节值得一起记住：kernel 里存 `(mean, rstd)` 这两个 $(B S,)$ 的小张量而不是 `var`，省一次 sqrt 和除法；`dw` / `db` 是跨 batch 的 reduction，要做两阶段规约（每个 block 先累加到 partial buffer，第二个 kernel 汇总），直接 `atomicAdd` 到 $(D,)$ 上会被 $B S$ 个 block 抢同一批地址的写冲突拖垮。

== gradcheck：怎么证明你推对了

`gradcheck` 用*数值差分*（中心差分）逐元素算出雅可比，和你的 `backward` 给出的解析雅可比逐项比对。它是*唯一*能抓出"公式推错但 shape 正确"的手段。

```python
from torch.autograd import gradcheck, gradgradcheck

x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
assert gradcheck(MyGELU.apply, (x,))                          # 一阶
assert gradgradcheck(MyGELU.apply, (x,))                      # 一阶 + 二阶
ok = gradcheck(RoundSTE.apply, (x,), raise_exception=False)    # 返回 False 而不抛异常
```

=== 必须用 float64

这是最容易被追问、也最容易答错的一点。中心差分的误差有两个来源：

#formula[$ (f(x+h) - f(x-h)) / (2h) = f'(x) + underbrace(O(h^2), "截断误差") + underbrace(O(epsilon_"mach" \/ h), "舍入误差") $]

两项一个随 $h$ 减小、一个随 $h$ 增大，最优 $h approx epsilon_"mach"^(1\/3)$，此时总误差约 $epsilon_"mach"^(2\/3)$：

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, right, right, left),
  [*dtype*], [*机器 eps*], [*可达相对误差*], [*够不够 `atol=1e-5`*],
  [fp32], [1.19e-7], [约 1e-5], [不够：数值误差本身就到了容差],
  [fp64], [2.22e-16], [约 1e-11], [足够，留了 6 个数量级余量],
)

用 fp32 跑 gradcheck，*数值微分自己的误差会掩盖真实的梯度错误*——公式错了 1% 它也可能通过，公式全对它也可能失败。所以 gradcheck 的输入一律 `dtype=torch.double`。

参数上还有三点：`eps`（差分步长）默认 `1e-6`，配 fp64 刚好，*不要因为"失败了"就去调大它，先怀疑公式*；`raise_exception=False` 返回 `bool` 而不抛异常，用途是*断言某个实现应该失败*（测 STE、测反面教材时必须用它，否则 pytest 直接被异常打断）；`nondet_tol` 只在算子有非确定性（atomicAdd 之类）时才用。

#warn[
  gradcheck 失败时先排除三个"不是你的错"的原因：输入落在不可导点上（ReLU 的 0、`abs` 的 0、`round` 的 .5）；用了 fp32；算子里有随机性（dropout 之类）导致两次 forward 结果不同。这三个排除完，才是公式真的推错了。
]

=== 二阶导：白拿与显式禁用

用 `create_graph=True` 反向时，autograd 会把 `backward()` *里面的运算本身也录进计算图*，再对这张图求一次导。所以只要满足两个条件，二阶导自动可得、你什么都不用写：backward 全部用*可微的 torch 算子*写（不要用 `.data` / `.detach()` / numpy / 原地写 buffer——这些会把链条切断，二阶导*静默变成 0*，比报错更可怕），并且没有加 `@once_differentiable`。

```python
class MyCube(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.pow(3)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return 3.0 * x.pow(2) * grad_out      # 全是可微算子

x = torch.tensor([2.0, -3.0, 0.5], requires_grad=True)
(g1,) = torch.autograd.grad(MyCube.apply(x).sum(), x, create_graph=True)
(g2,) = torch.autograd.grad(g1.sum(), x)
# g1 = 3x^2, g2 = 6x —— 二阶导没写一行额外代码
```

反过来，`@torch.autograd.function.once_differentiable`（加在 `@staticmethod` 下面）做两件事：在 `no_grad()` 下执行 backward（省掉建图开销），并给输出打标记，二阶导时*直接报错*而不是静默给错误结果。什么时候该加：backward 里调了 C++/CUDA 扩展、numpy 或任何不可微的东西时——这时二阶导本来就是错的，与其让用户拿到一个静默错误的 Hessian，不如显式报错。

真的需要二阶导的场景：WGAN-GP 的梯度惩罚（对梯度范数再求导）、MAML 等元学习、Hessian-vector product（见第 6 章）、以及一些二阶优化器。这些链路里只要有一个 `@once_differentiable` 的算子，整条就断了。

== Straight-Through Estimator：backward 可以不是真导数

```python
class RoundSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x.abs() <= 1.0)   # clipped STE：量化区间外不透传
```

`round` / `sign` / `argmax` 这类算子的真导数*几乎处处为 0*，梯度一到这里就断了，前面的层永远学不到东西。STE 的做法是"前向用离散值保证和推理一致，反向假装这一步是恒等函数"。两大用途：*量化感知训练（QAT）*（前向 fake-quantize 到 int8，反向按 fp32 更新）和*离散 latent*（VQ-VAE 的 codebook 查找、Gumbel-softmax 的硬采样分支）。

它是一个*有偏估计器*，不对应任何函数的真导数。能 work 的直觉是把 `round(x)` 看成 $x + "noise"$——一个加了零均值扰动的恒等映射，那么恒等的导数 1 就是个合理的代理。而 `grad * (|x| <= 1)` 这个 clip 不是可有可无的。落在量化区间外的输入已经饱和，再怎么推也不会改变输出，透传梯度纯属噪声，还会让权重一路漂向无穷。这叫 clipped STE 或 hardtanh STE，实践里几乎都用这个版本。

*一行写法*（面试里能说出来是加分项）：`y = x + (x.round() - x).detach()`。前向等于 `x.round()`，反向因为括号被 detach 了、梯度直接以系数 1 从 `y` 流到 `x`。但它*不带 clip*（区间外照样透传 1），也没法定制门函数，所以生产代码还是写 `Function`。

#warn[
  *STE 必然通不过 `gradcheck`，这是设计不是 bug。* `round` 是分段常数函数，数值差分给出的雅可比处处为 0（只要采样点不跨过 .5 边界），而你的解析梯度是 1，永远对不上。所以*不要因为 `gradcheck` fail 就去"修" STE*。被问到"你怎么验证 STE"，正确答案不是 gradcheck，而是：(1) 断言前向严格等于 `torch.round`；(2) 断言区间内梯度按预期透传、区间外为 0；(3) 端到端跑一个小 QAT 看量化后精度是否收敛。
]

== GradientReversal：DANN 的核心

```python
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd          # float 不能 save_for_backward
        return x.view_as(x)        # 恒等，返回新 view 避免 inplace 告警

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None
```

#figure(
  align(center, flow-boxes(boxes: ("feature extractor", "GRL (forward 恒等)", "domain classifier"))),
  caption: [DANN 的结构。forward 时 GRL 什么也不做；backward 时它把 domain classifier 回传的梯度乘 $-lambda$，于是特征提取器在*最大化* domain loss，而 classifier 在最小化它。一次前向、一个 optimizer 就实现了 min-max 对抗。],
) <fig-grl>

*用途.* domain adaptation 的目标是让特征提取器学到的表示*分不出*样本来自源域还是目标域。做法是接一个 domain classifier 去分辨，然后让特征提取器和它对着干。按 GAN 的思路做要交替更新两个网络、维护两个 optimizer、调交替比例；GRL 把这一切压成一个层：classifier 正常最小化自己的 loss，回传到特征提取器的梯度被翻转，等价于它在最大化同一个 loss。*一次 forward、一次 backward、一个 optimizer step。*

*这是"backward 不必是真导数"的最好例证.* autograd 只是忠实地按你写的 `backward` 做链式传播，它*没有任何机制*去校验你返回的东西是不是真的 VJP。理解这一点，就理解了 STE、梯度裁剪、`detach` 断路里各种"作弊"手法的合法性来源——它们全都是在合法地篡改反向传播。一个有趣的性质可以直接验证这件事：

```python
gradcheck(GradientReversal.apply, (x, 1.0), raise_exception=False)    # False：符号反了
gradcheck(GradientReversal.apply, (x, -1.0))                          # True：退化成恒等
```

$lambda = -1$ 时 backward 返回 $+g$，正好是恒等函数的真导数，于是 gradcheck 通过了。同一段代码，只因为一个常数就从"数学上错"变成"数学上对"——这恰好反衬出失败的唯一原因就是符号，也说明 gradcheck 检查的是*一致性*而不是*正确性*。

$lambda$ 通常带 schedule：训练初期设 0（等价于 `detach`，先让 domain classifier 自己学好），再按 $2\/(1 + e^(-10 p)) - 1$（$p$ 是训练进度）逐渐升到 1。早期就开满会让噪声梯度毁掉特征。

#warn[
  常见错法：把 forward 写成 `return -x`。那样*前向就变了*，domain classifier 看到的是取反后的特征，整个训练目标都不一样了。forward 必须是恒等，改的只有 backward。
]

== 什么时候真的需要自定义 Function

面试官问完实现，常会追一句"实际项目里你什么时候会写这个"。只有三类，其他情况都不需要：

+ *backward 有闭式解，比 autograd 逐 op 展开更省显存或更快*。autograd 会把每个中间量都存下来，而手写 backward 可以只存 `(mean, rstd)` 这种小量、或者用重算换显存。例子：fused LayerNorm / RMSNorm、fused softmax-cross-entropy、FlashAttention。
+ *forward 里调了不可微的外部代码*。autograd 看不进去，只能你自己告诉它梯度是什么。例子：自定义 CUDA / Triton kernel、通信原语（TP 里的 all-reduce）、第三方 C++ 库。
+ *需要"假梯度"*，即数学上的真导数没用或不存在。例子：STE（QAT / BNN）、GRL（DANN）、各种梯度手术。

*什么时候不需要，这一半更重要.* 绝大多数情况 autograd 自动展开就够了，而且更不容易错。只是"我觉得手写会快一点"就去写 Function，通常是净亏损：你放弃了 autograd 的正确性保证、放弃了白拿的二阶导、放弃了 `vmap` 兼容，还大概率被 `torch.compile` 甩在后面——Inductor 会把你手工融合的那几个 op 自动融掉。先 profile，确认这一段真的是瓶颈、且瓶颈真的来自 activation 显存或 kernel 数量，再动手。

#insight[
  自定义 Function 的成本是"你从此对这个节点的正确性全权负责"。收益必须能说清是省显存、省 kernel、还是实现了 autograd 表达不出来的语义。说不清就别写——这也是这道题的正确回答方式。
]

*与 `torch.compile` 的关系*一句话：`autograd.Function` 会导致 *graph break*。Dynamo 没法 trace 进你的 Python `backward`，只能把图切开，前后两段分别编译，中间回到 eager，跨算子 fusion 的机会就没了。torch 2.4+ 推荐的新路线是 `torch.library.custom_op` + `register_autograd`：把 op 注册成一个对编译器"不透明但已知"的算子（有 schema、有 `register_fake` 给出的 meta 实现、有单独注册的反向），Dynamo 就能把它当普通 op 放进图里而不 break。见第 7 章（dispatcher 与 op 注册）和第 12 章（graph break）。

== 面试考点

#interview[
  *Q1*：写一个 `autograd.Function` 要注意什么？backward 返回几个值？

  A：`forward` / `backward` 都是 `staticmethod`，调用用 `.apply()`（`MyOp()(x)` 在 torch 2.x 上直接报错）。*backward 返回值的个数严格等于 forward 输入的个数*（不含 `ctx`），非 tensor 参数（float/int/bool）返回 `None` 占位。tensor 一律用 `ctx.save_for_backward` 存，非 tensor 直接挂 `ctx.xxx`。用 `ctx.needs_input_grad` 跳过不必要的计算。forward 里原地改了输入要 `ctx.mark_dirty()`，返回不可微输出要 `ctx.mark_non_differentiable()`。
]

#interview[
  *Q2*：手推 softmax 的 backward。

  A：雅可比是 `dy_i/dx_j = y_i * (delta_ij - y_j)`，矩阵形式 `diag(y) - y y^T`。做 VJP：`dx_j = sum_i g_i y_i (delta_ij - y_j) = y_j (g_j - sum_i g_i y_i)`，向量化就是 `dx = (dy - (dy*y).sum(-1, keepdim=True)) * y`。关键是*不显式构造雅可比*（`(B,S,D,D)` 会 OOM），并且 backward 只需要输出 `y`——这就是 attention 反向只存 softmax 结果的原因。自检性质：`dx.sum(-1) == 0`。常见错法是写成 `dy*y*(1-y)`，那是 sigmoid 的导数。
]

#interview[
  *Q3*：手推 LayerNorm 的 backward。

  A：令 `g = dy * w`，`xhat = (x-mu)*rstd`。`x` 影响输出有三条路径（直通、经 mu、经 var），三条都算完合并化简，得 `dx = rstd * (g - mean(g) - xhat * mean(g*xhat))`，两个 mean 都沿归一化维。参数梯度 `dw = sum_batch(dy*xhat)`、`db = sum_batch(dy)`。几何上这就是把 `g` 对全 1 方向和 `xhat` 方向做正交投影——沿这两个方向移动 `x` 不改变输出，梯度分量必须为 0，所以 `dx.sum(-1)` 和 `(dx*xhat).sum(-1)` 都是 0。常见错法是只写 `dx = g * rstd`，loss 照样降、只有 gradcheck 抓得到。
]

#interview[
  *Q4*：什么是 STE？它能通过 gradcheck 吗？

  A：Straight-Through Estimator：前向做 `round`/`sign` 这类离散化，反向直接把梯度透传（通常还带 `|x| <= 1` 的 clip，因为区间外已饱和、透传的是纯噪声）。用在 QAT 和 VQ-VAE 这类离散 latent 上，因为这些算子的真导数几乎处处为 0，不作弊梯度就断了。它*必然通不过 gradcheck*——数值差分对分段常数函数给出的雅可比是 0，解析梯度是 1。这是设计不是 bug，验证要靠"前向严格等于 round + 梯度按预期透传 + 端到端收敛"。
]

#interview[
  *Q5*：gradcheck 为什么必须用 float64？

  A：中心差分的截断误差是 `O(h^2)`、舍入误差是 `O(eps/h)`，最优步长下总相对误差约 `eps^(2/3)`。fp32 的 eps 是 1.2e-7，能达到的精度只有 1e-5 量级，正好等于 gradcheck 默认的 `atol`——数值微分自身的误差就把真实的梯度错误淹没了。fp64 的 eps 是 2.2e-16，可达 1e-11，留了六个数量级余量。所以 gradcheck 的输入一律 `dtype=torch.double`。
]

#interview[
  *Q6*：自定义 Function 什么时候支持二阶导？

  A：两个条件：backward 里全部用可微的 torch 算子写（不碰 `.data` / `.detach()` / numpy / 原地写 buffer），并且没加 `@once_differentiable`。满足了就白拿——`create_graph=True` 时 autograd 会把 backward 里的运算本身也录进图再求一次导。加了 `@once_differentiable` 就是显式声明只支持一阶，二阶时直接报错（好过静默给错误的 Hessian）。用 `gradgradcheck` 一次验完一阶和二阶。需要二阶导的场景：WGAN-GP、MAML、Hessian-vector product。
]

#interview[
  *Q7*：backward 必须是 forward 的真导数吗？举个反例。

  A：不必须。autograd 只是忠实地按你写的 backward 做链式传播，引擎没有任何机制校验你返回的是不是真 VJP。反例：GRL（forward 恒等、backward 乘 $-lambda$，用一次 forward 实现 DANN 的 min-max 对抗，不用像 GAN 那样交替更新）、STE（forward 离散化、backward 透传）。判断标准从"数学上对不对"变成"训练动力学上想要什么"。代价是正确性由你承担——这类 Function 不能用 gradcheck 验，只能断言"梯度等于我设计的那个值"。
]

#interview[
  *Q8*：什么时候*不该*写自定义 Function？

  A：绝大多数时候。只是"觉得手写会快一点"就写通常是净亏损：放弃了 autograd 的正确性保证、放弃白拿的二阶导、放弃 `vmap` 兼容，还会*导致 graph break*（Dynamo 没法 trace 进你的 Python backward），反而拿不到 Inductor 的 fusion。真正的三个理由是：backward 有闭式解且能显著省显存（fused LayerNorm）、forward 调了不可微的外部代码、需要假梯度。新代码如果只是想包一个自定义 kernel，更推荐 `torch.library.custom_op` + `register_autograd`，它不会 break 图。
]
