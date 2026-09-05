#import "../template.typ": *

= Autograd：动态反向图的完整机制

面试问 PyTorch，绕不开 autograd。而且问法是分层的：初级问"`zero_grad` 为什么要调"，中级问"`retain_graph` 什么时候要开"，高级问"你自己写过 `autograd.Function` 吗，`ctx.needs_input_grad` 干什么用"。这一章把这条链从数学（VJP）一直讲到实现（version counter、`grad_fn.next_functions`），目标是：面试官往任何一层深挖，你都有话说。

本章只讲 autograd 引擎本身。它在 dispatcher 里是怎么一层 key 的，见第 7 章；activation 显存怎么算，见第 8 章；`torch.compile` 下的 AOTAutograd 怎么把这张图提前 trace 出来，见第 13 章。

== 反向图长什么样

autograd 图是 *forward 时动态构建的*。每执行一个 op，如果它的输入里有任何一个 `requires_grad=True`，PyTorch 就在输出 tensor 上挂一个 `grad_fn` 节点，并让这个节点的 `next_functions` 指回输入的 `grad_fn`。forward 跑完，反向图就已经建好了 —— `backward()` 只是沿着这些指针走一遍。

```python
import torch

x = torch.randn(4, 8, requires_grad=True)
W = torch.randn(8, 16, requires_grad=True)

h    = x @ W          # h.grad_fn    = <MmBackward0>
r    = h.relu()       # r.grad_fn    = <ReluBackward0>
loss = r.sum()        # loss.grad_fn = <SumBackward0>

loss.grad_fn.next_functions        # ((<ReluBackward0>, 0),)
r.grad_fn.next_functions           # ((<MmBackward0>, 0),)
h.grad_fn.next_functions           # ((<AccumulateGrad>, 0), (<AccumulateGrad>, 0))
```

最后一行是关键：`MmBackward0` 的两个输入 `x` 和 `W` 都是叶子，所以它们对应的节点是 `AccumulateGrad` —— 一个"把算出来的梯度加到 `.grad` 上"的特殊节点。整张图的叶子节点全是 `AccumulateGrad`。

#figure(
  align(center, autograd-graph(
    tensors: (
      ("x", "(4,8) leaf"),
      ("h", "(4,16)"),
      ("r", "(4,16)"),
      ("loss", "() scalar"),
    ),
    ops: (
      ("h = x @ W", "MmBackward0"),
      ("r = h.relu()", "ReluBackward0"),
      ("loss = r.sum()", "SumBackward0"),
    ),
  )),
  caption: [`loss = (x @ W).relu().sum()` 的 autograd 图。蓝色实线是 forward 建图方向，红色虚线是 `backward()` 的执行方向。`W` 那一支没画出来，它挂在 `MmBackward0` 的第二个 `next_functions` 上。],
) <fig-ag-basic>

#insight[
  "动态图"的准确含义不是"每次 forward 重新 trace"，而是 *反向图是 forward 的副产品*：没有单独的建图阶段，`grad_fn` 是在每个 op 返回时就地挂上去的。所以 Python 里的 `if` / `for` / 变长循环天然被支持 —— 走到哪条分支，图就长成什么样。代价是每步都要重建，这也是 `torch.compile` 要用 AOTAutograd 把图提前抓出来的原因。
]

== leaf、`requires_grad`、`.grad` 三者的关系

*leaf tensor（叶子）* 的定义：`grad_fn is None` 的 tensor。两类：用户直接创建的（`torch.randn(...)`、`nn.Parameter`），以及 `requires_grad=False` 的任何 tensor（它压根不参与建图）。

```python
x = torch.randn(3, requires_grad=True)
x.is_leaf          # True
y = x * 2
y.is_leaf          # False —— y 是运算产生的
z = torch.randn(3)
z.is_leaf          # True —— requires_grad=False 也算叶子
```

三者的关系一句话说清：*`requires_grad=True` 决定这个 tensor 要不要参与建图；`is_leaf` 决定 backward 的梯度要不要写进 `.grad`。* 只有同时满足 `is_leaf and requires_grad` 的 tensor，backward 才会往它的 `.grad` 里累加。

为什么这么设计？因为中间结果的梯度在训练里没人要。一个 70B 模型 forward 会产生几万个中间 tensor，如果每个都存 `.grad`，显存直接翻倍。优化器只需要参数（叶子）的梯度。

要看中间结果的梯度（调试梯度爆炸/消失时常做），用 `retain_grad()`：

```python
x = torch.randn(3, requires_grad=True)
h = x * 2
h.retain_grad()             # 必须在 backward 之前调
(h ** 2).sum().backward()
print(h.grad)               # tensor([...]) —— 拿到了
print(x.grad)               # 照常
```

#note[
  `retain_grad()` 只是让引擎在这个节点上额外挂一个 hook 把梯度存下来，不改变图结构，也不影响其他节点的梯度值。它的显存代价 = 这个 tensor 大小。调试完记得去掉。
]

== `requires_grad` 的传播规则

规则只有一条：*一个 op 的输出 `requires_grad = any(输入.requires_grad)`*。没有例外，没有隐式衰减。

```python
a = torch.randn(3, requires_grad=True)
b = torch.randn(3)                       # requires_grad=False
(a + b).requires_grad                    # True
(b + b).requires_grad                    # False
```

这条规则的直接后果：*冻结一部分网络时，只 `requires_grad_(False)` 是不够省显存的*。假设你冻了前 10 层、训后 10 层，前 10 层的参数确实不建图了，但只要输入本身携带梯度（比如你在做 LoRA、prompt tuning），或者第 11 层开始要梯度，那么第 11 层往后的 activation 全都要保存。真正省显存的是"冻结的前缀整段放进 `no_grad()`"：

```python
with torch.no_grad():
    feats = frozen_backbone(x)     # 这一段不建图，activation 用完即释放
out = head(feats)                  # 从这里开始建图
out.sum().backward()
```

#warn[
  `nn.Module.requires_grad_(False)` 只作用于 *参数和 buffer*，不作用于输入。冻 backbone 却发现显存没降，八成是没用 `no_grad()` 把这一段包起来，或者 backbone 后面还有 `requires_grad=True` 的 adapter 让整条链继续建图。
]

== backward 的执行：拓扑序与"累加"

`loss.backward()` 做的事：

+ 从 `loss.grad_fn` 出发，按 `next_functions` 做一次反向拓扑排序。一个节点只有当它 *所有下游（forward 意义上的消费者）* 的梯度都到齐了才能执行 —— 这就是为什么残差连接、多头共享输入这种"一个 tensor 被用了两次"的情况能正确求和。
+ 逐个执行 `grad_fn`。每个 `grad_fn` 的输入是"输出的梯度"，返回值是"输入的梯度"，个数严格对应 forward 的输入个数。
+ 走到 `AccumulateGrad`（叶子）时，把梯度 *累加* 到 `.grad`：`p.grad = p.grad + g`，而不是 `p.grad = g`。
+ 默认释放中间的 buffer（`save_for_backward` 存的东西），图结构本身还在，但再 backward 一次会报错。

第 3 点是全书最高频的面试题来源：

```python
for batch in loader:
    optimizer.zero_grad(set_to_none=True)   # ← 不写这行，梯度会跨 step 累积
    loss = model(batch).mean()
    loss.backward()
    optimizer.step()
```

#interview[
  面试官几乎必问："PyTorch 为什么不自动清梯度？" 标准答案：*因为累加是特性不是 bug*。梯度累积（gradient accumulation）实现大 batch、多个 loss 分别 backward 合并、RNN 里 BPTT 分段回传，全都依赖这个语义。如果引擎自动清零，这些场景就得自己发明一套缓存。把清零交给 optimizer，是把策略留给用户。
]

`zero_grad(set_to_none=True)` 是 torch 2.0+ 的默认值：它把 `.grad` 置成 `None` 而不是填 0。好处有两个：省一次 memset（大模型上不是小数目），以及让"没参与 forward 的参数"的 `.grad` 保持 `None`，optimizer 会跳过它们（而 `0` 会被当成真梯度，带 weight decay 或 momentum 的 optimizer 还是会更新参数）。

梯度累积就是直接用这个语义：连续 backward $N$ 次再 `step` 一次，每次的 loss 除以 $N$（否则等效学习率涨了 $N$ 倍）。完整训练循环写法见第 5 章。

== `backward()` 的参数与 `torch.autograd.grad()`

=== 为什么非标量必须传 `gradient`

`backward()` 计算的不是 Jacobian，而是 *vector-Jacobian product*（VJP）。`loss.backward()` 之所以能省略参数，是因为 `loss` 是标量，上游向量默认取 `torch.ones(())`。张量 `y` 没有天然的"上游向量"，所以必须显式给：

```python
y = model(x)                      # (B, C)，不是标量
y.backward(torch.ones_like(y))    # 等价于 y.sum().backward()
v = torch.randn_like(y)
y.backward(v)                     # 计算 v^T · (dy/dx)
```

不传就是这个报错：`RuntimeError: grad can be implicitly created only for scalar outputs`。修法要么 `.sum()` / `.mean()` 归约成标量，要么想清楚你的上游向量是什么。

=== `torch.autograd.grad` vs `backward`

```python
g, = torch.autograd.grad(loss, [x], create_graph=False)   # 返回 tuple，不写 x.grad
loss.backward()                                            # 写 x.grad，返回 None
```

区别有四点，面试里说全能加分：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [`backward()`], [`torch.autograd.grad()`],
  [返回值], [`None`], [梯度 tuple],
  [是否写 `.grad`], [写（累加到叶子）], [不写，不污染已有梯度],
  [作用范围], [整张图的所有叶子], [只算你指定的 `inputs`，其余分支被剪掉],
  [典型用途], [训练主循环], [二阶导、gradient penalty、meta-learning、影响函数],
)

第三点是性能上的实际差别：`torch.autograd.grad(loss, [x])` 只会执行"从 `loss` 到 `x` 的路径上"的节点，其他分支不跑。做 gradient penalty 时用 `grad` 而不是 `backward` 能省掉一次对全部参数求导。

`allow_unused=True` 用于 `inputs` 里可能有没参与 forward 的 tensor：不加会报 "One of the differentiated Tensors appears to not have been used in the graph"，加了会返回 `None`。torch 2.x 里还有 `materialize_grads=True`，把那些 `None` 换成全零 tensor，写循环时省一堆判断。

== `retain_graph`：什么时候必须开，为什么默认关

backward 走完一个节点，就把它 `save_for_backward` 的中间 buffer 释放掉。所以对同一张图 backward 第二次会报：

```text
RuntimeError: Trying to backward through the graph a second time
(or directly access saved tensors after they have already been freed).
```

需要 `retain_graph=True` 的真实场景只有一类：*同一张 forward 图要被回传多次*。

```python
h = shared_encoder(x)                 # 共享的 forward
loss_a = head_a(h).mean()
loss_b = head_b(h).mean()

loss_a.backward(retain_graph=True)    # 第一次，保留 buffer
loss_b.backward()                     # 第二次，用完释放
```

但上面这个例子其实有更好的写法：`(loss_a + loss_b).backward()` —— 一次 backward，梯度天然求和，不用留 buffer。*绝大多数 `retain_graph=True` 都是误用*，是把"我不知道为什么报错"用一个开关压住了。

#warn[
  滥用 `retain_graph=True` 的典型翻车：在训练循环里对每个 batch 都开，结果 activation 永远不释放，几十个 step 后 OOM。更隐蔽的一种是把上一步的 `loss` 或 hidden state 跨 step 带进下一步（RNN 里常见），整条历史图都被保活。RNN 的正确做法是 truncated BPTT：`hidden = hidden.detach()` 切断历史。

  自检方法：如果你写了 `retain_graph=True`，问自己"我到底 backward 了几次"。答案是 1 次，就该删掉它。
]

== `create_graph=True` 与二阶导

`create_graph=True` 让 *backward 过程本身也建图*，于是梯度 `g` 变成一个有 `grad_fn` 的普通 tensor，可以对它再求导。它隐含 `retain_graph=True`。

#figure(
  align(center, autograd-graph(
    tensors: (
      ("x", "(N,) leaf"),
      ("y", "() scalar"),
      ("g", "(N,) 带 grad_fn"),
      ("gv", "() scalar"),
    ),
    ops: (
      ("y = (x * x).sum()", "SumBackward0"),
      ("g = grad(y, x, create_graph=True)", "MulBackward0 (二阶图)"),
      ("gv = (g * v).sum()", "SumBackward0"),
    ),
  )),
  caption: [`create_graph=True` 时，一阶梯度 `g` 自己也挂上了 `grad_fn`，于是对 `gv` 再 backward 就得到 Hessian-vector product。这是所有二阶方法的公共骨架。],
) <fig-ag-second-order>

*Hessian-vector product（HVP）完整例子。* 直接算 Hessian 是 $O(N^2)$ 存储，实际永远算 HVP：

```python
import torch

def hvp(f, x, v):
    """返回 H @ v，其中 H = d^2 f / dx^2。只需两次反向，不实体化 Hessian。"""
    x = x.detach().requires_grad_(True)
    y = f(x)
    g, = torch.autograd.grad(y, x, create_graph=True)   # 一阶梯度，带图
    hv, = torch.autograd.grad((g * v).sum(), x)         # 对 g·v 再求导
    return hv

x = torch.randn(5)
v = torch.randn(5)
f = lambda t: (t ** 3).sum()          # f'' = 6t，所以 H = diag(6x)
print(torch.allclose(hvp(f, x, v), 6 * x * v))   # True
```

原理：$nabla_x (g^T v) = nabla_x ((nabla f)^T v) = H v$（$v$ 与 $x$ 无关，所以能提出来）。

*Gradient penalty（WGAN-GP 的核心）：*

```python
def gradient_penalty(D, real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = D(x_hat)
    g, = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),   # 非标量输出，必须给上游向量
        create_graph=True,                     # penalty 要参与主 loss 的 backward
        retain_graph=True,
    )
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

loss = d_fake.mean() - d_real.mean() + 10.0 * gradient_penalty(D, real, fake)
loss.backward()          # 这里会走二阶路径
```

#warn[
  `create_graph=True` 的两个坑。第一，*显存*：反向图被完整保留，峰值大致翻倍，而且很容易忘记它隐含 `retain_graph=True`，导致图不释放。第二，*不是所有 op 都二阶可导*：很多 fused kernel 在二阶时会报 "derivative for ... is not implemented"。自己写的 `autograd.Function` 也一样 —— 只有当它的 `backward` 本身完全由可微 op 组成时才支持二阶；标了 `@torch.autograd.function.once_differentiable` 的则明确声明"不支持"。遇到这类报错，通常只能把那一段换成纯 op 组合。
]

== 数学视角：链式法则与 VJP

设 forward 是一串函数复合 $L = f_n compose dots.c compose f_1 (x)$，中间量 $u_k = f_k (u_(k-1))$。链式法则给出

#formula[$ (partial L) / (partial x) = (partial L) / (partial u_n) dot (partial u_n) / (partial u_(n-1)) dot.c (partial u_1) / (partial x) $]

这串 Jacobian 连乘从左往右算，还是从右往左算，结果一样，代价天差地别。

- *反向模式（reverse mode）*：从左往右，每一步都是"行向量 $times$ 矩阵"，即 VJP。始终只操作向量，代价 $approx$ 一次 forward 的常数倍。
- *前向模式（forward mode）*：从右往左，每一步是"矩阵 $times$ 列向量"，即 JVP。一次只能得到"输出对某一个输入方向"的导数。

设输入维度 $n$、输出维度 $m$。要拿到完整 Jacobian，反向模式要跑 $m$ 次，前向模式要跑 $n$ 次。深度学习里 $n approx 10^9$（参数量）、$m = 1$（标量 loss），所以反向模式赢了九个数量级。这就是 autograd 叫 "backpropagation" 的全部理由。

#formula[$ overline(u)_(k-1) = J_k^T overline(u)_k, quad J_k = (partial u_k) / (partial u_(k-1)) $]

关键实现细节：*每个 `grad_fn` 从来不显式构造 $J_k$*，它只实现"给我上游向量 $overline(u)_k$，我返回 $J_k^T overline(u)_k$"。比如 `MmBackward0` 对 $y = x W$ 的 VJP 是 $overline(x) = overline(y) W^T$、$overline(W) = x^T overline(y)$ —— 两次 matmul，从没出现过那个 $(B H) times (B O)$ 的巨大 Jacobian。

#note[
  前向模式在 PyTorch 里也有：`torch.func.jvp` / `torch.autograd.forward_ad`。它在"输入维度小、输出维度大"时才划算，实际用途主要是算 JVP 做数值验证，以及某些 ODE/物理仿真场景。面试里提一句"PyTorch 也有 forward-mode AD，只是深度学习的维度形状不适合它"就够了。
]

== `detach` / `no_grad` / `inference_mode` / `enable_grad`

这四个东西经常被混着用，但语义差别很实在：

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left, left),
  [*机制*], [*是否建图*], [*in-place 限制*], [*version counter*], [*性能*],
  [`detach()`], [新 tensor 不带梯度，与原图断开], [与原 tensor 共享 storage，改它会改到原 tensor], [与原 tensor *共享同一个*，污染仍会被检测到], [零开销，不 copy],
  [`no_grad()`], [块内所有 op 都不建图], [允许，手写参数更新常用], [照常维护], [省掉建图与 activation 保存],
  [`inference_mode()`], [不建图，新建的是 *inference tensor*], [出了块就禁止 in-place], [*不维护*，也不走 view 记账], [比 `no_grad` 更快],
  [`enable_grad()`], [强制建图，可嵌在 `no_grad` 内反转], [无额外限制], [照常], [无],
)

```python
# no_grad 的经典用途：手写参数更新
with torch.no_grad():
    for p in model.parameters():
        p -= lr * p.grad          # 直接改 leaf，不建图

# enable_grad 嵌在 no_grad 里反转（eval 时要做 gradient-based 分析会用）
with torch.no_grad():
    feats = backbone(x)
    with torch.enable_grad():
        feats = feats.detach().requires_grad_(True)
        score = probe(feats).sum()
        g, = torch.autograd.grad(score, feats)
```

#warn[
  `inference_mode()` 里创建的 tensor 带一个特殊标记，*不能被 autograd 保存*。经典翻车：dataloader 或预处理函数被包进 `inference_mode()`，产出的 tensor 喂进训练，报错

  ```text
  RuntimeError: Inference tensors cannot be saved for backward.
  ```

  修法：要么把预处理换成 `torch.no_grad()`，要么在出块前 `.clone()` 成普通 tensor。规则是简单的：*纯推理服务用 `inference_mode()`，训练过程中的任何局部禁梯度用 `no_grad()`*。
]

`detach()` 与 `detach_()` 的差别：前者返回新 tensor（原 tensor 不变），后者原地把当前 tensor 变成叶子（切断它的 `grad_fn`）。RNN 里的 `hidden = hidden.detach()` 用前者。`.data` 是历史遗留的第五种写法，*它绕过 version counter*，所以能悄悄制造错误梯度而不报错 —— 新代码一律不要用 `.data`。

== `torch.autograd.Function`：自定义前反向

什么时候需要自己写：数学上有闭式导数比 autograd 逐 op 展开更快更稳（比如 softmax-cross-entropy 融合）；forward 里调了不可微的外部 kernel（自定义 CUDA、Triton）；想在反向里塞通信（DDP/TP 里的 all-reduce）或者改梯度（梯度反转）。

```python
class ScaledTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, clamp=False):
        y = scale * torch.tanh(x)
        # save_for_backward 只能存 tensor；非 tensor 挂 ctx 属性。
        # 存输出而不是输入：tanh 的导数能用输出反推，省一份显存。
        ctx.save_for_backward(y)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        grad_x = None
        # 只在真的需要时才算，省掉无用计算
        if ctx.needs_input_grad[0]:
            # d/dx [s*tanh(x)] = s*(1 - tanh(x)^2) = s - y^2/s
            grad_x = grad_out * (ctx.scale - y * y / ctx.scale)
        # 返回值个数必须 == forward 的输入个数（不含 ctx）
        # scale 是 float、clamp 是 bool，都不需要梯度 → None
        return grad_x, None, None

y = ScaledTanh.apply(x, 2.0)
```

必须记住的四条规则：

+ `forward` / `backward` 都是 `@staticmethod`，用 `Fn.apply(...)` 调用，*不要*直接 `Fn.forward(...)`（那样不建图）。
+ `backward` 的 *返回值个数必须等于 `forward` 的输入个数*（不含 `ctx`）。不需要梯度的位置返回 `None`。这是最高频的错误："expected N gradients but got M"。
+ 需要在 backward 里用到的 tensor 一律走 `ctx.save_for_backward(...)` + `ctx.saved_tensors`，*不要*直接 `ctx.x = x`。前者会走 version counter 检查，能抓到"存下来的东西被 in-place 改了"；后者绕过检查，静默给你错梯度。
+ `ctx.needs_input_grad` 是一个 bool tuple，长度等于 forward 输入个数。用它跳过不需要的分支。

=== 新版的 `setup_context` 写法

torch 2.0+ 推荐把"存 context"从 `forward` 里拆出来，这样 `forward` 是一个纯函数，能被 `torch.func`（`vmap` / `functorch`）和 `torch.compile` 正确处理：

```python
class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(x, scale):              # 注意：没有 ctx 参数
        return scale * x.sigmoid()

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, scale = inputs
        ctx.save_for_backward(output)   # 存输出比存输入省显存
        ctx.scale = scale

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        # d/dx [s*sigmoid(x)] = y * (1 - y/s)，用输出反推
        return grad_out * y * (1 - y / ctx.scale), None
```

两种写法都还支持。旧写法（`forward(ctx, ...)`）在纯训练场景没问题；写库、或者要跟 `vmap` / `torch.compile` 配合，就用 `setup_context`。

=== `mark_dirty` 与 `mark_non_differentiable`

- `ctx.mark_dirty(x)`：forward 里 *原地修改了输入* 时必须调。它告诉 autograd 去 bump 这个 tensor 的 version counter，否则后续会用到旧值算出错误梯度。
- `ctx.mark_non_differentiable(idx)`：forward 返回了不可微的输出（比如 `argmax` 出来的整数索引、mask）时调，autograd 就不会为它准备梯度、也不会在你没返回对应梯度时报错。

```python
class InplaceRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)               # 声明 x 被原地改了
        x.relu_()
        ctx.save_for_backward(x)
        return x
    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * (y > 0)
```

更多可直接背的模板（梯度反转层、straight-through estimator、fused 实现）在第 24 章。

== 用 `gradcheck` 验证自定义 backward

自己写的 backward 有没有算错，靠肉眼看不出来。`torch.autograd.gradcheck` 用有限差分逼近数值梯度，跟你的解析梯度对比：

```python
from torch.autograd import gradcheck

x = torch.randn(4, 5, dtype=torch.double, requires_grad=True)   # ← 必须 double
assert gradcheck(lambda t: ScaledTanh.apply(t, 2.0), (x,), eps=1e-6, atol=1e-4)

# 二阶（只有当你的 backward 本身由可微 op 组成时才会通过）
from torch.autograd import gradgradcheck
assert gradgradcheck(lambda t: ScaledTanh.apply(t, 2.0), (x,))
```

#warn[
  *`gradcheck` 必须用 `float64`。* 这是面试高频追问。原因：有限差分 $(f(x+epsilon) - f(x-epsilon)) / (2 epsilon)$ 的误差里有一项正比于 $"机器精度" / epsilon$。float32 的机器精度是 $approx 10^(-7)$，取 $epsilon = 10^(-6)$ 时这一项就有 $10^(-1)$ 量级，噪声比信号还大，必然 fail。float64 的机器精度 $approx 10^(-16)$，才有足够余量。

  其他常见 fail 原因：函数在测试点不可导（ReLU 在 0 附近、`abs`、`clamp` 边界 —— 换个输入点或加个偏移）；op 有随机性（dropout，要么固定 seed 要么设 `nondet_tol`）；输入没设 `requires_grad=True`。
]

== in-place 操作与 version counter

每个 tensor 的 storage 上带一个 *version counter*。任何 in-place op（后缀 `_` 的方法、`+=`、切片赋值 `x[0] = ...`）都会把它 +1。`save_for_backward` 存 tensor 时会顺手记下当时的版本号，backward 取出来时比对 —— 不一致就报错。这是 PyTorch 能"动态图 + in-place 还不算错梯度"的机制保障。

两个报错长得像，成因完全不同：

#warn[
  *报错 1：`a leaf Variable that requires grad is being used in an in-place operation.`*

  最小复现：

  ```python
  w = torch.randn(3, requires_grad=True)
  w.add_(1)          # RuntimeError
  ```

  成因：对 *叶子* 做 in-place，会让这个叶子的历史无法表达（它既是叶子又有了 `grad_fn`，矛盾）。真实场景是手写参数更新时忘了包 `no_grad`，或者做参数初始化/裁剪时直接改了 `nn.Parameter`。

  修法：包进 `no_grad()`，或者对 `.detach()` 出来的视图操作。
  ```python
  with torch.no_grad():
      w.add_(1)          # OK
  # 或
  w.detach().add_(1)     # OK，共享 storage
  ```
]

#warn[
  *报错 2：`one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3]], which is output 0 of AddBackward0, is at version 1; expected version 0 instead.`*

  最小复现：

  ```python
  x = torch.randn(3, requires_grad=True)
  y = x * 2
  z = y.pow(2)       # PowBackward0 存了 y（版本 0）
  y.add_(1)          # y 的版本变成 1
  z.sum().backward() # RuntimeError
  ```

  成因：某个 `grad_fn` 保存了 `y` 用于反向，`y` 却在 backward 之前被改了。注意 *不是所有 in-place 都会触发* —— 比如 `y.exp()` 的反向只需要输出，不需要 `y`，所以改 `y` 没事。这也是为什么这个 bug 常常"换个激活函数就好了"，很迷惑。

  修法：把那次 in-place 换成 out-of-place（`y = y + 1`），或在 in-place 之前 `y = y.clone()`。定位手段：`torch.autograd.set_detect_anomaly(True)` 会打印出"是哪个 forward op 产生了这个坏梯度"的栈，代价是慢好几倍，只在 debug 时开。

  典型来源：`nn.ReLU(inplace=True)` 接在需要保存输出的 op 后面、residual 里写了 `out += identity`、以及在 forward 里手动改 activation 做归一化。`out += identity` 换成 `out = out + identity` 就好。
]

#insight[
  version counter 只在 *autograd 记账* 的路径上生效。`x.data` 和 `inference_mode()` 都绕过它 —— 前者是历史遗留 API，后者是设计上就不做记账（换性能）。所以 "用了 `.data` 之后梯度悄悄变错" 是最难查的一类 bug：没有任何报错。新代码只用 `detach()`。
]

== "梯度是 `None`" 排查清单

这是调试时最常遇到的现象，按这个顺序查基本一次命中：

#table(
  columns: (0.9fr, 1fr, 1.4fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*原因*], [*判定*], [*修法*],
  [不是叶子], [`t.is_leaf == False`], [是中间量就用 `retain_grad()`；本该是参数就检查是不是被重新赋值了],
  [`requires_grad=False`], [`t.requires_grad == False`], [`t.requires_grad_(True)`；参数检查有没有被 `requires_grad_(False)` 冻住],
  [在 `no_grad` / `inference_mode` 里建的], [`out.grad_fn is None`], [把建图的那段挪出上下文],
  [被 `detach()` / `.data` 断开], [沿 `grad_fn` 往回走断在哪], [删掉那次 `detach`],
  [参数没参与 forward], [该参数的 `grad` 恒为 `None`], [检查分支逻辑；DDP 场景要 `find_unused_parameters=True`（见第 18 章）],
  [optimizer 没管这个参数], [参数不在 `param_groups` 里], [构造 optimizer 前先把模块搬到设备、加完所有子模块],
  [被 `set_to_none=True` 清了], [`.grad is None` 且在 `step` 之后], [正常行为，不是 bug],
)

```python
# 单个 tensor：把状态一次打全
print(t.is_leaf, t.requires_grad, t.grad_fn)

# 模块级批量检查：这些参数这一步根本没被更新
for n, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        print("no grad:", n)
```

== activation checkpointing 的 autograd 视角

`torch.utils.checkpoint.checkpoint(fn, *args)` 做的事，用 autograd 的语言说：*在 forward 时不保存 `fn` 内部的中间 activation，只保存输入；backward 走到这里时，先用保存的输入重跑一遍 `fn` 的 forward（这次建图），再正常反向。* 用一次额外的 forward 换掉整段的 activation 显存。

```python
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def forward(self, x):
        return self.mlp(self.attn(self.ln(x)) + x)

# 每个 block 只保存输入，内部 activation 全部丢弃
for blk in self.blocks:
    x = checkpoint(blk, x, use_reentrant=False)
```

`use_reentrant=False` 是新推荐实现（torch 2.1+ 起不显式传会告警，未来会变成默认）。它与旧的 reentrant 版本的差别：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [`use_reentrant=True`（旧）], [`use_reentrant=False`（新）],
  [实现方式], [在 backward 里再调一次 `autograd.Function`，嵌套进入引擎], [基于 saved-tensor hooks，不嵌套],
  [输入没有 `requires_grad`], [直接静默不算梯度], [正常工作],
  [`torch.autograd.grad()`], [不支持（只支持 `backward()`）], [支持],
  [backward hook], [触发时机不对，容易漏], [正常],
  [非 tensor 输出 / 变长输出], [不支持], [支持],
  [early stop], [无，整段都要重算], [有（`early_stop=True`，梯度算够就停）],
)

*RNG 一致性。* dropout 在 forward 和 recompute 时如果抽到不同的 mask，梯度就是错的（而且不会报错，只表现为收敛变差）。`checkpoint` 默认 `preserve_rng_state=True`：forward 时把 CPU 和当前 CUDA device 的 RNG state 存下来，recompute 之前恢复回去，跑完再切回来。代价是每个 checkpoint 段有一次 RNG state 的存取（很便宜）。

#warn[
  几个 checkpoint 的实际坑：

  - *`preserve_rng_state=True` 只保存"当前 device"的 RNG state*。如果 `fn` 内部切了 device 或用了多个 device，重算的随机数不保证一致。
  - 被 checkpoint 的函数应当是 *确定性* 的。除了 RNG，还包括：不要在 `fn` 里读写会被外部改动的全局状态，不要依赖上一次调用的副作用。`determinism_check="default"` 会检查重算的输出 shape/dtype 是否一致，但不检查数值。
  - 与 `torch.compile` 组合时行为在演进（编译器可能自己决定重算哪些 op，见第 13 章的 min-cut partitioner），版本敏感，升级 torch 后要重新 profile。
]

#insight[
  面试问"checkpointing 省了多少显存、花了多少时间"，标准答法是给公式不给假数字：$L$ 层均匀分段成 $sqrt(L)$ 段时，activation 显存从 $O(L)$ 降到 $O(sqrt(L))$，额外计算约等于 *多一次 forward*，即总计算量从 $2 F$（forward + backward $approx 2 times$）变成 $3 F$ 量级，因此吞吐损失通常在 20%--30%。具体数字必须自己 profile，跟模型形状、是否 memory-bound 强相关。
]

== hook：在反向路径上插手

=== `Tensor.register_hook`

签名 `t.register_hook(fn)`，`fn(grad) -> Tensor or None`。返回非 `None` 就 *替换* 这个梯度，返回 `None` 就只是旁观。触发时机：这个 tensor 的梯度算好、往下游传之前。

```python
# 用途 1：监控（不改梯度）
handle = h.register_hook(lambda g: print("grad norm:", g.norm().item()))
...
handle.remove()          # 记得摘，否则一直挂着

# 用途 2：梯度反转层（domain-adversarial training 的核心，一行搞定）
lambda_ = 0.5
feat.register_hook(lambda g: -lambda_ * g)

# 用途 3：per-tensor 梯度裁剪（比全局 clip_grad_norm_ 更早介入）
for p in model.parameters():
    p.register_hook(lambda g: g.clamp(-1.0, 1.0))
```

#note[
  梯度反转层也可以写成 `autograd.Function`（forward 恒等、backward 取负），两种写法等价。hook 版本更短，`Function` 版本能作为一个可复用的 module 插进 `nn.Sequential`，也更容易被 `torch.compile` 处理。第 24 章给 `Function` 版本。
]

hook 的一个重要工程用途：*DDP 的梯度同步就是挂在参数上的 hook*。参数 `p` 的梯度一算出来，hook 立刻触发，把它标记进 bucket；bucket 满就发一个异步 all-reduce。这就是 DDP 能让通信与 backward 重叠的机制（见第 18 章）。

=== `Module.register_full_backward_hook`

签名 `fn(module, grad_input, grad_output)`，在这个 module 的 *输入梯度* 全部算好后触发。返回一个新的 `grad_input` tuple 就能替换。

```python
def watch(mod, grad_in, grad_out):
    gi = grad_in[0]
    if gi is not None and not torch.isfinite(gi).all():
        print(f"NaN/Inf in grad_input of {mod.__class__.__name__}")

for m in model.modules():
    m.register_full_backward_hook(watch)      # 定位 NaN 从哪一层开始
```

`register_full_backward_hook` 与已废弃的 `register_backward_hook` 的区别：后者在 module 有多个输入/输出或内部有多个 op 时触发时机不正确（它实际挂在最后一个 op 上），新代码一律用 `full` 版本。

#warn[
  hook 有两个共同的坑。第一，*不摘会泄漏*：`register_hook` 返回的 handle 如果不 `remove()`，闭包捕获的对象（常常是整个 module 或一个大 tensor）就一直活着。第二，*hook 里做 `.item()` / `print` 会强制同步 GPU*，在训练主循环里挂满监控 hook 能把吞吐打掉一半。监控类 hook 建议按步数采样，比如每 100 步才真正算一次 norm。
]

== 面试考点

#interview[
  *Q1*：为什么每个 step 都要 `optimizer.zero_grad()`？

  A：因为 backward 是把梯度 *累加* 到 `.grad`，不是覆盖。累加是有意设计的：梯度累积做大 batch、多个 loss 分别回传、RNN 分段 BPTT 都依赖它。所以清零的时机交给用户。推荐 `set_to_none=True`，省一次 memset，而且能让没参与 forward 的参数保持 `None` 从而被 optimizer 跳过。
]

#interview[
  *Q2*：什么是 leaf tensor？为什么只有它有 `.grad`？

  A：`grad_fn is None` 的 tensor 是叶子，通常是用户创建的 tensor 和 `nn.Parameter`。backward 只把梯度写进"叶子且 `requires_grad`"的 `.grad`，因为中间 activation 的梯度训练里没人要，全存下来显存会翻倍。要看中间梯度调 `retain_grad()`。
]

#interview[
  *Q3*：非标量 tensor 调 `backward()` 为什么报错？

  A：autograd 算的是 VJP 不是 Jacobian，必须有一个上游向量。标量 loss 的上游向量默认取 1，所以能省略；非标量没有天然默认值，得自己传 `y.backward(v)`，语义是 $v^T (partial y) / (partial x)$。传 `ones_like(y)` 就等价于 `y.sum().backward()`。
]

#interview[
  *Q4*：`retain_graph=True` 什么时候要开？为什么默认关？

  A：只有"同一张 forward 图要 backward 多次"时才要开，比如共享 encoder 后两个 head 分别回传。默认关是因为 backward 走完会释放 `save_for_backward` 的中间 buffer，这是省显存的大头。滥用会导致 activation 不释放、几十步后 OOM。多数情况的正确解法是 `(loss_a + loss_b).backward()` 一次搞定。
]

#interview[
  *Q5*：`torch.autograd.grad()` 和 `backward()` 有什么区别？

  A：四点。`grad()` 返回梯度 tuple 且 *不写* `.grad`（不污染训练主循环累积的梯度）；只计算到你指定的 `inputs`，其他分支被剪掉所以更快；天然支持 `create_graph` 做高阶导；`backward()` 则是走遍整张图、写所有叶子的 `.grad`，是训练主循环的写法。
]

#interview[
  *Q6*：怎么算 Hessian-vector product？为什么不直接算 Hessian？

  A：两次反向。先 `g = grad(loss, x, create_graph=True)` 拿到带图的一阶梯度，再 `grad((g * v).sum(), x)` 得到 $H v$。因为 $nabla_x (g^T v) = H v$。不直接算 Hessian 是因为它是 $N times N$，$N$ 是参数量，存不下也算不动；而 HVP 的代价只是常数倍的 forward。
]

#interview[
  *Q7*：`detach()`、`no_grad()`、`inference_mode()` 分别什么时候用？

  A：`detach()` 断开单个 tensor 的历史，共享 storage、零开销，RNN 截断 BPTT 和"用中间量当常数"时用。`no_grad()` 让一整块代码不建图，训练中做验证、手写参数更新时用。`inference_mode()` 比 `no_grad()` 更激进，连 version counter 和 view 记账都不做，因此更快，但产出的 tensor 不能进 autograd —— 纯推理服务用它，训练过程里不要用。
]

#interview[
  *Q8*：自己写 `autograd.Function` 有哪些必须遵守的规则？

  A：`forward`/`backward` 都是 `@staticmethod`，用 `.apply()` 调；`backward` 返回值个数必须等于 `forward` 输入个数，不需要梯度的位置返回 `None`；要在反向用到的 tensor 用 `ctx.save_for_backward` 存（走 version counter 检查），非 tensor 挂 `ctx` 属性；用 `ctx.needs_input_grad` 跳过不需要的分支；forward 里改了输入要 `ctx.mark_dirty()`。写完用 `gradcheck` 在 float64 上验一遍。
]

#interview[
  *Q9*：`gradcheck` 为什么必须用 double？

  A：它用有限差分做数值梯度，误差里有一项是"机器精度 / eps"。float32 机器精度约 $10^(-7)$，配上 $10^(-6)$ 的 eps，噪声就到 $10^(-1)$ 量级，直接淹没信号。float64 机器精度约 $10^(-16)$，才有足够余量。此外还要保证测试点上函数可导（避开 ReLU 的 0 点）、op 是确定性的。
]

#interview[
  *Q10*：报错 "one of the variables needed for gradient computation has been modified by an inplace operation" 怎么查？

  A：说明某个 `grad_fn` 保存的 tensor 在 backward 前被 in-place 改了，autograd 靠 version counter 检测到。常见来源是 `nn.ReLU(inplace=True)`、residual 里的 `out += identity`、手动改 activation。修法是换成 out-of-place 或先 `clone()`。定位用 `torch.autograd.set_detect_anomaly(True)`，它会打印产生这个坏梯度的 forward 栈，代价是慢好几倍，只在 debug 时开。注意不是所有 in-place 都会触发 —— 反向只需要输出的 op（如 `exp`、`sigmoid`）改输入没事，所以这类 bug 常常表现得很随机。
]
