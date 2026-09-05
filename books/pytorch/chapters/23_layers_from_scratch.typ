#import "../template.typ": *

= 手写基础层：从零实现 torch.nn

前面四部分讲的是原理，这一部分开始是白板题。面试官最爱的开场是"来，手写一个 LayerNorm"——不给 IDE、不给文档、二十行之内写完，然后追问三个细节。这一章覆盖九个最高频的层。每个层都控制在白板能写完的规模，重点不在代码有多长，而在*那几个只有真写过才知道的细节*：`eps` 放在 sqrt 里还是外、方差除 N 还是 N-1、padding 补 0 还是补 `-inf`。

本章所有代码的可运行版本 + pytest 对齐测试在 `python/pytorch/interview/test_layers.py`（31 个测试，`pytest python/pytorch/interview/test_layers.py -q` 直接跑）。章节里的实现去掉了 jaxtyping 标注，逻辑与该文件一致。

*先说验收标准.* 面试里写完代码，下一句一定要是"我会这样验证它"。手写层的验证标准只有一个：*把官方层的权重拷进你的实现，同一个输入下 `torch.testing.assert_close` 通过*——shape 对了不算对，数值一致才算对。

```python
def copy_params(dst: nn.Module, src: nn.Module) -> None:
    """把 src 的同名 parameter / buffer 拷进 dst，然后就可以逐元素比了。"""
    with torch.no_grad():
        dst.load_state_dict(
            {k: v.clone() for k, v in src.state_dict().items()}, strict=False
        )
```

`strict=False` 容忍两边 key 不完全一样（比如你没实现 `num_batches_tracked`）。注意*权重不能用默认初始化后直接比*：两个模块各自 `reset_parameters` 一次，随机数不同，输出当然不同。带 buffer 的层（BatchNorm）除了输出，`running_mean` / `running_var` 也要逐步对齐——很多错误只在第二次 forward 才暴露。下面每个层结尾都给一句针对性的自证方法。

#insight[
  `assert_close` 默认的 `rtol/atol` 是按 dtype 定的（fp32 是 `rtol=1.3e-6, atol=1e-5`）。如果你的实现和官方只是*运算顺序*不同（比如先乘 weight 再除 std vs 反过来），fp32 下会差到 1e-6 量级，默认容差刚好能过。差到 1e-3 就不是浮点误差，是公式错了。
]

== Linear

最简单的一题，但下面两个追问能筛掉一半人。

```python
import math
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # a=sqrt(5) 是 nn.Linear 的历史默认值
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[1])
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # matmul 自动把前面所有维度当 batch，不需要手动 reshape 成 2D
        out = x @ self.weight.t()
        return out if self.bias is None else out + self.bias
```

*追问一：weight 为什么是 `(out, in)` 而不是 `(in, out)`？* 前向写成 $y = x W^T + b$，`W.t()` 看起来是"白白多一次转置"。四个理由：

+ *转置是免费的*。底层落到 `addmm(bias, x, W.t())`，cuBLAS 的 GEMM 本来就带 `transa` / `transb` 标志位，`W.t()` 只是改 stride，不产生任何拷贝。
+ *行主序下 cache 更友好*。`(out, in)` 让"同一个输出神经元的所有输入权重"在内存里连续。反向算 `grad_input = grad_out @ W` 时按行扫 `W`，一次 cache line 拿到的都是有用数据。
+ *量化 / 融合天然对齐*。per-output-channel 的 scale 正好按行切，`weight[i]` 就是第 $i$ 个通道的全部权重。
+ *现实原因*：所有预训练权重都是这个布局，改了就没法 `load_state_dict`。

*追问二：默认初始化到底是什么分布？* `nn.Linear` 用的是 `kaiming_uniform_(weight, a=math.sqrt(5))`。kaiming uniform 的边界代入 $a = sqrt(5)$ 后，$1 + a^2 = 6$ 与分子的 6 正好约掉：

#formula[$ "bound" = sqrt(6 / ((1 + a^2) dot "fan_in")) = sqrt(1 / "fan_in") = 1 / sqrt("fan_in") $]

也就是 $U(-1\/sqrt("fan_in"), +1\/sqrt("fan_in"))$——和 bias 的初始化区间完全一样。`a` 本意是 leaky-ReLU 的负斜率，但这里根本没有 leaky ReLU，`sqrt(5)` 纯粹是从 Torch7 继承的历史常数，作用只是把 kaiming 的公式凑回老的 $1\/sqrt("fan_in")$ 均匀分布。"为什么是 sqrt(5)"是高频追问，答案就是"历史遗留，等价于 $U(plus.minus 1\/sqrt("fan_in"))$"。

#warn[
  白板上直接写 `self.weight = nn.Parameter(torch.randn(out, in))` 是最常见的失分点：`randn` 的方差是 1，经过 $D$ 维内积后输出方差放大 $D$ 倍，堆几层就溢出。至少要写 `torch.randn(...) / math.sqrt(in_features)`，并说明这是在保持前向方差。
]

*自证*：`copy_params` 后与 `nn.Linear` 对齐；再断言 `weight.shape == (out, in)` 和 `weight.abs().max() <= 1 / sqrt(fan_in)`。

== Softmax

这题考的是数值稳定性，不是公式。

```python
class MySoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_max = x.amax(dim=self.dim, keepdim=True)
        e = torch.exp(x - x_max)
        return e / e.sum(dim=self.dim, keepdim=True)
```

*为什么减 max 后结果不变.* 对任意常数 $c$：

#formula[$ "softmax"(x + c)_i = e^(x_i + c) / (sum_k e^(x_k + c)) = (e^c dot e^(x_i)) / (e^c dot sum_k e^(x_k)) = "softmax"(x)_i $]

分子分母同乘 $e^c$，约掉。所以可以自由挑 $c$，取 $c = -max_k x_k$ 是最优选择：

这样所有指数的输入都 $<= 0$，`exp` 结果落在 $(0, 1\]$，*永远不会上溢*；同时分母里至少有一项是 $e^0 = 1$，*永远不会下溢成 0*，也就不会出现 `0/0 = NaN`。

*溢出的具体数字.* `exp` 的溢出门槛就是 $ln("dtype 的最大值")$。fp32 的 max 是 `3.40e38`，阈值 *88.72*（`torch.exp(tensor(89.))` 就是 `inf`）；fp16 的 max 只有 `65504`，阈值 *11.09*；bf16 的指数位和 fp32 一样宽，阈值同样 88.72，只是尾数少。

fp16 只有 11.09 的余量，这在实际训练里*太容易撞上*：没加 $1\/sqrt(d_k)$ 缩放的 attention logits、loss 爆炸的中间态、长序列上没归一化的分数，随便一个都能超。这就是混合精度里 softmax 一律在 fp32 里算的原因。

#warn[
  朴素写法 `torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)` 在 `x = [1e4, 1e4+1, 0]` 上会算出 `inf / inf = NaN`；稳定版返回正确结果。另一个隐蔽错误是 `keepdim=False`，广播维度错位后 shape 还可能是对的（比如方阵输入），数值全错。
]

最后一个高频细节：分类任务里不要写 `log(softmax(x))`，要用 `log_softmax`。后者把 log 和 exp 解析地约掉了，直接算 $x - max - log sum e^(x - max)$，少一次精度损失，也不会在概率接近 0 时 `log(0) = -inf`。

*自证*：与 `torch.softmax` 对齐；额外断言 `y.sum(-1) == 1` 和大输入下 `torch.isfinite(y).all()`。

== LayerNorm

```python
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)  # 除以 N
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x_hat = x_hat * self.weight + self.bias
        return x_hat
```

三个必答点。*第一，eps 在 sqrt 里面.*

#formula[$ y = (x - mu) / sqrt(sigma^2 + epsilon) dot w + b quad #text[（对）] quad quad y = (x - mu) / (sqrt(sigma^2) + epsilon) dot w + b quad #text[（错）] $]

区别在 $sigma^2 -> 0$ 的时候。放外面时 $sqrt(sigma^2)$ 的导数 $1\/(2 sqrt(sigma^2))$ 会爆炸；放里面则被 $epsilon$ 兜住，整条式子在常数输入上仍然有限。这是最常被抓的一个细节，而且*可以一行代码验*：喂一个常数张量（`var = 0`），eps 在里面时输出恒为 0，在外面时是 `NaN` 或 `inf`。

*必答点二：方差用有偏估计.* `x.var(dim, keepdim=True)` 默认是 `unbiased=True`（除以 $N-1$），直接用就和官方对不上。LayerNorm 要的是 `unbiased=False`（除以 $N$）：它不是在估计总体方差，只是在把这一条样本缩放到单位模长，除以 $N$ 才让 `x_hat` 的（有偏）方差正好是 1。归一化维度小的时候差异很明显——`head_dim=4` 时 $N\/(N-1) = 1.33$，肉眼可见。

*必答点三：归一化的维度是最后若干维.* `normalized_shape=(3, 8)` 表示沿*最后两维*一起做归一化，它必须与输入的尾部 shape 完全匹配。因为是逐样本的，LN 与 batch size 无关，也就*没有 running stats*，train / eval 行为完全一致——这是它和 BatchNorm 最本质的差别，也是它在变长序列和小 batch 上通吃的原因。

#warn[
  混合精度下的坑：官方 LN 会把统计量提升到 fp32 再算。bf16 只有 8 位尾数（$epsilon approx 0.0078$），$D = 4096$ 上直接 `x.mean()` 的累计误差足以让归一化结果偏掉。生产实现一律 `x.float()` 算完 `rstd` 再 cast 回去，RMSNorm 同理（而且它还要先平方，动态范围翻倍，更容易溢出）。
]

*自证*：与 `nn.LayerNorm` 对齐（记得给 `weight` / `bias` 灌随机值，否则 $w=1, b=0$ 掩盖了 affine 的 bug）；再喂常数张量验 eps 位置。

== RMSNorm

去掉均值中心化和 bias 的 LayerNorm，Llama / T5 / Gemma 都用它。

```python
class MyRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        ms = x.pow(2).mean(dim=dims, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight
```

#formula[$ "RMSNorm"(x) = x / sqrt(1/D sum_i x_i^2 + epsilon) ⊙ w $]

*为什么 LLM 都换成它.* RMSNorm 论文的核心论点：LayerNorm 的收益主要来自 *re-scaling 不变性*，而不是 re-centering 不变性。既然减均值那一步贡献不大，就砍掉它，换来两个实打实的好处：

*少一趟 reduction*——LayerNorm 要先求 mean 再求 var（或者用一趟 Welford），RMSNorm 只需要一个平方和，而归一化在 LLM 里是 memory-bound 的 kernel，省一趟 reduce 就是省一趟带宽。以及*少一组参数*——没有 bias，Llama 系列全是 weight-only，实测效果与 LayerNorm 相当。

*考点：`nn.RMSNorm` 的 eps 默认不是 1e-5.* `nn.RMSNorm`（torch 2.4+）的 `eps` 默认值是 *`None`*，不是 `1e-5`。`None` 时它取 `torch.finfo(x.dtype).eps`——fp32 下约 `1.19e-7`，比 `1e-5` 小两个数量级。

```python
ref = nn.RMSNorm(8)
ref.eps            # None
# 实际用的是 torch.finfo(torch.float32).eps == 1.1920928955078125e-07
```

对齐测试时不显式传 `eps`，你的 `1e-5` 版本和官方会差 1e-5 量级——刚好大过 `assert_close` 的默认 `atol=1e-5`，测试红了，你以为公式写错了，其实只是 eps 不同。*两边都显式写 `eps=1e-5`* 就好了。

#warn[
  `mean(x^2)` 不等于 `var(x)`，两者只在 `mean(x) = 0` 时才相等（$"var" = EE[x^2] - (EE[x])^2$）。写成 `x.var(...)` 是把 RMSNorm 又写回了半个 LayerNorm。这个错在随机初始化的激活上几乎看不出来（均值本来就接近 0），一旦上游有 bias 漂移就原形毕露。
]

*自证*：与 `nn.RMSNorm(8, eps=1e-5)` 对齐；再用手算参考值 `x / sqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 交叉验证，同时断言它*不等于* `var` 版本。

== BatchNorm1d

面试考点密度最高的一个层。

```python
class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        # buffer 而非 parameter：进 state_dict、跟着 .to(device)，但不参与梯度
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x):
        dims = (0,) if x.dim() == 2 else (0, 2)   # 除通道维 C 外全 reduce
        if self.training:
            mean = x.mean(dim=dims)
            var_biased = x.var(dim=dims, unbiased=False)      # 用于归一化
            with torch.no_grad():
                var_unbiased = x.var(dim=dims, unbiased=True)  # 用于 running
                self.running_mean.mul_(1 - self.momentum).add_(
                    self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var_unbiased)
                self.num_batches_tracked += 1
        else:
            mean, var_biased = self.running_mean, self.running_var

        shape = (1, -1) if x.dim() == 2 else (1, -1, 1)
        x_hat = (x - mean.view(shape)) * torch.rsqrt(var_biased.view(shape) + self.eps)
        if self.weight is not None:
            x_hat = x_hat * self.weight.view(shape) + self.bias.view(shape)
        return x_hat
```

*考点 a：归一化用有偏方差，`running_var` 存无偏方差.* 这不是笔误，是官方行为，也是最容易被追问到底的一点：

#formula[$ hat(x) = (x - mu) / sqrt(sigma^2_"biased" + epsilon), quad quad "running_var" <- (1 - m) dot "running_var" + m dot sigma^2_"unbiased" $]

两个分母不一样：归一化除 $N$，running 除 $N-1$。理由是两者的目的不同——`running_var` 是在*估计总体方差*，无偏才对；而归一化只是对当前 batch 做缩放，除 $N$ 才让 $hat(x)$ 的样本方差正好是 1。

后果是一个真实存在的坑：*同一个 batch 在 train 和 eval 下的输出有系统性差异*，比例大约是 $sqrt(N\/(N-1))$。$N = 32$ 时是 1.6%，$N = 4$ 时就是 15%。很多人报告"切到 eval 后指标掉了"，除了 running stats 没跑够，这个偏差也贡献了一部分。

```python
mine = MyBatchNorm1d(3, momentum=1.0)   # momentum=1 => running 直接等于本 batch
mine.train(); mine(x)                    # x: (N, 3)
mine.running_var / x.var(0, unbiased=False)   # 恒等于 N/(N-1)
```

*考点 b：PyTorch 的 momentum 和 Keras 的方向相反.* PyTorch：`running = (1 - momentum) * running + momentum * batch`。所以 `momentum=0.1` 的含义是*新观测占 10% 权重*。

TensorFlow / Keras：`moving = momentum * moving + (1 - momentum) * batch`，默认 `momentum=0.99` 的含义是*旧值占 99%*。

两者的关系是 $m_"torch" = 1 - m_"keras"$。迁移代码时把 `0.99` 原样抄进 PyTorch，就是"新观测占 99%"——running stats 变成了几乎只记得最后一个 batch，eval 时抖得一塌糊涂。反过来把 `0.1` 抄进 Keras，则是几乎完全不更新。

#warn[
  PyTorch 还允许 `momentum=None`，此时改用*累积平均*（`1 / num_batches_tracked` 作为权重），这就是 `num_batches_tracked` 这个 buffer 存在的唯一理由。手写版如果不实现它，`load_state_dict(strict=True)` 会因为缺 key 而失败。
]

*考点 c：小 batch 与分布式.* BN 的统计量是在 batch 维上算的，所以它的质量直接取决于 batch size：

- *`batch_size = 1` 时 train 模式下方差为 0*，`x_hat` 全是 0，输出恒等于 `bias`。检测目标检测这类每卡 batch 只有 2 的任务，BN 基本失效，所以后来都换成了 GroupNorm / LayerNorm。
- *DDP 下每张卡各算各的 BN*。`nn.BatchNorm2d` 的统计量只覆盖本卡的 local batch，8 卡每卡 4 张图，等效 BN batch 是 4 而不是 32。梯度会被 AllReduce，但*统计量不会*。
- 需要跨卡统计就用 `nn.SyncBatchNorm`（`SyncBatchNorm.convert_sync_batchnorm(model)` 一键替换）。代价是每个 BN 层每次 forward 多一次 AllGather（同步 `sum` 和 `sum_sq`），网络里 BN 层多的话开销可观。

#warn[
  忘了 `model.eval()` 是线上事故的经典原因：推理时 BN 仍在用当前 batch 的统计量，输出随 batch 组成变化，同一张图放在不同 batch 里结果不同。另一个方向的错误是 eval 时还在更新 `running_mean` / `running_var`——手写版一定要把更新逻辑放在 `if self.training` 里面。
]

*自证*：train 模式下连续跑 3 个 batch，每步都比对输出 + `running_mean` + `running_var`（只比第一步抓不到 momentum 方向写反）；再切 eval 比一次，并断言 eval 下 `running_mean` 不变。

== Dropout

```python
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x                       # eval 是恒等，直接返回原张量
        if self.p == 1.0:
            return torch.zeros_like(x)     # 否则 1/(1-p) 除零
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)
```

*为什么放大放在训练时（inverted dropout）.* 设 $m tilde "Bernoulli"(1-p)$。

- *朴素 dropout*：训练输出 $m ⊙ x$，期望是 $(1-p) x$。为了让推理和训练的分布对上，推理时必须乘 $(1-p)$。
- *inverted dropout*：训练输出 $m ⊙ x \/ (1-p)$，期望正好是 $x$。于是*推理路径就是纯恒等*。

#formula[$ EE[m ⊙ x / (1 - p)] = ((1-p) dot x) / (1-p) = x $]

把系数挪到训练侧有三个好处：*推理零开销、零分支*（部署 / 导出 ONNX / 图优化时 dropout 直接消失，不留任何算子，而推理是要跑几十亿次的那一侧）；*`p` 可以随时改*（dropout schedule、per-layer 不同的 `p`）而不影响已保存的推理逻辑；以及只有一套权重语义，不会出现"训练好的模型忘了乘 $(1-p)$"这种事故——而这在朴素方案下是必然会发生的。

#warn[
  两个高频错法。其一：eval 时还在 drop（忘了检查 `self.training`），表现是推理结果每次都不同。其二：用 `torch.rand(x.shape) > p` 得到 mask 后忘了除 `(1-p)`，推理期望值偏小 $(1-p)$ 倍，模型输出整体缩水，loss 看起来还在降但指标很差。
]

mask 必须*每次前向重新采样*，且同一个 mask 要同时用在 forward 和 backward（autograd 保存 mask 自动保证）。手写 CUDA kernel 时要么显式存 mask，要么在 backward 里用同一个 RNG offset 重放——这就是 Philox RNG 要记录 `(seed, offset)` 的原因，见第 10 章。

*自证*：`eval()` 下断言 `mine(x) is x`；`train()` 下用大张量统计置零比例 $approx p$、`y.mean() ≈ x.mean()`、非零元素恰好等于 $1\/(1-p)$；再连调两次断言两次输出不同。

== Embedding

```python
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)          # 官方默认 N(0, 1)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, idx):
        weight = self.weight
        if self.padding_idx is not None:
            # 前向数值不变，但这一行不在计算图里，梯度自然是 0
            weight = weight.clone()
            weight[self.padding_idx] = self.weight[self.padding_idx].detach()
        return weight[idx]
```

*本质是一次 gather，不是矩阵乘.* "embedding 等价于 one-hot 乘权重矩阵"在数学上对，实现上绝对不能这么做。one-hot 是 $(N, V)$ 的稠密矩阵，$V$ 通常 5 万到 15 万，而 gather 是 $O(N D)$、one-hot matmul 是 $O(N V D)$——$V = 128000, D = 4096$ 时差了五个数量级。反向也一样：gather 的反向是 *scatter-add*（同一个 token 在 batch 里出现多次，梯度累加到同一行），天然稀疏，可以配 `sparse=True` + `SparseAdam`。

*`padding_idx` 不只是初始化为 0.* 很多人以为 `padding_idx` 就是"把这一行初始化成全 0"。不够——*还必须屏蔽它的梯度*。理由：

- 如果 padding embedding 参与了 attention 或 pooling，梯度照样会流回这一行，把它从 0 推走；
- 就算梯度是 0，*带 weight decay 或 momentum 的优化器仍然会更新它*。AdamW 的 decoupled weight decay 是 `p -= lr * wd * p`，`p = 0` 时确实不动；但 momentum 缓冲里一旦攒了历史梯度，后续 step 会持续把它推离 0。

上面的实现用 `clone()` + `detach()` 把那一行从计算图里摘出去：前向数值一模一样，反向时这一行压根没有 `grad_fn` 指向 `self.weight`，梯度精确为 0。

#warn[
  "我把 padding 位置的 loss mask 掉了"不等价于设了 `padding_idx`。loss mask 只挡住了从 loss 直接过来的那一条路径；如果 padding token 的 embedding 还进了 attention 的 K/V、或者被 mean-pooling 平均进去了，梯度照样从别的路径流回来。
]

*weight tying*：很多 LM 让 input embedding 和输出投影共享同一个 $(V, D)$ 矩阵（`lm_head.weight = embed.weight`）。能共享正是因为这两者是同一个矩阵的两种用法——一边 gather，一边 matmul。省下 $V times D$ 个参数，在 $V = 128"k"$、$D = 4096$ 时是 5 亿参数。

*自证*：与 `nn.Embedding(..., padding_idx=pad)` 对齐前向；再各自 backward 一次，比对 `weight.grad` 全等，并断言 `weight.grad[pad]` 全 0 而非 padding 行非 0。

== Conv2d：im2col / unfold

手写卷积是最高频的一题。核心思路一句话：*把卷积变成一次矩阵乘*。

把每个滑窗位置上的 $(C_"in", K_H, K_W)$ 立方体拉平成一个长度 $K = C_"in" K_H K_W$ 的列向量，整张图就变成 $(K, L)$ 的矩阵（$L = O_H O_W$ 是滑窗个数）。卷积核 reshape 成 $(C_"out", K)$，一次矩阵乘就算完了。

#figure(
  align(center, shape-pipeline(stages: (
    ("input", "(N, Cin, H, W)", "原始特征图"),
    ("F.unfold", "(N, Cin*KH*KW, L)", "L = OH*OW 个滑窗，每窗展平成一列"),
    ("weight.reshape", "(Cout, Cin*KH*KW)", "每个 kernel 拉平成一行"),
    ("matmul", "(N, Cout, L)", "(Cout,K) @ (N,K,L) broadcast 成 batched GEMM"),
    ("view", "(N, Cout, OH, OW)", "把 L 拆回二维空间"),
  ))),
  caption: [im2col 的 shape 变化。关键是第二步：`F.unfold` 把不规则的滑窗访存 pattern 展平成规则的列，之后就完全是 GEMM 的地盘了。],
) <fig-im2col>

```python
import torch.nn.functional as F

def _pair(v):
    return (v, v) if isinstance(v, int) else v

def my_conv2d_unfold(x, weight, bias=None, stride=1, padding=0):
    n, _, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    cols = F.unfold(x, (kh, kw), stride=(sh, sw), padding=(ph, pw))  # (N, K, L)
    out = weight.reshape(c_out, -1) @ cols   # (Cout,K) @ (N,K,L) -> (N,Cout,L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out.view(n, c_out, oh, ow)
```

*输出尺寸公式（要能默写）.* 带 dilation 的完整版：

#formula[$ O_H = floor((H + 2 p_h - d_h (K_H - 1) - 1) / s_h) + 1 $]

无 dilation（$d = 1$）时化简成常用形式：

#formula[$ O_H = floor((H + 2 p_h - K_H) / s_h) + 1 $]

推论：`kernel=3, padding=1, stride=1` 恰好保持尺寸不变（"same" 卷积），这是 3×3 卷积统治 CNN 的实用原因之一。

*im2col 的显存代价.* `cols` 的元素个数是 $N times C_"in" K_H K_W times O_H O_W$。原图是 $N times C_"in" times H W$。stride=1、padding 保尺寸时，$O_H O_W approx H W$，于是

#formula[$ "显存放大倍数" approx K_H times K_W $]

3×3 卷积就是 9 倍。一个 $(32, 256, 56, 56)$ 的 fp16 activation 本身 51 MB，展开后 460 MB。这是 im2col 最大的缺点，也是别的卷积算法存在的理由：

*显式 im2col*（物化 col 矩阵后调 GEMM）只适合教学和 CPU 参考实现；*隐式 GEMM* 不物化 col，读 shared memory 时按需算索引，是 cuDNN 在绝大多数 shape 上的默认选择；*Winograd* 在 3×3、stride=1 上用变换换掉乘法次数；*FFT* 走频域点乘，只在大 kernel（7×7 以上）划算。面试里能说出"cuDNN 的 `IMPLICIT_GEMM` 就是不物化的 im2col"，比背公式加分。

#warn[
  最难查的一个错：`F.unfold` 展开后 `K` 维的排列顺序是 `(Cin, KH, KW)`，`weight.reshape(Cout, -1)` 的顺序也必须是 `(Cin, KH, KW)`。如果你在 reshape 之前 permute 过 weight（比如从 NHWC 权重转过来），shape 依然完全正确、程序不报错，但算出来的数完全是错的。所以卷积这题一定要跑数值对齐，不能只看 shape。
]

*自证*：`torch.testing.assert_close(my_conv2d_unfold(x, w, b, s, p), F.conv2d(x, w, b, s, p), rtol=1e-5, atol=1e-5)`，遍历 `stride` / `padding` 的几种组合以及非方形 kernel；再对两边 backward 比 `x.grad` 和 `w.grad`（unfold 本身可微，反向白拿）。

== MaxPool2d

同样用 `unfold` 展开滑窗，只是把 matmul 换成沿窗口维取 max。但有一个陷阱。

```python
class MyMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def forward(self, x):
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1

        # 关键：手动补 -inf，而不是让 unfold 补 0
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph), value=float("-inf"))
        cols = F.unfold(x, (kh, kw), stride=(sh, sw))   # (N, C*kh*kw, L)
        cols = cols.view(n, c, kh * kw, -1)
        return cols.amax(dim=2).view(n, c, oh, ow)
```

#warn[
  *`F.unfold` 的 `padding` 参数补的是 0*。对 conv 这正好符合语义（补 0 的位置贡献 0），但对 max pooling 是错的：如果特征图全是负数，补进来的 0 会成为窗口最大值，输出里凭空出现 0。

  ```python
  x = -torch.rand(1, 2, 5, 5) - 1.0                # 全部 < -1
  ninf = float("-inf")

  # 错：让 unfold 补 0，边缘窗口的最大值变成 0
  bad = F.unfold(x, 3, stride=2, padding=1).view(1, 2, 9, -1).amax(2)
  # 对：先补 -inf，padding 位置不参与竞争
  good = F.unfold(F.pad(x, (1, 1, 1, 1), value=ninf), 3, stride=2)
  good = good.view(1, 2, 9, -1).amax(2)

  bad.max()                                        # tensor(0.)      ✗
  torch.testing.assert_close(good, F.max_pool2d(x, 3, 2, 1).flatten(2))   # ✓
  ```

  官方 `max_pool2d` 的语义是"padding 位置不参与竞争"，等价于补 $-infinity$。这个 bug 只在输入全负时暴露，随机初始化的测试数据（有正有负）*抓不到*，必须专门构造。
]

*反向是稀疏的 scatter*：只有每个窗口里的 argmax 位置收到梯度，其余位置为 0——这就是 `return_indices=True` 存在的意义（喂给 `MaxUnpool2d`）；窗口重叠时（`stride < kernel`）同一个输入位置可能被多个窗口选中，梯度要累加。对比 AvgPool：它的反向是把梯度均摊到窗口内所有位置，处处非零；MaxPool 的梯度稀疏，训练早期容易让部分区域完全收不到信号。
- `ceil_mode=True` 会把输出尺寸公式里的 `floor` 换成 `ceil`，并保证最后一个窗口的起点仍在输入内。本实现只支持 `floor`，白板上说明这一点即可。

*自证*：遍历 `(k, s)` 组合与 `F.max_pool2d` 对齐；再专门用*全负输入 + padding* 跑一次，断言输出全 `< 0`。

== 汇总：九个层的对照表

#table(
  columns: (auto, auto, 1.6fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*手写实现*], [*官方对照*], [*最容易错的点*],
  [`MyLinear`], [`nn.Linear`], [weight 是 `(out, in)`；默认初始化是 $U(plus.minus 1\/sqrt("fan_in"))$，别写 `randn`],
  [`MySoftmax`], [`torch.softmax`], [不减 max → fp32 在 88.7、fp16 在 11.1 就 `inf`],
  [`MyLayerNorm`], [`nn.LayerNorm`], [`eps` 要在 sqrt 内；`var` 要 `unbiased=False`],
  [`MyRMSNorm`], [`nn.RMSNorm`], [`nn.RMSNorm` 的 `eps` 默认是 `None` → `finfo.eps`；`mean(x^2) != var(x)`],
  [`MyBatchNorm1d`], [`nn.BatchNorm1d`], [归一化用有偏方差、`running_var` 存无偏；momentum 语义与 Keras 相反],
  [`MyDropout`], [`nn.Dropout`], [inverted：训练时放大 $1\/(1-p)$；eval 必须恒等],
  [`MyEmbedding`], [`nn.Embedding`], [`padding_idx` 要屏蔽梯度，不只是初始化为 0],
  [`my_conv2d_unfold`], [`F.conv2d`], [`K` 维顺序必须是 `(Cin, KH, KW)`；im2col 显存放大 $K_H K_W$ 倍],
  [`MyMaxPool2d`], [`F.max_pool2d`], [padding 必须补 $-infinity$，不能靠 `unfold` 补 0],
)

#insight[
  这九个层里，有七个的"最容易错的点"都是*数值语义*而不是 shape 逻辑。所以白板写完之后，主动说一句"我会把官方权重拷进来跑 `assert_close`，并且专门构造能触发边界的输入（常数张量验 eps、全负输入验 pooling padding、大 logits 验 softmax）"——这一句话往往比代码本身更能说明你写过。
]

== 面试考点

#interview[
  *Q1*：`nn.Linear` 的默认初始化是什么分布？`a=math.sqrt(5)` 是什么意思？

  A：`kaiming_uniform_(weight, a=sqrt(5))`。kaiming uniform 的边界是 `sqrt(6 / ((1+a^2) * fan_in))`，代入 `a=sqrt(5)` 后 `1+a^2=6`，约掉就是 `1/sqrt(fan_in)`，即 `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`。`a` 本意是 leaky ReLU 的负斜率，但这里没有 leaky ReLU，`sqrt(5)` 纯粹是从 Torch7 继承的历史常数，作用就是把公式凑回老的均匀分布。
]

#interview[
  *Q2*：softmax 为什么要减最大值？不减会怎样？

  A：因为 `softmax(x + c) = softmax(x)`（分子分母同乘 `e^c` 约掉），减 max 后所有指数输入都 ≤ 0，`exp` 落在 `(0, 1]` 永不上溢，分母至少含一项 1 也不会下溢成 0。不减的话 fp32 在 `x > 88.7` 就 `inf`，fp16 在 `x > 11.1` 就 `inf`，最后 `inf/inf = NaN`。fp16 那 11 的余量在没做 QK 缩放的 attention 上很容易撞上。
]

#interview[
  *Q3*：BatchNorm 归一化时用有偏方差还是无偏方差？`running_var` 呢？

  A：归一化用*有偏*（除 N），`running_var` 存*无偏*（除 N−1）。因为 `running_var` 是在估计总体方差，要无偏；而归一化只是给当前 batch 做缩放，除 N 才让输出方差正好是 1。副作用是同一个 batch 在 train / eval 下输出差 `sqrt(N/(N-1))` —— N=4 时约 15%。
]

#interview[
  *Q4*：PyTorch 的 BN `momentum=0.1` 和 Keras 的 `0.99` 是一回事吗？

  A：不是，语义正好相反。PyTorch 是 `running = (1-m)*running + m*batch`，`m` 是*新观测的权重*；Keras 是 `moving = m*moving + (1-m)*batch`，`m` 是*旧值的权重*。换算关系是 `m_torch = 1 - m_keras`。迁移时把 0.99 直接抄进 PyTorch，等于让 running stats 只记得最后一个 batch。
]

#interview[
  *Q5*：为什么是 inverted dropout（训练时放大）而不是推理时缩小？

  A：训练输出 `m*x/(1-p)` 的期望正好是 `x`，于是推理路径退化成纯恒等：零开销、零分支，导出 ONNX 或做图优化时 dropout 直接消失。而且 `p` 可以随时改（dropout schedule）而不影响推理逻辑，也不会出现"训练好的模型忘了乘 (1-p)"这种事故。另外 `p=1` 要特判成全 0，否则 `1/(1-p)` 除零。
]

#interview[
  *Q6*：手写 conv2d，讲讲 im2col 的思路和代价；MaxPool 能不能照抄？

  A：`F.unfold` 把每个滑窗的 `(Cin, KH, KW)` 立方体展平成一列，得到 `(N, K, L)`，权重 reshape 成 `(Cout, K)`，一次 batched matmul 就是卷积；输出尺寸 `floor((H + 2p - k)/s) + 1`。代价是 col 矩阵放大约 `KH*KW` 倍显存，所以 cuDNN 用的是不物化 col 的隐式 GEMM，小 kernel 还有 Winograd。MaxPool *不能照抄*：`unfold` 的 padding 补 0，特征图全负时补进来的 0 会成为窗口最大值，必须先 `F.pad(x, ..., value=-inf)`。
]
