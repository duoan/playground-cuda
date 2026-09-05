"""PyTorch 面试手写题（一）：从零实现 torch.nn 的基础层。

这个文件里的每个类都控制在「白板 20 行以内」的规模，配套的 pytest 会把官方层的
权重拷进手写实现里，用 torch.testing.assert_close 做数值对齐 —— 这是验证
「我真的懂这个层」最硬的标准：shape 对了不算对，数值一致才算对。

阅读顺序建议：MyLinear -> MyLayerNorm/MyRMSNorm -> MyBatchNorm1d（考点最密集）
-> MyDropout -> MyEmbedding -> MySoftmax -> my_conv2d_unfold -> MyMaxPool2d。
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 1. Linear
# ---------------------------------------------------------------------------


class MyLinear(nn.Module):
    """y = x @ W.T + b。

    面试要点 —— 为什么 weight 存成 (out_features, in_features) 而不是 (in, out)？
      1. 前向写成 ``x @ W.T``，底层是 ``addmm(bias, x, W.t())``。cuBLAS 的
         GEMM 本来就带转置标志位，W.t() 不产生拷贝，所以「多一次转置」是免费的。
      2. 行主序下 (out, in) 让「同一个输出神经元的所有输入权重」在内存里连续。
         反向算 grad_input = grad_out @ W 时按行扫 W，cache 友好；而且做算子融合
         / 权重量化时，per-output-channel 的 scale 正好按行切，天然对齐。
      3. fan_in = W.shape[1]、fan_out = W.shape[0]，初始化代码不用关心是哪种布局。
      4. 现实原因：所有预训练权重都是这个布局，改了就没法直接 load_state_dict。

    常见错法：初始化直接用 ``torch.randn``（方差 1，深层网络必炸）；或者用
    kaiming_uniform 时忘了 nn.Linear 用的是 ``a=math.sqrt(5)`` 这个历史遗留参数，
    导致和官方初始化分布对不上。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # a=sqrt(5) 是 nn.Linear 的历史默认值：等价于 U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Float[Tensor, "... I"]) -> Float[Tensor, "... O"]:
        # matmul 会自动把前面所有维度当 batch，不需要手动 reshape 成 2D
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# 2. LayerNorm / RMSNorm
# ---------------------------------------------------------------------------


class MyLayerNorm(nn.Module):
    """沿最后若干维做归一化：y = (x - mean) / sqrt(var + eps) * w + b。

    面试要点：
      1. **eps 必须放在 sqrt 里面**：``sqrt(var + eps)`` 而不是 ``sqrt(var) + eps``。
         后者在 var -> 0 时导数 1/(2*sqrt(var)) 会爆炸，前者被 eps 兜住。这是最常
         被抓的一个细节。
      2. **var 用有偏估计**（除以 N，即 ``unbiased=False`` / ``correction=0``）。
         LayerNorm 不是在估计总体方差，它只是在对这一条样本做缩放，用 N-1 反而
         会让 N 很小时（比如 head_dim=4）结果和官方对不上。
      3. 归一化的维度是**最后 len(normalized_shape) 维**，不是 batch 维。和
         BatchNorm 的区别：LN 逐样本、与 batch size 无关，所以推理时没有
         running stats，train/eval 行为完全一致。
      4. 混合精度下官方会把统计量提升到 fp32 再算，手写版在 bf16 下直接算
         mean/var 容易掉精度。

    常见错法：``x.var(dim, keepdim=True)`` 默认 unbiased=True，直接用就和官方差一点。
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
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

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        # unbiased=False：除以 N 而不是 N-1
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x_hat = x_hat * self.weight + self.bias
        return x_hat


class MyRMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * w。相比 LayerNorm 去掉了均值中心化和 bias。

    面试要点：
      1. 为什么能去掉减均值？LayerNorm 的收益主要来自「re-scaling 不变性」而非
         「re-centering 不变性」（RMSNorm 论文的核心论点）。去掉 mean 少一次
         reduction，kernel 只需要一趟算平方和，LLM 里能省下可观的显存带宽。
      2. **注意 mean(x^2) 不等于 var(x)**，两者只有在 mean(x)=0 时才相等。写成
         ``x.var()`` 是错的。
      3. 没有 bias：Llama / T5 系列都是 weight-only，bias 对效果基本无贡献。
      4. torch 2.4+ 才有 nn.RMSNorm，且 ``eps=None`` 时默认取
         ``torch.finfo(dtype).eps``（fp32 约 1.19e-7），不是 1e-5！对齐测试里
         必须显式传 eps，否则会以为自己写错了。

    常见错法：先 cast 到 bf16 再算平方和 —— 平方会放大动态范围，长序列下容易溢出，
    生产实现都是在 fp32 里算 rstd 再 cast 回去。
    """

    def __init__(
        self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        dims = tuple(range(-len(self.normalized_shape), 0))
        ms = x.pow(2).mean(dim=dims, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


# ---------------------------------------------------------------------------
# 3. BatchNorm1d
# ---------------------------------------------------------------------------


class MyBatchNorm1d(nn.Module):
    """BatchNorm1d，支持 (N, C) 和 (N, C, L) 两种输入。

    面试考点密度最高的一个层，四个必答点：

    1. **momentum 的方向和别的框架相反**。PyTorch 是
       ``running = (1 - momentum) * running + momentum * batch``，
       momentum=0.1 表示「新观测占 10%」。TensorFlow/Keras 的 momentum=0.99
       表示「旧值占 99%」，语义正好是 1 - PyTorch 的 momentum。迁移代码时把
       0.99 直接抄过来 = 几乎完全不更新 running stats。

    2. **归一化用有偏方差，running_var 存无偏方差**。这不是笔误，是官方行为：
       - 归一化：``x_hat = (x - mu) / sqrt(var_biased + eps)``，除以 N；
       - 更新：``running_var = (1-m) * running_var + m * var_unbiased``，除以 N-1。
       原因是 running_var 是在**估计总体方差**（要无偏），而归一化只是对当前
       batch 做缩放（用 N 才让 x_hat 的方差正好是 1）。副作用：train 和 eval
       在同一个 batch 上的输出会有 N/(N-1) 量级的系统性差异，batch 很小时肉眼可见。

    3. **eps 加在 sqrt 里面**，同 LayerNorm。

    4. **train / eval 行为不同**：train 用当前 batch 统计并更新 running stats，
       eval 用 running stats 且不更新。忘了 ``model.eval()`` 是线上事故的经典原因；
       另一个坑是 batch_size=1 时 train 模式下方差为 0，输出全是 bias。

    常见错法：用 ``x.var(unbiased=True)`` 去归一化；或者 eval 时还在更新
    num_batches_tracked / running stats。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        # buffer 而非 parameter：会进 state_dict、会跟着 .to(device)，但不参与梯度
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Float[Tensor, "N C ..."]) -> Float[Tensor, "N C ..."]:
        assert x.dim() in (2, 3), "BatchNorm1d 只接受 (N, C) 或 (N, C, L)"
        # 除了通道维 C 之外的所有维度都要 reduce 掉
        dims = (0,) if x.dim() == 2 else (0, 2)

        if self.training:
            mean = x.mean(dim=dims)
            var_biased = x.var(dim=dims, unbiased=False)  # 用于归一化
            with torch.no_grad():
                var_unbiased = x.var(dim=dims, unbiased=True)  # 用于 running stats
                self.running_mean.mul_(1 - self.momentum).add_(
                    self.momentum * mean.detach()
                )
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var_unbiased
                )
                self.num_batches_tracked += 1
        else:
            mean, var_biased = self.running_mean, self.running_var

        shape = (1, -1) if x.dim() == 2 else (1, -1, 1)
        x_hat = (x - mean.view(shape)) * torch.rsqrt(var_biased.view(shape) + self.eps)
        if self.weight is not None:
            x_hat = x_hat * self.weight.view(shape) + self.bias.view(shape)
        return x_hat


# ---------------------------------------------------------------------------
# 4. Dropout
# ---------------------------------------------------------------------------


class MyDropout(nn.Module):
    """Inverted dropout：train 时置零并按 1/(1-p) 放大，eval 时恒等。

    面试要点 —— 为什么缩放放在**训练**时（inverted），而不是推理时？
      设 m ~ Bernoulli(1-p)。朴素 dropout 训练时输出 m*x，期望是 (1-p)*x，
      所以推理时必须乘 (1-p) 才能对上分布。inverted dropout 把这个系数挪到
      训练侧：输出 m*x/(1-p)，期望正好是 x，于是**推理路径就是纯恒等**。
      好处：
        1. 推理零开销、零分支，部署 / 导出 ONNX / 图优化时 dropout 直接消失；
        2. p 可以随时改（比如 dropout schedule）而不影响已保存的推理逻辑；
        3. 只有一套权重语义，不会出现「训练好的模型忘了乘 (1-p)」这种事故。

    另外两个考点：
      - **p=1 时 1/(1-p) 会除零**，官方对 p=1 特判为全 0 输出。
      - mask 必须每次前向重新采样，且**同一个 mask 要同时用在 forward 和 backward**
        （autograd 自动保证）。手写 CUDA kernel 时要保存 mask 或复用同一个 RNG offset。

    常见错法：eval 时还在 drop；或者用 ``torch.rand(x.shape) > p`` 得到 mask 后
    忘了缩放，导致推理期望值偏小。
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert 0.0 <= p <= 1.0, "dropout 概率必须在 [0, 1]"
        self.p = p

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        # empty().bernoulli_() 比 rand() > p 少一次比较，且和官方实现一致
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


# ---------------------------------------------------------------------------
# 5. Embedding
# ---------------------------------------------------------------------------


class MyEmbedding(nn.Module):
    """Embedding 本质就是 ``weight[idx]`` —— 一次 gather，不是矩阵乘。

    面试要点：
      1. 「embedding 等价于 one-hot 乘权重矩阵」在数学上对，但实现上绝不能这么做：
         one-hot 是 (N, V) 的稠密矩阵，V 通常 5 万到 15 万，显存和算力都浪费掉了。
         用 index 做 gather 是 O(N*D)，one-hot matmul 是 O(N*V*D)。
      2. **反向是 scatter-add**：同一个 token 在 batch 里出现多次，梯度要累加到
         同一行。这也是为什么 embedding 的梯度天然稀疏，可以用 ``sparse=True``
         配合 SparseAdam。
      3. **padding_idx 那一行梯度恒为 0**，且初始化为全 0。注意光把权重置零不够，
         梯度不屏蔽的话优化器（尤其带 weight decay / momentum 的）会把它推离 0。
         这里用 detach 把该行从计算图里摘出去来实现。
      4. 权重共享（weight tying）：很多 LM 让 embedding 和输出投影共享同一个矩阵，
         正因为两者是同一个 (V, D) 张量的 gather 和 matmul 两种用法。

    常见错法：把 padding 位的 loss mask 掉就以为够了 —— 那只挡住了 loss 路径，
    如果 padding embedding 还参与了 attention/pooling，梯度照样会流回来。
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)  # 官方默认 N(0, 1)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, idx: Int[Tensor, "..."]) -> Float[Tensor, "... D"]:
        weight = self.weight
        if self.padding_idx is not None:
            # clone 后把 padding 行替换成 detach 版本：前向数值不变，
            # 反向时这一行不在计算图里，梯度自然是 0
            weight = weight.clone()
            weight[self.padding_idx] = self.weight[self.padding_idx].detach()
        return weight[idx]


# ---------------------------------------------------------------------------
# 6. Softmax
# ---------------------------------------------------------------------------


class MySoftmax(nn.Module):
    """数值稳定版 softmax：先减去每行最大值再取 exp。

    面试要点：
      1. **为什么减 max 不改变结果**：softmax(x + c) = softmax(x) 对任意常数 c 成立，
         因为分子分母同时乘了 e^c。取 c = -max(x) 后所有指数的输入都 <= 0，
         exp 结果落在 (0, 1]，永远不会上溢；分母至少包含一项 e^0 = 1，也不会
         下溢成 0 导致 0/0 = NaN。
      2. fp32 的 exp 在输入 > 88 时就 inf，fp16 更早（> 11）。所以「输入是
         logits，量级不大」这个假设在实际训练里经常不成立（比如没加 QK scaling 的
         attention、或者 loss 爆炸的中间态）。
      3. **online softmax / FlashAttention** 就是把「求 max」「求 exp 和」
         这两趟 reduction 融合成一趟，边扫边修正之前累计的分母。手写 CUDA 时
         这是必考的下一步。
      4. 分类任务里不要 ``log(softmax(x))``，要用 ``log_softmax``：后者把
         log 和 exp 解析地约掉了（x - max - log(sum exp)），少一次精度损失。

    常见错法：``keepdim=False`` 导致广播维度错位；或者在 dim=0 上做 softmax
    却以为在 dim=-1 上做。
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x_max = x.amax(dim=self.dim, keepdim=True)
        e = torch.exp(x - x_max)
        return e / e.sum(dim=self.dim, keepdim=True)


# ---------------------------------------------------------------------------
# 7. Conv2d via im2col (unfold)
# ---------------------------------------------------------------------------


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else v


def my_conv2d_unfold(
    x: Float[Tensor, "N Cin H W"],
    weight: Float[Tensor, "Cout Cin KH KW"],
    bias: Float[Tensor, " Cout"] | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Float[Tensor, "N Cout OH OW"]:
    """im2col + GEMM 实现 conv2d —— 手写卷积的第一高频题。

    核心思想：把每个滑窗位置的 (Cin, KH, KW) 立方体拉平成一个长度 K=Cin*KH*KW 的
    列向量，整张图就变成 (K, L) 的矩阵（L = OH*OW 个滑窗）。卷积核 reshape 成
    (Cout, K)，一次矩阵乘 (Cout,K) @ (N,K,L) -> (N,Cout,L) 就算完了。

    面试要点：
      1. **为什么要 im2col**：卷积的访存 pattern 不规则，直接写六重循环打不满
         算力。转成 GEMM 之后可以直接吃 cuBLAS/oneDNN 几十年调优的成果。
         cuDNN 的 IMPLICIT_GEMM 算法就是这个思路，只是不真的物化 col 矩阵，
         而是在读 shared memory 时按需算索引（省掉 KH*KW 倍的显存放大）。
      2. **输出尺寸公式**：OH = floor((H + 2*ph - dilation*(KH-1) - 1) / sh) + 1。
         无 dilation 时化简成 (H + 2*ph - KH)//sh + 1。这个公式要能默写。
      3. **显存放大 KH*KW 倍**：3x3 卷积的 col 矩阵是原图的 9 倍，这是 im2col
         最大的缺点，也是 FFT/Winograd 卷积存在的理由。
      4. ``F.unfold`` 的 padding 是补 0，正好符合 conv 的语义；但下面 MaxPool
         就不能这么干（见 MyMaxPool2d）。
      5. ``(Cout, K) @ (N, K, L)`` 靠 broadcast 变成 batched matmul，不需要
         手动 expand 权重。

    常见错法：unfold 出来的 (N, K, L) 里 K 的排列顺序是 (Cin, KH, KW)，
    weight.view(Cout, -1) 的顺序也必须是 (Cin, KH, KW) —— 如果先 permute 过
    weight 就会静默地算出错误结果（shape 还是对的，最难查）。
    """
    n, _, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    cols = F.unfold(x, (kh, kw), stride=(sh, sw), padding=(ph, pw))  # (N, K, L)
    out = weight.reshape(c_out, -1) @ cols  # (Cout, K) @ (N, K, L) -> (N, Cout, L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out.view(n, c_out, oh, ow)


# ---------------------------------------------------------------------------
# 8. MaxPool2d
# ---------------------------------------------------------------------------


class MyMaxPool2d(nn.Module):
    """同样用 unfold 展开滑窗，只是把 matmul 换成沿窗口维取 max。

    面试要点：
      1. **padding 必须补 -inf，不能补 0**。这是本题唯一的陷阱：F.unfold 的
         padding 参数补的是 0，如果特征图全是负数（比如经过某些归一化后），
         补进来的 0 会成为窗口最大值，结果就错了。官方 max_pool2d 的语义是
         「padding 位置不参与竞争」，等价于补 -inf。
      2. **反向是稀疏的 scatter**：只有每个窗口里的 argmax 位置收到梯度，其余为 0。
         所以 ``return_indices=True`` 存在的意义就是给 MaxUnpool2d 用。
         窗口重叠时（stride < kernel）同一个输入位置可能被多个窗口选中，梯度要累加。
      3. 和 AvgPool 的区别：AvgPool 的反向是把梯度均摊，处处非零；MaxPool 的
         梯度稀疏，训练早期容易让部分区域完全收不到信号。
      4. ceil_mode 会改变输出尺寸公式（floor 换成 ceil），本实现只支持 floor。

    常见错法：直接 ``F.unfold(x, k, padding=p)`` 然后 max —— 见第 1 点。
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def forward(self, x: Float[Tensor, "N C H W"]) -> Float[Tensor, "N C OH OW"]:
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1

        # 关键：手动补 -inf，而不是让 unfold 补 0
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph), value=float("-inf"))
        cols = F.unfold(x, (kh, kw), stride=(sh, sw))  # (N, C*kh*kw, L)
        cols = cols.view(n, c, kh * kw, -1)
        out = cols.amax(dim=2)  # (N, C, L)
        return out.view(n, c, oh, ow)


# ===========================================================================
#                                  tests
# ===========================================================================


def _copy_(dst: nn.Module, src: nn.Module) -> None:
    """把 src 的同名参数/buffer 拷进 dst，用于对齐测试。"""
    with torch.no_grad():
        dst.load_state_dict(
            {k: v.clone() for k, v in src.state_dict().items()}, strict=False
        )


# ---- MyLinear -------------------------------------------------------------


def test_linear_matches_nn_linear():
    torch.manual_seed(0)
    ref = nn.Linear(6, 4)
    mine = MyLinear(6, 4)
    _copy_(mine, ref)

    x = torch.randn(3, 5, 6)
    torch.testing.assert_close(mine(x), ref(x))
    assert mine.weight.shape == (4, 6), "weight 必须是 (out, in)"


def test_linear_no_bias_and_init_range():
    torch.manual_seed(0)
    mine = MyLinear(16, 8, bias=False)
    assert mine.bias is None
    x = torch.randn(2, 16)
    torch.testing.assert_close(mine(x), x @ mine.weight.t())

    # kaiming_uniform(a=sqrt(5)) 等价于 U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    bound = 1 / math.sqrt(16)
    assert mine.weight.abs().max() <= bound + 1e-6


def test_linear_backward_flows():
    torch.manual_seed(0)
    mine = MyLinear(4, 3)
    x = torch.randn(5, 4, requires_grad=True)
    mine(x).sum().backward()
    assert x.grad is not None
    assert mine.weight.grad is not None and mine.bias.grad is not None


# ---- MyLayerNorm ----------------------------------------------------------


def test_layernorm_matches_nn_layernorm_last_dim():
    torch.manual_seed(0)
    ref = nn.LayerNorm(8, eps=1e-5)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    mine = MyLayerNorm(8, eps=1e-5)
    _copy_(mine, ref)

    x = torch.randn(2, 3, 8) * 5 + 2
    torch.testing.assert_close(mine(x), ref(x))


def test_layernorm_matches_nn_layernorm_multi_dim():
    """normalized_shape 是多维时（例如 (3, 8)）也要沿最后两维一起归一化。"""
    torch.manual_seed(1)
    ref = nn.LayerNorm((3, 8))
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    mine = MyLayerNorm((3, 8))
    _copy_(mine, ref)

    x = torch.randn(4, 3, 8)
    torch.testing.assert_close(mine(x), ref(x))


def test_layernorm_output_is_normalized():
    torch.manual_seed(2)
    mine = MyLayerNorm(64, eps=1e-5)
    x = torch.randn(2, 64) * 10 + 3
    y = mine(x)
    torch.testing.assert_close(y.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)
    # 有偏方差归一化后标准差应为 1（eps 带来的偏差可忽略）
    torch.testing.assert_close(y.var(-1, unbiased=False), torch.ones(2), atol=1e-4, rtol=0)


def test_layernorm_eps_inside_sqrt():
    """常数输入下 var=0：eps 在 sqrt 内时输出恒为 0，在外面则会 inf/NaN。"""
    mine = MyLayerNorm(4, eps=1e-5)
    x = torch.full((2, 4), 7.0)
    y = mine(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))


# ---- MyRMSNorm ------------------------------------------------------------


def test_rmsnorm_matches_nn_rmsnorm():
    """torch 2.10 有 nn.RMSNorm，但它 eps=None 时默认取 finfo.eps，必须显式传。"""
    assert hasattr(nn, "RMSNorm"), "torch < 2.4 没有 nn.RMSNorm，请改用手算参考值"
    torch.manual_seed(0)
    ref = nn.RMSNorm(8, eps=1e-5)
    nn.init.normal_(ref.weight)
    mine = MyRMSNorm(8, eps=1e-5)
    _copy_(mine, ref)

    x = torch.randn(2, 3, 8) * 3 + 1
    torch.testing.assert_close(mine(x), ref(x))


def test_rmsnorm_matches_manual_reference():
    """不依赖官方实现的手算参考值，顺便强调 mean(x^2) != var(x)。"""
    torch.manual_seed(3)
    mine = MyRMSNorm(8, eps=1e-6)
    x = torch.randn(2, 8) + 5.0  # 均值明显不为 0，var 和 mean(x^2) 差很多
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(mine(x), expected)

    wrong = x / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-6)
    assert not torch.allclose(mine(x), wrong), "mean(x^2) 与 var(x) 不应等价"


def test_rmsnorm_has_no_bias():
    mine = MyRMSNorm(4)
    assert not hasattr(mine, "bias") or mine.bias is None
    assert list(dict(mine.named_parameters())) == ["weight"]


# ---- MyBatchNorm1d --------------------------------------------------------


def test_batchnorm1d_train_matches_official_2d():
    torch.manual_seed(0)
    ref = nn.BatchNorm1d(5)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    mine = MyBatchNorm1d(5)
    _copy_(mine, ref)
    ref.train()
    mine.train()

    for _ in range(3):
        x = torch.randn(7, 5) * 2 + 1
        torch.testing.assert_close(mine(x), ref(x))
        # running stats 也要一路对齐
        torch.testing.assert_close(mine.running_mean, ref.running_mean)
        torch.testing.assert_close(mine.running_var, ref.running_var)
    assert int(mine.num_batches_tracked) == int(ref.num_batches_tracked) == 3


def test_batchnorm1d_train_matches_official_3d():
    torch.manual_seed(1)
    ref = nn.BatchNorm1d(4)
    mine = MyBatchNorm1d(4)
    _copy_(mine, ref)
    ref.train()
    mine.train()

    x = torch.randn(3, 4, 6)
    torch.testing.assert_close(mine(x), ref(x))
    torch.testing.assert_close(mine.running_var, ref.running_var)


def test_batchnorm1d_eval_matches_official():
    torch.manual_seed(2)
    ref = nn.BatchNorm1d(5)
    mine = MyBatchNorm1d(5)
    # 先在 train 模式下跑几步把 running stats 喂出非平凡值
    for _ in range(4):
        ref(torch.randn(8, 5) * 3 - 1)
    _copy_(mine, ref)
    ref.eval()
    mine.eval()

    x = torch.randn(2, 5)
    torch.testing.assert_close(mine(x), ref(x))
    # eval 下 running stats 不能被更新
    before = mine.running_mean.clone()
    mine(torch.randn(2, 5) * 100)
    torch.testing.assert_close(mine.running_mean, before)


def test_batchnorm1d_biased_vs_unbiased_var_gap():
    """经典考点实证：归一化用有偏方差，running_var 存无偏方差，两者差 N/(N-1)。"""
    torch.manual_seed(3)
    n = 5
    mine = MyBatchNorm1d(3, momentum=1.0)  # momentum=1 => running 直接等于本 batch
    mine.train()
    x = torch.randn(n, 3)
    mine(x)

    torch.testing.assert_close(mine.running_var, x.var(0, unbiased=True))
    ratio = mine.running_var / x.var(0, unbiased=False)
    torch.testing.assert_close(ratio, torch.full((3,), n / (n - 1)))


# ---- MyDropout ------------------------------------------------------------


def test_dropout_eval_is_identity():
    mine = MyDropout(0.5).eval()
    x = torch.randn(4, 8)
    torch.testing.assert_close(mine(x), x)
    assert mine(x) is x, "eval 应该直接返回原张量，零开销"


def test_dropout_train_zero_ratio_and_expectation():
    torch.manual_seed(0)
    p = 0.3
    mine = MyDropout(p).train()
    x = torch.ones(200, 500)
    y = mine(x)

    zero_ratio = (y == 0).float().mean().item()
    assert abs(zero_ratio - p) < 0.01, f"置零比例应接近 p={p}，实际 {zero_ratio}"
    # inverted dropout 的期望等于输入本身
    assert abs(y.mean().item() - 1.0) < 0.01
    # 非零元素恰好被放大成 1/(1-p)
    nonzero = y[y != 0]
    torch.testing.assert_close(nonzero, torch.full_like(nonzero, 1 / (1 - p)))


def test_dropout_p0_and_p1_edge_cases():
    x = torch.randn(3, 4)
    torch.testing.assert_close(MyDropout(0.0).train()(x), x)
    torch.testing.assert_close(MyDropout(1.0).train()(x), torch.zeros_like(x))


def test_dropout_masks_differ_between_calls():
    torch.manual_seed(0)
    mine = MyDropout(0.5).train()
    x = torch.ones(64, 64)
    assert not torch.equal(mine(x), mine(x)), "每次前向必须重新采样 mask"


# ---- MyEmbedding ----------------------------------------------------------


def test_embedding_matches_nn_embedding():
    torch.manual_seed(0)
    ref = nn.Embedding(10, 4)
    mine = MyEmbedding(10, 4)
    _copy_(mine, ref)

    idx = torch.randint(0, 10, (3, 5))
    torch.testing.assert_close(mine(idx), ref(idx))
    assert mine(idx).shape == (3, 5, 4)


def test_embedding_padding_idx_zero_grad():
    torch.manual_seed(0)
    pad = 2
    ref = nn.Embedding(10, 4, padding_idx=pad)
    mine = MyEmbedding(10, 4, padding_idx=pad)
    _copy_(mine, ref)

    # padding 行初始化为全 0
    torch.testing.assert_close(mine.weight[pad], torch.zeros(4))

    idx = torch.tensor([[1, pad, 3], [pad, pad, 5]])
    torch.testing.assert_close(mine(idx), ref(idx))

    mine(idx).pow(2).sum().backward()
    ref(idx).pow(2).sum().backward()
    torch.testing.assert_close(mine.weight.grad, ref.weight.grad)
    torch.testing.assert_close(mine.weight.grad[pad], torch.zeros(4))
    assert mine.weight.grad[1].abs().sum() > 0, "非 padding 行必须有梯度"


def test_embedding_backward_is_scatter_add():
    """同一个 token 重复出现时，梯度累加到同一行。"""
    torch.manual_seed(0)
    mine = MyEmbedding(5, 3)
    idx = torch.tensor([1, 1, 1, 4])
    mine(idx).sum().backward()
    torch.testing.assert_close(mine.weight.grad[1], torch.full((3,), 3.0))
    torch.testing.assert_close(mine.weight.grad[4], torch.ones(3))
    torch.testing.assert_close(mine.weight.grad[0], torch.zeros(3))


# ---- MySoftmax ------------------------------------------------------------


def test_softmax_matches_torch_softmax():
    torch.manual_seed(0)
    mine = MySoftmax(dim=-1)
    x = torch.randn(3, 4, 7)
    torch.testing.assert_close(mine(x), torch.softmax(x, dim=-1))
    torch.testing.assert_close(mine(x).sum(-1), torch.ones(3, 4))


def test_softmax_stable_with_large_inputs():
    """输入含 1e4 时，朴素 exp(x)/sum(exp(x)) 会 inf/inf = NaN。"""
    x = torch.tensor([[1e4, 1e4 + 1.0, 0.0], [-1e4, -1e4, 1.0]])
    y = MySoftmax(dim=-1)(x)
    assert torch.isfinite(y).all(), "稳定版 softmax 不能出现 NaN/Inf"
    torch.testing.assert_close(y, torch.softmax(x, dim=-1))
    torch.testing.assert_close(y.sum(-1), torch.ones(2))

    naive = torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)
    assert torch.isnan(naive).any(), "朴素写法在这个输入上应该炸掉（反面教材）"


def test_softmax_other_dim():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    torch.testing.assert_close(MySoftmax(dim=0)(x), torch.softmax(x, dim=0))


# ---- conv2d / maxpool2d ---------------------------------------------------


def test_conv2d_unfold_matches_f_conv2d_basic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    torch.testing.assert_close(
        my_conv2d_unfold(x, w, b), F.conv2d(x, w, b), rtol=1e-5, atol=1e-5
    )


def test_conv2d_unfold_stride_padding():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 9, 7)
    w = torch.randn(5, 3, 3, 3)
    b = torch.randn(5)
    for stride, padding in [(1, 1), (2, 1), (2, 0), ((2, 1), (1, 2))]:
        mine = my_conv2d_unfold(x, w, b, stride=stride, padding=padding)
        ref = F.conv2d(x, w, b, stride=stride, padding=padding)
        assert mine.shape == ref.shape, f"shape 不符 stride={stride} pad={padding}"
        torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-5)


def test_conv2d_unfold_no_bias_and_nonsquare_kernel():
    torch.manual_seed(2)
    x = torch.randn(1, 2, 6, 10)
    w = torch.randn(3, 2, 3, 5)
    torch.testing.assert_close(
        my_conv2d_unfold(x, w, None, padding=(1, 2)),
        F.conv2d(x, w, None, padding=(1, 2)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv2d_unfold_backward_matches():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 6, 6, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)

    my_conv2d_unfold(x, w, padding=1).pow(2).sum().backward()
    F.conv2d(x2, w2, padding=1).pow(2).sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w.grad, w2.grad, rtol=1e-4, atol=1e-4)


def test_maxpool2d_matches_f_max_pool2d():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    for k, s in [(2, 2), (3, 1), (3, 2), ((2, 3), (2, 1))]:
        mine = MyMaxPool2d(k, s)(x)
        ref = F.max_pool2d(x, k, s)
        assert mine.shape == ref.shape
        torch.testing.assert_close(mine, ref)


def test_maxpool2d_padding_uses_neg_inf():
    """全负输入 + padding：补 0 会算错，补 -inf 才对。"""
    torch.manual_seed(0)
    x = -torch.rand(1, 2, 5, 5) - 1.0  # 全部 < -1
    mine = MyMaxPool2d(3, stride=2, padding=1)(x)
    ref = F.max_pool2d(x, 3, stride=2, padding=1)
    torch.testing.assert_close(mine, ref)
    assert (mine < 0).all(), "padding 补 0 的错误实现这里会出现 0"


def test_maxpool2d_backward_is_sparse():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, requires_grad=True)
    MyMaxPool2d(2, 2)(x).sum().backward()
    # 2x2 无重叠池化：4 个窗口各只有 1 个位置拿到梯度
    assert (x.grad != 0).sum().item() == 4
    torch.testing.assert_close(x.grad.sum(), torch.tensor(4.0))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
