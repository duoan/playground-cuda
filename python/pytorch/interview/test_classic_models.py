"""PyTorch 面试手写题（三）：经典模型从零实现。

覆盖三条主线：
  - MLP / ResNet BasicBlock：CV 基本功，考残差连接和 BN 的摆放位置；
  - LSTMCell / LSTM / GRUCell：RNN 基本功，考门控公式和权重布局；
  - Bahdanau / Luong Attention + Seq2Seq：Transformer 之前的 attention，
    考打分函数的两种范式和 mask 的正确用法。

RNN 部分全部与 torch 官方实现做权重级数值对齐 —— 这里藏着几个官方文档里
一句话带过、但手写时 100% 会踩的坑（GRU 的 r 门作用位置、LSTM 的门顺序、
双 bias 的存在），都写在各自的 docstring 里。
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 1. MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """可配置深度 / 激活 / dropout 的多层感知机。

    面试要点：
      1. **最后一层不加激活也不加 dropout**。分类头后面要接 CrossEntropyLoss，
         而它内部已经含 log_softmax；再加一次激活会把 logits 压扁，训练直接崩。
         这是「写个 MLP」这道送分题里最常见的失分点。
      2. **dropout 放在激活之后**。放在激活之前的话，被置零的位置经过 ReLU 仍是 0
         （恰好等价），但换成 GELU/Sigmoid 就不等价了 —— sigmoid(0)=0.5，
         等于给网络注入了一个常数偏置。所以统一放激活后。
      3. **nn.Sequential vs ModuleList**：Sequential 自带 forward，适合纯串联；
         需要跳连 / 多分支时必须用 ModuleList（它只负责注册参数，不定义 forward）。
         用 python list 存子模块是致命错误 —— 参数不会被 ``.parameters()`` 收集，
         优化器和 ``.to(device)`` 都会漏掉它们。
      4. 如果要加 BN/LN，顺序是 Linear -> Norm -> Act -> Dropout；且此时前一层
         Linear 的 bias 可以省掉（会被 Norm 的均值减法吃掉）。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))  # 输出层：不加激活、不加 dropout
        self.net = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "B I"]) -> Float[Tensor, "B O"]:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. ResNet BasicBlock
# ---------------------------------------------------------------------------


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    # bias=False：后面紧跟 BN，BN 的 beta 已经承担了 bias 的作用，
    # 再加一个 conv bias 是纯粹的冗余参数（且会被 BN 的减均值完全抵消）
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet-18/34 的残差块：conv-bn-relu-conv-bn 之后加 shortcut 再 relu。

    面试要点：
      1. **第二个 BN 之后、加法之前不能有 ReLU**。正确顺序是
         ``out = relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))``。
         如果写成 ``relu(bn2(...)) + shortcut``，主分支输出恒非负，残差就只能
         「加」不能「减」，表达能力被砍一半。这是本题最高频的错法。
      2. **shortcut 什么时候需要 downsample**：当 stride != 1（空间尺寸变了）
         或 in_planes != planes（通道数变了）时，恒等映射的 shape 对不上，
         必须用 1x1 conv（stride 相同）+ BN 把它投影过去。注意这个 1x1 conv
         也要配 BN，否则两条分支的数值尺度不匹配。
      3. **downsample 用 1x1 stride=2 会丢掉 3/4 的像素**（只采样左上角）。
         ResNet-D / bag-of-tricks 的改法是先 2x2 AvgPool 再 1x1 conv stride=1。
         能提这一点说明真读过论文。
      4. **为什么残差有效**：反向时 ``d(out)/dx = d(F)/dx + I``，那个 +I 保证了
         梯度至少能原样传回去，深层网络不会因为连乘小于 1 的雅可比而消失。
      5. BasicBlock 的 expansion=1；Bottleneck（ResNet-50+）是 1x1-3x3-1x1
         且 expansion=4，输出通道是 planes*4。
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: Float[Tensor, "N C H W"]) -> Float[Tensor, "N C2 H2 W2"]:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))  # 注意：这里没有 relu
        return F.relu(out + identity)


# ---------------------------------------------------------------------------
# 3. LSTMCell / LSTM
# ---------------------------------------------------------------------------


class MyLSTMCell(nn.Module):
    r"""单步 LSTM，与 nn.LSTMCell 权重布局完全一致。

    公式（gates 顺序必须是 i, f, g, o）：
        i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)   输入门：新信息写入多少
        f = sigmoid(W_if x + b_if + W_hf h + b_hf)   遗忘门：旧 cell 保留多少
        g = tanh   (W_ig x + b_ig + W_hg h + b_hg)   候选值（注意是 tanh 不是 sigmoid）
        o = sigmoid(W_io x + b_io + W_ho h + b_ho)   输出门
        c' = f * c + i * g
        h' = o * tanh(c')

    **为什么把 4 个门拼成一次 matmul 再 chunk？**
      1. 4 次 (B,I)@(I,H) 的小 GEMM 变成 1 次 (B,I)@(I,4H) 的大 GEMM。小矩阵乘
         是 memory-bound 的：x 要被反复从显存读 4 遍，而算术强度上不去。合并后
         x 只读一遍，算术强度翻 4 倍，A100 上通常能快 2~3 倍。
      2. kernel launch 从 4 次降到 1 次。RNN 要跑 T 个时间步，T=100 时就是省下
         几百次 launch —— 短序列小 batch 下 launch 开销甚至能占一半时间。
      3. 所以 nn.LSTMCell 的 weight_ih 形状是 **(4*H, I)**，行方向按 i,f,g,o
         顺序拼接。手写时把顺序记成 "ifgo"（IFGO），记错了权重能 load 进去但
         结果完全不对，且不报错。

    **为什么有两个 bias（b_ih 和 b_hh）？** 数学上 b_ih + b_hh 完全可以合并成一个，
    这是纯粹的历史包袱 —— 为了和 cuDNN 的 API 对齐（cuDNN 分别存输入侧和隐层侧
    的 bias）。手写时如果只用一个 bias，和官方对齐就会差一个常数。

    **工程细节**：遗忘门 bias 常被初始化为 1（``b_if = 1``），让训练初期
    cell state 默认「记住」，缓解长程梯度消失。PyTorch 默认没这么做，
    但 Jozefowicz et al. 2015 证明这个 trick 很有效。
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 官方 RNN 系列统一用 U(-1/sqrt(H), 1/sqrt(H)) 初始化所有参数
        std = 1.0 / (self.hidden_size**0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(
        self,
        x: Float[Tensor, "B I"],
        state: tuple[Float[Tensor, "B H"], Float[Tensor, "B H"]] | None = None,
    ) -> tuple[Float[Tensor, "B H"], Float[Tensor, "B H"]]:
        b = x.shape[0]
        if state is None:
            zeros = x.new_zeros(b, self.hidden_size)
            state = (zeros, zeros)
        h, c = state

        # 一次算完 4 个门：(B, I) @ (I, 4H) + (B, H) @ (H, 4H) -> (B, 4H)
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            h, self.weight_hh, self.bias_hh
        )
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MyLSTM(nn.Module):
    """单层单向 LSTM：用 MyLSTMCell 沿时间步展开。

    面试要点：
      1. **batch_first 的取舍**：内部循环按时间步切片，``x[t]`` 在 (T, B, I)
         布局下是连续的一整块内存，而 (B, T, I) 下每个时间步是跨 stride 的。
         这就是官方默认 ``batch_first=False`` 的原因 —— 不是为了反人类，是为了
         让每步的 GEMM 拿到连续输入。实现上先 transpose 成 (T,B,I) 再循环，
         比在循环里反复做非连续索引快。
      2. **时间步之间有严格串行依赖**，这是 RNN 打不过 Transformer 的根本原因：
         T 步无法并行，GPU 利用率被时间维锁死。cuDNN 的优化手段是把「输入侧
         的 W_ih @ x」在所有时间步上一次性算完（它不依赖 h），只让 W_hh @ h
         留在循环里。这是一个很好的追问点。
      3. **返回值约定**：output 是所有时间步的 h（(T,B,H) 或 (B,T,H)），
         h_n/c_n 是最后一步的状态且带 num_layers 维 ——
         **形状是 (num_layers*num_directions, B, H)，不受 batch_first 影响**。
         这个不一致性坑过无数人。
      4. 双向 LSTM 不是「反着再跑一遍就完事」：反向的 h_n 取的是**序列起点**
         那一步的状态，且变长序列必须配 pack_padded_sequence，否则反向会从
         padding 开始跑。
    """

    def __init__(
        self, input_size: int, hidden_size: int, batch_first: bool = False
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = MyLSTMCell(input_size, hidden_size)

    def forward(
        self,
        x: Float[Tensor, "B T I"],
        state: tuple[Float[Tensor, "1 B H"], Float[Tensor, "1 B H"]] | None = None,
    ) -> tuple[
        Float[Tensor, "B T H"], tuple[Float[Tensor, "1 B H"], Float[Tensor, "1 B H"]]
    ]:
        if self.batch_first:
            x = x.transpose(0, 1)  # (B, T, I) -> (T, B, I)
        t, b, _ = x.shape

        if state is None:
            h = x.new_zeros(b, self.hidden_size)
            c = x.new_zeros(b, self.hidden_size)
        else:
            h, c = state[0][0], state[1][0]  # 去掉 num_layers 维

        outputs = []
        for step in range(t):
            h, c = self.cell(x[step], (h, c))
            outputs.append(h)
        output = torch.stack(outputs, dim=0)  # (T, B, H)

        if self.batch_first:
            output = output.transpose(0, 1)
        # h_n / c_n 始终是 (num_layers, B, H)，与 batch_first 无关
        return output, (h.unsqueeze(0), c.unsqueeze(0))


# ---------------------------------------------------------------------------
# 4. GRUCell
# ---------------------------------------------------------------------------


class MyGRUCell(nn.Module):
    r"""单步 GRU，与 nn.GRUCell 权重布局完全一致。

    公式（gates 顺序必须是 r, z, n）：
        r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)      重置门
        z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)      更新门
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))   候选状态  <<< 看清楚
        h' = (1 - z) * n + z * h

    **本题唯一也是最大的坑：r 到底乘在哪里？**
      原论文（Cho et al. 2014）写的是 ``n = tanh(W_in x + W_hn (r * h))``，
      也就是先把 r 作用在 h 上再做矩阵乘。但 PyTorch/cuDNN 实现的是
      ``n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`` —— **r 乘在矩阵乘之后，
      而且连 b_hn 一起乘进去了**。
      为什么？因为这样 ``W_hn h + b_hn`` 可以和 r、z 两个门的隐层项合并成
      **一次 (B,H)@(H,3H) 的 GEMM**；如果按原论文写，n 分支要等 r 算出来才能
      做它自己的矩阵乘，被迫拆成两次 GEMM 且串行。
      数学上两者不等价（因为 r 是逐元素的，r*(Wh) != W(r*h)），效果差异很小，
      但**做权重对齐时会直接对不上**。这是 GRU 手写题的标准答案分水岭。

    另外两点：
      1. ``h' = (1-z)*n + z*h`` —— PyTorch 里 z 是「保留旧状态的比例」。有些
         教材/框架写成 ``z*n + (1-z)*h``，语义正好相反，权重能加载但行为不同。
      2. GRU 只有 2 个门、3 组权重，参数量是 LSTM 的 3/4，且没有独立的 cell
         state。短序列上和 LSTM 打平，超长依赖上通常略逊。
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        std = 1.0 / (hidden_size**0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(
        self, x: Float[Tensor, "B I"], h: Float[Tensor, "B H"] | None = None
    ) -> Float[Tensor, "B H"]:
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)

        # 输入侧和隐层侧各一次 GEMM，之后再按门拆开
        gi = F.linear(x, self.weight_ih, self.bias_ih)  # (B, 3H)
        gh = F.linear(h, self.weight_hh, self.bias_hh)  # (B, 3H)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)  # 关键：r 乘的是 (W_hn h + b_hn) 整体
        return (1 - z) * n + z * h


# ---------------------------------------------------------------------------
# 5. Bahdanau / Luong Attention
# ---------------------------------------------------------------------------


class BahdanauAttention(nn.Module):
    r"""加性（additive / concat）注意力，Bahdanau et al. 2015。

        score(q, k) = v^T * tanh(W_q q + W_k k)

    面试要点：
      1. **加性 vs 乘性**：加性把 q 和 k 投到一个公共的 attn_dim 再过 tanh 打分，
         所以 **q 和 k 的维度可以不同**（encoder 双向 2H、decoder 单向 H 时特别
         方便），代价是要显式构造 (B, T_q, T_k, attn_dim) 的中间张量，
         显存和算力都是乘性的 attn_dim 倍。
      2. **d_k 较大时加性反而更稳**：点积的方差随 d_k 线性增长，不除
         sqrt(d_k) 的话 softmax 会饱和；加性因为有 tanh 天然被压在 [-1,1]，
         没有这个问题。Transformer 选了乘性 + 1/sqrt(d_k) 缩放，是因为乘性能
         直接调用高度优化的 matmul —— 论文原话就是「乘性在实践中快得多且省显存」。
      3. **mask 用 -inf 而不是 0**：要在 **softmax 之前** 把无效位置的 score
         设成 -inf，softmax 之后自然是 0。如果 softmax 之后再乘 0，剩下的权重
         就不再和为 1 了。
      4. **-inf 的陷阱**：如果某一行被全部 mask 掉（比如一个长度为 0 的样本），
         softmax 得到 0/0 = NaN。生产代码要么保证每行至少一个有效位置，
         要么用 ``torch.finfo(dtype).min`` 代替 -inf（结果是均匀分布而非 NaN）。
    """

    def __init__(self, query_dim: int, key_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        query: Float[Tensor, "B Tq Dq"],
        keys: Float[Tensor, "B Tk Dk"],
        values: Float[Tensor, "B Tk Dv"] | None = None,
        mask: Bool[Tensor, "B Tk"] | None = None,
    ) -> tuple[Float[Tensor, "B Tq Dv"], Float[Tensor, "B Tq Tk"]]:
        """mask 中 True 表示有效位置（和 nn.MultiheadAttention 的约定相反，注意）。"""
        if values is None:
            values = keys
        # (B, Tq, 1, A) + (B, 1, Tk, A) -> (B, Tq, Tk, A)
        feats = torch.tanh(self.w_q(query).unsqueeze(2) + self.w_k(keys).unsqueeze(1))
        scores = self.v(feats).squeeze(-1)  # (B, Tq, Tk)
        return _attend(scores, values, mask)


class LuongAttention(nn.Module):
    r"""乘性（multiplicative / general）注意力，Luong et al. 2015。

        score(q, k) = q^T W k        （general 形式）
        score(q, k) = q^T k          （dot 形式，要求 dim 相同）

    面试要点：
      1. **W 的作用是维度桥接 + 学一个双线性度量**。q 和 k 维度相同时可以退化成
         纯点积（``W = I``），但保留 W 通常更好 —— 它允许模型在一个学出来的
         子空间里比较相似度，而不是在原始特征空间。
      2. **实现只需两次 batched matmul**：``(q @ W) @ k.transpose(-1,-2)``，
         中间张量只有 (B, Tq, Tk)，比加性省一个 attn_dim 维度。这就是
         Transformer 采用它的原因。
      3. **Transformer 的 scaled dot-product 就是 dot 形式 + 1/sqrt(d_k)**。
         缩放的推导：q、k 各分量独立、均值 0 方差 1 时，q·k 的方差是 d_k，
         除以 sqrt(d_k) 才把方差拉回 1，避免 softmax 落入梯度接近 0 的饱和区。
         Luong 原文没有这一项 —— 因为当年 d_k 只有几百且用了 tanh 压过的 RNN
         输出，问题不明显。
      4. Luong 的另一个贡献是 **input feeding**：把上一步的 attention 输出拼回
         decoder 输入，让模型知道「上一步已经看过哪里了」。
    """

    def __init__(self, query_dim: int, key_dim: int, scaled: bool = False) -> None:
        super().__init__()
        self.w = nn.Linear(query_dim, key_dim, bias=False)
        self.scaled = scaled
        self.key_dim = key_dim

    def forward(
        self,
        query: Float[Tensor, "B Tq Dq"],
        keys: Float[Tensor, "B Tk Dk"],
        values: Float[Tensor, "B Tk Dv"] | None = None,
        mask: Bool[Tensor, "B Tk"] | None = None,
    ) -> tuple[Float[Tensor, "B Tq Dv"], Float[Tensor, "B Tq Tk"]]:
        if values is None:
            values = keys
        scores = self.w(query) @ keys.transpose(-1, -2)  # (B, Tq, Tk)
        if self.scaled:
            scores = scores / self.key_dim**0.5
        return _attend(scores, values, mask)


def _attend(
    scores: Float[Tensor, "B Tq Tk"],
    values: Float[Tensor, "B Tk Dv"],
    mask: Bool[Tensor, "B Tk"] | None,
) -> tuple[Float[Tensor, "B Tq Dv"], Float[Tensor, "B Tq Tk"]]:
    """两种 attention 共享的收尾：mask -> softmax -> 加权求和。"""
    if mask is not None:
        # (B, Tk) -> (B, 1, Tk)，对所有 query 位置广播
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = weights @ values  # (B, Tq, Tk) @ (B, Tk, Dv)
    return context, weights


# ---------------------------------------------------------------------------
# 6. Seq2Seq with attention
# ---------------------------------------------------------------------------


class Seq2Seq(nn.Module):
    """encoder LSTM + attention + decoder LSTMCell 的最小完整实现。

    面试要点：
      1. **decoder 必须逐步展开**，不能像 encoder 那样一把跑完：每一步的
         attention query 依赖上一步的 decoder hidden state。这正是
         Transformer decoder 能并行训练（用 causal mask 一次算完）而 RNN
         decoder 不能的关键区别。
      2. **teacher forcing**：训练时喂真实的上一个 token，推理时喂模型自己的
         预测。两者分布不一致就是 exposure bias，scheduled sampling 是常见缓解手段。
         这里 forward 走的是 teacher forcing 路径。
      3. **encoder 的最后状态作为 decoder 初始状态**是「桥接」的一种；有 attention
         之后其实可以直接用零初始化，因为信息主要通过 attention 流过去 ——
         attention 机制诞生的动机就是「不要把整个源句压进一个固定长度向量」。
      4. **context 怎么用**：这里把 context 和 decoder hidden 拼接后再过输出层
         （Luong 的做法）。Bahdanau 的做法是把 context 拼进下一步的 decoder 输入。
      5. **mask 必须传到 attention**：变长 batch 里 padding 位置如果参与 softmax，
         模型会学会去看 padding，验证集上表现和训练时对不上。
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.src_emb = nn.Embedding(src_vocab, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb_dim)
        self.encoder = MyLSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder_cell = MyLSTMCell(emb_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, hidden_dim)
        # 输出层吃 [decoder_hidden ; context] 的拼接
        self.out = nn.Linear(hidden_dim * 2, tgt_vocab)

    def forward(
        self,
        src: Float[Tensor, "B Ts"],
        tgt: Float[Tensor, "B Tt"],
        src_mask: Bool[Tensor, "B Ts"] | None = None,
    ) -> tuple[Float[Tensor, "B Tt V"], Float[Tensor, "B Tt Ts"]]:
        enc_out, (h_n, c_n) = self.encoder(self.src_emb(src))  # (B, Ts, H)
        h, c = h_n[0], c_n[0]

        tgt_emb = self.tgt_emb(tgt)  # (B, Tt, E)
        logits, attns = [], []
        for t in range(tgt.shape[1]):
            h, c = self.decoder_cell(tgt_emb[:, t], (h, c))
            # query 是当前 decoder hidden，(B, 1, H)
            context, attn = self.attention(h.unsqueeze(1), enc_out, mask=src_mask)
            logits.append(self.out(torch.cat([h, context.squeeze(1)], dim=-1)))
            attns.append(attn.squeeze(1))
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)


# ===========================================================================
#                                  tests
# ===========================================================================


# ---- MLP ------------------------------------------------------------------


def test_mlp_forward_shape():
    torch.manual_seed(0)
    m = MLP(8, [16, 12], 3)
    assert m(torch.randn(5, 8)).shape == (5, 3)


def test_mlp_grad_reaches_all_params():
    torch.manual_seed(0)
    m = MLP(8, [16, 12], 3, dropout=0.1)
    m(torch.randn(5, 8)).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"


def test_mlp_last_layer_has_no_activation():
    """输出层后面不能有激活：否则 logits 被压扁，接 CrossEntropyLoss 会崩。"""
    m = MLP(4, [8], 3)
    assert isinstance(m.net[-1], nn.Linear)
    # 输出必须能取到负值（有激活的话 ReLU 后恒非负）
    torch.manual_seed(0)
    assert (m(torch.randn(64, 4)) < 0).any()


def test_mlp_dropout_placement_and_depth():
    m = MLP(4, [8, 8], 2, activation=nn.GELU, dropout=0.5)
    kinds = [type(x).__name__ for x in m.net]
    assert kinds == [
        "Linear", "GELU", "Dropout",
        "Linear", "GELU", "Dropout",
        "Linear",
    ]


# ---- BasicBlock -----------------------------------------------------------


def test_basicblock_identity_branch_shape():
    torch.manual_seed(0)
    blk = BasicBlock(8, 8, stride=1).eval()
    x = torch.randn(2, 8, 16, 16)
    assert blk.downsample is None, "in==out 且 stride=1 时应走恒等 shortcut"
    assert blk(x).shape == x.shape


def test_basicblock_downsample_branch_shape():
    torch.manual_seed(0)
    blk = BasicBlock(8, 16, stride=2).eval()
    y = blk(torch.randn(2, 8, 16, 16))
    assert blk.downsample is not None, "通道或 stride 变化时必须有 downsample"
    assert y.shape == (2, 16, 8, 8)


def test_basicblock_is_residual():
    """把两个 conv 的权重置零，输出应精确等于 relu(shortcut)（这里就是 relu(x)）。"""
    torch.manual_seed(0)
    blk = BasicBlock(4, 4).eval()
    with torch.no_grad():
        blk.conv1.weight.zero_()
        blk.conv2.weight.zero_()
        blk.bn1.running_mean.zero_(), blk.bn1.running_var.fill_(1.0)
        blk.bn2.running_mean.zero_(), blk.bn2.running_var.fill_(1.0)
    x = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(blk(x), F.relu(x))


def test_basicblock_no_relu_before_addition():
    """主分支在加法前不过 relu，所以 bn2 的输出必须能取到负值。"""
    torch.manual_seed(0)
    blk = BasicBlock(4, 4).eval()
    x = torch.randn(4, 4, 8, 8)
    branch = blk.bn2(blk.conv2(F.relu(blk.bn1(blk.conv1(x)))))
    assert (branch < 0).any(), "残差主分支应保留负值（能做减法）"


def test_basicblock_conv_has_no_bias():
    blk = BasicBlock(4, 8, stride=2)
    assert blk.conv1.bias is None and blk.conv2.bias is None
    assert blk.downsample[0].bias is None


def test_basicblock_backward():
    torch.manual_seed(0)
    blk = BasicBlock(4, 8, stride=2)
    blk(torch.randn(2, 4, 8, 8)).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"


# ---- MyLSTMCell -----------------------------------------------------------


def test_lstmcell_matches_nn_lstmcell():
    torch.manual_seed(0)
    i_dim, h_dim, b = 6, 5, 3
    ref = nn.LSTMCell(i_dim, h_dim)
    mine = MyLSTMCell(i_dim, h_dim)
    with torch.no_grad():
        mine.load_state_dict(ref.state_dict())

    x = torch.randn(b, i_dim)
    h0, c0 = torch.randn(b, h_dim), torch.randn(b, h_dim)
    mh, mc = mine(x, (h0, c0))
    rh, rc = ref(x, (h0, c0))
    torch.testing.assert_close(mh, rh)
    torch.testing.assert_close(mc, rc)


def test_lstmcell_zero_init_state():
    torch.manual_seed(0)
    ref = nn.LSTMCell(4, 3)
    mine = MyLSTMCell(4, 3)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(2, 4)
    for m, r in zip(mine(x), ref(x)):
        torch.testing.assert_close(m, r)


def test_lstmcell_weight_layout_is_4h():
    mine = MyLSTMCell(6, 5)
    assert mine.weight_ih.shape == (20, 6) and mine.weight_hh.shape == (20, 5)
    assert mine.bias_ih.shape == (20,) and mine.bias_hh.shape == (20,)


def test_lstmcell_gate_order_is_ifgo():
    """把 i 门权重打满、其余置零，验证 chunk 出来的第 0 块确实是输入门。

    做法：让 i=1、f=0、g=1、o=1，则 c' = 0*c + 1*1 = 1，h' = tanh(1)。
    如果门顺序记错（比如 fifo），结果不会是这个值。
    """
    h_dim = 4
    cell = MyLSTMCell(2, h_dim, bias=True)
    big = 20.0  # sigmoid(20) ≈ 1, sigmoid(-20) ≈ 0
    with torch.no_grad():
        cell.weight_ih.zero_()
        cell.weight_hh.zero_()
        cell.bias_hh.zero_()
        # bias_ih 按 [i, f, g, o] 分四段
        cell.bias_ih.copy_(
            torch.cat([
                torch.full((h_dim,), big),    # i -> 1
                torch.full((h_dim,), -big),   # f -> 0
                torch.full((h_dim,), big),    # g -> tanh(20) ≈ 1
                torch.full((h_dim,), big),    # o -> 1
            ])
        )
    h, c = cell(torch.zeros(1, 2), (torch.ones(1, h_dim), torch.full((1, h_dim), 9.0)))
    torch.testing.assert_close(c, torch.ones(1, h_dim), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        h, torch.full((1, h_dim), float(torch.tanh(torch.tensor(1.0)))),
        rtol=1e-4, atol=1e-4,
    )


# ---- MyLSTM ---------------------------------------------------------------


def test_lstm_matches_nn_lstm_time_first():
    torch.manual_seed(0)
    i_dim, h_dim, t, b = 5, 4, 7, 3
    ref = nn.LSTM(i_dim, h_dim, num_layers=1, batch_first=False)
    mine = MyLSTM(i_dim, h_dim, batch_first=False)
    with torch.no_grad():
        mine.cell.weight_ih.copy_(ref.weight_ih_l0)
        mine.cell.weight_hh.copy_(ref.weight_hh_l0)
        mine.cell.bias_ih.copy_(ref.bias_ih_l0)
        mine.cell.bias_hh.copy_(ref.bias_hh_l0)

    x = torch.randn(t, b, i_dim)
    my_out, (my_h, my_c) = mine(x)
    ref_out, (ref_h, ref_c) = ref(x)
    torch.testing.assert_close(my_out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(my_h, ref_h, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(my_c, ref_c, rtol=1e-5, atol=1e-6)


def test_lstm_matches_nn_lstm_batch_first():
    torch.manual_seed(1)
    i_dim, h_dim, t, b = 5, 4, 6, 2
    ref = nn.LSTM(i_dim, h_dim, batch_first=True)
    mine = MyLSTM(i_dim, h_dim, batch_first=True)
    with torch.no_grad():
        mine.cell.weight_ih.copy_(ref.weight_ih_l0)
        mine.cell.weight_hh.copy_(ref.weight_hh_l0)
        mine.cell.bias_ih.copy_(ref.bias_ih_l0)
        mine.cell.bias_hh.copy_(ref.bias_hh_l0)

    x = torch.randn(b, t, i_dim)
    h0 = torch.randn(1, b, h_dim)
    c0 = torch.randn(1, b, h_dim)
    my_out, (my_h, my_c) = mine(x, (h0, c0))
    ref_out, (ref_h, ref_c) = ref(x, (h0, c0))

    assert my_out.shape == (b, t, h_dim)
    # h_n 的形状是 (num_layers, B, H)，不受 batch_first 影响
    assert my_h.shape == (1, b, h_dim)
    torch.testing.assert_close(my_out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(my_h, ref_h, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(my_c, ref_c, rtol=1e-5, atol=1e-6)


def test_lstm_last_output_equals_h_n():
    torch.manual_seed(0)
    mine = MyLSTM(4, 3, batch_first=True)
    out, (h_n, _) = mine(torch.randn(2, 5, 4))
    torch.testing.assert_close(out[:, -1], h_n[0])


# ---- MyGRUCell ------------------------------------------------------------


def test_grucell_matches_nn_grucell():
    torch.manual_seed(0)
    ref = nn.GRUCell(6, 5)
    mine = MyGRUCell(6, 5)
    mine.load_state_dict(ref.state_dict())

    x = torch.randn(3, 6)
    h0 = torch.randn(3, 5)
    torch.testing.assert_close(mine(x, h0), ref(x, h0))
    torch.testing.assert_close(mine(x), ref(x))


def test_grucell_reset_gate_placement_differs_from_paper():
    """考点实证：r*(W_hn h + b_hn) 与论文的 W_hn(r*h) 数值上不等价。"""
    torch.manual_seed(0)
    h_dim = 5
    ref = nn.GRUCell(6, h_dim)
    x, h = torch.randn(3, 6), torch.randn(3, h_dim)

    gi = F.linear(x, ref.weight_ih, ref.bias_ih)
    gh = F.linear(h, ref.weight_hh, ref.bias_hh)
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)

    pytorch_style = torch.tanh(i_n + r * h_n)
    w_hn = ref.weight_hh[2 * h_dim :]
    b_hn = ref.bias_hh[2 * h_dim :]
    paper_style = torch.tanh(i_n + F.linear(r * h, w_hn, b_hn))

    torch.testing.assert_close((1 - z) * pytorch_style + z * h, ref(x, h))
    assert not torch.allclose((1 - z) * paper_style + z * h, ref(x, h), atol=1e-4)


def test_grucell_update_gate_direction():
    """PyTorch 的 z 是「保留旧状态」的比例：z->1 时 h' == h。"""
    h_dim = 4
    cell = MyGRUCell(3, h_dim)
    with torch.no_grad():
        cell.weight_ih.zero_()
        cell.weight_hh.zero_()
        cell.bias_hh.zero_()
        cell.bias_ih.zero_()
        cell.bias_ih[h_dim : 2 * h_dim] = 30.0  # z -> 1
    h = torch.randn(2, h_dim)
    torch.testing.assert_close(cell(torch.zeros(2, 3), h), h, rtol=1e-5, atol=1e-5)


def test_grucell_weight_layout_is_3h():
    mine = MyGRUCell(6, 5)
    assert mine.weight_ih.shape == (15, 6) and mine.weight_hh.shape == (15, 5)


# ---- Attention ------------------------------------------------------------


def test_bahdanau_shapes_and_weights_sum_to_one():
    torch.manual_seed(0)
    b, tq, tk, dq, dk = 2, 3, 5, 6, 8
    attn = BahdanauAttention(dq, dk, 7)
    ctx, w = attn(torch.randn(b, tq, dq), torch.randn(b, tk, dk))
    assert ctx.shape == (b, tq, dk) and w.shape == (b, tq, tk)
    torch.testing.assert_close(w.sum(-1), torch.ones(b, tq))


def test_luong_shapes_and_weights_sum_to_one():
    torch.manual_seed(0)
    b, tq, tk, dq, dk = 2, 3, 5, 6, 8
    attn = LuongAttention(dq, dk)
    ctx, w = attn(torch.randn(b, tq, dq), torch.randn(b, tk, dk))
    assert ctx.shape == (b, tq, dk) and w.shape == (b, tq, tk)
    torch.testing.assert_close(w.sum(-1), torch.ones(b, tq))


def test_attention_mask_zeroes_invalid_positions():
    torch.manual_seed(0)
    b, tq, tk, d = 2, 3, 5, 4
    mask = torch.ones(b, tk, dtype=torch.bool)
    mask[0, 3:] = False  # 第 0 条样本只有前 3 个位置有效
    mask[1, 4:] = False

    for attn in [BahdanauAttention(d, d, 6), LuongAttention(d, d)]:
        _, w = attn(torch.randn(b, tq, d), torch.randn(b, tk, d), mask=mask)
        torch.testing.assert_close(w[0, :, 3:], torch.zeros(tq, 2))
        torch.testing.assert_close(w[1, :, 4:], torch.zeros(tq, 1))
        # mask 之后剩下的权重仍然和为 1
        torch.testing.assert_close(w.sum(-1), torch.ones(b, tq))


def test_attention_mask_does_not_leak_into_context():
    """padding 位置的 value 换成任意大数，context 不应改变。"""
    torch.manual_seed(0)
    b, tk, d = 2, 5, 4
    attn = LuongAttention(d, d)
    q = torch.randn(b, 1, d)
    k = torch.randn(b, tk, d)
    v = torch.randn(b, tk, d)
    mask = torch.ones(b, tk, dtype=torch.bool)
    mask[:, 3:] = False

    v2 = v.clone()
    v2[:, 3:] = 1e4
    ctx1, _ = attn(q, k, v, mask=mask)
    ctx2, _ = attn(q, k, v2, mask=mask)
    torch.testing.assert_close(ctx1, ctx2)


def test_luong_scaled_matches_manual_dot_product():
    torch.manual_seed(0)
    d = 8
    attn = LuongAttention(d, d, scaled=True)
    q, k = torch.randn(2, 3, d), torch.randn(2, 5, d)
    _, w = attn(q, k)
    expected = torch.softmax((attn.w(q) @ k.transpose(-1, -2)) / d**0.5, dim=-1)
    torch.testing.assert_close(w, expected)


def test_bahdanau_supports_mismatched_dims():
    """加性注意力的卖点：query 和 key 维度可以不同。"""
    torch.manual_seed(0)
    attn = BahdanauAttention(query_dim=5, key_dim=11, attn_dim=7)
    ctx, w = attn(torch.randn(2, 3, 5), torch.randn(2, 4, 11))
    assert ctx.shape == (2, 3, 11) and w.shape == (2, 3, 4)


def test_attention_fully_masked_row_produces_nan():
    """反面教材：整行被 mask 掉时 -inf 会导致 softmax 得到 NaN。"""
    torch.manual_seed(0)
    attn = LuongAttention(4, 4)
    mask = torch.zeros(1, 5, dtype=torch.bool)  # 全部无效
    _, w = attn(torch.randn(1, 2, 4), torch.randn(1, 5, 4), mask=mask)
    assert torch.isnan(w).all(), "全 mask 行应当暴露出 NaN（生产代码要显式规避）"


# ---- Seq2Seq --------------------------------------------------------------


def test_seq2seq_forward_shape():
    torch.manual_seed(0)
    sv, tv, b, ts, tt = 20, 15, 3, 6, 4
    model = Seq2Seq(sv, tv, emb_dim=8, hidden_dim=10)
    src = torch.randint(0, sv, (b, ts))
    tgt = torch.randint(0, tv, (b, tt))

    logits, attns = model(src, tgt)
    assert logits.shape == (b, tt, tv)
    assert attns.shape == (b, tt, ts)
    torch.testing.assert_close(attns.sum(-1), torch.ones(b, tt))


def test_seq2seq_respects_src_mask():
    torch.manual_seed(0)
    sv, tv, b, ts, tt = 20, 15, 2, 6, 3
    model = Seq2Seq(sv, tv, emb_dim=8, hidden_dim=10)
    src = torch.randint(0, sv, (b, ts))
    tgt = torch.randint(0, tv, (b, tt))
    mask = torch.ones(b, ts, dtype=torch.bool)
    mask[:, 4:] = False

    _, attns = model(src, tgt, src_mask=mask)
    torch.testing.assert_close(attns[:, :, 4:], torch.zeros(b, tt, 2))


def test_seq2seq_backward_reaches_all_params():
    torch.manual_seed(0)
    sv, tv = 20, 15
    model = Seq2Seq(sv, tv, emb_dim=8, hidden_dim=10)
    src = torch.randint(0, sv, (3, 6))
    tgt = torch.randint(0, tv, (3, 4))
    logits, _ = model(src, tgt)
    F.cross_entropy(logits.reshape(-1, tv), tgt.reshape(-1)).backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
    # embedding 只有被用到的行有梯度，其余为 0；至少要有非零行
    assert model.src_emb.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
