"""PyTorch 面试手写题：Transformer 全家桶。

覆盖面试里出现频率最高的一组题：
  1. scaled dot-product attention（缩放、mask、dropout 的细节）
  2. MultiHeadAttention（reshape/transpose 顺序这个经典 bug）
  3. GroupedQueryAttention / MQA（KV cache 显存优化）
  4. RoPE（旋转位置编码的两条核心性质）
  5. causal mask + padding mask 的组合
  6. KVCache（预分配 buffer + 游标）
  7. TransformerBlock（pre-norm + RMSNorm + SwiGLU）
  8. MiniGPT + generate（greedy / temperature / top-k / top-p）

每个实现都尽量和官方 API 对齐并写成测试，因为"能证明自己写的和
`F.scaled_dot_product_attention` / `nn.MultiheadAttention` 数值一致"
本身就是面试里最有说服力的答案。

Mask 约定（全文件统一）:
    bool mask 里 **True 表示"可见/保留"**，False 表示"屏蔽"。
    这和 `F.scaled_dot_product_attention` 一致；
    但和 `nn.MultiheadAttention` 相反（后者 True = 屏蔽）。
    这个不一致是 PyTorch 的历史包袱，也是很好的面试考点，见
    `test_mha_matches_torch_causal` 里的注释。
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

# =============================================================================
# 1. Scaled Dot-Product Attention
# =============================================================================


def scaled_dot_product_attention(
    query: Float[Tensor, "B H Sq D"],
    key: Float[Tensor, "B Hkv Sk D"],
    value: Float[Tensor, "B Hkv Sk Dv"],
    attn_mask: Bool[Tensor, "*broadcast Sq Sk"] | Float[Tensor, "*broadcast Sq Sk"] | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
) -> Float[Tensor, "B H Sq Dv"]:
    r"""手写 scaled dot-product attention。

    公式::

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    面试要点 1：**为什么要除以 sqrt(d_k)？**
        设 q、k 的各分量独立同分布、均值 0 方差 1，则
        ``q · k = sum_{i=1}^{d_k} q_i k_i`` 的方差是 ``d_k``（标准差 ``sqrt(d_k)``）。
        d_k 越大，logits 的动态范围越宽，softmax 越接近 one-hot（饱和）。
        而 softmax 饱和时雅可比 ``diag(p) - p p^T`` 趋近于 0，**梯度消失**，
        训练早期就学不动。除以 ``sqrt(d_k)`` 把 logits 方差拉回 1，
        使 softmax 停留在梯度良好的区域。
        常见错法：除以 ``d_k`` 或者 ``sqrt(d_model)`` —— 应该是**每个 head 的**维度。

    面试要点 2：**mask 为什么用 -inf 而不是 -1e9？**
        - fp32 下两者几乎等价（exp(-1e9) 下溢成 0）。
        - **fp16 下 -1e9 直接溢出成 -inf**，看似没事；但如果实现里写的是
          ``logits + (-1e9)``，而 logits 本身有较大正值，在 fp16 里
          ``-1e9`` 已经是 ``-inf``，加法结果仍是 -inf，勉强能用；
          真正的坑是有人写 ``-1e4``（fp16 可表示的安全值），此时若真实 logits
          也在 1e4 量级，masked 位置**不会被完全屏蔽**，信息泄漏。
        - 用 ``torch.finfo(dtype).min`` 或直接 ``float("-inf")`` 最稳。
          本实现用 ``-inf``，因为 masked_fill 后 softmax 会精确给出 0。
        - 唯一的坑：某一行**全部**被 mask 掉时，softmax(全 -inf) = NaN。
          padding mask 遇到"整条序列都是 pad"的样本就会踩到，
          工程上要么保证至少一个位置可见，要么事后 nan_to_num。

    面试要点 3：**dropout 加在哪？**
        加在 softmax **之后**、乘 V **之前**（即 drop 掉部分 attention 权重）。
        PyTorch 的 inverted dropout 会除以 (1-p)，所以推理时什么都不用做。
        常见错法：加在 logits 上（等价于随机屏蔽，语义完全不同）。

    Args:
        attn_mask: bool 型时 **True = 可见**；float 型时按**加性 bias** 处理
            （屏蔽位置填 -inf）。与 ``F.scaled_dot_product_attention`` 同义。
        is_causal: 为 True 时内部生成下三角 causal mask，不能与 attn_mask 同用。

    Returns:
        (输出, attention 权重)。返回权重是为了方便测试断言，
        真实实现（FlashAttention）不会物化这个 (Sq, Sk) 矩阵。
    """
    d_k = query.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # (B, H, Sq, D) @ (B, H, D, Sk) -> (B, H, Sq, Sk)
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        assert attn_mask is None, "is_causal 与 attn_mask 只能二选一"
        s_q, s_k = query.shape[-2], key.shape[-2]
        # 注意 start_pos: 当 Sq < Sk（decode 阶段）时，query 对齐到序列末尾
        attn_mask = causal_mask(s_q, s_k, start_pos=s_k - s_q, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask

    attn = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)

    out = torch.matmul(attn, value)
    return out, attn


def causal_mask(
    q_len: int,
    kv_len: int | None = None,
    start_pos: int = 0,
    device: torch.device | None = None,
) -> Bool[Tensor, "Sq Sk"]:
    """生成 causal（下三角）mask，**True = 可见**。

    ``causal_mask(S)`` 就是最经典的 (S, S) 下三角全 True 矩阵。

    面试要点：**decode 阶段的 mask 不是方阵。**
        用 KV cache 逐 token 解码时 q_len=1、kv_len=已缓存长度，
        此时 query 的绝对位置是 ``start_pos``，它能看到 **全部** 历史 key，
        所以 mask 应该是全 True 的 (1, kv_len)。
        很多人直接套 ``tril(ones(q_len, kv_len))`` 得到只看第 0 个 key 的
        错误 mask —— 这是 KV cache 实现里最常见的 bug 之一。
        这里用 ``q_pos >= k_pos`` 的通式覆盖所有情况。
    """
    if kv_len is None:
        kv_len = q_len
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + start_pos  # (Sq, 1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, Sk)
    return q_pos >= k_pos


def padding_mask(
    lengths: Int[Tensor, " B"], max_len: int, device: torch.device | None = None
) -> Bool[Tensor, "B 1 1 Sk"]:
    """由每个样本的真实长度生成 key padding mask，**True = 可见**。

    形状故意做成 ``(B, 1, 1, Sk)``，这样能直接和 ``(1, 1, Sq, Sk)`` 的
    causal mask 用 ``&`` 广播相与——这是"两种 mask 怎么组合"的标准答案：
    **逻辑与**（都可见才可见），而不是相加。
    """
    ar = torch.arange(max_len, device=device)
    return (ar.unsqueeze(0) < lengths.to(device).unsqueeze(1)).view(-1, 1, 1, max_len)


# =============================================================================
# 2. Multi-Head Attention
# =============================================================================


class MultiHeadAttention(nn.Module):
    """标准 MHA：fused qkv Linear -> split heads -> attention -> merge -> out proj。

    面试要点：**reshape / transpose 的顺序为什么不能写反？**
        Linear 输出是 ``(B, S, 3*d_model)``，切出来的 q 是 ``(B, S, d_model)``。
        内存里最后一维的排布是 ``[head0_d0..head0_dh-1, head1_d0, ...]``，
        也就是 **head 维在 d_head 维的外侧**。所以正确写法是::

            q.view(B, S, n_head, d_head).transpose(1, 2)   # -> (B, n_head, S, d_head)

        经典错法是::

            q.view(B, n_head, S, d_head)                    # 悄悄错，shape 却对！

        后者把 ``S * d_model`` 这段连续内存重新切成 ``n_head * (S * d_head)``，
        等价于"第 0 个 head 拿走了前 S/n_head 个 token 的全部通道"，
        语义完全乱掉。**它不会报错、shape 完全正确、loss 也会下降一点**，
        所以极难发现——这也是它成为经典面试题的原因。
        merge 回去时同理，必须 ``transpose(1, 2).contiguous().view(B, S, d_model)``，
        少了 contiguous 会因为 stride 不连续而 view 报错（这也是考点）。

    面试要点：**fused qkv 的好处**
        一次 GEMM 代替三次，kernel launch 少 2/3，且 M/N/K 更大更好打满 tensor core。
        代价是加载预训练权重时要自己拼 ``cat([Wq, Wk, Wv], dim=0)``——
        顺序必须是 q, k, v，和 ``nn.MultiheadAttention.in_proj_weight`` 一致。
    """

    def __init__(
        self, d_model: int, n_head: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: Float[Tensor, "B S H"],
        attn_mask: Bool[Tensor, "*broadcast Sq Sk"] | None = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "B S H"]:
        b, s, _ = x.shape

        qkv = self.qkv(x)  # (B, S, 3H)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 (B, S, H)

        # 唯一正确的 split-head 写法
        q = q.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, self.d_head).transpose(1, 2)

        out, _ = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout,
            is_causal=is_causal, training=self.training,
        )

        # merge heads：先 transpose 回去，再 contiguous 才能 view
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


# =============================================================================
# 3. Grouped-Query Attention（MQA 是 n_kv_head=1 的特例）
# =============================================================================


class GroupedQueryAttention(nn.Module):
    """GQA：n_head 个 query head 共享 n_kv_head 组 KV。

    面试要点：**GQA 省的是显存/带宽，不是计算量。**
        - KV cache 大小 ``2 * B * n_kv_head * S * d_head * dtype_size``，
          与 n_kv_head 成正比。从 MHA(n_kv_head=n_head) 到 MQA(n_kv_head=1)
          能把 KV cache 砍到 1/n_head，这是长上下文推理的关键。
        - 但 attention 本身的 FLOPs **不变**：kv 被 repeat 到 n_head 份后，
          ``Q K^T`` 依然是 n_head 个 (S, S) 矩阵。
        - 那为什么还快？因为 **decode 阶段是 memory-bound 的**：
          每生成一个 token 都要把整个 KV cache 从 HBM 读一遍，
          算术强度极低。KV cache 小 n_head 倍，读的字节数就少 n_head 倍。
        - MQA(n_kv_head=1) 压得最狠但质量掉得明显，GQA 取
          n_kv_head=8 之类的中间值，是"几乎不掉点 + 大部分收益"的折中。

    面试要点：**怎么扩展 kv head？**
        必须用 ``repeat_interleave``（``[k0,k0,k1,k1]``）而不是 ``repeat``
        （``[k0,k1,k0,k1]``）——因为 query head 是按组连续排布的，
        第 g 组的 n_rep 个 query head 要对上同一个 kv head。
        更省的写法是 ``expand``（不复制内存，靠 stride=0 广播），
        本实现用 expand + reshape 演示这个技巧。

    权重打包顺序与 MHA 兼容：当 ``n_kv_head == n_head`` 时
    ``qkv_proj`` 的输出维正好是 ``3 * d_model``，且切分顺序同为 q, k, v，
    因此可以直接拷贝 MHA 的权重（见 `test_gqa_equals_mha_when_full`）。
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_head == 0, "n_head 必须被 n_kv_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        out_dim = (n_head + 2 * n_kv_head) * self.d_head
        self.qkv_proj = nn.Linear(d_model, out_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    @staticmethod
    def repeat_kv(
        x: Float[Tensor, "B Hkv S D"], n_rep: int
    ) -> Float[Tensor, "B H S D"]:
        """把 n_kv_head 扩展成 n_head，语义等价 ``repeat_interleave(n_rep, dim=1)``。

        用 expand 而不是 repeat_interleave：expand 只改 stride 不拷贝内存，
        后面的 matmul 会自己处理广播（实际 cuBLAS 仍需 contiguous，
        但省下一次显式 kernel）。
        """
        if n_rep == 1:
            return x
        b, h_kv, s, d = x.shape
        return (
            x[:, :, None, :, :]
            .expand(b, h_kv, n_rep, s, d)
            .reshape(b, h_kv * n_rep, s, d)
        )

    def forward(
        self,
        x: Float[Tensor, "B S H"],
        cos: Float[Tensor, "S Dh2"] | None = None,
        sin: Float[Tensor, "S Dh2"] | None = None,
        attn_mask: Bool[Tensor, "*broadcast Sq Sk"] | None = None,
        is_causal: bool = False,
        kv_cache: "KVCache | None" = None,
        start_pos: int = 0,
    ) -> Float[Tensor, "B S H"]:
        b, s, _ = x.shape
        nh, nkv, dh = self.n_head, self.n_kv_head, self.d_head

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([nh * dh, nkv * dh, nkv * dh], dim=-1)

        q = q.view(b, s, nh, dh).transpose(1, 2)  # (B, nh, S, dh)
        k = k.view(b, s, nkv, dh).transpose(1, 2)  # (B, nkv, S, dh)
        v = v.view(b, s, nkv, dh).transpose(1, 2)

        if cos is not None:
            # RoPE 只作用于 q 和 k，**不作用于 v**（v 不参与相似度计算）
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            # 先写 cache 再取全量历史；k/v 存的是 **RoPE 之后** 的值，
            # 这样 decode 时不用重算历史位置的旋转。
            k, v = kv_cache.update(k, v)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        if is_causal and attn_mask is None:
            attn_mask = causal_mask(s, k.shape[-2], start_pos=start_pos, device=x.device)
            is_causal = False

        out, _ = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout,
            is_causal=is_causal, training=self.training,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


# =============================================================================
# 4. RoPE (Rotary Position Embedding)
# =============================================================================


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[Tensor, "S D2"], Float[Tensor, "S D2"]]:
    r"""预计算 RoPE 的 cos/sin 表，返回两个 ``(seq_len, dim // 2)`` 张量。

    面试要点：**RoPE 在做什么？**
        把 d 维特征拆成 d/2 个二维平面，第 i 个平面上按角度
        ``theta_i * m`` 旋转（m 是 token 的绝对位置）::

            theta_i = base^(-2i / d),   i = 0 .. d/2-1

        对第 i 个平面上的分量 ``(x_a, x_b)``::

            x_a' = x_a * cos(m * theta_i) - x_b * sin(m * theta_i)
            x_b' = x_a * sin(m * theta_i) + x_b * cos(m * theta_i)

        写成复数最清爽：把 ``(x_a, x_b)`` 看成复数 ``z``，
        RoPE 就是 ``z -> z * e^{i m theta}``。

    面试要点：**为什么说它是"相对"位置编码？**
        位置 m 的 query 与位置 n 的 key 做内积::

            <R_m q, R_n k> = Re[ (q e^{i m theta}) * conj(k e^{i n theta}) ]
                           = Re[ q conj(k) e^{i (m-n) theta} ]

        只依赖 ``m - n``。也就是说：**编码时用绝对位置，注意力里自动变成相对位置**。
        这是 RoPE 相对于可学习绝对位置编码的核心优势，也是最好的单元测试
        （见 `test_rope_inner_product_depends_only_on_relative_position`）。

    面试要点：**base 的作用与长度外推**
        base 越大，低频通道的波长越长，能表达的上下文越长。
        NTK-aware / linear 插值扩上下文本质就是改 base 或缩放位置索引。

    实现约定：本文件用 **split-half（GPT-NeoX / HF-Llama）** 排布，
    即把 x 前一半和后一半配对成复数；原论文用相邻两维配对（interleaved）。
    两种都对，**但必须和权重的训练时排布一致**，混用会静默掉点——
    这是移植 checkpoint 时的经典坑。
    """
    assert dim % 2 == 0, "RoPE 的维度必须是偶数"
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )  # (dim/2,)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (S, dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(
    x: Float[Tensor, "B H S D"],
    cos: Float[Tensor, "S D2"],
    sin: Float[Tensor, "S D2"],
) -> Float[Tensor, "B H S D"]:
    """对 ``(B, n_head, S, d_head)`` 施加 RoPE。cos/sin 是 ``(S, d_head//2)``。

    decode 时由调用方自己切片，例如 ``apply_rope(q, cos[pos:pos+1], sin[pos:pos+1])``。
    """
    x1, x2 = x.chunk(2, dim=-1)  # 各 (B, H, S, D/2)
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, S, D/2)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =============================================================================
# 5. KV Cache
# =============================================================================


class KVCache:
    """预分配 buffer 的 KV cache：``(B, n_kv_head, max_seq, d_head)`` + 一个游标。

    面试要点：**为什么预分配而不是 torch.cat？**
        ``cat`` 每步都重新分配 + 拷贝整个历史，decode N 步的总拷贝量是 O(N^2)，
        而且不停触发 allocator。预分配后每步只写 1 个 slot，是 O(N)。
        代价是要提前知道 max_seq，且显存按最坏情况占满
        （PagedAttention / vLLM 就是为了解决这个"预留浪费"而生的）。

    面试要点：**cache 里存 RoPE 之前还是之后的 k？**
        存 **之后**。RoPE 只依赖该 token 的绝对位置，写入时算一次即可；
        存之前的话每步都要对整个历史重算旋转，白白 O(N) 开销。

    面试要点：**KV cache 有多大？**
        ``2 * n_layer * B * n_kv_head * S * d_head * bytes``。
        比如 7B 模型 32 层、n_kv_head=32、d_head=128、fp16、B=1、S=4096：
        2*32*1*32*4096*128*2 B = 2 GB —— 比激活值大得多，
        所以才有 GQA / MQA / MLA / KV 量化这一整条优化线。
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_head: int,
        max_seq_len: int,
        d_head: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        shape = (batch_size, n_kv_head, max_seq_len, d_head)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.pos = 0  # 已填充的长度（也是下一个 token 的绝对位置）

    def reset(self) -> None:
        self.pos = 0

    def update(
        self, k: Float[Tensor, "B Hkv S D"], v: Float[Tensor, "B Hkv S D"]
    ) -> tuple[Float[Tensor, "B Hkv P D"], Float[Tensor, "B Hkv P D"]]:
        """写入新的 k/v（prefill 时 S>1，decode 时 S=1），返回**全部**历史。"""
        n = k.shape[2]
        assert self.pos + n <= self.max_seq_len, "KV cache 溢出，需要更大的 max_seq_len"
        self.k[:, :, self.pos : self.pos + n] = k
        self.v[:, :, self.pos : self.pos + n] = v
        self.pos += n
        # 返回 slice（view，不拷贝）
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]


# =============================================================================
# 6. Transformer Block: RMSNorm + SwiGLU + pre-norm
# =============================================================================


class RMSNorm(nn.Module):
    r"""RMSNorm: ``x / sqrt(mean(x^2) + eps) * g``。

    面试要点：与 LayerNorm 的差别是**不减均值、没有 bias**。
    少了一次 reduce，且实测效果与 LayerNorm 相当，
    所以 Llama 之后基本都用它。
    注意 ``mean(x^2)`` 要在 fp32 里算（低精度下平方容易溢出/损失精度）。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "... H"]) -> Float[Tensor, "... H"]:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    r"""SwiGLU FFN: ``W2( SiLU(W1 x) * W3 x )``。

    面试要点：**为什么是 3 个矩阵？**
        GLU 家族用一路做"门控"、一路做"内容"，逐元素相乘。
        为了让参数量和 2 矩阵的 ReLU FFN 持平（``2 * d * 4d``），
        hidden 要取 ``(2/3) * 4d = 8d/3``，Llama 还会向上对齐到 256 的倍数。
        常见错法：直接把 hidden 设成 4d，参数量悄悄多了 50%。
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # down

    def forward(self, x: Float[Tensor, "... H"]) -> Float[Tensor, "... H"]:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    r"""Pre-norm 的 decoder block（Llama 风格）。

    结构::

        x = x + Attn(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    面试要点：**pre-norm vs post-norm**
        - post-norm（原始 Transformer）：``x = Norm(x + Sublayer(x))``。
          残差流每层都被 Norm 重新缩放，**深层的梯度要穿过 L 个 Norm**，
          期望梯度范数随深度指数变化，必须靠 learning-rate warmup 才能训起来；
          去掉 warmup 直接发散。
        - pre-norm：``x = x + Sublayer(Norm(x))``。
          残差路径上是**纯恒等映射**，梯度可以无损直达底层，
          训练稳定得多，对 warmup 不敏感，是现在的默认选择。
        - 代价：pre-norm 的残差流方差随层数累加（每层都往上加东西），
          深层子层的相对贡献被稀释，表达能力略弱（"representation collapse"），
          所以最后要补一个 final norm；也有 DeepNorm / sandwich-norm 之类的折中。
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_head, n_kv_head or n_head, dropout=dropout, bias=False
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden, bias=False)

    def forward(
        self,
        x: Float[Tensor, "B S H"],
        cos: Float[Tensor, "S D2"] | None = None,
        sin: Float[Tensor, "S D2"] | None = None,
        attn_mask: Bool[Tensor, "*broadcast Sq Sk"] | None = None,
        is_causal: bool = True,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> Float[Tensor, "B S H"]:
        x = x + self.attn(
            self.attn_norm(x), cos, sin, attn_mask=attn_mask,
            is_causal=is_causal, kv_cache=kv_cache, start_pos=start_pos,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =============================================================================
# 7. MiniGPT
# =============================================================================


class MiniGPT(nn.Module):
    """embedding -> N x TransformerBlock -> final RMSNorm -> lm_head。

    面试要点：**weight tying（``lm_head.weight = tok_emb.weight``）**
        省下 ``V * d`` 个参数（小模型里能占总量一半以上），
        且有"输入输出共享同一套词表示"的正则效果。
        注意共享后 embedding 的梯度会同时来自两条路径。
        大模型（Llama 2/3）反而不 tie，因为 V*d 相对总量已经很小，
        解绑能多一点自由度。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        n_kv_head: int | None = None,
        ffn_hidden: int | None = None,
        max_seq_len: int = 512,
        tie_weights: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        self.d_head = d_model // n_head
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_head, self.n_kv_head, ffn_hidden)
                for _ in range(n_layer)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(max_seq_len, self.d_head, base=rope_base)
        # 用 buffer 存，随 .to(device) 一起搬；persistent=False 不进 state_dict
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        idx: Int[Tensor, "B S"],
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> Float[Tensor, "B S V"]:
        s = idx.shape[1]
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos : start_pos + s]
        sin = self.rope_sin[start_pos : start_pos + s]

        for i, block in enumerate(self.blocks):
            x = block(
                x, cos, sin, is_causal=True,
                kv_cache=None if caches is None else caches[i],
                start_pos=start_pos,
            )
        return self.lm_head(self.final_norm(x))

    def make_caches(
        self, batch_size: int, max_seq_len: int | None = None
    ) -> list[KVCache]:
        dev = self.tok_emb.weight.device
        dt = self.tok_emb.weight.dtype
        return [
            KVCache(
                batch_size, self.n_kv_head, max_seq_len or self.max_seq_len,
                self.d_head, dtype=dt, device=dev,
            )
            for _ in self.blocks
        ]


# =============================================================================
# 8. 采样 / generate
# =============================================================================


def top_k_top_p_filter(
    logits: Float[Tensor, "B V"], top_k: int | None = None, top_p: float | None = None
) -> Float[Tensor, "B V"]:
    """把被截断的 token 的 logit 置为 -inf。

    面试要点：**top-k 和 top-p(nucleus) 的区别**
        - top-k 保留概率最大的 k 个，**候选集大小固定**。
          分布很尖时（模型很确定）会强行引入本该被排除的低概率词；
          分布很平时又砍得太狠。
        - top-p 保留累计概率刚好超过 p 的最小集合，**候选集大小自适应**。
          确定时集合小、不确定时集合大，通常质量更好。
        - 实现细节：top-p 必须**至少保留 1 个 token**，
          否则当最大概率就 > p 时（比如 p=0.5 而 max prob=0.9）会把全部截掉。
          标准做法是把 shift 后的 mask 第 0 列强制置 False。
        - 顺序：先 temperature，再 top-k/top-p，最后 softmax 采样。
          （在截断后的集合上重新归一化，这正是 masked softmax 自动做的事。）
    """
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1, None]  # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = sorted_logits.softmax(-1).cumsum(-1)
        # 要移除的是"累计概率已经超过 p 之后"的 token
        remove = cumprobs - sorted_logits.softmax(-1) >= top_p
        remove[..., 0] = False  # 永远保留概率最高的那个
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    return logits


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: Int[Tensor, "B S"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> Int[Tensor, "B S_new"]:
    """自回归生成，返回 ``(B, S + max_new_tokens)``。

    面试要点：**KV cache 版和朴素版的区别**
        - 朴素版每步把整条 ``(B, t)`` 序列重新 forward，只取最后一个 logit，
          复杂度 O(N^2) 次 block 计算（总 O(N^3) attention FLOPs）。
        - cache 版分两段：**prefill** 一次性喂入 prompt（并行、compute-bound），
          **decode** 每步只喂 1 个 token（串行、memory-bound）。
          这两段的性能特征完全不同，是推理优化里 chunked-prefill、
          continuous batching 等技术的出发点。
        - 两者在 causal 下**数学上完全等价**，所以 greedy 结果必须逐位相同——
          这是验证 cache 实现正确性的黄金测试。

    面试要点：**temperature 怎么理解？**
        ``softmax(logits / T)``。T→0 时分布变成 one-hot，等价 greedy（argmax）；
        T→inf 时变成均匀分布。T<1 更保守，T>1 更发散。
        实现上 T=0 要单独走 argmax 分支，不然会除零。
    """
    model.eval()
    caches = model.make_caches(idx.shape[0]) if use_cache else None
    cur = idx

    for step in range(max_new_tokens):
        if use_cache:
            if step == 0:  # prefill
                inp, start = cur, 0
            else:  # decode：只喂最后一个 token
                inp, start = cur[:, -1:], caches[0].pos
            logits = model(inp, caches=caches, start_pos=start)
        else:
            logits = model(cur[:, -model.max_seq_len :])

        logits = logits[:, -1, :].float()  # (B, V)，采样统一在 fp32 做

        if temperature == 0.0:
            next_tok = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_top_p_filter(logits, top_k, top_p)
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=generator)

        cur = torch.cat([cur, next_tok], dim=1)

    return cur


# =============================================================================
#                                  TESTS
# =============================================================================

# ---------------------------- SDPA ----------------------------


def test_sdpa_matches_torch():
    """手写 SDPA 与 F.scaled_dot_product_attention 数值对齐（无 mask）。"""
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 7, 8)
    v = torch.randn(2, 3, 7, 8)

    mine, attn = scaled_dot_product_attention(q, k, v, training=False)
    ref = F.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(attn.sum(-1), torch.ones(2, 3, 5), rtol=0, atol=1e-6)


def test_sdpa_matches_torch_causal_and_mask():
    """causal 与显式 bool mask 两条路径都要与官方一致。

    顺带验证 PyTorch 的约定：bool attn_mask 里 True = 保留，
    等价于 float mask 里填 0（False 处填 -inf）。
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 6, 4) for _ in range(3))

    mine_c, _ = scaled_dot_product_attention(q, k, v, is_causal=True, training=False)
    torch.testing.assert_close(
        mine_c, F.scaled_dot_product_attention(q, k, v, is_causal=True),
        rtol=1e-5, atol=1e-6,
    )

    m = causal_mask(6)
    mine_m, _ = scaled_dot_product_attention(q, k, v, attn_mask=m, training=False)
    torch.testing.assert_close(mine_m, mine_c, rtol=1e-5, atol=1e-6)

    # float 加性 mask 路径
    mf = torch.zeros(6, 6).masked_fill(~m, float("-inf"))
    mine_f, _ = scaled_dot_product_attention(q, k, v, attn_mask=mf, training=False)
    torch.testing.assert_close(mine_f, mine_c, rtol=1e-5, atol=1e-6)


def test_sdpa_scaling_prevents_softmax_saturation():
    """量化演示"为什么要除以 sqrt(d_k)"：不缩放时 softmax 熵急剧塌缩。"""
    torch.manual_seed(0)
    d_k = 256
    q = torch.randn(1, 1, 64, d_k)
    k = torch.randn(1, 1, 64, d_k)

    unscaled = (q @ k.transpose(-2, -1)).softmax(-1)
    scaled = (q @ k.transpose(-2, -1) / math.sqrt(d_k)).softmax(-1)

    ent = lambda p: -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    # 未缩放的分布接近 one-hot（熵接近 0），缩放后接近均匀（熵上界是 log 64）
    assert ent(unscaled) < 0.5
    assert ent(scaled) > 0.8 * math.log(64)

    # 更直接的证据：softmax 的雅可比迹 sum p(1-p) 就是梯度能传下去的"通道量"。
    # 未缩放时它被压到接近 0（梯度消失），缩放后接近上界 1。
    jac = lambda p: (p * (1 - p)).sum(-1).mean().item()
    assert jac(unscaled) < 0.25
    assert jac(scaled) > 0.9
    assert jac(scaled) > 5 * jac(unscaled)


def test_sdpa_dropout_and_eval():
    """dropout 只在 training=True 时生效，且作用在 attention 权重上。"""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 4, 4) for _ in range(3))

    a, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=False)
    b, _ = scaled_dot_product_attention(q, k, v, training=False)
    torch.testing.assert_close(a, b)

    torch.manual_seed(1)
    c, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.9, training=True)
    assert not torch.allclose(c, b)


# ---------------------------- mask ----------------------------


def test_causal_mask_shape_and_upper_triangle_is_zero():
    """mask 后 attention 权重的严格上三角必须精确为 0（不是"很小"）。"""
    torch.manual_seed(0)
    m = causal_mask(5)
    assert m.shape == (5, 5)
    assert bool((m == torch.ones(5, 5, dtype=torch.bool).tril()).all())

    q, k, v = (torch.randn(1, 1, 5, 4) for _ in range(3))
    _, attn = scaled_dot_product_attention(q, k, v, attn_mask=m, training=False)
    assert attn.triu(diagonal=1).abs().max().item() == 0.0
    torch.testing.assert_close(attn.sum(-1), torch.ones(1, 1, 5), rtol=0, atol=1e-6)


def test_causal_mask_with_offset_for_decode():
    """decode 阶段（q_len=1, kv_len=P）的 mask 必须全 True。"""
    m = causal_mask(1, 5, start_pos=4)
    assert m.shape == (1, 5) and bool(m.all())
    # 反例：忘了 start_pos 就只能看到第 0 个 key
    wrong = causal_mask(1, 5, start_pos=0)
    assert int(wrong.sum()) == 1


def test_padding_mask_combines_with_causal():
    """padding mask 与 causal mask 用逻辑与组合；pad 位置权重严格为 0。"""
    torch.manual_seed(0)
    b, s, d = 2, 6, 4
    lengths = torch.tensor([6, 3])
    pm = padding_mask(lengths, s)  # (B, 1, 1, S)
    cm = causal_mask(s).view(1, 1, s, s)
    mask = pm & cm
    assert mask.shape == (b, 1, s, s)

    q, k, v = (torch.randn(b, 1, s, d) for _ in range(3))
    _, attn = scaled_dot_product_attention(q, k, v, attn_mask=mask, training=False)

    assert attn[0].triu(diagonal=1).abs().max().item() == 0.0
    # 第 1 个样本长度 3，key 位置 3..5 必须完全没有权重
    assert attn[1, :, :, 3:].abs().max().item() == 0.0
    # 前 3 个 query 行（都有可见 key）权重仍归一
    torch.testing.assert_close(
        attn[1, :, :3].sum(-1), torch.ones(1, 3), rtol=0, atol=1e-6
    )


# ---------------------------- MHA ----------------------------


def _copy_torch_mha_weights(mine: MultiHeadAttention, ref: nn.MultiheadAttention) -> None:
    """把 nn.MultiheadAttention 的权重搬到手写实现里。

    **对齐要点（面试素材）**：``nn.MultiheadAttention`` 在 embed_dim ==
    kdim == vdim 时把三个投影**打包成一个** ``in_proj_weight``，
    形状 ``(3 * E, E)``，行方向按 **[Wq; Wk; Wv]** 的顺序拼接
    （``in_proj_bias`` 同理是 ``(3E,)``）。
    只有当 kdim/vdim 不同时才会拆成 q_proj_weight / k_proj_weight / v_proj_weight
    （此时 in_proj_weight 是 None）。
    我们的 ``self.qkv`` 恰好也是 ``Linear(E, 3E)``，所以可以整块直接拷。
    """
    with torch.no_grad():
        mine.qkv.weight.copy_(ref.in_proj_weight)
        mine.qkv.bias.copy_(ref.in_proj_bias)
        mine.out_proj.weight.copy_(ref.out_proj.weight)
        mine.out_proj.bias.copy_(ref.out_proj.bias)


def test_mha_matches_torch_multiheadattention():
    """权重对齐后与 nn.MultiheadAttention(batch_first=True) 数值一致。"""
    torch.manual_seed(0)
    d_model, n_head, b, s = 16, 4, 2, 7
    ref = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.0)
    mine = MultiHeadAttention(d_model, n_head)
    _copy_torch_mha_weights(mine, ref)
    mine.eval()
    ref.eval()

    x = torch.randn(b, s, d_model)
    out_ref, _ = ref(x, x, x, need_weights=False)
    out_mine = mine(x)

    assert out_mine.shape == (b, s, d_model)
    torch.testing.assert_close(out_mine, out_ref, rtol=1e-5, atol=1e-5)


def test_mha_matches_torch_causal():
    """causal 情况也要一致。

    **对齐要点（面试素材）**：两套 API 的 bool mask 语义**正好相反**。
      - ``F.scaled_dot_product_attention(attn_mask=bool)``: True = **保留**
      - ``nn.MultiheadAttention(attn_mask=bool)``:          True = **屏蔽**
    所以下面 ref 传的是 ``triu(diagonal=1)``（上三角 True=屏蔽），
    而我们自己传的是 ``tril``（下三角 True=保留）。
    从一套 API 迁到另一套时忘了取反，会得到"只能看未来"的反向 causal，
    loss 会离奇地降得飞快（模型直接偷看答案），是很典型的事故。
    """
    torch.manual_seed(0)
    d_model, n_head, b, s = 16, 4, 2, 6
    ref = nn.MultiheadAttention(d_model, n_head, batch_first=True)
    mine = MultiHeadAttention(d_model, n_head)
    _copy_torch_mha_weights(mine, ref)
    mine.eval()
    ref.eval()

    x = torch.randn(b, s, d_model)
    torch_mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)  # True=屏蔽
    out_ref, _ = ref(x, x, x, attn_mask=torch_mask, need_weights=False)
    out_mine = mine(x, is_causal=True)

    torch.testing.assert_close(out_mine, out_ref, rtol=1e-5, atol=1e-5)


def test_mha_wrong_reshape_is_silently_different():
    """演示经典 bug：view(B, n_head, S, d_head) 不报错，但结果完全不同。"""
    torch.manual_seed(0)
    b, s, h, nh = 2, 6, 12, 3
    dh = h // nh
    x = torch.randn(b, s, h)

    right = x.view(b, s, nh, dh).transpose(1, 2)  # 正确
    wrong = x.view(b, nh, s, dh)  # 错误但 shape 一样

    assert right.shape == wrong.shape == (b, nh, s, dh)
    assert not torch.allclose(right, wrong)
    # 具体错在哪：错误写法的 head 0 拿的是前 s*dh 个元素，
    # 也就是前 dh/h * s 个 token 的**全部**通道
    torch.testing.assert_close(wrong[0, 0].reshape(-1), x[0].reshape(-1)[: s * dh])


def test_mha_backward():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    mha(x, is_causal=True).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(mha.qkv.weight.grad).all()


# ---------------------------- GQA ----------------------------


def test_gqa_equals_mha_when_full():
    """n_kv_head == n_head 时，GQA 就是 MHA（权重可直接对拷）。"""
    torch.manual_seed(0)
    d_model, n_head, b, s = 16, 4, 2, 5
    mha = MultiHeadAttention(d_model, n_head, bias=False)
    gqa = GroupedQueryAttention(d_model, n_head, n_kv_head=n_head, bias=False)
    with torch.no_grad():
        gqa.qkv_proj.weight.copy_(mha.qkv.weight)
        gqa.out_proj.weight.copy_(mha.out_proj.weight)
    mha.eval()
    gqa.eval()

    x = torch.randn(b, s, d_model)
    torch.testing.assert_close(gqa(x, is_causal=True), mha(x, is_causal=True),
                               rtol=1e-5, atol=1e-6)


def test_gqa_shapes_and_kv_param_savings():
    """MQA(n_kv_head=1) 的 shape 正确，且 qkv 投影参数确实变少。"""
    torch.manual_seed(0)
    d_model, n_head, b, s = 32, 8, 2, 6
    for n_kv in (8, 4, 1):
        gqa = GroupedQueryAttention(d_model, n_head, n_kv, bias=False)
        x = torch.randn(b, s, d_model)
        assert gqa(x, is_causal=True).shape == (b, s, d_model)
        d_head = d_model // n_head
        assert gqa.qkv_proj.weight.shape == ((n_head + 2 * n_kv) * d_head, d_model)

    # KV cache 显存与 n_kv_head 成正比：MQA 是 MHA 的 1/8
    mha_kv = KVCache(b, 8, 128, 4)
    mqa_kv = KVCache(b, 1, 128, 4)
    assert mha_kv.k.numel() == 8 * mqa_kv.k.numel()


def test_repeat_kv_uses_interleave_semantics():
    """repeat_kv 必须等价 repeat_interleave 而不是 repeat。"""
    x = torch.arange(2 * 2 * 1 * 3, dtype=torch.float32).view(2, 2, 1, 3)
    got = GroupedQueryAttention.repeat_kv(x, 3)
    torch.testing.assert_close(got, x.repeat_interleave(3, dim=1))
    assert not torch.allclose(got, x.repeat(1, 3, 1, 1))


# ---------------------------- RoPE ----------------------------


def test_rope_preserves_norm():
    """旋转是正交变换，不改变向量的 L2 范数（每个 head 都要检查）。"""
    torch.manual_seed(0)
    b, h, s, d = 2, 3, 10, 8
    cos, sin = build_rope_cache(s, d)
    x = torch.randn(b, h, s, d)
    y = apply_rope(x, cos, sin)

    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-6)
    # 每个二维子平面的范数也单独守恒（说明确实是逐平面旋转）
    x1, x2 = x.chunk(2, -1)
    y1, y2 = y.chunk(2, -1)
    torch.testing.assert_close(y1**2 + y2**2, x1**2 + x2**2, rtol=1e-5, atol=1e-6)


def test_rope_inner_product_depends_only_on_relative_position():
    """RoPE 的灵魂性质：<R_m q, R_n k> 只与 (m - n) 有关。

    这是"为什么 RoPE 是相对位置编码"的直接证据，也是最好的单元测试。
    """
    torch.manual_seed(0)
    d, s = 16, 32
    cos, sin = build_rope_cache(s, d)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def dot(m: int, n: int) -> float:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    for delta in (0, 1, 5, -3):
        vals = [dot(m, m - delta) for m in range(8, 16)]
        assert max(vals) - min(vals) < 1e-4, f"delta={delta} 时内积不恒定: {vals}"

    # 不同的相对距离应给出不同的内积（否则位置信息就白编码了）
    assert abs(dot(10, 10) - dot(10, 5)) > 1e-3


def test_rope_position_zero_is_identity():
    """位置 0 的旋转角为 0，RoPE 退化为恒等映射。"""
    torch.manual_seed(0)
    cos, sin = build_rope_cache(4, 8)
    torch.testing.assert_close(cos[0], torch.ones(4))
    torch.testing.assert_close(sin[0], torch.zeros(4))
    x = torch.randn(1, 1, 1, 8)
    torch.testing.assert_close(apply_rope(x, cos[:1], sin[:1]), x, rtol=1e-6, atol=1e-6)


def test_rope_cache_matches_closed_form():
    """cos/sin 表与公式 theta_i = base^(-2i/d) 逐元素吻合。"""
    d, s, base = 8, 5, 10000.0
    cos, sin = build_rope_cache(s, d, base=base)
    for m in range(s):
        for i in range(d // 2):
            theta = base ** (-2 * i / d)
            assert abs(cos[m, i].item() - math.cos(m * theta)) < 1e-6
            assert abs(sin[m, i].item() - math.sin(m * theta)) < 1e-6


# ---------------------------- KV cache ----------------------------


def test_kv_cache_buffer_and_cursor():
    """预分配 buffer 的写入位置与游标行为。"""
    cache = KVCache(2, 3, 8, 4)
    assert cache.k.shape == (2, 3, 8, 4) and cache.pos == 0

    k1 = torch.randn(2, 3, 5, 4)
    k_all, _ = cache.update(k1, k1)
    assert cache.pos == 5 and k_all.shape == (2, 3, 5, 4)
    torch.testing.assert_close(k_all, k1)

    k2 = torch.randn(2, 3, 1, 4)
    k_all, _ = cache.update(k2, k2)
    assert cache.pos == 6
    torch.testing.assert_close(k_all, torch.cat([k1, k2], dim=2))

    cache.reset()
    assert cache.pos == 0


def test_kv_cache_decode_matches_full_forward():
    """**验证 KV cache 正确性的黄金测试**。

    causal 下，"prefill 一段 + 逐 token decode" 每一步的 logits，
    必须与"一次性 forward 整条序列"取对应位置的 logits **逐位相同**。
    只要 mask 的 start_pos、RoPE 的位置切片、cache 写入下标三者中任一处错了，
    这个测试就会挂——它一次性覆盖了 KV cache 实现的所有常见 bug。
    """
    torch.manual_seed(0)
    v, b, s = 32, 2, 9
    model = MiniGPT(vocab_size=v, d_model=16, n_layer=2, n_head=4, n_kv_head=2,
                    max_seq_len=16)
    model.eval()
    idx = torch.randint(0, v, (b, s))

    with torch.no_grad():
        full = model(idx)  # (B, S, V)

        caches = model.make_caches(b)
        prefill = 4
        out = [model(idx[:, :prefill], caches=caches, start_pos=0)]
        for t in range(prefill, s):
            out.append(model(idx[:, t : t + 1], caches=caches, start_pos=t))
        incremental = torch.cat(out, dim=1)

    assert incremental.shape == full.shape
    torch.testing.assert_close(incremental, full, rtol=1e-5, atol=1e-5)


# ---------------------------- block / MiniGPT ----------------------------


def test_rmsnorm_matches_torch():
    """手写 RMSNorm 与 nn.RMSNorm 对齐（顺带确认没有减均值）。"""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16) * 3 + 1.0
    mine = RMSNorm(16, eps=1e-6)
    ref = nn.RMSNorm(16, eps=1e-6)
    torch.testing.assert_close(mine(x), ref(x), rtol=1e-5, atol=1e-6)
    # RMSNorm 不会把均值归零（这正是它与 LayerNorm 的区别）
    assert mine(x).mean().abs() > 1e-3


def test_swiglu_hidden_dim_and_param_count():
    """默认 hidden = 8d/3，参数量与 2 矩阵的 4d FFN 基本持平。"""
    d = 96
    ffn = SwiGLU(d)
    assert ffn.w1.out_features == int(8 * d / 3)
    swiglu_params = sum(p.numel() for p in ffn.parameters())
    vanilla_params = 2 * d * 4 * d
    assert abs(swiglu_params - vanilla_params) / vanilla_params < 0.01


def test_transformer_block_shape_and_grad():
    torch.manual_seed(0)
    b, s, d = 2, 6, 16
    blk = TransformerBlock(d, n_head=4, n_kv_head=2)
    cos, sin = build_rope_cache(s, d // 4)
    x = torch.randn(b, s, d, requires_grad=True)

    y = blk(x, cos, sin, is_causal=True)
    assert y.shape == (b, s, d)
    y.pow(2).mean().backward()

    assert torch.isfinite(x.grad).all()
    for n, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_prenorm_residual_is_identity_path():
    """pre-norm 的残差路径是恒等映射：把子层权重置零后，block 输出 == 输入。

    这正是 pre-norm 训练稳定的原因——梯度有一条无损通路直达底层。
    post-norm 做同样的事会得到 Norm(x) != x。
    """
    torch.manual_seed(0)
    blk = TransformerBlock(16, n_head=4)
    with torch.no_grad():
        blk.attn.out_proj.weight.zero_()
        blk.ffn.w2.weight.zero_()
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(blk(x, is_causal=True), x, rtol=1e-6, atol=1e-6)


def test_minigpt_forward_shape_and_backward():
    torch.manual_seed(0)
    v, b, s = 50, 2, 7
    model = MiniGPT(vocab_size=v, d_model=16, n_layer=2, n_head=4, max_seq_len=32)
    idx = torch.randint(0, v, (b, s))

    logits = model(idx)
    assert logits.shape == (b, s, v)

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, v), idx[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_minigpt_param_count_matches_hand_calculation():
    """参数量手算：面试里常被要求口算模型大小，这里把公式落到代码上。"""
    v, d, n_layer, n_head, n_kv, ffn_h = 50, 16, 2, 4, 2, 24
    d_head = d // n_head
    model = MiniGPT(v, d, n_layer, n_head, n_kv, ffn_hidden=ffn_h, max_seq_len=32)

    per_block = (
        d  # attn RMSNorm weight
        + (n_head + 2 * n_kv) * d_head * d  # fused qkv (bias=False)
        + d * d  # out_proj
        + d  # ffn RMSNorm weight
        + 3 * d * ffn_h  # SwiGLU 的 w1 / w3 / w2
    )
    expected = v * d + n_layer * per_block + d + d * v  # emb + blocks + final norm + head
    assert sum(p.numel() for p in model.parameters()) == expected

    # tie_weights 后 lm_head 与 embedding 共享，参数量正好少 V*d
    tied = MiniGPT(v, d, n_layer, n_head, n_kv, ffn_hidden=ffn_h,
                   max_seq_len=32, tie_weights=True)
    assert sum(p.numel() for p in tied.parameters()) == expected - v * d
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()


# ---------------------------- generate ----------------------------


def _tiny_model(vocab: int = 24) -> MiniGPT:
    torch.manual_seed(0)
    m = MiniGPT(vocab, d_model=16, n_layer=2, n_head=4, n_kv_head=2, max_seq_len=32)
    m.eval()
    return m


def test_generate_output_length_and_prefix_preserved():
    torch.manual_seed(0)
    model = _tiny_model()
    idx = torch.randint(0, 24, (3, 4))
    out = generate(model, idx, max_new_tokens=5, temperature=0.0)
    assert out.shape == (3, 9)
    torch.testing.assert_close(out[:, :4], idx)
    assert int(out.max()) < 24 and int(out.min()) >= 0


def test_generate_temperature_zero_equals_greedy():
    """T→0 等价 greedy：既测 T==0 的特判分支，也测极小 T 的数值行为。"""
    torch.manual_seed(0)
    model = _tiny_model()
    idx = torch.randint(0, 24, (2, 4))

    greedy = generate(model, idx, 6, temperature=0.0)
    tiny_t = generate(model, idx, 6, temperature=1e-5)
    assert torch.equal(greedy, tiny_t)


def test_generate_top_k_one_equals_greedy():
    """top_k=1 只剩最大 logit，采样必然选中它，等价 greedy。"""
    torch.manual_seed(0)
    model = _tiny_model()
    idx = torch.randint(0, 24, (2, 4))
    assert torch.equal(
        generate(model, idx, 6, temperature=1.0, top_k=1),
        generate(model, idx, 6, temperature=0.0),
    )
    # top_p 取极小值同理（至少保留 1 个的兜底逻辑生效）
    assert torch.equal(
        generate(model, idx, 6, temperature=1.0, top_p=1e-6),
        generate(model, idx, 6, temperature=0.0),
    )


def test_generate_with_and_without_kv_cache_agree():
    """带/不带 KV cache 的 greedy 结果必须逐 token 相同。"""
    torch.manual_seed(0)
    model = _tiny_model()
    idx = torch.randint(0, 24, (2, 5))
    with_cache = generate(model, idx, 8, temperature=0.0, use_cache=True)
    without = generate(model, idx, 8, temperature=0.0, use_cache=False)
    assert torch.equal(with_cache, without)


def test_top_k_top_p_filter_semantics():
    """截断函数本身的行为：候选集大小、-inf 位置、至少保留 1 个。"""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])

    kept = top_k_top_p_filter(logits.clone(), top_k=2)
    assert torch.isfinite(kept).sum().item() == 2
    assert torch.isfinite(kept[0, :2]).all()

    probs = logits.softmax(-1)  # 约 [0.64, 0.24, 0.09, 0.03, 0.01]
    kept = top_k_top_p_filter(logits.clone(), top_p=0.85)
    assert torch.isfinite(kept).sum().item() == 2, probs

    # 最大概率已经超过 top_p：必须兜底保留 1 个，而不是全部截掉
    kept = top_k_top_p_filter(logits.clone(), top_p=0.1)
    assert torch.isfinite(kept).sum().item() == 1


def test_generate_temperature_increases_diversity():
    """高温采样比低温采样产生更多不同的 token（统计性质，非严格保证）。"""
    torch.manual_seed(0)
    model = _tiny_model()
    idx = torch.randint(0, 24, (16, 3))
    g = torch.Generator().manual_seed(0)
    lo = generate(model, idx, 6, temperature=0.2, generator=g)[:, 3:]
    hi = generate(model, idx, 6, temperature=3.0, generator=g)[:, 3:]
    assert lo.unique().numel() < hi.unique().numel()


if __name__ == "__main__":
    import sys

    ns = dict(globals())
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all transformer tests passed", file=sys.stderr)
