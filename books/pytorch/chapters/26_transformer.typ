#import "../template.typ": *

= Transformer 全家桶：从 attention 到 generate

这是全书最重要的一章。"手写 multi-head attention" 是 LLM 相关岗位的必考题，"手写 RoPE"、"实现 KV cache"、"写一个 top-p 采样"紧随其后。这些题的共同特点是*看起来简单、有一堆静默出错的坑*：reshape 顺序写反不报错但学不出来、mask 语义取反会让 loss 降得离奇地快、KV cache 的位置偏移错了只在长序列上才暴露。

配套的可运行代码在 `python/pytorch/interview/test_transformer.py`（32 个 pytest，全部通过），每个实现都和 `F.scaled_dot_product_attention` / `nn.MultiheadAttention` / `nn.RMSNorm` 做了数值对齐——"我能证明我写的和官方逐元素相同"是这类题最有力的答案。

*全文的 mask 约定*：bool mask 里 *`True` = 可见*，`False` = 屏蔽。这和 `F.scaled_dot_product_attention` 一致，和 `nn.MultiheadAttention` 相反（后者 `True` = 屏蔽），后面会专门讲这个坑。

== Scaled dot-product attention

*题目*：手写 `F.scaled_dot_product_attention`，支持 bool / float mask、`is_causal`、dropout。

#formula[$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k) + M) V $]

```python
def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, training=True):
    d_k = query.shape[-1]
    # (B,H,Sq,D) @ (B,H,D,Sk) -> (B,H,Sq,Sk)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if is_causal:
        assert attn_mask is None, "is_causal 与 attn_mask 只能二选一"
        s_q, s_k = query.shape[-2], key.shape[-2]
        # decode 阶段 Sq < Sk，query 要对齐到序列末尾
        attn_mask = causal_mask(s_q, s_k, start_pos=s_k - s_q, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask          # float mask 按加性 bias 处理

    attn = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)

    return torch.matmul(attn, value), attn
```

=== 为什么要除以 $sqrt(d_k)$

这是本章最高频的追问，而且面试官想听的是*推导*，不是"为了数值稳定"。

设 $q, k in RR^(d_k)$ 的各分量独立同分布、均值 0、方差 1（这正是常见初始化 + LayerNorm 之后的近似状态）。内积是 $d_k$ 项之和：

#formula[
  $ q dot k = sum_(i=1)^(d_k) q_i k_i, quad EE[q_i k_i] = EE[q_i] EE[k_i] = 0 \
    "Var"(q_i k_i) = EE[q_i^2 k_i^2] = EE[q_i^2] EE[k_i^2] = 1 \
    "Var"(q dot k) = sum_(i=1)^(d_k) "Var"(q_i k_i) = d_k, quad sigma = sqrt(d_k) $
]

所以 logits 的标准差随 $sqrt(d_k)$ 增长：$d_k = 128$ 时典型幅度已经是 $plus.minus 11$，softmax 里最大项和次大项能差好几个 $e$ 的量级，分布接近 one-hot。

关键在下一步：*softmax 饱和时梯度会消失*。它的 Jacobian 是 $"diag"(p) - p p^T$，迹为 $sum_i p_i (1-p_i)$——这个量就是"能往下游传的梯度通道量"，$p$ 接近 one-hot 时趋于 0，$p$ 接近均匀时趋于上界 $1 - 1\/S$。

#formula[$ "tr"(J_"softmax") = sum_i p_i (1 - p_i) arrow.r cases(0 quad &"one-hot（饱和，梯度死）", 1 - 1\/S quad &"均匀（梯度最健康）") $]

除以 $sqrt(d_k)$ 把 logits 方差拉回 1，softmax 就停在梯度良好的区域。配套测试把这个论断量化了：$d_k = 256$、64 个 key 的随机 q/k 下，不缩放时注意力分布的熵 $< 0.5$（上界是 $log 64 approx 4.16$）、Jacobian 迹 $< 0.25$；缩放后熵 $> 0.8 log 64$、迹 $> 0.9$——*梯度通道量差了 5 倍以上*。

#warn[
  两个常见错法：除以 $d_k$（过度缩放，logits 全被压到 0 附近，attention 永远接近均匀，学不出选择性）；除以 $sqrt(d_"model")$（当 $d_"model" = n_"head" times d_"head"$ 时缩放过头 $sqrt(n_"head")$ 倍）。*必须是每个 head 的维度 $d_"head"$*。
]

=== mask 用 `-inf` 还是 `finfo.min`

#warn[
  *整行被 mask 掉时 `-inf` 会产生 NaN。* `softmax` 对全 $-infinity$ 的一行算的是 $exp(-infinity) = 0$ 全零，再除以和 0，得到 `0/0 = NaN`。然后 NaN 顺着反向传播把整个模型的梯度污染成 NaN，而 loss 曲线上看起来就是"训到某一步突然变 nan"。

  什么时候会遇到整行被 mask？padding mask 碰上"整条序列都是 pad"的样本（变长 batch + 数据清洗不干净）；或者 cross-attention 里源句为空。生产代码的两个选择：
  - 保证每行至少一个可见位置（在 dataloader 侧过滤空序列，或强制 pad 位置的第 0 列可见）；
  - 用 `torch.finfo(logits.dtype).min` 代替 `-inf`。全 mask 行会得到均匀分布——是无意义的值，但*有限*，不会污染梯度。

  另一个相关的错法是用魔数 `-1e4`。它在 fp16 里是可表示的"安全值"，但如果真实 logits 也在 $10^4$ 量级，被 mask 的位置*不会被完全屏蔽*，信息悄悄泄漏。要么 `-inf`，要么 `finfo.min`，不要自己编数。
]

=== dropout 加在哪，以及怎么自证

dropout 加在 softmax *之后*、乘 $V$ *之前*，即随机丢掉一部分 attention 权重。PyTorch 的 inverted dropout 会除以 $1-p$，推理时什么都不用做。常见错法是加在 logits 上——那等价于随机*屏蔽* key（被丢的概率会重新分配给其他 key），不是"减弱"而是"删除"，语义完全不同。

*怎么自证*：无 mask、`is_causal`、bool mask、float mask 四条路径分别和 `F.scaled_dot_product_attention` 比对；断言 `attn.sum(-1) == 1`；causal 下断言严格上三角*精确*为 0（`attn.triu(diagonal=1).abs().max() == 0.0`，不是"很小"）。

== MultiHeadAttention：那个静默出错的 reshape

*题目*：手写 MHA，要求能加载 `nn.MultiheadAttention` 的权重并数值对齐。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model, self.n_head = d_model, n_head
        self.d_head = d_model // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)   # fused
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, attn_mask=None, is_causal=False):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)      # 每个 (B,S,H)

        # 唯一正确的 split-head 写法
        q = q.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, self.d_head).transpose(1, 2)

        out, _ = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout,
            is_causal=is_causal, training=self.training)

        # merge：先 transpose 回去，再 contiguous 才能 view
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)
```

#figure(
  align(center, shape-pipeline(stages: (
    ("x", "(B, S, d_model)", "hidden states"),
    ("qkv(x)", "(B, S, 3*d_model)", "一次 fused Linear"),
    ("chunk(3, -1)", "(B, S, d_model) x3", "切成 q / k / v"),
    ("view", "(B, S, n_head, d_head)", "只拆最后一维，不动 S"),
    ("transpose(1,2)", "(B, n_head, S, d_head)", "head 提到 batch 维"),
    ("attention", "(B, n_head, S, d_head)", "每个 head 独立做 SDPA"),
    ("transpose(1,2)", "(B, S, n_head, d_head)", "非连续，必须 contiguous"),
    ("view + out_proj", "(B, S, d_model)", "merge 回去"),
  ))),
  caption: [MHA 的 shape 流。第 4→5 步*必须*是 `view` 再 `transpose` 两步，不能一步 `view(B, n_head, S, d_head)`。],
) <fig-mha-shape>

#warn[
  *全章最经典的 bug*：
  ```python
  q = q.view(b, self.n_head, s, self.d_head)   # 错！但 shape 完全正确
  ```
  Linear 输出最后一维的内存排布是 `[head0_d0 ... head0_d(dh-1), head1_d0, ...]`，即 *head 维在 `d_head` 维外侧、在 `S` 维内侧*。正确写法只切最后一维，语义是"把每个 token 的 hidden 分给各个 head"。错误写法把 `S * d_model` 这段连续内存重新切成 `n_head * (S * d_head)`，语义变成"第 0 个 head 拿走了前 $S \/ n_"head"$ 个 token 的*全部*通道"——相邻 hidden 维被错切给不同 head，序列也被切碎。

  它之所以经典：*shape 完全正确、不报任何错、loss 也会降一点*（毕竟还是个可微变换），只是永远学不到该有的水平。配套测试用 `wrong[0,0].reshape(-1) == x[0].reshape(-1)[:s*dh]` 把它钉死，直接展示错法拿到的是哪一段内存。

  merge 回去时同理：`out.transpose(1,2)` 之后 stride 不连续，*少了 `.contiguous()` 会让 `view` 直接报错*。这反而是好事——它至少会响。用 `reshape` 能绕过报错，但那是隐式拷贝，不如显式写清楚。
]

*fused qkv 的好处*：一次 GEMM 代替三次，kernel launch 少 2/3，且 $M \/ N \/ K$ 更大更容易打满 tensor core。代价是加载别人的预训练权重时要自己拼 `cat([Wq, Wk, Wv], dim=0)`，顺序必须是 q, k, v。

#insight[
  *考点：`nn.MultiheadAttention` 的权重是怎么存的。* 当 `embed_dim == kdim == vdim`（默认情况）时，它把三个投影*打包成一个* `in_proj_weight`，形状 `(3E, E)`，行方向按 `[Wq; Wk; Wv]` 顺序拼接（`in_proj_bias` 同理是 `(3E,)`），此时 `q_proj_weight` / `k_proj_weight` / `v_proj_weight` 全是 `None`。只有 `kdim` / `vdim` 与 `embed_dim` 不同时才拆成三个独立的 `*_proj_weight`，那时 `in_proj_weight` 是 `None`。所以如果你的 `self.qkv` 也是 `Linear(E, 3E)`，可以整块 `copy_(ref.in_proj_weight)` 直接对齐；写权重迁移脚本时不知道这个分支逻辑，就会在 `q_proj_weight is None` 上翻车。
]

#warn[
  *考点：两套 API 的 bool mask 语义正好相反。*
  - `F.scaled_dot_product_attention(attn_mask=bool)`：*`True` = 保留*
  - `nn.MultiheadAttention(attn_mask=bool)`：*`True` = 屏蔽*

  所以同一个 causal 约束，一边要传 `tril`（下三角 True），另一边要传 `triu(diagonal=1)`（上三角 True）。迁移时忘了取反，得到的是*反向 causal*：每个位置只能看未来、看不到过去。这个事故的特征非常好记——*loss 降得离奇地快*，因为位置 $t$ 能看到 $t+1$ 的 token，而训练目标恰好就是预测 $t+1$，模型直接在偷看答案。开发集指标也很好（同样的 mask），只有真正自回归生成时才发现全是胡言乱语。看到"loss 比论文低一大截"，第一件事就是查 mask 方向。
]

*怎么自证*：`mine.qkv.weight.copy_(ref.in_proj_weight)` 等四行对拷之后，双方 `eval()`，比对 `ref(x, x, x, need_weights=False)[0]` 和 `mine(x)`；causal 情况用上面说的 `triu` / `tril` 分别传，同样要逐元素相等。

== MHA / MQA / GQA：一条连续的谱

三者的区别只有一个数字：`n_kv_head`。

#table(
  columns: (auto, auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, left, left),
  [], [`n_kv_head`], [KV cache 相对大小], [代表模型],
  [MHA], [`n_head`], [$1$], [GPT-2、Llama-1、Llama-2 7B/13B],
  [GQA], [$1 < n_"kv" < n_"head"$], [$n_"kv" \/ n_"head"$], [Llama-2 70B、Llama-3 全系（8）、Mistral],
  [MQA], [$1$], [$1 \/ n_"head"$], [PaLM、Falcon、早期 StarCoder],
)

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, dropout=0.0, bias=False):
        super().__init__()
        assert n_head % n_kv_head == 0
        self.d_model, self.n_head, self.n_kv_head = d_model, n_head, n_kv_head
        self.n_rep, self.d_head = n_head // n_kv_head, d_model // n_head
        self.dropout = dropout
        out_dim = (n_head + 2 * n_kv_head) * self.d_head   # q 全量 + kv 缩减
        self.qkv_proj = nn.Linear(d_model, out_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    @staticmethod
    def repeat_kv(x, n_rep):                 # (B,Hkv,S,D) -> (B,Hkv*n_rep,S,D)
        if n_rep == 1:
            return x
        b, h_kv, s, d = x.shape
        return (x[:, :, None, :, :]                # expand 不拷贝内存
                .expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d))

    def forward(self, x, cos=None, sin=None, attn_mask=None, is_causal=False,
                kv_cache=None, start_pos=0):
        b, s, _ = x.shape
        nh, nkv, dh = self.n_head, self.n_kv_head, self.d_head
        q, k, v = self.qkv_proj(x).split([nh * dh, nkv * dh, nkv * dh], dim=-1)
        q = q.view(b, s, nh,  dh).transpose(1, 2)
        k = k.view(b, s, nkv, dh).transpose(1, 2)
        v = v.view(b, s, nkv, dh).transpose(1, 2)

        if cos is not None:              # RoPE 只作用于 q 和 k，不作用于 v
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:         # cache 里存 RoPE 之后的 k/v
            k, v = kv_cache.update(k, v)
        k, v = self.repeat_kv(k, self.n_rep), self.repeat_kv(v, self.n_rep)

        if is_causal and attn_mask is None:   # 自己算 mask 以带上 start_pos
            attn_mask = causal_mask(s, k.shape[-2], start_pos, x.device)
            is_causal = False
        out, _ = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout,
            is_causal=is_causal, training=self.training)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)
```

#insight[
  *GQA 省的是 KV cache 显存和访存带宽，不是计算量。* `repeat_kv` 之后 attention 本身的 FLOPs *一模一样*：query head 数没变，$Q K^T$ 依然是 $n_"head"$ 个 $(S, S)$ 矩阵，省的只有 qkv 投影那一小块参数。那为什么推理明显更快？因为 *decode 阶段是彻底 memory-bound 的*：每生成一个 token 都要把整个 KV cache 从 HBM 读一遍，而计算量只有 $O(S d)$，算术强度极低；KV cache 小 $n_"head" \/ n_"kv"$ 倍，读的字节数就少同样的倍数，端到端时间基本线性跟着降。MQA（$n_"kv" = 1$）压得最狠但质量掉得明显，GQA 取 8 之类的中间值是"几乎不掉点 + 拿到大部分收益"的折中，所以成了现在的默认。
]

*KV cache 的显存公式*必须能当场算出来（前面的 2 是 K、V 两份）：

#formula[
  $ M_"KV" &= 2 dot L dot B dot S dot n_"kv" dot d_"head" dot "bytes" \
    &= 2 times 32 times 1 times 4096 times 32 times 128 times 2 "B" = 2 "GiB" $
]

第二行代的是 *Llama-2 7B*（$L = 32$，MHA 所以 $n_"kv" = 32$，$d_"head" = 128$）、fp16、$B = 1$、$S = 4096$。7B 的 fp16 权重本身是 14 GB，*一条 4K 序列的 KV cache 就要 2 GiB*——batch 开到 8 就是 16 GiB，比权重还多。换成 GQA-8（$n_"kv" = 8$）直接降到 512 MiB。这就是 KV cache 优化（GQA / MQA / MLA / KV 量化 / PagedAttention）成为推理主战场的原因。

#warn[
  *`repeat_kv` 必须是 `repeat_interleave` 语义，不是 `repeat`。*
  `repeat_interleave(2)` 给 `[k0, k0, k1, k1]`，`repeat(2)` 给 `[k0, k1, k0, k1]`。query head 是*按组连续排布*的：第 0 组的 `n_rep` 个 query head 要对上同一个 kv head。用 `repeat` 会让 query head 配错 kv head——又是一个 shape 全对、不报错、静默掉点的坑。
  上面的实现用 `expand + reshape`：`expand` 只改 stride 不拷贝内存，`reshape` 时才物化一次，比 `repeat_interleave` 少一个 kernel。
]

*怎么自证*：`n_kv_head == n_head` 时权重对拷给 MHA，两者输出必须逐元素相等（GQA 是 MHA 的严格推广）；`repeat_kv(x, n)` 断言等于 `x.repeat_interleave(n, dim=1)` 且*不等于* `x.repeat(1, n, 1, 1)`。

== RoPE：为什么它是相对位置编码

*题目*：实现旋转位置编码，并说明为什么它能表达相对位置。

*旋转公式*。把 $d$ 维特征拆成 $d\/2$ 个二维平面，第 $i$ 个平面按角度 $m theta_i$ 旋转（$m$ 是 token 的绝对位置）：

#formula[
  $ theta_i = "base"^(-2 i \/ d), quad i = 0, 1, ..., d\/2 - 1 \
    mat(x'_a; x'_b) = mat(cos m theta_i, -sin m theta_i; sin m theta_i, cos m theta_i) mat(x_a; x_b) = R(m theta_i) mat(x_a; x_b) $
]

写成复数最清爽：把 $(x_a, x_b)$ 看成复数 $z$，RoPE 就是 $z arrow.r z e^(i m theta)$。

=== 相对性的证明

这是本题的核心，务必能当场推出来。二维旋转矩阵是正交的（$R(alpha)^T = R(alpha)^(-1) = R(-alpha)$）而且可加（$R(alpha) R(beta) = R(alpha + beta)$），于是位置 $m$ 的 query 和位置 $n$ 的 key 在第 $i$ 个平面上的内积是

#formula[
  $ ⟨R(m theta) q, R(n theta) k⟩ &= q^T R(m theta)^T R(n theta) k \
    &= q^T R(-m theta) R(n theta) k \
    &= q^T R((n - m) theta) k = f(q, k, n - m) $
]

*只依赖 $n - m$*。用复数写更短：$⟨R_m q, R_n k⟩ = "Re"[(q e^(i m theta)) overline((k e^(i n theta)))] = "Re"[q overline(k) e^(i (m - n) theta)]$。

#insight[
  RoPE 的精妙之处：*编码时用绝对位置（每个 token 独立旋转，不需要成对处理），注意力里自动变成相对位置*。这让它兼有绝对编码的实现简单性和相对编码的泛化能力，而且不占任何参数、不改变 attention 的计算图形状——这就是它彻底取代可学习绝对位置编码的原因。
]

=== 实现

```python
def build_rope_cache(seq_len, dim, base=10000.0, device=None,
                     dtype=torch.float32):
    assert dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                            dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (S, dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):                  # x: (B, H, S, D)
    x1, x2 = x.chunk(2, dim=-1)               # 各 (B, H, S, D/2)
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)    # (1, 1, S, D/2)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

三个实现细节：

- *cos/sin cache*。$theta_i$ 只依赖维度、$m theta_i$ 只依赖位置，都和数据无关，所以整张 `(max_seq, d_head/2)` 的表在 `__init__` 里算一次，用 `register_buffer(..., persistent=False)` 存：随 `.to(device)` 一起搬，但不进 `state_dict`——它是可重算的常量，存进 checkpoint 只是浪费。频率必须在 fp32 里算，bf16 的 mantissa 不足以区分相邻位置的角度。
- *`base=10000`*。$theta_i$ 从 1（$i = 0$）指数衰减到 $1\/"base"$（最高维）。低维通道转得快、编码局部位置；高维通道转得慢、波长最长约 $2 pi dot "base"$，编码全局位置。base 越大，最长波长越长，能覆盖的上下文越长。
- *维度配对有两种排布*。*interleaved*（原论文）把相邻两维 $(x_0, x_1), (x_2, x_3), ...$ 配成复数；*split-half*（GPT-NeoX / HF-Llama，也是上面的实现）把前一半和后一半配对 $(x_0, x_(d\/2)), (x_1, x_(d\/2+1)), ...$。

#warn[
  两种排布都是正确的 RoPE（都是逐平面旋转，相对性证明照样成立），但*必须和权重训练时用的排布一致*。混用不报错、shape 完全一样、模型还能输出通顺的文字，只是质量静默下降——因为 $W_q$ 学到的通道配对关系被打乱了。这是移植 checkpoint 时的经典事故，`convert_llama_weights_to_hf.py` 里那段看起来莫名其妙的权重 permute 就是在做这件事。
]

=== 两个正确性测试

RoPE 有两条能直接写成断言的数学性质，比"跑一下看 loss 降不降"强得多：

+ *旋转不改变范数*。$R$ 正交，所以 `apply_rope(x).norm(dim=-1) == x.norm(dim=-1)`；更细一点，每个二维子平面的范数也单独守恒（`y1**2 + y2**2 == x1**2 + x2**2`），这证明它确实是逐平面旋转，而不是某种混了通道的线性变换。
+ *内积只依赖相对位置*。固定 $q, k$，让 $m - n = delta$ 不变而 $m$ 变，`(rope(q, m) * rope(k, n)).sum()` 必须是常数。配套测试对 $delta in {0, 1, 5, -3}$ 各扫 8 个起点断言极差 $< 10^(-4)$，同时断言不同的 $delta$ 给出*不同*的内积（否则位置信息就白编码了）。

再加一条便宜的：位置 0 的旋转角是 0，`cos[0] == 1`、`sin[0] == 0`，`apply_rope(x, cos[:1], sin[:1]) == x`。

#note[
  *长上下文外推*一句话：训练时只见过 $S_"train"$ 以内的角度，直接推到更长会因为高频通道的角度进入没见过的区间而崩。三条主流做法都是在动角度——位置插值（把位置索引缩放到训练区间）、NTK-aware（调大 `base`，让高频通道少转一点）、YaRN（按通道波长分段处理，短波长插值、长波长保留，再配一个温度修正）。
]

== mask：causal、padding，以及它们的组合

```python
def causal_mask(q_len, kv_len=None, start_pos=0, device=None):
    """True = 可见。用 q_pos >= k_pos 的通式覆盖所有情况。"""
    if kv_len is None:
        kv_len = q_len
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + start_pos  # (Sq,1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)             # (1,Sk)
    return q_pos >= k_pos


def padding_mask(lengths, max_len, device=None):
    """(B,) 的真实长度 -> (B, 1, 1, Sk) 的 key mask，True = 可见。"""
    ar = torch.arange(max_len, device=device)
    return (ar.unsqueeze(0) < lengths.to(device).unsqueeze(1)).view(-1, 1, 1, max_len)
```

训练时 `causal_mask(S)` 就是最经典的 `(S, S)` 下三角全 True，等价于 `torch.ones(S, S, dtype=torch.bool).tril()`。

#warn[
  *decode 阶段的 causal mask 不是方阵。* 用 KV cache 逐 token 解码时 `q_len=1`、`kv_len=已缓存长度 P`，这个 query 的绝对位置是 `start_pos = P-1`，它能看到*全部* $P$ 个历史 key，所以 mask 应该是全 True 的 `(1, P)`。常见 bug 是直接套 `tril(ones(1, P))`——得到"只能看第 0 个 key"，模型每步都只看到序列开头。症状很有欺骗性：prefill 阶段（$q_"len" = S$）完全正常，只有开始逐 token 解码后输出才变成重复的胡话。上面 `q_pos + start_pos >= k_pos` 的通式一次覆盖 prefill 和 decode，不需要特判。
]

*两者的组合是逻辑与，不是相加*。padding mask 的形状故意做成 `(B, 1, 1, Sk)`，causal mask reshape 成 `(1, 1, Sq, Sk)`，直接 `&` 广播成 `(B, 1, Sq, Sk)`——语义是"两个条件都可见才可见"：

```python
mask = padding_mask(lengths, S) & causal_mask(S).view(1, 1, S, S)
```

用 float 加性 mask 相加也能得到同样效果（$-infinity + (-infinity) = -infinity$），但 bool 版更省显存且没有数值问题。

*怎么自证*：causal 下断言 attention 权重的严格上三角精确为 0；组合 mask 下断言"长度为 3 的样本在 key 位置 3 之后权重全 0"*且*"前 3 个 query 行的权重仍归一"。后半句很重要——它验证了归一化是在剩下的可见位置上重新做的。

== KV cache：为什么需要，怎么实现，怎么验

自回归生成时，第 $t$ 步要算 $q_t$ 对 $k_1..k_t$ 的 attention。朴素做法是把整条 `(B, t)` 序列重新 forward 一遍再取最后一个 logit——但 $k_1..k_(t-1)$ 只依赖前缀 token 和它们的绝对位置，*和这一步无关，上一步已经算过了*。缓存下来，每步的投影计算量从 $O(t)$ 降到 $O(1)$，生成 $N$ 个 token 的总量从 $O(N^2)$ 降到 $O(N)$。这是推理侧收益最大、也最不可能不做的优化。

=== 预分配 buffer + 游标

```python
class KVCache:
    def __init__(self, batch_size, n_kv_head, max_seq_len, d_head,
                 dtype=torch.float32, device=None):
        shape = (batch_size, n_kv_head, max_seq_len, d_head)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.pos = 0        # 已填充长度，也是下一个 token 的绝对位置

    def reset(self):        # 换一条 prompt 时复用 buffer，不重新分配
        self.pos = 0

    def update(self, k, v):
        """写入新 k/v（prefill 时 S>1，decode 时 S=1），返回全部历史。"""
        n = k.shape[2]
        assert self.pos + n <= self.max_seq_len, "KV cache 溢出"
        self.k[:, :, self.pos : self.pos + n] = k
        self.v[:, :, self.pos : self.pos + n] = v
        self.pos += n
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]   # view，不拷贝
```

*为什么不用 `torch.cat`*：`cat` 每步都重新分配一块更大的显存并拷贝*整个*历史，decode $N$ 步的总拷贝量是 $O(N^2)$，还不停触发 caching allocator 去找新 block（碎片化）。预分配后每步只写一个 slot，总量 $O(N)$，返回的还是 view。代价是要提前知道 `max_seq_len`，且显存按最坏情况占满——不管实际生成多长。vLLM 的 PagedAttention 就是为了解决这个"预留浪费"而生：把 cache 切成固定大小的 block 按需分配，像操作系统的分页。

*cache 里存 RoPE 之前还是之后的 k*：*之后*。RoPE 只依赖该 token 的绝对位置，写入时算一次即可；存之前的话每步都要对整个历史重算旋转，白付 $O(N)$ 开销，KV cache 的意义就没了。

=== prefill 与 decode：两个完全不同的阶段

#insight[
  这是整个推理优化领域的核心认识。*prefill*（喂 prompt）一次输入 $S$ 个 token，所有位置并行算，GEMM 的 $M$ 维是 $B times S$，*compute-bound*，能打满 tensor core，指标是 TTFT（time to first token）。*decode*（逐 token 生成）每步只有 1 个 token，GEMM 退化成 GEMV（$M = B$），且每步都要把*全部权重 + 全部 KV cache* 从 HBM 读一遍，*memory-bound*，指标是 TPOT（time per output token）。

  瓶颈不同，优化手段就完全不同：decode 侧要提高 batch（continuous batching 把不同请求的 decode 步凑成一个大 GEMM）、压 KV cache（GQA / 量化）、投机解码（用小模型一次猜多个 token，把 memory-bound 的串行步换成 compute-bound 的并行验证）；prefill 侧要 chunked prefill（把长 prompt 切块插到 decode 的空隙里，避免长 prompt 把所有请求的 decode 卡住）。
]

=== 验证正确性的黄金测试

#insight[
  causal 下，*"prefill 一段 + 逐 token decode" 每一步的 logits 必须与"一次性 forward 整条序列"取对应位置的 logits 逐位相同*——两者数学上完全等价。这一条测试同时覆盖了 KV cache 实现的所有常见 bug：mask 的 `start_pos`、RoPE 的位置切片 `cos[start:start+s]`、cache 写入的下标，三者中任何一处错了它都会挂。
]

```python
full = model(idx)                                  # (B, S, V) 一次算完
caches, prefill = model.make_caches(b), 4
out = [model(idx[:, :prefill], caches=caches, start_pos=0)]
for t in range(prefill, s):
    out.append(model(idx[:, t:t+1], caches=caches, start_pos=t))
torch.testing.assert_close(torch.cat(out, dim=1), full, rtol=1e-5, atol=1e-5)
```

同一个思路在 `generate` 层面还有一条：`use_cache=True` 和 `use_cache=False` 的 greedy 结果必须*逐 token 相同*。

== TransformerBlock：pre-norm、RMSNorm、SwiGLU

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()                                  # 平方和必须在 fp32 算
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)               # 关键
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)   # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)   # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head=None, ffn_hidden=None):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_head, n_kv_head or n_head)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden)

    def forward(self, x, cos=None, sin=None, is_causal=True,
                kv_cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, is_causal=is_causal,
                          kv_cache=kv_cache, start_pos=start_pos)
        return x + self.ffn(self.ffn_norm(x))
```

=== pre-norm vs post-norm

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [post-norm（原论文）], [pre-norm（现在的默认）],
  [公式], [$x arrow.l "Norm"(x + "Sub"(x))$], [$x arrow.l x + "Sub"("Norm"(x))$],
  [残差通路], [每层被 Norm 重新缩放], [纯恒等映射],
  [warmup], [必需，去掉直接发散], [不敏感],
  [深度], [超过 ~20 层难训], [百层量级没问题],
  [代价], [—], [残差流方差随层数累加，需补 final norm],
)

*为什么 pre-norm 稳*：残差路径上没有任何操作，梯度可以从最后一层无损直达第一层（和 ResNet 的 $+I$ 完全同理）。post-norm 的梯度要穿过 $L$ 个 Norm，每个 Norm 的 Jacobian 都会重新缩放梯度，期望范数随深度指数变化，所以必须靠 warmup 让训练早期别走大步。*代价*是 pre-norm 每层都往残差流上加东西而没人归一化，$x$ 的方差随层数累加，深层子层的输出相对于已经很大的残差流只是"小扰动"，相对贡献被稀释（有人叫 representation collapse）。所以 pre-norm 结构最后*必须*补一个 final norm，否则 `lm_head` 拿到的输入尺度会随层数漂移。DeepNorm、sandwich-norm 是两种折中。

#insight[
  *pre-norm 的残差通路是恒等*这句话可以直接写成断言：把 `attn.out_proj.weight` 和 `ffn.w2.weight` 全部置零（两个子层输出恒为 0），则 `block(x) == x` *精确成立*。post-norm 做同样的事得到的是 `Norm(x) != x`。一行代码就把两种结构的本质差别测出来了。
]

=== RMSNorm 与 SwiGLU

*RMSNorm* 与 LayerNorm 的差别：*不减均值、没有 bias*，只除以均方根。少一次 reduce（对 memory-bound 的 norm kernel 是实打实的收益），实测效果与 LayerNorm 相当，所以 Llama 之后基本都用它。$"mean"(x^2)$ 要在 fp32 里算——bf16 下平方容易损失精度，而且这个量在分母上，误差会被放大。

*SwiGLU 为什么是 3 个矩阵*：GLU 家族用一路做"门控"、一路做"内容"，逐元素相乘，即 $W_2 ("SiLU"(W_1 x) dot.op W_3 x)$。门控让 FFN 有了乘性交互（普通 FFN 只有加性），实测比同参数量的 ReLU/GELU FFN 好一截。

*中间维为什么取 $8 d \/ 3$*。普通 2 矩阵 FFN 取 hidden $= 4d$ 时参数量是 $2 dot d dot 4d = 8 d^2$；SwiGLU 有 3 个 $d times h$ 规模的矩阵，要参数量持平就得

#formula[$ 3 d h = 8 d^2 arrow.r.double h = 8/3 d approx 2.67 d $]

Llama 还会把它向上对齐到 256 的倍数（对 GEMM 的 tile 友好），所以 Llama-2 7B（$d = 4096$）的 FFN hidden 是 11008 而不是 $8 times 4096 \/ 3 = 10922.7$。

#warn[
  常见错法：直接把 SwiGLU 的 hidden 设成 $4d$。参数量变成 $3 dot 4 d^2 = 12 d^2$，比对照组多 50%——然后你拿它和 GELU FFN 比说"SwiGLU 更好"，实际上比的是一个更大的模型。这是论文复现里的经典不公平对比。
]

== MiniGPT 与参数量估算

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, n_kv_head=None,
                 ffn_hidden=None, max_seq_len=512, tie_weights=False,
                 rope_base=10000.0):
        super().__init__()
        self.n_kv_head, self.d_head = n_kv_head or n_head, d_model // n_head
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, self.n_kv_head, ffn_hidden)
            for _ in range(n_layer)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(max_seq_len, self.d_head, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)   # 不进 ckpt
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, caches=None, start_pos=0):
        s = idx.shape[1]
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos : start_pos + s]
        sin = self.rope_sin[start_pos : start_pos + s]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, is_causal=True, start_pos=start_pos,
                      kv_cache=None if caches is None else caches[i])
        return self.lm_head(self.final_norm(x))

    def make_caches(self, batch_size, max_seq_len=None):
        w = self.tok_emb.weight                      # 跟着模型的 dtype/device
        return [KVCache(batch_size, self.n_kv_head,
                        max_seq_len or self.max_seq_len, self.d_head,
                        w.dtype, w.device) for _ in self.blocks]
```

注意*没有位置 embedding*——位置信息全靠 RoPE 在每层的 attention 里注入。这是 Llama 系的标准做法，也解释了为什么 RoPE 模型换上下文长度不需要改任何参数形状。

*tied embedding 的取舍*：`lm_head.weight = tok_emb.weight` 省下 $V d$ 个参数。小模型上这块能占很大比例（$V = 32000$、$d = 768$ 时是 24.6M，而 12 层 block 才 85M），tie 了显然值得，还附带"输入输出共享同一套词表示"的正则效果。大模型（Llama 2/3）反而不 tie，因为 $V d$ 相对总量已经很小，解绑能多一点自由度。注意 tie 之后 embedding 的梯度会同时来自查表和输出投影两条路径。

=== 参数量估算公式

按上面的实现逐块数（忽略 norm 的 $d$ 个参数和所有 bias）：

#formula[
  $ "attn" &= underbrace((n_"head" + 2 n_"kv") d_"head" dot d, "fused qkv") + underbrace(d^2, "out proj") arrow.r 4 d^2 quad ("MHA 时") \
    "ffn" &= underbrace(3 d h, "SwiGLU 三个矩阵") = 8 d^2 quad (h = 8d\/3) \
    "per layer" &= 4 d^2 + 8 d^2 = 12 d^2 \
    N_"total" &approx 12 L d^2 + 2 V d quad (V d "只算一次，如果 tie") $
]

*$12 L d^2$ 是必须记住的数*。验算 Llama-2 7B（$L = 32$，$d = 4096$，$V = 32000$）：$12 times 32 times 4096^2 = 6.44 times 10^9$，加 $2 V d = 0.26 times 10^9$ 得 $6.70 times 10^9$，实际是 6.74B（差异来自 FFN hidden 取了 11008 而非 10922）。两个直接推论：*参数量对 $d$ 是平方、对 $L$ 是线性*，所以加宽比加深涨得快得多；$12 d^2$ 里 attention 只占 $4 d^2$，*FFN 才是参数量的大头*——这也是 MoE 只把 FFN 换成专家的原因。

配套测试把公式落到了代码上：手算 `per_block = d + (n_head + 2*n_kv)*d_head*d + d*d + d + 3*d*ffn_h`，断言它乘层数再加 emb / final norm / head 之后*精确等于* `sum(p.numel() for p in model.parameters())`。手算和 `numel()` 对不上，说明你对自己模型的结构理解有偏差。

== generate：greedy / temperature / top-k / top-p

```python
def top_k_top_p_filter(logits, top_k=None, top_p=None):
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1, None]     # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        cumprobs = probs.cumsum(-1)
        remove = cumprobs - probs >= top_p    # 累计概率已超过 p 之后的才删
        remove[..., 0] = False                # 永远保留概率最高的那个
        remove = remove.scatter(-1, sorted_idx, remove)        # 还原到原始序
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None,
             top_p=None, use_cache=True, generator=None):
    model.eval()
    caches = model.make_caches(idx.shape[0]) if use_cache else None
    cur = idx
    for step in range(max_new_tokens):
        if use_cache:
            if step == 0:                                  # prefill
                inp, start = cur, 0
            else:                                          # decode：只喂 1 个
                inp, start = cur[:, -1:], caches[0].pos
            logits = model(inp, caches=caches, start_pos=start)
        else:
            logits = model(cur[:, -model.max_seq_len:])

        logits = logits[:, -1, :].float()                  # 采样统一在 fp32 做
        if temperature == 0.0:
            next_tok = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_top_p_filter(logits, top_k, top_p)
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1,
                                         generator=generator)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur
```

*顺序是固定的*：先 temperature，再 top-k / top-p，最后 softmax 采样。先截断再除温度会改变截断的边界，得到的不是你想要的候选集。

*temperature*：$"softmax"("logits" \/ T)$。$T arrow.r 0$ 时分布退化成 one-hot，*等价于 greedy*（argmax）；$T arrow.r infinity$ 时变成词表上的均匀分布。$T < 1$ 更保守、$T > 1$ 更发散。实现上 $T = 0$ 必须单独走 argmax 分支，不然会除零。

*top-k vs top-p*：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [top-k], [top-p（nucleus）],
  [规则], [保留概率最大的 $k$ 个], [保留累计概率刚超过 $p$ 的最小集合],
  [候选集大小], [固定], [自适应],
  [分布很尖时], [强行引入本该排除的低概率词], [集合自动缩小到 1-2 个],
  [分布很平时], [砍得太狠], [集合自动放大],
)

#warn[
  *top-p 有两个必须写对的细节。*

  一是*截断判据*。要删的是"排在我前面的累计概率已经够了"的 token，即 `cumprobs - probs >= top_p`；写成 `cumprobs >= top_p`（含自己）会把恰好跨过阈值的那个 token 也删掉，候选集少一个。

  二是*至少保留 1 个 token*。$p$ 取得极小时（$p < $ 最大概率）上面的判据会把整行都标成 remove，所以必须显式 `remove[..., 0] = False` 兜底。漏了这行的后果是 `multinomial` 拿到全 0 的概率向量，报 `invalid multinomial distribution`。

  还有一步容易忘：排序后的 mask 要用 `scatter(-1, sorted_idx, remove)` 还原到*原始词表顺序*才能 apply 到 `logits` 上。忘了它会屏蔽掉一批随机 token，采样结果"还算通顺但偶尔莫名其妙"。
]

*怎么自证*：三条互相独立的等价性——`temperature=0` 与 `temperature=1e-5` 结果逐 token 相同；`top_k=1` 与 greedy 相同（只剩一个候选，采样必然选中它）；`top_p` 取极小值也与 greedy 相同（兜底逻辑生效）。加上前面说的 `use_cache=True/False` 一致性，`generate` 的正确性基本被锁死了。

== 复杂度：什么时候 attention 才是瓶颈

按 forward、每层、batch 为 1 算（$S$ 是序列长度，$d$ 是 hidden，$h = 8d\/3$）：

#formula[
  $ "FLOPs"_"attn-score" &= underbrace(2 S^2 d, Q K^T) + underbrace(2 S^2 d, "attn" dot V) = 4 S^2 d \
    "FLOPs"_"linear" &= underbrace(2 S dot 4 d^2, "qkv + out proj") + underbrace(2 S dot 8 d^2, "SwiGLU") = 24 S d^2 \
    "ratio" &= (4 S^2 d) / (24 S d^2) = S / (6 d) $
]

也就是说 *$S$ 超过 $6d$ 之后 attention 的打分部分才成为 FLOPs 主项*；causal mask 下只算下三角，这一项再减半，门槛推到 $S approx 12 d$。$d = 4096$ 时那是 $S approx 5 times 10^4$——所以在 8K、32K 这些常见上下文下，*FLOPs 主体仍然是那些 $d^2$ 的线性层*，说"attention 是 $O(S^2)$ 所以长上下文算不动"并不准确。

真正的瓶颈是*激活显存*：attention logits 是 $B dot n_"head" dot S^2$ 个元素，每层都要为反向保存一份。$B = 1$、$n_"head" = 32$、$S = 8192$、fp16 就是 $32 times 8192^2 times 2 = 4$ GiB，*每层*；32 层 128 GiB，单卡直接爆掉。这才是长上下文的真瓶颈。

#insight[
  *FlashAttention 不改变 FLOPs，改变的是 HBM 访存。* 它把 $Q K^T$ 分块、用 online softmax 在 SRAM 里增量归约，*从不把 $S times S$ 的矩阵物化到 HBM*，把 attention 的激活显存从 $O(S^2)$ 压到 $O(S)$（只存 softmax 的 statistics），同时因为省掉了那几次巨大的 HBM 往返，实际也更快。反向重算需要的中间量而不是读回来——用 FLOPs 换带宽，正是 memory-bound kernel 的标准套路。tiling、online softmax、访存分析的细节见 `books/cuda/`。
]

== 面试考点

#interview[
  *Q1*：attention 为什么要除以 $sqrt(d_k)$？除以 $d_k$ 行不行？

  A：$q, k$ 各分量独立零均值单位方差时，$q dot k = sum_i q_i k_i$ 是 $d_k$ 个方差为 1 的独立项之和，方差是 $d_k$、标准差 $sqrt(d_k)$。不缩放的话 logits 幅度随 $sqrt(d_k)$ 增长，softmax 进入饱和区，而它的 Jacobian $"diag"(p) - p p^T$ 在 one-hot 附近趋于 0，梯度消失、训练早期就学不动。除 $sqrt(d_k)$ 正好把方差拉回 1。除以 $d_k$ 是过度缩放，logits 全被压到 0 附近，attention 永远接近均匀分布，学不出选择性。注意用的是*每个 head 的* $d_"head"$，不是 $d_"model"$。
]

#interview[
  *Q2*：MHA 里 split head 为什么必须 `view(B,S,H,D).transpose(1,2)` 而不能 `view(B,H,S,D)`？

  A：Linear 输出最后一维的排布是 head 在 `d_head` 外侧、在 `S` 内侧，所以只能切最后一维再把 head 换到前面。直接 `view(B,H,S,D)` 会把 `S * d_model` 这段内存重新切分，语义变成"第 0 个 head 拿走前 $S\/H$ 个 token 的全部通道"——相邻 hidden 维被错切给不同 head，序列也被切碎。它 shape 完全正确、不报错、loss 还会降一点，只是永远学不到该有的水平，属于最难发现的一类 bug。
]

#interview[
  *Q3*：`nn.MultiheadAttention` 的 `attn_mask` 和 `F.scaled_dot_product_attention` 的有什么区别？

  A：bool mask 的语义*正好相反*。`F.scaled_dot_product_attention` 里 `True = 保留`，`nn.MultiheadAttention` 里 `True = 屏蔽`。同一个 causal 约束一边传 `tril` 一边传 `triu(diagonal=1)`。迁移时忘了取反会得到反向 causal——每个位置只能看未来，而训练目标就是预测下一个 token，等于直接偷看答案，症状是 loss 降得离奇地快、开发集也很好，只有真正自回归生成才暴露。
]

#interview[
  *Q4*：GQA 相比 MHA 省了什么？为什么推理更快？

  A：省的是 *KV cache 显存和访存带宽*，不是计算量。`repeat_kv` 之后 attention 的 FLOPs 一模一样（query head 数没变）。之所以更快，是因为 decode 阶段彻底 memory-bound：每生成一个 token 要把整个 KV cache 从 HBM 读一遍，KV cache 小 $n_"head"\/n_"kv"$ 倍，读的字节数就少同样的倍数。顺带能省下的还有 qkv 投影里 k/v 那部分参数。
]

#interview[
  *Q5*：估算一下 7B 模型的 KV cache 有多大。

  A：$M = 2 L B S n_"kv" d_"head" dot "bytes"$。Llama-2 7B：$L=32$、MHA 所以 $n_"kv"=32$、$d_"head"=128$，fp16、$B=1$、$S=4096$，代入是 2 GiB。权重本身 14 GB，batch 开到 8 时 KV cache（16 GiB）就超过权重了。这就是 GQA / MQA / MLA / KV 量化 / PagedAttention 这一整条优化线存在的理由。
]

#interview[
  *Q6*：RoPE 为什么是相对位置编码？

  A：RoPE 把特征拆成 $d\/2$ 个二维平面，位置 $m$ 的向量在第 $i$ 个平面上乘旋转矩阵 $R(m theta_i)$。二维旋转正交且可加，所以 $R(m theta)^T R(n theta) = R(-m theta) R(n theta) = R((n-m) theta)$，于是 $⟨R_m q, R_n k⟩ = q^T R((n-m)theta) k$，*只依赖 $n - m$*。精妙之处是编码时用绝对位置（每个 token 独立旋转），注意力里自动变成相对位置，兼有两者优点且不占参数。
]

#interview[
  *Q7*：怎么验证自己写的 RoPE 是对的？

  A：两条数学性质直接写成断言。一是*旋转不改变范数*：`apply_rope(x).norm(-1) == x.norm(-1)`，而且每个二维子平面的范数单独守恒（证明它确实是逐平面旋转）。二是*内积只依赖相对位置*：固定 $q, k$，让 $m - n$ 不变而 $m$ 变，内积必须是常数，同时不同的 $m-n$ 要给出不同的内积。再加一条便宜的：位置 0 的旋转是恒等。
]

#interview[
  *Q8*：KV cache 为什么要预分配，而不是每步 `torch.cat`？

  A：`cat` 每步重新分配更大的显存并拷贝整个历史，decode $N$ 步的总拷贝量 $O(N^2)$，还不停触发 allocator 造成碎片。预分配 `(B, n_kv, max_seq, d_head)` 的 buffer + 一个 `pos` 游标，每步只写一个 slot、返回 view，总量 $O(N)$。代价是要提前知道 `max_seq_len` 且按最坏情况占满显存——PagedAttention 就是把 cache 切成固定 block 按需分配来解决这个浪费。另外 cache 里存的应该是 *RoPE 之后* 的 k/v，否则每步都要对整个历史重算旋转。
]

#interview[
  *Q9*：prefill 和 decode 有什么区别？为什么这个区别重要？

  A：prefill 一次处理 $S$ 个 token，GEMM 的 $M$ 维是 $B times S$，*compute-bound*，指标是 TTFT；decode 每步只有 1 个 token，GEMM 退化成 GEMV，每步都要把全部权重和 KV cache 从 HBM 读一遍，*memory-bound*，指标是 TPOT。瓶颈不同所以优化手段完全不同：decode 靠提高 batch（continuous batching）、压 KV cache、投机解码；prefill 靠 chunked prefill 避免长 prompt 卡住其他请求的 decode。这是 LLM 推理优化的第一性认识。
]

#interview[
  *Q10*：pre-norm 和 post-norm 的区别？为什么现在都用 pre-norm？

  A：post-norm 是 $x arrow.l "Norm"(x + "Sub"(x))$，pre-norm 是 $x arrow.l x + "Sub"("Norm"(x))$。pre-norm 的残差通路是*纯恒等映射*，梯度能从顶层无损直达底层，训练稳定、对 warmup 不敏感；post-norm 的梯度要穿过 $L$ 个 Norm，期望范数随深度指数变化，必须靠 warmup 才能训起来，深了容易发散。代价是 pre-norm 的残差流方差随层数累加，深层子层的相对贡献被稀释，所以最后必须补一个 final norm。可以一行代码验证：把两个子层的输出投影置零，pre-norm 的 `block(x)` 精确等于 `x`。
]

#interview[
  *Q11*：SwiGLU 为什么有 3 个矩阵？中间维为什么取 $8d\/3$？

  A：GLU 家族一路做门控、一路做内容，逐元素相乘引入乘性交互，所以是 gate / up / down 三个矩阵：$W_2("SiLU"(W_1 x) dot.op W_3 x)$。普通 2 矩阵 FFN 取 hidden $=4d$ 时参数量 $8d^2$；SwiGLU 要持平就解 $3 d h = 8 d^2$ 得 $h = 8d\/3$。直接设 $h = 4d$ 会让参数量变成 $12d^2$，多 50%，那样比出来的"SwiGLU 更好"是不公平对比。Llama 还会把 $8d\/3$ 向上对齐到 256 的倍数，所以 7B 的 FFN hidden 是 11008。
]

#interview[
  *Q12*：Transformer 的复杂度是 $O(S^2)$，那 8K 上下文的瓶颈是 attention 吗？

  A：*FLOPs 上不是*。每层 attention 打分是 $4 S^2 d$，线性层是 $24 S d^2$，比值 $S\/(6d)$；causal 下再减半，门槛是 $S approx 12 d$。$d = 4096$ 要 $S approx 5 times 10^4$ 才让 attention 成为 FLOPs 主项，8K 时线性层仍占绝大多数。真正的瓶颈是*激活显存*：attention logits 是 $B n_"head" S^2$ 个元素，$B=1$、32 head、$S=8192$、fp16 就是每层 4 GiB。FlashAttention 解决的正是这个——它不改变 FLOPs，而是分块 + online softmax 让 $S times S$ 矩阵从不物化到 HBM，激活从 $O(S^2)$ 降到 $O(S)$，顺带因为省掉巨大的 HBM 往返而更快。
]
