"""PyTorch 面试手写题：进阶模型。

覆盖 CV / 多模态 / 生成 / PEFT 四条线上最常被要求手写的模块：
  1. ViT: PatchEmbed(conv trick) + cls token + 可学习位置编码
  2. CLIP: 对称 InfoNCE 对比损失
  3. MoE: top-k gating + dispatch + Switch 的 load-balancing aux loss
  4. DDPM: beta schedule / q_sample 闭式解 / 噪声预测 loss / 反向采样
  5. LoRA: W + (alpha/r) B A，B 零初始化

跑得快是刻意的：所有网络都是 toy 尺寸，考点在**逻辑**不在**规模**。
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

# =============================================================================
# 1. Vision Transformer
# =============================================================================


class PatchEmbed(nn.Module):
    """图像切 patch + 线性投影，用一个 conv2d 一步到位。

    面试要点：**为什么 conv2d(kernel=P, stride=P) 就等于"切 patch + Linear"？**
        ViT 的定义是：把 ``(C, H, W)`` 切成 ``(H/P) * (W/P)`` 个 ``(C, P, P)``
        的不重叠块，每块展平成 ``C*P*P`` 维再过一个共享的 ``Linear(C*P*P, D)``。
        而 ``Conv2d(C, D, kernel_size=P, stride=P)`` 的每个输出位置，
        正好是"取一个 P×P 且互不重叠的感受野、与权重做内积"——
        卷积核权重 ``(D, C, P, P)`` reshape 成 ``(D, C*P*P)`` 就是那个 Linear 的权重。
        **stride == kernel_size 是关键**：保证不重叠、不遗漏。
        好处是省掉显式的 unfold/reshape，一个 kernel 搞定，且对 cudnn 友好。
        （见 `test_patch_embed_equals_unfold_plus_linear`，直接把两条路跑成一样。）

    面试要点：**patch 数怎么算？** ``N = (H // P) * (W // P)``，
        加上 cls token 后序列长度是 ``N + 1``。
        attention 是 O(N^2)，所以 P 从 16 减到 8，序列长 4 倍、计算量 16 倍——
        这就是 ViT 难上高分辨率、要靠 Swin / 窗口注意力的原因。
    """

    def __init__(
        self, img_size: int = 32, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 48
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "图像边长必须能被 patch_size 整除"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B N D"]:
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """标准 ViT block：pre-norm LayerNorm + MHA + MLP(GELU)。

    面试要点：**和 GPT block 的区别只有两点**——
        (1) 没有 causal mask（图像 patch 之间是双向可见的）；
        (2) 用 LayerNorm + GELU MLP 而不是 RMSNorm + SwiGLU（历史原因，非本质）。
        位置编码用**可学习绝对位置**而不是 RoPE，因为图像的位置是二维的，
        改分辨率时要对位置编码做二维插值（``interpolate_pos_encoding``），
        这是微调 ViT 时最容易忘的一步。
    """

    def __init__(self, dim: int, n_head: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Float[Tensor, "B N D"]) -> Float[Tensor, "B N D"]:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    """PatchEmbed -> [cls] + pos -> N x ViTBlock -> norm -> 取 cls -> head。

    面试要点：**cls token 是什么？**
        一个可学习的、与输入无关的向量，拼在序列最前面。
        它自己没有图像内容，只能通过 attention 从各个 patch 聚合信息，
        因此天然成为"全局表示"。分类头只接它。
        替代方案是对所有 patch 做 mean pooling（DeiT III / SigLIP 用这个），
        效果相当甚至更好，还省一个 token；cls token 更多是从 BERT 继承的传统。

    面试要点：**位置编码为什么必须有？**
        attention 对输入顺序是**置换等变**的，不加位置编码的话
        把 patch 打乱结果完全不变——图像的空间结构就丢了。
        （见 `test_vit_without_pos_embed_is_permutation_equivariant`。）
        注意 cls token 也要占一个位置编码槽位，所以 pos_embed 是 ``(1, N+1, D)``。
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 48,
        depth: int = 2,
        n_head: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        # ViT 原论文用 trunc_normal(std=0.02) 初始化这两个
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([ViTBlock(embed_dim, n_head) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: Float[Tensor, "B C H W"], use_pos: bool = True
    ) -> Float[Tensor, "B K"]:
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)  # expand 不拷贝内存
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        if use_pos:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # 只取 cls token


# =============================================================================
# 2. CLIP 对比损失
# =============================================================================


def clip_contrastive_loss(
    image_emb: Float[Tensor, "B D"],
    text_emb: Float[Tensor, "B D"],
    logit_scale: float | Float[Tensor, ""] = 1.0,
) -> tuple[Float[Tensor, ""], Float[Tensor, "B B"]]:
    r"""CLIP 的对称 InfoNCE 损失。

    公式::

        i = normalize(image_emb),  t = normalize(text_emb)
        logits = logit_scale * i @ t.T                     # (B, B)
        labels = [0, 1, ..., B-1]                          # 对角线是正样本
        loss = (CE(logits, labels) + CE(logits.T, labels)) / 2

    面试要点：**为什么必须先 L2 normalize？**
        归一化后内积就是**余弦相似度**，取值被限制在 [-1, 1]，
        与向量模长无关。不归一化的话模型可以靠把正样本的模长撑大来降 loss，
        这是一条不学语义的捷径，训练会不稳定。
        归一化把"学表示"和"学置信度"解耦，置信度交给 logit_scale 统一控制。

    面试要点：**logit_scale 为什么是可学习的、且以 exp 参数化？**
        余弦相似度只在 [-1, 1]，直接送 softmax 分布几乎是均匀的，梯度极小，
        必须乘一个大的温度倒数（CLIP 训练收敛到约 100，即 temperature ≈ 0.01）。
        参数化成 ``exp(t)`` 是为了 (a) 保证恒正；(b) 让优化在对数尺度上进行，
        对这种跨数量级的标量更稳。CLIP 还会把它 clamp 到 ``ln(100)`` 以内防跑飞。

    面试要点：**为什么要对称（两个方向都算）？**
        单向 image→text 只约束"给定图片能选对文本"，
        模型可以把所有文本挤到一起、只让图片可分（表示坍缩的一种）。
        双向同时约束两个方向的检索，等价于把 (B, B) 相似度矩阵
        按行和按列都逼近单位阵，得到的是真正对齐的联合空间。
        实践上两个方向 loss 通常差不多，但去掉一个会明显掉检索指标。

    面试要点：**为什么 batch size 对对比学习特别重要？**
        每个样本的负样本数就是 ``B - 1``，全部来自 batch 内。
        InfoNCE 是互信息的下界，这个下界的上限就是 ``log(B)``——
        **B 越大，loss 能提供的信息量上界越高**。所以 CLIP 用 32k batch，
        并要用 all-gather 把所有 GPU 的 embedding 聚起来算全局 (N, N) 矩阵
        （只 gather embedding 不 gather 梯度会导致梯度错误，这是分布式实现的坑）。
        随机初始化时 loss 恰好是 ``log(B)``，是很好的 sanity check
        （见 `test_clip_loss_random_is_log_batch_size`）。

    Returns:
        (loss, logits)。返回 logits 方便测试断言和算 top-1 检索准确率。
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = logit_scale * (image_emb @ text_emb.t())  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)  # 按行：每张图找对应文本
    loss_t2i = F.cross_entropy(logits.t(), labels)  # 按列：每段文本找对应图
    return (loss_i2t + loss_t2i) / 2, logits


# =============================================================================
# 3. MoE：top-k gating + load balancing
# =============================================================================


def load_balancing_loss(
    gate_probs: Float[Tensor, "T E"],
    expert_indices: Int[Tensor, "T K"],
    num_experts: int,
) -> Float[Tensor, ""]:
    r"""Switch Transformer 的 load-balancing auxiliary loss。

    公式::

        f_e = (被路由到专家 e 的 (token, slot) 数) / (T * K)   # 硬分配的频率
        P_e = mean_t  gate_probs[t, e]                          # 软路由概率的均值
        aux = E * sum_e  f_e * P_e

    面试要点：**两项各是什么、为什么要这么配？**
        - ``f_e`` 来自 ``argmax/topk``，是**不可导**的（它只是个计数）。
        - ``P_e`` 是 softmax 的输出，**可导**，梯度就从这里流回 router。
        - 相乘的效果：某个专家如果既被大量选中（f 大）又被给了高概率（P 大），
          就会被重罚，router 因此被推着降低对它的打分，负载被摊平。
        - 常见错法：只用 ``P_e`` 的方差做 loss。那样只约束了软概率，
          实际的硬分配（真正决定显存和通信的东西）仍可能极不均衡。

    面试要点：**取值范围与最优值**
        由 Cauchy-Schwarz，当 f 和 P 都是均匀分布 ``1/E`` 时
        ``aux = E * E * (1/E)^2 = 1``，这是**最小值**；
        全部挤到一个专家时 ``aux = E * 1 * 1 = E``，是最大值。
        所以 aux 天然落在 ``[1, E]``，训练时看它是不是贴着 1 就知道路由健不健康。
        总 loss 通常是 ``task_loss + 0.01 * aux``。

    面试要点：**为什么 router 的 softmax 要在 fp32 里算？**
        (1) router logits 在训练中会被 aux loss 推得很尖，bf16 只有 8 位尾数，
            exp 之后 top-1 和 top-2 的概率可能变成完全相同的值；
        (2) **topk 的 tie-breaking 依赖精确比较**，一旦并列，不同 rank / 不同
            step 可能选出不同专家，导致 expert-parallel 下各卡路由结果不一致，
            all-to-all 直接对不上，是非常难查的非确定性 bug。
        所以 Switch/Mixtral 的实现里 router 一律 ``dtype=torch.float32``，
        算完再 cast 回去。
    """
    t, k = expert_indices.shape
    # 硬分配频率：把 (T, K) 的下标 one-hot 后在 token 维求和
    one_hot = F.one_hot(expert_indices.reshape(-1), num_experts).float()  # (T*K, E)
    f = one_hot.sum(0) / (t * k)  # (E,)，sum(f) == 1
    p = gate_probs.mean(0)  # (E,)，sum(p) == 1
    return num_experts * torch.sum(f * p)


class MoELayer(nn.Module):
    """Top-k sparse MoE 层，返回 (输出, aux_loss)。

    面试要点：**dispatch 为什么要按专家循环，而不是按 token 循环？**
        按 token 循环是 Python 级 O(T) 次小 kernel，慢到不能用。
        按专家循环是 O(E) 次、每次一个大 GEMM，E 通常只有 8~64。
        真正的生产实现还会先按专家做一次 ``argsort`` 把 token 排好序，
        变成一次 grouped GEMM（Megablocks / grouped_gemm），彻底去掉 Python 循环。

    面试要点：**capacity factor 去哪了？**
        本实现不丢 token（所有被路由的 token 都会算）。
        生产实现为了让每个专家的 buffer 是**静态形状**（TPU / expert parallel 必需），
        会设 ``capacity = capacity_factor * T * K / E``，超出的 token 直接被丢弃，
        只走残差连接。这就是 aux loss 存在的现实动机：
        负载越均衡，被丢的 token 越少。
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: Float[Tensor, "B S H"]
    ) -> tuple[Float[Tensor, "B S H"], Float[Tensor, ""]]:
        b, s, h = x.shape
        flat = x.reshape(-1, h)  # (T, H)，每个 token 独立路由

        # router 强制 fp32（见 load_balancing_loss 的 docstring）
        gate_probs = F.softmax(self.gate(flat).float(), dim=-1)  # (T, E)
        topk_probs, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        # top-k 内重新归一化，使每个 token 的权重和为 1。
        # 注意 top_k == num_experts 时这是恒等操作（原本就和为 1），
        # 所以此时会精确退化成 dense 加权和。
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)
        topk_probs = topk_probs.to(x.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            token_ids, slot_ids = torch.where(topk_idx == e)
            if token_ids.numel() == 0:
                continue
            y = self.experts[e](flat[token_ids])  # (M, H)
            out.index_add_(0, token_ids, y * topk_probs[token_ids, slot_ids, None])

        aux = load_balancing_loss(gate_probs, topk_idx, self.num_experts)
        return out.view(b, s, h), aux


# =============================================================================
# 4. DDPM
# =============================================================================


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Float[Tensor, " T"]:
    """DDPM 原论文的线性 schedule（为 T=1000 调的）。

    面试要点：直接照搬到 T=100 或小分辨率图像上会**加噪太快**，
    中间时刻信息已经毁光，模型学不到有用信号——所以 iDDPM 提出了 cosine。
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Float[Tensor, " T"]:
    r"""iDDPM 的 cosine schedule。

    先定义累积信号强度，再反推 beta::

        abar(t) = cos^2( (t/T + s) / (1 + s) * pi/2 )  /  cos^2( s/(1+s) * pi/2 )
        beta_t  = 1 - abar(t) / abar(t-1)      （clamp 到 0.999 防止数值爆炸）

    面试要点：**为什么它更好？**
        linear schedule 下 ``abar`` 在前 20% 步就掉到接近 0，
        绝大部分时间步都在"几乎纯噪声"里空转，样本利用率低。
        cosine 让 ``abar`` 在中段近似线性下降，噪声水平分布更均匀。
        小偏移 ``s`` 是为了避免 t≈0 时 beta 太小导致数值问题。
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    abar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    abar = abar / abar[0]
    betas = 1 - abar[1:] / abar[:-1]
    return betas.clamp(0.0, 0.999)


def q_sample(
    x0: Float[Tensor, "B ..."],
    t: Int[Tensor, " B"],
    noise: Float[Tensor, "B ..."],
    alphas_cumprod: Float[Tensor, " T"],
) -> Float[Tensor, "B ..."]:
    r"""前向扩散的**闭式解**：一步跳到任意时刻 t。

    公式::

        alpha_t   = 1 - beta_t
        abar_t    = prod_{s<=t} alpha_s
        q(x_t | x_0) = N( sqrt(abar_t) * x_0,  (1 - abar_t) I )
        =>  x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)

    面试要点：**为什么能一步到位？**
        逐步加噪 ``x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) eps_t``，
        把它递归展开，独立高斯的线性组合仍是高斯，方差直接相加：
        ``abar_t + (1 - abar_t) = 1``，系数正好凑成上式。
        这个性质是 DDPM 能高效训练的**根本原因**——
        训练时随机采一个 t 就能直接构造 (x_t, eps) 样本对，
        不用真的跑 t 步前向，否则训练成本是 O(T) 倍。

    面试要点：**广播怎么写？**
        ``abar`` 是 ``(T,)``，取出的 ``abar[t]`` 是 ``(B,)``，
        要 reshape 成 ``(B, 1, 1, ...)`` 才能和 ``(B, C, H, W)`` 相乘。
        常见错法是忘了补维，得到 (B, C, H, B) 的广播灾难或直接报错。
    """
    abar = alphas_cumprod[t]  # (B,)
    shape = (-1,) + (1,) * (x0.dim() - 1)  # (B, 1, 1, ...)
    abar = abar.view(shape)
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise


def ddpm_loss(
    model: nn.Module,
    x0: Float[Tensor, "B ..."],
    alphas_cumprod: Float[Tensor, " T"],
    t: Int[Tensor, " B"] | None = None,
) -> Float[Tensor, ""]:
    r"""DDPM 训练目标：**预测被加进去的噪声**。

    公式::

        t ~ Uniform{0..T-1},  eps ~ N(0, I)
        x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
        L_simple = E || eps - eps_theta(x_t, t) ||^2

    面试要点：**为什么预测噪声而不是直接预测 x0？**
        两者可以互相换算（``x0_hat = (x_t - sqrt(1-abar) eps_hat) / sqrt(abar)``），
        数学上等价，但**加权不同**：预测 eps 相当于给高噪声时刻更大的权重，
        实测样本质量更好。原论文的 ``L_simple`` 就是去掉了变分下界里
        随 t 变化的权重系数，简化成朴素 MSE——"简化反而更好"是这篇的关键经验。
        后来还有 v-prediction（预测 ``sqrt(abar) eps - sqrt(1-abar) x0``），
        在高 SNR 区间数值更稳，是 SD 2.x / 蒸馏常用的参数化。

    面试要点：**每个样本要采不同的 t**，不能整个 batch 共用一个，
        否则梯度方差巨大、等价于把 batch size 除以了时间维的多样性。
    """
    b = x0.shape[0]
    if t is None:
        t = torch.randint(0, alphas_cumprod.shape[0], (b,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    return F.mse_loss(model(x_t, t), noise)


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    shape: tuple[int, ...],
    betas: Float[Tensor, " T"],
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Float[Tensor, "B ..."]:
    r"""DDPM 的祖先采样（ancestral sampling）：从纯噪声倒着走回 x0。

    单步公式::

        x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - abar_t) * eps_theta(x_t, t) )
                  + sigma_t * z,      z ~ N(0, I)，且 **t == 0 时不加噪**

    其中 ``sigma_t^2`` 取 ``beta_t``（DDPM 论文的两个选择之一）。

    面试要点：**最后一步为什么不加噪声？**
        x_0 是最终输出，再加一次噪声等于白白往结果里掺噪点。
        对应 ``z = 0 if t == 0``，这是实现里最常漏的一行。

    面试要点：**为什么慢、DDIM 怎么救？**
        这里必须串行走 T 步，每步一次完整前向，T=1000 就是 1000 次网络调用。
        DDIM 把采样过程改写成**确定性**的非马尔可夫过程（``sigma_t = 0``），
        允许跳步（比如只走 50 步），且同一个训练好的模型可以直接复用。
    """
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    x = torch.randn(shape, device=device, generator=generator)

    for i in reversed(range(betas.shape[0])):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        mean = (x - betas[i] / (1.0 - abar[i]).sqrt() * eps) / alphas[i].sqrt()
        if i > 0:
            z = torch.randn(shape, device=device, generator=generator)
            x = mean + betas[i].sqrt() * z
        else:
            x = mean  # 最后一步不加噪
    return x


class ToyDenoiser(nn.Module):
    """一个 MLP 版的 eps_theta，带正弦时间嵌入。

    面试要点：**时间条件怎么注入？**
        用 Transformer 那套**正弦位置编码**把标量 t 映射成向量，
        再过一个小 MLP，然后加/拼到特征上（UNet 里是加到每个 ResBlock 的
        GroupNorm 之后，叫 FiLM 式调制）。
        直接把 t/T 当成一个标量特征拼进去效果差很多，
        因为网络需要在不同尺度上分辨时间，正弦编码天然提供了多尺度基。
        本类刻意用 MLP 而不是 UNet：**考点是扩散逻辑，不是网络结构**。
    """

    def __init__(self, dim: int, hidden: int = 64, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )

    def _sinusoidal(self, t: Int[Tensor, " B"]) -> Float[Tensor, "B Td"]:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self, x: Float[Tensor, "B D"], t: Int[Tensor, " B"]
    ) -> Float[Tensor, "B D"]:
        temb = self.time_mlp(self._sinusoidal(t))
        return self.net(torch.cat([x, temb], dim=-1))


# =============================================================================
# 5. LoRA
# =============================================================================


class LoRALinear(nn.Module):
    r"""LoRA 适配的 Linear::

        y = x W^T + b  +  (alpha / r) * (x A^T) B^T

    即在冻结的原权重旁挂一个低秩旁路 ``Delta W = (alpha/r) * B @ A``，
    其中 ``A: (r, in)``、``B: (out, r)``。

    面试要点：**B 为什么必须初始化为 0？**
        训练开始时必须有 ``Delta W = B @ A = 0``，让模型**精确等价于原模型**。
        否则一上来就给预训练权重叠加了一个随机扰动，
        相当于破坏了预训练知识，前几百步全在"修复"这个扰动，
        微调效果明显变差（尤其是数据少的时候）。
        A 和 B **不能都置零**——那样 ``dL/dA ∝ B^T(...) = 0``、
        ``dL/dB ∝ (...)A^T = 0``，两边梯度都恒为 0，永远学不动（对称性未破缺）。
        所以标准做法是 **A 随机（kaiming uniform）、B 全零**，
        既保证初始等价，又保证梯度非零（B 的梯度依赖 A，A 非零即可）。
        反过来（A=0, B 随机）在数学上也可行，但 A 的梯度依赖 B 的尺度，
        实践上 PEFT 库统一用前者。

    面试要点：**alpha / r 这个缩放是干嘛的？**
        让"换 r 时不用重调 learning rate"。r 越大，``B @ A`` 的元素量级
        越大（内积的项数变多），除以 r 正好抵消掉这个量级变化。
        实践中常固定 ``alpha = 2r`` 或 ``alpha = 16``。
        （后续的 rsLoRA 论证应该除以 ``sqrt(r)`` 而不是 r，在大 r 时更稳。）

    面试要点：**LoRA 到底省了什么？**
        - 可训练参数：``r*(in+out)`` vs ``in*out``，通常省 100~1000 倍。
        - **省得最多的其实是优化器状态**：AdamW 每个可训练参数要存
          m 和 v 两份 fp32，是参数本身的 8 倍。冻结 base 后这部分几乎归零。
        - **不省激活值显存**：反向仍要经过冻结的 base 权重传梯度到输入，
          前向激活照存不误。这是常见的误解。
        - 推理时可以 ``W' = W + (alpha/r) B A`` 合并回去，**零额外延迟**
          （见 `test_lora_merge_is_equivalent`）；不合并则多两次小 GEMM。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert r > 0
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)  # 冻结
        if bias:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        # A: kaiming uniform（和 nn.Linear 的默认初始化一致）；B: 全零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Float[Tensor, "... I"]) -> Float[Tensor, "... O"]:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def merged_weight(self) -> Float[Tensor, "O I"]:
        """把旁路合并进主干权重，用于零延迟部署。"""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#                                  TESTS
# =============================================================================

# ---------------------------- ViT ----------------------------


def test_patch_embed_shape_and_patch_count():
    torch.manual_seed(0)
    pe = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=48)
    assert pe.num_patches == (32 // 8) ** 2 == 16
    out = pe(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 48)

    # patch 变小 -> 序列变长（attention 是 O(N^2)，所以代价是平方级）
    pe2 = PatchEmbed(img_size=32, patch_size=4, embed_dim=48)
    assert pe2.num_patches == 64 == 4 * pe.num_patches


def test_patch_embed_equals_unfold_plus_linear():
    """证明 conv2d(k=P, s=P) 与"切块展平 + 共享 Linear"逐元素相同。"""
    torch.manual_seed(0)
    c, p, d, img = 3, 8, 48, 16
    pe = PatchEmbed(img_size=img, patch_size=p, in_chans=c, embed_dim=d)
    x = torch.randn(2, c, img, img)

    via_conv = pe(x)

    # 手动路径：unfold 出每个 (C, P, P) 块，展平后过等价的 Linear
    patches = F.unfold(x, kernel_size=p, stride=p)  # (B, C*P*P, N)
    patches = patches.transpose(1, 2)  # (B, N, C*P*P)
    w = pe.proj.weight.reshape(d, c * p * p)  # (D, C*P*P)
    via_linear = patches @ w.t() + pe.proj.bias

    torch.testing.assert_close(via_conv, via_linear, rtol=1e-5, atol=1e-5)


def test_minivit_forward_and_backward():
    torch.manual_seed(0)
    model = MiniViT(img_size=16, patch_size=4, embed_dim=32, depth=2, n_head=4, num_classes=5)
    x = torch.randn(3, 3, 16, 16)
    y = torch.randint(0, 5, (3,))

    logits = model(x)
    assert logits.shape == (3, 5)
    assert model.pos_embed.shape == (1, model.patch_embed.num_patches + 1, 32)

    F.cross_entropy(logits, y).backward()
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    # cls token 和位置编码都必须收到梯度
    assert model.cls_token.grad.abs().sum() > 0
    assert model.pos_embed.grad.abs().sum() > 0


def test_vit_without_pos_embed_is_permutation_equivariant():
    """不加位置编码时打乱 patch 顺序结果不变 —— 这就是位置编码必须存在的原因。"""
    torch.manual_seed(0)
    model = MiniViT(img_size=16, patch_size=4, embed_dim=32, depth=1, n_head=4, num_classes=5)
    model.eval()
    x = torch.randn(2, 3, 16, 16)

    # 把图像按 4x4 的 patch 网格做块置换（等价于打乱 patch 序列）
    def shuffle_patches(img: Tensor, perm: Tensor) -> Tensor:
        b, c, h, w = img.shape
        p, g = 4, h // 4
        blocks = img.view(b, c, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(b, g * g, c, p, p)
        blocks = blocks[:, perm]
        return blocks.view(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    perm = torch.randperm(model.patch_embed.num_patches)
    with torch.no_grad():
        no_pos_a = model(x, use_pos=False)
        no_pos_b = model(shuffle_patches(x, perm), use_pos=False)
        with_pos_a = model(x, use_pos=True)
        with_pos_b = model(shuffle_patches(x, perm), use_pos=True)

    torch.testing.assert_close(no_pos_a, no_pos_b, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(with_pos_a, with_pos_b, atol=1e-3)


# ---------------------------- CLIP ----------------------------


def test_clip_loss_perfect_alignment_is_near_zero():
    """图文 embedding 完全对齐（对角线相似度 1、其余接近 0）时 loss -> 0。"""
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(32, 64), dim=-1)
    loss, logits = clip_contrastive_loss(emb, emb.clone(), logit_scale=30.0)
    assert logits.shape == (32, 32)
    assert loss.item() < 1e-3
    # top-1 检索准确率 100%
    assert (logits.argmax(-1) == torch.arange(32)).all()


def test_clip_loss_random_is_log_batch_size():
    """随机 embedding 时 loss ≈ log(B) —— 对比学习最好用的 sanity check。

    直觉：高维随机向量两两近似正交，(B, B) 的 logits 几乎全 0，
    softmax 退化成均匀分布，CE = -log(1/B) = log(B)。
    训练时看到 loss 卡在 log(B) 不降，说明模型根本没学到对齐。
    """
    torch.manual_seed(0)
    for b in (32, 64):
        img = torch.randn(b, 512)
        txt = torch.randn(b, 512)
        loss, _ = clip_contrastive_loss(img, txt, logit_scale=1.0)
        assert abs(loss.item() - math.log(b)) < 0.05, (b, loss.item())


def test_clip_loss_is_symmetric_and_normalizes():
    """对称性：交换图文两侧 loss 不变；且 loss 与 embedding 模长无关。"""
    torch.manual_seed(0)
    img, txt = torch.randn(16, 32), torch.randn(16, 32)
    loss_a, _ = clip_contrastive_loss(img, txt, logit_scale=5.0)
    loss_b, _ = clip_contrastive_loss(txt, img, logit_scale=5.0)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-6)

    # L2 normalize 之后，把 embedding 整体放大 100 倍不影响任何结果
    loss_c, _ = clip_contrastive_loss(img * 100, txt * 0.01, logit_scale=5.0)
    torch.testing.assert_close(loss_a, loss_c, rtol=1e-4, atol=1e-5)


def test_clip_logit_scale_is_learnable_and_gradient_flows():
    """logit_scale 以 exp(param) 参数化，梯度能流到它自己。"""
    torch.manual_seed(0)
    log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
    img = nn.Parameter(torch.randn(8, 16))
    txt = nn.Parameter(torch.randn(8, 16))
    loss, _ = clip_contrastive_loss(img, txt, logit_scale=log_scale.exp())
    loss.backward()
    assert log_scale.grad is not None and log_scale.grad.abs() > 0
    assert img.grad.abs().sum() > 0 and txt.grad.abs().sum() > 0


# ---------------------------- MoE ----------------------------


def test_moe_forward_shape_and_backward():
    torch.manual_seed(0)
    moe = MoELayer(d_model=16, d_ff=32, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, aux = moe(x)
    assert y.shape == (2, 8, 16) and aux.shape == ()

    (y.pow(2).mean() + 0.01 * aux).backward()
    assert torch.isfinite(x.grad).all()
    # aux loss 让梯度流回 router（即使 topk 本身不可导）
    assert moe.gate.weight.grad.abs().sum() > 0


def test_load_balancing_loss_minimum_is_one():
    """完全均匀路由时 aux 取到最小值 1.0；全挤一个专家时取到最大值 E。"""
    t, e = 64, 4
    uniform_probs = torch.full((t, e), 1.0 / e)

    # 均匀硬分配：token i -> expert i % E
    idx = (torch.arange(t) % e).unsqueeze(1)  # (T, 1)，top_k=1
    aux = load_balancing_loss(uniform_probs, idx, e)
    torch.testing.assert_close(aux, torch.tensor(1.0), rtol=0, atol=1e-6)

    # 最坏情况：所有 token 都去 expert 0，且 router 也给它概率 1
    worst_probs = torch.zeros(t, e)
    worst_probs[:, 0] = 1.0
    worst_idx = torch.zeros(t, 1, dtype=torch.long)
    aux_worst = load_balancing_loss(worst_probs, worst_idx, e)
    torch.testing.assert_close(aux_worst, torch.tensor(float(e)), rtol=0, atol=1e-6)


def test_load_balancing_loss_is_bounded():
    """随机路由下 aux 始终落在 [1, E]，可用作训练时的健康度指标。"""
    torch.manual_seed(0)
    e, k, t = 8, 2, 256
    for _ in range(5):
        logits = torch.randn(t, e) * 3
        probs = logits.softmax(-1)
        idx = probs.topk(k, -1).indices
        aux = load_balancing_loss(probs, idx, e).item()
        assert 1.0 - 1e-5 <= aux <= e + 1e-5, aux


def test_moe_topk_equals_num_experts_is_dense():
    """top_k == num_experts 时精确退化为 dense 的概率加权和。"""
    torch.manual_seed(0)
    e = 4
    moe = MoELayer(d_model=8, d_ff=16, num_experts=e, top_k=e)
    moe.eval()
    x = torch.randn(2, 3, 8)

    y, _ = moe(x)

    flat = x.reshape(-1, 8)
    probs = F.softmax(moe.gate(flat).float(), dim=-1)
    dense = sum(probs[:, i, None] * moe.experts[i](flat) for i in range(e))
    torch.testing.assert_close(y.reshape(-1, 8), dense, rtol=1e-5, atol=1e-5)


def test_moe_only_topk_experts_contribute():
    """稀疏性：把某个未被任何 token 选中的专家权重改掉，输出不变。"""
    torch.manual_seed(0)
    moe = MoELayer(d_model=8, d_ff=16, num_experts=4, top_k=1)
    moe.eval()
    x = torch.randn(1, 4, 8)

    with torch.no_grad():
        _, _ = moe(x)
        probs = F.softmax(moe.gate(x.reshape(-1, 8)).float(), -1)
        chosen = set(probs.argmax(-1).tolist())
        unused = [i for i in range(4) if i not in chosen]
        assert unused, "构造失败：所有专家都被选中了"

        y0, _ = moe(x)
        moe.experts[unused[0]][0].weight.add_(100.0)
        y1, _ = moe(x)
    torch.testing.assert_close(y0, y1)


# ---------------------------- DDPM ----------------------------


def test_beta_schedules_are_valid():
    """两种 schedule 都要满足 0 < beta < 1，abar 单调递减，且首尾达到端点。"""
    for betas in (linear_beta_schedule(1000), cosine_beta_schedule(1000)):
        assert betas.shape == (1000,)
        assert (betas > 0).all() and (betas < 1).all()
        abar = torch.cumprod(1 - betas, 0)
        assert (abar.diff() < 0).all(), "累积 alpha 必须单调递减（信号逐步被噪声淹没）"
        assert abar[0] > 0.99, "t=0 时信号几乎无损"
        assert abar[-1] < 0.01, "t=T 时信号必须被彻底淹没"

    # cosine 的中段 abar 明显高于 linear：这正是它"加噪不那么急"的量化体现
    ab_lin = torch.cumprod(1 - linear_beta_schedule(1000), 0)
    ab_cos = torch.cumprod(1 - cosine_beta_schedule(1000), 0)
    assert ab_cos[500] > ab_lin[500]


def test_linear_schedule_breaks_when_timesteps_change():
    """**这是个陷阱题**：linear 的默认 beta 区间是为 T=1000 调的，换 T 会失效。

    beta 从 1e-4 线性到 0.02 时，``abar_T ≈ exp(-sum(beta))``，而
    ``sum(beta) ≈ T * (1e-4 + 0.02) / 2`` 与 T 成正比。
    T 从 1000 降到 100，累积噪声量直接少 10 倍，
    终点 abar 从 4e-5 变成 0.36 —— **前向过程根本没到纯噪声**。
    此时反向采样从 N(0, I) 起步，和训练时见过的 x_T 分布对不上，
    生成结果会系统性偏灰/偏糊。

    cosine schedule 因为是按归一化的 t/T 定义的，**天然对 T 免疫**。
    换 T 时忘了重调 beta 区间是自己实现 DDPM 最常踩的坑之一。
    """
    lin_1000 = torch.cumprod(1 - linear_beta_schedule(1000), 0)[-1]
    lin_100 = torch.cumprod(1 - linear_beta_schedule(100), 0)[-1]
    assert lin_1000 < 1e-3 and lin_100 > 0.3

    cos_1000 = torch.cumprod(1 - cosine_beta_schedule(1000), 0)[-1]
    cos_100 = torch.cumprod(1 - cosine_beta_schedule(100), 0)[-1]
    assert cos_1000 < 1e-3 and cos_100 < 1e-3


def test_q_sample_endpoints():
    """t=0 时 x_t ≈ x0；t=T-1 时 x_t 接近纯噪声（用统计量检验）。"""
    torch.manual_seed(0)
    t_max = 1000
    abar = torch.cumprod(1 - linear_beta_schedule(t_max), 0)
    x0 = torch.randn(512, 8) * 2.0
    noise = torch.randn_like(x0)

    x_start = q_sample(x0, torch.zeros(512, dtype=torch.long), noise, abar)
    torch.testing.assert_close(x_start, x0, rtol=0, atol=0.1)

    x_end = q_sample(x0, torch.full((512,), t_max - 1, dtype=torch.long), noise, abar)
    # 统计量 1：标准差接近 1（纯标准正态）
    assert abs(x_end.std().item() - 1.0) < 0.1
    # 统计量 2：与原信号几乎不相关
    corr = torch.corrcoef(torch.stack([x_end.flatten(), x0.flatten()]))[0, 1]
    assert abs(corr.item()) < 0.1
    # 统计量 3：与所加噪声高度相关
    corr_n = torch.corrcoef(torch.stack([x_end.flatten(), noise.flatten()]))[0, 1]
    assert corr_n.item() > 0.99


def test_q_sample_marginal_variance_is_preserved():
    """当 x0 本身是标准正态时，任意 t 下 x_t 的方差都应保持 1。

    因为 ``abar + (1 - abar) = 1``。这个恒等式是"方差保持型（VP）"扩散的定义，
    也是为什么系数是 sqrt(abar) 和 sqrt(1-abar) 而不是别的组合。
    """
    torch.manual_seed(0)
    abar = torch.cumprod(1 - cosine_beta_schedule(200), 0)
    x0 = torch.randn(4096, 4)
    for ti in (0, 50, 120, 199):
        t = torch.full((4096,), ti, dtype=torch.long)
        x_t = q_sample(x0, t, torch.randn_like(x0), abar)
        assert abs(x_t.std().item() - 1.0) < 0.05, ti


def test_q_sample_broadcasts_over_image_shape():
    """(B, C, H, W) 输入时 t 的广播不能写错。"""
    torch.manual_seed(0)
    abar = torch.cumprod(1 - linear_beta_schedule(50), 0)
    x0 = torch.randn(3, 3, 8, 8)
    t = torch.tensor([0, 25, 49])
    x_t = q_sample(x0, t, torch.randn_like(x0), abar)
    assert x_t.shape == x0.shape
    # t 越大离 x0 越远
    d = (x_t - x0).flatten(1).norm(dim=1)
    assert d[0] < d[1] < d[2]


def test_ddpm_loss_finite_and_trains():
    """loss 有限，且几步优化后能下降（说明梯度方向是对的）。"""
    torch.manual_seed(0)
    t_steps = 100
    abar = torch.cumprod(1 - cosine_beta_schedule(t_steps), 0)
    model = ToyDenoiser(dim=4, hidden=32)
    x0 = torch.randn(128, 4) * 0.5 + 1.0

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    first = ddpm_loss(model, x0, abar).item()
    for _ in range(30):
        opt.zero_grad()
        loss = ddpm_loss(model, x0, abar)
        loss.backward()
        opt.step()
    torch.manual_seed(0)
    last = ddpm_loss(model, x0, abar).item()

    assert math.isfinite(first) and math.isfinite(last)
    assert last < first, (first, last)


def test_ddpm_sample_shape_and_finite():
    """反向采样循环跑通，输出 shape 正确且数值有限。"""
    torch.manual_seed(0)
    betas = cosine_beta_schedule(20)  # 步数取小，保证测试快
    model = ToyDenoiser(dim=4, hidden=32)
    model.eval()

    out = ddpm_sample(model, (6, 4), betas)
    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()


def test_ddpm_sample_last_step_has_no_noise():
    """最后一步（t=0）必须不加噪声：用 eps_theta ≡ 0 的模型可以精确验证。

    eps 恒为 0 时，反向单步退化成 ``x_{t-1} = x_t / sqrt(alpha_t) + sigma_t z``，
    若最后一步仍加噪，重复采样的方差会明显偏大。
    这里直接检查确定性：固定 z 的来源后，去掉最后一步噪声的实现给出确定结果。
    """
    class ZeroEps(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    betas = linear_beta_schedule(5)
    alphas = 1.0 - betas
    model = ZeroEps()

    g = torch.Generator().manual_seed(0)
    out = ddpm_sample(model, (2, 3), betas, generator=g)

    # 手动复现：只有 i>0 的步骤加噪
    g2 = torch.Generator().manual_seed(0)
    x = torch.randn((2, 3), generator=g2)
    for i in reversed(range(5)):
        x = x / alphas[i].sqrt()
        if i > 0:
            x = x + betas[i].sqrt() * torch.randn((2, 3), generator=g2)
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)


# ---------------------------- LoRA ----------------------------


def test_lora_initial_output_equals_base_linear():
    """B 初始化为 0 => 训练开始时输出与原 Linear **完全**相同（不是"接近"）。"""
    torch.manual_seed(0)
    lora = LoRALinear(16, 8, r=4, alpha=8.0)
    x = torch.randn(3, 16)

    torch.testing.assert_close(lora(x), lora.base(x), rtol=0, atol=0)
    assert (lora.lora_B == 0).all()
    assert (lora.lora_A != 0).any(), "A 不能也是 0，否则梯度恒为 0 永远学不动"


def test_lora_only_ab_have_gradients():
    """base weight 被冻结：只有 A/B 拿到梯度。"""
    torch.manual_seed(0)
    lora = LoRALinear(16, 8, r=4)
    lora(torch.randn(3, 16)).pow(2).sum().backward()

    assert lora.base.weight.grad is None and lora.base.weight.requires_grad is False
    assert lora.base.bias.grad is None
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None
    # 初始时 B=0，所以 dL/dA ∝ B^T(...) = 0；而 dL/dB ∝ (...)A^T != 0。
    # 这正是"必须有一边非零"的直接体现：第一步只有 B 动，之后 A 才开始收到梯度。
    assert lora.lora_B.grad.abs().sum() > 0
    torch.testing.assert_close(lora.lora_A.grad, torch.zeros_like(lora.lora_A))


def test_lora_trainable_param_count():
    """可训练参数量 = r * (in + out)，与 base 的 in*out 差好几个数量级。"""
    in_f, out_f, r = 512, 512, 8
    lora = LoRALinear(in_f, out_f, r=r, alpha=16.0)

    assert lora.num_trainable() == r * (in_f + out_f) == 8192
    total = sum(p.numel() for p in lora.parameters())
    assert total == in_f * out_f + out_f + r * (in_f + out_f)
    assert lora.num_trainable() / (in_f * out_f) < 0.04  # 省 30 倍以上


def test_lora_merge_is_equivalent():
    """合并权重后用普通 Linear 前向，结果与 LoRA 前向一致 => 推理零额外延迟。"""
    torch.manual_seed(0)
    lora = LoRALinear(16, 8, r=4, alpha=8.0)
    with torch.no_grad():  # 让 B 非零，模拟训练之后的状态
        lora.lora_B.normal_(0, 0.1)

    x = torch.randn(5, 16)
    merged = F.linear(x, lora.merged_weight(), lora.base.bias)
    torch.testing.assert_close(lora(x), merged, rtol=1e-5, atol=1e-6)


def test_lora_scaling_and_the_rslora_critique():
    r"""alpha/r 缩放的实际效果，以及 rsLoRA 为什么说它"过校正"了。

    设 A、B 的元素独立、标准差分别是 sigma_A、sigma_B，则
    ``(B A)_ij = sum_{k=1}^{r} B_ik A_kj`` 是 r 项之和，标准差是
    ``sqrt(r) * sigma_A * sigma_B`` —— **随 r 以 sqrt(r) 增长**。

    于是：
      - 不缩放：Delta W 的量级 ``∝ sqrt(r)``，r 变大时更新会失控。
      - LoRA 的 ``alpha / r``：量级变成 ``∝ 1/sqrt(r)``，**矫枉过正**，
        r 越大反而更新越小，所以大 r 时必须调大 learning rate 才有效果
        （这正是"LoRA 提高 r 却不涨点"的常见原因）。
      - rsLoRA 的 ``alpha / sqrt(r)``：量级与 r **无关**，是理论上正确的缩放。

    这道题能问出候选人是真理解还是背下了公式。
    """
    torch.manual_seed(0)
    ranks = (2, 8, 32)
    raw, lora_scaled, rs_scaled = [], [], []
    for r in ranks:
        lora = LoRALinear(128, 128, r=r, alpha=16.0)
        with torch.no_grad():
            lora.lora_B.normal_(0, 1.0)  # sigma_B 与 r 无关
        ba = lora.lora_B @ lora.lora_A
        raw.append(ba.std().item())
        lora_scaled.append((lora.scaling * ba).std().item())
        rs_scaled.append(((16.0 / math.sqrt(r)) * ba).std().item())

    # 不缩放：从 r=2 到 r=32（16 倍）量级涨 sqrt(16)=4 倍
    assert 3.5 < raw[-1] / raw[0] < 4.5, raw
    # LoRA 的 alpha/r：反过来缩小 4 倍
    assert 3.5 < lora_scaled[0] / lora_scaled[-1] < 4.5, lora_scaled
    # rsLoRA 的 alpha/sqrt(r)：基本持平
    assert max(rs_scaled) / min(rs_scaled) < 1.15, rs_scaled


if __name__ == "__main__":
    import sys

    for name, fn in dict(globals()).items():
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all advanced-model tests passed", file=sys.stderr)
