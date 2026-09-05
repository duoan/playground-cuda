#import "../template.typ": *

= 进阶模型：ViT / CLIP / MoE / Diffusion / LoRA

第 26 章把 Transformer 拆到了零件级别。这一章往上走一层：面试里真正被要求"现场写一个"的进阶模块 —— ViT 的 patch embedding、CLIP 的对比损失、MoE 的 top-k 路由、DDPM 的加噪与采样、LoRA 的低秩旁路。

这五个模块的共同点是：*代码都不长（20--40 行），但每一个都埋着一两个"写出来能跑、跑出来是错的"的坑*。面试官问这些题不是想看你会不会调库，是想看你能不能说清"为什么这一行必须这么写"。所以下面每题都按同一个模板走：题目 → 考点 → 参考实现 → 常见错法 → *怎么自证*。最后那一步最值钱：能当场说出一个可验证的数值断言（"随机初始化时 CLIP loss 应该正好是 $log B$"），比任何口头解释都有说服力。

配套的可运行实现与全部断言在 `python/pytorch/interview/test_advanced_models.py`。

== ViT：把图像变成 token 序列

*题目.* 手写 `PatchEmbed` 和一个最小可用的 ViT 前向，说清 patch 数、序列长度、cls token 和位置编码各自的作用。

*考点.* conv 实现切 patch 的等价性；序列长度为什么是 $N + 1$；ViT 为什么吃数据。

=== 参考实现

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=48):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        # 关键：kernel == stride == patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                     # (B, C, H, W)
        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)
```

*为什么 `Conv2d(kernel=P, stride=P)` 等价于"切块 + Linear"？* ViT 的原始定义是：把 $(C, H, W)$ 切成 $N = (H \/ P) times (W \/ P)$ 个互不重叠的 $(C, P, P)$ 块，每块展平成 $C P^2$ 维，再过一个*所有 patch 共享*的 `Linear(C*P*P, D)`。而卷积的每个输出位置做的事就是"取一个 $P times P$ 的感受野、与卷积核做内积"。当 `stride == kernel_size` 时感受野恰好不重叠、不遗漏，于是每个输出位置正好对应一个 patch；卷积核 `(D, C, P, P)` reshape 成 `(D, C*P*P)` 就是那个 Linear 的权重矩阵，而卷积的权重共享正好对应"所有 patch 过同一个 Linear"。

好处是省掉显式的 `unfold` + `reshape`（`unfold` 会实体化一个 $C P^2 times N$ 的大矩阵，纯内存搬运），一个 cudnn kernel 搞定。

=== cls token 与位置编码

```python
    def __init__(self, ..., embed_dim=48, depth=2, num_classes=10):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))  # ← N+1
        nn.init.trunc_normal_(self.cls_token, std=0.02)   # ViT 原论文的初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)                        # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)         # expand 不拷贝内存
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:                        # pre-norm LN + MHA + MLP
            x = blk(x)
        return self.head(self.norm(x)[:, 0])           # 只取 cls
```

ViT-B/16 在 224 分辨率下：$N = (224\/16)^2 = 196$，加 cls token 后序列长 197，`pos_embed` 也是 197 行。ViT block 与 GPT block 的差别只有两点：*没有 causal mask*（图像 patch 双向可见），以及用 LayerNorm + GELU MLP 而不是 RMSNorm + SwiGLU（历史原因，非本质）。

*cls token 是什么.* 一个可学习的、与输入无关的向量，拼在序列最前面。它自己不携带任何图像内容，只能通过 attention 从各个 patch 聚合信息，因此天然成为"全局表示"。替代方案是对所有 patch 做 mean pooling（DeiT III、SigLIP 用这个），效果相当甚至更好还省一个 token；cls token 更多是从 BERT 继承的传统。

*位置编码为什么必须有.* attention 对输入顺序是*置换等变*的：把 patch 序列打乱，输出只是跟着一起打乱，每个 token 的表示完全不变 —— 图像的空间结构就丢了。ViT 用*可学习的绝对位置编码*而不是 RoPE，因为图像位置是二维的。

#warn[
  ViT 上最常踩的三个错：

  + *`pos_embed` 少了一行*：写成 `(1, N, D)` 会和 `(B, N+1, D)` 广播不上；更糟的是顺手写成 `x[:, 1:] += pos_embed`，变成 cls token 没有位置信息、所有 patch 整体偏移一格。
  + *换分辨率忘了插值位置编码*。`pos_embed` 的长度和 $N$ 绑死。从 224 微调到 384，$N$ 从 196 变成 576，必须把 patch 部分 reshape 成 $14 times 14$ 做二维双三次插值再拉平（cls 那一行单独保留），否则 `load_state_dict` 报 size mismatch，强行截断则精度崩盘。
  + *`cls_token.repeat(b, 1, 1)` 代替 `expand`*。功能一样但会真拷贝 $B$ 份内存；`expand` 只改 stride（见第 1 章），后面 `cat` 时才实体化一次。
]

=== 怎么自证

```python
# 1) conv 路径 == unfold + 共享 Linear，逐元素相同
patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # (B,N,C*P*P)
w       = pe.proj.weight.reshape(d, c * p * p)                  # (D, C*P*P)
torch.testing.assert_close(pe(x), patches @ w.t() + pe.proj.bias)

# 2) 关掉位置编码 => 打乱 patch 顺序输出不变（置换等变性）
torch.testing.assert_close(model(x, use_pos=False),
                           model(shuffle_patches(x, perm), use_pos=False))
assert not torch.allclose(model(x), model(shuffle_patches(x, perm)))
```

第二个测试是"位置编码为什么必须有"的直接实验证据，比说一遍定义有力得多。

=== 与 CNN 的对比：归纳偏置与复杂度

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [*CNN*], [*ViT*],
  [归纳偏置], [局部性 + 平移等变 + 层级感受野，硬编码在结构里], [几乎没有；只有"patch 内是局部的"],
  [感受野], [随深度线性增长], [第一层就是全局],
  [单层复杂度], [$O(H W C^2 k^2)$，与分辨率*线性*], [$O(N^2 D) + O(N D^2)$，与 $N$ *平方*],
  [数据需求], [小数据集也能训], [JFT-300M 级别才超过 CNN],
)

*"ViT 需要大数据"的准确说法*：CNN 把"相邻像素相关、物体平移后还是同一个物体"这两条先验直接写进结构，等于免费拿到一堆样本。ViT 把这些先验全交给数据去学 —— 数据够多时它学到的比手工先验更好（这是 ViT 赢的地方），数据不够时它连"局部性"都学不出来。DeiT 用蒸馏 + 强增强在 ImageNet-1k 规模补掉了这个 gap。复杂度那一行的实际后果：$P$ 从 16 减到 8，$N$ 变 4 倍，attention 计算量变 16 倍 —— 这就是 ViT 上不去高分辨率、要靠 Swin 的窗口注意力把 $O(N^2)$ 降回 $O(N)$ 的原因。

== CLIP：对称对比损失

*题目.* 手写 `clip_contrastive_loss(image_emb, text_emb, logit_scale)`。

*考点.* L2 normalize 为什么必须；`logit_scale` 为什么可学习且用 `exp`；为什么两个方向都要算；batch size 为什么是命门。

```python
def clip_contrastive_loss(image_emb, text_emb, logit_scale=1.0):
    image_emb = F.normalize(image_emb, dim=-1)         # (B, D)
    text_emb  = F.normalize(text_emb,  dim=-1)         # (B, D)

    logits = logit_scale * (image_emb @ text_emb.t())  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits,     labels)     # 按行：每张图找文本
    loss_t2i = F.cross_entropy(logits.t(), labels)     # 按列：每段文本找图
    return (loss_i2t + loss_t2i) / 2, logits
```

整个 loss 就是"把 $(B, B)$ 相似度矩阵按行和按列都逼近单位阵"。`labels = arange(B)` 是因为第 $i$ 张图的正样本就是第 $i$ 段文本 —— 对角线是正样本，同一行其余 $B - 1$ 个都是负样本。

=== 为什么必须先 L2 normalize

归一化后内积就是*余弦相似度*，取值被限死在 $[-1, 1]$、与向量模长无关。不归一化的话，模型可以靠"把正样本对的模长一起撑大"来降 loss —— 这是一条完全不学语义的捷径，训练会不稳定甚至发散。归一化把"学表示（方向）"和"学置信度（尺度）"解耦，后者统一交给 `logit_scale`。

#insight[
  L2 normalize 有一个直接可测的后果：*loss 对 embedding 的整体缩放完全不敏感*。把 `image_emb * 100`、`text_emb * 0.01` 喂进去，loss 一位不变。这是验证"你真的归一化了"最快的检查。
]

=== `logit_scale`：为什么可学习、为什么 `exp`、为什么 clamp

余弦相似度只在 $[-1, 1]$。直接送进 softmax，$B = 32768$ 时最大和最小 logit 只差 2，softmax 输出几乎是均匀分布，梯度小到学不动。所以必须乘一个大的"温度倒数"，CLIP 把它设成可学习参数，训练收敛到约 100（即 temperature $approx 0.01$）。

参数化成 $s = exp(t)$ 而不是直接学 $s$ 有两个理由：$exp$ 保证 $s > 0$（温度必须正）；优化在对数尺度上进行，对这种要跨 1--100 两个数量级的标量更稳（等价于对 $s$ 做乘性更新而不是加性更新）。clamp 上界（CLIP 官方 clamp 到 $ln 100$）是为了防跑飞 —— 只要模型已经把对角线做成最大值，无脑增大 $s$ 就能继续压 loss，$s$ 会单调爆炸到 softmax 溢出。

=== 为什么两个方向都要算

单向 image$arrow.r$text 只约束"给定图片能选对文本"。模型可以把所有文本 embedding 挤到一小块区域、只让图片彼此可分，照样把这个方向的 loss 降下去 —— 表示坍缩的一种。双向同时约束两个检索方向，得到的才是真正对齐的联合空间。实践上两个方向的 loss 数值通常差不多，但去掉一个会明显掉检索指标。

=== batch size 为什么对对比学习至关重要

每个样本的负样本数就是 $B - 1$，*全部来自 batch 内部*，没有别的来源。InfoNCE 是互信息的一个下界，而这个下界本身的上限是 $log B$：

#formula[$ I(X; Y) >= log B - cal(L)_"InfoNCE" $]

所以 $B$ 越大，loss 能提供的信息量上界越高。这就是 CLIP 用 32k batch 的原因，也是 MoCo（memory bank / queue）与 SimCLR（暴力大 batch）两条路线的分歧点。分布式实现要用 all-gather 把所有 GPU 的 embedding 聚起来算全局 $(N, N)$ 矩阵。

#warn[
  分布式对比学习的经典 bug：*只 gather embedding，没让梯度流回其他 rank*。`dist.all_gather` 是不可微的，返回的 tensor 没有 `grad_fn`，所以本 rank 只会收到"自己那一段 embedding"的梯度，其他 rank 贡献的负样本对本地梯度的贡献全丢了 —— loss 数值是对的（所以很难发现），梯度是错的。修法是用可微的 `torch.distributed.nn.all_gather`（它的 backward 里做 reduce-scatter），或手动把本地那一段拼回去。
]

=== 怎么自证

两个理论值都能算出来，这是这一题最漂亮的地方：

```python
# 1) 完美对齐：img == txt，对角线相似度 1、其余近 0（高维随机向量近似正交）
emb = F.normalize(torch.randn(32, 64), dim=-1)
loss, logits = clip_contrastive_loss(emb, emb.clone(), logit_scale=30.0)
assert loss.item() < 1e-3
assert (logits.argmax(-1) == torch.arange(32)).all()   # top-1 检索 100%

# 2) 随机 embedding：loss ≈ log(B)
for b in (32, 64):
    loss, _ = clip_contrastive_loss(torch.randn(b, 512), torch.randn(b, 512),
                                    logit_scale=1.0)
    assert abs(loss.item() - math.log(b)) < 0.05
```

第二条的推导（面试时现场给出来是加分项）：$D = 512$ 维的两个独立高斯向量归一化后，内积期望 0、标准差 $1 \/ sqrt(D) approx 0.044$。`logit_scale = 1` 时整个 $(B, B)$ 矩阵几乎全是 0，softmax 退化成均匀分布 $1\/B$，于是 $cal(L) = -log(1\/B) = log B$（$B = 32$ 是 3.47，$B = 64$ 是 4.16）。工程价值：*loss 卡在 $log B$ 不降，说明模型根本没在学对齐* —— 常见原因是学习率炸了、`logit_scale` 初始化太小、或者图文配对本身错位。这是对比学习最好用的健康度指标。

== MoE：top-k gating 与负载均衡

*题目.* 手写一个 top-k sparse MoE 层，返回输出和 load-balancing aux loss。

*考点.* dispatch 怎么写才不慢；aux loss 两项各是什么；router 为什么必须 fp32；MoE 省了什么、没省什么。

#note[
  本仓库另有一本 MoE 专书 `books/moe/`，把 dispatch、capacity、expert parallel、all-to-all 的工程细节讲透了。这一节只给面试需要的深度：写出正确的 20 行、说清 aux loss 的数学、回答"省显存吗"。
]

=== 参考实现

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model)) for _ in range(num_experts)])

    def forward(self, x):                              # (B, S, H)
        b, s, h = x.shape
        flat = x.reshape(-1, h)                        # (T, H)，逐 token 路由

        gate_probs = F.softmax(self.gate(flat).float(), dim=-1)      # 强制 fp32
        topk_probs, topk_idx = torch.topk(gate_probs, self.top_k, -1)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)   # 重归一化
        topk_probs = topk_probs.to(x.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):       # 按 expert 循环，不是按 token
            token_ids, slot_ids = torch.where(topk_idx == e)
            if token_ids.numel() == 0:
                continue
            y = self.experts[e](flat[token_ids])       # (M, H) 一个大 GEMM
            out.index_add_(0, token_ids,
                           y * topk_probs[token_ids, slot_ids, None])

        aux = load_balancing_loss(gate_probs, topk_idx, self.num_experts)
        return out.view(b, s, h), aux
```

#figure(
  align(center, moe-dispatch(n-tokens: 8, n-experts: 4,
                             routing: (0, 2, 0, 1, 3, 0, 2, 0),
                             title: "top-1 路由，E0 拿了一半 token")),
  caption: [负载不均衡的样子：E0 拿到 4 个 token，E1/E3 各只有 1 个。expert parallel 下 E0 所在的卡成为 straggler，其余卡空转等它 —— 这就是 aux loss 存在的现实动机。],
) <fig-moe-imbalance>

*为什么按 expert 循环而不是按 token 循环.* 按 token 循环是 Python 级 $O(T)$ 次小 kernel launch，$T$ 是几万，慢到不能用。按 expert 循环是 $O(E)$ 次、每次一个大 GEMM，而 $E$ 通常只有 8--64。生产实现会再进一步：先按 expert 做一次 `argsort` 把 token 排好序，变成一次 grouped GEMM（Megablocks、`grouped_gemm`），彻底消掉 Python 循环。

*top-k 内为什么要重归一化.* 让每个 token 的权重和为 1，保持输出量级与 dense 一致（否则 top-2 的权重和可能只有 0.3，输出被系统性缩小）。顺带一个好性质：`top_k == num_experts` 时这一步是恒等操作，整层精确退化成 dense 的概率加权和 —— 这就是这一题的自证方法。

=== load-balancing aux loss

```python
def load_balancing_loss(gate_probs, expert_indices, num_experts):
    t, k = expert_indices.shape
    one_hot = F.one_hot(expert_indices.reshape(-1), num_experts).float()
    f = one_hot.sum(0) / (t * k)      # (E,) 硬分配频率，sum(f) == 1
    p = gate_probs.mean(0)            # (E,) 软概率均值，sum(p) == 1
    return num_experts * torch.sum(f * p)
```

#formula[$ f_e = 1 / (T K) sum_(t, k) bb(1)[ "token" t "的第" k "个 slot 选了" e ], quad P_e = 1 / T sum_t p_(t e), quad "aux" = E sum_e f_e P_e $]

两项的分工是这个设计的全部精髓。$f_e$ 来自 `topk`，是硬分配的计数、*不可导* —— 它决定真实的显存占用和通信量。$P_e$ 是 softmax 的输出、*可导* —— 梯度全部从这一项流回 router。相乘的效果是：某个 expert 如果既被大量选中（$f_e$ 大）又被 router 给了高概率（$P_e$ 大），乘积就大、被重罚，router 因此被推着降低对它的打分。*用一个不可导的"实际负载"去加权一个可导的"意愿"*，是这类 straight-through 风格设计的通用套路。

*为什么均匀时最小值恰好是 1.0.* 关键观察是 $f$ 和 $P$ 不是独立的两个分布：硬分配是从软概率里 top-k 出来的，训练中 $f approx P$。代入后 aux $= E sum_e f_e^2$，而 $f$ 是概率向量，由 Cauchy--Schwarz

#formula[$ sum_e f_e^2 >= (sum_e f_e)^2 / E = 1 / E quad arrow.r.double quad "aux" >= E dot 1/E = 1 $]

等号成立当且仅当 $f$ 均匀。另一端：全挤到一个 expert 时 $f = P = (1, 0, dots, 0)$，aux $= E dot 1 = E$。所以 *aux 天然落在 $[1, E]$，训练时盯着它离 1 有多远就知道路由健不健康*。总 loss 通常是 `task_loss + 0.01 * aux`。

#warn[
  两个常见错法。

  第一，*只用 $P_e$ 的方差当 loss*（"让软概率均匀"）。这只约束了意愿，实际的硬分配仍可能极不均衡 —— 而决定显存和 all-to-all 通信的是硬分配。

  第二，*router 的 softmax 用 bf16/fp16*。两个后果：(a) bf16 只有 8 位尾数，`exp` 之后 top-1 和 top-2 的概率可能变成完全相同的浮点数；(b) `topk` 的 tie-breaking 依赖精确比较，一旦并列，不同 rank 可能选出不同 expert，expert-parallel 下 all-to-all 的收发形状直接对不上 —— 表现为随机 hang 或 shape 错，是最难查的一类非确定性 bug。fp16 更糟：router logits 会被 aux loss 推得很尖，`exp(11.1)` 就已经超过 fp16 的 65504 上限，直接 `inf`。所以 Switch / Mixtral 的实现里 router 一律 `.float()`，算完再 cast 回去。
]

=== expert collapse 与"省了什么"

*collapse 现象.* 少数 expert 拿走绝大多数 token，其余几乎不被选中 → 收不到梯度 → 更不会被选中，正反馈跑飞。*诊断*按顺序看三个指标：aux loss 离 1 有多远（贴着 $E$ 就是彻底坍缩）、每个 expert 的 token 计数直方图、被丢弃 token 的比例（有 capacity 时）。*只看 loss 曲线看不出来* —— 坍缩后的 MoE 就是一个参数量虚高的 dense 模型，task loss 照样降。*修法*：调大 aux 系数（0.01 $arrow.r$ 0.1，代价是牺牲一点 task loss）；router logits 加噪声（Noisy top-k gating）；加 router z-loss（惩罚 logits 的 log-sum-exp，防止整体变尖）；换 Expert Choice 路由（让 expert 挑 token，负载天然均衡）。

#insight[
  *MoE 省 FLOPs，不省显存。* top-2/64 的 MoE 每个 token 只过 2 个 expert，FLOPs 是 dense 的 $2\/64$；但*所有 64 个 expert 的权重都得装在显存里*，参数和优化器状态一分不少（AdamW 下每参数 12--16 字节）。这就是 MoE 必须配 expert parallel 的原因：把 expert 拆到不同卡上，用 all-to-all 换显存。完整答法还要加一句：*也不省 activation*，每个 token 的中间激活照存，反向还多一层 dispatch/combine 的 index 记账。所以 MoE 的真实收益是"同等训练 FLOPs 下容纳更多参数"，不是"同等显存下更快"。
]

=== 怎么自证

```python
# 1) aux 的两个端点值都是精确的
uniform = torch.full((64, 4), 0.25)
idx     = (torch.arange(64) % 4).unsqueeze(1)      # 完美均匀的硬分配
assert abs(load_balancing_loss(uniform, idx, 4).item() - 1.0) < 1e-6
worst = torch.zeros(64, 4); worst[:, 0] = 1.0      # 全挤 expert 0 => aux == E
assert abs(load_balancing_loss(worst, torch.zeros(64,1).long(), 4) - 4.0) < 1e-6

# 2) top_k == num_experts 时精确等于 dense 加权和 —— dispatch 的杀手测试
probs = F.softmax(moe.gate(flat).float(), -1)
dense = sum(probs[:, i, None] * moe.experts[i](flat) for i in range(E))
torch.testing.assert_close(moe(x)[0].reshape(-1, H), dense, rtol=1e-5, atol=1e-5)

# 3) 稀疏性：改一个没被选中的 expert 的权重，输出必须不变
```

第 2 条能一次抓住 `index_add_` 的下标、`slot_ids` 的对应关系、top-k 重归一化 —— 任何一处写错它都过不了。

== Diffusion：DDPM 的前向闭式解与反向采样

*题目.* 写出前向加噪的闭式解、DDPM 的训练目标、反向采样循环，并说清 `linear` 与 `cosine` beta schedule 的区别。

*考点.* 闭式解怎么推；为什么预测噪声；最后一步为什么不加噪；schedule 与 $T$ 的耦合。

=== 前向：一步跳到任意时刻

逐步加噪的定义是一条马尔可夫链（$alpha_t = 1 - beta_t$）：$x_t = sqrt(alpha_t) x_(t-1) + sqrt(beta_t) epsilon_t$，$epsilon_t tilde cal(N)(0, I)$。递归展开一步就是推导的*关键一步*，面试时把它写出来就够了：

#formula[$ x_t = sqrt(alpha_t alpha_(t-1)) x_(t-2) + underbrace(sqrt(alpha_t beta_(t-1)) epsilon_(t-1) + sqrt(beta_t) epsilon_t, "两个独立高斯之和") $]

独立高斯的线性组合仍是高斯、方差直接相加：$alpha_t beta_(t-1) + beta_t = alpha_t (1 - alpha_(t-1)) + 1 - alpha_t = 1 - alpha_t alpha_(t-1)$。也就是展开后系数仍满足"信号系数平方 + 噪声系数平方 = 1"。归纳下去，记 $overline(alpha)_t = product_(s <= t) alpha_s$：

#formula[$ q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I) quad arrow.r.double quad x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon $]

```python
def q_sample(x0, t, noise, alphas_cumprod):
    abar = alphas_cumprod[t]                        # (B,)
    shape = (-1,) + (1,) * (x0.dim() - 1)           # (B, 1, 1, ...)
    return abar.view(shape).sqrt() * x0 + (1.0 - abar.view(shape)).sqrt() * noise
```

#insight[
  这个闭式解是 DDPM 能被训练的*根本原因*：训练时随机采一个 $t$，一步就能构造出 $(x_t, epsilon)$ 样本对，不用真跑 $t$ 步前向，否则训练成本要乘 $O(T)$。它也顺带解释了系数为什么是 $sqrt(overline(alpha))$ 和 $sqrt(1 - overline(alpha))$：两者平方和为 1，所以 $x_0 tilde cal(N)(0, I)$ 时任意 $t$ 下 $x_t$ 的方差都保持 1。这叫 variance-preserving（VP）扩散，好处是网络在所有时间步看到的输入量级一致。
]

#warn[
  广播是这一题最容易写错的地方。`abar[t]` 是 `(B,)`，直接和 `(B, C, H, W)` 相乘会广播成 `(B,C,H,B)` 的灾难或直接报错。必须补维成 `(B,1,1,1)`；用 `view(shape)` 动态算而不是硬编码 4 维，才能同时兼容 `(B, D)` 和 `(B, C, H, W)`。
]

=== 训练目标：为什么是"预测噪声的 MSE"

```python
def ddpm_loss(model, x0, alphas_cumprod, t=None):
    if t is None:      # 每个样本独立采 t，不能整个 batch 共用一个
        t = torch.randint(0, alphas_cumprod.shape[0], (x0.shape[0],),
                          device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    return F.mse_loss(model(x_t, t), noise)
```

#formula[$ cal(L)_"simple" = EE_(t, x_0, epsilon) norm(epsilon - epsilon_theta (x_t, t))^2 $]

这个朴素 MSE 是*简化的变分下界*：完整 ELBO 是一串 KL 项 $sum_t D_"KL"(q(x_(t-1) | x_t, x_0) || p_theta (x_(t-1) | x_t))$，两边都是方差固定的高斯，KL 退化成均值之差的平方；把均值用 $epsilon_theta$ 重参数化后每项变成 $w_t norm(epsilon - epsilon_theta)^2$，权重 $w_t = beta_t^2 \/ (2 sigma_t^2 alpha_t (1 - overline(alpha)_t))$。DDPM 的关键经验是*把 $w_t$ 直接设为 1 反而效果更好* —— 原权重给低噪声时刻（任务只是"去掉一点点噪声"）压了极大的权重，扔掉它相当于把注意力挪到中高噪声区间。

*为什么预测 $epsilon$ 而不是 $x_0$.* 两者可互换（$hat(x)_0 = (x_t - sqrt(1 - overline(alpha)_t) hat(epsilon)) \/ sqrt(overline(alpha)_t)$），数学上等价，差别在*隐含加权*：以 $epsilon$ 为目标等价于给高噪声时刻更大权重，实测样本质量更好。后来还有 v-prediction（预测 $sqrt(overline(alpha)) epsilon - sqrt(1 - overline(alpha)) x_0$），高 SNR 区间数值更稳，是 SD 2.x 与蒸馏常用的参数化。`t` 必须*逐样本*采，整个 batch 共用一个 $t$ 会让梯度方差巨大。

=== 反向采样

#formula[$ x_(t-1) = 1 / sqrt(alpha_t) (x_t - beta_t / sqrt(1 - overline(alpha)_t) epsilon_theta (x_t, t)) + sigma_t z, quad z tilde cal(N)(0, I) $]

```python
@torch.no_grad()
def ddpm_sample(model, shape, betas, device=None, generator=None):
    alphas, abar = 1.0 - betas, torch.cumprod(1.0 - betas, dim=0)
    x = torch.randn(shape, device=device, generator=generator)
    for i in reversed(range(betas.shape[0])):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        mean = (x - betas[i] / (1.0 - abar[i]).sqrt() * eps) / alphas[i].sqrt()
        if i > 0:
            z = torch.randn(shape, device=device, generator=generator)
            x = mean + betas[i].sqrt() * z
        else:
            x = mean                      # ← 最后一步不加噪
    return x
```

$sigma_t^2 = beta_t$ 是论文给的两个选择之一（另一个是后验方差 $tilde(beta)_t = (1 - overline(alpha)_(t-1)) \/ (1 - overline(alpha)_t) dot beta_t$，效果接近）。*最后一步为什么不加噪*：$x_0$ 就是最终输出，再加一次噪声等于白往结果里掺噪点。对应 `if i > 0` 那一行 —— 实现里最常漏的一行，漏了的表现是生成图有一层可见颗粒噪声。*为什么慢*：必须串行走 $T$ 步、每步一次完整前向，$T = 1000$ 就是 1000 次调用且无法并行。DDIM 把采样改写成*确定性*的非马尔可夫过程（$sigma_t = 0$），允许跳步（只走 50 步），且同一个训练好的模型直接复用、不用重训。

=== beta schedule：linear 与 cosine

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps  # ← 归一化 t/T
    abar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    abar = abar / abar[0]
    return (1 - abar[1:] / abar[:-1]).clamp(0.0, 0.999)
```

cosine 先定义累积信号强度 $overline(alpha)(t)$，再反推 $beta_t = 1 - overline(alpha)_t \/ overline(alpha)_(t-1)$。它更好的原因：linear 下 $overline(alpha)$ 在前 20% 步就掉到接近 0，绝大部分时间步都在"几乎纯噪声"里空转，样本利用率低；cosine 让 $overline(alpha)$ 在中段近似线性下降，噪声水平分布更均匀。可测量的体现是 $overline(alpha)_(T\/2)$：cosine 明显高于 linear。

#warn[
  *`linear_beta_schedule` 的默认 beta 区间是为 $T = 1000$ 调的，换 $T$ 会静默失效* —— 这是自己实现 DDPM 最容易踩的坑，而且不报错。

  推导：$beta$ 从 $10^(-4)$ 线性到 $0.02$ 时 $overline(alpha)_T = product(1 - beta_t) approx exp(-sum beta_t)$，而 $sum beta_t approx T (10^(-4) + 0.02) \/ 2 = 0.01005 T$ —— *与 $T$ 成正比*。$T = 1000$ 时 $sum beta approx 10.05$、$overline(alpha)_T approx e^(-10.05) approx 4 times 10^(-5)$（合格，终点是纯噪声）；$T = 100$ 时 $sum beta approx 1.005$、$overline(alpha)_T approx e^(-1.005) approx 0.36$，*前向过程根本没到纯噪声* —— $x_T$ 里还留着 $sqrt(0.36) approx 0.6$ 倍的原始信号。

  而反向采样是从 $cal(N)(0, I)$ 起步的。训练时模型只见过"还带 60% 信号"的 $x_T$，采样时喂给它纯噪声，训练/采样分布对不上，生成结果系统性偏灰、偏糊 —— 但 loss 曲线完全正常，因为训练本身自洽。

  cosine schedule 因为按*归一化的 $t \/ T$* 定义（看代码里的 `t / timesteps`），$overline(alpha)$ 的形状与 $T$ 无关，天然免疫。自证：`torch.cumprod(1 - linear_beta_schedule(T), 0)[-1]` 在 $T = 1000$ 时 $< 10^(-3)$、$T = 100$ 时 $> 0.3$；cosine 在两个 $T$ 下都 $< 10^(-3)$。
]

=== 怎么自证

```python
# 1) schedule 合法性：0 < beta < 1，abar 单调递减，首尾到端点
abar = torch.cumprod(1 - betas, 0)
assert (betas > 0).all() and (betas < 1).all()
assert (abar.diff() < 0).all() and abar[0] > 0.99 and abar[-1] < 0.01

# 2) q_sample 两端：t=T-1 时是纯噪声 —— 用统计量而不是逐元素比
x_end = q_sample(x0, torch.full((512,), T - 1), noise, abar)
assert abs(x_end.std().item() - 1.0) < 0.1     # 标准正态
assert abs(corrcoef(x_end, x0)) < 0.1          # 与信号无关
assert corrcoef(x_end, noise) > 0.99           # 就是噪声本身

# 3) 最后一步不加噪：把网络换成恒返回零的 ZeroEps，反向单步退化成
#    x / sqrt(alpha_t) + sigma_t * z，固定 generator 后可手算轨迹逐位比对
```

第 3 条的技巧值得记：*把随机过程的一个分支固定成确定性，是测试生成模型的通用手法* —— `if i > 0` 写错立刻被抓出来。

== LoRA：低秩旁路

*题目.* 手写 `LoRALinear`，说清 A/B 的初始化、参数量、以及 `alpha / r` 这个缩放到底在干什么。

*考点.* B 为什么必须为 0；参数量省了多少；`alpha / r` 是不是真的让量级与 $r$ 无关（这题能筛掉背公式的人）。

#formula[$ y = x W^tack.b + b + (alpha / r) (x A^tack.b) B^tack.b, quad A in RR^(r times d_"in"), B in RR^(d_"out" times r) $]

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8.0, bias=True):
        super().__init__()
        self.r, self.scaling = r, alpha / r
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)         # 冻结主干
        if bias:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # ← 必须 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A),
                                       self.lora_B) * self.scaling

    def merged_weight(self):
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)
```

注意 `forward` 里是 `F.linear(F.linear(x, A), B)` 两次小 GEMM，*不是*先算 `B @ A` 再乘 `x`。前者计算量 $O(B r (d_"in" + d_"out"))$；后者要实体化一个 $d_"out" times d_"in"$ 矩阵、计算量 $O(d_"in" d_"out" (r + B))$ —— 写反了 LoRA 就一点都不省了。

=== B 为什么必须初始化为 0

训练开始时必须有 $Delta W = B A = 0$，让模型*精确等价于*原模型（不是"接近"，是逐位相同）。否则一上来就给预训练权重叠了个随机扰动，前几百步全在修复它，微调效果明显变差 —— 数据少时尤其明显。

*但 A 和 B 不能都置零*：那样 $partial cal(L) \/ partial A prop B^tack.b (dots) = 0$、$partial cal(L) \/ partial B prop (dots) A^tack.b = 0$，两边梯度恒为 0，对称性未破缺，永远学不动。标准做法是 *A 随机（kaiming uniform）、B 全零*：第一步只有 $B$ 收到梯度（它的梯度依赖 $A$，而 $A eq.not 0$），$B$ 动起来之后 $A$ 才开始收梯度。反过来（A=0、B 随机）数学上也可行，但 PEFT 库统一用前者。

=== 参数量：给具体数字

#formula[$ "可训练" = r (d_"in" + d_"out") quad "vs" quad "原权重" = d_"in" d_"out" $]

拿 LLaMA-7B 的一个 attention 投影 $d_"in" = d_"out" = 4096$ 举例：full FT 是 $4096^2 = 16.8$ M；$r = 8$ 是 $8 times 8192 = 65.5$ K（*0.39%*，省 256 倍）；$r = 16$ 是 131 K（0.78%，常用起点）；$r = 64$ 是 524 K（3.1%，接近 full FT 表现的经验上限）。

#insight[
  *LoRA 省得最多的其实是优化器状态，不是参数本身。* AdamW 给每个可训练参数存 $m$、$v$ 两份 fp32（参数量的 8 倍），加上 fp32 master weight 是 12 倍。冻结 base 后这 12 倍几乎全部归零。

  *LoRA 不省 activation 显存。* 反向仍要经过冻结的 base 权重把梯度传到输入，所以前向激活照存不误（`requires_grad=False` 只让参数不建图，不影响 activation 保存，见第 6 章）。"用了 LoRA 还是 OOM"通常就是这个原因，得靠 gradient checkpointing 解决。
]

=== `alpha / r`：一个流传很广的错误理解

标准说法是"$alpha \/ r$ 让 $Delta W$ 的量级与 $r$ 无关，所以换 $r$ 不用重调 lr"。*这个说法是错的*，而且能当场用两行代码证伪。

设 $A$、$B$ 的元素独立、标准差分别 $sigma_A$、$sigma_B$，则 $(B A)_(i j) = sum_(k=1)^r B_(i k) A_(k j)$ 是 $r$ 个独立项之和，方差相加、标准差按 $sqrt(r)$ 长：

#formula[$ "std"[(B A)_(i j)] = sqrt(r) sigma_A sigma_B prop sqrt(r) quad (italic("不是") prop r) $]

于是三种缩放的实际量级：

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, left),
  [*缩放*], [$Delta W$ *量级*], [*后果*],
  [不缩放], [$prop sqrt(r)$], [$r$ 变大更新失控],
  [LoRA 的 $alpha \/ r$], [$prop 1 \/ sqrt(r)$], [*矫枉过正*：$r$ 越大更新越小],
  [rsLoRA 的 $alpha \/ sqrt(r)$], [$prop 1$], [真正与 $r$ 无关],
)

#warn[
  $alpha \/ r$ 除多了一个 $sqrt(r)$。这直接解释了一个非常常见的现象：*"把 $r$ 从 8 提到 64，效果不但没涨反而略掉，除非同时把 lr 也调大"* —— 有效更新量级被压了 $sqrt(64\/8) approx 2.8$ 倍。这正是 rsLoRA（rank-stabilized LoRA）主张改用 $alpha \/ sqrt(r)$ 的全部理由；如果你在用标准 LoRA，"提高 $r$ 时把 lr 按 $sqrt(r)$ 一起放大"就是这个修正的手工版本。

  自证（`test_lora_scaling_and_the_rslora_critique`）：固定 $sigma_B$，对 $r in {2, 8, 32}$ 各算一次 `(B @ A).std()`。不缩放时 $r$ 从 2 到 32（16 倍）量级涨 $sqrt(16) = 4$ 倍；乘上 $alpha\/r$ 后反过来缩小 4 倍；用 $alpha\/sqrt(r)$ 则三个 $r$ 基本持平（相对差 $< 15%$）。
]

=== 合并回主干、QLoRA、以及怎么自证

推理时 `W' = W + (alpha/r) * B @ A` 合并进主干，之后就是一个普通 `Linear`，*零额外延迟*。不合并的代价是每层多两次小 GEMM（对 batch=1 解码尤其不划算，全是 memory-bound 的 kernel launch）。约束是合并后没法在同一个 base 上热切换多个 adapter —— 多租户 LoRA 服务（S-LoRA、vLLM 的 multi-LoRA）反而故意*不*合并，把多个 adapter 的 $A$、$B$ 批起来做一次 grouped GEMM。

*QLoRA 一句话*：把冻结的 base 权重量化到 4-bit（NF4）存着，前向时反量化成 bf16 参与计算，LoRA 旁路仍是 bf16 全精度训练。因为 base 不更新，量化误差是固定偏置而不是累积误差，所以撑得住 —— 效果是 65B 模型能在单张 48GB 卡上微调。

```python
# 1) 初始等价性：必须是 rtol=0, atol=0 的逐位相同，不是"接近"
torch.testing.assert_close(lora(x), lora.base(x), rtol=0, atol=0)
assert (lora.lora_B == 0).all() and (lora.lora_A != 0).any()

# 2) 只有 A/B 有梯度；且初始时 A 的梯度恒为 0（因为它依赖 B）
lora(x).pow(2).sum().backward()
assert lora.base.weight.grad is None and lora.lora_B.grad.abs().sum() > 0
torch.testing.assert_close(lora.lora_A.grad, torch.zeros_like(lora.lora_A))

# 3) 参数量与合并等价性
assert lora.num_trainable() == r * (in_f + out_f)
merged = F.linear(x, lora.merged_weight(), lora.base.bias)   # B 已改成非零
torch.testing.assert_close(lora(x), merged, rtol=1e-5, atol=1e-6)
```

第 2 条的最后一行断言一举两得：既验证了"B=0 $arrow.r.double$ A 的梯度为 0"这个数学结论，也证明了"两边都置零会死锁"的推理是对的。

== 面试考点

#interview[
  *Q1*：`Conv2d(kernel=P, stride=P)` 为什么等价于"切 patch + 共享 Linear"？ViT 的序列长度是多少？

  A：stride 等于 kernel 时卷积的感受野恰好互不重叠、不遗漏，每个输出位置对应一个 patch；卷积核 `(D, C, P, P)` reshape 成 `(D, C*P*P)` 就是那个 Linear 的权重，卷积的权重共享对应"所有 patch 过同一个 Linear"。序列长度是 $N + 1 = (H\/P)(W\/P) + 1$，所以 `pos_embed` 是 `(1, N+1, D)` —— cls token 也占一个位置槽。换分辨率微调时 $N$ 变了，必须把 `pos_embed` 的 patch 部分 reshape 成 $sqrt(N) times sqrt(N)$ 做二维插值再拉平（cls 那一行单独留），这是最容易漏的一步。
]

#interview[
  *Q2*：为什么说 ViT 需要大数据？

  A：CNN 把"局部性"和"平移等变"硬编码在结构里，等于免费拿到一堆先验；ViT 除了"patch 内是局部的"之外几乎没有归纳偏置，全靠数据学。所以在 ImageNet-1k 规模从头训会输给 CNN，到 JFT-300M 级别才反超 —— 数据够多时学出来的先验比手工的更好。DeiT 用蒸馏 + 强增强在 1k 规模补上了这个 gap。另外 attention 是 $O(N^2)$，$P$ 从 16 减到 8 计算量涨 16 倍，这是 ViT 上不去高分辨率、Swin 要做窗口注意力的原因。
]

#interview[
  *Q3*：CLIP 的 loss 为什么要 L2 normalize、为什么对称、`logit_scale` 为什么可学习且用 `exp`？

  A：归一化让内积变成余弦相似度、限死在 $[-1,1]$，堵死"靠撑大模长降 loss"这条不学语义的捷径，把表示（方向）和置信度（尺度）解耦。对称是因为单向只约束一个检索方向，模型可以把所有文本挤到一起、只让图片可分就把 loss 降下去；双向把 $(B,B)$ 矩阵按行和按列都逼近单位阵才是真对齐。`logit_scale` 可学习是因为余弦相似度太窄、softmax 几乎均匀、梯度学不动，必须乘一个大温度倒数（CLIP 收敛到约 100）；`exp` 参数化保证恒正且让优化在对数尺度进行；还要 clamp 到 $ln 100$，否则"无脑增大 scale"本身就能降 loss，$s$ 会爆炸到 softmax 溢出。
]

#interview[
  *Q4*：为什么 batch size 对对比学习特别重要？随机初始化时 loss 应该是多少？

  A：负样本数就是 $B - 1$，全部来自 batch 内、没有别的来源；InfoNCE 作为互信息下界，这个下界的上限是 $log B$，所以 $B$ 越大信息量上界越高（CLIP 用 32k）。随机初始化时高维 embedding 两两近似正交，logits 矩阵近似全 0，softmax 退化成 $1\/B$，于是 loss $= log B$（$B = 32$ 是 3.47）。训练时 loss 卡在 $log B$ 不降，说明根本没学到对齐 —— 这是最好用的 sanity check。分布式实现要注意 `dist.all_gather` 不可微，只 gather embedding 会静默算出错梯度。
]

#interview[
  *Q5*：MoE 的 load-balancing aux loss 是什么？为什么均匀时最小值是 1？

  A：$"aux" = E sum_e f_e P_e$，其中 $f_e$ 是硬分配频率（来自 `topk`，*不可导*），$P_e$ 是软概率均值（*可导*，梯度从这里流回 router）。相乘的效果是"既被大量选中又被给高分"的 expert 被重罚。因为硬分配来自软概率、训练中 $f approx P$，aux $= E sum_e f_e^2$，由 Cauchy--Schwarz $sum_e f_e^2 >= 1\/E$，所以 aux $>= 1$、均匀时取等；全挤一个 expert 时是 $E$。所以 aux 落在 $[1, E]$，是现成的路由健康度指标。常见错法是只用 $P_e$ 的方差做 loss —— 那只约束了软概率，真正决定显存和通信的硬分配仍可能极不均衡。
]

#interview[
  *Q6*：MoE 的 router softmax 为什么必须在 fp32 里算？

  A：数值上 fp16 撑不住尖锐的 router logits，`exp(11.1)` 就超过 65504 直接 `inf`；bf16 虽不溢出但只有 8 位尾数，top-1 和 top-2 的概率可能变成同一个浮点数。更致命的是 `topk` 的 tie-breaking 依赖精确比较，一旦并列，不同 rank 可能选出不同 expert，expert-parallel 下 all-to-all 的收发形状对不上，表现为随机 hang 或 shape 错 —— 非常难查。所以 Switch / Mixtral 一律 `.float()` 算完再 cast 回去。
]

#interview[
  *Q7*：MoE 省显存吗？expert collapse 怎么诊断？

  A：*不省。* 它省的是 FLOPs（top-2/64 只算 $2\/64$ 的计算量），但所有 64 个 expert 的权重都得装在显存里，参数和优化器状态一分不少；activation 也不省，还多一层 dispatch/combine 的记账。所以 MoE 必须配 expert parallel 把 expert 拆到不同卡上，真实价值是"同等训练 FLOPs 下容纳更多参数"。collapse 的诊断不能看 loss 曲线（坍缩后就是个参数虚高的 dense 模型，loss 照样降），要看 aux 离 1 有多远、expert 的 token 计数直方图、丢弃 token 比例。修法：调大 aux 系数、router 加噪、加 z-loss、或换 Expert Choice 路由。
]

#interview[
  *Q8*：写出 DDPM 前向的闭式解，并说明训练目标为什么是"预测噪声的 MSE"。

  A：$x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon$，$overline(alpha)_t = product_(s <= t)(1 - beta_s)$。推导的关键一步是把逐步加噪递归展开，两个独立高斯之和仍是高斯、方差相加，得 $alpha_t beta_(t-1) + beta_t = 1 - alpha_t alpha_(t-1)$，归纳即得；系数平方和为 1 所以是方差保持的。重要性：训练时随机采一个 $t$ 就能一步构造 $(x_t, epsilon)$ 样本对，否则训练成本乘 $O(T)$。目标是 MSE 因为完整 ELBO 展开后每项是 $w_t norm(epsilon - epsilon_theta)^2$，而 DDPM 发现把 $w_t$ 直接设为 1 效果反而更好（原权重给低噪声时刻压了过大权重）。预测 $epsilon$ 和预测 $x_0$ 可互换，但前者隐含给高噪声时刻更大权重，实测样本质量更好。
]

#interview[
  *Q9*：`linear_beta_schedule` 换 $T$ 会出什么问题？最后一步采样为什么不加噪？

  A：默认 $beta in [10^(-4), 0.02]$ 是为 $T = 1000$ 调的。$overline(alpha)_T approx exp(-sum beta_t)$，而 $sum beta_t approx 0.01 T$ 与 $T$ 成正比：$T = 1000$ 时终点 $overline(alpha) approx 4 times 10^(-5)$（合格），$T = 100$ 时变成 0.36，$x_T$ 里还留着 60% 的原始信号，*前向根本没到纯噪声*。而反向采样从 $cal(N)(0,I)$ 起步，训练和采样的分布对不上，生成结果偏灰偏糊，但 loss 曲线完全正常 —— 静默失效。cosine 按归一化的 $t\/T$ 定义，天然对 $T$ 免疫。最后一步不加噪是因为 $x_0$ 就是最终输出，再加噪等于往结果里掺噪点，对应 `if i > 0` 那一行，漏了会得到一层可见颗粒。
]

#interview[
  *Q10*：LoRA 的 B 为什么必须为 0？`alpha / r` 真的让 $Delta W$ 的量级与 $r$ 无关吗？LoRA 省了什么？

  A：B=0 保证初始 $Delta W = B A = 0$、模型逐位等价于原模型，否则一上来就叠了个随机扰动破坏预训练知识。两边都置零不行：$partial cal(L)\/partial A prop B^tack.b(dots) = 0$ 且 $partial cal(L)\/partial B prop (dots)A^tack.b = 0$，梯度恒零、对称性未破缺。`alpha / r` *不是*无关 —— $(B A)_(i j)$ 是 $r$ 项之和、标准差 $prop sqrt(r)$ 而不是 $prop r$，所以除以 $r$ 后实际量级 $prop 1\/sqrt(r)$，*矫枉过正*，这正是"提高 $r$ 却不涨点、必须同时调大 lr"的原因，rsLoRA 的修正是改用 $alpha\/sqrt(r)$。省的是可训练参数（4096 方阵、$r=8$ 时 0.39%）以及*占大头的优化器状态*（AdamW 的 $m$、$v$ 加 master weight 是参数量的 12 倍）；*不省 activation*，因为反向仍要过冻结的 base 把梯度传到输入。推理时可合并回主干做到零延迟，代价是没法热切换多 adapter。
]
