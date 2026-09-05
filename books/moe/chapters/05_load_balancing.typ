#import "../template.typ": *

= Load Balancing：Aux Loss、Z-Loss、Bias Tuning

MoE 训练面临一个根本冲突：*Router 要学最优路由*，但如果放任不管，router 学到的最优路由往往是"永远选那几个专家"——其他专家永远收不到梯度、变成 dead weights。

这一章讲三种（历史演进的）解决方案：GShard 时代的 aux loss、ST-MoE 加的 z-loss、DeepSeek-V3 的 aux-loss-free bias tuning。

== 问题重述：Expert Collapse

第 3 章"Backward"一节讲过 topk 反向传播只流 K 个专家。这意味着：

*如果 router 从未选过 expert $e$*，那么 $ell_e$（对应的 gate logit）从未有过直接梯度信号。它*只能*通过 softmax 的分母被间接影响——但这个梯度非常弱，尤其在 renorm 或 sigmoid gating 下几乎为零。

具体化：假设初始化时 router 有轻微偏好 expert 0，第一 step 后 expert 0 收到的 token 稍多、参数更新、下 step 更受青睐——正反馈循环 → expert 0 独占 → 其他 expert 死掉。

*现象*：训练时 $"std"(f_e) / "mean"(f_e)$ (per-expert token 比例的变异系数) 持续 $> 0.5$ 且不下降。

#warn[
  Expert collapse 不是罕见事件——*没有* load balancing 机制的 MoE 训练几乎 100% collapse。所有主流 MoE 都必须有某种均衡机制。选哪种是 trade-off，不选是 bug。
]

== 方案 1: GShard/Switch Aux Loss

Lepikhin et al. 2020 (GShard) 提出的 auxiliary loss，也是最经典的一种。

=== 数学定义

对一个 batch (总 $N$ 个 token)，定义两个 $E$ 维向量：

$ f_e = 1/N sum_(n=1)^N bb(1)[e "是 top-1 for token" n] $

即 "expert $e$ 被 token 作为 top-1 选中的比例"。$f in [0, 1]^E$，$sum f = 1$。

$ P_e = 1/N sum_(n=1)^N "gate_probs"[n, e] $

即 "gate 输出的 expert $e$ 概率的 batch 平均"。$P in [0, 1]^E$，$sum P = 1$。

Aux loss:

$ L_"aux" = E dot sum_(e=1)^E f_e dot P_e = E dot (f dot P) $

=== 为什么这样定义？

*理想情况*：$f_e = P_e = 1/E$（完全均匀），此时 $L_"aux" = E dot E dot (1/E)^2 = 1$——达到最小值。

*不均衡情况*：某个 expert 特别受宠，$f_e = P_e approx 1$，其他 $= 0$，$L_"aux" = E dot 1 dot 1 = E$——远大于 1，惩罚被放大 $E$ 倍。

关键设计：

- *$f_e$ 不可微*（indicator function），但*不 backprop 也没关系*——它只作为"这个 batch 实际发生了什么"的 scalar。
- *$P_e$ 可微*，通过 $P_e dot f_e$ 的乘法收到梯度。梯度大小与实际负载 $f_e$ 成正比：expert 越过载，$P_e$ 收到的抑制越强。

*直觉*：$L_"aux"$ 是 "预期负载 × 实际负载" 的相关系数放大版；当分布均匀时最小。

=== 完整训练 loss

$ L_"total" = L_"CE" + alpha sum_"layers" L_"aux"^((ell)) $

系数 $alpha$ 通常 $0.01$。太大会让 router 过分追求均匀、伤害精度；太小 collapse。

=== 实现

```python
def switch_aux_loss(gate_probs, expert_indices, num_experts):
    """
    gate_probs: (N, E) - router 输出概率
    expert_indices: (N, K) - topk 后的 expert id (int)
    """
    N, E = gate_probs.shape
    # f_e: top-1 命中率
    top1 = expert_indices[:, 0]  # (N,)
    f = torch.bincount(top1, minlength=E).float() / N  # (E,)

    # P_e: gate_probs 均值
    P = gate_probs.mean(dim=0)  # (E,)

    return E * (f * P).sum()
```

*Warning*: `bincount` 在 GPU 上是 sync 操作（要跨 block 归并），在 forward 里频繁调用会拖速度。生产实现用 `torch.scatter_add_` 或 pre-allocated buffer。

=== 单纯 aux loss 的问题

*(1) 梯度冲突*

Router 收到两种梯度：主 loss 说"这个 token 应该去 expert 3"，aux loss 说"你太偏爱 expert 3 了，去别的"。两个梯度可能同号可能异号——异号时 router 停在次优解。

*(2) 对 top-K 的偏差*

$f_e$ 只统计 *top-1*，忽略 top-2, top-3。如果 aux loss 只监督 top-1，router 可能学到 "top-1 均匀，但 top-2 全部去 expert 0"——绕开监督。修正：

$ f_e = 1/(N K) sum_(n, k) bb(1)[i_(n,k) = e] $

统计所有 top-K。这是 Switch/ST-MoE 用的版本。

*(3) $L_"aux"$ 的 scale 依赖 batch size*

$f_e$ 是 batch 内比例，batch 越大波动越小、$L_"aux"$ 越接近理论最小值——即使 router 没变好，$L_"aux"$ 也自然下降。评估 collapse 不能只看 $L_"aux"$。

== 方案 2: Router Z-Loss (ST-MoE)

Zoph et al. 2022 (ST-MoE) 在 aux loss 基础上加了第二项——*router z-loss*。

=== 目的

约束 `logsumexp(gate_logits)` 不要过大，防止：

- Router 在训练早期把 logit 推到 $10^3$ 量级 → softmax 极度锐化 → 熵坍缩
- fp16 溢出（`exp(gate_logits)` overflow）

=== 数学

$ L_z = 1/N sum_(n=1)^N (log sum_(e=1)^E exp(ell_(n,e)))^2 $

即 log-sum-exp 平方的 batch 平均。系数 $beta approx 0.001$。

$ L_"total" = L_"CE" + alpha sum_ell L_"aux"^((ell)) + beta sum_ell L_z^((ell)) $

=== 为什么"平方"

$L_z$ 就是要惩罚 |lse| 大——对称。平方比绝对值有更好的梯度（0 附近梯度小，惩罚集中在极端值）。

#insight[
  z-loss 本质是 *regularizer*，不是均衡机制。它是"aux loss + z-loss" 组合的一部分——单独用 z-loss 不能防止 collapse，但配上 aux loss 能显著改善训练稳定性和推理时的 fp16 兼容性。ST-MoE、GLaM、Palm-2 都用这套。
]

== 方案 3: DeepSeek-V3 的 Aux-Loss-Free 均衡

DeepSeek-V3 (2024) 提出一种*不用 aux loss* 的方法，用 *可学习偏置*：

=== 核心思路

给每个 expert 一个 scalar bias $b_e$，*只在路由决策时生效*：

$ hat(ell)_(n, e) = ell_(n, e) + b_e $
$ (w, i) = "topk"(sigma(hat(ell)), K) quad ("注意用 sigmoid 而非 softmax") $

*但计算 gate 权重*时仍用 $ell$（不含 bias），保证 $b$ 只影响*选谁*、不污染 gate 权重梯度：

$ tilde(w)_(n, k) = sigma(ell_(n, i_(n,k))) $

（DeepSeek-V3 的具体公式里还有一步 $K$ 内的归一化 + normalization。核心是 bias 只影响 top-k selection。）

=== 动态调整 $b_e$

$b_e$ *不是通过梯度下降学*——它是训练循环里手动调整的参数（放在 `buffer` 而不是 `parameter` 里）：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DeepSeekV3Router(nn.Module):
    """
    Aux-loss-free router (DeepSeek-V3, 2024).
    - gate logits + learnable bias 决定路由;
    - gate 权重 (发给下游 combine) 只用 gate logits (不含 bias);
    - bias 通过 rule-based 反馈调整, 不参与梯度.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 bias_lr: float = 1e-3):
        super().__init__()
        self.E, self.K   = num_experts, top_k
        self.gate        = nn.Linear(hidden_size, num_experts, bias=False)
        # bias 不参与 autograd — 用 buffer, 手动更新
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.bias_lr     = bias_lr

    def forward(self, x):
        # x: (N, H)
        logits = self.gate(x)                              # (N, E), 参与 autograd
        # 路由用 logits + bias, sigmoid 打分 (DeepSeek 用 sigmoid, 不是 softmax)
        scores = torch.sigmoid(logits + self.expert_bias)  # (N, E)
        # top-K 选择
        _, indices = torch.topk(scores, self.K, dim=-1)    # (N, K)

        # 给下游用的 combine 权重: 只用 logits (不含 bias!), 保持梯度纯净
        gate_scores = torch.sigmoid(logits)                # (N, E)
        weights = torch.gather(gate_scores, -1, indices)   # (N, K)
        weights = weights / weights.sum(-1, keepdim=True)  # renorm
        return weights, indices, logits

    @torch.no_grad()
    def update_bias(self, indices: torch.Tensor):
        """
        每个 step 结束调用 (或每 N step 攒一次).
        indices: 本次 forward 的 top-K expert ids, shape (N, K).
        """
        # 统计每个 expert 被选中的次数
        load = torch.bincount(indices.flatten(), minlength=self.E).float()
        mean = load.mean()
        # 过载 → bias↓; 欠载 → bias↑
        # sign(load - mean) ∈ {-1, 0, +1}, 更新方向反过来
        self.expert_bias -= self.bias_lr * torch.sign(load - mean)
```

$"bias_lr"$ 是小学习率（DeepSeek-V3 论文里用 0.001）。等价于反馈控制：过载专家的 bias 减少 → 下次更难被选 → 负载下降；欠载相反。

*分布式注意*：`load` 必须先 AllReduce 到全局（跨 DP + EP 组）再更新 bias，否则每 rank 各自往不同方向调，训练发散。生产实现：

```python
if dist.is_initialized():
    dist.all_reduce(load, op=dist.ReduceOp.SUM)
```

=== 为什么不用梯度？

因为 $b$ 只影响 topk 的*索引选择*（离散决策）——从 $b$ 到最终 loss 的映射经过 topk 这个不可导算子，梯度是 0。手动 rule-based 调整绕开了这个不可导性。

#insight[
  Aux-loss-free 的关键优势：*router 的梯度信号纯净*——只从主任务学"哪个专家对这个 token 好"，不需要同时学"我要负载均匀"。DeepSeek-V3 报告在 671B 规模上比 aux-loss 版本 loss 更低。
]

=== 代价

- 需要 sigmoid gating（不是 softmax），部分改变数值动力学。
- 需要额外的动态更新逻辑，不能纯 forward。
- 效果对 $gamma$ 敏感。

== 方案对比

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, center, left),
    [*方案*], [*Router 梯度纯度*], [*实现复杂度*], [*代表*], [*备注*],
    [Aux loss],           [中 (有干扰)], [低], [GShard, Mixtral], [alpha=0.01],
    [Aux + z-loss],       [中],           [中], [ST-MoE, GLaM],  [beta=0.001],
    [Aux-loss-free bias], [高 (无干扰)],  [中],  [DeepSeek-V3],  [gamma=0.001],
    [Expert Choice routing], [高],       [高],  [Zhou 2022],    [反向选 token],
    [Capacity + drop],    [(orthogonal)], [低], [Switch],       [限制上界],
  ),
  kind: table,
)

== Expert Choice：反向路由

Zhou et al. 2022 提出一种反直觉方案：*不是 token 选专家，而是专家选 token*。

每个专家从所有 $N$ 个 token 中，*按打分*选前 $C = "top-C"$ 个：

```python
# 常规: for each token, pick top-K experts
# expert choice: for each expert, pick top-C tokens

scores = gate_logits  # (N, E)
# 按 E 维 topk (每 expert 选 top-C 个 token)
top_scores, top_token_ids = torch.topk(scores.T, k=C, dim=-1)
# top_token_ids: (E, C)
```

结果：*每个专家恰好收 $C$ 个 token*，天然均衡；但*每个 token 可能被 0/1/2/… 个专家选中*，不是恰好 K。

$C = N K / E$ 时总计算量与 top-K 版本相同，但*同一个 token 可能被多个专家选*（如果它对多个专家都得分高），*也可能被 0 个专家选*（罕见但要处理，一般走 residual 跳过）。

优缺点：

- ✓ *完美负载均衡*，无需 aux loss
- ✓ Router 梯度纯净
- ✗ 破坏了 "每 token K 个专家" 的对称性——同一 token 收到的 expert 数量随机
- ✗ 训练与推理不一致（推理时通常仍用 top-K）
- ✗ 未成为主流（Google 内部用，开源社区少见）

== 实操：监控 collapse

训练时除了 loss 曲线，还必须监控以下指标（wandb 每 100 step log 一次）：

+ $f_e$ 直方图：per-expert token 比例
+ $"std"(f) / "mean"(f)$：均衡度指标（$< 0.1$ 良好，$> 0.5$ 需要介入）
+ Router 熵 $H(p) = -sum_e p_e log p_e$：随训练*先增后稳*才对
+ Drop rate：如果用 capacity，每 step 记录被 drop 的 token 数
+ $L_"aux"$ 值：应缓慢单调下降到 $approx 1$；稳定 $> 1.5$ 说明 collapse

一个够生产用的监控 hook（挂在 MoE 层上即可）：

```python
import torch, torch.distributed as dist

class MoEMonitor:
    """挂在每个 MoE 层, 每 log_every 个 step 汇总一次."""
    def __init__(self, num_experts: int, log_every: int = 100):
        self.E = num_experts
        self.log_every = log_every
        self.reset()

    def reset(self):
        self.load_sum   = torch.zeros(self.E, device="cuda")   # 累加每 step 的 load
        self.entropy_sum = torch.tensor(0.0, device="cuda")
        self.drop_sum    = torch.tensor(0.0, device="cuda")
        self.n_steps     = 0

    @torch.no_grad()
    def record(self, gate_probs: torch.Tensor, indices: torch.Tensor,
               n_dropped: int = 0):
        # gate_probs: (N, E)  softmax 后
        # indices:    (N, K)  被选专家
        load = torch.bincount(indices.flatten(), minlength=self.E).float()
        self.load_sum += load
        # 熵: 平均在 batch 上
        p = gate_probs.clamp_min(1e-9)
        self.entropy_sum += -(p * p.log()).sum(-1).mean()
        self.drop_sum    += n_dropped
        self.n_steps     += 1

    @torch.no_grad()
    def flush(self, step: int, writer=None):
        """返回一个 dict, 可以直接 log 到 wandb/tensorboard."""
        if dist.is_initialized():
            dist.all_reduce(self.load_sum)
            dist.all_reduce(self.entropy_sum)
            dist.all_reduce(self.drop_sum)

        f = self.load_sum / self.load_sum.sum()             # 归一化的 per-expert 比例
        stats = {
            "moe/load_std_over_mean": (f.std() / f.mean()).item(),
            "moe/router_entropy":     (self.entropy_sum / self.n_steps).item(),
            "moe/drop_rate":          (self.drop_sum    / self.n_steps).item(),
            "moe/min_load_frac":      f.min().item(),
            "moe/max_load_frac":      f.max().item(),
        }
        if writer is not None:
            for k, v in stats.items():
                writer.add_scalar(k, v, step)
            writer.add_histogram("moe/per_expert_load", f, step)
        self.reset()
        return stats


# 用法
monitor = MoEMonitor(num_experts=8, log_every=100)
for step, batch in enumerate(dataloader):
    weights, indices, gate_probs = router(batch)
    ...
    monitor.record(gate_probs, indices, n_dropped=drop_count)
    if step % monitor.log_every == 0:
        print(monitor.flush(step))
```

#note[
  见到 collapse 的三个补救顺序：(a) 检查 router weight init (std $≤ 0.02$)；(b) 提升 aux loss 系数（0.01 → 0.03 → 0.1）；(c) 换 aux-loss-free 或加 z-loss。如果都不行，可能是数据分布问题——不同专家对不同数据类型的偏好本身就有物理 skew（比如代码 vs 自然语言），一味均衡反而伤害精度。
]

== 面试考点

#interview[
  *Q1*: 为什么 aux loss 的定义里 $f$ 不可微没关系？

  A: $f$ 只做为 "已发生事件的 scalar"，与 $P$ 相乘后， $P$ 一侧可微、梯度通过 softmax 回传到 router。$f$ 相当于给 $P$ 加权重的"目标分布"——大 $f$（过载）的 expert 在其 $P$ 分量上有大梯度惩罚。
]

#interview[
  *Q2*: 为什么 aux loss 系数是 0.01 而不是 1.0？

  A: aux loss 只是辅助约束、不是主任务。系数太大 (>0.1) 会让 router 完全牺牲精度追求均匀，模型 loss 显著变差；太小 collapse。0.01 是经验值。DeepSeek 报告 aux-loss-free 在他们的规模下比 aux loss 好，可能因为 aux loss 系数在千亿模型上难以精调。
]

#interview[
  *Q3*: Z-loss 和 aux loss 的作用差异？

  A: aux loss 管*分布均衡*（防 collapse）；z-loss 管*数值稳定*（防 logit 爆炸）。前者约束 softmax 输出的形状，后者约束 softmax 输入的 magnitude。二者正交，可以（也应当）同时用。
]

#interview[
  *Q4*: DeepSeek 的 aux-loss-free 里 $b$ 不参与梯度，那 $b$ 是怎么初始化的？初始 collapse 怎么办？

  A: $b_e$ 初始化为 0——所有专家等价。第一个 batch 后统计负载、调整 $b$，几十个 step 内负载趋于均衡。文章里没细说 $gamma$ warmup 策略，但社区复现建议前 1000 step 用大一点的 $gamma = 0.01$，之后降到 0.001。
]

#interview[
  *Q5*: Expert Choice 里如果某 token 被 0 个专家选中怎么办？

  A: 通过 residual 分支跳过 MoE 层（$y = x$）。理论上极少发生（除非某 token 对所有专家都打分极低）；实操中往往加一个 shared expert（所有 token 都走）兜底，见 DeepSeek-MoE 的设计。
]

#interview[
  *Q6*: 如果 capacity + drop + aux loss 同时用，会不会冲突？

  A: 不会——它们互补。Aux loss 让 router 输出趋于均衡（减少 drop 概率）；capacity 是硬上限（防最坏情况）。生产上通常都开：aux loss $alpha = 0.01$ + capacity factor 1.25。DeepSeek-V3 更极端：既有 no-drop 目标又有 bias tuning，实测 drop 率 $< 0.1%$。
]
