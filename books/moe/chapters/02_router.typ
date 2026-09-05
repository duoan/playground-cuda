#import "../template.typ": *

= Router / Top-K Gating

Router 是 MoE 的"大脑"——每一 forward 都要给每个 token 决定"去哪些专家"。这是一个每步都执行、每层都执行、结果直接影响后续所有计算的模块。这一章我们把 router 从数学公式一直讲到 kernel 实现。

== 数学形式

给定 hidden state $x in RR^H$（单个 token），router 是：

$ ell = W_g x quad ("logits", ell in RR^E) $
$ p = "softmax"(ell) quad ("probs", p in RR^E) $
$ (w, i) = "top-k"(p, K) quad (w in RR^K, i in ZZ^K) $
$ tilde(w) = w / (sum_j w_j) quad ("renormalize") $

其中 $W_g in RR^(E times H)$ 是唯一的可学习参数——每层 MoE 一个。相对于专家 FFN，router 的参数量小到可以忽略（$E H approx 8 times 4096 = 3 times 10^4$ vs 单专家 $approx 10^8$）。

Batch 版本：输入是 $X in RR^(N times H)$（$N = B times S$），输出是：

- `expert_indices: (N, K)` — 每个 token 选的专家 id
- `expert_weights: (N, K)` — 每个 token 分给这些专家的权重（$sum = 1$）

对应 `test_moe.py`：

```python
gate_logits = self.gate(hidden_states)                # (N, E)
gate_probs  = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)
expert_weights, expert_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
```

== 用一个具体例子拉一遍 tensor

以 $N=4, E=4, K=2$ 为例。设 gate 输出 (softmax 后) 是：

#align(center)[
  #prob-heatmap(
    probs: ((0.10, 0.60, 0.05, 0.25),
            (0.40, 0.10, 0.45, 0.05),
            (0.05, 0.20, 0.15, 0.60),
            (0.30, 0.30, 0.30, 0.10)),
    row-labels: ("t0", "t1", "t2", "t3"),
    col-labels: ("E0", "E1", "E2", "E3"),
    topk: 2,
    title: "gate_probs (N=4, E=4), red border = top-2",
  )
]

Top-2 选出：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (center, center, center, center, center),
    [*token*], [*top-2 experts*], [*raw probs*], [*after renorm*], [*说明*],
    [t0], [E1, E3], [0.60, 0.25], [0.706, 0.294], [清晰主选],
    [t1], [E2, E0], [0.45, 0.40], [0.529, 0.471], [两个专家分数接近],
    [t2], [E3, E1], [0.60, 0.20], [0.750, 0.250], [清晰主选],
    [t3], [E0, E1], [0.30, 0.30], [0.500, 0.500], [路由熵高（"没主见"）],
  ),
  kind: table,
)

*Renormalization 前后* 数学不同：

- Renorm 前的 $w$：反映专家在*全 $E$ 个*里的相对置信；t3 的 (0.30, 0.30) 意味着 router 对 t3 没信心。
- Renorm 后的 $tilde(w)$：反映专家在*选中的 $K$ 个*里的相对权重；t3 的 (0.5, 0.5) 意味着两个专家平分。

#insight[
  Renorm 会*丢失*"router 对这个 token 有多少信心"这个信息。生产实现在这一步往往有两种取舍：Mixtral 用 `softmax(topk(logits))` 保留熵信息；GShard/本书 demo 用 renorm。DeepSeek-V3 用 sigmoid + bias（详见第 6 章"DeepSeek-V3 的 Aux-Loss-Free 均衡"），完全绕开了这个 trade-off。
]

从张量形状看，一次 router 前向：

#align(center)[
  #shape-pipeline(stages: (
    ("hidden_states", "(N, H)", "输入 token"),
    ("gate_logits", "(N, E)", "= X @ W_g, W_g: (E, H)"),
    ("gate_probs", "(N, E)", "softmax(fp32), 每行和=1"),
    ("topk output", "(N, K), (N, K)", "expert_weights, expert_indices"),
    ("renorm", "(N, K)", "每行和=1，可选"),
  ))
]

== Router 的三种变体

上面是最标准的一种。生产系统会看到几个变体：

*(A) softmax → topk → renorm* (本书 demo, GShard)

上文详解，*保留*所有 $E$ 个 logits 的相对信息，然后砍掉尾巴。

*(B) topk(logits) → softmax over K* (Mixtral)

```python
topk_logits, expert_indices = torch.topk(gate_logits, k=K, dim=-1)
expert_weights = F.softmax(topk_logits, dim=-1)
```

*不做全 $E$ 的 softmax*，直接对 top-k logits 做局部 softmax。计算量小（只对 K 个 exp），保留熵信息。缺点：反向传播时未选中的 $E - K$ 个 logits *完全无梯度*——router 早期训练收敛更依赖 aux loss。

*(C) sigmoid gating* (Switch 变种, DeepSeek-V3)

```python
gate_scores = torch.sigmoid(gate_logits)   # 每个专家独立打分, 无归一
```

不做 softmax，各专家独立 sigmoid 打分。DeepSeek-V3 在这个基础上加了动态 bias（详见第 6 章），实现无 aux loss 的负载均衡。

三种变体的差异不体现在推理精度上（可以互相 fine-tune 转换），但训练动力学差异明显：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, center, center, center),
    [*方案*], [*所有 logits 有梯度?*], [*保留熵信息?*], [*aux loss 依赖度*],
    [A. softmax+topk+renorm], [是 (通过 softmax)], [否 (被 renorm 抹掉)], [中],
    [B. topk+softmax], [否 (只有 top-K)], [是], [高],
    [C. sigmoid + bias], [是], [是 (无归一)], [无 (用 bias 替代)],
  ),
  kind: table,
)

== Backward：梯度怎么流？

Router 的反向传播比 dense FFN 微妙——因为 topk 是不可微操作。

*事实*：`torch.topk` 的 backward 只把梯度传给"被选中的 K 个位置"，其他 $E - K$ 位置的 logit 梯度是 0。

以变体 (A) 为例，$partial L / partial ell_j$ 只在 $j$ 是 top-k 之一时非零，梯度沿"prob-space renorm → softmax"链回传。

*结果*：如果 router 从未选过某个专家 $e$，那么 $ell_e$ 从未获得直接梯度信号——router 学不会去选它。这是 *expert collapse* 的直接来源。

#warn[
  "梯度只流 top-k" 是 MoE 的基本事实，不是 bug 也不是数值噪声。补救靠三条：(1) aux loss（第 6 章）；(2) router z-loss；(3) 权重初始化 + warmup，让早期路由本身就分散。DeepSeek 用可学习 bias 是第四条路。
]

*一个反直觉的问题*：既然梯度只流 top-k，为什么 Mixtral 变体 (B) 训练还能收敛？

答：靠 aux loss 给 router 一个"往均匀分布拉"的梯度——这个梯度是对 $p_i$ 求的（含所有 $E$ 个），会流回所有 logit。详见第 6 章的 aux loss 一节。

== 为什么 softmax 要用 fp32

`test_moe.py` 里这一句：

```python
gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)
```

明确指定 `dtype=torch.float32`。这不是可选优化，是*正确性要求*。

原因：softmax 里的 `exp(x)` 在 fp16/bf16 下容易 overflow / underflow：

- fp16: 最大值 65504，$exp(11) approx 6 times 10^4 approx$ 溢出边界，$x > 11$ 就 saturate。
- bf16: 指数位比 fp16 多，但尾数只有 7 bit，$sum "exp"$ 的精度差。

*后果如果不用 fp32*：所有 $p_i$ 变成 $1/E$，router 死掉——每个 token 均匀选专家，模型完全丧失路由能力。

#note[
  实践中 gate 的 Linear($H → E$) 输出本身可以是 bf16（因为 $E$ 很小，$W_g x$ 数值不易爆），只有 softmax 内部升 fp32。Torch 的 `F.softmax(..., dtype=torch.float32)` 会做 upcast + softmax + 输出还是 fp32，最后用户手动 `.to(bf16)`。
]

== Router 的 kernel 视角

从性能角度看，router 一步做的事：

$ ell = X W_g^T quad "(GEMM)" $
$ p = "softmax"(ell) quad "(每行独立)" $
$ (w, i) = "topk"(p, K) quad "(每行独立)" $

其中：

- GEMM: 形状 $(N, H) times (H, E)$，$N$ 大（$~10^4$）、$E$ 小（$~10$）、$H$ 中等（$~10^3$）。这是一个 skinny GEMM——tile 数少、可能 memory-bound。
- Softmax: 形状 $(N, E)$，每行 reduce，$E$ 很小（$~10$），一个 warp 甚至几个 lane 就搞定。
- Topk: 形状 $(N, E)$，每行选前 K 大。$E$ 小的时候直接 bitonic sort 或 $O(K E)$ 都行。

*Naive 实现*：分别用 `cublasGemmEx` + `softmax kernel` + `topk kernel` 三次 launch。中间 `logits (N, E)` 和 `probs (N, E)` 各写读一次 HBM。

*生产实现*：三步 fuse 成一个 kernel。中间态不落 HBM，只在寄存器/shared memory：

```
fused_router_kernel:
  read X row  (N-wise parallelism)
  compute W_g @ x → logits[E]         # in register (E small)
  softmax with fp32 accumulator       # in register
  bitonic topk-K                       # in register
  write expert_indices, expert_weights # 2 × (N, K) 输出
```

节省的 HBM 流量：一次 (N, E) 读 + 一次 (N, E) 写。当 $N = 10^6, E = 8, "fp16"$，约 $2 times 10^6 times 8 times 2 = 32$ MB——不多，但每层每 step 都要做，累加起来占端到端 forward 的 1-3%。

#insight[
  Router 的 kernel 优化收益有限（router 本身占端到端 5% 以内），但*所有生产框架都会 fuse 它*——一是因为写起来相对容易（问题结构简单），二是因为 kernel launch 数下降对 CUDA graph capture 友好。
]

对应的开源实现：vLLM `fused_moe.py::fused_topk`, SGLang `fused_moe_router`, Megatron-LM `fused_router_kernel`。

== Router 的可视化 debug

训练 MoE 有一个必备可视化：*每个 batch 里每个专家收到了多少 token*。

例：$E=8$ 时，理想是每个专家收 $N K / E$ 个 token。实际训练早期可能是：

#align(center)[
  #expert-load(
    (
      ("E0", 320),
      ("E1",  12),
      ("E2",   5),
      ("E3", 190),
      ("E4",   8),
      ("E5", 445),
      ("E6",  15),
      ("E7",  25),
    ),
    unit: "tokens",
    capacity: 128,
    width: 9,
    label-w: 1.2,
  )
]

红色虚线是 capacity 上限（如设 $C = "ceil"(N K / E times 1.25)$）。上图 E0/E3/E5 都会*超出 capacity 触发丢 token*，同时 E1/E2/E4/E6 严重欠载——这是典型的 expert collapse 早期表征，说明 aux loss 系数不够。

#note[
  生产训练时每 100 step 打印 $f_i$ (每专家 token 比例) 的直方图；如果 $"std"(f) / "mean"(f) > 0.5$ 持续超过 1000 step，通常需要：(a) 提升 aux loss 系数从 0.01 到 0.03；(b) 检查 router weight init（should be small, e.g. std=0.02）；(c) 考虑 warmup router 单独。
]

== 面试考点

#interview[
  *Q1*: 为什么 gate 的 softmax 必须 fp32？

  A: bf16/fp16 下 `exp` 容易饱和/下溢，$E$ 个 logits 差距被压缩，router 输出退化为均匀分布，模型丧失路由能力。fp32 只在 softmax 内部升精度，最终 gate_probs 可 downcast 回来 —— 计算量增加 $< 1%$，稳定性 huge。
]

#interview[
  *Q2*: `torch.topk` 会返回相同专家 id 两次吗？

  A: 不会。`topk` 返回的是 $E$ 个 logit 的 top-K *索引*，天然 unique。所以每个 token 的 `expert_indices[n, :]` 是 K 个不同 expert id，`index_add_` 里同一个 token_ids 值最多出现一次（在单个 expert_idx 循环内）。
]

#interview[
  *Q3*: Router 的反向传播中，未被选中的 $E - K$ 个 logits 怎么获得梯度？

  A: 三条路径：(1) 通过 softmax 分母 —— 变体 A 在 renorm 前，未选中 logits 仍参与 softmax 分母，有小梯度；(2) aux loss —— 直接对所有 $p_i$ 惩罚 (梯度流所有 logits)；(3) router z-loss —— 惩罚 `logsumexp(logits)^2`，也流所有 logits。变体 B (Mixtral) 只有 (2)(3)。
]

#interview[
  *Q4*: DeepSeek-V3 为什么弃用 aux loss？

  A: aux loss 会让 router 收到"两种冲突梯度"—— 一份来自主 loss (学"正确路由")，一份来自 aux loss (学"均匀分布")。冲突可能让 router 停在次优解。DeepSeek-V3 用可学习 bias 动态修正 (溢出专家 bias↓ / 欠载 bias↑)，*只影响路由决策*、不影响 gate 权重梯度，绕开冲突。见第 6 章。
]

#interview[
  *Q5*: 如果 router 的 `Linear(H, E)` 权重初始化太大会怎样？

  A: 初始 logits 方差大 → softmax 输出接近 one-hot → 早期路由固化到少数几个专家 → 其他专家永远收不到梯度 → 永久 collapse。Mixtral 官方 init 用 `std=0.02` 而不是 `std=1/sqrt(H)`，就是这个原因。
]

#interview[
  *Q6*: 一个 token 的 K 个专家选择顺序有意义吗？

  A: 数学上 —— 加权求和满足交换律，顺序无关。但 `k_ids` (topk 返回的位置索引) 有意义：它告诉你 "此 token 把这个专家当作 top-1 / top-2 / …"，用于从 `expert_weights (N, K)` 里取对应权重。顺序对*调试*和*aux loss 计算*有用，但对最终输出没影响。
]
