# Sparse Mixture-of-Experts (MoE) 完整指南

> 从零基础到工程实现、单机与分布式性能优化。
> 配套代码：`python/pytorch/test_moe.py`（本仓库最小可运行实现）。

---

## 目录

0. [直觉先行：一句话理解 MoE](#0-直觉先行一句话理解-moe)
1. [第 1 章：为什么要有 MoE？](#第-1-章为什么要有-moe)
2. [第 2 章：MoE 层的解剖图](#第-2-章moe-层的解剖图)
3. [第 3 章：Router / Top-K Gating 详解](#第-3-章router--top-k-gating-详解)
4. [第 4 章：Dispatch & Combine 的两种实现范式](#第-4-章dispatch--combine-的两种实现范式)
5. [第 5 章：逐行 Code Walkthrough（test_moe.py）](#第-5-章逐行-code-walkthroughtest_moepy)
6. [第 6 章：Load Balancing / Aux Loss / Router Z-loss](#第-6-章load-balancing--aux-loss--router-z-loss)
7. [第 7 章：单机训练性能优化](#第-7-章单机训练性能优化)
8. [第 8 章：分布式训练（EP / TP / DP / All-to-All）](#第-8-章分布式训练ep--tp--dp--all-to-all)
9. [第 9 章：数值稳定性、精度与常见陷阱](#第-9-章数值稳定性精度与常见陷阱)
10. [附录 A：符号表](#附录-a符号表)
11. [附录 B：延伸阅读](#附录-b延伸阅读)

---

## 0. 直觉先行：一句话理解 MoE

**Dense FFN**：每个 token 都走同一个大 FFN，参数量 = 计算量。
**MoE FFN**：把大 FFN 拆成 `E` 个"专家"（每个是小 FFN），每个 token 只走其中 `K` 个（`K ≪ E`）。

结果：**参数量 × E / K，计算量几乎不变**。这就是 MoE 的核心卖点——"更大模型，同等 FLOPs"。

```
Dense:                          MoE (E=4, K=1):

  token ──► [ BIG FFN ]           token ──► router ──► [expert_2]
  token ──► [ BIG FFN ]           token ──► router ──► [expert_0]
  token ──► [ BIG FFN ]           token ──► router ──► [expert_2]
  token ──► [ BIG FFN ]           token ──► router ──► [expert_3]
                                              │
                                              └─► expert_1 空闲

每个 token 都用同一份大权重       每个 token 只激活 1/E 的权重
```

---

## 第 1 章：为什么要有 MoE？

### 1.1 扩容瓶颈

Dense Transformer 提升能力有两条路：
- **加深 / 加宽**：参数量↑，FLOPs 也↑，训练/推理都变贵。
- **加数据**：数据不是无限的（尤其是高质量语料）。

**Scaling Law 的一个观察**：模型容量（参数）和计算量（FLOPs）在 dense 架构中是强绑定的。想要"更多知识"，就必须"更多计算"。

### 1.2 MoE 的破局思路

**条件计算 (Conditional Computation)**：不是每个 token 都需要"全部知识"。给它一个路由器，只激活相关的子网络。

- **总参数量 `P_total = E × P_expert`**（模型容量）
- **激活参数量 `P_active = K × P_expert`**（每个 token 的实际计算）
- **稀疏度 `K/E`**：Mixtral 8×7B 用 `E=8, K=2`，稀疏度 25%。

代价：
- 需要"路由"逻辑（多一层 Linear + top-k）
- 负载不均衡问题（有的专家忙死、有的闲死）
- 分布式通信复杂（跨机 all-to-all）

### 1.3 里程碑

| 模型 | 年份 | E | K | 备注 |
|---|---|---|---|---|
| Sparsely-Gated MoE (Shazeer et al.) | 2017 | 上千 | 2 | 首次大规模 MoE |
| GShard | 2020 | 2048 | 2 | Google，多语言翻译 |
| Switch Transformer | 2021 | 上万 | **1** | K=1 极简路由 |
| GLaM | 2021 | 64 | 2 | 1.2T 参数 |
| Mixtral 8x7B | 2023 | 8 | 2 | 开源 SOTA |
| DeepSeek-MoE / DeepSeekV2 | 2024 | 160+2(shared) | 6 | 细粒度 + 共享专家 |
| DeepSeek-V3 | 2024 | 256 | 8 | 671B 总参、37B 激活 |

---

## 第 2 章：MoE 层的解剖图

MoE 层是 **一个 Transformer block 内 FFN 子层的替换品**。其他部分（Attention、LayerNorm、残差）不变。

```
                     ┌───────────────── MoE Layer ─────────────────┐
                     │                                              │
   x (B, S, H) ──►   │  ┌────────┐   ┌─────────────────────────┐   │  ──► y (B, S, H)
                     │  │ Router │──►│ Dispatch → Experts → Combine│   │
                     │  └────────┘   └─────────────────────────┘   │
                     │       │                                     │
                     │       └──► gate_logits (用于 aux loss)        │
                     └──────────────────────────────────────────────┘
```

### 2.1 三大组件

1. **Router (Gate)**：一个 `Linear(H → E)` + softmax + top-k，输出：
   - `expert_indices ∈ (N, K)`：每个 token 选哪 K 个专家（专家 id）
   - `expert_weights ∈ (N, K)`：对应的加权系数（K 个和为 1）

2. **Experts**：`E` 个独立的 FFN，通常结构与 dense FFN 一致：
   - Mixtral 用 SwiGLU：`x → (W_up(x) * silu(W_gate(x))) → W_down`
   - 本仓库 demo 用最简的 `Linear → ReLU → Linear`

3. **Dispatch / Combine**：把 tokens **发**到对应专家、把专家输出**收**回原位置。工程实现的核心复杂度在这里。

### 2.2 端到端形状变化

以 `B=2, S=4, H=8, E=4, K=2` 为例：

```
input        (B=2, S=4, H=8)          原始输入
  │ view(-1, H)
  ▼
hidden       (N=8, H=8)                展平 batch × seq
  │ gate = Linear(H, E)
  ▼
gate_logits  (N=8, E=4)                每个 token 对每个专家的分数
  │ softmax + top-k
  ▼
expert_ids   (N=8, K=2)  ← 值 ∈ [0,E)
expert_wts   (N=8, K=2)  ← 归一化到 sum=1
  │ 逐专家：where + gather + FFN + scatter-add
  ▼
out          (N=8, H=8)                累加所有专家贡献
  │ view(B, S, H)
  ▼
output       (B=2, S=4, H=8)
```

---

## 第 3 章：Router / Top-K Gating 详解

Router 是 MoE 的"大脑"，决定每个 token 去哪些专家。

### 3.1 形式化

```
gate_logits  = X @ W_gate                # (N, E)
gate_probs   = softmax(gate_logits)      # (N, E)
(top_w, top_i) = topk(gate_probs, K)     # (N, K), (N, K)
weights      = top_w / top_w.sum(-1, keepdim=True)   # renormalize
```

### 3.2 Tensor 可视化（`N=4, E=4, K=2`）

**Step 1**: `gate_probs` 是每个 token 对每个专家的概率分布：

```
gate_probs =
             expert_0  expert_1  expert_2  expert_3
    token_0 [  0.10      0.60      0.05      0.25  ]
    token_1 [  0.40      0.10      0.45      0.05  ]
    token_2 [  0.05      0.20      0.15      0.60  ]
    token_3 [  0.30      0.30      0.30      0.10  ]
```

**Step 2**: Top-K=2 挑出每行最大的两个：

```
expert_indices =                    expert_weights (原始 top-k prob) =
    token_0 [  1,  3 ]                     [ 0.60, 0.25 ]
    token_1 [  2,  0 ]                     [ 0.45, 0.40 ]
    token_2 [  3,  1 ]                     [ 0.60, 0.20 ]
    token_3 [  0,  1 ]     (前两个)          [ 0.30, 0.30 ]
```

**Step 3**: 在 top-k 内 renormalize，使每行和为 1：

```
expert_weights (归一化后) =
    token_0 [ 0.706, 0.294 ]     ← 0.60/(0.60+0.25)
    token_1 [ 0.529, 0.471 ]
    token_2 [ 0.750, 0.250 ]
    token_3 [ 0.500, 0.500 ]
```

### 3.3 变体

| 方案 | 说明 | 代表 |
|---|---|---|
| **softmax → topk → renorm** | 本仓库、GShard | 最直观 |
| **topk → softmax(logits)** | 只对 top-k 的 logits 做 softmax | Mixtral |
| **Sigmoid gating** | 无归一化，各专家独立打分 | Switch 变种 |
| **Expert Choice** | 反过来：**每个专家挑固定数量的 token**，天然负载均衡 | Zhou et al. 2022 |

### 3.4 Noise / Jitter

训练时，Shazeer 2017 会加噪声鼓励探索：
```
gate_logits = X @ W_gate + softplus(X @ W_noise) * randn()
```
现代实现多数不用。DeepSeek 用 sigmoid + bias tuning 做负载均衡。

---

## 第 4 章：Dispatch & Combine 的两种实现范式

Router 出来之后，**如何把 token 送到 expert、把结果送回来**是 MoE 工程实现的核心。

### 4.1 范式 A：Scatter / Gather（教科书式，本仓库实现）

**思路**：外层循环遍历专家，每次找出所有路由到它的 token，跑一次小 GEMM。

```
for e in range(E):
    mask = (expert_indices == e)           # (N, K) bool
    token_ids, k_ids = where(mask)          # 命中的 (token, k) 位置
    x_e = hidden[token_ids]                 # gather: (M_e, H)
    y_e = experts[e](x_e)                   # (M_e, H)
    w_e = expert_weights[token_ids, k_ids]  # (M_e,)
    out.index_add_(0, token_ids, y_e * w_e[:, None])   # scatter-add
```

**优点**：清晰、无需自定义 kernel，PyTorch 一把梭。
**缺点**：
- E 次 kernel launch（当 E 大时 overhead 显著）
- 每个专家的 batch size `M_e` 变化，GPU 利用率抖动
- 无法把 E 个小 GEMM 融合成一个大 GEMM

**适用**：`E ≤ 16`、快速原型、学习理解。

### 4.2 范式 B：Permute + Grouped GEMM（生产级）

**思路**：先把 tokens 按 expert 排序打包，一次性用 **grouped GEMM** 处理所有专家。

```
Step 1  route:     每个 (token, k) → (expert_id, position)
Step 2  permute:   把 flatten 的 (N*K) 个 slot 按 expert_id 排序
                   得到 packed_input (N*K, H)，同时记录 inverse permutation
Step 3  group-gemm: 一次 kernel 完成所有专家的两层 FFN
Step 4  un-permute: 按 inverse perm 归位，乘以 gate 权重
Step 5  reduce:    top-k 内 K 项按 expert_weights 加权求和 → (N, H)
```

**可视化** (`N=3, K=2, E=3`)：

```
expert_indices =                       flatten & sort by expert →
    [[0, 2],
     [1, 0],       (t0,k0)=E0  ┐
     [2, 1]]       (t1,k1)=E0  ├─►  packed order:
                   (t1,k0)=E1  │     E0: t0k0, t1k1     ← 2 tokens
                   (t2,k1)=E1  │     E1: t1k0, t2k1     ← 2 tokens
                   (t0,k1)=E2  │     E2: t0k1, t2k0     ← 2 tokens
                   (t2,k0)=E2  ┘
```

Grouped GEMM 需要传入每个 group 的 offset：`group_sizes = [2, 2, 2]`。

**优点**：
- **1 次 kernel launch** 处理所有专家
- 可用 CUTLASS grouped-gemm、Triton、`torch._grouped_mm`
- 与 all-to-all（分布式）天然对齐

**缺点**：需要 permute/unpermute 的自定义算子，且要处理 dropped tokens（capacity）。

**代表实现**：Megablocks、Tutel、vLLM MoE、SGLang MoE、DeepSpeed-MoE。

### 4.3 范式 C：Block-Sparse GEMM（Megablocks）

Megablocks 的洞察：把 MoE 的所有专家权重拼成一个大 block-sparse 矩阵，用一次 **block-sparse GEMM**（SpMM）搞定，无需 padding、无需丢 token。

```
     ┌──────────────────────┐
     │ W_e0                 │
     │      W_e1            │
     │           W_e2       │
     │                W_e3  │
     └──────────────────────┘
      对角 block 结构，只有对应 block 有值
```

**优点**：0 padding、0 drop、原生变长；性能接近 dense GEMM。

---

## 第 5 章：逐行 Code Walkthrough（test_moe.py）

以下逐段解释本仓库的最小实现，配 tensor shape 注释。

### 5.1 Expert 定义


```13:28:/home/duo.an/workspaces/playground-cuda/python/pytorch/test_moe.py
class Expert(nn.Module):
    """一个普通的 2 层 FFN，用作单个专家。

    结构: Linear -> ReLU -> Linear，无 bias（与 Mixtral 对齐）。
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: Float[Tensor, "N H"]
    ) -> Float[Tensor, "N H"]:
        return self.fc2(F.relu(self.fc1(hidden_states)))
```

**说明**：一个专家就是一个普通 FFN。真实模型（Mixtral/DeepSeek）用 SwiGLU：
```python
return W_down(silu(W_gate(x)) * W_up(x))
```

### 5.2 Router 前向


```67:85:/home/duo.an/workspaces/playground-cuda/python/pytorch/test_moe.py
        hidden_states = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states.shape[0]

        # -------- 1. Router: 计算每个 token 被路由到各个专家的概率 --------
        # softmax 用 fp32 计算以保证数值稳定，最后再 cast 回原 dtype
        gate_logits = self.gate(hidden_states)  # (N, E)
        gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)

        # 取 top-k 专家及其原始概率
        # expert_weights: (N, K)，expert_indices: (N, K)
        expert_weights, expert_indices = torch.topk(
            gate_probs, k=self.top_k, dim=-1
        )
        # 在 top-k 内做一次归一化，使每个 token 的 K 个权重之和为 1
        # 注意: Mixtral 直接对 top-k logits 做 softmax，这里对概率再归一化，
        # 数值上等价于 sum-to-one，效果接近，实现更直观
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(dtype)
```

**Tensor 追踪** (`B=2, S=4, H=8, E=4, K=2` → `N=8`)：

```
hidden_states.view(-1, H)  ─►  (8, 8)
self.gate(·)               ─►  gate_logits: (8, 4)
softmax(dim=-1, fp32)      ─►  gate_probs:  (8, 4), 每行和=1
topk(K=2)                  ─►  expert_weights: (8, 2), expert_indices: (8, 2)
renormalize                ─►  expert_weights: (8, 2), 每行和=1
```

**为什么 softmax 要用 fp32？** 参见第 9 章 §9.1。

### 5.3 Dispatch 循环（重点）


```87:106:/home/duo.an/workspaces/playground-cuda/python/pytorch/test_moe.py
        # -------- 2. Dispatch: 逐专家计算 --------
        # 输出缓冲区，用 index_add_ 累加各专家的贡献
        out = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            # 找到所有把 expert_idx 选为 top-k 之一的 (token, k) 位置
            # token_ids: 命中该专家的 token 下标 (M,)
            # k_ids:     该 token 是把此专家选为第几个 (M,)，用于取对应权重
            # 其中 M 是路由到此专家的 token 数（负载不均衡）
            token_ids, k_ids = torch.where(expert_indices == expert_idx)
            if token_ids.numel() == 0:
                continue

            # 取出这些 token 的 hidden，喂给该专家
            expert_input = hidden_states[token_ids]  # (M, H)
            expert_output = self.experts[expert_idx](expert_input)  # (M, H)

            # 按 gate 权重缩放后累加到输出对应位置
            weights = expert_weights[token_ids, k_ids].unsqueeze(-1)  # (M, 1)
            out.index_add_(0, token_ids, (expert_output * weights).to(dtype))
```

**逐步图示** (以 `expert_idx=1` 为例，续用第 3.2 节的例子)：

```
expert_indices (8, 2) ==  1  →  mask (8, 2) bool:
                                [[T,F], [F,F], [F,T], [T,F], [F,F], [T,F], [F,T], [F,F]]

where(mask) →
    token_ids = [0, 2, 3, 5, 6]     ← 命中 expert_1 的 token 下标
    k_ids     = [0, 1, 0, 0, 1]     ← 各自把 expert_1 选为第几名

hidden_states[token_ids]  gather →  expert_input: (5, 8)
                                       ↓ Linear→ReLU→Linear
                                    expert_output: (5, 8)

expert_weights[token_ids, k_ids]  →  weights: (5,)
                                       ↓ unsqueeze
                                    weights: (5, 1)

out.index_add_(0, token_ids, expert_output * weights)
    ↑ 把结果按 token_ids 累加到 out 的对应行
```

**关键点**：
- `token_ids` 里可能有重复吗？**不会**——因为 `expert_indices` 的每一行是 top-k 内的 unique 专家 id，一个 token 不会两次选到同一个专家。所以 `index_add_` 每个 `token_ids` 值最多出现一次，退化为普通 scatter。
- 但**跨 expert_idx 循环**，同一个 token 会被多次 add 到 `out`（K 次），这正是 top-k 加权求和的实现。

### 5.4 完整 tensor 生命周期图

```
input (2,4,8)
   │ view
   ▼
hidden (8,8) ──────────────────────────────────┐
   │ gate                                       │
   ▼                                            │
gate_logits (8,4)                               │
   │ softmax                                    │
   ▼                                            │
gate_probs (8,4)                                │
   │ topk                                       │
   ▼                                            │
expert_indices (8,2)  expert_weights (8,2)      │
   │      │                                     │
   │      └────────┐                            │
   ▼               ▼                            │
for e in E:                                     │
    mask = idx==e   ─► token_ids, k_ids         │
                             │  ┌───────────────┘
                             ▼  ▼
                        gather → expert_e → weight → scatter-add → out (8,8)

out (8,8)
   │ view
   ▼
output (2,4,8)
```

---

## 第 6 章：Load Balancing / Aux Loss / Router Z-loss

### 6.1 问题：坍缩 (Expert Collapse)

如果不加约束，router 很容易学到"永远选同 K 个专家"的解——其他专家永远收不到梯度，等于浪费参数。

### 6.2 GShard/Switch Aux Loss

对每个 MoE 层附加辅助损失：

```
f_i = 该 batch 内路由到 expert i 的 token 比例        # (E,)
P_i = 该 batch 内 gate_probs 对 expert i 的平均概率   # (E,)
L_aux = E * Σ_i (f_i * P_i)
```

- `f_i` 不可微，但 `P_i` 可微；乘起来给 router 一个"往均匀分布拉"的梯度。
- 均匀分布时 `f_i = P_i = 1/E`，`L_aux = E * E * (1/E)^2 = 1`（最小值）。

系数通常 `α = 0.01`。

### 6.3 Router Z-loss (ST-MoE, Zoph et al. 2022)

约束 `logsumexp(gate_logits)` 不要太大，防止路由熵坍缩、防止 fp16 overflow：

```
L_z = (1/N) * Σ_n (logsumexp(gate_logits[n]))^2
```

系数通常 `β = 0.001`。

### 6.4 DeepSeek 的 Bias-tuning（无 aux loss 的负载均衡）

DeepSeek-V3 弃用 aux loss，改用 **可学习偏置** `b_i`：
- 路由时用 `gate_probs + b`
- 训练中动态调整 `b_i`：过载专家 `b_i -= γ`，欠载专家 `b_i += γ`
- 只影响路由决策，不影响加权（保持梯度纯净）

### 6.5 Capacity Factor（丢 token）

传统实现给每个专家一个"容量" `C = ceil(N * K / E * capacity_factor)`。超过 C 的 token 被丢弃（用残差路径跳过）。`capacity_factor = 1.25` 常见。

Megablocks 用 block-sparse 消除了这一步。

---

## 第 7 章：单机训练性能优化

### 7.1 瓶颈画像

跑一次 MoE forward 的时间分布（Naive PyTorch 实现，`E=8, K=2`）：

```
Router (Linear + softmax + topk)     ~5%
Dispatch (E 次 gather)               ~15%
Expert compute (E 次小 GEMM)         ~60%   ← 每个 GEMM 太小，不饱和
Combine (E 次 scatter_add)           ~15%
其他                                  ~5%
```

**核心问题**：E 次小 GEMM，SM 利用率低；kernel launch overhead 显著。

### 7.2 优化 1：Grouped GEMM（最重要）

把 E 个 `(M_e, H) @ (H, I)` 打包成一个 grouped GEMM：

```python
# PyTorch 2.4+
y = torch._grouped_mm(x_packed, w_stacked, group_sizes)  # 1 次 kernel
```

- CUTLASS `GroupedGemm` / Triton `grouped_matmul` / `torch._grouped_mm`
- **收益**：kernel launch 减少 `E→1`，SM 利用率↑，端到端 forward 常见 **2–4× 提速**。

### 7.3 优化 2：Permute / Unpermute 融合

Naive 实现在 dispatch/combine 阶段有多次 memory 往返。融合的做法：
- `permute_kernel`: 一次读原 hidden，一次写 packed hidden
- `unpermute_kernel`: 一次读 expert 输出，加权 + reduce，一次写 out

**代表**：`grouped_gemm.ops.permute/unpermute`（NVIDIA），Tutel `fast_dispatch`。

### 7.4 优化 3：Fused Router

把 `Linear + softmax(fp32) + topk + renorm` 融成一个 kernel，避免中间 `(N, E)` 的多次读写。SGLang、vLLM 都有 `fused_moe_router` kernel。

### 7.5 优化 4：Fused MoE Layer（激进）

极限做法：`router + permute + grouped_gemm + activation + grouped_gemm + unpermute + combine` 融成 **1 个** kernel。Triton 里可以实现，但灵活性差。

- 代表：vLLM `fused_moe_kernel`（推理侧）、DeepGEMM。

### 7.6 优化 5：Activation Recomputation

MoE FFN 的中间激活（`intermediate_size` × 4 常见）占显存大头。selective checkpointing：
- 只保存 expert 的 **输入** 和 **输出**（不保存中间 `I` 维激活）
- backward 时重跑一次 `fc1 + activation`
- 换约 30% 显存 vs 33% 额外算力

### 7.7 优化 6：TorchCompile

`torch.compile(model)` 对 MoE 的收益要看实现：
- **Naive scatter/gather 实现**：编译帮助有限（graph break 在 `where` + 数据依赖控制流）
- **静态 permute + grouped_gemm 实现**：可以吃到 fusion 红利

### 7.8 优化 7：显存优化

| 手段 | 收益 | 代价 |
|---|---|---|
| Expert 权重 bf16/fp8 | 显存 -50%/-75% | 精度 |
| Optimizer states 分片 (ZeRO-1/2/3) | 显存 -N× | 通信 |
| CPU offload experts | 大 | 慢 |
| MoE-specific: 冷专家 offload | 中 | 复杂 |

### 7.9 优化 8：Skew / Straggler 缓解

由于负载不均，某些专家 `M_e` 远大于均值，成为拖尾。缓解：
- **Drop tokens**（capacity）：牺牲少量精度
- **Expert Choice routing**：反向选择，天然均衡
- **Padding + block-sparse**（Megablocks）：让所有 group 对齐 block 边界

### 7.10 单机优化清单

```
□ 用 grouped GEMM 替代 for-loop expert
□ Router / permute / unpermute 各自 fuse
□ FFN 里 activation 用 SwiGLU 融合 kernel
□ bf16 训练 + fp32 softmax
□ Selective activation checkpointing
□ 打开 SDPA / FlashAttention（attention 部分）
□ profile 一遍看 expert size 分布是否严重 skewed
```

---

## 第 8 章：分布式训练（EP / TP / DP / All-to-All）

MoE 的分布式 = 常规 3D parallelism (DP × TP × PP) **加上** Expert Parallel (EP)。

### 8.1 四种并行的分工

| 并行 | 切什么 | 通信原语 |
|---|---|---|
| **DP** (Data Parallel) | 切 batch | AllReduce (grad) |
| **TP** (Tensor Parallel) | 切一个 op 内的张量 | AllReduce / AllGather |
| **PP** (Pipeline Parallel) | 切 layer | P2P send/recv |
| **EP** (Expert Parallel) | 切 experts（每卡放不同专家） | **All-to-All** |

### 8.2 Expert Parallel（核心）

**思路**：`E` 个专家分布在 `EP_size` 张卡上，每卡持有 `E / EP_size` 个专家。

每个 MoE 层的通信模式：

```
       每卡有本地的 tokens X_local (N_local, H)
       │
       ▼
   Router 在本地计算 (每卡独立)
       │
       ▼ ── all_to_all_1 ──►  tokens 按目标 expert 卡重排
                              每卡收到的是 "应该由我处理的 tokens"
       │
       ▼
   本地 Grouped GEMM 处理本地 experts
       │
       ▼ ── all_to_all_2 ──►  结果送回原 token 所在卡
       │
       ▼
   Combine + 加权 → 输出
```

**代价**：每个 MoE 层 2 次 all-to-all（fwd）+ 2 次（bwd），是 MoE 分布式训练的**主要通信开销**。

### 8.3 通信量分析

设：
- `N_local` = 每卡 token 数
- `K` = top-k
- `H` = hidden dim
- `EP_size` = expert parallel 世界大小

每次 all-to-all 每卡 send/recv 数据量：
```
data = N_local * K * H * dtype_bytes  (fwd 单向)
```

`EP_size=8, N_local=4096, K=2, H=4096, bf16` → 一次 all-to-all 每卡 128MB，一层两次 = 256MB × 2（fwd+bwd）。

### 8.4 EP + DP 组合

如果 `EP_size < world_size`，剩下的维度做 DP：

```
world_size = 64
EP_size    = 8      ← 每 8 卡放一份完整专家集
DP_size    = 8      ← 8 份数据并行副本

placement:
   EP group:  [rank 0, 1, 2, ..., 7]   持有 experts 0..7 (每卡 1 个)
              [rank 8, 9, ...,     15] 持有另一份 experts 0..7
              ...
   DP group:  [rank 0, 8, 16, ..., 56] 数据并行，梯度 AllReduce
```

对于 **non-expert 参数**（attention、gate、norm）：走 DP 全同步。
对于 **expert 参数**：只在同一 EP 组的 "同 rank" 之间做 AllReduce（=DP over EP）。

### 8.5 EP + TP

TP 切专家内的 GEMM（`H → I → H`）。要小心 all-to-all 和 TP AllReduce 的顺序。

Megatron-LM MoE 实现给出的正确顺序：
```
router (replicated across TP)
  → gate output (replicated)
  → all_to_all across EP
  → expert GEMM1 (col-parallel, split intermediate_size)
  → activation
  → expert GEMM2 (row-parallel)
  → AllReduce across TP (in expert)
  → all_to_all back across EP
  → combine
```

### 8.6 通信/计算 overlap

MoE 训练的关键工程优化：

1. **All-to-all vs expert compute overlap**
   - 用 NCCL async + CUDA stream
   - `all_to_all_1` 完成后立刻开始本地 expert；`all_to_all_2` 与下一层的 router 重叠

2. **Chunked all-to-all**
   - 把 all-to-all 拆成 K 个 chunk，每个 chunk 处理完就 fire 下一步
   - DeepSpeed-MoE、Tutel 都有实现

3. **Fine-grained overlap（DeepSeek-V3）**
   - 前向：dispatch → attn → mlp → combine 的四阶段 pipeline
   - 反向：反向拆两半，插入 all-to-all 的 grad

### 8.7 主要框架对比

| 框架 | EP 实现 | Grouped GEMM | 特色 |
|---|---|---|---|
| **DeepSpeed-MoE** | ✔ | ✔ | 早期成熟，与 ZeRO 集成 |
| **Megatron-LM MoE** | ✔ | ✔ | 与 TP/PP/SP 完整集成 |
| **Tutel** (微软) | ✔ | ✔ | 快速 all-to-all、动态 capacity |
| **Megablocks** | ✔ | 用 block-sparse 替代 | 无 padding、无 drop |
| **FastMoE / FasterMoE** | ✔ | ✔ | 早期开源实现 |

### 8.8 分布式训练配置示例（Mixtral 8×7B）

```
Total params:   47B (8 experts × 5.6B + shared)
Active params:  13B (2 experts + shared)
Setup:
    64× H100
    TP = 2      (attention & expert GEMM)
    EP = 8      (8 专家分布在 8 卡)
    DP = 4      (4 份数据并行)
    PP = 1      (单 pipeline)
    → 2 * 8 * 4 * 1 = 64 ✔
```

### 8.9 分布式调优清单

```
□ 用 grouped GEMM 打包本地专家计算
□ NCCL: NCCL_MIN_NCHANNELS / NCCL_NTHREADS 调 all-to-all 带宽
□ 检查 all-to-all 是否 overlap 到 expert compute
□ capacity_factor 或 Megablocks 消 padding
□ Router logits 在 all-to-all 之前保留一份（用于 aux loss、debug）
□ 每 N 步 log 每个专家的 load（f_i 分布）
□ bf16 参数 + fp32 optimizer + fp32 softmax
□ 打开 async grad allreduce（DP 维度）
□ 大 EP 时考虑 hierarchical all-to-all（intra-node NVLink + inter-node IB）
```

---

## 第 9 章：数值稳定性、精度与常见陷阱

### 9.1 Softmax 精度

**必须用 fp32** 计算 router 的 softmax（本仓库示例已经这样做）：

```python
gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)
```

原因：`E` 较大时 `logsumexp` 容易在 fp16/bf16 下溢或饱和，导致所有 prob 变成同一个值，router 死掉。

### 9.2 Renormalization vs. Softmax over top-K

两种实现在数学上不完全等价：

```
# 方案 A (本仓库、GShard)
p = softmax(logits)           # over all E
top_p, top_i = topk(p, K)
w = top_p / top_p.sum()

# 方案 B (Mixtral)
top_l, top_i = topk(logits, K)
w = softmax(top_l)            # over top K
```

方案 B 的 w 与 "被丢掉的专家的 logits" 无关（gradient 不流），方案 A 会有一点耦合。实践中差别很小，Mixtral 官方 checkpoint 用 B。

### 9.3 top-k 里出现相同专家？

**不会**。`torch.topk` 返回的 index 天然 unique（因为是从 `(N, E)` 每行选 K 个最大值的下标）。

### 9.4 index_add_ 的确定性

`torch.index_add_` 在 GPU 上**默认非确定性**（多个 index 相同的写入顺序未定义）。在本仓库实现中，每个 expert 循环内 `token_ids` unique，所以 add 顺序确定；但如果你用 packed 实现，要小心：
```python
torch.use_deterministic_algorithms(True)  # 需要时开启
```

### 9.5 空专家 (M_e = 0)

某些 batch 里某专家一个 token 都没收到。必须处理：
```python
if token_ids.numel() == 0:
    continue
```
否则在 grouped GEMM 里 group_size=0 也需要 kernel 正确处理。

反向传播时：**空专家的权重梯度是 0**（正常），但要防止把它误标为"loss 出错"。

### 9.6 Backward 只流过 top-k

Backward 时，只有被选中的专家会收到梯度；被 topk **淘汰掉的 logits 收不到直接梯度**（只通过 aux loss / z-loss / re-norm 间接收到）。这是稀疏 MoE 的固有属性，不算 bug。

**推论**：早期训练前几百步 router 可能只固定选某几个专家，其他专家几乎不更新——所以 warmup + aux loss 系数很关键。

### 9.7 保存/加载 checkpoint

分布式 MoE checkpoint 的坑：
- Expert 权重按 EP 切开存在不同 rank，加载时要能重新聚合/切分
- 恢复训练时 EP world size 可能变了：需要重新 shard
- 推荐用 `DTensor` 或 Megatron 的 dist ckpt 格式

### 9.8 推理场景差异

推理时（尤其是自回归 decoding，batch × seq 很小），MoE 的挑战完全不同：
- Token 极少 → grouped GEMM 每 group 只有 1–2 行 → 严重内存 bound
- 需要 expert 权重能快速加载（fp8/int4 量化）
- vLLM/SGLang 用 `fused_moe_kernel` + weight-only quant

本文档聚焦训练，推理细节参见 vLLM / SGLang 源码。

---

## 附录 A：符号表

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `S` | sequence length |
| `N = B × S` | token 总数（展平后）|
| `H` | hidden size |
| `I` | intermediate (FFN 内) size |
| `E` | 专家总数 |
| `K` | top-k |
| `M_e` | 路由到专家 e 的 token 数 |
| `C` | 每个专家的容量 (capacity) |
| `EP` | expert parallel size |
| `TP` | tensor parallel size |
| `DP` | data parallel size |

## 附录 B：延伸阅读

- **Shazeer et al. 2017** ["Outrageously Large Neural Networks"](https://arxiv.org/abs/1701.06538) — MoE 开山
- **GShard (Lepikhin et al. 2020)** [arxiv 2006.16668](https://arxiv.org/abs/2006.16668)
- **Switch Transformer (Fedus et al. 2021)** [arxiv 2101.03961](https://arxiv.org/abs/2101.03961)
- **ST-MoE (Zoph et al. 2022)** — router z-loss、stable training
- **Mixtral of Experts (Jiang et al. 2024)** [arxiv 2401.04088](https://arxiv.org/abs/2401.04088)
- **DeepSeek-V3 (2024)** [github/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) — auxiliary-loss-free 均衡、fine-grained overlap
- **Megablocks (Gale et al. 2022)** [arxiv 2211.15841](https://arxiv.org/abs/2211.15841) — block-sparse MoE
- **Tutel (Hwang et al. 2022)** [arxiv 2206.03382](https://arxiv.org/abs/2206.03382)
- **HuggingFace MoE blog** — https://huggingface.co/blog/moe

---

**本仓库入口**：
- 最小 PyTorch 实现：`python/pytorch/test_moe.py`
- 运行 smoke test：`python python/pytorch/test_moe.py`
