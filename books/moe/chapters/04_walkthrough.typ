#import "../template.typ": *

= 逐行 Code Walkthrough (test_moe.py)

这一章配合本仓库 `python/pytorch/test_moe.py` 逐段拆解——把前两章的抽象概念落到具体的 PyTorch 代码上。所有 tensor shape 都带具体数字，方便对着代码 debug。

跑法：

```bash
python python/pytorch/test_moe.py
# ok
```

代码 100 行左右，运行样例 $B=2, S=4, H=8, I=16, E=4, K=2$。

== 全文件结构

```python
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

class Expert(nn.Module): ...       # 单个专家 FFN
class MoE(nn.Module):    ...       # MoE 层

def test_moe_forward_shape(): ...  # smoke test 1
def test_moe_backward():     ...   # smoke test 2

if __name__ == "__main__":
    test_moe_forward_shape()
    test_moe_backward()
    print("ok")
```

三部分：Expert (单专家 FFN)、MoE 主体、两个 smoke test。

== Part 1: Expert 定义

```python
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

*说明*：一个专家就是一个 $(H → I → H)$ 的普通 FFN。

样例参数量：$H=8, I=16$，每个专家 $H times I + I times H = 256$ params，$E=4$ 时总 expert 参数 $= 1024$。

*生产模型的差异*：Mixtral/Llama 用 SwiGLU：

```python
class MixtralExpert(nn.Module):
    def __init__(self, H, I):
        self.w_gate = nn.Linear(H, I, bias=False)
        self.w_up   = nn.Linear(H, I, bias=False)
        self.w_down = nn.Linear(I, H, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

SwiGLU 比标准 MLP 多 50% 参数量 (3 个 Linear 而不是 2 个)，但训练/推理都实测更好。教学实现里为了简洁，本书用 ReLU 版。

== Part 2: MoE 主体——init

```python
class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size)
             for _ in range(num_experts)]
        )
```

关键点：

- `self.gate`：router 的唯一可学习参数，$(H → E)$。
- `self.experts`：用 `nn.ModuleList` 是为了让 `.parameters()` / `.to(device)` 递归看到子模块。用普通 `list` 会导致参数不注册。

样例配置 $H=8, I=16, E=4, K=2$：

- Router: 8 × 4 = 32 params
- Experts: 4 × 256 = 1024 params
- Total: 1056 params

真实 Mixtral-8×7B：

- Router: 4096 × 8 = 32,768 params × 32 layers = 1M params (可忽略)
- Experts per layer: 8 × (4096 × 14336 × 3) $approx$ 1.4B; 32 layers = 45B (主要参数)
- Attention + LN + embed: 剩下的 2B

== Part 3: MoE forward——展开和 router

```python
def forward(
    self, hidden_states: Float[Tensor, "B S H"]
) -> tuple[Float[Tensor, "B S H"], Float[Tensor, "BS E"]]:
    batch_size, seq_len, hidden_size = hidden_states.shape
    dtype = hidden_states.dtype

    # (B, S, H) -> (N, H)，N = B * S
    hidden_states = hidden_states.view(-1, hidden_size)
    num_tokens = hidden_states.shape[0]
```

样例：$(2, 4, 8) → (8, 8)$。这一步等价于*把 batch 和 seq 视为独立 token 序列*——因为 MoE 的路由是逐 token 独立的，两个维度合并不影响语义。

*易错点*：调用 `.view(-1, H)` 要求 tensor 是 contiguous。如果上游有 transpose，需要 `.reshape(-1, H)` 或先 `.contiguous()`。生产实现里 hidden_states 从 attention 出来通常是 contiguous 的，不用担心。

```python
    # -------- 1. Router --------
    gate_logits = self.gate(hidden_states)  # (N, E)
    gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)

    expert_weights, expert_indices = torch.topk(
        gate_probs, k=self.top_k, dim=-1
    )
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
    expert_weights = expert_weights.to(dtype)
```

Shape 追踪 ($N=8, E=4, K=2$)：

#align(center)[
  #shape-pipeline(stages: (
    ("self.gate(hidden)", "gate_logits: (8, 4)", "Linear(H=8, E=4)"),
    ("softmax(fp32).to(bf16)", "gate_probs: (8, 4)", "每行 sum=1"),
    ("torch.topk(K=2)", "weights, indices: (8, 2), (8, 2)", "expert_weights 未归一"),
    ("除以行和", "expert_weights: (8, 2)", "renorm sum=1"),
  ))
]

三个 subtle 点：

*(1) `dtype=torch.float32` 参数*

`F.softmax(..., dtype=torch.float32)` 会先 upcast `gate_logits` 到 fp32、再算 softmax、返回 fp32。然后 `.to(dtype)` 转回原 dtype (通常 bf16)。第 3 章"为什么 softmax 要用 fp32"解释过为什么必要。

*(2) `torch.topk` 的返回顺序*

返回的 K 个专家是*按 gate_probs 值降序*排的。所以 `expert_indices[n, 0]` 是 top-1，`expert_indices[n, 1]` 是 top-2。后续 dispatch 里 `k_ids = 0` 对应 top-1、`k_ids = 1` 对应 top-2。

*(3) renormalize 前后的语义*

在第 3 章解释过：renorm 前保留"router 置信度"信息，renorm 后是 top-K 内的相对权重。本书 demo 用 renorm，Mixtral 用 `softmax(topk(logits))`，两者数学等价性我们已经推过。

== Part 4: MoE forward——dispatch 循环

这是全文件最关键的 20 行：

```python
    # -------- 2. Dispatch --------
    out = torch.zeros_like(hidden_states)  # (N, H)

    for expert_idx in range(self.num_experts):
        # 找到所有把 expert_idx 选为 top-k 之一的 (token, k) 位置
        token_ids, k_ids = torch.where(expert_indices == expert_idx)
        if token_ids.numel() == 0:
            continue

        expert_input = hidden_states[token_ids]  # (M, H)
        expert_output = self.experts[expert_idx](expert_input)  # (M, H)

        weights = expert_weights[token_ids, k_ids].unsqueeze(-1)  # (M, 1)
        out.index_add_(0, token_ids, (expert_output * weights).to(dtype))
```

用 `expert_idx = 1` 一步拆解 (延续第 4 章的例子)：

*Line 1*: `torch.where(expert_indices == 1)`

- 输入: `expert_indices` shape $(8, 2)$
- 中间: `(expert_indices == 1)` 是 $(8, 2)$ 的 bool tensor
- 输出: 两个 $(M,)$ 的 int64 tensor，$M$ 是命中的 (token, k) 位置数

对样例数据 (随机，假设有 5 个 hit)：
```
token_ids = tensor([0, 2, 3, 5, 6])
k_ids     = tensor([0, 1, 0, 0, 1])
```

*Line 2*: `if token_ids.numel() == 0: continue`

处理 $M_e = 0$ 的边界情况——某专家这个 batch 一个 token 都没收到。生产实现要小心：如果这里不 continue，`hidden_states[[]]` 是 empty tensor，`Linear` 对 empty 输入是合法的，但 `.to(cuda)` 上有时会出错。范式 A 就地 continue 最安全。

*Line 3*: `hidden_states[token_ids]`

Fancy indexing，$(N, H) → (M, H)$，这是一次 gather kernel。生成的 tensor 是新分配的、connected 到 autograd 图。

*Line 4*: `self.experts[expert_idx](expert_input)`

单专家 FFN，$(M, H) → (M, H)$。内部两次 Linear + 一次 ReLU，共 3 个 kernel launch（本书教学实现，未 fuse）。

*Line 5*: `expert_weights[token_ids, k_ids].unsqueeze(-1)`

这是最容易看错的一句。

- `expert_weights` shape $(N, K) = (8, 2)$
- `token_ids` shape $(M,) = (5,)$
- `k_ids` shape $(M,) = (5,)$

`expert_weights[token_ids, k_ids]` 是 *2D fancy indexing*——每个 $(t, k)$ pair 取一个元素，返回 shape $(M,)$：

```
weights[i] = expert_weights[token_ids[i], k_ids[i]]

= expert_weights[[0,2,3,5,6], [0,1,0,0,1]]  # 逐元素配对
= [expert_weights[0, 0],   # t0 把 E1 选为 top-1 的权重
   expert_weights[2, 1],   # t2 把 E1 选为 top-2 的权重
   expert_weights[3, 0],   # ...
   expert_weights[5, 0],
   expert_weights[6, 1]]

  → shape (5,)
```

然后 `.unsqueeze(-1)` 变 $(5, 1)$，方便和 `(5, H)` broadcast 相乘。

#insight[
  `expert_weights[token_ids, k_ids]` 用了 numpy/pytorch 的*同轴 broadcasting fancy indexing*——两个 index tensor shape 必须一致，结果 shape 也一致。这不是 `expert_weights[token_ids][:, k_ids]`（后者是先取行，再对每行的所有列取 k_ids，结果是 (M, M)）。生产代码 review 常见 bug。
]

*Line 6*: `out.index_add_(0, token_ids, ...)`

Scatter-add：

```
for i in range(M):
    out[token_ids[i]] += (expert_output[i] * weights[i]).to(dtype)
```

`0` 是 dim (沿第 0 维即 token 维累加)。`token_ids` 在 *单次* `expert_idx` 循环内 unique（第 4 章"一个隐藏的正确性坑"解释过），所以这里等价于 gather-write，没有 race。但*跨* `expert_idx` 循环，`out[t]` 会被 K 次累加——这正是 top-K 加权求和。

*Line 7*: `.to(dtype)`

确保输出 dtype 一致。中间 `expert_output * weights` 可能因为 weights 是 fp32 (renorm 阶段) 而被 upcast——这里明确 downcast 回 bf16。

== Part 5: 收尾

```python
    out = out.view(batch_size, seq_len, hidden_size)
    return out, gate_logits
```

Reshape 回 $(B, S, H)$。返回 `gate_logits` 是为了外部可以用它算 aux loss（第 6 章）。生产实现通常返回 dict，包含 `router_probs`, `expert_indices`, `f_i` (per-expert token count) 等 debug 信息。

== 两个 smoke test

```python
def test_moe_forward_shape():
    torch.manual_seed(0)
    B, S, H, I, E, K = 2, 4, 8, 16, 4, 2
    moe = MoE(hidden_size=H, intermediate_size=I, num_experts=E, top_k=K)
    x = torch.randn(B, S, H)

    y, gate_logits = moe(x)

    assert y.shape == (B, S, H)
    assert gate_logits.shape == (B * S, E)
    assert torch.isfinite(y).all()
```

用 `torch.manual_seed(0)` 固定 router 初始化和输入，保证 test 可重复。检查 (1) 输出 shape 保持；(2) `gate_logits` 是 flatten 后的 $(N, E)$；(3) 输出全有限（没有 NaN/Inf）。

```python
def test_moe_backward():
    torch.manual_seed(0)
    B, S, H, I, E, K = 4, 16, 8, 16, 4, 2  # 更大 batch, 每个 expert 概率被 hit
    moe = MoE(hidden_size=H, intermediate_size=I, num_experts=E, top_k=K)
    x = torch.randn(B, S, H, requires_grad=True)

    y, _ = moe(x)
    y.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.gate.weight.grad is not None
    got_grad = [e.fc1.weight.grad is not None for e in moe.experts]
    assert any(got_grad)
```

关键设计：

*(1) 更大的 batch*：$B S = 64$，避免每个专家概率 $= 0$。$K = 2$，$E = 4$，理想每专家 $M_e = 32$——大概率*所有* expert 都被至少一个 token 选中。

*(2) `assert any(got_grad)`*：只要求*至少一个* expert 有梯度——第 3 章讲过，topk 反向只流 K 个专家，某个专家在 batch 里未被选中就没梯度。这个 test 不能 assert `all`。

== 完整 tensor 生命周期图

把整个 forward 汇成一张图：

#align(center)[
  #shape-pipeline(stages: (
    ("input", "(B, S, H) = (2, 4, 8)", ""),
    ("view", "(N, H) = (8, 8)", "flatten"),
    ("Linear (gate)", "gate_logits: (8, 4)", ""),
    ("softmax fp32", "gate_probs: (8, 4)", "每行 sum=1"),
    ("topk", "weights, indices: (8, 2), (8, 2)", "top-K experts"),
    ("renorm", "weights: (8, 2)", "sum=1"),
    ("dispatch loop (×E)", "out: (8, 8)", "scatter-add"),
    ("view", "output: (2, 4, 8)", ""),
  ))
]

== 常见改错练习

写完 MoE 你会想改的几个方向，各自的坑：

*(1) 想加 SwiGLU expert*

改 `Expert.forward`：

```python
def forward(self, x):
    return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

`init` 里加 `w_gate, w_up, w_down` 三个 Linear。注意*参数量增加 50%*。

*(2) 想改成 Mixtral 风格 router*

替换：

```python
# 原本
gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)
expert_weights, expert_indices = torch.topk(gate_probs, k=K, dim=-1)
expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

# Mixtral 风格
top_logits, expert_indices = torch.topk(gate_logits, k=K, dim=-1)
expert_weights = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(dtype)
```

结果几乎等价，但反向传播动力学不同（第 3 章"Backward"一节）。

*(3) 想加 aux loss*

返回 `gate_probs` 而不是 `gate_logits`，然后在训练脚本里：

```python
def aux_loss(gate_probs, expert_indices, num_experts):
    N, E = gate_probs.shape
    # f_i: 每个 expert 被 top-1 选中的比例
    f = torch.zeros(E, device=gate_probs.device)
    top1 = expert_indices[:, 0]
    f.scatter_add_(0, top1, torch.ones_like(top1, dtype=torch.float))
    f = f / N
    # P_i: gate_probs 的平均
    P = gate_probs.mean(dim=0)
    return E * (f * P).sum()
```

详情见第 6 章。

*(4) 想加 capacity + drop*

需要在 dispatch 前 truncate：

```python
capacity = int(num_tokens * K / E * capacity_factor)
# 每个 expert 保留前 capacity 个（按位置或按 gate prob）
```

范式 A 里做 drop 反直觉——先在 `expert_indices` 里 mark drop 位置 (e.g. 设为 -1)，`torch.where` 前跳过。范式 B 里 drop 天然（超 capacity 就截断 packed）。

== 面试考点

#interview[
  *Q1*: 为什么用 `view(-1, H)` 而不是 `reshape(-1, H)`？

  A: `view` 要求 contiguous 但不做拷贝——快；`reshape` 允许非 contiguous，会隐式拷贝。上游 attention 输出通常 contiguous，用 view 更快。如果无法确定，用 `reshape` 或先 `.contiguous().view(-1, H)`。
]

#interview[
  *Q2*: `expert_weights[token_ids, k_ids]` 和 `expert_weights[token_ids][:, k_ids]` 有什么区别？

  A: 前者是 *2D fancy indexing*，两个 index tensor 同步 broadcast，输出 shape $(M,)$；后者是先取行再取列，输出 shape $(M, M)$。工程 bug 高发点。
]

#interview[
  *Q3*: `index_add_` 里如果 `token_ids` 有重复，PyTorch 保证顺序吗？

  A: GPU 上*默认不保证*——多个原子加的顺序取决于硬件调度。要 reproducible 训练用 `torch.use_deterministic_algorithms(True)`，代价约 5-15% 慢。本书 demo 里 *单次 expert 循环内* `token_ids` 保证 unique，所以不涉及；跨循环的 K 次累加受此影响。
]

#interview[
  *Q4*: 为什么 `test_moe_backward` 只 assert `any` 而不是 `all`？

  A: topk 反向只把梯度流给被选中的 K 个专家；小 batch 下某个专家可能*一个 token 都没被选中*，参数梯度就是 None。`all` 会假阳性 fail；`any` 只验证 backward 至少走通一条路径。生产 unit test 里通常用更大的 $N$ 或 mock router 强制均匀。
]

#interview[
  *Q5*: 如果 `expert_indices` 里同一行有两个相同的 expert id（比如 `[3, 3]`），会怎样？

  A: `torch.topk` 保证返回值*不重复*——它是取索引，不同索引对应不同位置。要出现重复必须 gate_probs 完全相等 (tie)——tie 时 torch 的 tie-breaking 是 stable (按位置序返回)，结果 index 不同，*仍然 unique*。因此 `test_moe.py` 假设成立。
]

#interview[
  *Q6*: 有没有办法让 dispatch 循环用 `torch.compile` 加速？

  A: 部分能——router 部分 (Linear + softmax + topk) 编译收益明显。但循环内 `torch.where` + 动态形状 gather 会导致 graph break。要真正吃到编译红利，需要*换成范式 B*（静态 packed shape）。所以生产实现的正确姿势是先换范式，再上编译。
]
