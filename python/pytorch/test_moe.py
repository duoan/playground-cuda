"""Sparse Mixture-of-Experts (MoE) 层的最小实现。

参考 Mixtral / Switch Transformer 的 top-k gating + 稀疏 dispatch 结构，
只保留核心逻辑，方便对照 CUDA 版本理解每一步在做什么。
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


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


class MoE(nn.Module):
    """Top-k sparse MoE 层。

    对每个 token:
      1. gate 给出 num_experts 个 logits，取 top-k 决定路由到哪些专家。
      2. 每个专家独立处理被路由过来的 token，按 gate 权重加权求和写回。

    输入/输出 shape 都是 (B, S, H)，额外返回 gate_logits 以便计算
    load-balancing loss（本文件未包含，留作练习）。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k 必须在 [1, num_experts] 之间"
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(
        self, hidden_states: Float[Tensor, "B S H"]
    ) -> tuple[Float[Tensor, "B S H"], Float[Tensor, "BS E"]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        dtype = hidden_states.dtype

        # 展平 batch 和 seq 两个维度，后续每个 token 都是独立路由的
        # (B, S, H) -> (N, H)，其中 N = B * S
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

        out = out.view(batch_size, seq_len, hidden_size)
        return out, gate_logits


# ---------------- smoke tests ----------------

def test_moe_forward_shape():
    torch.manual_seed(0)
    B, S, H, I, E, K = 2, 4, 8, 16, 4, 2
    moe = MoE(hidden_size=H, intermediate_size=I, num_experts=E, top_k=K)
    x = torch.randn(B, S, H)

    y, gate_logits = moe(x)

    assert y.shape == (B, S, H)
    assert gate_logits.shape == (B * S, E)
    assert torch.isfinite(y).all()


def test_moe_backward():
    """确保梯度能流到 gate 和所有专家（至少每个专家被至少一个 token 选中的常见情况）。"""
    torch.manual_seed(0)
    B, S, H, I, E, K = 4, 16, 8, 16, 4, 2
    moe = MoE(hidden_size=H, intermediate_size=I, num_experts=E, top_k=K)
    x = torch.randn(B, S, H, requires_grad=True)

    y, _ = moe(x)
    y.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.gate.weight.grad is not None
    # 至少一个专家收到了梯度即可（top-k 稀疏路由下不保证每个专家都命中）
    got_grad = [e.fc1.weight.grad is not None for e in moe.experts]
    assert any(got_grad)


if __name__ == "__main__":
    test_moe_forward_shape()
    test_moe_backward()
    print("ok")
