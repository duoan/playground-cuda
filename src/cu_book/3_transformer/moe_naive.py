"""Naive Mixture-of-Experts for teaching / whiteboard interviews.

Reference implementation. Prioritizes clarity, not performance:
  - Straightforward for-loop over experts, no permute / grouped GEMM tricks.
  - Simple 2-layer MLP experts (Linear -> ReLU -> Linear).
  - Auxiliary load-balancing loss included, in the classic Switch-Transformer
    form so the "why does aux loss actually train the router" story works.

Shape symbols used throughout:
  T = number of tokens (batch dim after flattening)
  D = hidden_dim
  H = intermediate_dim
  E = num_experts
  K = num_experts_per_token   (K << E; typically 1 or 2)

Everything downstream of the router works at the token level, so the caller
should already have collapsed batch/seq into a single T dim.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """A single expert: a plain 2-layer MLP."""

    def __init__(
        self, hidden_dim: int, intermediate_dim: int, dtype=None, device=None
    ) -> None:
        factory_params = {"dtype": dtype, "device": device}
        super().__init__(**factory_params)
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=False, **factory_params)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False, **factory_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class NaiveMoE(nn.Module):
    """Top-K Mixture-of-Experts, naive reference version.

    Forward pass, step by step:
      1. Router scores each token against E experts.
      2. Take softmax to get a probability distribution over experts per token.
      3. Pick the top-K experts per token; renormalize their K weights to sum to 1.
      4. For each expert i, gather the tokens routed to it, run them through
         that expert, and scatter the weighted output back.
      5. Compute an auxiliary load-balancing loss so no expert gets starved.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        super().__init__()
        assert 1 <= num_experts_per_token <= num_experts
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Router = a Linear producing one logit per expert.
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # E independent experts. Simple, no weight sharing.
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, D]  input tokens (flatten batch/seq beforehand)

        Returns:
            y:        [T, D]  MoE output
            aux_loss: scalar   load-balancing loss (add to main loss during training)
        """
        T, D = x.shape
        E, K = self.num_experts, self.num_experts_per_token

        # --- 1. Route ------------------------------------------------------
        router_logits = self.router(x)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [T, E]

        # Pick K best experts per token by logit. `topk` returns distinct
        # indices, so a token never picks the same expert twice.
        topk_probs, topk_indices = torch.topk(router_probs, K, dim=-1)  # both [T, K]

        # Renormalize the K weights so they sum to 1 per token.
        # (We discarded E-K experts, so `topk_probs` no longer sums to 1.)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [T, K]

        # --- 2. Dispatch: run each expert on its tokens --------------------
        y = torch.zeros_like(x)  # [T, D]

        for i, expert in enumerate(self.experts):
            # Which (token, slot) positions picked expert i?
            slot_mask = topk_indices == i  # [T, K] bool
            # Which tokens have any slot pointing at expert i?
            # (At most one slot per token, since topk indices are distinct.)
            token_mask = slot_mask.any(dim=-1)  # [T] bool
            if not token_mask.any():
                continue  # no tokens for this expert this step; skip

            # For each such token, the gate weight it assigned to expert i.
            weight = topk_weights[slot_mask].unsqueeze(-1)  # [N_i, 1]

            # Run expert i on its tokens; scatter weighted output back.
            #   x[token_mask]           -> [N_i, D]
            #   expert(x[token_mask])   -> [N_i, D]
            y[token_mask] += weight * expert(x[token_mask])

        # --- 3. Auxiliary load-balancing loss (Switch-Transformer form) ---
        # Encourage each expert to see ~1/E of the tokens.
        #
        #   f_i = fraction of tokens that picked expert i as any of their K slots
        #         (hard count, non-differentiable, just an observable).
        #   P_i = mean gate probability assigned to expert i across all tokens
        #         (soft, differentiable, this is what the gradient flows through).
        #
        # aux = E * sum_i f_i * P_i
        #
        # Intuition: if the router concentrates all mass on one expert, both
        # f and P spike on that expert -> aux blows up.
        # If routing is uniform, f_i = P_i = 1/E, so aux = E * E * (1/E)^2 = 1.
        # The gradient flows only through P_i (f_i is discrete), but P_i is
        # weighted by f_i, so hard imbalance amplifies the soft penalty.
        with torch.no_grad():
            # one_hot slots -> sum over (token, slot) -> normalize by total slots
            slot_one_hot = F.one_hot(topk_indices, num_classes=E).float()  # [T, K, E]
            f = slot_one_hot.sum(dim=(0, 1)) / (T * K)  # [E]
        P = router_probs.mean(dim=0)  # [E]
        aux_loss = E * (f * P).sum()

        return y, aux_loss


if __name__ == "__main__":
    """Sanity check: forward runs, shapes match, gradient reaches the router."""
    torch.manual_seed(0)
    T, D, H, E, K = 8, 16, 32, 4, 2
    moe = NaiveMoE(
        hidden_dim=D, intermediate_dim=H, num_experts=E, num_experts_per_token=K
    )
    x = torch.randn(T, D, requires_grad=True)

    y, aux = moe(x)
    print(f"y shape:   {tuple(y.shape)}  (expect ({T}, {D}))")
    print(f"aux loss:  {aux.item():.4f}  (uniform routing gives ~1.0)")

    loss = y.sum() + aux
    loss.backward()
    print(
        f"router grad norm: {moe.router.weight.grad.norm().item():.4f}  (>0 means aux loss trains the router)"
    )
