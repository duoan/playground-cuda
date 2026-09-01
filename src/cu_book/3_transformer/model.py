import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn


def sdpa(
    q: Float[Tensor, "b n s_q h"],
    k: Float[Tensor, "b n s_kv h"],
    v: Float[Tensor, "b n s_kv h"],
) -> Float[Tensor, "b s_q h"]:
    s_q, s_kv = q.size(-2), k.size(-2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
        k.size(-1)
    )  # (b s s_q, s_kv)

    # mask
    scores = scores.masked_fill(
        mask=torch.triu(torch.ones((s_q, s_kv), dtype=torch.bool), diagonal=1),
        value=float("-inf"),
    )

    # softmax
    probs = torch.softmax(scores, dim=-1)  # (b, n, s_q, s_kv)

    atten = torch.matmul(
        probs, v
    )  # (b, n, s_q, s_kv) @ (b, n, s_kv, d) = (b, n, s_q, d)

    atten = rearrange(atten, "b n s_q d -> b s_q (n d)")

    return atten


class MultiHeadAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, n_heads: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv_proj = nn.Linear(
            in_features=hidden_dim,
            out_features=3 * hidden_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=False,
        )

    def forward(self, x: Float[Tensor, "b t c"]) -> Float[Tensor, "b t c"]:
        # 1. qkv projection and split heads
        q, k, v = rearrange(
            x,
            "b t (three h d) -> three b h t d",
            three=3,
            h=self.n_heads,
            d=self.head_dim,
        )

        # 2. scaled dot product attention
        attns = sdpa(q, k, v)

        # 3. out projection
        return self.out_proj(attns)


class KVCache:
    def __init__(self) -> None:
        self.k = None
        self.v = None

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.k is None and self.v is None:
            self.k, self.v = k, v
            return k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)  # type: ignore
            self.v = torch.cat([self.v, v], dim=2)  # type: ignore
            return self.k, self.v


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_seq_len: int = 4096,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )
        self.max_seq_len = max_seq_len
        mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        hidden_states: Float[Tensor, "B S D"],
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, "B S D"]:
        q_seq_len = hidden_states.size(1)
        assert q_seq_len < self.max_seq_len, (
            f"Only support sequence length < {self.max_seq_len} "
        )

        # Q, K, V projection, split heads and reshape
        q_states = rearrange(
            self.q_proj(hidden_states), "b s (n h) -> b n s h", h=self.head_dim
        )
        k_states = rearrange(
            self.k_proj(hidden_states), "b s (n h) -> b n s h", h=self.head_dim
        )
        v_states = rearrange(
            self.v_proj(hidden_states), "b s (n h) -> b n s h", h=self.head_dim
        )

        if kv_cache is not None:
            k_states, v_states = kv_cache.update(k_states, v_states)

        kv_seq_len = k_states.size(2)
        assert kv_seq_len <= self.max_seq_len, (
            f"Sequence length {kv_seq_len} exceeds max_seq_len {self.max_seq_len}"
        )

        # 2. match KV heads to Q heads using repeat_interleave
        # from (B, num_key_value_heads, S, head_dim) -> To: (B, num_attention_heads, S, head_dim)
        k_states = torch.repeat_interleave(
            k_states, repeats=self.num_key_value_groups, dim=1
        )
        v_states = torch.repeat_interleave(
            v_states, repeats=self.num_key_value_groups, dim=1
        )

        # 3. scaled dot product attention
        attn_weights = (
            einsum(q_states, k_states, "b n s_q h, b n s_kv h -> b n s_q s_kv")
            * self.scaling
        )
        mask = self.causal_mask[kv_seq_len - q_seq_len : kv_seq_len, :kv_seq_len].to(  # type: ignore
            hidden_states.device
        )
        attn_weights = attn_weights.masked_fill(mask=mask, value=float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q_states.dtype
        )
        attn_out = einsum(
            attn_weights,
            v_states,
            "b n s_q s_kv, b n s_kv h -> b s_q (n h)",
        )

        # 4. out projection
        return self.o_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward layer.

    Idea:
      Replace a single big FFN by `n_experts` smaller FFNs. For each token,
      route it to the top-K experts (K << n_experts), and combine only those
      experts' outputs weighted by the gate probabilities. Total parameters
      scale with n_experts, but per-token FLOPs stay ~ K * (one expert).

    Shape symbols used below:
      B = batch size
      S = sequence length
      D = d_model (feature dim)
      E = n_experts
      K = n_experts_top_k        # experts activated per token

    Auxiliary load-balancing loss (returned alongside y):
      Without a balancing signal the gate tends to collapse: it may route
      every token to a single expert, leaving the others cold. We add a small
      penalty that pushes both the mean gate probability and the empirical
      routing fraction toward the uniform 1/E distribution.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        n_experts_top_k: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_experts = n_experts
        self.n_experts_top_k = n_experts_top_k

        # Gate (a.k.a. router): produces one logit per expert for every token.
        # This is the only learnable part of the routing decision.
        self.gate = nn.Linear(d_model, n_experts)

        # Experts: each is a full FFN with the same architecture as the dense
        # case. They do NOT share weights.
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff) for _ in range(n_experts)]
        )

    def forward(self, x: Float[Tensor, "B S D"]):
        B, S, _ = x.shape
        num_tokens = B * S  # total tokens across the batch

        # ---------------- 1. Gating ----------------
        # One logit per (token, expert) pair.
        gate_logits = self.gate(x)  # [B, S, E]

        # Full softmax over experts. Used ONLY for the aux loss below.
        # Kept differentiable end-to-end so aux loss actually trains the gate.
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, S, E]

        # Pick the K best experts per token by raw logit.
        #   topk_logits:  the K largest logits per token
        #   topk_indices: which experts those K logits belong to, in [0, E)
        # Note: torch.topk returns distinct indices, so a token never picks
        # the same expert twice -- this is what makes the mask trick below
        # safe (see routing loop).
        topk_logits, topk_indices = torch.topk(
            gate_logits, k=self.n_experts_top_k, dim=-1
        )  # both [B, S, K]

        # Re-normalize among the K chosen experts, so the K weights sum to 1
        # per token. This is standard MoE (Switch / GShard / Mixtral style):
        # we don't reuse the full-E softmax weights, only the K selected ones.
        topk_weights = F.softmax(topk_logits, dim=-1)  # [B, S, K]

        # ---------------- 2. Dispatch to experts ----------------
        # Output accumulator, same shape as input.
        y = torch.zeros_like(x)  # [B, S, D]

        # Loop over experts. For each expert i we gather all tokens routed to
        # it, run them through, and scatter the weighted result back into y.
        # This is O(E) Python-level launches -- easy to read, not the fastest.
        # (Grouped-matmul / megablocks style implementations avoid this loop.)
        for i, expert in enumerate(self.experts):
            # Which (token, slot) positions selected expert i?
            expert_slot_mask = topk_indices == i  # [B, S, K] bool

            # Which tokens have any slot pointing at expert i?
            # Since topk_indices has no duplicates per token, at most one of
            # the K slots per token is True in expert_slot_mask, so
            #   token_mask.sum() == expert_slot_mask.sum()
            # -- this equality is what makes the += broadcast below shape-correct.
            token_mask = expert_slot_mask.any(dim=-1)  # [B, S] bool

            # No tokens picked this expert this step. Skip it (also avoids a
            # zero-batch forward, which some layers dislike).
            if not token_mask.any():
                continue

            # Advanced indexing:
            #   topk_weights[expert_slot_mask]  -> [N] 1-D, where
            #   N = expert_slot_mask.sum() = number of tokens routed to expert i.
            # Unsqueeze to [N, 1] so it broadcasts against the [N, D] expert output.
            expert_weight = topk_weights[expert_slot_mask].unsqueeze(-1)

            # Gather the input rows for tokens going to expert i:
            #   x[token_mask]      shape [N, D]
            # Run the expert:
            #   expert(x[token_mask])  shape [N, D]
            # Weight and scatter-add into y at the same positions:
            #   y[token_mask]       shape [N, D]
            y[token_mask] += expert(x[token_mask]) * expert_weight

        # ---------------- 3. Auxiliary load-balancing loss ----------------
        # Motivation: encourage every expert to see ~1/E of the tokens.
        # We combine two per-expert statistics, both compared against uniform:
        #
        #   (a) prob_fraction[i] = mean over tokens of gate_probs[..., i].
        #       "How much probability mass, on average, does the gate assign
        #        to expert i?"  --  DIFFERENTIABLE (comes from gate_probs).
        #
        #   (b) hard_fraction[i] = fraction of top-K slots that landed on i.
        #       "How often was expert i actually picked?"
        #       NOT differentiable (topk / bincount are discrete), so this
        #       term is a constant wrt gate parameters -- it shows up in the
        #       reported loss value but produces no gradient. It is kept as
        #       an observable, not a training signal.
        #
        # The gate is actually shaped by (a). If you want the classic
        # Switch-Transformer aux ( E * sum_i f_i * P_i ), it plugs f_i and
        # P_i together so the differentiable P_i carries the discrete f_i's
        # signal into the gradient -- this file uses a simpler MSE variant.
        with torch.no_grad():
            # Flatten [B, S, K] -> [B*S*K] indices, count occurrences per expert.
            hard_counts = torch.bincount(
                topk_indices.reshape(-1),
                minlength=self.n_experts,
            ).float()  # [E]
            # Normalize: total slots dispatched = num_tokens * K.
            hard_fraction = hard_counts / (num_tokens * self.n_experts_top_k)  # [E]

        # Average gate probability per expert across all tokens.
        prob_fraction = gate_probs.mean(dim=(0, 1))  # [E]

        # Target: perfectly uniform routing.
        uniform = torch.full_like(prob_fraction, 1.0 / self.n_experts)  # [E]

        aux_loss = F.mse_loss(prob_fraction, uniform) + F.mse_loss(
            hard_fraction, uniform
        )

        return y, aux_loss


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_attention_heads: int,
        n_key_value_heads: int,
        n_experts: int = 0,
        n_experts_top_k: int = 1,
        parallel: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.parallel = parallel
        self.use_moe = n_experts > 0
        self.ffn = (
            MoE(d_model, d_ff, n_experts, n_experts_top_k)
            if self.use_moe
            else FeedForward(d_model, d_ff)
        )
        self.input_layernorm = nn.RMSNorm(normalized_shape=d_model)
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=d_model)
        self.self_attn = GroupQueryAttention(
            d_model, n_attention_heads, n_key_value_heads
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states

        if self.parallel:
            hidden_states = self.input_layernorm(hidden_states)
            attn_output = self.self_attn(hidden_states)
            ffn_output = self.ffn(hidden_states)
            hidden_states = residual + attn_output + ffn_output
            return hidden_states
        else:
            hidden_states = self.input_layernorm(hidden_states)
            # self attention
            attn_output = self.self_attn(hidden_states)
            hidden_states = residual + attn_output

            # feedforward
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            ffn_output = self.ffn(hidden_states)

            return residual + ffn_output
