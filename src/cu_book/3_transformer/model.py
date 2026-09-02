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
        self,
        hidden_size: int,
        num_attention_heads: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.qkv_proj = nn.Linear(
            in_features=hidden_size,
            out_features=3 * hidden_size,
            bias=False,
        )

        self.out_proj = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        # 1. qkv projection, then split heads
        qkv = self.qkv_proj(x)  # [b, t, 3 * hidden_size]
        q, k, v = rearrange(
            qkv,
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.num_attention_heads,
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
        assert q_seq_len <= self.max_seq_len, (
            f"Only support sequence length <= {self.max_seq_len} "
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
        # from (B, num_key_value_heads, S, head_dim) -> (B, num_attention_heads, S, head_dim)
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
            "b n s_q s_kv, b n s_kv h -> b n s_q h",
        )
        # merge heads: [b, n, s_q, h] -> [b, s_q, n * h]
        attn_out = rearrange(attn_out, "b n s_q h -> b s_q (n h)")

        # 4. out projection
        return self.o_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # can fuse gate and up
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FusedFeedForward(nn.Module):
    """SwiGLU FFN with gate_proj and up_proj fused into a single Linear.

    Same math as `FeedForward`, but one matmul produces [..., 2*D_ff] which
    is then chunk'd into gate | up. Saves one kernel launch and reads the
    input activation once instead of twice (LLaMA / Mixtral / Qwen convention).
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class GroupedExperts(nn.Module):
    """SwiGLU FFN experts with weights stacked as [E, ...] for grouped GEMM.

    gate_proj and up_proj are fused into a single `gate_up_proj` of shape
    [E, 2*D_ff, D] (LLaMA / Mixtral / Qwen-MoE convention). One grouped GEMM
    produces [N, 2*D_ff], then `chunk(2)` splits back into gate/up. This
      - saves one kernel launch (2 grouped GEMMs total instead of 3),
      - reads the activation only once for both projections,
      - lets the GEMM tile stream both halves of the weight contiguously.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # nn.Linear-style shape (out, in), stacked over experts.
        # gate and up fused along the out dim: first D_ff rows = gate, next D_ff = up.
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Match nn.Linear default init (Kaiming uniform on 2D slices).
        for w in (self.gate_up_proj, self.down_proj):
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(w[i], a=math.sqrt(5))

    def forward(
        self,
        x_permuted: Float[Tensor, "N D"],  # tokens grouped by target expert
        offs: Tensor,  # [E]  int32, END offset per expert (torch._grouped_mm convention)
    ) -> Tensor:
        """Run all experts as two grouped GEMMs (fused gate/up + down)."""
        # grouped_mm expects b as [E, K, N]; our weights are (out, in) so transpose last two dims.
        # [N, D] @ [E, D, 2*D_ff] -> [N, 2*D_ff], split into gate | up
        gate_up = torch._grouped_mm(  # type: ignore
            x_permuted, self.gate_up_proj.transpose(-1, -2), offs=offs
        )  # [N, 2*D_ff]
        gate, up = gate_up.chunk(2, dim=-1)  # each [N, D_ff]
        hidden = F.silu(gate) * up  # [N, D_ff]
        out = torch._grouped_mm(  # type: ignore
            hidden, self.down_proj.transpose(-1, -2), offs=offs
        )  # [N, D]
        return out


class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward layer.

    Idea:
      Replace a single big FFN by `num_local_experts` smaller FFNs. For each
      token, route it to the top-K experts (K << num_local_experts), and
      combine only those experts' outputs weighted by the gate probabilities.
      Total parameters scale with num_local_experts, but per-token FLOPs
      stay ~ K * (one expert).

    Shape symbols used below:
      B = batch size
      S = sequence length
      D = hidden_size (feature dim)
      E = num_local_experts
      K = num_experts_per_tok        # experts activated per token

    Auxiliary load-balancing loss (returned alongside y):
      Without a balancing signal the gate tends to collapse: it may route
      every token to a single expert, leaving the others cold. We add a small
      penalty that pushes both the mean gate probability and the empirical
      routing fraction toward the uniform 1/E distribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_local_experts: int,
        num_experts_per_token: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_local_experts = num_local_experts
        self.num_experts_per_token = num_experts_per_token

        # Router: one logit per (token, expert). The only learnable part of routing.
        self.router = nn.Linear(hidden_dim, num_local_experts)

        # Experts: SwiGLU FFN weights stacked as [E, ...] for grouped GEMM.
        # Requires bf16/fp16 at forward time (torch._grouped_mm has no fp32 path).
        self.experts = GroupedExperts(num_local_experts, hidden_dim, intermediate_dim)

    def forward(self, x: Float[Tensor, "B S D"]):
        # MoE is a purely token-level op; batch/seq have no semantics for routing.
        # Flatten to [T, D] (T = B*S) so everything downstream is 1D over tokens.
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # [T, D]

        router_topk_indices, router_topk_weights, router_probs = self._route(x_flat)
        y_flat = self._dispatch(x_flat, router_topk_indices, router_topk_weights)
        aux_loss = self._aux_loss(router_topk_indices, router_probs)

        y = y_flat.reshape(B, S, D)  # restore original shape
        return y, aux_loss

    def _route(self, x: Float[Tensor, "T D"]):
        """Score every (token, expert) pair, pick top-K, normalize the K weights.

        Returns
            router_topk_indices: [T, K]  which experts each token picked, in [0, E)
            router_topk_weights: [T, K]  weights over the K picks, sum to 1
            router_probs:        [T, E]  full softmax, kept for aux loss (differentiable)
        """
        router_logits = self.router(x)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [T, E]

        # topk over raw logits (equivalent to topk-of-softmax, cheaper).
        # topk returns distinct indices, so no token picks an expert twice --
        # this makes the mask trick in _dispatch safe.
        router_topk_logits, router_topk_indices = torch.topk(
            router_logits, k=self.num_experts_per_token, dim=-1
        )  # both [T, K]

        # Softmax over the K chosen experts so weights sum to 1 per token
        # (Switch / GShard / Mixtral style renormalization).
        router_topk_weights = F.softmax(router_topk_logits, dim=-1)  # [T, K]

        return router_topk_indices, router_topk_weights, router_probs

    def _dispatch(
        self,
        x: Float[Tensor, "T D"],
        router_topk_indices: Tensor,  # [T, K]
        router_topk_weights: Tensor,  # [T, K]
    ) -> Tensor:
        """Permute -> grouped GEMM -> unpermute (Megatron / Tutel style).

        1. permute:    sort all T*K token-slots by target expert-id so tokens
                       going to the same expert become contiguous in memory.
        2. compute:    one grouped GEMM kernel (torch._grouped_mm) runs all
                       experts at once on their contiguous slices.
        3. unpermute:  scatter results back to the original (token, slot)
                       layout, multiply by top-K weights, sum over K.
        """
        E, K = self.num_local_experts, self.num_experts_per_token
        T, D = x.shape

        # ---------------- 1. permute ----------------
        # T*K "dispatch slots" total (each token contributes K).
        flat_expert = router_topk_indices.reshape(-1)  # [T*K]  which expert per slot
        flat_weight = router_topk_weights.reshape(-1)  # [T*K]  weight per slot

        # perm[j] = original slot index that now sits at position j after
        # sorting by expert-id. Stable sort keeps within-expert order deterministic.
        perm = torch.argsort(flat_expert, stable=True)  # [T*K]
        # Inverse permutation: inv_perm[perm[j]] = j -- used by unpermute.
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        # Each slot j maps to source token j // K. After sorting by expert,
        # source_token[perm] gives the token id at each permuted position.
        source_token = torch.arange(T, device=x.device).repeat_interleave(K)  # [T*K]
        x_permuted = x[source_token[perm]]  # [T*K, D]  contiguous by expert

        # torch._grouped_mm needs END offsets per expert as int32 on device
        # (length E, not E+1). Avoid torch.bincount here: its output shape is
        # data-dependent, which forces torch.compile to graph-break. scatter_add
        # into a fixed-shape [E] buffer is equivalent and stays traceable.
        counts = torch.zeros(E, dtype=torch.int64, device=x.device).scatter_add_(
            0, flat_expert, torch.ones_like(flat_expert)
        )  # [E]
        offs = counts.cumsum(0).to(torch.int32)  # [E]

        # ---------------- 2. compute ----------------
        y_permuted = self.experts(x_permuted, offs)  # [T*K, D]

        # ---------------- 3. unpermute ----------------
        # Undo the sort, weight by top-K gate weight, sum over K slots per token.
        y_slots = y_permuted[inv_perm]  # [T*K, D]
        y_slots = y_slots * flat_weight.unsqueeze(-1)  # [T*K, D]
        y = y_slots.reshape(T, K, D).sum(dim=1)  # [T, D]
        return y

    def _aux_loss(
        self,
        router_topk_indices: Tensor,  # [T, K]
        router_probs: Tensor,  # [T, E]
    ) -> Tensor:
        """Load-balancing loss: push both hard-count and soft-prob per expert to 1/E.

        Two per-expert statistics, both compared against uniform:
          prob_fraction[i] = mean_t router_probs[t, i]  -- DIFFERENTIABLE, trains the router
          hard_fraction[i] = frac of top-K slots on i   -- NOT differentiable (observable)

        Only the prob term produces gradient; the hard term is a monitored value
        that shows up in the loss but has no grad w.r.t. router params.
        (Classic Switch aux `E * sum_i f_i * P_i` couples the two so hard counts
        also shape the gradient -- this file uses a simpler MSE-to-uniform variant.)
        """
        E = self.num_local_experts
        num_slots = router_topk_indices.numel()  # T * K

        with torch.no_grad():
            flat = router_topk_indices.reshape(-1)
            # scatter_add into a fixed [E] buffer (torch.compile-friendly;
            # torch.bincount has data-dependent output shape and breaks the graph).
            hard_counts = torch.zeros(
                E, dtype=torch.float32, device=router_topk_indices.device
            ).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))  # [E]
            hard_fraction = hard_counts / num_slots  # [E]

        prob_fraction = router_probs.mean(dim=0)  # [E]
        uniform = torch.full_like(prob_fraction, 1.0 / E)  # [E]

        return F.mse_loss(prob_fraction, uniform) + F.mse_loss(hard_fraction, uniform)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_local_experts: int = 0,
        num_experts_per_tok: int = 1,
        parallel: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.parallel = parallel
        self.use_moe = num_local_experts > 0
        self.ffn = (
            MoE(hidden_size, intermediate_size, num_local_experts, num_experts_per_tok)
            if self.use_moe
            else FeedForward(hidden_size, intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(normalized_shape=hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=hidden_size)
        self.self_attn = GroupQueryAttention(
            hidden_size, num_attention_heads, num_key_value_heads
        )

    def _run_ffn(self, x: Tensor) -> Tensor:
        """Call the FFN, unpack the MoE tuple, stash aux_loss for read-out."""
        if self.use_moe:
            y, aux_loss = self.ffn(x)
            self.aux_loss = aux_loss  # last-forward aux loss; caller reads and clears
        else:
            y = self.ffn(x)
            self.aux_loss = None
        return y

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states

        if self.parallel:
            hidden_states = self.input_layernorm(hidden_states)
            attn_output = self.self_attn(hidden_states)
            ffn_output = self._run_ffn(hidden_states)
            return residual + attn_output + ffn_output
        else:
            hidden_states = self.input_layernorm(hidden_states)
            # self attention
            attn_output = self.self_attn(hidden_states)
            hidden_states = residual + attn_output

            # feedforward
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            ffn_output = self._run_ffn(hidden_states)

            return residual + ffn_output
