"""Chapter 8: Expert Parallel (EP) for MoE — minimal dispatch/combine.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/09_ep_moe.py
    (NCCL / GPU required — dispatch uses all_to_all.)

What this demo shows:
    - E experts total, split evenly across W ranks (each rank owns E/W).
    - Router picks top-1 expert per token (top-1 for simplicity; top-k
      would repeat every token k times).
    - Dispatch: each rank sends tokens to the rank that owns the chosen
      expert, via all_to_all_single with per-rank split sizes.
    - Local compute: each rank runs its own E/W experts.
    - Combine: reverse all_to_all to send results back to originating rank.
    - Router weights get applied here (top-1 → identity weight = 1).

Verification:
    Compare against a single-rank MoE that runs all E experts locally on
    the concatenated batch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, assert_close,
    is_gpu_available,
)


class Expert(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class EPMoE(nn.Module):
    def __init__(self, hidden: int, inter: int, n_experts: int):
        super().__init__()
        world = get_world_size()
        rank = get_rank()
        assert n_experts % world == 0
        self.hidden = hidden
        self.n_experts = n_experts
        self.n_local_experts = n_experts // world
        self.world = world
        self.rank = rank
        # Router: 1 linear that outputs logits over all E experts.
        self.router = nn.Linear(hidden, n_experts, bias=False)
        # Local experts.
        self.experts = nn.ModuleList([Expert(hidden, inter)
                                      for _ in range(self.n_local_experts)])

    def load_from_full(self, full_router: nn.Linear, full_experts: nn.ModuleList):
        with torch.no_grad():
            self.router.weight.copy_(full_router.weight)
            for i, e in enumerate(self.experts):
                src = full_experts[self.rank * self.n_local_experts + i]
                e.w1.weight.copy_(src.w1.weight)
                e.w2.weight.copy_(src.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, H). Returns (B, S, H)."""
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)               # (N, H) where N = B*S
        N = x_flat.shape[0]

        # ---- Router ----
        logits = self.router(x_flat)             # (N, E)
        gate = F.softmax(logits, dim=-1)
        top1_prob, top1_idx = gate.max(dim=-1)   # (N,), (N,) top-1 expert per token
        # Which rank owns each token's chosen expert?
        target_rank = top1_idx // self.n_local_experts  # (N,)
        local_expert_idx = top1_idx % self.n_local_experts

        # ---- Sort tokens by target rank so dispatch becomes contiguous ----
        sort_idx = torch.argsort(target_rank)
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(N, device=x.device)
        sorted_x = x_flat[sort_idx]
        sorted_target = target_rank[sort_idx]
        sorted_local_exp = local_expert_idx[sort_idx]
        sorted_prob = top1_prob[sort_idx]

        # Counts to each rank.
        counts_to = torch.bincount(sorted_target, minlength=self.world)  # (W,)

        # ---- Exchange counts so recv side knows sizes ----
        counts_from = torch.empty_like(counts_to)
        dist.all_to_all_single(counts_from, counts_to)

        # ---- Dispatch: send sorted_x, sorted_local_exp, sorted_prob ----
        split_to = counts_to.tolist()
        split_from = counts_from.tolist()

        recv_x = torch.empty(sum(split_from), H, device=x.device, dtype=x.dtype)
        dist.all_to_all_single(recv_x, sorted_x,
                               output_split_sizes=split_from,
                               input_split_sizes=split_to)

        recv_local_exp = torch.empty(sum(split_from), device=x.device, dtype=torch.long)
        dist.all_to_all_single(recv_local_exp, sorted_local_exp,
                               output_split_sizes=split_from,
                               input_split_sizes=split_to)

        recv_prob = torch.empty(sum(split_from), device=x.device, dtype=x.dtype)
        dist.all_to_all_single(recv_prob, sorted_prob,
                               output_split_sizes=split_from,
                               input_split_sizes=split_to)

        # ---- Local compute: run each local expert on its tokens ----
        out_local = torch.zeros_like(recv_x)
        for e_idx in range(self.n_local_experts):
            mask = (recv_local_exp == e_idx)
            if mask.any():
                out_local[mask] = self.experts[e_idx](recv_x[mask])
        # Apply router weight before combine (equivalent to after; it's a scalar).
        out_local = out_local * recv_prob.unsqueeze(-1)

        # ---- Combine: reverse a2a to send back to originating rank ----
        combined = torch.empty(sum(split_to), H, device=x.device, dtype=x.dtype)
        dist.all_to_all_single(combined, out_local,
                               output_split_sizes=split_to,
                               input_split_sizes=split_from)

        # Undo the sort permutation
        out_flat = combined[inv_sort]
        return out_flat.view(B, S, H)


def main():
    if not is_gpu_available():
        print("EP demo requires NCCL (GPU). Skipping.")
        return

    rank, world, device = setup()
    torch.manual_seed(0)
    B, S, H, I = 2, 8, 32, 64
    E = 4 * world

    # ---- Build reference MoE (single-rank, all E experts local) ----
    ref_router = nn.Linear(H, E, bias=False).to(device)
    ref_experts = nn.ModuleList([Expert(H, I).to(device) for _ in range(E)])

    def ref_forward(x):
        x_flat = x.reshape(-1, H)
        logits = ref_router(x_flat)
        gate = F.softmax(logits, dim=-1)
        top1_prob, top1_idx = gate.max(dim=-1)
        out = torch.zeros_like(x_flat)
        for i in range(E):
            m = (top1_idx == i)
            if m.any():
                out[m] = ref_experts[i](x_flat[m]) * top1_prob[m].unsqueeze(-1)
        return out.view_as(x)

    # Broadcast ref weights so every rank has identical ref.
    for p in ref_router.parameters():
        dist.broadcast(p.data, src=0)
    for e in ref_experts:
        for p in e.parameters():
            dist.broadcast(p.data, src=0)

    # ---- Build EP MoE and load matching weights ----
    ep = EPMoE(H, I, E).to(device)
    ep.load_from_full(ref_router, ref_experts)

    x = torch.randn(B, S, H, device=device)
    y_ref = ref_forward(x)
    y_ep = ep(x)

    assert_close(y_ep, y_ref, rtol=1e-4, atol=1e-4, name="EP MoE forward")
    rprint("EP MoE forward matches single-rank MoE ✓", rank=0)

    cleanup()


if __name__ == "__main__":
    main()
