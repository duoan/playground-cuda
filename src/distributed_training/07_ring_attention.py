"""Chapter 7: Ring Attention with Flash-style LSE merge.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/07_ring_attention.py

What this demo shows:
    - Sequence is split into W shards along the S dim (one per rank).
    - Each rank holds Q_i, K_i, V_i corresponding to its shard.
    - A ring rotates K/V around all ranks. At each step, each rank
      computes a *partial* attention block Q_local × K_current with
      the correct causal mask, and merges into an accumulator using
      the log-sum-exp identity (the same trick FlashAttention uses).
    - After W steps, every rank has attended over the full K/V range.

Numerics:
    Merge two partial (O_a, lse_a) and (O_b, lse_b) into (O_new, lse_new):
        lse_new = logaddexp(lse_a, lse_b)
        O_new   = O_a * exp(lse_a - lse_new) + O_b * exp(lse_b - lse_new)

    This is exact because softmax is stable under such re-normalization.

Verification:
    Compare against a single-rank F.scaled_dot_product_attention over the
    full sequence.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import setup, cleanup, get_rank, get_world_size, rprint, assert_close


# ---- attention with LSE ----------------------------------------------------


def attn_block(q, k, v, causal_mask=None):
    """Compute (O, LSE) for one block.
    q: (B, A, Sq, d)
    k: (B, A, Sk, d)
    v: (B, A, Sk, d)
    causal_mask: (Sq, Sk) with 0/-inf, or None
    Returns O: (B, A, Sq, d), lse: (B, A, Sq)
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,A,Sq,Sk)
    if causal_mask is not None:
        scores = scores + causal_mask
    # If the entire row is -inf (mask hides all K positions for this Q),
    # we return zeros and lse=-inf so the merge treats this block as empty.
    lse = torch.logsumexp(scores, dim=-1)              # (B,A,Sq)
    p = torch.exp(scores - lse.unsqueeze(-1))          # (B,A,Sq,Sk)
    out = torch.matmul(p, v)                           # (B,A,Sq,d)
    return out, lse


def merge_attn(O_a, lse_a, O_b, lse_b):
    """Merge two partial attention results with the FlashAttention identity."""
    lse_new = torch.logaddexp(lse_a, lse_b)   # (B,A,Sq)
    w_a = torch.exp(lse_a - lse_new).unsqueeze(-1)
    w_b = torch.exp(lse_b - lse_new).unsqueeze(-1)
    # Handle -inf blocks: exp(-inf - -inf) is nan → treat as 0.
    w_a = torch.nan_to_num(w_a, nan=0.0)
    w_b = torch.nan_to_num(w_b, nan=0.0)
    O_new = O_a * w_a + O_b * w_b
    return O_new, lse_new


def make_causal_mask(rank_q: int, rank_k: int, S_local: int, device, dtype):
    """Return a (S_local, S_local) mask for the block where Q comes from
    rank_q's shard and K comes from rank_k's shard, under GLOBAL causal
    masking (i.e., Q position i can attend to K position j iff j <= i)."""
    if rank_q > rank_k:
        # entire block is in the past → all allowed
        return torch.zeros(S_local, S_local, device=device, dtype=dtype)
    if rank_q < rank_k:
        # entire block is in the future → mask everything
        return torch.full((S_local, S_local), float("-inf"),
                          device=device, dtype=dtype)
    # rank_q == rank_k: standard causal within block
    m = torch.zeros(S_local, S_local, device=device, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones_like(m), diagonal=1).bool(),
                   float("-inf"))
    return m


# ---- Ring Attention -------------------------------------------------------


def ring_attention(q_local, k_local, v_local, causal=True):
    """q/k/v_local: (B, A, S/W, d). Returns (B, A, S/W, d)."""
    world = get_world_size()
    rank = get_rank()
    B, A, S_local, d = q_local.shape
    device, dtype = q_local.device, q_local.dtype

    left = (rank - 1) % world
    right = (rank + 1) % world

    # Start: Q_local vs K_local, V_local (block owned by same rank)
    mask = make_causal_mask(rank, rank, S_local, device, dtype) if causal else None
    O_acc, lse_acc = attn_block(q_local, k_local, v_local, mask)

    k_cur, v_cur = k_local, v_local
    for step in range(1, world):
        # Rotate K, V one step around the ring.
        # After `step` rotations, we hold K/V originally from rank
        # `(rank - step) % world`.
        k_next = torch.empty_like(k_cur)
        v_next = torch.empty_like(v_cur)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, k_cur.contiguous(), right),
            dist.P2POp(dist.irecv, k_next, left),
            dist.P2POp(dist.isend, v_cur.contiguous(), right),
            dist.P2POp(dist.irecv, v_next, left),
        ])
        for r in reqs:
            r.wait()
        k_cur, v_cur = k_next, v_next
        src_rank = (rank - step) % world
        mask = make_causal_mask(rank, src_rank, S_local, device, dtype) if causal else None

        # Skip blocks that are entirely masked out for stability.
        if causal and rank < src_rank:
            continue

        O_new, lse_new = attn_block(q_local, k_cur, v_cur, mask)
        O_acc, lse_acc = merge_attn(O_acc, lse_acc, O_new, lse_new)

    return O_acc


# ---- verification ---------------------------------------------------------


def main():
    rank, world, device = setup()
    torch.manual_seed(0)
    B, A, S, d = 2, 4, 8 * world, 16  # S divisible by W
    causal = True

    # Full q, k, v — replicated on every rank (deterministic seed).
    q_full = torch.randn(B, A, S, d, device=device)
    k_full = torch.randn(B, A, S, d, device=device)
    v_full = torch.randn(B, A, S, d, device=device)

    # Reference: single-rank
    ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=causal)

    # Our ring: each rank holds only its shard
    q_l = q_full.chunk(world, dim=2)[rank].contiguous()
    k_l = k_full.chunk(world, dim=2)[rank].contiguous()
    v_l = v_full.chunk(world, dim=2)[rank].contiguous()
    out_l = ring_attention(q_l, k_l, v_l, causal=causal)

    # Reconstruct full and compare
    pieces = [torch.empty_like(out_l) for _ in range(world)]
    dist.all_gather(pieces, out_l.contiguous())
    out_full = torch.cat(pieces, dim=2)

    assert_close(out_full, ref, rtol=1e-4, atol=1e-4,
                 name=f"Ring Attention causal={causal}")
    rprint("Ring Attention output matches SDPA reference ✓", rank=0)

    # Also test non-causal (bidirectional) just to make sure the merge is
    # correct even without masking.
    ref_bi = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=False)
    out_l_bi = ring_attention(q_l, k_l, v_l, causal=False)
    pieces = [torch.empty_like(out_l_bi) for _ in range(world)]
    dist.all_gather(pieces, out_l_bi.contiguous())
    out_full_bi = torch.cat(pieces, dim=2)
    assert_close(out_full_bi, ref_bi, rtol=1e-4, atol=1e-4,
                 name="Ring Attention non-causal")
    rprint("Ring Attention non-causal also matches ✓", rank=0)

    cleanup()


if __name__ == "__main__":
    main()
