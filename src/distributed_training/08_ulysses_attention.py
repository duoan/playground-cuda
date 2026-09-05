"""Chapter 7: Ulysses (DeepSpeed-Ulysses) sequence parallelism.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/08_ulysses_attention.py
    (NCCL / GPU required — gloo has no all_to_all.)

What this demo shows:
    - Input to each rank: (B, S/W, A, d)   — sharded along SEQ.
    - Do all-to-all along dims (seq → head): (B, S, A/W, d).
    - Run standard attention locally over the full sequence, using only
      A/W heads.
    - Do all-to-all back (head → seq): (B, S/W, A, d).
    - Total: 2 a2a per forward, no ring compute.

Constraint:
    A must be divisible by W (Ulysses fundamental limit; the book's Ch.7
    covers hybrid USP to lift this).

Verification:
    Compare with a single-rank full SDPA over all heads and full seq.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, assert_close,
    is_gpu_available,
)


def all_to_all_seq_to_head(x: torch.Tensor) -> torch.Tensor:
    """(B, S/W, A, d) → (B, S, A/W, d) via all-to-all.

    Split input along dim=2 (A) into W pieces; a2a; concat along dim=1 (S).
    """
    world = get_world_size()
    B, S_local, A, d = x.shape
    assert A % world == 0
    A_local = A // world
    # Split A into W chunks along dim=2; each chunk goes to a different rank.
    # send list length must equal world; each entry shape (B, S_local, A_local, d).
    send_list = list(x.chunk(world, dim=2))
    send = torch.cat([s.contiguous() for s in send_list], dim=0)  # (W*B, S_local, A_local, d)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    # recv is (W*B, S_local, A_local, d) where the W blocks are S shards from other ranks.
    # Reassemble: split back into W pieces along dim=0, then concat along dim=1 (S).
    recv_list = list(recv.chunk(world, dim=0))
    out = torch.cat(recv_list, dim=1)  # (B, W*S_local, A_local, d) = (B, S, A_local, d)
    return out


def all_to_all_head_to_seq(x: torch.Tensor) -> torch.Tensor:
    """(B, S, A/W, d) → (B, S/W, A, d) via all-to-all. Inverse of above."""
    world = get_world_size()
    B, S, A_local, d = x.shape
    assert S % world == 0
    S_local = S // world
    # Split S into W chunks along dim=1; each chunk goes to a different rank.
    send_list = list(x.chunk(world, dim=1))
    send = torch.cat([s.contiguous() for s in send_list], dim=0)
    # send shape: (W*B, S_local, A_local, d)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    # recv (W*B, S_local, A_local, d) with W blocks being A shards from other ranks.
    recv_list = list(recv.chunk(world, dim=0))
    out = torch.cat(recv_list, dim=2)  # (B, S_local, A, d)
    return out


def ulysses_attention(q_local, k_local, v_local, causal=True):
    """q/k/v_local: (B, S/W, A, d). Returns (B, S/W, A, d)."""
    # a2a to head-parallel layout
    q = all_to_all_seq_to_head(q_local)  # (B, S, A/W, d)
    k = all_to_all_seq_to_head(k_local)
    v = all_to_all_seq_to_head(v_local)

    # SDPA expects (B, A, S, d), so permute
    q = q.transpose(1, 2)  # (B, A/W, S, d)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # (B, A/W, S, d)
    out = out.transpose(1, 2).contiguous()  # (B, S, A/W, d)

    # a2a back to seq-parallel layout
    out = all_to_all_head_to_seq(out)  # (B, S/W, A, d)
    return out


def main():
    if not is_gpu_available():
        # a2a in gloo is unsupported. Demo requires NCCL.
        print("Ulysses demo requires NCCL (GPU). Skipping.")
        return

    rank, world, device = setup()
    torch.manual_seed(0)
    B = 2
    A = 4 * world       # must be divisible by W
    S = 8 * world       # must be divisible by W
    d = 16
    causal = True

    # Reference: full-seq full-head attention on every rank (deterministic).
    q_full = torch.randn(B, S, A, d, device=device)
    k_full = torch.randn(B, S, A, d, device=device)
    v_full = torch.randn(B, S, A, d, device=device)

    q_perm = q_full.transpose(1, 2)  # (B, A, S, d)
    k_perm = k_full.transpose(1, 2)
    v_perm = v_full.transpose(1, 2)
    ref = F.scaled_dot_product_attention(q_perm, k_perm, v_perm, is_causal=causal)
    ref = ref.transpose(1, 2)  # (B, S, A, d)

    # Shard input along seq for this rank
    q_l = q_full.chunk(world, dim=1)[rank].contiguous()
    k_l = k_full.chunk(world, dim=1)[rank].contiguous()
    v_l = v_full.chunk(world, dim=1)[rank].contiguous()

    out_l = ulysses_attention(q_l, k_l, v_l, causal=causal)  # (B, S/W, A, d)

    # Gather along seq
    pieces = [torch.empty_like(out_l) for _ in range(world)]
    dist.all_gather(pieces, out_l.contiguous())
    out_full = torch.cat(pieces, dim=1)

    assert_close(out_full, ref, rtol=1e-4, atol=1e-4, name="Ulysses attention")
    rprint("Ulysses attention matches full SDPA reference ✓", rank=0)

    cleanup()


if __name__ == "__main__":
    main()
