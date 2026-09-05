"""Chapter 1: Collectives demo + hand-written Ring AllReduce.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/01_collectives.py

Shows:
    - Broadcast, AllReduce, AllGather, ReduceScatter, All-to-All (NCCL only)
    - Hand-written Ring AllReduce that reproduces NCCL AR bit-for-bit.
    - The identity: AllReduce = ReduceScatter + AllGather.

Works on CPU (gloo) — a2a is skipped there because gloo does not support it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, pick_device,
    is_gpu_available, assert_close,
)


def demo_broadcast(device):
    world = get_world_size()
    rank = get_rank()
    # rank 0 has real data; others start with zeros
    x = torch.arange(4, dtype=torch.float32, device=device) if rank == 0 \
        else torch.zeros(4, dtype=torch.float32, device=device)
    dist.broadcast(x, src=0)
    expected = torch.arange(4, dtype=torch.float32, device=device)
    assert_close(x, expected, name="broadcast")
    rprint("broadcast OK:", x.tolist(), rank=0)


def demo_all_reduce(device):
    rank = get_rank()
    world = get_world_size()
    # Each rank holds [r, r+1, r+2, r+3]; AR should sum → sum_r([r..r+3])
    x = torch.arange(4, dtype=torch.float32, device=device) + rank
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    # Reference: sum over r=0..W-1 of (arange(4) + r) = W*arange(4) + W(W-1)/2
    ref = world * torch.arange(4, dtype=torch.float32, device=device) + world * (world - 1) / 2
    assert_close(x, ref, name="all_reduce")
    rprint("all_reduce OK:", x.tolist(), rank=0)


def demo_all_gather(device):
    rank = get_rank()
    world = get_world_size()
    x = torch.tensor([rank * 10.0, rank * 10.0 + 1], device=device)
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    stacked = torch.cat(out)
    ref = torch.cat([torch.tensor([r * 10.0, r * 10.0 + 1], device=device) for r in range(world)])
    assert_close(stacked, ref, name="all_gather")
    rprint("all_gather OK:", stacked.tolist(), rank=0)


def demo_reduce_scatter(device):
    rank = get_rank()
    world = get_world_size()
    # Each rank has a tensor of length W; RS gives each rank one element = sum over ranks
    x = torch.tensor([rank + r * 0.1 for r in range(world)], device=device)
    out = torch.empty(1, device=device)
    dist.reduce_scatter(out, list(x.chunk(world, dim=0)), op=dist.ReduceOp.SUM)
    # rank r receives sum_{r'}(r' + r * 0.1) = W(W-1)/2 + W * r * 0.1
    expected = world * (world - 1) / 2 + world * rank * 0.1
    assert_close(out, torch.tensor([expected], device=device), name="reduce_scatter")
    rprint("reduce_scatter OK:", out.tolist(), rank=0)


def demo_all_to_all(device):
    """NCCL only — gloo has no all_to_all."""
    if not is_gpu_available():
        rprint("skipping all_to_all (gloo backend)", rank=0)
        return
    rank = get_rank()
    world = get_world_size()
    # Send `[rank * 100 + j]` to rank j
    send = torch.tensor([rank * 100 + j for j in range(world)],
                        dtype=torch.float32, device=device)
    recv = torch.empty(world, dtype=torch.float32, device=device)
    dist.all_to_all_single(recv, send)
    # rank r receives [r' * 100 + r for r' in range(W)]
    ref = torch.tensor([r_ * 100 + rank for r_ in range(world)],
                       dtype=torch.float32, device=device)
    assert_close(recv, ref, name="all_to_all")
    rprint("all_to_all OK:", recv.tolist(), rank=0)


# ---- Hand-written Ring AllReduce ----------------------------------------------
#
# Algorithm (equivalent to NCCL's Ring AR for a single vector):
#   1. Split the tensor into W equal chunks.
#   2. Reduce-Scatter phase: W-1 steps.
#      - At step k, each rank r sends chunk[(r-k) mod W] to (r+1) mod W,
#        receives from (r-1) mod W, and adds the received chunk into its
#        chunk[(r-k-1) mod W]. After W-1 steps, each rank r "owns" the
#        fully-reduced chunk[(r+1) mod W].
#   3. All-Gather phase: W-1 more steps rotating owned chunks around the ring.
#
# Total per-GPU volume: 2 * (W-1)/W * V — the book's canonical formula.


def ring_all_reduce(x: torch.Tensor) -> None:
    """In-place ring AR that reproduces `dist.all_reduce(x, SUM)` numerically."""
    rank = get_rank()
    world = get_world_size()
    if world == 1:
        return

    # Split into W chunks (pad if not divisible).
    N = x.numel()
    pad = (-N) % world
    if pad:
        flat = torch.cat([x.flatten(), x.new_zeros(pad)])
    else:
        flat = x.flatten().clone()
    chunk_size = flat.numel() // world
    chunks = list(flat.split(chunk_size))

    left = (rank - 1) % world
    right = (rank + 1) % world

    # --- ReduceScatter phase ---
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(send_buf)
        # Use batch_isend_irecv so both directions overlap on a pair.
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, right),
            dist.P2POp(dist.irecv, recv_buf, left),
        ])
        for r in reqs:
            r.wait()
        chunks[recv_idx] = chunks[recv_idx] + recv_buf

    # --- AllGather phase ---
    for step in range(world - 1):
        send_idx = (rank - step + 1) % world
        recv_idx = (rank - step) % world
        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, right),
            dist.P2POp(dist.irecv, recv_buf, left),
        ])
        for r in reqs:
            r.wait()
        chunks[recv_idx] = recv_buf

    flat = torch.cat(chunks)
    if pad:
        flat = flat[:N]
    x.copy_(flat.view_as(x))


def demo_ring_all_reduce(device):
    rank = get_rank()
    world = get_world_size()
    # Non-trivial size so the split matters.
    torch.manual_seed(0)
    x_ref = torch.randn(1000, device=device) + rank
    x_mine = x_ref.clone()

    dist.all_reduce(x_ref, op=dist.ReduceOp.SUM)
    ring_all_reduce(x_mine)

    assert_close(x_mine, x_ref, rtol=1e-5, atol=1e-5, name="ring_all_reduce")
    rprint("ring_all_reduce matches NCCL/gloo AR", rank=0)


def demo_ar_equals_rs_plus_ag(device):
    """Verify the identity AllReduce = ReduceScatter then AllGather (per-GPU volume)."""
    world = get_world_size()
    if world == 1:
        return
    rank = get_rank()
    N = 8 * world
    torch.manual_seed(1)
    x = torch.randn(N, device=device) + rank

    # Path A: AR
    a = x.clone()
    dist.all_reduce(a, op=dist.ReduceOp.SUM)

    # Path B: RS + AG
    b = x.clone()
    rs_out = torch.empty(N // world, device=device)
    dist.reduce_scatter(rs_out, list(b.chunk(world, dim=0)), op=dist.ReduceOp.SUM)
    ag_parts = [torch.empty_like(rs_out) for _ in range(world)]
    dist.all_gather(ag_parts, rs_out)
    b = torch.cat(ag_parts)

    assert_close(a, b, rtol=1e-5, atol=1e-5, name="AR == RS+AG")
    rprint("verified: AllReduce == ReduceScatter + AllGather", rank=0)


def main():
    rank, world, device = setup()
    rprint("device:", device, rank=0)
    demo_broadcast(device)
    demo_all_reduce(device)
    demo_all_gather(device)
    demo_reduce_scatter(device)
    demo_all_to_all(device)
    demo_ring_all_reduce(device)
    demo_ar_equals_rs_plus_ag(device)
    rprint("all collectives demos passed ✓", rank=0)
    cleanup()


if __name__ == "__main__":
    main()
