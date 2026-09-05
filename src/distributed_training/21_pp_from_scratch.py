"""Chapter 6: pipeline parallelism, built from scratch.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/21_pp_from_scratch.py

The engine lives in `common/pipeline.py` (~250 lines). This file is the proof
that it works, and a walk through the three things that make PP harder than
its picture suggests:

    1. autograd does not cross a process boundary   - you hand-roll the chain
       rule: recv a gradient, call backward, send the input gradient upstream
    2. the naive P2P order deadlocks                - reproduced here, for real
    3. the receiver does not know the shape         - so somebody must say

Then all four schedules -- GPipe, 1F1B, interleaved 1F1B, zero-bubble -- run
against the same engine and must produce gradients bit-comparable to a
single-process reference. A schedule that is fast and wrong is easy to write;
the reference check is what makes the exercise real.

Costs and bubbles are in 22_pp_schedule_sim.py.

Backend note: pinned to gloo. These demos use P2P tags to disambiguate
activations from gradients on the same rank pair, and NCCL ignores tags.
Real frameworks avoid tags by pairing ops positionally inside one
batch_isend_irecv per step.
"""

from __future__ import annotations

import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F_

from common import assert_close, cleanup, rprint, setup
from common.pipeline import (
    PipeLinear, PipelineEngine, SCHEDULES, WGradTape, exchange, recv_meta,
    send_meta,
)

HID, N_MICRO, MICRO_B = 24, 8, 4
LAYERS_PER_CHUNK = 2


# ---- the model ------------------------------------------------------------


class Chunk(nn.Module):
    """A slice of the layer stack. One rank owns V of these."""

    def __init__(self, n_layers: int, tape: WGradTape | None = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [PipeLinear(HID, HID, tape) for _ in range(n_layers)])

    def forward(self, x):
        for l in self.layers:
            x = F_.gelu(l(x))
        return x


def reference_layers(n_total: int, device) -> list[PipeLinear]:
    """The full stack, identical on every rank, used as ground truth."""
    torch.manual_seed(1234)
    return [PipeLinear(HID, HID).to(device) for _ in range(n_total)]


def load_chunk(chunk: Chunk, ref: list[PipeLinear], global_chunk: int):
    """Copy the layers this chunk is responsible for out of the reference.

    A real implementation never materialises the full model -- each rank
    constructs only its own layers, because the whole point is that the model
    does not fit. Here we build it everywhere so we have a ground truth.
    """
    lo = global_chunk * LAYERS_PER_CHUNK
    with torch.no_grad():
        for i, l in enumerate(chunk.layers):
            l.weight.copy_(ref[lo + i].weight)


def make_data(device):
    g = torch.Generator().manual_seed(7)
    xs = [torch.randn(MICRO_B, HID, generator=g).to(device) for _ in range(N_MICRO)]
    ys = [torch.randn(MICRO_B, HID, generator=g).to(device) for _ in range(N_MICRO)]
    return xs, ys


def loss_fn(pred, target):
    return F_.mse_loss(pred, target)


# ---- part 1: the autograd boundary ---------------------------------------


def make_pairs(world: int):
    """Sub-groups used by parts 1-3.

    `new_group` is collective: EVERY rank in the world must call it, in the
    same order, even ranks that will not be members. Creating them lazily
    inside an `if rank < 2:` block is a classic way to hang a job.
    """
    pair_ok = dist.new_group([0, 1])
    pair_slow = dist.new_group([0, 1], timeout=datetime.timedelta(seconds=2))
    return pair_ok, pair_slow


def part1_autograd_boundary(rank, P, device, pair):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 1  autograd does not cross a process boundary", rank=0)
    rprint("=" * 76, rank=0)

    rprint("  On stage 1, `loss.backward()` cannot reach stage 0's weights: the", rank=0)
    rprint("  graph was severed by the send/recv. So you splice it by hand.", rank=0)
    rprint("", rank=0)
    rprint("  forward   x = recv(...).requires_grad_(True)   <- a NEW leaf", rank=0)
    rprint("            y = stage(x);  send(y)", rank=0)
    rprint("  backward  gy = recv(...)                       <- dL/dy from below", rank=0)
    rprint("            torch.autograd.backward(y, gy)       <- fills W.grad AND x.grad", rank=0)
    rprint("            send(x.grad)                         <- dL/dx goes upstream", rank=0)

    # Two-stage worked example: split f(x) = L2(gelu(L1(x))) across ranks.
    torch.manual_seed(99)
    w1 = torch.randn(HID, HID, device=device) * 0.1
    w2 = torch.randn(HID, HID, device=device) * 0.1
    x0 = torch.randn(MICRO_B, HID, device=device)

    # Single-process ground truth.
    a = w1.clone().requires_grad_(True)
    b = w2.clone().requires_grad_(True)
    out = F_.gelu(x0 @ a.t()) @ b.t()
    out.pow(2).sum().backward()

    if rank < 2:
        if rank == 0:
            wa = w1.clone().requires_grad_(True)
            y = F_.gelu(x0 @ wa.t())
            exchange(y, 1, None, None, device=device, tag=0, group=pair)
            gy = exchange(None, None, tuple(y.shape), 1, device=device,
                          tag=1, group=pair)
            torch.autograd.backward(y, gy)          # splice happens here
            assert_close(wa.grad, a.grad, rtol=1e-4, atol=1e-5,
                         name="stage 0 weight grad")
            rprint("\n  stage 0's weight grad matches the single-process value, and it", rank=0)
            rprint("  was produced without stage 0 ever seeing the loss.", rank=0)
        else:
            xr = exchange(None, None, (MICRO_B, HID), 0, device=device,
                          tag=0, group=pair)
            xr = xr.detach().requires_grad_(True)
            wb = w2.clone().requires_grad_(True)
            (xr @ wb.t()).pow(2).sum().backward()
            exchange(xr.grad, 0, None, None, device=device, tag=1, group=pair)
    dist.barrier()

    rprint("\n  Two details people get wrong here:", rank=0)
    rprint("  * `.detach().requires_grad_(True)` is not defensive coding. Without", rank=0)
    rprint("    it the recv buffer has no grad_fn and no .grad, so there is", rank=0)
    rprint("    nothing to send upstream.", rank=0)
    rprint("  * use `torch.autograd.backward(y, gy)`, not `y.backward()` on a", rank=0)
    rprint("    non-scalar. Only the last stage has a scalar to start from; every", rank=0)
    rprint("    other stage starts from a gradient it was handed.", rank=0)


# ---- part 2: the deadlock ------------------------------------------------


def part2_deadlock(rank, P, device, pair_ok, pair_slow):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 2  the naive P2P order deadlocks (reproduced)", rank=0)
    rprint("=" * 76, rank=0)

    rprint("  1F1B steady state: stage s sends an activation DOWN while stage s+1", rank=0)
    rprint("  sends a gradient UP. If both do `send(); recv()`, both block in send", rank=0)
    rprint("  waiting for a recv that the other will only post afterwards.", rank=0)

    if rank < 2:
        peer = 1 - rank
        out_t = torch.full((MICRO_B, HID), float(rank))
        in_t = torch.zeros(MICRO_B, HID)

        # (a) naive: send first on both sides. Uses the short-timeout group so
        # the demo reports the hang instead of becoming one. Note that a gloo
        # timeout CLOSES the pair, so everything after this needs a fresh
        # group -- which is also true in production: a P2P timeout does not
        # leave you a usable process group to retry on.
        deadlocked = False
        try:
            dist.send(out_t, dst=peer, group=pair_slow)
            dist.recv(in_t, src=peer, group=pair_slow)
        except RuntimeError as e:
            deadlocked = "imeout" in str(e) or "imed out" in str(e)
        if rank == 0:
            rprint(f"\n  (a) send-then-recv on both sides -> deadlock: {deadlocked}", rank=0)
        assert deadlocked

        # (b) odd/even ordering: break the symmetry by rank parity.
        in_t.zero_()
        if rank % 2 == 0:
            dist.send(out_t, dst=peer, group=pair_ok)
            dist.recv(in_t, src=peer, group=pair_ok)
        else:
            dist.recv(in_t, src=peer, group=pair_ok)
            dist.send(out_t, dst=peer, group=pair_ok)
        ok_parity = in_t[0, 0].item() == float(peer)

        # (c) post everything non-blocking, then wait. No ordering rule needed.
        got = exchange(out_t, peer, (MICRO_B, HID), peer, device=device,
                       tag=5, group=pair_ok)
        ok_batch = got[0, 0].item() == float(peer)

        if rank == 0:
            rprint(f"  (b) odd/even ordering        -> exchanged correctly: {ok_parity}",
                   rank=0)
            rprint(f"  (c) batch_isend_irecv        -> exchanged correctly: {ok_batch}",
                   rank=0)
        assert ok_parity and ok_batch
    dist.barrier()

    rprint("\n  (b) works but is fragile: every new message you add has to be", rank=0)
    rprint("  parity-ordered too, and interleaved schedules break the tidy", rank=0)
    rprint("  'one send one recv per step' assumption it relies on.", rank=0)
    rprint("  (c) is what the engine uses and what Megatron does. Posting the", rank=0)
    rprint("  ops before waiting removes the ordering constraint entirely, and", rank=0)
    rprint("  it also lets the two transfers overlap on the wire.", rank=0)


# ---- part 3: shape negotiation ------------------------------------------


def part3_shapes(rank, P, device, pair):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 3  the receiver has to allocate before it receives", rank=0)
    rprint("=" * 76, rank=0)

    rprint("  `dist.recv` needs a tensor to write into, so the receiver must know", rank=0)
    rprint("  the shape and dtype BEFORE the data arrives. With fixed shapes you", rank=0)
    rprint("  hard-code it. With packing or variable sequence lengths you cannot,", rank=0)
    rprint("  and guessing wrong is a silent truncation or a hang.", rank=0)

    if rank < 2:
        shapes = [(2, HID), (5, HID), (1, HID)]     # a varlen batch
        if rank == 0:
            for s in shapes:
                t = torch.full(s, float(s[0]))
                send_meta(t, 1, tag=20, group=pair)
                exchange(t, 1, None, None, device=device, tag=21, group=pair)
        else:
            got = []
            for _ in shapes:
                shape, dtype = recv_meta(0, tag=20, group=pair)
                t = exchange(None, None, shape, 0, device=device, dtype=dtype,
                             tag=21, group=pair)
                got.append(tuple(t.shape))
            assert got == shapes, got
            rprint(f"\n  receiver learned shapes {got} from an 8-int header", rank=0)
    dist.barrier()

    rprint("  Cost is one tiny message per tensor, which is why frameworks make it", rank=0)
    rprint("  optional: Megatron's `--variable-seq-lengths` turns the handshake on", rank=0)
    rprint("  and it is off by default. The engine here does the same -- pass", rank=0)
    rprint("  act_shape to skip it, leave it None to negotiate.", rank=0)


# ---- part 4: run every schedule, check against one reference -------------


def run_schedule(name: str, rank: int, P: int, device, xs, ys, V: int):
    """Build this rank's chunks, execute the schedule, return grads by layer."""
    tape = WGradTape() if name == "zero-bubble" else None
    n_global = V * P
    ref = reference_layers(n_global * LAYERS_PER_CHUNK, device)

    chunks = []
    for v in range(V):
        c = Chunk(LAYERS_PER_CHUNK, tape).to(device)
        load_chunk(c, ref, v * P + rank)
        chunks.append(c)

    engine = PipelineEngine(chunks, rank, P, N_MICRO, loss_fn=loss_fn,
                            tape=tape, device=device,
                            act_shape=(MICRO_B, HID))
    gen = SCHEDULES[name]
    sched = gen(rank, P, N_MICRO, V) if name == "interleaved" else gen(rank, P, N_MICRO)
    engine.run(sched, xs, ys)

    grads = {}
    for v, c in enumerate(chunks):
        for i, l in enumerate(c.layers):
            grads[(v * P + rank) * LAYERS_PER_CHUNK + i] = l.weight.grad.clone()
    return grads, engine


def reference_grads(n_global: int, device, xs, ys):
    """Single process, all m micro-batches, gradient accumulation."""
    ref = reference_layers(n_global * LAYERS_PER_CHUNK, device)
    for l in ref:
        l.weight.requires_grad_(True)
        l.weight.grad = None
    for i in range(N_MICRO):
        h = xs[i]
        for l in ref:
            h = F_.gelu(l(h))
        (loss_fn(h, ys[i]) / N_MICRO).backward()
    return {k: l.weight.grad for k, l in enumerate(ref)}


def part4_verify(rank, P, device, xs, ys):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 4  four schedules, one engine, one reference", rank=0)
    rprint("=" * 76, rank=0)

    for name in ("gpipe", "1f1b", "zero-bubble", "interleaved"):
        V = 2 if name == "interleaved" else 1
        grads, engine = run_schedule(name, rank, P, device, xs, ys, V)
        ref = reference_grads(V * P, device, xs, ys)
        worst = 0.0
        for layer, g in grads.items():
            assert_close(g, ref[layer], rtol=1e-4, atol=1e-6,
                         name=f"{name} layer {layer}")
            worst = max(worst, (g - ref[layer]).abs().max().item())

        stats = torch.tensor([worst, float(engine.peak_live),
                              float(engine.n_send + engine.n_recv)], device=device)
        gathered = [torch.zeros_like(stats) for _ in range(P)]
        dist.all_gather(gathered, stats)
        peaks = [int(t[1].item()) for t in gathered]
        p2p = [int(t[2].item()) for t in gathered]
        err = max(t[0].item() for t in gathered)
        rprint(f"  {name:<13} max|grad-ref| = {err:.2e}   peak live acts/stage = "
               f"{peaks}   P2P msgs = {p2p}", rank=0)

    rprint("\n  All four match the reference, so the schedules differ only in when", rank=0)
    rprint("  work happens -- never in what is computed. That is the invariant to", rank=0)
    rprint("  assert in a test when you write your own scheduler.", rank=0)
    rprint("\n  Note the peak-live column: GPipe holds m=8 on every stage, 1F1B", rank=0)
    rprint("  holds P-rank, interleaved holds more than either. And the P2P count", rank=0)
    rprint("  is what interleaving actually costs you.", rank=0)


def main():
    rank, world, device = setup(backend="gloo")
    if world < 2:
        rprint("needs >= 2 ranks: torchrun --nproc-per-node=4 ...")
        cleanup()
        return
    P = world
    xs, ys = make_data(device)
    pair_ok, pair_slow = make_pairs(world)

    part1_autograd_boundary(rank, P, device, pair_ok)
    part2_deadlock(rank, P, device, pair_ok, pair_slow)
    part3_shapes(rank, P, device, pair_ok)
    part4_verify(rank, P, device, xs, ys)

    rprint("\n" + "=" * 76, rank=0)
    rprint("pipeline engine verified against the single-process reference", rank=0)
    rprint("=" * 76, rank=0)
    cleanup()


if __name__ == "__main__":
    main()
