"""Chapter 6: Pipeline Parallel — GPipe and 1F1B schedules.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/06_pp_1f1b.py

What this demo shows:
    - Split a stack of `nn.Sequential`-like layers into P stages, one per rank.
    - Cut a global batch into m micro-batches.
    - Two schedulers:
        * GPipe: all forwards, then all backwards. Keeps activations for
          all m micro-batches on every stage.
        * 1F1B (PipeDream): steady-state alternation. Peak activation memory
          is only P micro-batches, not m.
    - Bubble ratio (P-1)/(m+P-1) matches theory.

Verification:
    Both schedulers should produce the same parameter grads as a single-rank
    reference that runs the model on the SAME concatenated batch, one micro
    at a time with grad accumulation.

We use P2P `send`/`recv` for stage-to-stage activation transfer. Works on
gloo (CPU) too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import setup, cleanup, get_rank, get_world_size, rprint, assert_close


# ---- Toy pipelined model ---------------------------------------------------


class Stage(nn.Module):
    """A single pipeline stage: n_layers of Linear+GELU."""

    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = F.gelu(l(x))
        return x


def build_full_model(hidden: int, n_layers: int, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers
        def forward(self, x):
            for l in self.layers:
                x = F.gelu(l(x))
            return x
    return Full()


def build_stage_from_full(full_model: nn.Module, n_stages: int, stage: int,
                          hidden: int) -> Stage:
    """Take the correct slice of layers for this stage."""
    n_layers = len(full_model.layers)
    per_stage = n_layers // n_stages
    my_layers = list(full_model.layers[stage * per_stage:(stage + 1) * per_stage])
    s = Stage(hidden, per_stage)
    with torch.no_grad():
        for dst, src in zip(s.layers, my_layers):
            dst.weight.copy_(src.weight)
    return s


# ---- P2P helpers ------------------------------------------------------------


# We use non-blocking isend so that a stage can post its send BEFORE the
# peer is ready to receive without deadlocking. Every isend returns a
# Work handle we must keep alive (and wait on) until the peer receives.
#
# Message tags let a pair of adjacent stages have both an activation
# (forward direction) and a gradient (backward direction) in flight
# simultaneously without confusing them.
TAG_ACT = 1
TAG_GRAD = 2

_PENDING_SENDS: list = []  # keep isend handles alive until wait_all


def send_to(x: torch.Tensor, dst: int, tag: int = 0) -> None:
    # Detach + clone: the buffer must remain valid until the peer recv
    # completes. Cloning is safest for a teaching demo.
    buf = x.detach().clone().contiguous()
    work = dist.isend(buf, dst=dst, tag=tag)
    _PENDING_SENDS.append((work, buf))  # hold ref


def recv_from(shape, dtype, src: int, device, tag: int = 0) -> torch.Tensor:
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src, tag=tag)
    return buf


def wait_all_sends():
    for w, _ in _PENDING_SENDS:
        w.wait()
    _PENDING_SENDS.clear()


# ---- GPipe scheduler --------------------------------------------------------


def run_gpipe(stage_module: Stage, micro_batches: List[torch.Tensor],
              stage_id: int, n_stages: int, hidden: int, device) -> torch.Tensor | None:
    """Returns the loss on the LAST stage (None elsewhere).

    Warning: keeps activations of all m micro-batches on every stage
    (unlike 1F1B). This is what makes GPipe memory-heavy.
    """
    m = len(micro_batches)
    saved_inputs: List[torch.Tensor] = []
    saved_outputs: List[torch.Tensor] = []

    # ---- forward all m ----
    for i in range(m):
        if stage_id == 0:
            x = micro_batches[i].to(device).detach().requires_grad_(True)
        else:
            x = recv_from(micro_batches[0].shape, torch.float32,
                          stage_id - 1, device, tag=TAG_ACT)
            x = x.detach().requires_grad_(True)
        y = stage_module(x)
        saved_inputs.append(x)
        saved_outputs.append(y)
        if stage_id != n_stages - 1:
            send_to(y.detach(), stage_id + 1, tag=TAG_ACT)

    # ---- backward all m in reverse ----
    total_loss = torch.tensor(0.0, device=device)
    for i in reversed(range(m)):
        if stage_id == n_stages - 1:
            loss = saved_outputs[i].sum()  # placeholder loss
            total_loss = total_loss + loss.detach()
            grad_out = torch.autograd.grad(loss, saved_outputs[i], retain_graph=False)[0]
        else:
            grad_out = recv_from(saved_outputs[i].shape, torch.float32,
                                 stage_id + 1, device, tag=TAG_GRAD)
        # Backward through this stage's compute
        saved_outputs[i].backward(grad_out)
        # Send input-grad upstream
        if stage_id != 0:
            send_to(saved_inputs[i].grad.detach(), stage_id - 1, tag=TAG_GRAD)

    wait_all_sends()
    return total_loss if stage_id == n_stages - 1 else None


# ---- 1F1B scheduler --------------------------------------------------------


def run_1f1b(stage_module: Stage, micro_batches: List[torch.Tensor],
             stage_id: int, n_stages: int, hidden: int, device) -> torch.Tensor | None:
    """PipeDream 1F1B steady-state schedule.

    Phases per stage s (0-indexed):
        - Warm-up:  n_warm = n_stages - 1 - s forwards
        - Steady:   m - n_warm alternating (F, B)
        - Cool-down: n_warm backwards
    """
    m = len(micro_batches)
    n_warm = n_stages - 1 - stage_id

    saved_inputs: List[torch.Tensor] = []
    saved_outputs: List[torch.Tensor] = []
    next_fwd = 0
    next_bwd = 0

    def do_forward():
        nonlocal next_fwd
        i = next_fwd
        next_fwd += 1
        if stage_id == 0:
            x = micro_batches[i].to(device).detach().requires_grad_(True)
        else:
            x = recv_from(micro_batches[0].shape, torch.float32,
                          stage_id - 1, device, tag=TAG_ACT)
            x = x.detach().requires_grad_(True)
        y = stage_module(x)
        saved_inputs.append(x)
        saved_outputs.append(y)
        if stage_id != n_stages - 1:
            send_to(y.detach(), stage_id + 1, tag=TAG_ACT)

    def do_backward():
        nonlocal next_bwd
        i = next_bwd
        next_bwd += 1
        if stage_id == n_stages - 1:
            loss = saved_outputs[i].sum()
            grad_out = torch.autograd.grad(loss, saved_outputs[i], retain_graph=False)[0]
        else:
            grad_out = recv_from(saved_outputs[i].shape, torch.float32,
                                 stage_id + 1, device, tag=TAG_GRAD)
        saved_outputs[i].backward(grad_out)
        if stage_id != 0:
            send_to(saved_inputs[i].grad.detach(), stage_id - 1, tag=TAG_GRAD)
        # We could free saved_inputs[i]/saved_outputs[i] here — that's
        # what makes 1F1B activation-memory light. We leave them for
        # clarity of the demo.

    # Warm-up
    for _ in range(min(n_warm, m)):
        do_forward()
    # Steady
    remain_fwd = m - min(n_warm, m)
    for _ in range(remain_fwd):
        do_forward()
        do_backward()
    # Cool-down
    while next_bwd < m:
        do_backward()

    wait_all_sends()
    return None  # not used for numerical check here


# ---- verification -----------------------------------------------------------


def collect_grads(stage_module: Stage) -> torch.Tensor:
    """Flatten all layer weights' grads into one vector for comparison."""
    return torch.cat([l.weight.grad.flatten() for l in stage_module.layers])


def main():
    rank, world, device = setup()
    P = world
    hidden = 16
    n_layers = 4 * P  # 4 layers per stage
    m = 8            # micro-batches
    micro_shape = (2, hidden)  # small "batch"

    # Build full ref model + stage view (same weights on every rank via bcast).
    full = build_full_model(hidden, n_layers, seed=0)
    with torch.no_grad():
        for p in full.parameters():
            dist.broadcast(p.data.to(device), src=0)
            p.data = p.data.to(device)
    stage_mod = build_stage_from_full(full, P, rank, hidden).to(device)

    # Build the micro-batches. Every rank generates the SAME (seed=0)
    # so that the ref computation matches.
    g = torch.Generator(device="cpu").manual_seed(123)
    micro_batches = [torch.randn(*micro_shape, generator=g) for _ in range(m)]

    # ---- Ref: single-rank sequential forward+backward with grad accum ----
    ref_full = build_full_model(hidden, n_layers, seed=0).to(device)
    with torch.no_grad():
        # Copy from the broadcasted `full` to ensure identical init.
        for pr, ps in zip(ref_full.parameters(), full.parameters()):
            pr.copy_(ps)
    for i in range(m):
        x = micro_batches[i].to(device).detach().requires_grad_(True)
        y = ref_full(x)
        y.sum().backward()   # accumulates grad
    # Now ref_full.layers[k].weight.grad is the reference for stage k.

    # ---- 1F1B run ----
    # Zero grads on stage_mod
    for p in stage_mod.parameters():
        p.grad = None
    run_1f1b(stage_mod, micro_batches, rank, P, hidden, device)

    # Compare stage-local grads against the corresponding slice of ref_full.
    per_stage = n_layers // P
    ref_slice = list(ref_full.layers[rank * per_stage:(rank + 1) * per_stage])
    for l_mine, l_ref in zip(stage_mod.layers, ref_slice):
        assert_close(l_mine.weight.grad, l_ref.weight.grad,
                     rtol=1e-4, atol=1e-4,
                     name=f"1F1B grad stage={rank}")
    rprint("1F1B grads match single-rank reference ✓", rank=0)

    # ---- GPipe run (fresh grads) ----
    for p in stage_mod.parameters():
        p.grad = None
    run_gpipe(stage_mod, micro_batches, rank, P, hidden, device)
    for l_mine, l_ref in zip(stage_mod.layers, ref_slice):
        assert_close(l_mine.weight.grad, l_ref.weight.grad,
                     rtol=1e-4, atol=1e-4,
                     name=f"GPipe grad stage={rank}")
    rprint("GPipe grads match single-rank reference ✓", rank=0)

    # ---- Report bubble ----
    bubble = (P - 1) / (m + P - 1)
    rprint(f"P={P}, m={m}, bubble ratio = {bubble:.2%} (theory)", rank=0)

    cleanup()


if __name__ == "__main__":
    main()
