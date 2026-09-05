"""A minimal but real pipeline-parallel engine.

The design choice that makes pipeline parallelism tractable: *a schedule is
data*, not control flow. Every schedule below compiles to a flat list of
instructions like

    F0.0  F1.0  F0.1  B0.1  F2.0  B0.0  ...

and ONE interpreter executes any of them. GPipe, 1F1B, interleaved 1F1B and
zero-bubble differ only in which list they emit. That is also how
`torch.distributed.pipelining` and Megatron-Core are structured, and it is
what lets you reason about (or simulate, or diff) a schedule without running
a single GEMM.

Contents:
    Instr / schedule generators   - the four schedules as pure functions
    WGradTape / PipeLinear        - splitting backward into B and W
    p2p helpers                   - shape handshake, deadlock-free exchange
    PipelineEngine                - the interpreter

Vocabulary used throughout:
    P   pipeline stages (ranks)
    m   micro-batches per optimizer step
    V   virtual chunks per rank (interleaved; V=1 means plain 1F1B)
    g   GLOBAL chunk index, 0 .. V*P-1, laid out as g = v * P + rank
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

# Instruction opcodes.
F = "F"      # forward
B = "B"      # backward, fused: input-grad AND weight-grad
BI = "BI"    # backward, input-grad only   (zero-bubble: unblocks upstream)
BW = "BW"    # backward, weight-grad only  (zero-bubble: blocks nobody)


@dataclass(frozen=True)
class Instr:
    op: str
    mb: int              # micro-batch id
    chunk: int = 0       # VIRTUAL chunk id on this rank, 0 .. V-1

    def __str__(self) -> str:
        return f"{self.op}{self.mb}" if self.chunk == 0 else f"{self.op}{self.mb}.{self.chunk}"


# ---- schedule generators --------------------------------------------------
# Each returns the instruction list for ONE rank. Note how little code each
# schedule is once the executor is factored out.


def sched_gpipe(rank: int, P: int, m: int) -> list[Instr]:
    """All forwards, then all backwards. Holds m activations on every stage."""
    return [Instr(F, i) for i in range(m)] + [Instr(B, i) for i in range(m)]


def sched_1f1b(rank: int, P: int, m: int) -> list[Instr]:
    """PipeDream 1F1B. Same bubble as GPipe, but only P-rank live activations.

    The warm-up depth is what bounds memory: stage 0 runs P-1 forwards before
    its first backward, the last stage runs 1.
    """
    n_warm = min(P - 1 - rank, m)
    out = [Instr(F, i) for i in range(n_warm)]
    for i in range(m - n_warm):
        out += [Instr(F, n_warm + i), Instr(B, i)]
    out += [Instr(B, i) for i in range(m - n_warm, m)]
    return out


def _interleave_maps(P: int, V: int, m: int):
    """Megatron's step -> (chunk, micro-batch) maps for interleaved 1F1B.

    A "step" k counts forward (or backward) passes on this rank, 0 .. m*V-1.
    Ranks share these maps, which is exactly why their sends and receives
    line up without any global coordination.
    """
    group = P * V

    def chunk_of(k: int, forward: bool) -> int:
        c = (k % group) // P
        return c if forward else V - 1 - c

    def mb_of(k: int) -> int:
        return (k // group) * P + (k % P)

    return chunk_of, mb_of


def sched_interleaved_1f1b(rank: int, P: int, m: int, V: int = 2) -> list[Instr]:
    """Interleaved 1F1B (virtual pipeline). Bubble shrinks by ~V.

    Each rank owns V non-adjacent chunks, so a micro-batch goes around the
    ring V times. More P2P, more live activations, smaller bubble.
    """
    assert m % P == 0, "interleaved 1F1B needs m divisible by P"
    chunk_of, mb_of = _interleave_maps(P, V, m)
    total = m * V
    n_warm = min((P - rank - 1) * 2 + (V - 1) * P, total)

    out = [Instr(F, mb_of(k), chunk_of(k, True)) for k in range(n_warm)]
    for j in range(total - n_warm):
        kf, kb = n_warm + j, j
        out += [Instr(F, mb_of(kf), chunk_of(kf, True)),
                Instr(B, mb_of(kb), chunk_of(kb, False))]
    out += [Instr(B, mb_of(k), chunk_of(k, False))
            for k in range(total - n_warm, total)]
    return out


def sched_zero_bubble(rank: int, P: int, m: int) -> list[Instr]:
    """Zero-bubble ZB-H1: 1F1B with backward split into BI and BW.

    BI produces the input gradient and unblocks the upstream stage, so it must
    run as early as 1F1B would. BW produces the weight gradient and blocks
    nobody, so it is deferred into the cool-down, where 1F1B just waits.

    The warm-up bubble survives (there is no W to do yet), which is why ZB-H1
    reduces the bubble rather than removing it. ZB-2P fills that too, at the
    cost of more live activations.
    """
    n_warm = min(P - 1 - rank, m)
    out = [Instr(F, i) for i in range(n_warm)]
    for i in range(m - n_warm):
        out += [Instr(F, n_warm + i), Instr(BI, i)]
    # Cool-down: each BI we still owe, paired with a W we already owe.
    owed_w = deque(range(m - n_warm))
    for i in range(m - n_warm, m):
        out.append(Instr(BI, i))
        if owed_w:
            out.append(Instr(BW, owed_w.popleft()))
    out += [Instr(BW, i) for i in list(owed_w) + list(range(m - n_warm, m))]
    return out


SCHEDULES = {
    "gpipe": sched_gpipe,
    "1f1b": sched_1f1b,
    "interleaved": sched_interleaved_1f1b,
    "zero-bubble": sched_zero_bubble,
}


# ---- splitting backward into B and W --------------------------------------


class WGradTape:
    """Collects deferred weight-gradient work during a BI pass.

    A normal `backward()` computes dL/dx and dL/dW in one traversal. Zero-bubble
    needs them separately, so the linear op below hands dL/dW back as a closure
    instead of accumulating it. This is the whole mechanism -- no autograd
    patching, just a custom Function that declines to do half its job.
    """

    def __init__(self):
        self.sink: list = []

    def push(self, w, gy, x):
        self.sink.append((w, gy, x))

    def take(self) -> list:
        out, self.sink = self.sink, []
        return out

    @staticmethod
    def apply(work: list) -> None:
        for w, gy, x in work:
            gw = gy.reshape(-1, gy.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
            w.grad = gw if w.grad is None else w.grad + gw


class _LinearSplitBW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tape):
        ctx.save_for_backward(x, w)
        ctx.tape = tape
        return x @ w.t()

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        ctx.tape.push(w, gy.detach(), x.detach())   # W: deferred
        return gy @ w, None, None                   # B: now


class PipeLinear(nn.Module):
    """Linear that can defer its weight gradient onto a tape."""

    def __init__(self, n_in: int, n_out: int, tape: WGradTape | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.normal_(self.weight, std=0.02)
        self.tape = tape

    def forward(self, x):
        if self.tape is None:
            return x @ self.weight.t()
        return _LinearSplitBW.apply(x, self.weight, self.tape)


# ---- P2P ------------------------------------------------------------------


_DTYPE_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_CODE_DTYPES = {v: k for k, v in _DTYPE_CODES.items()}


def send_meta(x: torch.Tensor, dst: int, tag: int, group=None) -> None:
    """Tell the peer what is coming. Needed whenever shapes can vary."""
    hdr = torch.tensor([x.dim(), _DTYPE_CODES[x.dtype]] + list(x.shape) + [0] * (6 - x.dim()),
                       dtype=torch.int64)
    dist.send(hdr, dst=dst, tag=tag, group=group)


def recv_meta(src: int, tag: int, group=None) -> tuple[tuple[int, ...], torch.dtype]:
    hdr = torch.empty(8, dtype=torch.int64)
    dist.recv(hdr, src=src, tag=tag, group=group)
    ndim = int(hdr[0])
    return tuple(int(v) for v in hdr[2:2 + ndim]), _CODE_DTYPES[int(hdr[1])]


def exchange(send_t: torch.Tensor | None, send_to: int | None,
             recv_shape: tuple | None, recv_from: int | None,
             *, device, dtype=torch.float32, tag: int = 0,
             group=None) -> torch.Tensor | None:
    """One deadlock-free bidirectional P2P step.

    Naively doing `send(); recv()` on both sides of a pair deadlocks the moment
    both sides want to send first -- which is exactly the 1F1B steady state,
    where a stage sends an activation down while the stage below sends a
    gradient up. Posting every op non-blocking first and waiting afterwards
    removes the ordering constraint entirely.
    """
    ops, out = [], None
    if send_t is not None and send_to is not None:
        ops.append(dist.P2POp(dist.isend, send_t.detach().contiguous(), send_to,
                              group=group, tag=tag))
    if recv_shape is not None and recv_from is not None:
        out = torch.empty(recv_shape, dtype=dtype, device=device)
        ops.append(dist.P2POp(dist.irecv, out, recv_from, group=group, tag=tag))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    return out


# ---- the interpreter ------------------------------------------------------


class PipelineEngine:
    """Executes any instruction list produced above.

    Holds, per virtual chunk, a FIFO of in-flight micro-batches. FIFO is not a
    convenience: every schedule here backwards a chunk's micro-batches in the
    same order it forwarded them, and the P2P matching relies on it.
    """

    def __init__(self, chunks: list[nn.Module], rank: int, P: int, m: int,
                 *, loss_fn=None, tape: WGradTape | None = None,
                 group=None, device=None, act_shape: tuple | None = None):
        self.chunks, self.rank, self.P, self.m = chunks, rank, P, m
        self.V = len(chunks)
        self.n_global = self.V * P
        self.loss_fn, self.tape, self.group = loss_fn, tape, group
        self.device = device or torch.device("cpu")
        self.act_shape = act_shape          # None => negotiate shapes at runtime

        self.live = [deque() for _ in chunks]     # (mb, x_leaf, y) per chunk
        self.pending_w: dict[tuple[int, int], list] = {}
        self.pending_sends: list = []
        self.peak_live = 0
        self.losses: list[torch.Tensor] = []
        self.n_send = self.n_recv = 0

    # -- P2P: the rule that keeps this deadlock-free --
    #
    # Nobody ever blocks in a send. Sends are posted and their handles parked;
    # only receives block. That makes a circular wait impossible, because a
    # cycle needs at least one rank stuck in a send.
    #
    # Get this wrong and 1F1B deadlocks in the steady state, where stage s is
    # sending an activation down at the same moment stage s+1 is sending a
    # gradient up (see part 2 of 21_pp_from_scratch.py). Megatron takes the
    # other route and fuses the two into one call --
    # `send_forward_recv_backward` -- which avoids the cycle the same way and
    # additionally lets the two transfers share one collective.
    #
    # A parked isend owns its buffer until it completes, hence the clone.

    def _send(self, x: torch.Tensor, dst: int, tag: int) -> None:
        buf = x.detach().contiguous().clone()
        work = dist.isend(buf, dst=dst, tag=tag, group=self.group)
        self.pending_sends.append((work, buf))     # buf must outlive the send
        self.n_send += 1

    def _recv(self, shape, src: int, dtype, tag: int) -> torch.Tensor:
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=src, tag=tag, group=self.group)
        self.n_recv += 1
        return buf

    def _drain_sends(self) -> None:
        for work, _ in self.pending_sends:
            work.wait()
        self.pending_sends.clear()

    # -- topology helpers --
    def _g(self, v: int) -> int:
        return v * self.P + self.rank

    def _tag(self, mb: int, g: int, is_grad: bool) -> int:
        """Unique per (micro-batch, boundary, direction).

        With V>1 and P=2 a rank pair carries both activations and gradients in
        the same direction, so ordering alone cannot disambiguate them. Real
        frameworks instead pair the ops positionally inside one
        batch_isend_irecv per step; tags are the simpler equivalent and are why
        these demos pin the gloo backend (NCCL ignores tags).
        """
        return 2 * (mb * self.n_global + g) + int(is_grad)

    def _peer(self, pipe_rank: int) -> int:
        """Pipeline position -> GLOBAL rank.

        P2P takes global ranks even when you pass a sub-group, and your
        pipeline neighbour is not rank +-1 as soon as DP or TP is in play (it
        is rank +- DP*TP under Megatron's ordering). Translating here rather
        than at the call sites is the difference between a working engine and
        one that hangs the moment somebody enables data parallelism.
        """
        if self.group is None:
            return pipe_rank
        return dist.get_global_rank(self.group, pipe_rank)

    def _prev(self) -> int:
        return self._peer((self.rank - 1) % self.P)

    def _next(self) -> int:
        return self._peer((self.rank + 1) % self.P)

    # -- instruction handlers --
    def _forward(self, mb: int, v: int, inputs, labels):
        g = self._g(v)
        if g == 0:
            # Real input data, not an activation. It needs no gradient (there
            # is no upstream stage to send one to) and it may well be integer
            # token ids, which cannot carry one anyway.
            x = inputs[mb].to(self.device)
        else:
            tag = self._tag(mb, g - 1, False)
            if self.act_shape is None:
                shape, dtype = recv_meta(self._prev(), tag, self.group)
            else:
                shape, dtype = self.act_shape, torch.float32
            x = self._recv(shape, self._prev(), dtype, tag)
            # The autograd boundary: a fresh leaf, so that backward() will
            # deposit dL/dx in x.grad for us to ship upstream.
            x = x.detach().requires_grad_(True)

        y = self.chunks[v](x)

        if g == self.n_global - 1:
            loss = self.loss_fn(y, labels[mb].to(self.device)) / self.m
            self.losses.append(loss.detach())
            y = loss
        else:
            tag = self._tag(mb, g, False)
            if self.act_shape is None:
                send_meta(y, self._next(), tag, self.group)
            self._send(y, self._next(), tag)

        self.live[v].append((mb, x, y))
        self.peak_live = max(self.peak_live, sum(len(d) for d in self.live))

    def _backward(self, mb: int, v: int, *, weight: bool):
        g = self._g(v)
        mb_q, x, y = self.live[v].popleft()
        assert mb_q == mb, f"FIFO violated on chunk {v}: expected {mb_q}, got {mb}"

        if g == self.n_global - 1:
            grad_out = None                      # y is the scalar loss
        else:
            grad_out = self._recv(tuple(y.shape), self._next(), y.dtype,
                                  self._tag(mb, g, True))

        torch.autograd.backward(y, grad_out)

        if self.tape is not None:
            work = self.tape.take()
            if weight:
                WGradTape.apply(work)
            else:
                self.pending_w[(mb, v)] = work

        if g != 0:
            self._send(x.grad, self._prev(), self._tag(mb, g - 1, True))

    def _weight(self, mb: int, v: int):
        WGradTape.apply(self.pending_w.pop((mb, v)))

    # -- driver --
    def run(self, schedule: list[Instr], inputs, labels=None):
        for ins in schedule:
            if ins.op == F:
                self._forward(ins.mb, ins.chunk, inputs, labels)
            elif ins.op == B:
                self._backward(ins.mb, ins.chunk, weight=True)
            elif ins.op == BI:
                self._backward(ins.mb, ins.chunk, weight=False)
            elif ins.op == BW:
                self._weight(ins.mb, ins.chunk)
            else:
                raise ValueError(ins.op)
        self._drain_sends()
        assert not any(self.live), "in-flight activations left over"
        assert not self.pending_w, "deferred weight grads never applied"
        return sum(self.losses) if self.losses else None
