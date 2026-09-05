"""Chapter 5: Sequence Parallel (SP) combined with Tensor Parallel (TP).

Run:
    torchrun --nproc-per-node=4 src/distributed_training/05_sp_with_tp.py

What this demo shows:
    - The LN/Dropout parts of a block are sharded along SEQ dimension.
    - Entering the TP region: AllGather along seq to reconstruct the
      full sequence for the TP compute.
    - Leaving the TP region: ReduceScatter along seq → back to (B, S/TP, H).
    - Total comm volume = same as pure TP (AG+RS = AR). Activation memory
      of LN/Dropout is `1/TP` of the pure-TP baseline.

We build a simple block:  LN (sp) -> FFN (tp) -> LN (sp) -> FFN (tp),
and compare with a single-rank equivalent.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import setup, cleanup, get_rank, get_world_size, rprint, assert_close

# Reuse TP linears from previous demo
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from importlib import import_module
tp_mod = import_module("04_tp_column_row")
ColumnParallelLinear = tp_mod.ColumnParallelLinear
RowParallelLinear = tp_mod.RowParallelLinear


# ---- SP <-> TP boundary primitives ------------------------------------------


class _AllGatherSeq(torch.autograd.Function):
    """Forward: AllGather along seq (dim=1). Backward: ReduceScatter."""

    @staticmethod
    def forward(ctx, x):  # x: (B, S/W, H)
        world = get_world_size()
        ctx.world = world
        pieces = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(pieces, x.contiguous())
        return torch.cat(pieces, dim=1)  # (B, S, H)

    @staticmethod
    def backward(ctx, dy):  # dy: (B, S, H)
        world = ctx.world
        chunks = list(dy.chunk(world, dim=1))
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
        return out


class _ReduceScatterSeq(torch.autograd.Function):
    """Forward: ReduceScatter along seq. Backward: AllGather."""

    @staticmethod
    def forward(ctx, x):  # x: (B, S, H) — partial sum across TP
        world = get_world_size()
        ctx.world = world
        chunks = list(x.chunk(world, dim=1))
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, dy):
        world = ctx.world
        pieces = [torch.empty_like(dy) for _ in range(world)]
        dist.all_gather(pieces, dy.contiguous())
        return torch.cat(pieces, dim=1)


ag_seq = _AllGatherSeq.apply
rs_seq = _ReduceScatterSeq.apply


# ---- SP-aware ColumnParallelLinear / RowParallelLinear ---------------------
#
# The difference from Ch.4's version: instead of copy_to_tp (identity fwd /
# AR bwd), we do AllGather along seq (fwd) / ReduceScatter along seq (bwd).
# And instead of AR at the end (row-parallel), we do ReduceScatter along
# seq (which is exactly equivalent to AR then chunk).


class ColumnParallelLinearSP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = ColumnParallelLinear(in_features, out_features)

    def load_from_full(self, w):
        self.lin.load_from_full(w)

    def forward(self, x):  # x: (B, S/W, H) sequence-sharded
        x = ag_seq(x)      # → (B, S, H) full
        # bypass the internal copy_to_tp: we already handled the seq gather
        return F.linear(x, self.lin.weight, self.lin.bias)


class RowParallelLinearSP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = RowParallelLinear(in_features, out_features)

    def load_from_full(self, w):
        self.lin.load_from_full(w)

    def forward(self, x):  # x: (B, S, in/W) already tp-sharded on last dim
        y = F.linear(x, self.lin.weight, None)  # (B, S, out) partial sum
        y = rs_seq(y)  # → (B, S/W, out) sum-then-scatter along seq
        if self.lin.bias is not None:
            y = y + self.lin.bias
        return y


# ---- SP-aware FFN and Block -------------------------------------------------


class SPFFN(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.w1 = ColumnParallelLinearSP(hidden, inter)
        self.w2 = RowParallelLinearSP(inter, hidden)

    def forward(self, x):  # x: (B, S/W, H)
        return self.w2(F.gelu(self.w1(x)))


class SPBlock(nn.Module):
    """LN → FFN → residual, all with SP+TP.

    - Input:  (B, S/W, H)  (seq-sharded)
    - LN:     local, per-shard  — but LN over hidden dim is fully local
      because it doesn't cross seq positions.
    - FFN:    SPFFN — internally AG at entry, RS at exit.
    - Output: (B, S/W, H)  (seq-sharded again)
    """

    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)  # replicated (small)
        self.ffn = SPFFN(hidden, inter)

    def forward(self, x):
        # LN over hidden dim on the local seq shard is exact — no comm.
        return x + self.ffn(self.ln(x))


# ---- verification -----------------------------------------------------------


def main():
    rank, world, device = setup()
    torch.manual_seed(0)
    B, S, H, I = 2, 8 * world, 32, 64  # seq must be divisible by W

    # ---- reference: single-rank block ----
    ref_ln = nn.LayerNorm(H).to(device)
    ref_w1 = nn.Linear(H, I, bias=False).to(device)
    ref_w2 = nn.Linear(I, H, bias=False).to(device)

    def ref_forward(x):
        return x + ref_w2(F.gelu(ref_w1(ref_ln(x))))

    x_full = torch.randn(B, S, H, device=device, requires_grad=True)

    y_ref = ref_forward(x_full)
    y_ref.sum().backward()

    # ---- our SP+TP block ----
    sp = SPBlock(H, I).to(device)
    with torch.no_grad():
        sp.ln.weight.copy_(ref_ln.weight)
        sp.ln.bias.copy_(ref_ln.bias)
        sp.ffn.w1.load_from_full(ref_w1.weight)
        sp.ffn.w2.load_from_full(ref_w2.weight)

    # Shard input along seq for each rank
    x_shard = x_full.detach().chunk(world, dim=1)[rank].clone().requires_grad_(True)
    y_shard = sp(x_shard)
    y_shard.sum().backward()

    # Reconstruct full y from all shards
    y_pieces = [torch.empty_like(y_shard) for _ in range(world)]
    dist.all_gather(y_pieces, y_shard.contiguous().detach())
    y_full_reconstructed = torch.cat(y_pieces, dim=1)
    assert_close(y_full_reconstructed, y_ref.detach(),
                 rtol=1e-4, atol=1e-4, name="SP+TP forward")

    # Reconstruct full dx from all shards
    dx_pieces = [torch.empty_like(x_shard) for _ in range(world)]
    dist.all_gather(dx_pieces, x_shard.grad.contiguous())
    dx_full = torch.cat(dx_pieces, dim=1)
    assert_close(dx_full, x_full.grad, rtol=1e-4, atol=1e-4,
                 name="SP+TP dx")
    rprint("SP+TP block matches single-rank ref (forward + dx) ✓", rank=0)

    cleanup()


if __name__ == "__main__":
    main()
