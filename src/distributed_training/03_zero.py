"""Chapter 4: ZeRO-1 / ZeRO-2 / ZeRO-3 from scratch.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/03_zero.py

What this demo shows:
    - Stage-1: shard optimizer state along DP axis. Grad AR is normal
      (or RS), each rank updates only its slice of Adam m/v/master, then
      AllGather the updated weight slice.
    - Stage-2: also shard grad. Use ReduceScatter instead of AllReduce for
      grads → each rank only holds its shard's grad.
    - Stage-3 (= FSDP FULL_SHARD): also shard weights. Before every
      forward/backward of a "unit" we AllGather the weight; after the
      unit we free it.

We build it around a single big linear (as one flat "unit") to keep the
code short. This is enough to demonstrate the communication pattern.

Verification:
    Run 1 optimizer step of the ZeRO version vs a full single-rank model
    that received the SAME initial weights and the SUM of all ranks' data.
    They should match bit-for-bit modulo AR ordering.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, assert_close,
)


# ---- helper: split a flat vector into W equal shards ------------------------

def _shard_size(numel: int, world: int, rank: int) -> tuple[int, int]:
    base, rem = divmod(numel, world)
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return start, size


# ---- ZeRO wrappers ----------------------------------------------------------


class ZeROOptimizer:
    """Adam that shards optim state (Stage-1). Accepts a list of *flat* params.

    In real ZeRO, params are flattened & sharded. We flatten per-param for
    simplicity and only shard optim state within each param. Semantics are
    identical because we shard each param independently on the same axis.
    """

    def __init__(self, params: List[nn.Parameter], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, stage: int = 1):
        assert stage in (1, 2, 3)
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.stage = stage
        self.step_num = 0
        self.world = get_world_size()
        self.rank = get_rank()

        # Per-param sharded state
        self.m = []       # first moment, shard-local
        self.v = []       # second moment, shard-local
        self.master = []  # FP32 master weight, shard-local
        self.shard_slices = []  # (start, size) per param

        for p in self.params:
            n = p.data.numel()
            start, size = _shard_size(n, self.world, self.rank)
            self.shard_slices.append((start, size))
            self.m.append(torch.zeros(size, device=p.device, dtype=torch.float32))
            self.v.append(torch.zeros(size, device=p.device, dtype=torch.float32))
            self.master.append(p.data.flatten()[start:start + size].to(torch.float32).clone())

    @torch.no_grad()
    def step(self) -> None:
        """One optimizer step, ZeRO-style.

        Assumes .grad is already reduced across DP (either by user before
        calling, or via the ZeRO-2 pipeline that used RS in place of AR).
        For Stage-1 we assume standard AR (grad is the SUM across ranks
        already, so we divide by world for mean).
        """
        self.step_num += 1
        b1t = 1 - self.b1 ** self.step_num
        b2t = 1 - self.b2 ** self.step_num

        for p, m, v, master, (start, size) in zip(
                self.params, self.m, self.v, self.master, self.shard_slices):

            if size == 0:
                continue

            # Local grad shard
            g = p.grad.flatten()[start:start + size].to(torch.float32)
            if self.stage == 1:
                # AR already produced SUM; convert to mean.
                g = g / self.world

            # Adam update on shard only
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / b1t
            v_hat = v / b2t
            update = m_hat / (v_hat.sqrt() + self.eps)
            master.add_(update, alpha=-self.lr)

            # Write master back to BF16/FP32 weight — SHARD ONLY.
            # For Stage-1/2 we must AllGather to make weight consistent on
            # all ranks before next forward.
            local_weight = p.data.flatten()
            local_weight[start:start + size].copy_(master.to(p.data.dtype))

        # Stage-1/2: AllGather updated weight slice across ranks
        for p in self.params:
            n = p.data.numel()
            flat = p.data.flatten().contiguous()
            # Reduce-scatter-like: each rank has its slice authoritative
            # Everyone must see all slices. Use all_gather_into_tensor.
            pieces = [torch.empty(_shard_size(n, self.world, r)[1],
                                  device=p.device, dtype=p.data.dtype)
                      for r in range(self.world)]
            start_r, size_r = self.shard_slices[self.params.index(p)]
            pieces[self.rank] = flat[start_r:start_r + size_r].contiguous()
            gathered = [torch.empty_like(pc) for pc in pieces]
            dist.all_gather(gathered, pieces[self.rank])
            new_flat = torch.cat(gathered)
            p.data.copy_(new_flat.view_as(p.data))


def reduce_grads_stage1_or_2(params: List[nn.Parameter], stage: int) -> None:
    """Reduce gradients across DP. Stage-1/2 differ in AR vs RS but the
    downstream update in this teaching implementation is the same — we
    keep it simple and always AR here; the *savings* of RS in Stage-2
    are described in the book. What Stage-2 truly saves is *storing*
    the full grad; we simulate that by zeroing out non-owned shards
    after the reduction."""
    world = get_world_size()
    rank = get_rank()
    for p in params:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if stage >= 2:
            # keep only owned shard of grad; free the rest
            n = p.grad.numel()
            start, size = _shard_size(n, world, rank)
            flat = p.grad.flatten()
            keep = flat[start:start + size].clone()
            flat.zero_()
            flat[start:start + size].copy_(keep)


# ---- Stage-3 (FSDP-lite) on a single Linear ---------------------------------
#
# A real FSDP wraps a whole submodule and uses a FlatParameter + hooks.
# For teaching we implement Stage-3 on ONE nn.Linear — enough to show the
# AllGather-on-forward / ReduceScatter-on-backward pattern.


class Stage3Linear(nn.Module):
    """FSDP-lite for a SINGLE nn.Linear. Enough to show the AG/RS pattern.

    - weight is stored sharded along dim-0 (output dim, i.e. row-parallel-ish
      but sharded flat).
    - forward: AllGather weight → run linear → discard.
    - backward autograd is handled by us: we override forward to a custom
      autograd.Function so the AG happens under a torch.no_grad context and
      the RS happens in backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world = get_world_size()
        self.rank = get_rank()

        # Sharded weight: (out_features / W, in_features)
        assert out_features % self.world == 0, "keep it simple for the demo"
        self.shard_out = out_features // self.world
        self.weight_shard = nn.Parameter(
            torch.empty(self.shard_out, in_features)
        )
        nn.init.kaiming_uniform_(self.weight_shard, a=5 ** 0.5)
        # sync init across ranks with a broadcast so every rank starts
        # with the same "conceptual" full weight (different shards).
        # Trick: build a full weight on rank 0, broadcast, then keep own shard.
        with torch.no_grad():
            full = torch.empty(out_features, in_features,
                               device=self.weight_shard.device)
            if self.rank == 0:
                nn.init.kaiming_uniform_(full, a=5 ** 0.5)
            dist.broadcast(full, src=0)
            self.weight_shard.copy_(full[self.rank * self.shard_out:
                                         (self.rank + 1) * self.shard_out])

        # bias omitted for brevity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Stage3LinearFn.apply(x, self.weight_shard,
                                     self.out_features, self.world, self.rank)


class _Stage3LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_shard, out_features, world, rank):
        # AllGather weight into full
        pieces = [torch.empty_like(weight_shard) for _ in range(world)]
        dist.all_gather(pieces, weight_shard.contiguous())
        full_weight = torch.cat(pieces, dim=0)  # (out, in)
        y = x @ full_weight.T
        ctx.save_for_backward(x, weight_shard)
        ctx.world = world
        ctx.rank = rank
        ctx.out_features = out_features
        # discard full_weight immediately (leaves scope)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight_shard = ctx.saved_tensors
        world, rank = ctx.world, ctx.rank
        # Recompute the full weight for computing dx (mimics FSDP prefetch)
        pieces = [torch.empty_like(weight_shard) for _ in range(world)]
        dist.all_gather(pieces, weight_shard.contiguous())
        full_weight = torch.cat(pieces, dim=0)
        dx = dy @ full_weight
        # Compute grad on full weight, then ReduceScatter down to shard.
        # dy: (..., out), x: (..., in). We need dW: (out, in).
        dy_flat = dy.reshape(-1, dy.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        dW_full = dy_flat.T @ x_flat  # (out, in)
        # ReduceScatter over ranks along dim 0 (out).
        shard_out = weight_shard.shape[0]
        # First AllReduce, then keep our slice. (In real FSDP: RS directly.)
        # We do RS here explicitly.
        rs_out = torch.empty_like(weight_shard)
        dist.reduce_scatter(rs_out, list(dW_full.chunk(world, dim=0)),
                            op=dist.ReduceOp.SUM)
        # PyTorch DDP semantics = mean; we divide.
        rs_out.div_(world)
        return dx, rs_out, None, None, None


# ---------- verification -----------------------------------------------------


def verify_stage_1_or_2(stage: int, device):
    """Compare 1-step ZeRO update against a single-rank optimizer over
    the concatenated gradients."""
    world = get_world_size()
    rank = get_rank()
    torch.manual_seed(0)
    N = 64  # small param
    p_ref = nn.Parameter(torch.randn(N, device=device))
    p_mine = nn.Parameter(p_ref.detach().clone())

    # per-rank grad (different)
    torch.manual_seed(100 + rank)
    g_local = torch.randn(N, device=device)

    # ---- ref: single-rank AdamW on the MEAN of per-rank grads
    g_all = [torch.empty_like(g_local) for _ in range(world)]
    dist.all_gather(g_all, g_local)
    g_mean = torch.stack(g_all).mean(0)
    opt_ref = torch.optim.Adam([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    p_ref.grad = g_mean.clone()
    opt_ref.step()

    # ---- mine: distribute g_local, AR/RS to get mean, then ZeRO update
    p_mine.grad = g_local.clone()
    reduce_grads_stage1_or_2([p_mine], stage=stage)
    # After AR-with-mean-conversion inside ZeROOptimizer:
    zero = ZeROOptimizer([p_mine], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, stage=stage)
    zero.step()

    assert_close(p_mine.data, p_ref.data, rtol=1e-4, atol=1e-4,
                 name=f"ZeRO-{stage} vs single-rank Adam")
    rprint(f"ZeRO-{stage} matches single-rank Adam ✓", rank=0)


def verify_stage_3(device):
    """Verify Stage-3 Linear forward/backward numerics."""
    world = get_world_size()
    rank = get_rank()
    torch.manual_seed(0)
    in_f, out_f = 16, 4 * world  # divisible

    # Build ref full-linear on every rank with same init (shard on rank 0
    # then re-broadcast identically inside Stage3Linear).
    with torch.no_grad():
        full = torch.empty(out_f, in_f, device=device)
        if rank == 0:
            nn.init.kaiming_uniform_(full, a=5 ** 0.5)
        dist.broadcast(full, src=0)

    ref = nn.Linear(in_f, out_f, bias=False).to(device)
    with torch.no_grad():
        ref.weight.copy_(full)

    mine = Stage3Linear(in_f, out_f).to(device)
    # Overwrite shard so both start bit-identical to `full`
    with torch.no_grad():
        mine.weight_shard.copy_(full[rank * mine.shard_out:
                                     (rank + 1) * mine.shard_out])

    # IMPORTANT: use IDENTICAL x on every rank so the reference computation
    # and the sharded computation see the same data. In FSDP practice each
    # rank sees a different DP shard of x; here we test the *op semantics*
    # of the sharded linear itself, so replicate x.
    torch.manual_seed(1234)  # same on every rank
    x_all = torch.randn(3, in_f, device=device)
    x_ref = x_all.detach().clone().requires_grad_(True)
    x = x_all.detach().clone().requires_grad_(True)

    y_mine = mine(x)
    y_ref = ref(x_ref)
    assert_close(y_mine, y_ref, rtol=1e-4, atol=1e-4, name="Stage-3 forward")

    y_mine.sum().backward()
    y_ref.sum().backward()
    assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4, name="Stage-3 dx")

    # Weight grad: ref has full (out, in); mine has shard (shard_out, in).
    # Because x is identical on every rank, dW_full is identical on every
    # rank. RS summing gives world * dW_full; our /world converts to mean.
    # So mine.weight_shard.grad should equal ref_shard exactly.
    ref_shard = ref.weight.grad[rank * mine.shard_out:(rank + 1) * mine.shard_out]
    assert_close(mine.weight_shard.grad, ref_shard,
                 rtol=1e-4, atol=1e-4, name="Stage-3 dW shard")
    rprint("Stage-3 (FSDP-lite Linear) matches ref forward/backward ✓", rank=0)


def main():
    rank, world, device = setup()
    verify_stage_1_or_2(1, device)
    verify_stage_1_or_2(2, device)
    verify_stage_3(device)
    rprint("all ZeRO stage demos passed ✓", rank=0)
    cleanup()


if __name__ == "__main__":
    main()
