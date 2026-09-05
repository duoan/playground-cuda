"""Chapter 5: Tensor Parallel — ColumnParallelLinear + RowParallelLinear.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/04_tp_column_row.py

What this demo shows:
    - Megatron-style column-then-row pattern for FFN and Attention.
    - ColumnParallelLinear: shard weight along OUT dim; no comm on forward
      (activation comes out sharded on OUT). Input is replicated.
    - RowParallelLinear: shard weight along IN dim; input comes in sharded
      on last dim; forward ends with AllReduce.
    - FFN(x) = Row( GELU( Column(x) ) ) needs exactly one AllReduce per
      forward → matches the book's Ch.5 calculation.
    - Attention: Q/K/V column-parallel (shard heads), Output row-parallel
      + AR.

Verification:
    Compare TP forward output & input-grad with a single-rank ref that
    starts from the same full-weight init.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, assert_close,
)


# ---- primitives -------------------------------------------------------------


class _CopyToTPRegion(torch.autograd.Function):
    """Identity forward; AllReduce on backward. Used at column-parallel entry."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, dy):
        dist.all_reduce(dy, op=dist.ReduceOp.SUM)
        return dy


class _ReduceFromTPRegion(torch.autograd.Function):
    """AllReduce on forward; identity on backward. Used at row-parallel exit."""

    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, dy):
        return dy


copy_to_tp = _CopyToTPRegion.apply
reduce_from_tp = _ReduceFromTPRegion.apply


# ---- ColumnParallelLinear ---------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """Weight: (out/W, in). Forward output: (..., out/W). No comm on forward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.world = get_world_size()
        self.rank = get_rank()
        assert out_features % self.world == 0
        self.in_features = in_features
        self.out_features = out_features
        self.shard_out = out_features // self.world
        self.weight = nn.Parameter(torch.empty(self.shard_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.shard_out))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor) -> None:
        """Set weight from a full (out, in) tensor for verification."""
        self.weight.copy_(full_weight[self.rank * self.shard_out:
                                      (self.rank + 1) * self.shard_out])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is replicated. Register backward hook via CopyToTPRegion so
        # that grads on x get AllReduced (because column-parallel means
        # every TP rank contributes to input gradient).
        x = copy_to_tp(x)
        return F.linear(x, self.weight, self.bias)


# ---- RowParallelLinear ------------------------------------------------------


class RowParallelLinear(nn.Module):
    """Weight: (out, in/W). Input MUST come in sharded on last dim. Forward
    ends with AllReduce, producing a full output on every rank."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.world = get_world_size()
        self.rank = get_rank()
        assert in_features % self.world == 0
        self.in_features = in_features
        self.out_features = out_features
        self.shard_in = in_features // self.world
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_in))
        if bias:
            # bias must NOT be sharded (added after AR) — usually done post-AR
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor) -> None:
        """Set weight from a full (out, in) tensor for verification."""
        self.weight.copy_(full_weight[:, self.rank * self.shard_in:
                                        (self.rank + 1) * self.shard_in])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in/W)
        y = F.linear(x, self.weight, None)  # partial sum
        y = reduce_from_tp(y)                # AllReduce over TP
        if self.bias is not None:
            y = y + self.bias
        return y


# ---- TP FFN and TP Attention -----------------------------------------------


class TPFFN(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(hidden, inter)  # → (..., inter/W)
        self.w2 = RowParallelLinear(inter, hidden)      # (..., inter/W) → (..., hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TPAttention(nn.Module):
    """MHA with heads sharded across TP."""

    def __init__(self, hidden: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert n_heads % get_world_size() == 0
        assert hidden % n_heads == 0
        self.world = get_world_size()
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_h = hidden // n_heads
        self.n_heads_local = n_heads // self.world
        self.causal = causal
        # Q/K/V column-parallel — output is (..., hidden/W) → n_heads_local heads
        self.wq = ColumnParallelLinear(hidden, hidden)
        self.wk = ColumnParallelLinear(hidden, hidden)
        self.wv = ColumnParallelLinear(hidden, hidden)
        # Output row-parallel: input is (..., hidden/W), output full hidden
        self.wo = RowParallelLinear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        # Each is (B, S, hidden/W)
        q = self.wq(x).view(B, S, self.n_heads_local, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads_local, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads_local, self.d_h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        # (B, n_heads_local, S, d_h) → (B, S, hidden/W)
        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden // self.world)
        return self.wo(out)


# ---- verification -----------------------------------------------------------


def verify_ffn(device):
    world = get_world_size()
    torch.manual_seed(0)
    B, S, H, I = 2, 8, 32, 64
    # Ref FFN
    ref_w1 = nn.Linear(H, I, bias=False).to(device)
    ref_w2 = nn.Linear(I, H, bias=False).to(device)
    x_ref = torch.randn(B, S, H, device=device, requires_grad=True)
    y_ref = ref_w2(F.gelu(ref_w1(x_ref)))
    y_ref.sum().backward()

    tp = TPFFN(H, I).to(device)
    tp.w1.load_from_full(ref_w1.weight)
    tp.w2.load_from_full(ref_w2.weight)
    x = x_ref.detach().clone().requires_grad_(True)
    y = tp(x)
    y.sum().backward()

    assert_close(y, y_ref, rtol=1e-4, atol=1e-4, name="TP FFN forward")
    assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4, name="TP FFN dx")
    rprint("TPFFN forward+backward matches single-rank FFN ✓", rank=0)


def verify_attention(device):
    world = get_world_size()
    torch.manual_seed(0)
    B, S, H, A = 2, 8, 32, 4 * world  # heads must be divisible by W

    from common.toy_model import ToyAttention
    ref = ToyAttention(H, A, causal=True).to(device)

    tp = TPAttention(H, A, causal=True).to(device)
    tp.wq.load_from_full(ref.wq.weight)
    tp.wk.load_from_full(ref.wk.weight)
    tp.wv.load_from_full(ref.wv.weight)
    tp.wo.load_from_full(ref.wo.weight)

    x_ref = torch.randn(B, S, H, device=device, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref)
    y = tp(x)
    y_ref.sum().backward()
    y.sum().backward()

    assert_close(y, y_ref, rtol=1e-4, atol=1e-4, name="TP attn forward")
    assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4, name="TP attn dx")
    rprint("TPAttention matches single-rank ToyAttention ✓", rank=0)


def main():
    rank, world, device = setup()
    verify_ffn(device)
    verify_attention(device)
    rprint("all TP demos passed ✓", rank=0)
    cleanup()


if __name__ == "__main__":
    main()
