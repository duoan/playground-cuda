"""Chapter 3: DDP from scratch — bucket + backward hook + async AllReduce.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/02_ddp_from_scratch.py

What this demo shows:
    - How PyTorch's DistributedDataParallel actually works underneath.
    - Group parameters into fixed-size buckets by declaration order (like
      PyTorch DDP does, but reverse-order so that the buckets that finish
      first in backward are dispatched first).
    - Register a `Tensor.register_hook` on every param's grad; when all
      params in a bucket have their grads ready, issue an async AllReduce
      on the bucket buffer.
    - Wait on all pending handles at `finish_backward()`, then average.

Verification:
    - Run 1 forward+backward with our DDP and 1 with `torch.nn.parallel.DDP`,
      assert grads and params match after `optimizer.step()`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from common import (
    setup, cleanup, get_rank, get_world_size, rprint, pick_device,
    ToyTransformer, assert_close,
)
from common.toy_model import make_random_batch


# ---------- MyDDP: hand-written DDP -------------------------------------------


class _Bucket:
    """One AllReduce unit. Holds a flat buffer and per-param views."""
    __slots__ = ("params", "buffer", "ready_count", "handle", "size")

    def __init__(self, params: List[nn.Parameter], device: torch.device,
                 dtype: torch.dtype):
        self.params = params
        self.size = sum(p.numel() for p in params)
        self.buffer = torch.zeros(self.size, device=device, dtype=dtype)
        self.ready_count = 0
        self.handle = None

    def copy_in(self, p: nn.Parameter, offset: int) -> None:
        self.buffer[offset:offset + p.numel()].copy_(p.grad.flatten())

    def copy_out(self, p: nn.Parameter, offset: int, world: int) -> None:
        # Reduce = SUM; DDP semantics = mean → divide by world.
        p.grad.copy_((self.buffer[offset:offset + p.numel()] / world).view_as(p.grad))


class MyDDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_bytes: int = 25 * 1024 * 1024):
        super().__init__()
        self.module = module
        self.world = get_world_size()
        self.bucket_bytes = bucket_bytes

        # ---- broadcast rank-0 params so every rank starts from same weights
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

        # ---- form buckets in REVERSE parameter order (mimics PyTorch DDP)
        # Backward produces grads output-side first, so buckets grouped from
        # the output end become ready earliest; issuing their AR first gives
        # more overlap with earlier layers' backward compute.
        params = [p for p in module.parameters() if p.requires_grad]
        params_rev = list(reversed(params))
        self._buckets: List[_Bucket] = []
        self._param_to_bucket: dict[int, tuple[_Bucket, int]] = {}
        cur, cur_bytes = [], 0
        dtype = params[0].dtype
        device = params[0].device
        elem_size = params[0].element_size()

        def flush():
            nonlocal cur, cur_bytes
            if not cur:
                return
            b = _Bucket(cur, device, dtype)
            offset = 0
            for p in cur:
                self._param_to_bucket[id(p)] = (b, offset)
                offset += p.numel()
            self._buckets.append(b)
            cur, cur_bytes = [], 0

        for p in params_rev:
            p_bytes = p.numel() * elem_size
            if cur_bytes + p_bytes > bucket_bytes and cur:
                flush()
            cur.append(p)
            cur_bytes += p_bytes
        flush()

        # ---- register hooks
        for p in params:
            p.register_hook(self._make_hook(p))

    def _make_hook(self, p: nn.Parameter):
        def hook(grad):
            # We assign grad to p later via autograd; here just count ready.
            # Note: we use `p.grad` inside `_maybe_launch` because at hook
            # firing time PyTorch has already accumulated grad into p.grad.
            # But hook receives the fresh grad tensor of THIS backward; the
            # p.grad may still be stale if there were accumulations from
            # prior micro-batches. We rely on standard case: 1 backward.
            b, offset = self._param_to_bucket[id(p)]
            # p.grad is set by autograd right after hook returns; we need
            # `grad` here (the incoming tensor) — copy immediately.
            b.buffer[offset:offset + p.numel()].copy_(grad.flatten())
            b.ready_count += 1
            if b.ready_count == len(b.params):
                b.handle = dist.all_reduce(b.buffer, op=dist.ReduceOp.SUM,
                                           async_op=True)
            return grad
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_backward(self) -> None:
        """Call after `loss.backward()` and before `optimizer.step()`."""
        for b in self._buckets:
            assert b.handle is not None, "bucket AR not launched — some param missed backward"
            b.handle.wait()
            # scatter reduced grads back into per-param .grad tensors
            offset = 0
            for p in b.params:
                p.grad.copy_((b.buffer[offset:offset + p.numel()] / self.world).view_as(p.grad))
                offset += p.numel()
            # reset for next iteration
            b.ready_count = 0
            b.handle = None
            b.buffer.zero_()


# ---------- verification ------------------------------------------------------


def run_myddp(model, ids, tgt, optim):
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1))
    optim.zero_grad(set_to_none=False)  # keep .grad tensors alive for hook
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    loss.backward()
    model.finish_backward()
    optim.step()
    return loss.item()


def run_torchddp(model, ids, tgt, optim):
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def main():
    rank, world, device = setup()

    B, S, V, H = 2, 32, 512, 128
    torch.manual_seed(0)
    ref_model = ToyTransformer(n_layers=2, hidden=H, n_heads=4, inter=4 * H,
                               vocab=V, seq=S).to(device)
    my_model = ToyTransformer(n_layers=2, hidden=H, n_heads=4, inter=4 * H,
                              vocab=V, seq=S).to(device)
    # Copy ref weights into ours so both start identical BEFORE broadcast.
    my_model.load_state_dict(ref_model.state_dict())

    # Wrap
    my_ddp = MyDDP(my_model, bucket_bytes=1 * 1024 * 1024)  # small buckets, more AR
    torch_ddp = nn.parallel.DistributedDataParallel(
        ref_model,
        device_ids=[device.index] if device.type == "cuda" else None,
        bucket_cap_mb=1,
    )

    my_opt = torch.optim.SGD(my_ddp.parameters(), lr=0.01)
    ref_opt = torch.optim.SGD(torch_ddp.parameters(), lr=0.01)

    # Each rank gets a DIFFERENT batch (that's the whole point of DP).
    ids, tgt = make_random_batch(B, S, V, device, seed=100 + rank)

    my_loss = run_myddp(my_ddp, ids, tgt, my_opt)
    ref_loss = run_torchddp(torch_ddp, ids, tgt, ref_opt)

    rprint(f"my_loss={my_loss:.6f} ref_loss={ref_loss:.6f}", rank=0)

    # Every rank should have identical params after step (that's what DDP guarantees).
    for (n1, p1), (n2, p2) in zip(my_ddp.module.named_parameters(),
                                  torch_ddp.module.named_parameters()):
        assert n1 == n2
        assert_close(p1.data, p2.data, rtol=1e-4, atol=1e-4, name=f"param {n1}")

    rprint("MyDDP matches torch.nn.parallel.DDP after 1 step ✓", rank=0)
    cleanup()


if __name__ == "__main__":
    main()
