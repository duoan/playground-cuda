"""Numerical verification helpers.

Every parallel demo computes its result then reconstructs the equivalent
single-rank baseline and asserts they match. This is the single most
useful thing we can do to prove a parallel implementation is correct.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def assert_close(a: torch.Tensor, b: torch.Tensor, *, rtol: float = 1e-4,
                 atol: float = 1e-4, name: str = "") -> None:
    """torch.allclose with a helpful message."""
    a, b = a.detach(), b.detach()
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs()
    max_diff = diff.max().item() if diff.numel() > 0 else 0.0
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        rel = (diff / (b.abs() + 1e-12)).max().item()
        raise AssertionError(
            f"{name}: max |a-b|={max_diff:.3e}, max rel={rel:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )


def gather_and_assert_close(local: torch.Tensor, full: torch.Tensor, dim: int,
                            *, group=None, name: str = "",
                            rtol: float = 1e-4, atol: float = 1e-4) -> None:
    """AllGather `local` along `dim`, compare with `full` on every rank."""
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        assert_close(local, full, rtol=rtol, atol=atol, name=name)
        return
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local.contiguous(), group=group)
    gathered = torch.cat(parts, dim=dim)
    assert_close(gathered, full, rtol=rtol, atol=atol, name=name)
