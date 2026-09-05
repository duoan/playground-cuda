"""Small helpers used across every demo.

Design goals:
- Work on CPU (Gloo) so demos run without GPUs. Prefer NCCL when CUDA is
  available. All demos check `pick_device()` and route tensors accordingly.
- Deterministic: seed derived from a fixed base + rank when the caller
  wants divergent per-rank data, or same seed for shared init.
- Small: everything here should fit on one screen.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def is_gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def pick_backend() -> str:
    """NCCL on GPU, gloo on CPU. Never mix — gloo lacks a2a but supports basics."""
    return "nccl" if is_gpu_available() else "gloo"


def pick_device() -> torch.device:
    if is_gpu_available():
        # torchrun sets LOCAL_RANK; fall back to 0 for single-GPU debugging.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def setup(seed_base: int = 42, deterministic: bool = True,
          backend: str | None = None) -> tuple[int, int, torch.device]:
    """Init process group + set device + seed. Returns (rank, world, device).

    Must be called at the top of every demo's main(). Pass `backend="gloo"` to
    force CPU collectives — needed by demos that rely on P2P tags, which NCCL
    ignores.
    """
    backend = backend or pick_backend()
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()

    device = torch.device("cpu") if backend == "gloo" else pick_device()
    if is_gpu_available() and backend != "gloo":
        torch.cuda.set_device(device)

    seed = seed_base + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_gpu_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(False)  # keep loose; matmul isn't strict
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return rank, world, device


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def rprint(*args, rank: int | None = None, **kwargs) -> None:
    """Print with `[rank r/W]` prefix. If `rank` given, only that rank prints."""
    r = get_rank()
    w = get_world_size()
    if rank is not None and r != rank:
        return
    print(f"[rank {r}/{w}]", *args, **kwargs, flush=True)
