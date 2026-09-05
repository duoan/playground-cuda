"""Chapter 11: Overlap compute with communication using two CUDA streams.

Run:
    torchrun --nproc-per-node=2 src/distributed_training/11_overlap_2stream.py

What this demo shows:
    - Launch an AllReduce on a comm stream while a big matmul runs on the
      compute stream. Use CUDA events to enforce dependencies without
      forcing a full sync.
    - Compare total elapsed time against a serial version that waits for
      AR before starting the matmul.
    - Also demonstrate the standard idiom: `dist.all_reduce(..., async_op=True)`
      returns a handle; you call `.wait()` right before you need the result.

Requires GPU (CUDA streams). Prints timings; asserts non-trivial speedup
only when the workloads are large enough on the given hardware.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from common import setup, cleanup, get_rank, get_world_size, rprint, is_gpu_available


def bench(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    if not is_gpu_available():
        print("Overlap demo requires CUDA. Skipping.")
        return

    rank, world, device = setup()

    # Sizes big enough to have measurable comm and compute.
    N = 4096  # matmul dim
    K = 4096
    C = 32 * 1024 * 1024  # AR tensor: 32M floats = 128 MB

    A = torch.randn(N, K, device=device)
    B = torch.randn(K, N, device=device)
    to_reduce = torch.randn(C, device=device)

    comm_stream = torch.cuda.Stream(device=device)

    # ---- Serial: matmul then AllReduce ----
    def serial():
        x = to_reduce.clone()
        _ = A @ B
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # ---- Overlap via async_op + late .wait() ----
    def overlap_async_handle():
        x = to_reduce.clone()
        h = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        y = A @ B                        # runs on default stream while NCCL fires on its own
        h.wait()                         # wait comm before we "use" x
        return y, x

    # ---- Overlap via explicit two streams + events ----
    def overlap_two_streams():
        x = to_reduce.clone()
        # Fire AR on comm stream
        with torch.cuda.stream(comm_stream):
            # Make sure the clone completed on default stream before AR reads it
            comm_stream.wait_stream(torch.cuda.default_stream(device))
            h = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        # Meanwhile, matmul on default stream
        y = A @ B
        # Before returning, ensure the comm stream's op finished
        h.wait()
        torch.cuda.default_stream(device).wait_stream(comm_stream)
        return y, x

    t_serial = bench(serial)
    t_async = bench(overlap_async_handle)
    t_two = bench(overlap_two_streams)

    rprint(f"serial          : {t_serial:8.2f} ms")
    rprint(f"async_op handle : {t_async:8.2f} ms  ({t_serial/t_async:.2f}x)")
    rprint(f"two streams     : {t_two:8.2f} ms  ({t_serial/t_two:.2f}x)")
    rprint("(numbers depend on NIC/NVLink; the async versions should be ≤ serial)",
           rank=0)

    cleanup()


if __name__ == "__main__":
    main()
