"""CUDA vector add example sized for meaningful profiling."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CUDA add workload that is large enough to inspect with Nsight Compute."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1 << 24,
        help="Number of float32 elements in the CUDA tensor.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations to run before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured add iterations to run on the same tensor.",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=3.0,
        help="Scalar value added on each iteration.",
    )
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.")
        return 1

    if args.size <= 0:
        print("--size must be positive.")
        return 1

    if args.warmup < 0 or args.repeat <= 0:
        print("--warmup must be >= 0 and --repeat must be > 0.")
        return 1

    tensor = torch.zeros(args.size, dtype=torch.float32).to('cuda', non_blocking=True)

    for _ in range(args.warmup):
        tensor.add_(args.value)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(args.repeat):
        tensor.add_(args.value)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    expected_value = args.value * (args.warmup + args.repeat)
    sample_count = min(8, args.size)
    sample = tensor[:sample_count].cpu()

    print("App: python_vector_add")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor size: {args.size}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measured iterations: {args.repeat}")
    print(f"Add value per iteration: {args.value}")
    print(f"Expected final value: {expected_value}")
    print(f"Average iteration time (ms): {elapsed_ms / args.repeat:.4f}")
    print(f"Tensor sample: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
