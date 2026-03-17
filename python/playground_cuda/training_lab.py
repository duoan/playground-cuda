"""A small PyTorch training job with configurable bottlenecks for profiling practice."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from collections.abc import Iterator
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small training job and intentionally shift the bottleneck between data, PyTorch, and kernels."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "data", "torch", "kernel"),
        default="baseline",
        help="Preset that changes the model and input pipeline to emphasize a specific bottleneck.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps to run.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input feature dimension.")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden feature dimension.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. If omitted, mode-specific defaults are used.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=None,
        help="Extra sleep inside __getitem__ to simulate slow input pipelines.",
    )
    parser.add_argument(
        "--cpu-transform-depth",
        type=int,
        default=None,
        help="How many CPU transform rounds to run per sample before batching.",
    )
    parser.add_argument(
        "--micro-ops",
        type=int,
        default=None,
        help="Number of tiny eager ops to run in the torch-bound model.",
    )
    parser.add_argument(
        "--pointwise-depth",
        type=int,
        default=None,
        help="How many pointwise blocks to run in the kernel-bound model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device to use. Defaults to "cuda" and falls back to CPU if CUDA is unavailable.',
    )
    parser.add_argument(
        "--amp",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="Use autocast during forward/loss for mixed precision experiments.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile to compare eager and compiled execution.",
    )
    parser.add_argument(
        "--profile",
        choices=("none", "torch"),
        default="none",
        help="Enable torch.profiler and print aggregated operator tables.",
    )
    parser.add_argument(
        "--trace-dir",
        default="traces/torch_profiler",
        help="Directory where torch.profiler trace files are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-first-steps",
        type=int,
        default=1,
        help="Number of initial steps to exclude from steady-state timing summaries.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a structured JSON summary for this run.",
    )
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "baseline":
        args.num_workers = 2 if args.num_workers is None else args.num_workers
        args.sleep_ms = 0.0 if args.sleep_ms is None else args.sleep_ms
        args.cpu_transform_depth = 1 if args.cpu_transform_depth is None else args.cpu_transform_depth
        args.micro_ops = 0 if args.micro_ops is None else args.micro_ops
        args.pointwise_depth = 0 if args.pointwise_depth is None else args.pointwise_depth
    elif args.mode == "data":
        args.num_workers = 0 if args.num_workers is None else args.num_workers
        args.sleep_ms = 2.0 if args.sleep_ms is None else args.sleep_ms
        args.cpu_transform_depth = 6 if args.cpu_transform_depth is None else args.cpu_transform_depth
        args.micro_ops = 0 if args.micro_ops is None else args.micro_ops
        args.pointwise_depth = 0 if args.pointwise_depth is None else args.pointwise_depth
    elif args.mode == "torch":
        args.num_workers = 2 if args.num_workers is None else args.num_workers
        args.sleep_ms = 0.0 if args.sleep_ms is None else args.sleep_ms
        args.cpu_transform_depth = 1 if args.cpu_transform_depth is None else args.cpu_transform_depth
        args.micro_ops = 48 if args.micro_ops is None else args.micro_ops
        args.pointwise_depth = 0 if args.pointwise_depth is None else args.pointwise_depth
        args.hidden_dim = max(args.hidden_dim // 2, 512)
    else:
        args.num_workers = 2 if args.num_workers is None else args.num_workers
        args.sleep_ms = 0.0 if args.sleep_ms is None else args.sleep_ms
        args.cpu_transform_depth = 1 if args.cpu_transform_depth is None else args.cpu_transform_depth
        args.micro_ops = 0 if args.micro_ops is None else args.micro_ops
        args.pointwise_depth = 8 if args.pointwise_depth is None else args.pointwise_depth
        args.hidden_dim = max(args.hidden_dim, 4096)
        args.batch_size = max(args.batch_size, 512)
    return args


class SyntheticClassificationDataset:
    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        num_classes: int,
        sleep_ms: float,
        cpu_transform_depth: int,
        seed: int,
    ) -> None:
        import torch

        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, input_dim, generator=generator)
        self.labels = torch.randint(num_classes, (num_samples,), generator=generator)
        self.sleep_ms = sleep_ms
        self.cpu_transform_depth = cpu_transform_depth

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[object, object]:
        import torch

        sample = self.features[index].clone()
        label = self.labels[index]

        if self.sleep_ms > 0:
            time.sleep(self.sleep_ms / 1000.0)

        for _ in range(self.cpu_transform_depth):
            sample = torch.tanh(sample * 1.01 + 0.01)
            sample = sample.roll(shifts=1, dims=0)

        return sample, label


def make_model(args: argparse.Namespace):
    import torch

    class BaselineNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(args.input_dim, args.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_dim, args.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_dim, args.num_classes),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)

    class TorchBoundNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(args.input_dim, args.hidden_dim)
            self.output_proj = torch.nn.Linear(args.hidden_dim, args.num_classes)
            self.micro_ops = args.micro_ops

        def forward(self, x):  # type: ignore[override]
            x = self.input_proj(x)
            for _ in range(self.micro_ops):
                x = torch.relu(x + 0.01)
                x = x * 0.99
                x = x + torch.sigmoid(x) * 0.01
            return self.output_proj(x)

    class KernelHeavyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(args.input_dim, args.hidden_dim)
            self.output_proj = torch.nn.Linear(args.hidden_dim, args.num_classes)
            self.pointwise_depth = args.pointwise_depth

        def forward(self, x):  # type: ignore[override]
            x = self.input_proj(x)
            for _ in range(self.pointwise_depth):
                gate = torch.sigmoid(x)
                x = torch.nn.functional.gelu(x) * gate
                x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
                x = x + 0.05
            return self.output_proj(x)

    if args.mode in {"baseline", "data"}:
        return BaselineNet()
    if args.mode == "torch":
        return TorchBoundNet()
    return KernelHeavyNet()


@contextlib.contextmanager
def profiled_region(name: str, torch_module, use_nvtx: bool) -> Iterator[None]:
    with torch_module.profiler.record_function(name):
        if use_nvtx:
            torch_module.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch_module.cuda.nvtx.range_pop()
        else:
            yield


def build_profiler(args: argparse.Namespace, torch_module, use_cuda: bool):
    if args.profile != "torch":
        return None

    activities = [torch_module.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch_module.profiler.ProfilerActivity.CUDA)

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    active_steps = max(1, min(args.steps - 2, 6))

    return torch_module.profiler.profile(
        activities=activities,
        schedule=torch_module.profiler.schedule(wait=1, warmup=1, active=active_steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch_module.profiler.tensorboard_trace_handler(str(trace_dir)),
    )


def autocast_context(args: argparse.Namespace, torch_module, device_type: str):
    if device_type != "cuda" or args.amp == "none":
        return contextlib.nullcontext()

    dtype = torch_module.bfloat16 if args.amp == "bf16" else torch_module.float16
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def cycle_loader(loader) -> Iterator[tuple[object, object]]:
    while True:
        for batch in loader:
            yield batch


def main() -> int:
    import torch

    args = apply_mode_defaults(parse_args())
    torch.manual_seed(args.seed)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    device = torch.device(requested_device)
    use_cuda = device.type == "cuda"

    dataset = SyntheticClassificationDataset(
        num_samples=max(args.steps * args.batch_size, args.batch_size * 4),
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        sleep_ms=args.sleep_ms,
        cpu_transform_depth=args.cpu_transform_depth,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = make_model(args).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    step_iterator = cycle_loader(loader)
    step_times_ms: list[float] = []

    profiler = build_profiler(args, torch, use_cuda)
    if profiler is not None:
        profiler.__enter__()

    try:
        for step in range(args.steps):
            if use_cuda:
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            with profiled_region("data_loader", torch, use_cuda):
                features, targets = next(step_iterator)

            with profiled_region("h2d", torch, use_cuda):
                features = features.to(device, non_blocking=use_cuda)
                targets = targets.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(args, torch, device.type):
                with profiled_region("forward", torch, use_cuda):
                    logits = model(features)
                    loss = loss_fn(logits, targets)

            with profiled_region("backward", torch, use_cuda):
                loss.backward()

            with profiled_region("optimizer", torch, use_cuda):
                optimizer.step()

            if use_cuda:
                torch.cuda.synchronize()
            step_end = time.perf_counter()
            step_times_ms.append((step_end - step_start) * 1000.0)

            if profiler is not None:
                profiler.step()

            if step in {0, args.steps - 1}:
                print(f"step={step:03d} loss={loss.item():.4f} step_ms={step_times_ms[-1]:.2f}")
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)

    print("App: training_lab")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input dim: {args.input_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num workers: {args.num_workers}")
    print(f"Sleep per sample (ms): {args.sleep_ms}")
    print(f"CPU transform depth: {args.cpu_transform_depth}")
    print(f"Micro ops: {args.micro_ops}")
    print(f"Pointwise depth: {args.pointwise_depth}")
    print(f"torch.compile: {args.compile}")
    print(f"AMP mode: {args.amp}")
    print(f"Average step time (ms): {sum(step_times_ms) / len(step_times_ms):.2f}")
    steady_state_step_times = step_times_ms[min(args.skip_first_steps, max(len(step_times_ms) - 1, 0)) :] or step_times_ms
    steady_state_step_ms = sum(steady_state_step_times) / len(steady_state_step_times)
    steady_state_samples_per_s = args.batch_size / (steady_state_step_ms / 1000.0)
    print(f"Steady-state step time (ms): {steady_state_step_ms:.2f}")
    print(f"Step time p50 (ms): {statistics.median(steady_state_step_times):.2f}")
    print(f"Steady-state throughput (samples/s): {steady_state_samples_per_s:.2f}")
    print(f"Final loss: {loss.item():.4f}")

    if profiler is not None:
        print("\nTop operators by self CPU time:")
        print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
        if use_cuda:
            print("\nTop operators by CUDA time:")
            print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=12))
            print(f"Torch profiler trace written to: {Path(args.trace_dir).resolve()}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "mode": args.mode,
            "device": str(device),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": args.num_classes,
            "num_workers": args.num_workers,
            "sleep_ms": args.sleep_ms,
            "cpu_transform_depth": args.cpu_transform_depth,
            "micro_ops": args.micro_ops,
            "pointwise_depth": args.pointwise_depth,
            "amp": args.amp,
            "compile": args.compile,
            "skip_first_steps": args.skip_first_steps,
            "average_step_ms": sum(step_times_ms) / len(step_times_ms),
            "steady_state_step_ms": steady_state_step_ms,
            "step_time_p50_ms": statistics.median(steady_state_step_times),
            "steady_state_samples_per_s": steady_state_samples_per_s,
            "step_times_ms": step_times_ms,
            "final_loss": loss.item(),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
