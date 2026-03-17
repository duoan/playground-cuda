"""A single-machine DDP lab that works without multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-based multi-process experiments that teach key DDP performance behaviors."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "skew", "comm", "barrier"),
        default="baseline",
        help="Which distributed pathology to emphasize.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size.")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input feature dimension.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden feature dimension.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes.")
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=20.0,
        help="How much extra host delay to inject into the slow rank in skew mode.",
    )
    parser.add_argument(
        "--slow-rank",
        type=int,
        default=0,
        help="Which rank becomes the straggler in skew mode.",
    )
    parser.add_argument(
        "--comm-mb",
        type=int,
        default=64,
        help="Approximate extra all-reduce payload size in MB for comm mode.",
    )
    parser.add_argument(
        "--barrier-every",
        type=int,
        default=1,
        help="Insert a dist.barrier every N steps in barrier mode.",
    )
    parser.add_argument(
        "--barrier-sleep-ms",
        type=float,
        default=15.0,
        help="Extra rank-0 host work to simulate before the barrier in barrier mode.",
    )
    parser.add_argument(
        "--skip-first-steps",
        type=int,
        default=2,
        help="Number of warmup steps to exclude from steady-state summaries.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a rank-aggregated JSON summary from rank 0.",
    )
    return parser.parse_args()


def setup_dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return dist, rank, world_size, local_rank


class SyntheticDataset:
    def __init__(self, num_samples: int, input_dim: int, num_classes: int, seed: int) -> None:
        import torch

        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, input_dim, generator=generator)
        self.labels = torch.randint(num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def maybe_inject_skew(args: argparse.Namespace, rank: int) -> None:
    if args.mode == "skew" and rank == args.slow_rank:
        time.sleep(args.sleep_ms / 1000.0)


def maybe_run_extra_collective(args: argparse.Namespace, dist, tensor, rank: int) -> None:
    import torch

    if args.mode == "comm":
        element_count = max(1, args.comm_mb * 1024 * 1024 // 4)
        buffer = torch.ones(element_count, dtype=torch.float32)
        dist.all_reduce(buffer)
    elif args.mode == "barrier" and args.barrier_every > 0:
        if (tensor + 1) % args.barrier_every == 0:
            # Simulate rank-0-only work such as logging or checkpoint prep that
            # becomes visible to every rank once a synchronization point is added.
            if rank == 0 and args.barrier_sleep_ms > 0:
                time.sleep(args.barrier_sleep_ms / 1000.0)
            dist.barrier()


def main() -> int:
    import torch
    import torch.distributed as dist_torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    args = parse_args()
    torch.set_num_threads(1)
    dist, rank, world_size, _ = setup_dist()
    torch.manual_seed(1234 + rank)

    dataset = SyntheticDataset(
        num_samples=max(args.steps * args.batch_size * 2, args.batch_size * world_size * 4),
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        seed=2025,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    iterator = cycle_loader(loader)

    model = torch.nn.Sequential(
        torch.nn.Linear(args.input_dim, args.hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(args.hidden_dim, args.num_classes),
    )
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    step_times_ms: list[float] = []
    local_data_wait_ms: list[float] = []
    local_compute_ms: list[float] = []

    for step in range(args.steps):
        if step == 0:
            sampler.set_epoch(0)

        step_start = time.perf_counter()

        data_start = time.perf_counter()
        maybe_inject_skew(args, rank)
        features, labels = next(iterator)
        data_end = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        maybe_run_extra_collective(args, dist, step, rank)
        optimizer.step()
        step_end = time.perf_counter()

        local_data_wait_ms.append((data_end - data_start) * 1000.0)
        local_compute_ms.append((step_end - data_end) * 1000.0)
        step_times_ms.append((step_end - step_start) * 1000.0)

    local_summary = {
        "rank": rank,
        "world_size": world_size,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "sleep_ms": args.sleep_ms,
        "comm_mb": args.comm_mb,
        "barrier_every": args.barrier_every,
        "barrier_sleep_ms": args.barrier_sleep_ms,
        "step_times_ms": step_times_ms,
        "data_wait_ms": local_data_wait_ms,
        "compute_ms": local_compute_ms,
        "final_loss": float(loss.item()),
    }

    gathered: list[dict[str, object]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_summary)

    if rank == 0:
        skip = min(args.skip_first_steps, max(args.steps - 1, 0))
        steady_rank_means = {
            item["rank"]: statistics.mean(item["step_times_ms"][skip:] or item["step_times_ms"])  # type: ignore[index]
            for item in gathered
        }
        steady_data_means = {
            item["rank"]: statistics.mean(item["data_wait_ms"][skip:] or item["data_wait_ms"])  # type: ignore[index]
            for item in gathered
        }
        steady_compute_means = {
            item["rank"]: statistics.mean(item["compute_ms"][skip:] or item["compute_ms"])  # type: ignore[index]
            for item in gathered
        }

        per_step_max = []
        per_step_min = []
        for step in range(skip, args.steps):
            values = [item["step_times_ms"][step] for item in gathered]  # type: ignore[index]
            per_step_max.append(max(values))
            per_step_min.append(min(values))

        slowest_rank = max(steady_rank_means, key=steady_rank_means.get)
        global_batch = args.batch_size * world_size
        slowest_step_ms = steady_rank_means[slowest_rank]
        global_throughput = global_batch / (slowest_step_ms / 1000.0)

        print("App: ddp_lab")
        print(f"Mode: {args.mode}")
        print(f"World size: {world_size}")
        print(f"Per-rank batch size: {args.batch_size}")
        print(f"Global batch size: {global_batch}")
        print(f"Slowest steady-state rank: {slowest_rank}")
        print(f"Slowest steady-state step time (ms): {slowest_step_ms:.2f}")
        print(f"Global steady-state throughput (samples/s): {global_throughput:.2f}")
        print(f"Average rank skew per step (ms): {statistics.mean(m - n for m, n in zip(per_step_max, per_step_min)):.2f}")
        print("")
        for rank_id in sorted(steady_rank_means):
            print(
                f"rank={rank_id} step_ms={steady_rank_means[rank_id]:.2f} "
                f"data_ms={steady_data_means[rank_id]:.2f} compute_ms={steady_compute_means[rank_id]:.2f}"
            )

        if args.summary_json:
            summary = {
                "mode": args.mode,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_batch_size": global_batch,
                "slowest_rank": slowest_rank,
                "slowest_step_ms": slowest_step_ms,
                "global_throughput_samples_per_s": global_throughput,
                "average_rank_skew_ms": statistics.mean(m - n for m, n in zip(per_step_max, per_step_min)),
                "per_rank_step_ms": steady_rank_means,
                "per_rank_data_ms": steady_data_means,
                "per_rank_compute_ms": steady_compute_means,
                "rank_summaries": gathered,
            }
            path = Path(args.summary_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
