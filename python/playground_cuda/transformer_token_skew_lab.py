"""A small distributed lab for variable-length transformer token skew."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny transformer-like DDP lab that highlights token skew from variable sequence lengths."
    )
    parser.add_argument(
        "--strategy",
        choices=("sample", "bucket", "token"),
        default="sample",
        help="How each rank forms batches from variable-length samples.",
    )
    parser.add_argument(
        "--length-mode",
        choices=("fixed", "variable"),
        default="variable",
        help="Whether all samples have the same sequence length or a heavy-tailed distribution.",
    )
    parser.add_argument("--steps", type=int, default=12, help="Number of optimization steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size for sample/bucket strategies.")
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Approximate padded-token budget per rank for token strategy.",
    )
    parser.add_argument("--min-seq-len", type=int, default=64, help="Minimum sequence length in variable mode.")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length in variable mode.")
    parser.add_argument("--fixed-seq-len", type=int, default=128, help="Sequence length used in fixed mode.")
    parser.add_argument(
        "--outlier-prob",
        type=float,
        default=0.08,
        help="Probability of sampling a max-length outlier sequence in variable mode.",
    )
    parser.add_argument("--hidden-dim", type=int, default=48, help="Transformer hidden size.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-classes", type=int, default=16, help="Number of classification targets.")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Tokenizer vocabulary size.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for dataset generation.")
    parser.add_argument(
        "--skip-first-steps",
        type=int,
        default=2,
        help="Number of warmup steps to exclude from steady-state summaries.",
    )
    parser.add_argument("--summary-json", default=None, help="Optional path to write a rank-aggregated JSON summary.")
    return parser.parse_args()


def setup_dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return dist, rank, world_size, local_rank


def generate_lengths_and_labels(args: argparse.Namespace, world_size: int) -> tuple[list[int], list[int]]:
    total_examples = max(args.steps * args.batch_size * world_size * 8, args.max_tokens_per_batch * world_size // 32)
    rng = random.Random(args.seed)
    lengths: list[int] = []
    labels: list[int] = []

    for _ in range(total_examples):
        if args.length_mode == "fixed":
            seq_len = args.fixed_seq_len
        else:
            log_min = math.log(args.min_seq_len)
            log_max = math.log(args.max_seq_len)
            seq_len = int(round(math.exp(rng.uniform(log_min, log_max))))
            if rng.random() < args.outlier_prob:
                seq_len = args.max_seq_len
            seq_len = max(args.min_seq_len, min(args.max_seq_len, seq_len))

        lengths.append(seq_len)
        labels.append(rng.randrange(args.num_classes))

    return lengths, labels


def chunked(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size) if items[index : index + chunk_size]]


def build_local_batches(
    args: argparse.Namespace,
    lengths: list[int],
    rank: int,
    world_size: int,
) -> list[list[int]]:
    indices = list(range(len(lengths)))
    rng = random.Random(args.seed + 17)
    rng.shuffle(indices)
    local_indices = indices[rank::world_size]

    if args.strategy == "sample":
        return chunked(local_indices, args.batch_size)

    if args.strategy == "bucket":
        local_sorted = sorted(local_indices, key=lengths.__getitem__)
        batches = chunked(local_sorted, args.batch_size)
        rng.shuffle(batches)
        return batches

    local_sorted = sorted(local_indices, key=lengths.__getitem__)
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_len = 0
    for index in local_sorted:
        seq_len = lengths[index]
        next_max_len = max(current_max_len, seq_len)
        projected_padded_tokens = next_max_len * (len(current_batch) + 1)
        if current_batch and projected_padded_tokens > args.max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_max_len = 0
            next_max_len = seq_len
        current_batch.append(index)
        current_max_len = next_max_len

    if current_batch:
        batches.append(current_batch)

    rng.shuffle(batches)
    return batches


def cycle_batches(batches: list[list[int]]):
    while True:
        for batch in batches:
            yield batch


class TinyTransformerClassifier:  # pragma: no cover - exercised via integration runs
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, num_classes: int) -> None:
        import torch

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.module = torch.nn.Sequential()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.qkv = torch.nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.ffn1 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.ffn2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def parameters(self):
        for module in (
            self.embedding,
            self.norm1,
            self.qkv,
            self.proj,
            self.norm2,
            self.ffn1,
            self.ffn2,
            self.classifier,
        ):
            yield from module.parameters()

    def __call__(self, input_ids, lengths):
        import torch
        import torch.nn.functional as F

        x = self.embedding(input_ids)
        batch_size, seq_len, _ = x.shape
        mask = torch.arange(seq_len, device=input_ids.device)[None, :] < lengths[:, None]

        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = self.hidden_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~attn_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        x = residual + self.proj(attended)

        residual = x
        x = self.norm2(x)
        x = self.ffn2(F.gelu(self.ffn1(x)))
        x = residual + x

        x = x.masked_fill(~mask[..., None], 0.0)
        pooled = x.sum(dim=1) / lengths[:, None].clamp_min(1)
        return self.classifier(pooled)


def collate_variable_batch(
    batch_indices: list[int],
    lengths: list[int],
    labels: list[int],
    vocab_size: int,
    seed: int,
):
    import torch

    batch_lengths = [lengths[index] for index in batch_indices]
    padded_len = max(batch_lengths)
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.zeros((len(batch_indices), padded_len), dtype=torch.long)
    targets = torch.tensor([labels[index] for index in batch_indices], dtype=torch.long)

    for row, seq_len in enumerate(batch_lengths):
        input_ids[row, :seq_len] = torch.randint(1, vocab_size, (seq_len,), generator=generator)

    return input_ids, torch.tensor(batch_lengths, dtype=torch.long), targets


def mean_after_skip(values: list[float], skip: int) -> float:
    kept = values[skip:] or values
    return statistics.mean(kept)


def main() -> int:
    import torch
    import torch.distributed as dist_torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    args = parse_args()
    torch.set_num_threads(1)
    dist, rank, world_size, _ = setup_dist()
    torch.manual_seed(args.seed + rank)

    lengths, labels = generate_lengths_and_labels(args, world_size)
    local_batches = build_local_batches(args, lengths, rank, world_size)
    batch_iterator = cycle_batches(local_batches)

    core_model = TinyTransformerClassifier(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    )
    module = torch.nn.Module()
    module.embedding = core_model.embedding
    module.norm1 = core_model.norm1
    module.qkv = core_model.qkv
    module.proj = core_model.proj
    module.norm2 = core_model.norm2
    module.ffn1 = core_model.ffn1
    module.ffn2 = core_model.ffn2
    module.classifier = core_model.classifier

    def forward(input_ids, lengths_tensor):
        return core_model(input_ids, lengths_tensor)

    module.forward = forward  # type: ignore[method-assign]
    model = DDP(module)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    step_times_ms: list[float] = []
    data_wait_ms: list[float] = []
    compute_ms: list[float] = []
    useful_tokens_per_step: list[int] = []
    padded_tokens_per_step: list[int] = []
    sample_counts_per_step: list[int] = []
    max_seq_len_per_step: list[int] = []
    padding_ratio_per_step: list[float] = []

    for step in range(args.steps):
        step_start = time.perf_counter()

        data_start = time.perf_counter()
        batch_indices = next(batch_iterator)
        input_ids, batch_lengths, targets = collate_variable_batch(
            batch_indices,
            lengths,
            labels,
            args.vocab_size,
            seed=args.seed + rank * 10_000 + step,
        )
        data_end = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, batch_lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        step_end = time.perf_counter()

        useful_tokens = int(batch_lengths.sum().item())
        padded_tokens = int(batch_lengths.max().item()) * len(batch_indices)
        padding_ratio = 0.0 if padded_tokens == 0 else 1.0 - useful_tokens / padded_tokens

        step_times_ms.append((step_end - step_start) * 1000.0)
        data_wait_ms.append((data_end - data_start) * 1000.0)
        compute_ms.append((step_end - data_end) * 1000.0)
        useful_tokens_per_step.append(useful_tokens)
        padded_tokens_per_step.append(padded_tokens)
        sample_counts_per_step.append(len(batch_indices))
        max_seq_len_per_step.append(int(batch_lengths.max().item()))
        padding_ratio_per_step.append(padding_ratio)

    local_summary = {
        "rank": rank,
        "world_size": world_size,
        "strategy": args.strategy,
        "length_mode": args.length_mode,
        "batch_size": args.batch_size,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "step_times_ms": step_times_ms,
        "data_wait_ms": data_wait_ms,
        "compute_ms": compute_ms,
        "useful_tokens_per_step": useful_tokens_per_step,
        "padded_tokens_per_step": padded_tokens_per_step,
        "sample_counts_per_step": sample_counts_per_step,
        "max_seq_len_per_step": max_seq_len_per_step,
        "padding_ratio_per_step": padding_ratio_per_step,
        "final_loss": float(loss.item()),
    }

    gathered: list[dict[str, object]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_summary)

    if rank == 0:
        skip = min(args.skip_first_steps, max(args.steps - 1, 0))

        per_rank_step_ms = {
            item["rank"]: mean_after_skip(item["step_times_ms"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_data_ms = {
            item["rank"]: mean_after_skip(item["data_wait_ms"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_compute_ms = {
            item["rank"]: mean_after_skip(item["compute_ms"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_useful_tokens = {
            item["rank"]: mean_after_skip(item["useful_tokens_per_step"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_padded_tokens = {
            item["rank"]: mean_after_skip(item["padded_tokens_per_step"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_padding_ratio = {
            item["rank"]: mean_after_skip(item["padding_ratio_per_step"], skip)  # type: ignore[index]
            for item in gathered
        }
        per_rank_sample_count = {
            item["rank"]: mean_after_skip(item["sample_counts_per_step"], skip)  # type: ignore[index]
            for item in gathered
        }

        per_step_max_time = []
        per_step_min_time = []
        per_step_max_tokens = []
        per_step_min_tokens = []
        per_step_global_useful = []
        per_step_global_padded = []

        for step in range(skip, args.steps):
            time_values = [item["step_times_ms"][step] for item in gathered]  # type: ignore[index]
            token_values = [item["useful_tokens_per_step"][step] for item in gathered]  # type: ignore[index]
            padded_values = [item["padded_tokens_per_step"][step] for item in gathered]  # type: ignore[index]
            per_step_max_time.append(max(time_values))
            per_step_min_time.append(min(time_values))
            per_step_max_tokens.append(max(token_values))
            per_step_min_tokens.append(min(token_values))
            per_step_global_useful.append(sum(token_values))
            per_step_global_padded.append(sum(padded_values))

        slowest_rank = max(per_rank_step_ms, key=per_rank_step_ms.get)
        slowest_step_ms = per_rank_step_ms[slowest_rank]
        mean_global_useful_tokens = statistics.mean(per_step_global_useful)
        mean_global_padded_tokens = statistics.mean(per_step_global_padded)
        useful_tokens_per_s = mean_global_useful_tokens / (slowest_step_ms / 1000.0)
        padded_tokens_per_s = mean_global_padded_tokens / (slowest_step_ms / 1000.0)
        average_padding_ratio = 0.0 if mean_global_padded_tokens == 0 else 1.0 - mean_global_useful_tokens / mean_global_padded_tokens
        average_rank_time_skew_ms = statistics.mean(m - n for m, n in zip(per_step_max_time, per_step_min_time))
        average_rank_token_spread = statistics.mean(m - n for m, n in zip(per_step_max_tokens, per_step_min_tokens))

        print("App: transformer_token_skew_lab")
        print(f"Strategy: {args.strategy}")
        print(f"Length mode: {args.length_mode}")
        print(f"World size: {world_size}")
        if args.strategy == "token":
            print(f"Per-rank token budget: {args.max_tokens_per_batch}")
        else:
            print(f"Per-rank sample batch size: {args.batch_size}")
        print(f"Slowest steady-state rank: {slowest_rank}")
        print(f"Slowest steady-state step time (ms): {slowest_step_ms:.2f}")
        print(f"Global useful token throughput (tokens/s): {useful_tokens_per_s:.2f}")
        print(f"Global padded token throughput (tokens/s): {padded_tokens_per_s:.2f}")
        print(f"Average padding ratio: {average_padding_ratio:.3f}")
        print(f"Average rank time skew (ms): {average_rank_time_skew_ms:.2f}")
        print(f"Average rank token spread per step: {average_rank_token_spread:.1f}")
        print("")
        for rank_id in sorted(per_rank_step_ms):
            print(
                f"rank={rank_id} step_ms={per_rank_step_ms[rank_id]:.2f} "
                f"data_ms={per_rank_data_ms[rank_id]:.2f} compute_ms={per_rank_compute_ms[rank_id]:.2f} "
                f"useful_tokens={per_rank_useful_tokens[rank_id]:.1f} padded_tokens={per_rank_padded_tokens[rank_id]:.1f} "
                f"padding_ratio={per_rank_padding_ratio[rank_id]:.3f} samples={per_rank_sample_count[rank_id]:.1f}"
            )

        if args.summary_json:
            summary = {
                "strategy": args.strategy,
                "length_mode": args.length_mode,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "max_tokens_per_batch": args.max_tokens_per_batch,
                "slowest_rank": slowest_rank,
                "slowest_step_ms": slowest_step_ms,
                "global_useful_tokens_per_s": useful_tokens_per_s,
                "global_padded_tokens_per_s": padded_tokens_per_s,
                "average_padding_ratio": average_padding_ratio,
                "average_rank_time_skew_ms": average_rank_time_skew_ms,
                "average_rank_token_spread": average_rank_token_spread,
                "per_rank_step_ms": per_rank_step_ms,
                "per_rank_data_ms": per_rank_data_ms,
                "per_rank_compute_ms": per_rank_compute_ms,
                "per_rank_useful_tokens": per_rank_useful_tokens,
                "per_rank_padded_tokens": per_rank_padded_tokens,
                "per_rank_padding_ratio": per_rank_padding_ratio,
                "per_rank_sample_count": per_rank_sample_count,
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
