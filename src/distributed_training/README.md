# distributed_training — teaching-level implementations

Companion code for `books/distributed-training/` (the interview handbook).
Each script implements ONE concept from the book at the smallest scale
that still exhibits the correct communication pattern, then asserts
numerical equivalence to a single-rank reference.

## Design constraints

- **Only `torch.distributed` primitives.** No FSDP, no Megatron, no
  DeepSpeed. Every collective is spelled out.
- **CPU-friendly.** Where possible (chapters 1–7, 10), demos default to
  the gloo backend and run without GPUs. Demos 8, 9, 11 need NCCL and
  are skipped on CPU.
- **Self-verifying.** Every demo ends with `assert torch.allclose(...)`
  against a single-rank reference and prints `✓` on success.
- **Small.** Aim for ≤ 200 lines per demo. Prefer clarity over speed.

## Layout

| File                          | Concept                                         | Book §  |
|-------------------------------|-------------------------------------------------|---------|
| `01_collectives.py`           | AR/AG/RS/A2A + hand-written Ring AllReduce      | Ch. 1   |
| `02_ddp_from_scratch.py`      | DDP: bucket + backward hook + async AR          | Ch. 3   |
| `03_zero.py`                  | ZeRO-1 / -2 / -3 (FSDP-lite Linear)             | Ch. 4   |
| `04_tp_column_row.py`         | Column-/Row-parallel Linear, TP FFN & Attention | Ch. 5   |
| `05_sp_with_tp.py`            | Sequence Parallel combined with TP              | Ch. 5   |
| `06_pp_1f1b.py`               | GPipe + 1F1B pipeline schedulers                | Ch. 6   |
| `07_ring_attention.py`        | Ring Attention with Flash-style LSE merge       | Ch. 7   |
| `08_ulysses_attention.py`     | DeepSpeed-Ulysses (all-to-all on head dim)      | Ch. 7   |
| `09_ep_moe.py`                | Expert Parallel MoE dispatch / combine          | Ch. 8   |
| `10_seq_packing.py`           | FFD packing + varlen attention                  | Ch. 12  |
| `11_overlap_2stream.py`       | Compute/comm overlap via CUDA streams           | Ch. 11  |
| `12_optimizers.py`            | AdamW / Lion / LAMB / Muon + torch.optim parity | Ch. P2  |
| `13_lr_schedules.py`          | warmup+cosine / WSD / WSD-rewarm                | Ch. P3  |
| `14_nan_reproducer.py`        | 6 canonical NaN pathologies + fixes             | Ch. P4  |
| `15_training_monitor.py`      | grad_norm / activation / weight-norm alerts     | Ch. P4  |
| `16_agent_rollout_sched.py`   | agentic rollout: sync vs async vs partial       | Ch. 15  |
| `17_rlvr_verifier.py`         | RLVR verifier pool: timeout / cache / capacity  | Ch. 15  |
| `18_traj_masking.py`          | trajectory loss mask, GRPO degeneracy, ratio    | Ch. 15  |
| `19_cp_loss_and_metrics.py`   | CP cross-token reductions: loss / aux / pooling | Ch. 7   |
| `20_cp_dataloader_halo.py`    | CP data path: zigzag, positions, mask, halo     | Ch. 7   |
| `21_pp_from_scratch.py`       | PP engine: autograd boundary, deadlock, 4 schedules | Ch. 6 |
| `22_pp_schedule_sim.py`       | PP cost: measured bubble, Gantt, activation peak | Ch. 6  |
| `23_pp_integration.py`        | PP wiring: rank layout, partitioning, tied weights | Ch. 6 |

`common/pipeline.py` holds the pipeline engine the three PP demos share: the
schedule generators, the B/W split, the P2P layer and the ~40-line interpreter
that executes any of them. Read it before the demos.

`common/` holds shared helpers: `dist_utils.py` (init/rank/device),
`toy_model.py` (mini Transformer for verification), `verify.py`
(`assert_close`).

## Running

```bash
cd src/distributed_training

# CPU (gloo) — works on any machine
make demo-01
make demo-02
make demo-03
make demo-04
make demo-05
make demo-06
make demo-07
make demo-10        # single-process, no torchrun

# Training-fundamentals demos (single-process, no torchrun)
make demo-12        # optimizers
make demo-13        # LR schedules
make demo-14        # NaN reproducer
make demo-15        # training monitor

# GPU (NCCL) only
NPROC=2 make demo-08
NPROC=2 make demo-09
NPROC=2 make demo-11

# Everything CPU-friendly at once
make all
```

Change world size:

```bash
NPROC=8 make demo-06
```

## What "verified" means

Every parallel-execution demo:

1. Builds a reference computation that could run on a single rank.
2. Sets identical initial weights on the reference and the parallel model
   (via `broadcast` or explicit copy).
3. Runs one forward (and often one backward) on both.
4. Reconstructs the parallel output from all ranks (`all_gather`) and
   calls `torch.allclose` against the reference with `rtol/atol = 1e-4`.

If a demo prints `✓` at the end without an `AssertionError`, the
communication pattern is numerically correct.

## What this is *not*

- Not production. No FP8, no async prefetch, no bucket-fusion, no NCCL
  tuning. See Megatron-Core / TorchTitan / FSDP2 for that.
- Not exhaustive. USP, DualPipe, ZeroBubble, DeepEP, MegaScale-scale
  fault tolerance are described in the book but not implemented here —
  their code either exists upstream or is far beyond a teaching demo.
- Not benchmarked. Only `11_overlap_2stream.py` prints timings; the
  others prioritize correctness proofs.
