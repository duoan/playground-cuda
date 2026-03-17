# DDP Experiments on a One-GPU Machine

If you only have one GPU, you still cannot reproduce the full DDP performance story, but you can reproduce a surprisingly large part of the reasoning workflow.

The key idea is to separate:

- what truly requires multiple GPUs and NCCL
- what can already be learned from multiple ranks on CPU using `gloo`

This repo now includes a small CPU-based distributed lab at `python/playground_cuda/ddp_lab.py`.
It also includes a batch experiment runner at `python/playground_cuda/ddp_lab_blog_experiments.py` that regenerates the charts and timing tables used by the DDP blog post.

## What You Can Learn on One GPU

With a one-GPU machine, you can still learn these DDP concepts well:

- the slowest rank determines step time
- data skew on one rank stalls all ranks
- barriers and host-side synchronization create visible stalls
- extra communication volume can dominate when per-rank work is small
- global throughput is the right metric, not one-rank step time alone

## What You Cannot Reproduce Faithfully

These need at least two real GPUs to be meaningful:

- NCCL bandwidth and topology effects
- overlap between CUDA backward kernels and NCCL all-reduce on different streams
- NVLink vs PCIe behavior
- true multi-GPU tensor-core scaling
- rank-local GPU memory pressure differences

So the honest framing is:

- this lab teaches the DDP mental model
- it does not replace real multi-GPU measurement

## The Lab Modes

`ddp_lab.py` supports four modes:

- `baseline`: balanced multi-rank CPU training
- `skew`: one rank is intentionally slower in the input path
- `comm`: adds a large extra `all_reduce` every step
- `barrier`: injects explicit synchronization stalls

## How to Run It

All examples below work on your machine because they use CPU `gloo`.

### 1. Baseline

```bash
torchrun --standalone --nproc_per_node=4 \
  -m playground_cuda.ddp_lab \
  --mode baseline \
  --steps 20
```

What to look for:

- per-rank `step_ms` should be close
- rank skew should be small

### 2. Simulate a Straggler Rank

```bash
torchrun --standalone --nproc_per_node=4 \
  -m playground_cuda.ddp_lab \
  --mode skew \
  --steps 20 \
  --sleep-ms 20 \
  --slow-rank 0
```

What to look for:

- rank 0 `data_ms` becomes much larger
- slowest rank dominates global throughput
- average rank skew increases

This is the easiest way to internalize the most important DDP rule:

> one slow rank is enough to make the whole job look slow

### 3. Simulate Communication Pressure

```bash
torchrun --standalone --nproc_per_node=4 \
  -m playground_cuda.ddp_lab \
  --mode comm \
  --steps 20 \
  --comm-mb 64
```

What to look for:

- all ranks slow down more uniformly
- `compute_ms` does not explain the regression by itself

This approximates the situation where communication becomes too expensive relative to per-rank compute.

### 4. Simulate Bad Synchronization

```bash
torchrun --standalone --nproc_per_node=4 \
  -m playground_cuda.ddp_lab \
  --mode barrier \
  --steps 20 \
  --barrier-every 1
```

What to look for:

- step time rises for everyone
- rank skew may stay modest, but throughput still drops

This is a good stand-in for the class of problems caused by unnecessary synchronization in real DDP jobs.

## Saving Structured Results

You can also save a JSON summary:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m playground_cuda.ddp_lab \
  --mode skew \
  --steps 20 \
  --summary-json reports/ddp_skew.json
```

The summary includes:

- slowest rank
- slowest steady-state step time
- global throughput
- average rank skew
- per-rank step/data/compute timing

## Regenerating the Full Blog Dataset

If you want the whole experiment pack instead of one run at a time, use:

```bash
uv run python -m playground_cuda.ddp_lab_blog_experiments
```

That command writes:

- `reports/ddp_lab_blog/timings.csv`
- `reports/ddp_lab_blog/results.json`
- `reports/ddp_lab_blog/charts/*.svg`
- per-case raw logs under `reports/ddp_lab_blog/*.log`

## How This Connects Back to Real DDP

The mapping from this CPU lab to real multi-GPU DDP is:

- `skew` mode teaches you how rank-local data delays stall everyone
- `comm` mode teaches you how too much synchronization work hurts scaling
- `barrier` mode teaches you why unnecessary coordination destroys throughput

For transformer training, a very common real equivalent of `skew` mode is variable sequence length. Even if every rank sees the same number of examples, one rank may receive a batch with far more total tokens, which makes collation, transfer, attention, and backward all take longer.

Then, when you move to a real multi-GPU machine:

1. you already know what rank skew looks like conceptually
2. you already know why the slowest rank matters
3. you are ready to use `torch.profiler` and `nsys` to find the real source of the stall

## Suggested Learning Order for Your Machine

Because you only have one GPU, I would use this order:

1. Run the single-GPU training lab:

```bash
uv run python -m playground_cuda.training_lab --mode baseline --steps 20
```

2. Learn data / torch / kernel bottlenecks from the existing training lab.

3. Run the CPU DDP lab:

```bash
torchrun --standalone --nproc_per_node=4 -m playground_cuda.ddp_lab --mode baseline
torchrun --standalone --nproc_per_node=4 -m playground_cuda.ddp_lab --mode skew --sleep-ms 20
torchrun --standalone --nproc_per_node=4 -m playground_cuda.ddp_lab --mode comm --comm-mb 64
torchrun --standalone --nproc_per_node=4 -m playground_cuda.ddp_lab --mode barrier --barrier-every 1
```

4. Read the DDP blog with those experiments in mind:

- `docs/pytorch_ddp_performance_tuning_blog.md`

That sequence will give you most of the intuition you need before you ever sit down on a multi-GPU box.
