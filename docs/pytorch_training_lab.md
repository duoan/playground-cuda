# PyTorch Training Lab

This lab gives you one training script that can deliberately become:

- `baseline`: a healthy-enough reference run
- `data`: input pipeline bound
- `torch`: eager/Python overhead bound
- `kernel`: GPU kernel and memory-system bound

The script lives at `python/playground_cuda/training_lab.py`.

## Step 1: Establish a baseline

Run a short job without any profiler first.

```bash
uv run python -m playground_cuda.training_lab --mode baseline --steps 20
```

What to record:

- average step time
- GPU utilization from `nvidia-smi dmon` or `watch -n 0.5 nvidia-smi`
- whether the run is clearly CPU-limited or GPU-limited

## Step 2: Use torch.profiler first

`torch.profiler` is the fastest way to answer: "Is the time going into data loading, Python/operator overhead, or CUDA kernels?"

```bash
uv run python -m playground_cuda.training_lab \
  --mode baseline \
  --steps 12 \
  --profile torch \
  --trace-dir traces/torch_baseline
```

What to look for:

- If `data_loader` dominates and CUDA time is low, the job is input bound.
- If many tiny `aten::` ops dominate self CPU time, the job is PyTorch/eager overhead bound.
- If CUDA time is concentrated in a few heavy ops, move to Nsight tools and inspect those kernels.

Open the trace in TensorBoard if you want a timeline. This is optional and requires TensorBoard to be installed in the environment.

```bash
uv run python -m tensorboard.main --logdir traces
```

## Step 3: Reproduce a data bottleneck

```bash
uv run python -m playground_cuda.training_lab \
  --mode data \
  --steps 12 \
  --profile torch \
  --trace-dir traces/torch_data
```

Typical signals:

- `data_loader` region is large
- GPU kernels have gaps between steps
- step time improves if you increase `--num-workers`

Things to try:

- increase `--num-workers`
- reduce `--sleep-ms`
- reduce `--cpu-transform-depth`
- keep `pin_memory=True` and `.to(..., non_blocking=True)`

## Step 4: Reproduce a PyTorch overhead bottleneck

```bash
uv run python -m playground_cuda.training_lab \
  --mode torch \
  --steps 12 \
  --profile torch \
  --trace-dir traces/torch_eager
```

Typical signals:

- many tiny `aten::add`, `aten::relu`, `aten::sigmoid` style ops
- high self CPU time even though GPU kernels are small
- lots of kernel launches with little work per launch

Things to try:

- `torch.compile`
- fuse or rewrite chains of tiny eager ops
- increase batch size so each launch does more work

Compare eager vs compiled:

```bash
uv run python -m playground_cuda.training_lab --mode torch --steps 20
uv run python -m playground_cuda.training_lab --mode torch --steps 20 --compile
```

## Step 5: Use Nsight Systems for the whole-step timeline

Use `nsys` after `torch.profiler` has told you which category the problem belongs to.

```bash
nsys profile \
  --trace cuda,nvtx,osrt \
  --sample none \
  --output reports/training_torch \
  .venv/bin/python -m playground_cuda.training_lab --mode torch --steps 10
```

The script emits named regions:

- `data_loader`
- `h2d`
- `forward`
- `backward`
- `optimizer`

What to look for in Nsight Systems:

- long CPU gaps before GPU work
- host-to-device copies not overlapping compute
- too many tiny kernels in `forward`
- whether `backward` or `optimizer` dominates the step

## Step 6: Use Nsight Compute on a hot kernel

Use `ncu` only after you have already identified a hot kernel family from `torch.profiler` or `nsys`.

Kernel-focused example:

```bash
ncu --target-processes all \
  --kernel-name-base demangled \
  -k "regex:.*vectorized_elementwise_kernel.*" \
  -c 1 \
  .venv/bin/python -m playground_cuda.training_lab --mode kernel --steps 4
```

What to look for:

- `Memory Throughput` vs `Compute Throughput`
- occupancy
- launch size (`Grid Size`, `Block Size`, `Waves Per SM`)
- whether the kernel is memory bound, compute bound, or launch bound

Common interpretations:

- low compute + high memory: memory-bound pointwise kernels
- low occupancy + tiny grid: launch configuration or problem size is too small
- many similar pointwise kernels: candidate for fusion or `torch.compile`

## A practical decision tree

1. Start with plain timing.
2. Use `torch.profiler` to classify the bottleneck.
3. If the problem is data-side, optimize the input pipeline first.
4. If the problem is eager/Python-side, reduce small ops or try `torch.compile`.
5. If the problem is inside a hot CUDA kernel, move to `nsys` then `ncu`.

## Good next experiments

- Change only one variable at a time.
- Compare `--mode torch` with and without `--compile`.
- Compare `--mode data --num-workers 0` vs `--num-workers 4`.
- Compare `--mode kernel --amp none` vs `--amp bf16`.
