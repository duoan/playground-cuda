# playground-cuda

A CUDA playground for learning GPU programming from both C++/libtorch and Python/PyTorch.

## Layout

- `src/`: each `.cu` file is its own runnable app
- `examples/`: each `.cu` file is its own runnable app
- `python/`: installable Python package with matching PyTorch examples
- `libtorch/`: vendored libtorch distribution downloaded on first configure

## Common commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
cmake --build build --target run
```

```bash
uv sync
uv run python -m playground_cuda.device_query
uv run python -m playground_cuda.vector_add
uv run python -m playground_cuda.training_lab --mode baseline --steps 20
```

## Notes

- The C++ side uses vendored `libtorch`, while the Python side uses the `torch` package from the `uv`-managed environment in `.venv/`.
- The base Python project is intentionally lightweight. Add PyTorch to the environment with `uv` using the wheel/index that matches your CUDA setup before running the GPU examples.
- The Python training lab lives in `python/playground_cuda/training_lab.py` and is designed for learning how to diagnose data, PyTorch, and kernel bottlenecks with `torch.profiler`, `nsys`, and `ncu`.
- Any `src/*.cu` or `examples/*.cu` file becomes a build target with the same basename.
- Put kernels directly inside the example `.cu` that uses them.
- Use the `Debug Active CUDA File` launch configuration in VS Code to run the currently open `.cu` file with `cuda-gdb`.
- Use `uv sync` after pulling Python changes so the local `.venv/` stays in sync with `pyproject.toml` and `uv.lock`.
