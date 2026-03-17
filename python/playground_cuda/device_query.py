"""Minimal CUDA device inspection using PyTorch."""

from __future__ import annotations


def main() -> int:
    import torch

    print("App: python_device_query")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        return 0

    tensor = torch.rand((4, 4), device="cuda", dtype=torch.float32)
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor mean: {tensor.mean().item()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
