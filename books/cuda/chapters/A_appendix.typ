= 附录 A：参考资料

== 硬件参考卡（A100 80GB SXM4，本书实测环境）

- Compute capability: 8.0
- SM 数量: 108
- 每 SM 最大 warp 数: 64（2048 threads）
- 每 SM 寄存器: 65,536 × 32-bit
- 每 SM shared memory: 164 KB（配置 L1 = 28 KB 时；总 192 KB）
- L2 cache: 40 MB
- HBM 带宽: 2039 GB/s
- FP32 峰值: 19.5 TFLOPS
- FP16 tensor core: 312 TFLOPS
- BF16 tensor core: 312 TFLOPS
- TF32 tensor core: 156 TFLOPS

== H100 SXM5（Hopper，仅供 Ch.10 参考）

- Compute capability: 9.0
- SM 数量: 132
- HBM3 带宽: 3350 GB/s
- FP8 tensor core: 1979 TFLOPS
- FP16 tensor core: 989 TFLOPS
- TMA (Tensor Memory Accelerator)
- Thread Block Cluster + Distributed Shared Memory

== 常用 nvcc flag

- `-arch=sm_80`：目标 CC。
- `-lineinfo`：让 `ncu` 能对到源码行。
- `--ptxas-options=-v`：打印每个 kernel 的寄存器 / smem / spill 用量。
- `-Xptxas -dlcm=ca`：L1 cache 策略（default `ca` = cache all）。
- `-maxrregcount=N`：全局限制寄存器数（会强迫 spill）。
- `--use_fast_math`：让 SFU/MUFU 用近似路径（例如 `sinf`、`__expf`）。
- `-Xcompiler -fPIC`：需要生成 shared library 时。

== 参考资料

- CUDA C++ Programming Guide (latest)
- CUDA C++ Best Practices Guide
- PTX ISA
- Nsight Compute Kernel Profiling Guide（也可以看附录 B）
- Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024
- CUTLASS 文档与源码
- Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"
- Lei Mao 的 CUDA 系列 blog
