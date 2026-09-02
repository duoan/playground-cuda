= 前言

== 这本书是写给谁的

这本书面向已经会写基本 CUDA 代码、希望把 kernel 优化能力打磨到*能在面试里被系统性追问*那种深度的读者。

在阅读本书之前，你应该：

- 会 C++，读得懂指针、模板、`reinterpret_cast`。
- 知道 `__global__` / `__device__` 的区别，会用 `cudaMalloc` / `cudaMemcpy`。
- 对 grid / block / thread 的基本层级*有个印象*——不熟也没关系，下一章「CUDA 基本概念速查」会把这些术语和索引推导讲清楚，之后正文里遇到卡壳可以随时翻回去。

如果 CUDA API 完全没接触过，建议先跑通一个 `cudaMalloc + kernel launch + cudaMemcpy` 的 hello world 再回来读，会更有画面感。

== 这本书讲什么

每一章围绕*一个 kernel* 展开，从最朴素的实现开始，一步步爬优化 ladder，直到接近硬件峰值。ladder 上的每一层都会讲清楚：

- 上一版为什么慢（用具体的性能瓶颈概念：memory-bound / compute-bound / occupancy / bank conflict / warp divergence 等）。
- 这一版做了什么改变，改变对应硬件上的什么行为。
- 什么时候用不上这个技巧，或者会变成过度优化。

高级章节会涉及：warp shuffle、shared memory bank conflict 与 swizzle、`cp.async` 异步拷贝、tensor core（`wmma` / `mma.sync`）、Flash-Attention 的 online softmax 与分块推导、FP8 / MXFP4 等新数据类型。

== 怎么读

- *顺序读*：ladder 之间有依赖。第 4 章 matmul 用到第 2 章 reduction 的 shuffle 技巧；第 8 章 flash-attention 用到第 3 章 softmax 的数值稳定性推导。
- *配代码读*：每一章的代码都在仓库 `src/cuda/` 下，`make build/0X_xxx && ./build/0X_xxx` 就能跑。文中引用的 kernel 版本都对应源代码里的具体函数。
- *把面试考点框读进去*：每章末尾会用紫色框标出这个 kernel 常见的面试追问，答案在正文里都能找到依据。

== 记号约定

- $B$, $S$, $H$：batch / sequence / hidden 三个维度。
- $"bx", "by", "tx", "ty"$：`blockIdx.x/y`, `threadIdx.x/y`。
- warp = 32 个 lane。lane id = `threadIdx.x % 32`。
- "row" 通常指最内层维度上的一行（例如 layernorm 里一个 (batch, seq) 位置的 $H$ 维向量）。
- 性能数字（GB/s、TFLOPS）除非注明，都在 A100 40GB PCIe / SM80 上测得。
