= 前言

== 这本书是写给谁的

这本书面向已经写过 Transformer、想把 *Sparse Mixture-of-Experts (MoE)* 从"看过论文"读到"能自己写、能上分布式、能在面试里被系统性追问"那种深度的读者。

在阅读本书之前，你应该：

- 会用 PyTorch，读得懂 `nn.Linear` / `nn.Module`。
- 熟悉标准 Transformer block：Attention + FFN + LayerNorm + residual。
- 对 `torch.softmax` / `torch.topk` / `torch.where` / `index_add_` 这些张量操作有基本认知。
- 对分布式训练的 DP / TP / PP 三种并行有*一个概念*——不熟也没关系，第 8 章会把 EP（Expert Parallel）从零推一遍。

如果 Transformer FFN 层完全没接触过，建议先读姊妹卷《CUDA Kernel 优化实战》第 6 章 (MLP)，回来再读本书。

== 这本书讲什么

MoE 表面看是"把一个大 FFN 拆成 E 份，每 token 只走 K 份"，但真正的复杂度在*工程*：

- Router 怎么设计？softmax + topk 就够了吗？
- 每个 batch 里 M 个专家的 token 数是变量，怎么做 batched GEMM？
- 单机 8 卡时 experts 怎么分？跨机时 all-to-all 怎么 overlap？
- 训练不稳定怎么办？专家坍缩怎么办？

每一章围绕一个具体主题展开——从零基础的路由直觉，到 grouped GEMM 的实现，再到 DeepSeek-V3 那种规模的分布式训练。每章都会讲清楚：

- 这一步*为什么需要*（对比朴素方案的缺陷）。
- 具体实现里*每个 tensor 的 shape 变化*（配 tensor 可视化图）。
- 什么场景*用不上*（不是所有优化都总有用）。
- 面试里常见的追问。

== 怎么读

- *顺序读*：章节间有依赖。第 5 章 code walkthrough 用到第 3、4 章的路由 & dispatch 概念；第 7 章单机性能优化建立在第 4 章的 dispatch 范式上；第 8 章分布式建立在第 7 章的 grouped GEMM 之上。
- *配代码读*：所有代码来自本仓库 `python/pytorch/test_moe.py`，可以直接跑：

  ```bash
  python python/pytorch/test_moe.py
  ```

  第 5 章会逐行讲解这份代码。

- *把面试考点框读进去*：每章末尾用紫色框标出常见面试追问，答案在正文里都能找到依据。附录 D 汇总了 20 道系统面试题（含参考答案）。

== 记号约定

- $B$, $S$, $H$：batch size / sequence length / hidden dim
- $N = B times S$：flatten 之后的 token 总数
- $E$：专家总数（num_experts）
- $K$：top-k，每个 token 激活的专家数
- $M_e$：路由到专家 $e$ 的 token 数（负载不均，$sum_e M_e = N K$）
- $I$：FFN 内的 intermediate dim（通常 $I approx 4H$）
- $C$：capacity，每个专家在 batch 内允许接收的最大 token 数
- $"EP"$ / $"TP"$ / $"DP"$ / $"PP"$：expert / tensor / data / pipeline parallel 的世界大小
- 张量 shape 用 `(a, b, c)` 标注，如 `expert_indices: (N, K)`

性能数字除非注明，都在 A100 80GB SXM4 / H100 SXM5 上给出量级估算，不承诺精确复现。生产系统请以 `nsys` / `ncu` 或框架自带 profiler 为准。

== 与相关材料的关系

- 本书假设的最小 PyTorch 实现在 `python/pytorch/test_moe.py`，是理解 MoE 的"读物"，*不是*生产实现。
- 生产实现细节参考 Megatron-LM MoE、DeepSpeed-MoE、Tutel、Megablocks、vLLM `fused_moe`。本书讲*为什么这些框架要这样做*，源码级细节请直接读代码。
- Mixtral、DeepSeek-V3 等具体模型的 checkpoint 分析在附录 A。

好，翻页开始。
