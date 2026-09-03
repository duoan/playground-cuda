#import "../template.typ": *

= 附录 C：CUDA 性能优化思路总结

这是一份跨章节的方法论 recap——把书里 10 章零散讲的优化技巧收成一份"该按什么顺序想问题"的 checklist，同时给出每个技巧的*触发条件*和*量化收益*。

面试里被问"你怎么优化一个 kernel"，最好的答案是照着这个流程讲。

== 第 0 步：定性 —— 定位瓶颈

在写任何优化之前，先回答一个问题：*这个 kernel 是 memory-bound 还是 compute-bound？*

方法：算 arithmetic intensity (AI)。

$ "AI" = frac("useful FLOPs per launch", "bytes moved between HBM and SM per launch") $

和硬件 ridge point 比较：

- A100 FP32：$"ridge" = 19.5 "TFLOPS" / 2.04 "TB/s" = 13 "FLOP/B"$
- A100 tensor core FP16：$312 / 1.5 = 208 "FLOP/B"$
- H100 tensor core FP8：$~2000 / 3.3 approx 606 "FLOP/B"$

规则：

- AI $<<$ ridge → memory-bound（vector add, reduce, softmax, layernorm）
- AI $>>$ ridge → compute-bound（matmul with large tile, MLP prefill）
- AI $approx$ ridge → mixed，两条路都能优化

`ncu --set roofline` 会自动把 kernel 画到 roofline 图上，一眼看出属于哪种。

#insight[
  瓶颈定位错，后面所有优化都白做。memory-bound kernel 上加 tensor core 一点用没有；compute-bound kernel 上做 float4 也基本没用。
]

== Memory-bound 优化路径

从"能压满 HBM 带宽"这个终极目标出发倒推：

=== 第 1 步：合并访问 (coalesced access)

*触发条件*：任何全局内存读写。

*怎么做*：让同一个 warp 里 32 个 lane 访问连续的 128 字节。用 `blockDim.x * gridDim.x` 的 stride，而不是每 thread 处理连续 K 个元素。

*收益*：破坏合并访问会让带宽掉到 1/32。做对了是"及格线"，没什么加分。

*本书出现位置*：Ch.1 vector_add naive、Ch.4 matmul 的 B 矩阵访问模式。

=== 第 2 步：向量化 load/store

*触发条件*：kernel 已合并访问但 HBM % < 80%。

*怎么做*：`float4` / `int4` / `half8`，一次读写 128 bit（`LDG.E.128` / `STG.E.128`）。要求 16 字节对齐；`cudaMalloc` 保证数组头对齐。

*收益*：
- HBM 已经打满时：0-5%（vector add 就是这个情况）
- HBM 还没满且瓶颈是 MSHR 或 fetch pipeline：20-50%

*本书出现位置*：Ch.1 v4/v5、Ch.3 softmax vectorized、Ch.5 layernorm vectorized。

=== 第 3 步：Occupancy 检查

*触发条件*：ncu 报告 `sm__warps_active` < 50% 或 `long_scoreboard` stall 高。

*怎么做*：
- 减少每 thread 寄存器（用 `__launch_bounds__` 或 `maxrregcount` 强制）
- 减少 shared memory per block
- 调 block size 让 SM 放得下更多 block

*收益*：让 SM 上驻留更多 warp 来掩盖 memory latency。memory-bound kernel 上从 25% occ 涨到 50% 常见提升 30%+。但 occupancy > 50% 后收益递减，别死追 100%。

*本书出现位置*：Ch.4 matmul（寄存器压力 vs occupancy 权衡）。

=== 第 4 步：Grid-stride / block-per-row / warp-per-row 的选择

*触发条件*：kernel 的问题结构（vector / batch of rows / matrix）决定分工。

*怎么做*：
- 数据 1D 且规模 >> SM 数量：*grid-stride*
- 每行独立且 H 中等（$<= 1024$）：*block-per-row*，用 shared memory reduction
- 每行独立且 H 小（$<= 32$）：*warp-per-row*，用 warp shuffle

*收益*：分工选对 vs 选错常见 2-10x 差距（Ch.2 reduce naive vs warp shuffle）。

*本书出现位置*：Ch.2 reduce、Ch.3 softmax、Ch.5 layernorm。

=== 第 5 步：Warp shuffle 替代 shared memory reduction

*触发条件*：block 内需要 reduction，且规模 $<= 32$（warp 大小）。

*怎么做*：`__shfl_down_sync` / `__shfl_xor_sync`，寄存器内部直接交换，跳过 shared memory。

*收益*：
- 少一次 smem 写 + 一次 smem 读
- 少一次 `__syncthreads`
- 常见 20-40% 的 reduction 阶段加速

*硬件*：Volta 起，shuffle 走 register file crossbar，不占 smem port。

*本书出现位置*：Ch.2 v6、Ch.3 softmax warp、Ch.5 layernorm warp。

== Compute-bound 优化路径

=== 第 1 步：Tiling → 提升 AI

*触发条件*：AI 太低（例如 naive matmul 的 0.25 FLOP/B）。

*怎么做*：把数据分块 (tile) 载入 shared memory / registers，让一个 tile 被复用多次。

- BM×BK 的 A tile + BK×BN 的 B tile，一个 CTA 计算 BM×BN 的输出
- AI 提升到 $frac(2 "BM BN BK", 2 "BK (BM+BN)") = frac("BM BN", "BM+BN")$
- BM=BN=128 时 AI = 64 FLOP/B，远超 ridge → 转为 compute-bound

*收益*：本书 Ch.4 matmul naive → tiled 加速 30-100x。

*本书出现位置*：Ch.4 matmul、Ch.8-10 flash-attention。

=== 第 2 步：Register tiling / thread tiling

*触发条件*：shared memory tile 已存在，但 tensor core 或 CUDA FMA 用率仍低。

*怎么做*：让一个 thread 处理 TM×TN 个输出元素，中间结果全在寄存器；shared memory tile 只被读*一次到寄存器*，然后计算完全从寄存器出。

*收益*：进一步减少 smem load，让 FFMA/HMMA 流水线占满。Ch.4 v4 中让 tensor pipe 从 20% 涨到 60%+。

=== 第 3 步：Tensor Core (WMMA / mma.sync)

*触发条件*：数据类型是 FP16 / BF16 / TF32 / FP8 / INT8。

*怎么做*：
- Ampere 起：`mma.sync.aligned.m16n8k16` 或 CUTLASS 抽象
- Hopper 起：WGMMA (`wgmma.mma_async`)，异步
- 注意 fragment layout（`ldmatrix` 指令搬 smem → mma 的 register layout）

*收益*：FP16 vs FP32 tensor core throughput 差 16x（A100：312 vs 19.5 TFLOPS）。

*本书出现位置*：Ch.4 讨论，Ch.10 flash-attention v3 详细展开。

=== 第 4 步：Async copy (cp.async / TMA)

*触发条件*：数据搬运和计算可以重叠。

*怎么做*：
- Ampere: `cp.async.cg.shared.global` 直接从 HBM 异步搬到 shared memory，不经过 register。配 `cp.async.commit_group` + `cp.async.wait_group` 做流水线。
- Hopper: TMA (`cp.async.bulk.tensor`)，硬件独立单元管，支持多维 tensor。

*收益*：GEMM 里 double-buffered pipeline 让 memory 完全和 compute 重叠。Ch.4 pipeline kernel 20-30% 加速。

*本书出现位置*：Ch.4 v5、Ch.10 flash-attention v3。

=== 第 5 步：Warp specialization (producer-consumer)

*触发条件*：Hopper 硬件，且 kernel 里有明显的"搬运 vs 计算"两阶段。

*怎么做*：一部分 warp 专门做 TMA（producer），另一部分做 wgmma（consumer），用 mbarrier 同步。

*收益*：Flash-Attention v3 相比 v2 加速 1.5-2x（H100 FP16）。

*本书出现位置*：Ch.10 flash-attention v3。

== 通用路径：Kernel fusion

*触发条件*：多个 kernel 依次调用，中间结果只被下一个 kernel 用（不 export 出去）。

*怎么做*：
- Elementwise fusion：几个 pointwise op 合成一个 kernel（PyTorch `torch.compile` 自动）
- Epilogue fusion：GEMM 出结果后立刻在寄存器里做 bias / activation / scale
- Full fusion：整段 attention 合成一个 kernel（flash-attention）

*收益*：
- 省 kernel launch overhead：每次 ~5 μs
- 省中间结果 HBM 读写：AI 提升，可能从 memory-bound 转为 compute-bound
- Attention 的 fused kernel（flash）省了 O(N²) 中间存储

*本书出现位置*：Ch.1 v5（小规模）、Ch.6 MLP、Ch.8-10 flash-attention。

== 通用路径：数值精度

- FP32 是安全但慢
- FP16 accumulate FP32 是训练标配（AMP）
- BF16 accumulate FP32 更稳（LLM 训练）
- FP8 / MXFP4 需要 block scaling 保精度

关注点：
- Softmax 的 `subtract max` 必须做，不然 exp overflow
- LayerNorm 累加用 fp32，输入输出用 fp16/bf16
- Flash-Attention v3 里 P 矩阵的 FP8 量化范围需要 rescale

*本书出现位置*：Ch.3、Ch.5、Ch.10。

== 决策树

面试遇到 "怎么优化 kernel X" 时，按下面顺序问：

1. *AI 是多少？roofline 落哪？*
2. Memory-bound：
   a. 合并访问对吗？
   b. HBM % 到多少了？（`ncu` 抓 `dram__bytes.pct`）
   c. 尝试 float4？
   d. Occupancy 够吗？
   e. Reduction 用 warp shuffle 了吗？
3. Compute-bound：
   a. 有 shared memory tile 吗？
   b. Register tile 到 TM×TN 了吗？
   c. 用 tensor core 了吗？（数据类型能用 FP16/BF16 吗？）
   d. 用 cp.async / TMA 做双缓冲了吗？
   e. Bank conflict 消掉了吗？（swizzle）
4. 通用：
   a. 能 fuse 上下游 kernel 吗？
   b. 精度可以降吗？（FP32 → FP16 → FP8）
   c. Launch overhead 显著吗？（CUDA Graph）

== 常见误区

#warn[
  *不测就优化*。凭直觉写"更快"的版本，实测常常慢。本书 vector_add 章 tiled 版本就慢于 naive。
]

#warn[
  *追 100% occupancy*。occupancy 50-75% 通常最优；100% 意味着寄存器用得太少，往往拖慢 compute。
]

#warn[
  *盯着 HBM % 优化*。HBM % 只是"有多少 cycle DRAM 在传数据"的比例，短 kernel 上会低估。实测 effective GB/s 更可靠。
]

#warn[
  *盲目 vectorize*。数据不对齐 / 访问模式不连续时，float4 会静默降级到多个窄访问，甚至更慢。
]

#warn[
  *忽略 numerical stability*。FP16 累加 softmax / layernorm 会 blow up。省时间但训练崩。
]

== 收尾

优化的正确姿势：

1. 有 baseline（正确、能跑）。
2. `ncu` 抓 metric 找瓶颈。
3. 一次只改一件事。
4. 再 `ncu` 抓，对比。
5. 如果*没变快甚至变慢*，回滚，猜错了。
6. 如果*变快了*，思考"为什么"——把它写成注释 / 文档，别让下一个人重犯。

这也是这本书每一章的写法：给出 ladder 上每一版的实测数字，让你看到什么优化真有用、什么优化只是"看起来聪明"。
