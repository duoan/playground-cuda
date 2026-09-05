#import "../template.typ": *

= 前言

== 这本书是写给谁的

这本书是给那些*已经能独立训通一个 dense 中等模型（1-10B）、想拿分布式训练岗位 offer* 的人写的面试通关手册。

在阅读之前你应该：

- 熟悉 PyTorch，能自己写一个 Transformer training loop。
- 对 CUDA、cuBLAS、NVLink、Infiniband 这些名词不陌生（不需要写 kernel）。
- 用过 `torch.distributed` 或至少跑过 `torchrun --nproc-per-node`。
- 知道 AMP / gradient accumulation / checkpoint 是什么，但可能没深挖。

如果你从来没接触过分布式，也可以直接读——第 1 章会把集合通信 + 带宽模型从零讲一遍。但节奏是"面试拿分"而不是"科普"，所以每一章都会*直接给数字、公式、坑*。

== 这本书讲什么，不讲什么

*讲的*：

+ *分布式训练的第一性原理*：内存怎么算、通信怎么算、compute/comm 怎么 overlap，Roofline 拉到哪里。
+ *每种并行策略的核心动机 + 数学 + 通信模式 + Megatron/DeepSpeed/FSDP 的具体实现差别*——DP, ZeRO 1/2/3, FSDP, TP, SP, PP, CP, EP。
+ *长序列并行*：Ring Attention, Ulysses, USP hybrid，怎么支持 1M ctx。
+ *MoE 训练*：EP + all-to-all + DualPipe + DeepEP（本书简讲，深入部分请看姊妹卷《Sparse MoE 训练实战》）。
+ *精度*：AMP / BF16 / FP8 (Transformer Engine, DeepSeek-V3 recipe) / FP4，loss scaling 与 outlier 处理。
+ *激活优化*：selective recompute, activation offload, TE checkpoint, DeepSeek 的模块级 recompute。
+ *Overlap*：DDP bucket, ZeRO param gather, TP-SP overlap, PP overlap (1F1B / ZeroBubble / DualPipe / Megatron FWD-BWD merged)。
+ *Data loader*：packing (BFD / FFD / streaming), variable-length batching, IterableDataset shard 均衡, mosaicml streaming。
+ *多模态特殊问题*：变长图像/视频/音频、encoder/decoder 隔离、跨模态负载不均、data-packing 的特殊难度。
+ *RL 训练*：RLHF/PPO/GRPO 的分布式，rollout/train 分离，vLLM/SGLang 集成，weight sync 的三种方案。
+ *工程稳定性*：MegaScale 级 checkpoint、容错、慢卡诊断、监控指标。
+ *面试题库 + 面试故事模板*：100 道高频题 + 20 个"框架都做了这些，你还能加什么"式差异化优化案例。

*不讲的*：

- CUDA kernel 手写（去看姊妹卷《CUDA Kernel 优化实战》）
- MoE 内部 router / gating / grouped GEMM 的细节（去看姊妹卷《Sparse MoE 训练实战》）
- 具体框架 API 的完整教程（Megatron-LM / DeepSpeed 的官方文档已经写得很好，我们只讲"怎么在面试里说清楚它做了什么以及为什么"）
- 推理（vLLM / SGLang）的深入优化——RL 章会讲一点，但推理是另一本书

== 怎么读

- *顺序读*：第 1-2 章是所有后续章节的公共基础。第 3-8 章是并行策略，可以按顺序深入。第 9-11 章是优化技巧，正交于并行策略。第 12-15 章是工程实战。
- *对着 profile 读*：所有优化都对应 nsys/torch profiler 上的具体现象。手边打开一个 profiler screenshot 会更快。
- *把面试考点框读透*：每章末尾紫框是高频追问，答案在正文里都有依据。
- *面试故事*（黄框）是"你可以怎么用这个知识点在面试里讲一个 STAR 故事"的模板，不是照抄的话术。核心的话你必须真的做过。

== 参考文献与致谢

本书内容主要综合了以下工作，具体引用见附录 B：

- *DP / ZeRO 家族*：DeepSpeed (Rajbhandari 2020), ZeRO-Offload, ZeRO-Infinity, FSDP (Zhao 2023)
- *Megatron 家族*：Megatron-LM (Shoeybi 2019), Megatron-Turing NLG, Megatron-Core MoE (2026), Reducing Activation Recomputation (2022, sequence parallel)
- *Pipeline*：GPipe (Huang 2018), PipeDream (Narayanan 2019), Interleaved 1F1B (Narayanan 2021), ZeroBubble (Qi 2023), DualPipe (DeepSeek 2024)
- *长上下文*：Ring Attention (Liu 2023), DeepSpeed-Ulysses (Jacobs 2023), USP (Fang 2024), Striped Attention (Brandon 2024)
- *MoE*：GShard (Lepikhin 2020), Switch (Fedus 2021), Mixtral, DeepSeek-V3 (2024), Megablocks (Gale 2022), Tutel (Hwang 2023), DeepEP
- *精度*：Transformer Engine, DeepSeek-V3 FP8 recipe, MS-AMP
- *系统*：MegaScale (Jiang 2024), MegaScale-MoE, Alpa (Zheng 2022), veScale, torchtitan
- *RL*：InstructGPT (Ouyang 2022), OpenRLHF, verl (ByteDance), TRL, DeepSpeed-Chat, NeMo-Aligner
- *Data*：Mosaicml Streaming, MosaicBench, Nemotron data pipeline，torchdata

== 记号约定

- $B$：micro-batch size；$"GBS"$：global batch size
- $S$ / $L$：sequence length
- $H$：hidden dim；$I$：FFN intermediate dim
- $A$：attention heads；$d_h = H / A$：per-head dim
- $L$（在 layer 上下文里）或 $N_"layers"$：Transformer layer 数
- $N$：模型参数量（B = billion 参数）
- $"DP"$ / $"TP"$ / $"PP"$ / $"CP"$ / $"EP"$ / $"SP"$：各种并行的 world size
- $W$ = world_size = $"DP" times "TP" times "PP" times "CP" times "EP"$（视配置）
- $"bs"$：以 bytes 计的精度大小（BF16 = 2, FP32 = 4, FP8 = 1）
- 硬件默认 H100 SXM5 80GB（HBM 3.35 TB/s，NVLink4 900 GB/s bidi，BF16 989 TFLOPS Tensor Core）；性能数字是量级估算，实测请以 profile 为准。

== 一句话导航

如果你只有 30 分钟，先读：

+ 第 2 章 §"Roofline + 三堵墙"
+ 第 4 章 §"ZeRO-3 = FSDP，全书重点公式"
+ 第 6 章 §"1F1B vs DualPipe bubble 对比表"
+ 第 11 章 §"什么时候 overlap 值得做"
+ 附录 D 前 20 道高频题

翻页开始。
