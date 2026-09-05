#import "../template.typ": *

= 附录 B：核心参考文献

按主题组织，标注每篇 paper 在本书哪一章用到。

== 集合通信与硬件

- Patarasuk & Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations," 2009. → 第 1 章 Ring AllReduce
- Sanders et al., "Two-tree algorithms for full bandwidth broadcast, reduction and scan," 2009. → 第 1 章 Tree AllReduce
- NVIDIA, NCCL developer guide. → 第 1 章 环境变量
- NVIDIA, "NVLink and NVSwitch technical overview," 2023-2024. → 第 1 章 硬件

== Data Parallel

- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017. → 第 3 章 linear LR scaling
- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," 2020. → 第 3 章 DDP bucket
- You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes," 2019 (LAMB). → 第 3 章 large batch

== ZeRO / FSDP

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC'20. → 第 4 章 ZeRO 1/2/3
- Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training," ATC'21. → 第 4, 10 章 offload
- Rajbhandari et al., "ZeRO-Infinity," SC'21. → 第 4 章 NVMe offload
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel," VLDB'23. → 第 4 章 FSDP
- Torchtitan blog: https://github.com/pytorch/torchtitan → 第 4, 5 章 FSDP2 + TP

== Tensor / Sequence Parallel

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019. → 第 5 章 TP
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022. → 第 5 章 SP + selective recompute
- Wang et al., "Reducing Activation Recomputation in Large Transformer Models with Async Tensor Parallel," 2024. → 第 5 章 async TP

== Pipeline Parallel

- Huang et al., "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism," NeurIPS'19. → 第 6 章 GPipe
- Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training," SOSP'19. → 第 6 章 1F1B
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," SC'21. → 第 6 章 Interleaved 1F1B
- Qi et al., "Zero Bubble Pipeline Parallelism," ICLR'24. → 第 6 章 ZeroBubble
- DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024. → 第 6, 8, 9 章 DualPipe / DeepEP / FP8

== 长上下文 / Context Parallel

- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context," 2023. → 第 7 章 Ring
- Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," 2023. → 第 7 章 Ulysses
- Brandon et al., "Striped Attention: Faster Ring Attention for Causal Transformers," 2024. → 第 7 章 Striped/Zigzag
- Fang et al., "USP: A Unified Sequence Parallelism Approach for Long Context Generative AI," 2024. → 第 7 章 USP
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision," 2024. → 第 7, 10 章

== MoE

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," 2017. → 第 8 章 起源
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR'21. → 第 8 章 EP 起源
- Fedus et al., "Switch Transformer: Scaling to Trillion Parameter Models," 2021. → 第 8 章 Switch
- Rajbhandari et al., "DeepSpeed-MoE," ICML'22. → 第 8 章 hierarchical a2a
- Hwang et al., "Tutel: Adaptive Mixture-of-Experts at Scale," MLSys'23. → 第 8 章 adaptive dispatch
- Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts," MLSys'23. → 第 8 章 dropless
- Jiang et al., "Mixtral of Experts," 2023. → 第 8 章 Mixtral
- DeepSeek-AI, "DeepSeek-V2 Technical Report," 2024. → 第 8 章 DeepSeek-MoE
- DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024. → 第 8, 9 章
- Cui et al., "Scalable Training of Mixture-of-Experts Models with Megatron-Core," 2026. → 第 2, 8 章 三堵墙
- DeepEP: https://github.com/deepseek-ai/DeepEP → 第 8 章
- DualPipe: https://github.com/deepseek-ai/DualPipe → 第 6, 8 章

== 精度 / FP8

- Micikevicius et al., "Mixed Precision Training," 2017. → 第 9 章 GradScaler
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," 2019. → 第 9 章 BF16
- Micikevicius et al., "FP8 Formats for Deep Learning," 2022. → 第 9 章 FP8
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine → 第 9 章
- Peng et al., "FP8-LM: Training FP8 Large Language Models," 2023 (MS-AMP). → 第 9 章
- DeepSeek-V3 tech report §3 FP8 recipe. → 第 9 章 blockwise
- NVIDIA MXFP8 spec (Blackwell). → 第 9 章

== Activation / Recompute

- Chen et al., "Training Deep Nets with Sublinear Memory Cost," 2016. → 第 10 章 checkpoint 起源
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022. → 第 10 章 selective
- Cui et al., Megatron-Core MoE tech report §Fine-grained recompute, 2026. → 第 10 章 fine-grained

== 大规模系统

- Jiang et al., "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs," NSDI'24. → 第 16 章
- Meta, "The Llama 3 Herd of Models," 2024. → 第 4, 6, 15 章
- ByteDance, "MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production," EuroSys'26 (arXiv:2505.11432). → 第 8 章
- Zheng et al., "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning," OSDI'22. → 第 5, 6 章 自动化并行
- veScale (ByteDance internal): https://github.com/volcengine/veScale → 第 4-6 章 DTensor
- Yuan et al., "Nemotron-4 340B Technical Report," 2024. → 第 12, 13 章 data pipeline

== 数据 / Packing

- MosaicML Streaming: https://github.com/mosaicml/streaming → 第 12 章
- WebDataset: https://github.com/webdataset/webdataset → 第 12 章
- Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining," NeurIPS'23. → 第 12 章 mixture
- Ye et al., "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance," 2024. → 第 12 章

== Multimodal

- Dehghani et al., "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution," NeurIPS'23. → 第 13 章
- Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS'23. → 第 13 章
- Liu et al., "LLaVA-NeXT," 2024. → 第 13 章 any-resolution
- Wang et al., "Qwen2-VL," 2024. → 第 13 章 dynamic resolution
- Chen et al., "InternVL 1.5-2.5," 2024. → 第 13 章
- Chen et al., "PaLI-3," 2023. → 第 13 章
- Yang et al., "CogVideoX," 2024. → 第 13 章 3D VAE

== RL / RLHF

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022. → 第 14 章 PPO
- Bai et al., "Constitutional AI," 2022. → 第 14 章
- Rafailov et al., "Direct Preference Optimization," 2023. → 第 14 章 DPO
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," 2024. → 第 14 章 GRPO
- DeepSeek-AI, "DeepSeek-R1," 2025. → 第 14 章
- Kimi Team, "Kimi K1.5," 2025. → 第 14 章 rollout 优化
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (verl), EuroSys'25. → 第 14 章
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF → 第 14 章
- NVIDIA NeMo-Aligner: https://github.com/NVIDIA/NeMo-Aligner → 第 14 章
- vLLM: https://github.com/vllm-project/vllm → 第 14 章
- SGLang: https://github.com/sgl-project/sglang → 第 14 章

== RLVR / Agentic RL

- Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training," 2024. → 第 15 章，RLVR 一词的出处
- DeepSeek-AI, "DeepSeek-R1," 2025. → 第 15 章，规则 reward、弃用 neural PRM/ORM 的理由
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," 2025. → 第 15 章，clip-higher / dynamic sampling / token-level loss / overlong shaping
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO), 2025. → 第 15 章，长度归一化与 std 归一化的偏差
- Yue et al., "VAPO: Efficient and Reliable RL for Advanced Reasoning Tasks," 2025. → 第 15 章，value-based 路线
- Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR," 2025. → 第 15 章，为什么 RLVR 评测需要对照组
- Kimi Team, "Kimi K1.5," 2025. → 第 15 章，partial rollout
- Kimi Team, "Kimi K2," 2025. → 第 15 章，agentic 数据合成与可验证 reward
- Qwen Team, "Qwen3 Technical Report," 2025. → 第 15 章，RLVR 在工业配方里的位置
- Wei et al., "SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution," 2025. → 第 15 章，verifier 太贵时的替代 reward
- Jin et al., "Search-R1: Training LLMs to Reason and Leverage Search Engines with RL," 2025. → 第 15 章，检索 token masking
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (verl), EuroSys'25. → 第 15 章，agent loop 与异步 rollout
- SkyRL: https://github.com/NovaSky-AI/SkyRL → 第 15 章，长 horizon agentic RL
- DeepSeek-AI, "DeepSeekMath" (GRPO), 2024. → 第 15 章，GRPO 原始定义

== Scaling Laws

- Kaplan et al., "Scaling Laws for Neural Language Models," 2020. → 第 2 章
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), 2022. → 第 2 章
- Hu et al., "Minimum Compute for Model Training with Data-Constrained Environments," 2024. → 第 2 章
