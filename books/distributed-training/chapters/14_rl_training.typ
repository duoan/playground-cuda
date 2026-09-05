#import "../template.typ": *

= RL 训练的分布式：从 PPO 到 GRPO 到 rollout/train 分离

RLHF 训练看起来只是 fine-tune 一个 policy model，实际上是分布式系统里最复杂的场景之一——同一 iteration 里要跑 4-6 个模型（policy, ref, reward, value, critic, tokenizer），且 policy 要 *generate* 出长序列（inference workload），然后再做 gradient update（training workload）。这一章讲清楚 PPO/GRPO/DPO 的分布式架构、rollout/train 分离、weight sync、vLLM/SGLang 集成、以及 verl/OpenRLHF 等主流框架的关键取舍。

#note[
  这一章讲的是*经典 RLHF*：reward 来自一个学出来的 reward model，rollout 是一次性的单轮生成。2024 年底之后工业界主流已经迁移到 RLVR（reward 换成确定性 verifier）和 Agentic RL（rollout 变成多轮的 LLM ↔ 环境循环），它们的系统形态差异很大 —— 多出一个 CPU verifier 集群、rollout 期 GPU 会闲置、轨迹长度方差上百倍、训练侧多出 token masking 这个大坑。这些内容在*第 15 章*。建议按顺序读：本章建立 rollout/train 分离的基本框架，第 15 章讲它在新范式下怎么变。
]

== 从 PPO 说起

Standard RLHF PPO 循环（InstructGPT / Anthropic 风格）：

```
for iter in range(N):
    # 1. Rollout: policy generates responses
    prompts = sample_batch()
    responses = policy.generate(prompts, max_new_tokens=1024)
    
    # 2. Reward
    rewards = reward_model(prompts, responses)
    values  = value_model(prompts, responses)
    
    # 3. Compute advantages (GAE)
    old_logprobs = policy_logprobs_of(responses)
    ref_logprobs = ref_model_logprobs_of(responses)
    advantages   = gae(rewards, values)
    
    # 4. PPO update (k epochs on same rollout data)
    for epoch in range(K):
        for mini_batch in split(rollout_data):
            new_logprobs = policy(mini_batch)
            ratio = exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-eps, 1+eps) * advantages
            kl    = old_logprobs - new_logprobs
            loss  = -min(surr1, surr2) + beta * kl
            loss.backward()
            optim.step()
```

*涉及模型*：
- Policy (training + generation)
- Reference (frozen, for KL)
- Reward (frozen, scoring)
- Value / Critic (training, GAE)

*涉及 workload*：
- Rollout: autoregressive generation (inference-style, memory-bound)
- Reward inference (forward-only)
- Ref inference (forward-only)
- Policy training (fwd + bwd + optim)
- Value training (fwd + bwd + optim)

== GRPO (DeepSeek): 干掉 value model

DeepSeek-Math / DeepSeek-R1 提出 GRPO (Group Relative Policy Optimization)：

+ 不训 value model —— 直接用 group baseline: rollout $N$ 个 response per prompt，advantage = reward - group_mean_reward
+ 无 critic → 少一个 model to train
+ 更省显存，训练更快，效果不差 (DeepSeek-R1 用 GRPO 训 reasoning)

*简化的 loss*：

$ A_i = (r_i - "mean"(r_1..r_N)) / "std"(r_1..r_N) $
$ L = - EE_i [min("ratio"_i A_i, "clip"("ratio"_i) A_i)] + beta dot "KL"("policy" | "ref") $

*分布式意义*：一个 model 少训一个 GPU 组，节省 20-30% 资源。GRPO 变体（DAPO, DrGRPO, ...）都保持这个结构。

== Rollout 是 inference workload：与 training 完全不同

Training 一个 step 处理 GBS × seq_len tokens；rollout 是 autoregressive，每 sample 逐 token generate:

- BS=1024, seq_len=1024, autoregressive → 需要 1024 forward passes (per token)
- 每 forward 只处理 BS tokens (decode phase, M=BS)
- Memory-bound（small-M GEMM）
- KV cache 大：$2 L B S H "bs"$，占大量显存

*所以 rollout ≠ training 的 forward*。用 training framework 直接跑 rollout 会：
+ 效率极低（training 的 forward 是 dense compute-bound；decode 是 memory-bound）
+ 无 KV cache 优化
+ 无 continuous batching
+ 无 PagedAttention

*生产做法*：用 dedicated inference engine (vLLM, SGLang, TensorRT-LLM) 跑 rollout，训练用 Megatron/FSDP。

== Rollout / Train 分离架构

*架构*：

```
GPU pool A (Rollout):   vLLM / SGLang instances
   - Serve policy generate()
   - Serve reward inference
   - Serve ref inference
   ↓ (sync generated data)
GPU pool B (Training):  Megatron / FSDP
   - Policy training (PPO update)
   - Value training (if PPO)
```

两 pool 可以是：
+ *Colocated*：同一批 GPU，rollout 完停 vLLM 起 training，交替进行。OpenRLHF v1 类
+ *Separate*：不同 GPU 组，rollout pool 和 training pool 各自独立。verl / OpenRLHF v2 / NeMo-Aligner 用

*Colocated 的优劣*：
- 优：GPU 利用率高 (100% 都在用)
- 劣：切换需要卸载 vLLM state / 重启 —— 秒级 overhead，且 rollout 与 training 时长必须匹配

*Separate 的优劣*：
- 优：两阶段并行，rollout 与 training overlap
- 劣：GPU 分配比例难调（rollout 慢 → training 等；反之亦然）
- 复杂：需要 weight sync 与数据传输机制

现代 RL 框架（verl, OpenRLHF, HybridFlow）都支持两种模式，实测哪个快取决于 model size 和 rollout length。

== Weight Sync: 从 training 到 rollout

每次 PPO update 之后，rollout engine 用旧 weight——不对。要把新 weight sync 到 rollout engine。

三种方案：

=== 方案 1: NCCL broadcast (Colocated)

Rollout 与 training 在同一批 GPU (colocated)。Training 更新后：

```python
# Training rank 直接把 policy weight overwrite 到 vLLM weight
for name, p in policy.named_parameters():
    vllm_engine.get_model().state_dict()[name].copy_(p)
```

优：instant，无跨 GPU 传输。劣：需要 model 在 CPU 里 (rollout 和 training 交替占 GPU)，或 vLLM 支持 hot-update。

=== 方案 2: NCCL group broadcast (Separate)

Rollout 与 training 在不同 GPU 组。定义 cross-group NCCL group，broadcast weight:

```python
# Training rank 0 → rollout rank 0..N
cross_group = dist.new_group(ranks=[train_rank_0, *rollout_ranks])
for name, p in policy.named_parameters():
    dist.broadcast(p, src=train_rank_0, group=cross_group)
    vllm.load_weight(name, p)
```

优：直接 GPU-to-GPU。劣：broadcast 慢 (整模型 130 GB for 70B，即使 NVLink 也几百 ms)。生产：只 broadcast changed weight (LoRA scenario) 或用 pipelined broadcast overlap 与 training。

=== 方案 3: Shared memory / NVLink DMA (Colocated + shared)

Nvidia FlexPPO / ByteDance verl 用：把 policy weight 存在共享 GPU buffer，training 更新后 vLLM 直接读同一 buffer。

优：零 copy。劣：需要 hack vLLM 内部 weight loader，framework-specific。

== 主流 RL 框架架构

=== OpenRLHF (from OpenLLMAI)

- 早期 colocated (v1)，v2 支持 separate + Ray-based orchestration
- Ray Actor 抽象每个 role: `PolicyActor`, `RolloutActor`, `RewardActor`, `RefActor`
- 用 vLLM 做 rollout，DeepSpeed / FSDP 做 training
- 支持 PPO/DPO/GRPO/KTO/REINFORCE++

=== verl (ByteDance, HybridFlow)

- 论文 "HybridFlow: A Flexible and Efficient RLHF Framework"
- 核心：Hybrid mode - rollout 和 training 都可以配置为 colocated 或 separated
- SPMD + Ray 混合调度
- Weight sync 用 NCCL broadcast 优化
- 生产用 (kimi K1.5 训练)

=== NeMo-Aligner (NVIDIA)

- Megatron-based
- Colocated + weight offload
- 支持 DPO / SPIN / SteerLM

=== TRL (HuggingFace)

- Single-node / small-scale friendly
- 学术友好，生产大规模不行
- 集成 PEFT (LoRA)

=== DeepSpeed-Chat

- 早期 (2023) 的三阶段 pipeline (SFT + RM + PPO)
- 老，很少用

== Long-context / Reasoning RL 的挑战

DeepSeek-R1 类 reasoning 训练：response 长度 8K-32K+，rollout 一个 sample 要几分钟。

*问题*：
+ Rollout time 主导 (占 90%+)
+ Rollout 期间 training GPU 空闲
+ KV cache 大 (32K × 70B = 200+ GB per instance)

*优化*：
+ *Speculative decoding*：小 model draft + 大 model verify，rollout 加速 2-3×
+ *PagedAttention*：动态 KV cache 管理，多 sample 共享 memory
+ *Continuous batching*：sample 完成时不等 batch，立刻起新 sample
+ *Rollout 与 training 并行*：async pipeline，training rank 一直在做 update on old rollout
+ *"On-policy" vs "off-policy"*：允许 rollout 用略旧的 weight，避免每 step sync

Kimi K1.5 tech report 里描述用 vLLM + custom scheduler 做 rollout，rollout throughput > 1000 samples/min。

== Reward Model 的分布式

Reward model 是 forward-only，需要 serve inference：

- Same size as policy (7B-70B typical)，一张卡装不下
- 用 TP inference (vLLM TP=8)
- 每 rollout iteration 打分 batch × N candidates → 有的框架把 reward 和 rollout colocate 到同 pool

*进阶*：reward model ensemble (多个 reward + averaging)，或 process reward (每 step 打分)。分布式复杂度直接乘 N。

== KL divergence 的分布式计算

PPO 里 KL 是 policy 与 ref 的对数比。ref model forward = full model forward → 需要一份 GPU。

*常见做法*：ref model 与 policy 同 pool (放在 rollout pool 里 forward)，或独立 pool（简单但耗 GPU）。

*Kimi K1.5 优化*：ref 与 policy 共享 base，用 LoRA delta，ref forward = policy forward - LoRA output → 只需一个 forward。

== 显存分析：一次 PPO iteration 的 GPU 占用

以 70B policy + 70B ref + 70B reward + 70B value on 128 H100:

Training pool (64 GPU):
- Policy: FSDP HYBRID_SHARD, TP=8, DP=8 → 70B / 8 shard × 16 bytes = 17.5 GB/GPU (fully sharded)
- Value: 同上, 17.5 GB/GPU
- Activation for training: ~20 GB/GPU
- Total: ~50 GB/GPU, well within 80GB

Rollout pool (64 GPU):
- Policy (vLLM TP=8): 70B × 2 (BF16) / 8 = 17.5 GB
- Ref (vLLM TP=8): 17.5 GB
- Reward (vLLM TP=8): 17.5 GB
- KV cache pool: ~30 GB (大 batch)
- Total: ~80 GB, tight

*即 128 GPU 训 70B RL setup 显存刚好卡满*。放宽：quantize ref/reward 到 FP8 (省一半)。

== 一个 GRPO 训练配置样例 (verl-like)

```yaml
# 128 H100 训 Qwen2-72B with GRPO
policy_model:
  size: 72B
  training:
    dp_size: 8
    tp_size: 8
    pp_size: 1
    fsdp: HYBRID_SHARD
    optimizer: precision-aware AdamW
    activation_checkpoint: selective
    precision: bf16

rollout:
  engine: vllm
  tp_size: 8
  n_instances: 8      # 8 vllm engines
  max_new_tokens: 4096
  temperature: 1.0
  n_rollouts_per_prompt: 16    # GRPO group size

reward:
  type: rule_based_math          # 或 reward model
  # 或
  # engine: vllm
  # model: reward-72b
  # tp_size: 8

ref:
  colocate_with_rollout: true
  precision: fp8               # 省显存

training_loop:
  batch_size: 1024              # prompts per iter
  ppo_epochs: 1                # GRPO 通常 1 epoch
  minibatch_size: 32
  lr: 1e-6
  kl_coef: 0.001
  clip_range: 0.2
  
weight_sync:
  frequency: every_iter
  method: nccl_broadcast
  overlap_with_training: true    # 下 iter forward 时 broadcast
```

*行业参考*：DeepSeek-R1 用 GRPO 训 671B MoE，成本估算 rollout 占总 RL 训练时间 70%+。Kimi K1.5 类似。

== 常见 RL 训练坑

+ *Reward hacking*：模型学到"骗 reward model"而非真解决问题。检测：val 集 reward vs 人工评估的 gap。
+ *KL blow up*：某个 mini-batch KL 突然大，梯度爆。做 KL clip 或 adaptive KL coefficient。
+ *Rollout diversity 掉*：温度 0 让 policy 越训越 deterministic。GRPO 里 rollout 用 temp=1.0，evaluation 用 temp=0。
+ *Ref-policy drift*：训久了 policy 与 ref 差异过大，KL 项 blow up。定期 update ref = policy (like 每 1000 step)。
+ *vLLM & training weight desync*：weight broadcast 忘了 sync，rollout 用旧 weight，advantage 有偏。
+ *长 rollout 的 length exploit*：模型学到"输出长 = reward 高"，输出全是废话。用 length penalty。
+ *"Format collapse"*：GRPO 里 group std=0 时 advantage=inf。加 small epsilon 到 std。

== 面试考点

#interview[
  *Q1*: RLHF 训练里为什么要 rollout / train 分离？

  A: Rollout 是 autoregressive generation (memory-bound decode, small-M GEMM)，training 是 dense fwd+bwd (compute-bound, large-M)。用 training framework 直接跑 rollout 效率极低（无 KV cache, 无 continuous batching）。分离让 rollout 用 vLLM/SGLang 拿 5-10× 加速；training 用 Megatron/FSDP 拿满 MFU。
]

#interview[
  *Q2*: GRPO 相比 PPO 的核心简化？

  A: 去掉 value model / critic。用 group baseline：一个 prompt rollout N 个 response，advantage = (reward - group_mean) / group_std。少一个 model 训，省 20-30% GPU。DeepSeek-R1 证明效果不差。变种 DAPO / DrGRPO 都基于此。
]

#interview[
  *Q3*: 权重从 training 同步到 rollout engine 的三种方案？

  A: (1) Colocated 直接 copy: 同 pool 交替，无跨节点传输，但要停 vLLM；(2) NCCL cross-group broadcast: 训练完后 broadcast 到 rollout 组，几百 ms overhead；(3) Shared buffer: 训练与 rollout 共 GPU buffer，零 copy，需 hack vLLM。verl / OpenRLHF v2 主要用 (2) + overlap with training。
]

#interview[
  *Q4*: DeepSeek-R1 训练里 rollout 占多少时间？怎么优化？

  A: Reasoning task response 长 (8K-32K)，rollout 占 70%+ RL 训练时间。优化：(a) vLLM continuous batching + PagedAttention；(b) speculative decoding；(c) rollout 与 training async pipeline（下 iter training 时上 iter rollout 继续）；(d) 允许 slight off-policy (K epoch 用一次 rollout data)。Kimi K1.5 报告用 (a)-(c) 组合。
]

#interview[
  *Q5*: RLHF 里 KL divergence 项怎么算？ref model 放哪？

  A: KL 用 policy 与 ref 的 logprob 差。ref model = full forward 一次。放 rollout pool（forward-only, 与 policy generate 同 GPU 用 vLLM 起两个 instance）；或独立 pool；或用 LoRA delta 表达 (ref forward = policy - LoRA output)，Kimi K1.5 做法，只需 1 forward。
]

#interview[
  *Q6*: rollout 与 training 用不同 GPU 组时怎么分配比例？

  A: 依赖 rollout time vs training time 比例。DeepSeek-R1 类长 reasoning：rollout : training ≈ 4 : 1 → GPU 分 80% rollout, 20% training。短 response (chat, math easy) 可能 1:1。生产会先 profile 一个 iter 确定比例；有些框架 (verl) 支持动态调整。
]

#interview[
  *Q7*: 一次 GRPO iteration 里 GRPO 的 group size 与 batch size 怎么关系？

  A: `group_size N` = 每 prompt rollout N 个 response 算 group baseline；`batch_size B` = 每 iter 用 B 个 prompt。总 rollout samples = B × N。GRPO 论文 N=64；DeepSeek-R1 N=16-32。B × N 太大 rollout 时间长，太小 baseline 方差大。经验：N=8-32, B=256-1024。
]

#interview[
  *Q8*: 你的 RL 训练 MFU 只有 15%，比 SFT 40% 差很多，正常吗？

  A: 正常。RL 里"MFU"不好定义，因为 rollout 是 inference-style 计算 (decode 只有 small-M GEMM, 天然 memory-bound)。SFT 是 dense fwd+bwd (compute-bound)。真实指标应该看：(a) rollout throughput (samples/s); (b) training update tokens/s; (c) end-to-end iterations per hour。GPU 挂 rollout 占比 70%+ 会拉低 "training MFU"，但整体 wall-clock 反而更快。
]
