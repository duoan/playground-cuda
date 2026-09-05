#import "../template.typ": *

= 附录 E：面试故事 —— "框架都做了，你还能加什么"

面试官最爱问的一类问题：*"Megatron / DeepSpeed 都已经做了 XX，你在训练里还能做什么优化？"*——这是考察你能不能真读过 profile、能不能提出 delta 优化。

这一附录不是"帮你编经历"——里面的故事要基于*你实际做过*的项目才有说服力。这里给的是：

+ *STAR 框架*：把技术优化包装成面试故事的结构
+ *20 个高频"差异化优化" 案例*：都是公开报告 / paper 里描述过的、可以延伸自己的经历讲的具体优化点
+ *"框架 X 已做，我加了 Y"* 的对比表：让你的回答有 delta

== STAR 框架：一个技术故事的结构

STAR = *Situation, Task, Action, Result*。用于把"我做过 X 优化"结构化成 90 秒的完整故事。

*一个模板*：

- *Situation (30 秒)*：什么模型、什么 setup、遇到什么现象。数字量化 (MFU X%, step Y ms, comm 占 Z%)
- *Task (10 秒)*：你负责什么，目标是什么
- *Action (40 秒)*：你怎么诊断的，试过什么方案，为什么选定这个。这里最能体现深度。
- *Result (10 秒)*：结果数字。MFU 从 X → Y，step time 从 A → B，节省 GPU-hours。

*不要*：只说"我调 flag"、"我加机器"、"我 fine-tune 参数"。要说：*为什么这么做，怎么发现问题，别的选项为什么不行*。

== 20 个可复用的"差异化优化"案例

以下每个案例都基于*公开的* paper / tech report / repo，可用来延伸自己经历讲。斜体是"面试口径"版本。

=== 案例 1: DDP bucket size 调优

*背景*：训 30B model on 128 H100，MFU 32%，nsys 看到 backward 结束还有大段 AR 等待，overlap 只 60%。

*诊断*：DDP 默认 bucket=25 MB，30B 模型有 200+ bucket，每个 AR 都有 30 µs 启动 overhead，累积起来 6 ms/step 只是启动。

*Action*：调 `bucket_cap_mb=200`，同时 `gradient_as_bucket_view=True`。

*Result*：MFU 32% → 41%，step time -14%。

_"Megatron 默认 bucket 25 MB 是通用值，实测发现 30B 模型上启动 overhead 累积 6 ms/step，调到 200 MB 之后 overlap 从 60% 提到 90%+，MFU +9 个百分点。"_

=== 案例 2: FSDP `limit_all_gathers` 调优

*背景*：FSDP FULL_SHARD + BACKWARD_PRE 训 70B，OOM。

*诊断*：BACKWARD_PRE 太激进 prefetch，同时持有 3-4 层 param unshard 峰值超显存。

*Action*：设 `limit_all_gathers=True`，最多 prefetch 2 层。

*Result*：显存峰值 -12 GB，可跑更大 batch。

=== 案例 3: HSDP node 内 wrap 粒度

*背景*：HSDP 训 Llama-3 70B on 64 卡 (8 node × 8 GPU)，intra-node AG 慢。

*诊断*：node 内 8 卡的 shard 单位太小 (每 param 8 份，NCCL 效率低)。

*Action*：改用 `_HYBRID_SHARD_ZERO2`，intra-node 只 shard grad + optim，weight replicate；跨节点仍 DDP。

*Result*：intra-node comm -40%，MFU +6%。

=== 案例 4: TP + SP 里 Async TP

*背景*：Llama 70B TP=8 训，AR 占 forward 22%。

*Action*：升级到 TE `LayerNormMLP` fused kernel + `--tp-comm-overlap`。GEMM tile 与 AR chunk overlap。

*Result*：TP comm 22% → 6%，MFU +12%。

=== 案例 5: MoE overlap 从 naive 到 Megatron merged

*背景*：Mixtral 8×7B on 64 H100，MoE dispatch a2a 占 forward 28%。

*诊断*：naive 实现 dispatch/expert/combine 完全串行。

*Action*：开 `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`，CUDA_DEVICE_MAX_CONNECTIONS=8。

*Result*：a2a 占比 28% → 4%，MFU 38% → 51%。

_"这个不是我发明的，Megatron 2024 加的功能，但当时集群还没升到新版 Megatron。我 backport 了 patch，验证在生产不 break。收益 +13% MFU。"_

=== 案例 6: DeepEP V1 → V2 SM carve-out

*背景*：MoE 训 with DeepEP V1，GEMM 效率只 65% (SM 被 comm 抢)。

*诊断*：DeepEP V1 吃 20 SM。

*Action*：升级 V2，把 comm SM 降到 4-6。

*Result*：GEMM 效率 65% → 85%，端到端 +18%。

=== 案例 7: FA varlen 与 CUDA graph 兼容

*背景*：多模态训练里 seq_len 大方差 (500-16K)，FA varlen kernel launch overhead 累积占 8% step。

*Action*：seq_len 按 buckets ([1024, 2048, 4096, 8192, 16384]) 归一化，per-bucket capture CUDA graph。

*Result*：kernel launch overhead 8% → 1%。

=== 案例 8: Vision encoder 分离 device group

*背景*：VLM 训 32B (LLM) + 0.4B ViT，MFU 25%。

*诊断*：ViT 与 LLM 共 TP=8，ViT 用不上 TP，AR 空开销 + rank imbalance。

*Action*：ViT 用独立 process group (DP=world, no TP)；LLM 保 TP=8+PP+FSDP。

*Result*：MFU 25% → 38%。

=== 案例 9: Sequence length balancing (SFT)

*背景*：SFT 训 Qwen2-72B，DP=8，step time p99/p50 = 1.4 (严重 straggler)。

*诊断*：log 每 rank 每 batch total tokens，发现 rank 3 常有 8K seq，其他 rank 500-2K seq。

*Action*：换成 packing (fixed 8192 length) + FA varlen。所有 rank total tokens 恒定。

*Result*：p99/p50 = 1.05，MFU +25%。

=== 案例 10: Precision-aware optimizer + FP8 activation

*背景*：训 30B 单机 8xH100 pretrain，activation OOM (每卡 78 GB / 80 GB 边缘)。

*Action*：`--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` (省 12P → 6P optim state, 每卡 -12 GB)，加 FP8 activation (再 -8 GB)。

*Result*：显存 78 → 58 GB，可以扩 batch 2×，wall-clock -30%。

=== 案例 11: Fine-grained recompute

*背景*：MoE 训 Mixtral 8×22B，OOM 需要开 full recompute，但 recompute 让 a2a 通信量翻倍 (backward 时 forward 再跑一遍触发 dispatch)。

*Action*：改用 `--recompute-granularity selective --recompute-modules moe_act mlp core_attn`，只 recompute 计算部分，不触发 a2a。

*Result*：显存与 full recompute 相当，compute overhead 从 32% → 6%。

=== 案例 12: In-memory checkpoint

*背景*：训 70B on 512 卡，checkpoint 保存 5 分钟 stop training，每 30 min 一次，5% overhead。

*Action*：改 async checkpoint (D2H → CPU pinned mem, 15s)，配合 in-memory peer backup (存到隔壁 node CPU DDR)，disk backup 每 6h 一次。

*Result*：checkpoint 阻塞 training 时间 5 min → 15s，overhead 5% → 0.5%。恢复时间也从 20 min → 30s。

=== 案例 13: Data loader worker + prefetch tuning

*背景*：训 loss curve 正常，但 GPU util 只 78%。

*诊断*：nsys 看每 step 开头有 15 ms idle，正好等 loader。

*Action*：num_workers 4 → 16, prefetch_factor 2 → 8, persistent_workers=True。

*Result*：GPU util 78% → 92%，MFU +10%。

=== 案例 14: NCCL 环境变量批调优

*背景*：新集群 (8 node × 8 H100 with IB NDR)，AR 只跑到 40 GB/s (理论 100 GB/s)。

*Action*：跑 nccl-tests + grid search `NCCL_MIN_NCHANNELS, NCCL_NTHREADS, NCCL_BUFFSIZE, NCCL_IB_HCA` 组合。找到 optimal config。

*Result*：AR 跑到 85 GB/s，DP training step -12%。

_"NCCL 参数没有标准答案，每集群都要实测。我把 grid search 脚本 open source 出来了。"_

=== 案例 15: RL rollout / training 比例调优

*背景*：GRPO 训 32B，rollout 用 vLLM (16 GPU) + training (16 GPU)，一 iter 6 min。

*诊断*：log 每阶段时间，rollout 5 min training 1 min → training GPU 空 4 min。

*Action*：调 rollout : training = 12 : 20 GPU。

*Result*：iter time 6 → 3.5 min，同硬件 iter/hour +70%。

=== 案例 16: DDP grad clip 顺序修正

*背景*：训到某 step loss spike，后续训炸。

*诊断*：grad clip 放在 micro-batch backward 之后，此时 grad 是 partial (未 AR)，clip 无意义。

*Action*：`clip_grad_norm_` 移到 `optim.step()` 之前 (AR 完成后)。

*Result*：训练稳定，不再 spike。

_"这是很常见的 bug，尤其在 gradient accumulation 下。DDP 的 AR 在 backward 完成时才做，clip 在 AR 前 clip 的是 partial grad，不影响 optim.step，无效但看似有效。fix 后训练稳定。"_

=== 案例 17: 长 ctx 训练 RoPE base 调整

*背景*：把 8K ctx pretrain model 扩到 128K ctx SFT，val loss 不降甚至升。

*诊断*：RoPE base=10000 对长 pos 有 extrapolation 失效，attention pattern 崩。

*Action*：RoPE base → 500000（YaRN / LongRoPE 建议），或用 NTK-aware scaling。

*Result*：128K val loss 下降，perplexity ≈ 8K baseline。

=== 案例 18: Multi-source data weight 调优 with DoReMi

*背景*：预训 7B 用 5 类 domain 数据，naive weight (按 size) val loss 不理想。

*Action*：跑 1B proxy model + DoReMi 学 optimal weight，重训 7B。

*Result*：downstream benchmarks (HumanEval, MMLU) 平均 +2 分。

=== 案例 19: MoE aux-loss-free 切换

*背景*：训 128 experts MoE 用 aux loss (α=1e-2)，router 梯度受 aux 干扰，主 loss 收敛慢。

*Action*：换 DeepSeek aux-loss-free (bias tuning, γ=1e-3)，α 降到 1e-4 作 fallback。

*Result*：main loss 收敛快 30%，final val loss -0.05。

=== 案例 20: RL 中 KL blow up 处理

*背景*：GRPO 训 10 iter 后 KL 突然 5.0+，policy 输出退化。

*Action*：(a) adaptive KL coef (KL > target × 2 时 coef × 1.5)；(b) 每 1000 step update ref = policy avoid drift。

*Result*：KL 稳定在 0.5-1.0 范围，训练继续。

== "框架 X 已做，我加了 Y" delta 对比表

面试官问"Megatron 已经有 XX，你的贡献是什么"时，给个具体 delta：

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*基线做的*], [*常见 delta 优化*], [*量化收益*],
    [DDP bucket 25 MB],
    [调 200 MB + grad-as-view],
    [+5-10% MFU],
    [FSDP FULL_SHARD],
    [切 HSDP，跨节点用 DDP],
    [+30-50% wall-clock],
    [FSDP BACKWARD_PRE],
    [加 limit_all_gathers 控制显存峰值],
    [显存 -10 GB (avoid OOM)],
    [Megatron TP+AR],
    [`--tp-comm-overlap` (async TP)],
    [+10-15% MFU],
    [Megatron 1F1B PP],
    [`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`],
    [MoE +10-15% MFU],
    [DeepEP V1],
    [升级 V2 (SM 20 → 4-6)],
    [+15-20% MFU],
    [Naive optim state FP32],
    [`--use-precision-aware-optimizer`],
    [-12 GB/GPU (nearly free)],
    [Full activation recompute],
    [Fine-grained (avoid a2a re-trigger)],
    [compute overhead 33% → 6%],
    [Sync checkpoint 5 min],
    [Async + in-memory peer backup],
    [Overhead 5% → 0.5%],
    [Fixed batch by count],
    [Sequence packing + FA varlen],
    [+50-100% throughput],
    [Vision + LLM 同 TP=8],
    [Split device group],
    [+10-20% MFU],
    [Naive RL rollout in trainer],
    [vLLM + colocated + weight sync],
    [Rollout 5-10× 加速],
    [PPO with value model],
    [GRPO (drop value)],
    [-25-30% GPU need],
    [BF16 grad AR],
    [FP32 grad AR at W > 1024],
    [Loss 曲线稳定],
    [Rank 0 gather checkpoint],
    [DCP sharded + resharding],
    [Checkpoint 5× 快 + elastic],
  ),
  kind: table,
  caption: [常见"delta 优化"对比表。用来在面试里快速 anchor "我在 X 基础上做的 Y" 的具体 delta。],
)

== 面试里怎么回答 "MFU 只 30%，你怎么优化"

*错误答法*：一上来堆技术名词——"我用 DualPipe + FP8 + FlashAttention 3 + selective recompute"。面试官会立刻打断问"为什么？"

*正确答法（framework）*：

+ *先诊断*：跑 nsys 一步 profile，判断 bottleneck 属于三堵墙的哪一堵
+ *如是 memory bound*：先开 precision-aware optim (免费)，再看 activation recompute
+ *如是 comm bound*：查是 DP AR / TP AR / EP a2a / PP P2P 哪种，各自有对应 overlap 方案
+ *如是 compute inefficient*：查 GEMM M 维、kernel fusion、CUDA graph
+ 每一步都*量化* MFU 变化，让面试官看到你的诊断能力

*典型对话*：

> 面试官: "我们 MoE 训练 MFU 只 35%，你会怎么优化？"
> 
> 你: "我先想问几个问题——(1) 你们模型多大？多少 expert？(2) 集群多大？H100 还是 H800？跨节点 IB 是什么速率？(3) profile 里 comm 占多少比例？如果 comm > 30%，我会先 attack a2a，考虑 DeepEP V2 + Megatron `--overlap-moe-expert-parallel-comm`；如果 comm < 10%，那 bottleneck 不在这儿，可能是 activation recompute 过多或 grouped GEMM 小-M。基于三堵墙框架诊断，才能对症下药。举个我做过的案例……"

这种回答会立刻把面试从"背 flag"变成"平等技术讨论"。

== 面试故事的 anti-pattern

+ *"我调 flag"*：说了等于没说
+ *"我用 A100"*：硬件不是你的贡献
+ *"我实现了 XX"*（但显然是抄框架源码）：面试官一问细节就露馅
+ *无数字*："训练变快了" 没意义，说"MFU 从 X 到 Y" 或 "step time 从 A 到 B"
+ *技术堆砌*：一次提 10 个优化，每个都浅——不如深挖 1-2 个
+ *推给团队*："这是我们团队做的"——一定要有*你个人*的技术贡献

== 一个完整的 STAR 故事示例

*Situation*: "上一份工作里我们训 30B dense LLM on 128 H100，用 Megatron TP=8/PP=2/DP=8, seq=8K, BF16。刚起手时 MFU 只 32%，比 Megatron 官方 benchmark 差 15 个点。"

*Task*: "我负责找出并 close 这个 gap，目标 45% MFU。"

*Action*:
"跑了 nsys 一 step profile，看到几个问题：
+ DDP grad AR 完全串行在 backward 结束后，overlap 只 60%——查 code 发现 bucket_cap=25MB 默认；
+ TP forward 时 AR kernel 占 SM，GEMM idle 5%；
+ activation 显存 70/80 GB，被迫开 full recompute，compute overhead 33%。

对应 (1) 调 bucket 200MB + grad-as-view + persistent_workers；(2) 升级 TE 到支持 async TP (`--tp-comm-overlap`)；(3) 关 full recompute 改 selective + FP8 activation，activation 降到 45 GB。"

*Result*: "MFU 32% → 47%，step time -32%，训一个 epoch 从 5 天 → 3.4 天。团队后来把这三个 patch 合到 baseline recipe，其他模型也受益。"

_每一步都能追问细节：面试官问"为什么 FP8 activation 而不是 offload？"你要能答"activation 大头是 residual + LN input，offload 走 PCIe 有 CPU 与 dataloader 竞争风险；FP8 是 kernel 内 quant/dequant 无 PCIe，且 Hopper native 支持。"_

== 结语

分布式训练面试问的从来不是"你会不会调 flag"，而是"你能不能诊断 + 有 delta 贡献"。这本书从头到尾都在建立诊断框架 (三堵墙、Roofline、通信量公式)，让你面对任何具体问题都能*先分析再动手*。

面试前一晚：翻附录 D 的 100 题 + 附录 E 的对比表 + 附录 A 的数字速查。面试当场：先问清楚 setup，用三堵墙 anchor，用 STAR 讲你做过的事。

祝拿 offer。
