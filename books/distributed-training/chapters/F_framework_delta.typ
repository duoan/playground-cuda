#import "../template.typ": *

= 附录 F：Megatron 上的改造 —— 教你回答"你们框架改了什么"

面试里最能拉开档次的一类问题：

#quote(block: true)[
  "你们用 Megatron / DeepSpeed，那这些框架已经把 TP/PP/EP 都做好了、把 FlashAttention 集成了、把 async ckpt 也做了。你在上面*具体改造了什么*，让模型训得又快又稳？"
]

只答"我们用了 Megatron 官方 recipe"或"我们调了几个 flag"，会被立刻判定为"tuner"，不是"framework builder"。真正加分的回答是把你的改造分*四个维度*说清楚：

+ *算子层*：kernel/GEMM/attn/norm/a2a — 哪些是自研，哪些是替换上游
+ *算法层*：optimizer/router/schedule/packing — 哪些是"跟随论文改"，哪些是自己 tune 出来的
+ *稳定性*：spike skip/NaN heal/grad protect/ckpt guard — 让长训不炸的"保护罩"
+ *监控 + 基础设施*：从 grad_norm dashboard 到 elastic restart 到 topo-aware placement

如果你的岗位还覆盖 post-training，同一个问题会以"你们用 verl，那 verl 都做好了，你做了什么"的形式再问一遍 —— 第六节按同样四个维度给了 RL 栈的版本。

这一附录把 2023–2025 主流大厂 (Meta / DeepSeek / Kimi / Qwen / OpenAI-inferred / Anthropic-inferred) 公开的框架改造拆解，配 patch 伪代码，形成你可以拿去说 "这个我做过 / 我知道怎么做" 的具体清单。

#warn[
  这里的每一项都是*公开可查*的技术。你在真实面试中要说的应当是*你实际做过的*那部分。这一章的目的是让你能*认出*一个改造属于哪个维度、量级多大、trade-off 在哪；不是让你"背下来假装做过"—— 面试官一追问细节就会露馅。
]

== 全景图：四个维度的改造清单

#figure(
  table(
    columns: (auto, 1fr, 1.4fr),
    stroke: 0.4pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*维度*], [*涉及范围*], [*一句话定位*],
    [*算子层*],  [kernel / GEMM / attn / norm / a2a],
                 [每个 op 都能再快 X%，是"物理加速"],
    [*算法层*],  [optimizer / router / schedule / packing],
                 [跟随论文 + 自己 tune，是"每 token 学更多"],
    [*稳定性*],  [spike skip / NaN heal / ckpt guard],
                 [长训不炸的"保护罩"，决定能否训完],
    [*监控 + Infra*], [dashboard / elastic / topology],
                     [7×24 无人值守的"地基"],
  ),
  caption: [框架改造的四个维度。上层做加速，下层做兜底；越往下越决定"是否能训完"，越往上越决定"训多快"。面试里 4 个维度都要能各举 1-2 个例子。],
) <fig-4-dims>

== 一. 算子层改造

Megatron 官方已经集成 FlashAttention、Transformer Engine (TE)、grouped GEMM。但生产集群还有 5 类高频改造：

=== 1.1 替换 attention kernel

*基线*：TE 内的 `DotProductAttention` (调 cuDNN backend 或 FlashAttention-2)

*改造*：
- *FlashAttention-3* (Hopper only, 2024 Q3+ 版本)：H100 上 forward ~1.5–2× 更快。但 varlen path 到 2024 Q4 才稳。生产集群需 backport patch，加 shape guard fallback。
- *Ring FlashAttention* (Zhu 2023)：长上下文时 CP 场景默认 kernel。Megatron `--context-parallel-size` 默认调 cuDNN，替换成 Ring FA 后 128K seq 快 30%。
- *TriDao ThunderKittens / Mamba-attention*：experimental，用于 SSM 或 hybrid 结构。

*Patch 位置*：`megatron/core/transformer/attention.py::DotProductAttention.forward`，加 `attn_backend` config。

*Trade-off*：新 kernel 常有 shape / dtype 限制 (如 head_dim ∈ {64, 128})。要写 fallback：
```python
if attn_backend == "flashattn3":
    try:
        out = fa3_kernel(q, k, v, ...)
    except (RuntimeError, ValueError):
        # shape/dtype not supported → fallback
        out = fa2_kernel(q, k, v, ...)
        _fa3_fallback_counter.inc()   # track how often we fallback
```

*收益*：H100 long-ctx +15–30% MFU。Kimi K2 报告 FA3 是 long-ctx 稳定 recipe 的关键。

=== 1.2 Fused RMSNorm / SwiGLU / RoPE

*基线*：TE `LayerNormLinear` (LN + linear fused)，`RotaryPositionEmbedding` 独立 kernel

*改造*：写*自己的* Triton fused kernel，把 "RMSNorm + Linear + SwiGLU gate + RoPE" 尽可能 fuse 成 1-2 个 kernel。

*为什么*：
+ TE fused kernel 是 CUDA C++，*难扩展*——想加个 QK-norm 或改 eps upcast 就得改 C++ 重编
+ Triton 是 Python，*快速迭代*——2 天能出一个 prototype
+ Fused RMSNorm+Linear+RoPE 三 op 合一，H100 上比 TE 快 8-12% (Kimi K2 报告)

*Patch*：把 `megatron/core/models/gpt/gpt_layer_specs.py` 里的 `norm_and_linear` module 换成自定义 Triton 版本，注册进 spec。

*典型 fused kernel 伪代码*：

```python
@triton.jit
def fused_rmsnorm_qkv_rope_kernel(
    X_ptr, W_ptr, cos_ptr, sin_ptr, Y_ptr,
    stride_x, stride_w, stride_y,
    H, D_head, N_head, eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1. Load X and compute RMSNorm (var in FP32)
    x = tl.load(X_ptr + offsets)
    x_f32 = x.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(x_f32 * x_f32) / H + eps)
    x_norm = (x_f32 * inv_rms).to(x.dtype)
    # 2. GEMM: qkv = x_norm @ W_qkv
    qkv = tl.dot(x_norm, W_qkv)
    # 3. Split into q, k, v and apply RoPE to q, k
    q, k, v = split(qkv)
    cos = tl.load(cos_ptr + ...)
    sin = tl.load(sin_ptr + ...)
    q = rope(q, cos, sin)
    k = rope(k, cos, sin)
    # 4. Write out
    tl.store(Y_ptr + q_offsets, q)
    ...
```

*Trade-off*：
- Triton 在 shape 变化时会 re-compile (200 ms+)，需要开 `@triton.autotune` 或 warmup
- 只对 BF16 优化；FP8 走 TE 走 C++
- kernel bug 更难 debug，需要单元测试对齐 unfused reference (`torch.allclose(out_fused, out_ref, atol=1e-3)`)

*收益*：H100 上 fused normalized MLP layer +8-12% wall-clock。

=== 1.3 Grouped GEMM 替换 for MoE

*基线*：Megatron 默认 MoE 每个 expert 独立 GEMM，`for e in experts: y_e = act(x_e @ W_e)`。expert 多时 (>32) launch overhead 大。

*改造*：换成 *grouped GEMM* —— 一次 kernel 处理所有 expert 的 GEMM。CUTLASS 3.5+ 原生支持，或用 `torch._grouped_mm` (Torch 2.4+)。

*Patch 位置*：`megatron/core/transformer/moe/moe_layer.py::SequentialMLP.forward`。

*典型对比*：

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 6pt,
  align: (left, right, right),
  [*方式*], [*expert 数*], [*fwd time / MoE layer*],
  [Loop of GEMM (baseline)], [256], [4.2 ms],
  [Grouped GEMM (CUTLASS)],  [256], [1.1 ms],
  [Fused with dispatch (DeepEP)], [256], [0.8 ms],
)

*收益*：DeepSeek-V3 报告 grouped GEMM 是 fine-grained expert (256+) 可行的前提；否则单纯 expert 数增加会被 launch overhead 吃掉。

*进阶*：把 dispatch a2a + grouped GEMM + combine a2a *完全 fuse* 成一个 mega-kernel（DeepEP V2 做的事）—— 但工程复杂度高，多数团队直接用 DeepEP，不自研。

=== 1.4 自研 all-to-all kernel（DeepEP-style）

*基线*：`torch.distributed.all_to_all_single` → 走 NCCL。NCCL a2a 用 20+ SM 做通信，抢 GEMM 的 SM。

*改造*：
- *DeepEP* (DeepSeek 开源，2024)：用 IBGDA (InfiniBand GPU-Direct Async) 让 GPU 直接发 RDMA verbs，不经过 CPU proxy。SM 占用降到 4-6。
- *NVSHMEM*：NVIDIA 官方类似方案，Megatron 2025 起集成。

*Patch 位置*：`megatron/core/transformer/moe/token_dispatcher.py::TokenDispatcher`。把 `dist.all_to_all_single` 换成 `deep_ep.dispatch()` + `deep_ep.combine()`。

*要点*：
+ 需要网络 driver 支持 IBGDA（Mellanox MLNX_OFED 5.4+）
+ 需要 GPU peer memory access 权限（root/CAP_SYS_ADMIN 或 privileged container）
+ intra-node vs inter-node 用不同 code path（intra 走 NVLink direct，inter 走 IBGDA）—— hierarchical a2a

*收益*：MoE MFU 从 35% → 51% 是 Kimi / DeepSeek 报告的典型数字。DeepEP V1 → V2 又能再 +15%。

*Trade-off*：与 CUDA graph 不完全兼容（IBGDA 需要 host-side unblocking event），需要 mode switch。

=== 1.5 CUDA graph capture

*基线*：Megatron 2024 起支持 `--enable-cuda-graph`，但仅覆盖 forward 主 path。

*改造*：
- 把 forward+backward 都进 graph
- Varlen 场景按 seq_len bucket 分别 capture (bucket 数 5-8)
- 与 dataloader stream 隔离，避免 CPU launch 抖动打断 graph replay

*Patch 位置*：`megatron/training/training.py::train_step`，加 graph capture wrapper：

```python
if step == warmup_step + 1 and use_cuda_graph:
    # capture on first "real" step (after warmup, shapes stable)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = model(static_input)
        loss.backward()
    _graph_cache[bucket_id] = graph

if step > warmup_step + 1:
    static_input.copy_(real_input)
    graph.replay()      # replaces train_step body
```

*收益*：小 batch (< 4 seq/rank) 或短 seq (< 2K) 场景 CPU launch overhead 从 12% → 1%。GPT-4 rumored 用到，Meta Llama-3 405B 训练用类似技术。

*Trade-off*：dynamic shape / control flow 全崩，dropout mask 需要 external RNG state。开 graph 之前必须先能保证 shape 稳定 —— packing 是前提。

=== 算子层改造的整体清单

#figure(
  table(
    columns: (auto, 1.5fr, 1.2fr, auto),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left, center),
    [*改造点*], [*基线 (Megatron 官方)*], [*生产改造*], [*典型收益*],
    [Attention kernel], [TE cuDNN / FA2],       [FA3 + fallback guard],           [+15-30% MFU],
    [Norm+Linear fused], [TE C++],              [Triton (norm+linear+RoPE)],      [+8-12%],
    [MoE GEMM],        [loop of GEMM],          [Grouped GEMM (CUTLASS)],         [1 layer 3-4×],
    [MoE a2a],         [NCCL a2a],              [DeepEP / NVSHMEM (IBGDA)],       [MoE +15-20% MFU],
    [Graph capture],   [部分 forward path],     [完整 fwd+bwd + varlen buckets],  [小 batch +10-15%],
    [Precision cast],  [TE 内部管理],           [关键路径手动 upcast (LN var, QK^T accum)], [训练稳定性],
    [Communication SM], [NCCL 默认],            [`NCCL_MAX_NCHANNELS=8` + 自定 topology], [±5% MFU],
  ),
  kind: table,
  caption: [算子层的 7 类常见改造。每一项都可以做深，"哪一项是你亲手写的" 是面试差异化关键。],
) <table-op-delta>

#interview[
  面试深挖：*"你说你写了 fused Triton kernel，遇到什么坑？"*

  正确答法（体现真做过）：
  + *shape 不稳定 → autotune 缓存失效*：Triton 遇新 shape 重编 200 ms，varlen 场景灾难。修：加 shape rounding（`M = ceil(M / 128) * 128`）或 pre-warm 所有可能 shape。
  + *BF16 accumulator drift*：Triton 默认 accum 用输入 dtype。RMSNorm var 要 explicit `tl.float32` upcast，否则 loss slow-degrade。
  + *不同 head_dim 需要不同 BLOCK*：写 dispatch table，`if D == 128: BLOCK_D = 128; elif D == 64: BLOCK_D = 64`。
  + *nsys 抓不到 Triton 内部 sub-kernel*：只能看到一个 Triton launch。需要用 `TRITON_KERNEL_DUMP_DIR` 落 PTX 再逐行分析。
]

== 二. 算法层改造

算子改造是"物理加速"，算法改造是"每 token 学更多"。这一节更影响 loss curve 与收敛速度。

=== 2.1 Optimizer 替换：AdamW → Muon / Lion

*基线*：Megatron `--optimizer adam`（默认 AdamW，`beta=(0.9, 0.95)`, `wd=0.1`）

*改造*：
- *Muon* (Kimi K2, 2025)：2D+ 权重用 Muon (Newton-Schulz orthogonalize)，1D 权重仍 AdamW。Kimi 报告 loss 曲线更平滑，无 spike。
- *Lion* (Google 2023)：`sign(momentum)` 直接 apply，显存少 50%。少数团队试。

*Patch 位置*：`megatron/core/optimizer/__init__.py::get_megatron_optimizer`。新增 `MuonOptimizer` class，参数分组 (2D → Muon, 1D → AdamW)：

```python
def build_muon_optimizer(model):
    matrix_params, vector_params = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in n:
            matrix_params.append(p)
        else:
            vector_params.append(p)
    return CompositeOptimizer([
        Muon(matrix_params, lr=2e-3, beta=0.95, wd=0.1),
        AdamW(vector_params, lr=4e-4, betas=(0.9, 0.95), wd=0.1),
    ])
```

参考实现见 `src/distributed_training/12_optimizers.py`。

*Trade-off*：
- Muon 与 FSDP FULL_SHARD 不兼容（Newton-Schulz 要 full matrix），只能 ZeRO-1 → 参数复制 → 每卡显存翻倍
- Kimi K2 因此选 ZeRO-1 + PP + EP，*不用 FSDP*
- Muon 需要 4-5× 大的 LR，需要重新调 warmup + wd 组合

*面试点*：*"为什么 Kimi 用 Muon 而 DeepSeek 用 AdamW？"* 答：Kimi 押注 Muon 的稳定性 + MuP scaling；DeepSeek 已经用 FP8+ZeRO+MoE 组合叠了太多变量，不想再引入 Muon 的额外风险。都是合理选择，取决于团队 risk appetite。

=== 2.2 MoE 路由：aux-loss → aux-loss-free

*基线*：Megatron `--moe-aux-loss-coeff 1e-2`，用 Shazeer 2017 的 load-balance loss。

*改造*：DeepSeek-V3 提出的 *bias-based aux-loss-free* balancing。每个 expert 加动态 bias，"最近少被选的" bias 自动加大 → 下次更容易被选。无需在 loss 里加 aux 项。

*Patch 位置*：`megatron/core/transformer/moe/router.py::TopKRouter.forward`：

```python
# baseline
scores = softmax(logits)
topk_scores, topk_idx = scores.topk(k, dim=-1)
aux_loss = compute_load_balance_loss(topk_idx)  # ← add to total loss

# aux-loss-free
scores = softmax(logits) + self.expert_bias        # ← bias added to logits
topk_scores, topk_idx = scores.topk(k, dim=-1)
# update bias every step (EMA of imbalance)
with torch.no_grad():
    hit_rate = compute_hit_rate(topk_idx)           # per-expert hit fraction
    imbalance = hit_rate - hit_rate.mean()
    self.expert_bias -= self.bias_lr * torch.sign(imbalance)
```

*收益*：DeepSeek 报告 main loss 收敛快 20-30%（没有 aux 干扰主 loss 梯度），expert 利用率 CV \<5%。

*Trade-off*：
- `bias_lr` 需要 tune（DeepSeek 用 1e-3，太大会震荡）
- `expert_bias` 需要放 checkpoint (每 expert 一个 scalar → 千个数)
- 与 `--moe-expert-capacity-factor` 交互复杂，需要禁用 capacity drop 或改成 soft drop

=== 2.3 Multi-Token Prediction (MTP)

*基线*：Megatron 默认预测下一个 token。

*改造*：额外加 $n$ 个"预测未来第 $n$ 个 token"的 head (DeepSeek-V3 用 n=1，即 MTP head 预测 t+2)。Loss = main NTP loss + `α * MTP loss`。

*Patch 位置*：`megatron/core/models/gpt/gpt_model.py::GPTModel`，加 `mtp_head` module + 修改 loss 计算。

```python
class GPTModelWithMTP(GPTModel):
    def __init__(self, ...):
        super().__init__(...)
        self.mtp_head = MTPHead(hidden_size, vocab_size, depth=1)

    def forward(self, ..., labels):
        hidden = self.transformer(...)
        main_logits = self.lm_head(hidden)
        loss_main = F.cross_entropy(main_logits, labels)

        # MTP: predict labels shifted by +1 more
        mtp_logits = self.mtp_head(hidden, main_logits)
        loss_mtp = F.cross_entropy(mtp_logits, labels_shifted)

        return loss_main + self.mtp_alpha * loss_mtp
```

*收益*：
- 训练时：main loss 略降（MTP 相当于额外 supervision，正则化效果）
- 推理时：MTP head 可作 speculative decoding 的 draft，2-3× inference 加速

*Trade-off*：模型参数增 1-2%（一个 head + 一个 transformer block），显存 & compute 增。

=== 2.4 MLA 实现（DeepSeek 系）

*基线*：Megatron 默认 MHA / GQA。

*改造*：实现 *Multi-Head Latent Attention*：KV 压缩到低秩 $c_"kv"$，attention 时上采样。K 分成 rotated (RoPE) 和 non-rotated 部分。

*Patch 位置*：新建 `megatron/core/transformer/mla_attention.py`，重写 QKV projection 逻辑 + 修改 KV cache（推理时）。

*关键代码骨架*：
```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, ...):
        # down-project to compressed KV
        self.W_dkv = ColumnParallelLinear(H, d_c + d_rope)
        # up-project back (per head)
        self.W_uk = ColumnParallelLinear(d_c, n_head * d_head_nope)
        self.W_uv = ColumnParallelLinear(d_c, n_head * d_head)
        # Q: partial nope + partial rope
        self.W_q  = ColumnParallelLinear(H, n_head * (d_head_nope + d_rope))

    def forward(self, x):
        c_kv, k_rope = self.W_dkv(x).split([d_c, d_rope], dim=-1)
        # ↑ this is what gets cached at inference (much smaller than K/V)
        k_nope = self.W_uk(c_kv).view(..., n_head, d_head_nope)
        v      = self.W_uv(c_kv).view(..., n_head, d_head)
        q_nope, q_rope = self.W_q(x).view(..., n_head, -1)
                                        .split([d_head_nope, d_rope], dim=-1)
        # apply RoPE only to the rope-partition
        q_rope = rope(q_rope, ...); k_rope = rope(k_rope, ...)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.unsqueeze(-2).expand(-1, -1, n_head, -1)], dim=-1)
        return flash_attn(q, k, v)
```

*Trade-off*：
- TP 切分复杂：`W_uk`/`W_uv` 是 per-head，需要沿 head 维 column parallel
- FA kernel 需要接受 rotated + non-rotated 拼接的 head_dim（FA3 才原生支持）
- 训练/推理代码 diverge (KV cache 形态不同)，需要双 code path

*收益*：KV cache 缩到 GQA-8 的 1/4，128K 推理场景内存瓶颈解除。训练 loss 与 GQA 打平。

=== 2.5 Sequence packing 精细化

*基线*：Megatron 支持 `--packed-sequence`，按 fixed length 顺序拼接。

*改造*：
- *BFD (Best-Fit-Decreasing)* 或 *FFD (First-Fit-Decreasing)* packing，减少 padding。参考实现 `src/distributed_training/10_seq_packing.py`。
- 加 *cost-based balancing*：不同 batch 的 attention cost ∝ $sum_i s_i^2$（因为 attention 是 $O(S^2)$）。DP rank 之间平衡 cost 而不是 token 数。
- *cross-document attention mask*：pack 里的不同 document 之间必须 mask 隔离，用 FA 的 varlen path (`flash_attn_varlen_func`) 或 block-diagonal mask。

*Patch 位置*：`megatron/core/datasets/gpt_dataset.py::_pack_documents`，改 packing 策略。

*Trade-off*：
- BFD 需要 look-ahead 一个 buffer（如 1000 个 sample），流式训练需要 online BFD
- cost balancing 会让 rank 间收到不同 seq 数 → hidden state 数不同 → 需要 padding to max 或者 uneven collective（gloo 支持，NCCL 需要 pad）

*收益*：Padding 率从 15% → 2%，等效 +13% throughput。Meta Llama-3 训练报告 packing 是 105B → 405B 训完的关键。

=== 2.6 LR schedule tweak：WSD + mini-restart

*基线*：Megatron 默认 `--lr-decay-style cosine`。

*改造*：
- WSD (Warmup-Stable-Decay)，见 P3 章
- Mini-restart：训到 60% 时 LR × 0.3 rewarm 500 步再 stable
- Curriculum-aware LR：切换 data phase 时短 warmup 200 步

*Patch 位置*：`megatron/core/optimizer_param_scheduler.py::OptimizerParamScheduler`：

```python
def get_lr(self, step):
    if step < self.warmup:
        return self.peak * step / self.warmup

    # WSD stable
    if step < self.decay_start and step < self.rewarm_step:
        return self.peak

    # Mini-restart
    if self.rewarm_step <= step < self.rewarm_step + self.rewarm_warmup:
        prog = (step - self.rewarm_step) / self.rewarm_warmup
        return self.peak * (self.rewarm_ratio +
                            (1 - self.rewarm_ratio) * prog * 0)  # ramp
    # ... etc
```

*收益*：Kimi K2 报告 mini-restart 让 loss 曲线在 60% training 时不停滞（原本会有平台期）。

== 三. 稳定性改造

这一部分决定"能不能训完 30 天不炸"。生产集群里，稳定性问题永远是 P0，其次才是 MFU。

=== 3.1 Loss spike 自动 skip

*基线*：Megatron 默认 grad clip = 1.0，spike 时 clip 但 optim step 仍执行。

*改造*：
```python
# in train_step, right before optim.step()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

# Track EMA of grad_norm for outlier detection
_grad_norm_ema.update(grad_norm.item())
threshold = _grad_norm_ema.median * 10   # 10× median

if grad_norm > threshold or torch.isnan(grad_norm) or torch.isinf(grad_norm):
    # SKIP this update
    logger.warning(f"[step {step}] skip: grad_norm={grad_norm} "
                   f"vs 10×median={threshold}")
    _skip_counter.inc()
    optim.zero_grad()
    # OPTIONAL: log the offending batch for post-mortem
    torch.save({"input_ids": batch, "step": step},
               f"/spike_logs/step_{step}.pt")
    return  # skip optim.step + scheduler.step
```

*Patch 位置*：`megatron/training/training.py::train_step`

*收益*：Meta OPT-175B 报告 45+ 次 manual restart，改成 auto-skip 后可零人工干预。Anthropic Claude 训练报告类似机制。

*Trade-off*：
- skip 太多会让 loss 收敛慢（每个 skip 相当于 free 一个 batch）
- 需要 tune threshold：太严 (5×) 会误 skip 正常 spike，太松 (100×) 会漏掉真正的 divergence
- 需要 alert：连续 3 步 skip → page 值班人

=== 3.2 NaN / Inf 自愈

*基线*：NaN 出现就 crash。

*改造*：分层 NaN 处理：

```python
# 1. Detect NaN in loss
if torch.isnan(loss) or torch.isinf(loss):
    # 2. Rollback to previous "clean" state
    if _clean_state_available():
        model.load_state_dict(_clean_state)
        optim.load_state_dict(_clean_optim_state)
        logger.warning(f"[step {step}] NaN loss, rolled back to step {step-N}")
        continue

    # 3. If no clean state (very early), just skip
    optim.zero_grad()
    _nan_counter.inc()
    if _nan_counter.value > 5:
        # ... hard crash, need human intervention
        raise RuntimeError("Too many consecutive NaNs")

# 4. Every K clean steps, snapshot as new "clean" state (in memory)
if step % 100 == 0 and grad_norm < threshold:
    _clean_state = {k: v.clone() for k, v in model.state_dict().items()}
    _clean_optim_state = deepcopy(optim.state_dict())
```

*Patch 位置*：`megatron/training/training.py::train_step`

*收益*：训练一次 spike 不会导致重启（原本要从 checkpoint restore 20 min，现在 rollback in-memory 200 ms）。DeepSeek-V3 训练报告类似机制。

*Trade-off*：
- 需要额外显存存 clean state (整个 model + optim ~ 20 GB for 70B on H100)
- 若 spike 是数据引起的（poisoned batch），rollback + skip 后需要 skip 该 batch 避免立即再触发

=== 3.3 Gradient 保护 (unit-norm + skip huge)

*基线*：`clip_grad_norm_(model.parameters(), max_norm=1.0)` 全局 clip。

*改造*：
- *Per-layer clip* (in addition to global)：某一层 grad 特别大时单独 clip 该层，不影响全局 norm
- *Gradient masking on outlier*：某个 param 的 grad 单元 > $mu + 10 sigma$，替换为 median
- *No-op step*：`||grad||` > 1000 时直接 optim.zero_grad() 不 step

```python
# Per-layer clip
for name, p in model.named_parameters():
    if p.grad is None: continue
    local_norm = p.grad.norm()
    if local_norm > 10.0:   # 10 is per-layer cap
        p.grad.mul_(10.0 / local_norm)
        _layer_clip_counter[name].inc()

# Global clip after
total_norm = clip_grad_norm_(model.parameters(), 1.0)

# Log which layer contributed most to spike
if total_norm > 5.0:
    top = sorted(_layer_clip_counter.items(), key=lambda x: -x[1].value)[:3]
    logger.warning(f"[step {step}] high grad_norm={total_norm}, "
                   f"top layers: {top}")
```

*收益*：Llama-3 训练报告用类似 per-layer clip，某一层的病态 grad（如 embedding norm 出问题）不会污染全局。

=== 3.4 Checkpoint double-buffer + async save

*基线*：Megatron 支持 `--async-save`，但 optim state save 时会 block train 1-2 min（大 model）。

*改造*：*三级 checkpoint*：

+ *L1 (in-memory backup on peer)*：每 100 step，state dict → 隔壁 node CPU DDR。约 3 s（NVLink 到 CPU 再到 IB）。用于 in-flight NaN rollback。
+ *L2 (disk async)*：每 1000 step，flush 到本地 NVMe。约 30 s，与 train 并行。
+ *L3 (remote object store)*：每 10000 step 或 6h，upload 到 S3 / OSS，用于灾备。

*Patch 位置*：`megatron/training/checkpointing.py::save_checkpoint`

*架构*：

#figure(
  align(center, op-stack(steps: (
    ("train step N",         "GPU busy",           "full"),
    ("D2H → CPU pinned",     "background",         "shard-h"),
    ("copy peer CPU (L1)",   "NVLink+IB, ~3 s",    "comm"),
    ("NVMe async (L2)",      "background, ~30 s",  "shard-s"),
    ("upload S3 (L3)",       "hourly",             "comm"),
  ), width: 7.0, cell-h: 0.55)),
  caption: [三级 checkpoint 架构。L1 覆盖 in-flight NaN rollback，L2 覆盖单 node 挂，L3 覆盖整个集群失效。生产训练里 L1+L2 是标配。],
) <fig-ckpt-3level>

*收益*：
- Checkpoint 阻塞 train 时间 5 min → 15 s（Meta OPT 论文数字）
- 恢复时间 20 min → 30 s（in-memory recovery）
- Kimi K2 报告 30 天训练 zero manual intervention 是靠 L1+L2

*Trade-off*：
- L1 需要选 peer node（跳过同 node 避免同时挂）
- 需要 SSD 冗余 (RAID) 或每 node 独立 disk

=== 3.5 Deterministic training

*基线*：Megatron 未强制 deterministic。dataloader RNG, dropout, kernel non-determinism 都会让"同 seed 复现"失败。

*改造*：
```python
# In training entry point:
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# For distributed shuffling
gen = torch.Generator()
gen.manual_seed(base_seed + dp_rank)   # per-DP-rank seed

# Log all RNG state to checkpoint
ckpt["python_rng"] = random.getstate()
ckpt["numpy_rng"]  = np.random.get_state()
ckpt["torch_rng"]  = torch.get_rng_state()
ckpt["cuda_rng"]   = torch.cuda.get_rng_state_all()
ckpt["gen_rng"]    = gen.get_state()
```

*收益*：debug 时能复现问题（哪个 step 出 NaN，rerun 保证 hit 同一 batch）。生产 tune 时可以做 controlled experiment。

*Trade-off*：deterministic mode 有 3-5% 性能损失（因为禁用了 non-det kernel），生产训练里通常*只在 debug 期开*，正式 run 关掉。

=== 稳定性改造清单

#figure(
  table(
    columns: (auto, 1.4fr, 1.4fr, auto),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left, center),
    [*改造点*], [*基线*], [*生产改造*], [*收益*],
    [Loss spike],  [手动重启],           [auto skip step + alert],           [-90% manual intervention],
    [NaN 处理],    [crash],              [in-memory rollback 3 层],          [恢复时间 20 min → 30 s],
    [Grad clip],   [global norm 1.0],    [per-layer + skip huge],            [避免单层污染全局],
    [Checkpoint],  [同步 5 min block],   [async + L1/L2/L3 三级],            [overhead 5% → 0.5%],
    [Determinism], [关],                 [debug 时开，全 RNG state 存 ckpt], [可复现，可 A/B],
    [Elastic restart], [无],             [torchrun --max-restarts + 自动重连], [硬件挂不停 job],
  ),
  kind: table,
) <table-stability-delta>

== 四. 监控改造

"能不能观测到问题"决定"问题多快被发现"。Megatron 默认只 print loss 到 stdout，生产集群远远不够。

=== 4.1 训练主指标 dashboard

*基线*：Megatron log 每 N step 一次 loss + grad_norm 到 tensorboard。

*改造*：至少监控这 15 项（Meta / DeepSeek 报告都提到）：

#figure(
  table(
    columns: (auto, 1fr, 1.4fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*类别*], [*指标*], [*用途*],
    [Loss],     [train loss, val loss, per-domain loss], [收敛监控],
    [Grad],     [grad_norm (global, per-layer), grad_var], [稳定性],
    [Weight],   [weight_norm per param, weight/init 比值], [drift 监控],
    [Activation], [per-layer activation min/max/mean/std], [NaN 提前预警],
    [Optim],    [effective lr, m/v ratio, `sqrt(v)`],      [Adam 健康度],
    [Throughput], [MFU, HFU, samples/sec, tokens/sec/GPU], [效率],
    [Memory],   [allocated GB, reserved GB, peak, fragmentation], [OOM 预警],
    [Comm],     [AR bandwidth, a2a bandwidth, P2P latency], [通信健康],
    [Rank imbalance], [per-rank step time p50/p99, straggler idx], [straggler 检测],
    [Hardware], [GPU util%, HBM util%, NVLink util%, IB util%], [硬件健康],
    [Temperature], [GPU temp, throttle events],               [降频预警],
    [ECC error], [SBE/DBE count per GPU],                    [硬件寿命],
    [NCCL], [error count, timeout events, comm hang],        [网络故障],
    [Dataloader], [buffer depth, worker idle time, sample skip], [I/O 瓶颈],
    [Skip step], [count of skipped steps + reason],           [稳定性 audit],
  ),
  kind: table,
  caption: [生产训练 dashboard 必备 15 类指标。多数团队实现是 Prometheus + Grafana + DCGM exporter + 自研 Python exporter。],
) <table-monitor-metrics>

*Patch 位置*：新建 `megatron/training/monitor.py`，注册各类指标 exporter；在 `train_step` 结尾统一发到 Prometheus pushgateway。

参考实现见 `src/distributed_training/15_training_monitor.py`（简化版）。

=== 4.2 Straggler 检测

*基线*：无。慢 rank 拖慢整体，但没告警。

*改造*：
```python
def train_step(step):
    t0 = time.time()
    # ... normal train step ...
    t_local = time.time() - t0

    # Gather per-rank step time
    all_times = [torch.zeros(1) for _ in range(world_size)]
    torch.distributed.all_gather(all_times,
                                  torch.tensor([t_local], device="cuda"))
    times = [t.item() for t in all_times]
    median = statistics.median(times)

    for r, t in enumerate(times):
        if t > median * 1.3:   # 30% slower than median
            _straggler_counter[r].inc()

    # Escalate: if same rank straggler for 10 consecutive steps
    for r, cnt in _straggler_counter.items():
        if cnt.recent(10) >= 10:
            logger.error(f"[straggler] rank {r} slow for 10 steps, "
                         f"check GPU health!")
            emit_alert(f"straggler-{r}")
```

*Patch 位置*：`megatron/training/training.py::train_step`

*收益*：一个 slow GPU（吃 ECC）会拖慢整个 job 30%+。早期发现（10 步内）可以 drain 该 node 换机器。Meta MegaScale 论文详细描述。

=== 4.3 NCCL 拓扑探测

*基线*：Megatron 假设 rank ↔ GPU 是简单顺序映射。

*改造*：训练开始前跑 `nccl-tests`，画出实际 all-reduce 带宽矩阵，检查：
- 同一 node 8 卡应该 NVLink 300+ GB/s
- 跨 node 应 IB 100 GB/s
- 若某 pair 只跑到 40 GB/s → 网络故障或 topology 错

```python
def probe_topology():
    """Run all_reduce on each pair and log bandwidth."""
    results = {}
    for i in range(world_size):
        for j in range(i + 1, world_size):
            group = dist.new_group([i, j])
            if rank in [i, j]:
                bw = measure_ar_bandwidth(group)
                results[(i, j)] = bw

    # Report anomalies
    intra_node = [bw for (i, j), bw in results.items()
                  if i // 8 == j // 8]
    inter_node = [bw for (i, j), bw in results.items()
                  if i // 8 != j // 8]
    logger.info(f"intra-node AR: median={median(intra_node):.0f} GB/s")
    logger.info(f"inter-node AR: median={median(inter_node):.0f} GB/s")

    for pair, bw in results.items():
        expected = 300 if pair[0] // 8 == pair[1] // 8 else 100
        if bw < expected * 0.7:
            logger.warning(f"pair {pair} slow: {bw} GB/s (expected {expected})")
```

*Patch 位置*：`megatron/training/initialize.py::initialize_megatron`（在 model init 之前）

*收益*：训练开始前发现慢 node 并 fail-fast，避免训 1h 后才发现 straggler。Meta MegaScale 论文的 topology-aware placement 就是这套。

=== 4.4 Loss / gradient anomaly detection

*基线*：无。用户肉眼看 tensorboard。

*改造*：online anomaly detection（简单版本）：
```python
class OnlineAnomalyDetector:
    def __init__(self, window=100):
        self.buffer = collections.deque(maxlen=window)

    def check(self, value, name):
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(value)
            return None

        median = statistics.median(self.buffer)
        mad = statistics.median([abs(v - median) for v in self.buffer])
        z_score = 0.6745 * (value - median) / (mad + 1e-8)   # MAD-based z

        self.buffer.append(value)

        if abs(z_score) > 6:
            return f"{name} anomaly: {value:.4f} (z={z_score:.1f})"
        return None

# in training loop
det = OnlineAnomalyDetector()
for step in range(...):
    loss = model(...)
    msg = det.check(loss.item(), "loss")
    if msg:
        alert_slack(msg)     # or pagerduty for grad
```

*收益*：loss spike 出现 30 s 内自动 alert，值班人可以在 2 分钟内决定是否 rollback / abort。

*Trade-off*：MAD-based z 对突然分布 shift（比如 curriculum 换）也会 alert，需要在 phase 切换时 reset buffer。

=== 监控改造清单

#figure(
  table(
    columns: (auto, 1.4fr, 1.5fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*改造点*], [*基线*], [*生产改造*],
    [Dashboard],     [loss 到 tensorboard],          [15+ 类指标 → Prometheus + Grafana],
    [Straggler],     [无],                           [gather per-rank step time + auto alert],
    [Topology probe], [无],                          [启动前 nccl-tests 全对 + anomaly report],
    [Anomaly detect], [无],                          [MAD-based z on loss / grad_norm],
    [Alert],         [无 / stdout],                  [Slack / PagerDuty 分级 (info / warn / page)],
    [Post-mortem],   [无],                          [spike batch 存盘 + step log 5 min 前后],
  ),
  kind: table,
) <table-monitor-delta>

== 五. Infra + 数据/系统接入改造

这一节是"能不能起手就能训 30 天"的地基。

=== 5.1 RDMA / IBGDA / topology-aware placement

*基线*：Megatron 假设网络拓扑理想。生产集群 rack 层次、SU 层次会让某些 rank pair 更慢。

*改造*：
- 用 `nvidia-smi topo -m` 获取 intra-node topology
- 用 UFM (InfiniBand fabric manager) 拿 inter-node topology
- Rank 分配时把 TP group 放同 NVLink 域内，DP group 尽量跨 rack 分散（fault tolerance），PP group 沿 rack 排列（P2P 局部化）

*Patch 位置*：`megatron/core/parallel_state.py::initialize_model_parallel`。默认按顺序切 rank，改成 topology-aware：

```python
def initialize_model_parallel_topo_aware(tp, pp, dp):
    # Read topology
    topo = read_gpu_topology()  # {rank: {"node": N, "rack": R, "su": S}}
    # Group ranks so that:
    #   TP ranks share same node
    #   PP ranks are within same rack (P2P locality)
    #   DP ranks span racks (fault isolation)
    tp_groups = group_by_node(topo, size=tp)
    pp_groups = interleave_racks(tp_groups, size=pp)
    dp_groups = ...
```

*收益*：Meta MegaScale 报告 topology-aware placement 让 8K H100 集群通信 -20%，故障 blast radius 减半。

=== 5.2 Elastic training

*基线*：Megatron 无 elastic。一个 node 挂，整个 job 挂，需要 SLURM 重排。

*改造*：`torchrun --max-restarts=100 --standalone` + 自定 rendezvous backend：
- Node 挂 → torchrun 检测到 heartbeat 丢失 → 剩余 node 重连 rendezvous
- 从最新 L1/L2 checkpoint 恢复
- 若 hot spare 可用，自动 include
- 若无 spare，reshape (DP−1) 继续训（需要 gradient accumulation 补偿）

*Patch 位置*：`torchrun` wrapper + `megatron/training/initialize.py` 加 dynamic mesh support。

*收益*：Meta MegaScale 报告 90% 硬件故障可以 elastic 恢复（无人工），mean-time-to-recovery 从 20 min → 3 min。

*Trade-off*：需要 checkpoint 支持 reshape（DCP sharded checkpoint 才行，`torch.save` 的老 ckpt 不支持）。

=== 5.3 Resumable dataloader

*基线*：Megatron `--data-path` + `--split` 假设从头开始或从 step N 精确恢复。dataloader RNG state 存 checkpoint。

*改造*：
- *Streaming*：数据在 S3 / OSS，边读边训，不预下载
- *Fault tolerant*：某个 shard 读失败 → skip 并 alert，不阻塞训练
- *Resumable at any step*：checkpoint 里存 `(shard_id, offset, RNG_state)`，恢复精确到 sample 级
- *Multi-source mixing*：不同数据源按 weight 采样，weight 可以中途改（curriculum）

```python
class StreamingResumableLoader:
    def __init__(self, shards, weights, seed):
        self.shards = shards
        self.weights = weights
        self.rng = numpy.random.default_rng(seed)
        self.cursor = {s: 0 for s in shards}

    def __iter__(self):
        while True:
            source = self.rng.choice(len(self.shards), p=self.weights)
            shard = self.shards[source]
            try:
                sample, self.cursor[shard] = read_next(shard, self.cursor[shard])
            except IOError:
                _shard_error_counter[shard].inc()
                continue   # skip broken shard, don't block
            yield sample

    def state_dict(self):
        return {"cursor": self.cursor, "rng_state": self.rng.bit_generator.state}
```

*Patch 位置*：新建 `megatron/data/streaming_dataset.py`。

*收益*：
- 训练不受磁盘故障影响
- 从任意 step 精确恢复
- 支持"训到 100B token 时换 data mix"这种在线调整

=== 5.4 Determinism + resumability 的 audit

*改造*：每 N step 做一次 self-audit：
- 从 checkpoint 恢复 (in-memory)
- 用相同 batch 跑一步
- 比较 loss / grad_norm 是否与原始 run 一致
- 不一致就报警——checkpoint 有 bug

```python
if step % 1000 == 0 and step > 0:
    # Snapshot current state
    snap_model = deepcopy(model.state_dict())
    snap_optim = deepcopy(optim.state_dict())
    orig_loss = last_loss

    # Restore from checkpoint, replay one step
    model.load_state_dict(load_ckpt(step - 1)["model"])
    replayed_loss = model(same_batch).item()

    if abs(replayed_loss - orig_loss) > 1e-4:
        alert(f"checkpoint {step - 1} does not replay same loss "
              f"({orig_loss} vs {replayed_loss})")

    # Restore live state
    model.load_state_dict(snap_model)
    optim.load_state_dict(snap_optim)
```

*收益*：checkpoint bug (常见：优化器状态未同步、RNG 未存全) 提前发现，避免"训完发现无法 continual pretrain"。

== 六. RL 训练栈上的改造

如果你的岗位覆盖 post-training，"你们框架改了什么"这个问题会再问一遍，只不过基线从 Megatron 换成了 verl / OpenRLHF —— 而这两个框架本身就是搭在 Megatron / FSDP 上的，所以前五节的改造在 RL 栈里*依然成立*，只是上面又叠了一层专属的坑。

这一节按同样的四个维度组织。原理和背景在第 15 章，这里只列"基线给了什么 / 你还能加什么"。

=== 6.1 引擎层：rollout 引擎与权重同步

*基线*：verl / OpenRLHF 给你 vLLM 或 SGLang 后端、NCCL cross-group weight broadcast、基本的 agent loop。

高频改造：

- *Prefix cache pinning*。多轮 agentic rollout 里，序列等环境执行那几秒会被 LRU 淘汰 KV，回来重新 prefill。给"等待中"的序列加 pin 或换出到 CPU，命中率能从 40% 级拉到 90% 级。改的是推理引擎的淘汰策略，不是 RL 框架本身。
- *权重同步路径*。默认 NCCL broadcast 每轮几百 ms。同机部署时可以走 CUDA IPC 共享 buffer 做零拷贝；跨机则把 broadcast 与下一轮 rollout 的 prefill 重叠。
- *轨迹级连续批处理*。把 continuous batching 从 token 粒度抬到轨迹粒度，让等环境的轨迹自动退出运行批次。这是 GPU 利用率从 19% 到 88% 的那一步。
- *FP8 rollout*。rollout 是 inference，对精度容忍度比训练高，量化到 FP8 能显著提吞吐；但要验证 rollout 与训练的 logprob 差距没有因此扩大（见 6.4 的监控项）。

=== 6.2 verifier 与环境层：一个全新的 CPU 栈

*基线*：框架通常只给你一个 `reward_fn` 回调，剩下全是你的。

这一层基本*没有基线可言*，也因此最容易讲出个人贡献：

- 异步 verifier 池 + 每样本硬超时（子进程 + `SIGKILL`，不能用线程）
- `hash(prompt, response)` 结果缓存 + 组内去重，实测省 30–60% 调用
- 沙箱隔离：gVisor / Firecracker、断网、只读 FS、cgroup 限 CPU/内存/pid
- 环境失败分类：`infra_failure` 与 `policy_failure` 分开，前者剔除样本而不是给 reward 0
- 答案抽取器的加固：取最后一个 `\\boxed{}`、括号匹配而非正则、拒绝多答案

#insight[
  面试里如果你只讲得出 6.1，会被归类为"用框架的人"；能讲 6.2 才说明你真的建过 RLVR 的生产流水线 —— 因为这一层开源框架给得最少，每个团队都得自己造，而每个自己造过的人都踩过同一批坑（sympy 挂死、模型改写测试文件、fork bomb）。
]

=== 6.3 算法层：把论文里的修正接进来

*基线*：框架给的通常是原始 GRPO / PPO。

- DAPO 四件套：clip-higher、dynamic sampling、token-level loss、overlong reward shaping
- Dr. GRPO 的两处去偏：去掉 $1\/|o_i|$ 长度归一化与 $1\/"std"$
- 去 KL（省掉 ref model 整个池），配合监控兜底
- 难度分层采样：维护每题历史准确率，优先采 0.2–0.8 区间
- partial rollout 的跨版本记账：限制轨迹最多跨 $K$ 个 policy 版本

*要点*：这些都是"跟随论文改"，说的时候要讲清*为什么在你的场景需要它*。比如 dynamic sampling 不是白上的——它让每 iteration 的 rollout 量变成动态的，调度器要支持流式补采，这是实打实的工程改动。

=== 6.4 训练层与监控：RL 专属的那几个指标

训练层最重要的是 *token masking*（观测 token 不进 loss，归一化分母用 `mask.sum()`），细节见第 15 章 4.1/4.2 —— 那是本书里"最容易写错且最不容易发现"的一处。

监控上，预训练那套（loss / grad_norm / MFU）在 RL 里*远远不够*，得再加一组：

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*指标*], [*为什么要看*], [*健康值*],
    [epoch-0 裁剪比例],
    [rollout 与训练引擎 logprob 是否自洽],
    [≈ 0],
    [prefix cache 命中率],
    [多轮 rollout 是否在重复 prefill],
    [> 85%],
    [verifier 队列深度],
    [CPU 池是否成为瓶颈],
    [不持续增长],
    [verifier 超时率],
    [是否有病态输出把沙箱卡死],
    [\< 1%，突增即告警],
    [有效组占比],
    [多少 rollout 产生了真实梯度],
    [> 50%],
    [policy 熵],
    [是否在走向熵坍缩],
    [不单调下降],
    [工具调用成功率],
    [agentic 场景的真实业务指标],
    [随训练上升],
    [infra 失败率],
    [环境侧 SLO],
    [\< 1%],
    [平均轮数 / 截断率],
    [轨迹是否在被预算腰斩],
    [截断率 \< 10%],
  ),
  kind: table,
  caption: [RL 训练相比预训练需要额外监控的九项指标。前两项最能体现深度：多数人不知道 epoch-0 的裁剪比例应该是 0，也不知道 prefix cache 命中率会悄悄掉到 40%。],
) <table-rl-monitor>

#warn[
  RL 里*最危险的监控习惯是盯 loss*。Agentic RL 的 loss 下降几乎不能说明任何事 —— token masking 写错时 loss 照样平滑下降，同时 agent 正在退化成"不调工具直接编答案"。必须以任务成功率、工具调用成功率、平均轮数这些业务指标为准。
]

=== RL 栈改造清单

#figure(
  table(
    columns: (auto, 1.1fr, 1.3fr, auto),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left, center),
    [*改造点*], [*基线 (verl / OpenRLHF)*], [*生产改造*], [*典型收益*],
    [Rollout 调度], [同步批式 / 基础 agent loop], [轨迹级连续批处理],
    [GPU 利用率 19%→88%],
    [Prefix cache], [引擎默认 LRU], [等待中序列 pinning / CPU 换出],
    [命中率 40%→90%],
    [长尾轨迹], [等最长那条], [partial rollout + 跨版本记账],
    [makespan −40%+],
    [权重同步], [NCCL broadcast], [CUDA IPC 零拷贝 / 与 prefill 重叠],
    [每轮省数百 ms],
    [Verifier], [一个 `reward_fn` 回调], [异步池 + 超时 + cache/去重 + 沙箱],
    [调用量 −30\~60%],
    [环境失败], [算作 reward 0], [`infra_failure` 剔除 + SLO 告警],
    [去掉训练信号噪声],
    [RL 算法], [原始 GRPO], [DAPO 四件套 + Dr. GRPO 去偏],
    [有效组占比翻倍],
    [Ref model], [常驻一个池], [去 KL，显存转给 KV cache],
    [省一个 70B 池],
    [训练侧], [单轮假设], [轨迹 token masking + `mask.sum()` 归一化],
    [修复"越训越差"],
    [监控], [loss / reward 曲线], [九项 RL 专属指标（见上表）],
    [问题从天级到分钟级],
  ),
  kind: table,
  caption: [RL 训练栈的十类改造。与前五节的区别在于：verifier 与环境这一层开源框架几乎没有基线，是最容易做出个人贡献、也最容易在面试里讲出深度的地方。],
) <table-rl-delta>

#interview[
  *面试深挖*：_"RL 训练你们用 verl，那 verl 都做好了，你做了什么？"_

  这问题和开篇那个 Megatron 版本是同一道题。可信的回答结构：

  + *先指出 verl 没做的那一层*：verifier 池、沙箱隔离、环境失败分类 —— 框架只给一个 `reward_fn` 回调，生产流水线是自己搭的。
  + *再讲一个引擎层的深挖*：比如 prefix cache 命中率只有 40% 的排查过程，从"为什么开了 cache 还是慢"到"等环境时 KV 被 LRU 淘汰"到"扩 KV 池 + pinning"。
  + *再讲一个正确性 bug*：token masking 或 `numel()` 归一化，说清"loss 正常但业务指标下降"这个信号是怎么锁定它的。
  + *最后给数字*：iteration 时间、GPU 利用率、任务成功率各降/升了多少。

  第 15 章第八节有一个完整的 STAR 版本可以直接借鉴结构。
]

== 七. 完整 STAR 故事：把四个维度串起来

*Situation (30 s)*：

"我们上一个模型是 32B dense LLM，在 512 张 H100 上从 Meta Llama-2 baseline 起训。跑起来第一版是 Megatron 官方 recipe：TP=4, PP=8, DP=16, BF16 mixed, SelectiveRecompute。MFU 大约 34%，比 Meta 报告的 42% 差 8 个点；同时前 200 步有过 3 次 loss spike，每次都要手动 restart，7 天训了 500B token，团队想 push 到 3 T token 需要 6-7 周。"

*Task (10 s)*：

"我负责在两周内把 MFU 提到 45%+，并让训练能 7×24 无人值守 push 到 3T token。"

*Action (2 min)*：

"我按四个维度动手：

*算子层*：跑 nsys 一步 profile，发现三处：
- GEMM 里 RMSNorm + Linear + RoPE 是三个独立 kernel，launch overhead 8%。写了 Triton fused kernel（RMSNorm var 强制 FP32 upcast），单元测试对齐 unfused reference。
- TE FA2 kernel 在 seq=8K 时 tail 长，升级到 FA3 后 +12% MFU。加 shape guard，头 100 步 dtype/head_dim 不匹配就 fallback 到 FA2，count fallback 到 dashboard。
- DDP grad AR bucket=25 MB，200 个 bucket → 6 ms/step launch overhead。改成 200 MB + `gradient_as_bucket_view=True`，overlap 60% → 92%。

*算法层*：
- 观察前几步 grad_norm 波动大，`beta_2=0.999` (Megatron 老默认) → 改 0.95，warmup 从 500 → 2000 步。这是标配但 baseline recipe 没跟上。
- 数据侧上 BFD packing（原来 fixed-length，padding 15%）→ padding 2%，等效 +13% throughput。

*稳定性*：
- 3 次 loss spike 后加了 auto-skip：`grad_norm > 10× 20-step median` 就 skip 该 step 并存 poisoned batch。之后再没手工重启。
- Checkpoint 从同步 5 min block 改成 async + L1 (peer node CPU DDR, 3 s) + L2 (local NVMe, 30 s)。in-flight NaN 用 L1 rollback 200 ms 恢复，硬故障用 L2 20 s 恢复。

*监控 + Infra*：
- Prometheus + Grafana dashboard，15 类指标。straggler detect 每 10 步 gather per-rank time，慢 30% 的 rank alert。启动阶段跑 nccl-tests 全对，一次 fail-fast 掉了 2 张有问题的卡。
- topology-aware rank 分配：TP 组 pin 同 node，PP 组沿 rack 排列。inter-rack AR 降 15%。
- torchrun elastic + max-restarts=100，一次真实硬件故障 3 分钟自动恢复。"

*Result (20 s)*：

"MFU 34% → 46%，push 到 3T token 从 6-7 周 → 4 周。训练期间 12 次硬件故障全部 elastic 恢复，0 次手动 restart。团队把这套改造（Triton fused kernel + skip step + L1 backup + straggler detect）合并到内部 baseline recipe，后续 2 个模型无需重新做这些活。"

_面试官可以从任何一句往下追问细节 —— 每个环节都得能 hold 住。这就是为什么这一附录写得深：你不能只知道"我们改了 kernel"，得知道 fused 时哪一步要 FP32 upcast、fallback 怎么写、shape guard 怎么测。_

== 八. 面试 anti-pattern：什么情况不能这么答

*不能*把四个维度全都说"我做过"，除非你真做过。可信的分工：

- *一个 IC*：一般深度覆盖 1-2 个维度（比如算子 + 监控），其他维度"知道 & review 过 patch"
- *tech lead*：4 个维度都 review 过，但只有 1-2 个亲手写
- *manager*：知道每个维度是谁做的、trade-off 是什么，但不写代码

*一定要老实*：某个维度不熟就说 "这块是我 team 里另一位同学做的，我知道大概做了 X 但细节不敢瞎讲"—— 这比 "我啥都会" 加分得多。

*不能*说的话：
- "我们把 Megatron 改了很多"（无具体）
- "我实现了 XX"（如果实际上是抄开源）
- "MFU 从 30 提到 60"（数字过于夸张，业界 dense LLM 顶到 50% 就很高）
- "我训了 XX B model"（模型大小是资源不是贡献）
- "我们没遇到过 loss spike"（不真实；spike 是常态，重点是怎么处理）

*一定要*说的话：
- 具体的 patch 位置（哪个文件、哪个函数）
- 具体的 trade-off（为什么这么改，别的方案为什么不行）
- 具体的数字（MFU、step time、恢复时间）
- 具体的 debug 故事（怎么发现问题的）

== 九. 一页速记：改造维度总表

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.4pt + gray,
    inset: 6pt,
    align: (left, left),
    [*维度*], [*高频改造点（面试前至少能说出 2 个 + 1 个 trade-off）*],
    [*算子层*],
    [Triton fused (RMSNorm+Linear+RoPE) · FA3 + shape-guard fallback · Grouped GEMM · DeepEP a2a (IBGDA) · CUDA graph capture (varlen bucket)],
    [*算法层*],
    [Muon (Kimi K2) · aux-loss-free MoE (DeepSeek-V3) · MTP head · MLA · BFD sequence packing · WSD + mini-restart LR],
    [*稳定性*],
    [auto-skip spike (grad_norm 10× median) · NaN in-memory rollback · per-layer grad clip · L1/L2/L3 三级 checkpoint · deterministic mode 可 debug],
    [*监控 + Infra*],
    [15+ metric Prometheus dashboard · straggler detect (per-rank step time) · nccl topology probe · elastic + `--max-restarts` · streaming resumable loader],
    [*RL 栈* (§六)],
    [轨迹级连续批处理 · prefix cache pinning · partial rollout 跨版本记账 · 异步 verifier 池 (超时/cache/去重/沙箱) · `infra_failure` 剔除 · DAPO 四件套 · token masking + `mask.sum()` · epoch-0 裁剪比例监控],
  ),
  caption: [Megatron 与 RL 栈改造速记。面试前一晚翻这张表，每个维度至少能说出 2 个具体点 + 1 个 trade-off。做 post-training 的岗位重点看最后一行。],
) <fig-4dim-recap>

*最后一句话*：Megatron 和 verl 都是 baseline，不是终点。所有开源框架都留有大量 "生产集群才知道要改" 的空白 —— 你的贡献就是填这些空白，而 RLVR 的 verifier 与环境那一层空白最大。面试里能把自己相关的那几个维度都举出具体例子（哪怕不都是自己写的），就已经把自己从"tuner"升到"framework builder"了。
