#import "../template.typ": *

= Overlap: 通信与计算的重叠

这一章把散落在前几章的 overlap 技术系统整理一遍，给出选择框架和 profile 诊断方法。分布式训练 30% 以上的性能优化最终都是"让通信和计算同时跑"。

#figure(
  align(center, timeline(
    streams: (
      ("serial     ", (("compute", 10), ("comm", 6))),
      ("async_op   ", (("compute", 10), ("comm", 2))),
      ("two-stream ", (("compute", 10), ("comm", 0))),
    ),
    unit: 0.45, bar-h: 0.55,
    title: "compute + comm overlap 效果对比",
  )),
  caption: [同一 workload 三种执行策略。serial 是 compute→comm；async_op 用 `async_op=True` 部分隐藏；two-stream 用独立 stream + CUDA event，通信完全隐藏在 compute 之下。见 `src/distributed_training/11_overlap_2stream.py`。],
) <fig-overlap-serial-vs-async>

== Overlap 的物理基础

GPU 上通信 (NCCL) 与 compute (matmul) 是不同的 CUDA kernel。如果它们在*不同 CUDA stream* 上，硬件会尝试并行调度。

*NVIDIA Hopper 硬件层面的限制*：

+ NCCL kernel 占若干 SM (default 2-8 per channel)
+ Compute kernel 占其余 SM
+ 两者共享 HBM 带宽
+ 通信 kernel 用 PCIe / NVLink / IB，与 HBM 不冲突

所以*理想 overlap 是 100% 隐藏*：只要 SM 分得开、HBM 带宽够。

*CUDA_DEVICE_MAX_CONNECTIONS* 决定 host 到 device 的 hardware queue 数。默认 8，实际 overlap 需要 ≥ 通信 stream 数。生产建议 `export CUDA_DEVICE_MAX_CONNECTIONS=8`。

== Overlap 的分层：从 op 到 pipeline

从最细到最粗：

+ *Level 0 — Op 内 overlap*：单 kernel 内部把 compute 和 async load overlap（Flash Attention 内部, CUTLASS pipeline）。用户不看，kernel 作者负责。
+ *Level 1 — Op 间 overlap*：AllReduce 与相邻的 GEMM overlap（Megatron async TP, DDP bucket AR）
+ *Level 2 — Layer 间 overlap*：本层 comm 与下层 compute overlap（FSDP forward prefetch, backward prefetch）
+ *Level 3 — Micro-batch 间 overlap*：本 micro-batch 的 backward 与下一 micro-batch 的 forward overlap（1F1B pipeline）
+ *Level 4 — Iteration 间 overlap*：optim step 与下 step forward overlap（fused optimizer with async grad reduce）

大多数框架自动处理 L0-L1；L2-L4 需要手动 tune 或选择合适 flag。

== 逐 level 详解

=== Level 1: DDP bucket AllReduce 与 backward compute

第 3 章讲过。核心：backward 是 output → input 反向走，DDP 把 param 按顺序 bucket，一个 bucket 所有 grad 就绪即起 async AR，与前面 layer 的 backward compute overlap。

*代码要点*：

```python
# 检查是否 overlap：
for name, p in model.named_parameters():
    p.register_hook(lambda g, n=name: print(f"grad ready: {n} at step {torch.cuda.Event().record()}"))
```

失败案例：`find_unused_parameters=True` 会破 overlap（DDP 要等所有 grad 判定），`gradient_as_bucket_view=False` 有额外 copy。

*诊断*：nsys 里看 `ncclAllReduce` kernel 与前面 `sgemm` kernel 是否在不同 stream + 时间叠加。理想 overlap ratio > 90%。

=== Level 1: TP AllReduce 与 GEMM overlap (Async TP)

Megatron `--tp-comm-overlap` + TE `LayerNormLinear` fused。

思路：GEMM 拆 tile，per-tile 边算边发通信。

*示意*：

#figure(
  align(center, timeline(streams: (
    ("GEMM", (("compute", 15),)),
    ("AR",   (("bubble", 3), ("comm", 12))),
  ), unit: 0.5, title: none)),
  caption: [Async TP：GEMM 全长 15 单位；AR 从第 3 单位（tile 0 算完）开始，与后续 tile 计算并行，总时长 ≈ max(15, 3+12) = 15。串行方式则是 15 + 12 = 27。overlap 消掉 ~45% 时间。],
) <fig-async-tp>

*收益*：TP=8 的 AR 通信占比 15-20% → 5-8%。开销：kernel 复杂化，需要 TE 支持。

=== Level 2: FSDP forward/backward prefetch

FSDP `BackwardPrefetch.BACKWARD_PRE`。第 4 章讲过。

思路：backward 第 $l$ 层开始前，就 issue 第 $l+1$ 层的 AllGather；第 $l$ 层 backward compute 时 AG 已经在跑，第 $l+1$ 层 compute 起来时 param 已经就绪。

*代价*：显存峰值高（同时持有 $l$ 层和 $l+1$ 层的 param）。OOM 时降到 `BACKWARD_POST`。

*前向类似*：`forward_prefetch=True` 让 $l$ 层 compute 时 issue $l+1$ 层 AG。

=== Level 2: MoE dispatch / combine 与 expert compute overlap

一层 MoE 内部三阶段串行时通信占 40%+：

#figure(
  align(center, timeline(streams: (
    ("serial", (("comm", 3), ("compute", 4), ("comm", 3))),
  ), unit: 0.5, title: none)),
  caption: [MoE 串行：dispatch a2a → 本地 expert compute → combine a2a。通信占 6/10 = 60%。],
) <fig-moe-serial>

改成 2-stream overlap：

#figure(
  align(center, timeline(streams: (
    ("comm stream", (("comm", 3), ("bubble", 4), ("comm", 3))),
    ("comp stream", (("bubble", 3), ("compute", 4), ("bubble", 3))),
  ), unit: 0.5, title: none)),
  caption: [MoE 2-stream：dispatch 与前一 op 尾巴 overlap；combine 与下一层 attention 起始 overlap。CUDA event 控制依赖。通信占比降到 ~20%。],
) <fig-moe-2stream>

Megatron `--overlap-moe-expert-parallel-comm`。第 8 章有代码示例。

=== Level 3: 1F1B micro-batch 交叉

Pipeline 里 stage $s$ 交叉做 $F_i$ 和 $B_j$。stream 层面：

#figure(
  align(center, timeline(streams: (
    ("compute", (("compute", 2), ("compute", 2), ("compute", 2),
                 ("compute", 2), ("compute", 2), ("compute", 2),
                 ("compute", 2), ("compute", 2), ("compute", 2))),
    ("P2P",     (("bubble", 1), ("comm", 1), ("bubble", 1), ("comm", 1),
                 ("bubble", 1), ("comm", 1), ("bubble", 1), ("comm", 1),
                 ("bubble", 1), ("comm", 1), ("bubble", 1), ("comm", 1),
                 ("bubble", 1), ("comm", 1), ("bubble", 1), ("comm", 1),
                 ("bubble", 1), ("comm", 1))),
  ), unit: 0.38, title: none)),
  caption: [1F1B pipeline stream 视图：compute stream 依次 F0/F1/B0/F2/B1/…；P2P stream 与 compute 并行发送每个 $F_i$ 的 activation 到 stage $s{+}1$。P2P 与 compute 天然 overlap（不同 stream），1F1B 已 built-in。],
) <fig-1f1b-streams>

=== Level 3.5: Megatron FWD-BWD Merged

`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`。第 6, 8 章讲过。

思路：同一 iteration 里，相邻 micro-batch 的 forward compute 与 backward+comm 排到两条 stream。B 与 W 分开让"data-grad B" 可以先算继续 pipeline 推进，"weight-grad W" 见缝插针。

对 MoE：EP a2a 占 30-40% → \< 5%。当前 Megatron 默认推荐。

=== Level 4: DualPipe

第 6 章。bidirectional pipeline + 4-phase overlap，pipeline bubble 与 a2a 几乎完全隐藏。代价 2× 权重。DeepSeek-V3 用。

=== Level 4: Optimizer step 与下 step forward overlap

比较冷门。思路：optim step 完成 rank $r$ 的 shard 时，rank $r'$ 已经开始下 step forward。fused Adam 内部 async。

TE 的 `apply_optimizer_step_with_overlap` 提供，但生产用得不多——optim step 时间通常 \< 5% 的 step time。

== Overlap 值不值得做

*量化的判断*：

$ "gain"_"overlap" = min("comm_time", "compute_time") $

- 如果 comm_time > compute_time：overlap 后 step = comm_time（compute 白算白 free）
- 如果 comm_time < compute_time：overlap 后 step = compute_time
- 如果两者相等：overlap 后 step ≈ max/2

*示例*：一层 dense forward 100 ms compute + 20 ms comm，overlap 后 100 ms（省 20%）。
一层 MoE 100 ms compute + 80 ms comm (跨节点 a2a)，overlap 后 100 ms（省 44%）。

*Overlap 收益越大的场景*：跨节点、MoE、大模型、TP > 4。在小模型 + 快网络场景，overlap 收益 \< 5% 不值得代码复杂度。

== Overlap 的常见 bug

+ *stream 依赖漏了*：async op handle 忘了 wait，导致 kernel 用还没到的数据。表现：偶发 nan 或 hang。修：CUDA event 记录 + wait_event
+ *`CUDA_DEVICE_MAX_CONNECTIONS=1`*：默认 1 时 stream 串行执行，overlap 失效。检查：`echo $CUDA_DEVICE_MAX_CONNECTIONS`
+ *NCCL kernel 抢 SM*：DeepEP V1 吃 20 SM，导致 GEMM 掉 20%。用 DeepEP V2 或减少 channel
+ *host-side 同步*：`.item()`, `.tolist()`, `dist.barrier()` 都是 D2H sync，破坏 CUDA graph 和 overlap
+ *Bucket 太小/太大*：DDP 里太小则 AR 次数多延迟主导；太大则 overlap 空间小
+ *AR 的 op 顺序*：`dist.all_reduce(..., async_op=True)` 返回 handle 后必须在正确点 wait，否则 grad 用未同步数据

== 用 nsys profile overlap

```bash
nsys profile -t cuda,cudnn,cublas,nvtx,osrt \
    --stats=true \
    -o my_run.nsys-rep \
    torchrun --nproc-per-node=8 train.py
```

打开 my_run.nsys-rep，看：

+ *Timeline*：多条 stream row，NCCL kernel 和 GEMM kernel 应该在同一时间列上（overlap）
+ *NCCL kernel*：搜 `nccl` 前缀。看 duration、颜色（AR vs AG vs a2a）
+ *stream row 内的 gap*：如果 stream 内有明显 gap 但另一 stream 有 kernel，那是没 overlap 好——依赖关系错了

一个 healthy MoE profile：expert compute stream 和 a2a stream 大部分时间都有 kernel，几乎没 gap。

== 一份 overlap checklist

在训练脚本里检查：

```bash
# Environment
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_DEBUG=WARN

# DDP (if using)
model = DDP(model, bucket_cap_mb=200, gradient_as_bucket_view=True)
# 不开 find_unused_parameters

# FSDP
FSDP(model,
     backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
     forward_prefetch=True,
     limit_all_gathers=True)   # 控制 prefetch 内存峰值

# Megatron / MoE
--tp-comm-overlap
--overlap-grad-reduce
--overlap-param-gather
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
```

跑 profile 确认：一个 step 里 NCCL kernel 时间 / step time < 15% (dense) 或 < 20% (MoE)。超过就有 overlap 空间。

== 面试考点

#interview[
  *Q1*: DDP overlap 靠什么实现？

  A: Bucket + backward hook + async AllReduce。DDP 在 init 时把 param 分 25 MB bucket；每 param 注册 backward hook；一个 bucket 内所有 grad ready 立刻 issue async AR (在专门 NCCL stream)；backward 继续往前算。AR 与前面层的 backward compute 时间重叠。要求 `find_unused_parameters=False` 才不会破 overlap。
]

#interview[
  *Q2*: FSDP forward_prefetch 与 backward_prefetch 分别什么时机？

  A: forward_prefetch: 第 $l$ 层 forward 开始 → 立刻 issue 第 $l+1$ 层的 AllGather。backward_prefetch (BACKWARD_PRE): 第 $l$ 层 backward 开始前 → issue 第 $l-1$ 层 AG（backward 反向走，"下一步"就是 $l-1$）。两个都是"往未来看"，代价是显存峰值高（同时持两层 param）。
]

#interview[
  *Q3*: 为什么 CUDA_DEVICE_MAX_CONNECTIONS=1 时 overlap 失效？

  A: 这个 env var 控制 host 到 device 的 hardware queue 数。=1 意味着 host 只能一次下发一个 stream 的命令，多 stream 变成串行。生产要 >= max(concurrent stream count)，Hopper+ 一般设 8。
]

#interview[
  *Q4*: nsys 里怎么判断 overlap 好不好？

  A: 看 timeline 的多 stream row。理想：compute stream 和 comm stream 时间列 overlap，两者都几乎无 gap。不好的信号：(a) comm stream 有 kernel 但 compute stream 有 gap → compute 在等 comm 结果；(b) 两 stream kernel 完全错开（不同时间列）→ 没 overlap；(c) NCCL kernel 内部有 gap → NCCL 自己在等（可能是 rank 不齐, 慢卡）。
]

#interview[
  *Q5*: Async TP 相比标准 TP 的收益怎么估？

  A: 标准 TP 每层 2 AR，AR 时间约 GEMM 时间的 15-20%（H100 NVL8）。开 async 后大部分 AR 与 GEMM overlap，AR 剩 5-8%。假设 TP 通信总占 20%，overlap 到 6%，则 step time -14%。对训 Llama-70B 是 5-10 天时间的节约。
]

#interview[
  *Q6*: MoE 训练 overlap 后为什么能大幅提速？

  A: MoE 一层 4 次 a2a (fwd/bwd dispatch+combine)，占 forward 15-40% 时间。dense 只有 TP AR (10-20%)。overlap 到 \<5% 后，MoE 省的绝对时间比 dense 多得多。DeepSeek DualPipe + DeepEP 是极端案例，让 a2a 从 30-40% 隐藏到 ~0%。
]

#interview[
  *Q7*: 一个 step 里 host-side 出现 `.item()` 会有什么后果？

  A: `.item()` 是 D2H sync，host 阻塞等 device queue 排空。破坏：(1) CUDA graph capture (graph 内不能有 sync)；(2) stream overlap (sync 后 host 要重新 issue 命令，NCCL kernel 可能延迟启动)；(3) profile 里看到很长的 idle time。生产训练循环里避免任何 `.item()` / `.tolist()` / `.cpu()`，log 用 `torch.tensor` 累积到 log step 才 D2H。
]

#interview[
  *Q8*: DualPipe 与 Megatron FWD-BWD merged 都能 overlap 到 90%+，为什么大多数生产选 FWD-BWD merged?

  A: DualPipe 需要 2× weight 副本（bidirectional pipeline 每端存一份），memory wall 加剧。FWD-BWD merged 只需 CUDA_DEVICE_MAX_CONNECTIONS>1 + `--delay-wgrad-compute`，1× 权重，配置改一个 flag 就开。overlap 收益都在 90%+，实测差异 < 3%。DualPipe 只在训 671B+ 且 IB comm 极度受限时值得。
]
