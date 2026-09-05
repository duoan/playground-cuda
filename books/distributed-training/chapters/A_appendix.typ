#import "../template.typ": *

= 附录 A：数字速查表

面试里 30 秒能算出的量级估算是加分项。这一附录把全书数字集中，用来临阵抱佛脚。

== 硬件带宽 / 算力

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, left),
    [*硬件*], [*数值*], [*说明*],
    [A100 SXM4 80GB (BF16)],          [312 TFLOPS],  [Tensor Core],
    [A100 HBM],                        [2.0 TB/s],    [],
    [A100 NVLink 3 (bidi)],           [600 GB/s],    [8 卡 NVL 域],
    [H100 SXM5 80GB (BF16)],          [989 TFLOPS],  [Tensor Core],
    [H100 SXM5 80GB (FP8)],           [1979 TFLOPS], [E4M3/E5M2],
    [H100 HBM3],                       [3.35 TB/s],   [80GB HBM3],
    [H100 NVLink 4 (bidi)],           [900 GB/s],    [8 卡 NVL 域],
    [H100 NVSwitch 3 (aggregate)],    [3.6 TB/s],    [8-way non-blocking],
    [H200 HBM3e],                      [4.8 TB/s],    [141GB],
    [B200 (BF16)],                     [2250 TFLOPS], [~2.3× H100],
    [B200 (FP8)],                      [4500 TFLOPS], [],
    [B200 (FP4)],                      [9000 TFLOPS], [inference-only initially],
    [B200 NVLink 5 (bidi)],           [1800 GB/s],   [],
    [GB200 NVL72 (aggregate)],        [~130 TB/s],   [72 卡 domain],
    [Infiniband NDR 400G],            [50 GB/s],     [uni-directional],
    [Infiniband XDR 800G (2025)],     [100 GB/s],    [B200 时代新一代],
    [Ethernet 400G RoCEv2],           [~40 GB/s],    [effective],
    [PCIe Gen5 x16],                   [63 GB/s],     [uni],
    [DDR5 per socket],                 [~90 GB/s],    [],
  ),
  kind: table,
  caption: [硬件带宽/算力速查。BF16 与 FP8 峰值算力是同一硬件两个数字，实际实现要看 kernel。],
)

== 参数量与显存

#figure(
  table(
    columns: (auto, 1.5fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left),
    [*模型*], [*参数量公式 & 数字*],
    [Transformer], [$N approx 12 L H^2 + V H$],
    [AdamW 混精 memory/param], [16 bytes (2 wt + 4 wt_fp32 + 2 grad + 4 m + 4 v)],
    [Precision-aware AdamW], [10 bytes (m,v 用 BF16)],
    [Activation (unfused)], [$approx 34 L B S H$ bytes],
    [Activation (TE fused)], [$approx 17 L B S H$],
    [Activation (full recomp)], [$approx 2 L B S H$],
    [Attention (naive)], [$B S^2 A$ per layer (huge)],
    [Attention (FA)], [$B S H$ per layer],
    [KV cache], [$2 L S H "bs"$ per sample],
  ),
  kind: table,
)

== FLOPs

- Forward per token: $2 N$
- Full train per token: $6 N$ (加 attention 项 $12 L S H$)
- 总 training FLOPs: $6 N D$

Wall-clock 时间：$T ("s") = 6 N D / ("cards" times "MFU" times "peak")$

MFU 参考：
- Dense LLM: 40-55%
- MoE: 35-50%
- FP8 training: 30-45%
- RL (SFT loop): 30-45%
- RL (rollout dominated): 15-25%

== 通信量速查

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, auto, left),
    [*操作*], [*per-GPU vol*], [*备注*],
    [Broadcast/Reduce (root)], [$V$], [O(V)],
    [AllReduce (Ring)], [$2V(W-1)/W approx 2V$], [$W$ 大恒定 2V],
    [AllGather], [$V(W-1)/W approx V$], [],
    [ReduceScatter], [$V(W-1)/W approx V$], [],
    [All-to-All], [$V(W-1)/W approx V$], [每 rank vol],
    [DDP grad AR], [$2P$ per step], [$P$ = param bytes],
    [ZeRO-1/2], [$2P$], [同 DDP],
    [ZeRO-3 / FSDP], [$3P$], [多 50%],
    [ZeRO-3 activation AG], [$P$ (fwd) + $P$ (bwd)], [prefetchable],
    [TP AllReduce], [$2 B S H "bs" / "layer"$], [每层, 每方向],
    [SP AG+RS], [$2 B S H "bs" / "layer"$], [同 TP vol],
    [PP P2P], [$B S H "bs"$ per stage transition], [每 micro-batch],
    [MoE dispatch a2a], [$B S K H "bs" / "EP"$ (approx)], [每层每方向],
    [Ring attention P2P], [$2 B S H "bs"$ total per layer], [独立于 CP],
    [Ulysses a2a], [$4 B S H "bs" (1 - 1/"CP") / "layer"$], [],
  ),
  kind: table,
)

== Bubble & Overlap 收益

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, auto, left),
    [*方案*], [*Bubble ratio*], [*代价*],
    [GPipe], [$(P-1)/(m+P-1)$], [activation × $m$],
    [1F1B], [$(P-1)/(m+P-1)$], [activation × $P$],
    [Interleaved 1F1B], [$(P-1)/(V m + P - 1)$], [activation × $V$, P2P × $V$],
    [ZeroBubble (ZB1P)], [$(P-1)(F+B-2W)/"total"$], [autograd hack],
    [DualPipe], [$(P/2-1)(F\&B+B-3W)$], [*2× weight*],
    [Megatron FWD-BWD merged], [~1F1B, comm hidden], [ 1×, 需 stream ≥ 2],
  ),
  kind: table,
)

Overlap 收益典型值 (dense / MoE):

- DDP: 90%+ auto (bucket_cap ≥ 100 MB)
- FSDP: 85-95% (BACKWARD_PRE + forward_prefetch)
- TP async: 15-20% → 5-8%
- MoE EP (no overlap): 30-40% comm
- MoE EP (Megatron merged): \< 5%
- MoE EP (DualPipe): ~0%

== 训练成本参考

行业公开数字：

- GPT-3 175B on 300B tokens: $3.14 times 10^23$ FLOPs, ~3,640 pf-days
- Llama-2 70B: ~1.7M A100-hours ≈ \$2M
- Llama-3 70B: ~6.4M H100-hours ≈ \$15-20M
- Llama-3 405B: ~30M H100-hours ≈ \$70M
- DeepSeek-V3 671B: 2.664M H800-hours ≈ \$5.3M (efficient)
- Mixtral 8×7B: undisclosed，估 300K H100-hours
- Chinchilla law: $D approx 20 N$ tokens optimal

== 长上下文常见数字

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, right),
    [*模型 / seq*], [*Activation (BF16, 1 layer, B=1)*], [*KV cache*],
    [7B, 8K], [~256 MB], [~1 GB/sample],
    [7B, 32K], [~1 GB], [~4 GB],
    [7B, 128K], [~4 GB], [~16 GB],
    [7B, 1M], [~32 GB (needs CP)], [~128 GB],
    [70B, 128K], [~40 GB (needs CP+TP)], [~160 GB],
    [70B, 1M], [~320 GB (CP+TP+FSDP+recomp)], [~1.2 TB],
  ),
  kind: table,
  caption: [长序列显存参考（含 FA，含 residual）。1M ctx 训练必须组合 CP + TP + FSDP + full recomp。],
)

== 训练稳定性阈值

- Grad norm alert threshold: > 100
- Grad norm spike (skip step): > 10 × running mean
- Loss NaN: 立刻 save debug ckpt + rollback
- LR warmup steps: min(2000, D/1000)
- KL blow up (RL): > 5.0

== 常用环境变量

```bash
# NCCL 稳定与性能
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_ALGO=Auto
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=512
export NCCL_BUFFSIZE=8388608
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=1

# Watchdog
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=1

# Torch mem alloc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Distributed
export TORCH_DISTRIBUTED_DEBUG=INFO       # 或 DETAIL
export FI_PROVIDER=efa                     # AWS EFA
export FI_EFA_USE_DEVICE_RDMA=1
```
