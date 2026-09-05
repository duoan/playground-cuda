#import "../template.typ": *

= 集合通信与带宽模型

分布式训练的一切性能分析，最终都能落到*一次集合通信要几字节 × 走什么带宽 × 有没有被 overlap*。这一章把公式和数字讲清楚，后面每章都会用。

== 硬件层次：从 SM 到 IB

一个训练集群的通信媒介，按带宽从高到低：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, right, left),
    [*介质*], [*典型带宽 (单向)*], [*延迟*], [*作用域*],
    [SM ↔ HBM],       [~1.7 TB/s (H100)],  [~400 cycles], [单卡],
    [NVLink 4 (H100)],[450 GB/s (uni) / 900 GB/s (bidi)], [~1 µs], [同一 NVL8 / NVL72 域],
    [NVSwitch (GB200 NVL72)], [900 GB/s per link × 18], [~1 µs], [72 卡 flat 域],
    [PCIe Gen5 x16],  [63 GB/s (uni)],     [~2 µs],   [CPU↔GPU / GPU↔NIC],
    [Infiniband NDR (400 Gb/s)], [50 GB/s (uni)],  [~2 µs], [跨节点],
    [Infiniband HDR (200 Gb/s)], [25 GB/s],        [~2 µs], [老集群],
    [Ethernet RoCEv2 400G], [~40 GB/s effective], [~5 µs], [跨节点],
  ),
  kind: table,
  caption: [常见通信媒介带宽。注意 NVLink 与 IB 差 *~10×*，跨节点通信永远是瓶颈；GB200 NVL72 通过把 NVLink 域扩到 72 卡缓解了这一点。],
)

*一个关键比例*：H100 单卡 BF16 算力 989 TFLOPS，HBM 3.35 TB/s，NVLink 双向 900 GB/s，IB 单向 50 GB/s。也就是说：

$ "compute" : "HBM" : "NVLink" : "IB" approx 1000 : 3.35 : 0.9 : 0.05 " (TB/s vs TFLOPS)" $

Arithmetic intensity（每字节多少 FLOPs）在 HBM 层需要 ~295 FLOP/byte 才能 roof，NVLink 层需要 ~1100，IB 层需要 ~20000。这是所有 overlap 策略的物理基础。

== 集合通信原语

八个 NCCL 提供的原语，是所有分布式训练框架的字面构件。

#figure(
  table(
    columns: (auto, 1.4fr, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, right, right),
    [*Op*], [*语义*], [*每卡 send*], [*每卡 recv*],
    [Broadcast],     [rank 0 广播 $V$ 给所有],                      [$V$],       [$V$],
    [Reduce],        [所有 rank 数据 reduce 到 rank 0],             [$V$],       [$V$],
    [AllReduce],     [reduce 且结果广播到所有],                     [$2 V (W-1)/W$], [$2 V (W-1)/W$],
    [AllGather],     [rank $i$ 有 $V/W$，最终每 rank 拥有完整 $V$], [$V(W-1)/W$],[$V(W-1)/W$],
    [ReduceScatter], [reduce 后每 rank 拿 $V/W$ 分片],              [$V(W-1)/W$],[$V(W-1)/W$],
    [Scatter],       [rank 0 把 $V/W$ 分片发给每 rank],             [$V(W-1)/W$ (0号)], [$V/W$],
    [Gather],        [rank 0 收集所有 rank 的 $V/W$],               [$V/W$],     [$V(W-1)/W$ (0号)],
    [All-to-All],    [rank $i$ 的第 $j$ 段发给 rank $j$],           [$V(W-1)/W$],[$V(W-1)/W$],
  ),
  kind: table,
  caption: [八种集合通信及每卡 volume（$V$ = 单卡起始/结束数据量，$W$ = 世界大小）。AllReduce 的 volume 是 AllGather 或 RS 的 2×，这是 ZeRO/FSDP 分片省 comm 的公式基础。],
)

*两个关键身份*，面试常考：

$ "AllReduce" = "ReduceScatter" + "AllGather" $
$ "vol"_"AllReduce" = 2 times "vol"_"AllGather" = 2 times "vol"_"ReduceScatter" $

第一条是算法层面（NCCL Ring AllReduce 就是这两步 fuse）；第二条是通信量层面（AllReduce 走两遍数据，AG/RS 只走一遍）。ZeRO-2 (grad AR) → ZeRO-3 (param AG + grad RS) 看似省了 comm，其实*通信量相同*——只是切换了通信模式。

#figure(
  align(center, collective-diag(op: "AR", n: 4, slots: 4, cell: 0.4)),
  caption: [AllReduce：每卡持有一份数据，op 后每卡都得到"全体归约"的结果（深色 = 归约后）。],
) <fig-coll-ar>

#figure(
  align(center, collective-diag(op: "AG", n: 4, slots: 4, cell: 0.4)),
  caption: [AllGather：每卡持自己那一份 shard，op 后每卡拿到*全部* shard 的拼接。],
) <fig-coll-ag>

#figure(
  align(center, collective-diag(op: "RS", n: 4, slots: 4, cell: 0.4)),
  caption: [ReduceScatter：每卡有 W 份数据，op 后每卡只留自己"该负责的那一份"，且该份已被归约。],
) <fig-coll-rs>

#figure(
  align(center, collective-diag(op: "A2A", n: 4, slots: 4, cell: 0.4)),
  caption: [All-to-All：转置。rank $i$ 的第 $j$ 块发给 rank $j$。带宽压力最大——常成为 MoE 训练瓶颈。],
) <fig-coll-a2a>

=== Ring AllReduce 算法

#figure(
  align(center, ring-diagram(n: 4, labels: ("R0", "R1", "R2", "R3"),
    title: "4 卡 Ring")),
  caption: [Ring AllReduce 的物理拓扑：所有 rank 组成一个环，每一步只与相邻 rank 通信。共 $2(W-1)$ 步：前 $W-1$ 步 ReduceScatter，后 $W-1$ 步 AllGather。手写实现见 `src/distributed_training/01_collectives.py::ring_all_reduce()`，与 NCCL 结果 bit-for-bit 对齐。],
) <fig-ring>


NCCL 默认在中大规模 (> 8 rank) 用 Ring 算法。分 $W-1$ 步 ReduceScatter + $W-1$ 步 AllGather，每步每卡发送 $V/W$：

$ T_"ring" = 2(W-1) times (V/W) / B + 2(W-1) times alpha $

- $B$：单向带宽
- $alpha$：每步延迟（NCCL 里主要是 kernel launch + IB QP setup）
- $V/W$：每步 payload

当 $W$ 大时，$T$ 主要由 $2V/B$ 主导（与 $W$ 无关），加 $2 W alpha$ 的延迟项。这就是 ring 好的原因：*带宽项与 $W$ 无关*，只有延迟随 $W$ 线性增长。

=== Tree AllReduce

Ring 的延迟项 $O(W)$ 在小 payload 上会主导。NCCL 提供 Tree 算法：

$ T_"tree" = 2 log_2(W) times V/B + 2 log_2(W) times alpha $

带宽项从 $O(V)$ 变成 $O(V log W)$——大 payload 时更差，但延迟项只有 $O(log W)$——小 payload 更快。NCCL 自动根据 payload 选。手动切换：`NCCL_ALGO=Tree` 或 `Ring`。

#insight[
  面试爱问"为什么 ring allreduce 通信量与世界大小无关"。回答：每步每卡传 $V/W$，共 $2(W-1)$ 步 → total per-GPU volume $= 2V(W-1)/W -> 2V$。这个 "$2V$" 就是 AllReduce 的黄金常数——不管多少卡都是它。
]

=== NCCL Ring vs NVLS

从 NCCL 2.19 起 H100/H200 支持 *NVLS (NVLink SHARP)*：AllReduce 里把 reduction 逻辑下放到 NVLink switch 芯片里做，节省一次 GPU 参与。带来 *~1.3×* AllReduce 带宽。开启：`NCCL_NVLS_ENABLE=1`（部分 setup 需要显式打开）。

== 带宽项 vs 延迟项：Message size 决定一切

一次 AllReduce 的时间：

$ T = alpha_"latency" + V / B_"effective" $

$V$ 小的时候（< 1 MB）延迟项主导，$V$ 大的时候带宽项主导。转折点约 1-8 MB (依赖 NCCL 参数)。这决定了：

+ *DDP bucket size* 要 ≥ 25 MB (PyTorch 默认)，太小会被延迟吃掉
+ *ZeRO-3 flat param* 要 shard 得够大，`FULL_SHARD` 每个 flat group ≥ 100 MB 是好的
+ *TP AllReduce* 一层 activation 通常 10-100 MB，天然合适
+ *EP All-to-All* 单层 100 MB - 1 GB，永远带宽项主导

== 通信量的三种度量

面试里经常混淆，务必区分：

+ *Per-GPU volume* (`sendrecv`)：一张卡自己发出/收到多少字节。上面表格给的是这个。
+ *Aggregate volume*：$W$ 张卡加起来。$"agg" = W times "per-GPU"$。
+ *Bisection bandwidth*：网络切成两半时穿过 cut 的总流量。跨节点集群里 bisection 决定 all-to-all 的性能。

Ring AllReduce 的 per-GPU 是 $2V$（bandwidth 侧），aggregate 是 $2 V W$（跨所有 rank 的总数据流）。IB fabric 的 bisection 通常是 $O("nodes")$，所以 all-to-all 在跨 100+ 节点会退化。

== 延迟分解

一次跨节点 send/recv 的延迟组成：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, left),
    [*阶段*], [*典型耗时*], [*说明*],
    [CUDA kernel launch],      [5-20 µs],   [每次 collective 都要 launch NCCL kernel],
    [NCCL 内部同步],           [10-30 µs],  [rank 之间的同步握手],
    [PCIe DMA (GPU→NIC)],      [~1 µs],     [大 message 里可忽略],
    [IB fabric (Switch hops)], [~1-3 µs],   [每 hop ~500 ns，spine-leaf 通常 2-3 hops],
    [NIC→PCIe→GPU (对端)],     [~1 µs],     [],
    [接收方 kernel launch],    [5-20 µs],   [],
    [Total (small msg)],       [~30-80 µs], [$alpha$ 约等于这个数],
  ),
  kind: table,
  caption: [跨节点集合通信延迟分解。这就是为什么小 message 会被延迟吃掉——payload 只有几 KB 时，30 µs 的启动开销让有效带宽掉到几 GB/s。],
)

*IBGDA (Infiniband GPUDirect Async)*：让 GPU 直接下发 IB verb，绕过 CPU，把 launch 相关的 20+ µs 干到 < 5 µs。DeepEP、UCX 都在用。

== 简易 profile：nsys 里怎么看通信

一个典型的 nsys trace 里：

```
[cuda kernel] gemm_kernel_0            2.34 ms
[cuda kernel] ncclKernel_AllReduce_..  0.85 ms   <- 通信
[cuda kernel] elementwise_add          0.12 ms
```

关键看的三点：

+ *NCCL kernel 与其他 kernel 是否 overlap*（不同 stream）
+ *NCCL kernel 内的 gap*（是不是 IB 卡了）
+ *通信占总 step 时间的 %*

Blackwell / Hopper 上还能用 NCCL profiler `NCCL_DEBUG_SUBSYS=COLL,PROFILE`。

== NCCL 调优 checklist

生产训练必调的环境变量：

```bash
# 通用
export NCCL_DEBUG=WARN                  # 或 INFO 排错时用
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=8    # 允许多 stream 并行 (Hopper 上建议 ≥4)
export TORCH_NCCL_HIGH_PRIORITY=1

# Ring / Tree
export NCCL_ALGO=Auto                   # 通常自动选好
export NCCL_MIN_NCHANNELS=8             # 更多 SM channel
export NCCL_NTHREADS=512

# IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3   # 用足所有 IB HCA
export NCCL_IB_GID_INDEX=3              # RoCEv2 常用
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_SL=1                     # Service Level

# H100 NVLS
export NCCL_NVLS_ENABLE=1

# Buffer
export NCCL_BUFFSIZE=8388608            # 8MB, 大 message 稳定

# Debug 慢通信时
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
```

*每个集群都要用 nccl-tests 实测选参数*——不同拓扑差别巨大。生产建议跑：

```bash
mpirun -np 64 --hostfile hosts \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

看输出的 `bus BW` 列，理想 H100 8 卡是 ~370 GB/s (NVLink) 或 30-40 GB/s (跨节点)。低于这个就要 debug。

== 拓扑感知的进程排布

*错误* 的做法：

```bash
torchrun --nproc-per-node=8 --nnodes=8 ...
# 默认 rank 0-7 在 node 0, rank 8-15 在 node 1, ...
# TP=8, DP=8: TP 组默认取连续 rank -> TP 组内是 NVLink OK
# 但如果 TP=2, DP=32: TP 组 = (0,1),(2,3),... 都在同一 node OK
# 反过来 PP=2 时 stage 之间会跨 node
```

*建议*：手动设置 process group 时确认三件事：

+ *TP 组必须在同一 NVLink 域内*（NVL8 就是同一 node，NVL72 允许一个 rack）
+ *DP 组的 AllReduce 可以跨节点*（带宽敏感度低）
+ *PP 组的 P2P 尽量在同 node 或同 spine*
+ *EP 组根据规模：EP ≤ 8 尽量同 node，EP > 8 只能跨*

Megatron-LM 的 `torch.distributed.new_group` 排列方式看 `megatron/core/parallel_state.py`；FSDP 用 `DeviceMesh` 更直观。

== 面试考点

#interview[
  *Q1*: AllReduce 的通信量为什么是 $2V(W-1)/W$，"2" 从哪来？

  A: Ring AllReduce = ReduceScatter + AllGather。两步各走 $V(W-1)/W$，总 $2V(W-1)/W$。$W$ 大时约等于 $2V$。这就是 AllReduce "两倍数据"的直觉。
]

#interview[
  *Q2*: DDP 的 gradient AllReduce 与 ZeRO-3 的 param AllGather + grad ReduceScatter 通信量对比？

  A: 假设模型参数 $P$ 字节：
  - DDP：$2P$ 字节 (grad AR)
  - ZeRO-3：forward 时 $P$ (param AG) + backward 时 $P$ (param AG) + $P$ (grad RS) = $3P$ 字节

  ZeRO-3 通信量 *多 50%* 但内存少得多（optim + grad + param 都切）。这是显存-通信 tradeoff。
]

#interview[
  *Q3*: 什么时候 Tree AllReduce 比 Ring 好？

  A: 小 payload + 大 world size。Ring 的延迟项是 $2(W-1) alpha$，$W=1024$ 时约 60 ms 只算延迟。Tree 只 $2 log_2(W) alpha ≈ 0.6$ ms。GPT-3 训练的 embedding sync 之类小 message 会自动切 Tree。
]

#interview[
  *Q4*: 为什么 NVLS (NVLink SHARP) 能加速 AllReduce？

  A: SHARP 让 reduction 在 NVSwitch 芯片里做——每卡只需 send 一次数据到 switch，switch 里做 sum 后广播回，把 $2V(W-1)/W$ 的 per-GPU volume 降到 $V/W + V/W = 2V/W$，即 *~1.5-2×* 提速。类似 InfiniBand SHARP 在网络上做。
]

#interview[
  *Q5*: bisection bandwidth 是什么？为什么 all-to-all 特别敏感？

  A: 网络切成任意等分两半，穿过切面的总带宽。all-to-all 里每对 rank 都要通信，跨节点流量 $prop n^2$；只有 bisection $prop n$ 的 fat-tree 才能不阻塞。100+ 节点的 IB fabric 里 bisection 常常 oversubscribe，all-to-all 掉到理论 1/3。
]

#interview[
  *Q6*: NCCL 的 `NCCL_MIN_NCHANNELS` 是什么？调高有代价吗？

  A: NCCL 用多个 CUDA channel（每 channel ~2 SM）来并行传输。channel 越多带宽越接近峰值，但*吃 SM*，与 kernel compute 争资源。H100 上 8 channel 大约 16 SM (12% of 132)，能拿到 90%+ 带宽；DeepEP V1 吃 20 SM 让 GEMM 掉 20% 效率——所以 V2 降到 4-6。这是通信-计算的 SM 分配 tradeoff。
]

#interview[
  *Q7*: 如果 NCCL AllReduce 突然变慢一半，你怎么排查？

  A: 按顺序：
  + `NCCL_DEBUG=INFO` 看是不是切换到 Tree/Ring 了（payload 边界）；
  + `nccl-tests` 测 raw 带宽，排除模型代码；
  + `ibstat` / `ethtool` 看 IB/Eth 是不是有 error 或 rate 掉；
  + `nvidia-smi topo -m` 看 GPU 与 NIC 的 affinity 是不是错了；
  + 排查慢卡：`NCCL_DEBUG_SUBSYS=COLL` + 每 rank 打时间戳，找拖后腿的 rank；
  + Node NIC 硬件重启（IB 偶尔会陷入 slow-mode）。
]
