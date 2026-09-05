#import "../template.typ": *

= Tensor Parallel 与 Sequence Parallel

Tensor Parallel (TP) 由 Megatron-LM (Shoeybi 2019) 提出，把单个 matmul 沿某一维切给多张卡，在层内完成一次 AllReduce 拼回结果。Sequence Parallel (SP, Korthikanti 2022) 补上 TP 没切的 LN / Dropout activation。两者结合是 dense LLM 训练标配。

== TP 的核心思想：一层 GEMM 里就切

考虑一个 FFN：$Y = X W_1 W_2$，$W_1 in RR^(H, 4H)$，$W_2 in RR^(4H, H)$。

*Column parallel* $W_1$：把 $W_1$ 沿 output ($4H$) 维切成 $W_1 = ["W"_1^((1)), "W"_1^((2))]$。

$ X W_1 = [X "W"_1^((1)), X "W"_1^((2))] $

每卡各自算，得到 activation 沿 output 维切分。*不需要通信*。

#figure(
  align(center, tp-partition(mode: "column", tp: 4, w: 4.5, h: 1.4,
    title: "Column-parallel weight (out dim sharded across TP=4)")),
  caption: [列并行：$W$ 沿 out 维（4H）切，输入 X 复制。每卡独立算 $X W^((i))$，输出天然沿 out 维切分，不需通信。],
) <fig-tp-col>

*Row parallel* $W_2$：把 $W_2$ 沿 input ($4H$) 维切成 $W_2 = mat("W"_2^((1)); "W"_2^((2)))$。

$ Y = X W_2 = X^((1)) "W"_2^((1)) + X^((2)) "W"_2^((2)) $

每卡拿到自己切分的 input 与 row，各算部分和；最后 *AllReduce* 求和拼回。

#figure(
  align(center, tp-partition(mode: "row", tp: 4, w: 4.5, h: 1.4,
    title: "Row-parallel weight (in dim sharded across TP=4)")),
  caption: [行并行：$W$ 沿 in 维切，输入必须已按 in 维切分。每卡算部分和，最后一次 AllReduce 求和。],
) <fig-tp-row>

*连起来*：$X -> X W_1$ (column, 无通信) → activation ($Y_1$) 沿 $4H$ 切分 → 送 GELU（elementwise，无通信）→ 送 $W_2$ (row) → *AllReduce*。整个 FFN 只 1 次 AllReduce（forward）+ 1 次（backward）。

#figure(
  align(center, sp-tp-flow(steps: (
    ("(B,S,H)", "@col W1"),
    ("(B,S,I)", "GELU"),
    ("(B,S,I)", "@row W2"),
    ("(B,S,H)", "AR ↓"),
    ("(B,S,H)", none),
  ), box-w: 1.8)),
  caption: [Megatron FFN 的 column-then-row 数据流。整个 FFN 只在最后一次 AllReduce（forward），backward 再一次。],
) <fig-tp-ffn-flow>

代码见 `src/distributed_training/04_tp_column_row.py`——用 200 行实现了 `ColumnParallelLinear` + `RowParallelLinear` + TP FFN + TP Attention，并与单卡 baseline 数值对齐（`torch.allclose`）。

这就是 Megatron 的 "column-then-row" pattern。所有 Transformer 组件 (attention, MLP) 都能这样组织。

=== Attention 的 TP 分解

Multi-head attention 天然按 head 切：

- $W_Q, W_K, W_O$ 沿 head 维 column parallel（每 head 独立）
- Attention compute（$Q K^T$, softmax, $P V$）本地做，*不需要通信*
- $W_O$ row parallel + AllReduce

每个 attention 层 forward 1 次 AllReduce，backward 1 次。加上 FFN 的两次，一整层 4 次 AllReduce。

*GQA / MQA*：head 数少（Grouped Query）时 K/V 沿 head 切会遇到 head 不够分的问题。做法：K/V 复制到多 TP rank（重复存储），Q 正常切。或者反过来 KV 沿 head 切 Q 复制——各 TP 组内做。

== 通信量与效率

每层 forward + backward 4 次 AllReduce。activation 大小 = $B S H times "bs"$。TP=8 时每次 AR：

$ "vol"_"TP" = 2 (W-1)/W times B S H times "bs" approx 2 B S H "bs" $

$B=1, S=4096, H=8192, "bs"=2$ (BF16): $= 128$ MB per AR。32 层 × 4 次 × 128 MB = 16 GB per step。

#cost-table(
  header: ([TP=], [每 AR bytes], [每层 AR 数], [400GB/s NVLink 每层耗时]),
  ([2], [$ (B S H) / 2 · b · 2$], [4], [~0.16 ms]),
  ([4], [$3/4 · B S H · b · 2$], [4], [~0.24 ms]),
  ([8], [$7/8 · B S H · b · 2$], [4], [~0.28 ms]),
  ([16], [$15/16 · B S H · b · 2$], [4], [~0.30 ms]),
)

注意随 TP 翻倍通信 bytes 增速 → per-GPU compute 减半，*通信/计算比翻倍*。这就是 TP 不能拉太大的根本原因。参考 `src/distributed_training/estimators.py::comm_volumes()` 里 TP 部分。

*TP 只能在高带宽域内用*——AllReduce 频率是 DP 的 32×+ (每层 vs 每 step)。跨节点 IB 不够快，会把 step time 拖垮。

*铁律*：$"TP" <= "同 NVLink 域大小"$。H100 NVL8 只能 TP=8，GB200 NVL72 可以 TP 到 72（但没意义，见下）。

=== 为什么 TP > 8 通常不划算

+ *AllReduce 时间*：$T_"AR" prop (W-1)/W$，$W=8 -> 87.5%$，$W=16 -> 94%$，$W=64 -> 98%$。加一倍卡通信只增 6%——但*compute 每卡少一半*。带宽利用率骤降。
+ *Activation 也在 TP 组内切*：TP=8 → activation 每卡 $1/8$；TP=64 → 每卡 $1/64$。但每次 AllReduce 都要传送近 $H$ 的完整 activation → 通信量与 TP 无关，*即 TP 越大越不划算*。

DeepSeek-V3 明确*不用 TP*，全用 EP + FSDP。GB200 NVL72 上 Megatron 也建议 TP ≤ 8 而不是拉到 72。

== Sequence Parallel (SP)：把 LN 也切了

Megatron TP 里，一层 forward 的顺序：

#figure(
  align(center, op-stack(steps: (
    ("Input X",              "(B, S, H)",        "full"),
    ("LayerNorm",            "(B, S, H)",        "full"),
    ("Dropout",              "(B, S, H)",        "full"),
    ("QKV proj (col-para)",  "(B, S, H/T)",      "shard-h"),
    ("Attention (local)",    "(B, S, H/T)",      "shard-h"),
    ("Out proj (row-para)",  "partial sum",      "shard-h"),
    ("AllReduce",            "(B, S, H)",        "comm"),
    ("LayerNorm",            "(B, S, H)",        "full"),
    ("Dropout",              "(B, S, H)",        "full"),
    ("FFN col + row",        "partial sum",      "shard-h"),
    ("AllReduce",            "(B, S, H)",        "comm"),
  ), width: 7.2, cell-h: 0.55)),
  caption: [Megatron TP 一层 forward 顺序。黄色格表 hidden 维已切分（每 rank 只留 $1/T$），蓝色格表张量仍是完整 $(B,S,H)$ —— 这就是 LN/Dropout 显存痛点：TP 完全没帮上忙。粉色格是每层的两次 AllReduce。],
) <fig-megatron-tp-flow>

*痛点*：LN 和 dropout 的 activation 是 $B S H$（完整），TP 完全没帮上忙。对 $B=1, S=32K, H=8192$：一个 layer 的 LN activation = 512 MB。80 层 = 40 GB，光 LN activation 就爆显存。

*SP 的解决*：LN + Dropout 沿 *sequence 维* 切分。这段 $X in (B, S/"TP", H)$，每卡处理 $1/"TP"$ 长度的序列。

*通信改造*：
- LN 出来 (B, S/TP, H) → 进 QKV projection 前需要 AllGather 到 (B, S, H)
- Output projection 出来 (B, S, H, partial) → 直接 ReduceScatter 到 (B, S/TP, H)

原来 forward 一层的 2 次 AllReduce 变成 2 次 AllGather + 2 次 ReduceScatter。回顾第 1 章：$"AllReduce" = "AG" + "RS"$，*通信量完全相同*！

#figure(
  align(center, sp-tp-flow(steps: (
    ("(B,S/T,H)", "AG"),
    ("(B,S,H)",   "TP"),
    ("(B,S,H)",   "RS"),
    ("(B,S/T,H)", none),
  ), box-w: 1.9)),
  caption: [SP + TP 数据流。SP 区间张量沿 seq 维切成 $(B,S/T,H)$（LN/Dropout 只吃 $1/T$ 显存）；进入 TP 区间前 AllGather 到 $(B,S,H)$；TP compute 完 ReduceScatter 回 SP 区间。],
) <fig-sp-tp-flow>

*收益*：LN/Dropout activation 显存从 $2 B S H$ 降到 $2 B S H / "TP"$，别的照常。TP=8 时 activation 省 87.5%。

代价：0（通信量不变），仅代码复杂化。所有 Megatron 生产训练都开 SP。

Megatron flag: `--sequence-parallel`。

#note[
  SP 把 activation 沿 seq 维切开之后，*模型里所有跨 token 计算的模块都被切断了*——loss 的分母、MoE 的 router aux loss、pooling 头、grad norm、以及 label shift 这类只差一个 token 的位移。这些改造与 CP 完全共用一套规则（SP 和 CP 本质都是 sequence sharding），统一放在 §7 讲。

  这里只强调一点：LN/Dropout 本身*不需要任何通信*就能在 seq-shard 上正确执行，因为它们沿 hidden 维归一化、逐 token 独立——这正是 SP 能零通信成立的原因。
]

=== TP-only 与 TP+SP 数据流对比

#figure(
  align(center, sp-tp-flow(steps: (
    ("(B,S,H)", "TP"),
    ("(B,S,H)", "AR"),
    ("(B,S,H)", none),
  ), box-w: 2.0)),
  caption: [*TP-only*：整条链上张量都是完整 $(B,S,H)$，每层一次 AllReduce。LN/Dropout activation 显存无节省。],
) <fig-tp-only>

#figure(
  align(center, sp-tp-flow(steps: (
    ("(B,S/T,H)", "AG"),
    ("(B,S,H)",   "TP"),
    ("(B,S,H)",   "RS"),
    ("(B,S/T,H)", none),
  ), box-w: 1.9)),
  caption: [*TP + SP*：SP 区间张量沿 seq 维切成 $S/T$，只在 TP 区间还原为完整 $(B,S,H)$。通信量与 TP-only 完全相同（$"AG"+"RS"="AR"$），但 LN/Dropout 显存降到 $1/T$。],
) <fig-tp-plus-sp>

AG + RS 与原来的 AR *完全等价*，NCCL 也可以 fuse。

== Async TP / TP overlap

TP 的 AllReduce 是同步的——AG 完才能 compute。Megatron-Core 2024 加了 async TP：

- `--tp-comm-overlap`：把 AR 拆成"partial + delayed"，与 GEMM overlap
- 依赖 Transformer Engine (TE) 的 fused `LayerNormMLP` / `LayerNormLinear`——kernel 里 chunked GEMM，边算边通信

*收益*：TP=8 通信占 15-20% → 5-8%。生产 Llama 70B 训练一定开。

== Expert Tensor Parallel (ETP)

MoE 场景下：expert 内的 GEMM (H → I → H) 也可以 TP 切。Megatron-Core flag: `--expert-tensor-parallel-size 2`。

*用法*：如果 expert 权重太大 ($I$ 很大)，单卡装不下。等价于把 EP 组内的每个 expert 再切 ETP 份，需要在 expert 内再做 AllReduce。

DeepSeek-V3 用 EP=64 + ETP=1 (不切 expert)，因为 fine-grained expert 权重本来就小。Mixtral 8×22B 有时用 EP=8 + ETP=2。

== TP × PP × DP × EP 的拓扑排布

*Rank 分配*的物理意义（Megatron 默认顺序）：

```
world = TP × CP × EP × DP × PP
             (最内)             (最外)
```

即：TP 组是相邻 rank（同 node 内的 NVLink 高带宽），DP 组跨 node，PP 组更外层。

例：world=64, TP=8, PP=2, DP=4:
- TP 组: {0-7}, {8-15}, ..., 每组同 node
- DP 组: {0, 8, 16, 24}, ..., 跨 node
- PP 组: {0, 32}, {1, 33}, ..., 跨 node

#figure(
  align(center, topology-grid(
    rows: 2, cols: 8, cell: 0.55,
    // Row 0 = PP stage 0 (GPUs 0-7 on node A, 8-15 on node B, etc.)
    // Row 1 = PP stage 1
    // Groups color by TP set: TP=8 → 1 group per row × per 8 GPUs.
    // We colorize by DP index for visibility (each DP replica same color).
    groups: (
      // DP 0/1/2/3 on stage 0
      (0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0),
    ),
    title: "TP=8 组: 每行的 8 张卡（同 NVLink 域）",
  )),
  caption: [world=64 TP=8 PP=2 DP=4 的物理布局。每 8 张卡组成一个 TP 组（需要同 NVLink 域）；两行是 PP 两个 stage；DP 沿多个 node 复制。],
) <fig-topo-tp8-pp2>

*调整原则*：
+ TP 组必须*同 NVLink 域*（NVL8 最多 TP=8）
+ EP 组尽量*同 NVLink 域*，否则 all-to-all 会慢（DeepSeek 用 hierarchical a2a 缓解）
+ DP 组可以跨节点（AllReduce 对带宽宽容）
+ PP 组可以跨节点（P2P send 只在阶段过渡时发生）
+ CP 组*同 NVLink 域*最好（下章）

== 与 FSDP 组合

Megatron-Core 传统 TP+PP+DP 是自己实现的三维网格。torchtitan 用 FSDP2 + TP：

```python
mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))

for layer in model.layers:
    parallelize_module(layer, mesh["tp"], {
        "attention.wq":  ColwiseParallel(),
        "attention.wk":  ColwiseParallel(),
        "attention.wv":  ColwiseParallel(),
        "attention.wo":  RowwiseParallel(),
        "mlp.w1":        ColwiseParallel(),
        "mlp.w3":        ColwiseParallel(),  # SwiGLU 的 gate
        "mlp.w2":        RowwiseParallel(),
    })
    fully_shard(layer, mesh=mesh["dp"])

fully_shard(model, mesh=mesh["dp"])
```

用 DTensor 的 `ColwiseParallel` / `RowwiseParallel` 标注每个 module，PyTorch 底层自动插入 AllGather/AllReduce。torchtitan Llama-3 70B 用这个跑到 40%+ MFU。

== TP 的坑

+ *`LayerNorm` weight/bias 是 replicated*：每 TP rank 都有一份，梯度合并要 AR (DDP 自动)。不合并会 drift
+ *`nn.Embedding` 的 TP*：Vocab 大 (128K+) 时 embedding 是大头。Megatron 用 "vocab-parallel embedding"：vocab 沿 TP 切，index 到不同 rank，output 用 AllReduce
+ *loss 计算*：cross_entropy 的 logits 是 (B, S, V)，V 大时是巨大 activation。Megatron 有 fused vocab-parallel cross entropy，直接在切分的 logits 上算。TE 也有 `parallel_cross_entropy`
+ *确定性 seed*：TP 组内 dropout mask 必须*完全一致*（因为不同 rank 处理同一 sample 的不同切片，dropout 语义要求跨 rank 一致）。set seed by (rank // TP)
+ *梯度合并顺序*：TP replicated param (LN weight) 的梯度需要在 TP 组内 AR + DP 组间 AR。顺序：先 TP，再 DP。DDP 里 `process_group=tp_group` 单独 wrap
+ *checkpoint save/load*：TP 权重是 shard 的，checkpoint 要按 TP rank 分片存。Megatron 有 dist ckpt，torchtitan 用 `torch.distributed.checkpoint`

== 面试考点

#interview[
  *Q1*: 为什么 Megatron 的 FFN 是"column-then-row"，反过来能行吗？

  A: 反过来 (row-then-column) 每层需要 2 次 AllReduce (row 之后一次，column 之后又一次)——但 column 出来的 activation 无需 AR。column-then-row 只需 row 之后一次 AR。语义等价但通信量差 2×。所以选前者。
]

#interview[
  *Q2*: TP=8 一层 forward 通信几次？activation 显存怎么算？

  A: FFN 1 次 AR，attention 1 次 AR，总 2 次 forward + 2 次 backward = 4 次。开 SP 后是 2 次 AG + 2 次 RS = 4 次（volume 相同）。activation 显存：TP 只切 QKV/FFN 中间 activation ($1/"TP"$)，LN 部分不切。开 SP 后 LN activation 也切 → 全部 $1/"TP"$。
]

#interview[
  *Q3*: Sequence Parallel 通信量与只用 TP 一样，那 SP 为什么"值得做"？

  A: SP 不改变通信量（AG+RS = AR），但把 LN / Dropout 的 activation 显存从 $B S H$ 降到 $B S H / "TP"$。对长序列（S=32K+）这一项能省 GB 级显存，允许更大 batch 或更长 seq。零通信代价的免费显存。
]

#interview[
  *Q4*: 为什么 TP 不超过 8？

  A: (1) 物理: NVLink 域 8 卡, 跨域走 IB 太慢；(2) AR 通信量随 TP 增长: $2(W-1)/W$ 逼近 2 后不再省, 但 compute 每卡还是 $1/W$ 降；(3) activation 沿 TP 切但 AR 通信量不变——TP 越大越不划算。GB200 NVL72 上 Megatron 建议 TP ≤ 8 而非 72。
]

#interview[
  *Q5*: Async TP 是怎么 overlap 的？

  A: `--tp-comm-overlap` (Megatron + TE)。把 GEMM 拆成 tile，AR 也拆成 chunked ReduceScatter+AllGather。GEMM tile $k$ 算完后立刻起 chunk $k$ 的通信，tile $k+1$ 继续算，通信与后续 tile compute overlap。要 kernel 支持（TE `LayerNormLinear` / `LayerNormMLP` fused）。TP 通信占比 15-20% → 5-8%。
]

#interview[
  *Q6*: GQA (num_kv_heads=8) 在 TP=8 情况下 K/V 怎么切？

  A: K/V 只有 8 个 head，正好每卡 1 个，OK。如果 TP=16 而 num_kv_heads=8：K/V 每卡 0.5 head 不行——要么复制 K/V (每 2 个 TP rank 存一份完整 KV head)，要么调 TP=8。Llama-2/3 里 GQA=8 是特意为 TP=8 设的。
]

#interview[
  *Q7*: TP + ZeRO-3 一起用有意义吗？

  A: 有。ZeRO-3 沿 DP 组切 param，TP 沿 TP 组切 tensor——正交维度。TP=8 且 DP=64 时 param 切 512 份。torchtitan 就是这样跑 Llama-3 70B。FSDP1 + TP 组合有 bug，用 FSDP2。
]

#interview[
  *Q8*: 为什么 embedding layer 也要 TP 切？

  A: Llama-3 vocab=128256，embedding weight = 128256 × 8192 × 2 bytes = 2 GB。TP=8 切 4 份不算很省，但 loss 计算里 logits (B, S, V) 是 4 GB (S=4K)，*不切显存爆*。Megatron 的 vocab-parallel embedding 把 V 沿 TP 切，logits 也切分，最后 fused cross_entropy 直接在切分 logits 上算，省下 loss activation。
]
