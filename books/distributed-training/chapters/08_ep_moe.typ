#import "../template.typ": *

= Expert Parallel 与 MoE 训练概览

MoE 训练的详细内容（router, dispatcher, all-to-all, DeepEP, DualPipe）都在姊妹卷《Sparse MoE 训练实战》里。这一章只讲*与其他并行策略的集成点*和*面试里最容易混淆的概念区分*，避免重复。

== EP 的核心：把 expert 沿 rank 切

MoE 层的稀疏性 = "每 token 只走 $K/E$ 的专家"。切 EP 就是把 $E$ 个专家分给 $"EP"$ 张卡，每卡持 $E/"EP"$ 个 expert 的完整权重。

#figure(
  align(center, moe-dispatch(
    n-tokens: 8, n-experts: 4, cell: 0.55,
    routing: (0, 2, 1, 0, 3, 2, 1, 3),
    title: "MoE dispatch: 8 tokens → 4 experts (top-1 routing)",
  )),
  caption: [Router 给每 token 打分选 top-k expert。EP 情况下 expert 分散在多卡上，同色 token 必须*聚到*对应 rank——这一步就是 all-to-all dispatch。反向再一次 all-to-all 把结果 combine 回原 rank。],
) <fig-moe-dispatch>

代码见 `src/distributed_training/09_ep_moe.py`——用一次 `all_to_all_single` 完成 dispatch，另一次做 combine，与单卡 full-experts baseline 数值对齐。

*四种并行在 MoE 里的分工*：

#figure(
  table(
    columns: (auto, 1.4fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [*并行*], [*在 MoE 里干嘛*],
    [DP],   [batch 沿样本切，非-expert 部分 grad AllReduce],
    [TP],   [切 attention head + expert 内 GEMM 沿 $I$ 维 (ETP)],
    [PP],   [沿 layer 切，MoE 层与 dense attention 层交替 stage],
    [CP],   [attention 沿 seq 切，MoE 部分 seq 也切],
    [EP],   [MoE 专属：expert 沿 expert-id 维切，dispatch/combine 用 all-to-all],
  ),
  kind: table,
  caption: [五种并行在 MoE 里的角色。EP 是 MoE 独有；其他四个在 dense 与 MoE 里作用相同。],
)

== EP × DP 的关系

*非-expert 参数*（attention, gate, LN, embedding）：跨 EP 组*完全复制*——因为它们本来就该完整存在每卡，与 expert 分片无关。这些参数的 grad 用 DP AllReduce（也可能跨 EP 组）。

*Expert 参数*：EP 组内切分（每 rank 持不同 expert），EP 组间也可能有 replica（若 EP < world_size）。这时 expert 的 grad 需要"expert-DP 组"内 AllReduce。

Megatron-Core 的 `--expert-model-parallel-size` 与 `--data-parallel-size` 组合：

- `EP=8, DP=8, world=64`：8 个 EP 组，每组 8 卡；每 EP 组的 expert 有 8 份 replica，跨 8 个 DP rank 做 AllReduce
- `EP=64, DP=1, world=64`：单一 EP 组，无 DP replica，无需 expert-DP AR

第二种通信少但受限于 `batch = MBS × PP` (无 DP 加大)。生产常用 EP × DP 混合。

*Parallel Folding* (Megatron-Core 2026)：打破 EP ≤ DP 约束，Attention 与 MoE 用独立 process group。允许配置：$"attn-DP" = 32, "expert-EP" = 64, "expert-DP" = 2$ 之类不对称。

== 通信量与 dense 的对比

Dense Transformer 一层 forward + backward：
- DP: 每 step $2P$
- TP: 每层 2 AR (fwd+bwd) × 4 层组件 = 8 AR/step, activation-sized
- PP: P2P

MoE 一层多出：
- 2 次 dispatch a2a (fwd + bwd)
- 2 次 combine a2a (fwd + bwd)
- 每次 volume $approx B S K H / "EP" times "bs"$

对 $B=1, S=4K, K=2, H=4K, "EP"=8, "bs"=2$: 单层 a2a = 4 × 2 × 4096 × 4096 × 2 / 8 = 32 MB。32 层 = 1 GB per step。跨节点 IB 50 GB/s 需要 20 ms —— 不 overlap 就吃掉 10% step time。

这就是为什么 MoE overlap (DualPipe / FWD-BWD merged) 比 dense 更重要。

== 与 CP 的组合：MoE 长序列训练

Kimi K2, DeepSeek-V3.2 等长上下文 MoE 需要 EP + CP + DP。

*注意点*：
+ MoE 层的 dispatch a2a 与 CP 的 Ring/Ulysses P2P *会争带宽*
+ 一般让 EP 与 CP 走不同 device mesh 维度：EP 在同节点 NVLink，CP 沿另一维度
+ FSDP 沿"pure DP"轴切 param，不含 EP 组内的 expert
+ *router 的 load-balancing aux loss 必须在 CP 组上归约 expert 直方图*。$f_e$（token 比例）和 $P_e$（平均概率）都是"对 token 求平均"，CP 下每卡只看到 $1\/"CP"$ 的 token。只统计本卡会让一个全局均衡的 router 被判为极度不均衡，梯度方向反而把各卡推离全局最优——征状是 load-balance metric 长期震荡不收敛。同理适用于 z-loss 和任何从 per-rank token 数推出的 capacity / drop rate。详见 §7 与 `19_cp_loss_and_metrics.py`

*示例* Kimi K2 32-expert MoE, 1M ctx on 512 H100:
```
world = 512
TP = 8         (NVLink)
CP = 16        (USP: Ulysses 8 × Ring 2)
EP = 32        (32 experts, one per rank in EP group)
PP = ?         (取决于层数)
DP = auto-fill
```

CP 组与 EP 组尽量正交（不共享 NVLink 带宽），实测里 CP 走同 node 8 卡 NVLink，EP 跨节点 IB（EP a2a 用 hierarchical / DeepEP 缓解）。

== EP 通信量公式速查

从 MoE 书里搬过来最核心的几个公式：

*Dispatch (forward)*，每卡 send:

$ "vol"_"dispatch" approx N_"local" K H "bs" times ("EP" - 1)/"EP" $

*Combine (forward)*：同 dispatch，方向反。

*一层 forward+backward total per-GPU*: 4 × dispatch = $4 N_"local" K H "bs"$（$"EP"$ 大时）。

*与 $E$ (总 expert 数) 无关*——每 token 还是走 $K$ 个专家，通信量不变。这是 DeepSeek-V3 用 256 experts 却通信量与 8 experts 一样的原因。

== ETP: Expert Tensor Parallel

*背景*：如果单个 expert 权重太大（$I$ 大），单卡装不下 → 把 expert 内的 GEMM 沿 $I$ 维 TP 切。

Megatron flag: `--expert-tensor-parallel-size 2` (与 `--tensor-model-parallel-size` 独立)。

*用法*：
- Mixtral 8×22B: $I=16384$，单 expert weight = 128 MB，*不需要 ETP*
- 假想 100B expert-size：需要 ETP=2 或 4

DeepSeek-V3 用 EP=64, ETP=1 (fine-grained expert 都很小)。Qwen3-235B MoE 类似。

== 与 PP 的互动: MoE 层 vs Dense 层

一个 61 层的 DeepSeek-V3 里，MoE 层与 dense (attention only) 层交替：

- 前 3 层：dense (只 attention + MLP，无 MoE)
- 中间：dense MLP + MoE FFN 组合
- 每层的 attention 部分与 MoE FFN 部分并列

PP 切分时要考虑各 stage 的 FLOPs 均衡：
- MoE 层 FLOPs ≈ 2 × attention FLOPs（有 gate + expert compute）
- 单纯按层数均分 → MoE 集中的 stage 慢
- Megatron `--custom-pipeline-partitioning` 手动分：dense 密集 stage 多几层，MoE stage 少几层

== 常见配置模板

从 MoE 书里搬过来的两份典型配置：

*Mixtral 8×7B on 64 H100* (标准 MoE):
```
TP=1, PP=4, EP=8, DP=2
Dispatcher: alltoall
overlap: --overlap-moe-expert-parallel-comm --delay-wgrad-compute
Recompute: selective
Precision: BF16
```

*DeepSeek-V3-like fine-grained (256 exp) on 512 H100+*:
```
TP=1, PP=16, EP=64, DP=2 (或更多 for GBS)
Dispatcher: flex + deepep
overlap: DualPipe (or FWD-BWD merged for simpler)
Recompute: fine-grained modules
Precision: FP8 blockwise
```

详细每个 flag 的含义与调参见 MoE 书第 8 章。

== 面试考点

#interview[
  *Q1*: 为什么 EP 是 MoE 独有，dense 模型不用？

  A: EP 切 "expert-维度"，dense 模型没有 expert——每层就是一个 FFN，切它的方式是 TP (沿 hidden dim) 或 SP。MoE 有 E 个独立 FFN，天然可以按 expert-id 切，稀疏的路由让通信只对*涉及*的 tokens 做（all-to-all），比 dense TP 的 全量 AllReduce 更符合稀疏语义。
]

#interview[
  *Q2*: EP=8 与 TP=8 都能把 expert 权重切 8 份，选哪个？

  A: EP。TP 切 expert 后每卡还是有 $E$ 个"半专家"—— dense 化，浪费稀疏性。EP 让每卡持*完整*的 $E/8$ 个专家，通信只在 dispatch 时对*被路由的 tokens*做，与 expert 数无关。EP 通信 $prop B S K H$，TP 通信 $prop B S H$（但每层都 AR），EP 更符合 MoE 稀疏本质。
]

#interview[
  *Q3*: EP 组的 all-to-all 与 TP 组的 AllReduce 会争带宽吗？

  A: 会。EP 组与 TP 组物理上共享 NVLink/IB。生产做法：让 EP 与 TP 走不同 device mesh 维度，让 TP 组同 NVLink 域 (NVL8)、EP 组尽量也同域 (EP ≤ 8)。EP > 8 只能跨节点，用 hierarchical a2a 或 DeepEP 减轻。DeepSeek-V3 干脆不用 TP。
]

#interview[
  *Q4*: 为什么 MoE 训练 overlap 比 dense 更重要？

  A: dense 一层 forward 只有 attention output projection 和 FFN 后的 2 次 AR（TP 组）；MoE 多了 2 次 all-to-all (dispatch + combine)。a2a 是 EP 组间的大 payload 通信，占 step 15-40%（未 overlap）。overlap 到 \<5% 需要 DualPipe 或 Megatron FWD-BWD merged。dense 里通信占比小，overlap 收益也小。
]

#interview[
  *Q5*: expert-DP 与 attention-DP 分开有什么用？

  A: Parallel Folding。传统 Megatron 要求 EP × 内的 DP 与 attention 的 DP 相同（因为 world_size 拆分是全局的）。Parallel Folding 允许："attention 部分" DP=32, "expert 部分" EP=64+DP=2——非 expert 部分获得更大有效 batch (32×MBS)，expert 部分利用大 EP 减少通信。DeepSeek 类 fine-grained MoE 常用。
]

#interview[
  *Q6*: fine-grained MoE (256 experts) 比 vanilla MoE (8 experts) 显存/通信量各是几倍？

  A: 通信量：*相同*（不依赖 E）。显存：weight 相同（相同总参数），但 grouped GEMM 每 expert 的 M 维小 8×（每 batch 里每 expert 平均更少 tokens）→ 更容易 memory-bound。所以 fine-grained MoE 需要更好的 grouped GEMM kernel (DeepGEMM, TE GroupedLinear) 才不掉率。
]
