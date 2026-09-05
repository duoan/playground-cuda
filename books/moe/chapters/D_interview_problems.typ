#import "../template.typ": *

= 附录 D：MoE 系统面试题 20 问

以下 20 题是 MoE 系统面试的高频问题，覆盖从基础概念到分布式实现。答案里注明了参考章节，可以回去精读。

== 基础概念 (Q1-Q5)

#interview[
  *Q1*: 用一句话解释 MoE 相比 dense FFN 的核心思想。

  A: 把一个大 FFN 拆成 $E$ 个小专家，用 router 让每个 token 只走 $K$ 个专家 ($K << E$)。参数量 $times E/K$，激活计算量几乎不变。见第 2 章"一句话直觉"。
]

#interview[
  *Q2*: Mixtral 8×7B 里 "8×7B" 是什么意思？总参数量约 47B 是怎么来的？

  A: 8 个专家，每个专家的 FFN 部分约 5.5B 参数。总参数 = 8 × 5.5B (experts) + 5B (attention + embedding + norm) ≈ 47B。"8×7B" 是营销措辞而非精确算术。见第 2 章"里程碑速览"。
]

#interview[
  *Q3*: 为什么现代 MoE 论文都强调"稀疏激活"？稀疏在哪一层？

  A: 稀疏在 *expert 选择* 上——每 token 只激活 $K/E$ 的专家。FLOPs 稀疏（大部分 expert 不计算），但*显存不稀疏*（所有专家权重都要装）。区别于 "activation sparsity"（激活值多为 0）。见第 2 章"MoE 的核心思想"。
]

#interview[
  *Q4*: MoE 相比 dense 训练需要更多什么？

  A: 主要三样：(a) *显存*，因为 $E$ 倍权重；(b) *通信*（分布式时 all-to-all）；(c) *代码复杂度*（router + dispatch + load balancing）。计算量是节约的，但工程复杂度显著上升。见第 2 章"代价：三个新问题"。
]

#interview[
  *Q5*: 什么情况下 MoE 不适合？

  A: (1) 单机小模型 (< 10B) — dense 更简单，性能相当；(2) 严格显存受限 (edge deployment) — MoE 权重全存装不下；(3) $B S$ 极小 (自回归 decode) — expert 内 GEMM 小到没效率；(4) 训练数据不足 — collapse 严重。
]

== Router 与负载均衡 (Q6-Q10)

#interview[
  *Q6*: 为什么 router 的 softmax 必须 fp32？给一个具体的 overflow 数字。

  A: fp16 上限 65504，`exp(11.1) ≈ 66000` 就溢出为 inf。训练中期 logit 值可到 ±15，`exp(15) ≈ 3.3e6` 必然 nan。fp32 支持 `exp(88)` 无 overflow。见第 3 章"为什么 softmax 要用 fp32"和第 9 章。
]

#interview[
  *Q7*: `torch.topk` 会返回相同 expert id 两次吗？

  A: 不会。`topk(gate_probs, k=K)` 从 $E$ 个不同的位置选 $K$ 个索引，索引本身天然 unique。除非 tie（值完全相等）——tie 处理是 stable ordering，返回的还是不同 index。见第 3 章面试考点 Q2。
]

#interview[
  *Q8*: Expert collapse 是什么？怎么诊断？怎么修？

  A: Router 学到"永远选那几个 expert"，其他 expert 收不到梯度，参数浪费。诊断：$"std"(f_e) / "mean"(f_e) > 0.5$ 持续。修：(1) 检查 init (`std=0.02`)；(2) 提升 aux loss $alpha = 0.01 -> 0.03$；(3) 加 z-loss；(4) 换 DeepSeek 的 aux-loss-free bias tuning。见第 6 章"问题重述"和"Aux Loss"。
]

#interview[
  *Q9*: Aux loss 的公式为什么是 $E dot (f dot P)$？各项什么意思？

  A: $f_e$ = expert $e$ 被 top-1 选中的比例 (不可微)，$P_e$ = gate_probs 均值 (可微)。均匀分布时 $f = P = 1/E$，$L_"aux" = E dot E dot (1/E)^2 = 1$（最小）。乘 $E$ 是为了让最小值 = 1（scale invariant）。见第 6 章 "GShard/Switch Aux Loss"。
]

#interview[
  *Q10*: DeepSeek-V3 为什么弃用 aux loss？bias tuning 怎么工作？

  A: aux loss 给 router 冲突梯度（主任务 vs 均衡）。DeepSeek 用可学习 bias $b_e$，*只影响 topk 选择*（$hat(ell) = ell + b$）、不影响 gate 权重梯度。$b$ 用简单反馈规则调整：过载专家 bias↓，欠载 bias↑。router 梯度纯净。见第 6 章"DeepSeek-V3 的 Aux-Loss-Free 均衡"。
]

== 实现与性能 (Q11-Q15)

#interview[
  *Q11*: 逐 expert for-loop (范式 A) vs Grouped GEMM (范式 B) 差在哪？

  A: (1) A 有 E 次 kernel launch，B 只有 1 次；(2) A 里每个 GEMM 的 M 维小 ($< 256$)，TensorCore 吃不满；B 内部全局 tile scheduling，SM 利用率高；(3) B 需要 permute+unpermute overhead。端到端 B 比 A 快 1.5-2×。见第 4 章"范式 A/B" 和第 7 章。
]

#interview[
  *Q12*: `expert_weights[token_ids, k_ids]` 的行为？为什么不是 `expert_weights[token_ids][:, k_ids]`？

  A: 前者是 2D fancy indexing，两个 index 同步 broadcast，输出 shape $(M,)$：每个 $(t, k)$ pair 取一个元素。后者是先取行再对每行的全部列取 k_ids，输出 shape $(M, M)$。是常见 bug。见第 5 章"Dispatch 循环"。
]

#interview[
  *Q13*: `index_add_` 在 GPU 上有确定性问题吗？MoE 里这个问题在哪些地方触发？

  A: 有——多个原子加的顺序取决于 CUDA 调度。在范式 A 的单次 expert 循环内 `token_ids` unique（不触发），但跨 expert 循环、同一 token 被 K 次 add — 顺序影响低位 bit。要 reproducible 训练需 `torch.use_deterministic_algorithms(True)`。见第 4 章"一个隐藏的正确性坑"和第 9 章。
]

#interview[
  *Q14*: Grouped GEMM 里 permute 阶段有多重要？为什么要 fuse 进 GEMM prologue？

  A: 独立 permute 要写读一次 $(N K, H)$ tensor 到 HBM，约几百 MB traffic。融进 GEMM prologue 直接从 hidden gather 到 shared memory，省掉一次 HBM 往返，端到端 +10-15%。见 §6, v3。
]

#interview[
  *Q15*: 单机 MoE 的显存开销主要来自哪里？

  A: (1) *权重*：$E$ 倍 FFN 权重是最大头；(2) *optimizer state*：AdamW 12 bytes/param × 权重量；(3) *激活*：`packed_hidden (N K, I)` 是最大 activation，selective ckpt 可省 30%；(4) grad 累积。见 §6。
]

== 分布式 (Q16-Q20)

#interview[
  *Q16*: 为什么 MoE 需要 Expert Parallel (EP)，用 TP/DP 不够吗？

  A: DP 每卡都要装完整 MoE — $E times$ 权重装不下。TP 切 expert 内 GEMM，切完后每卡还是有 $E/"TP"$ 个"部分专家"——本质是 dense 化，没有利用稀疏。EP 让每卡持有*完整 expert 的一子集*，权重 $times 1/"EP"$，通信只在被路由的 token 上做（a2a），符合 MoE 稀疏语义。见第 8 章"为什么需要 Expert Parallel"。
]

#interview[
  *Q17*: All-to-all 的通信量公式？为什么与 $E$ 无关？

  A: 每 rank 通信量约 $N_"local" K H times "bytes"$。传的是 token 的 hidden vector，每 token 走 $K$ 个专家、$K$ 份数据发出去、$K$ 份收回来。$E$ 只决定专家池大小，不改变每 token 的通信。$N_"local"=4K, K=2, H=4K, "bf16"$ → 128 MB per a2a。见第 8 章"通信量分析"。
]

#interview[
  *Q18*: 层次化 all-to-all 是什么？收益从哪来？

  A: 把 (N nodes × 8 GPUs)-way a2a 拆成 intra-node (NVLink, 400 GB/s) + inter-node (IB, 50 GB/s) + intra-node。IB 带宽消耗从 $O(N^2 K H)$ 降到 $O(N K H)$。100+ 节点场景 wall-clock 差 10×。DeepSpeed-MoE、Tutel 都实现。见第 8 章"层次化 All-to-All"。
]

#interview[
  *Q19*: DeepSeek-V3 的 fine-grained overlap 怎么工作？为什么能加速？

  A: 把每 MoE 层拆成 4 阶段 pipeline: dispatch a2a → attention → mlp → combine a2a。多个 micro-batch 交替，每 stage 一直有 work、通信持续 overlap 计算。传统实现通信占 forward 15-30%，overlap 后近 0%。代价：async collective + 精心排布的 CUDA stream。见第 8 章"Overlap"。
]

#interview[
  *Q20*: 分布式 MoE 训练中 aux loss 的 $f_e$ 需要 AllReduce 吗？

  A: 需要（跨 DP 组）。$f_e$ 是"全局 batch 内每 expert 的负载比例"—— 每 rank 各算得到的只是"本 rank 的负载"。训练早期可能本 rank collapse 到 expert 0、别 rank collapse 到 expert 1，local aux loss 都 0，*但 global 不均衡*。必须 AllReduce($f$)。这是分布式 MoE 训练最常被遗漏的一步。见第 9 章"Send/recv counts 不匹配"和相关小节。
]

== 加分题：动手实操

如果面试官愿意手撸代码，可能会问：

*B1*: 现场写一个最小 MoE forward (禁用 grouped GEMM)。

- 参考本书 §4 逐段实现。关键点：`view(-1, H)`、`F.softmax(dtype=fp32)`、`torch.topk`、renorm、`for expert_idx in range(E)` 循环、`torch.where`、`expert_weights[token_ids, k_ids]`、`index_add_`。

*B2*: 现场推导 aux loss 的最小值证明。

- 均匀分布 $f = P = 1/E$ 时 $L = E dot sum_e (1/E)(1/E) = E dot E dot 1/(E^2) = 1$。要证是最小值：拉格朗日 or Cauchy-Schwarz。

*B3*: 现场画 EP=4, K=2, N_local=4 的 all-to-all 前后 tensor 布局。

- 画一个 $4 times 4$ 的 (rank, expert) grid，每 rank 4 个 token 各走 2 个 expert，画出 send/recv counts 和 packed_input shape。

*B4*: 现场写一段 grouped GEMM 的 permute (不用 argsort)。

- Counting sort: `bincount` → `cumsum` → 得每 expert 的 start offset → scan `expert_indices` 填 permute_idx。生产实现用这个避免 argsort。
