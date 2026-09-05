#import "../template.typ": *

= MoE 是什么，为什么要有

在动手写代码前，先花一章建立*直觉*——为什么 dense Transformer 撑不下去、稀疏 MoE 到底解决什么问题、代价是什么。这一章不涉及具体实现，但每一段都对应后面章节要动手做的事。

== 一句话直觉

*Dense FFN*：每个 token 都走同一个大 FFN。参数量 = 计算量。

*MoE FFN*：把大 FFN 拆成 $E$ 个"专家"（每个是小 FFN），每个 token 只走其中 $K$ 个（$K << E$）。

结果：*总参数量 $times E / K$，激活计算量几乎不变*。这是 MoE 的核心卖点——"同样的 FLOPs，更大的模型"。

用图对比一下（$E=4, K=1$）：

#align(center)[
  #flow-boxes(
    boxes: ("token", "BIG FFN", "output"),
  )
  #v(0.4em)
  _Dense: 每个 token 都过唯一的大 FFN_
]

#v(0.6em)

#align(center)[
  #flow-boxes(
    boxes: ("token", "router", "expert_2", "output"),
  )
  #v(0.4em)
  _MoE (K=1): 每个 token 只激活 4 个专家中的 1 个_
]

== 为什么 dense 撑不下去：scaling law 的视角

从 Kaplan 2020 / Chinchilla 2022 起，我们已经很清楚：*loss 关于参数量 $P$、数据量 $D$、计算量 $C$ 是幂律关系*。想让 loss 更低，只有三条路：加参数、加数据、加计算。

Dense 架构的问题：三者是*强绑定*的。

$ C_"forward" approx 2 P N_"tokens" $

要更强的模型，就得更多参数；参数一多，同数据下的 FLOPs 就同步增加。训练一个 dense 70B 用 15T token 需要约 $C = 2 times 7 times 10^10 times 1.5 times 10^13 = 2.1 times 10^24$ FLOP——这大概是 6000× H100 训 3 个月。要 700B？直接 10 倍。

#insight[
  Dense scaling 的"物理"局限：*每个 token 都要读一遍所有参数*。参数量线性 $arrow.r$ 每 token 内存流量线性 $arrow.r$ 训推成本线性。想打破这个绑定，必须让"读多少参数"和"总参数量"解耦——这就是 conditional computation 的思想。
]

== MoE 的核心思想：条件计算

*Conditional Computation* (Bengio et al. 2013, Shazeer et al. 2017)：不是所有 token 都需要全部知识。给每个 token 一个 "router"，只激活相关子网络。

用数学讲：把一个 FFN 变成 $E$ 个 FFN 的加权和，但权重是*稀疏*的：

$ y = sum_(e=1)^E g_e (x) dot "FFN"_e (x) $

其中 $g_e(x) in [0, 1]$ 且 $sum_e g_e = 1$，且 $g$ 只在 $K$ 个专家上非零——其他专家*不计算*，等于 0 加进去。

于是：

$ P_"total" = E dot P_"expert" quad ("模型容量") $
$ P_"active" = K dot P_"expert" quad ("每 token 计算") $
$ "稀疏度" = K / E $

Mixtral 8×7B：$E=8, K=2$ → 稀疏度 25%，总 47B 参数、激活 13B。DeepSeek-V3：$E=256+1, K=8+1$ → 稀疏度 ~3.5%，671B/37B。

#note[
  Mixtral 名字里"8×7B"是营销术语——*不是* 8 份 7B 参数相加。每个专家共享 attention/embedding，专家只是 FFN 部分。真实参数约 47B，激活约 13B（含 attention）。
]

== 代价：三个新问题

天下没有免费的午餐。MoE 引入三个 dense 时代没有的问题：

*1. 路由决策要学*

Router 是可学习的，训练早期它可能给出"总是选专家 0"的解——其他专家收不到梯度，等价于浪费 $E-1$ 份参数。解决方案是负载均衡损失（第 6 章）或 DeepSeek 那种 auxiliary-loss-free 的 bias tuning。

*2. 负载天然不均*

即使 router 收敛良好，某个 batch 里*恰好* 大部分 token 都路由到 expert 0 是完全可能的。这意味着：

- 显存里 $E$ 个专家权重都得存
- 但计算时只有 1-2 个专家真在忙，其他 SM 空闲
- 大 batch 下"每个专家的 batch size"是 skewed 分布——GEMM 的 M 维忽大忽小

第 7 章会讲怎么用 grouped GEMM / block-sparse 应对。

*3. 分布式通信复杂*

Dense 训练里，一个 FFN 的通信量就是权重梯度 all-reduce。MoE 里如果把 experts 分布在多张卡上（expert parallel），*每一层都要做两次 all-to-all*——一次把 token 送到对的专家、一次把结果送回来。第 8 章是这本书最长的章节，就在讲这件事。

#warn[
  常见误解：既然专家稀疏，MoE 训推都会更便宜。*错。* MoE 只是 FLOP 便宜；显存开销（要装下所有专家）和通信开销（all-to-all）都比 dense 更高。所以 MoE 是"更大模型，同 FLOPs"，*不是*"同能力，更少资源"。
]

== 一张图看清 MoE 层的位置

MoE 层是 *Transformer block 内 FFN 子层的替换品*，其他部分 (attention、LN、residual) 不变：

```
   Transformer block (dense):              Transformer block (MoE):

   ┌─────────────────────────┐             ┌─────────────────────────┐
   │  MHSA + residual + LN   │             │  MHSA + residual + LN   │
   │           │             │             │           │             │
   │           ▼             │             │           ▼             │
   │      ┌─────────┐        │             │    ┌──────────────┐     │
   │      │   FFN   │        │             │    │ MoE (E,K)    │     │
   │      └─────────┘        │             │    │  = router +  │     │
   │           │             │             │    │    K experts │     │
   │           ▼             │             │    └──────────────┘     │
   │  residual + LN → out    │             │           │             │
   └─────────────────────────┘             │           ▼             │
                                            │  residual + LN → out    │
                                            └─────────────────────────┘
```

替换粒度可以是"每层都换"（Switch Transformer、Mixtral）或"每隔 2 层换一层"（GShard、DeepSeek-MoE 部分变种）。

== 里程碑速览

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, center, left),
    [*模型*], [*年份*], [*$E$*], [*$K$*], [*特色*],
    [Sparsely-Gated MoE], [2017], [$>1000$], [2], [开山，per-layer huge $E$],
    [GShard], [2020], [2048], [2], [翻译，capacity + drop],
    [Switch Transformer], [2021], [$>10^4$], [1], [$K=1$ 极简路由],
    [GLaM], [2021], [64], [2], [1.2T 参数],
    [Mixtral 8×7B], [2023], [8], [2], [开源 SOTA，SwiGLU expert],
    [DBRX], [2024], [16], [4], [细粒度专家],
    [DeepSeek-V2 / V3], [2024], [$160+$/$256+$], [6/8], [细粒度 + shared expert],
    [Qwen-1.5-MoE], [2024], [60+], [4], [上采样自 dense],
  ),
  caption: [MoE 里程碑。近年趋势：单层专家数从"上千"降到"几十~几百"（细粒度专家），$K$ 从 1 升到 6-8（表达力）。],
  kind: table,
)

== 本书 optimization ladder 概览

作为 metadata：本书讲的"MoE"优化 ladder 大致长这样（每个 rung 后面会详细讲）：

#ladder(
  ("v0: naive scatter/gather", "for-loop E 个专家 + `where` + `index_add_`", "教学；E ≤ 16"),
  ("v1: grouped GEMM", "permute 到 packed 排序，一次 kernel 跑完 E 个专家", "生产单机；2-4× 加速"),
  ("v2: fused router",         "Linear+softmax+topk+renorm 融成 1 个 kernel", "轻量收益，减一次 HBM 读写"),
  ("v3: block-sparse (Megablocks)", "无 padding、无 drop token", "vs v1 更好的 skew 容忍"),
  ("v4: EP + async all-to-all", "跨卡专家并行，通信/计算 overlap", "8+ 卡训练必需"),
  ("v5: fine-grained overlap (DSv3)", "dispatch/attention/mlp/combine 四阶段 pipeline", "1000+ 卡训练"),
)

后面每一章基本都在这条 ladder 上爬。第 5 章的教学代码对应 v0；第 7 章讲 v1-v3；第 8 章讲 v4-v5。

== 面试考点

#interview[
  *Q1*: MoE 相比 dense 的核心权衡是什么？

  A: *参数量*和*计算量*解耦。用 $E$ 倍参数换 $K/E$ 的 FLOPs（相对同参数量的 dense），但代价是：显存要装完整参数（$E times P_"expert"$）、多一层路由计算、跨卡时 all-to-all 通信开销。
]

#interview[
  *Q2*: "Mixtral 8×7B" 是不是 8 个独立的 7B 模型？

  A: 不是。只有 FFN 部分是 8 份 experts；attention、embedding、LN 都共享。实际总参数约 47B，激活约 13B。名字来源于"8 experts, each ~7B params in FFN"的营销修辞。
]

#interview[
  *Q3*: 稀疏 MoE 和 ensemble 有什么区别？

  A: 关键区别是*路由*。Ensemble 让每个成员看所有输入、取平均或投票；MoE 让 router *学会*把每个 token 送到相应专家，专家专注不同"技能"。而且 MoE 只激活 K 个专家，ensemble 全跑一遍。
]

#interview[
  *Q4*: 为什么现在的 MoE 都是"few experts, larger each"（Mixtral 8 expert）而不是 Shazeer 2017 的"many tiny experts"？

  A: 早期 many-tiny 在小 batch 下每个专家几乎无 token，GEMM 内存 bound + launch overhead 主导。少而大的专家（few large）在同参数量下 GEMM 密度更高。但 DeepSeek 又转回"many fine-grained" + shared expert 的混合方向——工程 trade-off 还在演化，没有唯一正确答案。
]
