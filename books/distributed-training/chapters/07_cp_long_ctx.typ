#import "../template.typ": *

= Context Parallel：Ring Attention, Ulysses 与长序列训练

Context Parallel (CP)，或"sequence parallel"（与 Megatron 那个 SP 是不同东西——命名混乱），把 attention 沿 sequence 维切分到多张卡。1M+ context 训练必需，也是 2023-2024 最火的方向。

*澄清命名*：
- *Megatron SP (Sequence Parallel)*：LN/Dropout 沿 seq 切，与 TP 组合，不动 attention（第 5 章）
- *Context Parallel (CP)*：attention 也沿 seq 切，需要跨卡通信 K/V（本章）
- *Ulysses / Ring Attention*：CP 的两种实现路径

== 为什么需要 CP：attention 的 quadratic 显存

Standard attention 显存 $prop B S^2 A$（存 attention matrix）。FlashAttention 把 matrix on-the-fly 算，显存降到 $O(B S H)$。但*compute*仍然 $O(S^2 H)$——64K → 1M 序列 compute 增加 256×。

*Activation*：一层 forward 存 Q, K, V, output = $4 B S H$。$B=1, S=10^6, H=8192, "bs"=2$: 单层 activation 64 GB。80 层 = 5 TB。单卡当然不行。

*方案*：把 $S$ 切成 $"CP"$ 份，每卡持 $S/"CP"$ 长的 Q/K/V。activation 降到 $4 B S H / "CP"$。但 attention 是"每 Q 都要看所有 K/V"——需要跨卡通信。

== Ulysses：All-to-All 沿 head 维交换

Jacobs et al. 2023 (DeepSpeed-Ulysses)。核心思路：

+ Input: 每卡 $Q_i, K_i, V_i in RR^(B, S/"CP", A, d_h)$（seq 切分）
+ *All-to-All*：沿 seq 维 gather + 沿 head 维 scatter。每卡拿到 $Q_i^"a2a", K_i^"a2a", V_i^"a2a" in RR^(B, S, A/"CP", d_h)$（head 切分）
+ *本地 attention*：$O_i = "attention"(Q_i^"a2a", K_i^"a2a", V_i^"a2a")$。每卡处理 $A/"CP"$ 个 head 的完整 seq。attention 计算完全本地
+ *All-to-All 逆*：从 head-shard 回 seq-shard，得 $O_i in RR^(B, S/"CP", A, d_h)$

#figure(
  align(center, a2a-diag(n: 4, cell: 0.42, title: "All-to-All (Ulysses): 每 rank 把自己 4 份中的第 j 份发给 rank j")),
  caption: [Ulysses 用一次 all-to-all 把"seq-shard、all-heads"重排成"full-seq、head-shard"，attention 完全本地，反向再一次 a2a 换回来。],
) <fig-a2a-ulysses>

*通信量*：每卡 2 次 all-to-all（forward）+ 2 次（backward），每次 $B S H / "CP"$ 数据。

$ "vol"_"ulysses" = 4 times B S H times "bs" / "CP" times ("CP" - 1)/"CP" $

对 $B=1, S=32K, H=8192, "CP"=8$: 每卡 per-layer per-step ≈ 4 × 512MB × 7/8 = 1.75 GB。

*限制*：
+ *$A gt.eq "CP"$*：head 数必须 ≥ CP，否则 head shard 分不下
+ Head 少的模型 (Llama 7B A=32) 只能 CP ≤ 32；GQA (KV heads 少) 更受限
+ All-to-all 通信量 $prop S H$，$S$ 大到 1M 时通信量单层几 GB

*优势*：attention compute 全本地——kernel 效率高，与 FlashAttention 完全兼容，无需改 kernel。

== Ring Attention：K/V 沿环流动

Liu et al. 2023。另一种切法：

+ Input: 每卡 $Q_i, K_i, V_i in RR^(B, S/"CP", A, d_h)$
+ *不做 all-to-all*，而是把 K/V 沿 ring 逐步*旋转*
+ 每一步：本卡 Q 与当前持有的 K/V 段做 attention，累加到 output
+ 同时 async send K/V 到下一 rank
+ $"CP"$ 步后每 Q 都看过了所有 K/V

*算法*：

```python
# 每卡拥有 Q_i, K_i, V_i (本地 shard)
K_cur, V_cur = K_i, V_i     # 起点是自己
O_acc, lse_acc = init()      # accumulator (Flash-style online softmax)

for step in range(CP):
    # 与当前 K, V 做局部 attention
    O_i, lse_i = flash_attention(Q_i, K_cur, V_cur, causal=causal_mask(i, step))
    O_acc, lse_acc = merge_flash(O_acc, lse_acc, O_i, lse_i)

    # async 送 K, V 到下一 rank; 从上一 rank recv 下一段
    K_next = dist.p2p_recv_and_send(K_cur, next_rank, prev_rank)
    V_next = dist.p2p_recv_and_send(V_cur, next_rank, prev_rank)
    K_cur, V_cur = K_next, V_next
```

*Flash-style merge*：每步局部 attention 输出 $(O^((s)), "lse"^((s)))$，用 online softmax 的合并规则累加。这是 Ring Attention 的关键 trick，让分步计算等价于一次完整 attention。

#figure(
  align(center, ring-attn-diagram(n: 4, r: 1.7,
    title: "Ring Attention: 4 卡各持 1/4 的 Q/K/V, K/V 沿 ring 旋转")),
  caption: [每 rank 用自己的 Q 和"当前持有的 K/V"做局部 attention，然后把 K/V async 送到下一 rank；4 步后每个 Q 都看过了所有 K/V。总通信量 $2 B S H · b$，与 CP 无关。],
) <fig-ring-rotate>

代码见 `src/distributed_training/07_ring_attention.py`——包含 causal mask、Flash-lse merge、单卡 SDPA 对拍。

*通信量*：每步 P2P send $B S H / "CP"$ (K + V 一起)。$"CP"$ 步 total per-GPU：

$ "vol"_"ring" = 2 times "CP" times (B S H / "CP") = 2 B S H "bs" $

*与 Ulysses 相比*：Ring 的每步 P2P 更小 (~50 MB range) 但*次数多* ($"CP"$ 次)；Ulysses 是 1-2 次大 collective。Ring 的通信量与 CP 无关（$2 B S H$ 恒定），Ulysses 与 CP 有关。

*限制*：
+ *与 head 数无关*——CP 可以任意大 (up to seq_len)
+ 但每步 attention kernel 的 $K$ 维小 (= $S/"CP"$)，效率低
+ P2P ring 的延迟累积——CP 大时通信步数多

*优势*：
+ CP 可以扩到 1024 (匹配序列长度)
+ 通信量 fixed
+ 与 FlashAttention 兼容 (通过 flash-attn 的 lse 合并接口)

=== Causal Mask 的处理

Autoregressive LM 里 attention 是 causal 的——position $i$ 的 Q 只看 position $j lt.eq i$ 的 K/V。Ring 里每 rank 拿到不同 shard 的 K/V，需要正确 mask。

*问题*：如果 rank 0 持 tokens [0, S/CP)，rank 1 持 [S/CP, 2S/CP)，做 attention 时 rank 1 的 K/V 对 rank 0 的 Q 是*未来*——应该跳过。

*naive 做法*：只在 K/V shard 的 index ≤ Q shard index 时做 attention，其他 step 跳过。但这样 rank 0 一直闲、rank last 一直忙——*load imbalance 2×*。

*Striped Attention* (Brandon et al. 2024)：把 seq 重新排列，让每 rank 处理"从头到尾均匀的一带"，负载均衡。

*Zigzag*：类似思路，rank $r$ 持有 pattern $[r, 2N-1-r]$——头尾对称，每 rank 都持有前半 + 后半。

torch-loong / vLLM / OpenDiT 都用 Zigzag 或 Striped。Megatron-LM `context_parallel_size` flag 用类似方案。

== USP: Unified Sequence Parallel (Hybrid)

Fang et al. 2024 (USP)。观察：Ulysses 和 Ring 各有优劣：
- Ulysses: head-limited, 通信量 $prop S H / "CP"$，1-2 大 collective
- Ring: head-free, 通信量 $prop S H$，$"CP"$ 小 P2P

*USP*：CP = $"CP"_"ulysses" times "CP"_"ring"$，先 Ulysses 沿 head 切一层，再 Ring 沿 seq 切一层。

*例*：$A=64, S=256K$，想 CP=32。
- 纯 Ulysses：$"CP" lt.eq A = 64$ OK，通信量 $32 B S H / 32 approx B S H$，短 seq 强
- 纯 Ring：$32$ 步 P2P，长 seq 时 kernel 效率低
- USP $"CP"_u = 4, "CP"_r = 8$：head 切 4 份 (还剩 16 head/rank)，seq 再切 8 份。综合最优

Long-context 训练里 USP 是当前 SoTA。Kimi/Moonshot 1M ctx、Gemini 1M ctx 都用类似 hybrid。

== FlashAttention v3 + CP

FA3 (Shah et al. 2024) 支持 `cp_group`，直接把 ring attention 集成到 kernel 里：

```python
from flash_attn.flash_attn_interface import flash_attn_func
out = flash_attn_func(q, k, v, causal=True,
                      cp_group=cp_process_group,
                      cp_ring=True)   # ring attention inside kernel
```

kernel 内部把 P2P 与 tile 计算 overlap，比 Python-level ring 快 20-30%。Hopper 用 async TMA 让通信基本 hidden。

== 通信量对比表

对 $B=1, S=64K, H=8192, "bs"=2$, CP=8：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, right, left),
    [*Method*], [*Comm / layer / step*], [*Head 限制*], [*Kernel efficiency*],
    [Ulysses],   [$4 B S H "bs" (1-1/"CP") = 3.5 "GB"$], [$A gt.eq "CP"$], [high (local attn)],
    [Ring],      [$2 B S H "bs" = 2 "GB"$], [无],          [medium (small-K)],
    [USP],       [between],                 [$A gt.eq "CP"_u$], [high],
    [FA3 fused CP], [same as ring but async], [无],       [very high],
  ),
  kind: table,
  caption: [长序列 CP 方案对比。Ring 通信量与 CP 无关但 kernel 效率打折；Ulysses 反之；USP hybrid 兼顾。],
)

== Long-context 训练的其他关键点

CP 只解决 attention。长 seq 还有其他显存来源：

+ *Position embedding cache*：RoPE 的 cos/sin cache = $S times d_h$，$S=1M$ 时 = 500 MB。用 cached 或 fused RoPE (TE)
+ *LN activation*：$B S H$ 沿 seq 切（Megatron SP）
+ *FFN activation*：$B S I$ 沿 seq 切（Megatron SP + activation checkpoint）
+ *KV cache in generation*：inference 时 KV cache = $2 L S H "bs"$ per sample，$S=1M$ 直接 500 GB/sample，需要 PagedAttention

*Recompute*：Megatron `--recompute-granularity full` 让 attention layer 全 recomp，只存 input，backward 重算——省 activation 但 compute 加 30%。CP + full recompute 是 1M ctx 训练常用组合。

== SP/CP 的全栈适配：attention 之外的所有 cross-token 模块

前面几节把 attention 讲透了，但*把 CP 接进一个真实训练栈，attention 只占改动量的一小半*。原因很简单：CP 沿 sequence 维切分，于是*任何跨 token 计算或跨 token 归约的模块都被切断了*，而 attention 只是其中最显眼的一个。

这类 bug 有一个共同的、极其讨厌的性质：

#warn[
  *它们不报错*。进程不 crash，NCCL 不超时，loss 曲线看上去完全正常——只是下降得慢一点，或者某个 metric 一直不收敛。你把 CP 从 1 调到 8，吞吐涨了 6 倍，验证 loss 差了 3%，很容易归因成"长序列本来就难训"。

  Ring/Ulysses 写错了会直接数值爆炸，你当天就能发现；这一节的每一个坑都能安静地活到模型交付。
]

本节的代码在 `src/distributed_training/19_cp_loss_and_metrics.py`（跨 token 归约）与 `20_cp_dataloader_halo.py`（数据侧与局部算子），全部用单卡全序列结果做对拍，下面引用的数字都来自这两个 demo 的实际输出（CP=4）。

=== 先看一张判定表

接 CP 时先把模型过一遍，对每个模块问同一个问题：*它的输出依赖它自己那一段以外的 token 吗？*

#figure(
  table(
    columns: (1.75fr, 0.5fr, 1.8fr, 1.55fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, center, left, left),
    [*模块*], [*跨\ token*], [*需要的处理*], [*写错的征状*],

    [RMSNorm / LayerNorm], [否], [无。沿 hidden 归一化，逐 token 独立], [—],
    [FFN / SwiGLU / 逐元素], [否], [无], [—],
    [Attention], [是], [Ring / Ulysses / USP（前几节）], [数值直接错，立刻发现],
    [RoPE / 位置编码], [是#super[\*]], [必须用*全局* position_ids], [长序列外推变差],
    [Causal / doc mask], [是], [从全局 position 与 doc id 重建], [跨文档泄漏],
    [Cross-entropy loss], [是], [全局 token 数作分母], [loss 偏移、等效 LR 变化],
    [梯度尺度], [是], [明确 CP 是副本维还是分片维], [梯度差 CP 倍],
    [Label shift / MTP], [是], [切分*前*做位移，或 halo], [边界 token 配错],
    [Sliding window / conv], [是], [halo 交换 $w-1$ 个 token], [接缝处输出错],
    [Mamba / SSM scan], [是], [传递递归 state，或 chunked scan], [接缝处输出错],
    [MoE router aux loss], [是], [全局 expert 直方图], [负载均衡永不收敛],
    [Pooling / reward 头], [是], [定位 owner rank 后归约], [分数取自序列中间],
    [Metrics (ppl / acc)], [是], [按 token 数加权归约], [监控读数系统性偏移],
    [Grad norm / clipping], [是], [只在*分片*维求和], [范数虚高 $sqrt("CP")$ 倍],
    [Dropout RNG], [是#super[\*]], [mask 的切分方式要与张量一致], [正确性与单卡不可对拍],
  ),
  kind: table,
  caption: [SP/CP 适配判定表。#super[\*] RoPE 与 Dropout 本身逐 token 独立，但都依赖 token 的*全局身份*（位置、RNG offset），所以同样要改。前两行是提醒：不要给它们加通信。],
) <tab-cp-checklist>

#insight[
  表里最容易被忽略的是*前两行*。面试里常见的过度设计是"CP 下 LayerNorm 要不要 all-reduce"——不要。LN/RMSNorm 沿 hidden 维求均值方差，每个 token 自己算自己的，与序列怎么切完全无关。这也正是 Megatron SP 能把 LN 放在 seq-shard 上零通信执行的原因（§5）。

  会真正需要通信的是 BatchNorm 这类沿 batch/序列维统计的算子——而 LLM 里基本不用它，这本身就是原因之一。
]

=== Loss：分母必须是全局有效 token 数

这是最高频的一个。CP 下每卡只有 $S\/"CP"$ 个 token，而 cross-entropy 的归约（求平均）是*跨 token* 的。

关键在于：真实 batch 里*各卡的有效 token 数差别极大*。SFT 的 loss mask 把 prompt 全部屏蔽，右侧还有 padding。demo 里 $S=64$、CP=4，各卡有效 token 数是 `[0, 12, 16, 2]`——rank 0 一个都没有。同时各卡的平均 loss 也不同（序列后段上下文更多、更好预测），实测 `[--, 3.91, 3.31, 1.01]`。

四种写法，只有最后一种对：

#figure(
  table(
    columns: (2.1fr, 1fr, 2.5fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right, left),
    [*写法*], [*实测值*], [*为什么*],
    [参考值（单卡全序列）], [3.3984], [—],
    [本地 mean 后跨 CP 求平均], [`NaN`], [rank 0 算了 $0\/0$，一次 all-reduce 全场变 NaN],
    [同上，但跳过空 shard], [2.7457], [$0.81 times$。把每卡权重强行拉平成 $1\/"CP"$],
    [分母用本地总位置数], [1.5930], [$0.47 times$。`numel()` 而非 `mask.sum()`],
    [*先归约分子分母，再相除*], [*3.3984*], [唯一正确],
  ),
  kind: table,
  caption: [CP 下 cross-entropy 的四种归约。第三行是最危险的：数值有限、曲线平滑、完全错误。],
) <tab-cp-loss-denom>

正确写法就一句话——*把 sum 和 count 分别 all-reduce，最后才做除法*：

```python
# 每卡本地：只求和，不求平均
local_sum = (per_token_loss * loss_mask).sum()
local_cnt = loss_mask.sum()

# 跨 CP 组归约分子与分母（count 不需要梯度）
global_cnt = local_cnt.clone()
dist.all_reduce(global_cnt, group=cp_group)

loss = local_sum / global_cnt      # 本卡对全局 loss 的贡献
```

#note[
  第三行"跳过空 shard"值得单独说，因为它是唯一能长期存活的版本。它把每张卡的权重都变成 $1\/"CP"$，与该卡实际持有多少真 token 无关。后果不是一个常数偏移，而是*样本权重被重新分配*：短答案样本被系统性放大，长答案被压小。你会看到模型偏好短输出，然后去调 length penalty——治的是症状。

  这和 §15 里 agentic RL 的 `mask.numel()` vs `mask.sum()` 是同一个 bug 的两种形态：一个沿 turn 维，一个沿 CP 维。
]

=== 梯度尺度：CP 组到底是副本维还是分片维

上面的 loss 写对了，还有第二个坑，而且更隐蔽——*它只错在梯度里，loss 打印出来的数完全正确*。

每卡算的是 `local_sum / global_cnt`，即"本卡对全局 loss 的贡献"。这些贡献需要*求和*才等于全局 loss 的梯度。但 CP 组内参数是*副本*，而 Megatron 把 CP 维折进 data-parallel 的梯度归约里——DP 归约是*求平均*。平均一堆部分和，就丢了一个 $"CP"$ 因子：

#figure(
  table(
    columns: (2.4fr, 1.2fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right),
    [*CP 维的梯度归约方式*], [$|g| \/ |g_"ref"|$],
    [AVG（DP 默认）], [0.2500 $= 1\/"CP"$],
    [SUM], [1.0000],
  ),
  kind: table,
  caption: [CP=4 实测。两种写法的 loss 打印值一模一样，只有梯度差 4 倍——看曲线永远看不出来。],
) <tab-cp-grad-scale>

征状是"感觉 LR 小了 CP 倍"：loss 下降偏慢、grad_norm 偏小、把 LR 乘 CP 倍就好了——于是很多人就真的把 LR 乘上去当作调参结论，而没意识到是归约约定错了。

两种修法都对，选一种并写进单测：

+ 保持 AVG 归约，把 loss 乘 $"CP"$
+ CP 维用 SUM 归约，只在真正的 DP 维上做 AVG

#warn[
  框架之间的约定不一致，这是升级 Megatron / 换 FSDP 时最容易踩的回归。DeepSpeed-Ulysses、Megatron-CP、torchtitan 对"谁负责这个 $"CP"$ 因子"的默认值都不同。接 CP 时的第一个测试就应该是：*固定数据、CP=1 与 CP=N 跑一步，比对参数更新量*。
]

=== 位置与 mask：必须作为 batch 的字段跟着 token 走

==== 为什么先要 zigzag 置换

连续切分（rank $r$ 拿 $[r L, (r+1)L)$）在 causal mask 下负载严重不均：rank 0 的 query 只能看极少的 key，最后一张卡几乎要看全序列。demo 实测每卡的未被 mask 的 $(q,k)$ 对数：

$ "contiguous": [36, 100, 164, 228] quad "zigzag": [132, 132, 132, 132] $

collective 走最慢的那张卡，所以代价是 $max\/"mean"$ 而不是更夸张的 $max\/min$——实测 $1.73 times$，CP 越大越趋近 $2 times$。

*Zigzag* 把序列切成 $2"CP"$ 个 chunk，rank $r$ 取 chunk $r$ 与 chunk $2"CP"-1-r$，一前一后配对。每卡的 $(q,k)$ 对数变成严格相等（上面第二组数字），这也是 Megatron、vLLM、torchtitan 的默认做法。

#figure(
  table(
    columns: (auto, ) + (1fr,) * 8,
    stroke: 0.5pt + gray,
    inset: 4.5pt,
    align: center,
    [], [*c0*], [*c1*], [*c2*], [*c3*], [*c4*], [*c5*], [*c6*], [*c7*],
    [*rank 0*], table.cell(fill: rgb("#dbeafe"))[■], [], [], [], [], [], [], table.cell(fill: rgb("#dbeafe"))[■],
    [*rank 1*], [], table.cell(fill: rgb("#dcfce7"))[■], [], [], [], [], table.cell(fill: rgb("#dcfce7"))[■], [],
    [*rank 2*], [], [], table.cell(fill: rgb("#fef3c7"))[■], [], [], table.cell(fill: rgb("#fef3c7"))[■], [], [],
    [*rank 3*], [], [], [], table.cell(fill: rgb("#fce7f3"))[■], table.cell(fill: rgb("#fce7f3"))[■], [], [], [],
  ),
  kind: table,
  caption: [CP=4 的 zigzag 布局：序列切成 8 个 chunk，rank $r$ 取 chunk $r$ 与 $7-r$。每张卡都同时持有序列的前段和后段，causal 负载因此均衡。代价是*每卡的 shard 不再是一段连续区间*——这是后面所有麻烦的根源。],
) <tab-zigzag-layout>

==== 三样东西必须跟着 token 一起走

Zigzag 之后，rank 0 持有的全局位置是 `[0,1,2,3, 28,29,30,31]`。任何从 `arange(local_len)` 重新推导出来的东西都是错的：

+ *全局 position_ids*。RoPE 是相对位置编码，但它靠*绝对位置作差*实现——每卡用本地 $0..L-1$ 会让所有相对距离都错。这是最经典的静默 bug：短序列时误差被 attention softmax 吸收掉一部分，长序列外推能力直接崩。
+ *document ids*。packing 场景下（§12）一条序列里塞了多个文档，attention 不能跨文档。切分后每卡需要自己那段的 doc id 才能重建 mask，varlen kernel 的 `cu_seqlens` 也要按本地 shard 重算。
+ *置换本身*。输出要还原回原始顺序才能和 label 对齐、才能给下游模块用。

mask 不能再用"下三角"或任何基于本地 index 的形状，必须从全局量重建：

```python
# q 在本卡（长度 L），k 覆盖全序列（Ring/Ulysses 拿到的那份）
causal   = pos_q[:, None] >= pos_k[None, :]      # 全局 position 比较
same_doc = doc_q[:, None] == doc_k[None, :]      # packing 的文档隔离
mask = causal & same_doc
```

demo 里把"本地 position + 本地 index mask"和上面的正确写法都跑了一遍，对拍单卡结果：

$ "wrong": max|Delta| = 3.557 quad "correct": max|Delta| = 2.4 dot 10^(-7) $

#insight[
  工程上的结论很直接：*把 position_ids、doc_ids、以及置换索引都做成 batch 的显式字段*，由 data loader 产出、和 token 一起切分，模型里任何地方都不要用 `arange` 现场生成。

  这条规则听起来平淡，但它是 CP 代码能不能长期维护的分水岭。一旦允许某个模块自己算位置，之后每加一个 CP 变体（zigzag、striped、USP 的两级切分）都要去找一遍所有 `arange`。
]

=== Data loader：CP 组内必须拿到同一条样本

一个 CP 组*合起来*才持有一条序列，所以组内每张卡必须拿到*同一个*样本，然后沿 seq 切分。这意味着 sampler 的索引必须用 `dp_rank`，不能用 `global_rank`：

```python
cp_rank, dp_rank = rank % CP, rank // CP     # global = dp_rank * CP + cp_rank

sample_id = step * n_dp + global_rank        # 错：组内两张卡拿到不同样本
sample_id = step * n_dp + dp_rank            # 对
```

demo 里 world=4 = DP2 $times$ CP2，错的写法让一个 CP 组看到 `[28, 29]` 两个不同样本，对的写法看到 `[14, 14]`。

后果有两层，都不报错：

+ rank 0 持有文档 A 的前半，rank 1 持有文档 B 的后半，attention 会*把两者当成一条序列*来算
+ 实际 global batch size 变成配置值的 $"CP"$ 倍，等效 LR 与 token 预算全部对不上

*shuffle 的随机种子有完全相同的要求*——`base + global_rank` 会让组内两卡打乱出不同顺序，必须用 `base + dp_rank`。同一条规则也适用于 TP 组和 PP stage：*只有 DP 维才允许数据不同*。

#note[
  这个 bug 在小规模冒烟测试里几乎测不出来：CP=2、跑 100 步，loss 照样下降（模型在学两个半截文档的拼接，仍然是有信号的语言建模任务）。可靠的检查是一行断言——把 `sample_id` 在 CP 组内 all-gather，要求全相等。这类断言应该常驻在 data loader 里，代价可以忽略。
]

=== 需要 halo 交换的模块

有一类算子只依赖*邻近*的几个 token，它们不需要全局通信，但需要从邻居拿一小段"边缘数据"（halo）。

==== Label shift：一个 token 的位移就跨界了

Next-token prediction 里 `labels = tokens[1:]`。本卡最后一个 token 的 label 在下一卡上。在 shard 内部 `roll(-1)` 会静默配错：

#figure(
  table(
    columns: (0.9fr, 1.1fr, 2.8fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right, left),
    [*布局*], [*配错的 pair*], [*说明*],
    [contiguous], [3 / 32 $= "CP"-1$], [每卡末尾一个，最后一卡那个本来无 label],
    [zigzag], [6 / 32 $= 2"CP"-2$], [每卡两段、两个接缝；rank $"CP"-1$ 的两个 chunk 恰好相邻，它的内部接缝碰巧对了],
  ),
  kind: table,
  caption: [CP=4、$S=32$ 实测。zigzag 下错得更多，而且"少错一个"纯属 chunk 配对的巧合——这类偶然正确性正是让人误判问题范围的东西。],
) <tab-label-shift>

正确做法几乎不花钱：*在切分之前对整条序列做一次位移*，然后照常切。不需要任何通信。

```python
labels = torch.cat([tokens[1:], tokens[-1:]])   # 切分前做，一次搞定
labels_local = labels[perm[rank * L : (rank + 1) * L]]
```

MTP / speculative decoding 的头要往后看 $k$ 个 token，同一个技巧一次处理完所有 $k$。反过来说，如果你只能在切分后处理，就得做宽度 $k$ 的 halo——而 zigzag 下"下一个位置"可能在任意一张卡上，退化成一次不规则 gather。

==== Sliding window / conv / SSM

这些算子的 halo 宽度由感受野决定，且*必须*通信：

- *Causal conv1d*（kernel $K$）：需要前 $K-1$ 个 token。demo 里 $K=4$、CP=4，零填充版本错了 $9 = (K-1)("CP"-1)$ 个位置，halo 版本与单卡完全一致
- *Sliding-window attention*（窗口 $w$）：halo $= w-1$
- *Mamba / SSM 的递归扫描*：需要的不是 token 而是前一卡的*递归 state*，这会把各卡*串行化*，除非用 chunked scan 重写
- 多模态里 ViT / audio 前端的 depthwise conv 同理（§13）

#warn[
  注意误差的形状：它被限制在每个接缝的 $K-1$ 行里，*占比随 shard 变长而下降*。$S=32$、CP=4 时错 28%，$S=128"K"$、CP=8 时错 $0.002%$——loss 上完全看不见，但模型在每个 shard 边界都学到了一小段错误的卷积响应。

  这也解释了为什么*带局部算子的模型通常不用 zigzag*：zigzag 下一个 rank 有两段不连续区间，要从两个不同的 rank 各拿一次 halo。连续切分的负载不均是可以量化、可以接受的 $1.7 times$，而不规则 halo 是工程复杂度。这是一个真实的设计取舍，值得在面试里主动讲出来。
]

=== MoE router：aux loss 需要全局 expert 直方图

Load-balancing loss 形如 $"aux" = E sum_e f_e P_e$，其中 $f_e$ 是路由到 expert $e$ 的 token *比例*、$P_e$ 是 router 对 $e$ 的*平均*概率。两个都是"对 token 求平均"，所以 CP 下两个都要全局归约。

只统计本卡会得到一个荒谬的结论。demo 构造了一个*全局完美均衡*的 router（rank $r$ 的所有 token 都去 expert $r$，于是每个 expert 恰好拿到 $1\/E$）：

#figure(
  table(
    columns: (2.2fr, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right, right),
    [*统计范围*], [*aux loss*], [*$|nabla|$*],
    [参考值（全序列）], [1.0000], [0.00],
    [只统计本卡], [3.4802], [0.131],
  ),
  kind: table,
  caption: [CP=4、$E=4$ 实测。全局直方图下 $f_e = 1\/E$ 均匀，aux loss 退化为常数 1（理论最小值），梯度恰好为零——router 已经均衡，无需修正。只统计本卡则给出 $3.48 times$ 的惩罚和一个不该存在的梯度。],
) <tab-moe-aux-cp>

方向更糟：本卡视角看到的是"我的 token 全去了一个 expert"，梯度在*自己那个 expert 的 logit 上是正的*（实测 $+0.0283$），也就是*把每张卡都往远离全局最优的方向推*。aux loss 在反对它本该促成的事情。

征状是 load-balance metric 反复震荡、永不收敛，而 expert 利用率看起来又没那么差。同一处理适用于 z-loss、以及任何从 per-rank token 数推出来的 capacity / drop rate（§8）。

=== 归约到单点的头：pooling、reward model、GAE

有些头把整条序列*归约成一个向量或一个标量*，CP 下这个"点"只落在某一张卡上。

Reward model 是最典型的：分数取自最后一个有效 token 的 hidden state。demo 里最后一个有效 token 的全局位置是 49，只在 rank 3 上。各卡各自取"自己 shard 的最后一个有效 token"的话：

$ max|Delta| "per rank" = ["rank 0: 空 shard，索引越界", 3.669, 3.610, 0.000] $

只有 owner（rank 3）是对的，rank 0 因为整段都是 padding 直接崩在空索引上。

正确做法用 one-hot 选择 + 一次可微 all-reduce，非 owner 贡献严格为零：

```python
sel = torch.zeros(L, device=h.device)
if rank == owner:
    sel[global_last - rank * L] = 1.0
h_last = all_reduce_sum(sel @ h_local, cp_group)   # forward 求和，backward 恒等

# mean pooling 就是前面 loss 那套分子/分母分别归约
h_mean = all_reduce_sum((h_local * mask_l[:, None]).sum(0)) / global_count
```

这里的 `all_reduce_sum` 必须是*可微*版本（Megatron 里的 `f`/`g` 算子）：forward 跨组求和、backward 恒等。裸的 `dist.all_reduce` 是 in-place 的，autograd 完全不知道其他卡的存在。

同类模块还有：embedding 模型的 mean pooling、分类头的 `[CLS]`、以及 RL 里的 *GAE*——广义优势估计是沿序列的反向递推，CP 下要么把 reward/value 序列聚合到一张卡上算，要么做跨卡串行扫描（§15）。

=== Metrics：按 token 数加权，且只 exp 一次

监控指标和 loss 是同一个问题，但更容易被放过，因为"监控偏一点无所谓"——直到你用它做早停或选 checkpoint。

#figure(
  table(
    columns: (1.6fr, 1fr, 1.3fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right, right, right),
    [*指标*], [*参考*], [*每卡平均后再平均*], [*加权归约*],
    [perplexity], [25.027], [21.647], [25.027],
    [token accuracy], [0.667], [0.854], [0.667],
  ),
  kind: table,
  caption: [CP=4 实测。accuracy 偏高 28%，因为空 shard 与短 shard 被赋予了同样的权重。],
) <tab-cp-metrics>

perplexity 还多一层错误：$exp$ 是凸函数，*先在每卡 exp 再平均*会因 Jensen 不等式产生偏差。必须*先归约 log-loss，最后 exp 一次*：

$ "ppl" = exp((sum_r sum_i m_(r i) ell_(r i)) / (sum_r sum_i m_(r i))) $

按数据域、按任务分别记的 loss 全部同理——每个分桶都要自己的 (sum, count) 对。

=== Grad norm：只在分片维求和

梯度裁剪需要全局 $||g||$。CP 组内参数是*副本*，梯度归约之后每张卡持有的已经是完整梯度，*不能再跨 CP 求和*。但"把 $|g|^2$ 在全 world 上 all-reduce"是分片参数的标准写法，照搬过来就会：

#figure(
  table(
    columns: (2.4fr, 1fr, 1.4fr),
    stroke: 0.5pt + gray,
    inset: 5.5pt,
    align: (left, right, right),
    [*写法*], [$||g||$], [*clip 系数* ($max=1.0$)],
    [正确（不跨 CP 求和）], [0.8465], [1.000（不裁剪）],
    [错误（跨 CP 求和）], [1.6930], [0.591],
  ),
  kind: table,
  caption: [CP=4 实测：范数虚高 $sqrt("CP") = 2 times$。本该完全不触发裁剪的一步，被把更新压到了 59%。],
) <tab-cp-gradnorm>

这是一次*静默的 LR 削减*，而且 grad_norm 监控帮不了你——记下来的就是那个虚高的值，看上去一切正常。

一条规则涵盖所有并行维：

#formula[
  $|g|^2$ *只在参数被分片的组上求和*：ZeRO/FSDP shard、TP 的 column/row shard、EP 的 expert 维。

  *绝不在参数是副本的组上求和*：CP、DP replica、TP 下被复制的 LN / bias。
]

Megatron 用 `param.tensor_model_parallel` 和 `param.shared` 这两个标记来区分，自己改框架时最容易漏的就是新加的参数忘了打标记（§F）。

=== Dropout RNG：mask 的切分要与张量一致

Dropout 逐元素独立，看起来不跨 token，但它依赖 RNG 状态，而"该不该同步 RNG"取决于*它作用的张量是怎么切的*：

- 张量沿 seq 切（Megatron SP 的 LN/Dropout 区、CP 的所有 activation）：各卡是*不同的 token*，mask 就*应该*不同——但必须由全局 offset 可复现地生成，否则你无法与单卡对拍
- 张量在 TP 组内是*复制*的（例如 attention 之前的 input）：mask *必须逐位相同*，否则各 TP 分支算的其实不是同一个前向

Megatron 为此维护两套 RNG tracker（`get_cuda_rng_tracker`）。这也是"CP=1 与 CP=N 对拍"时必须先关掉 dropout 的原因——否则你在对比两个不同的随机前向。

=== 验证方法论：逐模块对拍

上面每一条都是"不报错的错"，所以*唯一可靠的手段是与单卡参考实现做位级对拍*，而且要逐模块做，不能只看最终 loss。

一个可以直接落地的流程：

+ *固定一切随机性*：关 dropout、固定数据、固定初始化种子
+ *从内往外逐层对拍*，出错立刻定位。每一层能抓到的东西不同，跳过任何一层都会漏掉一类 bug：

  #figure(
    table(
      columns: (1.7fr, 2.3fr),
      stroke: 0.5pt + gray,
      inset: 5.5pt,
      align: (left, left),
      [*对拍对象*], [*只有这一层能抓到*],
      [attention 输出 vs 单卡 SDPA], [Ring/Ulysses 的 lse 合并、causal mask 分块],
      [加上 position + mask], [本地 arange 当位置、mask 没重建、逆置换写错],
      [完整 forward 的 hidden states], [halo 缺失（conv/window）、doc 隔离失效],
      [loss 标量], [分母用了本地 count 或 numel],
      [*每个参数的梯度*], [*梯度尺度的 $"CP"$ 因子——loss 完全正确*],
      [一步更新后的参数], [grad-norm 虚高、clipping 提前触发],
    ),
    kind: table,
    caption: [CP 实现的逐层对拍。倒数第二行是最容易被跳过、也最容易出问题的一层。],
  ) <tab-cp-verify>

+ *断言常驻*：`sample_id` 在 CP 组内相等、`loss_mask.sum()` 全局大于零、position_ids 单调且覆盖完整区间。这些检查每步的开销可以忽略，但能挡住绝大多数回归
+ *CP 扫描*：同一份数据跑 CP $in {1, 2, 4, 8}$，要求 loss 与参数更新量在数值误差内一致。这是最强的一个测试，`19_*.py` 与 `20_*.py` 就是按这个思路写的

#insight[
  第 5 步（梯度对拍）是分水岭。只对拍 loss 的团队一定会漏掉梯度尺度那个 $"CP"$ 因子，因为 loss 完全正确。面试里如果被问"你怎么保证 CP 实现是对的"，回答"和单卡比 loss 一致"是不够的，说到"比对参数更新量"才说明真的踩过坑。
]

=== 面试考点

#interview[
  *Q1*: 开了 CP 之后，除了 attention，还有哪些模块需要改？

  按"是否跨 token"分类回答，比背清单更能体现理解：*跨 token 归约*的（loss 分母、MoE aux loss 的 expert 直方图、pooling 头、metrics、grad norm）要把 (sum, count) 分别归约再相除；*依赖全局 token 身份*的（RoPE position_ids、doc mask、dropout RNG offset）要让这些量随 token 一起切分而不是本地重算；*依赖邻近 token*的（label shift、MTP、sliding window、conv、SSM）要做 halo 交换。反过来，逐 token 独立的模块（RMSNorm/LayerNorm、FFN、逐元素算子）*不需要任何通信*——能主动说出这一点，说明你不是在乱加 all-reduce。

  *Q2*: CP 下 loss 怎么算？为什么不能各卡算完平均一下？

  因为各卡的有效 token 数不同（prompt masking + padding，极端情况下某卡为 0，直接 $0\/0$ 出 NaN）。必须本地只求和，把分子 $sum m ell$ 和分母 $sum m$ 分别 all-reduce，最后相除。追问"跳过空卡再平均行不行"——不行，那等于把每卡权重拉平成 $1\/"CP"$，实际效果是按样本长度重新加权，模型会偏好短输出。

  *Q3*: loss 算对了，为什么梯度还可能差 CP 倍？

  每卡产出的是"对全局 loss 的贡献"（分子是本地和、分母是全局数），这些贡献需要*求和*。但 CP 维常被折进 DP 的梯度归约，而 DP 是*求平均*，于是丢掉一个 $"CP"$ 因子。修法：要么 loss 乘 $"CP"$，要么 CP 维改成 SUM 归约。关键点是*loss 打印值完全正确*，只有梯度错，所以必须靠对拍参数更新量来发现，看曲线只会误判成"LR 需要调大"。

  *Q4*: 为什么要 zigzag？它带来了什么新问题？

  连续切分在 causal mask 下负载不均，代价是 $max\/"mean" arrow 2 times$。zigzag 让 rank $r$ 取 chunk $r$ 与 $2"CP"-1-r$，每卡的 $(q,k)$ 对数严格相等。新问题是*shard 不再连续*：position_ids 和 doc_ids 必须按同样的置换分发、mask 要从全局 position 重建而不能用下三角、输出要逆置换、而 halo 类算子会退化成从两个不同 rank 各取一次——所以带 conv/sliding-window/SSM 的模型往往宁愿承受连续切分的 $1.7 times$ 不均。

  *Q5*: CP 组内各卡应该拿到相同还是不同的数据？怎么验证？

  相同——一个 CP 组合起来才是一条序列。sampler 和 shuffle 种子都要用 `dp_rank` 索引，不能用 `global_rank`。写错了不会报错：attention 会把两个半截文档当成一条序列，同时 global batch size 悄悄变成配置值的 CP 倍。验证只需一行——把 `sample_id` 在 CP 组内 all-gather 后断言全相等，建议常驻。

  *Q6*: 为什么 CP 下 grad norm 容易算大 $sqrt("CP")$ 倍？

  因为把分片参数的写法（跨组 all-reduce $|g|^2$）套到了副本参数上。CP 组内参数是副本，梯度归约后每卡已持有完整梯度，不该再求和。规则是：*只在参数被分片的组上求和*（ZeRO/FSDP、TP shard、EP），*不在副本组上求和*（CP、DP replica、TP 复制的 LN）。范数虚高会让裁剪提前触发，等于静默削减 LR，而且 grad_norm 日志记的正是那个虚高值，查不出来。
]

== 完整 CP 训练配置样例

*Megatron-LM 训 Llama 8B, S=128K, 8 卡*：

```bash
torchrun --nproc-per-node=8 pretrain_gpt.py \
  --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
  --seq-length 131072 --max-position-embeddings 131072 \
  \
  --tensor-model-parallel-size 1 \
  --context-parallel-size 8 \                    # <-- CP=8
  --sequence-parallel \                          # Megatron SP for LN
  --use-distributed-optimizer \
  \
  --micro-batch-size 1 --global-batch-size 8 \
  --recompute-granularity full \                 # 长 seq 必须
  --recompute-num-layers 1 \
  --distributed-timeout-minutes 60 \             # 长 seq 慢
  \
  --bf16 --use-flash-attn --swiglu \
  --normalization RMSNorm --position-embedding-type rope \
  --rotary-base 500000                          # 长 ctx 一般加大 base
```

带 CP + FlashAttention v3 + selective recompute，H100 8 卡跑 Llama-8B S=128K，MFU ~35%。

== 面试考点

#interview[
  *Q1*: Ulysses 与 Ring Attention 的核心区别是什么？

  A: Ulysses 用 all-to-all 沿 head 维交换，一次通信后每卡拿到完整 seq 的部分 head——attention 本地做。Ring 用 P2P 沿 ring 逐步旋转 K/V——每步只做 local partial attention，累加 Flash lse。Ulysses 简单快但 CP ≤ head 数；Ring 无 head 限制但 kernel 效率低。
]

#interview[
  *Q2*: Ring Attention 里怎么处理 causal mask？

  A: naive 做法是 rank $r$ 只在收到 rank $r' lt.eq r$ 的 K/V 时算 attention，其他 skip。但这导致 rank 0 一直闲、rank last 一直忙，负载不均 2×。改进用 Striped Attention 或 Zigzag：把 seq permutation 让每 rank 都持"前半+后半"pattern，负载均衡。Megatron/vLLM 生产实现都用 Zigzag。
]

#interview[
  *Q3*: 训 1M context，你会怎么组合并行？

  A: 假设 512 卡 H100。举例：TP=8 × CP=16 (USP hybrid = Ulysses 4 × Ring 4) × PP=2 × DP=2。CP 主承担 seq 切；TP 切 attention head；FSDP2 切 weight。加 full activation recompute + FA3 with `cp_ring`。RoPE base scaled。
]

#interview[
  *Q4*: 为什么 Ulysses 的通信量与 CP 有关但 Ring 与 CP 无关？

  A: Ulysses 一次 all-to-all volume = $V(W-1)/W approx V$（$V = B S H$），与 CP 弱相关。Ring 每步 P2P $V/"CP"$，共 $"CP"$ 步 → total $V$，与 CP 完全无关。所以 CP 大时 Ring 通信量优势明显（尤其单步 kernel 效率恢复后）。
]

#interview[
  *Q5*: 一个 GQA num_kv_heads=8 的模型，能用 CP=16 的 Ulysses 吗？

  A: 不能——KV head 只有 8, 无法切 16 份。要么 (a) CP=8 (Ulysses)；(b) 用 Ring；(c) 用 USP 分解：Ulysses=8 × Ring=2 = CP 16。生产常用 (c)。
]

#interview[
  *Q6*: CP 之后的 output projection 需要通信吗？

  A: 不需要额外通信。Attention output 是 $(B, S/"CP", A, d_h)$（seq 切分），output projection $W_O$ 是 row-parallel 或全 replicate。如果 $W_O$ replicate，直接本地 matmul 得 $(B, S/"CP", H)$——顺利传给 FFN。如果与 TP 组合，output proj 是 row-parallel，需要 TP AllReduce（不是 CP）。
]

#interview[
  *Q7*: FlashAttention 与 CP 怎么集成？

  A: FA 内部的 online softmax 已经支持"分块累加"（LSE 合并）。Ring Attention 只需在每步 P2P 之后调 FA + 合并 LSE 到 accumulator。FA3 直接提供 `cp_ring` kernel，把 P2P 用 CUDA async 与 tile compute overlap，比 Python-level ring 快。
]

#interview[
  *Q8*: 长序列训练里 activation 显存的主要来源？CP 之外还能省什么？

  A: 层级 $B S H$（LN, Dropout, residual）＋ $B S I$（FFN intermediate）＋ Q/K/V/O activation。CP 切 seq 只解决 attention 内的部分。剩下的靠：(1) Megatron SP 沿 seq 切 LN/Dropout；(2) FSDP 沿 DP 切 weight；(3) `--recompute-granularity full` 只存 layer input；(4) activation offload 到 CPU（DeepSpeed / TE）。DeepSeek-V3 Table 3 里 activation 占显存 66%，是 long-ctx 的首要优化对象。
]
