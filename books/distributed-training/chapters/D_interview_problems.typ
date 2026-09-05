#import "../template.typ": *

= 附录 D：130 题面试题库

按主题分类。每题给参考答案 + 章节引用。用于面试前一天扫一遍。

Q1–Q100 覆盖并行策略、精度、overlap、多模态与经典 RLHF；Q101–Q120 是 RLVR 与 Agentic RL 的专项，2025 年之后面 RL infra 岗基本必问；Q121–Q130 专门考 SP/CP 开启后 attention 之外那些模块的适配——这一组问题最能区分"用过 CP"和"接过 CP"。

== 基础 / 通信 (Q1-Q15)

#interview[
  *Q1*: 一句话解释 AllReduce = ReduceScatter + AllGather 的通信量恒等式。

  A: AllReduce per-GPU 通信量 = $2 V (W-1)/W$；分解成 ReduceScatter ($V(W-1)/W$) + AllGather ($V(W-1)/W$) 恰好 = AllReduce。这是 NCCL Ring 算法的实现基础，也是 ZeRO/FSDP 分片通信量分析的公式基础。§1
]

#interview[
  *Q2*: 一次 Ring AllReduce 与 world size 有什么关系？

  A: bandwidth 项恒为 $2V$（与 W 无关），latency 项 $2(W-1)alpha$ 随 W 线性增。所以 payload 大时 Ring 与 W 无关；小 payload + 大 W 时切 Tree。§1
]

#interview[
  *Q3*: NVLink 与 IB 带宽差多少？这对并行策略选择意味着什么？

  A: H100 NVLink 4 = 900 GB/s (bidi)，IB NDR = 50 GB/s (uni)。差约 10-18×。所有高频率通信 (TP AR / EP a2a) 必须在 NVL 域内；DP AR 频率低，跨节点 OK。§1
]

#interview[
  *Q4*: bisection bandwidth 是什么？为什么 all-to-all 特别敏感？

  A: 把网络切成两半，穿过 cut 的总流量。all-to-all 里每对 rank 都通信，跨节点流量 $prop n^2$；只有 bisection $prop n$ 的 fat-tree 才不阻塞。100+ 节点 IB 常 oversubscribe，a2a 掉到理论 1/3。§1
]

#interview[
  *Q5*: NCCL Ring vs Tree 什么时候切换？

  A: Ring 带宽项 $2V$ 常数，延迟项 $2(W-1)alpha$；Tree 带宽项 $2 V log W$ 更大，延迟 $2 log W dot alpha$ 更小。小 payload (< 1 MB) + 大 W 用 Tree；大 payload 用 Ring。NCCL 自动，用 `NCCL_ALGO` 强制。§1
]

#interview[
  *Q6*: NVLS (NVLink SHARP) 加速原理？

  A: reduction 下放到 NVSwitch 芯片，每卡只 send 一次到 switch，switch 做 sum 再广播。per-GPU volume 从 $2V(W-1)/W$ 降到 $2V/W$，~1.5-2× 加速。`NCCL_NVLS_ENABLE=1` 开启。§1
]

#interview[
  *Q7*: `CUDA_DEVICE_MAX_CONNECTIONS` 是什么？

  A: host 到 device 的 hardware queue 数。=1 时 stream 串行，overlap 失效。Hopper+ 建议 8，允许多 stream 并行 issue command。是所有 overlap 优化的前置。§1, §11
]

#interview[
  *Q8*: IBGDA 是什么？为什么能加速小 message？

  A: Infiniband GPUDirect Async，GPU 直接下发 IB verb，绕过 CPU。small message 里 20 µs 的 CPU launch overhead → < 5 µs，DeepEP / UCX 用。§1
]

#interview[
  *Q9*: 一个跨节点 collective 的延迟由哪几部分组成？

  A: (1) NCCL kernel launch (~10-20 µs); (2) NCCL 内部 sync (~10-30 µs); (3) PCIe DMA (~1 µs); (4) IB switch hops (~2-3 µs); (5) 对端 kernel launch (~10-20 µs)。total ~30-80 µs α。§1
]

#interview[
  *Q10*: 心算：Llama-70B on 512 卡, MFU=45%, D=1T tokens，训多久？

  A: FLOPs = 6 × 70e9 × 1e12 = 4.2e23。cards × MFU × peak = 512 × 0.45 × 989e12 = 2.28e17 FLOPs/s。time = 4.2e23 / 2.28e17 = 1.84e6 s ≈ 21 天。§2
]

#interview[
  *Q11*: 参数量 = 12 L H² + V H，"12" 从哪来？

  A: 每层 attention Q/K/V/O 各 $H^2$ (4)，FFN $H -> 4H -> H$ = $8 H^2$ (8)。共 12。GLU 有三 linear ($H -> 8/3 H, H -> 8/3 H, 8/3 H -> H$) 仍 8。§2
]

#interview[
  *Q12*: 混合精度训练 16 bytes/param，来源？

  A: BF16 weight (2) + FP32 master weight (4) + BF16 grad (2) + FP32 momentum m (4) + FP32 variance v (4) = 16。Precision-aware (m,v BF16) 是 10。§2, §9
]

#interview[
  *Q13*: FLOPs = 6 N D 里的 "6"？

  A: 每 param forward 用 1 次 (2 FLOPs)，backward 求 input grad (2)、weight grad (2)，共 6。忽略 attention 的 $S^2$ 项 (在 $S << H$ 时可忽略)。§2
]

#interview[
  *Q14*: Roofline ridge point 定义？H100 是多少？

  A: $I^* = pi_"peak" / beta_"peak"$，超过 $I^*$ 时 kernel 变 compute-bound。H100 BF16 = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/byte。§2
]

#interview[
  *Q15*: 三堵墙是什么？举例说明它们互相耦合。

  A: Memory / Communication / Compute Efficiency Wall。案例：EP 切 MoE 解 memory → 引入 a2a 暴露 comm → DualPipe overlap → 需 2× weight → memory 又紧 → FP8 → kernel 碎片化 → compute 掉率 → grouped GEMM + CUDA graph 缓解。§2
]

== DDP / FSDP / ZeRO (Q16-Q30)

#interview[
  *Q16*: DDP overlap 是怎么实现的？

  A: bucket + backward hook + async AR。param 分 25 MB bucket；backward 时某 bucket 所有 grad ready 立刻起 async AllReduce；与前面 layer backward compute overlap。§3
]

#interview[
  *Q17*: `no_sync()` 什么时候用？

  A: gradient accumulation 期间。前 n-1 个 micro-batch 不做 grad AR，只最后一个同步，省 (n-1) 次通信。§3
]

#interview[
  *Q18*: `find_unused_parameters=True` 为什么慢？

  A: DDP 每 step 遍历所有 param + hook counting 判定谁参与 forward autograd graph。CPU overhead 10-30%。重构模型让所有 param 参与更好。§3
]

#interview[
  *Q19*: BF16 训练为什么不需要 GradScaler？

  A: BF16 exponent 8 bit 与 FP32 同，动态范围 $10^-38$ 到 $10^38$，无 underflow。FP16 exponent 5 bit 范围小才需要 scale。§3, §9
]

#interview[
  *Q20*: ZeRO 三个 stage 分别切什么？每卡显存？

  A: Stage-1 切 optim state ($4P + 12P/W$)；Stage-2 加切 grad ($2P + 14P/W$)；Stage-3 加切 weight ($16P/W$，= FSDP FULL_SHARD)。§4
]

#interview[
  *Q21*: ZeRO-3 通信量比 DDP 多 50%，为什么还用？

  A: 大模型 DDP 装不下。ZeRO-3 显存 $16P/W$ 可训 100B+。多 50% 通信换 W 倍显存是不对等交换，没得选。§4
]

#interview[
  *Q22*: HSDP (HYBRID_SHARD) 相对 pure FSDP 好处？

  A: 节点内 FULL_SHARD (NVLink AG/RS)，节点间 DDP (IB AR grad shard)。跨节点通信从 $3P$ 降到 $2P/n$，wall-clock 5-10× 加速。Llama-3 用。§4
]

#interview[
  *Q23*: FSDP `auto_wrap_policy` 怎么选？

  A: 每 Transformer layer 一个 unit。太大退化 DDP，太小 AG 次数爆炸。`transformer_auto_wrap_policy(transformer_layer_cls={YourLayer})`。§4
]

#interview[
  *Q24*: FSDP + activation checkpoint 顺序？

  A: 先 apply AC 再 FSDP wrap。反过来 FSDP 内部 AG 与 AC detach 冲突。§4, §10
]

#interview[
  *Q25*: `use_orig_params=True` 好处？

  A: optimizer 看到原始 `named_parameters()` 而不是 FlatParameter，可按 name 设 lr/wd/frozen。§4
]

#interview[
  *Q26*: FSDP2 相比 FSDP1 改进？

  A: (1) per-param sharding (无 FlatParameter)；(2) DTensor 后端；(3) 与 TP composable；(4) 更好的 optim 交互。torchtitan 用。§4
]

#interview[
  *Q27*: FSDP `BackwardPrefetch.BACKWARD_PRE` 与 `POST` 差别？

  A: PRE: 第 $l$ 层 backward 开始前 issue $l-1$ 层 AG，overlap 更多但内存峰值高。POST: $l$ 层结束后再 issue，overlap 少内存少。OOM 时 PRE → POST。§4, §11
]

#interview[
  *Q28*: BF16 grad AR 什么时候要 upcast FP32？

  A: world_size > 512-1024 时 BF16 累加误差累积 → loss 抖动。`MixedPrecision(reduce_dtype=torch.float32)` 或 Megatron `--grad-reduce-in-fp32`。通信翻倍，精度稳。§4, §9
]

#interview[
  *Q29*: ZeRO-Offload / CPU Offload 什么时候用？

  A: 单机大模型 fine-tune (7B on 4090 24GB)；或超大 model + spare CPU memory + PCIe 有余量。生产多卡训练很少用（有钱直接加 GPU）。§4, §10
]

#interview[
  *Q30*: FSDP 与 TP 有什么本质区别？

  A: FSDP 切"存储"，compute 时 AG 回全 weight 算——activation 不切。TP 切"使用"，compute 直接在 shard weight 上算——activation 沿 TP 维切。所以 FSDP activation 大 TP activation 小，但 FSDP 通信在 param level (与 batch 无关)，TP 通信在 activation level。§4, §5
]

== TP / SP / PP (Q31-Q50)

#interview[
  *Q31*: Megatron TP 里 FFN 的 column-then-row pattern 通信量？

  A: column parallel 出来 activation 沿 output 切分，无需通信；后接 GELU (elementwise)；再 row parallel 后需 1 次 AR。整个 FFN forward 1 AR，backward 1 AR。§5
]

#interview[
  *Q32*: Attention 层的 TP 分解？

  A: QKV/O projection 沿 head 维 column parallel (Q/K/V) → attention 本地做 → output projection row parallel + AR。1 AR forward + 1 backward。§5
]

#interview[
  *Q33*: Sequence Parallel 是什么？通信量变吗？

  A: LN/Dropout 沿 seq 切，进 TP 前 AG，出 TP 后 RS。原来的 AR 变 AG+RS，通信量*完全相同* ($V(W-1)/W$ each)。activation 显存 $B S H$ → $B S H / "TP"$。§5
]

#interview[
  *Q34*: 为什么 TP 不超过 8？

  A: (1) NVL 域只 8 卡, TP > 8 跨节点极慢; (2) AR volume $2(W-1)/W$ 逼近 2 后不省, compute 每卡 $1/W$ 降; (3) activation 沿 TP 切但 AR volume 不变。GB200 NVL72 上 Megatron 建议 TP ≤ 8。§5
]

#interview[
  *Q35*: Async TP 怎么 overlap？

  A: `--tp-comm-overlap` + TE `LayerNormLinear`。GEMM 拆 tile，per-tile 边算边发通信；tile k 算完起 chunk k 通信，tile k+1 计算与 chunk k 通信 overlap。TP 通信占比 15-20% → 5-8%。§5, §11
]

#interview[
  *Q36*: GQA num_kv_heads=8 在 TP=8 与 TP=16 分别怎么处理？

  A: TP=8: K/V 每卡 1 head, OK。TP=16: K/V 每卡 0.5 head 不行——复制 K/V (每 2 rank 存一份) 或换 TP=8。Llama-2/3 GQA=8 是为 TP=8 设的。§5
]

#interview[
  *Q37*: vocab-parallel embedding 是什么？

  A: 大 vocab (128K+) 时 embedding weight 大 + logits activation 大。沿 vocab 维 TP 切 embedding，对应 index 到 partial logits，最后 fused vocab-parallel cross_entropy 直接在 shard logits 上算——省 loss activation。§5
]

#interview[
  *Q38*: GPipe 与 1F1B 的 bubble 一样，为什么选 1F1B？

  A: 1F1B activation 显存从 $m$ 份降到 $P$ 份。$m/P$ 倍显存节省，其他一致。§6
]

#interview[
  *Q39*: Interleaved 1F1B bubble ratio？代价？

  A: $(P-1)/(V m + P-1)$，$V$ 倍改善。代价：P2P × V，activation × V，代码复杂。$V=4, P=8$ bubble 从 46.7% → 21.9%。§6
]

#interview[
  *Q40*: ZeroBubble 怎么做到 ~0 bubble？

  A: backward 拆 B (input-grad, 阻塞) + W (weight-grad, 不阻塞)。B 尽早算, W 见缝插针填 bubble。$B approx W$ 时 bubble → 0。代价：autograd hack。§6
]

#interview[
  *Q41*: DualPipe vs ZeroBubble？

  A: DualPipe = ZB + bidirectional injection + 4-phase overlap。bidirectional 让 pipeline 两端同时注入 → bubble 更小，但需 2× 权重。§6
]

#interview[
  *Q42*: DualPipe vs Megatron FWD-BWD merged 生产选哪个？

  A: 大多数场景 FWD-BWD merged。DualPipe 需 2× weight，overlap 收益差不到 3%。除非训 500B+ MoE 且 IB comm 极度受限，否则 DualPipe overkill。§6
]

#interview[
  *Q43*: PP 组该跨节点还是同节点？

  A: 跨节点 OK。PP 用 P2P，只 pair-to-pair，通信频率低 (每 stage 过渡一次)。跨节点 IB 够用。TP/EP 才必须同 NVLink。§6
]

#interview[
  *Q44*: PP 训练里最后一层为什么特别慢？

  A: LM head + loss + logits (B, S, V) 巨大 activation & compute。Backward 从这里开始。tie embedding + LM head 减 param, 或最后 stage 少几个 transformer layer 给 vocab head 留余量。§6
]

#interview[
  *Q45*: PP + TP 排布？

  A: TP 组必须同 NVL 域。TP=8, PP=8 时: TP 组 = 每 node 8 卡, PP 组 = 跨 8 个 node 的对应 rank。Megatron 默认这么排。§6
]

#interview[
  *Q46*: 为什么 PP 让 micro-batch 越多 bubble 越小？

  A: bubble ratio = $(P-1)/(m+P-1)$，$m$ 大分母大 bubble 小。GPipe/1F1B 建议 $m >> P$，但太大 activation 爆。$m=P$ 时 bubble 50%，$m=4P$ 时 20%。§6
]

#interview[
  *Q47*: ZeroBubble 的 W (weight-grad) 计算为什么更慢？

  A: 原 fused B+W 里 activation 只 load 一次；分开后 W 时要再 load 一次 activation → HBM 带宽 x2。kernel 效率 -10-20%。但 bubble 减少的收益 > 效率损失。§6
]

#interview[
  *Q48*: 你怎么选 PP 深度？

  A: 每 stage 层数 4-8 是 sweet spot。太少 bubble 大且 P2P overhead 高；太多 stage compute > 通信, overlap 空间小。Llama-3 405B: 126 layer / PP=16 ≈ 8 layer/stage。§6
]

#interview[
  *Q49*: `--delay-wgrad-compute` 是什么？

  A: 把 backward 的 W (weight-grad) 计算延后，允许 B 继续 pipeline 前进——ZeroBubble 的 W-only 分离。让 stream overlap 更多。Megatron 与 `--overlap-moe-expert-parallel-comm` 配合。§6, §11
]

#interview[
  *Q50*: PP 组的 P2P 用 NCCL 还是 MPI？

  A: 现代 PyTorch/Megatron 用 NCCL P2P (`send`/`recv`)，走 CUDA-aware。老代码有用 MPI 或 Gloo，但性能差。生产必用 NCCL。§6
]

== 长上下文 / CP (Q51-Q60)

#interview[
  *Q51*: Ulysses vs Ring Attention?

  A: Ulysses = all-to-all 沿 head 切换，attention 全本地。Ring = P2P 逐步旋转 K/V，累加 Flash lse。Ulysses 简单快但 CP ≤ head 数；Ring 无 head 限制但小-K kernel 效率低。§7
]

#interview[
  *Q52*: Ring Attention causal mask 怎么处理？

  A: naive: rank $r$ 只处理 K/V shard index ≤ $r$，负载不均 2×。Striped/Zigzag: seq permutation 让每 rank 都持"前半+后半"pattern，负载均衡。生产用 Zigzag。§7
]

#interview[
  *Q53*: USP 是什么？

  A: Unified Sequence Parallel。CP = CP_ulysses × CP_ring。先 Ulysses 沿 head 切一层，再 Ring 沿 seq 切一层。兼顾 head 少 + seq 长场景。Kimi 1M ctx 用。§7
]

#interview[
  *Q54*: Ulysses 通信量与 CP 有关但 Ring 与 CP 无关？

  A: Ulysses 一次 a2a $V(W-1)/W approx V$。Ring 每步 P2P $V/"CP"$, $"CP"$ 步 total $V$——与 CP 无关。所以 CP 大时 Ring 通信不涨。§7
]

#interview[
  *Q55*: 训 1M context 你怎么组合并行？

  A: 512 卡 H100 例：TP=8 × CP=16 (USP: Ulysses 4 × Ring 4) × PP=2 × DP=2。CP 主承担 seq 切，TP 切 head，FSDP2 切 weight。加 full activation recompute + FA3 with cp_ring。§7
]

#interview[
  *Q56*: GQA num_kv_heads=8 能用 CP=16 Ulysses 吗？

  A: 不能——KV head 只 8。要么 CP=8 (Ulysses)；要么 Ring；要么 USP: Ulysses=8 × Ring=2 = CP=16。§7
]

#interview[
  *Q57*: Ring Attention 里 Flash lse 合并公式？

  A: Flash online softmax 支持分块累加。每步得局部 $(O^((s)), "lse"^((s)))$，合并：$O_"new" = O_"old" e^("lse"_"old" - "lse"_"max") + O^((s)) e^("lse"^((s)) - "lse"_"max")$，$"lse"_"new" = "logsumexp"("lse"_"old", "lse"^((s)))$。§7
]

#interview[
  *Q58*: FA3 与 Ring Attention 集成有什么优势？

  A: `cp_group` + `cp_ring=True`。P2P 与 tile compute 在 kernel 内 async overlap，Hopper 用 TMA 让通信几乎 hidden。比 Python-level ring 快 20-30%。§7
]

#interview[
  *Q59*: RoPE 长 ctx (1M) 训练要注意什么？

  A: (1) RoPE base 加大 (`--rotary-base 500000+`)，缓解 extrapolation degradation；(2) RoPE cos/sin cache 大 (S × d_h = 500 MB @ 1M)，用 cached RoPE；(3) 组合 CP 时 pos_id 沿 seq 切正确传给 RoPE fused kernel。§7
]

#interview[
  *Q60*: 长 ctx 训练里 activation 主要来源和优化？

  A: LN/Dropout $B S H$, FFN intermediate $B S I$, Q/K/V/O activation。CP 切 seq attention 内；Megatron SP 切 LN/Dropout；full recomp 只存 layer input；activation offload PCIe overlap。DeepSeek-V3 Table 3 activation 66% 显存。§7, §10
]

== MoE / EP (Q61-Q70)

#interview[
  *Q61*: EP vs TP 切 expert，选哪个？

  A: EP。TP 切 expert 后每卡还有 E 个"半专家"—— dense 化。EP 让每卡持完整 $E/"EP"$ 个专家，通信只对被路由 tokens 做。EP 通信 $prop B S K H$ (稀疏)，TP 通信 $prop B S H$ 但每层 AR。EP 符合 MoE 稀疏本质。§8
]

#interview[
  *Q62*: All-to-all 通信量为什么与 E 无关？

  A: a2a 传每 token 的 hidden vector，total = $B S K H$。K 决定通信量（一 token 走几专家），E 决定池大小但每 token 还是走 K 个——加多专家不加通信。DS-V3 用 256 experts 通信量与 8 experts 相同。§8
]

#interview[
  *Q63*: 层次化 all-to-all 收益？

  A: NVLink 400 GB/s vs IB 50 GB/s。intra-node NVLink 聚合 → 单次 inter-node IB → intra-node 散发。IB 带宽消耗从 $O(N^2 K H)$ 降到 $O(N K H)$，100+ 节点差 10×。§8
]

#interview[
  *Q64*: DeepEP V2 相比 V1 改进？

  A: V1 占 20 SM (GEMM 掉 20%)。V2 warp specialization + auto-tuned chunk 只占 4-6 SM。让 fine-grained MoE 可行 (256 experts topk=8)。§8
]

#interview[
  *Q65*: MoE 一层 forward 通信几次？

  A: 2 次 a2a (dispatch + combine)。若 TP 组合还有 2 AR (attention output + FFN)。加 backward 就是 8 次集合通信/层。§8
]

#interview[
  *Q66*: 为什么 MoE overlap 比 dense 更重要？

  A: MoE 一层 4 次 a2a 占 15-40% forward 时间；dense 只 2 次 AR (10-20%)。overlap 到 \<5% 后，MoE 省绝对时间 > dense。DualPipe 是极端案例。§8, §11
]

#interview[
  *Q67*: aux loss vs aux-loss-free？

  A: aux loss 给 router 冲突梯度 (main task vs balance)。DeepSeek aux-loss-free: 可学 bias 只影响 topk 选择不影响 gate 梯度；简单反馈规则调 bias。router 梯度纯净。§8
]

#interview[
  *Q68*: Parallel Folding 是什么？

  A: 打破 EP ≤ DP 约束，attention 与 expert 用独立 process group。允许 attention-DP=32, expert-EP=64+DP=2 非对称配置。DeepSeek fine-grained MoE 类模型用。§8
]

#interview[
  *Q69*: MoE + CP 一起用的注意事项？

  A: (1) MoE a2a 与 CP P2P 争带宽——让 EP/CP 走不同 device mesh 维度；(2) EP 沿一维 (跨节点用 DeepEP 缓解), CP 沿另一维；(3) MoE token 数不均 + CP 序列不均双重不确定——用 capacity padding + fixed-length packing。§7, §8
]

#interview[
  *Q70*: fine-grained MoE (256 experts) 与 vanilla (8 experts) 的差别？

  A: 通信量*相同* (E 无关)。显存 weight 相同 (总 param 相同)。grouped GEMM 每 expert M 维 8× 更小 → 易 memory-bound。需要 DeepGEMM / TE GroupedLinear 好的 kernel 才不掉率。§8
]

== 精度 / 优化 (Q71-Q80)

#interview[
  *Q71*: BF16 vs FP16 训练？

  A: BF16 exponent 8 bit (FP32 同) 动态范围大，无需 GradScaler；mantissa 7 bit 精度稍差。FP16 exp 5 bit 需 GradScaler 防 underflow；mantissa 10 bit 精度好但 range 小。H100 时代基本弃 FP16。§9
]

#interview[
  *Q72*: FP32 master weight 为什么必需？

  A: BF16 mantissa 7 bit，$"weight" - "lr" times "update"$ 里 update 通常小 1000+ 倍，BF16 直接减 round 到 0。FP32 master 累积小 update，broadcast BF16 用于 fwd/bwd。§9
]

#interview[
  *Q73*: FP8 Delayed Scaling vs Current Scaling?

  A: Delayed 用上次 amax 估计 scale，无额外 reduce，快但对 outlier 敏感。Current 每次求当前 amax，overhead 3-5%，精度稳。TE default = Delayed；DeepSeek 用 blockwise (每 tile 一个 scale)。§9
]

#interview[
  *Q74*: DeepSeek blockwise FP8 recipe?

  A: Weight 按 $(128,128)$ tile 各存一个 FP32 scale；activation 按 $(1,128)$ tile scale。粒度细，outlier 只影响自己 tile。三路 GEMM (Fprop/Dgrad/Wgrad) 全 FP8，只 optim state 与 master weight FP32。val loss error < 0.25% vs BF16。§9
]

#interview[
  *Q75*: FP8 里 attention softmax 为什么不用 FP8？

  A: softmax 需 max + exp + sum + divide。FP8 mantissa 3 bit 精度不足以做 exp 累加。FA-FP8 只把 $Q K^T$ 和 $P V$ GEMM 用 FP8，softmax stats 全 FP32。§9
]

#interview[
  *Q76*: activation checkpoint compute overhead 为什么 33%？

  A: 1 fwd + 2 bwd = 3F baseline；checkpoint 加 1 fwd 重算 = 4F。overhead = 4/3 - 1 = 33%。§10
]

#interview[
  *Q77*: FA 之后 activation checkpoint 还有必要？

  A: 有。FA 只省 $B S^2 A$ (attention matrix)；activation 还有 $B S H$ (LN, residual)、$B S I$ (FFN)。长 seq 累积 GB 级。selective recompute 对 Q/K/V matmul activation checkpoint 有用。§10
]

#interview[
  *Q78*: MoE 场景为什么不整层 recompute？

  A: 整层 recompute 会重触发 a2a 通信——通信开销翻倍。fine-grained recompute 只重算计算部分 (moe_act, expert intermediate)，跳过 dispatch/combine activation。§10
]

#interview[
  *Q79*: Precision-Aware Optimizer 代价？

  A: 几乎 0。fused kernel 里 BF16↔FP32 cast 微小 latency。收益 optim state -50%。Megatron 2024 新 flag。§10
]

#interview[
  *Q80*: FSDP 与 activation checkpoint 组合顺序？

  A: 先 apply_activation_checkpointing 再 FSDP wrap。反了 FSDP 内部 AG 与 AC detach 冲突，backward 会重新 AllGather 但 AC 已 discard activation。§10
]

== Overlap / 系统 (Q81-Q90)

#interview[
  *Q81*: DDP overlap 依赖什么？

  A: (1) `find_unused_parameters=False`；(2) bucket_cap_mb ≥ 25 (推荐 100-500 大模型)；(3) `gradient_as_bucket_view=True`；(4) NCCL 异步 stream；(5) CUDA_DEVICE_MAX_CONNECTIONS > 1。§3, §11
]

#interview[
  *Q82*: FSDP forward_prefetch 与 backward_prefetch 时机？

  A: forward_prefetch: 层 $l$ 开始时 issue $l+1$ AG。backward_prefetch (PRE): 层 $l$ backward 开始前 issue $l-1$ AG。都"往未来看"，代价显存峰值高。§4, §11
]

#interview[
  *Q83*: Async TP 收益？

  A: TP=8 AR 占 15-20% → 5-8%，step time -14%。TE `LayerNormLinear` fused kernel 支持。§5, §11
]

#interview[
  *Q84*: MoE 场景开 `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` 收益？

  A: EP a2a 占 30-40% → \< 5%，overlap 93%。不需 2× weight (对比 DualPipe)。§8, §11
]

#interview[
  *Q85*: nsys 里怎么判 overlap 好坏？

  A: 多 stream row。理想: comm stream 和 compute stream 时间列 overlap，两者少 gap。不好: comm stream 有 kernel 但 compute stream gap → compute 等 comm；两 stream 完全错开 → 无 overlap。§11
]

#interview[
  *Q86*: `.item()` / `.tolist()` 在训练循环里的后果？

  A: D2H sync，host 阻塞。破坏 (1) CUDA graph capture；(2) stream overlap；(3) profile 出现 idle time。log 用 tensor 累积到 log step 才 D2H。§11
]

#interview[
  *Q87*: 训练 dataloader wait 占 step > 5%，怎么排查？

  A: 加 time.time 前后测。检查 (1) num_workers < 8；(2) prefetch_factor 只 2；(3) tokenizer 单进程慢；(4) 数据慢盘；(5) preprocess 重 (大图 decode)。分别对症。§12
]

#interview[
  *Q88*: `persistent_workers=True` 什么用？

  A: DataLoader worker 进程每 epoch 不重启，省 fork + tokenizer 加载时间 (每 epoch ~30s)。生产训练必开。§12
]

#interview[
  *Q89*: sequence packing 与 batch padding 差别？

  A: padding: batch 内 pad 到 max seq，浪费 30-70% compute。packing: 拼接短 seq 到固定长，attention mask (或 FA varlen cu_seqlens) 保证不跨 doc attend。padding ratio \< 5%，吞吐 +50-100%。§12
]

#interview[
  *Q90*: `torch.utils.data.IterableDataset` 里 rank 分 shard 常见 bug？

  A: (1) shard 数不能整除 world × workers，某些 rank 少数据 → hang; (2) 忘 `worker_init_fn`，多 worker 拉同一 shard; (3) shard 内无 shuffle; (4) resume 时未保存 iterator state; (5) 长 seq 分给某 rank 造成不均。§12
]

== 多模态 / RL / 稳定性 (Q91-Q100)

#interview[
  *Q91*: 多模态 vision encoder 与 LLM 为什么并行策略分开？

  A: ViT 300M-1B 一张卡装得下用 DDP/FSDP；LLM 70B+ 需 TP+PP+FSDP。同 TP=8 setup 让 ViT 白开 TP AR。分离后 MFU +10-20%。§13
]

#interview[
  *Q92*: NaViT packing 与文本 packing 差别？

  A: 文本沿 seq 拼；NaViT 把不同尺寸图 patch 化后 pack 到一个 sequence，attention mask 图间不 attend。unit 是 image patch block。让 batch 混合不同分辨率图像。§13
]

#interview[
  *Q93*: DDP 里图像数不均衡后果？

  A: rank 0 十张图, rank 1 零张 → ViT forward 差 10×。DP AR sync 时快 rank 空等，MFU -20-30%。做 cost-based balancing (n_images + text_len 二维桶)。§13
]

#interview[
  *Q94*: 视频训练最大显存瓶颈？

  A: 视频 tokens 爆炸 (8 frames × 576 tokens = 4608)。做法 (a) Q-Former/Perceiver 压缩; (b) temporal pooling; (c) 3D VAE 直接编 spatio-temporal。CogVideoX 用 (c) 把 (49,480,720) 压到 (13,60,90) latent。§13
]

#interview[
  *Q95*: RLHF 为什么要 rollout / train 分离？

  A: rollout 是 autoregressive decode (memory-bound, small-M)；train 是 dense fwd+bwd (compute-bound, large-M)。用 training 跑 rollout 效率极低 (无 KV cache, 无 continuous batch)。分离后 vLLM/SGLang 5-10× 加速 rollout。§14
]

#interview[
  *Q96*: GRPO 相对 PPO 简化在哪？

  A: 去 value model。用 group baseline: N 个 rollout 算 advantage = (r - group_mean) / group_std。少训一 model，省 20-30% GPU。DeepSeek-R1 证明有效。§14
]

#interview[
  *Q97*: weight 从 training 到 rollout engine 同步方法？

  A: (1) Colocated 直接 copy (同 pool 交替)；(2) NCCL cross-group broadcast (不同 pool)；(3) shared GPU buffer 零 copy。verl/OpenRLHF 主要用 (2) + overlap with training。§14
]

#interview[
  *Q98*: 10000 卡训 3 个月，容错怎么设计？

  A: 三层：(1) 监控 dcgm + IB + 温度; (2) NCCL timeout + async error; (3) async checkpoint 15 min + rank-agnostic + spare node auto-replace + resharding。目标 MTBF 2h → downtime \< 5%。§16
]

#interview[
  *Q99*: async checkpoint 为什么比 sync 快？

  A: sync: gather 到 rank 0 → write NAS，几分钟停 training。async: D2H 到 CPU pinned mem (10s) + background 写 disk (分钟级)，training 不阻塞。停 training 时间从几分钟 → < 10s。§16
]

#interview[
  *Q100*: loss NaN 你怎么处理？

  A: (a) save debug ckpt (含最近 batch + activation stats); (b) 检查 grad_norm; (c) skip 这 batch 继续训——多数能自恢复; (d) 若连续 NaN, rollback 到 100 步前 ckpt 并降 LR 20%; (e) audit batch 找 outlier data。Llama-3 训练 466 次 spike, 90% 用 (c) 处理。§16
]

== RLVR (Q101-Q110)

#interview[
  *Q101*: RLVR 相比经典 RLHF，算力结构变了什么？

  A: 多出一个 *CPU verifier 集群*（沙箱跑测试 / 符号比对），同时可能少掉 reward model 和 ref model 两个 GPU 池（去 KL 的配方）。从"纯 GPU"变成"GPU + 大规模 CPU 沙箱"。新瓶颈是 verifier 吞吐与尾延迟。§15
]

#interview[
  *Q102*: DeepSeek-R1 为什么明确弃用 neural PRM/ORM？

  A: 三个理由：大规模 RL 下 reward hacking 不可避免；对抗 hacking 要不断重训 RM，代价高；RM 本身准确率成为 policy 的天花板。数学/代码这类有客观对错的任务改用规则 reward（答案比对 + 格式检查）就没有这些问题。§15
]

#interview[
  *Q103*: 四类 verifier 的延迟量级各是多少？哪个决定架构？

  A: 数学符号比对 1–50 ms；代码跑单测 0.1–10 s；Lean/Coq 编译 秒\~分钟；LLM-as-judge ms\~秒（且又变回 GPU 负载）。数学便宜到可同步跑，代码/形式化必须独立池 + 异步。决定架构的不是均值而是*尾延迟*：一组 16 个响应里只要 1 个卡住，整组 advantage 出不来。§15
]

#interview[
  *Q104*: 写出 verifier 池的容量公式，并说明 $t_v$ 该代什么值。

  A: $W gt.eq G dot r dot (1-h) dot t_v \/ u$（$G$ rollout GPU 数，$r$ 每卡每秒响应，$h$ 缓存+去重命中率，$u$ 目标利用率）。$t_v$ *代均值*——排队稳定性只取决于平均服务时间，按 p90 代入是白买 2–3 倍机器。重尾的影响不在容量而在组内最大值，那是另一个问题。§15
]

#interview[
  *Q105*: 为什么加 verifier worker 解决不了 GRPO 的等待？

  A: GRPO 要等一组 $N$ 个响应*全部*验完才能算 advantage，优化器等的是组内*最大值*。重尾下 $E["max of" 16] approx 4 - 5 times E["单次"]$，且相当比例的组里至少有一个撞超时——那部分等待与 worker 数无关。压组最大值只能靠超时封顶和去重（减少从尾部抽样的次数），根本解法是异步验证让它离开关键路径。§15
]

#interview[
  *Q106*: verifier 侧性价比最高的两个优化是什么？

  A: 结果缓存 + 组内去重。GRPO 一个 prompt 采 $N=16$ 个响应，温度不高时完全相同的响应很常见（正确解法就那么几种），按 `hash(prompt, response)` 去重实测能省 30–60% 的 verifier 调用，零成本。其次是异步化，让 verify 与下一批 rollout 重叠。§15
]

#interview[
  *Q107*: 二值 reward 下 GRPO 为什么会大量白烧 rollout？

  A: 组内全对或全错时 $"std" = 0$ 且 $R_i - "mean" = 0$，advantage 全 0，整组零梯度。训练早期大量题全错、后期大量题全对，无效组占比能到 30–50%。解法：DAPO dynamic sampling（过采样后丢掉 acc 为 0 或 1 的组），或按历史准确率做难度分层采样优先取 0.2–0.8 的题。§15
]

#interview[
  *Q108*: dynamic sampling 对调度器提出了什么新要求？

  A: 每个 iteration 的 rollout 量不再是常数。过滤掉无效组后要继续补采直到凑满 batch，容量规划从"生成 N 组"变成"生成到 N 组存活"。仿真里存活率约 62%，意味着要多采 1.6 倍。调度器必须支持带反馈的流式采样。§15
]

#interview[
  *Q109*: Dr. GRPO 指出 GRPO 的哪两处偏差？

  A: (1) 按响应长度归一化 $1\/|o_i|$ 让长错误答案获得不成比例的权重，系统性鼓励响应变长——这就是"越训越长但没变强"的成因之一；(2) 按组标准差归一化 $1\/"std"$ 放大了极端难度题的权重。处方是去掉这两项、改用 token 级求和。DAPO 的 token-level loss 动机一致。§15
]

#interview[
  *Q110*: 长时间 RLVR 训练熵坍缩怎么办？为什么很多配方去掉了 KL？

  A: 熵坍缩链条：熵降 → 输出趋同 → 组内 $N$ 个采样几乎一样 → GRPO 没有差异 → 梯度消失。DAPO 的 clip-higher 把裁剪区间拆成非对称的 $[1-0.2, 1+0.28]$，给低概率 token（探索来源）留上升空间。去 KL 是因为 RLVR 的目标恰恰是学会 base 不会的长链推理，KL 是纯阻力；系统上还能省掉 ref model 一整个池，显存全给 KV cache。代价是失去跑偏的刹车，要靠监控语言混杂、格式崩坏和通用能力回退兜底。§15
]

== Agentic RL (Q111-Q120)

#interview[
  *Q111*: 多轮 agentic rollout 与单轮 rollout 在系统上差在哪？

  A: 四点：环境执行期间 GPU 闲置（工具调用几秒）；每轮都要 prefill，不复用 prefix cache 的话总代价 $O(T^2)$；轨迹长度方差从约 4× 涨到约 100×（1 轮 500 token \~ 50 轮 100K token）；训练侧多出 token masking。§15
]

#interview[
  *Q112*: 为什么 prefix cache 是 agentic RL 的生死线？开了还可能没用是怎么回事？

  A: 第 $k+1$ 轮输入 = 第 $k$ 轮完整上下文 + 新观测，不复用则 $T$ 轮 prefill 是 $O(T^2)$，20 轮要多付一个数量级。真正的坑是"开了但命中率低"：序列等环境那几秒里 KV cache 被 LRU 淘汰，回来又要重新 prefill。查引擎的 prefix cache 命中率指标；修法是扩大 KV 池（去 KL 省下的 ref 显存正好给它）、对等待中的序列 pinning 或换出 CPU、限制并发轨迹数让活跃集合装得下。§15
]

#interview[
  *Q113*: 同步批式 rollout 为什么必须换成异步？

  A: 同步是"这批一起走第 1 轮、一起等环境、再一起走第 2 轮"，每轮都要等这批里最慢的环境调用，GPU 纯闲。异步/连续批处理把几百条轨迹同时挂在引擎上，等环境的自然退出运行批次。仿真里 GPU 利用率 19% → 88%，makespan 快 4.6×。§15
]

#interview[
  *Q114*: 你用 p99/p50 衡量 agentic rollout 的 straggler，合理吗？

  A: 不合理，这个指标在这里会说反话。同步批式的 p99/p50 反而比异步更"健康"（2.85 vs 2.94），因为 barrier 把所有短轨迹都拖慢到最慢那条的节奏——离散度小是因为大家一起被饿死，百分位比值区分不了"没有 straggler"和"全都是 straggler"。同理单轨迹延迟也是红鲱鱼：异步故意让几百条在飞，单条延迟必然变差。该看 *makespan 和 GPU 利用率*，因为整个 iteration 组装完之前没有东西会消费单条轨迹。§15
]

#interview[
  *Q115*: partial rollout 解决什么？代价是什么？三种处理方式？

  A: 解决轨迹长度重尾导致的 straggler（同步收集一个 batch 要等最长那条）。做法是给每轮设 token/轮数预算，没跑完的存状态下轮接着跑（Kimi K1.5）。代价是 off-policy：一条轨迹跨了多个 policy 版本。三种处理从严到松：逐 token 记版本并用对应 logprob 算重要性比；限制最多跨 $K$ 个版本超了就丢（最常见）；直接当 on-policy。§15
]

#interview[
  *Q116*: 权重更新时几百条轨迹跑到一半，怎么办？

  A: 三选一。Drain 等全跑完再换权重——长尾让 GPU 空转，退回 straggler 问题；Abort 直接丢——浪费已生成 token，长轨迹损失最大；Carry-over 跨版本继续（即 partial rollout）——需要 off-policy 记账。长 horizon 场景下前两个都太贵，主流框架都往 carry-over 走。§15
]

#interview[
  *Q117*: 多轮轨迹里哪些 token 要 mask？不 mask 有什么后果？

  A: 只训 assistant 生成的 token，工具观测 token 必须 mask。三个后果：观测常占 60–80% token，绝大部分梯度花在教模型预测它原理上无法知道的工具输出；模型学会自己编造观测，推理时不调工具直接幻觉结果；重要性比对观测 token 没有意义，那些 token 不是 $pi_"old"$ 采样出来的。最阴险的是*它不会让训练崩*——loss 照降、grad_norm 正常，只能靠工具调用成功率发现。§15
]

#interview[
  *Q118*: `loss.sum() / mask.numel()` 和 `/ mask.sum()` 差在哪？

  A: 用 `numel()` 时分母含观测 token，等于按"这条轨迹的工具输出有多啰嗦"在缩放学习率——而那是环境决定的，不是你决定的。同样 5 轮、assistant 干了同样多活的三条轨迹，只因为工具从计算器换成网页抓取，loss 就能差 *40×*：啰嗦工具的轨迹被压到无关紧要，简洁的主导整个 batch。正确写法是 `/ mask.sum().clamp(min=1)`。§15
]

#interview[
  *Q119*: rollout 引擎和训练引擎的 logprob 对不上，有什么影响？怎么监控？

  A: 直接拿 rollout 的 logprob 当 $log pi_"old"$，第一个内层 epoch 本该恒为 1 的重要性比会偏离，造成本不该发生的裁剪，梯度被白砍——而且砍掉的恰恰是两边分歧最大的低概率探索 token。根因是 kernel、batch 组织、BF16 累加顺序、并行切分都不同。处方是用训练引擎重新前向算 $log pi_"old"$。监控要看*第一个 epoch 的裁剪比例*（应接近 0），不能只看 mean |dlogp|：分歧分布是重尾的，均值只有 0.09 时裁剪比例已经到 5%。§15
]

#interview[
  *Q120*: 沙箱 OOM、外部 API 限流导致的失败，该给什么 reward？

  A: 不能给 0。那是基础设施失败，不是 policy 的行为后果，给 0 等于往训练信号里注入随机噪声。正确做法是标为 `infra_failure`、从 batch 剔除、单独打点，并把 infra 失败率当 SLO 管——超过 1% 是要修的系统故障。考点是能不能区分"环境的错"和"模型的错"。§15
]

== SP/CP 全栈适配 (Q121-Q130)

#interview[
  *Q121*: CP 下 cross-entropy 怎么算？各卡算完平均一下行不行？

  A: 不行。各卡有效 token 数差别极大（prompt masking + padding，极端情况某卡为 0 → $0\/0$ 出 NaN，一次 all-reduce 全场变 NaN）。必须本地只求和，把分子 $sum m ell$ 与分母 $sum m$ 分别 all-reduce，最后相除。追问"跳过空卡再平均"——那等于把每卡权重拉平成 $1\/"CP"$，实测偏 $0.81 times$，效果是按答案长度重新加权样本，模型会偏好短输出。§7
]

#interview[
  *Q122*: loss 算对了，为什么梯度还可能差 CP 倍？

  A: 每卡产出的是"对全局 loss 的贡献"（本地分子 / 全局分母），这些贡献要*求和*才是全局梯度。但 CP 维常被折进 DP 的梯度归约，而 DP 是*求平均*，于是丢掉一个 $"CP"$ 因子（实测 $|g|$ 恰好是 $1\/"CP"$）。修法：loss 乘 CP，或 CP 维改 SUM 归约。要点是 *loss 打印值完全正确*，看曲线只会误判成"LR 要调大"，必须对拍参数更新量才能发现。§7
]

#interview[
  *Q123*: 一个 CP 组内各卡该拿相同还是不同的数据？怎么验证？

  A: 相同——一个 CP 组合起来才是一条序列。sampler 索引与 shuffle 种子都要用 `dp_rank`，不能用 `global_rank`。写错不报错：attention 把两个半截文档当一条序列算，同时 global batch size 悄悄变成配置值的 CP 倍，token 预算和等效 LR 全错。验证一行——`sample_id` 在 CP 组内 all-gather 后断言全相等，建议常驻。同规则适用 TP 组和 PP stage：只有 DP 维允许数据不同。§7, §12
]

#interview[
  *Q124*: 为什么需要 zigzag 切分？它引入了什么新麻烦？

  A: 连续切分在 causal mask 下负载不均——rank 0 的 query 几乎看不到 key，最后一卡要看全序列，实测每卡 $(q,k)$ 对数 $[36,100,164,228]$。collective 走最慢那卡，代价是 $max\/"mean"$（实测 $1.73 times$，CP 越大趋近 $2 times$），不是更夸张的 $max\/min$。zigzag 让 rank $r$ 取 chunk $r$ 与 $2"CP"-1-r$，对数严格相等。新麻烦：shard 不再连续，position_ids/doc_ids 要按同置换分发、mask 不能用下三角、输出要逆置换、halo 要从两个不同 rank 各取一次。§7
]

#interview[
  *Q125*: CP 下 RoPE 用本地 `arange(0, S/CP)` 当位置会怎样？

  A: 全错但不报错。RoPE 是相对位置编码，靠*绝对位置作差*实现，本地位置让所有相对距离都错。短序列时误差被 softmax 部分吸收，表现为"能训但外推能力差"；zigzag 下更糟，rank 0 持有的全局位置是 `[0,1,2,3,28,29,30,31]`，本地 arange 与之毫无关系。正确做法：把 `position_ids` 做成 batch 的显式字段，由 DataLoader 产出并随 token 一起切分，模型里不准用 `arange` 现场生成。§7
]

#interview[
  *Q126*: 开 CP 之后，哪些模块*不需要*额外同步？

  A: 所有逐 token 独立的模块——RMSNorm/LayerNorm（沿 hidden 维归一化，与序列怎么切无关）、FFN/SwiGLU、所有逐元素算子。这正是 Megatron SP 能把 LN/Dropout 放在 seq-shard 上零通信执行的原因。会真正需要跨 token 统计的是 BatchNorm 这类沿 batch/序列维求统计量的算子，而 LLM 基本不用它。这题是反向考察：能主动说"不要给 LN 加 all-reduce"，说明不是在乱加通信。§5, §7
]

#interview[
  *Q127*: CP 下 MoE 的 load-balancing aux loss 要怎么改？

  A: $"aux" = E sum_e f_e P_e$ 里 $f_e$（token 比例）和 $P_e$（平均概率）都是"对 token 求平均"，必须在 CP 组上归约 expert 直方图和概率和。只统计本卡的后果不只是数值偏大（实测 $3.48 times$）：一个*全局完美均衡*的 router 会被每张卡判为"我的 token 全去了一个 expert"，梯度在自己那个 expert 的 logit 上为正，*把各卡推离全局最优*——正确梯度此时恰为零。征状是 load-balance metric 长期震荡不收敛。z-loss 与 capacity/drop rate 同理。§7, §8
]

#interview[
  *Q128*: reward model 的打分头在 CP 下为什么会错？

  A: 分数取自最后一个有效 token 的 hidden state，而这个 token 只落在*某一张卡*上。各卡各取"自己 shard 的最后一个有效 token"的话，只有 owner 是对的，其他卡拿的是序列中间的 token，整段是 padding 的卡还会崩在空索引上。正确做法：one-hot 选中后做一次*可微* all-reduce（forward 求和、backward 恒等），非 owner 贡献严格为零。裸 `dist.all_reduce` 是 in-place 的，autograd 不知道其他卡存在。同类还有 mean pooling、`[CLS]` 头、以及 RL 的 GAE（沿序列反向递推）。§7, §15
]

#interview[
  *Q129*: 为什么 CP 下 grad norm 容易算大 $sqrt("CP")$ 倍？

  A: 把分片参数的写法（跨组 all-reduce $|g|^2$）套到了副本参数上。CP 组内参数是副本，梯度归约后每卡已持有完整梯度，不该再求和。规则：*只在参数被分片的组上求和*（ZeRO/FSDP shard、TP column/row shard、EP expert 维），*不在副本组上求和*（CP、DP replica、TP 下复制的 LN/bias）。后果是裁剪提前触发、更新被静默压小（实测 clip 系数从 1.0 变 0.59），等于偷偷降 LR——而 grad_norm 日志记的正是那个虚高值，查不出来。Megatron 用 `param.tensor_model_parallel` / `param.shared` 区分。§7
]

#interview[
  *Q130*: 哪些算子在 CP 下需要 halo 交换？宽度多少？

  A: 只依赖邻近 token 的算子：causal conv1d 需要前 $K-1$ 个 token；sliding-window attention 需要 $w-1$；Mamba/SSM 扫描需要的不是 token 而是前一卡的*递归 state*，会把各卡串行化，除非改 chunked scan；多模态 ViT/audio 前端的 depthwise conv 同理。label shift / MTP 理论上也是 halo，但更好的做法是*在切分前对整条序列做位移*，零通信解决所有 $k$。注意误差形状：只错在每个接缝的 $K-1$ 行，占比随 shard 变长而下降（$S=128"K"$、CP=8 时只错 $0.002%$），loss 上完全看不见——所以带局部算子的模型往往宁可承受连续切分的 $1.7 times$ 负载不均，也不用 zigzag。§7
]
