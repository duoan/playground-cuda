#import "../template.typ": *

= 张量并行与流水并行

DDP 和 FSDP 切的都是"样本"和"状态"：DP 沿 batch 维复制计算，FSDP 再把参数/梯度/优化器状态沿 DP 组切开。它们从来没有把*一层的计算本身*切开。当一层已经装不下、或者 batch 小到不能再切时，只剩两条路：把单个 matmul 横向切给多卡（Tensor Parallel），或者把层纵向切成 stage 排流水（Pipeline Parallel）。这一章讲这两种"模型并行"的机制、通信量、以及为什么它们在集群里的摆放位置不能随便定。DTensor 的声明式写法见第 21 章。

== 什么时候必须超越 FSDP

三个信号，任何一个成立就该考虑 TP / PP：

+ *单层装不下*。FSDP 是按 module 做 AllGather 的：forward 到某一层时，这一层的完整参数必须在一张卡上物化。$H = 16"K"$ 的 MLP 单个权重就是 $16"K" times 64"K" times 2 = 2$ GB，加上 fp32 master weight 和优化器状态峰值直接爆。FSDP 无法把"一次 matmul"切开。
+ *batch 已经小到不能再切*。DP 的并行度上限是 global batch size。1024 卡而 global batch 只有 1024 个样本时，`micro_bsz = 1`，再加卡只能靠模型并行。而且 micro batch 太小 GEMM 的 M 维退化，SM 利用率掉。
+ *激活显存是瓶颈*。FSDP 不切激活：每张卡跑完整的 $(B, S, H)$ 前向，激活量与单卡训练同阶。TP 把中间激活沿 hidden 维切成 $1/T$，PP 让每张卡只持有 $L/P$ 层的激活。

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left),
  [*策略*], [*切什么*], [*不切什么*], [*每 step 通信*],
  [DDP], [batch], [参数、梯度、激活], [梯度 AllReduce，1 次],
  [FSDP], [batch + 参数/梯度/优化器状态], [激活、单层计算], [每层 AllGather + ReduceScatter],
  [TP], [单层的权重与中间激活], [batch、层数], [每层 2 次 AllReduce（fwd）],
  [PP], [层（stage）], [单层计算、batch], [stage 边界 P2P],
)

#insight[
  四种并行切的是四个正交的维度，所以可以叠加。选择顺序是固定的：先用 DP/FSDP（通信最轻），装不下再加 TP（限机内），层数太多再加 PP（可跨机）。
]

== Tensor Parallel：MLP 的 column-then-row

以 Transformer 的 MLP 为例。忽略 bias，两层权重是

$ Y = "GELU"(X W_1) W_2, quad W_1 in RR^(H times 4H), quad W_2 in RR^(4H times H) $

TP 组大小记作 $T$。有两种切法，选哪一种、以什么顺序组合，决定了通信次数。

=== Linear1：column parallel（沿输出维切）

把 $W_1$ 沿输出维（$4H$）切成 $T$ 块 $W_1 = [W_1^((0)), ..., W_1^((T-1))]$，每块 $H times 4H\/T$。输入 $X$ 在 TP 组内是 *replicate* 的，每张卡算

$ Z^((i)) = X W_1^((i)) in RR^(B times S times 4H\/T) $

*没有任何通信*：每张卡拿到输出的一段切片。

#figure(
  align(center, tp-partition(mode: "column", tp: 4, w: 4.6, h: 1.4,
    title: "W1: column parallel（沿 out 维切给 4 张卡）")),
  caption: [column parallel：$W_1$ 沿输出维切，输入 replicate，输出天然沿 hidden 维 shard。前向零通信。],
) <fig-tp-col>

=== 激活函数：正好能在 shard 上直接做

GELU / SiLU 是逐元素的，$"GELU"(Z)^((i)) = "GELU"(Z^((i)))$ ——每张卡对自己手上那 $4H\/T$ 列直接做，结果就是完整结果的对应切片。*这就是 column 必须在前的原因*：column parallel 的输出切分方式恰好与逐元素算子兼容。

=== Linear2：row parallel（沿输入维切）

$W_2$ 沿输入维（$4H$）切成 $T$ 块，每块 $4H\/T times H$。输入必须已经沿同一维切开——上一步给的正是这个形状。每张卡算

$ Y^((i)) = "GELU"(Z^((i))) W_2^((i)) in RR^(B times S times H) $

每块都是完整形状，但只是*部分和*。要拿到 $Y = sum_i Y^((i))$ 必须做一次 *AllReduce*。

#figure(
  align(center, tp-partition(mode: "row", tp: 4, w: 4.6, h: 1.4,
    title: "W2: row parallel（沿 in 维切给 4 张卡）")),
  caption: [row parallel：$W_2$ 沿输入维切，输入必须已按同一维 shard。每卡得到形状完整但数值为部分和的输出，需要一次 AllReduce 求和。],
) <fig-tp-row>

=== 为什么这个顺序只要一次 AllReduce

整条链：$X$（replicate）$-> W_1$ column（无通信）$->$ GELU（无通信）$-> W_2$ row $->$ *AllReduce* $-> Y$（replicate）。前向一次 AllReduce。

反过来（row-then-column）会怎样：$W_1$ 若是 row parallel，输入 $X$ 得先沿 $H$ 维切开，输出是部分和 → 必须先 AllReduce 才能算 GELU（逐元素算子需要*完整的数值*，部分和上做 GELU 是错的：$"GELU"(a + b) != "GELU"(a) + "GELU"(b)$）；AllReduce 之后 $W_2$ 若是 column parallel，输出又是 shard 的，下一层 LayerNorm 需要完整 hidden 才能算 → 再 AllReduce 一次。同样的数学结果，通信次数 2×。

#insight[
  Megatron-LM 的核心 trick 一句话：*让唯一的非线性算子落在"输出已经切好、且切法与逐元素兼容"的位置上*，这样两个 GEMM 之间不需要同步，只在最后把部分和合起来。高频题，能把"GELU 在部分和上不成立"说出来就到位了。
]

=== 手写一遍：`f` 和 `g` 两个对偶算子

TP 的全部通信可以收进两个自定义 autograd Function，它们互为对偶：`f` 前向恒等、反向 AllReduce；`g` 前向 AllReduce、反向恒等。

```python
# torchrun --nproc-per-node 2 tp_mlp.py   （2× A100，TP=2）
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

class _CopyToTP(torch.autograd.Function):     # f: fwd identity / bwd AllReduce
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        dist.all_reduce(g, group=ctx.group)   # 各卡的 dL/dX 求和
        return g, None

class _ReduceFromTP(torch.autograd.Function):  # g: fwd AllReduce / bwd identity
    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous()
        dist.all_reduce(x, group=group)        # 部分和求和
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None                         # 每卡的 dL/dY 本来就相同
```

为什么反向是这两个方向：前向 replicate 的 $X$ 被 $T$ 张卡各用了一次，链式法则要求把 $T$ 份 $partial L\/partial X$ 加起来（`f` 的反向 AllReduce）；前向做过 AllReduce 的 $Y$ 在每张卡上数值相同，反向传下来的 $partial L\/partial Y$ 也相同，直接透传（`g` 的反向恒等）。

```python
class ColumnParallelLinear(nn.Module):
    """weight: (out_f // tp, in_f)；输入 replicate，输出沿 out 维 shard。"""
    def __init__(self, in_f, out_f, group):
        super().__init__()
        self.group, tp = group, dist.get_world_size(group)
        assert out_f % tp == 0, "out_features 必须能被 tp 整除"
        self.weight = nn.Parameter(torch.empty(out_f // tp, in_f).normal_(std=0.02))
        self.bias = nn.Parameter(torch.zeros(out_f // tp))

    def forward(self, x):                      # x: (..., in_f) replicated
        x = _CopyToTP.apply(x, self.group)
        return F.linear(x, self.weight, self.bias)   # (..., out_f // tp)

class RowParallelLinear(nn.Module):
    """weight: (out_f, in_f // tp)；输入已沿 in 维 shard，输出 replicate。"""
    def __init__(self, in_f, out_f, group):
        super().__init__()
        self.group, tp = group, dist.get_world_size(group)
        assert in_f % tp == 0
        self.weight = nn.Parameter(torch.empty(out_f, in_f // tp).normal_(std=0.02))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):                      # x: (..., in_f // tp) sharded
        y = _ReduceFromTP.apply(F.linear(x, self.weight), self.group)
        return y + self.bias                   # bias 必须在 AllReduce 之后加
```

#warn[
  `RowParallelLinear` 的 bias *必须在 AllReduce 之后加*。放在前面就会被加 $T$ 次。这是手写 TP 最经典的静默错误：不报错、shape 全对、loss 也在降，只是慢慢偏掉。
]

校验方式：单进程构造完整的 `nn.Linear`，把 `weight` 按对应维度切给两张卡，比较 TP 版输出与单卡输出的 `torch.allclose(rtol=1e-5)`，以及两边 `weight.grad` 拼回后是否一致。数值不会逐位相同（求和顺序不同），但应在 1e-5 量级内。

=== Attention 的 TP：按 head 切

Multi-head attention 天然按 head 分块，因为不同 head 之间没有任何计算耦合：

- `q_proj` / `k_proj` / `v_proj`：column parallel（等价于按 head 切），每卡持有 `num_heads / tp` 个 head
- $Q K^T$、softmax、$P V$：*完全本地*，不需要通信（这是按 head 切的全部意义）
- `o_proj`：row parallel + 一次 AllReduce

约束：`num_heads % tp == 0`。GQA 还多一条：`num_kv_heads % tp == 0`。Llama-3 的 `num_kv_heads = 8` 就是照着 TP $<= 8$ 设的；`tp = 16` 时 KV head 不够分，只能让每 2 个 rank 共享一份 KV（复制存储），或者把 TP 调回 8。

#warn[
  QKV 通常 fuse 成一个 `(3H, H)` 的权重。切它的时候*不能*直接按 `3H` 维均分成 $T$ 段——那会把 Q 的后半和 K 的前半切进同一张卡。必须先 reshape 成 `(3, num_heads, head_dim, H)`，按 `num_heads` 维切，再 flatten 回去。改 TP 数时加载老 checkpoint 出乱码，一多半是这里错了。
]

=== 通信量：为什么 TP 必须在 NVLink 域内

每个 transformer layer 的 AllReduce 次数：

#cost-table(
  header: ([阶段], [AllReduce 次数 / 层], [张量], [触发点]),
  ([forward], [2], [$(B, S, H)$], [attention 的 `o_proj` 之后 + MLP 的 down proj 之后]),
  ([backward], [2], [$(B, S, H)$], [两个 column-parallel 入口的 `f` 反向]),
)

一层 4 次，$L$ 层一个 step 就是 $4 L$ 次 AllReduce。对比 DDP：整个 step 只有一次梯度 AllReduce（分 bucket 但总量固定）。*TP 的同步频率是 DP 的两三个数量级*。

单次 AllReduce 每张卡进出的字节数（ring 算法）：

#formula[$ "bytes" = 2 dot (T-1)/T dot B dot S dot H dot "itemsize" $]

代入 $B = 1$、$S = 4096$、$H = 8192$、bf16、$T = 4$：$B S H dot 2 = 64$ MiB，乘 $2 times 3\/4$ 得约 96 MiB 每次。32 层 $times 4$ 次 $= 128$ 次，一个 step 搬约 12 GiB。这是数量级估算（假设理想 ring、忽略 kernel 启动与非连续拷贝）。

两个结论直接跟着出来：

+ *$T$ 翻倍，通信量几乎不变（$2(T-1)\/T$ 从 1.5 涨到 1.75、1.875），但每卡计算量减半*。所以通信/计算比随 $T$ 线性恶化，TP 的扩展性天生有限。
+ *这些 AllReduce 全在关键路径上*：`o_proj` 的结果 AllReduce 完才能算 LayerNorm，没法像 DDP 梯度那样和反向计算 overlap。

#warn[
  *铁律：`tp_size` $<=$ 单机 NVLink 域内的卡数。* 跨机做 TP 等于把每层两次全量激活搬上 IB/以太网，step time 会被通信吃掉大半。本书环境是 2× A100-SXM4 NVLink 直连，所以 `tp_size` 上限是 2。8 卡 A100/H100 机器上是 8。
]

=== Embedding 与 loss

vocab 大（128K+）时这两处是显存与通信的另一个大头。*vocab-parallel embedding*：把 embedding 表沿 vocab 维切成 $T$ 份，rank $i$ 持有 $["v"_i, "v"_(i+1))$ 这段。前向每张卡把落在自己区间外的 token id mask 掉、查表、把 mask 掉的行置零，然后一次 AllReduce 求和——因为每个 token 只会命中一张卡，求和等价于选择。

*parallel cross-entropy*：logits 是 $(B, S, V)$，$V = 128"K"$、$S = 4096$、bf16 时单个张量就是 1 GiB，AllGather 回完整 logits 再算 loss 非常浪费。做法是让 logits 保持沿 vocab 维 shard，只在 TP 组内交换两个标量：每个 token 的 $max$（数值稳定用）和 $sum exp$（softmax 分母）。通信量从 $O(B S V)$ 降到 $O(B S)$。Megatron 的 `vocab_parallel_cross_entropy` 就是这个；PyTorch 侧对应 `torch.distributed.tensor.parallel.loss_parallel`（配合 DTensor 用，见第 21 章）。

#warn[
  LayerNorm / RMSNorm 的 weight 在 TP 组内是 *replicate* 的，梯度必须在 TP 组内额外 AllReduce 一次，否则各 rank 的这份参数会慢慢 drift（现象只是 loss 不降，没有任何报错）。dropout 同理：replicate 区间的 mask 必须在 TP 组内一致，shard 区间的 mask 必须不同，所以 seed 要按 `rank // tp` 和 `rank` 分别设。
]

== Sequence Parallel：把 TP 没切到的那部分切掉

TP 只切了 QKV/MLP 中间那段激活。LayerNorm、Dropout、residual 这些区间的张量仍是完整的 $(B, S, H)$——每张卡都存一份完整副本，TP 完全没帮上忙。$B = 1$、$S = 32"K"$、$H = 8192$、bf16 时一层的 LN 输入就是 512 MiB，80 层光这一项就是 40 GB。

Sequence Parallel 的做法：这些区间的张量沿 *sequence 维*切成 $(B, S\/T, H)$。LayerNorm 沿 hidden 维归一化、逐 token 独立，Dropout 也逐元素，所以在 seq 切片上*直接算就是对的，零通信*。

通信随之改造：进入 TP 区间前把 $(B, S\/T, H)$ AllGather 成 $(B, S, H)$；TP 区间出来的部分和直接 ReduceScatter 回 $(B, S\/T, H)$。而 $"AllReduce" = "ReduceScatter" + "AllGather"$，所以：

#insight[
  SP 的通信量与纯 TP *完全相同*（一次 AR 换成一次 RS + 一次 AG），但把 LayerNorm / Dropout / residual 的激活显存从 $B S H$ 降到 $B S H \/ T$。零通信代价换显存，长序列训练必开。
]

代价只有实现复杂度：所有跨 token 的计算（loss 分母、grad norm、pooling）都要意识到 seq 已经被切开。PyTorch 侧对应 `SequenceParallel` 这个 `ParallelStyle`，见第 21 章。

== Pipeline Parallel：按层切 stage

把 $L$ 层切成 $P$ 段，每段一张卡（或一个 TP 组）。stage 之间只在边界传一个激活张量，通信量与层数无关——这是 PP 唯一的优点，也是它能跨机的原因。代价是流水线气泡。把一个 global batch 拆成 $M$ 个 micro-batch 排流水，前向记 $t_f$、反向记 $t_b$：每个 stage 都要等前面 $P-1$ 个 stage 先把第一个 micro-batch 传过来，也要等后面 $P-1$ 个 stage 把最后一个 micro-batch 的梯度传回来，这段等待是 $(P-1)(t_f + t_b)$；总时长是 $(M + P - 1)(t_f + t_b)$。

#formula[$ "bubble" = ((P-1)(t_f + t_b)) / ((M + P - 1)(t_f + t_b)) = (P-1) / (M + P - 1) $]

$P = 4$、$M = 8$ 是 $3\/11 approx 27.3%$，$M = 32$ 时是 $3\/35 approx 8.6%$——*降 bubble 唯一便宜的办法是加大 $M$*。

#warn[
  PP 有一件事任何时空图都不会告诉你：*autograd 图在 `send`/`recv` 处断开*。loss 只在最后一个 stage 上存在，stage 0 上调 `loss.backward()` 是不可能的。每个 stage 必须手工把 recv 到的激活 `detach().requires_grad_(True)` 变成新 leaf，反向时用 `torch.autograd.backward(y, grad_y)` 从下游给的梯度接着往回传，再把 `x.grad` 发给上游。用 `torch.distributed.pipelining` 时这些由框架代劳，但面试会问。
]

=== GPipe：所有 F 完再所有 B

#figure(
  align(center, pipeline-schedule(stages: 4, schedule: (
    (("F", 8), ("_", 6), ("B", 8)),
    (("_", 1), ("F", 8), ("_", 4), ("B", 8), ("_", 1)),
    (("_", 2), ("F", 8), ("_", 2), ("B", 8), ("_", 2)),
    (("_", 3), ("F", 8), ("B", 8), ("_", 3)),
  ), cell: 0.5, title: "GPipe: P=4, M=8")),
  caption: [GPipe：每个 stage 把 8 个 micro-batch 的前向全部做完再开始反向。空闲集中在中间一整块，bubble $= 3\/11 approx 27.3%$。stage 0 必须同时持有 8 份激活。图中取 $t_f = t_b$，每个 B 格是一次完整的 backward（真实比例约 $t_b approx 2 t_f$，不影响 bubble 公式）。],
) <fig-gpipe>

问题在显存：stage 0 在第一次反向之前已经跑完 $M$ 个前向，$M$ 份激活全部滞留。于是"加大 $M$ 降 bubble"和"激活爆显存"被绑在了一起。

=== 1F1B：稳定期一前一后

#figure(
  align(center, pipeline-schedule(stages: 4, schedule: (
    (("F", 4), ("_", 3), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1),
     ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("_", 1), ("B", 1), ("_", 1),
     ("B", 1), ("_", 1), ("B", 1)),
    (("_", 1), ("F", 3), ("_", 2), ("B", 1), ("F", 1), ("B", 1), ("F", 1),
     ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1),
     ("_", 1), ("B", 1), ("_", 1), ("B", 1), ("_", 1)),
    (("_", 2), ("F", 2), ("_", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1),
     ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1),
     ("F", 1), ("B", 1), ("_", 1), ("B", 1), ("_", 2)),
    (("_", 3), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1),
     ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1), ("B", 1), ("F", 1),
     ("B", 1), ("_", 3)),
  ), cell: 0.5, title: "1F1B: P=4, M=8")),
  caption: [1F1B：warm-up 阶段 stage $s$ 先做 $P-s$ 个前向，之后进入"一前一后"的稳定期，最后 cool-down 排空。总时长与 GPipe *完全相同*（22 个单位，bubble 仍是 27.3%），空闲只是被打散了。],
) <fig-1f1b>

#insight[
  *1F1B 的收益不是 bubble——它和 GPipe 的 bubble 一模一样。* 收益是激活峰值：stage $s$ 在第一次反向前只跑了 $P - s$ 个前向，所以峰值从 $M$ 份降到 $P - s$ 份（$P=4, M=8$ 时 stage 0 是 4 份而不是 8 份）。这解开了"降 bubble"与"爆显存"的耦合，让你敢把 $M$ 开到 $4P$ 以上。这是高频题，答错的人极多。
]

=== interleaved 1F1B：每卡持有多个非连续 stage

把模型切成 $v P$ 段，每张卡持有 $v$ 段（rank $r$ 持有第 $r, r+P, r+2P, ...$ 段）。micro-batch 在环上绕 $v$ 圈，每段的计算量变成 $1\/v$，等效于把流水线拉长而 $t_f$ 变小：

#formula[$ "bubble"_"interleaved" = (P-1) / (v M + P - 1) $]

$P = 4$、$M = 8$、$v = 2$：$3\/19 approx 15.8%$，比 $v=1$ 的 27.3% 差不多减半。代价是 P2P 通信次数 $times v$（每个 micro-batch 每圈都要过一遍所有 stage 边界），而且激活峰值比 1F1B *更高*而不是更低。所以先把 $M$ 加满，$M$ 被 global batch 卡住之后才动 $v$。

=== zero-bubble / DualPipe

一次 backward 其实是两件事：算 $partial L\/partial X$（记 B，会阻塞上游，必须尽早做）和算 $partial L\/partial W$（记 W，不阻塞任何人）。把 W 拆出来塞进 cool-down 的空隙，bubble 能再降一半左右——这是 zero-bubble 的全部思路。填不满是因为 warm-up 阶段还没有任何梯度回来，手上没有 W 可做。DeepSeek-V3 的 DualPipe 更激进：从流水线两端同时注入 micro-batch，bubble 系数从 $P-1$ 降到 $P\/2-1$，代价是 2 份权重副本。

=== PP 的工程难点

+ *$M >= 4P$ 才划算*。$M = P$ 时 bubble 高达 $(P-1)\/(2P-1) approx 50%$。而 $M$ 的上限是 global batch size 除以 micro batch size，所以 PP 深度受 batch 大小约束。
+ *stage 负载必须均衡*。流水线跑在最慢那一段的速度上，不均衡是每个 micro-batch 都要付的减速，与 bubble 相乘而不是被 bubble 吸收。主要不均衡来源是首尾：embedding 很便宜，而 `lm_head` 要做 $(B, S, V)$ 的 GEMM 加 cross-entropy，$V = 128"K"$ 时相当于好几个 transformer block。按耗时分层而不是按层数平均分，或者给首尾 stage 少放几层。
+ *P2P 的死锁*。1F1B 稳定期里 stage $s$ 要往下发激活、stage $s+1$ 同时要往上发梯度，两边都"先 send 后 recv"就双向卡死。解法是 `batch_isend_irecv` 把这一步所有 send/recv 先 post 再一起 wait。
+ *只有最后一个 stage 有 loss*。dataloader 只在首尾两端起作用（stage 0 要 token、末 stage 要 label），而同一个 PP 组内所有 rank 必须拿到同一个样本。tied embedding（首尾共享）的梯度还要在首尾两个 rank 之间单独 AllReduce 一次。
=== PyTorch 原生 `torch.distributed.pipelining`（2.4+）

三个概念：`pipeline()` 把模型按 `split_spec` 切成 stage（基于 `torch.export` 追踪）、`PipelineStage` 描述某一段在拓扑里的位置、`Schedule*` 负责排程并驱动 `step()`。

```python
# torchrun --nproc-per-node 2 pp_demo.py    （2 卡 = 2 stage）
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

class Toy(nn.Module):
    def __init__(self, h=512, n=4):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(h, h) for _ in range(n)])
    def forward(self, x): return self.layers(x)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
rank = dist.get_rank()                    # 单机时与 local_rank 相同；stage 编号用全局 rank

model = Toy().to(dev)
n_mb = 8                                  # micro-batch 数
mb_x = torch.randn(4, 512, device=dev)    # 一个 micro-batch 的样例输入

pipe = pipeline(model, mb_args=(mb_x,),
                split_spec={"layers.2": SplitPoint.BEGINNING})
stage = pipe.build_stage(rank, dev)       # 本 rank 那一段
sched = ScheduleGPipe(stage, n_microbatches=n_mb, loss_fn=nn.MSELoss())

x = torch.randn(4 * n_mb, 512, device=dev)   # 完整 batch，框架自己切 micro-batch
y = torch.randn(4 * n_mb, 512, device=dev)
opt = torch.optim.AdamW(stage.submod.parameters(), lr=1e-3)
opt.zero_grad()

if rank == 0:
    sched.step(x)                  # 第一个 stage 只喂输入
else:
    losses = []
    out = sched.step(target=y, losses=losses)   # 最后一个 stage 提供 label
    print("loss", torch.stack(losses).mean().item())
opt.step()
dist.destroy_process_group()
```

把 `ScheduleGPipe` 换成 `Schedule1F1B` 就切换排程，接口不变；`ScheduleInterleaved1F1B` 需要一个 rank 持有多个 stage（传 stage 列表）。

#note[
  `pipelining` 的 API 在 2.4 到 2.10 之间仍有调整（`build_stage` / `PipelineStage` 两条构造路径、`scale_grads` 等参数），确切签名以对应版本文档为准。稳定的部分是三个概念本身：split、stage、schedule。生产上 Megatron-LM / torchtitan 的实现更成熟。
]

== 组合并行：3D / 4D 的维度排布

排布原则只有一条：*通信最重的维度放在最内层（rank 相邻、同 NVLink 域），最轻的放最外层（可跨机）*。按通信重量排序是 TP > EP > CP > PP > DP，所以标准顺序是

#formula[$ "rank" = "dp" dot (P dot T) + "pp" dot T + "tp" $]

也就是 TP 变化最快、DP 最慢。给定 world size 16、$(d, p, t) = (2, 2, 4)$：

#figure(
  align(center, topology-grid(rows: 4, cols: 4, cell: 0.9,
    groups: ((0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1)),
    group-labels: ((0, "dp=0"), (1, "dp=1")),
    title: "world=16, dp=2, pp=2, tp=4")),
  caption: [每一行 4 张卡是一个 TP 组，必须落在同一台机器的 NVLink 域内（$G_0..G_3$ 一台，$G_4..G_7$ 另一台）。上两行是 pp stage 0、下两行是 stage 1。DP 组是跨行取同一列，例如 ${G_0, G_4}$；PP 组是 ${G_0, G_8}$，跨机走 P2P。],
) <fig-3d-mesh>

具体映射（`tp` 最快）：

#table(
  columns: (auto, auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (center, center, center, center, left),
  [*rank*], [*dp*], [*pp*], [*tp*], [*所属组*],
  [0–3], [0], [0], [0–3], [TP 组 {0,1,2,3}，同机],
  [4–7], [0], [1], [0–3], [TP 组 {4,5,6,7}，同机],
  [8–11], [1], [0], [0–3], [DP 组 {0,8} / {1,9} / …],
  [12–15], [1], [1], [0–3], [PP 组 {0,4} / {8,12} / …],
)

#warn[
  手写这套映射最常见的 bug：pipeline 邻居是 rank $plus.minus T$（这里 $plus.minus 4$），不是 $plus.minus 1$。而且 `dist.send/recv` 即使传了子进程组，`src`/`dst` 参数要的仍是*全局* rank，用 `dist.get_global_rank(group, i)` 翻译。算错了如果那个 rank 恰好存在，表现是挂住而不是报错。
]

用 `DeviceMesh` 描述同一件事只要一行，而且不用自己算 rank（见第 21 章）：

```python
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh("cuda", (2, 2, 4), mesh_dim_names=("dp", "pp", "tp"))
mesh["tp"]          # 本 rank 所在的 TP 子 group（相邻 4 个 rank）
mesh["dp"]          # 本 rank 所在的 DP 子 group（stride = 8）
```

=== 怎么选并行配置

给定模型大小和集群，按这个顺序走，每一步只在上一步不够时才启用：

+ *先算显存*。参数 + 梯度 + Adam 状态在混合精度下约 $16 P$ 字节（bf16 参数与梯度各 $2P$、fp32 master weight 与两个 moment 各 $4P$）。单卡放不下就上 FSDP，看 $16 P \/ N + "激活"$ 是否放得下。
+ *能只用 FSDP 就只用 FSDP*。通信最轻、代码最简单、checkpoint 最省事。
+ *单层放不下，或激活显存爆，才加 TP*，且 `tp_size` 不超过单机卡数（本书环境 2）。优先 $T = 8$（8 卡机）；$T > 8$ 几乎总是亏的。开 TP 就顺手开 SP。
+ *层数多到 TP + FSDP 仍装不下，才加 PP*，让 PP 跨机。检查 $M >= 4P$ 是否成立、stage 是否均衡。
+ *剩下的卡全给 DP/FSDP*（$"dp" = "world" \/ (t dot p)$），然后跑 10 个 step 看 step time 与显存峰值，用 profiler 看通信占比（见第 11 章）：TP 通信占比超过 20% 说明 $T$ 太大或跨了 NVLink 域。

本书的 2× A100 环境能真跑的只有 $(d, p, t) in {(2,1,1), (1,2,1), (1,1,2)}$ 这几种组合，但概念和调试手段与 512 卡完全一样。

== 面试考点

#interview[
  *Q1*：Megatron 的 MLP 为什么是 column-then-row？反过来行不行？

  A：行是行，但通信翻倍。column parallel 的输出沿 hidden 维 shard，而 GELU 是逐元素的，在 shard 上直接算就对，所以两个 GEMM 之间零通信，只在 row parallel 之后把部分和 AllReduce 一次。反过来的话，row parallel 的输出是部分和，而 $"GELU"(a+b) != "GELU"(a) + "GELU"(b)$，必须先 AllReduce 才能过激活函数；后面 column parallel 的输出又是 shard 的，下一层 LayerNorm 需要完整 hidden，还得再 AllReduce。语义等价，通信 2×。
]

#interview[
  *Q2*：TP=8 的一个 transformer layer，前向反向一共几次 AllReduce？

  A：前向 2 次（attention 的 `o_proj` 之后、MLP 的 down proj 之后），反向 2 次（两个 column-parallel 入口处对 $partial L\/partial X$ 求和），一层共 4 次。32 层就是 128 次，而 DDP 整个 step 只有 1 次梯度 AllReduce——TP 的同步频率高两三个数量级，这就是它必须待在 NVLink 域里的原因。
]

#interview[
  *Q3*：为什么 `tp_size` 不能超过单机卡数？$T$ 从 8 加到 16 会发生什么？

  A：单次 AllReduce 的字节数是 $2(T-1)\/T dot B S H dot b$，$T$ 从 8 到 16 只从 $1.75$ 涨到 $1.875$ 倍——几乎不变，但每卡计算量减半，通信/计算比翻倍。同时这些 AllReduce 在关键路径上无法与计算 overlap。跨机之后带宽掉一个数量级，step time 直接被通信吃掉。所以 `tp_size` $<=$ NVLink 域大小，实践上 $T <= 8$。
]

#interview[
  *Q4*：Sequence Parallel 的通信量和纯 TP 一样，那它省了什么？

  A：省激活显存。TP 只切了 QKV/MLP 中间那段，LayerNorm / Dropout / residual 区间的张量仍是完整 $(B,S,H)$。SP 把这些区间沿 seq 维切成 $(B, S\/T, H)$——LN 沿 hidden 归一化、逐 token 独立，所以在 seq 切片上零通信直接算。原来的一次 AllReduce 拆成 ReduceScatter + AllGather，二者之和等于 AllReduce，通信量不变，但这部分激活降到 $1/T$。$S = 32"K"$ 时能省 GB 级。
]

#interview[
  *Q5*：GPipe 和 1F1B 的 bubble 分别是多少？1F1B 到底赢在哪？

  A：*完全相同*，都是 $(P-1)\/(M+P-1)$（$P=4, M=8$ 时 27.3%）。1F1B 赢的是激活峰值：stage $s$ 在第一次反向前只跑 $P-s$ 个前向，峰值从 $M$ 份降到 $P-s$ 份。为什么这很重要：降 bubble 最便宜的手段就是加大 $M$，而 GPipe 的显存正比于 $M$，把两件事绑死了；1F1B 解开耦合，才让 $M >= 4P$ 变得可行。
]

#interview[
  *Q6*：interleaved 1F1B 把 bubble 降到多少？代价是什么？

  A：$(P-1)\/(v M + P - 1)$，每卡持有 $v$ 个非连续 stage。$P=4, M=8, v=2$ 时从 27.3% 降到 15.8%。代价：micro-batch 要在环上绕 $v$ 圈，P2P 通信次数 $times v$；激活峰值比普通 1F1B 更高。所以先加 $M$（不花显存），$M$ 被 global batch 卡住了才动 $v$。
]

#interview[
  *Q7*：PP 里为什么不能直接 `loss.backward()`？

  A：autograd 图在 `send`/`recv` 处断成 $P$ 段，loss 只在最后一个 stage 上，stage 0 上没有任何从 loss 连回自己权重的路径。做法是手工提供每段的起点：把 recv 到的激活 `detach().requires_grad_(True)` 变成新 leaf，反向时 recv 下游给的 `grad_output`，调 `torch.autograd.backward(y, grad_y)`（不是 `y.backward()`，非标量会抛异常），它会同时填好本段的 `weight.grad` 和 `x.grad`，再把 `x.grad` 发给上游。第一个 stage 的输入*不要* `requires_grad_`，它常是 int64 的 token id。
]

#interview[
  *Q8*：PP 的 stage 怎么切？为什么不能按层数平均分？

  A：按耗时分。embedding 很便宜，`lm_head` 要做 $(B,S,V)$ 的 GEMM 加 cross-entropy，$V=128"K"$ 时相当于好几个 transformer block。流水线跑在最慢那段的速度上，所以不均衡是每个 micro-batch 都要付的减速，与 bubble 相乘而不是被吸收。常见做法：给首尾 stage 少放几层，或把 loss 单独放一个 virtual chunk。另外大模型不能"先建完整模型再切"，分层方案必须在分配权重之前就定下来。
]

#interview[
  *Q9*：TP、PP、DP 三种并行在集群里怎么摆？

  A：通信最重的放最内层。顺序 `rank = dp*(P*T) + pp*T + tp`，TP 变化最快、落在连续 rank（同 NVLink 域），因为它每层搬两次完整激活；PP 给最大 stride 让它跨机，因为它只在 stage 边界传一次激活、而且是 P2P 不占整个 fabric；DP/FSDP 最外层，它对带宽最宽容且能与反向计算 overlap。所以 pipeline 邻居是 rank $plus.minus T$ 而不是 $plus.minus 1$。
]

#interview[
  *Q10*：给一个 70B 模型和 64 张 A100-80G，你怎么配并行？

  A：先算显存：$70"B" times 16$ 字节约 1.1 TB 状态，除以 64 卡是 17.5 GB，FSDP 装得下参数部分。但单层 $H = 8192$ 的权重和激活加上通信 buffer 会紧，且激活不切。我会从 `tp=8`（机内 NVLink，顺带开 SP 把 LN 激活切掉）+ `fsdp=8` 起步，先不上 PP——64 卡这个规模 PP 的 bubble 和实现复杂度都不值得。跑 10 步看显存峰值和通信占比：TP 通信超过 20% 就把 `tp` 降到 4；显存还是不够就加 activation checkpointing（比加 PP 便宜得多）。只有层数多到 TP+FSDP 也装不下、且卡数上千时才引入 PP，并保证 $M >= 4P$。
]
