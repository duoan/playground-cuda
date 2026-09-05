#import "../template.typ": *

= DeviceMesh 与 DTensor

第 20 章的 TP 是"命令式"的：每个 op 前后自己插 AllReduce，每种切法写一个 module。这套写法能跑，但它把*张量的分布信息*藏在了类名里——`ColumnParallelLinear` 的输出是 shard 的、`RowParallelLinear` 的输入必须是 shard 的，这些约束只存在于文档和你的脑子里，编译器和框架都不知道。DTensor 反过来：把分布信息写进张量本身，让通信由"两个张量的分布不一致"自动推导出来。这是 PyTorch 分布式在 2.x 时代的主线，FSDP2、TP、DCP、`torch.compile` 全部落在这一层上。

== 手写并行的痛点

Megatron 风格的代码有四个具体问题：

+ *组合并行时代码爆炸*。TP 一套 module，加上 FSDP 又一套包装，再加 PP 又一层。$k$ 种并行两两交互，要考虑的组合是 $O(k^2)$ 而不是 $O(k)$。"这个 tensor 现在是 TP shard 还是 FSDP shard 还是两者都是"，只能靠人脑追。
+ *checkpoint 与并行度绑死*。权重按 `tp=8` 切成 8 份存下来，换成 `tp=4` 就读不回来——因为文件里只有裸的 local tensor，没有"我是第几片、总共几片、沿哪一维切"这些元信息。
+ *每个 op 都要手写通信*。加一个新算子（比如某种 fused norm）就要重新想一遍它的输入输出该是什么分布、要不要补通信。
+ *`torch.compile` 看不懂*。手写的 `dist.all_reduce` 对 Dynamo 是一个黑盒副作用，图会在这里断开（见第 12 章的 graph break）。

DTensor 要解决的就是"分布信息缺失"这一个根因。

== DeviceMesh：给 world 的每一维起名字

`DeviceMesh` 把 world 的 $N$ 个 rank 组织成一个 $n$ 维网格，每一维起一个名字，然后你就可以按名字取子通信组，不用自己算 rank 了。

```python
# torchrun --nproc-per-node 8 mesh_demo.py
import torch
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
# mesh.mesh 是 arange(8).reshape(2, 4)：
#   dp=0 -> [0, 1, 2, 3]
#   dp=1 -> [4, 5, 6, 7]

mesh["tp"]                  # 本 rank 所在的 TP 子 mesh（rank 3 → [0,1,2,3]）
mesh["dp"]                  # 本 rank 所在的 DP 子 mesh（rank 3 → [3, 7]）
mesh["tp"].get_group()      # 拿到底层 ProcessGroup，可以直接喂 dist.all_reduce
mesh["tp"].get_local_rank() # 本 rank 在 TP 维的坐标
mesh.ndim, mesh.shape       # 2, (2, 4)
```

rank 到坐标的映射是*行优先*：`arange(world).reshape(mesh_shape)`，所以*最后一维变化最快*，最后一维上相邻的坐标对应连续的 global rank。

#insight[
  连续的 global rank 通常落在同一台机器（同一个 NVLink 域），所以*最后一维必须留给通信最重的并行*。标准写法是 `("dp", "pp", "tp")` 或 `("dp", "tp")`——TP 放最后。把 mesh 写成 `("tp", "dp")` 不会报错，也能算出正确结果，但 TP 的 AllReduce 会跨机，step time 差几倍。这是 DeviceMesh 唯一的性能陷阱。
]

#figure(
  align(center, topology-grid(rows: 2, cols: 4, cell: 0.9,
    groups: ((0, 0, 0, 0), (1, 1, 1, 1)),
    group-labels: ((0, "dp=0"), (1, "dp=1")),
    title: "init_device_mesh(\"cuda\", (2, 4), (\"dp\", \"tp\"))")),
  caption: [每一行是一个 TP 组（rank 连续，走机内 NVLink）；每一列是一个 DP 组（stride $= 4$，跨机走 IB）。`mesh["tp"]` 取行、`mesh["dp"]` 取列。],
) <fig-mesh-2d>

== DTensor：local tensor + mesh + placements

一个 `DTensor` 逻辑上是一个全局张量，物理上由每个 rank 手里的一块 local tensor 拼成。它携带三样东西：

- `_local_tensor`：本 rank 实际持有的那块内存（用 `to_local()` 取）
- `device_mesh`：分布在哪个 mesh 上
- `placements`：一个长度等于 `mesh.ndim` 的元组，*逐维*说明这个张量沿该 mesh 维怎么分布

三种 placement：

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*placement*], [*含义*], [*local shape（mesh 维大小 $= n$）*],
  [`Shard(d)`], [沿张量的第 $d$ 维切给这一 mesh 维上的 $n$ 个 rank], [第 $d$ 维变成 $1\/n$],
  [`Replicate()`], [每个 rank 持有完整副本，数值相同], [与全局 shape 相同],
  [`Partial(op)`], [每个 rank 持有*部分和*，还没规约；全局值 $=$ 各 rank 按 `op` 规约的结果], [与全局 shape 相同],
)

`Shard` 和 `Replicate` 是直觉的，`Partial` 是这套设计里真正关键的一环。

#insight[
  `Partial` 让"需要 AllReduce"这件事变成一个*可以延迟兑现的标记*。第 20 章的 row-parallel matmul 输出是部分和，手写版本必须立刻 AllReduce 才敢往下传；DTensor 里它只是 `Partial("sum")`，什么时候必须变成 `Replicate` 或 `Shard`，由后面的算子决定。如果下一步正好是 ReduceScatter 能满足的分布（SP 场景），就一次 ReduceScatter 解决，省掉一次 AllGather；如果连着几个逐元素线性算子，规约还能继续往后推。手写代码做不到这种"通信重排"，因为信息不在张量里。
]

== placement 转换与 collective 的对应关系

这张表是全章的核心。DTensor 不需要为每个算子手写通信，它只需要知道"当前 placement"和"算子要求的 placement"，剩下的是一次查表：

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left),
  [*从*], [*到*], [*collective*], [*为什么*],
  [`Shard(i)`], [`Replicate()`], [AllGather], [每个 rank 缺别人的切片，要全部收齐],
  [`Replicate()`], [`Shard(i)`], [无通信（本地切片）], [完整数据已在手上，按坐标切一刀丢掉其余],
  [`Partial`], [`Replicate()`], [AllReduce], [部分和求和，且结果每个 rank 都要],
  [`Partial`], [`Shard(i)`], [ReduceScatter], [求和 + 只保留自己那一段，比 AllReduce 更省],
  [`Shard(i)`], [`Shard(j)`], [AllToAll], [沿 $i$ 维收拢、同时沿 $j$ 维散开],
  [`Replicate()`], [`Partial`], [本地缩放（除以 $n$）], [反向里出现；不是通信],
)

三个能直接背的推论：

+ *`AllReduce = ReduceScatter + AllGather`* 在这张表里是显式的：`Partial -> Replicate` 等于 `Partial -> Shard(i)` 再 `Shard(i) -> Replicate`。这正是 Sequence Parallel 省显存不加通信的原因（第 20 章）。
+ *`Replicate -> Shard` 是免费的*，所以"能 shard 就早点 shard"。
+ *TP 的一整套通信不需要任何新概念*：column-parallel 的输入要 `Replicate`、输出是 `Shard(-1)`；row-parallel 的输入要 `Shard(-1)`、输出是 `Partial`。把这两条接起来，中间那一步是 `Shard(-1) -> Shard(-1)`（无通信），末尾 `Partial -> Replicate` 就是那唯一一次 AllReduce。第 20 章手写的 `f` / `g` 两个 Function，在 DTensor 里是查表的结果。

=== matmul 的三条规则就够解释整个 TP

转换表说的是"怎么改分布"，还差一半：*算子怎么从输入分布推出输出分布*。以 $C = A B$ 为例，只看被切的那一维落在哪：

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left),
  [*A 的分布*], [*B 的分布*], [*C 的分布*], [*对应什么*],
  [`Shard(0)`], [`Replicate`], [`Shard(0)`], [沿 batch/token 维切，DP 与 SP 的日常],
  [`Replicate`], [`Shard(1)`], [`Shard(1)`], [column parallel：切输出维],
  [`Shard(1)`], [`Shard(0)`], [`Partial("sum")`], [row parallel：切了收缩维，结果是部分和],
)

第三条是全章最值得记的一句：*被切的维如果正好是 matmul 的收缩维（$A$ 的列 = $B$ 的行），结果一定是 `Partial`*。所有"什么时候需要 AllReduce"的问题都归到这一条——不是因为 TP 有个约定，而是因为求和被分到了不同的卡上。

#note[
  反向传播的 placement 是前向的对偶：前向 `Replicate` 的输入，反向的梯度是 `Partial`（每个 rank 算出一份部分梯度，需要求和）；前向 `Partial -> Replicate` 的输出，反向梯度是 `Replicate`（直接透传）。这恰好复现了手写版本里 `f` 反向 AllReduce、`g` 反向恒等的结构——不是巧合，是同一条规则的两种写法。
]

== 用 DTensor 表达 TP

`parallelize_module` 接受一个 "module 名 → `ParallelStyle`" 的字典，把对应 module 的参数换成 DTensor 并注册输入/输出的 placement 转换。

```python
# torchrun --nproc-per-node 2 tp_dtensor.py   （2× A100，tp=2）
import torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
)

class MLP(nn.Module):
    def __init__(self, h=1024):
        super().__init__()
        self.up = nn.Linear(h, 4 * h)
        self.down = nn.Linear(4 * h, h)
    def forward(self, x): return self.down(torch.nn.functional.gelu(self.up(x)))

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("tp",))

model = MLP().cuda()
model = parallelize_module(model, mesh["tp"], {
    "up":   ColwiseParallel(),      # weight → Shard(0)，输出 Shard(-1)
    "down": RowwiseParallel(),      # weight → Shard(1)，输入 Shard(-1)，输出 Replicate
})

# up.weight 现在是 DTensor：全局 shape (4h, h)，local shape (4h/tp, h)
w = model.up.weight
print(rank, type(w).__name__, tuple(w.shape), tuple(w.to_local().shape), w.placements)

x = torch.randn(8, 1024, device="cuda")   # 普通 tensor，被当作 Replicate 处理
y = model(x)                              # 内部只有一次 AllReduce
y.sum().backward()
dist.destroy_process_group()
```

生成的通信与第 20 章手写版本*完全等价*：`ColwiseParallel` 默认 `input_layouts=Replicate()`、`output_layouts=Shard(-1)`；GELU 在 `Shard(-1)` 上逐元素做，不触发转换；`RowwiseParallel` 期望输入 `Shard(-1)`（正好匹配，无通信），输出 `Partial` 再按 `output_layouts=Replicate()` 落地成一次 AllReduce。区别只是这些通信是查表推出来的，不是你写出来的。

几个常用的 style：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*style*], [*参数怎么切*], [*默认 in / out placement*],
  [`ColwiseParallel`], [`weight` 沿 dim 0（输出维）切], [in `Replicate` / out `Shard(-1)`],
  [`RowwiseParallel`], [`weight` 沿 dim 1（输入维）切], [in `Shard(-1)` / out `Replicate`],
  [`SequenceParallel`], [`LayerNorm` / `RMSNorm` 的参数 `Replicate`], [in/out `Shard(1)`（沿 seq 维）],
  [`PrepareModuleInput`], [不切参数，只转换输入], [由 `desired_input_layouts` 指定],
  [`loss_parallel`], [context manager，让 CE 在 vocab-shard 的 logits 上算], [logits `Shard(-1)`],
)

开 SP 时把 norm 层标成 `SequenceParallel()`，再用 `ColwiseParallel(input_layouts=Shard(1))` 告诉 attention 入口"输入是沿 seq 切的"——转换表自动把它变成 AllGather，出口的 `RowwiseParallel(output_layouts=Shard(1))` 自动变成 ReduceScatter。第 20 章那一整段手写 AG/RS 改造，在这里是两个参数。

#warn[
  `use_local_output=True`（`Colwise`/`Rowwise` 的默认值）会把模块输出转回普通 `Tensor`，方便和不感知 DTensor 的代码对接，但也*丢掉了分布信息*——后面的算子无法再自动推导通信。要串联多个 TP 模块并让 DTensor 全程接管，中间层要传 `use_local_output=False`。`SequenceParallel` 的默认值本来就是 `False`。
]

== FSDP2 就是 mesh 上的 `Shard(0)`

`fully_shard`（FSDP2，torch 2.6+ 稳定）把每个参数换成 `placements=(Shard(0),)` 的 DTensor，前向按 module AllGather 成 `Replicate`、用完再丢掉，梯度以 `Partial -> Shard(0)`（ReduceScatter）落地。所以 ZeRO-3 在这套语言里不是一个特殊机制，只是"参数沿 DP 维 `Shard(0)`"这一个 placement。

于是 TP + FSDP 的组合变成了在 2D mesh 上给参数写两维 placement：

```python
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

for layer in model.layers:                      # 先 TP：沿 tp 维切
    parallelize_module(layer, mesh["tp"], {
        "attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(),
        "attn.wv": ColwiseParallel(), "attn.wo": RowwiseParallel(),
        "mlp.up":  ColwiseParallel(), "mlp.down": RowwiseParallel(),
    })
    fully_shard(layer, mesh=mesh["dp"])         # 再 FSDP：沿 dp 维切
fully_shard(model, mesh=mesh["dp"])

# attn.wq.weight.placements == (Shard(0), Shard(0))
#   第 0 个是 dp 维（FSDP 沿参数 dim 0 切），第 1 个是 tp 维（Colwise 沿 dim 0 切）
```

`placements` 的第 $k$ 项对应 `mesh` 的第 $k$ 维，两维互不干扰。这就是"组合并行变简单"的全部：*不需要为每种组合写新类，只需要多写一维 placement*。

#note[
  FSDP2 与 TP 组合必须用 `fully_shard`，老的 `FSDP` 包装类（FSDP1）与 TP 混用有已知问题。判断代码是哪一代：FSDP1 是 `FSDP(model, ...)` 返回包装对象，FSDP2 是 `fully_shard(model)` 原地改造。
]

== 手工构造与转换 DTensor

四个 API 覆盖绝大多数用法，语义上的区别是"输入是全局张量还是本地张量"：

```python
import torch
from torch.distributed.tensor import (
    DTensor, Shard, Replicate, Partial, distribute_tensor,
)

# 1) distribute_tensor: 输入是【全局】张量，由 src rank 广播/切分下去
big = torch.randn(1024, 512)                       # 每个 rank 上都有（或只有 rank 0 有效）
dt = distribute_tensor(big, mesh["tp"], [Shard(0)])
dt.shape, dt.to_local().shape                      # (1024,512) / (256,512) at tp=4

# 2) from_local: 输入是【本地】张量，你声称它已经是某个分布的一片
#    不做任何通信，也不校验（run_check=False 是默认），全局 shape 由 local shape 推出
local = torch.randn(256, 512)
dt2 = DTensor.from_local(local, mesh["tp"], [Shard(0)])
dt2.shape                                          # (1024, 512)

# 3) redistribute: 改 placement，按转换表插 collective
dt3 = dt.redistribute(mesh["tp"], [Replicate()])   # AllGather
dt4 = dt3.redistribute(mesh["tp"], [Shard(1)])     # 本地切片，无通信

# 4) to_local: 拿回本 rank 的那块内存，脱离 DTensor 体系
plain = dt.to_local()                              # 普通 Tensor，autograd 仍能穿过
```

三个容易踩的语义细节：

- `distribute_tensor` 要求*所有 rank 传进来的全局张量在逻辑上是同一个*（默认 `src_data_rank=0`，从 rank 0 分发），常用于初始化；`from_local` 相反，它信任你手里的 local 片，不通信、不校验，用错了会静默算错。
- `redistribute` 是可微的：反向会自动插入对偶的 collective（`AllGather` 的反向是 `ReduceScatter`）。
- `to_local()` 之后你就自己负责了。想让梯度以特定 placement 回来，用 `to_local(grad_placements=[...])` 显式声明。

#warn[
  DTensor 上的所有操作都是 *SPMD 的*：每个 rank 必须执行同一串 DTensor 操作。在 `if rank == 0:` 里 `redistribute` 或者对 DTensor 做 `print(dt)`（会触发 AllGather）都会导致部分 rank 参与 collective、部分不参与，直接 hang。排查见第 22 章。
]

== 收益与代价

收益：

+ *声明式*：通信从"手写"变成"由 placement 不一致推导"，加新算子不用重新想通信。
+ *checkpoint 可 resharding*：DTensor 自带 mesh 与 placement，`torch.distributed.checkpoint` 存的是"这块内存是全局张量的哪一段"，所以 `tp=8` 存的 checkpoint 能用 `tp=4` 加载（见第 22 章）。这是 DTensor 最实际的好处。
+ *组合并行是加维度而不是加代码*：$O(k)$ 而不是 $O(k^2)$。
+ *`torch.compile` 友好*：collective 以正规算子的形式进图，Inductor 能把通信与计算重排、overlap，而不是遇到 `dist.all_reduce` 就 graph break。

代价：

+ *placement 推导有 CPU 侧开销*：每个 op 都要查 sharding 规则、可能构造新的 DTensor 对象。小模型/小 batch 时这部分 Python overhead 可能变得可见，`torch.compile` 能吃掉大部分。
+ *调试栈更深*：一个报错要穿过 DTensor dispatch、sharding propagation、DeviceMesh 三层，而分不清全局 shape 与 local shape 是最常见的困惑。
+ *算子覆盖度仍在补齐*：自定义算子和冷门算子可能没有 sharding rule，报 "operator does not have a sharding strategy"。解法是自己注册规则，或者在这一段 `to_local()` 手写。

调试 DTensor 的第一步固定是把这三样一起打印：

```python
def dbg(name, t):
    if isinstance(t, DTensor):
        print(f"[{dist.get_rank()}] {name} global={tuple(t.shape)} "
              f"local={tuple(t.to_local().shape)} {t.placements} {t.device_mesh}")
    else:
        print(f"[{dist.get_rank()}] {name} plain={tuple(t.shape)}")
```

`global` 与 `local` 相同而 placement 是 `Shard(...)`，说明某处 `from_local` 的声明和实际数据不符；placement 长期停在 `Partial` 说明规约被推迟到了你没预期的位置。

torchtitan 是 DTensor path 的官方参考实现（FSDP2 + TP + PP + DCP 的组装方式），要看生产写法直接读它。

== 面试考点

#interview[
  *Q1*：`DeviceMesh` 解决什么问题？为什么 TP 要放在 mesh 的最后一维？

  A：它把 world 组织成命名的 $n$ 维网格，`mesh["tp"]` 直接给出子通信组，不用手算 `rank = dp*(P*T) + pp*T + tp` 这类映射。rank 到坐标是行优先，最后一维变化最快，也就是最后一维上相邻的坐标对应*连续的 global rank*、通常在同一台机器的 NVLink 域内。TP 每层要搬两次完整激活，通信最重，必须落在这里。写成 `("tp", "dp")` 不报错但 TP 会跨机，慢几倍。
]

#interview[
  *Q2*：DTensor 由哪几部分组成？`placements` 的长度是多少？

  A：local tensor + device mesh + placements。`placements` 的长度等于 `mesh.ndim`，第 $k$ 项描述这个张量沿 mesh 第 $k$ 维怎么分布。所以 2D mesh 上 TP + FSDP 的参数是 `(Shard(0), Shard(0))`——第一项是 FSDP 沿 dp 维切，第二项是 Colwise 沿 tp 维切，互不干扰。
]

#interview[
  *Q3*：`Partial` 是什么？为什么它是关键？

  A：`Partial(op)` 表示每个 rank 手上是*部分和*，全局值等于各 rank 按 `op` 规约后的结果。它把"这里需要一次 AllReduce"从"必须立刻执行的动作"变成"可以延迟兑现的标记"。于是 row-parallel matmul 的输出可以先带着 `Partial` 往下走，等真正需要完整数值时才规约；如果下游要的是 seq-shard，一次 ReduceScatter 就够，省掉 AllGather。手写代码做不到这种重排，因为分布信息不在张量里。
]

#interview[
  *Q4*：写出 placement 转换与 collective 的对应关系。

  A：`Shard(i) -> Replicate` 是 AllGather；`Replicate -> Shard(i)` 是本地切片、无通信；`Partial -> Replicate` 是 AllReduce；`Partial -> Shard(i)` 是 ReduceScatter；`Shard(i) -> Shard(j)` 是 AllToAll。DTensor 就是靠这张表把"当前 placement"和"算子要求的 placement"之间的差异翻译成 collective。顺带能看出 $"AllReduce" = "ReduceScatter" + "AllGather"$。
]

#interview[
  *Q5*：`ColwiseParallel` + `RowwiseParallel` 生成的通信和 Megatron 手写的一样吗？

  A：一样。Colwise 的输入要 `Replicate`、输出 `Shard(-1)`；逐元素激活在 `Shard(-1)` 上不触发转换；Rowwise 期望输入 `Shard(-1)`（正好匹配，零通信），输出是 `Partial`，落地成 `Replicate` 时就是那唯一一次 AllReduce。前向一次、反向一次，与手写的 `f`/`g` 完全对应。区别只在于通信是查表推出来的还是人写的。
]

#interview[
  *Q6*：`distribute_tensor` 和 `DTensor.from_local` 有什么区别？

  A：`distribute_tensor` 的输入是*全局*张量，默认从 `src_data_rank=0` 分发并按 placement 切开，有通信；`from_local` 的输入是*本地*片，你声称它已经是某个分布的一片，不通信也不校验（`run_check` 默认 `False`），全局 shape 由 local shape 与 mesh 推出。前者用于初始化/加载，后者用于把已有的手写并行代码接进 DTensor。`from_local` 声明错了不会报错，会静默算错。
]

#interview[
  *Q7*：FSDP2 和 DTensor 什么关系？为什么它让"组合并行"变简单了？

  A：`fully_shard` 就是把参数变成 mesh 上 `Shard(0)` 的 DTensor，前向 AllGather 成 `Replicate` 用完即丢，梯度 `Partial -> Shard(0)` 即 ReduceScatter。所以 ZeRO-3 不是特殊机制，只是一个 placement。TP + FSDP 于是变成 2D mesh 上的两维 placement `(Shard(0), Shard(0))`——不需要为每种组合写新类，只需要多写一维。注意必须用 FSDP2 的 `fully_shard`，FSDP1 与 TP 混用有已知问题。
]

#interview[
  *Q8*：DTensor 的代价是什么？什么时候不该用？

  A：三点。placement 推导有 CPU 侧开销，小模型小 batch 时可能可见（`torch.compile` 能吃掉大部分）；调试栈更深，报错要穿过 dispatch / sharding propagation / mesh 三层，看 shape 时要分清全局还是 local；算子覆盖度仍在补齐，自定义算子可能没有 sharding rule，需要自己注册或局部 `to_local()` 手写。如果只是单机 DDP 或纯 FSDP 且不打算改并行度，用不上 DTensor 的表达力；一旦要组合两种以上并行、或者要求 checkpoint 能换并行度恢复，DTensor 就是当前最省事的路。
]
