#import "../template.typ": *

= torch.distributed 基础：通信原语与进程组

分布式面试的第一关不是 DDP，是"你知道 AllReduce 到底传了多少字节吗"。这一章把 `torch.distributed` 拆成三层：底下是集合通信原语（谁给谁发多少数据）、中间是进程组与 rendezvous（这些进程怎么互相找到）、上面才是 DDP / FSDP / TP 这些并行策略。原语层的通信量公式是所有并行策略成本分析的地基，也是被问到最多的地方——第 18、19、20 章的每一个"通信量 = 几倍参数量"的结论，都是从这一章的几条公式推出来的。

== 为什么要分布式：两道墙

*显存墙*。混合精度 AdamW 训练一个 $Phi$ 参数的模型，每卡至少要放这些（$Phi$ 是参数*个数*，下面都是字节）：

#formula[$ M = underbrace(2 Phi, "bf16 参数") + underbrace(2 Phi, "bf16 梯度") + underbrace(4 Phi, "fp32 master") + underbrace(4 Phi + 4 Phi, "Adam " m\, v) = 16 Phi $]

7B 模型：$16 times 7 times 10^9 = 112$ GB。A100 只有 80 GB，*连一步都跑不了*，activation 还没算。这条墙靠 ZeRO / FSDP 拆（第 19 章），或者靠 TP / PP 把模型本身切开（第 20 章）。

*时间墙*。Transformer 训练的 FLOPs 有个好用的估算式（forward 2、backward 4）：$"FLOPs" approx 6 Phi dot N_"tokens"$。7B 模型训 1T token 就是 $4.2 times 10^22$ FLOPs；单张 A100 bf16 峰值 312 TFLOPS、取 50% MFU 得 $1.56 times 10^14$ FLOPS，需要 $2.7 times 10^8$ 秒 $approx$ *8.5 年*。1024 卡才压到 3 天。这条墙靠 data parallel 拆。

=== 并行策略全景

#table(
  columns: (auto, 1.5fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*策略*], [*切什么*], [*详见*],
  [DP (data parallel)], [切 batch，每卡持完整模型，梯度 AllReduce], [第 18 章],
  [ZeRO / FSDP], [DP 的显存优化版：把参数/梯度/优化器状态沿 DP 组切片], [第 19 章],
  [TP (tensor parallel)], [切单个矩阵乘的权重，一层内部就要通信], [第 20 章],
  [PP (pipeline parallel)], [按层切成 stage，stage 间只传 activation], [第 20 章],
  [CP / SP (context / sequence)], [切 sequence 维，长上下文时 activation 装不下才用], [第 20 章],
  [EP (expert parallel)], [MoE 的 expert 分到不同卡，用 `all_to_all` 路由 token], [仓库的 MoE 书],
)

实际的大规模训练是这些的乘积（"3D / 4D 并行"）：`world_size = dp × tp × pp`。怎么组合、每一维放多大，取决于下面要讲的硬件拓扑。

#insight[
  记住这条分工：*DP 通信的是梯度（与 batch 无关，与参数量成正比）*，*TP / CP 通信的是 activation（与 batch·seq 成正比）*，*PP 通信的是 stage 边界的 activation（最省，但有 bubble）*。面试里几乎所有"该用哪个并行"的问题都能从这一句推出来。
]

== 集合通信原语的精确语义

先统一记号：$N$ = world size，$D$ = *这次通信涉及的全量数据字节数*。所有原语都用同一个 $D$ 表示，这样它们的通信量才能直接比。

#table(
  columns: (auto, auto, auto, auto, 1.2fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left, left, left),
  [*原语*], [*每卡输入*], [*每卡输出*], [*每卡收发量*], [*典型用途*],
  [`all_reduce`], [$D$], [$D$（全卡相同）], [$2 (N-1) \/ N dot D$], [DDP 梯度同步；求全局 loss],
  [`reduce_scatter`], [$D$], [$D \/ N$], [$(N-1) \/ N dot D$], [FSDP 梯度规约到自己那片],
  [`all_gather`], [$D \/ N$], [$D$], [$(N-1) \/ N dot D$], [FSDP unshard 参数；收集 logits],
  [`broadcast`], [src 有 $D$], [$D$], [$approx D$], [初始参数同步；广播 seed],
  [`reduce`], [$D$], [dst 有 $D$], [$approx D$], [把 metric 聚到 rank 0],
  [`all_to_all`], [$D$（$N$ 块）], [$D$（$N$ 块）], [$(N-1) \/ N dot D$], [MoE dispatch；SP 的 activation 重排],
  [`gather`], [$D \/ N$], [dst 有 $D$], [dst 收 $(N-1) \/ N dot D$], [收集每卡的评测结果],
  [`scatter`], [src 有 $D$], [$D \/ N$], [src 发 $(N-1) \/ N dot D$], [从 rank 0 分发数据],
  [`barrier`], [无], [无], [$approx 0$], [同步点（写 checkpoint 前后）],
)

几个容易被追问的点：

- *`all_reduce` / `reduce` 都是 in-place*：直接改传入的张量，不返回新张量。`reduce` 只有 `dst` 上的结果有意义，其他 rank 的张量是未定义的中间值——别去读。
- *`gather` / `scatter` 的形状不对称*：只有 `dst` 需要传 `gather_list`（长度 $N$ 的张量列表），其他 rank 传 `None`。
- *`all_to_all` 的语义是"转置"*：rank $i$ 的第 $j$ 块发给 rank $j$，rank $j$ 收到的第 $i$ 块来自 rank $i$。
- *`op` 支持 `SUM` / `AVG` / `MAX` / `MIN` / `PRODUCT`*。`ReduceOp.AVG` 由 NCCL 原生支持，比"SUM 完再除"省一次 elementwise kernel。

```python
import torch, torch.distributed as dist

# 每 rank 一个 (4,) 张量；world_size = 2
t = torch.full((4,), float(dist.get_rank() + 1), device="cuda")
dist.all_reduce(t)                     # rank0/1 都得到 [3,3,3,3]
dist.all_reduce(t, op=dist.ReduceOp.AVG)   # 都得到 [1.5,...]（NCCL 原生支持）

# 扁平版本比 list 版本快：少一次 stack/split
full  = torch.arange(8, device="cuda", dtype=torch.float32)   # D = 8 个元素
shard = torch.empty(4, device="cuda")
dist.reduce_scatter_tensor(shard, full)      # 每卡拿到求和后的一半

out = torch.empty(8, device="cuda")
dist.all_gather_into_tensor(out, shard)      # 拼回全量
```

#note[
  list 版本的 `all_gather(tensor_list, t)` / `reduce_scatter(out, input_list)` 要在 Python 侧构造张量列表、NCCL 内部还得拼成连续 buffer。*一律优先用 `all_gather_into_tensor` / `reduce_scatter_tensor`*，它们接受扁平张量，也是 FSDP 内部实际调用的入口。
]

=== 核心恒等式：AllReduce = ReduceScatter + AllGather

这是本章最高频的一道题。AllReduce 的语义是"每卡都拿到全体的规约结果"，实现上分两半：

+ *ReduceScatter*：把 $D$ 切成 $N$ 片，rank $i$ 负责规约第 $i$ 片。结束后 rank $i$ 手里是"第 $i$ 片的全局和"。
+ *AllGather*：每个 rank 把自己那片全局和广播给所有人。结束后每卡都有完整结果。

两个阶段各自的每卡收发量都是 $(N-1)\/N dot D$，所以：

#formula[$ D_"AR" = D_"RS" + D_"AG" = 2 (N - 1) / N dot D approx 2 D $]

#insight[
  *AllReduce 的每卡通信量 $approx 2 D$，与 $N$ 几乎无关*（$N -> infinity$ 时收敛到 $2D$，$N = 2$ 时是 $D$）。这是 DDP 能扩到上千卡的根本原因，也是"为什么 ZeRO-1/2 通信量和 DDP 一样"的直接推论：ZeRO 只是把 AllReduce 拆回它的两个组成部分，各留一半在不同位置用，总量没变（第 19 章）。
]

NCCL 的 `all_reduce_perf` 报的 `busbw` 就是按这个系数换算的：$"busbw" = D \/ T times 2(N-1)\/N$。所以估时间可以直接写：

#formula[$ T_"AR" approx 2 (N - 1) / N dot D / B_"bus" $]

== Ring AllReduce：为什么与 N 无关

朴素做法：所有 rank 把数据发给 rank 0，rank 0 求和再广播。rank 0 要收发 $2(N-1)D$，*通信量随 $N$ 线性增长*，而且它的网卡是唯一瓶颈。这就是 `nn.DataParallel` 的病根（第 18 章）。

Ring 的做法：把 $N$ 个 rank 排成环，每个 rank 只跟左邻居收、右邻居发。

#figure(
  align(center, ring-diagram(n: 4, labels: ("R0", "R1", "R2", "R3"),
                             title: "Ring：每卡只与两个邻居通信，带宽全用满")),
  caption: [4 卡 ring。每条边同时在传不同的数据分片，所以 $N$ 张网卡是*并行*工作的，
    而不是像 rank-0 聚合那样串行挤在一张卡上。],
) <fig-ring>

把数据切成 $N$ 片，跑两个各 $N-1$ 步的阶段：

+ *ReduceScatter*：第 $k$ 步，rank $i$ 把手上第 $(i - k) mod N$ 片发给右邻居，同时把左邻居发来的片*加到*自己对应的片上。$N-1$ 步后，rank $i$ 的第 $i$ 片已累加了全部 $N$ 个 rank 的贡献。
+ *AllGather*：同样绕环 $N-1$ 步，但这次是*覆盖*而不是累加，把每片的最终值传遍全环。

每步每卡收发 $D\/N$ 字节，两阶段共 $2(N-1)$ 步：

#formula[$ D_"ring" = 2 (N-1) dot D / N = 2 (N-1) / N dot D $]

和上一节的公式完全一致——*Ring 是通信量最优的 AllReduce 实现*，达到了理论下界。

为什么必须分两个阶段？因为规约和分发是两件事：$N-1$ 步只够让*每片*的和汇聚到*某一个* rank（这就是 ReduceScatter 的输出），要让所有 rank 都看到所有片，必须再绕一圈。想省掉第二圈，只能是 ReduceScatter 本身就够用的场景——那正是 ZeRO-2 / FSDP 干的事。

#note[
  Ring 的代价是*延迟*：$2(N-1)$ 步，每步一个网络往返，延迟随 $N$ 线性增长。小消息（几十 KB 级）上延迟项主导，NCCL 会自动切到 *Tree* 算法（延迟 $O(log N)$，通信量略高），或者在有 NVSwitch 的机器上用 *NVLS*（NVLink SHARP，在交换机里做规约）。可以用 `NCCL_ALGO` 强制，但默认的自动选择通常是对的。
]

== `init_process_group` 到底做了什么

```python
import os, torch, torch.distributed as dist

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)          # 必须在建 PG 之前
    dist.init_process_group(backend="nccl")    # 其余参数从环境变量读
    return local_rank
```

一行 `init_process_group(backend="nccl")` 背后是四件事：

+ *读环境变量*（`init_method="env://"` 是默认）：`RANK`（全局唯一编号 $0..N-1$）、`WORLD_SIZE`（$N$）、`MASTER_ADDR`、`MASTER_PORT`。注意 `LOCAL_RANK`（机内编号）*不被 `init_process_group` 使用*，它是给你自己调 `set_device` 用的。
+ *Rendezvous*：所有进程连到 `MASTER_ADDR:MASTER_PORT` 上的一个 *TCPStore*（rank 0 起 server，其他 rank 当 client）。TCPStore 是个简单的分布式 KV，用来交换"每个 rank 的 NCCL unique id、网卡地址"这些 bootstrap 信息，并做一次 barrier 确认所有 $N$ 个 rank 都到齐。
+ *初始化 backend*：NCCL 在这一步探测拓扑（谁和谁之间有 NVLink、有几张 IB 网卡）、建 ring / tree、分配通信 buffer。
+ *建默认 ProcessGroup*：之后所有不带 `group=` 参数的 collective 都走它。

#warn[
  `init_process_group` 是*阻塞*的：它会等到所有 `WORLD_SIZE` 个进程都完成 rendezvous 才返回，默认超时 30 分钟（NCCL backend 是 10 分钟）。所以"卡在 `init_process_group` 不动"永远是*有 rank 没起来*或者*`WORLD_SIZE` 设错了*，不是 NCCL 的问题。先去数进程个数。
]

TCPStore 只在初始化和少量元数据交换时用，*真正的数据通信不走它*——建好之后是 NCCL 直接走 NVLink / IB。所以 `MASTER_ADDR` 的网络质量对训练吞吐没影响。

=== backend 怎么选

- *`nccl`*：GPU 训练的唯一选择。走 NVLink / IB，支持全部原语，是唯一能打满带宽的。
- *`gloo`*：CPU-only 环境、`gather_object` 这类 CPU 侧集合，以及*本地调试*——没有多卡的机器上用 `backend="gloo"` + CPU 张量，`torchrun --nproc_per_node=4` 起 4 个进程，collective 语义与 NCCL 完全一致。
- *`mpi`*：需要编译时开启，只在和已有 MPI 作业集成时用。

一个进程组可以同时挂两个 backend（`backend="cpu:gloo,cuda:nccl"`），DDP 内部就靠这个在 CPU 上做一些控制流相关的 collective。

== `torchrun`：谁来设这些环境变量

手写 `MASTER_ADDR` 和逐个 rank 起进程太痛苦，`torchrun`（`torch.distributed.run`）就是那个 launcher：

```bash
# 单机 2 卡（本仓库的环境）
torchrun --nproc_per_node=2 train.py

# 2 机 × 8 卡，c10d rendezvous（推荐，不需要额外服务）
torchrun --nnodes=2 --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=node0:29500 \
         --rdzv_id=my_job train.py
```

#table(
  columns: (auto, 1.5fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*参数*], [*作用*],
  [`--nproc_per_node`], [每机起几个进程。GPU 训练就等于每机 GPU 数（也可写 `gpu` 自动取）],
  [`--nnodes`], [机器数。可写 `2:4` 表示弹性范围（配合 `--max_restarts`）],
  [`--rdzv_backend=c10d`], [用 rank 0 上内建的 TCPStore 做 rendezvous。取代老的 `--master_addr/--master_port` 写法],
  [`--rdzv_endpoint`], [rendezvous 服务地址 `host:port`。所有节点填同一个],
  [`--rdzv_id` / `--max_restarts`], [作业标识；worker 挂掉后允许整组重启几次（弹性训练）],
)

`torchrun` 为每个子进程设的环境变量：`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、`GROUP_RANK`（第几台机器）、`MASTER_ADDR`、`MASTER_PORT`、`TORCHELASTIC_RUN_ID`。你的脚本只需要读 `LOCAL_RANK`，其余交给 `init_process_group`。

#warn[
  *忘记 `torch.cuda.set_device(local_rank)` 是最高频的坑。* 不设的话所有进程的 "current device" 都是 `cuda:0`：8 个进程的模型和 NCCL buffer 全挤在 GPU 0 上，表现是"GPU 0 OOM、其他 7 张卡 0% 利用率"，或者在 NCCL 初始化时直接 hang（同一张卡上多个 rank 建 ring 会死锁）。

  ```python
  # 错
  dist.init_process_group("nccl")
  model = Model().cuda()                        # 全去 cuda:0

  # 对
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)             # 放在最前面
  dist.init_process_group("nccl")
  model = Model().to(f"cuda:{local_rank}")
  ```

  也可以用 `dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))` 显式绑定，较新版本还能让 NCCL 提前 eager 初始化，把建 communicator 的开销从第一次 collective 挪到这里。
]

== ProcessGroup 与 sub-group

3D 并行需要"只在某几个 rank 之间通信"——TP 组内 AllReduce activation、DP 组内 AllReduce 梯度。这就是 sub-group：

```python
# world_size = 8, 想切成 tp=2 × dp=4
rank = dist.get_rank()
tp_groups, dp_groups = [], []

# 关键：所有 rank 都要走完这两个循环，创建全部 group
for i in range(4):                                   # tp 组：{0,1} {2,3} {4,5} {6,7}
    g = dist.new_group(ranks=[2 * i, 2 * i + 1])
    tp_groups.append(g)
for i in range(2):                                   # dp 组：{0,2,4,6} {1,3,5,7}
    g = dist.new_group(ranks=[i, i + 2, i + 4, i + 6])
    dp_groups.append(g)

my_tp = tp_groups[rank // 2]
my_dp = dp_groups[rank % 2]
dist.all_reduce(grad, group=my_dp)                   # 只在 DP 组内规约
```

#warn[
  *`dist.new_group` 必须被所有 rank 以相同顺序调用，哪怕这个 rank 不在 `ranks` 里。* 它内部要做一次跨全 world 的元数据交换（分配 group id、建 NCCL communicator），少一个 rank 参与就永久 hang。

  ```python
  # 错：只有偶数 rank 调用 → 全体 hang 在这里
  if rank % 2 == 0:
      g = dist.new_group(ranks=[0, 2])

  # 对：全体调用，各自挑自己的
  g02 = dist.new_group(ranks=[0, 2])
  g13 = dist.new_group(ranks=[1, 3])
  my_group = g02 if rank % 2 == 0 else g13
  ```

  这也是手搓 3D 并行最容易翻车的地方。生产代码用 `init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))` 让 torch 建好全部 sub-group，再用 `mesh["dp"].get_group()` 取（第 21 章）。
]

另一个易错点：sub-group 里 rank 编号有两套。`dist.broadcast(t, src=...)` 的 `src` 是*全局* rank；手里只有 group 内下标时改用 `group_src=` 参数，或 `dist.get_global_rank(group, group_rank)` 转换。搞混会 broadcast 到错的源，而且*不会报错*。

== 同步、异步，以及 collective 的调用契约

`async_op=True` 让 collective 立刻返回一个 work handle：

```python
handle = dist.all_reduce(grad, async_op=True)
other_work()                    # 与通信 overlap
handle.wait()                   # 之后 grad 才可读
```

#insight[
  NCCL 的 collective 是在*另一条 CUDA stream* 上排的，所以"同步"和"异步"跟你想的可能不一样：`async_op=False` 时 `wait()` 被隐式调用，而 NCCL 的 `wait()` 做的是 `cudaStreamWaitEvent`——*给当前 stream 插一个依赖，CPU 并不阻塞*。真正让 CPU 阻塞要设 `TORCH_NCCL_BLOCKING_WAIT=1`（调试用）。这就是为什么 DDP 能把 AllReduce 藏进 backward：GPU 侧的依赖关系已经表达清楚了，CPU 只管往下发 kernel。
]

*Collective 的调用契约*（这条比任何 API 细节都重要）：

+ 组内*每个* rank 都必须调用，一个不落。
+ 调用的*顺序*必须一致。
+ 每次调用的 *shape、dtype、op、group* 必须一致。

违反任何一条 → hang，或者更糟：数据被静默错配。

#warn[
  最常见的违反方式是*控制流在不同 rank 上分叉*：

  ```python
  # 错：不同 rank 的 loss 不同 → 有的 rank 进 if 有的不进 → 集合调用不匹配 → hang
  if loss.item() > threshold:
      dist.all_reduce(stats)

  # 错：rank 0 多做一次 collective
  if rank == 0:
      dist.all_reduce(extra)        # 其他 rank 没有对应调用

  # 对：先把判据同步成全卡一致的值
  flag = torch.tensor([1.0 if loss.item() > threshold else 0.0], device="cuda")
  dist.all_reduce(flag, op=dist.ReduceOp.MAX)
  if flag.item() > 0:
      dist.all_reduce(stats)
  ```

  排查手法：设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 让超时后进程带栈退出而不是永久 hang，再看每个 rank 停在第几个 collective——用 flight recorder 直接 dump 未完成的 collective（第 22 章）。
]

`dist.barrier()` 是个纯同步点，没有数据。用在"rank 0 写完文件其他 rank 才能读"这种地方：

```python
if rank == 0:
    torch.save(state, "ckpt.pt")
dist.barrier(device_ids=[local_rank])    # 其他 rank 等 rank 0 写完
state = torch.load("ckpt.pt", map_location=f"cuda:{local_rank}")
```

NCCL 的 `barrier()` 实际是拿一个 dummy 张量做 AllReduce，会用到 current device——所以显式传 `device_ids` 比依赖 `set_device` 更稳。

== 硬件拓扑决定并行策略怎么摆

#table(
  columns: (auto, auto, 1.2fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*链路*], [*量级*], [*说明*],
  [NVLink 3.0（A100 SXM4）], [单向 300 GB/s], [12 条 link × 25 GB/s；NVIDIA 标称的 600 GB/s 是双向合计],
  [NVSwitch（DGX A100 内 8 卡）], [任意两卡同上], [全互联，不用担心"哪两张卡更近"],
  [PCIe 4.0 x16], [单向 \~32 GB/s], [没有 NVLink 的机器（如 A100 PCIe 版）走这条],
  [InfiniBand / RoCE 单网卡], [100–400 Gbps = 12.5–50 GB/s], [跨机。DGX 级机器有 8 张，聚合后仍与机内差一个数量级],
)

*机内比跨机快一个数量级*，结论就一句话：*通信最密集的并行维度放在机内*。TP 每层要做两次 activation AllReduce，必须压在 NVLink 域内（所以 `tp` 一般不超过单机卡数，A100/H100 上就是 8）；DP 的梯度 AllReduce 每 step 只做一次而且能和 backward overlap，跨机是可以接受的。

以 2 机 × 4 卡、`tp=2` × `dp=4` 为例（`world_size = 8`，rank 编号按 tp 最内层排）：

#figure(
  align(center, topology-grid(rows: 2, cols: 4,
    groups: ((0, 0, 1, 1), (2, 2, 3, 3)),
    group-labels: ((0, "tp0"), (1, "tp1"), (2, "tp2"), (3, "tp3")),
    title: "TP 组：每 2 卡一组，全部在机内 NVLink")),
  caption: [每行是一台机器。TP 组 `{G0,G1} {G2,G3} {G4,G5} {G6,G7}` 都不跨行——
    每层的 activation AllReduce 全走 NVLink。],
) <fig-topo-tp>

#figure(
  align(center, topology-grid(rows: 2, cols: 4,
    groups: ((0, 1, 0, 1), (0, 1, 0, 1)),
    group-labels: ((0, "dp0"), (1, "dp1")),
    title: "DP 组：跨机，但每 step 只 AllReduce 一次")),
  caption: [DP 组 `{G0,G2,G4,G6}` 和 `{G1,G3,G5,G7}` 跨了机器边界。
    梯度 AllReduce 每 step 一次且能与 backward overlap，跨机延迟被藏住。],
) <fig-topo-dp>

排 rank 的通用原则：*把通信频率最高的维度放在 rank 编号的最内层*，这样同组的 rank 编号连续、物理上也最近。Megatron 的默认顺序是 `tp` 最内、然后 `cp`、`dp`、`pp` 最外，正是按通信频率从高到低排的。用 `nvidia-smi topo -m` 看每对 GPU 之间实际是 `NV12`（12 条 NVLink）、`PIX`（同 PCIe switch）还是 `SYS`（跨 NUMA），别凭机型猜。

== NCCL 环境变量速查

#table(
  columns: (auto, 1.4fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*变量*], [*用途*],
  [`NCCL_DEBUG=INFO`], [打印拓扑探测、ring 构建、用了哪张网卡。排查一切分布式问题的第一步；太吵就用 `NCCL_DEBUG_SUBSYS=INIT,GRAPH` 过滤],
  [`NCCL_SOCKET_IFNAME=eth0`], [指定 bootstrap 用的网卡。多网卡机器上不设可能选到 `docker0` 之类的死网卡然后 hang],
  [`NCCL_IB_DISABLE=1`], [禁用 IB/RoCE 退回 TCP socket。用来确认"是不是 IB 配置的问题"，别在生产开],
  [`NCCL_P2P_DISABLE=1`], [禁用 GPU 间直连（NVLink / PCIe P2P），退回经主机内存。同上，诊断用],
  [`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`], [torch 侧 watchdog：collective 超时后主动 abort 进程并打栈，而不是永久 hang。默认值随版本变化，显式设 1 最稳],
  [`TORCH_NCCL_BLOCKING_WAIT=1`], [让 `work.wait()` 真的阻塞 CPU。定位"哪一行卡住"时用，会掉性能],
  [`NCCL_ALGO=Ring`], [强制算法（`Ring` / `Tree` / `NVLS` / `CollNet`）。默认自动选，一般只在对比测试时手动设。同族的 `NCCL_PROTO` 强制协议（`LL` / `LL128` / `Simple`）],
)

#note[
  `TORCH_NCCL_*` 前缀是 torch 2.2 起的命名（旧名 `NCCL_ASYNC_ERROR_HANDLING` 等已废弃）。这些是 *PyTorch* 读的变量，不是 NCCL 库读的——所以在 NCCL 文档里查不到。
]

== 通信量估算：一套通用方法

任何并行策略的每 step 通信量都可以套这个式子：

#formula[$ D_"step" = underbrace(Phi, "元素个数") times underbrace(b, "dtype 字节") times underbrace(c, "策略系数") $]

$c$ 就是各章要推的那个数：DDP 是 $2$（一次 AllReduce），ZeRO-3 是 $3$（两次 AllGather + 一次 ReduceScatter），TP 是 $4 times L$ 量级（每层前后各一次 AllReduce，且乘的不是 $Phi$ 而是 activation 大小）。

*具体算一遍*：7B 模型、bf16 梯度、DP 组大小 $N$：

- 一份梯度 $D = 7 times 10^9 times 2 = 14$ GB
- $N = 2$（本仓库环境）：$2 times (2-1)\/2 times 14 = 14$ GB
- $N = 8$：$2 times (8-1)\/8 times 14 = 24.5$ GB
- $N = 64$：$2 times (64-1)\/64 times 14 = 27.6$ GB

从 8 卡到 64 卡通信量只涨 13%，*这就是 "AllReduce 与 $N$ 无关" 的实感*。时间上代入 $T approx D_"AR" \/ B_"bus"$。若机内 NVLink 域的 busbw 是 200 GB/s 量级，8 卡的 24.5 GB 约 120 ms；跨机换成单网卡 25 GB/s 量级就要 1 s 量级。所以*跨机 DP 必须靠 overlap 把这一秒藏进 backward*，藏不住就得上 gradient accumulation 减少 AllReduce 次数（第 18 章），或者上 HYBRID_SHARD 把大部分通信压回机内（第 19 章）。

#warn[
  上面的时间是*基于假设带宽的量级估算*，不是实测。要报数就自己跑 `nccl-tests`（`./build/all_reduce_perf -b 8M -e 2G -f 2 -g 2`）看 `busbw` 列。别拿标称带宽当实际带宽——小消息受延迟主导，实测值可能只有标称的一半。
]

更深的推导（3D 并行的通信量矩阵、bubble 分析、大规模拓扑感知调度）见仓库的《大模型分布式训练面试通关手册》。

== 面试考点

#interview[
  *Q1*：AllReduce 和 ReduceScatter + AllGather 是什么关系？通信量各是多少？

  A：AllReduce 就是这两个的组合。ReduceScatter 把数据切 $N$ 片，让 rank $i$ 拿到第 $i$ 片的全局和，每卡收发 $(N-1)\/N dot D$；AllGather 再把每片广播回全环，同样 $(N-1)\/N dot D$。合起来 $2(N-1)\/N dot D approx 2D$。这个恒等式直接解释了 ZeRO-1/2 为什么通信量和 DDP 一样：它只是把这两半用在了不同位置。
]

#interview[
  *Q2*：Ring AllReduce 为什么通信量与 world size 几乎无关？

  A：数据切成 $N$ 片，每卡每步只发 $D\/N$，两阶段共 $2(N-1)$ 步，总量 $2(N-1)\/N dot D$，$N -> infinity$ 时收敛到 $2D$。关键是 $N$ 张网卡是*并行*工作的——环上每条边同时在传不同的片，不像"都发给 rank 0"那样串行挤在一张卡的带宽上。代价是延迟：$2(N-1)$ 步，延迟随 $N$ 线性增长，所以小消息 NCCL 会切到 Tree（$O(log N)$ 延迟）。
]

#interview[
  *Q3*：`init_process_group` 做了哪些事？

  A：四件。读环境变量 `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT`；所有 rank 连到 rank 0 起的 TCPStore 做 rendezvous，交换 NCCL unique id 等 bootstrap 信息并 barrier 确认全员到齐；初始化 backend（NCCL 探测 NVLink/IB 拓扑、建 ring）；建默认 ProcessGroup。它是阻塞的，卡住基本都是有 rank 没起来或 `WORLD_SIZE` 设错。数据通信不走 TCPStore。
]

#interview[
  *Q4*：为什么必须 `torch.cuda.set_device(local_rank)`？不做会怎样？

  A：不做的话所有进程的 current device 都是 `cuda:0`，模型、activation、NCCL buffer 全挤在 GPU 0 上——现象是 GPU 0 OOM、其他卡零利用率，或者 NCCL 建 communicator 时直接 hang（同一张卡上多个 rank 建 ring 会死锁）。要在 `init_process_group` 之前调用。注意 `LOCAL_RANK` 是机内编号，`RANK` 是全局编号，`set_device` 用的是前者。
]

#interview[
  *Q5*：两个 rank 调用 collective 的顺序不一致会发生什么？怎么排查？

  A：hang。collective 的契约是"组内每个 rank 都调用、顺序一致、shape/dtype/op/group 一致"，NCCL 按调用顺序配对，顺序错位就永久互等。最常见的诱因是控制流在不同 rank 上分叉（`if loss.item() > x:`、`if rank == 0:` 里做了 collective）。修法是先把判据 AllReduce 成全卡一致的值再分支。排查用 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 让超时后带栈退出，再用 flight recorder dump 未完成的 collective。
]

#interview[
  *Q6*：`dist.new_group(ranks=[0,1])` 只需要 rank 0 和 1 调用吗？

  A：不。*所有 rank 都必须调用，且顺序一致*，哪怕自己不在 `ranks` 里——因为它内部有一次跨全 world 的元数据交换。少一个 rank 参与就永久 hang。这也是手搓 3D 并行 mesh 最容易出错的地方，生产代码用 `init_device_mesh` 让 torch 建。
]

#interview[
  *Q7*：NCCL 和 gloo 怎么选？

  A：GPU 训练一律 nccl——它走 NVLink / IB，支持全部原语，是唯一能打满带宽的选择。gloo 用在三个地方：CPU-only 环境、没有多卡的本地调试（`gloo` + CPU 可以在单机跑 4 个进程验证分布式逻辑）、以及 `gather_object` 这类需要 CPU 侧序列化的集合。也可以 `backend="cpu:gloo,cuda:nccl"` 同时挂两个。
]

#interview[
  *Q8*：为什么 TP 要放在机内，DP 可以跨机？

  A：通信频率和通信量的量级差。TP 每层 forward / backward 各要一次 activation AllReduce，一个 32 层模型一个 step 就是上百次集合通信，而且通信量与 batch·seq 成正比；DP 只在 backward 末尾对梯度做一次 AllReduce，与 batch 无关且能和 backward overlap。机内 NVLink 单向 300 GB/s，跨机单张 IB 网卡 12.5–50 GB/s，差一个数量级——所以把通信最密集的维度压进 NVLink 域，`tp` 一般不超过单机卡数。
]

#interview[
  *Q9*：`async_op=True` 之后的 `work.wait()` 在等什么？会阻塞 CPU 吗？

  A：NCCL collective 排在一条独立的 CUDA stream 上，`wait()` 默认做的是 `cudaStreamWaitEvent`——给当前 stream 插一条依赖，*CPU 不阻塞*，只是保证后续 kernel 在通信完成后才跑。想让 CPU 真阻塞要设 `TORCH_NCCL_BLOCKING_WAIT=1`（调试用）。这个机制正是 DDP 能把梯度 AllReduce 藏进 backward 计算的基础：依赖关系在 GPU 侧表达，CPU 只管继续发 kernel。
]
