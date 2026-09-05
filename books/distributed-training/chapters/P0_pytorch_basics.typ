#import "../template.typ": *

= PyTorch 与 autograd：分布式训练的底盘

这一章讲面试里高频、但常常被"直接跳过"的 PyTorch 底盘知识。不重复官方 tutorial，只讲 *与分布式训练/性能强绑定* 的部分：Tensor 存储模型、autograd 图、hook 时机、`torch.compile` 边界、DTensor、CUDA stream/event、`memory_format`。理解这些，才能读懂 Megatron、FSDP、`torch.distributed` 的源代码。

== Tensor: storage / view / stride

一个 `Tensor` 是"底层 `Storage` 的视图"。理解这一点能省下 90% 的 "shape 对但值不对" 的调试时间。

```python
x = torch.arange(12).reshape(3, 4)     # shape=(3,4), stride=(4,1), storage=[0..11]
y = x.T                                 # shape=(4,3), stride=(1,4), storage 同上，无 copy
y.is_contiguous()                       # False
y.contiguous()                          # 触发 copy → 新 storage
```

*为什么要 `.contiguous()`：* NCCL send/recv、all_gather 底层都直接读取连续 storage；非连续张量会 assertion fail 或悄悄传错。生产代码里 P2P 前的 `.contiguous()` 都是必需。

*view vs reshape：* `view` 只做 stride 变换，要求连续；`reshape` 自动决定是否 copy。

```python
x = torch.zeros(4, 8)
x.view(2, 16)         # OK
x.T.view(8, 4)        # RuntimeError: not contiguous
x.T.reshape(8, 4)     # OK (silent copy)
```

*memory_format：* channels-last (NHWC) 用于 CNN；LLM 场景不涉及。分布式训练里唯一相关的是 `.contiguous(memory_format=torch.contiguous_format)`——多数场景默认已是 contiguous。

#interview[
  面试题：*"为什么 all_gather 前要 .contiguous()？非连续 tensor 会怎样？"* 答：NCCL 直接读 base pointer + stride=1 假设，non-contiguous 会传出乱序数据。PyTorch 新版本会 assert，旧版本静默出错。

  进阶：*"a2a 时输入 shape (B, S/W, A, d) 你怎么保证是 contiguous 的？"* 答：`.chunk(dim=2)` 出来的每片*不*连续（因为切的是 stride 大的维度），需要显式 `.contiguous()`；否则 `all_to_all_single` 会失败。
]

== Autograd: 图、Function、hook

Autograd 是一张*动态构建的有向图*。每个 `Tensor` 有 `.grad_fn`（若是运算得到）指向创建它的 `Function`；`Function` 有 `.next_functions` 指向输入。

*核心 API：*
- `torch.autograd.Function` — 自定义前反向
- `Tensor.register_hook(fn)` — 该 tensor 有 grad 时触发（backward 阶段）
- `Module.register_forward_hook`, `register_full_backward_hook` — 模块级
- `Tensor.retain_grad()` — 非叶子 tensor 保留 `.grad`（默认 backward 完就丢）

*hook 触发顺序（面试常问）：*

```
backward starts (from loss)
  → for each Function in reverse:
       run Function.backward() → produce grad w.r.t. inputs
       for each input tensor:
           if it has registered hook → run hook with grad
       grad accumulates into input.grad (if leaf)
       if input is not leaf, propagate to input.grad_fn
```

*DDP 的 backward hook 就是绑在参数（叶子 tensor）的 `.grad` 上*。参数 `p` 的 grad 一算出，`p.register_hook(lambda g: bucket.mark_ready(p))` 立即触发，然后 bucket 满就发 async AR。这也是为什么 `find_unused_parameters=True` 会破 overlap——DDP 无法确定某个 bucket 何时"齐全"，只能等 backward 全结束。

*自定义 Function 的正确写法：*

```python
class RingAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x
    @staticmethod
    def backward(ctx, dy):
        dist.all_reduce(dy, group=ctx.group)
        return dy, None  # 每个 forward 输入对应一个 grad
```

*常见陷阱：*
- `forward` 里 in-place 修改输入 tensor → autograd 报错，除非在 `mark_dirty()`
- `backward` 返回的数量必须与 `forward` 输入的数量相同（`None` 占位不需要 grad 的参数）
- `ctx.save_for_backward` 只能存 tensor；其他用 `ctx.some_attr = value`
- forward 里的 `ctx.save_for_backward(x)` 会让 `x` 一直保活到 backward，若 `x` 是大 activation → 显存占满

*完整例子：Ring AllReduce 前向 = 后向对称* 是 DP 通信的关键——正因为 forward AR 会让 grad 也需要 AR，DDP 才能只在 bucket 里发一次通信就完成"所有 rank 的 grad 求和"。

#insight[
  面试深挖：*"如果你不显式做 backward AR，会怎样？"* 答：DDP 里就是靠 `Function` 的对称性——forward 时 replicate input（每卡拿全 batch 的分片），backward 时 grad 自动 AR。若在自定义 `Function.forward` 里做了 AR，就必须在 backward 里也做——否则不同 rank 的 grad 会不一致，params drift。
]

== Hook 类型速查

#table(
  columns: (2fr, 1.4fr, 1.6fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*Hook*], [*触发*], [*常见用途*],
  [`Tensor.register_hook`],
    [该 tensor 的 grad 计算好时],
    [DDP、grad 监控、grad-clip unscale],
  [`Module.register_forward_pre_hook`],
    [forward 前],
    [activation offload、input shape trace],
  [`Module.register_forward_hook`],
    [forward 后],
    [activation checkpoint、mem tracker],
  [`Module.register_full_backward_hook`],
    [该 module 的输入 grad 算好后],
    [gradient magnitude tracker],
  [`Module.register_state_dict_pre_hook`],
    [`state_dict()` 保存前],
    [checkpoint 剥离 buffer],
  [`Module.register_load_state_dict_post_hook`],
    [`load_state_dict` 后],
    [FSDP 参数重构],
)

== `torch.compile`: JIT 边界与分布式

`torch.compile()` 用 TorchDynamo 抓取 python 字节码，用 AOTAutograd + PrimTorch 展开成 op 序列，用 Inductor 编译成 Triton kernel。收益：kernel fusion (10-30% speedup on H100)、消除 Python overhead。

*与分布式的相互作用（面试重点）：*

+ *通信不能 trace*：`dist.all_reduce` 是 Python-side call，TorchDynamo 遇到会 *graph break*（切断编译图）。Torch 2.2+ 用 `torch._C._distributed_c10d._register_process_group` 让 NCCL op 变成 traceable，Torch 2.4+ 有 `torch._inductor.config.reorder_for_compute_comm_overlap` 可以在编译图里自动 reorder，让 compute/comm overlap。

+ *DDP + compile*：先 wrap DDP 再 compile → 每个 bucket 是一个 subgraph；先 compile 再 DDP → 一个大图。生产用第一种。

+ *FSDP + compile*：Torch 2.3+ 支持，需要 `use_orig_params=True`。仍有若干边界问题（graph break at unshard），预期 2026 稳定。

+ *TP + compile*：`torch.distributed.tensor.parallel` 的 DTensor 与 compile 一起可用 (torchtitan 主线支持)。

+ *动态 shape*：`torch.compile(dynamic=True)` 才能编译一次跑不同 shape，否则每种 shape 都要重编——分布式训练 packing 场景常见的坑。

```python
# 生产推荐 idiom
model = MyModel().cuda()
model = FSDP(model, use_orig_params=True, sharding_strategy=FULL_SHARD)
model = torch.compile(model, mode="reduce-overhead", dynamic=False)
```

#warn[
  `mode="reduce-overhead"` 使用 CUDA graphs，与 CUDA stream 上的动态 P2P 通信不兼容。PP 场景要用 `mode="default"`。
]

== DTensor：分布式 tensor 抽象

Torch 2.1 引入。DTensor 把 sharding meta（"这个 tensor 在 dim=1 沿 tp 组切"）附加到 tensor 上，让 op 自动推断需要什么 collective。

```python
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard, Replicate

mesh = DeviceMesh("cuda", torch.arange(8).view(2, 4),
                  mesh_dim_names=("dp", "tp"))

W = torch.randn(4096, 4096)
W_dt = distribute_tensor(W, mesh["tp"], placements=[Shard(0)])
# W_dt: DTensor sharded on dim 0 across 4 TP ranks; each rank holds (1024, 4096)

y = W_dt @ x_dt  # DTensor 会自动推断 Shard(0) @ Replicate → 需要 AllGather
```

*为什么 DTensor 是趋势：*
- Megatron/DeepSpeed 里 TP 是手写 `ColumnParallelLinear` / `RowParallelLinear` 每个 op；DTensor 让"把 tensor 沿某维切"变成声明式
- torchtitan、torchtnt、FSDP2 全部基于 DTensor
- 与 `torch.compile` 集成好——因为 sharding meta 是 tensor 属性，编译期就能拿到

*面试考点*：*"DTensor 与 FSDP/TP 手写实现相比，性能怎样？"* 答：目前 (2026 Q3) DTensor 的 latency 略高 (~5% due to placement resolution)，但 torchtitan 已达到 Megatron 90% MFU。长期看 DTensor 会成为主流，因为工程复杂度低太多。

== CUDA stream / event / graph

*Stream*：CUDA 内的执行队列，同一 stream 内串行，不同 stream 之间并行。默认 stream 是 `torch.cuda.default_stream()`。

*Event*：同步点。`event.record(stream)` 打点，`stream2.wait_event(event)` 让 stream2 等到该 event 触发。

*用法：让通信与计算 overlap*：

```python
comm_stream = torch.cuda.Stream()

# 主 stream 做计算 y = A @ B
# comm stream 做 AllReduce x

evt_a = torch.cuda.Event()
evt_a.record()   # 记录 default stream 上当前进度

with torch.cuda.stream(comm_stream):
    comm_stream.wait_event(evt_a)         # 等 default stream 到达此点
    dist.all_reduce(x, async_op=True)

y = A @ B         # 与 AR 并行

torch.cuda.default_stream().wait_stream(comm_stream)   # 用 x 前等 AR 完
```

这就是 Ch11 里 overlap 的物理机制。手写实现见 `src/distributed_training/11_overlap_2stream.py`。

*CUDA graph*：把一段 stream 上的 kernel launch 序列固化成"图"，之后每次 replay 只需 1 次 launch overhead。收益：小 batch/短 seq 时 CPU launch overhead 可占 20-30%，graph 消灭它。`torch.compile(mode="reduce-overhead")` 底层用 CUDA graph。局限：*shape 必须静态、控制流不能变*。所以动态 packing / PP boundary 都不能上 graph。

== `torch.distributed`: 从 backend 到 process group

`torch.distributed.init_process_group(backend="nccl")` 做的事：
+ 从 `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` 读参数（`torchrun` 帮你设好）
+ 与其他 rank rendezvous（TCP store）
+ 初始化 NCCL/GLOO backend

*Process group*：一个通信组的抽象。默认组 = 全部 world。要做 sub-group AR（比如仅 TP 组内）：

```python
tp_group = dist.new_group(ranks=[0, 1, 2, 3])
dist.all_reduce(x, group=tp_group)  # 仅在这 4 卡内 AR
```

Megatron/FSDP 内部维护多个 group：TP group, DP group, PP group, EP group，Rank ↔ 各 group 的映射由 `mesh` 决定。

*NCCL communicator 内存*：每个 process group 会创建一个 NCCL communicator，占 ~100 MB HBM。你有 TP+DP+PP+EP+CP 5 个维度 → 5 个 comm × 5 个 group per dim = 25 个 group，占几 GB。这就是"过多 sub-group 吃显存"的原因，实际会 lazy init。

*超时*：`init_process_group(timeout=timedelta(minutes=30))`——默认 10 min，长 checkpoint 保存或大 batch 慢 rank 场景需拉大。

== 常用调试/性能 API

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  [*API*], [*用途*],
  [`torch.cuda.memory_allocated()`], [当前 HBM 占用],
  [`torch.cuda.max_memory_allocated()`], [step 峰值，OOM 分析必用],
  [`torch.cuda.memory_summary()`], [详细 breakdown by size bin],
  [`torch.cuda.reset_peak_memory_stats()`], [每 step 前清零，追踪峰值来源],
  [`torch.profiler.profile(...)`], [kernel timeline + memory + comm],
  [`torch._C._cuda_setSyncDebugMode(1)`], [任何隐式 sync 报错，抓 dataloader 阻塞],
  [`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`], [NCCL hang 时 raise 而不是死等],
  [`TORCH_DISTRIBUTED_DEBUG=DETAIL`], [打印每个 collective 的 shape/rank],
  [`torch.cuda.set_per_process_memory_fraction(0.9)`], [限制单进程显存，防跑飞],
)

*生产模板：*
```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # 只在 debug 时
export NCCL_DEBUG=WARN                     # 平时；hang 时 =INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0              # =1 只在 debug crash 时
```

== 面试考点

#interview[
  *Q1*：`register_hook` 何时触发？与 `register_full_backward_hook` 有何不同？

  A：`register_hook` 是 tensor 级，backward 走到该 tensor 时（其 grad 被计算）触发。`register_full_backward_hook` 是 module 级，输入 grad 全部算好后触发。DDP 用前者绑参数 grad；activation offload 用后者绑 module 输入。
]

#interview[
  *Q2*：`.view()` 和 `.reshape()` 什么区别？分布式代码里用哪个？

  A：`view` 要求连续，只改 stride 不 copy；`reshape` 自动 fallback 到 copy。分布式代码 (NCCL) 强制要求连续，一般显式 `.contiguous().view(...)`，避免 reshape 里的"看似省事但会 silent copy 占显存"的坑。
]

#interview[
  *Q3*：`torch.compile` 遇到 `dist.all_reduce` 会发生什么？如何 workaround？

  A：老版本 (< 2.2) 触发 graph break，把编译图切成两段。新版本注册了 functional collective ops (`torch._C._distributed_c10d`), 可以 inline 进图。生产用 `torch.compile(mode="default")` + 让 comm 在 Python 层（不 compile 的 wrapper）里发；或用 Torch 2.4+ 的 `reorder_for_compute_comm_overlap`。
]

#interview[
  *Q4*：CUDA stream 和 CUDA graph 的关系？

  A：Stream 是执行队列（decorate op 顺序 + 并行），graph 是"把 stream 上的一串 kernel launch 序列快照下来"。graph replay 消除 CPU-side launch overhead，但要求 shape 静态。`torch.compile(mode="reduce-overhead")` 底层用 graph。
]

#interview[
  *Q5*：DTensor 与 Megatron 里的 `ColumnParallelLinear` 相比？

  A：语义等价。DTensor 是声明式（"这个 tensor 沿 tp 切"，op 自动推断需要 AG/RS/AR）；Megatron 是命令式（每个 op 手写通信）。DTensor 更易读、易组合，但当前 latency 略高，正在追赶。torchtitan 是 DTensor path 的旗舰实现。
]

#interview[
  *Q6*：`init_process_group(timeout=...)` 为什么重要？

  A：默认 10 min。如果 rank 之间 barrier / collective 未在 10 min 内会合，会 raise。大规模训练里 checkpoint save 可能几分钟，dataloader 卡顿也常见 → 需要拉到 30-60 min。同时开 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，避免整个 job 死锁等 hang。
]

#interview[
  *Q7*：你怎么知道你的 DDP overlap 生效了？

  A：三种验证：
  + `nsys profile` 看 timeline：`ncclAllReduce` kernel 与前面 `sgemm` kernel 在不同 stream + 时间叠加
  + `torch.profiler` 看 "GPU busy time"：若 busy ≈ wall-clock，comm 全 hidden
  + 手动实验：`comm_stream + Event` 显式 overlap 与不 overlap 对比 iter time。
]
