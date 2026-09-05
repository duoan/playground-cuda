#import "../template.typ": *

= DDP：原理、实现与调优

DDP 是分布式面试的必答题，而且面试官问的几乎总是同一个点：*梯度 AllReduce 怎么和 backward 计算重叠*。答案是 bucket + backward hook 这一套机制，它把"backward 算完再通信"的串行结构改成了流水线。这一章把 DDP 从数学（为什么是梯度平均）讲到实现（bucket 怎么分、hook 什么时候触发），再到那一长串会让训练 hang 住的坑。上一章的 $D_"AR" approx 2D$ 是这里所有成本分析的前提。

== 为什么 `nn.DataParallel` 被废弃

`nn.DataParallel`（DP，注意和"data parallel"这个泛称区分）是*单进程多线程*的实现，一个 step 干这些事：

+ rank 0 把 batch `scatter` 到各卡
+ 每卡各起一个 Python 线程跑 forward
+ 输出 `gather` 回 GPU 0，在 GPU 0 上算 loss
+ backward 的梯度 `gather` 回 GPU 0，GPU 0 上做 optimizer step
+ 更新后的参数 `broadcast` 回其他卡

四个致命问题：

- *GIL 争抢*：多个线程在同一个 Python 解释器里发 kernel，CPU 侧的 launch 被 GIL 串行化。模型层数越多、kernel 越小，这个开销越致命。
- *GPU 0 是瓶颈也是热点*：所有 output、loss、梯度都汇聚到 GPU 0，它的显存占用显著高于其他卡（经常是"7 张卡 40 GB，GPU 0 OOM"），算力和带宽也不均衡。
- *每 step 重新 broadcast 参数*：因为只有 GPU 0 更新了参数，别的卡必须重新拿一份。这是纯粹的浪费——DDP 里每卡都能自己更新出相同的参数。
- *无 overlap，且不能跨机*：`gather` 必须等 backward 全部结束；单进程模型天然只能单机。

DDP 的答案是*每张卡一个独立进程*：绕开 GIL、原生跨机、用 AllReduce 替代 gather + broadcast，让每卡对称。

#insight[
  一句话概括差异：*DP 是"一个 master 收发数据"，DDP 是"所有卡对称地交换数据"*。前者的通信量随卡数线性增长且集中在一张网卡上，后者按上一章的 Ring AllReduce 是 $2(N-1)\/N dot D$ 且带宽全用满。
]

== DDP 的数学基础

DDP 一个 step 的语义：

+ 全局 batch 按 rank 切开，rank $i$ 拿到 micro-batch $B_i$（大小 $b$）
+ rank $i$ 在 $B_i$ 上算 loss $cal(L)_i$，backward 得到 local 梯度 $g_i = nabla cal(L)_i$
+ 对 $g_i$ 做 AllReduce 求*平均*，每卡得到相同的 $macron(g)$
+ 每卡各自跑 optimizer step

=== 为什么是平均而不是求和

因为要和"单卡跑一个 $N b$ 的大 batch"等价。单卡大 batch 的 loss 通常定义成样本均值：

#formula[$ cal(L) = 1 / (N b) sum_(i=1)^N sum_(x in B_i) ell(x) = 1 / N sum_(i=1)^N cal(L)_i $]

对它求梯度就是 $macron(g) = 1/N sum_i g_i$——正是梯度的*算术平均*。如果 AllReduce 用 SUM 不除，得到的梯度是正确值的 $N$ 倍，等价于把学习率悄悄放大了 $N$ 倍，训练很快就炸。

DDP 内部自动做这个除法：`ReduceOp.SUM` 之后除 `world_size`（较新版本直接用 NCCL 原生的 `ReduceOp.AVG`，省一次 elementwise kernel）。所以*你的训练脚本里 loss 不需要手动除 world size*——它已经是 local batch 的均值了。

#warn[
  这个等价性有个前提：*每个 rank 的 local batch 大小必须相同*。如果最后一个 batch 各 rank 样本数不等，简单平均就给小 batch 的样本更高的权重了。用 `DistributedSampler` 时它会自动 pad 到整除；用 token 级 loss（LLM 训练里 `sum(loss) / num_tokens`）时各 rank 的有效 token 数天然不等，*严格做法是先 AllReduce token 数再归一化*，否则 loss 和梯度都有偏。
]

=== 为什么初始参数必须一致

AllReduce 只同步*梯度*，不同步*参数*。只要初始参数一致、梯度一致、optimizer 状态一致，每卡独立 step 就必然算出相同的参数——DDP 全靠这个不变量维持一致性。所以：

- *构造时 broadcast 参数*：`DDP(model)` 在 `__init__` 里把 rank 0 的参数和 buffer 广播给所有 rank（由 `init_sync=True` 控制，默认开）。这样即使你每个 rank 用了不同的随机种子，也不会跑飞。
- *`broadcast_buffers=True`（默认）*：每次 forward 前把 rank 0 的 *buffer*（`register_buffer` 注册的东西，如 BatchNorm 的 `running_mean` / `running_var`）广播给所有 rank。buffer 不是参数、没有梯度，不参与 AllReduce，所以只能靠广播保持一致。

#note[
  纯 Transformer 模型（LayerNorm / RMSNorm，没有 running stats）通常没有需要同步的 buffer，`broadcast_buffers=False` 可以省掉每 step 一次 broadcast。有 BatchNorm 时不能关。
]

== bucket + backward hook：本章核心

Naive 的实现是"backward 全部跑完，再对所有梯度做一次大 AllReduce"。问题是这两段完全串行：通信期间 GPU 空转，一个 step 的时间是 $T_"compute" + T_"comm"$。

DDP 的做法是把梯度切成若干 *bucket*，*每个 bucket 一就绪就立刻发一次 async AllReduce*，让通信和还没算完的 backward 重叠。

#figure(
  align(center, timeline(
    streams: (
      ("naive  compute", (("compute", 16),)),
      ("naive  comm   ", (("wait", 16), ("comm", 8))),
      ("bucket compute", (("compute", 16),)),
      ("bucket comm   ", (("wait", 4), ("comm", 3), ("wait", 1), ("comm", 3),
                          ("wait", 1), ("comm", 3), ("wait", 1), ("comm", 3))),
    ),
    unit: 0.42, bar-h: 0.5,
    title: "bucket 把 AllReduce 藏进 backward",
  )),
  caption: [上两行 naive DDP：backward 结束才开始通信，step time $= T_"compute" + T_"comm"$。
    下两行 bucket DDP：每个 bucket 就绪就发 async AllReduce，通信藏在后续 backward 里，
    理想情况 step time $approx T_"compute" +$ 最后一个 bucket 的通信时间。],
) <fig-ddp-overlap>

机制拆开看：

+ *构造时分桶*：`DDP.__init__` 把所有 `requires_grad=True` 的参数按顺序装进 bucket，默认每桶 25 MB（`bucket_cap_mb`）。同一个 bucket 里的梯度会被拷进一块连续的 buffer，这样只发一次 AllReduce 而不是每个参数一次。
+ *注册 autograd hook*：给每个参数的 `AccumulateGrad` 节点挂钩子。参数的梯度算完 → hook 触发 → 把这个梯度标记 ready，并拷进它所属 bucket 的 buffer。
+ *bucket 齐了就发*：某个 bucket 里所有参数都 ready → 立刻 `all_reduce(bucket.buffer, async_op=True)`，*不等 backward 结束*。
+ *backward 结束时收*：DDP 在 backward 末尾等所有 pending 的 work handle，然后把 bucket buffer 里的平均梯度写回各个 `p.grad`。

=== 为什么按参数的反向顺序分桶

backward 是从 loss 往输入走的，*最后一层的梯度最先算出来*。而 `model.parameters()` 的顺序是前向定义顺序（第一层在前）。所以 DDP 分桶时把参数列表*倒过来*：第 0 个 bucket 装的是模型最后几层的参数。

这样第 0 个 bucket 在 backward 刚开始几毫秒就齐了，AllReduce 立刻发出去，有整个 backward 的时间去藏它。如果按正向顺序分，第 0 个 bucket 要等 backward 走到第一层才齐——那一刻 backward 已经结束，退化成 naive 实现。

#insight[
  "为什么反向顺序" 是这道题的题眼：*overlap 的可用窗口 = 从这个 bucket 就绪到 backward 结束的时间*。让最先算出梯度的参数分在最先发的 bucket，就是让每个 bucket 的窗口最大化。
]

=== bucket 大小的取舍

单次 AllReduce 的时间可以粗略写成"延迟 + 带宽项"：$T approx alpha + D\/B$。$n$ 个 bucket 就是 $n alpha + D\/B$。

- *桶太小*：$n$ 大，$n alpha$ 主导，而且每次 AllReduce 都要走一遍 NCCL 的 kernel launch 和同步开销。
- *桶太大*：backward 走到很后面这个桶才齐，能藏的时间变短；极端情况一个桶装下整个模型，就是 naive 实现。

25 MB 是 PyTorch 的通用默认值。7B+ 模型或跨机（$alpha$ 大）时调大到 100–500 MB 通常更好，但*具体值必须实测*——用 profiler 看 AllReduce kernel 是不是压在 compute kernel 下面。

```python
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=100,
    gradient_as_bucket_view=True,     # 省一次拷贝，大模型建议开
    broadcast_buffers=False,          # 纯 Transformer 没有需要同步的 buffer
)
```

`gradient_as_bucket_view=True` 让 `p.grad` 直接是 bucket buffer 上的一个 view，而不是独立张量。默认（`False`）时每个梯度要从 `p.grad` 拷进 bucket buffer、AllReduce 完再拷回来——开了这个选项就省掉这两次拷贝，同时也省下一份梯度的显存。

#warn[
  开了 `gradient_as_bucket_view=True` 之后不要对 `p.grad` 做原地替换（`p.grad = some_new_tensor`），那会切断 view 关系让 DDP 拷贝失效。用 `p.grad.copy_(...)` 或 `optimizer.zero_grad(set_to_none=True)`（DDP 会在下一次 backward 重建 view）。
]

== `find_unused_parameters` 的代价

DDP 判断"bucket 齐了"的依据是"这个 bucket 里每个参数的 hook 都触发过"。如果某个参数这一步没参与 forward，它的 hook 永远不触发，那个 bucket 永远不齐，AllReduce 永远不发——*其他 rank 在等这次 AllReduce，全体 hang*。

`find_unused_parameters=True` 的做法是：backward 一开始就从 loss 的 autograd 图往回遍历一遍，找出哪些参数*不*在图里，提前把它们标记成 ready。

代价有两块：

- *每 step 一次图遍历*：纯 CPU 开销，模型越大越明显。
- *破坏 overlap*：这次遍历要在 backward 起步阶段完成才能确定 bucket 的就绪条件，实践中会推迟第一批 AllReduce 的发出时机，overlap 窗口变窄。

什么时候不得不开：模型里真有条件分支（`if task == "a": use head_a`）、MoE 只激活部分 expert、多任务模型每 step 只用一个 head。

更好的做法，按优先级排：

+ *改模型让所有参数都参与*。MoE 的常见手法是在 aux loss 里加一个乘 0 的项，让所有 expert 权重都在图里（梯度是 0，但 hook 会触发）。
+ *`static_graph=True`*：如果"哪些参数参与"在每个 step 都一样（只是不是全部参与），用这个。它在第一个 step 记录下参与的参数集合和 backward 的执行顺序，之后每 step 复用，*不需要重新遍历*。还能顺带优化 activation checkpointing 下的重入问题。
+ 实在动态就只能 `find_unused_parameters=True`。

#warn[
  `static_graph=True` 和 `find_unused_parameters=True` 不要同时开——`static_graph` 已经涵盖了 unused 参数的情况，同时设会被忽略或报错。而且 `static_graph=True` 要求图*真的*静态：如果 step 之间参与的参数集合会变，行为是未定义的（表现为梯度错或 hang）。
]

== `no_sync()`：梯度累积时别通信

梯度累积的语义是"累 $K$ 个 micro-batch 的梯度再 step 一次"。累积中间的梯度是不完整的，同步它没有意义——只在最后一个 micro-step 做一次 AllReduce 就够了，因为梯度累加和 AllReduce 都是线性算子，交换顺序不影响结果。

```python
from contextlib import nullcontext

ACCUM = 8
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    is_last = (i % ACCUM == ACCUM - 1)
    # 前 K-1 步进 no_sync：hook 照样触发累加梯度，但不发 AllReduce
    ctx = nullcontext() if is_last else model.no_sync()
    with ctx:
        loss = criterion(model(x), y) / ACCUM     # 除以 ACCUM 保持梯度是均值
        loss.backward()
    if is_last:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

要点三条：

- `loss / ACCUM` 是必须的，否则梯度是 $K$ 个 micro-batch 的*和*而不是均值，等于把 LR 放大 $K$ 倍。
- `no_sync()` 省掉 $K-1$ 次全模型 AllReduce。$K = 8$ 时通信次数降到 $1\/8$。
- *`clip_grad_norm_` 必须在最后一步、AllReduce 之后做*。累积期间梯度是 partial 的，对它 clip 得到的 norm 没有意义。

#formula[$ "GBS" = "MBS" times "accum" times N_"dp" $]

#note[
  如果 backward 本来就能完全藏住 AllReduce，`no_sync()` 的收益不大——省下的通信本来就是免费的。它在*跨机、通信藏不住*的场景才是关键优化。反过来，用了 `no_sync()` 就相当于把 $K$ 步的通信压到一步，那一步的通信更难藏，需要重新调 `bucket_cap_mb`。
]

== DDP 与 BatchNorm

BatchNorm 在每卡上*只用本卡的 local batch* 算 mean / var。8 卡、每卡 MBS=4 时，BN 看到的实际是 batch size 4 的统计量，不是 32——*DDP 下 BN 的 effective batch 被切小了*。batch 小到个位数时 BN 的统计噪声会明显伤害收敛。

`SyncBatchNorm` 让 BN 的统计量跨卡同步：

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   # 必须在包 DDP 之前
model = DDP(model, device_ids=[local_rank])
```

代价：*每个 BN 层每次 forward 都要做一次 AllReduce*（同步 $sum x$ 和 $sum x^2$，可以合并成一次）。ResNet-50 有 53 个 BN 层，就是每 step 多 53 次小消息 AllReduce——小消息受延迟主导，开销比通信量看起来严重得多，而且这些 AllReduce 在 forward 里，*没有 backward 计算可以拿来 overlap*。

结论：CV 模型 per-GPU batch 很小时才值得开；LLM 一律用 LayerNorm / RMSNorm，没有这个问题。

== DDP + `torch.compile`

推荐顺序是*先 `DDP` 再 `compile`*：

```python
model = DDP(model, device_ids=[local_rank])
model = torch.compile(model)
```

DDP 的 AllReduce 是靠 autograd hook 触发的，而 Dynamo 默认会把整个 backward 编译成一个大的 fused graph——如果 backward 是一整块，所有梯度是在图末尾一次性产出的，hook 全在同一时刻触发，*bucket 的渐进就绪性质就消失了，overlap 归零*。

所以 Inductor 里有个专门的 `ddp_optimizer`（由 `torch._dynamo.config.optimize_ddp` 控制，默认开）：它读取 DDP 的 `bucket_cap_mb`，*按 bucket 边界把图切成多个子图*。每个子图的 backward 一跑完就产出对应 bucket 的全部梯度、触发 AllReduce，overlap 结构被保留下来。代价是引入了 graph break（第 15 章讲了怎么权衡）。

#note[
  这也解释了为什么 DDP + compile 的 bucket 数量会影响编译产物：桶越多、切的子图越多、fusion 机会越少。相比 eager，compile 场景下可以适度*调大* `bucket_cap_mb` 换更大的子图。
]

== 手写一个极简 DDP

白板题常见问法："不用 `nn.parallel.DistributedDataParallel`，你怎么实现梯度同步？"

```python
import os, torch, torch.nn as nn, torch.distributed as dist

class MiniDDP(nn.Module):
    """极简 DDP：参数 broadcast + 每个梯度就绪即 async AllReduce。"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        self._works = []
        # 1) 初始参数对齐：所有 rank 从 rank 0 拿一份
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
        # 2) 每个参数挂 post-accumulate hook，梯度写好后触发
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._on_grad)

    def _on_grad(self, param):
        param.grad.div_(self.world)                        # 先除，再 SUM = 平均
        self._works.append(dist.all_reduce(param.grad, async_op=True))

    def forward(self, *a, **kw):
        return self.module(*a, **kw)

    def finish_backward(self):
        """在 loss.backward() 之后、optimizer.step() 之前调用。"""
        for w in self._works:
            w.wait()
        self._works.clear()

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    torch.manual_seed(dist.get_rank())     # 故意每卡不同：数据不同，初始化靠 broadcast 对齐
    model = MiniDDP(nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 8)).cuda())
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(5):
        x = torch.randn(16, 64, device="cuda")
        loss = model(x).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.finish_backward()        # 等梯度 AllReduce 完成
        opt.step()

    # 校验：所有 rank 的参数应完全一致
    w = next(model.module.parameters()).data
    ref = w.clone(); dist.broadcast(ref, src=0)
    assert torch.equal(w, ref), "参数发散了"
    if local_rank == 0:
        print("ok")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

`torchrun --nproc_per_node=2 mini_ddp.py` 运行。这个版本相比真 DDP 缺三样东西，正好是加分回答：*没有分桶*（每个参数一次 AllReduce，小消息太多）、*没有 buffer 同步*、*没有 unused parameter 处理*。

#note[
  用 `register_post_accumulate_grad_hook`（torch 2.1+）而不是 `p.register_hook`：后者拿到的是"即将被累加的梯度"，在梯度累积场景下会在 `p.grad` 更新*之前*触发，语义不对。前者保证 `p.grad` 已经写好。
]

== DDP 常见坑清单

#warn[
  *1. 忘记 `torch.cuda.set_device(local_rank)`。* 所有进程挤在 `cuda:0`，表现是 GPU 0 OOM 或 NCCL 初始化 hang。第 17 章有详细说明。

  *2. 忘记 `sampler.set_epoch(epoch)`。* `DistributedSampler` 的 shuffle 种子是 `seed + epoch`，不调用就每个 epoch 洗出同样的顺序——训练看起来正常，但等价于反复看同一个排列的数据。
  ```python
  for epoch in range(E):
      sampler.set_epoch(epoch)     # 少这一行，静默降低数据多样性
      for batch in loader: ...
  ```

  *3. 不同 rank 走了不同分支 → collective 不匹配 → hang。* 典型是 `if loss.item() > thr:` 里做 log / eval / 存 checkpoint。判据必须先同步成全卡一致（第 17 章）。

  *4. 只在 rank 0 存 checkpoint 但没 barrier。* rank 0 在写盘（几十秒），其他 rank 已经冲进下一个 step 并发起 AllReduce → 超时。存完要 `dist.barrier(device_ids=[local_rank])`。反过来，*不要让所有 rank 都写同一个路径*，会互相截断文件。

  *5. `state_dict` 的 `module.` 前缀。* `DDP(model).state_dict()` 的 key 全带 `module.` 前缀，直接 `load_state_dict` 到裸 model 会 key 不匹配。存的时候用 `model.module.state_dict()`。

  *6. 各 rank 的 batch 数不同 → 最后一步 hang。* 数据集不能被 `world_size × batch_size` 整除时，快的 rank 已经退出循环，慢的 rank 还在 AllReduce。三种修法：`DistributedSampler(..., drop_last=True)`（丢尾部数据）、`DataLoader(..., drop_last=True)`、或者用 `model.join()` context（让提前结束的 rank 继续参与 shadow collective）。
  ```python
  with model.join():
      for batch in loader:
          loss = model(batch).mean(); loss.backward(); opt.step()
  ```

  *7. eval 时忘记关梯度同步。* `with torch.no_grad():` 下没有 backward，DDP 不会通信，但 `broadcast_buffers=True` 时 forward 前仍有 broadcast。eval 直接用 `model.module` 更干净。
]

== 扩展性与大 batch

DDP 的通信量 $2(N-1)\/N dot D$ 与 $N$ 几乎无关，但*延迟*不是：Ring 有 $2(N-1)$ 步，每步一个网络往返，所以 $N$ 越大 AllReduce 的固定延迟越高（NCCL 用 Tree 算法把它压到 $O(log N)$）。这是 DDP 在千卡规模仍能工作、但边际收益递减的原因。

DDP 扩卡是 *weak scaling*：卡数翻倍，global batch 也翻倍。GBS 变了就要重调超参：

#formula[$ "lr" = "lr"_"base" times ("GBS") / ("GBS"_"base") $]

这是 linear scaling rule（Goyal et al. 2017）：batch 翻倍则梯度的方差减半，可以走更大的步子。两个必须配套的东西：

- *warmup*：训练开头参数变化剧烈，直接用放大后的 LR 会炸。前几百到几千步从很小的 LR 线性升到目标值。
- *上限*：linear scaling 在 batch 特别大时失效（ResNet-50 上大约 8k 之后）。LLM 预训练更常用 $"lr" prop sqrt("GBS")$ 或者干脆按经验表查。

大 batch 还有个泛化问题：*同样的 epoch 数下，大 batch 的 optimizer step 次数更少*，模型倾向收敛到更"尖"的极小值，测试集表现变差。实践中靠 LR warmup + 更长训练 + LARS / LAMB 这类 layer-wise 自适应优化器缓解。更细的推导见仓库的《大模型分布式训练面试通关手册》。

== 面试考点

#interview[
  *Q1*：DDP 里 backward 和 AllReduce 怎么 overlap？

  A：DDP 在构造时把参数按*反向顺序*分成 bucket（默认 25 MB），给每个参数挂 autograd hook。某个参数的梯度算完就触发 hook 标记 ready 并拷进 bucket buffer；一个 bucket 全 ready 就立刻发 `async_op=True` 的 AllReduce，*不等 backward 结束*。因为 backward 是从最后一层往前算，最先就绪的 bucket 有整个 backward 的时间窗口去藏它的通信。backward 末尾统一 `wait()` 所有 handle。
]

#interview[
  *Q2*：为什么 bucket 要按参数的反向顺序分？

  A：overlap 的可用窗口 = 从 bucket 就绪到 backward 结束的时间。backward 从 loss 往输入走，最后一层的梯度最先出来，所以把最后几层的参数放在第 0 号 bucket，它就在 backward 刚开始时就绪，窗口最大。如果按 `model.parameters()` 的正向顺序分，第 0 号 bucket 要等 backward 走到第一层才齐，那时已经没有计算可以重叠了，退化成 naive 实现。
]

#interview[
  *Q3*：`nn.DataParallel` 为什么被废弃？

  A：四点。单进程多线程被 GIL 卡住 kernel launch；所有 output / loss / 梯度 gather 到 GPU 0，它成为算力、带宽和显存的三重瓶颈（经常只有 GPU 0 OOM）；每 step 要把更新后的参数 broadcast 回其他卡，纯浪费；gather 必须等 backward 全部结束，无法 overlap，且单进程不能跨机。DDP 用"每卡一进程 + 梯度 AllReduce"把这四条全解决了。
]

#interview[
  *Q4*：DDP 的梯度是平均还是求和？为什么？

  A：平均。要和"单卡跑 $N$ 倍大的 batch"等价——大 batch 的 loss 是样本均值，$cal(L) = 1/N sum_i cal(L)_i$，求梯度就是各 rank 梯度的算术平均。用 SUM 不除相当于把 LR 放大 $N$ 倍。DDP 内部自动做（`ReduceOp.SUM` 后除 world size，或直接用 NCCL 的 `ReduceOp.AVG`），所以脚本里 loss 不用手动除。前提是各 rank local batch 大小相同；LLM 里按 token 归一化时各 rank token 数不等，严格做法是先 AllReduce token 数。
]

#interview[
  *Q5*：`bucket_cap_mb` 调大调小分别有什么影响？

  A：单次 AllReduce 时间 $approx alpha + D\/B$，$n$ 个桶就是 $n alpha + D\/B$。调小 → 桶多，延迟项 $n alpha$ 累积、NCCL launch 开销上升。调大 → 单次通信效率高，但要等更多参数就绪，overlap 窗口变窄；极端情况一个桶装完整个模型就是 naive 实现。25 MB 是通用默认；7B+ 或跨机（$alpha$ 大）时 100–500 MB 通常更好，但必须用 profiler 实测确认 AllReduce 压在 compute 下面。
]

#interview[
  *Q6*：`find_unused_parameters=True` 为什么慢？有什么替代？

  A：DDP 判断 bucket 就绪的依据是"桶里每个参数的 hook 都触发过"，某参数没参与 forward 就永远不触发、AllReduce 永远不发、全体 hang。这个选项通过每 step 遍历一次 autograd 图找出未参与的参数并提前标 ready——CPU 开销 + 推迟第一批 AllReduce 的发出时机、overlap 变窄。替代方案按优先级：改模型让所有参数都在图里（MoE 常加一个乘 0 的 aux 项）；参与集合固定的话用 `static_graph=True`（第一步记录后复用，不再遍历）；真动态才开它。
]

#interview[
  *Q7*：`no_sync()` 做什么？为什么这样是对的？

  A：梯度累积时让前 $K-1$ 个 micro-step 只累加梯度、不发 AllReduce，只在最后一步同步一次，省掉 $K-1$ 次全模型 AllReduce。正确性来自"梯度累加和 AllReduce 都是线性算子，交换顺序不影响结果"：先本地累加再 AllReduce 一次，等于每步都 AllReduce 再累加。两个配套点：loss 要除 `ACCUM` 保持均值语义；`clip_grad_norm_` 只能在最后一步、AllReduce 之后做。
]

#interview[
  *Q8*：DDP 下用 BatchNorm 有什么问题？`SyncBatchNorm` 代价是什么？

  A：BN 只用本卡 local batch 算 mean / var，8 卡每卡 MBS=4 时 BN 看到的是 batch 4 而不是 32，统计噪声大伤收敛。`SyncBatchNorm.convert_sync_batchnorm(model)`（要在包 DDP 之前调）让统计量跨卡同步。代价是*每个 BN 层每次 forward 一次 AllReduce*：ResNet-50 有 53 个 BN 层就是 53 次小消息通信，受延迟主导，而且在 forward 里*没有 backward 计算可以拿来 overlap*。per-GPU batch 很小时才值得。LLM 用 LayerNorm，没这个问题。
]

#interview[
  *Q9*：为什么不同 rank 的 batch 数不同会 hang？怎么修？

  A：AllReduce 要求组内所有 rank 都调用。数据集不能被 `world_size × batch_size` 整除时，数据少的 rank 先退出循环，多的 rank 还在发 AllReduce，等不到配对就一直等到超时。修法三种：`DistributedSampler(drop_last=True)` 或 `DataLoader(drop_last=True)` 丢掉尾部数据；或者用 `with model.join():`，让提前结束的 rank 继续参与 shadow collective，直到所有 rank 都跑完。
]

#interview[
  *Q10*：DDP 扩到更多卡时，为什么要调学习率？

  A：DDP 是 weak scaling——卡数翻倍 global batch 也翻倍。GBS 变大后梯度方差变小，可以走更大的步子，linear scaling rule 说 $"lr" prop "GBS"$（Goyal 2017），必须配 warmup（开头几百到几千步线性升 LR，否则放大后的 LR 会在参数剧变阶段炸掉）。这个规则在 batch 特别大时失效，LLM 预训练更常用 $sqrt("GBS")$ 缩放。另外同 epoch 数下大 batch 的 step 数更少，泛化会变差，需要更长训练或 LARS / LAMB 这类 layer-wise 优化器。
]
