#import "../template.typ": *

= 显存：allocator、峰值与 OOM

显存题是训练框架岗位的必考题，而且是最容易分辨"真练过"和"只看过博客"的一道。面试官问的不是 `torch.cuda.empty_cache()` 怎么调，而是：你的模型为什么占这么多、峰值在哪一步、`reserved` 比 `allocated` 大 20 GB 说明什么、为什么明明还剩 30 GB 却报 OOM。这一章把显存拆成"算得出来的静态部分"和"必须测出来的动态部分"，然后讲 caching allocator 的机制、碎片化的成因，最后给一套 OOM 排查流程。省显存的分布式手段（ZeRO / FSDP）只在这里列出定位，细节见第 19 章。

== 显存都花在哪

一张卡上的显存分成六块，性质完全不同：

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*成分*], [*大小*], [*随 batch 变化*],
  [参数], [由参数量 $P$ 与 dtype 决定], [否],
  [梯度], [与参数同 shape], [否],
  [优化器状态], [Adam 是 2 份 moment（+ fp32 master）], [否],
  [激活], [$prop$ batch $times$ seq $times$ hidden $times$ layers], [*是*],
  [临时 buffer], [kernel workspace、AllReduce 的 flat buffer、`cat` 的输出], [部分],
  [CUDA context + NCCL], [每进程约 0.5--1 GB，加每个 communicator 的 buffer], [否],
)

前三块可以纯手算，这是面试白板上要能当场推的。用 $P$ 表示参数个数，单位是 byte：

#formula[
  $ M_"static" = underbrace(2 P, "bf16 参数") + underbrace(2 P, "bf16 梯度") + underbrace(4 P, "fp32 master") + underbrace(8 P, "Adam " m\, v) = 16 P $
]

有意思的是，纯 fp32 训练 + Adam 也是 $4P + 4P + 8P = 16 P$——AMP 省的不是这块，省的是激活和算力。真正把 $16 P$ 压下去要动优化器：纯 bf16 训练（不留 master weight、moment 也用 bf16）是 $8 P$，8-bit optimizer 是 $2P + 2P + 4P + 2P approx 10 P$。

代进 7B：$16 times 7 times 10^9 = 112$ GB。

#figure(
  align(center, mem-bar((
    ("params bf16", 14),
    ("grads bf16", 14),
    ("fp32 master", 28),
    ("Adam m", 28),
    ("Adam v", 28),
  ))),
  caption: [7B 模型 bf16 混合精度 + AdamW 的静态显存分解，合计 112 GB。*一张 80 GB 的 A100 装不下一个 7B 的训练态*，激活还没算。这就是 ZeRO / FSDP 存在的理由。],
) <fig-mem-7b>

#insight[
  面试标准答案的骨架：静态显存 $approx 16 P$ byte（bf16 + AdamW），激活显存另算且与 batch 线性相关。7B 在单卡 80 GB 上放不下训练态，但推理只要 $2 P = 14$ GB——这个对比能一句话说清"训练比推理贵在哪"。
]

== 激活显存为什么是大头

激活是 forward 里被 autograd 存下来、等 backward 用的中间张量（机制见第 6 章）。它随 batch 线性涨，所以是唯一一个"你能调"的大头。

一个 Transformer layer 大约要存这些：LayerNorm 的输入与 `rstd`、QKV 投影的输入、attention 的输出、FFN 的输入与中间的 $4h$ 张量、GELU 的输入、dropout mask。粗算成"若干份 $b times s times h$"：

#formula[
  $ A_"layer" approx c dot b dot s dot h dot "sizeof(dtype)", quad c approx 10 tilde.op 20 $
]

$c$ 取决于实现细节（有没有 fuse、存不存 GELU 输入），量级上取 16 就够做估算。代进 $b=4$、$s=4096$、$h=4096$、bf16、32 层：

$ 16 times 4 times 4096 times 4096 times 2 "B" approx 2.1 "GB" quad "每层" arrow.r quad times 32 approx 69 "GB" $

激活比全部静态显存还大。而且注意这里没有 $b dot a dot s^2$ 那一项——那是手写 attention 把 $s times s$ 的 score 矩阵 materialize 出来时才有的，$s=4096$、$a=32$、$b=4$ 时它单独就是 $4 times 32 times 4096^2 times 2 "B" approx 4.3$ GB 每层。用 `F.scaled_dot_product_attention`（FlashAttention 后端）就没有这一项，这是 SDPA 最大的显存收益，比它的速度收益更值得在面试里提。

#note[
  这些系数是量级估算，不是测量值。真要知道自己模型的激活占多少，跑一个 step 然后 `max_memory_allocated() - 静态部分`，或者直接看下面的 memory snapshot。
]

== caching allocator：为什么 `empty_cache()` 通常没用

`cudaMalloc` / `cudaFree` 是同步调用，一次几十到几百微秒，而训练一个 step 要分配几千次张量。所以 PyTorch 自己实现了一层 caching allocator：向驱动一次性申请大块 segment，之后在这些 segment 里切 block 给你，释放时*不还给驱动*，只放回自己的空闲链表。

于是有了两个数，这是最高频的显存题：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [`memory_allocated()`], [你的张量真正占用的字节数，allocator 记账得到],
  [`memory_reserved()`], [allocator 从 CUDA 驱动手里拿到的总字节数],
)

`reserved >= allocated` 恒成立。`nvidia-smi` 看到的是 `reserved` 再加 CUDA context，所以它总比 `memory_allocated()` 大一截——这不是泄漏。

allocator 的三个关键设计：

- *按 size 分池*。小于 1 MB 的请求走 small pool（从 2 MB 的 segment 里切），大于 1 MB 走 large pool（20 MB 或按需的 segment）。block 大小向上圆整到 512 B 的倍数。分池是为了让大量小张量不去污染大块连续空间。
- *stream-aware*。每个 block 记着自己是在哪个 stream 上分配的。只有同 stream 的后续请求才能直接复用它，跨 stream 复用必须先同步——这条在第 9 章的 `record_stream()` 那节会咬人。
- *不还给驱动*。所以第二个 step 起几乎不再调 `cudaMalloc`，稳态下分配是纯用户态操作。

`torch.cuda.empty_cache()` 做的事就是把空闲 segment 还给驱动。它*不会*减少 `memory_allocated`，只会减少 `memory_reserved`。什么时候有用：你要在同一进程里把显存让给别的库（比如 vLLM、TensorRT、另一个 CUDA context）。什么时候没用（也就是绝大多数情况）：训练 loop 里为了"防 OOM"而周期性调用——它只是把缓存丢掉，下个 step 再重新 `cudaMalloc` 拿回来，纯亏时间。

#warn[
  在训练 loop 里每 step 调 `torch.cuda.empty_cache()` 是典型的 cargo cult。它引入同步、把 allocator 的缓存清空，让每个 step 都要重新走 `cudaMalloc`，step time 明显变差，而且并不能解决真正的峰值超限。唯一合理的用法是"阶段切换"处：训练结束、开始 eval 之前，或者要交显存给别的进程/库时调一次。
]

== 碎片化

判据只有一条：*`reserved` 明显大于 `allocated`（比如差 20% 以上）就是碎片*。

典型报错长这样：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 79.14 GiB of which 5.62 GiB is free.
Process ... has 73.52 GiB memory in use. Of the allocated memory
68.10 GiB is allocated by PyTorch, and 1.83 GiB is reserved by PyTorch
but unallocated.
```

关键在最后一句：`reserved but unallocated` 有 1.83 GB，`free` 还有 5.62 GB，却申请不到 2 GB 连续空间。这就是碎片——空闲字节总量够，但没有一块*连续*的够大。

成因几乎总是"分配大小在变"：

- 变长 seq：`s` 每个 batch 都不同，激活张量的大小跟着变，旧 block 的尺寸和新请求对不上
- 动态 batch / bucketing dataloader
- eval 用的 batch 和 train 不一样大，两套尺寸交替
- 中途插入的大临时分配（一次 `torch.cat` 出一个巨大张量）把 segment 切碎

对策一是 `expandable_segments`：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

它把固定大小的 segment 换成"可以往后扩展的虚拟地址区间"（用 CUDA 的 virtual memory management API）。这样一个 segment 能随需求长大，而不是"要 2.5 GB 但手里只有一堆 2 GB 的 segment"。变长 shape 的场景下这个开关经常是零成本的净收益，值得默认打开试一次。

其他可调项（同一个环境变量，逗号分隔）：`max_split_size_mb:N` 禁止把大于 N MB 的 block 切开给小请求；`garbage_collection_threshold:0.8` 在 reserved 超过 80% 上限时主动回收空闲 block 而不是直接 OOM。

#note[
  `expandable_segments` 在较新版本里已经相当稳定，但它改变的是底层分配方式，遇到过与某些自定义 CUDA 扩展（自己拿 pointer 做 IPC 的）不兼容。上线前测一遍功能正确性。
]

== 排查工具

第一层是计数器，插在训练 loop 里几乎零成本：

```python
import torch

torch.cuda.reset_peak_memory_stats()
train_one_step()
print(f"cur  alloc {torch.cuda.memory_allocated()  / 2**30:.2f} GiB")
print(f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"cur  resv  {torch.cuda.memory_reserved()   / 2**30:.2f} GiB")
print(f"peak resv  {torch.cuda.max_memory_reserved()  / 2**30:.2f} GiB")
```

`reset_peak_memory_stats()` 每 step 清零，才能定位"峰值出现在哪一步"而不是"整个 job 的峰值"。`torch.cuda.memory_summary()` 打一张按 size bin 分的详细表，`torch.cuda.memory_stats()` 返回同样内容的 dict，里面 `num_alloc_retries` 这个 key 特别有用：它大于 0 就说明 allocator 曾经因为拿不到空间而被迫 `empty_cache()` 重试，这是碎片的直接证据。

第二层是 memory snapshot，能看到"每一块显存是哪行 Python 代码分配的"：

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=200_000)

for step in range(3):            # 跑几个 step 到稳态就够了
    train_one_step()

torch.cuda.memory._dump_snapshot("mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)   # 关掉
```

把 `mem.pickle` 拖到 #link("https://pytorch.org/memory_viz")[pytorch.org/memory_viz]（纯前端，文件不上传）。这张图的横轴是时间、纵轴是显存，每个色块是一次分配，点进去有完整的 Python 调用栈。你要在图里找三样东西：最高的那根柱子（峰值时刻）、峰值时最厚的那几层（谁占的）、还有*从头贯穿到尾的横条*（这就是泄漏：一块显存从分配到 dump 都没释放）。

#warn[
  `_record_memory_history` 是带下划线的私有 API，签名在版本间变过（`enabled` 参数取 `"all"` / `"state"` / `None`）。它开着的时候会记录每次分配的调用栈，有明显开销，*只在排查时开*，不要留在生产训练脚本里。
]

== 省显存的手段，按性价比排序

#ladder(
  ([`set_to_none=True`], [`zero_grad(set_to_none=True)` 让 grad 张量真正释放而不是填 0],
   [省 $2P$–$4P$，零代价]),
  ([AMP / bf16], [参数与激活都减半，激活是大头所以收益大], [省激活约一半，还更快]),
  ([SDPA / FlashAttention], [不 materialize $s times s$ 的 score 矩阵],
   [省掉 $b a s^2$ 那一项]),
  ([micro-batch + 梯度累积], [激活与 batch 线性相关，切小就线性下降], [几乎零代价]),
  ([activation checkpointing], [只存 layer 边界，backward 时重算],
   [激活降一个数量级，算力 +30%]),
  ([8-bit optimizer], [moment 用 int8 + 分块量化存], [$16P arrow.r 10P$]),
  ([参数 / 优化器 offload], [把 fp32 master 与 moment 放 CPU], [省 $12P$，受 PCIe 限制]),
  ([ZeRO / FSDP], [参数、梯度、优化器状态按 DP 维度切开], [静态显存除以 DP 度，见第 19 章]),
)

前四条应该无脑上，第五条要算一下值不值，后三条是"实在装不下才动"的手段。面试里被问"你怎么省显存"，按这个顺序讲就是满分答案——重点是*先说哪些是零成本的*，再说哪些是拿算力/带宽换显存的。

== activation checkpointing 的取舍

原理：forward 时对被包住的段落不存中间激活，只存输入；backward 需要时重跑一遍 forward 把激活算回来。

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# 在模型的 forward 里逐层包
for blk in self.blocks:
    x = checkpoint(blk, x, use_reentrant=False)
```

`use_reentrant=False` 必须显式写。老的 reentrant 实现不支持在 checkpoint 段里用 `.grad` / hook，也不支持部分输出不需要梯度，且要求输入必须有 `requires_grad`；新实现基于 `TorchDispatchMode`，限制少得多。torch 2.x 里不传这个参数会 warning，未来默认值会翻转。

代价：多一遍 forward。整层 checkpoint 的话，backward 原本约等于 2 倍 forward 的计算量，现在变成 3 倍 forward，所以总计算量涨约 $1/3$，实测通常落在 +25% 到 +35%（取决于被重算的部分是 memory-bound 还是 compute-bound）。收益：每层激活从 $c dot b s h$ 降到约 $1 dot b s h$，上面那个 69 GB 的例子降到约 4.3 GB。

*selective checkpointing* 是更好的折中：只重算便宜的 op，把贵的 op（matmul、attention）的输出留下来。norm、GELU、dropout 这些 elementwise/memory-bound op 重算几乎不花时间，但它们的激活占了相当比例；反过来 matmul 重算最贵。

```python
import torch
from torch.utils.checkpoint import (
    checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy,
)

_save_ops = {torch.ops.aten.mm.default, torch.ops.aten._scaled_dot_product_flash_attention.default}

def policy_fn(ctx, op, *args, **kwargs):
    if op in _save_ops:
        return CheckpointPolicy.MUST_SAVE      # 贵的：存下来
    return CheckpointPolicy.PREFER_RECOMPUTE   # 便宜的：重算

def ctx_fn():
    return create_selective_checkpoint_contexts(policy_fn)

x = checkpoint(blk, x, use_reentrant=False, context_fn=ctx_fn)
```

工程上更常见的粗粒度版本是"每 $k$ 层 checkpoint 一层"（`if i % 2 == 0`），实现两行就够，效果是显存和算力都取中间值。面试里能说出"selective 而不是全 checkpoint"就已经比大多数人细了。

#insight[
  checkpointing 是用算力买显存。买它的唯一正当理由是：省下来的显存能让你把 batch 开大，而更大的 batch 带来的 GPU 利用率提升 > 重算多花的 30% 算力。如果开了 checkpoint 但 batch 没变大，你就是纯亏。
]

== OOM 排查流程

按顺序做，不要跳步：

+ *看报错里的三个数*。`Tried to allocate X`、`free Y`、`reserved but unallocated Z`。如果 $Y$ 很小 → 真的不够，走第 2 步；如果 $Y$ 不小但 $X > $ 最大连续块 → 碎片，走第 5 步。
+ *确认是稳态峰值还是某一步的尖峰*。每 step 打 `max_memory_allocated()` 并 `reset_peak_memory_stats()`。如果第 1 个 step 就 OOM，是静态部分算错了；如果跑了几百步才 OOM，是某种累积。
+ *看 snapshot 找最大分配和贯穿到底的横条*。峰值通常出现在 backward 刚开始（激活最多的时刻）或者 optimizer step 里（要临时开与参数同 shape 的 buffer）。
+ *检查有没有累积带图的张量*。这是"跑几百步才 OOM"的头号原因，见下一节。
+ *碎片处理*：先开 `expandable_segments:True`，再考虑把变长 seq pad 到几个固定档位（bucketing），让分配尺寸收敛到有限几种。
+ *检查 eval / 推理路径*。忘了 `torch.no_grad()` 或 `model.eval()`，eval 就会像训练一样存激活，而 eval 的 batch 往往更大。
+ *检查 dataloader*。`pin_memory=True` 占的是 CPU 内存不是显存；但如果你在 `Dataset.__getitem__` 里 `.cuda()`，或者把整个数据集 `.to('cuda')` 预加载，那就是显存。
+ *还是不够*，才动第 19 章的分布式手段。

== 那些不是泄漏的"泄漏"

Python 有 GC，CUDA 显存也由 allocator 管，*真正的显存泄漏在 PyTorch 里非常罕见*。你遇到的 99% 是"你自己拿着引用不放"。三种典型写法：

#warn[
  *1. 把 `loss` 而不是 `loss.item()` 存进 list。* `loss` 带着 `grad_fn`，`grad_fn` 引用着整张 autograd 图，图引用着这个 step 的*全部激活*。存 1000 个 step 就是 1000 份激活。

  ```python
  losses.append(loss)             # 泄漏：整张图被保活
  losses.append(loss.item())      # 对：取标量（有一次同步，见第 9 章）
  losses.append(loss.detach())    # 也对：想留在 GPU 上做平均时用这个
  ```

  *2. 把带 grad 的 tensor 存进外部容器 / buffer。*

  ```python
  self.cache = hidden              # 泄漏：hidden.grad_fn 保活整张图
  self.cache = hidden.detach()     # 对
  ```

  同理，accumulate 指标要写 `total += loss.detach() * n`，写成 `total += loss * n` 会把每个 step 的图串成一条越来越长的链。

  *3. hook 里持有引用。*

  ```python
  acts = {}
  def hook(mod, inp, out):
      acts[mod] = out              # 泄漏：out 一直被 dict 拿着
  handle = m.register_forward_hook(hook)
  # 排查完必须 handle.remove()，且存 out.detach().cpu()
  ```

  hook 的 handle 不 `remove()` 也是常见的隐性泄漏源——尤其是在一个循环里反复注册。
]

还有两个容易忽略的：`torch.cuda.max_memory_allocated()` 记的是历史峰值，看着不降不代表当前占着；异常处理里 `except` 块捕获的 traceback 会持有栈帧，栈帧里的局部变量（包括那个巨大的 activation）跟着被保活，这就是"OOM 之后重试还是 OOM"的原因，处理办法是在 `except` 里先 `del` 掉相关变量。

== 面试考点

#interview[
  *Q1*：`memory_allocated` 和 `memory_reserved` 有什么区别？为什么 `nvidia-smi` 的数比 `memory_allocated` 大？

  A：`allocated` 是活着的张量占的字节；`reserved` 是 PyTorch caching allocator 从驱动那里申请到的总量，包含它缓存起来没还的空闲 block。`nvidia-smi` 看到的约等于 `reserved` 加 CUDA context（每进程约 0.5--1 GB）加 NCCL buffer，所以必然更大。两者差得多说明缓存里有大量空闲 block，即碎片。
]

#interview[
  *Q2*：`torch.cuda.empty_cache()` 什么时候有用？

  A：几乎只有一种场景有用——要把显存让给同进程里的其他库或其他 CUDA context。它把空闲 segment 还给驱动，只降 `reserved`，不降 `allocated`，也不能降低你的峰值需求。放在训练 loop 里每 step 调是负优化：清掉缓存后每步都要重新走同步的 `cudaMalloc`。
]

#interview[
  *Q3*：报 "tried to allocate 2 GiB" 但 `nvidia-smi` 显示还剩 6 GB，怎么解释？

  A：碎片化。空闲字节的*总量*够，但没有一块连续的 2 GiB。看报错里的 `reserved but unallocated` 和 `memory_stats()["num_alloc_retries"]` 确认。成因是分配尺寸在变（变长 seq、动态 batch）。对策：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或把 seq 长度 bucketing 到有限档位。
]

#interview[
  *Q4*：估算一个 7B 模型 bf16 混合精度 + AdamW 训练要多少显存。

  A：静态部分 $16P$ byte：bf16 参数 $2P$ + bf16 梯度 $2P$ + fp32 master $4P$ + Adam 的 $m$ 和 $v$ 各 $4P$，代进 7B 是 112 GB。加上激活（与 batch $times$ seq 线性相关）和约 1 GB 的 context。结论：单张 80 GB A100 装不下，必须 ZeRO/FSDP 切分或 offload。推理只要 $2P = 14$ GB。
]

#interview[
  *Q5*：训练跑了几百个 step 才 OOM，第一个怀疑对象是什么？

  A：累积了带 `grad_fn` 的张量。最经典的是 `losses.append(loss)` 而不是 `loss.item()`——`loss` 引用整张 autograd 图，图引用整个 step 的激活。同类还有 `total_loss += loss`（应该 `+= loss.detach()`）、forward hook 把 `out` 存进 dict 不 `detach`、module 上挂 `self.cache = hidden`。用 memory snapshot 找"从头贯穿到尾的横条"能一眼定位。
]

#interview[
  *Q6*：activation checkpointing 省多少、贵多少？什么时候不该用？

  A：每层激活从"十几份 $b s h$"降到约"一份 $b s h$"，量级上是一个数量级；代价是 backward 时多跑一遍 forward，总计算量涨约三分之一（实测常见 +25%--35%）。不该用的情况：显存本来够用，或者省下的显存没被用来加大 batch——那就是纯亏算力。更优的做法是 selective checkpointing，只重算 norm/GELU 这类 memory-bound op，保留 matmul 和 attention 的输出。
]

#interview[
  *Q7*：caching allocator 为什么是 stream-aware 的？

  A：每个 block 记录分配它的 stream。释放后只能被同 stream 的请求直接复用，因为 allocator 只能保证同 stream 内的顺序执行。跨 stream 复用需要同步。这带来一个实际的坑：在非默认 stream 上创建、在默认 stream 上使用的张量，必须调 `tensor.record_stream(stream)`，否则它可能在消费 kernel 还没跑完时就被回收复用，产生静默的数据损坏。详见第 9 章。
]

#interview[
  *Q8*：`zero_grad(set_to_none=True)` 和 `set_to_none=False` 差在哪？

  A：`False` 是把 `.grad` 张量原地填 0，张量还占着显存；`True` 是把 `.grad` 置为 `None`，张量被释放，下次 backward 时重新分配。前者省一次分配但常占 $2P$--$4P$ 显存，后者省显存。torch 2.0 起默认值就是 `True`。副作用要知道：`grad` 变成 `None` 后，任何直接读 `p.grad` 做统计的代码要处理 `None`；带 momentum 的 SGD 在"某参数这步没有梯度"时，两种模式的更新行为会不一样（`None` 时跳过，填 0 时仍然按 momentum 更新）。
]
