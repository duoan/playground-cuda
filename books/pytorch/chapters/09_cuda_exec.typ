#import "../template.typ": *

= CUDA 执行模型：异步、stream、同步点

这一章解释一件事：*你的 Python 代码和 GPU 上真正发生的事情，在时间上是错开的*。`y = a @ b` 返回时，那个 matmul 很可能一个 SM 都还没碰。理解这一点之后，一连串面试题就都通了——为什么你用 `time.time()` 测的 kernel 时间是假的、为什么 `loss.item()` 会让训练变慢、为什么通信能和计算重叠、CUDA graph 到底省了什么。这些是 AI Infra 面试的白板常客，尤其"写一段正确的 GPU 计时代码"几乎必考。

== kernel launch 是异步的

CPU 调一次 `torch.mm`，走完 dispatcher（第 7 章）之后最终是一次 `cudaLaunchKernel`。这个调用把 kernel 塞进 stream 的队列就立刻返回，*不等 GPU 执行*。

#figure(
  align(center, flow-boxes(boxes: (
    "Python op", "dispatcher", "cudaLaunchKernel", "stream queue", "SM 执行",
  ), box-w: 2.4)),
  caption: [一次 op 的路径。前三步在 CPU 上，返回后 CPU 就继续往下跑；后两步在 GPU 上异步发生。稳态下 CPU 通常领先 GPU 几百次 launch。],
) <fig-launch>

这个设计是为性能服务的：单次 launch 的 CPU 开销大约几微秒，而一个 kernel 在 GPU 上可能跑几十到几百微秒。异步让 CPU 得以"跑在前面"，把队列填满，GPU 就不会因为等 CPU 而空转。

直接后果是*你不能用 CPU 时钟测 GPU 时间*：

```python
import time, torch

x = torch.randn(4096, 4096, device="cuda")

t0 = time.perf_counter()
y = x @ x
t1 = time.perf_counter()
print(f"{(t1 - t0) * 1e3:.3f} ms")   # 测到的是 launch 开销，不是 matmul 时间
```

这段代码在 A100 上会打出几十微秒级的数字，而一个 $4096^3$ 的 fp32 matmul 显然不可能这么快。它测的是"把 kernel 塞进队列"的时间。

正确的计时模板（这是白板题的标准答案，背下来）：

```python
import torch

def bench(fn, warmup: int = 20, iters: int = 100) -> float:
    """返回 fn 的中位数耗时（毫秒）。fn 不应该包含 CPU 同步。"""
    for _ in range(warmup):          # 1) warmup：JIT、cudnn 选算法、allocator 稳态
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()               # 2) event 打在 stream 里，测的是 GPU 时间
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))   # 毫秒，float

    samples.sort()                   # 3) 取中位数，不取平均：平均被离群值污染
    return samples[len(samples) // 2]

print(bench(lambda: x @ x))
```

三个要点缺一不可：*warmup*（第一次调用包含 cudnn/cublas 的算法选择、Triton 编译、allocator 首次 `cudaMalloc`）、*用 CUDA event 或 `synchronize()`*、*多次取中位数*。第 11 章会给一个更省事的封装 `torch.utils.benchmark.Timer`，它自动处理这三件事，日常测量优先用它。

#insight[
  CUDA event 记录的是"GPU 走到 stream 里这个位置的时刻"，所以 `start.elapsed_time(end)` 是纯 GPU 时间，不含 CPU launch 开销。如果你想测的恰恰是"CPU 会不会成为瓶颈"，那就要用 wall-clock 测整个 loop 并对比 GPU 时间——两者的差就是 CPU 侧的开销。
]

== 隐式同步点

异步是默认行为，但很多操作会强制 CPU 等 GPU。这张表是面试直接问的：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*操作*], [*为什么同步*],
  [`t.item()`], [要把标量搬回 CPU，必须等产生它的 kernel 跑完],
  [`t.cpu()` / `t.to("cpu")`], [D2H 拷贝，且目标是非 pinned 内存时必然同步],
  [`t.numpy()` / `t.tolist()`], [同上，本质是 D2H],
  [`print(t)`], [格式化要读取数值 → D2H],
  [`if t:` / `bool(t)` / `float(t)`], [Python 需要真值 → D2H],
  [`t.nonzero()` / `t[mask]`], [输出 shape 依赖数据，CPU 要知道 shape 才能分配],
  [`torch.cuda.synchronize()`], [显式等整个 device],
  [`Event.synchronize()` / `Work.wait()`], [显式等一个点],
  [allocator 拿不到空间], [触发 `cudaFree` 重试，`cudaFree` 是同步的],
  [`assert` / `torch.isnan(t).any().item()`], [`.item()` 的伪装形式，最容易漏],
)

`t.nonzero()` 这类"输出 shape 依赖数据"的 op 是最不容易想到的一类。同理还有 `torch.unique`、`torch.masked_select`、以及任何用 tensor 值做 Python 控制流的地方。

*为什么 `loss.item()` 会拖慢训练*：稳态下 CPU 领先 GPU 几百次 launch，队列是满的。`loss.item()` 一来，CPU 必须停下来等 GPU 把这个 step 的所有 kernel 排干；等它拿到值继续往下跑时，*队列已经空了*，GPU 只能一边等 CPU 重新填队列一边零星地干活。损失的时间约等于"CPU 发完一个 step 的 launch 所需的时间"。小模型、小 batch 时这个数很可观。

```python
# 差：每 step 一次同步
for x, y in loader:
    loss = train_step(x, y)
    logger.log(loss.item())

# 好：在 GPU 上累积，按 log 间隔（比如 50 step）才同步一次
loss_acc = torch.zeros((), device="cuda")
for i, (x, y) in enumerate(loader):
    loss = train_step(x, y)
    loss_acc += loss.detach()
    if (i + 1) % 50 == 0:
        logger.log(loss_acc.item() / 50)   # 50 步只同步 1 次
        loss_acc.zero_()
```

注意用 `loss.detach()` 而不是 `loss`，否则会把整张 autograd 图串起来（第 8 章）。

*怎么抓出自己代码里的隐式同步*：

```python
import torch

torch.cuda.set_sync_debug_mode("warn")    # 或 "error" 直接抛异常
# 底层等价写法：torch._C._cuda_setSyncDebugMode(1)

train_one_step()                          # 任何隐式同步都会报出调用位置

torch.cuda.set_sync_debug_mode("default")
```

`"error"` 模式在排查"这个 step 里到底谁在同步"时最好用——它直接给你一个带完整 Python 栈的异常。注意 `torch.cuda.synchronize()` 这类*显式*同步不会被它报出来，它只管隐式的。

#warn[
  别把 `CUDA_LAUNCH_BLOCKING=1` 当性能工具。它让每次 launch 都同步，唯一用途是让 CUDA 报错的栈落在真正出错的那一行（否则异步执行下，报错会出现在之后某个不相关的 op 上）。它会让训练慢好几倍，调完必须去掉。
]

== stream：并行的执行队列

*Stream* 是一个 FIFO 的 kernel 队列。规则只有两条：同一个 stream 内的 kernel 严格按提交顺序执行；不同 stream 之间*没有*顺序保证，硬件资源够就并行跑。

PyTorch 默认把所有 op 提交到当前 device 的默认 stream，所以你写的代码天然是顺序语义的。要并行就得自己开 stream，并用 event 手动建立依赖关系：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*API*], [*语义*],
  [`s = torch.cuda.Stream()`], [新建一个 stream],
  [`with torch.cuda.stream(s):`], [块内的 op 提交到 `s`],
  [`torch.cuda.current_stream()`], [当前 stream],
  [`s2.wait_stream(s1)`], [`s2` 后续的 kernel 等 `s1` 当前已提交的全部完成],
  [`e = torch.cuda.Event(); e.record(s1)`], [在 `s1` 里打一个点],
  [`s2.wait_event(e)`], [`s2` 等这个点],
)

`wait_stream` 和 `wait_event` 都是*在 GPU 上*建立依赖，CPU 不阻塞——这是它们和 `synchronize()` 的根本区别。

下面是一个真能跑的 overlap 例子：把"搬下一个 batch 上 GPU"和"算当前 batch"重叠起来。这是 dataloader 之后最常见的一处 overlap 机会。

```python
import torch

copy_stream = torch.cuda.Stream()

def prefetch(batch_cpu: torch.Tensor) -> torch.Tensor:
    """在 copy_stream 上异步搬运。batch_cpu 必须是 pinned memory。"""
    with torch.cuda.stream(copy_stream):
        return batch_cpu.to("cuda", non_blocking=True)

it = iter(loader)
nxt = prefetch(next(it))

for _ in range(n_steps):
    cur = torch.cuda.current_stream()
    x = nxt
    cur.wait_stream(copy_stream)    # 主 stream 等这次拷贝完成
    x.record_stream(cur)            # 关键：告诉 allocator 主 stream 也在用 x

    try:                            # 先发起下一次拷贝，再开始计算
        nxt = prefetch(next(it))
    except StopIteration:
        nxt = None

    loss = model(x).sum()           # 与 nxt 的 H2D 拷贝并行
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
```

#figure(
  align(center, stack(dir: ttb, spacing: 16pt,
    timeline(streams: (
      ("compute", (("wait", 5), ("compute", 15), ("wait", 5), ("compute", 15))),
      ("copy",    (("comm", 5), ("wait", 15), ("comm", 5), ("wait", 15))),
    ), title: "无 prefetch：每 step 先等拷贝，再算"),
    timeline(streams: (
      ("compute", (("wait", 5), ("compute", 15), ("compute", 15))),
      ("copy",    (("comm", 5), ("comm", 5), ("wait", 10), ("comm", 5))),
    ), title: "有 prefetch：拷贝 k+1 与 compute k 并行"),
  )),
  caption: [两个 stream 的时间线（示意，比例不代表实测）。overlap 之后拷贝时间被藏进计算时间里，前提是拷贝时间 < 计算时间。],
) <fig-overlap>

#note[
  `dist.all_reduce(..., async_op=True)` *不需要*你自己开 stream——`ProcessGroupNCCL` 内部就有专属 stream，返回的 `Work` 对象调 `work.wait()` 是让当前 stream 去等它，而不是阻塞 CPU。所以 DDP 的 overlap 是自动的（第 18 章）。自己开 stream 主要用于拷贝、以及把两段互不依赖的计算并行。
]

== stream 与 allocator 的坑：`record_stream`

caching allocator 是 stream-aware 的（第 8 章）：每个 block 记着自己在哪个 stream 上分配。当这个 block 被释放时，allocator 认为"只要 *那个 stream* 上的后续操作能保证顺序，就可以立刻复用它"。

#warn[
  如果一个张量在 stream A 上分配、在 stream B 上被使用，而 Python 侧的引用在 B 的 kernel 跑完之前就消失了，allocator 会以为它自由了，把这块显存分给 A 上的下一个请求。于是 B 的 kernel 读到的是别人写进去的数据——*静默的数值错误，不报错、不崩，只是 loss 变成 NaN 或者悄悄训坏*。

  ```python
  # 错：x 在 copy_stream 上分配，在默认 stream 上使用
  with torch.cuda.stream(copy_stream):
      x = cpu_batch.to("cuda", non_blocking=True)
  torch.cuda.current_stream().wait_stream(copy_stream)
  y = model(x)          # 用了，但 allocator 不知道默认 stream 在用它

  # 对：显式登记
  with torch.cuda.stream(copy_stream):
      x = cpu_batch.to("cuda", non_blocking=True)
  torch.cuda.current_stream().wait_stream(copy_stream)
  x.record_stream(torch.cuda.current_stream())
  y = model(x)
  ```

  `record_stream(s)` 的语义是"这块显存要等 `s` 上当前已提交的工作全部完成之后才可以被复用"。规则很简单：*任何跨 stream 使用的张量都要 `record_stream`*。这类 bug 极难调试，因为它依赖时序，小规模测试往往复现不出来。
]

== pinned memory 与 `non_blocking=True`

主机内存默认是 pageable 的，操作系统可以随时把它换出去。DMA 引擎不能直接读这种内存，所以一次 pageable H2D 拷贝实际是"驱动先拷到内部的 staging buffer，再 DMA"——中间那步是同步的 CPU 拷贝。

*所以 `non_blocking=True` 只在 pinned（page-locked）内存上才真的异步。* 在 pageable 内存上传 `non_blocking=True` 不报错也不生效，这是最常见的"我设了却没效果"。

```python
x = torch.randn(1024, 1024)                     # pageable
x.to("cuda", non_blocking=True)                 # 实际同步

x = torch.randn(1024, 1024).pin_memory()        # pinned
x.to("cuda", non_blocking=True)                 # 真异步
```

`DataLoader(pin_memory=True)` 就是让 worker 把 batch 放进 pinned 内存，这样训练循环里的 `.to("cuda", non_blocking=True)` 才有意义。两者要*配对使用*，只开一个没用。

代价：pinned 内存不能被换出，占的是"操作系统不能动"的物理内存；`pin_memory()` 本身是一次同步的分配 + 拷贝，不便宜。所以不要在热路径里反复 `pin_memory()`，要么让 DataLoader 做，要么自己预分配一组固定的 pinned buffer 循环用。

D2H 方向同理：`t.to("cpu", non_blocking=True)` 要目标是 pinned 才异步，而且*返回后你不能立刻读它*——必须先同步。忘了这一点就会读到未写完的数据。

== CUDA graph：把一串 launch 固化成一次

每个 kernel 都要一次 `cudaLaunchKernel`，CPU 开销几微秒。一个 Transformer 的 step 有几千个 kernel，CPU 侧就是几毫秒到十几毫秒的固定开销。当 kernel 本身很小（小 batch、LLM 的 decode 阶段一次只算一个 token）时，*这个开销会成为瓶颈，GPU 大部分时间在等 CPU 发指令*。

CUDA graph 的做法是：录一遍，把整个 launch 序列（含它们之间的依赖）存成一张图；之后 replay 只有一次提交开销。

```python
import torch

model = MyModel().cuda().eval()

# 静态输入输出 buffer —— 地址在 capture 后就不能变了
static_x = torch.randn(8, 512, device="cuda")

# 1) warmup：必须在 side stream 上跑几步，让 cudnn/cublas 选好算法、
#    让 allocator 进入稳态，否则 capture 会把这些一次性行为录进去
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_y = model(static_x)
torch.cuda.current_stream().wait_stream(s)

# 2) capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(static_x)

# 3) replay：只改 buffer 的内容，不能换 buffer
for batch in batches:
    static_x.copy_(batch)      # 原地写入，地址不变
    g.replay()
    consume(static_y)          # static_y 每次 replay 后被覆盖，要用就先 copy 走
```

限制清单（面试会挨个问）：

- *shape 必须静态*。换 batch size 或 seq len 就要重新 capture 一张图。变长输入通常 pad 到几个固定档位，每档一张图。
- *地址必须固定*。输入输出都得是预分配的 buffer，用 `copy_` 灌数据。这也意味着 replay 之后 `static_y` 会被下一次 replay 覆盖。
- *不能有 CPU 同步*。被 capture 的代码里出现 `.item()`、`print(tensor)`、`torch.cuda.synchronize()` 会直接失败。
- *不能有数据依赖的控制流*。`if t.max() > 0:` 这种在 capture 时走的哪个分支就永远是哪个分支。
- *不能有随机性副作用之外的 CPU 逻辑*。RNG 是特殊处理过的（graph 会捕获 generator 状态），但普通 Python 逻辑在 replay 时根本不执行。

如果只是想给一个 `nn.Module` 上 graph，`torch.cuda.make_graphed_callables` 会帮你把 forward 和 backward 都 capture 好，可以直接参与 autograd：

```python
model = torch.cuda.make_graphed_callables(model, (static_x,), num_warmup_iters=3)
loss = model(static_x).sum()
loss.backward()                # backward 也是 replay
```

实践中你几乎不会手写这些——`torch.compile(mode="reduce-overhead")` 底层就是 CUDA graph，它顺带帮你处理了 buffer 管理和 shape 分档。细节见第 15 章。

#insight[
  CUDA graph 消除的是*CPU launch 开销*，一点也不加速 kernel 本身。所以判据很清晰：如果你的 GPU 已经吃满（util 接近 100%、timeline 上 kernel 之间没有空隙），graph 带来的收益是零。它只在 CPU-bound 的场景有意义。
]

== CPU-bound 还是 GPU-bound

这是优化的第一个分叉口，判错方向就白干。三个由粗到细的方法：

+ *`nvidia-smi dmon` 或 `nvidia-smi -l 1` 看 SM 利用率*。持续 90%+ 说明 GPU 忙；在 40%--70% 之间跳说明有空隙。注意 util 的定义是"这一采样周期内有没有 kernel 在跑"，一个只用 1 个 SM 的 kernel 也算 100%，所以它只能排除 CPU-bound，不能证明 GPU 用得好。
+ *加倍 batch size 看 step time*。step time 涨得远小于 2 倍 → 之前 GPU 没吃满，大概率 CPU-bound 或访存瓶颈。step time 接近线性涨 → GPU-bound。这个实验两分钟就能做，性价比最高。
+ *看 profiler 或 `nsys` 的 timeline 找 gap*。kernel 之间有肉眼可见的空白，且空白期 CPU 那一行在忙 → CPU launch 跟不上。空白期 CPU 也闲 → 在等 data 或等通信。第 11 章讲怎么读这张图。

CPU-bound 的典型对策：`torch.compile`（减少 kernel 数量）、CUDA graph（消除 launch 开销）、去掉 loop 里的 `.item()` 和 `print`、把 Python 侧的 per-parameter 循环换成 `torch._foreach_*` 或 fused optimizer。

== 多进程与 CUDA

*CUDA context 不能跨 `fork` 使用。* 一旦父进程初始化过 CUDA（哪怕只是 `torch.cuda.is_available()` 之后建了 context），`fork` 出来的子进程碰 CUDA 就会报 `Cannot re-initialize CUDA in forked subprocess`。原因是 CUDA driver 的状态里有大量文件描述符和 mmap 区域，`fork` 只复制内存不复制这些内核态资源。

```python
import torch.multiprocessing as mp

def worker(rank):
    torch.cuda.set_device(rank)
    ...

if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)          # spawn，不是 fork
    # 或者显式：mp.set_start_method("spawn", force=True)
```

`DataLoader(num_workers=4)` 在 Linux 上默认用 `fork`，这没问题——*只要 worker 里不碰 CUDA*。这也是"不要在 `Dataset.__getitem__` 里 `.cuda()`"的硬性理由之一。

设备可见性有两层，容易混：

- `CUDA_VISIBLE_DEVICES=2,3` 是*进程级*的过滤，进程内看到的 `cuda:0` 就是物理的 GPU 2。它必须在 CUDA 初始化*之前*设好，在 Python 里 `os.environ[...] = ...` 得写在 `import torch` 触发初始化之前才有效——最稳的是在 shell 里设。
- `torch.cuda.set_device(local_rank)` 是设进程的*默认 device*，之后 `torch.randn(..., device="cuda")` 落在这张卡上。分布式训练每个进程启动后第一件事就该是这一句，否则所有进程默认都用 `cuda:0`，NCCL 会直接挂或者显存爆掉。

`torchrun` 会帮你设好 `LOCAL_RANK`，标准开头是：

```python
import os, torch, torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)               # 必须在 init_process_group 之前
dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
```

== 面试考点

#interview[
  *Q1*：写一段正确测量一个 CUDA op 耗时的代码，说明每一步为什么必要。

  A：warmup 若干次（排除 cudnn/cublas 算法选择、Triton 编译、allocator 首次 `cudaMalloc`）→ `torch.cuda.synchronize()` → 用 `torch.cuda.Event(enable_timing=True)` 在 stream 里打 start/end，`end.synchronize()` 后读 `start.elapsed_time(end)` → 跑几十次取*中位数*。直接用 `time.perf_counter()` 包住一个 op 测到的是 launch 开销（几微秒），不是 kernel 时间。日常测量用 `torch.utils.benchmark.Timer`，它把这些都封装好了。
]

#interview[
  *Q2*：列举 PyTorch 里的隐式同步点。

  A：`.item()`、`.cpu()` / `.numpy()` / `.tolist()`、`print(tensor)`、`bool(tensor)` 或用 tensor 值做 `if`、输出 shape 依赖数据的 op（`nonzero`、`masked_select`、`unique`、布尔索引）、`Work.wait()`、以及 allocator 空间不足时触发的 `cudaFree` 重试。用 `torch.cuda.set_sync_debug_mode("error")` 可以把它们全抓出来。
]

#interview[
  *Q3*：为什么训练 loop 里每 step 打 `loss.item()` 会变慢？

  A：稳态下 CPU 领先 GPU 几百次 launch，队列是满的。`.item()` 强制 CPU 等 GPU 排干队列；等它拿到值时队列已空，GPU 得一边等 CPU 重新填一边零星工作。损失约等于 CPU 发完一个 step 的 launch 所需时间，小模型上占比可观。做法是在 GPU 上用 `loss.detach()` 累积，每 N 步同步一次。
]

#interview[
  *Q4*：`torch.cuda.synchronize()`、`Event.synchronize()`、`stream.wait_stream()` 三者区别？

  A：`torch.cuda.synchronize()` 阻塞 CPU 直到整个 device 上所有 stream 排空；`Event.synchronize()` 阻塞 CPU 直到某一个点；`stream.wait_stream()` / `wait_event()` *不阻塞 CPU*，只是在 GPU 上给这个 stream 插一条依赖边。做 overlap 只能用第三种，用前两种就把并行性同步掉了。
]

#interview[
  *Q5*：跨 stream 用同一块 tensor 要注意什么？

  A：必须 `tensor.record_stream(consuming_stream)`。caching allocator 按分配 stream 记账，只要 Python 引用消失它就认为可复用；如果消费 kernel 在另一个 stream 上还没跑完，这块显存就会被别的分配覆盖，产生*静默的数值错误*。`record_stream` 告诉 allocator 必须等那个 stream 上已提交的工作完成才能复用。这类 bug 依赖时序，小规模复现不出来。
]

#interview[
  *Q6*：`non_blocking=True` 什么时候真的异步？

  A：只有源（H2D）或目标（D2H）是 pinned / page-locked 内存时。pageable 内存的拷贝需要驱动先做一次同步的 staging 拷贝，`non_blocking=True` 会被静默忽略。所以 `DataLoader(pin_memory=True)` 和 `.to("cuda", non_blocking=True)` 必须配对。代价是 pinned 内存不可换出，且 `pin_memory()` 本身是一次同步分配。
]

#interview[
  *Q7*：CUDA graph 解决什么问题？有什么限制？什么时候没用？

  A：解决 CPU 侧的 kernel launch 开销——把成千上万次 `cudaLaunchKernel` 压成一次 replay。限制：shape 必须静态、输入输出地址必须固定（用 `copy_` 灌数据）、被 capture 的段里不能有 CPU 同步和数据依赖的控制流。GPU 已经吃满时收益为零，它只在 CPU-bound 场景（小 batch、LLM decode）有意义。`torch.compile(mode="reduce-overhead")` 底层就是它。
]

#interview[
  *Q8*：为什么多进程用 CUDA 必须 `spawn` 不能 `fork`？`DataLoader` 的 worker 为什么可以 `fork`？

  A：CUDA context 里有大量文件描述符和 mmap 映射，`fork` 只复制内存不复制这些内核态资源，子进程碰 CUDA 会报 `Cannot re-initialize CUDA in forked subprocess`。`DataLoader` 的 worker 用 `fork` 没问题是因为它们只做 CPU 侧的数据处理，从不触碰 CUDA——这也是"不要在 `Dataset.__getitem__` 里 `.cuda()`"的硬理由。
]
