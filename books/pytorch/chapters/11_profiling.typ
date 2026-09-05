#import "../template.typ": *

= 性能分析与优化清单

面试里问"你会怎么优化训练速度"，最差的回答是背一串 API（"上 AMP、上 compile、上 flash attention"），最好的回答是描述一个*定位流程*：先测出瓶颈在哪一层，再针对那一层动手。这一章讲怎么正确测量（计时是白板高频题）、怎么读 profiler 的 timeline、怎么用 roofline 和 MFU 两个数值直觉判断"这个优化值不值得做"，最后给一份按性价比排序的清单。CUDA kernel 级别的细节（`ncu` 的各项指标、occupancy 调优）见仓库里的 CUDA 书 `books/cuda/`。

== 先测再改

*优化的第一原则：不测量就是猜*。而且要按层次测，每一层的结论决定下一层看哪里：

+ *端到端 step time*。最重要的一个数，因为它是你实际关心的东西。同时记 tokens/s 或 samples/s，这才是跨配置可比的指标（batch 一变，step time 就没法比了）。
+ *拆成 data / compute / comm 三块*。三个问题：dataloader 跟得上吗？GPU 在算的时候在干什么？分布式通信藏住了吗？这一层决定了你该去优化哪个子系统，*绝大多数人跳过这一层直接去调 kernel，然后发现瓶颈根本不在那*。
+ *单 kernel*。到这一层才谈 fusion、occupancy、访存 pattern。

#figure(
  align(center, stacked-bar(entries: (
    ("compute", 58), ("comm", 14), ("dataloader", 12), ("optimizer", 9), ("gap/launch", 7),
  ), title: "典型 step time 分解（示意）")),
  caption: [一个 step 的时间去哪了。*这张图的数字是示意*，你必须自己测——但形状是有代表性的：compute 是主体，而剩下 40% 里往前每一块都可能是你的瓶颈。先知道自己这张图长什么样，再决定优化谁。],
) <fig-step-breakdown>

怎么*测*出这张图？最省事的办法是三次消融，每次去掉一块，看 step time 掉多少：

```python
# 1) 只跑 dataloader，不算：得到 data 的独立耗时
for x, y in loader:
    pass

# 2) 用一个固定的常量 batch 反复训练，绕开 dataloader：
#    step time 明显下降 → data 是瓶颈
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
for _ in range(n):
    train_step(x, y)

# 3) 单卡跑同样的 micro-batch（不起 DDP）：
#    与多卡的 per-step 时间差就是没藏住的通信
```

消融比读 trace 快得多，也不容易看错。确认了瓶颈在哪一块之后，再用 profiler 进那一块看细节。

== 正确计时

第 9 章讲了为什么 `time.perf_counter()` 包一个 op 测出来的是假的（launch 是异步的），也给了手写的 CUDA event 模板。日常用不着手写——`torch.utils.benchmark.Timer` 自动处理 warmup、同步、自适应重复次数和离群值统计：

```python
import torch
import torch.utils.benchmark as benchmark

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

t = benchmark.Timer(
    stmt="x @ x",
    globals={"x": x},
    label="matmul",
    sub_label="4096^3 bf16",
    num_threads=1,
)
m = t.blocked_autorange(min_run_time=1.0)
print(m)              # 打印 median / IQR / 测量次数，IQR 太大会自动警告
print(m.median)       # 秒
```

它的输出里最该看的是 *IQR（四分位距）占 median 的比例*。超过 10% 说明系统有波动（别的进程在抢卡、时钟在降频、你自己代码里有 CPU 抖动），这时候任何"优化了 5%"的结论都不可信，先把环境弄干净。

测一整个训练 step（含 backward 和 optimizer）时用同样的方法，但要注意：

```python
def step():
    opt.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()
    opt.step()

m = benchmark.Timer(stmt="step()", globals={"step": step}).blocked_autorange(min_run_time=5.0)
```

#warn[
  测 `torch.compile` 时 warmup 必须够长。第一次调用要跑 Dynamo trace + Inductor 编译 + Triton 编译，可能是几十秒；而且*每种新的输入 shape 都会触发一次重编译*。如果你的 warmup 只有 3 次，测出来的"compile 更慢"全是编译时间。做法：warmup 至少跑到 step time 稳定（打印每次的耗时看它什么时候收敛），并且 warmup 和测量用*同样的 shape*。
]

== `torch.profiler`

Timer 告诉你"多久"，profiler 告诉你"时间花在哪个 op、哪个 kernel 上"。完整用法：

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

prof_schedule = schedule(
    skip_first=3,     # 前 3 步完全不管（还在 warmup）
    wait=1,           # 空转 1 步
    warmup=1,         # profiler 自己预热 1 步（丢弃这步的数据）
    active=3,         # 真正采集 3 步
    repeat=1,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./tb_log"),
    record_shapes=True,      # 记录输入 shape，能看出是哪个 shape 慢
    profile_memory=True,     # 记录每个 op 的显存增减
    with_stack=True,         # 记录 Python 调用栈（开销不小）
    with_flops=True,         # 对 matmul/conv 估算 FLOPs
) as prof:
    for step in range(12):
        train_one_step()
        prof.step()          # 必须调，否则 schedule 不会推进
```

`schedule` 那几个参数是必须理解的：profiler 本身有开销，全程开着会把 timeline 拖变形，所以只在稳态的几个 step 上采集。`prof.step()` 忘了写是最常见的错误——schedule 不推进，最后什么都没采到。

看结果有两条路。快速看排行榜：

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 按 shape 分组，定位"是哪个 shape 慢"
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="self_cuda_time_total", row_limit=20))
```

`cuda_time_total` 含子调用，`self_cuda_time_total` 只算这个 op 自己。找热点用 `self_*`（否则最上面永远是 `nn.Module.__call__` 这种包壳），看"某个模块整体占多少"用非 self 的。

#note[
  较新版本里这些列名从 `cuda_*` 逐步改成了 `device_*`（为了兼容非 NVIDIA 后端），`sort_by="self_device_time_total"` 和 `"self_cuda_time_total"` 目前都能用。看到 deprecation warning 就换成 `device_*` 的写法。
]

另一条路是看 timeline，信息量大得多：

```python
prof.export_chrome_trace("trace.json")
```

用 #link("https://ui.perfetto.dev")[ui.perfetto.dev] 打开（比老的 `chrome://tracing` 好用很多，支持 SQL 查询）。如果用了 `tensorboard_trace_handler`，直接 `tensorboard --logdir ./tb_log` 看 PyTorch Profiler 插件。

给自己的代码段打标签，让 timeline 可读：

```python
from torch.profiler import record_function

with record_function("## data ##"):
    x, y = next(loader_it)
with record_function("## forward ##"):
    out = model(x)
with record_function("## backward ##"):
    loss.backward()
```

== 怎么读 timeline

打开 trace 之后按这个顺序看：

+ *先看 GPU 那几行有没有空隙（gap）*。这是最重要的一眼。GPU 行密不透风 → GPU-bound，去看单 kernel；有明显空白 → 先搞清楚空白期在等什么。
+ *空白期看 CPU 行在干什么*。CPU 忙着发 launch → CPU-bound（Python 开销 / kernel 太碎），对策是 `torch.compile`、CUDA graph、fused optimizer。CPU 也闲 → 在等外部：等 dataloader（能看到 `## data ##` 那段变长）或等通信。
+ *找 `ncclAllReduce` 之类的通信 kernel 在哪一行*。如果它和 compute kernel 在*不同 stream 且时间上重叠*，overlap 生效了；如果 compute 那行在通信期间是空的，overlap 没生效（第 18 章）。
+ *找占时最长的几个 kernel*。看名字就能判断类别：`*_gemm_*` / `cutlass*` 是 matmul，`elementwise_kernel` / `vectorized_elementwise` 是 elementwise，`reduce_kernel` 是规约。如果 elementwise 的总时间和 GEMM 一个量级，说明有大量没融合的小 op，`torch.compile` 会有明显收益。
+ *看 kernel 之间有没有 `Memcpy` / `Memset`*。`Memcpy DtoH` 出现在训练 loop 里几乎总是 bug（某处在 `.item()` 或 `.cpu()`）。`Memcpy DtoD` 多说明有大量 `.contiguous()` 或不必要的 copy。

`nsys` 和 `ncu` 的分工一句话：*`nsys` 看时间线上的宏观分布*（哪个 kernel、多少 gap、通信有没有重叠），能力和 `torch.profiler` 的 trace 重叠但更底层、能看到驱动和 NCCL 内部；*`ncu` 看单个 kernel 的微观指标*（achieved occupancy、DRAM 吞吐、是 memory-bound 还是 compute-bound），一次只看一个 kernel，开销极大。判断"该不该 fuse 这个 op"用 `ncu`，判断"该优化哪个 op"用 `nsys` 或 `torch.profiler`。具体命令和指标解读见 `books/cuda/`。

== roofline 直觉：这个 op 值不值得 fuse

一个 kernel 的算术强度（arithmetic intensity）定义为它做的浮点运算数除以它搬运的字节数：

#formula[
  $ I = "FLOPs" / "bytes moved" quad ["单位" : "FLOP/byte"] $
]

拿它和硬件的"分界强度" $I^* = P_"peak" / B$ 比：小于 $I^*$ 就是 memory-bound（HBM 带宽是瓶颈，SM 在等数据），大于就是 compute-bound。

A100-SXM4-80GB：bf16 稠密峰值约 312 TFLOPS，HBM 带宽约 2.0 TB/s。

#formula[
  $ I^*_"A100,bf16" = (312 times 10^12) / (2.0 times 10^12) approx 156 " FLOP/byte" $
]

156 是个很高的门槛，所以*绝大多数 op 都是 memory-bound*。代几个例子看：

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, right, left),
  [*op*], [*$I$ 量级*], [*结论*],
  [`y = x + 1`（bf16）], [$0.25$], [读 2 B 写 2 B 做 1 FLOP，极度 memory-bound],
  [LayerNorm], [$O(1)$], [几次遍历、每元素十几个 FLOP，memory-bound],
  [Softmax], [$O(1)$], [同上，而且要多遍（max、sum、除）],
  [GEMM $n times n times n$（bf16）], [$n\/3$], [$n = 4096$ 时约 1365，compute-bound],
  [GEMM，$n = 468$], [$156$], [恰好在分界线上],
)

GEMM 那一行值得单独说：bf16 方阵乘法读写 3 个 $n^2$ 矩阵共 $6 n^2$ 字节、做 $2 n^3$ 次运算，所以 $I = 2 n^3 \/ (6 n^2) = n\/3$。解 $n\/3 = 156$ 得 $n approx 470$。也就是说*A100 上小于约 470 的方阵 GEMM 也是 memory-bound*——这解释了为什么小 batch、小 hidden size 的模型再怎么优化 kernel 也吃不满算力。

#insight[
  这个直觉的实际用法：一串连续的 elementwise / norm op（比如 `x = norm(x); x = x * w; x = gelu(x)`）每个都要把整个张量从 HBM 读一遍写一遍。融合成一个 kernel 之后，中间结果留在寄存器和 shared memory 里，*省掉的是 $k-1$ 次 HBM 往返*。所以 fusion 的收益上界就是"省掉的 HBM 流量 ÷ 带宽"，可以直接估出来。反过来，两个大 GEMM 之间融合没什么意义——它们本来就 compute-bound。这就是 `torch.compile`（第 14 章）主要在优化 memory-bound 部分的原因。
]

== MFU：一个数说清训练效率

MFU（Model FLOPs Utilization）= 模型有效计算量 ÷ 硬件峰值。分子用著名的 $6 N$ 近似：训练一个 token，forward 约 $2N$ 次乘加（每个参数一次乘一次加），backward 约 $4N$（对输入和对权重各一遍），$N$ 是非 embedding 参数量。

#formula[
  $ "MFU" = (6 N dot T) / P_"peak", quad T = "每卡每秒处理的 token 数" $
]

长序列时还要补上 attention 的部分，它不含参数但要算：每 token 每层约 $12 s h$ 次运算（$s$ 是 seq len，$h$ 是 hidden），所以完整式子是 $6 N + 12 L s h$。当 $s$ 和 $h$ 同量级时这一项不可忽略。

算一个例子（*假设*测到每卡 3000 tokens/s，7B 模型，A100 bf16）：

$ "MFU" = (6 times 7 times 10^9 times 3000) / (312 times 10^12) = (1.26 times 10^14) / (3.12 times 10^14) approx 40% $

公开的大模型训练报告里，A100 上调好的稠密 Transformer 通常落在 35%--50% 这个区间；低于 30% 说明有明显的可优化空间，高于 55% 在稠密训练上很少见。

#note[
  面试被问"你的训练 MFU 多少"时，答案里必须包含*怎么算的*：分子用 $6N$ 还是 $6N + 12 L s h$、$N$ 含不含 embedding、分母用稠密峰值还是稀疏峰值（NVIDIA 标的稀疏峰值是稠密的 2 倍，用它会让 MFU 看起来腰斩）。还有 MFU 和 HFU 的区别：HFU（Hardware FLOPs Utilization）把 activation checkpointing 重算的那部分也算进分子，所以开了 checkpoint 之后 HFU 会比 MFU 高一截。能主动说清这些，比报一个漂亮数字更有说服力。
]

== 优化清单，按性价比排序

#ladder(
  ([AMP / bf16], [激活和参数的读写都减半，且走 Tensor Core，峰值算力高一个档],
   [几乎必做]),
  ([`torch.compile`], [把连续的 memory-bound op 融合，省 HBM 往返和 launch 次数],
   [一行代码，见第 15 章]),
  ([SDPA / Flash], [不 materialize $s times s$ 的 score 矩阵，省显存也省 HBM 往返],
   [长序列最大单项]),
  ([fused optimizer], [`AdamW(..., fused=True)`，把 per-param 的 Python 循环压成少量 kernel],
   [CPU-bound 时明显]),
  ([`set_to_none`], [grad 不填 0 而是直接释放], [省显存，少一批 memset]),
  ([`channels_last`], [NHWC 布局才能让 cudnn 走 Tensor Core 路径], [仅 CNN，LLM 无关]),
  ([加大 batch], [小 batch 下 GEMM 是 memory-bound，Tensor Core 吃不满],
   [受显存限制]),
  ([dataloader 调优], [`num_workers` + `pin_memory` + `persistent_workers`],
   [确认是瓶颈再做]),
  ([activation ckpt], [用算力换显存，换来的显存拿去加大 batch], [算力 +30%，见第 8 章]),
  ([CUDA graph], [消除 CPU 侧的 launch 开销], [仅 CPU-bound，见第 9 章]),
  ([通信 overlap], [让 AllReduce 藏进 backward 的计算里], [分布式，见第 18 章]),
)

前三条基本无脑上，中间几条要先确认瓶颈在那一层，后面几条都有明确的前置条件——用错了不但不快，还会因为多出来的转换开销变慢。

#figure(
  align(center, hbar-chart(
    (("eager fp32", 100), ("+ bf16 autocast", 55), ("+ torch.compile", 45), ("+ SDPA", 38)),
    unit: "",
  )),
  caption: [示意图，非实测数据。数值是归一化的 step time（eager fp32 $= 100$），只表达"逐项叠加、收益递减"的形状。真实数字强烈依赖模型结构、shape 和 GPU 型号——同一套优化在小模型上可能只有几个百分点（因为它本来就 CPU-bound），在长序列模型上 SDPA 一项可能就占大半收益。*每一项都要自己测。*],
) <fig-opt-ladder>

== "优化了但没变快"

#warn[
  排查顺序：

  + *瓶颈不在那里*。你优化了 GEMM，但 40% 的时间在等 dataloader。回到第一节，先做那张 step time 分解图。
  + *被 CPU 掩盖了*。GPU kernel 快了一倍，但 CPU 发 launch 的速度没变，step time 一点不动——因为原本就是 CPU-bound。判据：`torch.profiler` 里 GPU 总忙时远小于 wall-clock。
  + *`torch.compile` 的编译时间没排除*。warmup 不够，或者输入 shape 每个 batch 都在变（变长 seq）导致反复重编译。看 `torch._dynamo.utils.compile_times()` 或日志里的 recompile 计数。
  + *batch 太小*。$n < 470$ 的 GEMM 在 A100 bf16 上是 memory-bound（见 roofline 那节），Tensor Core 的峰值根本用不上。这时候正确的优化是加大 batch，不是调 kernel。
  + *测量噪声大于收益*。`Timer` 报的 IQR 超过 median 的 10% 时，"提升 5%" 是噪声。把别的进程停掉、锁定时钟、多测几轮。
  + *优化被别的地方吃掉了*。典型是加了 `channels_last` 之后模型里某个 op 不支持这个布局，于是每次都要转回 contiguous，转换的开销抵消了收益。在 trace 里找突然多出来的 `Memcpy DtoD` 或 `contiguous` kernel。
]

== 面试考点

#interview[
  *Q1*：你会怎么定位一个训练任务的性能瓶颈？

  A：分三层。先量端到端 step time 和 tokens/s；再用 `torch.profiler` 加 `record_function` 把 step 拆成 data / forward / backward / optimizer / comm，看哪块占比异常；最后才看单个 kernel。看 trace 的第一眼是*GPU 行有没有空隙*：没空隙 → GPU-bound，去看 kernel；有空隙且 CPU 忙 → CPU-bound（launch 开销、Python 开销）；有空隙且 CPU 也闲 → 在等 data 或等通信。
]

#interview[
  *Q2*：为什么用 `time.time()` 测 GPU 算子时间是错的？正确怎么做？

  A：kernel launch 是异步的，`time.time()` 测到的是 launch 开销（微秒级），不是 kernel 执行时间。正确做法：warmup 若干次 → `torch.cuda.synchronize()` → 用 `torch.cuda.Event(enable_timing=True)` 打 start/end → 多次取中位数。生产上直接用 `torch.utils.benchmark.Timer.blocked_autorange()`，它自动处理这三件事，还会报 IQR 让你知道测量可不可信。
]

#interview[
  *Q3*：算术强度是什么？怎么用它判断一个 op 该不该优化？

  A：算术强度 $I$ = FLOPs / 搬运字节数。和硬件的分界强度 $I^* = $ 峰值算力 / 带宽 比较：A100 bf16 是 $312 "TFLOPS" \/ 2.0 "TB/s" approx 156$ FLOP/byte。低于它是 memory-bound。elementwise（$I approx 0.25$）、LayerNorm、softmax 都远低于 156，优化方向是减少 HBM 往返（融合）；大 GEMM 的 $I approx n\/3$，$n = 4096$ 时约 1365，是 compute-bound，优化方向是提高 Tensor Core 利用率。
]

#interview[
  *Q4*：kernel fusion 的收益上限怎么估？

  A：$k$ 个连续的 memory-bound op，每个都要把张量从 HBM 读一遍写一遍。融合后中间结果留在寄存器/shared memory，省掉 $k-1$ 次往返。收益上界 = 省掉的字节数 ÷ HBM 带宽，可以直接算出来。所以 `torch.compile` 对"一长串 elementwise + norm"收益大，对"两个大 GEMM"基本没收益——后者本来就 compute-bound。
]

#interview[
  *Q5*：MFU 怎么算？你的训练 MFU 多少？

  A：$"MFU" = 6 N T \/ P_"peak"$，$N$ 是非 embedding 参数量、$T$ 是每卡每秒 token 数、$P_"peak"$ 用*稠密*峰值（A100 bf16 是 312 TFLOPS）。$6N$ 来自 forward $2N$ + backward $4N$；长序列要补 attention 的 $12 L s h$。回答时必须交代清楚分子用了哪个式子、$N$ 含不含 embedding、分母用稠密还是稀疏峰值。公开报告里 A100 上调好的稠密 Transformer 常在 35%--50%。另外要能区分 MFU 和 HFU：后者把 activation checkpointing 的重算算进分子。
]

#interview[
  *Q6*：`torch.profiler` 的 `schedule` 为什么要设 `wait` / `warmup` / `active`？

  A：profiler 自身有开销（尤其 `with_stack=True`），全程开着会把 timeline 拖变形，而且产出的 trace 大到打不开。所以只在稳态的少数几个 step 上采集：`skip_first` 跳过 warmup 阶段，`warmup` 让 profiler 自己预热（这几步数据丢弃），`active` 才是真正采集。同时必须在每个 step 末尾调 `prof.step()` 推进 schedule——忘了写是最常见的错误，结果是什么都没采到。
]

#interview[
  *Q7*：`nsys` 和 `ncu` 分别什么时候用？

  A：`nsys` 是时间线级别的，看整体分布——哪些 kernel、gap 在哪、通信有没有和计算重叠、多卡之间有没有 straggler。用它回答"该优化哪个 op"。`ncu` 是单 kernel 级别的，给 achieved occupancy、DRAM 吞吐、L2 命中率这些微观指标，用它回答"这个 kernel 离硬件上限还差多远、瓶颈是访存还是算力"。`ncu` 会 replay kernel，开销极大，只对少数几个热点 kernel 用。
]

#interview[
  *Q8*：优化之后 step time 没变，可能是什么原因？

  A：最常见是*瓶颈不在你优化的地方*——比如 GPU kernel 快了但整体是 CPU-bound 或在等 dataloader，判据是 profiler 里 GPU 忙时远小于 wall-clock。其次：`torch.compile` 的编译时间没从测量里排除，或输入 shape 在变导致反复重编译；batch 太小以致 GEMM 本身就 memory-bound；测量噪声大于收益（`Timer` 的 IQR 超过 median 的 10%）；以及优化引入了新开销（比如 `channels_last` 遇到不支持的 op，每次都要转回 contiguous）。
]
