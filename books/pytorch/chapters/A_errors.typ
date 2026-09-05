#import "../template.typ": *

= 附录 A：常见报错速查

拿着报错原文来查这一章。小节标题就是报错里最有辨识度的那段字符串，直接搜关键字。每条给三样：*报错原文*（torch 2.10 实际触发的输出，老版本措辞可能略有出入，为排版做了换行）、*成因*（通常不止一种）、*最小修法*。

面试也考这些——"你遇到过什么棘手的 bug"、"CUDA OOM 怎么排查"本质上就是这张表。答的时候别只说修法，要说*怎么定位到成因*。

== `view size is not compatible with input tensor's size and stride`

```text
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces).
Use .reshape(...) instead.
```

*成因.* `view` 要求新 shape 能用一组合法 stride 走完原 storage，前提是被合并的那些维度连续。来源几乎总是这三个之一：前面做过 `transpose` / `permute`（stride 被交换）；做过非最外维的切片如 `x[:, 1:3]`；或者是 `expand` 出来的广播视图（stride=0）。

*修法.* `x.reshape(12)`（能 view 就 view，不能就自动 copy）或 `x.contiguous().view(12)`（显式表明这里有一次拷贝）。两者结果一样，差别是意图表达：性能敏感路径用后者，让 review 的人一眼看到开销。判定规则见第 1 章。

== `Expected all tensors to be on the same device`

两种措辞，来自不同的检查点：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu!
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on
cpu, different from other tensors on cuda:0 (... wrapper_CUDA_addmm)
```

*成因.* 模型 `.cuda()` 了但 batch 没有，或反过来（`.to()` 对 module 是原地改、对 tensor 是返回新对象，`x.to("cuda")` 不写等号就白做了，见第 3 章）；forward 里 `torch.zeros(...)` / `torch.arange(...)` 没指定 device，默认落 CPU；常量存成普通属性而不是 `register_buffer`，`model.cuda()` 搬不动它；多卡时没设 `torch.cuda.set_device(local_rank)`，新建张量全跑到 `cuda:0`。

*修法.* 新建张量一律从已有张量取 device，不要硬编：

```python
pos  = torch.arange(S, device=x.device)         # 不是 device="cuda"
mask = torch.zeros_like(x, dtype=torch.bool)    # *_like 继承 device + dtype
self.register_buffer("freqs", freqs)            # 常量走 buffer，跟着 .to() 走
```

== dtype 不匹配：`must have the same dtype` / `should be the same`

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
RuntimeError: Input type (c10::Half) and bias type (float) should be the same
RuntimeError: Expected query, key, and value to have the same dtype, but got
query.dtype: c10::Half key.dtype: float and value.dtype: float instead.
```

老版本里 `expected scalar type Float but found Half` 是同一类问题的另一种措辞。

*成因.* 手动 `.half()` 了模型但输入还是 fp32（最常见）；*AMP 边界*——`autocast` 只对白名单 op 自动 cast，其低精度输出流出上下文后再和 fp32 做 matmul 就炸；自定义 `autograd.Function` 不在白名单里；加载 bf16 checkpoint 到 fp32 模型，或 LoRA 与 base 精度不一致。

*修法.* 让 AMP 管精度，不要手动 `.half()` 整个模型；需要局部对齐时显式 cast 到目标那一边：

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x)              # 边界只在这里
loss.backward()                  # backward 不放进 autocast
y = y.to(w.dtype)                # 局部对齐，别指望隐式提升
```

另外，`autocast` 里不要缓存中间结果给下个 iteration 用——存下来的是低精度张量，下次可能落在 `autocast` 外面。AMP 完整用法见第 5 章。

== `CUDA out of memory. Tried to allocate ...`

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4096.00 GiB.
GPU 0 has a total capacity of 79.25 GiB of which 78.83 GiB is free. ... Of the
allocated memory X is allocated by PyTorch, and Y is reserved by PyTorch but
unallocated. If reserved but unallocated memory is large try setting
PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

*先读这条消息本身*，答案已经写在里面。关键是两个数的差：`allocated` 是真正被张量占着的，`reserved` 是 caching allocator 从驱动拿到、还没还回去的。`allocated` 逼近上限是*真 OOM*；`reserved` 远大于 `allocated`（比如 70 GB vs 40 GB）却申请不到几百 MB 连续块，是*碎片*。

*排查顺序.*

+ 看 `Tried to allocate` 的数值。异常巨大（几百 GB）说明是 shape 算错了，不是显存不够——去查那一行的 reshape / broadcast。
+ `torch.cuda.memory_allocated()` 对比 `memory_reserved()`，区分真 OOM 与碎片。
+ 真 OOM：减 batch、开 AMP、开 activation checkpointing、换 FSDP（性价比排序见第 8 章）。碎片：设 `PYTORCH_ALLOC_CONF=expandable_segments:True`，对变长 shape 场景效果显著。
+ 定位谁吃的：`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`，用官方 memory viz 页面看。

#warn[
  `torch.cuda.empty_cache()` 只把 reserved 还给驱动，*不减少 allocated*，对真 OOM 毫无帮助，还会让下次分配走慢路径。只在"把显存让给同机另一个进程"时才用。
]

== `a leaf Variable that requires grad is being used in an in-place operation`

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place
operation.
```

*成因.* 直接原地修改 `requires_grad=True` 的叶子张量（也就是 `nn.Parameter`）。autograd 需要叶子的值稳定，禁止这么干。最常见是手动改权重：`w += 1`。

*修法.* 用 `torch.no_grad()` 包起来——optimizer 内部就是这么做的：

```python
with torch.no_grad():
    w += 1; w.clamp_(-1, 1)
```

`w.data += 1` 也能跑，但绕过了 version counter，不推荐。初始化用 `nn.init.*`（内部已在 `no_grad` 里）。

== `one of the variables needed for gradient computation has been modified by an inplace operation`

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.FloatTensor [3]], which is output 0 of
ExpBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly
detection ... torch.autograd.set_detect_anomaly(True).
```

*成因.* forward 时某个 op 把张量存进了 `ctx`（`save_for_backward`），之后这个张量被原地改了，backward 时 version counter 对不上。报错里的 `output 0 of ExpBackward0` 是关键线索：*被改的是 exp 的输出*。典型来源：残差写成 `x += block(x)`；`relu_` / `add_` 紧跟在保存输出的 op（`exp`、`sigmoid`、`tanh`、`softmax`）后面；自写 `Function` 里 `save_for_backward` 存了会被复用的 buffer。

*修法.* 把那处原地操作改成非原地。定位靠报错里的 `XxxBackward0` 名字，找不到就开 anomaly detection：

```python
x = x + block(x)                                # 不是 x += block(x)
h = F.relu(h)                                   # 不是 F.relu(h, inplace=True)
with torch.autograd.set_detect_anomaly(True):   # 只在 debug 时开，很慢
    loss.backward()
```

不是所有原地操作都会炸：`relu_` 接在 `Linear` 后面是安全的，因为 `addmm` 的 backward 不需要它的输出。会炸的只有"输出被 backward 用到"的那些 op（第 6 章的 version counter 一节）。

== `Trying to backward through the graph a second time`

```text
RuntimeError: Trying to backward through the graph a second time (or directly
access saved tensors after they have already been freed). ... Specify
retain_graph=True if you need to backward through the graph a second time ...
```

*成因.* 报错文本建议加 `retain_graph=True`，但*绝大多数情况下这是错的建议*。真实成因按概率排：(1) *把带图的张量跨 step 累积了*，最经典是 `total_loss += loss`——`loss` 带着整张图，累加后上一步的图仍被引用，下次 backward 会走回去；(2) RNN / 状态机里 hidden state 跨 step 传下去忘了 `detach()`；(3) 一次 forward 配了两次 `backward()`（GAN 里 D 和 G 共用一次 forward）。只有第 3 种是 `retain_graph=True` 的正当用法，而且更好的解法通常是重新 forward 一次或用 `autograd.grad`。

*修法.* `total_loss += loss.item()`（记录标量用 `.item()` 或 `.detach()`）；`h = h.detach()`（RNN 跨 step 截断）。

#warn[
  盲目加 `retain_graph=True` 会让图一直不释放，症状是显存随 step 单调上涨、几十步后 OOM——一个报错换成另一个更难查的报错。
]

== `element 0 of tensors does not require grad and does not have a grad_fn`

```text
RuntimeError: element 0 of tensors does not require grad and does not have a
grad_fn
```

*成因.* 你 `backward()` 的那个张量根本不在图上：整段 forward 被包在 `torch.no_grad()` / `inference_mode()` 里（推理代码复制过来忘了删）；中间 `.detach()` 了，或走了 `.item()` / `.numpy()` / `.tolist()` 再转回张量；所有参数都被 `requires_grad_(False)` 冻住（只训 LoRA 但忘了给 A/B 打开）；loss 是 `torch.tensor(...)` 现造的常量。

*修法.* 从 loss 往回查一步：

```python
print(loss.requires_grad, loss.grad_fn)   # 期望 True, <SumBackward0 ...>
print([n for n, p in model.named_parameters() if p.requires_grad])  # 别是空的
```

`grad_fn` 是 `None` 说明断点在 loss 这一层；不为 `None` 但某个参数没梯度，看下一节。

一个细节：`inference_mode()` 产生的张量比 `no_grad()` 更"死"，它带了 inference tensor 标记，之后即使离开上下文也不能参与 autograd。验证代码用 `no_grad()` 更保险。

== `.grad` 是 `None`（不报错，但更难查）

没有报错，`optimizer.step()` 照跑，就是参数不动。按这个顺序查：

+ *它是叶子吗？* 只有叶子的 `.grad` 会被填。`p = w.cuda()` 之后 `p` 不是叶子——正确写法是先建 module 再 `.cuda()`。访问非叶子的 `.grad` 会有一条 UserWarning 提示你用 `.retain_grad()`。
+ *`requires_grad` 是 True 吗？路径上有没有 `detach()` / `no_grad()` / `.data`？*
+ *它参与 forward 了吗？* 定义了但 `forward` 里没用到的层梯度当然是 `None`（DDP 下还会直接 hang，见第 18 章 `find_unused_parameters`）。
+ *是不是在 `backward()` 之前、或 `zero_grad(set_to_none=True)` 之后看的？* 那时本来就是 `None`。
+ *optimizer 拿到这个参数了吗？* `model.cuda()` 在 `SGD(model.parameters())` *之后*执行的话，optimizer 里存的是旧对象。顺序永远是先 `.to(device)` 再建 optimizer。

一行定位：`print([n for n, p in model.named_parameters() if p.requires_grad and p.grad is None])`。

== `CUDA error: device-side assert triggered`

```text
.../ATen/native/cuda/Indexing.cu:1515: indexSelectSmallIndex: block: [0,0,0],
thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
torch.AcceleratorError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so
the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

*成因.* 几乎总是*索引越界*。两个高频来源：`nn.Embedding` 的某个 token id $>=$ `num_embeddings`（词表和 tokenizer 对不上、忘了 special token）；`cross_entropy` / `nll_loss` 的某个 label $>=$ `num_classes` 或为负数。同一个 bug 在 CPU 上报的是可读得多的 `IndexError: index out of range in self` 和 `IndexError: Target 5 is out of bounds.`。

*修法.* 先用 `CUDA_LAUNCH_BLOCKING=1 python train.py` 让 stack trace 指向真正出错那一行，再在数据入口补上 `assert ids.max() < emb.num_embeddings` 和 `assert labels.min() >= 0 and labels.max() < num_classes`。

#warn[
  CUDA context 被 assert 打挂后不可恢复，后续任何 CUDA 调用都报同一个错。看到它必须重启进程，不要在同一进程里继续 debug。异步语义见第 9 章。
]

== `Expected more than 1 value per channel when training`

```text
ValueError: Expected more than 1 value per channel when training, got input
size torch.Size([1, 4])
```

*成因.* BatchNorm 在 train 模式下要算 batch 内方差，batch size 为 1 时方差无定义。最常出现在：最后一个不完整 batch 只剩 1 个样本；或者 eval 时忘了 `model.eval()`。

*修法.* `DataLoader(..., drop_last=True)` 丢掉尾 batch；推理前 `model.eval()`；小 batch 场景换 `nn.GroupNorm` / `nn.LayerNorm`。

== `Sizes of tensors must match except in dimension` / `stack expects each tensor to be equal size`

```text
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size
3 but got size 4 for tensor number 1 in the list.
RuntimeError: stack expects each tensor to be equal size, but got [2, 3] at
entry 0 and [3, 3] at entry 1
```

*成因.* 两个函数契约不同，混用是主因：`cat` 沿已有维度拼接，*除拼接维外其余维度必须完全一致*，不增加维度；`stack` 新建一维，*所有输入 shape 必须完全一致*。变长序列走 `collate_fn` 时最容易撞上——不同样本长度不同却直接 `torch.stack`。

*修法.* 变长序列用 `torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)` 先 pad 再 stack。报错里的 "tensor number 1" 是列表下标，`print([t.shape for t in xs])` 直接看出是哪一个。

== `Error(s) in loading state_dict: Missing key(s) / Unexpected key(s)`

```text
RuntimeError: Error(s) in loading state_dict for Linear:
	Missing key(s) in state_dict: "weight", "bias".
	Unexpected key(s) in state_dict: "module.weight", "module.bias".
```

*成因.* 看两组 key 的差异形状就能判断：整齐地差一个 `module.` 前缀 $arrow.r$ checkpoint 是从 DDP / DataParallel 包装后的模型存的；差一个 `_orig_mod.` 前缀 $arrow.r$ 是从 `torch.compile` 包装后的模型存的；只多不少（多出 `num_batches_tracked` 之类）$arrow.r$ 版本差异或 buffer 的 `persistent` 设置变了；结构真变了（加层、改名）才需要改代码。

*修法.* 存的时候就存干净的，这是根治：`torch.save(model.module.state_dict(), path)`（DDP 存内层）、`torch.save(model._orig_mod.state_dict(), path)`（compile 存原模型）。已经存脏了就在 load 时剥前缀：`sd = {k.removeprefix("module."): v for k, v in sd.items()}`。

#warn[
  `strict=False` 能让它跑起来，但会*静默忽略*所有对不上的 key——权重没加载完，训练看着正常但效果莫名其妙差。真要用就检查返回值 `missing, unexpected = model.load_state_dict(sd, strict=False)` 并打印出来人工确认。
]

== `DataLoader worker (pid xxx) is killed by signal`

```text
RuntimeError: DataLoader worker (pid 12345) is killed by signal: Killed.
It is possible that dataloader's workers are out of shared memory. Please try
to raise your shared memory limit.
RuntimeError: DataLoader worker (pid(s) 12345) exited unexpectedly
```

*成因.* worker 是子进程，它死了主进程只看得到信号。三种来源：*共享内存不足*——worker 通过 `/dev/shm` 把 batch 传回主进程，Docker 默认 `--shm-size=64m`，大 batch 或大图像立刻爆，这是容器里最常见的原因；*宿主机 OOM 被内核 kill*（`signal: Killed` = SIGKILL），worker 是 fork 出来的，Python 对象的引用计数会让 copy-on-write 失效，实际内存远超预期；*worker 里抛了异常*，这种通常能看到 `Caught ValueError in DataLoader worker process 0.` 加原始 traceback，好查得多。

*修法.* 容器加 `--shm-size=8g` 或 `--ipc=host`；debug 第一步永远是 `num_workers=0`，报错会原样抛出来，八成问题当场清楚。降 `num_workers`、把 Dataset 里的大 Python list 换成 numpy array 也能缓解内存问题（第 4 章）。

== `Cannot re-initialize CUDA in forked subprocess`

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```

*成因.* CUDA context 不能跨 `fork` 继承。父进程只要初始化过 CUDA（哪怕只是建了一个 cuda 张量），fork 出来的子进程碰 CUDA 就炸。Linux 上 multiprocessing 默认就是 fork。

*修法.* 用 spawn：`mp.set_start_method("spawn", force=True)`（放在 `if __name__ == "__main__"` 里），或直接 `mp.spawn(worker, nprocs=world_size)`。DataLoader 的 worker 里也不要碰 CUDA：augment 留在 CPU，`.cuda(non_blocking=True)` 在主进程做。生产上更简单的路子是用 `torchrun` 起独立进程，从根上避开 fork。

== `Address already in use`

```text
torch.distributed.DistNetworkError: The server socket has failed to listen on
any local network address. port: 29500, useIpv6: false, code: -98,
name: EADDRINUSE, message: address already in use
```

*成因.* rank 0 要在 `MASTER_PORT` 上建 TCPStore，端口被占了：上次训练崩了进程没清干净；同机跑了两个 job 都用默认的 29500；或端口被别的服务占用。

*修法.*

```bash
pkill -f train.py                                              # 先清残留
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 ...   # 让它自己挑端口
```

或者 `MASTER_PORT=$((20000 + RANDOM % 10000))` 随机一个。多 job 共机时把端口按 job id 算进启动脚本，不要靠人记。

== NCCL：`unhandled system error` 与 `Watchdog caught collective operation timeout`

```text
NCCL WARN Bootstrap : no socket interface found
ncclInternalError: Internal check failed. ... unhandled system error
[Rank 3] Watchdog caught collective operation timeout: WorkNCCL(...,
OpType=ALLREDUCE, Timeout(ms)=600000) ran for 600351 milliseconds before
timing out.
```

这两条是分布式面试的高频题，成因完全不同。

*`unhandled system error` 一般是环境问题*：网卡选错（多网卡机器上 NCCL 挑了张不通的，比如 `docker0`）$arrow.r$ `NCCL_SOCKET_IFNAME=eth0` 指定；IB 不可用却在走 IB $arrow.r$ `NCCL_IB_DISABLE=1` 先退回 socket 验证连通性；容器 `/dev/shm` 太小或没开 `--ipc=host`。排查第一步永远是 `NCCL_DEBUG=INFO`，它会打印挑了哪张网卡、走什么传输。

*`Watchdog caught collective operation timeout` 是 collective 对不齐*，按概率排：

+ *某个 rank 没调这个 collective*。`if rank == 0: dist.all_reduce(...)` 这种写法必挂——所有 rank 必须以相同顺序调用相同的 collective。
+ *shape / dtype 不一致*。变长 batch 下不同 rank 算出的 tensor 大小不同。
+ *某个 rank 挂在别处*：等 IO、OOM 重试、或已经抛异常退出——其他 rank 集体超时。
+ *慢卡*：某张卡降频或被别的进程占用，超过 timeout（默认 10 分钟）。

*定位.* `TORCH_NCCL_DESYNC_DEBUG=1` 会在超时时打印每个 rank 卡在哪个 collective，直接看出谁没跟上；配合 `py-spy dump --pid <pid>` 看 Python 栈。完整 hang 排查流程见第 22 章。

== `torch._dynamo.exc.Unsupported`

```text
torch._dynamo.exc.Unsupported: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
    Dynamo does not support tracing dynamic control flow.
  Hint: Use `torch.cond` to express dynamic control flow.
from user code:  File "model.py", line 42, in forward
    if x.sum() > 0:
```

*成因.* 这是一个 *graph break*，只在 `fullgraph=True` 下才变成异常（默认 `fullgraph=False` 时它静默切图，代价是编译收益打折）。常见触发源：依赖张量值的控制流（`if x.sum() > 0`、`while loss > tol`）；`print` / `pdb` / 写日志；`.item()` / `.tolist()` / `.cpu()` 这类强制同步；Dynamo 不认识的 C 扩展和第三方库。

*修法.* 报错里的 `from user code:` 已经指到具体行。

```python
y = torch.where(cond, a, b)     # 张量条件用 where，不用 if
torch._dynamo.graph_break()     # 主动在这里切，接受两段图
@torch._dynamo.disable          # 整个函数不编译
def log_metrics(loss): ...
```

*先不加 `fullgraph=True` 跑一遍*，用 `TORCH_LOGS="graph_breaks"` 看有几个 break、在哪，确认没有致命的再打开 `fullgraph=True` 锁住。机制见第 12 章。

== `hit config.recompile_limit`

```text
torch._dynamo hit config.recompile_limit (8)
   function: 'forward' (model.py:42)
   last reason: 0/7: tensor 'x' size mismatch at index 0. expected 16, actual 17
```

*成因.* 同一个函数被反复重编译，超过上限后 Dynamo *静默 fallback 到 eager*。这是"加了 `torch.compile` 但没变快"的头号原因——它是 warning 不是 error，很容易被日志淹掉。触发源：输入 shape 一直变（变长序列、尾 batch）；forward 依赖一个每次都不同的 Python 标量（`step`、`temperature`）；每次传进来新建的 Python 对象（config dataclass、dict）。

*修法.* `torch.compile(model, dynamic=True)` 编动态 shape 版本；`DataLoader(..., drop_last=True)` 消掉尾 batch；把变动的标量做成张量参数；或者把 shape bucket 化（pad 到 128 的倍数）。定位一律用 `TORCH_LOGS="recompiles"`，它会打印每次重编的具体 guard 原因。调参见第 15 章。

== loss 变成 NaN

不是报错，但排查成本最高。按这个顺序查，每一步都能二分定位：

+ *定位第一个 NaN 在哪一步*。`assert torch.isfinite(loss), step`。是某一步突然开始（数据 / lr spike）还是逐渐发散（lr 太大）？
+ *查输入与数值稳定性*。`assert torch.isfinite(x).all()` 排除脏数据、除零归一化、全 pad 样本（softmax 整行被 mask 会出 NaN）；再查有没有手写 `log(softmax(x))` 而不用 `log_softmax`、`log(p)` 时 `p` 可能为 0、`sqrt(x)` 在 0 处梯度是 inf。
+ *查 lr 与 warmup*。lr 太大、warmup 太短、加载 checkpoint 时 scheduler 状态没恢复导致 lr 跳回峰值。
+ *查 AMP*。fp16 动态范围只到 65504，激活或梯度溢出会变 inf 再变 NaN。`GradScaler` 处理的是梯度*下溢*，处理不了前向溢出。*换 bf16 是最快的验证手段*——bf16 指数位和 fp32 一样宽，换过去就好了基本可确诊是 fp16 range 问题。
+ *查梯度爆炸*。打印 `clip_grad_norm_` 的返回值（它返回裁剪*前*的范数），看是不是某步突然涨了几个数量级。
+ *定位到具体 op*：`torch.autograd.set_detect_anomaly(True)` 会在产生 NaN 梯度的 backward 节点抛异常并打印 forward 调用栈。很慢，只在 debug 时开。

#insight[
  前向正常、反向 NaN，几乎总是"梯度公式在某点无定义"：`sqrt(0)`、`log(0)`、`norm()` 在零向量处，以及 mask 用 `-inf` 而不是 `-1e9` 导致整行被 mask 时 softmax 出 NaN。加 `eps` 或换成有限大的负数即可。
]

== 报错关键字 $arrow.r$ 去看哪一章

#table(
  columns: (1fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*报错关键字*], [*相关章节*],
  [`view size is not compatible` / `is invalid for input of size`], [第 1、2 章],
  [`Expected all tensors to be on the same device`], [第 1、3 章],
  [`must have the same dtype` / `Input type ... and bias type`], [第 1 章、第 5 章（AMP）],
  [`CUDA out of memory` / 碎片 / `expandable_segments`], [第 8 章],
  [`leaf Variable ... in-place` / `modified by an inplace operation`], [第 1、6 章],
  [`backward through the graph a second time` / `retain_graph`], [第 6 章],
  [`does not have a grad_fn` / `.grad` 是 `None`], [第 6 章],
  [`device-side assert triggered` / `CUDA_LAUNCH_BLOCKING`], [第 9 章],
  [`Expected more than 1 value per channel`], [第 3 章、第 18 章（SyncBN）],
  [`Sizes of tensors must match` / `stack expects each tensor`], [第 2、4 章],
  [`Missing key(s)` / `Unexpected key(s)` / `module.` 前缀], [第 3、18、22 章],
  [`DataLoader worker ... killed by signal` / shm 不足], [第 4 章],
  [`Cannot re-initialize CUDA in forked subprocess`], [第 4、9 章],
  [`address already in use` / `MASTER_PORT`], [第 17 章],
  [`NCCL WARN` / `Watchdog caught collective operation timeout`], [第 17、22 章],
  [`torch._dynamo.exc.Unsupported` / graph break], [第 12、15 章],
  [`hit config.recompile_limit` / 反复重编译], [第 12、15 章],
  [loss NaN / inf / 数值不稳定], [第 2、5、10 章],
  [两次跑结果不一致 / TF32 / 加了 compile 没变快], [第 10、11、15 章],
)
