#import "../template.typ": *

= 快问快答题库

前面 28 章每章结尾的「面试考点」是*按主题深挖*的题：一道题能聊五分钟。这一章是另一种东西 —— *收口题库*，全是"一句话判断"型的短题，用于最后一周的冲刺自测。

怎么用它：*先盖住答案，出声把答案说出来，再对照*。默读会骗自己 —— 你以为知道，但说不出流畅的两句话，面试时就会卡。一轮下来把说不出的标记下来，只复习标记的那些，第二轮通常只剩 20% 不到。每道题的目标是 *15 秒内说完*；说到 30 秒还没说完，说明你在现场组织语言而不是在回忆结论，这一条要回去重看对应章节。

最后一节「开放题与故障排查」不一样：那 10 道题的答案是*结构化的排查/决策清单*，不是一句话。这类题是高级岗位的分水岭 —— 面试官不在意你猜的根因对不对，在意你有没有一套"从现象到根因"的系统方法。练它们的方式是*把清单的顺序和每一步的判据说出来*，而不是背结论。

== 基础与 Tensor

#interview[
  *Q*：`x.T` 能用在 3-D 张量上吗？

  A：不能。`.T` 对非 2-D 张量已经废弃（新版本直接报错），因为它的语义是"反转所有维度"，对高维几乎总不是你想要的。要转最后两维用 `x.mT` 或 `x.transpose(-2, -1)` —— batched matmul 场景下一律用这两个。
]

#interview[
  *Q*：`torch.empty` 和 `torch.zeros` 差在哪？什么时候必须用 `zeros`？

  A：`empty` 只分配显存不写值，内容是上一次这块显存的残留；`zeros` 多一次 memset。所以只有"马上就会被整块覆盖"的 buffer 才用 `empty`（比如 `torch.empty(n, out=...)` 的接收端）。要累加进去的、要当 mask 的、要作为 `index_add_` 目标的，一律 `zeros` —— 用 `empty` 会得到随机 NaN，而且是不可复现的。
]

#interview[
  *Q*：`nn.Parameter(t)` 会拷贝 `t` 吗？

  A：不会，它和 `t` *共享 storage*。所以之后原地改 `t` 会改到参数上。要独立就写 `nn.Parameter(t.clone())`。同理 `torch.as_tensor(x)` 尽量不拷贝，而 `torch.tensor(x)` *总是*拷贝 —— 热路径上用 `as_tensor` 或 `from_numpy`。
]

#interview[
  *Q*：`x.contiguous()` 在 `x` 已经连续时会 copy 吗？

  A：不会，直接返回 `x` 本身（同一个对象）。所以在不确定的地方无脑加 `.contiguous()` 的代价是零 —— 除非真的不连续。反过来说，看到 `.contiguous()` 出现在 profiler 的热点里，说明上游确实产生了非连续张量，该去看的是那里。
]

#interview[
  *Q*：`x.numel() * x.element_size()` 一定等于它占的显存吗？

  A：不一定，这算的是*逻辑元素*占的字节。如果 `x` 是一个 view/slice，它背后的 storage 可能大得多（`x = big[0]` 会让整个 `big` 都活着）。看真实占用要用 `x.untyped_storage().nbytes()`。这也是"切了一小片却没省显存"这类问题的根源。
]

#interview[
  *Q*：`x.storage_offset()` 什么时候非 0？

  A：切片跳过了开头的元素时，比如 `x[1:]`、`x[:, 2:]`。它和 shape、stride 一起构成 view 的完整描述：`storage[offset + sum(i_k * stride_k)]`。手写 kernel 或做 `from_blob` 时忘了 offset 是经典 bug。
]

#interview[
  *Q*：`torch.cat` 和 `torch.stack` 的区别？

  A：`cat` 在*已有*维度上拼接，输出维度数不变；`stack` 新建一个维度，输出维度数 +1，且要求所有输入 shape 完全相同。`stack([a, b], 0)` 等价于 `cat([a[None], b[None]], 0)`。
]

#interview[
  *Q*：`a[mask]` 的输出 shape 是静态的吗？

  A：不是，它是*数据相关*的 —— 输出长度等于 `mask.sum()`，只有真的跑一遍才知道。后果是：`torch.compile` 遇到它会 graph break 或引入 unbacked symint，`torch.export` 需要显式约束，CUDA graph 直接不支持。想保持静态 shape 就改成"乘 mask 保留原形状"或 `masked_fill`。
]

#interview[
  *Q*：为什么不建议给 `torch.arange` 传浮点 `step`？

  A：元素个数是 `ceil((end - start) / step)` 算出来的，浮点误差会让它在边界上多一个或少一个元素，而且跨平台不一致。要固定个数就用 `torch.linspace(start, end, steps)`。
]

#interview[
  *Q*：`torch.tensor(1) / torch.tensor(2)` 得到什么？

  A：`0.5`（一个 float 张量），不是整数 0。`/` 是 true division，会把整型提升成默认浮点 dtype。要整除用 `//` 或 `torch.div(..., rounding_mode="floor")`。这在算 index 时最容易出事 —— 得到 float 索引会直接报错。
]

#interview[
  *Q*：`torch.equal` 和 `torch.eq` 的区别？

  A：`torch.equal(a, b)` 返回一个 Python `bool`，同时比较 shape 和全部数值，不广播；`torch.eq(a, b)`（即 `==`）返回逐元素的 bool 张量并且*会广播*。所以 `if a == b:` 在张量上会报 "Boolean value of Tensor is ambiguous"。浮点比较一律用 `torch.allclose` 或 `torch.testing.assert_close`。
]

#interview[
  *Q*：`model.half()` 和 AMP 有什么区别？

  A：`half()` 把*所有*参数和 buffer 永久转成 fp16，优化器状态也变成 fp16，很容易在几百步后发散。AMP 保留 fp32 的参数，只在 forward 时对*选定的 op*（matmul、conv）autocast 成低精度，softmax / LayerNorm / loss 归约仍在 fp32。训练用 AMP，纯推理才考虑 `half()`。
]

#interview[
  *Q*：为什么混合精度训练需要 fp32 的 master weight？

  A：因为 `lr * grad` 常常比权重本身的 ULP（最小可分辨间隔）还小。fp16 只有 10 位尾数、bf16 只有 7 位，`w + tiny` 会被直接舍回 `w`，更新静默丢失（swamping）。用 fp32 存权重、低精度只用于计算，就避开了这个问题。
]

#interview[
  *Q*：`nn.Module.float()` / `half()` 是原地的吗？

  A：是（`nn.Module.to()` 系列对 module 都是原地的，返回 `self` 只是为了链式写法）。而 `Tensor.to()` *不是*原地的，必须接返回值。这个不对称是新手最常翻车的地方之一。
]

== autograd 与训练

#interview[
  *Q*：`loss.backward()` 之后 `loss` 这个张量还能用吗？

  A：能，`loss` 本身的数值不变，`loss.item()` 照常。被释放的是图里的中间 buffer（`save_for_backward` 存的东西），所以不能再 backward 第二次。但要注意：*不要把 `loss` 张量存进 list 做日志* —— 那会让整张图跟着活着，几十步后 OOM。存 `loss.item()` 或 `loss.detach()`。
]

#interview[
  *Q*：`model.train()` / `model.eval()` 具体改变了哪些层的行为？

  A：只有依赖"当前是训练还是推理"的层：Dropout 系（eval 时变恒等）和 BatchNorm 系（eval 时用 running stats 而不是当前 batch 统计）。LayerNorm、Linear、Conv 完全不受影响。所以一个只有 LayerNorm 的 Transformer，忘了 `eval()` 唯一的后果是 dropout 没关。
]

#interview[
  *Q*：`scheduler.step()` 和 `optimizer.step()` 的顺序？搞错会怎样？

  A：*先 `optimizer.step()`，再 `scheduler.step()`*，每个 iteration 各一次（epoch 级 scheduler 则每个 epoch 一次）。反过来会让第一步用错误的 lr，整条曲线偏移一格；更常见的错是在构造完 scheduler 后又手动多调一次 `step()`，效果一样。torch 2.x 会对明显的顺序错误发 warning。
]

#interview[
  *Q*：`p.grad` 的 dtype 一定和 `p` 一样吗？

  A：是的，autograd 保证 `p.grad` 和 `p` 的 dtype、shape、device 都一致。AMP 下参数是 fp32，所以梯度也是 fp32 —— 低精度只存在于 forward 的中间计算里。这也是为什么 AMP 省的是 activation 显存，不是梯度显存。
]

#interview[
  *Q*：一个参数同时出现在两个 `param_group` 里会怎样？

  A：它会被更新*两次*，等效 lr 翻倍，而且不报错。这在"按 weight decay 分组"时很容易发生 —— 分组逻辑用 `name` 匹配，某个参数同时命中两条规则。自检方法：`sum(len(g["params"]) for g in opt.param_groups)` 必须等于 `len(list(model.parameters()))`，并且把所有 `id(p)` 放进 set 检查无重复。
]

#interview[
  *Q*：weight decay 该不该作用在 bias 和 LayerNorm 的参数上？

  A：不该。这些参数是"平移/缩放"性质的，把它们拉向 0 没有正则意义，反而会损害表达能力（LayerNorm 的 `weight` 被拉到 0 等于关掉这一层）。标准做法是分两个 param group：`ndim >= 2` 的（权重矩阵、embedding）给 wd，`ndim < 2` 的（bias、norm 的 weight/bias）wd 设 0。GPT-2 / LLaMA 的官方训练脚本都是这么分的。
]

#interview[
  *Q*：`F.dropout` 忘了传 `training=self.training` 会怎样？

  A：`F.dropout` 的 `training` 默认是 `True`，*不会*跟着 `model.eval()` 变。所以推理时 dropout 照样在丢，输出带随机性、指标莫名偏低。这是用 functional API 而不用 `nn.Dropout` 模块的最大风险 —— `nn.Dropout` 会自动读 `self.training`。
]

#interview[
  *Q*：Dropout 在 eval 时要不要除以 `1-p`？

  A：不用，因为 PyTorch 用的是 *inverted dropout*：训练时保留下来的激活已经被除过 `1-p` 了，所以期望和 eval 时一致，eval 直接恒等输出即可。老教材里"训练不缩放、推理乘 `1-p`"是另一种等价写法，但推理时多一次乘法，所以现代框架都选前者。
]

#interview[
  *Q*：`model.zero_grad()` 和 `optimizer.zero_grad()` 有区别吗？

  A：作用范围不同。`optimizer.zero_grad()` 只清 `param_groups` 里的参数，`model.zero_grad()` 清这个 module 的所有参数。绝大多数情况两者等价；不等价的场景是"optimizer 只管了模型的一部分"（比如冻结了 backbone、或者有多个 optimizer），这时候用哪个要想清楚。两者的 `set_to_none` 默认都是 `True`。
]

#interview[
  *Q*：`cross_entropy` 的 `target` 能是 float 吗？

  A：能，但语义不同。整型 `(N,)` 是类别索引，走 NLL 路径；浮点 `(N, C)` 被当成*概率分布*（torch 1.10+ 支持），走软标签路径。所以知识蒸馏、mixup 的软标签可以直接喂。混淆这两种会得到 "expected scalar type Long but found Float" 或者一个数值完全不对的 loss。
]

#interview[
  *Q*：`nn.Embedding` 的 `sparse=True` 有什么用？代价是什么？

  A：让它的梯度变成稀疏张量（只包含这一步真正用到的 row），省掉"整个 `(V, D)` 稠密梯度"的显存和 memset —— 词表 10 万时这是几百 MB。代价是*只有少数 optimizer 支持稀疏梯度*（`SGD`、`SparseAdam`、`Adagrad`），AdamW 不支持；而且和 DDP 的梯度桶机制配合很差。所以大模型基本不用它，靠 tied embedding 和 TP 切分解决。
]

#interview[
  *Q*：Adam 的 `beta2` 调大（比如 0.999 $arrow.r$ 0.9999）会怎样？

  A：二阶矩的平均窗口从约 1000 步拉长到约 1 万步，对梯度尺度的突变反应更迟钝 —— 好处是更稳（大 batch、长训练常这么调），坏处是遇到真实的 loss spike 恢复更慢，而且前期需要更长的 warmup（因为 bias correction 的分母 $1-beta_2^t$ 更久才接近 1）。
]

== 显存与性能

#interview[
  *Q*：batch size 翻倍，显存一定翻倍吗？

  A：不会。只有 activation 部分随 batch 线性增长；参数、梯度、优化器状态是*固定开销*，与 batch 无关。所以小模型大 batch 时接近线性，大模型小 batch 时几乎不变。这个拆分也是估显存的正确方式：先算固定项（每参数 16 字节量级的 AdamW 混合精度），剩下的才是 activation 预算。
]

#interview[
  *Q*：两个进程共用一张卡，为什么特别容易 OOM？

  A：因为每个进程有*独立的 caching allocator*，各自缓存的空闲块对另一个进程完全不可见、也不会让出。加上每个 CUDA context 本身就要几百 MB，两个进程的"可用显存"远小于总量的一半。所以同卡多进程要么设 `PYTORCH_CUDA_ALLOC_CONF` 限制单进程上限，要么干脆用 MPS / 单进程多流。
]

#interview[
  *Q*：显存碎片怎么判断？`expandable_segments` 干什么的？

  A：判据是 `memory_reserved() - memory_allocated()` 很大却仍然 OOM —— 说明空闲显存被切成了很多小块，凑不出一个连续的大块。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 让 allocator 用可增长的虚拟地址段而不是固定大小的 segment，变长 shape 的训练上通常能显著缓解碎片。
]

#interview[
  *Q*：`nvidia-smi` 显示 GPU-Util 100%，说明算力跑满了吗？

  A：*完全不能这么推。* GPU-Util 的定义是"过去一段采样窗口里，至少有一个 kernel 在执行的时间比例"。一个极小的 kernel 连续不断地跑，利用率也是 100%，但 SM 占用率可能只有 2%。要看真实算力利用得算 MFU，或者用 `ncu` 看 achieved occupancy 和 memory throughput。
]

#interview[
  *Q*：KV cache 占多少显存？给公式。

  A：$2 times L times S times H_"kv" times d_h times B times "bytes"$ —— 系数 2 是 K 和 V 两份。LLaMA-2-7B（$L = 32$、$H_"kv" d_h = 4096$、bf16）算下来是*每 token 每层 16 KB、全模型 512 KB/token*，4k 上下文单条序列就是 2 GB。这就是 GQA / MQA（减小 $H_"kv"$）和 PagedAttention（减少碎片浪费）要解决的问题。
]

#interview[
  *Q*：`torch.inference_mode()` 能省显存吗？

  A：能省一点，但主要不是靠它。它不建图，所以中间 activation 用完即释放（这部分和 `no_grad()` 一样）；额外省的是 view 记账和 version counter 的元数据，量很小。它真正的收益是*速度* —— 少了记账开销。要在推理上省显存得靠低精度、量化、KV cache 管理。
]

#interview[
  *Q*：gradient checkpointing 和 CPU offload 怎么选？

  A：先 checkpointing。它换的是*计算*（多一次 forward，吞吐掉 20%--30%），而 offload 换的是 *PCIe 带宽* —— 单卡 A100 的 PCIe 是 32 GB/s 量级，比 HBM 慢 60 倍，很容易让 GPU 空转等数据。只有 checkpointing 已经开满还是不够、或者要 offload 的是"整个 step 只用一次"的优化器状态时，offload 才划算。
]

#interview[
  *Q*：`persistent_workers=True` 解决什么问题？

  A：默认每个 epoch 结束时 DataLoader 会销毁所有 worker 进程，下个 epoch 重新 fork —— 每次都要重新 import、重建 Dataset 对象、重开文件句柄。epoch 短或 Dataset 初始化重（比如要加载索引文件）时，这个开销能占到几个百分点。设成 `True` 就让 worker 活到 DataLoader 被销毁为止。代价是 worker 一直占着内存。
]

#interview[
  *Q*：为什么推理时 batch 越大吞吐越高但延迟越差？

  A：因为解码阶段是 memory-bound —— 每生成一个 token 都要把全部权重从 HBM 读一遍，这个代价与 batch 无关。所以把 $B$ 条请求批在一起，权重只读一次，吞吐几乎线性涨。但单条请求必须等整个 batch 一起算完，还可能要等凑批，所以 P99 延迟变差。continuous batching（vLLM）就是在这个权衡上做优化。
]

#interview[
  *Q*：`max_memory_allocated()` 什么时候需要 reset？

  A：它记录的是*进程启动以来*的峰值，所以想测"某一段代码的峰值"必须先 `torch.cuda.reset_peak_memory_stats()`。典型用法：warmup 几步（让 allocator 缓存稳定）→ reset → 跑一个 step → 读峰值。不 reset 就会一直读到 warmup 或模型加载时的旧峰值，得出错误结论。
]

#interview[
  *Q*：`non_blocking=True` 拷到 GPU 后立刻在 CPU 上改源张量会怎样？

  A：数据竞争 —— 拷贝还在 DMA 引擎里跑，你已经把源 buffer 改了，GPU 上可能拿到新旧混合的内容，而且不报错、不可复现。规则是：异步拷贝的源 buffer 在拷贝完成前不能碰。要么用 `torch.cuda.current_stream().synchronize()`，要么用双 buffer 轮换（DataLoader prefetch 就是这么做的）。
]

#interview[
  *Q*：为什么大模型训练要把 `fused` / `foreach` optimizer 打开？

  A：因为朴素实现是"逐参数一个小 kernel"。一个 7B 模型有几百个参数张量，AdamW 每步要对每个做 5--6 次 elementwise op，就是几千次 kernel launch，纯 CPU 开销能占到 step 的百分之几十。`foreach=True`（默认）用 multi-tensor apply 把同 dtype 同 device 的张量批成一个 kernel；`fused=True` 更进一步，把整个 AdamW 更新融进一个 kernel。代价是 `fused` 要求参数都在 CUDA 上且是浮点，且对 `state_dict` 的兼容性略敏感。
]

== torch.compile

#interview[
  *Q*：`torch.compile(model)` 和 `model.compile()` 有区别吗？

  A：有，而且是个实际的坑。`torch.compile(model)` 返回一个 `OptimizedModule` 包装，它的 `state_dict` 的 key 全部多了 `_orig_mod.` 前缀 —— 存下来的 checkpoint 加载回未编译的模型会全部 key mismatch。`model.compile()`（torch 2.2+）是*原地*编译，key 不变。存 checkpoint 时要么用后者，要么存 `model._orig_mod.state_dict()`。
]

#interview[
  *Q*：`torch.compile` 能加速 backward 吗？

  A：能，而且反向往往是收益更大的一半。AOTAutograd 在编译时就把反向图一起 trace 出来交给 Inductor（见第 13 章），所以反向也享受 fusion 和 kernel 生成。这也是为什么 compile 只需要包住 forward：`loss.backward()` 走的是编译好的反向图，不需要（也不能）单独编译。附带效应是反向的 activation 保存策略由 min-cut partitioner 决定，所以显存占用可能和 eager 不同。
]

#interview[
  *Q*：`fullgraph=True` 是干什么的？

  A：把 graph break 从"静默降级"变成"直接报错"。它不会让代码变快，它是个*调试开关*：想确认整个模型真的编成了一张图（CUDA graph、export 的前置条件）就打开它，报错信息会指出是哪一行导致了 break。日常训练不要开，因为很多 break 是无害的。
]

#interview[
  *Q*：`dynamic=True` / `False` / `None` 分别什么意思？

  A：`None`（默认）是*自动*：第一次按静态 shape 编译，看到第二种 shape 时把变化的维度标成动态、重编译一次，之后不再重编。`True` 从一开始就假设动态，省掉那次多余的静态编译（变长输入场景推荐）。`False` 强制静态，每种 shape 都单独编一份 —— shape 种类少且想要极致性能时用。
]

#interview[
  *Q*：编译过的模型能直接 `torch.save` 吗？

  A：不能可靠地存 —— `OptimizedModule` 里挂着编译后的 callable，pickle 不了。正确做法是*只存 `state_dict`*（注意上面说的 key 前缀问题），加载时重新构造模型再 compile。要真正序列化编译产物得走 `torch.export` + AOTInductor（见第 16 章）。
]

#interview[
  *Q*：编译区里 `print(x)` 会发生什么？

  A：一次 graph break。`print` 需要张量的*真实值*，而 Dynamo 在 trace 时手里只有 FakeTensor，所以它只能把图切开、退回 eager 执行这一行、再重新开图。调试时想看中间值，用 `TORCH_LOGS="+graph_breaks,output_code"` 或者 `torch._dynamo.graph_break()` 显式标记，别靠 `print`。
]

#interview[
  *Q*：Inductor 的编译缓存存在哪？换机器能复用吗？

  A：默认在 `/tmp/torchinductor_$USER`（可以用 `TORCHINDUCTOR_CACHE_DIR` 改）。同机器上二次启动能命中，冷启动时间从几十秒降到几秒。换机器*有条件*可以复用：需要相同的 torch 版本、相同的 GPU 架构（`sm_80` 的产物不能给 `sm_90` 用）、相同的编译配置。CI 里预热这个目录是很常见的优化。
]

#interview[
  *Q*：`torch._dynamo.config.cache_size_limit` 是什么？超了会怎样？

  A：同一个代码位置允许缓存的已编译版本数上限（默认 8）。每种不同的 guard 组合（shape、dtype、Python 类型、某个 bool 标志）都占一份。超过之后 Dynamo *放弃编译、静默退回 eager*，只在 `TORCH_LOGS="recompiles"` 里留一行日志 —— 表现就是"跑了一会儿突然变慢且没人报错"。看到它超限不要直接调大上限，先去查为什么有这么多版本（通常是 shape 没 bucket，见第 15 章）。
]

== 分布式

#interview[
  *Q*：`torchrun` 和 `mp.spawn` 怎么选？

  A：新代码一律 `torchrun`。它帮你设好 `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `MASTER_ADDR` 等环境变量，支持多机、支持 `--max-restarts` 做故障重启、和 rendezvous 后端集成。`mp.spawn` 是在一个 Python 进程里 fork 子进程，只能单机，而且父进程一旦提前 import 了 CUDA 就会出问题。
]

#interview[
  *Q*：`RANK`、`LOCAL_RANK`、`WORLD_SIZE` 分别是什么？

  A：`RANK` 是全局进程编号（$0$ 到 `WORLD_SIZE`$-1$），`WORLD_SIZE` 是总进程数，`LOCAL_RANK` 是本机内的编号。*绑设备用 `LOCAL_RANK`*（`torch.cuda.set_device(local_rank)`），*判断"是不是主进程"用 `RANK == 0`*。混用是新手最常见的错：用 `RANK` 绑设备，第二台机器上就会去访问不存在的 `cuda:8`。
]

#interview[
  *Q*：哪些事只该 rank0 做？

  A：写日志和 TensorBoard、保存 checkpoint（除非用 DCP，见第 22 章）、下载数据集或预训练权重、打印进度条、上报监控指标。核心判据是*有外部副作用的操作*。反过来，*所有 collective 必须全 rank 都调用* —— 把 `all_reduce` 写在 `if rank == 0:` 里就是直接 hang。
]

#interview[
  *Q*：`dist.barrier()` 什么时候真的需要？

  A：只有当"某个 rank 的非通信副作用"必须先于其他 rank 的动作时才需要 —— 典型是 rank0 下载/解压数据集，其他 rank 等它完成再打开文件。普通训练循环里*不需要* barrier，因为 collective 本身就是同步点，多余的 barrier 只会掩盖负载不均衡（把 straggler 的等待从通信挪到 barrier 上，profiler 里更难看出来）。NCCL 后端调 barrier 记得传 `device_ids`。
]

#interview[
  *Q*：DDP 下各 rank 的 dropout mask 应该一样吗？

  A：*不应该一样*，而且默认就不一样（各 rank 的 RNG 状态独立演化）。因为各 rank 处理的是不同数据，dropout 本来就该独立采样 —— 强行同步等于把有效 batch 的正则化多样性砍掉。必须一样的是*模型初始化*，DDP 构造时会从 rank0 broadcast 一次参数和 buffer 来保证这一点。
]

#interview[
  *Q*：用了 `DistributedSampler` 还要不要给 `DataLoader` 传 `shuffle=True`？

  A：不要，会直接报错（`sampler` 和 `shuffle` 互斥）。shuffle 交给 sampler：`DistributedSampler(ds, shuffle=True)`，并且每个 epoch 调 `sampler.set_epoch(epoch)`，否则每个 epoch 的顺序完全一样。
]

#interview[
  *Q*：8 卡 DDP、per-GPU batch 16、梯度累积 4 步，effective batch size 是多少？

  A：$16 times 8 times 4 = 512$。三个因子都要算进去 —— 这是调 lr 时最容易少算一项的地方（尤其是加了累积之后忘了同步调整 lr 和 warmup 步数）。另外要注意"总 step 数"的定义：optimizer 更新的次数是 dataloader 迭代次数的 $1\/4$，scheduler 的 `total_steps` 必须按前者算。
]

#interview[
  *Q*：为什么 DDP 要求所有 rank 的模型结构完全相同？

  A：因为 DDP 在构造时按*参数的注册顺序*划分梯度 bucket，所有 rank 必须得到完全一致的 bucket 划分和 AllReduce 顺序。哪怕只是某个 rank 多了一个 buffer 或者层的定义顺序不同，通信就会错位 —— 表现为 hang（等一个永远不来的 collective）或者梯度被张冠李戴（更可怕，因为不报错）。
]

#interview[
  *Q*：`gradient_as_bucket_view=True` 干什么的？

  A：让 `p.grad` 直接指向 DDP 通信 bucket 里的一段（view），而不是单独分配一份再拷进 bucket。省掉一整份梯度的显存（约 $|P|$）和一次拷贝。副作用是 `p.grad` 的生命周期和 bucket 绑定，不要长期持有它的引用，也不要对它做改变 storage 的原地操作。
]

#interview[
  *Q*：多机训练 `MASTER_ADDR` 该填谁？

  A：*rank 0 所在节点*的地址（且必须是其他节点真正能连上的那个网卡地址，不是 `127.0.0.1`）。所有节点填同一个值。配套的 `MASTER_PORT` 要选一个空闲端口，同一台机器上跑多个任务时必须错开。连不上通常是防火墙或者填了内网/外网中不通的那一个。
]

#interview[
  *Q*：NCCL 报 "unhandled system error" 第一步查什么？

  A：先加 `NCCL_DEBUG=INFO` 拿到真实原因 —— 这个报错本身几乎不含信息。最常见的三类根因：网卡选错（设 `NCCL_SOCKET_IFNAME`）、共享内存不足（容器里 `/dev/shm` 太小，`--shm-size` 调大或设 `NCCL_SHM_DISABLE=1` 验证）、P2P/IB 不可用（`NCCL_P2P_DISABLE=1` 试一下能不能跑通来定位）。
]

== 手写题速答

#interview[
  *Q*：LayerNorm 和 RMSNorm 差几个操作？

  A：RMSNorm 去掉了*减均值*和*偏置 $beta$*，只除以均方根：$x \/ sqrt("mean"(x^2) + epsilon) dot gamma$。省一次求均值的 reduction 和一次减法，反向也少一项。实测在 LLM 上效果不掉，所以 LLaMA 之后基本成了默认选择。
]

#interview[
  *Q*：attention 为什么要除以 $sqrt(d_k)$？

  A：$q dot k$ 是 $d_k$ 个乘积之和，输入方差为 1 时它的方差是 $d_k$、标准差 $sqrt(d_k)$。$d_k = 128$ 时 logits 的量级就到 $plus.minus 11$，softmax 严重饱和、梯度接近 0。除以 $sqrt(d_k)$ 把方差归一化回 1。注意除的是 $sqrt(d_k)$（单头维度）而不是 $sqrt(d_"model")$。
]

#interview[
  *Q*：multi-head attention 的 head 数变多，参数量变吗？

  A：不变。$W_Q$、$W_K$、$W_V$、$W_O$ 都是 $d_"model" times d_"model"$，与 head 数无关 —— head 只是把这个矩阵的输出在通道维上*切分*成 $h$ 段，$d_h = d_"model" \/ h$。变的是 attention 矩阵的个数（$h$ 个 $S times S$）和 softmax 的粒度。所以 head 数是纯粹的"表达方式"超参，不影响参数预算。
]

#interview[
  *Q*：KV cache 为什么只 cache K/V 不 cache Q？

  A：因为解码时只需要*当前这一个 token* 的 $q$，它每步都是新的、下一步用不上；而所有历史 token 的 $k$、$v$ 每一步都要被重新用一次。这个不对称正是"prefill 是 compute-bound、decode 是 memory-bound"的根源。
]

#interview[
  *Q*：RoPE 和可学习绝对位置编码的关键区别？

  A：绝对 PE 是*加*在 embedding 上的一张可学习表，长度写死；RoPE 是对 $q$、$k$ 做一个与位置相关的*旋转*，使得注意力分数只依赖相对位置 $i - j$。后果是 RoPE 没有可学习参数、外推到更长上下文更容易（配合 NTK / YaRN 插值），而绝对 PE 换长度必须重训或插值（ViT 那一套，见第 27 章）。
]

#interview[
  *Q*：causal mask 应该填 `-inf` 还是一个很大的负数？

  A：用 `torch.finfo(dtype).min` 或直接 `-inf`，*不要*硬编码 `-1e9` —— fp16 的最大值是 65504，`-1e9` 会直接变成 `-inf` 之外的溢出行为。用 `-inf` 的唯一风险是"整行全被 mask"时 softmax 得到 `0/0 = NaN`（padding 全掩的行会遇到），所以变长序列要么保证每行至少有一个可见位置，要么事后把 NaN 行清零。
]

#interview[
  *Q*：`nn.Embedding` 的反向是什么操作？

  A：`index_add_` / `scatter_add_` —— 把上游梯度按 token id 累加进一个 `(V, D)` 的零张量。所以同一个 token 在 batch 里出现多次，梯度会正确累加；也所以这个反向是*原子加*，在 GPU 上是非确定性的（见第 10 章，`use_deterministic_algorithms` 会换一个慢但确定的实现）。
]

#interview[
  *Q*：dropout 放在 residual 相加之前还是之后？

  A：之前 —— `x = x + Dropout(Sublayer(LN(x)))`。dropout 作用在*子层的输出*上，residual 那条通路必须保持干净，否则梯度高速公路被随机切断，深层网络训不起来。放错位置的表现是收敛明显变慢但不报错。
]

#interview[
  *Q*：weight tying 能省多少参数？

  A：省一整个 $V times d$ 的输出投影。GPT-2 small（$V = 50257$、$d = 768$）省 38.6 M，占 124 M 总参数的 *31%*；LLaMA-7B（$V = 32000$、$d = 4096$）省 131 M，只占 *2%*。所以它对小模型是关键优化、对大模型基本可忽略 —— LLaMA 就没有 tie。
]

#interview[
  *Q*：SwiGLU 的 FFN 中间维度为什么常取 $8d\/3$ 而不是 $4d$？

  A：为了*保持参数量不变*。标准 FFN 有 2 个 $d times h$ 矩阵，$h = 4d$ 时参数是 $8d^2$；SwiGLU 有 3 个（gate、up、down），要让 $3 d h = 8 d^2$ 就得 $h = 8d\/3$。实现里还会把它向上取整到 128 或 256 的倍数以对齐 tensor core。
]

#interview[
  *Q*：AdamW 的优化器状态占多少？

  A：$m$ 和 $v$ 各一份、都是 fp32，所以是*参数数量的 2 倍元素、8 字节每参数*。混合精度训练下还要加 fp32 master weight（4 字节），合计 12 字节；再加 bf16 的参数和梯度各 2 字节，就是常说的*每参数 16 字节*。7B 模型光这部分就是 112 GB —— 这是 ZeRO / FSDP 存在的全部理由（见第 19 章）。
]

#interview[
  *Q*：手写一个 op 之后，怎么快速确认它和 `torch.nn` 的实现等价？

  A：三步。*数值对齐*：随机输入上 `torch.testing.assert_close`，覆盖多组 shape 和边界（batch=1、序列长 1、含负数）。*梯度对齐*：同一份输入分别 backward，比对所有参数的 `.grad`；自定义 `autograd.Function` 还要在 float64 上过 `gradcheck`。*退化对齐*：找一个超参使你的实现精确退化成官方的某个已知情形（`gamma=0` 退化成 CE、`top_k=E` 退化成 dense）—— 这类测试抓 bug 的效率最高。
]

== 开放题与故障排查

这一节的题没有一句话答案。面试官想看的是*你有没有一套从现象到根因的系统方法*，以及你会不会在信息不足时先问清楚。回答的通用结构是：*先复述我的理解和假设 → 给出排查顺序和每步的判据 → 说明什么情况下推翻假设换方向*。

#interview[
  *Q*：训练前 20 万步都正常，某一步 loss 突然变 NaN，怎么排查？

  A：第 10 章给了 NaN 的高发位置清单（`log(0)`、除零、`sqrt` 的反向、softmax 整行被 mask、fp16 溢出）。这道题的重点不是那个清单，而是*"跑了很久才炸"这个时间信息该怎么用* —— 它排除了"实现本身就错"，把嫌疑集中在"某个特定 batch"或"某个缓慢累积的量"上。

  第一步是*把范围缩到一个 step*。三个判据同时看：`loss.item()`、`grad_norm`（`clip_grad_norm_` 的返回值本来就要算）、以及 AMP 的 `scaler.get_scale()`。如果 `grad_norm` 在炸之前几十步就在持续攀升 $arrow.r$ 是*缓慢发散*，看 lr / warmup / $beta_2$，属于超参问题。如果前一步一切正常、这一步直接 NaN $arrow.r$ 是*这个 batch 的问题*，去 dump 它。如果 `scaler` 的 scale 在反复减半 $arrow.r$ 是 fp16 溢出，换 bf16。

  第二步是*定位到层*。用 `register_full_backward_hook` 给每层挂一个 `isfinite` 检查，找到第一个出 NaN 的层；需要精确到行就开 `set_detect_anomaly(True)` 复现一次（慢好几倍，只用于复现）。注意区分"forward 就 NaN"和"forward 正常但 backward NaN"——后者指向导数在边界上发散的 op。

  第三步是*确认根因可复现*：把那个 batch 单独喂进去，看能不能稳定重现。能重现就是数据问题（越界 label、全 padding 的序列、脏样本）；不能重现就要怀疑非确定性来源（原子加、多卡不一致、显存越界的自定义 kernel）。

  最后说工程上的预防：*`grad_norm` 和每层激活的 max 应该是常态监控指标*。NaN 几乎总有几十步的前兆，有这两条曲线就能在事后五分钟定位，没有就只能重跑。再加一道廉价守卫：`if not torch.isfinite(total_norm): skip this step`，别让 NaN 进参数 —— 一旦进了参数，后面所有 step 都是 NaN，现场就被破坏了。
]

#interview[
  *Q*：给你 8 卡 A100-80G，训一个 13B 模型，你怎么配并行？

  A：先算显存账。13B 参数在 bf16 混合精度 + AdamW 下，固定开销约 $13 times 10^9 times 16 = 208$ GB，8 卡总共 640 GB —— 装得下，但单卡 80 GB 放不下 26 GB 的固定开销加 activation，所以*必须切*。

  方案排序：(1) *首选 FSDP / ZeRO-3*，`auto_wrap_policy` 按 transformer block 包，`HYBRID_SHARD` 在单机 8 卡上通常不必要（NVLink 带宽足够）。8 卡全切之后每卡固定开销降到 26 GB 左右，剩下 50+ GB 给 activation，配 gradient checkpointing 就能开到不错的序列长度和 batch。(2) 如果序列很长导致 activation 仍然爆，再叠 *TP=2 或 4*（机内 NVLink，绝不跨机）+ FSDP 在剩下的维度上做 DP，并开 sequence parallel 摊掉 LayerNorm 的 activation。(3) *PP 在单机 8 卡上一般不用* —— bubble 和实现复杂度都不划算，PP 是跨机才有价值。

  然后要问清的前提：序列长度多少（决定 activation 是不是主要矛盾）、要不要全参微调还是 LoRA（LoRA 的话优化器状态几乎归零，单卡 DDP 就够了）、吞吐目标是多少。最后一定说：*先跑一个 profile 拿 MFU，再决定继续调哪一维* —— 配置是量出来的不是猜出来的。
]

#interview[
  *Q*：如何判断训练是 CPU bound 还是 GPU bound？

  A：三个层次，从便宜到贵。

  (1) *最快的判据*：把 dataloader 换成一个反复吐同一个预先搬到 GPU 上的假 batch，其他不动。step time 明显变快 $arrow.r$ 瓶颈在数据侧（CPU / IO）；几乎不变 $arrow.r$ 瓶颈在 GPU 计算或通信。这个实验五分钟能做完，应该是第一步。

  (2) *看指标*：`nvidia-smi dmon` 或 DCGM 看 GPU-Util 的*时间序列*而不是瞬时值 —— 数据瓶颈的典型特征是"忙一段、空一段"的锯齿。同时看 CPU 的 `%usr`（如果某个 worker 进程吃满一个核，说明预处理是瓶颈）和磁盘 `iostat`。注意 GPU-Util 100% 不代表算力跑满（见前面那题）。

  (3) *看 profile*：`torch.profiler` 开 CPU + CUDA，看 timeline 上 GPU 那一行有没有空隙，以及空隙对齐到哪个 CPU 事件。空隙前面是 `enumerate(loader)` 就是数据；是 `all_reduce` 就是通信；GPU 密不透风但吞吐还是低，就去算 MFU、看是不是 kernel 本身效率低。

  确认是 CPU bound 之后的优化顺序：加 `num_workers` → `pin_memory` + `non_blocking` → `persistent_workers` → 把预处理挪到 GPU 或离线预处理成二进制格式（webdataset / mmap）→ 最后才考虑换更快的解码库（见第 4 章）。
]

#interview[
  *Q*：训练 loss 一路正常下降，但验证指标很差，怎么查？

  A：先分清是*过拟合*还是*有 bug*，两者的曲线形状不同：过拟合是 val loss 先降后升，bug 是 val 从一开始就不对或者和 train 差一个数量级。

  bug 优先级：(1) *train/eval 预处理不一致* —— 归一化的均值方差、resize 方式、tokenizer 的截断策略，这是最高频的原因。(2) *忘了 `model.eval()`* —— dropout 还在丢、BN 还在用当前 batch 统计。(3) *BN 的 running stats 有问题* —— batch 太小导致统计量噪声大，或者用了 EMA 却没同步 buffer（见第 28 章）。(4) *数据泄漏或标签错位* —— 划分时同一个来源的样本同时进了 train 和 val，或者 val 的 label 顺序被 shuffle 打乱了。(5) *评测代码本身写错* —— 先用"训练集的一小部分"当验证集跑一遍，如果这样指标也差，问题一定在评测路径而不是泛化。

  最后这个技巧值得单独记：*用 train 的子集当 val 是分离"泛化问题"和"代码问题"最快的实验。*
]

#interview[
  *Q*：复现一篇论文，指标差 2 个点，怎么定位？

  A：按"影响量级"从大到小查，别从超参开始猜。(1) *数据* —— 版本、预处理、增强、划分方式；数据差异通常就能解释好几个点。(2) *有效 batch size 和总训练步数* —— 论文常报"8 卡 × bs 32"，你在 2 卡上跑 bs 32 就差了 4 倍，lr 和 warmup 都得跟着改。(3) *学习率 schedule 的细节* —— warmup 步数、`min_lr`、是按 step 还是按 epoch。(4) *正则项* —— weight decay 有没有排除 bias/norm、label smoothing、EMA、mixup 的概率。(5) *评测协议* —— 是否用 EMA 权重、是否多尺度/TTA、指标定义是否一致。(6) 最后才是初始化和随机种子 —— 单个种子的方差常有 0.3--0.5 个点，*先用 3 个种子确认这 2 个点是不是噪声*。

  方法论上说一句会加分：*一次只改一个变量，并且优先改那些能把差距一次性抹掉大半的*。
]

#interview[
  *Q*：训练跑了几百步之后才 OOM，怎么办？

  A："跑了一会儿才 OOM"这个现象本身就是最强的线索 —— 说明*有东西在单调增长*，不是单步峰值不够。

  查这几处：(1) *把带图的张量存进了 list* —— `losses.append(loss)` 而不是 `loss.item()`，整张 autograd 图被保活，这是第一嫌疑人。(2) *变长输入* —— 序列变长时峰值随之涨，加上碎片就在某个长 batch 上炸；判据是看 `memory_reserved` 是否远大于 `memory_allocated`，修法是按长度分桶、或开 `expandable_segments`。(3) *误用 `retain_graph=True`* 或者跨 step 带 hidden state 没 `detach()`。(4) *缓存类的字典无界增长* —— 自己写的 KV cache、metrics 累加器。(5) *eval 循环没包 `no_grad()`* —— 每次验证都攒一堆 activation。

  工具：`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` 拿到分配栈，在 `pytorch.org/memory_viz` 上看是哪一行的分配在持续增长（见第 8 章）。这是这类问题的决定性手段，比逐行注释快得多。
]

#interview[
  *Q*：同一份代码换到新集群慢了 30%，怎么查？

  A：先*分层*：单卡慢，还是只有多卡慢？跑一个单卡 step time 对比就能分开，这决定了后面查计算还是查通信。

  单卡就慢：查 GPU 型号和功耗墙（`nvidia-smi -q -d PERFORMANCE` 看有没有 SW Power Cap 或热降频）、CUDA / cuDNN / torch 版本差异（TF32 默认值、cuDNN benchmark、新版 SDPA 后端选择都会变）、CPU 侧的 dataloader（新集群的存储可能慢得多，或者 `num_workers` 相对新的 CPU 核数不合适）。

  只有多卡慢：查拓扑（`nvidia-smi topo -m`，是不是从 NVLink 掉到了 PCIe）、网卡选择（`NCCL_DEBUG=INFO` 看它实际用了哪个 interface，有没有走上了慢的管理网口）、是否有 straggler（每 rank 记 step time，看是不是固定某张卡慢）、`/dev/shm` 大小。

  两边都要做的一件事：*在两个集群上跑同一个 profile，对比 step time 的 breakdown*（compute / comm / dataloader 三段的绝对值）。哪一段变长了，答案就在那一段里，不用继续猜。
]

#interview[
  *Q*：单卡训练正常，换 DDP 之后 loss 不降（不是 hang），怎么查？

  A：DDP 下"能跑但不收敛"的根因集中在几处。(1) *lr 没跟着有效 batch 调* —— 8 卡的有效 batch 是 8 倍，沿用单卡 lr 相当于 lr 小了 8 倍，表现就是"降得特别慢"。(2) *各 rank 的初始化不一致* —— 正常情况 DDP 构造时会 broadcast，但如果你在构造 DDP *之后*才改了参数（比如加载 checkpoint、重新初始化某一层），各 rank 就分叉了；验证方法是对所有参数做一次 `all_reduce` 求和再比对哈希。(3) *数据没正确切分* —— 忘了 `DistributedSampler`，每个 rank 都在跑全量数据，等于一个 epoch 重复了 8 次；或者忘了 `set_epoch`，每个 epoch 顺序完全一样。(4) *BatchNorm* —— 每张卡只用自己那 $1\/8$ 的 batch 算统计量，小 batch 下噪声大，换 `SyncBatchNorm` 或 GroupNorm。(5) *loss 归约方式不一致* —— 各 rank 的 batch 大小不等时，简单平均梯度就不等于全局平均。

  自检的一招：*把 world size 设成 1 用 `torchrun` 跑*。如果这样正常、2 卡就不对，问题一定在"跨 rank 的一致性"这一类；如果 world size 1 就已经不对，问题在你为分布式改的那些代码本身。
]

#interview[
  *Q*：给你一个别人写的训练脚本，怎么在一天内把吞吐提一倍？

  A：*先测量，按投入产出比排序，每改一步重测*。

  第一小时只做一件事：拿到 step time 的 breakdown（dataloader / compute / comm）和当前 MFU。没有这个数字，后面全是瞎猜。

  然后按"改动小、收益大"排：(1) 确认开了 *AMP*（bf16），这通常是最大的单项收益。(2) *dataloader*：`num_workers`、`pin_memory`、`persistent_workers`，如果 breakdown 显示数据占比高，这里几行代码就能吃掉一大块。(3) 消掉训练循环里的*隐式同步*（每步的 `.item()`、`print`、`assert` 张量条件）—— 见第 9 章。(4) `zero_grad(set_to_none=True)`、`torch.backends.cuda.matmul.allow_tf32 = True`。(5) *`torch.compile`* —— 收益通常 10%--40%，但要先确认 shape 稳定、没有密集 graph break。(6) 如果显存有余量，*加大 batch* 并按比例调 lr（memory-bound 的 op 直接受益）。(7) 通信占比高就调 `bucket_cap_mb`、关掉不需要的 `find_unused_parameters`。

  最后说一句边界：*不改变数值语义的优化先做完，再考虑会动收敛行为的（换 dtype、换 batch size、换 fused optimizer）*，而且后者必须跑一段曲线确认收敛没退化。
]

#interview[
  *Q*：估算训练一个 7B 模型需要多少卡多少天。

  A：用 Chinchilla + MFU 两步就能给出量级。

  (1) *总 FLOPs*：经验公式 $C approx 6 N D$（$N$ 参数量、$D$ 训练 token 数，系数 6 来自 forward 2 倍加 backward 4 倍）。7B 按 Chinchilla 最优配 20 token/参数是 140B token，$C approx 6 times 7 times 10^9 times 1.4 times 10^11 approx 5.9 times 10^21$ FLOPs。

  (2) *算力*：A100 的 bf16 峰值是 312 TFLOPS，实际 MFU 取 40%--50%（这是 7B 稠密模型在良好实现下的合理区间），即约 $1.3 times 10^14$ FLOP/s 每卡。

  (3) 单卡时间 $approx 5.9 times 10^21 \/ 1.3 times 10^14 approx 4.5 times 10^7$ 秒 $approx$ 525 卡·天。所以 *64 张 A100 约 8 天，128 张约 4 天*（假设扩展效率接近线性，实际要打 85%--95% 的折扣）。

  面试时要主动交代三件事：*系数 6 的来源*、*MFU 是假设值必须实测*、*没算上 checkpoint 恢复和失败重跑的开销（真实项目按 1.2--1.5 倍留余量）*。愿意把假设摆出来、并说"这些数我会先用小规模跑一个 scaling 实验校准"，比给一个精确到小数点的答案更让人放心。
]
