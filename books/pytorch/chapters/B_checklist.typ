#import "../template.typ": *

#let ck = box(width: 7pt, height: 7pt, stroke: 0.5pt + rgb("#6b7280"), inset: 0pt)

= 附录 B：自测清单与复习路线

这一章不讲新知识，只做三件事：让你知道*自己漏了什么*、*该按什么顺序补*、*面试前一晚看什么*。

自测清单里的每一条都是"能力项"不是"知识点"——判据是*合上书能不能讲出来*，讲得磕磕巴巴等于不会。勾不上的按右边的章号回去读。

== 自测清单

=== Part 1 基础（第 1--5 章）

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center),
  [], [*能力项*], [*章*],
  [#ck], [能用 stride 公式解释 tensor 的内存布局，说清哪些 op 是纯元数据操作], [1],
  [#ck], [能说清 `view` 和 `reshape` 的区别，以及 `view` 失败的确切条件], [1],
  [#ck], [能区分 `expand` 和 `repeat`（stride=0 的视图 vs 真拷贝）], [1],
  [#ck], [能列出"共享内存"和"复制"两类操作，说清 `clone` / `detach` / `.data` 的差别], [1],
  [#ck], [能说清 bf16 和 fp16 的差别，以及为什么大模型训练偏好 bf16], [1],
  [#ck], [能手推广播规则，说清 `matmul` 的批维广播怎么算], [2],
  [#ck], [能区分 basic indexing（返回视图）和 advanced indexing（返回拷贝）], [2],
  [#ck], [能用 `gather` / `scatter_add_` 写出 top-k 路由这类操作], [2],
  [#ck], [能解释 softmax 为什么要减 max，`log_softmax` 为什么比 `log(softmax)` 稳], [2],
  [#ck], [能说清 `nn.Module` 的四张注册表，以及为什么 python list 装子模块是致命错误], [3],
  [#ck], [能说清 `register_buffer` 与普通属性的差别、`persistent=False` 的用途], [3],
  [#ck], [能说清 `train()` / `eval()` 只影响哪两类层，以及 `eval()` 和 `no_grad()` 的正交性], [3],
  [#ck], [能说清 `num_workers` 的进程模型、`pin_memory` 为什么能加速], [4],
  [#ck], [能写一个处理变长序列的 `collate_fn`，并说清多 worker 下 RNG 的坑], [4],
  [#ck], [能判断 dataloader 是不是瓶颈（给出具体的测量方法）], [4],
  [#ck], [能默写训练循环七行，并解释每一行为什么在那个位置], [5],
  [#ck], [能说清 `zero_grad(set_to_none=True)` 省了什么、有什么行为差异], [5],
  [#ck], [能写出正确的 AMP + `GradScaler` + 梯度裁剪的调用顺序（顺序错了会怎样）], [5],
  [#ck], [能说清梯度累积的等价性条件，以及 BatchNorm 为什么是例外], [5],
  [#ck], [能说清 AdamW 和 Adam+L2 的区别（不是同一个东西）], [5],
)

=== Part 2 原理（第 6--11 章）

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center),
  [], [*能力项*], [*章*],
  [#ck], [能画出一个两层网络的反向图，标出 `grad_fn` 和叶子], [6],
  [#ck], [能说清 leaf / `requires_grad` / `.grad` 三者的关系，以及 `requires_grad` 的传播规则], [6],
  [#ck], [能给出 `.grad` 是 `None` 的完整排查清单（至少 5 条）], [6],
  [#ck], [能说清 `retain_graph` 什么时候真的需要，以及乱加的后果], [6],
  [#ck], [能区分 `detach` / `no_grad` / `inference_mode` 三者的语义差别], [6],
  [#ck], [能写一个 `autograd.Function` 并用 `gradcheck` 验证（含为什么必须 float64）], [6, 24],
  [#ck], [能解释 version counter 是什么、什么样的原地操作会炸], [6],
  [#ck], [能用 60 秒讲清一次 `a + b` 从 Python 到 kernel 的分层路径], [7],
  [#ck], [能说清 dispatcher 的 op $times$ DispatchKey 二维分发，以及 autograd 是一层 key], [7],
  [#ck], [能算出一个 Transformer 训练的显存构成（参数/梯度/优化器/激活各占多少）], [8],
  [#ck], [能解释 caching allocator，说清为什么 `empty_cache()` 通常没用], [8],
  [#ck], [能区分真 OOM 和碎片，并给出各自的处理手段], [8],
  [#ck], [能说清 kernel launch 是异步的，并列出至少 4 个隐式同步点], [9],
  [#ck], [能说清 `non_blocking=True` 生效的前提（pinned memory）], [9],
  [#ck], [能判断一个训练是 CPU-bound 还是 GPU-bound，并说出判据], [9, 11],
  [#ck], [能解释"设了 seed 结果还是不一样"的至少 3 种原因], [10],
  [#ck], [能说清 TF32 是什么、在 A100 上默认状态如何、什么时候要关], [10],
  [#ck], [能用 `torch.profiler` 抓一段 trace 并读出瓶颈在哪], [11],
  [#ck], [能算 MFU，并说清它为什么是判断训练效率的单一指标], [11],
)

=== Part 3 torch.compile（第 12--16 章）

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center),
  [], [*能力项*], [*章*],
  [#ck], [能说出 compile 的四层（Dynamo / AOTAutograd / 后端 / Inductor）各自负责什么], [12--14],
  [#ck], [能解释 Dynamo 怎么接管 Python 执行（字节码层面），以及为什么它比 TorchScript 成功], [12],
  [#ck], [能解释 guard 是什么，为什么编译产物是特化的], [12],
  [#ck], [能说清 graph break 是什么、常见触发源、怎么定位（`TORCH_LOGS`）], [12],
  [#ck], [能解释 AOTAutograd 为什么要在 dispatcher 层 trace，而不是在 Python 层], [13],
  [#ck], [能解释 functionalization 解决什么问题], [13],
  [#ck], [能说清 min-cut partitioner 在权衡什么（显存 vs 重算）], [13],
  [#ck], [能说清 Inductor 的主要收益来自 fusion，以及为什么 matmul 不靠 fusion], [14],
  [#ck], [能说清 `mode="reduce-overhead"` 就是 CUDA graph，以及它的使用前提], [15],
  [#ck], [能说清动态 shape 的三种处理方式，以及重编译超限的症状], [15],
  [#ck], [知道 compile 后数值有微小差异是预期行为，能解释为什么], [15],
  [#ck], [能说清 `torch.export` 和 `torch.compile` 的定位差别], [16],
  [#ck], [能说清 `jit.trace` 和 `jit.script` 各自的失败模式，以及为什么新项目不该用], [16],
)

=== Part 4 分布式（第 17--22 章）

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center),
  [], [*能力项*], [*章*],
  [#ck], [能精确说出 AllReduce / AllGather / ReduceScatter / AllToAll 的语义和通信量], [17],
  [#ck], [能推 Ring AllReduce 的 $2(N-1)/N dot |P|$，并解释为什么与 $N$ 基本无关], [17],
  [#ck], [能说清 `init_process_group` 干了什么、`torchrun` 设了哪些环境变量], [17],
  [#ck], [能说清 collective 的调用契约（所有 rank、相同顺序、相同 shape）], [17],
  [#ck], [能说清 `nn.DataParallel` 为什么被废弃], [18],
  [#ck], [能讲清 DDP 的 bucket + backward hook 机制，以及为什么要分 bucket], [18],
  [#ck], [能说清 `find_unused_parameters=True` 的代价和替代方案], [18],
  [#ck], [能说清 `no_sync()` 在梯度累积里的作用], [18],
  [#ck], [能说清 ZeRO 三级各切什么、省多少显存、多付多少通信], [19],
  [#ck], [能描述 FSDP 一次 forward/backward 的 gather-compute-reshard 流程], [19],
  [#ck], [能说清 `HYBRID_SHARD` 的动机，以及 FSDP1 与 FSDP2 的差别], [19],
  [#ck], [能画出 MLP 的 TP column-then-row 切法，说清为什么这个顺序只需一次 AllReduce], [20],
  [#ck], [能说清 Sequence Parallel 补了 TP 的哪块短板], [20],
  [#ck], [能算 PP 的 bubble 比例，说清 1F1B 相比 GPipe 省的是显存不是 bubble], [20],
  [#ck], [能说清 DeviceMesh / DTensor 的 placement 与 collective 的对应关系], [21],
  [#ck], [能给出分布式 hang 的完整排查流程（这是最高频的分布式面试题）], [22],
  [#ck], [能说清分片 checkpoint 怎么存、怎么在不同并行度之间 resume], [22],
)

=== Part 5 手写题（第 23--29 章）

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center),
  [], [*能力项*], [*章*],
  [#ck], [写完任何一个层，能马上说出"我会怎么验证它和官方实现一致"], [23],
  [#ck], [能说清 LayerNorm 的 `eps` 在 sqrt 里面还是外面，以及为什么], [23],
  [#ck], [能说清 BatchNorm 的有偏/无偏方差分别用在哪里（train vs running stats）], [23],
  [#ck], [能说清 Dropout 为什么在训练时除以 $1-p$（inverted dropout）], [23],
  [#ck], [能说清 `autograd.Function` 的 `save_for_backward` 与 `ctx.xxx =` 的区别], [24],
  [#ck], [能解释 STE 为什么过不了 `gradcheck`，而这是设计使然], [24],
  [#ck], [能默写 LSTM 的四个门公式，说清 PyTorch 的门顺序和双 bias], [25],
  [#ck], [能说清 GRU 的 reset gate 作用位置与原论文的差别], [25],
  [#ck], [能默写 SDPA，说清 $sqrt(d_k)$ 缩放为什么必要], [26],
  [#ck], [能默写 MHA，并说出 reshape/transpose 顺序写错时为什么不报错却算错], [26],
  [#ck], [能说清 GQA/MQA 省的是什么（KV cache 显存，不是计算）], [26],
  [#ck], [能说清 RoPE 的两条核心性质（保范数、内积只依赖相对位置）], [26],
  [#ck], [能写 KV cache 并说清预填充与解码两阶段的差别], [26],
  [#ck], [能说清 ViT 的 PatchEmbed 为什么可以用一个 `Conv2d(k=P, s=P)` 实现], [27],
  [#ck], [能写 CLIP 的对称 InfoNCE loss，说清随机初始化时 loss 应该约等于 $ln B$], [27],
  [#ck], [能写 MoE 的 top-k gating 和 load-balancing aux loss，说清后者的最小值是多少], [27],
  [#ck], [能写 LoRA，说清 B 为什么零初始化、merge 之后为什么零推理开销], [27],
  [#ck], [能写 label smoothing CE 并与 `F.cross_entropy(label_smoothing=)` 对齐], [28],
  [#ck], [能手写 AdamW 的一步更新（含 bias correction），并说清与 Adam+L2 的差异], [28],
  [#ck], [能说清 `clip_grad_norm_` 是全局范数而不是逐张量，且不会放大梯度], [28],
)

== 按岗位定制复习路线

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray, inset: 5pt, align: (left, left, left),
  [*岗位*], [*重点*], [*可以浅一点*],
  [算法 / 模型\ （偏训练调优）],
  [Part 1 全部、Part 2 的第 6/8/10 章、Part 5 全部。AMP、梯度累积、显存优化、手写模型是主战场],
  [Part 3 会用 `torch.compile` 即可；Part 4 只需 DDP 的完整机制 + FSDP 的概念（切什么、省多少）],
  [AI Infra /\ 训练框架],
  [Part 2 全部（dispatcher、显存、CUDA 执行、profiling 是核心）、Part 3 全部、Part 4 全部。Part 5 的第 26 章 Transformer 必须能手写],
  [Part 5 的第 25/27 章（经典模型、扩散/多模态）了解即可],
  [推理 / 部署],
  [Part 1、Part 2 的第 8/9 章（显存与 CUDA 执行）、Part 3 全部（尤其第 16 章 export/AOTInductor）、第 26 章的 KV cache 与 GQA],
  [Part 4 只需知道 TP 在推理里怎么用；Part 5 的训练技巧类（第 28 章）可跳],
  [应届 / 转行],
  [Part 1 打牢（这是筛人的第一道关）、第 6 章 autograd 讲清楚、Part 5 的第 23/25/26 章能手写],
  [Part 3、Part 4 知道名词和一句话定位即可，不要在这上面花时间——面试官不会指望应届生答],
)

#insight[
  岗位不确定时按这个优先级：第 1 章 $arrow.r$ 第 6 章 $arrow.r$ 第 5 章 $arrow.r$ 第 26 章 $arrow.r$ 第 8 章 $arrow.r$ 第 18 章。这六章覆盖了 PyTorch 面试里出现频率最高的一批题，不管什么岗位都会问。
]

== 时间预算

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray, inset: 5pt, align: (left, left, left),
  [*预算*], [*读什么*], [*跳过什么*],
  [1 天],
  [本章最后的"面试前一晚速览"完整过一遍；第 29 章题库快速自测；错的题只翻对应小节，不读整章],
  [所有原理章的推导过程、所有代码。这一档的目标是"不在送分题上翻车"，不是"答出难题"],
  [1 周],
  [第 1、5、6 章精读（约 2 天）；第 8、18、19 章（1 天）；第 12、15 章（1 天）；第 23、26 章边读边写代码（2 天）；最后半天做第 29 章题库],
  [第 7、10、13、14、16、20、21 章；Part 5 的第 25、27 章],
  [1 个月],
  [顺序读完全书。Part 5 的每道题自己敲一遍再跑对应 pytest；Part 2、Part 4 的每章读完合上书复述一遍面试考点框],
  [不跳。有余力的话按前言里的指引去读 `books/distributed-training/` 或 `books/cuda/`],
)

== 手写题清单

难度分三档：*L1* 必须能写（写不出来直接挂）、*L2* 应该能写（写不出来要能讲清思路和坑）、*L3* 能讲思路即可（面试官通常不会真让你写完）。

代码列给的是 `python/pytorch/interview/` 下的文件名。

#table(
  columns: (auto, 1fr, auto, auto, auto),
  stroke: 0.4pt + gray, inset: 4pt, align: (center, left, center, center, left),
  [], [*题目*], [*难度*], [*章*], [*代码*],
  [#ck], [Linear（含 weight 布局为什么是 `(out, in)`）], [L1], [23], [`test_layers.py`],
  [#ck], [Softmax（减 max 的数值稳定）], [L1], [23], [`test_layers.py`],
  [#ck], [LayerNorm], [L1], [23], [`test_layers.py`],
  [#ck], [RMSNorm], [L1], [23], [`test_layers.py`],
  [#ck], [BatchNorm1d（train/eval 双路径 + running stats）], [L2], [23], [`test_layers.py`],
  [#ck], [Dropout（inverted，eval 恒等）], [L1], [23], [`test_layers.py`],
  [#ck], [Embedding（backward 是 scatter-add）], [L2], [23], [`test_layers.py`],
  [#ck], [conv2d（im2col / unfold + GEMM）], [L3], [23], [`test_layers.py`],
  [#ck], [MaxPool2d（padding 补 `-inf` 不是 0）], [L2], [23], [`test_layers.py`],
  [#ck], [`autograd.Function` 写 ReLU（三件套模板）], [L1], [24], [`test_custom_autograd.py`],
  [#ck], [手推 Softmax 的 backward（雅可比不用显式建）], [L2], [24], [`test_custom_autograd.py`],
  [#ck], [手推 LayerNorm 的 backward], [L3], [24], [`test_custom_autograd.py`],
  [#ck], [STE（量化直通估计）], [L2], [24], [`test_custom_autograd.py`],
  [#ck], [GradientReversal（域自适应）], [L2], [24], [`test_custom_autograd.py`],
  [#ck], [用 `gradcheck` / `gradgradcheck` 验证自己的实现], [L1], [24], [`test_custom_autograd.py`],
  [#ck], [MLP（最后一层不加激活这个送分点）], [L1], [25], [`test_classic_models.py`],
  [#ck], [ResNet BasicBlock（相加之后才 ReLU）], [L2], [25], [`test_classic_models.py`],
  [#ck], [LSTMCell（门顺序 i-f-g-o、双 bias）], [L2], [25], [`test_classic_models.py`],
  [#ck], [多层 LSTM（时间维循环 + 层间堆叠）], [L3], [25], [`test_classic_models.py`],
  [#ck], [GRUCell（reset gate 的作用位置）], [L2], [25], [`test_classic_models.py`],
  [#ck], [Bahdanau / Luong attention + mask], [L2], [25], [`test_classic_models.py`],
  [#ck], [scaled dot-product attention], [L1], [26], [`test_transformer.py`],
  [#ck], [MultiHeadAttention（reshape/transpose 顺序）], [L1], [26], [`test_transformer.py`],
  [#ck], [causal mask + padding mask 的组合], [L1], [26], [`test_transformer.py`],
  [#ck], [GroupedQueryAttention / MQA], [L2], [26], [`test_transformer.py`],
  [#ck], [RoPE（cache 构造 + apply）], [L2], [26], [`test_transformer.py`],
  [#ck], [KV cache（预分配 buffer + 游标）], [L2], [26], [`test_transformer.py`],
  [#ck], [TransformerBlock（pre-norm + RMSNorm + SwiGLU）], [L2], [26], [`test_transformer.py`],
  [#ck], [MiniGPT + generate（temperature / top-k / top-p）], [L3], [26], [`test_transformer.py`],
  [#ck], [ViT 的 PatchEmbed + cls token + pos embed], [L2], [27], [`test_advanced_models.py`],
  [#ck], [CLIP 对称 InfoNCE loss], [L2], [27], [`test_advanced_models.py`],
  [#ck], [MoE top-k gating + load-balancing aux loss], [L2], [27], [`test_advanced_models.py`],
  [#ck], [DDPM 的 `q_sample` 闭式解 + 噪声预测 loss], [L3], [27], [`test_advanced_models.py`],
  [#ck], [DDPM 反向采样循环], [L3], [27], [`test_advanced_models.py`],
  [#ck], [LoRALinear（B 零初始化 + merge）], [L1], [27], [`test_advanced_models.py`],
  [#ck], [label smoothing cross entropy], [L2], [28], [`test_training_tricks.py`],
  [#ck], [focal loss（gamma=0 退化为 CE）], [L2], [28], [`test_training_tricks.py`],
  [#ck], [mixup / cutmix], [L1], [28], [`test_training_tricks.py`],
  [#ck], [ModelEMA（buffer 是拷贝不是平均）], [L2], [28], [`test_training_tricks.py`],
  [#ck], [warmup + cosine scheduler], [L1], [28], [`test_training_tricks.py`],
  [#ck], [SGD with momentum / Nesterov], [L2], [28], [`test_training_tricks.py`],
  [#ck], [AdamW（含 bias correction）], [L2], [28], [`test_training_tricks.py`],
  [#ck], [`clip_grad_norm_`（全局范数，不放大）], [L1], [28], [`test_training_tricks.py`],
)

L1 一共 15 题。*这 15 题必须全部能在白板上一次写对*，这是硬门槛；L2 有 22 题，是拉开差距的部分；L3 的 6 题只要能讲清思路。

== 面试前一晚：一页速览

+ *`view` vs `reshape`*：`view` 只改元数据、要求兼容 stride；`reshape` 能 view 就 view，不能就 copy。transpose/切片之后 `view` 会失败。
+ *`zero_grad(set_to_none=True)`*（2.0+ 默认）：把 `.grad` 置 `None` 而不是填 0，省一次 memset 和一份显存；副作用是没参与 forward 的参数梯度是 `None` 而不是 0。
+ *AMP 三件套顺序*：`autocast` 里只放 forward $arrow.r$ `scaler.scale(loss).backward()` $arrow.r$ `scaler.unscale_(opt)` $arrow.r$ `clip_grad_norm_` $arrow.r$ `scaler.step(opt)` $arrow.r$ `scaler.update()`。裁剪必须在 unscale 之后，否则裁的是放大过的梯度。bf16 不需要 GradScaler。
+ *`.grad` 是 `None`*：非叶子 / `requires_grad=False` / 没参与 forward / 路径上有 `detach` / 在 `backward` 之前看的。
+ *`retain_graph`*：报错让你加，但九成情况是你把带图的张量跨 step 累积了（`total_loss += loss`）。
+ *显存构成*（fp32 参数量 $P$ + Adam）：参数 $4P$ + 梯度 $4P$ + 一阶二阶动量 $8P$ = $16P$，激活另算且通常是大头。混合精度下另加 fp32 master weights。
+ *caching allocator*：`empty_cache()` 只还 reserved 不减 allocated，对真 OOM 无用。`reserved` 远大于 `allocated` 才是碎片。
+ *kernel launch 是异步的*：计时必须 `torch.cuda.synchronize()` 或用 CUDA event。`.item()` / `print(tensor)` / `.cpu()` 都是隐式同步点。
+ *DDP bucket*：梯度按 bucket（默认 25 MB）分组，一个 bucket 的梯度算完就发 AllReduce，实现通信与 backward 计算 overlap。不分 bucket 就只能等全部 backward 结束才通信。
+ *ZeRO 三级*：ZeRO-1 切优化器状态、ZeRO-2 加切梯度、ZeRO-3 加切参数。ZeRO-3 = FSDP，通信量是 DDP 的 1.5 倍（多一次参数 AllGather）。
+ *`torch.compile` 四层*：Dynamo 抓字节码出 FX graph（会 graph break）$arrow.r$ AOTAutograd 出前反向联合图 $arrow.r$ 后端 $arrow.r$ Inductor 生成 Triton kernel。主要收益来自 fusion 省 HBM 往返和 kernel launch。
+ *attention 缩放*：`softmax(Q K^T / sqrt(d_k)) V`。不除 $sqrt(d_k)$ 的话内积方差随 $d_k$ 线性增长，softmax 饱和、梯度消失。
+ *MHA 的 reshape 顺序*：`(B,S,H) -> view(B,S,nh,hd) -> transpose(1,2) -> (B,nh,S,hd)`。先 transpose 再 view 是经典 bug，不报错但算错。
+ *KV cache*：解码时只有新 token 的 Q，K/V 从 cache 取，把每步的 $O(S^2)$ 降到 $O(S)$。显存代价是 $2 dot L dot S dot H dot$ dtype 字节，GQA/MQA 就是来砍这一项的。
+ *AdamW vs Adam+L2*：Adam+L2 把 weight decay 加进梯度，会被自适应学习率的分母除掉，实际衰减强度和梯度大小耦合；AdamW 把它解耦，直接从权重里减 $"lr" times "wd" times w$。
+ *BatchNorm vs LayerNorm*：BN 沿 batch 维统计，依赖 batch size，train/eval 行为不同，分布式下要 SyncBN；LN 沿特征维统计，与 batch 无关，所以 Transformer 用它。
+ *pre-norm vs post-norm*：pre-norm 的残差路径是纯恒等，深层训练稳定、通常不需要 warmup；post-norm 效果上限略高但难训。现代 LLM 一律 pre-norm。
+ *梯度累积等价性*：loss 要除以累积步数才等价于大 batch。BatchNorm 是例外（统计量只在 micro-batch 内算）。DDP 下前 $k-1$ 步要用 `no_sync()`。
+ *graph break*：`print`、`.item()`、依赖张量值的 `if` 都会切图。`fullgraph=True` 把它变成报错，`TORCH_LOGS="graph_breaks"` 定位。
+ *TF32*：A100 上 fp32 matmul 可以走 TF32 tensor core（10 位尾数），快很多、精度略降。torch 2.10 上 `backends.cuda.matmul.allow_tf32` 默认 `False`、`backends.cudnn.allow_tf32` 默认 `True`——两个开关默认值不一样，这是个爱考的细节。

最后一条不写在纸上但比上面所有都重要：*不会的题按"划边界 $arrow.r$ 归约到会的 $arrow.r$ 说怎么验证"三步答*，别硬编。见前言最后一节。
