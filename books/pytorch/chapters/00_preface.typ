#import "../template.typ": *

= 前言

== PyTorch 面试到底在考什么

一场 PyTorch 面试基本按三层递进往下挖，每一层都可能是终点：

+ *API 层*：`view` 和 `reshape` 有什么区别？`zero_grad` 为什么要调？——答对了只能证明你写过代码。
+ *原理层*：`view` 为什么会失败？失败条件用 stride 怎么表达？`set_to_none=True` 为什么更省显存？——答对了说明你知道底下发生了什么。
+ *手写层*：白板上二十行写个 LayerNorm，写完解释 `eps` 为什么在 sqrt 里面。——答对了说明你不是背的。

市面上的材料要么是 API 教程（只有第一层），要么是八股题库（第二层的结论，没有推导过程，追问一句就穿帮）。这本书的写法是把三层焊在一起：每个知识点从"面试怎么问"出发，讲清机制，然后给一段真能跑的代码。

*定位是高频题，不是冷门八股。* 每一章的题都是真会被问到的——`view` vs `reshape`、autograd 的 `None` 梯度、DDP 为什么分 bucket、graph break 是什么、手写 MHA。凡是"面试官几乎不会问"的边角 API，一律没写。

环境是 torch 2.10.0+cu128 / CUDA 12.8 / A100-SXM4-80GB。版本敏感的地方都标了版本，PyTorch 2.x 的版本差异单独放在附录 C。

== 全书结构

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*部分*], [*章节*], [*读完能答上什么*],
  [Part 1\ 基础], [1--5], [张量占多少内存、哪个 op 偷偷 copy 了；广播和 `einsum`；`nn.Module` 的四张注册表和 `state_dict`；DataLoader 的 worker/`pin_memory`/`collate_fn`；训练循环七行里每一行的考点（AMP、梯度累积、裁剪、调度器）],
  [Part 2\ 原理], [6--11], [反向图怎么建、`.grad` 为什么是 `None`、`retain_graph` 什么时候真的需要；一次 `a + b` 走过 dispatcher 的哪几层；显存花在哪、caching allocator 和碎片；kernel launch 的异步语义与隐式同步点；为什么设了 seed 结果还不一样；怎么用 profiler 定位瓶颈],
  [Part 3\ compile], [12--16], [Dynamo 怎么改 Python 字节码、guard 和 graph break；AOTAutograd 怎么把反向也编出来、min-cut partitioner 的取舍；Inductor 的 fusion 收益从哪来；生产里 `mode` / 动态 shape / 编译时间怎么调；`torch.export` 与 AOTInductor 的部署路线],
  [Part 4\ 分布式], [17--22], [集合通信原语的精确语义与通信量估算；DDP 的 bucket + backward hook；ZeRO 三级省多少、FSDP 的 gather/reshard 流程；TP 的 column-then-row 与 PP 的 bubble；DeviceMesh/DTensor；hang 怎么排查、分片 checkpoint 怎么存],
  [Part 5\ 手写题], [23--29], [白板写 Linear / LayerNorm / BatchNorm / conv2d；自定义 `autograd.Function` 与 `gradcheck`；LSTM / GRU / Bahdanau attention；MHA / GQA / RoPE / KV cache / MiniGPT；ViT / CLIP loss / MoE / DDPM / LoRA；label smoothing / mixup / EMA / warmup-cosine / 手写 AdamW；最后一章是限时题库],
  [附录], [A--C], [A：拿着报错原文查成因和修法；B：五百多条自测项 + 按岗位的复习路线 + 面试前一晚速览；C：PyTorch 2.x 版本差异，答"你用的什么版本"这类题],
)

Part 1--4 有依赖（Part 3 假设你读过第 6 章的 autograd，Part 4 假设你读过第 5 章的训练循环）；Part 5 是独立的，任何时候都能翻开练。

== 配套代码

Part 5 的每道题都有一份可运行的参考实现 + pytest 对齐测试，在 `python/pytorch/interview/`。总共 6 个文件、约 6100 行、186 个测试。

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left),
  [*文件*], [*对应章*], [*测试数*], [*内容*],
  [`test_layers.py`], [23], [31], [Linear / LayerNorm / RMSNorm / BatchNorm1d / Dropout / Embedding / Softmax / conv2d / MaxPool2d],
  [`test_custom_autograd.py`], [24], [33], [`autograd.Function` 三件套、STE、GradientReversal、`gradcheck` / `gradgradcheck`],
  [`test_classic_models.py`], [25], [31], [MLP / ResNet BasicBlock / LSTMCell / GRUCell / Bahdanau / Luong / Seq2Seq],
  [`test_transformer.py`], [26], [32], [SDPA / MHA / GQA / RoPE / mask / KVCache / SwiGLU / MiniGPT / generate],
  [`test_advanced_models.py`], [27], [26], [ViT / CLIP loss / MoE 路由与 aux loss / DDPM / LoRA],
  [`test_training_tricks.py`], [28], [33], [label smoothing / focal / mixup / cutmix / EMA / warmup-cosine / 手写 SGD 与 AdamW / clip-grad-norm],
)

跑法（全部是 CPU 测试，秒级）：

```bash
cd /path/to/playground-cuda
pytest python/pytorch/interview/ -q                       # 全跑，186 passed
pytest python/pytorch/interview/test_transformer.py -q    # 单文件
pytest python/pytorch/interview/ -q -k "rope or kv_cache" # 按关键字挑
```

这些测试的价值不在"能跑"，在*验收标准*：几乎每道题都把官方实现的权重拷进手写版本，用 `torch.testing.assert_close` 做逐元素比对。面试里写完代码接一句"我会把 `nn.LSTMCell` 的权重拷进来做数值对齐"，比任何解释都有说服力。

== 三种读法

*(a) 一周突击。* 时间只够读一遍的话：第 1、5、6 章（张量 + 训练循环 + autograd，这三章的题占了基础面的一半），第 8、18、19 章（显存 + DDP + FSDP），第 12、15 章（Dynamo + compile 实践），Part 5 的第 23、26 章。然后直接做第 29 章的限时题库自测，错哪题回去补哪章。

*(b) 系统学习。* 顺序读，每章的代码自己敲一遍——特别是 Part 5，看懂和写出来是两回事。手写题写完就跑对应的 pytest 验证，不要停留在"我觉得对"。

*(c) 查漏。* 直接翻附录 B 的自测清单，逐条问自己"不看书能不能答出来"，勾不上的条目按后面标的章号回去读。这是复习到后期最有效率的用法。

== 前置知识，与不讲什么

假设你会 Python、写过 PyTorch 训练脚本、知道什么是反向传播和 Adam。

不讲：Python 语法基础；深度学习入门理论（梯度下降怎么来的、CNN 为什么有效）；具体业务模型的调参经验（数据配比、超参搜索）；也不讲 CUDA kernel 怎么写。

== 与仓库里另外三本书的关系

这本是横向的入口书，另外三本是纵向的深挖：

- `books/distributed-training/`：分布式训练实战。本书 Part 4 讲到"能在面试里把 DDP/FSDP/TP/PP 说清楚"为止；ZeRO 的完整推导、Ring Attention、DualPipe、FP8、MegaScale 级容错在那本。
- `books/cuda/`：CUDA kernel 优化。本书第 9、14 章讲到"知道 kernel launch 和 fusion 在干什么"为止；怎么手写一个 kernel、怎么用 `ncu` 调 occupancy 在那本。
- `books/moe/`：Sparse MoE 专题。本书第 27 章只写到面试常考的 top-k gating + load-balancing loss；grouped GEMM、all-to-all、capacity 设计在那本。

面到分布式或推理岗、且这本书的 Part 4 已经能顺畅讲下来的话，去读那两本。

== 被问到不会的题，怎么答

这是最被低估的一项能力。面试官问到你不会的东西，硬编是最差的选项——编错一个细节，后面所有回答的可信度一起打折。

正确的答法有三步：

+ *说清你知道的边界。* "FSDP2 我只用过 `fully_shard` 的默认配置，`reshard_after_forward` 的取舍我没实测过。" 明确划线，面试官立刻知道该往哪边追问，反而显得你对自己的知识有把握。
+ *把它归约到你会的东西。* "但它本质上是 ZeRO-3，参数在 forward 前 AllGather、用完 reshard，通信量是 DDP 的 1.5 倍——按这个模型推，`reshard_after_forward=False` 应该是拿显存换掉一次 AllGather。" 这一步展示的是推理能力，比背对答案更值钱。
+ *说你会怎么验证。* "我会跑一个 profile 看 AllGather 有没有消失，再用 `torch.cuda.max_memory_allocated` 对比峰值。" 能说出验证手段，说明你平时是靠测量而不是靠猜。

三步说完，一道不会的题也能拿到分。整本书的面试考点框都是按这个思路写的：给结论，也给结论怎么来的，以及怎么自己验证。

翻页开始。
