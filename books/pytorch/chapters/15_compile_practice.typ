#import "../template.typ": *

= 生产里用 torch.compile

前三章讲机制，这章讲怎么用不踩坑。面试里这部分问得比机制更细，因为它直接暴露你有没有真的把 `torch.compile` 推上过生产：参数怎么选、动态 shape 怎么办、数值对不上算不算 bug、和 AMP/checkpoint/DDP 怎么组合、什么时候干脆别用。

== 参数逐个说清

```python
torch.compile(model=None, *, fullgraph=False, dynamic=None,
              backend="inductor", mode=None, options=None, disable=False)
```

=== mode

torch 2.10 实测的合法取值（传错会报 `Unrecognized mode=...`）：`default`、`lite`、`reduce-overhead`、`max-autotune-no-cudagraphs`、`max-autotune`。

#table(
  columns: (auto, 1.5fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*mode*], [*做什么 / 适用场景*], [*代价*],
  [`default`],
    [fusion + codegen，不开 CUDA graph、不做 GEMM autotune。训练的默认选择],
    [编译时间数秒到数十秒],
  [`reduce-overhead`],
    [`default` + CUDA graph。小 batch、decode、CPU 开销占比高的场景],
    [shape 必须固定、要独占显存池、限制多（见下）],
  [`max-autotune`],
    [`reduce-overhead` + GEMM/卷积 autotune + coordinate descent 调参],
    [编译时间常涨到 3 倍以上，收益不保证],
  [`max-autotune-no-cudagraphs`],
    [要 autotune 但不能用 CUDA graph（有动态 shape、有 CPU 同步、PP 通信）时用这个],
    [同上，减去 CUDA graph 的限制],
  [`lite`],
    [torch 2.10 新增。默认对所有区域 fallback，只编译显式标注的区域，同时关掉大部分图级 pass],
    [收益也相应小；适合"只想编一小块"的渐进式落地],
)

=== dynamic

- `dynamic=None`（默认）：先按静态 shape 编一份；同一维度出现第二个不同的值时，Dynamo 自动把它符号化重编一份动态的（automatic dynamic shapes，由 `torch._dynamo.config.automatic_dynamic_shapes = True` 控制）。
- `dynamic=True`：一上来就假设 shape 是符号的。少编几次，但生成的 kernel 拿不到具体 size，某些 tiling 决策会保守。
- `dynamic=False`：禁止符号化，每种 shape 都单独编一份。这是最容易撞 `recompile_limit` 的设置。

=== fullgraph / backend / options

- `fullgraph=True`：遇到第一个 graph break 直接抛 `Unsupported`。开发期必用，见第 12 章。
- `backend`：`torch._dynamo.list_backends()` 在这套环境上实测返回 `['cudagraphs', 'inductor', 'openxla', 'tvm']`；调试用的 `"eager"` / `"aot_eager"` 不在这个列表里但可用（第 13 章）。
- `options`：直接塞 Inductor 配置，比 `mode` 细。`mode` 和 `options` 不能同时给。

```python
model = torch.compile(model, options={"max_autotune_gemm": True,
                                      "triton.cudagraphs": False})
```

== reduce-overhead 就是 CUDA graph

`mode="reduce-overhead"` 的实质是把 Inductor 生成的那串 kernel launch 用 CUDA graph 捕获下来，之后每步只 replay 一次。收益是彻底消掉 per-kernel 的 CPU launch 开销。

实测：8 层 `Linear(512, 512) + GELU`，bf16，`no_grad`，batch = *1*，A100-SXM4-80GB：

#figure(
  align(center, hbar-chart(
    (("eager", 0.371), ("compile default", 0.413), ("reduce-overhead", 0.127)),
    unit: "ms", width: 7,
  )),
  caption: [batch=1 时 GPU 几乎没活干，耗时全是 launch 开销。CUDA graph 直接把它砍掉 2.9×；而 `default` 因为 kernel 数没少多少，反而比 eager 慢一点。],
)

这组数字是理解 `reduce-overhead` 的关键：*它和 fusion 是两条独立的收益来源*。batch 大、GPU 饱和时它几乎没用；batch 小、层数多、单 kernel 很短时它是唯一的解。LLM 推理的 decode 阶段（每步只算一个 token）是最典型的受益场景。

限制（都是 CUDA graph 本身的限制）：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*限制*], [*后果与对策*],
  [输入 shape 必须固定],
    [每种 shape 要重新捕获一张图。动态 shape 场景先 pad 到桶],
  [输入张量地址必须固定],
    [PyTorch 用一个静态输入 buffer + `copy_` 处理这件事；但如果你自己持有输出张量的引用跨 step 使用，会读到被下一次 replay 覆盖的数据],
  [捕获期间不能有 CPU 同步],
    [`.item()`、`.cpu()`、`print(tensor)` 都会让捕获失败或触发 fallback],
  [独占一个显存池],
    [峰值显存可能上升。换配置必须重测 `max_memory_allocated()`],
  [动态 P2P 通信不兼容],
    [pipeline parallel 的 send/recv 在 graph 里地址固定不了，PP 场景用 `default` 或 `max-autotune-no-cudagraphs`],
)

#warn[
  最常见的 CUDA graph 事故：把编译后模型的输出直接存进一个 list 攒起来（比如收集每步的 logits），过几步之后发现前面的值全被改了。CUDA graph 的输出写在固定地址上，下一次 replay 就覆盖。要跨 step 保留就 `.clone()`。需要手动切分 step 边界时用 `torch.compiler.cudagraph_mark_step_begin()`。
]

== 动态 shape

变长序列是 `torch.compile` 在生产里最大的敌人。每个新 shape 都是一次 guard 失败：

```text
[0/1] [__recompiles] Recompiling function f in train.py:3
[0/1] [__recompiles]     triggered by the following guard failure(s):
[0/1] [__recompiles]     - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 5
```

=== automatic dynamic shapes

好消息是默认行为已经帮了大忙。实跑一个函数、输入长度依次为 4、5、6：第 5 次触发一次 recompile（上面那条日志），到 6 就*不再重编*了——Dynamo 发现这一维变过，自动把它换成符号 `s0`，编出一份对该维通用的图。

坏消息是这个机制只在"变化模式简单"时有效。多维同时变、或者 batch 与 seq 组合出很多种情况时，编译份数照样爆。用 `dynamic=False` 强制静态则完全没有这层保护——实测 16 种不同长度的输入，编到第 8 份就撞上限：

```text
W [0/8] torch._dynamo hit config.recompile_limit (8)
W [0/8]    function: 'f' (/tmp/v4_cachelimit.py:3)
W [0/8]    last reason: 0/7: tensor 'x' size mismatch at index 0. expected 11, actual 12
```

之后这个函数永久 fallback 到 eager，*不报错*。

=== 用 `mark_dynamic` 精确标注

想精确控制哪一维动态，用 `torch._dynamo.mark_dynamic`。它的完整签名（实测）：

```python
torch._dynamo.mark_dynamic(t, index, *, hint_override=None,
                           min=None, max=None, specialize_on=None)
```

```python
x = torch.randn(6, 16)
torch._dynamo.mark_dynamic(x, 0, min=2, max=128)   # 第 0 维动态，范围给出来
```

标注之后 Dynamo 抓到的图多了一个 `SymInt` 参数，实跑输出：

```text
def forward(self, s13 : torch.SymInt, L_t_ : torch.Tensor):
    l_t_ = L_t_
    sin = l_t_.sin();  l_t_ = None
    sum_1 = sin.sum(-1);  sin = None
    return (sum_1,)
```

给 `min` / `max` 不是可选的装饰：范围能让 Inductor 选更合理的 tiling，也能让 shape 相关的断言在编译期就求解掉，而不是在图里留一堆运行时检查。`maybe_mark_dynamic` 是"能动态就动态，不行就算了"的软版本。

=== 0/1 specialization

*shape 为 0 或 1 的维度总是被特化*，即使你写了 `dynamic=True`。实测：`dynamic=True` 编译一个函数，依次喂 batch = 1、2、3、4，只在 1 → 2 时发生一次 recompile：

```text
Recompiling function g in <string>:3
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 1, actual 2
```

原因是 size 1 会改变广播语义（`(1, N)` 能广播到 `(M, N)`，`(2, N)` 不能），size 0 会让很多 op 走空张量的特殊路径。所以符号 shape 的隐含约束是 `s >= 2`。

#warn[
  这个规则在推理服务里会咬人：batch=1 的请求和 batch=8 的请求走两份不同的编译产物。如果你的 warmup 只用 batch=1 跑过，线上第一个非 1 的 batch 会在请求路径上触发一次几十秒的编译。warmup 必须覆盖所有会出现的桶，包括 1。
]

=== bucketing：朴素但最常用

比 `mark_dynamic` 更常见的生产解法是把输入 pad 到固定的几个长度：

```python
BUCKETS = (128, 256, 512, 1024, 2048)

def pad_to_bucket(x, lengths):
    s = x.shape[1]
    tgt = next(b for b in BUCKETS if b >= s)
    return torch.nn.functional.pad(x, (0, 0, 0, tgt - s))
```

代价是浪费一部分计算（平均而言，桶分得越粗浪费越多），换来的是编译份数固定为桶的个数、每份都是静态 shape（于是 `reduce-overhead` 也能用）。桶的数量要留在 `recompile_limit` 以内，或者相应调大它。

== 编译时间

冷启动从几秒到几分钟。实测 `Linear(4096,4096)+GELU+Linear(4096,4096)` 这么小的模型，干净缓存下默认 mode 就要 4.8 s，`max-autotune` 要 13.6 s。真实的 LLM 训练脚本第一步花几分钟是常态。

缓解手段，按性价比排：

+ *只编热点模块，不编整个 model。* 一个 transformer 的 N 层是同构的，编一层就够——`nested_compile_region` 或者直接 compile 单个 block 类，能把编译时间从 O(N) 压到 O(1)。
  ```python
  for blk in model.blocks:
      blk.forward = torch.compile(blk.forward, dynamic=False)
  ```
+ *把动态维标好。* 每次 recompile 都是一次完整的编译时间。这是最大的单项收益。
+ *缓存落在持久盘上。* `TORCHINDUCTOR_CACHE_DIR` 指到持久目录，按 torch 版本 + GPU 型号分开。注意缓存只覆盖 Inductor 及以后，实测冷 4.8 s 热 3.8 s（第 14 章）。
+ *别在 CI 或调参循环里开 `max-autotune`。* 它的收益需要实测确认，编译时间的代价是确定的。

== 调试流程

按这个顺序走，不要跳步：

+ *`fullgraph=True`* —— 先确认能不能整图编。挂了就有异常和源码行号，直接改代码消除 graph break。
+ *`torch._dynamo.explain(fn)(args)`* —— 不想让它抛异常时，一次拿到 `graph_count` / `graph_break_count` / 每个 break 的原因和 user stack。
+ *`TORCH_LOGS="graph_breaks,recompiles"`* —— 跑真实训练，看有没有稳态阶段还在反复重编。`recompiles_verbose` 能看到所有失败的 guard 检查，不只是第一个。
+ *确认没撞上限* —— `grep "hit config.recompile_limit"`。撞了就先解决 shape，别急着调大 limit。
+ *`TORCH_LOGS="output_code"`* —— 怀疑"编了但没融合"时看生成的 kernel 数量和名字。
+ *怀疑正确性* —— `torch._dynamo.config.suppress_errors`（默认就是 `False`，别去改它）确保编译错误抛出来而不是静默 fallback；然后用 `backend="eager"` → `"aot_eager"` → `"inductor"` 三段夹逼定位是哪一层的问题（第 13 章）。

```bash
TORCH_LOGS="graph_breaks,recompiles" TORCH_LOGS_OUT=/tmp/compile.log python train.py
grep -c "__graph_breaks" /tmp/compile.log
grep "hit config.recompile_limit" /tmp/compile.log
```

== 数值差异不是 bug

编译后的结果*不会*和 eager bitwise 相同。三个原因：

+ *fusion 改了运算顺序。* 归约的累加顺序、中间结果是否落回低精度，都会变。
+ *可能换了 GEMM 实现。* `max-autotune` 下尤其明显（cuBLAS → Triton template，split-k 策略不同）。
+ *TF32 / 精度策略。* Inductor 生成的 Triton kernel 里的 `ALLOW_TF32` 和 `ACC_TYPE` 未必和 ATen kernel 的选择一致（第 14 章那段 `config_args` 里能直接看到这两项）。

实测 `Linear(1024,4096)+GELU+Linear(4096,1024)`、bf16、A100：`torch.equal(eager, compiled)` 为 `False`，最大绝对误差 `0.0078125`（正好是该量级下 bf16 的一个 ulp）。用 `torch.testing.assert_close` 卡容差，`rtol=atol=1e-2` 通过。

怎么验证"等价"：

- *单步*：`torch.testing.assert_close(a, b, rtol=..., atol=...)`，容差按 dtype 定（bf16 放到 1e-2 量级是合理的，fp32 可以到 1e-5）。不要用 `torch.equal`，也不要在 bf16 上要求 `rtol=1e-5`。
- *训练*：比 loss 曲线。跑几百步，compile 和 eager 的 loss 应该在噪声范围内重合。这是唯一有说服力的判据——单步 1e-2 的相对误差看着大，但它和"换一个随机种子"的影响是同一量级。
- *怀疑真有 bug 时*：用 `backend="aot_eager"` 对比，把 Inductor 摘出去。

#warn[
  比 loss 曲线时先把随机性对齐：`dropout` 的 RNG、dataloader 的 shuffle、`torch.use_deterministic_algorithms`。否则你比的是两条不同的随机轨迹，看不出编译到底有没有引入偏差。相关细节见第 10 章。
]

== 常见坑清单

#warn[
  *在 `forward` 里 `print` / `.item()` / `.cpu()`。* 现象：`explain` 报一堆 break，加速接近零。修法：调试打印挪到 `forward` 外，或者包一层 `@torch.compiler.disable`；loss 的 `.item()` 放到 `backward()` 之后的训练循环里。
]

#warn[
  *每步换 shape。* 现象：日志里 `__recompiles` 刷屏，稳态阶段还在编译。修法：`mark_dynamic` 标注动态维，或 pad 到桶。别用 `dynamic=False`。
]

#warn[
  *`recompile_limit` 静默 fallback。* 现象：训练速度慢慢退回 eager，没有报错。修法：上线前 `grep "hit config.recompile_limit"`；根治靠减少 shape 变化。
]

#warn[
  *第一步慢被当成"训练变慢"。* 现象：加了 compile 之后第一个 step 几十秒。修法：性能统计从第 10 步之后开始算；先跑一次覆盖所有桶的 warmup。
]

#warn[
  *编译 optimizer step 但忘了 `capturable=True`。* 现象：optimizer 里的 `step` 计数、`lr` 之类是 CPU 标量，每次都触发 guard 失败或 CPU 同步。修法：`torch.optim.AdamW(..., capturable=True)` 让状态全在 GPU 上。实测在 torch 2.10 上 `torch.compile` 包住 `opt.step()` 配 `capturable=True` 可以正常跑。
]

#warn[
  *`if self.training:` 导致两份图。* 这不是坑而是必然：`training` 是 guard 的一部分。实测 `model.eval()` 之后再调用，日志给出 `- 0/0: fn._modules['1'].training == True`，触发一次重编。train 和 eval 各一份编译产物是正常的，只要别在训练循环里反复来回切。
]

#warn[
  *compile 与 hook 的交互。* 注册在编译区域内部张量上的 backward hook，AOTAutograd 在前向 trace 时看不到，可能被忽略或引起 graph break。要靠它就开 `torch._dynamo.config.compiled_autograd = True`（第 13 章），或者把 hook 挪到编译边界之外的参数上。
]

#warn[
  *`torch.compile` 包了 dataloader 或 metric 逻辑。* 这些代码里全是 Python 对象操作，Dynamo 会疯狂 break 且毫无收益。只 compile 纯计算的部分。
]

== 与其他特性组合

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*特性*], [*怎么配*],
  [AMP / autocast],
    [推荐 `autocast` 在外、`compile` 在内（`compile(model)` 然后在训练循环里 `with autocast(...)`）。放在被编译的函数*里面*也能 trace——实测 `graph_count 1, breaks 0`——但放外面语义更清楚，且 autocast 的 dtype 会进 guard],
  [activation checkpoint],
    [必须 `use_reentrant=False`。粒度放到整个 block。能整图编译的模型优先改用 `activation_memory_budget`（第 13 章）],
  [DDP],
    [先 `DDP(model)` 再 `torch.compile`。`torch._dynamo.config.optimize_ddp` 默认 `True`（等价于 `"ddp_optimizer"`），按 DDP bucket 大小把图切成多段，让通信能早发从而与计算 overlap。另有 `"python_reducer"`（需要 compiled autograd，不靠 graph break 做 overlap）和 `"no_optimization"`（不切图，但没有 overlap）],
  [FSDP2],
    [较新版本支持度已经不错，DTensor 的 sharding 元信息在编译期可见。仍建议对单个 block compile 而不是整个 FSDP 包装体],
  [optimizer],
    [`torch.compile` 可以编 `opt.step()`，把逐参数的 elementwise 更新融成少量 kernel（对参数多的模型收益明显）。前提是 `capturable=True` 或用 `foreach` 实现],
  [pipeline parallel],
    [别用 `reduce-overhead`；动态 P2P 与 CUDA graph 不兼容],
)

生产推荐 idiom：

```python
model = MyModel().cuda()

# 1) 分布式包装在最外，compile 在里面（或直接 compile 每个 block）
model = DDP(model, device_ids=[local_rank])

# 2) 开发期先用 fullgraph=True 把图打通，确认后再放开
model = torch.compile(model, dynamic=False)   # shape 稳定时；变长就去掉或 mark_dynamic

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

# 3) autocast 在 compile 外层
for step, batch in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step == WARMUP:                 # 4) 性能统计从 warmup 之后开始
        timer.reset()
```

配套的环境变量：

```bash
export TORCHINDUCTOR_CACHE_DIR=/persistent/inductor_cache/torch2.10-a100
export TORCH_LOGS="graph_breaks,recompiles"     # 只在验证阶段开
export TORCH_LOGS_OUT=/tmp/compile.log
```

== 什么时候不该用

+ *shape 高度动态且没法 bucket。* 比如输入 shape 由数据内容决定（检测框数量、稀疏图的节点数）。编译份数控制不住，收益被重编吃光。
+ *模型很小、CPU 开销占主导，且 shape 不固定。* CPU 开销大本该用 `reduce-overhead`，但 shape 不固定又用不了 CUDA graph，两头堵。
+ *频繁改代码的实验阶段。* 改一行模型就全量重编。用 `torch.compiler.set_stance("force_eager")` 或 `disable=True` 一键关掉，跑通了再开。
+ *编译时间不可接受。* 短时任务（跑几十个 step 的 ablation）、或者按请求冷启动的 serverless 推理。后者的正确答案是 `torch.export` + AOTInductor（第 16 章）。
+ *已经被通信或数据加载卡住。* step time 的 60% 花在 AllReduce 或 dataloader 上时，把计算再快 20% 也看不出来。先 profile（第 11 章）。

#insight[
  面试里被问"你怎么决定要不要上 compile"，答案的结构应该是：先 profile 确认瓶颈在 GPU 计算上；看 kernel 分布判断是 memory-bound 的碎 kernel 多（fusion 有戏）还是 launch 开销占主（CUDA graph 有戏）；再看 shape 稳不稳定决定 `dynamic` 和能不能用 `reduce-overhead`；最后用 loss 曲线验证数值等价。给不出这条链的人，通常是抄了别人的 `torch.compile(model)` 然后发现没变快。
]

== 面试考点

#interview[
  *Q1*：`torch.compile` 的 `mode` 有哪些？分别什么时候用？

  A：torch 2.10 上是 `default`、`lite`、`reduce-overhead`、`max-autotune-no-cudagraphs`、`max-autotune`。`default` 是训练默认，只做 fusion 和 codegen。`reduce-overhead` 额外开 CUDA graph，用在小 batch / decode / CPU 开销占主的场景。`max-autotune` 再加 GEMM autotune，编译时间常涨 3 倍以上且收益不保证，要实测。有动态 shape 或 CPU 同步、不能用 CUDA graph 但想 autotune 时用 `max-autotune-no-cudagraphs`。
]

#interview[
  *Q2*：`mode="reduce-overhead"` 的原理和限制？

  A：原理是用 CUDA graph 捕获 Inductor 生成的 kernel launch 序列，之后每步 replay 一次，消掉 per-kernel 的 CPU launch 开销。实测 8 层 `Linear(512,512)` bf16 batch=1 在 A100 上：eager 0.371 ms、默认 compile 0.413 ms、reduce-overhead 0.127 ms。限制：shape 必须固定、输入输出地址固定（跨 step 保留输出必须 `.clone()`）、捕获期不能有 CPU 同步、独占一个显存池、与动态 P2P 通信不兼容（PP 别用）。
]

#interview[
  *Q3*：变长序列为什么会反复 recompile？怎么解决？

  A：shape 是 guard 的一部分，每个新 shape 触发 guard 失败 → 重编。Dynamo 的 automatic dynamic shapes 会在某一维出现第二个值时自动把它符号化，所以简单情况第 3 种 shape 起就不再重编；但多维同时变时编译份数照样爆，撞到 `recompile_limit`（默认 8）就永久 fallback 到 eager 且只打一条 warning。解法：`torch._dynamo.mark_dynamic(x, dim, min=..., max=...)` 标注动态维并给出范围；或者更常用的 pad 到固定的几个桶（桶数留在 limit 以内，且每桶都是静态 shape 所以还能用 `reduce-overhead`）。
]

#interview[
  *Q4*：什么是 0/1 specialization？

  A：shape 为 0 或 1 的维度总是被特化，即使写了 `dynamic=True`。因为 size 1 改变广播语义（`(1,N)` 能广播到 `(M,N)`，`(2,N)` 不能）、size 0 会走空张量的特殊路径。所以符号 shape 隐含 `s >= 2`。实测 `dynamic=True` 下 batch 依次为 1、2、3、4，只在 1 → 2 时重编一次。工程后果：推理服务里 batch=1 和 batch>1 是两份产物，warmup 必须覆盖包括 1 在内的所有桶，否则线上第一个非 1 的 batch 会在请求路径上触发几十秒编译。
]

#interview[
  *Q5*：compile 之后结果和 eager 不一样，是 bug 吗？

  A：不 bitwise 相同是正常的。原因：fusion 改变了归约的累加顺序和中间结果的精度、可能换了 GEMM 实现（`max-autotune` 下尤其）、TF32 与累加 dtype 的选择可能和 ATen kernel 不同。实测 bf16 的两层 MLP，最大绝对误差 0.0078（约一个 bf16 ulp），`assert_close(rtol=1e-2, atol=1e-2)` 通过。验证方式：单步用 `assert_close` 放宽容差（bf16 到 1e-2 量级），训练用 loss 曲线在几百步内重合，怀疑真有 bug 时用 `backend="aot_eager"` 把 Inductor 摘出去对比。
]

#interview[
  *Q6*：讲一下你排查 `torch.compile` 性能问题的流程。

  A：先 `fullgraph=True` 确认能不能整图编，挂了就按异常里的源码行改；不想抛异常就用 `torch._dynamo.explain` 看 `graph_break_count` 和每个 break 的原因；跑真实训练开 `TORCH_LOGS="graph_breaks,recompiles"`（配 `TORCH_LOGS_OUT` 落盘）看稳态还在不在重编；grep `hit config.recompile_limit` 确认没静默 fallback；怀疑"编了但没融合"就看 `TORCH_LOGS="output_code"` 里 kernel 的数量和名字；怀疑正确性用 `eager` → `aot_eager` → `inductor` 三段夹逼定位层。
]

#interview[
  *Q7*：DDP 和 `torch.compile` 谁包谁？为什么？

  A：先 `DDP(model)` 再 `torch.compile`。因为 `torch._dynamo.config.optimize_ddp` 默认开（`"ddp_optimizer"`），它会按 DDP 的 bucket 大小把图切成多段，让每个 bucket 的梯度一算完就能发 AllReduce，从而和后面的计算 overlap；如果反过来 compile 出一整张大图，所有梯度到图的最后才出来，通信完全串行在计算之后。另有 `"python_reducer"` 模式配合 compiled autograd，能不靠切图就 overlap。
]

#interview[
  *Q8*：怎么压缩编译时间？

  A：性价比从高到低：只 compile 热点模块——transformer 的 N 层同构，compile 单个 block 或用 `nested_compile_region` 把编译时间从 O(N) 压到 O(1)；把动态维标好，因为每次 recompile 都是一次完整编译；`TORCHINDUCTOR_CACHE_DIR` 指到持久盘并按 torch 版本 + GPU 型号分开（注意只覆盖 Inductor 及以后，实测冷 4.8 s 热 3.8 s）；别在 CI 和调参循环里开 `max-autotune`。彻底消掉启动开销要靠 `torch.export` + AOTInductor。
]

#interview[
  *Q9*：什么情况下不该用 `torch.compile`？

  A：shape 由数据内容决定又没法 bucket（检测框数、稀疏图节点数）——编译份数控制不住；模型小、CPU 开销占主但 shape 又不固定——想用 CUDA graph 用不了，两头堵；频繁改模型代码的实验阶段——改一行全量重编，用 `set_stance("force_eager")` 或 `disable=True` 关掉；编译时间不可接受的短时任务或按请求冷启动的推理（后者该用 AOTInductor）；瓶颈本来就在通信或 dataloader 上——先 profile 再说。
]
