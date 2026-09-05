#import "../template.typ": *

= TorchInductor：codegen 与 fusion

前两层只是把 Python 变成了图，一个 kernel 都没省。真正产生加速的是 Inductor：它把图 lowering 成一个 loop-level 的 IR，决定哪些 op 可以塞进同一个 kernel，然后生成代码——GPU 上生成 Triton，CPU 上生成 C++/OpenMP。

面试里"`torch.compile` 为什么快"的标准答案只有一句：*大部分逐点和归约操作是 memory-bound 的，融合成一个 kernel 就省掉了中间张量往返 HBM 的开销。* 这一章把这句话拆开算清楚，并贴上实测的 Triton 代码。

== 流水线

#figure(
  align(center, shape-pipeline(stages: (
    ("FX Graph", "post-grad ATen/prim ops", "AOTAutograd 交来的图"),
    ("Inductor IR", "loop-level: 循环 + 索引表达式", "lowering"),
    ("Scheduler", "SchedulerNode 图 + fusion 决策", "vertical / horizontal fusion"),
    ("Triton / C++", "kernel 源码 + wrapper", "codegen"),
    ("cubin / .so", "编译产物 + 缓存", "autotune + FX graph cache"),
  ))),
  caption: [Inductor 内部五步。关键在中间两步：lowering 到 loop-level IR 让 fusion 变成"合并循环"这个可判定的问题。],
)

lowering 这一步的核心是把每个 op 表示成"输出的每个元素怎么由输入索引算出来"。`add` 变成 `out[i] = a[i] + b[i]`，`gelu` 变成一串标量运算，`layer_norm` 变成一个带归约的循环。表示成这个形式之后，*fusion 就等价于"能不能把两个循环合并成一个"*——一个有明确判定条件的问题，而不是靠一堆手写的 pattern match 规则。

== 为什么 fusion 是主要收益来源

A100-SXM4-80GB 的两个关键数字：HBM 带宽 *2.04 TB/s*，bf16 tensor core 峰值 *312 TFLOPS*。两者的比值给出这块卡的 ridge point（算术强度拐点）：

#formula[$ I_"ridge" = (312 times 10^12) / (2.04 times 10^12) approx 153 " FLOP/byte" $]

意思是：一个 kernel 每从 HBM 读写 1 字节，必须做到 153 次浮点运算，才能把 tensor core 喂饱。低于这个强度的 kernel 一定是 memory-bound——加速的唯一途径是*少读写字节*。

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, right, right, left),
  [*操作*], [*FLOP*], [*bytes (bf16)*], [*算术强度*],
  [`tanh(x)`（单独一个 kernel）], [$N$], [$4 N$], [约 0.25，比 ridge 低 600 倍],
  [`x + y`], [$N$], [$6 N$], [约 0.17],
  [`layer_norm`（一行 1024）], [约 $5 N$], [$4 N$], [约 1.25],
  [`mm` $4096^3$], [$2 dot 4096^3$], [$3 dot 4096^2 dot 2$], [约 1365，远高于 ridge],
)

#insight[
  elementwise、activation、norm、dropout 全部低于 ridge point 两三个数量级，它们的耗时就等于"读写了多少字节 ÷ 带宽"。N 个这样的 op 串起来，eager 下要往返 HBM 大约 2N 次；融合成一个 kernel 后只有一次读、一次写。省下来的时间就是全部收益。GEMM 完全在另一侧，它的优化路径不是 fusion（见后文）。
]

=== 实测

A100-SXM4-80GB，bf16，`x` 是 `(8192, 8192)`（每次读或写 134.2 MB），函数是

```python
def f(x):
    return torch.sigmoid(torch.tanh(x) * 2.0 + 1.0) * x
```

用 `torch.profiler` 数 kernel、用 CUDA event 计时（`no_grad`，50 次 warmup，200 次取平均）：

#figure(
  align(center, hbar-chart(
    (("eager (5 kernel)", 0.868), ("torch.compile (1 kernel)", 0.172)),
    unit: "ms", width: 7,
  )),
  caption: [A100-SXM4-80GB / bf16 / `(8192, 8192)` 实测。5.06× 的来源全部是省下的 HBM 往返。],
)

对得上账：eager 的 5 个 kernel 一共约 11 次 134.2 MB 的 HBM 往返（`tanh` 读写各一次、`mul` 一次、`add` 一次、`sigmoid` 一次，最后 `* x` 要读两个输入写一个输出），共约 1476 MB，除以 0.868 ms 得约 1.70 TB/s——已经接近这块卡的带宽上限，说明 eager 的每个 kernel 本身写得没问题。融合后只剩 2 次往返共 268 MB，除以 0.172 ms 得约 1.56 TB/s。*带宽利用率几乎没变，变的是要搬的字节数：11 次 → 2 次。*

#note[
  这也解释了为什么 `torch.compile` 在小张量上经常没有收益甚至变慢：张量小到 kernel 时间被 launch 开销和 kernel 内在效率主导时，省 HBM 往返省不出什么，而 Inductor 生成的 Triton kernel 未必比 ATen 手调的 kernel 更高效。实测 `gelu(LayerNorm(x)) * 1.5`、张量 `(8, 512, 1024)` fp32（每次往返 16.7 MB），eager 3 个 kernel 0.0657 ms、compile 1 个 kernel 0.0632 ms，只有 1.04×。小 shape 的正确解法是 `mode="reduce-overhead"`（第 15 章）。
]

== fusion 的种类与阻碍

*vertical fusion*（producer-consumer）：`a = f(x)` 后紧跟 `b = g(a)`，把 `g` 的计算接在 `f` 的寄存器里做完，`a` 根本不写回 HBM。这是绝大多数收益的来源。

*horizontal fusion*：两个互不依赖但迭代空间相同的 op（比如同 shape 的两个 elementwise），塞进同一个 kernel 分摊 launch 开销和索引计算。收益小得多。

什么阻止 fusion：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*阻碍*], [*为什么*],
  [归约后接不同迭代空间的 op],
    [归约把 `(B, S, H)` 变成 `(B, S, 1)`，后面再作用回 `(B, S, H)` 需要跨 block 广播；只有当归约维能整块放进一个 program 时（persistent reduction）才融得动],
  [中间结果有图外的使用者],
    [该张量是图的输出、或被 `.item()` 之类读走，必须真的物化到 HBM],
  [内存布局冲突],
    [两个 op 的最优 tiling/循环序不同，硬融进去会导致非合并访存，Inductor 的启发式会拒绝],
  [graph break],
    [段边界上的张量必须落盘。第 12 章的账在这里结算],
  [寄存器压力 / spill],
    [融合链太长会超出寄存器预算，spill 到 local memory 反而更慢；Inductor 有 `spill_threshold` 之类的保护],
)

== 看真实生成的 Triton kernel

把 `LayerNorm(1024) + GELU` 作用在 `(8, 512, 1024)` fp32 上，`TORCH_LOGS="output_code"` 实跑，Inductor 生成的是*一个* Triton kernel（eager 是两个：`vectorized_layer_norm_kernel` 加一个 GELU 的 `vectorized_elementwise_kernel`）：

```python
@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32',
                 'in_ptr1': '*fp32', 'in_ptr2': '*fp32',
                 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, ...},
    inductor_meta={'kernel_name': 'triton_per_fused_gelu_native_layer_norm_0',
                   'num_load': 3, 'num_store': 1, 'num_reduction': 4, ...},
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1,
                                              in_ptr2, xnumel, r0_numel,
                                              XBLOCK : tl.constexpr):
    xnumel = 4096                      # 8 * 512 行
    r0_numel = 1024                    # 归约维（normalized_shape）
    R0_BLOCK: tl.constexpr = 1024      # 整行放进一个 program → persistent reduction
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex
    tmp0  = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)                  # x
    tmp21 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')  # weight
    tmp23 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')  # bias
    tmp1  = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3  = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5  = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp8  = (tmp5 / 1024.0)            # mean
    tmp9  = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tmp16 = (tmp13 / 1024.0)           # var
    tmp19 = libdevice.rsqrt(tmp16 + 1e-05)
    tmp20 = (tmp0 - tmp8) * tmp19      # 归一化
    tmp24 = tmp20 * tmp21 + tmp23      # affine
    tmp26 = tmp24 * 0.5                # ↓ gelu 的 decomposition
    tmp28 = tmp24 * 0.7071067811865476
    tmp29 = libdevice.erf(tmp28)
    tmp32 = tmp26 * (tmp29 + 1.0)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp32, None)
```

三件事值得注意：

+ *一次 `tl.load`，一次 `tl.store`。* LayerNorm 的两趟归约、affine、GELU 的四步全在寄存器里完成。这就是前面那笔账的具体形态。
+ *GELU 被分解了。* `0.5`、`0.7071067811865476`、`libdevice.erf` 就是第 13 章讲的 decomposition 的产物——Inductor 从来没见过 `aten.gelu`。
+ *kernel 名字自带来源。* `triton_per_fused_gelu_native_layer_norm_0`：`per` 表示 persistent reduction（`poi` 是 pointwise，`red` 是普通 reduction，`tem` 是 template），后面是被融进来的 ATen op 名。profiler timeline 里看到 kernel 名就知道融了什么，这是排查 fusion 的第一手信息。

看代码的两个入口：

```bash
TORCH_LOGS="output_code"  python train.py 2>&1 | tee /tmp/oc.log   # 打到 stderr
TORCH_COMPILE_DEBUG=1     python train.py                          # 落盘，含各阶段 IR
TORCH_LOGS="ir_pre_fusion,ir_post_fusion,fusion,schedule" python train.py  # 看融合决策本身
```

== matmul 不靠 fusion

GEMM 的算术强度远在 ridge point 之上，它需要的是 tiling、双缓冲、tensor core 指令编排——这些 cuBLAS/CUTLASS 已经做到极致了。Inductor 的策略是*选*，而不是自己生成。

默认 `mode` 下，`Linear(2048, 2048) + ReLU`（bf16）实跑生成的 wrapper 是：

```python
buf0 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
# Original ATen: [aten.t, aten.addmm]
extern_kernels.mm(arg2_1,
    reinterpret_tensor(arg0_1, (2048, 2048), (1, 2048), 0), out=buf0)
buf1 = buf0; del buf0  # reuse
triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 4194304, stream=stream0)
```

读法：GEMM 交给 `extern_kernels.mm`（就是 cuBLAS）；权重的转置用 `reinterpret_tensor` 换 stride 完成，*没有拷贝*；bias 加法和 ReLU 融进一个 Triton pointwise kernel，并且直接写回 `buf0` 复用显存（`buf1 = buf0`）。

`mode="max-autotune"` 会额外把 Triton GEMM template 拉进候选池，benchmark 之后选最快的。同一个例子实跑，autotune 日志和选中的 kernel：

```text
SingleProcess AUTOTUNE benchmarking takes 0.7233 seconds and 0.0031 seconds
    precompiling for 20 choices
```

```python
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_0', ...,
  'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32',
                  'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8,
                  'ALLOW_TF32': False}}
```

`tem` 前缀说明这次选中了 Triton template（不是 cuBLAS），并且 autotune 定下了 tile 尺寸。

*epilogue fusion* 就发生在这个 template 上：Triton template 的尾部留了一个可以插入 pointwise 计算的钩子，于是 bias、activation、scale 可以在 GEMM 的累加器还在寄存器里的时候就算掉，省掉一次 `(M, N)` 的往返。

#warn[
  epilogue fusion 会不会真的发生取决于 scheduler 的 benchmark 结果，不是必然。上面那次 `max-autotune-no-cudagraphs` 的实跑里，Inductor 选了 Triton template 但把 bias + ReLU 留在了独立的 `triton_poi_fused_addmm_relu_1` 里。判断方法只有一个：看 `output_code` 里 kernel 的数量和名字，不要靠推断。
]

=== max-autotune 的代价

同一个 `Linear(4096,4096) + GELU + Linear(4096,4096)`，bf16，输入 `(4096, 4096)`，A100 上用干净的 `TORCHINDUCTOR_CACHE_DIR` 实测：

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, right, right),
  [*配置*], [*稳态 step*], [*冷启动编译*],
  [eager], [1.314 ms], [—],
  [`mode` 默认], [1.331 ms], [4.8 s],
  [`mode="max-autotune"`], [1.438 ms], [13.6 s],
)

#insight[
  这个例子里 `max-autotune` *更慢*，编译时间还贵了 3 倍。原因是这个 shape 上 cuBLAS 已经接近最优（1.314 ms 对应约 209 TFLOPS，A100 bf16 峰值 312），Triton template 赢不了，而 autotune 的候选池里挑出来的那个还带了额外的 epilogue 开销。结论是：`max-autotune` 是要*实测*才能用的选项，尤其在标准 shape 的 GEMM 上没有免费午餐。它真正有价值的场景是不规则 shape、小 batch 的 GEMM、以及能吃到 epilogue fusion 的地方。
]

== 编译缓存

Inductor 的产物落在 `TORCHINDUCTOR_CACHE_DIR`，默认 `/tmp/torchinductor_$USER`。实测目录结构：

```text
/tmp/torchinductor_duo.an/
├── fxgraph/       # FX graph cache：图 + 配置的 hash → 编译结果
├── aotautograd/   # AOTAutograd 阶段的缓存
├── triton/        # 编译好的 Triton kernel（cubin）
└── <hash>/        # 生成的 Python wrapper 与 kernel 源码
```

#table(
  columns: (1fr, 1.3fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*缓存*], [*键与作用*],
  [FX graph cache #linebreak() 配置项 `fx_graph_cache`，默认开],
    [图结构 + 输入元信息 + Inductor 配置 + PyTorch 版本 + GPU 架构 的 hash → 整个 Inductor 编译结果],
  [Triton / autotune cache],
    [kernel 源码 hash → cubin；autotune 选出的最优 config 也缓存，避免重复 benchmark],
  [remote cache #linebreak() 配置项 `fx_graph_remote_cache`],
    [同上但存在共享后端，供集群里多个 job 复用；默认未启用],
)

为什么第一次慢：Dynamo 符号执行 + AOTAutograd trace + Inductor lowering/fusion + Triton 编译（要调 `ptxas`）+ 可能的 autotune benchmark，全部串在第一次调用里。

缓存命中的条件很严：图结构、输入的 shape/dtype/device、Inductor 与 Dynamo 的配置、PyTorch 版本、GPU compute capability，任何一项变了都不命中。所以*改一行模型代码、换一张卡型号、升一次 torch 版本都会全量重编*。

#warn[
  缓存只省掉 Inductor 及以后的部分。同一个例子实测：冷缓存 4.8 s，热缓存（新进程，同 `TORCHINDUCTOR_CACHE_DIR`）3.8 s——只快了 1 s。Dynamo 的符号执行和 AOTAutograd 的 trace 每个进程都要重做一遍，不进 FX graph cache。所以"加了缓存冷启动就不慢了"是错的；真正想省启动时间要靠 `torch.export` + AOTInductor 把编译彻底移到离线（第 16 章）。
]

#warn[
  `/tmp` 在很多集群上是 tmpfs 或有清理策略，容器重启就没了；多个 job 并发写同一个 cache dir 也见过踩坏的案例。生产上把 `TORCHINDUCTOR_CACHE_DIR` 指到持久盘上的一个按"torch 版本 + GPU 型号"分开的路径。
]

== Inductor 和手写 Triton 的边界

Inductor 生成的 Triton 已经能覆盖绝大多数 elementwise/reduction 的融合。什么时候还要自己写：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*场景*], [*为什么 Inductor 不够*],
  [FlashAttention 这类算法级重写],
    [它不是"把几个循环合并"，而是改变了算法（online softmax + 分块，把 $O(S^2)$ 的中间矩阵彻底消掉）。Inductor 只做保语义的循环变换，想不出这种改写。用 `F.scaled_dot_product_attention`，它会派发到 FlashAttention 后端],
  [跨归约的复杂融合],
    [归约 → 广播 → 再归约这种链，Inductor 的启发式常常拒绝融合],
  [特殊数据布局 / 量化],
    [4-bit 打包权重、block-scaled fp8、稀疏格式，索引逻辑无法用 Inductor 的 IR 表达],
  [极致占用率调优],
    [手动控制 `num_warps`、`num_stages`、swizzle、异步拷贝流水],
)

手写 Triton 的写法、tl 语义和 autotune 见仓库里的 `src/triton/`；CUDA 层面的 tiling 与 tensor core 见 `books/cuda/`。融进 `torch.compile` 的正确姿势是把手写 kernel 注册成 custom op（`torch.library`），这样 Dynamo 不会 break、AOTAutograd 也能拿到它的反向。

== 面试考点

#interview[
  *Q1*：`torch.compile` 为什么快？

  A：主要收益是 kernel fusion。elementwise、activation、norm、dropout 的算术强度在 0.2 到几 FLOP/byte，而 A100 的 ridge point 是 312 TFLOPS ÷ 2.04 TB/s ≈ 153 FLOP/byte，低了两三个数量级，全是 memory-bound。eager 下 N 个这样的 op 要往返 HBM 约 2N 次，融成一个 kernel 只剩一次读一次写。次要收益是省 Python 和 kernel launch 开销（`mode="reduce-overhead"` 上 CUDA graph 才是这部分的主力）。
]

#interview[
  *Q2*：给个具体数字，fusion 能省多少？

  A：A100-SXM4-80GB、bf16、`(8192, 8192)`，`sigmoid(tanh(x)*2+1)*x` 实测：eager 5 个 kernel 0.868 ms，compile 1 个 kernel 0.172 ms，5.06×。对账：eager 约 11 次 134 MB 的 HBM 往返，共 1476 MB，除以耗时约 1.70 TB/s（已接近带宽上限）；融合后 2 次往返共 268 MB，约 1.56 TB/s。带宽利用率没变，字节数少了 5.5 倍。
]

#interview[
  *Q3*：Inductor 是怎么决定能不能融合的？

  A：lowering 时把每个 op 表示成 loop-level IR（"输出每个元素怎么由输入索引算出来"），fusion 就变成"两个循环能不能合并"。最常见是 vertical fusion（producer-consumer，中间结果留在寄存器）；horizontal fusion（同迭代空间的独立 op）收益小。阻碍：归约改变了迭代空间、中间结果是图的输出必须物化、tiling/布局冲突导致非合并访存、graph break 切断了图、融合链太长导致寄存器 spill。
]

#interview[
  *Q4*：matmul 会被 fusion 优化吗？

  A：不会，GEMM 的算术强度远高于 ridge point，是 compute-bound，靠的是 tiling 和 tensor core 编排。Inductor 默认直接调 `extern_kernels.mm`（cuBLAS）；`mode="max-autotune"` 会把 Triton GEMM template 加进候选池 benchmark 后选最快的。GEMM 上唯一的 fusion 是 epilogue fusion——把 bias/activation/scale 接在累加器还在寄存器时算掉，省一次 `(M, N)` 的往返，但只有走 Triton template 时才可能。
]

#interview[
  *Q5*：`mode="max-autotune"` 一定更快吗？

  A：不一定。实测 `Linear(4096,4096)+GELU+Linear` bf16 在 A100 上：eager 1.314 ms，默认 mode 1.331 ms，max-autotune 1.438 ms，而冷启动编译从 4.8 s 涨到 13.6 s。标准 shape 的 GEMM 上 cuBLAS 已经接近最优（1.314 ms 约 209 TFLOPS，峰值 312），Triton template 赢不了。max-autotune 有价值的场景是不规则 shape、小 batch GEMM、以及能吃到 epilogue fusion 的地方——必须实测。
]

#interview[
  *Q6*：怎么看 Inductor 生成了什么？

  A：`TORCH_LOGS="output_code"` 打出完整的 Python wrapper + Triton kernel 源码；`TORCH_COMPILE_DEBUG=1` 落盘且包含各阶段 IR；`TORCH_LOGS="ir_pre_fusion,ir_post_fusion,fusion,schedule"` 看融合决策本身。看不完全部代码时先看 kernel 名字：`triton_poi_*` 是 pointwise，`triton_per_*` 是 persistent reduction，`triton_red_*` 是普通 reduction，`triton_tem_*` 是 GEMM template，名字后半段直接列出被融进来的 ATen op。
]

#interview[
  *Q7*：为什么第一次调用很慢？缓存能解决吗？

  A：第一次要跑完 Dynamo 符号执行、AOTAutograd trace、Inductor lowering/fusion、Triton 编译（调 `ptxas`），可能还有 autotune benchmark。FX graph cache（默认开，落在 `/tmp/torchinductor_$USER`）只缓存 Inductor 及以后，键是"图结构 + 输入元信息 + 配置 + torch 版本 + GPU 架构"的 hash。实测冷 4.8 s、热 3.8 s——Dynamo 和 AOTAutograd 每个进程都要重做。要真正消掉启动开销得用 `torch.export` + AOTInductor 把编译移到离线。
]

#interview[
  *Q8*：什么时候 `torch.compile` 不够，得自己写 Triton？

  A：Inductor 只做保语义的循环变换，想不出算法级的改写。FlashAttention 是典型：它靠 online softmax + 分块消掉 $O(S^2)$ 的中间矩阵，这是换算法不是换循环（实践中直接用 `F.scaled_dot_product_attention` 派发到 FlashAttention 后端）。其他要手写的场景：跨归约的复杂融合链、4-bit 打包权重或 block-scaled fp8 这类特殊布局、需要手动控制 `num_warps`/`num_stages`/异步流水的极致调优。手写 kernel 要用 `torch.library` 注册成 custom op 才能干净地嵌进 `torch.compile`。
]
