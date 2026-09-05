#import "../template.typ": *

= Dispatch 与 Combine 的三种实现范式

Router 只给出"每个 token 该去哪些专家"的*决策*；真正把数据搬过去、把结果搬回来的是 *dispatch* 和 *combine*。这一章是本书工程细节最密的一章——三种范式（scatter/gather、grouped GEMM、block-sparse）之间的取舍决定了单机性能上限，也决定了分布式实现难度。

== 问题精确化

给定：

- `hidden_states: (N, H)`
- `expert_indices: (N, K)`，每行 K 个 unique expert id ∈ [0, E)
- `expert_weights: (N, K)`，每行 K 个 fp probability，和为 1
- $E$ 个 `experts[e]: (H,) → (H,)`（本质是 FFN）

目标：算出

$ y_n = sum_(k=0)^(K-1) tilde(w)_(n, k) dot "FFN"_(i_(n,k))(x_n) $

产出 `output: (N, H)`。

*核心难点*：每个 token 走 K 个专家、每个专家的 batch size $M_e$ 是运行时决定的变量。这不能直接翻译成一个静态形状的 GEMM。

== 范式 A：Scatter / Gather（本书教学实现）

最直观的实现——*外层循环遍历专家*，每次找出所有路由到它的 token，跑一次小 FFN：

```python
out = torch.zeros_like(hidden_states)          # (N, H)

for e in range(E):
    # 1. 找出命中此专家的 (token, k) 位置
    token_ids, k_ids = torch.where(expert_indices == e)
    if token_ids.numel() == 0:
        continue

    # 2. gather: 从 hidden 里取出这些 token
    x_e = hidden_states[token_ids]              # (M_e, H)

    # 3. compute: 单专家 FFN
    y_e = experts[e](x_e)                       # (M_e, H)

    # 4. weight: 乘以对应 gate 权重
    w_e = expert_weights[token_ids, k_ids].unsqueeze(-1)  # (M_e, 1)

    # 5. scatter-add: 累加回 output 对应位置
    out.index_add_(0, token_ids, y_e * w_e)
```

Shape 流：

#align(center)[
  #shape-pipeline(stages: (
    ("mask", "(N, K) bool = (idx == e)", "布尔矩阵"),
    ("where", "(M_e,) × 2  [token_ids, k_ids]", "找 True 位置"),
    ("gather", "(M_e, H)", "hidden_states[token_ids]"),
    ("expert compute", "(M_e, H)", "experts[e](·)"),
    ("weight", "(M_e, H)", "×gate_weight per token"),
    ("scatter-add", "→ out (N, H)", "index_add_"),
  ))
]

=== 图解一步 dispatch

以第 3 章 §"用一个具体例子拉一遍 tensor" 的例子 ($N=4, E=4, K=2$)，我们看 `expert_idx = 1` 的循环：

```
expert_indices = [[1, 3],       # t0 top-1=E1, top-2=E3
                  [2, 0],       # t1 top-1=E2, top-2=E0
                  [3, 1],       # t2 top-1=E3, top-2=E1
                  [0, 1]]       # t3 top-1=E0, top-2=E1

(expert_indices == 1) =
                 [[T, F],
                  [F, F],
                  [F, T],
                  [F, T]]

where(...) →
    token_ids = [0, 2, 3]       # 3 个 token 命中 E1
    k_ids     = [0, 1, 1]       # t0: E1 是 top-1；t2/t3: E1 是 top-2

gather →      hidden_states[[0,2,3]] : (3, H)
FFN →         experts[1](·)          : (3, H)
weight →      expert_weights[[0,2,3], [0,1,1]] : (3,)
                                     = [0.706, 0.250, 0.500]
scatter-add → out[[0,2,3]] += y * w
```

用 dispatch 图看更清楚：

#align(center)[
  #dispatch-diagram(
    routes: ((1, 3), (2, 0), (3, 1), (0, 1)),
    weights: ((0.706, 0.294), (0.529, 0.471),
              (0.250, 0.750), (0.500, 0.500)),
    n-experts: 4,
    title: "Dispatch: N=4 tokens → E=4 experts, K=2 (箭头粗细 = gate 权重)",
  )
]

=== 复杂度与瓶颈

单次 forward 触发的 kernel launch：

- Router: 3 次 (Linear, softmax, topk) — 可 fuse 到 1 次
- Dispatch 循环: E 次 gather + E 次 FFN (2 GEMM + 1 activation) + E 次 `index_add_`
- 合计 $O(6 E + 1)$ 次 kernel launch

对 $E = 8$：约 50 次 launch × 5 μs = 250 μs 的固定开销——在小 batch 下比 FFN 本身还贵。

*核心问题*：每个专家的 GEMM 是 $(M_e, H) times (H, I)$，$M_e$ 可能很小（数十）。TensorCore 需要 M 至少 128 才能吃满，$M_e = 30$ 的 GEMM 效率约 15%。E 个这样的小 GEMM 加起来，SM 利用率极低。

#warn[
  范式 A 只适合*教学*和*调研原型*。在 $E > 4$ 或 batch 大时，一定要转到范式 B。本书 `test_moe.py` 用范式 A，是为了让读者能读懂 —— 生产上*没有一个 MoE 框架用这个实现*。
]

=== 一个隐藏的正确性坑

`torch.index_add_` 在 GPU 上默认*非确定性*——多个 index 相同的写入可能顺序不同。但在范式 A 中，*单个 expert 循环内* `token_ids` 保证 unique（因为 `expert_indices[n, :]` 每行 K 个专家 id 互不相同 → 一个 token 在同一个 `expert_idx == e` 循环内最多出现一次）。所以本书 demo 里 `index_add_` 退化为普通 scatter，是确定的。

*但是*：跨 `expert_idx` 循环，同一个 token 被 `index_add_` K 次——这是设计需要的（top-k 加权求和）。这里 add 的顺序可能受 kernel launch 调度影响：

```python
out[t0] = w_(t0,0) * FFN_{i_(t0,0)}(x_(t0)) + w_(t0,1) * FFN_{i_(t0,1)}(x_(t0))
```

由于浮点加法非结合，*两次 add 的顺序影响低位 bit*。要严格 reproducible 训练：

```python
torch.use_deterministic_algorithms(True)
```

代价约 5-15% 慢。

== 范式 B：Permute + Grouped GEMM（生产级）

范式 A 的根本痛点：*E 次独立 GEMM*。如果能把 E 次 $(M_e, H) times (H, I)$ 打包成*一次* grouped GEMM，就能大幅提升效率。

=== 核心思路：把 tokens 按专家排序

给每个 (token, k) pair 分配一个 slot（共 $N K$ 个 slot），把 slot 按 `expert_indices` 排序打包：

```
输入:
    expert_indices = [[1, 3],
                      [2, 0],
                      [3, 1],
                      [0, 1]]           # shape (N=4, K=2)

flatten & 排序:
    slots (before) = [(t0,k0)→E1, (t0,k1)→E3,
                       (t1,k0)→E2, (t1,k1)→E0,
                       (t2,k0)→E3, (t2,k1)→E1,
                       (t3,k0)→E0, (t3,k1)→E1]

按 expert 排序 (稳定排序):
    permute_order = [3,     6,               # E0: (t1,k1), (t3,k0)
                     0,     5,     7,        # E1: (t0,k0), (t2,k1), (t3,k1)
                     2,                       # E2: (t1,k0)
                     1,     4]                # E3: (t0,k1), (t2,k0)

    group_sizes = [2, 3, 1, 2]                # 每个专家的 token 数

    total = 8 = N*K
```

=== Permute 后的 memory layout

用图看（同一 GPU 上的内存布局，无跨卡通信）：

#align(center)[
  #a2a-diagram(
    n-ranks: 1,
    before: (((1, 1), (3, 1), (2, 1), (0, 1), (3, 1), (1, 1), (0, 1), (1, 1)),),
    after:  (((0, 2), (1, 3), (2, 1), (3, 2)),),
    title: "Permute (single-GPU): 8 个 (token, k) slot 按目标 expert 分组打包",
    gap-y: 1.6,
    before-label: "before (原始 slot 顺序):",
    after-label:  "after  (按 expert 排序):",
    arrow-label:  "argsort by expert_id",
  )
]

打包后的 tensor `packed_input: (N*K, H)` 里，第 0-1 行是 E0 的 token，第 2-4 行是 E1 的 token，以此类推。

#note[
  这里*不是*分布式 all-to-all——只是本卡的内存 gather。真正的 all-to-all（跨 GPU 的 dispatch）出现在第 8 章，长得像但语义完全不同。
]

=== Grouped GEMM

Grouped GEMM 一次调用完成 E 个不同 shape 的 GEMM：

```python
# 概念签名（PyTorch 2.4+ 提供 torch._grouped_mm）
packed_out = torch._grouped_mm(
    packed_input,           # (N*K, H)
    W_stacked,              # (E, H, I) — 所有专家权重堆叠
    group_sizes,            # (E,) int — 每个专家的 token 数
)
# packed_out: (N*K, I)
```

Kernel 内部：CUTLASS 的 `GroupedGemm` template 根据 `group_sizes` 在 CTA 层生成 E 个 sub-problem tile，动态分配 SM 资源。相当于一次 launch 完成 E 个 GEMM。

*收益*：kernel launch 从 $E → 1$；tile scheduler 全局最优（能把小 group 和大 group 混合调度）；SM 利用率从 15% → 60-80%。

=== 完整范式 B 流程

```
1. router → expert_indices (N,K), expert_weights (N,K)
2. permute:
     - 展平 (N,K) → (N*K,) 得 flat_expert_ids
     - argsort 得 permute_idx
     - packed_input = hidden_states.repeat_interleave(K)[permute_idx]
     - 或用 fused kernel 直接 gather 到 packed
3. grouped_gemm_1: packed_input @ W_up  → packed_hidden (N*K, I)
4. activation: (SwiGLU / ReLU / GELU) elementwise
5. grouped_gemm_2: packed_hidden @ W_down → packed_out (N*K, H)
6. weight & unpermute:
     - packed_out *= expert_weights[permute_idx]  (broadcast)
     - out[permute_idx.argsort()] = packed_out  (inverse permute)
7. sum over K:
     - out.view(N, K, H).sum(dim=1) → (N, H)
```

Shape 全链路：

#align(center)[
  #shape-pipeline(stages: (
    ("hidden", "(N, H)", ""),
    ("router", "expert_indices (N, K), expert_weights (N, K)", ""),
    ("permute", "packed_input (N*K, H) + group_sizes (E,)", "按 expert 排序"),
    ("grouped GEMM 1", "packed_hidden (N*K, I)", "1 kernel"),
    ("activation", "(N*K, I)", "elementwise"),
    ("grouped GEMM 2", "packed_out (N*K, H)", "1 kernel"),
    ("weight + unpermute", "(N, K, H)", "inverse perm + × w"),
    ("reduce over K", "(N, H)", "sum(dim=1)"),
  ))
]

=== 范式 B 的代价

不是免费午餐：

*(1) Permute / unpermute 是新增开销*

Permute 本身要一次全内存 gather：$N K H$ elements 的读写。当 $N=10^5, K=2, H=4096, "bf16"$，约 3 GB traffic。fusion 到 grouped-gemm 的 prologue 里可以摊平大半，但独立版本会显著。

*(2) $M_e = 0$ 的空专家要正确处理*

grouped GEMM kernel 必须允许 `group_sizes[e] = 0`——传入的 offset 不变，跳过。CUTLASS `GroupedGemm` 支持，但早期版本有 bug。

*(3) capacity 与 drop token*

有些实现给每个 expert 设 capacity $C = "ceil"(N K / E times c_"factor")$，超出的 token 直接*丢弃*。丢弃 token 通过 residual 分支跳过 FFN。范式 B 里 drop 天然对齐：packed 到 group $e$ 的 slot 超过 $C$ 就截断。

- 优点：packed shape 完全静态 (`(E, C, H)`)、GEMM tuning 更好。
- 缺点：训练早期 collapse 时大量 drop → 损失 & 信息丢失。

*(4) 需要自定义 kernel*

不是 vanilla PyTorch 一行搞定——需要 `torch._grouped_mm` 或框架自带的 `MoELayer`。这是 Tutel / Megatron-MoE / DeepSpeed-MoE 的核心 IP。

#insight[
  范式 B 的性能取决于*permute+unpermute 是否 fuse 进 grouped GEMM*。裸调 `torch._grouped_mm` 只 fuse 了 GEMM，permute 仍是独立 kernel；生产实现（Tutel `fast_dispatch`、Megatron `moe_permute`）会把 permute 融进 GEMM 的 prologue，把 3 次 (N*K, H) 的 HBM 往返压到 1 次。
]

== 范式 C：Block-Sparse GEMM（Megablocks）

Megablocks (Gale et al. 2022) 的洞察：*所有 expert 权重可以拼成一个大 block-diagonal 矩阵*，然后用一次*块稀疏*GEMM 完成计算。

=== Block-sparse 视角

先按范式 B 那样把 tokens 排好——`packed_input: (N*K, H)`。每一行对应一个 (token, k) slot，属于某个 expert $e$。Megablocks 的洞察是：*把所有 expert 权重视为一个大的 block-diagonal 矩阵*，然后一次 block-sparse GEMM 就完成全部计算：

$ underbrace(Y, (N K"," E I)) = underbrace(hat(X), (N K"," E H)) times underbrace(W_"big", (E H"," E I)) $

其中 $W_"big"$ 是把 $E$ 个 $W_"up" in RR^(H times I)$ 沿对角线拼起来，只有对角上的 $E$ 个 block 非零：

```
W_big =  ┌────────────────────────────┐
         │ W_up^{E0}                  │
         │            W_up^{E1}       │
         │                       ...  │
         │              W_up^{E_{E-1}}│
         └────────────────────────────┘  (E-1)/E 稀疏
```

$hat(X)$ 是 packed_input 的"扩展"：第 $i$ 行只在 slot 所属 expert 的列区间放原来的 hidden vector、其余全零。这样 $hat(X) times W_"big"$ 每行只在对应 expert 的输出区间产生非零结果，行为等价于范式 B 的 grouped GEMM——但 kernel 上真正跑的是 block-sparse SpMM（CUTLASS block-sparse、Sputnik、Megablocks 自带的 `stk`）。

生产实现里 $hat(X)$ 不会真的 materialize——它是逻辑视图，SpMM kernel 直接从 packed_input + slot→expert 的映射 gather。整个过程等价于对 packed layout 加一次 block-sparse 元数据，*不需要 padding、不需要 drop*。

=== 关键优势

*(1) 完全无 padding、无 drop*

不需要 capacity，$M_e$ 可以是任意值——block-sparse GEMM 天然处理变长。

*(2) 与 dense GEMM 性能可比*

只要 block 大小 $≥ 128$，CUTLASS 的 block-sparse kernel 效率接近 dense。Megablocks 论文报告 in-block density = 1、跨 block sparse pattern，dense-equivalent throughput 达到 90%+。

*(3) 训练早期 collapse 也不 drop*

对训练动力学友好——router 可以慢慢学，不会因为 drop 导致信号丢失。

=== 关键代价

*(1) Kernel 复杂*

需要 block-sparse metadata (block row/col indices)，permute 逻辑更复杂。这是 Megablocks 的核心 IP。

*(2) 分布式集成难度高*

Block-sparse metadata 在 all-to-all 之后需要重新计算——这也是为什么 Megablocks 早期只在单机上。后来 (2024) 有 distributed megablocks 集成到 Databricks DBRX。

== 三种范式对比

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, center, center),
    [*维度*], [*A. scatter*], [*B. grouped GEMM*], [*C. block-sparse*],  [*说明*],
    [Kernel launch 数], [$O(E)$], [$O(1)$], [$O(1)$], [B/C 大幅减少],
    [SM 利用率], [low ($< 30%$)], [high ($60%+$)], [very high ($80%+$)], [取决于 tile 大小],
    [支持 drop token], [手动实现], [是 (常用)], [不需要], [C 是最大优势],
    [Permute overhead], [无 (循环内 gather)], [显式 permute], [显式 permute], [B/C 需要 permute fuse],
    [空专家 $M_e = 0$], [continue], [需支持], [自然处理], [B 早期 kernel 有 bug],
    [代码复杂度], [~50 行], [~500 行 + kernel], [~2000 行 + kernel], [C 最难],
    [分布式集成], [不适用], [成熟 (Megatron, Tutel)], [新 (DBRX, 2024)], [B 是主流],
  ),
  kind: table,
)

== 选择建议

不同场景下选哪个范式，一个 rough guide：

- *学习/调研*：范式 A。写起来快、跑起来对、能读懂。
- *单机 $E ≤ 8$*：范式 A 或 B 均可；B 收益 $2-3 times$，但对 batch 小的场景 (< 512 tokens) 收益递减。
- *单机 $E > 8$ 或大 batch*：范式 B 是标准。
- *需要 drop-free 训练*：范式 C。
- *多机 EP*：范式 B + all-to-all；范式 C 集成难度高，选型看框架支持。

#note[
  个人建议：即使目标是生产系统，*先写一遍范式 A 的正确版*、跑通 unit test，*然后*把 FFN 部分替换成范式 B。范式 A 的输出是 "reference"，用来对拍 B/C 的正确性——特别是 drop token、capacity、边界条件的处理。
]

== 面试考点

#interview[
  *Q1*: 范式 A (scatter) 主要的性能问题是什么？

  A: (1) $E$ 次 kernel launch overhead；(2) 每个专家 GEMM 的 M 维小 ($M_e$ 数十)，TensorCore 无法吃满；(3) $M_e$ 随 batch 抖动，无法预测性能；(4) `index_add_` 的 memory access pattern 不连续。总体表现：SM 利用率 $< 30%$。
]

#interview[
  *Q2*: Grouped GEMM 为什么比 for-loop 快？

  A: 一次 kernel 内部做全局 tile 调度：把 E 个不同大小的 sub-problem 打散成 tile，动态负载均衡；SM 一直有 work，不需要等 kernel launch。等价于把 E 次串行 launch 换成一次 launch + 内部并行。
]

#interview[
  *Q3*: `permute_idx` 和 `permute_idx.argsort()` 的关系？

  A: 逆排列。若 `permute_idx = [3, 6, 0, 5, 7, 2, 1, 4]`，则 `argsort(permute_idx) = [2, 6, 5, 0, 7, 3, 1, 4]`——这是 unpermute 用的 index。生产实现里 permute + unpermute 的 index 只算一次，缓存到 workspace 复用。
]

#interview[
  *Q4*: Capacity factor 应该设多少？

  A: 训练 $c_"factor" = 1.25$ 是 GShard/Switch 经验值；推理常用 $c_"factor" ≥ 2$ 或直接 no-drop (Megablocks 路径)。理论下界：如果 router 输出接近均匀，$M_e approx N K / E$，$c_"factor" = 1$ 就够；训练早期 collapse 时 $M_e$ 可以是 $10 N K / E$，$c_"factor" = 2$ 都不够——所以 drop 是必然事件，只是频率问题。
]

#interview[
  *Q5*: Megablocks vs grouped GEMM 的核心差异？

  A: Grouped GEMM 需要 permute 到 packed layout，capacity + drop 才能保证 shape 静态；Megablocks 用 block-sparse GEMM 天然处理变长，*不 drop*，代价是 kernel 复杂度和分布式集成难。Megablocks 训练动力学更好（无 drop），但工程负担重。
]

#interview[
  *Q6*: 如果不用 `torch.where`，还能怎么实现 range B 的 permute？

  A: (1) `expert_ids.flatten().argsort()` 直接给 permute index；(2) counting sort 因为 $E$ 小 (`bincount + cumsum` 得 group offsets)；(3) fused kernel 直接从 `expert_indices` 生成 sorted flat index + offsets 一次搞定。生产实现用 (3)，因为 argsort 有隐式排序开销。
]
