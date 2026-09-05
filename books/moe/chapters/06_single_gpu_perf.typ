#import "../template.typ": *

= 单机训练性能优化

前面五章讲了 MoE *能算对*，这一章讲*怎么算得快*。目标是单机场景下（无跨机通信），把 MoE 层从"naive 慢 3-5×"打到"接近 dense FFN 的效率"。

分布式（EP、all-to-all）留到第 8 章——那需要先理解本章的 grouped GEMM 基础。

== 瓶颈画像：先看开销分布

在 A100/H100 上跑一个中等 MoE 层（$B S = 8192, H = 4096, I = 4 H, E = 8, K = 2$），naive PyTorch 实现（范式 A，for-loop）的 forward wall-clock 时间*大致*按下面比例分布（数量级估算，实测值随 batch size 与硬件浮动）：

#align(center)[
  #time-share-bar(
    (
      ("Router (Linear + softmax + topk)",  8.0),
      ("Dispatch (E 次 gather)",           17.0),
      ("Expert FFN (E 次 2-GEMM + act)",   56.0),
      ("Combine (E 次 index_add)",         14.0),
      ("其他 (view / cast / launch)",       5.0),
    ),
    width: 8.5,
    label-w: 5.4,
  )
]

关键观察：

*(1) Expert FFN 占大头（约 56%）*，但*效率极差*——每个 sub-GEMM 的 M 维只有 $tilde.op B S / E = 8192 / 8 = 1024$（假设完美均衡）或更小（实际 skewed）。TensorCore 需要 $M >= 128$ 才吃满，$M_e = 30$ 量级时 GEMM 效率跌到 15-30%。

*(2) Dispatch + Combine 加起来约 31%*——这在 dense FFN 里完全不存在，是 MoE 的额外税。

*(3) Router 只占 8%*，优化收益有限。

*一个 dense FFN 的对比*：同规模 dense FFN 的 GEMM 效率 80-90%——MoE 有巨大的头顶空间。

== 优化 ladder

#ladder(
  ("v0: naive for-loop",             "第 5 章 test_moe.py",                     "基线"),
  ("v1: grouped GEMM",               "permute + torch._grouped_mm",            "1.5-2×"),
  ("v2: v1 + fused router",          "router 融合成 1 kernel",                  "+3-5%"),
  ("v3: v2 + fused permute",         "permute 融进 grouped GEMM prologue",       "+10-15%"),
  ("v4: v3 + fused activation",      "SwiGLU 融成 1 kernel",                    "+5%"),
  ("v5: v4 + block-sparse",          "无 drop / 无 padding",                    "+5-10% (drop 时更多)"),
  ("v6: v5 + torch.compile",         "auto tuning + kernel fuse",              "+5-15% (case-dependent)"),
)

从 v0 到 v6，端到端 MoE 层加速 3-4×。生产系统 (Megatron-MoE, Tutel) 一般在 v3/v4 附近。本章逐个讲每一层的原理与代价。

== v1: Grouped GEMM

第 4 章"范式 B"已经讲过 permute + grouped GEMM 的原理。这里聚焦*为什么快*。

=== 单次 grouped GEMM 的收益公式

假设 $E$ 个专家，每个 $M_e$ 个 token，做 $(M_e, H) times (H, I)$：

*Naive (v0)*: E 次独立 GEMM

- 每次 kernel launch: $tilde.op 5 mu s$
- Tile scheduling: 每 GEMM 独立
- SM 利用率: 若 $M_e < 256$，仅约 30%

*Grouped (v1)*: 1 次 grouped GEMM

- 1 次 launch
- Kernel 内部把 E 个 sub-problem 拆成 tile 全局调度
- SM 利用率: $60-80%$

*总节省*: launch overhead $(E-1) times 5 mu s = 35 mu s$ (for E=8) + GEMM 加速约 $1.5 times$。

对 $E=8$，$M_e approx 250$，端到端 MoE forward 从 $tilde.op 600 mu s$ 降到 $tilde.op 300 mu s$。

=== 代码骨架

```python
def moe_grouped_gemm(hidden, W_up, W_down, expert_indices, expert_weights):
    N, H = hidden.shape
    E, _, I = W_up.shape  # W_up: (E, H, I), W_down: (E, I, H)
    K = expert_indices.shape[1]

    # 1. 展平 (N, K) → (N*K,), 记录每个 slot 的目标 expert
    flat_expert = expert_indices.view(-1)                    # (N*K,)
    flat_weight = expert_weights.view(-1)                    # (N*K,)
    # 每个 slot 对应的 token id
    flat_token  = torch.arange(N, device=hidden.device).repeat_interleave(K)

    # 2. 按 expert 排序 → permute
    permute_idx = flat_expert.argsort(stable=True)           # (N*K,)
    perm_tokens = flat_token[permute_idx]                    # (N*K,) 排序后的 token id
    packed_input = hidden[perm_tokens]                        # (N*K, H)
    packed_weight = flat_weight[permute_idx]                  # (N*K,)

    # 3. 每个 expert 的 token 数
    group_sizes = torch.bincount(flat_expert, minlength=E)   # (E,)

    # 4. Grouped GEMM (fc1)
    packed_hidden = torch._grouped_mm(packed_input, W_up, group_sizes)
    # (N*K, I)

    # 5. Activation
    packed_hidden = F.relu(packed_hidden)  # 或 SwiGLU

    # 6. Grouped GEMM (fc2)
    packed_out = torch._grouped_mm(packed_hidden, W_down, group_sizes)
    # (N*K, H)

    # 7. Weight
    packed_out = packed_out * packed_weight.unsqueeze(-1)

    # 8. Unpermute: 逆排列 + reduce over K
    inverse_idx = permute_idx.argsort()  # (N*K,)
    unpacked = packed_out[inverse_idx]    # (N*K, H) — 恢复到 (token, k) 排列
    out = unpacked.view(N, K, H).sum(dim=1)  # (N, H)

    return out
```

=== 需要注意的点

*(1) `stable=True` 排序*

必需——保证同一 expert 内部 token 的相对顺序稳定，方便 unpermute。unstable 排序会让 debug 变噩梦。

*(2) `torch._grouped_mm` 的 API 变化*

PyTorch 2.4+ 提供，但 API 在演进。生产实现用 CUTLASS/Triton 的 grouped GEMM。开源可参考 `grouped_gemm` (NVIDIA) 或 SGLang/vLLM 里的实现。

*(3) $M_e$ 可以为 0*

`group_sizes[e] = 0` 时 kernel 应该跳过——早期 CUTLASS 版本有 bug 会 nan。测试时先跑 $E = 2, M_1 = 0, M_2 = N K$ 的极端 case 验证。

*(4) $W_"stacked"$ 的 layout*

`W_up: (E, H, I)` 意味着 $E$ 份权重*连续存储*。加载时对每个 group，kernel 会从 `W_up[e]` 起始位置取。这个 layout 也影响后续的 EP 权重分片——不同 EP rank 持有不同 slice `W_up[e_start:e_end]`。

== v2: Fused Router

Router 的 Linear + softmax + topk + renorm 是 4 次 kernel launch。融成一个 kernel 的关键 trick：*$E$ 很小*（$≤ 32$ 常见），一整行 logits 可以塞进 register/shared memory，从头到尾不落 HBM。

=== Triton 参考实现

只写"logits → softmax(fp32) → top-K → renorm"这四步的 fusion（假设 GEMM 之后接这个 kernel；或者更激进的实现把 gate GEMM 也 fuse 进来，见下文）。

```python
import torch, triton
import triton.language as tl

@triton.jit
def fused_topk_softmax_kernel(
    logits_ptr,          # (N, E), 输入
    weights_ptr,         # (N, K), 输出 renormed weights
    indices_ptr,         # (N, K), 输出 expert ids
    N, E: tl.constexpr,  # N runtime, E compile-time (小, ≤32)
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    row_mask = row_ids < N

    # -- 1. load logits (BLOCK_N, E) --
    col = tl.arange(0, E)                                          # (E,)
    ptrs = logits_ptr + row_ids[:, None] * E + col[None, :]        # (BLOCK_N, E)
    x = tl.load(ptrs, mask=row_mask[:, None], other=-float("inf"))
    x = x.to(tl.float32)                                            # fp32 accumulator

    # -- 2. softmax (fp32, numerically stable) --
    x = x - tl.max(x, axis=1, keep_dims=True)
    e = tl.exp(x)
    p = e / tl.sum(e, axis=1, keep_dims=True)                       # (BLOCK_N, E)

    # -- 3. top-K by repeated masked argmax (K 小, unrolled) --
    for k in tl.static_range(K):
        v = tl.max(p, axis=1, keep_dims=True)                       # (BLOCK_N, 1)
        idx = tl.argmax(p, axis=1)                                  # (BLOCK_N,)
        # 写回本轮 top-1
        out_w = weights_ptr + row_ids * K + k
        out_i = indices_ptr + row_ids * K + k
        tl.store(out_w, tl.reshape(v, (BLOCK_N,)), mask=row_mask)
        tl.store(out_i, idx.to(tl.int32),          mask=row_mask)
        # mask 掉本轮 winner，进入下一轮
        p = tl.where(col[None, :] == idx[:, None], -float("inf"), p)

    # -- 4. renormalize (读回 K 个权重，除以 sum) --
    #    如果 K 很小可以在上面循环里就累一个 sum，然后最后一次写；这里为清晰分开。
    w_ptrs = weights_ptr + row_ids[:, None] * K + tl.arange(0, K)[None, :]
    w = tl.load(w_ptrs, mask=row_mask[:, None])
    w = w / tl.sum(w, axis=1, keep_dims=True)
    tl.store(w_ptrs, w, mask=row_mask[:, None])


def fused_topk(logits: torch.Tensor, K: int, BLOCK_N: int = 64):
    N, E = logits.shape
    weights = torch.empty((N, K), dtype=torch.float32, device=logits.device)
    indices = torch.empty((N, K), dtype=torch.int32,   device=logits.device)
    grid = (triton.cdiv(N, BLOCK_N),)
    fused_topk_softmax_kernel[grid](
        logits, weights, indices,
        N=N, E=E, K=K, BLOCK_N=BLOCK_N,
    )
    return weights, indices
```

用法：

```python
gate_logits = hidden @ W_gate.T            # (N, E), bf16 GEMM 依然走 cuBLAS
w, idx = fused_topk(gate_logits, K=2)      # (N, K), (N, K)
```

=== 关键点

*(1) fp32 accumulator 是刚需*

`x.to(tl.float32)` 是这个 kernel 的正确性核心——第 3 章讲过 bf16 下 softmax 会崩。这也是为什么 fused router 一定要自己写：`F.softmax(..., dtype=torch.float32)` 内部会 materialize 一个 fp32 中间 tensor，而这里全程在 register。

*(2) top-K 的选择*

上面用"K 次 masked argmax"是 $O(K E)$，对 $K ≤ 8, E ≤ 32$ 完全够用。想更快可以用 bitonic sort 的 static kernel（vLLM `fused_moe.py` 的做法），但代码复杂 3 倍、收益 $< 1%$。

*(3) 一步到位的 gate + router fusion*

上面只 fuse 了 softmax/topk/renorm。更激进的实现把 gate GEMM 也 fuse 进来：$X$ (bf16, $(N, H)$) → $W_g$ (bf16, $(H, E)$) → 一个 kernel 直接输出 (weights, indices)。$E$ 很小时 GEMM 是 memory-bound skinny GEMM，用 Triton 手写反而比 cuBLAS 快 20-30%。SGLang 的 `moe_align_block_size` 和 vLLM `fused_moe_router` 都走这条路。

*收益*: 3-5% 端到端。相对小但*免费*——因为 CUDA graph 需要静态图，fuse 后减少 launch 有额外的 scheduling 收益。

*代表实现*: `vllm/model_executor/layers/fused_moe/fused_moe.py::fused_topk`（同样的 Triton 结构，做了更多 tuning）; SGLang `moe_align_block_size`。

== v3: Fused Permute

Permute (`hidden[perm_tokens]`) 本质是一个 gather，读 $(N K, H)$ elements。有两条 fusion 思路：

+ *独立 permute kernel*（简单，也能省 10-15%）—— 把 gather 写成一个 Triton kernel，比 `hidden[perm_tokens]` 少一次中间 tensor 的 HBM 往返。
+ *融进 grouped GEMM prologue*（激进）—— 不 materialize packed_input，直接从 hidden gather 到 GEMM 的 shared memory。生产实现基本都走这条。

=== 方案 1: 独立 permute 的 Triton 实现

```python
@triton.jit
def permute_kernel(
    src_ptr,          # (N, H) hidden_states
    dst_ptr,          # (M, H) packed_input, M = N*K
    perm_tokens_ptr,  # (M,) 每个 slot 的源 token id
    M, H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 每个 program 负责一行 packed_input
    m = tl.program_id(0)
    src_row = tl.load(perm_tokens_ptr + m)
    h_off = tl.arange(0, BLOCK_H)
    # 循环整行（H 可能 > BLOCK_H）
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + h_off
        mask = offs < H
        v = tl.load(src_ptr + src_row * H + offs, mask=mask)
        tl.store(dst_ptr + m       * H + offs, v, mask=mask)


def permute(hidden, perm_tokens, BLOCK_H=128):
    N, H = hidden.shape
    M    = perm_tokens.numel()
    dst  = torch.empty((M, H), dtype=hidden.dtype, device=hidden.device)
    permute_kernel[(M,)](hidden, dst, perm_tokens, M=M, H=H, BLOCK_H=BLOCK_H)
    return dst
```

反向就是 `index_add_`（对同一 src_row 累加 K 次 dst 梯度），也可以写成一个 atomic add Triton kernel。

=== 方案 2: 融进 grouped GEMM 的 prologue

伪代码：

```
grouped_gemm_with_permute_kernel(hidden, W_up, perm_tokens, group_offsets, ...):
    // 每个 CTA 处理一个 GEMM tile: BLOCK_M × BLOCK_N
    m0, group_id = tile_to_group(blockIdx, group_offsets)

    // 加载 BLOCK_M 行——原本从 packed_input 读，现在从 hidden gather
    for i in range(BLOCK_M):
        token_id = perm_tokens[m0 + i]
        As[i, :] = hidden[token_id, :]           // gather 直接到 shared memory

    // 加载 W_up[group_id] 的 BLOCK_N 列到 Bs
    // Normal tiled GEMM ...
```

Triton 里可以直接把 gather 写在 `tl.load` 的 pointer 表达式里：

```python
# grouped GEMM inner loop 里，读 A tile 的部分
row_ids   = m0 + tl.arange(0, BLOCK_M)                      # (BLOCK_M,)
token_ids = tl.load(perm_tokens_ptr + row_ids)              # gather idx
a_ptrs    = hidden_ptr + token_ids[:, None] * H + k_offs[None, :]
a         = tl.load(a_ptrs, mask=...)
# 后续 tl.dot(a, b) 走 tensor core
```

*收益*: 10-15%。省了一次 $(N K, H)$ 的 HBM 读写。当 $N K H$ 大（千 MB 级）时收益显著。

*代价*: kernel 复杂——permute 的 gather pattern 会破坏内存 coalescing，需要 shared memory reshuffle 优化。NVIDIA 的 `grouped_gemm` 库、Megatron 的 `moe_permute` op、Tutel 的 `fast_dispatch` 都做了这个 fusion。裸调 `torch._grouped_mm` *不*包含 fused permute，需要额外 wrap。

#insight[
  从 v1 到 v3 是"MoE 层从教学到生产"的核心转变。v1 用现成 API 就能实现；v3 需要 CUDA/CUTLASS 级别的自定义 kernel。生产实现的 90% 复杂度在 v3 这一步。
]

== v4: Fused SwiGLU

SwiGLU 里 $"SiLU"(x) dot y$ 是 elementwise 但*不可交换*——需要 gate 和 up 两路都在 register 才能算。生产的做法是让一次 grouped GEMM 同时产出 gate 和 up，然后在 epilogue 里直接 SiLU + 乘。

=== Weight concat 布局

关键工程 trick：把 `W_gate` 和 `W_up` 沿输出维 concat：

```python
# 训练开始时构造一次
W_gate_up = torch.cat([W_gate, W_up], dim=-1)      # (E, H, 2*I)
```

于是一次 grouped GEMM 直接出 $(N K, 2I)$：

```python
def swiglu_grouped_gemm(packed_input, W_gate_up, W_down, group_sizes):
    # packed_input: (M, H),  W_gate_up: (E, H, 2*I),  W_down: (E, I, H)
    gate_up = torch._grouped_mm(packed_input, W_gate_up, group_sizes)  # (M, 2I)
    hidden  = fused_silu_glu(gate_up)                                   # (M, I)
    out     = torch._grouped_mm(hidden, W_down,    group_sizes)         # (M, H)
    return out
```

=== `fused_silu_glu` 的 Triton 实现

$"SwiGLU"(x, y) = "SiLU"(x) dot y = (x / (1 + e^(-x))) dot y$。写成一个 kernel（含反向，因为训练要）：

```python
@triton.jit
def silu_glu_fwd_kernel(
    inp_ptr,        # (M, 2*I) — 前 I 列是 gate，后 I 列是 up
    out_ptr,        # (M, I)
    M, I: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)                     # 一个 program 负责一行的一段
    row = pid // tl.cdiv(I, BLOCK)
    col0 = (pid %  tl.cdiv(I, BLOCK)) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    mask = (row < M) & (cols < I)

    gate = tl.load(inp_ptr + row * 2 * I + cols,     mask=mask).to(tl.float32)
    up   = tl.load(inp_ptr + row * 2 * I + I + cols, mask=mask).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    out  = silu * up

    tl.store(out_ptr + row * I + cols, out.to(inp_ptr.dtype.element_ty), mask=mask)


def fused_silu_glu(x: torch.Tensor) -> torch.Tensor:
    M, two_I = x.shape
    assert two_I % 2 == 0
    I = two_I // 2
    y = torch.empty((M, I), dtype=x.dtype, device=x.device)
    BLOCK = 128
    grid  = (M * triton.cdiv(I, BLOCK),)
    silu_glu_fwd_kernel[grid](x, y, M=M, I=I, BLOCK=BLOCK)
    return y
```

对应的 backward 也一样简单：$partial L / partial x_"gate" = partial L / partial "out" dot y dot "SiLU"'(x)$，$partial L / partial x_"up" = partial L / partial "out" dot "SiLU"(x)$，写一个双输出 kernel 就行。

=== 更激进：Option B — 完整 fused MoE mega-kernel

vLLM 的 `fused_moe.py::fused_experts` 把 permute + grouped GEMM 1 + SwiGLU + grouped GEMM 2 + weight + unpermute 全部融成*一个* kernel（推理侧，无 backward）。训练用得少——灵活性差、backward 难写。

*收益*: SwiGLU epilogue fusion 约 5%。完整 mega-kernel 在小 batch 下额外 +10%。

== v5: Block-Sparse (Megablocks)

第 4 章"范式 C"已讲过原理。这里补充训练场景的对比：

*Grouped GEMM (v1-v4)*:
- 需要 capacity + drop，或者手动 padding
- Drop 造成信息丢失，尤其训练早期 collapse 时严重
- Kernel 简单，主流 (Megatron, Tutel, DeepSpeed)

*Block-Sparse (Megablocks)*:
- 无 drop，训练动力学更好
- Kernel 复杂 (block-sparse metadata)
- 单机成熟，分布式仍在演化

一个具体数字：Megablocks 论文报告在 1.3B MoE 上 vs 相同 capacity 的 grouped GEMM 版本，*同 wall-clock 训练*下 loss 低 0.02——不多，但免费。

工程选择：如果 team 有 CUDA 人力，Megablocks 值得投入；如果依赖开源框架，grouped GEMM 是稳妥选择。

== v6: torch.compile

`torch.compile` 对 MoE 的作用取决于代码结构：

*(1) 范式 A (for-loop) 上编译*

有 graph break（`torch.where` + 动态 shape gather）。收益有限（$< 5%$）。

*(2) 范式 B (grouped GEMM) 上编译*

Router 和 permute 部分能编译，grouped GEMM 本身是 pre-compiled kernel（编译器不动它）。整体收益 10-15%。

*(3) 完整 fused MoE 上编译*

如果已经 v5 级别 fused，`torch.compile` 主要吃 Python overhead 层（每 forward 的调度）——CUDA graph capture 有额外收益。

生产实现里 `torch.compile` 更多用在 non-MoE 部分（attention、norm）；MoE 部分依赖手写 kernel。

== 显存优化

MoE 显存开销与 dense 的差别：

*(1) 权重*：$E$ 倍。$E=8$ 就是 8 倍 FFN 权重。这是 MoE 的固有开销。

*(2) 激活*：与 dense 相当。中间 packed_hidden $(N K, I)$ 只比 dense 大 $K$ 倍。

*(3) Optimizer state*：与权重同比例增大。AdamW 每参数 12 bytes (bf16 训练下 fp32 momentum + fp32 variance + fp32 master + bf16 param + bf16 grad ≈ 4+4+4 = 12 for state, param/grad separate)，是模型权重的 6-8×。

显存优化手段：

*(a) ZeRO-1/2/3 分片*：optimizer state / grad / param 沿 DP 维度切。DeepSpeed 集成。

*(b) Selective activation ckpt*：packed_hidden $(N K, I)$ 是最大激活，checkpoint 掉最省内存。代价 30% 显存 vs 30% 额外算力。

*(c) Expert offload*：冷专家 (低命中率) 参数 offload 到 CPU/NVMe，需要时 prefetch。稀疏专家（DeepSeek 256 experts）这个策略特别有效——推理侧 vLLM 已经实现。

*(d) 量化权重*：bf16 → fp8 (Hopper) 或 int4 (推理)。训练侧 fp8 主参数 + fp32 master 是当前趋势 (TransformerEngine)。

== 面试考点

#interview[
  *Q1*: Grouped GEMM 相比 for-loop GEMM 的核心收益来源？

  A: 三点：(1) 减少 $E-1$ 次 kernel launch overhead；(2) 全局 tile scheduling —— 小 group 和大 group 混合调度、SM 常驻满载；(3) L2 缓存复用 —— weight 数据在 GEMM 内跨 group 尝试保留。SM 利用率从 30% 提升到 60-80%。
]

#interview[
  *Q2*: 为什么 permute 融合到 GEMM prologue 收益大？

  A: Permute 独立做时，$(N K, H)$ 的 tensor 要写一次 HBM 再读一次；融合后从原始 hidden 直接 gather 到 GEMM shared memory，*不产生*中间 tensor 的 HBM 往返。当 $N K H$ 是几百 MB 量级时，节省的 HBM 带宽换算成 latency 就是 10-15% 端到端。
]

#interview[
  *Q3*: 训练时能不能不 drop token？

  A: 三种方式：(1) 极大 capacity factor (≥ 2)，牺牲显存 + kernel 效率；(2) 用 Megablocks (block-sparse)，从算法上避免；(3) 用 Expert Choice routing，天然均衡。Mixtral 训练用 (1) 兜底，DBRX 用 (2)。
]

#interview[
  *Q4*: 单机 MoE 里 activation checkpointing 该 checkpoint 什么？

  A: `packed_hidden: (N*K, I)` 是最大 activation ($4K$ x dense FFN)，first choice。次选是 `gate_probs (N, E)` 但小。*不要* checkpoint `expert_indices` — 是 int，重算成本高（要重跑 softmax + topk）。
]

#interview[
  *Q5*: MoE 的 SM 利用率天生比 dense 低吗？

  A: 不一定。dense FFN 是一个大 GEMM，SM 利用率 80-90%；MoE grouped GEMM 因为 group size 可能小、L2 复用差、tile 调度不完美，天花板约 70-80%。差距 10-15% —— 用大 batch (让 $M_e$ 更大) 可以缩小。
]

#interview[
  *Q6*: 如果我在 forward 里发现某个 expert 显存溢出，能动态 offload 吗？

  A: 训练侧不建议——forward/backward 都需要该 expert 的权重，来回搬 CPU 会显著拖慢。正确做法：(1) EP 分布式，让 expert 分布到不同 GPU；(2) 上 fp8 减半权重内存；(3) offload 只对*冷 expert* 有意义 (推理侧 vLLM 用得多)。
]
