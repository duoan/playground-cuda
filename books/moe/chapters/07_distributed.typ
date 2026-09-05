#import "../template.typ": *

= 分布式训练：EP、All-to-All 与 Fine-Grained Overlap

单机 MoE 的性能上限是单张 GPU 的算力和显存。要训 100B+ 的 MoE 模型，必须跨机分布式。这是本书最长、也最容易出错的一章。

我们的组织方式：先建立 *3 堵墙* 的诊断框架（Memory / Communication / Compute Efficiency），再依次讲每堵墙 Megatron-Core、DeepSeek-V3、MegaScale-MoE 这些生产系统各自怎么打。这三堵墙*强耦合*——解决一堵往往会暴露另一堵，工业界的所谓"最佳实践"其实就是在三者之间反复权衡的结果。

== MoE 训练的三堵墙 (Three Walls)

Megatron-Core MoE tech report#footnote[Cui et al., _Scalable Training of Mixture-of-Experts Models with Megatron-Core_, 2026.] 把 MoE 分布式训练遇到的问题归纳成三堵互相制约的墙：

#figure(
  table(
    columns: (auto, 1.3fr, auto, 1.6fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, center, left),
    [*Wall*], [*根因*], [*度量指标*], [*典型症状*],
    [*Memory Wall*],
    [E 个 expert 的 weight/grad/opt state 全部常驻，但每 token 只激活 K 个],
    [GB/GPU],
    [DeepSeek-V3 BF16、PP4×VPP4×EP64、256 GPU：*199.5 GB/GPU*——H100 80GB 装不下],
    [*Communication Wall*],
    [EP 需要 dispatch + combine 两次 all-to-all；跨节点带宽骤降 (IB 50 GB/s vs NVLink 160 GB/s)],
    [step time 中通信占比],
    [未优化时 EP a2a 占 forward *20-60%*；DeepSeek-V3 类跨节点场景可达 *~60%*],
    [*Compute Efficiency Wall*],
    [细粒度 expert → 小 GEMM + 大量 kernel launch + host sync],
    [SM 利用率 / MFU],
    [细粒度 MoE 每 expert 只有 ~128 tokens，远低于 TensorCore 饱和阈值 (M ≥ 128)；dropless routing 引入 D2H sync],
  ),
  kind: table,
  caption: [MoE 训练的三堵墙。三者*强耦合*：解决内存墙 → 引入 EP → 暴露通信墙 → overlap → 需要更大 batch → FP8 → kernel 碎片化 → CUDA Graph。],
)

*a2a 通信量公式*（每卡每次 dispatch）：

$ "vol"_"a2a" approx T dot K dot h dot (E P - 1) / (E P) $

一次 MoE 层 forward + backward 有 *4 次* a2a（2 dispatch + 2 combine），32 层模型每步 total ~ *16-64 GB* per GPU。跨节点未 overlap 时这直接决定 step time。

#insight[
  3 walls 的意义是给出*诊断顺序*：先量出你当前 bottleneck 在哪堵墙，再选对应武器。Megatron-Core 的 `--profile` 与 DeepSeek profile-data#footnote[https://github.com/deepseek-ai/profile-data] 都提供了区分这三类开销的 trace 工具。
]

=== 参考基线：一个 8 × 8 × 7B MoE 的开销分布

以 Mixtral 8×7B (32 层、$H=4096, I=14336, E=8, K=2$) 在 64 × H100 上、`TP=1, PP=4, EP=8, DP=2, MBS=1, seq=4096` 的 baseline 数字（Megatron-Core v0.9 官方 benchmark 附近）：

#align(center)[
  #time-share-bar(
    (
      ("Attention (dense)",              22.0),
      ("MoE Router + Gate",               3.0),
      ("MoE Dispatch a2a (fwd+bwd)",     18.0),
      ("MoE Expert grouped GEMM",        30.0),
      ("MoE Combine a2a (fwd+bwd)",      15.0),
      ("Optim + DP grad sync",            8.0),
      ("其他 (LN / cast / launch)",       4.0),
    ),
    width: 8.5,
    label-w: 5.4,
  )
]

*读法*：Attention + Expert 加起来 52%——这是"有效计算"；剩下 48% 里 33% 是 MoE 特有的 a2a。这就是 3 walls 的具体体现——你能把 33% 的通信 overlap 到接近 0，就能拿到理论上限的 ~1.7× 加速；这正是 DeepSeek DualPipe 与 Megatron-Core `--overlap-moe-expert-parallel-comm` 的战场。

数字都是数量级估算，实际 profile 请以 nsys / DeepSeek profile-data 为准。

== 四种并行的分工

MoE 分布式训练涉及*四个正交*的并行维度：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, left),
    [*并行*], [*切什么*], [*通信原语*], [*对 MoE 的作用*],
    [DP (Data Parallel)], [batch 沿样本切], [AllReduce (grad)], [每 replica 有完整模型],
    [TP (Tensor Parallel)], [单个 GEMM 内切张量], [AllReduce / AllGather], [切 attention/expert 内的 GEMM],
    [PP (Pipeline Parallel)], [沿 layer 切], [P2P send/recv], [切 Transformer stack],
    [EP (Expert Parallel)], [专家沿 expert 维切], [*All-to-All*], [MoE 独有],
  ),
  kind: table,
)

四种并行可以*任意组合*：

$ "world_size" = "DP" times "TP" times "PP" times "EP" $

例如 Mixtral 8×7B 训练常见配置：world = 64 卡，$"TP"=2, "EP"=8, "DP"=4, "PP"=1$，$2 times 8 times 4 = 64$。

== 为什么需要 Expert Parallel

单卡 A100 80GB 能装多大 MoE？

- Mixtral 8×7B：47B params × 2 bytes (bf16) = 94 GB — *装不下*
- 加上 optimizer state (12 bytes × 47B = 564 GB) — 需要 8+ 卡

必须切模型。四种切法在 MoE 里的适用性：

*(1) 纯 DP*：每卡都装全 MoE。不适用——单卡装不下。

*(2) TP 切 expert GEMM*：把 $W_"up" in RR^(H, I)$ 沿 $I$ 维切。可行，但 $E$ 个 expert 都被切 TP 份——每卡显存 = $E/("TP")$ expert 权重。等价于每卡还是有 $E/("TP")$ 个"半专家"，扩展性有限。

*(3) EP 切 expert*：$E$ 个专家分布在 $"EP"$ 张卡上，每卡持有 $E/"EP"$ 个专家。这是最 natural 的切法——*模型稀疏性 = 通信稀疏性*。

*(4) PP 切 layer*：一样切分，与 MoE 内部结构无关。

EP 是 MoE 训练的*核心*，其他三种是配套。这一章的 80% 内容是 EP。

== Wall 1 · Memory Wall：显存怎么省

单纯用 EP 切 expert 权重就能省，但还不够。DeepSeek-V3 tech report Table 3 给出的显存分解（BF16、256 GPU、PP4×VPP4×EP64）：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, left),
    [*组成*], [*GB/GPU*], [*占比*],
    [Weight + Grad],        [36.4],  [18%],
    [Optimizer state],      [32.1],  [16%],
    [Activation],           [131.0], [66%],
    [*Total*],              [*199.5*], [100%],
  ),
  kind: table,
  caption: [DeepSeek-V3 671B 显存分解。三大头里 activation 占 2/3——这决定了 memory wall 的主要武器是 *激活优化* (recomp + offload + FP8)，而不是 weight 分片。],
)

Megatron-Core 和 DeepSeek 各自的 memory-wall 优化菜单：

为节约横向空间，我们把"Megatron flag"单独列在下方一段，表里只保留*机制*与*收益*：

#figure(
  table(
    columns: (1.1fr, 2.2fr, 1.7fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*优化*], [*机制*], [*DS-V3 / 参考收益*],
    [Memory-Efficient Permutation],
    [Router 权重 $tilde(w)$ 乘到 activation 上*再进* FC2，数学等价（无 bias 时），省掉为 router backward 保存的 per-expert 输出],
    [激活 ~*26.3 GB* → 换来更多 recompute 空间],
    [Fine-grained Recomputation],
    [模块级 output-discarding checkpoint，只重算 cheap 子模块（MLA up-proj、SwiGLU、LN），*不*整层 recompute (否则重触发 a2a)],
    [MLA up-proj 30.4 + SwiGLU 3.8 + LN 8.2 = *42.4 GB*，额外算力 $< 5%$],
    [Fine-grained Activation Offloading],
    [模块级 D2H/H2D 加独立 CUDA stream，与 1F1B 的 a2a 兼容 overlap],
    [mem *-10.7%*，吞吐 *-1.6%*；换 mapping 后 Qwen3-235B 反而 *+15%* 吞吐],
    [Precision-aware Optimizer],
    [Adam moment 存 BF16/FP8，update 时 fused kernel 里 upcast 到 FP32],
    [Optimizer state *~50%*],
    [Optimizer CPU Offload],
    [forward/backward 时 optim state 在 CPU，step 前拉回],
    [*15-20 GB*，iter *+0.1-0.2 s*],
    [FSDP + EP (Dual DeviceMesh)],
    [Dense 层 FSDP over DP；Expert 层 FSDP over EDP；AllGather/RS 限定在小 EDP 组内，非-uniform sharding + NCCL UBR 零拷贝],
    [Llama3 405B FSDP comm *~10%↓*],
    [FP8/FP4 Activation],
    [线性层输入按 FP8 存 checkpoint，激活占用 *~50%↓*],
    [激活 ~*16 GB*（DS-V3 activation 12%）],
    [Distributed Optimizer (ZeRO-1)],
    [Adam state 沿 DP 分片，通常与 FSDP 二选一],
    [—],
    [Capacity-based Drop],
    [超 capacity 的 token 走 residual 跳过 FFN，activation shape 静态化便于 CUDA graph],
    [训练早期防 OOM],
  ),
  kind: table,
  caption: [Memory Wall 常用武器一览。DS-V3 收益数字来自 Megatron-Core Tech Report Table 3-5。],
)

对应的 Megatron-Core CLI flag：

#figure(
  table(
    columns: (1.1fr, 2.9fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left),
    [*优化*], [*Megatron flag*],
    [Memory-Efficient Permutation],
    [默认 (v0.13+) + `--moe-permute-fusion`],
    [Fine-grained Recomputation],
    [`--recompute-granularity selective`\
     `--recompute-modules mla_up_proj layernorm moe_act moe mlp core_attn`],
    [Fine-grained Activation Offloading],
    [`--fine-grained-activation-offloading`\
     `--offload-modules expert_fc1 moe_act ...`],
    [Precision-aware Optimizer],
    [`--use-precision-aware-optimizer`\
     `--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`],
    [Optimizer CPU Offload],
    [`--optimizer-cpu-offload`],
    [FSDP + EP (Dual DeviceMesh)],
    [Megatron-FSDP 集成 (v0.15+)，无独立 flag，配合 `--use-distributed-optimizer`],
    [FP8/FP4 Activation],
    [`--fp8-format e4m3 --fp8-recipe blockwise`],
    [Distributed Optimizer (ZeRO-1)],
    [`--use-distributed-optimizer`],
    [Capacity-based Drop],
    [`--moe-expert-capacity-factor 1.0 --moe-pad-expert-input-to-capacity`],
  ),
  kind: table,
  caption: [Memory Wall 优化对应的 Megatron-Core 命令行 flag。],
)

#insight[
  切 memory 的顺序有讲究：*先切 optimizer state* (ZeRO-1 / precision-aware) → *再切 activation* (recompute + offload + FP8) → *最后切 weight* (EP + TP)。因为 optimizer state 与 weight 无关但占用最大，切它不影响计算通信；activation 决定 batch size 上限；weight 切碎会引入通信。反过来做通常会陷入"切了一堆通信却省不了多少内存"。
]

== EP 数据流：从单机到 EP

=== 单机 (EP=1) 回顾

第 5 章的 MoE forward：

```
X: (N, H) → router → indices, weights
                   ↓
                dispatch → each expert 本地计算 → combine
                   ↓
                out: (N, H)
```

所有 expert 都在本卡，dispatch 是本卡 memory scatter。

=== EP=8 时会发生什么

假设 world_size=8 (全 EP，无 DP/TP/PP)，$E=8$，每卡持有 1 个专家。每卡有本地 tokens $X_"local": (N_"local", H)$。

*Router 计算*：*每卡独立*算自己 tokens 的路由。因为 gate weight `W_g` 是完整的（$W_g$ 很小，全 rank 都存一份），每卡都能算完整 topk。

*但 dispatch 遇到问题*：本卡上的 token 可能被路由到 *其他卡上的 expert*。这就需要*跨卡把 token 送到目标 expert 所在的卡*。

这就是 all-to-all。

=== All-to-All 通信模式

*All-to-all* = 每个 rank 向每个其他 rank 各发一份数据。语义：

```
before:            after all-to-all:
  rank 0: [a0, b0, c0, d0]      rank 0: [a0, a1, a2, a3]
  rank 1: [a1, b1, c1, d1]      rank 1: [b0, b1, b2, b3]
  rank 2: [a2, b2, c2, d2]      rank 2: [c0, c1, c2, c3]
  rank 3: [a3, b3, c3, d3]      rank 3: [d0, d1, d2, d3]
```

即 rank $i$ 发送第 $j$ 段给 rank $j$，接收所有 rank 发给自己的段。

MoE 的用法：*rank $i$ 的第 $j$ 段 = 应该由 expert $j$ (在 rank $j$) 处理的 tokens*。

=== EP forward 完整流程

```
每卡有 X_local: (N_local, H)
      ↓
router 本地 → expert_indices (N_local, K), expert_weights (N_local, K)
      ↓
按 expert 归属 rank 排序 → packed_input (N_local × K, H)
   同时构造 send_counts (EP,) — 发给每个 rank 多少 token
      ↓
[all_to_all_1] packed_input, send_counts
      ↓
本卡收到 recv_input (M_local, H)
   其中 M_local = 应由本卡上的 expert 处理的所有 token 总数
      ↓
本地 grouped GEMM: recv_input → local experts → recv_output (M_local, H)
      ↓
[all_to_all_2] recv_output, recv_counts (是发送时的 send_counts 的转置)
      ↓
本卡收回 (N_local × K, H)
      ↓
乘以 expert_weights + unpermute + reduce → out (N_local, H)
```

=== 完整 PyTorch 实现（教学版）

用 `torch.distributed.all_to_all_single` 写出来的极简 EP MoE，端到端能跑。跳过了 fused kernel、只保证语义正确：

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

class EPMoE(nn.Module):
    """
    Expert-Parallel MoE (教学版).
    - world_size = EP, 无 DP/TP;
    - 每 rank 持有 num_experts // EP 个 expert 的 W_up / W_down.
    """
    def __init__(self, hidden: int, inter: int, num_experts: int,
                 top_k: int, ep_group=None):
        super().__init__()
        self.ep_group   = ep_group or dist.group.WORLD
        self.ep_size    = dist.get_world_size(self.ep_group)
        self.ep_rank    = dist.get_rank(self.ep_group)
        assert num_experts % self.ep_size == 0
        self.E, self.K  = num_experts, top_k
        self.E_local    = num_experts // self.ep_size          # 本卡专家数

        # Router 每卡 replicate
        self.gate       = nn.Linear(hidden, num_experts, bias=False)
        # 本地 expert 权重
        self.W_up       = nn.Parameter(torch.empty(self.E_local, hidden, inter))
        self.W_down     = nn.Parameter(torch.empty(self.E_local, inter,  hidden))
        nn.init.kaiming_uniform_(self.W_up)
        nn.init.kaiming_uniform_(self.W_down)

    def forward(self, x):
        N, H = x.shape
        # -- 1. router (每卡独立) --
        logits = self.gate(x)                                    # (N, E)
        probs  = F.softmax(logits, dim=-1, dtype=torch.float32).to(x.dtype)
        weights, indices = torch.topk(probs, self.K, dim=-1)     # (N, K), (N, K)
        weights = weights / weights.sum(-1, keepdim=True)

        # -- 2. 展平 + 按目标 rank 排序 --
        flat_expert = indices.flatten()                          # (N*K,)
        flat_weight = weights.flatten()                          # (N*K,)
        flat_token  = torch.arange(N, device=x.device).repeat_interleave(self.K)

        target_rank = flat_expert // self.E_local                # 每 slot 该发往哪个 rank
        perm        = target_rank.argsort(stable=True)
        send_tokens = x[flat_token[perm]]                        # (N*K, H) 打包好待发
        send_weights = flat_weight[perm]                         # (N*K,)
        send_expert  = flat_expert[perm]                         # (N*K,) 目标 expert (全局 id)

        send_counts = torch.bincount(target_rank, minlength=self.ep_size)
        # 3.a 先 all-to-all 交换 counts, 拿到 recv_counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        M_local = int(recv_counts.sum().item())

        # -- 3.b all-to-all 数据: (dispatch) --
        recv_tokens = torch.empty((M_local, H),
                                  dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            recv_tokens, send_tokens,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes =send_counts.tolist(),
            group=self.ep_group,
        )
        # 相同 pattern 也发 expert id 和 weight, 用于后面的 grouped GEMM
        recv_expert = torch.empty(M_local, dtype=send_expert.dtype, device=x.device)
        recv_weight = torch.empty(M_local, dtype=send_weights.dtype, device=x.device)
        dist.all_to_all_single(recv_expert, send_expert,
                               output_split_sizes=recv_counts.tolist(),
                               input_split_sizes =send_counts.tolist(),
                               group=self.ep_group)
        dist.all_to_all_single(recv_weight, send_weights,
                               output_split_sizes=recv_counts.tolist(),
                               input_split_sizes =send_counts.tolist(),
                               group=self.ep_group)

        # -- 4. 本地 expert 计算 (grouped GEMM) --
        # 把全局 expert id 转本地
        local_expert = recv_expert - self.ep_rank * self.E_local  # ∈ [0, E_local)
        # 按 local expert 再排一次序 (方便 grouped GEMM)
        perm2        = local_expert.argsort(stable=True)
        packed_in    = recv_tokens[perm2]
        group_sizes  = torch.bincount(local_expert, minlength=self.E_local)
        # torch._grouped_mm 需要 PyTorch 2.4+; 无则退化到 for-loop
        h  = torch._grouped_mm(packed_in, self.W_up,   group_sizes)
        h  = F.silu(h)                                            # 演示用 SiLU
        y  = torch._grouped_mm(h,         self.W_down, group_sizes)
        # unpermute 回 recv 顺序
        expert_out = torch.empty_like(y)
        expert_out[perm2] = y
        # 乘 gate weight
        expert_out = expert_out * recv_weight.unsqueeze(-1)

        # -- 5. all-to-all 送回 (combine) --
        # send_counts / recv_counts 的角色互换
        back_tokens = torch.empty_like(send_tokens)
        dist.all_to_all_single(
            back_tokens, expert_out,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes =recv_counts.tolist(),
            group=self.ep_group,
        )

        # -- 6. inverse permute + reduce over K --
        # back_tokens 里的顺序仍是 perm 后的; 恢复到 flat (N*K,) 原序
        flat_out = torch.empty_like(back_tokens)
        flat_out[perm] = back_tokens
        out = flat_out.view(N, self.K, H).sum(dim=1)              # (N, H)
        return out, logits
```

跑起来（单机 4 卡模拟 EP=4）：

```bash
torchrun --nproc_per_node=4 test_ep_moe.py
```

=== 生产级实现的差别

上面的教学版有几个明显的性能问题，生产实现都会修：

+ *4 次独立 all-to-all*（1 次 counts + 1 次 tokens + 1 次 expert ids + 1 次 weights）—— 生产实现把 expert id / weight 编码进 token 首几个字节，只做 1 次 counts + 1 次数据 all-to-all
+ *permute / unpermute 是独立 kernel* —— fuse 进 grouped GEMM prologue（第 7 章 §"v3 Fused Permute"）
+ *all-to-all 是同步的* —— 生产用 `dist.all_to_all_single(..., async_op=True)` 拿 handle，让下游 attention / next-layer forward overlap
+ *`bincount` 是 GPU-CPU sync* —— 用固定 shape + shape hints 避免（Tutel 的做法）
+ *torch autograd 会为 all_to_all 自动生成反向*，但需要用 Megatron 的 `_AllToAll.apply` 或 DeepEP 的封装才能拿到 async backward

参考实现：Megatron-LM `megatron/core/transformer/moe/token_dispatcher.py`, DeepEP (DeepSeek 开源) `deep_ep/buffer.py`。

用图表达：

#align(center)[
  #a2a-diagram(
    n-ranks: 4,
    // 各 rank 上的 token 分布 (示例 N_local=4, K=1, 每 rank 各选不同 expert)
    before: (
      ((0, 2), (1, 1), (3, 1)),
      ((0, 1), (2, 2), (1, 1)),
      ((1, 2), (2, 1), (3, 1)),
      ((0, 1), (3, 3)),
    ),
    // all-to-all 后：每 rank 收到属于本地 expert 的 token
    after: (
      ((0, 2), (0, 1), (0, 1)),
      ((1, 1), (1, 1), (1, 2)),
      ((2, 2), (2, 1)),
      ((3, 1), (3, 1), (3, 3)),
    ),
    title: "All-to-all (dispatch): tokens 按目标 expert 分发",
  )
]

== 通信量分析

设：
- $N_"local"$: 每卡 token 数
- $K$: top-K
- $H$: hidden dim
- $"EP"$: EP world size

*Forward 的 all-to-all*：每卡向其他 rank 送 $tilde.op N_"local" K / "EP" times H$ 数据（假设路由均衡），总 send/recv volume：

$ "vol"_"a2a" approx N_"local" K H times "bytes"_"dtype" $

对 $N_"local"=4096, K=2, H=4096, "bf16"$：

$ "vol"_"a2a" = 4096 times 2 times 4096 times 2 = 128 "MB" $

单层 MoE forward 有 *2 次* all-to-all（dispatch + combine），backward 有 *2 次*（反向送 grad + 反向送 activation grad），总 *4 次 × 128 MB = 512 MB* per layer per step。

一个 32 层的 MoE 训练：$32 times 512 "MB" = 16 "GB"$ per step 的 all-to-all 通信量。在 3.2 Tbps NVLink (400 GB/s bidi) 下需要 $tilde.op 40$ ms 通信；在 400 Gbps IB (50 GB/s) 下需要 $tilde.op 320$ ms。这就是为什么大规模 MoE 训练*通信/计算 overlap* 极其重要。

#insight[
  一个反直觉：MoE 的通信量 *不* 依赖于 $E$（专家总数），只依赖于 $N K H$。加多专家不增加通信量——每 token 还是走 $K$ 个专家。这也是 DeepSeek-V3 敢用 256 experts 的原因：通信开销与 8 experts 相同。
]

== Wall 2 · Communication Wall：a2a 怎么打

MoE 通信优化在过去两年出了三代方案，一代比一代激进。理解这条演进对定位自己的问题很关键：

+ *NCCL alltoall*（baseline）：`torch.distributed.all_to_all_single`；一对 rank 一个 IB pair，*每对都占带宽*，跨节点极慢。
+ *Hierarchical a2a*（DeepSpeed-MoE、Tutel）：intra-node NVLink 聚合 → 单次 inter-node IB → intra-node 散发。
+ *IB + NVLink 并行调度 + warp specialization*（DeepEP、HybridEP）：一次 dispatch 让 IB 和 NVLink *同时*工作，且 combine 里融合 reduction。

=== 方案 1：Hierarchical a2a（DeepSpeed-MoE 风格）

三步：intra-node → inter-node → intra-node。

```python
def hierarchical_a2a(x, send_counts, ep_group):
    node_size = 8                        # GPUs per node
    n_nodes   = ep_group.size() // node_size
    x_by_node = rearrange_by_target_node(x, send_counts)
    x1 = intra_a2a(x_by_node, node_size)   # NVLink 400 GB/s
    x2 = inter_a2a(x1,       n_nodes)      # IB     50  GB/s
    return intra_a2a(x2,     node_size)    # NVLink 再散发
```

IB 带宽消耗从 $O(N^2 K H)$ 降到 $O(N K H)$。100+ 节点上 wall-clock 差 10×。缺点：三段串行，IB 与 NVLink 各自独占带宽的时候另一方闲着。

=== 方案 2：DeepEP — IB + NVLink 并行 + warp specialization

DeepSeek 开源#footnote[https://github.com/deepseek-ai/DeepEP] 的 a2a 库，是 DeepSeek-V3 训练用的原型。两个关键改进：

*(a) IB / NVLink 并行调度*：目标节点内 *同 in-node index* 的 GPU 作为"落地点"——token 先 IB 送到 target node 的对应 index GPU，那边同时用 NVLink 二次分发；IB 和 NVLink 在时间上*完全重叠*而不是串行。

*(b) Warp specialization + auto-tuned chunk size*：20 SM 划分为 10 个 comm channel，dispatch 三阶段 (IB send / IB→NVLink forward / NVLink recv) 由不同 warp 负责；combine 对称，NVLink→IB 阶段*在 kernel 内完成 reduction*。

配合 *Node-Limited Routing*（每 token 最多发往 $M = 4$ 个节点），一次 dispatch 的 IB fan-out 有上界，通信量可预测。

API 长这样（V2）：

```python
import deep_ep
buf = deep_ep.Buffer(group, num_nvl_bytes=..., num_rdma_bytes=...)

# forward dispatch
recv_x, handle = buf.dispatch(
    x, topk_idx=..., topk_weights=...,
    num_experts=E, num_sms=6,        # V2 只要 4-6 SM
    async_finish=True,               # 拿到 handle 之后可以做别的
)
# backward dispatch 就是 combine
out = buf.combine(y, handle=handle, topk_weights=..., num_sms=6)
```

*性能数字*（H800 + CX7，8K tokens/batch, $H=7168$, top-8, FP8 dispatch / BF16 combine）：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, center, right, right, center),
    [*模式*], [*EP*], [*Dispatch BW*], [*Combine BW*], [*SM 数*],
    [Intranode (NVLink)], [8],  [153 GB/s],  [158 GB/s], [24 (V1)],
    [Internode (RDMA V1)], [64], [ 51 GB/s],  [ 50 GB/s], [24],
    [Internode (RDMA V2)], [32], [ 90 GB/s],  [ 81 GB/s], [12],
    [Internode (RDMA V2)], [64], [ 61 GB/s],  [ 61 GB/s], [ 6],
  ),
  kind: table,
  caption: [DeepEP dispatch/combine 有效带宽。V2 相比 V1 用 *4× 更少的 SM* 达到同等带宽——省下来的 SM 归还给 grouped GEMM。],
)

*代价*：DeepEP 占 *20 SM/GPU*（V1）或 4-6 SM（V2），带来大约 *20% GEMM 效率损失*（V1）。V2 才让 fine-grained MoE 真正可行。

=== 方案 3：HybridEP（NVIDIA GB200/MNNVL）— TMA + IBGDA

GB200 NVL72 场景下节点内 NVLink 域扩大到 72 卡，NVIDIA 提出 HybridEP：*TMA (Tensor Memory Accelerator) + IBGDA*，intra-node NVLink 与 inter-node RDMA fusion；combine 在 kernel 内完成 reduction。

Megatron flag: `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (center, right, right, right),
    [*EP*], [*Dispatch (HybridEP / alltoall)*], [*ratio*], [*GB200 / H100*],
    [8],   [391 / 735 µs], [1.88×], [GB200],
    [64],  [675 / 930 µs], [1.38×], [GB200],
    [8],   [661 / 1265 µs], [1.91×], [H100],
    [64],  [4626 / 9164 µs], [1.98×], [H100],
  ),
  kind: table,
  caption: [HybridEP vs NCCL naive alltoall，通信 kernel 时间（Megatron-Core Tech Report Table 7）。H100 上 EP=64 时接近 2× 加速；GB200 上因 NVLink 域扩大，naive 就已经很好，收益变小。],
)

对 256 experts、top-8 的 DeepSeek-V3 类模型，HybridEP 相比 DeepEP 端到端 *+14%*（NVIDIA blog 数据）。

=== Megatron-Core 的 dispatcher 类型选择

Megatron-Core 把上述方案封装成 `--moe-token-dispatcher-type` 的 4 个选项：

#figure(
  table(
    columns: (auto, 1.4fr, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*Dispatcher*], [*适用*], [*Flag*],
    [`allgather`],   [仅 TP、小 EP、大 top-K],           [`--moe-token-dispatcher-type allgather` (默认)],
    [`alltoall`],    [标准 EP > 1],                       [`--moe-token-dispatcher-type alltoall`],
    [`flex` + DeepEP],   [H100/B200 跨节点、fine-grained MoE], [`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`],
    [`flex` + HybridEP], [GB200 NVL72 / MNNVL],               [`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`],
  ),
  kind: table,
  caption: [Megatron-Core MoE token dispatcher 类型。旧的 `alltoall_seq` 从 v0.13 起 deprecated，请忽略。],
)

=== 关键 tricks 小结

+ *Node-Limited Routing*：每 token 最多送 M 个 node。DeepSeek $M=4$，配合 8 卡/节点，等效*扩展到 ~13 个可选 expert*（不是全 256），通信量恒定。
+ *FP8 dispatch*：activation 在 dispatch 前量化到 FP8，通信量 *-50%*，combine 仍用 BF16 保精度。
+ *Warp specialization*：一个 kernel 里不同 warp 各司其职（send / forward / recv），避免各自 launch。
+ *SM carve-out*：早期方案 (DeepEP V1) 占用 20 SM，DeepEP V2 降到 4-6 SM 才让 GEMM 有喘息空间。

== EP × TP × DP 组合

生产训练里 EP 单独用是奢侈。真实配置往往 EP × TP × DP：

*配置示例*: Mixtral 8×7B on 64 × H100

```
world_size = 64
TP = 2      # 切 attention 和 expert 内 GEMM
EP = 8      # 8 个 expert 分布 8 卡
DP = 4      # 4 份数据并行 replica
PP = 1

placement (illustrative):
   TP-group: [rank i, i+1]         for i in 0, 2, 4, ..., 62
   EP-group: 每 EP 组 8 卡各持 1 expert
   DP-group: 跨 EP 组的相同 slot 位置
```

*非 expert 参数* (attention, gate, LN)：TP 内切 + DP 全同步。梯度用标准 AllReduce。

*Expert 参数*：EP 组间*不同 expert*（无需 sync）；DP 组间*同一 expert 的多份 replica*（AllReduce）。

*Forward 通信序列*（一个 MoE 层）：

```
1. Attention:
   - QKV Linear (TP col-parallel)
   - Attention compute (local)
   - Output Linear (TP row-parallel) + AllReduce(TP)

2. Gate:
   - Linear(H, E) (replicated, small)
   - softmax + topk (local)

3. Dispatch:
   - all-to-all across EP-group
   - (packed input arrives)

4. Expert:
   - grouped_gemm(x, W_up) — W_up 沿 I 维 TP-partitioned
   - activation
   - grouped_gemm(h, W_down) — W_down 沿 I 维 TP-partitioned
   - AllReduce(TP) 因 W_down 是 row-parallel

5. Combine:
   - all-to-all across EP-group
   - weight + unpermute
```

*每 MoE 层通信次数*：2 次 AllReduce(TP) + 2 次 all-to-all(EP)。加上 backward 就是 4+4 = 8 次集合通信/层。

== Overlap: 隐藏通信

MoE 训练的核心工程优化：*让通信和计算重叠*。有几个层次：

=== Level 1: All-to-all 与 expert compute overlap

单层内可做的：
- `all_to_all_1` 开始 → 立刻开始处理已到达的 tokens (chunked all-to-all)
- 或：本层 `all_to_all_2`（combine 送回）与*下层* attention overlap，两条 CUDA stream 并行。

```python
# 一层 MoE 内, 2 stream overlap
comm_stream = torch.cuda.Stream()   # 只做通信
comp_stream = torch.cuda.current_stream()

# 1) dispatch a2a 起在 comm_stream
with torch.cuda.stream(comm_stream):
    recv_tokens = torch.empty(...)
    a2a_handle = dist.all_to_all_single(
        recv_tokens, send_tokens,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes =send_counts.tolist(),
        async_op=True,
    )
# 2) 等 dispatch 完成 (但让 comp_stream 通过 event 等待, 不阻塞 host)
a2a_handle.wait()
event_dispatch = torch.cuda.Event()
event_dispatch.record(comm_stream)
comp_stream.wait_event(event_dispatch)

# 3) expert compute 在 comp_stream 上跑
with torch.cuda.stream(comp_stream):
    y = grouped_gemm(recv_tokens, W_up)
    y = F.silu(y)
    y = grouped_gemm(y, W_down)

# 4) combine a2a 起在 comm_stream, 与下一层的 attention 并行
event_compute = torch.cuda.Event()
event_compute.record(comp_stream)
with torch.cuda.stream(comm_stream):
    comm_stream.wait_event(event_compute)
    back_handle = dist.all_to_all_single(
        back_tokens, y,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes =recv_counts.tolist(),
        async_op=True,
    )
# 5) 下一层的 attention 可以在 comp_stream 上*立刻*开始
#    只有需要 back_tokens 的地方才 back_handle.wait()
```

关键点是 CUDA event 而不是 stream.wait_stream —— event 只等特定操作完成，让 stream 之间的依赖尽可能松。

=== Level 2: 跨层 pipeline

$"layer"_l$ 的 `all_to_all_2` 完成的同时，$"layer"_(l+1)$ 的 attention 已经开始（因为 attention 与 MoE 独立）。这需要 async grad accumulation + micro-batch pipelining。

=== Level 3: DeepSeek-V3 DualPipe

DeepSeek-V3 (2024) 展示了最激进的 overlap 策略——把每 PP chunk 拆成 4 个细粒度阶段：

+ Attention
+ All-to-all dispatch
+ MLP (MoE expert 计算)
+ All-to-all combine

再借鉴 ZeroBubble 把 backward 拆成 *B (input grad)* 和 *W (weight grad)*，然后 *双向* 注入 micro-batch (pipeline 两端同时进)。每个 forward chunk 与配对的 backward chunk 在同一时刻用不同 SM 跑——通信 kernel 分走 20 SM 做 IB/NVLink，剩下 112 SM 做 GEMM，*通信被计算完全隐藏*。

*Bubble 对比*（Tech Report Table 2；F、B、W 分别是 forward / input-grad / weight-grad 时间，F&B 是 forward+backward 重叠后的时间）：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*Method*], [*Bubble*], [*每设备参数副本*],
    [1F1B],      [$(P P - 1)(F + B)$],                   [1×],
    [ZB1P],      [$(P P - 1)(F + B - 2 W)$],             [1×],
    [*DualPipe*],[$(P P slash 2 - 1)(F \& B + B - 3 W)$],[*2×*],
  ),
  kind: table,
  caption: [pipeline bubble 系数。$P P = 16$ 时 1F1B 是 15 个 stage，DualPipe 是 7 个——但代价是 *2× 参数副本*（bidirectional 每端各存一份）。activation 也从 $P P$ 增到 $P P + 1$。],
)

*Wall-clock 收益*：DeepSeek 声称"a2a 与 PP 通信近乎零开销"，但没给单一 %。参考 profile-data 里的 Chrome trace，$"comp":"comm" approx 1:1$ 的 setup 下 DualPipe 让 comm 完全隐藏。

*工程代价*：需要 2× 权重、需要自己实现 pipeline schedule、需要与 DeepEP 深度耦合。开源实现在 https://github.com/deepseek-ai/DualPipe。

=== Level 4: Megatron-Core 1F1B FWD-BWD Merged

DualPipe 的 2× 参数副本对内存墙不友好。Megatron-Core 提出一个折中：*不改 pipeline schedule (仍是 1F1B)，但在同一个 iteration 里把相邻 micro-batch 的 FWD 和 BWD 合并*，两条 CUDA stream 交错——一条 compute、一条 comm。

关键 flag：

```bash
--overlap-moe-expert-parallel-comm    # 打开 FWD/BWD merged 调度
--delay-wgrad-compute                 # 把 backward 拆成 B (data grad) + W (weight grad)
                                      # 打破 B/dispatch → B/mlp 的依赖
```

需要环境变量 `CUDA_DEVICE_MAX_CONNECTIONS>1` 允许多 stream。

*收益*：EP 通信占比 30-40% → *< 5%*，overlap ratio 达 *93%*，不需要 2× 权重。

=== Overlap 策略选择

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, center, center, center, left),
    [*策略*], [*额外内存*], [*a2a 隐藏率*], [*代码复杂度*], [*何时用*],
    [naive],                 [1×],  [~0%],   [低], [< 32 卡 debug],
    [Level 1 (2-stream)],    [1×],  [~50%],  [低], [64-256 卡快速上手],
    [Level 2 (跨层 pipeline)], [1×],  [~70%],  [中], [512+ 卡、attention 有余量],
    [DualPipe],              [*2×*],[~95%],  [高], [1000+ 卡且能吃下 2× 权重],
    [1F1B FWD-BWD merged],   [1×],  [~93%],  [中], [Megatron-Core 用户，*推荐*],
  ),
  kind: table,
  caption: [四代 overlap 策略选择。绝大多数场景 Level 4 已经够——DualPipe 的 2× 权重成本很少划算。],
)

#insight[
  Overlap 的收益取决于*通信与计算的比例*。$"comm":"comp" = 1:1$（DS-V3 跨节点）时 overlap 能省近 50% wall-clock；$"comm":"comp" = 1:5$（Mixtral 8×7B 单机）时 overlap 只值几个 %。先 profile 再上 overlap，别为了 overlap 而 overlap。
]

== Wall 3 · Compute Efficiency Wall：kernel 碎片化

第 7 章讲的 grouped GEMM / permute fusion 都是这堵墙的武器。分布式场景又加了两个新问题：

+ *Dropless MoE 的 D2H sync*：`torch.bincount(target_rank)` 是 GPU 上跑但要返回 CPU 用来做 `all_to_all_single` 的 split_sizes——一次 D2H sync 每层每 step 都发生，破坏 CUDA graph capture。
+ *动态 shape*：每卡收到多少 token（$M_"local"$）每步不同，kernel launch 时的 shape 都要重新解析、autotune cache miss。

=== Sync-Free MoE + Device-initiated Grouped GEMM

*核心*：把 shape 信息留在 GPU 上，让 grouped GEMM kernel 从 device memory 读 group sizes，*不做 D2H sync*。配合预分配 upper-bound buffer（按 capacity 上限），整层可以进 CUDA graph。

```python
# 常规做法 (有 D2H sync)
send_counts_gpu = torch.bincount(target_rank, minlength=EP)
send_counts_cpu = send_counts_gpu.tolist()          # <-- D2H sync!
dist.all_to_all_single(recv, send,
                       output_split_sizes=recv_counts_cpu,
                       input_split_sizes =send_counts_cpu, ...)

# Sync-free (Megatron-Core / HybridEP)
# send/recv 用固定 upper-bound buffer, counts 用 device tensor 传给 fused kernel
recv, counts_gpu = hybridep.dispatch(x, target_rank, capacity_per_expert=C)
h = device_grouped_gemm(recv, W_up, counts_gpu)     # kernel 内 gather counts
```

需要与 `--moe-expert-capacity-factor` + `--moe-pad-expert-input-to-capacity` 一起用（换回 capacity + drop 的开销）。

=== CUDA Graph（Partial vs Full）

Dropless MoE 只有 attention + router + moe_preprocess 是静态的，可以*部分* graph：

```bash
--cuda-graph-impl transformer_engine
--cuda-graph-modules attn moe_router moe_preprocess
```

DS-V3 GB200 上报告 *+10% 端到端*，代价额外 *~7 GB* 内存。

*Full graph* 只有在 capacity + pad 之后（shape 静态）才能做，Sync-Free + HybridEP 是解锁 full graph 的关键组合。

=== ECHO：Elastic Cloning for Hot Experts

Megatron-Core Tech Report 提出的负载均衡新方法。观察：即使有 aux-loss-free，某些 batch 里仍会有 1-2 个"过热" expert，导致同 EP 组里其他 rank 干等。ECHO 动态把热 expert *clone 到空闲 rank*，用 bin-packing 最小化 clone 数；backward 时把梯度 reduce 回 home rank。

对 fine-grained MoE 尤其有用——256 experts 场景下 heat map 不均是常态。

=== 三墙武器统一映射

```
Parameter-Compute Mismatch (E >> K)
      │
      ├─ Memory Wall ──► EP + FSDP-Dual + Fine-grained recompute
      │                  Precision-aware optim + FP8 activation
      │
      ├─ Communication Wall ──► DeepEP/HybridEP + Node-Limited Routing
      │                         DualPipe / 1F1B-merged overlap + FP8 dispatch
      │
      └─ Compute Efficiency Wall ──► Grouped GEMM + Router/Permute Fusion
                                     CUDA Graph + Sync-Free + ECHO
      三墙互相暴露：解决内存 → 引入 EP → 暴露通信 → overlap
                  → 需更大 batch → FP8 → kernel 碎片化 → CUDA Graph
```

== NCCL 调优

MoE all-to-all 通信性能极大依赖 NCCL 参数。生产建议：

```bash
export NCCL_ALGO=Tree              # 或 Ring，根据拓扑测试
export NCCL_PROTO=Simple           # 大 message 稳定
export NCCL_MIN_NCHANNELS=8        # 增加并发 channel
export NCCL_NTHREADS=512           # 每 channel 线程数
export NCCL_BUFFSIZE=8388608       # 8MB buffer

# IB tuning
export NCCL_IB_HCA=mlx5_0,mlx5_1   # 双网卡
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22

# Debug
export NCCL_DEBUG=INFO
```

具体值必须*实测*——不同集群拓扑差别巨大。NVIDIA NCCL Tests 可以基准。

== 主要框架

先看 kernel / a2a 层的框架谱系：

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, center, left),
    [*框架*], [*EP*], [*TP+EP*], [*Grouped GEMM*], [*特色*],
    [DeepSpeed-MoE],  [✔], [部分],  [✔],           [早期成熟，与 ZeRO 集成],
    [Megatron-LM MoE],[✔], [✔完整],[✔],           [NVIDIA 官方，工业标准],
    [Tutel],          [✔], [✔],   [✔],           [微软，快速 a2a、动态 capacity],
    [Megablocks],     [✔], [部分], [替代为 sparse],[无 drop，DBRX 用],
    [DeepEP],         [✔], [✔],   [靠 grouped GEMM 库], [DeepSeek 官方 a2a 库，V2 只吃 4-6 SM],
    [HybridEP],       [✔], [✔],   [✔],           [NVIDIA GB200 NVL72 / MNNVL],
  ),
  kind: table,
  caption: [MoE 框架谱系。DeepEP / HybridEP 是 *a2a-only* 库，需要嵌入 Megatron-Core / 自研训练框架里用。],
)

新项目的现实选择：

+ *H100 单机* (< 64 卡)：Megatron-Core MoE + `alltoall` dispatcher，够用
+ *H100/H800 跨节点* (> 128 卡)：Megatron-Core + `flex` + `deepep` backend
+ *GB200 NVL72*：Megatron-Core + `flex` + `hybridep`
+ *完全自研*：参考 DeepSeek-V3 论文 + DeepEP 源码

=== 生产系统对比：Megatron-Core / MegaScale-MoE / DeepSeek-V3

三个当前最完整的公开系统。选一个当模板抄比自己从头搭快得多：

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, left),
    [*维度*], [*Megatron-Core MoE*], [*MegaScale-MoE* (ByteDance)], [*DeepSeek-V3*],
    [并行],
    [任意 TP/EP/PP/DP + Parallel Folding],
    [SP (attention) + EP (expert)；MoE 层限 node 内],
    [PP16 × EP64 × ZeRO-1，*无 TP*],
    [a2a],
    [DeepEP / HybridEP 可插],
    [自研 tile-level fusion + swizzling],
    [DeepEP + warp specialization],
    [Pipeline],
    [1F1B FWD-BWD merged (`--overlap-moe-expert-parallel-comm`)],
    [MegaScale 继承的 DP/PP overlap],
    [*DualPipe* (bidirectional)],
    [精度],
    [BF16 / FP8 blockwise / MXFP8],
    [BF16 + FP8 optional],
    [FP8 E4M3 训练 (Fprop/Dgrad/Wgrad)],
    [路由],
    [aux-loss / seq_aux / global_aux / *aux-loss-free*],
    [aux loss + token drop],
    [*aux-loss-free* + seq-wise 小 α + Node-Limited Routing],
    [Grouped GEMM],
    [TE GroupedLinear / CUTLASS grouped],
    [自研 GroupedGEMM + fuse gather/scatter],
    [DeepGEMM (FP8)],
    [规模验证],
    [Mixtral 8×22B *49.3% MFU*，Qwen2-57B *39.0% MFU* (1024 H100)],
    [352B / 1440 H800 / *1.41M tok/s*, *1.88×* vs Megatron-LM],
    [671B / 2048 H800 / *180K GPU-hrs per T tokens*],
    [开源],
    [✔ (NVIDIA/Megatron-LM)],
    [✘ 论文 only],
    [部分：DeepEP / DualPipe / DeepGEMM 开源，训练主脚本未开],
  ),
  kind: table,
  caption: [三个生产 MoE 训练系统对比。Megatron-Core 优势在特性齐全 + 官方支持；MegaScale-MoE 优势在 intra-op tile fusion；DeepSeek-V3 优势在极致 overlap + FP8 + 大规模验证。],
)

=== 各系统的关键收益 (公开数字)

*MegaScale-MoE*（arXiv:2505.11432，EuroSys '26）核心贡献:

+ 用 *SP + EP* 替代 Megatron 的 TP + TP：SP+EP vs TP+TP *MFU +14.9-32.9%*（6 个 MoE 模型）
+ *Intra-op fusion*：Attention 里 `A2A+GEMM`、`GEMM+A2A`；Expert 里 `AG+scatter+GroupedGEMM`、`GroupedGEMM+gather+RS`——把 tile 级依赖打散，通信+算子时间缩短 *1.2-4.7×*
+ *Adaptive a2a ↔ AG+RS*：top-K > 6 时切换到 AG+RS 更省
+ *Selective Activation Rematerialization*：显存 -45~57%，吞吐几乎不变

*DeepSeek-V3 训练 infra*（arXiv:2412.19437 §3）核心：

+ *DualPipe*：bubble 系数从 $(P P - 1)(F + B)$ 降到 $(P P slash 2 - 1)(F \& B + B - 3 W)$
+ *DeepEP*：IB + NVLink 并行调度 + warp specialization，V2 只吃 4-6 SM
+ *FP8 E4M3 全链路*：三路 GEMM (Fprop/Dgrad/Wgrad) 全用 FP8，val loss error *< 0.25%* vs BF16
+ *Node-Limited Routing* ($M = 4$)：通信量恒定，独立于总 expert 数
+ *aux-loss-free balancing*：$gamma = 0.001$，前 14.3T tokens 用，后 500B tokens 关掉

*Megatron-Core MoE*（arXiv:2603.07685）核心：

+ *Parallel Folding*：打破 EP ≤ DP 约束，Attention 与 MoE 用独立 process group
+ *Fine-grained recompute / offload*：模块级 checkpoint，MLA up-proj / SwiGLU / LN 单独 recompute
+ *EP overlap*：`--overlap-moe-expert-parallel-comm` + `--delay-wgrad-compute`，通信占比 30-40% → *< 5%*
+ *Sync-Free MoE + ECHO*：让 dropless MoE 也能进 CUDA graph；ECHO 处理热 expert bin-packing

== 完整训练配置样例

给两份可直接抄的配置——一份 Mixtral 8×7B（标准 MoE），一份 DeepSeek-V3 类 (fine-grained + FP8)。都用 Megatron-Core flag。

=== A. Mixtral 8×7B on 64 × H100

```bash
# 64 GPU, TP=1 EP=8 PP=4 DP=2, MBS=1 GBS=256 seq=4096
torchrun --nproc-per-node=8 --nnodes=8 \
  --node-rank=$NODE_RANK --master-addr=$MASTER pretrain_gpt.py \
  \
  # --- 模型 ---
  --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 \
  --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 32768 \
  --swiglu --normalization RMSNorm --position-embedding-type rope \
  \
  # --- MoE ---
  --num-experts 8 --moe-router-topk 2 \
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2 \
  --moe-grouped-gemm --moe-permute-fusion --moe-router-fusion \
  --moe-token-dispatcher-type alltoall \
  \
  # --- 并行 ---
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 4 \
  --expert-model-parallel-size 8 \
  --num-layers-per-virtual-pipeline-stage 8 \
  --sequence-parallel \
  --use-distributed-optimizer \
  \
  # --- Overlap ---
  --overlap-grad-reduce --overlap-param-gather \
  --overlap-moe-expert-parallel-comm --delay-wgrad-compute \
  \
  # --- 训练 ---
  --micro-batch-size 1 --global-batch-size 256 \
  --bf16 --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine \
  --train-iters 100000
```

参考 throughput：Megatron-Core v0.9 官方 benchmark 报告约 *468 TFLOPS/GPU* (BF16, 64 GPU, `alltoall` dispatcher)。dense Mistral 7B 同 setup 是 *492 TFLOPS/GPU*——MoE 差 ~5%，主要来自 a2a 未 100% overlap。

=== B. DeepSeek-V3 类 fine-grained MoE (FP8, 256 GPU+)

```bash
# 关键 flags, 省略常规参数

# --- MoE: fine-grained + shared + aux-loss-free ---
--num-experts 256 --moe-shared-expert-intermediate-size 2048 \
--moe-router-topk 8 --moe-router-score-function sigmoid \
--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3 \
--moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-4 \
--moe-router-num-groups 8 --moe-router-topk-limited-devices 4 \
--moe-router-dtype fp32 \

# --- Dispatcher: cross-node ---
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep \
--moe-shared-expert-overlap \

# --- 并行: PP + EP (无 TP) ---
--expert-tensor-parallel-size 1 \
--expert-model-parallel-size 64 \
--pipeline-model-parallel-size 16 \
--num-layers-per-virtual-pipeline-stage 4 \

# --- FP8 ---
--fp8-format e4m3 --fp8-recipe blockwise --fp8-param-gather \
--moe-router-padding-for-fp8 \

# --- Memory ---
--recompute-granularity selective \
--recompute-modules mla_up_proj layernorm moe_act moe mlp core_attn \
--fine-grained-activation-offloading \
--offload-modules expert_fc1 moe_act \
--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 \

# --- Overlap ---
--overlap-moe-expert-parallel-comm --delay-wgrad-compute \

# --- CUDA graph ---
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess \

# --- 稳定 ---
--manual-gc --manual-gc-interval 10
```

环境变量：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=8         # 多 stream overlap
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_MIN_NCHANNELS=8
```

DeepSeek-V3 论文报告的成本：*180K H800-GPU-hours per T tokens* ≈ 3.7 天 @ 2048 GPUs。全 14.8T tokens pretrain *2.664M GPU-hrs*，按 \$2/GPU-hr 估算 *~\$5.3M*。

== 常见坑

*(1) All-to-all 的 send/recv counts 不匹配*

每个 rank 发出的 counts 必须等于对应 rank 收到的 counts。这需要*先做一次 all-to-all 交换 counts*，再做数据 all-to-all。生产实现 (Tutel `all_to_all`) 内部封装了这一步。

*(2) Backward 的 all-to-all*

`torch.distributed.all_to_all_single` 的 autograd 反向自动生成——但要*手动指定 async_op* 或用 Megatron 的 `MoELayer` 封装，否则默认阻塞。

*(3) Checkpoint 保存/加载 的 expert sharding 一致性*

EP=8 训练的 checkpoint 换到 EP=4 加载时，需要*重新 shard*。用 `torch.distributed.checkpoint` 或 Megatron dist ckpt 格式，保存元数据。

*(4) Router 在 TP 组内 replicate*

Gate 是小 tensor，全 rank 都存一份；但*梯度合并*要 AllReduce(TP)。不然不同 TP rank 上 router 参数 drift。

*(5) $M_"local"$ imbalance*

跨 rank 的 tokens 数 (all-to-all 后每卡实际负载) 可能不均——同一步内某卡收 2×，某卡 0.5×。expert 内 grouped GEMM 是这卡本地的，不同卡计算时间不同 → 同步 barrier 拖慢整个 group。缓解：capacity 上限、Expert Choice、Megablocks。

== 面试考点

#interview[
  *Q1*: EP 和 TP 都可以切 expert，为什么 EP 优先？

  A: TP 切 expert 后每卡还是有 $E/("TP")$ 个"部分专家"——本质是 dense 化。EP 让每卡持有*完整专家*的一子集——利用 MoE 的稀疏性*降低单卡权重*，同时通信只对涉及的 token 做（all-to-all），比 TP 的 AllReduce 更符合 MoE 稀疏语义。
]

#interview[
  *Q2*: All-to-all 的通信量为什么与 $E$ 无关？

  A: All-to-all 传的是*每 token 的 hidden vector*，总量 = 每卡 tokens × K × H。K 决定通信量（1 个 token 走多少专家），$E$ 决定专家池大小、但每 token 走的还是 $K$ 个——加多专家不增加通信。这也是 DeepSeek 用 256 experts 却通信开销不变的原因。
]

#interview[
  *Q3*: 层次化 all-to-all 的收益怎么来的？

  A: NVLink 400 GB/s vs IB 50 GB/s。同一 8 卡节点内的通信用 NVLink，跨节点用 IB。层次化把 (8×N)-way all-to-all 拆成 8-way NVLink + N-way IB + 8-way NVLink，等价的 IB 带宽消耗从 $O(N^2 K H)$ 降到 $O(N K H)$。对 100+ 节点场景差 10×。
]

#interview[
  *Q4*: MoE 训练里 aux loss 的梯度怎么在分布式下汇总？

  A: aux loss 每 rank 本地算（gate_probs 本地全）——但 $f_e$ (per-expert token count) 需要跨 DP 组 AllReduce，不然是"per-DP-rank load"不是"global load"。生产实现里这个 AllReduce 常被遗忘，导致 aux loss 训不到位。
]

#interview[
  *Q5*: EP 组内的 grouped GEMM，输入 shape 是什么？

  A: 本地 expert 数 = $E/"EP"$，每个 expert 收到的 tokens 数 = $M_"local,e"$ (all-to-all 之后)。所以 grouped GEMM 输入是 packed $(sum_e M_"local,e", H)$，group_sizes 是 $E/"EP"$ 个 int。所有 EP rank *各跑一份*，彼此独立。
]

#interview[
  *Q6*: DeepSeek-V3 的 fine-grained overlap 为什么能显著加速？

  A: 4 阶段 pipeline (dispatch a2a / attn / mlp / combine a2a) 让 GPU 一直有 work——通信不再是 blocking。传统实现里通信占 forward 15-30%，overlap 后接近 0%。代价是调度复杂度：需要 async collective + 精心排布的 CUDA stream。
]

#interview[
  *Q7*: 单节点 8 卡上 EP=8 vs EP=4+TP=2 哪个好？

  A: 看模型。$E ≥ 8$ 时 EP=8 通信更稀疏 (a2a 只在 EP 组内)。但 EP=8 要求 $"world_size" >= 8$，且 expert 数必须整除 EP；EP=4+TP=2 更灵活、每卡持 2 expert。生产 (Megatron) 允许 hybrid，实测选。
]

#interview[
  *Q8*: MoE 的 pipeline parallel (PP) 有什么特殊？

  A: PP 沿 layer 切与 MoE 内部结构正交。但 MoE 层的 all-to-all 会阻塞 pipeline bubble——上游 PP stage 的 all-to-all 未完成，下游 stage 无法开始。DeepSeek 用了"pipeline 内的 fine-grained overlap"缓解，但复杂度非常高。工业界一般 PP 层数少 (2-4)，让 MoE 主要走 EP。
]
