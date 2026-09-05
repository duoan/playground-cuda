#import "../template.typ": *

= 数值稳定性、精度与常见陷阱

这一章是"实操 debug 手册"——把前几章散落的正确性问题、数值坑集中在一处，加上一些经验教训。写 MoE 代码的过程 = 踩坑的过程；先看到坑，再写代码。

== 分类

MoE 的坑大致分五类：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [*类别*], [*举例*],
    [数值精度],       [softmax fp32、logit overflow、accumulator dtype],
    [边界情况],       [空专家、空 batch、全 masked 行、$K = E$],
    [路由 semantics], [renorm vs softmax(topk)、tie-break、bias 时序],
    [Autograd],       [topk 反向、index_add 确定性、all-to-all backward],
    [分布式],         [count 不匹配、TP router replication、ckpt reshard],
  ),
  kind: table,
)

== 数值精度类

=== Softmax 必须 fp32

第 3 章已详解，这里补一个 fp16 训练*不 upcast* 的具体后果：

假设 $E = 32$，router logits 在训练中期近似 $N(0, 5)$ 分布。fp16 下 $exp(5) approx 148$，$exp(-5) approx 0.0067$，比值 $2 times 10^4$——尚可。但训练早期若某个 logit 由于初始化偏大到 15：

- $exp(15) approx 3.3 times 10^6$ — fp16 max 65504，*直接 inf*
- Sum = inf，除法后 all nan
- 一步 backward 后 gate weight 变 nan
- 整个模型死亡

用 fp32 accumulator：$exp(15) = 3.3 times 10^6$ 是合法 fp32 值，softmax 正常输出约 1 给该 expert（尖锐但有效）。

*诊断*：训练中间若发现 gate weight 出现 nan 或极大值，第一 check：softmax 是不是 fp32？

=== Accumulator 与 GEMM dtype

Grouped GEMM 内部有两个 dtype：

- Input/output: bf16 (或 fp16)
- Accumulator: 应该 fp32

如果 kernel 用 bf16 accumulator（错误配置），大 K 维（如 $I = 14336$）的 dot product 会积累精度误差 $10^(-2)$ 量级——梯度失真。

CUTLASS `GroupedGemm` 默认 fp32 accumulator。自己写 Triton kernel 要显式：

```python
tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)
```

=== 训练 vs 推理的精度切换

生产 pipeline 常见：

- 训练：bf16 参数 + fp32 optimizer + fp32 softmax
- 推理：bf16 参数（或 fp8/int4） + fp16 softmax

推理换 fp16 softmax 的风险：如果训练时 gate logits 学到较大 magnitude（router 熵低），推理 fp16 可能 overflow。ST-MoE 的 z-loss 就是为了训练时约束这一点。

*诊断工具*：训练后 log `max(abs(gate_logits))` per batch，应该 $< 20$。

== 边界情况类

=== 空专家 $M_e = 0$

某 batch 里某专家 0 token。行为要求：

*范式 A*: `if token_ids.numel() == 0: continue` — 必须。

*范式 B*: grouped GEMM 的 `group_sizes[e] = 0` 必须被 kernel 正确跳过。测试 fixture：手动构造 `expert_indices` 让某个 expert 从未出现，check `torch._grouped_mm` 输出正确。

*Backward*: 空专家的 $partial L / partial W_e = 0$（无贡献，无梯度）。但*optimizer step 仍要执行*—— AdamW 会给 momentum 一个小 decay。不要跳过 optimizer step。

=== 空 batch $N = 0$

不太常见（batch size 通常大于 1），但 GAN/RL 场景可能。处理：

```python
if hidden_states.shape[0] == 0:
    return hidden_states, torch.empty(0, self.num_experts,
                                       device=hidden_states.device)
```

否则 `torch.topk` 对 (0, E) 的输入行为 undefined。

=== $K = E$ (全激活)

MoE 退化为 dense FFN 但计算重复。合法但性能差。用户可能出于 debug 目的这么做，代码不该报错。测试建议加：

```python
def test_moe_k_equals_e():
    moe = MoE(H=8, I=16, E=4, K=4)  # K = E
    y, _ = moe(x)
    # 应该数值等价于 sum of all expert outputs weighted by softmax
```

=== 全 masked 行

Attention 场景，某 token 的所有 attention 被 mask 掉后 (`X = -inf`)，gate 输入是全 `-inf`：

$ ell = W_g dot (-oo) → -oo $
$ p = "softmax"(-oo, -oo, ..., -oo) → "nan" $

必须在 gate 之前检查/替换全 mask 行：

```python
mask = hidden_states.isfinite().all(dim=-1)  # (N,)
safe_hidden = torch.where(mask.unsqueeze(-1), hidden_states, 0.0)
gate_logits = self.gate(safe_hidden)
# 对 masked token 输出 0 而非计算 MoE
```

生产实现要更 robust，用 `attention_mask` 显式传入。

== 路由 semantics 类

=== Renorm 前后的 activation

前几章讲过两种 router 变体，还有一个隐含的坑：`expert_weights` 传给后续计算的*精度*。

如果 renorm 用 fp32 但后面 mul 用 bf16：

```python
weights_fp32 = weights_fp32 / weights_fp32.sum(dim=-1, keepdim=True)
weights = weights_fp32.to(bf16)
# 后面: expert_output * weights
```

`weights` 的 bf16 表示可能让 `sum ≠ 1.0` 精确成立——一个 token 的 K 个权重和是 0.998 或 1.002。数学上小误差，*但* 训练稳定性：如果整个模型对 "gate weight 精确 sum=1" 有暗含依赖（很多 aux loss 计算），会有几个 percent 的 loss 波动。

*Fix*: 保持 renorm 后的 weights 在 fp32，直到最后 output cast 时才降回。

=== Tie-break 行为

`torch.topk` 处理 tie 时选*最先出现的*——但 CUDA 上多线程/多 block 的 tie 处理是*deterministic*但*platform-specific*。CPU vs GPU 结果可能不同：

```python
# CPU
torch.topk(torch.tensor([1.0, 1.0, 1.0, 0.5]), k=2)
# → values=[1, 1], indices=[0, 1]

# GPU 有时是
# → values=[1, 1], indices=[0, 2]  (depending on kernel)
```

对训练不重要（初始化会打破 tie），但对 unit test 是问题——用 GPU-generated `expert_indices` 和 CPU reference 对比时要跳过 tie 情况。

=== DeepSeek Bias 更新时机

Bias tuning 每 step 更新 $b_e$——放*什么时候*？

- Forward 前？会污染 forward 用的 bias（本 step 用的和本 step 更新的不同）
- Forward 后 / optimizer.step 后？OK
- 每 100 step 一次？可行但 lag 大

DeepSeek 官方实现是"forward 后立刻用本 step 的负载统计更新"，让下一 step 的 forward 用新 bias。

要注意：多 rank 情况下*统计要 AllReduce 一次*——本 rank 只知道本 rank 的负载。

== Autograd 类

=== Topk 反向的意义

第 3 章讲过：topk 反向只把梯度给 top-K 位置。这不是 bug，但要理解：

```python
w, i = torch.topk(x, k=2)   # x: (N, E), w: (N, 2), i: (N, 2)
loss = w.sum()
loss.backward()
# x.grad 只有 top-2 位置有值，其他 = 0
```

*陷阱*: 如果你在 topk 输出上再做 gather，然后 backward，autograd 会给出正确梯度——只是"未选中位置"的梯度是 0。不要以为 gradient checker fail 了。

=== `index_add_` 的确定性

第 4 章"一个隐藏的正确性坑"讲过：`index_add_` 在 GPU 上默认非确定性。对训练可复现性有影响，但不影响正确性。

```python
# 打开确定性
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

代价 5-15% 慢。生产训练一般*不*开——loss curve 差异远小于训练噪声。

=== All-to-all 的 autograd

`torch.distributed.all_to_all_single` 支持 autograd——它的反向自动是 all_to_all with reversed counts。但要求：

- Send 和 recv counts 是 static (不依赖于 tensor 值)
- Async_op=False (aka blocking mode, 默认)

Async 版本 (`async_op=True`) 返回 work handle，autograd*不会自动等待*——需要手动 `.wait()`。生产实现用 Megatron 的 `MoELayer` 会封装好。

== 分布式类

=== Send/recv counts 不匹配

Rank A 说要发 100 token 给 rank B，但 rank B 说要收 90 token——mismatch，all-to-all 死锁或数据 corruption。

*Fix*: 每个 all-to-all 之前先做一次 counts exchange：

```python
send_counts = compute_locally(...)  # (world_size,)
recv_counts = torch.zeros_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)   # 交换 counts
# 现在 recv_counts 是 "别 rank 要给我发的量"
data = dist.all_to_all_single_uneven(data, recv_counts, send_counts)
```

Tutel 和 Megatron 内部封装了这个流程。

=== TP 组内 Router replication

Gate weight `W_g: (E, H)` 很小，全 TP rank 都存一份。但 forward 里每个 TP rank 各算一次——*结果应该相同* (deterministic if same input)。

问题：$H$ 维在 TP 里被切了，attention output 出来是*每 TP rank 各持一份 replicated hidden*吗？

- Megatron TP 里 output linear 是 row-parallel，*最后有 AllReduce(TP)*——之后 hidden 在 TP 内 replicated。
- 但如果 sequence parallel (SP) 开启，hidden 在 TP 内是*沿 seq 切的*——不 replicated。

Router *需要* full hidden。所以在 SP 场景下 gate 前要 AllGather(TP) 恢复完整 hidden。

生产实现里这个 AllGather 是隐藏的：Megatron MoE 层内部自动插入，用户不感知。

=== Checkpoint reshard

训练时 EP=8，恢复时想改 EP=4（比如换更少节点）——需要重新 shard expert weights：

```
原 EP=8: rank 0 持 W_up[0], rank 1 持 W_up[1], ..., rank 7 持 W_up[7]
新 EP=4: rank 0 持 W_up[0:2], rank 1 持 W_up[2:4], rank 2 持 W_up[4:6], rank 3 持 W_up[6:8]
```

推荐用 `torch.distributed.checkpoint` (2.x+) 或 Megatron 的 dist ckpt 格式——它们把 sharding 元数据编码进 checkpoint，加载时自动 reshard。手写文件 IO 极易出错。

== 训练稳定性与调试

=== Loss 突然 NaN 的排查顺序

1. *Router 输出*: `torch.isfinite(gate_probs).all()` — 如果 nan，softmax 精度问题
2. *Expert 输入*: `torch.isfinite(packed_input).all()` — attention 出的 nan 传下来的
3. *Expert 输出*: 同上，通常是权重梯度爆炸
4. *Aux loss*: 有时 aux loss 系数太大压制主 loss，check ratio
5. *All-to-all*: 通信中数据 corruption (rare，NCCL 稳定)

日志要 log 每层 `gate_logits.abs().max()` 和 `packed_input.abs().max()`——nan 出现前通常有几百 step 的"数值爬升"预警。

=== Reproducibility

MoE 完全 reproducible 训练 (bit-exact) 需要：

- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG` 环境变量
- 固定 all-to-all 顺序 (NCCL 有随机性)
- 固定 grouped GEMM tile scheduler

工业界一般不追求 bit-exact——只要 loss curve 在噪声范围内一致就 OK。

=== 训练 vs 推理的路由差异

*训练*: 需要 aux loss (监督) + drop token (保证 GEMM shape)
*推理*: 不需要 aux loss；drop 会伤害精度，用大 capacity 或 Megablocks

如果生产 pipeline 是 "训练时 drop、推理时不 drop"，会有 train/eval mismatch。缓解：训练用 Megablocks 或大 capacity，保证 drop 率 $< 0.1%$。

DeepSeek-V3 训练/推理都 no-drop，一致性好。

== 常见坑速查表

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*症状*], [*可能原因*], [*Fix*],
    [Loss 训到一半 NaN], [Router softmax 非 fp32], [显式 dtype=torch.float32],
    [某专家不学习], [Router collapse 早期], [aux loss ↑, init std ↓],
    [Loss 曲线抖动], [aux loss 系数过大], [降到 0.005-0.01],
    [Gate logits 越来越大], [无 z-loss], [加 z-loss $beta=0.001$],
    [Grouped GEMM nan], [空专家 $M_e=0$], [跳过或用支持的 kernel],
    [推理精度掉], [训练 drop 太多], [提高 capacity 或 Megablocks],
    [多机训练卡住], [a2a counts mismatch], [先 exchange counts],
    [Reshard 后 loss 突变], [Ckpt 格式不对], [用 dist ckpt API],
    [多次跑 loss 不一致], [`index_add` 非确定], [`use_deterministic_algorithms`],
    [Gate 梯度爆炸], [batch 全走 1 个专家], [提升 aux loss 或降 LR],
  ),
  kind: table,
)

== 面试考点

#interview[
  *Q1*: Softmax 不用 fp32 会出什么问题？给一个具体数值例子。

  A: fp16 下 $exp(11.1) approx 6.6 times 10^4$，接近 fp16 上限 65504。如果 gate logit 到 15，$exp(15) approx 3.3 times 10^6$ 直接溢出为 inf，softmax 输出 nan。fp32 支持到 $exp(88) approx 1.6 times 10^{38}$，安全。
]

#interview[
  *Q2*: `torch.topk` 处理相等值时的顺序保证？

  A: CPU 上稳定 (按位置排)；GPU 上 deterministic 但 kernel-specific，不同 PyTorch 版本可能不同。要 reproducible 就在 CPU 上做 topk，或加小 noise 打破 tie。
]

#interview[
  *Q3*: 训练时 drop token，推理时不 drop，会有什么问题？

  A: Train/eval mismatch — 训练时 router 学到 "选中但被 drop" 的 token 无梯度信号，推理时该 token 走完全部专家，行为分布不同。缓解：训练用大 capacity ($≥ 2$) 或 Megablocks 减少 drop；或 evaluate loss 时也开相同 drop 策略。
]

#interview[
  *Q4*: 分布式 MoE 里，aux loss 的 $f_e$ 需要跨 rank 同步吗？

  A: 需要。$f_e$ 是"这个 batch (globally) 每 expert 的负载比例"——如果每 rank 各自算，得到的是"本 rank 的负载"，训练早期本 rank 恰好 collapse 到 expert 0，其他 rank collapse 到 expert 1，local aux loss 都 0，*但 global 不均衡*。必须 AllReduce($f$) across DP-group。
]

#interview[
  *Q5*: 为什么生产 MoE 训练不追求 bit-exact reproducibility？

  A: `index_add_`、`all_to_all` 的非确定性会引入低位 bit 差异；bit-exact 需要牺牲 5-15% 性能。工业界只要 loss curve 在噪声范围内一致（$1sigma$ 内）就够。学术论文报告 3 seed avg + std 即可。
]

#interview[
  *Q6*: Router 参数 (gate weight) 在 EP 组内是 replicated 还是 sharded？

  A: Replicated (小)。每 EP rank 都算完整 router 输出，然后基于 expert_indices 决定 all-to-all 目标。梯度也 replicated——但通过 DP AllReduce 保持一致 (跨 DP 组时)。
]

#interview[
  *Q7*: 遇到 loss NaN，检查顺序是什么？

  A: (1) `gate_probs.isfinite()`；(2) input to expert `.isfinite()`；(3) `W_expert.grad.abs().max()` 是否爆炸；(4) aux loss 数值；(5) NCCL log 有无 corruption。90% 是 (1)——softmax 精度问题。
]

#interview[
  *Q8*: EP=8 训练完的 checkpoint 换到 EP=4 加载，实操怎么做？

  A: 用 `torch.distributed.checkpoint.save/load` (PyTorch 2.x) 或 Megatron 的 dist ckpt。它们保存"每个 expert 的完整 tensor + placement 元数据"，加载时按目标 EP 自动 reshard。手写 file IO 极易出错（记住 expert 排列顺序、参数字典键名一致）。
]
