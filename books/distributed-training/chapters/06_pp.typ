#import "../template.typ": *

= Pipeline Parallel：从零造一个出来

PP 的*理论*只有三个公式，一页纸讲得完：bubble 是 $(P-1)/(m+P-1)$，1F1B 省显存，interleaved 把 bubble 除以 $V$。真正的难点全在实现，而实现里最关键的那件事，任何一张时空图都不会告诉你：

#warn[
  *autograd 不跨进程*。`loss` 只在最后一个 stage 上存在，stage 0 上调 `loss.backward()` 是不可能的——计算图在 `send`/`recv` 那里就断了。所以 PP 的核心不是"怎么排 F 和 B"，而是*手工把链式法则在网络边界上接起来*。排程只是接好之后的调度问题。
]

这一章按"造轮子"的顺序走：先接 autograd 边界，再把 schedule 变成数据，再解决 P2P 的死锁与形状协商，再拆 B/W，最后接进真实训练循环。每一步都有可运行、可验证的代码：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left),
    [*文件*], [*内容*],
    [`common/pipeline.py`], [引擎本体：4 个 schedule 生成器、B/W 拆分、P2P 层、约 40 行的解释器],
    [`21_pp_from_scratch.py`], [autograd 边界、真实复现 P2P 死锁、形状协商、四种 schedule 对齐单卡参考],
    [`22_pp_schedule_sim.py`], [依赖图模拟：实测 bubble 对闭式公式、ASCII 时空图、activation 峰值],
    [`23_pp_integration.py`], [rank 排布、按耗时分层、首尾权重共享、loss 缩放与 DP 同步时机],
  ),
  kind: table,
  caption: [本章代码。四种 schedule 的梯度与单进程参考的最大误差是 `0.00e+00`——不是"接近"，是逐位相同。],
) <tab-pp-code>

== 第一步：在进程边界上手工接链式法则

单卡上 `loss.backward()` 之所以能工作，是因为 autograd 有一张从 loss 一直连回第一层权重的图。`send`/`recv` 是普通的数据拷贝，不是 autograd op，所以图在 stage 边界断成 $P$ 段。你要做的是给每一段单独提供"起点"。

协议只有四行，但每一行都有讲究：

```python
# forward
x = recv(...).detach().requires_grad_(True)   # 造一个新的 leaf
y = stage(x)
send(y)

# backward
gy = recv(...)                     # 下游告诉你 dL/dy
torch.autograd.backward(y, gy)     # 同时填好 W.grad 和 x.grad
send(x.grad)                       # 把 dL/dx 交给上游
```

三个容易写错的地方：

+ *`.detach().requires_grad_(True)` 不是防御性代码*。recv 出来的 buffer 没有 `grad_fn`、也没有 `.grad`。不把它变成 leaf，backward 之后 `x.grad` 是 `None`，你就没有任何东西可以往上游发——而且不会报错，只是上游所有层的梯度恒为零。

+ *用 `torch.autograd.backward(y, gy)`，不是 `y.backward()`*。只有最后一个 stage 手上有标量 loss；其他每个 stage 都是"拿到一个梯度，从这里继续往回传"。`y.backward()` 对非标量会直接抛异常。

+ *第一个 stage 的输入不要 `requires_grad_`*。stage 0 拿到的是真实数据而不是 activation，没有上游可以接收梯度。更实际的原因：它往往是 int64 的 token id，PyTorch 会直接报 `only Tensors of floating point dtype can require gradients`。我自己写引擎时就踩了这个——在纯 float 的玩具模型上完全看不出来，一接上 embedding 就炸。

各 stage 手上有什么、没有什么，是这套协议的全部信息：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, center),
    [], [*stage 0*], [*中间 stage*], [*最后 stage*],
    [输入], [真实数据], [recv 的 activation], [recv 的 activation],
    [输入要 grad 吗], [不要], [要（新 leaf）], [要（新 leaf）],
    [标量 loss], [没有], [没有], [有],
    [backward 起点], [收到的 `gy`], [收到的 `gy`], [`loss`（无需 `gy`）],
    [往上游发], [不发], [`x.grad`], [`x.grad`],
    [需要 label 吗], [不需要], [不需要], [需要],
  ),
  kind: table,
  caption: [PP 的每个 stage 都在跑同一段代码，但边界条件三种。绝大多数"梯度全零"或"loss 不降"的 PP bug 都是某一格填错了。],
) <tab-pp-boundary>

`21_pp_from_scratch.py` 的 Part 1 把这件事做成了可验证的最小例子：两卡切开 $f(x) = W_2 dot "gelu"(W_1 x)$，stage 0 *从未见过 loss*，但它算出的 $partial L\/partial W_1$ 与单进程完全一致。

#insight[
  面试里被问"PP 怎么实现"，从这里开始讲，而不是从时空图开始讲。能说清"我要手工构造 leaf、手工提供 `grad_outputs`、把 `x.grad` 当消息发出去"，就说明你真的写过；只会画 F/B 方格图的人讲不到这一层。
]

== 第二步：把 schedule 变成数据

朴素写法是给每种 schedule 写一套 for 循环，于是 GPipe、1F1B、interleaved、zero-bubble 就是四份互相抄的代码，改一个 bug 要改四处。

更好的做法——也是 `torch.distributed.pipelining` 和 Megatron-Core 背后的结构——是让 schedule 变成一个*指令列表*，然后写一个解释器：

```python
@dataclass(frozen=True)
class Instr:
    op: str        # F / B / BI / BW
    mb: int        # micro-batch id
    chunk: int = 0 # 这张卡上的第几个 virtual chunk
```

四种 schedule 于是各自只有几行。1F1B 是最典型的：

```python
def sched_1f1b(rank, P, m):
    n_warm = min(P - 1 - rank, m)
    out = [Instr("F", i) for i in range(n_warm)]           # warm-up
    for i in range(m - n_warm):                            # steady
        out += [Instr("F", n_warm + i), Instr("B", i)]
    out += [Instr("B", i) for i in range(m - n_warm, m)]    # cool-down
    return out
```

`n_warm = P - 1 - rank` 这一行就是 1F1B 的全部内容：stage 0 要先跑 $P-1$ 个 forward 才等到第一个梯度回来，最后一个 stage 只跑 1 个。

打印出来（$P=4, m=8$）：

```
stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
stage 1: F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
stage 2: F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
stage 3: F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
```

这样做的好处不只是省代码。schedule 成了数据之后，你可以在*不跑任何 GEMM* 的前提下把它打印、diff、灌进模拟器算 bubble、检查它是否会死锁。`22_pp_schedule_sim.py` 全部指标都是从这些列表算出来的。

=== interleaved 的 (chunk, micro-batch) 映射

interleaved 1F1B 唯一复杂的地方是"第 $k$ 步该算哪个 chunk 的哪个 micro-batch"。Megatron 的映射是：

#formula[
  $"chunk"(k) = floor((k mod P V) \/ P), quad "mb"(k) = floor(k \/ P V) dot P + (k mod P)$
]

backward 方向把 chunk 反过来：$V - 1 - "chunk"(k)$。全局 chunk 编号按 $g = v dot P + "rank"$ 排布，也就是说 rank $r$ 持有 chunk $r, r+P, r+2P, dots$，micro-batch 在环上绕 $V$ 圈。

#note[
  这两个映射式*与 rank 无关*——每张卡算出来的 $("chunk", "mb")$ 序列完全相同，只是各自处在 warm-up/steady/cool-down 的不同位置。这正是为什么各 rank 的 send 和 recv 能自动对上而不需要任何全局协调。想清楚这一点，interleaved 就不神秘了。

  warm-up 深度是 $2(P-1-r) + (V-1)P$：前一项是普通 1F1B 的往返，后一项是"micro-batch 还要再绕 $V-1$ 圈才回到你手上"。
]

=== 解释器与 FIFO 纪律

解释器本体就是一个 dispatch，加上*每个 chunk 一个 FIFO 队列*：

```python
def run(self, schedule, inputs, labels):
    for ins in schedule:
        if   ins.op == "F":  self._forward(ins.mb, ins.chunk, inputs, labels)
        elif ins.op == "B":  self._backward(ins.mb, ins.chunk, weight=True)
        elif ins.op == "BI": self._backward(ins.mb, ins.chunk, weight=False)
        elif ins.op == "BW": self._weight(ins.mb, ins.chunk)
    self._drain_sends()
```

forward 时把 `(mb, x_leaf, y)` push 进队尾，backward 时从队首 popleft。FIFO 不是随手选的：本章四种 schedule 对同一个 chunk 都是*按 forward 的顺序* backward 的，而 P2P 的配对也依赖这个顺序。引擎里留了一行断言，它能在几十行之内定位一个写错的 schedule：

```python
mb_q, x, y = self.live[v].popleft()
assert mb_q == mb, f"FIFO violated on chunk {v}: expected {mb_q}, got {mb}"
```

顺带一个可以直接背的结论，四种 schedule 通用：

#formula[
  activation 峰值 $=$ 该 stage 在*第一次 backward 之前*跑掉的 forward 个数
]

代入即得：GPipe 是 $m$（它根本没有 steady 阶段），1F1B 是 $P - "rank"$，interleaved 是 $2(P-1-r) + (V-1)P + 1$。文件 `22` 模拟出来的值与文件 `21` 实测的值逐位相同：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, left),
    [*schedule*], [*stage 0 峰值*], [*stage 3 峰值*], [*P2P 消息数（每 stage）*],
    [GPipe],       [8],  [8], [16 / 32 / 32 / 16],
    [1F1B],        [4],  [1], [16 / 32 / 32 / 16],
    [zero-bubble], [4],  [1], [16 / 32 / 32 / 16（另有 6 份 W 张量滞留）],
    [interleaved ($V=2$)], [11], [5], [48 / 64 / 64 / 48],
  ),
  kind: table,
  caption: [$P=4, m=8$ 实测。interleaved 的 P2P 消息数是 3×，这是它换 bubble 付的真实代价；显存也比 1F1B 高而不是低。],
) <tab-pp-peak>

== 第三步：P2P 的三个工程问题

=== 死锁：1F1B 稳态一定会撞上

这是最经典的 PP bug。1F1B 稳态下，stage $s$ 正要把 activation 往*下*发，而 stage $s+1$ 正要把 gradient 往*上*发。如果两边都写成"先 send 再 recv"：

```
stage s   : send(act -> s+1)   阻塞，等 s+1 来收
stage s+1 : send(grad -> s)    阻塞，等 s 来收
```

两边都卡在 send 里，等着对方去执行那个永远不会到达的 recv。`21_pp_from_scratch.py` 的 Part 2 用一个 2 秒超时的进程组把它*真实复现*出来（不是画图说明），然后对比三种解法：

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*做法*], [*为什么能解*], [*代价 / 风险*],
    [奇偶定序],
    [偶数 rank 先 send 后 recv，奇数 rank 反过来，用 rank 奇偶性打破对称],
    [脆。每加一种消息都要重新排一遍；interleaved 打破了"每步恰好一收一发"的前提，这套就不成立了],
    [`batch_isend_irecv`],
    [把这一步所有 send/recv 先全部 post，再一起 wait，顺序约束直接消失],
    [真实框架的做法。还顺带让两个方向的传输在线路上重叠],
    [send 永不阻塞],
    [send 一律 post 后把 handle 存起来，只有 recv 才阻塞。循环等待需要至少有一个 rank 卡在 send 上，所以环不可能形成],
    [本章引擎的做法。要自己管 buffer 生命周期，且要在 step 结束前 drain],
  ),
  kind: table,
  caption: [三种解法都对，选一种并且知道为什么。第三种最容易讲清楚——"没人阻塞在 send 上"是一个可以一句话证明的不变量。],
) <tab-pp-deadlock>

Megatron 里第二种做法对应的函数就叫 `send_forward_recv_backward`（以及 `send_backward_recv_forward`）——看到这个函数名，你就知道它存在的唯一理由是躲开上面那个环。

#warn[
  一个只有踩过才知道的细节：gloo 的 P2P 超时会*关闭整条 pair*，超时之后这个进程组就不能再用了，后续操作直接报 `Application timeout caused pair closure`。生产上也是这个性质——P2P 超时不会给你留一个还能重试的进程组，只能重建。所以文件 `21` 里死锁演示用的是独立的短超时进程组，后面三种解法各用干净的组。
]

还有一个 buffer 生命周期的坑：`isend` 立刻返回，但它*持有*那块内存直到传输完成。把待发的 activation 原地改掉，或者让它被回收，收到的就是垃圾。引擎里这一行的 `clone()` 不是随手加的：

```python
def _send(self, x, dst, tag):
    buf = x.detach().contiguous().clone()      # isend 期间必须一直有效
    work = dist.isend(buf, dst=dst, tag=tag, group=self.group)
    self.pending_sends.append((work, buf))     # 连 buf 一起存，否则会被 GC
```

=== 形状协商：接收方得先分配才能接收

`dist.recv` 需要一个已经分配好的 tensor 来写入，所以*接收方必须在数据到达之前就知道 shape 和 dtype*。固定形状时硬编码就行，一旦上了 packing 或变长序列就不行了，而猜错的后果是静默截断或者挂住。

做法是先发一个小 header：

```python
def send_meta(x, dst, tag, group):
    hdr = torch.tensor([x.dim(), DTYPE_CODE[x.dtype]] + list(x.shape) + pad,
                       dtype=torch.int64)
    dist.send(hdr, dst=dst, tag=tag, group=group)
```

代价是每个 tensor 多一条极小的消息，所以框架都把它做成开关：Megatron 的 `--variable-seq-lengths` 默认*关闭*。引擎同理——传 `act_shape` 就跳过协商，传 `None` 就每次协商。

=== tag 还是位置配对

同一对 rank 之间怎么区分"这条是 activation"和"这条是 gradient"？

多数情况下不用区分：activation 走 $r arrow.r r+1$，gradient 走 $r+1 arrow.r r$，方向不同即 `(src, dst)` 不同，天然不会混。但 $V > 1$ 且 $P = 2$ 时会退化——rank 1 到 rank 0 这条链上同时跑着 activation（chunk 1 $arrow.r$ 2）和 gradient（chunk 3 $arrow.r$ 2、chunk 1 $arrow.r$ 0），单靠顺序区分不了。

两条出路：

- *tag*：$"tag" = 2 (m dot N_g + g) + "dir"$，其中 $N_g$ 是全局 chunk 总数，简单直接。但 *NCCL 忽略 tag*，只有 gloo 支持——这就是本章 demo 把 backend 钉在 gloo 的原因。
- *位置配对*：每一步把该步所有 op 塞进一个 `batch_isend_irecv`，靠"两边步骤结构相同"来保证第 $j$ 个 op 对上第 $j$ 个 op。这是真实框架的做法，也是它们不需要 tag 的原因。

== 第四步：把 backward 拆成 B 和 W

zero-bubble 的前提是把一次 backward 拆成两半：

- *B（input-grad）*：$partial L\/partial X$，*会阻塞*上游 stage，必须尽早算
- *W（weight-grad）*：$partial L\/partial W$，*不阻塞任何人*，可以塞到任意空隙里

问题是 PyTorch 的 `backward()` 一次遍历就把两个都算了。怎么拆？

天真的做法是调两次 `torch.autograd.grad`，一次求 `inputs=[x]`、一次求 `inputs=params`——但那是把整个反向图遍历了两遍，白算一倍。

正确做法是写一个*拒绝干完一半活*的 autograd Function：

```python
class _LinearSplitBW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tape):
        ctx.save_for_backward(x, w)
        ctx.tape = tape
        return x @ w.t()

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        ctx.tape.push(w, gy.detach(), x.detach())   # W：记账，先不算
        return gy @ w, None, None                   # B：现在就算
```

于是 `BI` 就是普通的 `torch.autograd.backward(y, gy)`（它会把整段的 input-grad 算完，并把所有 W 的活记在 tape 上），`BW` 就是把 tape 排空：

```python
for w, gy, x in work:
    gw = gy.reshape(-1, gy.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
    w.grad = gw if w.grad is None else w.grad + gw
```

没有 autograd 补丁，没有 monkey patch，两个 GEMM 各算一次。

#insight[
  代价在显存，而且账要算清楚：`BI` 释放了 activation，但 tape 上还压着 $x$ 和 $g_y$ 等着 `BW` 用。$P=4, m=8$ 时 stage 0 的 activation 峰值和 1F1B 一样是 4，但另外滞留 6 份 W 张量（文件 `22` 的 Part 4 实测）。所以 zero-bubble 是*用显存换 bubble*，不是白拿。
]

== 第五步：接进真实训练循环

到这里引擎能跑了，但离能训练还差五件事，而这五件事一张时空图都不会画。全部在 `23_pp_integration.py` 里，并且端到端验证过：PP × DP + 权重共享的梯度必须等于单进程跑完整 global batch。

=== rank 排布：你的邻居不是 rank ± 1

Megatron 默认排布是 TP 变化最快、PP 最慢：

#formula[
  $"global_rank" = "pp" dot (D T) + "dp" dot T + "tp"$
]

这个顺序不是随便定的。TP 每层要来回搬两次完整 activation，所以 TP 组必须落在一个 NVLink 域里，也就是*连续的 global rank*；PP 每个 stage 边界只搬一次 activation，所以给它最大的 stride，让它去跨节点。

$"world"=8, T=2, D=2, P=2$：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (center, center, center, center, center, center, center),
    [*rank*], [*tp*], [*dp*], [*pp*], [*node*], [*TP 组*], [*PP 组*],
    [0], [0], [0], [0], [0], [0,1], [0,4],
    [1], [1], [0], [0], [0], [0,1], [1,5],
    [2], [0], [1], [0], [0], [2,3], [2,6],
    [3], [1], [1], [0], [0], [2,3], [3,7],
    [4], [0], [0], [1], [1], [4,5], [0,4],
    [5], [1], [0], [1], [1], [4,5], [1,5],
    [6], [0], [1], [1], [1], [6,7], [2,6],
    [7], [1], [1], [1], [1], [6,7], [3,7],
  ),
  kind: table,
  caption: [TP 组永远是连续 rank，不跨节点；PP 组 stride $= D T = 4$，正好跨节点。],
) <tab-pp-layout>

#warn[
  *手写 PP 最常见的 bug*：pipeline 邻居是 rank $plus.minus D T$，不是 rank $plus.minus 1$。而且 `dist.send/recv` 即使传了子进程组，`src`/`dst` 要的仍然是*全局* rank，不是组内序号。

  我自己的引擎就中过这一枪：文件 `21` 里 PP 组恰好是整个 world，组内序号与全局 rank 相同，四种 schedule 全部通过；一到文件 `23` 开了 DP=2、PP 组变成 `[1, 3]`，立刻 `ValueError: Global rank 0 is not part of group`。引擎里现在有一层显式翻译：

  ```python
  def _peer(self, pipe_rank):
      if self.group is None:
          return pipe_rank
      return dist.get_global_rank(self.group, pipe_rank)
  ```

  更阴的情况是它*不报错*：如果错算出的那个 rank 恰好存在，你就在往一张持有相同层的卡发 activation，表现是挂住而不是异常。
]

=== 分层：层数相等不等于工作量相等

32 层、$h = 4096$、vocab 128K 的模型，按模块相对耗时（一个 transformer block $= 1.0$）：embedding 约 $0.5$（查表很便宜），lm_head 约 $5.0$——head 的 GEMM 本身就是 $h V \/ 12 h^2 approx 2.6$ 个 block，再加上在 $(B, S, 128"K")$ logits 上做 cross-entropy。

按层数平均切 4 段：负载 $[7.5, 9.0, 9.0, 12.0]$，$max\/"mean" = 1.28$×。按耗时贪心切：$[9.5, 10.0, 9.0, 9.0]$，$1.07$×。

#insight[
  流水线跑在最慢那一段的速度上，所以 $1.28$× 的不均衡是*每个 micro-batch 都要付*的 $1.28$× 减速——它和 bubble 相乘，而不是被 bubble 藏起来。

  常见解法：给首尾 stage 少分几个 block，或者把 loss 单独放一个 virtual chunk。Megatron 有手动指定分层的 flag。
]

还有一个约束容易忽略：*不能先建完整模型再切*。70B 以上没有任何一张卡装得下，每张卡只能构造自己那几层——也就是说分层方案必须在*任何权重被分配之前*就定下来，只能靠一张成本表，不能靠跑一遍 profile。

=== 权重共享：embedding 和 lm_head 落在不同 stage 上

embedding 在 stage 0，lm_head 在 stage $P-1$，两者 tie 在一起——但它们在*不同的进程*上。每一端只算到梯度的一部分：stage 0 从查表那条路径拿到一部分，最后一个 stage 从投影那条路径拿到另一部分。

所以首尾两个 rank 要单独组一个进程组（Megatron 就叫 embedding group），backward 之后在组内 all-reduce：

```python
emb_group = dist.new_group([first_pp_rank, last_pp_rank])   # 每个 dp 坐标一个
...
dist.all_reduce(emb.grad, op=SUM, group=emb_group)
```

文件 `23` 实测：每一端各自只持有 tied 梯度的一部分（该配置下约 69%），all-reduce 之后两端都拿到完整的。

#warn[
  漏掉这个 all-reduce *不会崩*。两份 embedding 副本会在几个 step 内漂开，于是模型用一张表编码、用另一张表解码。现象只是 loss 不再下降——这类 bug 极难查，因为所有 assert 都过、所有 shape 都对。
]

两个反直觉的补充，都是文件 `23` 里验证过的：

+ *两次 reduce 的顺序无关紧要*。embedding group 上的 SUM 和 DP 组上的 AVERAGE 作用在不相交的 rank 集合上，两者都是线性的，可交换。真正要求的是它们都在 clip 之前完成。

+ *但 grad-norm 会把共享参数数两遍*。all-reduce 之后首尾两端持有的是*同一份完整梯度*，在 pipeline 组上求平方和就重复计入了。实测膨胀 $1.27$×（embedding 占主导时最坏是 $sqrt(2)$）。norm 被放大意味着 clipping 比你设定的更早介入，等效学习率悄悄变小。规则和第 7 章 CP 那里完全一样：*每个被复制的参数只在一张卡上计入 norm*。

=== loss 缩放与 DP 同步时机

两件事必须同时对：

- 每个 micro-batch 的 loss 除以 $m$，这样 $m$ 次累加之后等于对整个 batch 求平均
- 梯度 all-reduce *每个 step 一次*，不是每个 micro-batch 一次——后者是 $m$ 倍通信量。原生 DDP 会在每次 backward 都触发 all-reduce，所以前 $m-1$ 个 micro-batch 必须包在 `no_sync()` 里

文件 `23` 把两件事一起验证了：引擎内部除以 $m$，之后在 DP 组上做一次 SUM 再除以 $D$，结果与单进程跑完 $D times m$ 个 micro-batch 的 global batch 逐位一致。

=== 数据路由

stage 0 要 token 但从不需要 label，最后一个 stage 要 label 但从不需要 token。中间 stage 两个都不要。

```python
inputs = toks if pp_rank == 0          else [None] * m
labels = labs if pp_rank == pp_size-1  else [None] * m
```

给每个 stage 都发全套不会算错，但白占 host-to-device 带宽，而且会掩盖"某个 stage 用错了张量"这类 bug。真实实现里 dataloader 只在 PP 组的两端起作用，而*同一个 PP 组内所有 rank 必须拿到同一个样本*——这条和第 7 章 CP 组的要求是同一回事。

== 调度族谱

引擎搭好之后，各种 schedule 就只是不同的指令生成器了。下面是它们的定位，配文件 `22` 的实测值（$P=4, m=8$，$V=2$，全部 schedule 总工作量相同均为 24 个单位——这一栏是防止"我的优化其实只是少算了"的护栏）。

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center, left),
    [*schedule*], [*实测 bubble*], [*闭式*], [*换来 / 付出*],
    [GPipe],
    [27.3%], [$(P-1)\/(m+P-1)$],
    [基线。收敛性与不开 PP 完全一致],
    [1F1B],
    [27.3%], [$(P-1)\/(m+P-1)$],
    [bubble *完全相同*，activation 从 $m$ 份降到 $P-"rank"$ 份],
    [interleaved ($V$)],
    [15.8%], [$(P-1)\/(V m+P-1)$],
    [bubble 除以 $V$；P2P 次数 ×3，显存反而更高],
    [ZeroBubble ZB-H1],
    [14.3%], [—],
    [W 填满 cool-down；warm-up 那一半填不掉，另需显存压 W 张量],
    [DualPipe],
    [—], [$(P\/2-1)(F\&B + B - 3W)$],
    [双向注入 + 4 相 overlap；*2× 权重副本*],
  ),
  kind: table,
  caption: [$P=4, m=8$。前三行的实测值与闭式公式吻合到小数点后若干位（文件 `22` 里是断言，不是打印）。],
) <tab-pp-family>

=== 三个结论值得单独记

*GPipe 和 1F1B 的 bubble 完全一样*。这与直觉相反——大多数人以为 1F1B 更快。它不更快，它只是更省显存。$P=4, m=8$ 时两者都是 27.3%，文件 `22` 里断言了这一点。

那为什么所有生产系统都用 1F1B？因为 GPipe 的 activation 是 $m$ 份而 1F1B 是 $P - "rank"$ 份，而*缩小 bubble 唯一最便宜的手段就是加大 $m$*。GPipe 把"降 bubble"和"爆显存"绑在了一起，1F1B 解开了这个耦合——这才是它赢的地方。

```
              m=2     m=4     m=8    m=16    m=32    m=64
1F1B, P=4   60.0%   42.9%   27.3%   15.8%    8.6%    4.5%
```

*zero-bubble 到不了零*。$27.3% arrow.r 14.3%$，砍掉一半。W 能填满 cool-down，但 warm-up 阶段还没有任何梯度回来，*根本没有 W 可做*。要填那一半得上 ZB-2P，代价是更多 activation 滞留。

时空图上三者的差别一眼就看出来（每格一个时间单位，`F`=forward，`B`=fused backward，`b`=B-only，`w`=W-only，`.`=bubble）：

#block(breakable: false)[
```
gpipe  (bubble 27.3%)                 1f1b  (bubble 27.3%)
 s0 |FFFFFFFF.........BBBBBBBBBBBBBBBB|  |FFFF......BBFBBFBBFBBFBB.BB.BB.BB|
 s1 |.FFFFFFFF......BBBBBBBBBBBBBBBB..|  |.FFF....BBFBBFBBFBBFBBFBB.BB.BB..|
 s2 |..FFFFFFFF...BBBBBBBBBBBBBBBB....|  |..FF..BBFBBFBBFBBFBBFBBFBB.BB....|
 s3 |...FFFFFFFFBBBBBBBBBBBBBBBB......|  |...FBBFBBFBBFBBFBBFBBFBBFBB......|
```
]

#block(breakable: false)[
```
zero-bubble  (bubble 14.3%)
 s0 |FFFF...bFbFbFbFb.bwbwbwwwwww|
 s1 |.FFF..bFbFbFbFbFb.bwbwwwwwww|
 s2 |..FF.bFbFbFbFbFbFb.bwwwwwwww|
 s3 |...FbFbFbFbFbFbFbFbwwwwwwww.|
```
]

读形状，不只读数字。GPipe 的空闲是*中间一整块*——每个 stage 把 $m$ 个 forward 全跑完，然后一起等反向波传过来。1F1B 空闲总量一模一样，但分布在 warm-up 的等待加 cool-down 的小缝里。注意*没变的那个量*：bubble。1F1B 赢的是"第一次 backward 只需等 $P-1-"rank"$ 个 forward 而不是 $m$ 个"，这是显存性质，不是时间性质。zero-bubble 则把右侧斜坡用 `w` 填掉，左侧原样留着——正如 warm-up 那个论证预测的。

*两个降 bubble 的旋钮代价不同*。加大 $m$ 在 1F1B 下不花额外显存，所以永远先动它。加大 $V$ 要付 P2P 往返和 activation，所以它是"$m$ 已经被你想要的 global batch 卡住了"之后才用的。

```
interleaved, m=8, P=4:   V=1: 27.3%   V=2: 15.8%   V=4: 8.6%   V=8: 4.5%
```

=== DualPipe 与 Megatron FWD-BWD merged

DeepSeek-V3 的 DualPipe 是最激进的方案：pipeline 两端*同时注入* micro-batch，中间 rank 同时跑正反两个方向，再把每个 chunk 拆成 attention / dispatch / MLP / combine 四相与通信交叉。bubble 系数从 $P-1$ 降到 $P\/2-1$。代价是*2× 权重副本*，外加要自己 hack autograd 层的调度、与 DeepEP 深度耦合。

Megatron 2024 给了个折中：*不改 schedule*（仍 1F1B/interleaved），但在同一 iteration 里合并相邻 micro-batch 的 fwd 和 bwd，两个 CUDA stream 并行——stream 0 跑 forward compute，stream 1 跑 backward compute 加 a2a 通信，配合 `--delay-wgrad-compute` 的 B/W 拆分打破依赖。需要 `CUDA_DEVICE_MAX_CONNECTIONS>1`。MoE 场景下 EP a2a 占比从 30--40% 压到 \< 5%，overlap 93%，且不需要 2× 权重。

什么时候值得上 DualPipe：$>$ 1000 卡、权重吃得下 2× 副本、且 comm 与 compute 接近 1:1。绝大多数场景 Megatron 的 merged 方案就够——90% 的收益，不用 2× 权重。

== PP 通信量

每个 micro-batch 每个 stage 边界过一次 P2P activation：$B S H times "bytes"$。

一个 step 总量 $approx m times (P-1) times 2 times B S H times 2$（forward + backward，bf16）。$B=1, S=8"K", H=8192, m=32, P=8$：约 60 GB per step。

*但 P2P 只在 pair 之间*，不占整个 fabric。跨节点走 IB，节点内走 NVLink，每对 pair 各自一条管道，与 collective 的性质完全不同。所以 PP 通信量看着大，实际很少成为瓶颈。

要避免的 pattern：PP 组跨节点且 P2P 消息又小又多时，延迟主导而非带宽主导。合并消息或者 chunked send。

== PP 与其他并行的组合

PP 与 TP/DP/EP 完全正交。生产配置：

*Llama-3 405B on 16K H100*：TP $= 8$（NVL 域内）、CP $= 16$、PP $= 16$、DP $= 8$。$16"K" = 8 times 16 times 16 times 8$ ✓

*DeepSeek-V3 671B on 2048 H800*：TP $= 1$（不用 TP）、EP $= 64$、PP $= 16$（DualPipe）、DP $= 2$。

选 PP 深度的经验：让每 stage 层数 $L\/P approx 4-8$。太少（每 stage 1--2 层）bubble 大且 P2P overhead 占比高；太多则 stage 内 compute 远大于通信，overlap 空间小。

== PP 的坑

+ *邻居算错*：rank $plus.minus D T$ 而非 $plus.minus 1$；子进程组里 `send/recv` 仍要全局 rank
+ *第一个 stage 的输入设了 `requires_grad`*：token id 是 int64，直接报错
+ *忘了把 recv 出来的 activation 变成 leaf*：上游梯度恒零，且不报错
+ *tied embedding 漏掉 embedding-group all-reduce*：不崩，只是 loss 不降
+ *grad-norm 重复计入 tied 参数*：norm 膨胀最多 $sqrt(2)$，clipping 提前介入
+ *每个 micro-batch 都做 DP all-reduce*：$m$ 倍通信量，忘了 `no_sync()`
+ *Layer 不均衡*：embedding 与 output head 比中间层重，按耗时分而非按层数分
+ *LN / dropout seed*：PP 组内每个 micro-batch 用不同 seed，但跨 stage 要保持一致
+ *`isend` 的 buffer 被改写或回收*：收到垃圾数据，且是间歇性的
+ *变长序列没做形状协商*：静默截断或挂住
+ *KV cache in generation*：inference 时每 stage 维护自己的 stage-local KV
+ *ckpt 的 PP 层分*：save 按 PP 存、load 要匹配；改 PP 数需要重新切分
+ *`micro_batch_size = 1` 的 attention*：$B=1$ 时 GEMM 的 M 维太小，掉利用率，可以 pack seq

== 面试考点

#interview[
  *Q1*: PP 里 `loss.backward()` 为什么不能用？你怎么办？

  A: 计算图被 `send`/`recv` 切成了 $P$ 段，loss 只在最后一个 stage 上，stage 0 上没有任何从 loss 连回自己权重的路径。做法是手工提供每一段的起点：forward 时把 recv 到的 activation `detach().requires_grad_(True)` 变成新 leaf；backward 时 recv 到下游给的 `grad_output`，调 `torch.autograd.backward(y, gy)`（不是 `y.backward()`，非标量会抛异常），它会同时填好本段的 `W.grad` 和 `x.grad`；再把 `x.grad` 发给上游。注意第一个 stage 的输入*不要* `requires_grad_`——它是数据不是 activation，而且通常是 int64 token id。
]

#interview[
  *Q2*: 1F1B 相对 GPipe 的核心收益是什么？

  A: *不是 bubble*——两者 bubble 完全相同，都是 $(P-1)/(m+P-1)$。收益是 activation 从 $m$ 份降到 $P - "rank"$ 份。为什么这很重要：降 bubble 最便宜的手段是加大 $m$，而 GPipe 的显存正比于 $m$，把"降 bubble"和"爆显存"绑死了；1F1B 解开这个耦合，所以你能放心把 $m$ 开到 $4P$ 以上。
]

#interview[
  *Q3*: 手写 1F1B 时 P2P 怎么排才不死锁？

  A: 稳态下 stage $s$ 要往下发 activation，同时 stage $s+1$ 要往上发 gradient；两边都"先 send 后 recv"就双向卡死。三种解法：奇偶定序（脆，interleaved 下不成立）；`batch_isend_irecv` 把该步所有 op 先 post 再一起 wait（Megatron 的 `send_forward_recv_backward`）；或者让 send 永不阻塞——只 post 不 wait，只有 recv 阻塞，这样循环等待需要至少一个 rank 卡在 send 上，环不可能形成。第三种要自己管 `isend` 的 buffer 生命周期（必须 clone，且要连 buffer 一起持有到 wait）。
]

#interview[
  *Q4*: 变长序列下 PP 怎么传 activation？

  A: `recv` 需要预先分配好的 tensor，所以接收方必须先知道 shape/dtype。发一个小 header（ndim + dtype code + dims）再发数据。代价是每个 tensor 多一条极小消息，所以框架做成开关（Megatron `--variable-seq-lengths`，默认关）。猜错形状的后果是静默截断或挂住，不是报错。
]

#interview[
  *Q5*: ZeroBubble 怎么把 B 和 W 拆开？为什么不能直接调两次 `autograd.grad`？

  A: 调两次会把反向图遍历两遍，白算一倍。正确做法是写一个自定义 autograd Function：backward 里只算 `gy @ w` 返回 input-grad，把 weight-grad 需要的 $(w, g_y, x)$ 推到一个 tape 上不算。于是 `BI` 就是普通 `backward()`（顺带把所有 W 的活记账），`BW` 就是排空 tape 做 `gy.T @ x`。两个 GEMM 各算一次，不需要 patch autograd。代价是显存：`BI` 释放了 activation，但 tape 上压着 $x$ 和 $g_y$ 等 `BW`，所以 zero-bubble 是拿显存换 bubble。
]

#interview[
  *Q6*: 为什么 ZeroBubble 到不了零 bubble？

  A: W 只能填 cool-down。warm-up 阶段还没有任何梯度回来，手上*没有 W 可做*，那一半 bubble 填不掉。实测 $P=4, m=8$ 从 27.3% 降到 14.3%，正好一半左右。要填 warm-up 得上 ZB-2P，代价是更多 activation 滞留。
]

#interview[
  *Q7*: interleaved 1F1B 每一步该算哪个 chunk 的哪个 micro-batch？

  A: $"chunk"(k) = floor((k mod P V)/P)$，$"mb"(k) = floor(k/(P V)) dot P + (k mod P)$，backward 方向 chunk 取 $V-1-"chunk"(k)$。全局 chunk 按 $g = v P + "rank"$ 排，rank $r$ 持有 $r, r+P, dots$。关键是这两个映射*与 rank 无关*——每张卡算出的序列相同，只是处在不同阶段，所以 send/recv 自动对上，不需要全局协调。warm-up 深度 $2(P-1-r) + (V-1)P$。
]

#interview[
  *Q8*: PP+TP+DP 一起开，pipeline 邻居是谁？

  A: Megatron 默认 $"rank" = "pp" dot (D T) + "dp" dot T + "tp"$，所以邻居是 $"rank" plus.minus D T$，*不是* $plus.minus 1$。这个顺序的理由：TP 每层搬两次完整 activation，必须在 NVLink 域内，也就是连续 rank；PP 每个边界只搬一次，给最大 stride 让它跨节点。另外 `dist.send/recv` 即使传了子组，`src`/`dst` 仍要全局 rank（用 `dist.get_global_rank(group, i)` 翻译）。算错了如果那个 rank 恰好存在，表现是挂住而非报错。
]

#interview[
  *Q9*: embedding 和 lm_head tie 在一起，但在首尾两个 stage 上，怎么处理？

  A: 首尾两个 rank 单独组一个进程组（embedding group），backward 之后在组内 all-reduce 这个梯度——因为每一端只算到一部分（stage 0 走查表路径，末 stage 走投影路径）。漏掉不会崩，两份副本几个 step 内漂开，模型用一张表编码另一张表解码，现象只是 loss 不降。补充两点：这次 reduce 与 DP 的 all-reduce 顺序无关（都是线性、作用在不相交 rank 集上，可交换），但都必须在 clip 之前完成；而 grad-norm 会把 tied 参数数两遍，norm 最多膨胀 $sqrt(2)$，导致 clipping 提前介入、等效 LR 变小，所以要保证每个复制参数只在一张卡上计入。
]

#interview[
  *Q10*: PP 下 loss 怎么缩放，DP 梯度什么时候同步？

  A: 每个 micro-batch 的 loss 除以 $m$，$m$ 次累加后等于整个 batch 的均值。DP 梯度 all-reduce *每 step 一次*，不是每 micro-batch 一次，否则是 $m$ 倍通信量——用原生 DDP 就要把前 $m-1$ 个 micro-batch 包在 `no_sync()` 里。验证方法：PP × DP 的结果应当与单进程跑完 $D times m$ 个 micro-batch 的 global batch 逐位一致。
]

#interview[
  *Q11*: PP 分层为什么不能按层数平均分？

  A: 模块耗时不等。32 层 $h=4096$ vocab 128K 时，embedding 约 0.5 个 block、lm_head 约 5 个 block（head 的 GEMM 就是 $h V/12h^2 approx 2.6$ 个 block，再加 $(B,S,128"K")$ logits 上的 CE）。按层数平均切 4 段是 1.28× 不均衡，按耗时切是 1.07×。流水线跑在最慢那段的速度上，所以这个 1.28× 是每个 micro-batch 都要付的减速，与 bubble 相乘而不是被 bubble 吸收。另外注意：不能先建完整模型再切（70B+ 装不下），分层必须在分配权重之前从成本表决定。
]

#interview[
  *Q12*: PP 组该跨节点还是同节点？

  A: 跨节点 OK，而且应该让它跨。PP 用 P2P，只有 pair-to-pair 通信，占 fabric 少；且通信频率低（每 stage 边界一次，不像 TP 每层多次）。跨节点 IB 带宽足够。TP/EP 才必须在 NVLink 域内。
]

#interview[
  *Q13*: PP 训练最后一层为什么特别慢？

  A: 最后一个 stage 是 LM head + loss，vocab 128K 时 $(B, S, V)$ 的 logits 在 compute 和 activation 上都远超一个 transformer block，而且 backward 从这里起步。做法：把 embedding 与 LM head tie 到同一 stage（省参数），或给最后一个 stage 少分几个 transformer layer 给 vocab head 留余量，或把 loss 单独放一个 virtual chunk。
]

#interview[
  *Q14*: PP=16 时选 DualPipe 还是 Megatron FWD-BWD merged？

  A: 看 comm/compute 比。DeepSeek-V3 那种 setup（comm:compute $approx$ 1:1、跨节点 MoE a2a 主导）用 DualPipe 值得。Megatron 官方推荐大多数场景用 FWD-BWD merged（1F1B + delayed wgrad + 双 stream）——90% 收益，不需要 2× 权重。除非你训 500B+ MoE 且 IB 是瓶颈，否则 DualPipe overkill。
]
