#import "../template.typ": *

= 索引、广播与形状变换

这一章是"shape 对了但值不对"这类 bug 的集中营。广播规则、fancy indexing 会不会 copy、`gather` 的 index 该是什么形状、`chunk` 为什么少给了一块——这些题面试官爱问，因为它们直接对应线上事故。第 1 章讲了 stride 决定"哪些操作是视图"，这一章讲的是"这些操作的语义到底是什么"。

== 广播规则

只有三条，从*最右边的维度*开始对齐往左比：

+ 两边长度相等 $arrow.r$ 合法；
+ 有一边是 1 $arrow.r$ 合法，把它 `expand` 到另一边（stride 设 0，见第 1 章）；
+ 一边维度不存在（更短）$arrow.r$ 合法，在左侧补 1。

其它情况报 "The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension"。

```python
(8, 1, 6, 1) 与    (7, 1, 5)  ->  (8, 7, 6, 5)   # OK
(  256, 256, 3) 与       (3,)  ->  (256, 256, 3)  # OK
(  256, 256, 3) 与(256, 256)  ->  报错（3 vs 256）
```

危险的地方在于*广播成功也可能是错的*。三个真实踩坑：

```python
# 坑 1：最经典的 loss bug
pred   = torch.randn(64, 1)        # 模型忘了 squeeze
target = torch.randn(64)           # label 是 1-D
(pred - target).shape              # (64, 64) —— 不报错！
((pred - target) ** 2).mean()      # 算的是所有 pair 的距离，loss 假性偏大且几乎不下降

# 坑 2：忘了 keepdim
x = torch.randn(32, 32)            # B 恰好等于 S，最阴的情况
x / x.sum(-1)                      # (32,32)/(32,)：第 j *列* 被第 j 行的和除，错
x / x.sum(-1, keepdim=True)        # 正确

# 坑 3：mask 维度对齐
scores = torch.randn(4, 8, 16, 16) # (B, H, S, S)
mask   = torch.zeros(4, 16).bool() # (B, S) padding mask
scores.masked_fill(mask, -1e4)     # 报错：4 vs 16
scores.masked_fill(mask[:, None, None, :], -1e4)   # 正确：(B,1,1,S)
```

#insight[
  防广播 bug 的三个习惯：reduce 一律带 `keepdim=True`；构造 mask 时显式写出所有 `None` 让 rank
  一眼可数；关键位置 `assert pred.shape == target.shape`。`torch.broadcast_shapes` 可以不分配内存先算 shape。
]

== basic indexing vs advanced indexing

*basic indexing*（整数、`slice`、`None`、`Ellipsis`）返回*视图*，因为结果能用一组新的
`(shape, stride, offset)` 表达。*advanced indexing*（index 是 list / 整型 tensor / bool tensor）
返回*副本*，因为取出来的位置一般不是等差的。

```python
x = torch.arange(20).reshape(4, 5)
x[1:3, 2:4]        # view：shape (2,2) stride (5,1) offset 7
x[..., None, 2]    # view
x[[0, 2]]          # copy：advanced
x[x > 10]          # copy：bool mask（内部走 nonzero，在 GPU 上还会隐式同步）
x[[0, 2]] = 0      # 有效！赋值走 index_put_，直接写原张量
```

#warn[
  advanced indexing 返回副本，所以 `x[[0,2]] += 1` 生效（Python 会翻译成 `__setitem__`），
  但 `x[[0,2]].add_(1)` *不生效*——改的是临时副本，且不会报错。
  bool mask 同理：`x[mask].zero_()` 是无效操作。要就地改用 `x[mask] = 0` 或 `x.masked_fill_(mask, 0)`。
]

=== 多个 index tensor 的语义（高频题）

多个 index tensor 之间是*先互相广播、再逐元素配对取值*，不是"先取行再取列"。

```python
x = torch.arange(35).reshape(5, 7)
i = torch.tensor([0, 1, 2])
j = torch.tensor([3, 4, 5])

x[i, j]            # shape (3,)：取 x[0,3], x[1,4], x[2,5] —— 逐元素配对
x[i][:, j]         # shape (3,3)：先选 3 行，再从这 3 行里选 3 列 —— 笛卡尔积
x[i[:, None], j]   # shape (3,3)：等价于上面，i 变成 (3,1) 与 j 的 (3,) 广播成 (3,3)
```

要"对角线取值"用 `x[i, j]`，要"子矩阵"用 `x[i[:, None], j]`。这是 gather 与 index_select
语义差别的根源，也是最容易在白板上写错的一行。

#note[
  混用 basic 与 advanced 时，结果维度的位置有个规则：如果所有 advanced index 是*相邻*的，
  广播出来的维度就留在原位；如果被 slice 隔开了，广播维度会被搬到*最前面*。
  例：`x` shape `(2,3,4,5)`，`x[:, idx, idx2, :]` 得到 `(2, B, 5)`；
  而 `x[idx, :, idx2]` 得到 `(B, 3)`——`B` 跑到最前。不确定就 print shape，不要靠记忆。
]

== gather / scatter / index 系列

#table(
  columns: (auto, 1.2fr, 1.1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*op*], [*语义*], [*典型用途*],
  [`gather(dim, index)`],
    [`index` 与 `input` 同 ndim，输出 shape = `index.shape`。逐位置沿 `dim` 取],
    [取每个 token 的 label logit、beam search 收集],
  [`scatter_(dim, index, src)`],
    [`gather` 的逆：`self[.., index[..], ..] = src[..]`],
    [one-hot、label smoothing],
  [`scatter_add_(dim, index, src)`],
    [同上但*累加*，重复 index 全部相加],
    [稀疏梯度归约、MoE 的 combine],
  [`index_select(dim, index)`],
    [`index` 是 1-D，沿 `dim` 整片选，返回 copy],
    [取子词表、按 rank 选 shard],
  [`index_add_(dim, index, src)`],
    [沿 `dim` 整片累加],
    [embedding 反向、MoE unpermute],
  [`masked_fill_(mask, v)`],
    [`mask` 为 True 的位置填标量 `v`（`mask` 可广播）],
    [attention causal / padding mask],
  [`where(cond, a, b)`],
    [三方广播后逐元素选],
    [数值保护、条件分支向量化],
)

`gather` 的 index 形状是最容易记错的：*`index` 的 ndim 必须等于 `input` 的 ndim，
除了 `dim` 那一维，其它维度的 size 必须能与 `input` 对上；输出 shape 就等于 `index.shape`。*

```python
logits = torch.randn(4, 10)                 # (B, C)
labels = torch.tensor([3, 1, 9, 0])         # (B,)

# 取每个样本正确类别的 logit
picked = logits.gather(1, labels[:, None]).squeeze(1)     # (4,)
# 等价的 advanced indexing 写法，更好读
picked = logits[torch.arange(4), labels]                  # (4,)

# one-hot：scatter_ 在零矩阵上打点
onehot = torch.zeros(4, 10).scatter_(1, labels[:, None], 1.0)
# 更简单的等价写法
onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
```

#warn[
  `torch.where` 会*同时求值两个分支*，所以它不能用来避开 `nan`：

  ```python
  x = torch.tensor([0.0, 1.0], requires_grad=True)
  y = torch.where(x > 0, torch.log(x), torch.zeros_like(x))
  y.sum().backward()
  x.grad     # tensor([nan, 1.]) —— log(0) 的 -inf 在反向里污染了梯度
  ```

  正确做法是先把输入夹到安全区间：`torch.log(x.clamp_min(1e-12))`，
  或者用 `masked_fill` 把非法位置换成合法值再算。
]

`scatter_` 遇到重复 index 时"哪个写赢"是未定义的（GPU 上不确定）；需要确定性就用
`scatter_add_` 或 `index_add_`。注意 `index_add_` / `scatter_add_` 的浮点累加顺序在 GPU 上
也是不确定的，逐 bit 复现要看第 10 章。

== einsum 速成

`einsum` 用下标字符串描述"哪些维度相乘、哪些维度求和"：*出现在输入但不出现在输出的下标被 sum 掉，
其它下标逐元素配对。*

```python
A = torch.randn(3, 4); B = torch.randn(4, 5)
torch.einsum("ik,kj->ij", A, B)          # matmul：k 被消掉

Q = torch.randn(2, 8, 128, 64)           # (B, H, S, D)
K = torch.randn(2, 8, 128, 64)
V = torch.randn(2, 8, 128, 64)
S = torch.einsum("bhqd,bhkd->bhqk", Q, K)        # attention score，省掉 transpose
O = torch.einsum("bhqk,bhkd->bhqd", S.softmax(-1), V)

torch.einsum("i,j->ij", torch.randn(3), torch.randn(4))   # outer product (3,4)
torch.einsum("bij,bjk->bik", X, Y)       # bmm
torch.einsum("ii->i", M)                 # 取对角线
torch.einsum("bhd->bh", T)               # 沿最后一维 sum
torch.einsum("...ij->...ji", T)          # 转置，支持 ellipsis
```

*什么时候用*：下标一多（4-5 维带 batch 和 head），`einsum` 比一串 `permute` + `reshape` + `bmm`
好读得多，而且不会写错维度顺序。*什么时候不用*：热点路径。`einsum` 内部会把式子降解成
`permute` + `reshape` + `bmm`，其中 `reshape` 可能触发 copy；它也拿不到 flash attention 这类
融合 kernel。attention 请直接调 `F.scaled_dot_product_attention`，别自己 einsum。

#note[
  三个以上操作数时收缩顺序会显著影响 FLOPs，PyTorch 通过 `opt_einsum` 自动挑
  （`torch.backends.opt_einsum.enabled`，默认开）。两个操作数时没有选择空间。
]

== 形状变换 API 的选择

#figure(
  align(center, shape-pipeline(stages: (
    ("hidden", "(B, S, H)", "H = n_heads * d_head"),
    ("qkv = Linear(H, 3H)", "(B, S, 3H)", "一次 fused GEMM"),
    ("unflatten(-1, (3, N, D))", "(B, S, 3, N, D)", "纯元数据，零 copy"),
    ("permute(2, 0, 3, 1, 4)", "(3, B, N, S, D)", "non-contiguous"),
    ("unbind(0)", "3 x (B, N, S, D)", "拆出 q / k / v"),
    ("attn out -> transpose(1,2)", "(B, S, N, D)", "此时 non-contiguous"),
    ("reshape(B, S, H)", "(B, S, H)", "这里必然发生一次 copy"),
  ))),
  caption: [多头 attention 的形状流水线。只有最后一步真的搬内存，
    前面全是 stride 改写。把 `reshape` 写成 `contiguous().view(...)` 能让这次 copy 显式可见。],
) <fig-mha-shapes>

#table(
  columns: (auto, 1.4fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*API*], [*什么时候用*],
  [`view(shape)`], [确定输入连续、想让不连续时*报错*而不是静默 copy],
  [`reshape(shape)`], [不确定连续性、可以接受 copy],
  [`flatten(start, end)`], [合并一段相邻维度，比手算乘积安全],
  [`unflatten(dim, sizes)`], [拆一个维度，`(B,S,H) -> (B,S,N,D)` 的首选],
  [`squeeze(dim)`], [去掉指定的 size-1 维。*一定带 dim*],
  [`unsqueeze(dim)`], [插一个 size-1 维；等价于索引里的 `None`],
  [`movedim(src, dst)`], [只想搬一两个维度，比 `permute` 少写一堆下标],
  [`permute(*dims)`], [完全重排，必须列出所有维度],
)

`-1` 的推断规则：最多一个 `-1`，其余维度的乘积必须整除 `numel()`，否则报
"shape is invalid for input of size N"。`view(-1)` 是"拉平成 1-D"。
注意 `numel() == 0` 时 `-1` 无法推断，会直接报错。

#warn[
  *不带参数的 `squeeze()` 是 batch size = 1 时的经典事故源。*

  ```python
  x = torch.randn(1, 1, 768)      # batch=1 的推理请求
  x.squeeze().shape               # (768,) —— batch 维也被吃了，后面全错
  x.squeeze(1).shape              # (1, 768) —— 正确
  ```

  训练时 batch=32 一切正常，上线遇到单条请求就炸。永远写 `squeeze(dim)`。
  torch 2.0+ 支持一次给多个：`squeeze((1, 2))`。
]

== cat / stack / split / chunk

`cat` 沿*已有*维度拼接，ndim 不变，除 `dim` 外其它维度必须一致。
`stack` 造一个*新*维度，所有输入 shape 必须完全相同。

```python
a, b = torch.randn(2, 3), torch.randn(2, 3)
torch.cat([a, b], dim=0).shape      # (4, 3)
torch.stack([a, b], dim=0).shape    # (2, 2, 3)
torch.stack(xs, 0)                  # == torch.cat([x[None] for x in xs], 0)
```

反向的两个 API 差别更值得记：

```python
t = torch.arange(10)
[c.shape[0] for c in t.split(3)]    # [3, 3, 3, 1]  —— 每块固定 3，最后一块剩多少给多少
[c.shape[0] for c in t.chunk(4)]    # [3, 3, 3, 1]  —— 要 4 块，ceil(10/4)=3
t.split([2, 5, 3])                  # 显式给每块大小，最精确的写法
```

#warn[
  *`chunk(n)` 可能返回少于 `n` 块。* 它先算 `ceil(numel/n)` 作为块大小，再按这个大小切：

  ```python
  len(torch.arange(9).chunk(4))      # 3 —— ceil(9/4)=3，切成 3+3+3，第 4 块不存在
  len(torch.arange(9).tensor_split(4))   # 4 —— 3+2+2+2，保证块数
  ```

  在按 rank 切数据、按 GPU 数切 batch 的代码里，`chunk` 会让某些 rank 拿不到数据然后 collective 挂死。
  要"保证 n 块"用 `tensor_split(n)`；要"精确指定大小"用 `split([...])`。
]

`split` / `chunk` / `unbind` 返回的都是视图。但沿 stride 较大的维度切出来的片*不连续*，
交给 NCCL 之前要 `.contiguous()`（第 17 章）。

== matmul 的广播规则

`torch.matmul`（也就是 `@`）按输入的 ndim 分情况处理：

#table(
  columns: (auto, auto, 1.2fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*输入*], [*输出*], [*行为*],
  [`(n)` @ `(n)`], [`()`], [点积，返回 0-D 标量],
  [`(m,n)` @ `(n,p)`], [`(m,p)`], [普通矩阵乘 = `mm`],
  [`(m,n)` @ `(n)`], [`(m)`], [矩阵-向量 = `mv`],
  [`(n)` @ `(n,p)`], [`(p)`], [左边补 1 变 `(1,n)`，算完再去掉],
  [`(B,m,n)` @ `(B,n,p)`], [`(B,m,p)`], [batch 矩阵乘 = `bmm`],
  [`(B,H,m,n)` @ `(n,p)`], [`(B,H,m,p)`], [最后两维做矩阵乘，*前面的 batch 维广播*],
  [`(1,H,m,n)` @ `(B,1,n,p)`], [`(B,H,m,p)`], [batch 维按标准广播规则对齐],
)

核心一句：*最后两维做矩阵乘，前面所有维度按标准广播规则对齐。*

`mm` / `bmm` / `mv` 是不带广播、不带 ndim 推断的严格版本：`mm` 只吃 2-D，`bmm` 只吃 3-D 且两边
batch 维必须相等。写库代码时用它们更好——*形状写错会立刻报错，而不是被广播规则"救活"成一个错的 shape*。

#note[
  `matmul` 广播 batch 维时可能需要物化被广播的一侧（把 stride=0 展开成真实内存），
  batch 维很大时这是隐藏的显存开销，`(1,H,m,n) @ (B,1,n,p)` 这种双向广播尤其要留意。
  另外 `nn.Linear` 内部走 `addmm`，把 `bias + A @ B` 融成一个 kernel。
]

== reduce：dim 与 keepdim

`sum` / `mean` / `max` / `norm` / `argmax` 这类 op 的三个约定：

- `dim` 不给 $arrow.r$ 对*全部元素* reduce，返回 0-D 标量。
- `dim` 给了 $arrow.r$ 该维消失（`keepdim=False`，默认），或变成 1（`keepdim=True`）。
- `dim` 可以是 tuple：`x.sum(dim=(1, 2))`。但 `max` / `min` 只接受单个 `dim`，
  且返回 `(values, indices)` 命名元组——直接当张量用会报错。

```python
x = torch.randn(4, 8, 16)
x.sum().shape                   # ()
x.sum(dim=1).shape              # (4, 16)
x.sum(dim=1, keepdim=True).shape   # (4, 1, 16)
x.sum(dim=(1, 2)).shape         # (4,)
v, i = x.max(dim=-1)            # 两个返回值
x.amax(dim=-1).shape            # 只要值、支持多 dim 时用 amax/amin
x.mean()                        # 整型张量会报错，先 .float()
```

`keepdim=True` 能救广播 bug，因为它让结果的 rank 与输入一致，后续广播必然按你期望的维度对齐：

```python
x = torch.randn(32, 32)
x - x.mean(-1, keepdim=True)    # 逐行去均值：正确
x - x.mean(-1)                  # (32,32) - (32,) 按最后一维对齐：逐列去均值，错
```

所以*凡是 reduce 的结果还要参与与原张量的运算，就带 `keepdim=True`*。
norm / softmax / attention 里到处都是这个模式。

== 数值稳定：softmax、logsumexp、log_softmax

朴素 softmax 会溢出：`x_i = 1000` 时 $e^(x_i)$ 在 fp32 里就是 `inf`，`inf/inf` 得 `nan`。
标准做法是先减去最大值——因为分子分母同乘一个常数不改变结果：

#formula[$ "softmax"(x)_i = e^(x_i) / (sum_j e^(x_j)) = e^(x_i - m) / (sum_j e^(x_j - m)), quad m = max_k x_k $]

减完之后所有指数的自变量都 $<= 0$，$e^(x_i - m) in (0, 1]$，永不上溢；分母至少有一项等于 1，
永不为 0。下溢只会让某些项变成 0，不影响结果的正确性。

`logsumexp` 用同一个技巧，把这个模式单独封成一个算子：

#formula[$ "LSE"(x) = log sum_i e^(x_i) = m + log sum_i e^(x_i - m) $]

于是 log-softmax 只是一次减法：

#formula[$ log "softmax"(x)_i = x_i - "LSE"(x) $]

*为什么 `log_softmax(x)` 比 `log(softmax(x))` 稳定：* 后者要先把概率算出来，
当某个 $x_i$ 远小于最大值时 `softmax` 的结果下溢成 0，`log(0)` 得 $-infinity$，梯度变 `nan`。
前者直接算 $x_i - "LSE"(x)$，中间从不出现"接近 0 的概率"，再小的 logit 也只得到一个很负的有限值，
而且少一次舍入。

```python
x = torch.tensor([0.0, -1000.0])
torch.log(torch.softmax(x, -1))      # tensor([0., -inf])
torch.log_softmax(x, -1)             # tensor([    0., -1000.])  正确
torch.logsumexp(x, -1)               # tensor(0.)
```

所以损失函数一律用 logits 版本：`F.cross_entropy(logits, labels)` 内部是 `log_softmax` + `nll_loss`，
`F.binary_cross_entropy_with_logits` 用了等价的 `log(1+exp(-|x|))` 变形。
*不要自己先 `softmax` / `sigmoid` 再喂给 `nll_loss` / `bce`。*

#insight[
  混精下这条更要紧。bf16 只有 7 位尾数，手写 `exp` / `log` 组合会迅速丢精度。
  PyTorch 的 `softmax` / `log_softmax` / `cross_entropy` 在 half 输入上会*内部升到 fp32* 累加再降回来，
  这是你自己写 `exp(x)/exp(x).sum()` 拿不到的。
]

== 面试考点

#interview[
  *Q1*：广播规则是什么？举一个"广播成功但结果是错的"的例子。

  A：从最右维度对齐往左比，每个维度要么相等、要么其中一个是 1、要么不存在（补 1）。
  经典错例：`pred` 是 `(N,1)`、`target` 是 `(N,)`，`pred - target` 广播成 `(N,N)`，
  算出的是所有 pair 的差，loss 不报错但根本不收敛。防御手段是 reduce 带 `keepdim=True`
  加上关键位置 `assert pred.shape == target.shape`。
]

#interview[
  *Q2*：`x[i, j]` 和 `x[i][:, j]` 有什么区别？

  A：`x[i, j]` 把 `i` 和 `j` 先互相广播再*逐元素配对*取值，`i`、`j` 都是长 3 的 1-D 时结果是 `(3,)`，
  取的是 `x[0,3], x[1,4], x[2,5]`。`x[i][:, j]` 是先选 3 行再选 3 列，结果 `(3,3)` 的笛卡尔积。
  想用一次索引拿到子矩阵，写 `x[i[:, None], j]`。
]

#interview[
  *Q3*：哪些索引返回视图、哪些返回副本？有什么实际影响？

  A：basic indexing（整数、slice、`None`、`Ellipsis`）返回视图，因为结果能用新的
  `(shape, stride, offset)` 表达。advanced indexing（整型/bool tensor、list）返回副本。
  影响是 `x[mask].zero_()` 或 `x[[0,2]].add_(1)` 改的是临时副本、静默无效，
  必须写成 `x[mask] = 0`（走 `index_put_`）或 `x.masked_fill_(mask, 0)`。
  另外 bool mask 索引在 GPU 上内部走 `nonzero`，会隐式同步。
]

#interview[
  *Q4*：`gather` 的 `index` 形状有什么要求？怎么取每个样本正确类别的 logit？

  A：`index` 的 ndim 必须等于 `input` 的 ndim，输出 shape 就等于 `index.shape`。
  `logits` 是 `(B,C)`、`labels` 是 `(B,)` 时要写
  `logits.gather(1, labels[:, None]).squeeze(1)`；
  等价且更好读的写法是 `logits[torch.arange(B), labels]`。
]

#interview[
  *Q5*：`scatter_` 和 `scatter_add_` 的区别？

  A：`scatter_` 是赋值，重复 index 时哪个写赢在 GPU 上是未定义的；
  `scatter_add_` 是累加，重复 index 全部相加，语义确定（但浮点累加顺序仍不保证逐 bit 复现）。
  one-hot 用 `scatter_`，把稀疏更新聚合回稠密张量（embedding 反向、MoE combine）用 `scatter_add_`
  或 `index_add_`。
]

#interview[
  *Q6*：`chunk(4)` 一定返回 4 块吗？

  A：不一定。`chunk(n)` 先算 `ceil(numel/n)` 当块大小再切，`arange(9).chunk(4)` 只返回 3 块
  （3+3+3）。按 rank 切数据时这会让某些 rank 空手然后 collective 挂死。
  要保证块数用 `tensor_split(n)`，要精确控制大小用 `split([...])`。
  另外 `chunk` / `split` 沿 stride 大的维度切出来的片是非连续的，交给 NCCL 前要 `contiguous()`。
]

#interview[
  *Q7*：`matmul`、`@`、`bmm`、`mm` 该用哪个？

  A：`@` 就是 `matmul`：最后两维做矩阵乘，前面的 batch 维按广播规则对齐，还会根据 ndim
  自动退化成点积 / `mv`。`mm` 只吃 2-D，`bmm` 只吃 3-D 且 batch 维必须严格相等。
  写库代码优先用 `mm` / `bmm`，因为形状写错会立刻报错，不会被广播"救活"成一个错的 shape。
]

#interview[
  *Q8*：为什么 `log_softmax` 比 `log(softmax(x))` 稳定？`cross_entropy` 该喂什么？

  A：`softmax` 先算出概率，logit 很小的位置会下溢成 0，`log(0)` 得 $-infinity$，梯度变 `nan`。
  `log_softmax` 直接算 $x_i - "LSE"(x)$，其中 $"LSE"(x) = m + log sum e^(x_i - m)$，
  中间不出现接近 0 的概率，且少一次舍入。所以 `F.cross_entropy` 必须喂 *logits*，
  它内部就是 `log_softmax` + `nll_loss`；自己先 `softmax` 再取 log 是错的写法。
]
