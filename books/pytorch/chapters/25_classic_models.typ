#import "../template.typ": *

= 经典模型：MLP / CNN / RNN / Seq2Seq

这一章开始是白板题。面试官不会再问"`view` 和 `reshape` 有什么区别"，他会说："来，手写一个 ResNet BasicBlock。" 这类题的评分点不在"能不能写出来"——大部分候选人都能写个七八成——而在那几个只有真正实现过才知道的细节：ReLU 到底加在残差相加之前还是之后、GRU 的重置门乘在矩阵乘之前还是之后、`h_n` 的形状受不受 `batch_first` 影响。这一章把这些点全部挑出来。

配套的可运行代码 + 权重级对齐测试在 `python/pytorch/interview/test_classic_models.py`（31 个 pytest，全部通过）。RNN 部分全都和 `torch.nn` 官方实现做了逐元素数值比对——因为"我能证明我写的和 `nn.LSTMCell` 完全一致"是这道题最有说服力的答案。

本章的代码是白板规模：省掉了 jaxtyping 标注和长 docstring，逻辑与配套文件一字不差。Transformer 全家桶在第 26 章。

== MLP：送分题里的失分点

*题目*：写一个可配置深度的 MLP，输入 `in_dim`、隐层列表 `hidden_dims`、输出 `out_dim`。

*关键考点*。这道题真正在考三件事：容器怎么选、激活/norm/dropout 的顺序、以及最后一层能不能加激活。

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))   # 输出层：不激活、不 dropout
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

*`nn.Sequential` 还是 `nn.ModuleList`？* 判据只有一条：*forward 是不是纯串联*。是就用 `Sequential`（自带 `forward`，少写一个循环）；有跳连、多分支、需要拿中间层输出的必须用 `ModuleList`——它只注册子模块、不定义 `forward`，怎么连由你写。

#warn[
  用 Python 原生 `list` 存子模块是致命错误：
  ```python
  self.layers = [nn.Linear(d, d) for _ in range(4)]   # 参数不会被注册！
  ```
  `nn.Module.__setattr__` 只拦截 `Parameter` / `Module` / `ModuleList` 这些类型，普通 `list` 里的东西对它是不可见的。后果是 `model.parameters()` 收不到它们（optimizer 不更新）、`model.to("cuda")` 搬不动它们（forward 直接报 device mismatch）、`state_dict()` 存不下来。这个 bug 在小网络上表现为"loss 降得很慢"，极容易被误判成学习率问题。见第 3 章。
]

*激活放在 norm 前还是后？* 加了 norm 之后的标准顺序是 `Linear → Norm → Act → Dropout`。norm 的作用是把进入非线性之前的分布拉到零均值单位方差，让激活工作在梯度最好的区间；放到激活之后就失去了这个意义。附带一条：此时前一层 `Linear` 的 `bias` 可以省掉，它会被 norm 的减均值完全吃掉。dropout 则放在激活*之后*——对 ReLU 前后等价（被置零的位置过 ReLU 仍是 0），但换成 GELU / Sigmoid 就不等价了，`sigmoid(0) = 0.5` 相当于给网络注入一个常数偏置。

#warn[
  *最后一层不加激活也不加 dropout*，这是这道送分题最常见的失分点。分类头后面接的 `nn.CrossEntropyLoss` 内部已经含 `log_softmax`，它要的是原始 logits（可正可负、无界）。再套一层 ReLU 会把所有负 logit 压成 0，模型永远无法表达"这个类不可能"；套一层 `softmax` 则是做了两次 softmax，梯度被压得极小，训练看起来"能动但学不出来"。
]

*怎么自证*：断言 `isinstance(m.net[-1], nn.Linear)`，再喂一批随机输入检查输出*能取到负值*（有激活的话 ReLU 后恒非负，断言会挂）；层序直接比对 `[type(x).__name__ for x in m.net]`。

== ResNet BasicBlock：ReLU 加在哪

*题目*：手写 ResNet-18/34 的残差块，支持 stride 和通道数变化。

*关键考点*：ReLU 的位置、shortcut 什么时候需要投影、conv 后接 BN 时为什么 `bias=False`。

```python
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))        # 注意：这里没有 relu
        return F.relu(out + identity)
```

#warn[
  *最高频的错法：在加法之前多写一个 ReLU。*
  ```python
  out = F.relu(self.bn2(self.conv2(out)))   # 错
  return out + identity
  ```
  后果是残差主分支的输出*恒非负*，残差只能"加"不能"减"，表达能力直接砍掉一半——网络再也无法学出"把这个通道的响应压下去"。它不会报错，shape 完全正确，loss 也会下降，只是精度上不去。正确顺序是 `relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))`：*ReLU 必须在相加之后*。
]

*shortcut 什么时候需要 downsample*：`stride != 1`（空间尺寸变了）或 `in_planes != planes`（通道数变了）时，恒等映射的 shape 对不上，必须用 1×1 conv（同样的 stride）把它投影过去。注意这个 1×1 conv *也要配一个 BN*，否则两条分支的数值尺度不匹配，相加后主分支会被淹掉。

#note[
  能加分的一句：1×1 conv + stride=2 的 downsample *丢掉了 3/4 的像素*（每 2×2 窗口只采左上角那个）。ResNet-D / bag-of-tricks 的改法是先 2×2 AvgPool 再 1×1 conv stride=1，把被跳过的像素平均进去。这说明你真读过论文而不只是抄过代码。
]

*conv 后接 BN 时为什么 `bias=False`*：BN 做的是 $(x - mu) \/ sigma dot gamma + beta$，conv 的 bias 是逐通道常数，先被 $x - mu$ 的减均值项完全抵消，再由 BN 自己的 $beta$ 重新提供。它既不改变函数空间，还白占参数和一次访存。ResNet 全网的 conv 都是 `bias=False`。

*为什么残差能训深*。设块的映射是 $y = F(x) + x$，则 $partial y \/ partial x = partial F \/ partial x + I$。那个 $+I$ 是关键：$L$ 层堆起来，梯度的展开式里存在一项等于恒等（把所有 $partial F \/ partial x$ 取零那一项），梯度可以原样传回最底层；没有它的话梯度是 $L$ 个 Jacobian 连乘，谱范数略小于 1 就指数衰减到 0。"残差提供了一条恒等通路"是标准答案，能顺手写出这个式子会好得多。

*怎么自证*：把两个 conv 的权重全部置零、BN 设成恒等（`running_mean=0`、`running_var=1`、eval 模式），此时主分支恒为 0，block 的输出必须*精确等于* `F.relu(x)`——这一条同时验证了残差连线接对和 ReLU 在加法之后。另一条：随机权重下检查 `bn2(conv2(...))` 的输出*能取到负值*，证明加法前没有 ReLU。

== LSTMCell：4 个门与两套 bias

*题目*：手写 `nn.LSTMCell`，要求能直接 `load_state_dict(nn.LSTMCell(...).state_dict())` 并数值对齐。

*关键考点*：门的公式与顺序、为什么一次 matmul 算完 4 个门、为什么有两套 bias。

#formula[
  $ i &= sigma(W_(i i) x + b_(i i) + W_(h i) h + b_(h i)) quad &&"输入门：新信息写入多少" \
    f &= sigma(W_(i f) x + b_(i f) + W_(h f) h + b_(h f)) quad &&"遗忘门：旧 cell 保留多少" \
    g &= tanh(W_(i g) x + b_(i g) + W_(h g) h + b_(h g)) quad &&"候选值（tanh，不是 sigmoid）" \
    o &= sigma(W_(i o) x + b_(i o) + W_(h o) h + b_(h o)) quad &&"输出门：cell 露出多少" \
    c' &= f ⊙ c + i ⊙ g \
    h' &= o ⊙ tanh(c') $
]

```python
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:   # 声明成 None，state_dict 的键才和官方一致
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        std = 1.0 / hidden_size ** 0.5          # 官方 RNN 系列的初始化
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, state=None):
        if state is None:
            z = x.new_zeros(x.shape[0], self.hidden_size)
            state = (z, z)
        h, c = state
        # 一次算完 4 个门：(B,I)@(I,4H) + (B,H)@(H,4H) -> (B,4H)
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(h, self.weight_hh, self.bias_hh))
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
```

*为什么一次大 matmul 而不是 4 次小的*。两条理由，面试官通常要听到第二条：*算术强度*——4 次 `(B,I)@(I,H)` 变成 1 次 `(B,I)@(I,4H)`，小矩阵乘是彻底 memory-bound 的，分开写要把 `x` 从 HBM 读 4 遍，合并后只读一遍、算术强度提升约 4 倍；*kernel launch*——RNN 要跑 $T$ 个时间步，$T = 100$ 时省下的是几百次 launch，短序列小 batch 下 launch 开销能占到整体一半以上。这就是 `nn.LSTMCell` 的 `weight_ih` 形状是 `(4*H, I)` 而不是四个 `(H, I)` 的原因：行方向按 *i, f, g, o* 顺序拼接。

#warn[
  *门顺序记成 "ifgo"（IFGO）。* 记错了权重照样能 `load_state_dict` 进去、shape 全对、不报任何错，但结果完全不对——比如把遗忘门当输入门用，网络行为彻底改变。这是"能跑但错"的典型，只有做数值对齐才能发现。
]

#insight[
  *PyTorch 有两套 bias（`bias_ih` 和 `bias_hh`），数学上完全冗余。* $b_(i i) + b_(h i)$ 可以合并成一个数，参数量白多一份 $4H$。留着纯粹是为了对齐 cuDNN 的 API——cuDNN 分别存输入侧和隐层侧的 bias，PyTorch 的 `nn.LSTM` 直接把它的权重布局透出来了。手写时只写一个 bias，功能上没问题，但和官方做数值对齐时会*差一个常数*，然后你会花半小时怀疑自己的门顺序。
]

*`c` 和 `h` 的区别*。`c` 是"内部长期记忆"，只被逐元素的加法和乘法更新，从不过矩阵乘；`h` 是"对外输出"，是 `c` 经 `tanh` 压缩再被输出门筛选的结果。维度相同但角色完全不同：下一层和输出层看到的是 `h`，`c` 只在时间维上流动。

*为什么 LSTM 能缓解梯度消失*。cell state 的更新是 $c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t$，对 $c_(t-1)$ 求导得 $partial c_t \/ partial c_(t-1) = "diag"(f_t)$——*逐元素相乘，不是矩阵乘*。跨 $T$ 步的梯度是 $product_t f_t$，只要遗忘门保持接近 1，梯度就能近似无损地沿 `c` 传回去，和残差连接的 $+I$ 是同一个思路。而原始 RNN 的 $partial h_t \/ partial h_(t-1) = W_(h h)^T "diag"(tanh')$ 是 $T$ 个矩阵连乘，$W_(h h)$ 的谱半径稍微偏离 1 就指数爆炸或消失。

#note[
  工程 trick：把遗忘门的 bias 初始化成 1（`bias_ih` 的第 2 段填 1），让训练早期 $f approx 1$，cell state 默认"记住"。PyTorch 默认没这么做，但 Jozefowicz et al. 2015 系统验证过这个 trick 有效。
]

*怎么自证*：`mine.load_state_dict(nn.LSTMCell(I,H).state_dict())` 之后喂同一个 `(x, h0, c0)`，`torch.testing.assert_close` 比对 `h'` 和 `c'`。验门顺序可以更狠：把 `weight` 全置零、只用 `bias_ih` 把四段分别设成 `[+20, -20, +20, +20]`（`sigmoid(±20)` 已经饱和到 1/0），则 $i=1, f=0, g approx 1, o=1$，于是 $c' = 1$、$h' = tanh(1)$。顺序记错的话这两个值都对不上。

== LSTM：沿时间步展开，以及 `h_n` 的形状陷阱

*题目*：用上面的 cell 实现单层单向 LSTM，支持 `batch_first`。

```python
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = MyLSTMCell(input_size, hidden_size)

    def forward(self, x, state=None):
        if self.batch_first:
            x = x.transpose(0, 1)              # (B,T,I) -> (T,B,I)
        t, b, _ = x.shape
        if state is None:
            h = x.new_zeros(b, self.hidden_size)
            c = x.new_zeros(b, self.hidden_size)
        else:
            h, c = state[0][0], state[1][0]    # 去掉 num_layers 维

        outputs = []
        for step in range(t):
            h, c = self.cell(x[step], (h, c))
            outputs.append(h)
        output = torch.stack(outputs, dim=0)    # (T,B,H)

        if self.batch_first:
            output = output.transpose(0, 1)
        # h_n / c_n 始终是 (num_layers, B, H)
        return output, (h.unsqueeze(0), c.unsqueeze(0))
```

*`batch_first=False` 为什么是默认值*——不是反人类，是内存布局。循环里每步要取 `x[step]`，`(T,B,I)` 布局下这是*一整块连续内存*，直接喂给 GEMM；`(B,T,I)` 下每个时间步都是跨 stride 的切片，GEMM 前得先物化一次。所以实现上先 `transpose` 成 `(T,B,I)` 再循环，比在循环里反复做非连续索引快。

#insight[
  *`h_n` / `c_n` 的形状不受 `batch_first` 影响，恒为 `(num_layers * num_directions, B, H)`。* 只有 `output` 会在 `(T,B,H)` 和 `(B,T,H)` 之间切换。这个不一致坑过所有人——写 `h_n[0]` 想拿第一个样本，实际拿到的是第一层。这是本题最常被追问的一点。
]

#warn[
  *时间步之间有严格的串行依赖*，这是 RNN 打不过 Transformer 的根本原因：$T$ 步无法并行，GPU 利用率被时间维锁死。cuDNN 的优化手段是把"输入侧的 $W_(i *) x$"在所有时间步上*一次性算完*（它不依赖 `h`），只把 $W_(h *) h$ 留在循环里——这是很好的追问点，因为它解释了为什么 `nn.LSTM` 比你手写的 for 循环快好几倍，而不是靠什么魔法。
]

#note[
  双向 LSTM 不是"反着再跑一遍就完事"：反向那一支的 `h_n` 取的是*序列起点*那一步的状态；而且变长序列必须配 `pack_padded_sequence`，否则反向会从 padding 开始跑，把 pad 的信息混进真实 token。
]

*怎么自证*：和 `nn.LSTM(I, H, num_layers=1)` 逐权重对拷（`ref.weight_ih_l0` 等四个），比对 `output`、`h_n`、`c_n` 三个返回值；再断言 `out[:, -1] == h_n[0]`，最后一个时间步的输出就是 `h_n`，这条恒等式在单向单层下必须成立。

== GRUCell：重置门乘在哪

*题目*：手写 `nn.GRUCell` 并与官方数值对齐。

#formula[
  $ r &= sigma(W_(i r) x + b_(i r) + W_(h r) h + b_(h r)) quad &&"重置门" \
    z &= sigma(W_(i z) x + b_(i z) + W_(h z) h + b_(h z)) quad &&"更新门" \
    n &= tanh(W_(i n) x + b_(i n) + r ⊙ (W_(h n) h + b_(h n))) quad &&"候选状态" \
    h' &= (1 - z) ⊙ n + z ⊙ h $
]

```python
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih   = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh   = nn.Parameter(torch.zeros(3 * hidden_size))
        std = 1.0 / hidden_size ** 0.5
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, h=None):
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
        gi = F.linear(x, self.weight_ih, self.bias_ih)   # (B, 3H)
        gh = F.linear(h, self.weight_hh, self.bias_hh)   # (B, 3H)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)     # 关键：r 乘的是 (W_hn h + b_hn) 整体
        return (1 - z) * n + z * h
```

#insight[
  *本题唯一也是最大的考点：`r` 到底乘在矩阵乘之前还是之后。*
  原论文（Cho et al. 2014）写的是 $n = tanh(W_(i n) x + W_(h n) (r ⊙ h))$——先把 `r` 作用在 `h` 上，再做矩阵乘。
  PyTorch / cuDNN 实现的是 $n = tanh(W_(i n) x + b_(i n) + r ⊙ (W_(h n) h + b_(h n)))$——*`r` 乘在矩阵乘之后，而且把 $b_(h n)$ 也一起乘进去了*。
  动机纯粹是性能：这样 $W_(h n) h + b_(h n)$ 就能和 `r`、`z` 两个门的隐层项合并成*一次* `(B,H)@(H,3H)` 的 GEMM；按原论文写，`n` 分支必须等 `r` 算出来才能做它自己的矩阵乘，被迫拆成两次 GEMM 且串行。
  因为 `r` 是逐元素的，$r ⊙ (W h) != W (r ⊙ h)$，两者*数学上不等价*。实际效果差异很小，但做权重对齐时会直接对不上。
]

#warn[
  第二个坑是更新门的方向。PyTorch 的 $h' = (1-z) n + z h$ 里，*`z` 是"保留旧状态"的比例*：$z arrow.r 1$ 时 $h' = h$。有些教材和框架写成 $z n + (1-z) h$，语义正好相反。权重能加载进去，行为完全不同。
]

*另外两点*：GRU 只有 3 组权重（`weight_ih` 是 `(3H, I)`），参数量是 LSTM 的 3/4；没有独立的 cell state，长期记忆和输出合并成一个 `h`，短序列上和 LSTM 基本打平、超长依赖上通常略逊。

*怎么自证*：`load_state_dict(nn.GRUCell(...).state_dict())` 后逐元素对齐。更能体现理解的是把两种写法都算出来：用官方的 `weight_hh[2H:]` 手动构造 `paper_style = tanh(i_n + F.linear(r * h, w_hn, b_hn))`，断言它*不等于* `ref(x, h)` 而 PyTorch 写法相等。这一条测试就是这道题的分水岭。

== Bahdanau vs Luong：Transformer 之前的 attention

*题目*：实现加性（additive）和乘性（multiplicative）两种 attention，支持 padding mask。

打分函数的对比就是这道题的全部：

#formula[
  $ "Bahdanau (additive)": quad &"score"(q, k) = v^T tanh(W_q q + W_k k) \
    "Luong (general)": quad &"score"(q, k) = q^T W k \
    "Luong (dot)": quad &"score"(q, k) = q^T k $
]

```python
class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim,   attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, values=None, mask=None):
        if values is None:
            values = keys
        # (B,Tq,1,A) + (B,1,Tk,A) -> (B,Tq,Tk,A)
        feats = torch.tanh(self.w_q(query).unsqueeze(2)
                           + self.w_k(keys).unsqueeze(1))
        scores = self.v(feats).squeeze(-1)          # (B,Tq,Tk)
        return _attend(scores, values, mask)

class LuongAttention(nn.Module):
    def __init__(self, query_dim, key_dim, scaled=False):
        super().__init__()
        self.w = nn.Linear(query_dim, key_dim, bias=False)
        self.scaled, self.key_dim = scaled, key_dim

    def forward(self, query, keys, values=None, mask=None):
        if values is None:
            values = keys
        scores = self.w(query) @ keys.transpose(-1, -2)   # (B,Tq,Tk)
        if self.scaled:
            scores = scores / self.key_dim ** 0.5
        return _attend(scores, values, mask)

def _attend(scores, values, mask):
    if mask is not None:                     # mask: (B,Tk)，True = 有效
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ values, weights         # (B,Tq,Dv), (B,Tq,Tk)
```

*参数量与计算量的差异*，这是选择的依据：

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [Bahdanau（加性）], [Luong（乘性）],
  [参数], [$W_q$、$W_k$、$v$，共 $(D_q + D_k) A + A$], [$W$，共 $D_q D_k$],
  [中间张量], [`(B, Tq, Tk, A)`], [`(B, Tq, Tk)`],
  [算子], [broadcast add + tanh + 一次 `(·,1)` 投影], [两次 batched matmul],
  [`Dq != Dk`], [天然支持], [靠 $W$ 桥接],
  [硬件友好度], [elementwise 为主，memory-bound], [纯 GEMM，能打满 tensor core],
)

关键差别是那个多出来的 $A$ 维：加性注意力必须显式物化 `(B, Tq, Tk, A)` 的中间张量，显存和访存都是乘性版的 $A$ 倍。Transformer 选乘性，论文原话就是"乘性在实践中快得多且省显存"——因为它能直接调用高度优化的 matmul。

#note[
  一个反直觉的点：$d_k$ 很大时*加性反而更稳*。点积的方差随 $d_k$ 线性增长，不缩放的话 softmax 会饱和；加性因为外面套了 `tanh`，打分天然被压在 $[-1, 1]$，没有这个问题。Transformer 的解法不是换回加性，而是乘性 + $1 \/ sqrt(d_k)$ 缩放——推导见第 26 章。Luong 原文没有这一项，因为当年 $d_k$ 只有几百且输入是被 `tanh` 压过的 RNN 输出，问题不明显。
]

*mask 的处理*有两条规则。一是*在 softmax 之前*把无效位置的 score 设成 $-infinity$，softmax 之后自然是精确的 0；如果 softmax 之后再乘 0，剩下的权重就不再和为 1 了，等于给不同样本用了不同的温度。二是 mask 形状为 `(B, Tk)`，`unsqueeze(1)` 成 `(B, 1, Tk)` 对所有 query 位置广播——padding 是 key 侧的属性，与 query 无关。

#warn[
  *`-inf` 的陷阱*：某一行被*全部* mask 掉时（比如 batch 里混进一个长度为 0 的样本），`softmax(全 -inf)` 是 `exp(-inf)` 全 0 除以 0，结果是 NaN，然后 NaN 顺着反向传播污染掉整个模型的梯度。两个解法：要么在 dataloader 侧保证每条序列至少一个有效位置，要么用 `torch.finfo(scores.dtype).min` 代替 `-inf`（全 mask 行会得到均匀分布，是无意义但有限的值）。
  配套测试里专门留了一个 `test_attention_fully_masked_row_produces_nan` 断言 NaN *会*出现——把坑固化成反面教材，比在注释里写一句提醒有用。
]

*怎么自证*：断言 `weights.sum(-1) == 1`；断言 masked 位置的权重*精确*为 0（不是"很小"）；最强的一条是把 padding 位置的 `value` 换成 `1e4` 这种大数，检查 `context` 完全不变——这证明 mask 真的挡住了信息流，而不只是让权重看起来小。

== Seq2Seq：encoder + decoder + attention

*题目*：用上面的组件拼一个带 attention 的 seq2seq，支持 teacher forcing 和 source mask。

```python
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_dim):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb_dim)
        self.encoder = MyLSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder_cell = MyLSTMCell(emb_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, tgt_vocab)   # 吃 [h ; context]

    def forward(self, src, tgt, src_mask=None):
        enc_out, (h_n, c_n) = self.encoder(self.src_emb(src))  # (B,Ts,H)
        h, c = h_n[0], c_n[0]                                  # 桥接初始状态

        tgt_emb = self.tgt_emb(tgt)                            # (B,Tt,E)
        logits, attns = [], []
        for t in range(tgt.shape[1]):
            h, c = self.decoder_cell(tgt_emb[:, t], (h, c))    # teacher forcing
            context, attn = self.attention(h.unsqueeze(1), enc_out, mask=src_mask)
            logits.append(self.out(torch.cat([h, context.squeeze(1)], dim=-1)))
            attns.append(attn.squeeze(1))
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)
```

#figure(
  align(center, shape-pipeline(stages: (
    ("src ids", "(B, Ts)", "源句 token"),
    ("src_emb", "(B, Ts, E)", "Embedding"),
    ("enc_out", "(B, Ts, H)", "encoder 全部时间步的 h"),
    ("query", "(B, 1, H)", "decoder 第 t 步的 hidden"),
    ("attn", "(B, 1, Ts)", "对源位置的分布，行和为 1"),
    ("context", "(B, 1, H)", "attn @ enc_out，加权求和"),
    ("[h ; ctx]", "(B, 2H)", "拼接后过输出层"),
    ("logits_t", "(B, V)", "第 t 步的词表分布"),
  ))),
  caption: [seq2seq 的 shape 流。decoder 每步都要重跑虚线以下的四步，$T_t$ 步串行——这正是 Transformer decoder 用 causal mask 一次算完、而 RNN decoder 做不到的地方。],
) <fig-seq2seq-shape>

*decoder 为什么必须逐步展开*，而 encoder 可以一把跑完？因为每一步的 attention query 是*上一步的 decoder hidden state*，而 encoder 的输入是已知的完整源句，没有这个依赖。这正是 RNN decoder 训练慢的根源：Transformer decoder 训练时所有位置的 query 都来自已知的 target 序列，配上 causal mask 可以一次算完 $T_t$ 步。

*teacher forcing 与 exposure bias*。训练时喂真实的上一个 token（代码里的 `tgt_emb[:, t]`），推理时只能喂模型自己的预测。两者输入分布不一致，这就是 *exposure bias*：模型从没见过"自己犯了错之后的状态"，推理早期一偏就一路崩下去。常见缓解手段是 scheduled sampling（训练中按概率混入自己的预测）。反过来不用 teacher forcing 的话训练初期几乎学不动，因为随机的预测提供不了有效梯度。

*decoder 的初始 hidden 从哪来*。这里用 encoder 的最后状态 `h_n[0]`，叫"桥接"。但有个反直觉的事实：*有了 attention 之后直接零初始化也能训得差不多好*——源句信息主要通过 attention 流过去，而不是挤在这个定长向量里；attention 诞生的动机本来就是"不要把整个源句压进一个定长向量"。桥接更多是历史惯例加一点热启动的好处。

*context 怎么用*：这里把 context 和 decoder hidden 拼起来过输出层（Luong 的做法）；Bahdanau 的做法是把 context 拼进*下一步*的 decoder 输入，让 context 参与状态更新，理论上更强但实现更绕。

#warn[
  *`src_mask` 必须传到 attention。* 变长 batch 里如果 padding 位置参与了 softmax，模型会学会去看 padding——而 padding 的分布在训练集和验证集里可能完全不同（比如验证集句子更长、padding 更少），于是验证指标和训练时对不上，而且这种偏差随 batch 组成随机波动，非常难排查。
]

*怎么自证*：`logits.shape == (B, Tt, V)`、`attns.sum(-1) == 1`；传 mask 后断言 `attns[:, :, valid_len:]` 精确为 0；最后跑一次 `cross_entropy(...).backward()`，检查*每一个* named parameter 都拿到了有限梯度——seq2seq 组件多，最容易出的问题是某个模块因为写错连线而完全没参与 forward。

== RNN 家族对比

#table(
  columns: (auto, auto, auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, left, left),
  [], [门数], [状态], [参数量（`weight` + `bias`）], [适用],
  [RNN], [0], [`h`], [$H I + H^2 + 2H$], [几乎不用了，梯度消失严重],
  [LSTM], [4], [`h`, `c`], [$4(H I + H^2 + 2H)$], [长依赖、需要显式记忆通路],
  [GRU], [3], [`h`], [$3(H I + H^2 + 2H)$], [参数敏感场景、小数据集],
)

表里的"门数"按 PyTorch 的权重布局口径数（候选值 $g$ / $n$ 也算一路，因为它在 `weight_ih` 里占一段）；严格说 LSTM 是 3 个门 + 1 个候选，GRU 是 2 个门 + 1 个候选。参数量的推导很简单：每一路需要一个 $(H, I)$ 的输入侧权重、一个 $(H, H)$ 的隐层侧权重、两个长度 $H$ 的 bias，乘上路数就是全部。所以 *GRU 的参数量恰好是 LSTM 的 3/4*，这是个常被要求口算的数。

== 面试考点

#interview[
  *Q1*：ResNet BasicBlock 里 ReLU 加在残差相加之前还是之后？写错了会怎样？

  A：*之后*，`relu(bn2(conv2(...)) + shortcut(x))`。加在相加之前的话主分支输出恒非负，残差就只能加不能减，网络无法学出"抑制某个通道"，表达能力砍半。它不报错、shape 全对、loss 也会降，只是精度上不去，属于最难发现的一类错。
]

#interview[
  *Q2*：conv 后面接 BN 时为什么要 `bias=False`？

  A：BN 是 $(x - mu)\/sigma dot gamma + beta$，conv 的 bias 是逐通道常数，先被减均值项完全抵消，再由 BN 的 $beta$ 重新提供。所以它既不扩大函数空间，还多占参数和一次访存。同理 `Linear → LayerNorm` 时 Linear 的 bias 也可以省。
]

#interview[
  *Q3*：为什么残差连接能训很深的网络？

  A：$partial y \/ partial x = partial F \/ partial x + I$，那个 $+I$ 给梯度留了一条恒等通路，$L$ 层堆起来梯度里始终存在一项等于恒等。没有它的话梯度是 $L$ 个 Jacobian 连乘，谱范数略小于 1 就指数衰减。LSTM 的 cell state（$partial c_t \/ partial c_(t-1) = "diag"(f_t)$）是同一个思路的时间维版本。
]

#interview[
  *Q4*：LSTM 为什么把 4 个门拼成一次 matmul？

  A：两点。一是算术强度：4 次 `(B,I)@(I,H)` 要把 `x` 从 HBM 读 4 遍，合成 1 次 `(B,I)@(I,4H)` 后只读一遍，小 GEMM 本来是 memory-bound 的，这一改直接提升 4 倍强度。二是 kernel launch：$T$ 个时间步 $times$ 省 3 次 launch，短序列小 batch 下 launch 开销能占一半。这也是 `nn.LSTMCell.weight_ih` 形状是 `(4H, I)`、按 i/f/g/o 顺序拼接的原因。
]

#interview[
  *Q5*：`nn.LSTMCell` 为什么有 `bias_ih` 和 `bias_hh` 两套 bias？

  A：数学上完全冗余，$b_(i i) + b_(h i)$ 可以合并成一个。留着纯粹是为了对齐 cuDNN 的 API（cuDNN 分别存输入侧和隐层侧的 bias），是历史包袱。实际影响：手写版只写一个 bias 的话，功能没问题但和官方做数值对齐会差一个常数。
]

#interview[
  *Q6*：`nn.LSTM` 的 `batch_first=True` 会改变哪些返回值的形状？

  A：*只有 `output`*，从 `(T,B,H)` 变成 `(B,T,H)`。`h_n` 和 `c_n` 恒为 `(num_layers * num_directions, B, H)`，不受影响。这个不一致性的后果是有人写 `h_n[0]` 想拿第一个样本，实际拿到的是第一层。默认 `batch_first=False` 是因为循环里取 `x[t]` 在 time-first 布局下是连续内存。
]

#interview[
  *Q7*：PyTorch 的 GRU 和原论文有什么区别？为什么？

  A：重置门 `r` 的作用位置。论文是 $tanh(W_(i n) x + W_(h n)(r ⊙ h))$，PyTorch/cuDNN 是 $tanh(W_(i n) x + b_(i n) + r ⊙ (W_(h n) h + b_(h n)))$——`r` 乘在矩阵乘*之后*，还把 $b_(h n)$ 一起乘了进去。动机是让三路 gate 的隐层项合并成一次 `(B,H)@(H,3H)` GEMM；按论文写，`n` 分支得等 `r` 算完才能做矩阵乘，被迫拆两次且串行。因为 `r` 是逐元素的，两者数学上不等价，做权重对齐会直接对不上。
]

#interview[
  *Q8*：Bahdanau 和 Luong attention 的区别？Transformer 为什么选后者？

  A：打分函数不同。Bahdanau 是加性 $v^T tanh(W_q q + W_k k)$，要物化 `(B,Tq,Tk,A)` 的中间张量；Luong 是乘性 $q^T W k$，只要 `(B,Tq,Tk)`。Transformer 选乘性是因为它能直接落到高度优化的 matmul 上，硬件友好度天差地别。代价是点积的方差随 $d_k$ 增长会让 softmax 饱和，所以补了 $1\/sqrt(d_k)$；加性因为有 tanh 压着，天然没这个问题。
]

#interview[
  *Q9*：为什么 Transformer 取代了 RNN？

  A：核心是*并行性*，不是表达能力。RNN 的 $h_t$ 依赖 $h_(t-1)$，$T$ 步严格串行，GPU 利用率被时间维锁死，且任意两个位置的信息要走 $O(T)$ 步才能相遇（梯度也要穿过 $O(T)$ 个 Jacobian）。Transformer 训练时所有位置一次算完，任意两位置的路径长度是 $O(1)$，唯一代价是 attention 的 $O(S^2)$ 复杂度——在 $S$ 不太大时用换来的并行度和硬件利用率完全值得。这也解释了为什么 RNN 在*推理*上反而有优势：它的状态是 $O(1)$ 的，而 Transformer 的 KV cache 是 $O(S)$，这是 Mamba / RWKV 这条线的出发点。
]
