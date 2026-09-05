#import "../template.typ": *

= nn.Module 的一切

`nn.Module` 是 PyTorch 里唯一有"状态"的抽象，也是面试最容易问出深浅的地方。表面看它就是 `__init__` 加 `forward`，实际上是四张注册表加一套 `__setattr__` 魔法，所有诡异行为（参数没进 optimizer、`load_state_dict` 报 missing key、`eval()` 后指标突变）都能从这四张表解释。训练循环与 optimizer 的配合见第 5 章，hook 在分布式里的用法见第 18 章。

== 四张注册表与 `__setattr__`

`nn.Module.__init__` 建立的核心状态是四个有序字典：

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*注册表*], [*装什么*], [*进 `state_dict`*],
  [`_parameters`], [`nn.Parameter`，要被 optimizer 更新的张量], [是],
  [`_buffers`], [`register_buffer` 注册的张量，不需要梯度], [`persistent=True` 才进],
  [`_modules`], [子 `nn.Module`], [递归展开],
  [hook 字典（`_forward_hooks` 等）], [各类回调], [否],
)

`__setattr__` 被重写过：`self.x = v` 时先看 `v` 的类型，`nn.Parameter` 进 `_parameters`，`nn.Module` 进 `_modules`，普通 tensor 落到 `__dict__`（*什么都不进*）。四张表决定三件事：谁被 `parameters()` 遍历到（进 optimizer）、谁被 `state_dict()` 存下来、谁被 `.to(device)` 搬走。

#figure(
  align(center, flow-boxes(boxes: (
    "module(x)", "_call_impl", "pre_hooks",
    "forward()", "fwd_hooks", "output",
  ), box-w: 2.2, gap-x: 0.4)),
  caption: [`module(x)` 而不是 `module.forward(x)` 的原因：`__call__` 走 `_call_impl`，
    `forward_pre_hooks` 与 `forward_hooks` 只在这条路上触发。直接调 `forward()`
    会绕过所有 hook，也绕过 DDP / FSDP 的 unshard 逻辑。],
) <fig-module-call>

=== `nn.Parameter` 为什么要包一层

`nn.Parameter` 是 `Tensor` 子类，只加两件事：默认 `requires_grad=True`，以及*作为类型标记*让 `__setattr__` 认出它。没有它就得靠"需不需要梯度"来猜，而这个判断在冻结参数时立刻失效——被冻结的参数依然要进 checkpoint、依然要被 `.cuda()` 搬走。类型标记与梯度开关必须解耦。

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()                          # 必须先调，否则四张表不存在
        self.w = nn.Parameter(torch.randn(4, 4))    # -> _parameters
        self.register_buffer("mu", torch.zeros(4))  # -> _buffers
        self.fc = nn.Linear(4, 4)                   # -> _modules
        self.scale = torch.ones(4)                  # -> __dict__，孤儿！

m = M().cuda()
m.w.device, m.mu.device      # cuda:0, cuda:0
m.scale.device               # cpu —— .cuda() 没搬它，forward 里一用就 device mismatch
```

== register_buffer 与 persistent=False

buffer 是"属于模型状态、但不参与梯度更新"的张量。它跟着 `.to(device)` 走、跟着 `state_dict` 存，但不出现在 `parameters()` 里，所以 optimizer 看不见它。

```python
self.register_buffer("running_mean", torch.zeros(dim))       # 存进 ckpt
self.register_buffer("causal_mask", mask, persistent=False)  # 不存进 ckpt
self.register_buffer("cos_cached", cos, persistent=False)    # RoPE 缓存
```

`persistent=False` 用于*能从超参重算出来的东西*：causal mask、RoPE 的 sin/cos 表、位置编码常量。好处是省 checkpoint 体积，且改 `max_seq_len` 时不会因形状不匹配而 `load_state_dict` 失败。

*为什么 BN 的 `running_mean` / `running_var` 是 buffer 而不是 parameter：* 它们是对训练数据分布的*统计量*、靠 EMA 累积，不是梯度学出来的（BN 里靠梯度学的是 `weight` / `bias`，那两个才是 Parameter）；但推理完全依赖它们，必须存进 checkpoint。"要持久化 + 不要梯度"正好是 buffer 的定义。BN 还有第三个 buffer `num_batches_tracked`，用于 `momentum=None` 时的累积平均。

== 容器：ModuleList / ModuleDict / Sequential

#table(
  columns: (auto, 1.3fr, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*容器*], [*语义*], [*有 forward*],
  [`nn.Sequential`], [按顺序调用，自带 `forward`，只能单输入单输出], [有],
  [`nn.ModuleList`], [像 list 一样索引 / 遍历，`forward` 要自己写], [无],
  [`nn.ModuleDict`], [像 dict 一样按 key 取，顺序 = 插入顺序], [无],
  [`nn.ParameterList` / `ParameterDict`], [同上，但装 `nn.Parameter`], [无],
)

```python
# 有分支 / 残差 / 需要 layer index -> ModuleList，forward 自己写
self.layers = nn.ModuleList([Block(dim) for _ in range(n_layers)])
# 纯串行无分支 -> Sequential，省一段 forward
self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
# 按名字取 -> ModuleDict
self.heads = nn.ModuleDict({"cls": nn.Linear(d, 10), "reg": nn.Linear(d, 1)})
```

#warn[
  *用 python `list` / `dict` 装 module，参数不会被注册。* 这是最经典的静默 bug：

  ```python
  class Bad(nn.Module):
      def __init__(self):
          super().__init__()
          self.layers = [nn.Linear(4, 4) for _ in range(3)]   # 普通 list！

  m = Bad()
  len(list(m.parameters()))    # 0 —— optimizer 拿到空参数列表
  m.state_dict().keys()        # 空 —— checkpoint 存不下这些层
  m.cuda()                     # 这些 Linear 还在 CPU 上
  ```

  症状是"loss 不动"或 device mismatch，而不是报错。换成 `nn.ModuleList([...])` 即可，同理 `self.w = [p1, p2]` 要换 `nn.ParameterList`。自查一句：`assert sum(p.numel() for p in m.parameters()) > 0`。
]

== state_dict 与 load_state_dict

`state_dict()` 返回一个 `OrderedDict`，key 是点分路径，value 是 *`detach()` 后的张量，与模型参数共享同一块 storage*。所以它不是快照：

```python
sd = model.state_dict()
optimizer.step()          # 参数原地更新
sd["fc.weight"]           # 也跟着变了！
```

要真快照（EMA、best-model 缓存）必须显式 clone：`{k: v.detach().clone() for k, v in model.state_dict().items()}`。想拿到还挂在 autograd 图上的原始 Parameter 用 `state_dict(keep_vars=True)`。

`load_state_dict()` 反过来是*原地 copy*：内部在 `torch.no_grad()` 下做 `param.copy_(input_param)`，不重新绑定对象。所以 optimizer 里持有的 Parameter 引用在 load 之后依然有效，不需要重建 optimizer。代价是 shape 必须对上，否则报 "size mismatch for ...: copying a param with shape ..."。

```python
missing, unexpected = model.load_state_dict(sd, strict=False)
# missing:    模型里有、checkpoint 里没有的 key（新加的层、persistent=False 的 buffer）
# unexpected: checkpoint 里有、模型里没有的 key（删掉的层、多余的 module. 前缀）
assert not unexpected, f"checkpoint 里有模型不认识的 key: {unexpected}"
print(f"未加载（应当只有新 head）: {missing}")
```

`strict=False` 的正当用途只有两类：把预训练 backbone 加载到带新 head 的模型；跳过 `persistent=False` 的 buffer。*用完必须检查 `missing` / `unexpected`*——否则一个拼错的前缀会让整个 backbone 静默保持随机初始化，训练"能跑但收敛很差"。

=== DDP 的 `module.` 前缀

`DistributedDataParallel` 把原模型挂在 `.module` 上，所以它 `state_dict()` 的每个 key 都多一层 `module.` 前缀。存的时候剥掉，加载时就不用管：

```python
torch.save(model.module.state_dict(), path)      # 推荐：存裸模型，与并行方式解耦

# 已经存成带前缀的了，加载时剥掉：
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
sd = torch.load(path, map_location="cpu", weights_only=True)
consume_prefix_in_state_dict_if_present(sd, "module.")
model.load_state_dict(sd)
```

#note[
  torch 2.4 起 `torch.load` 的 `weights_only` 默认是 `True`（只反序列化张量，不执行 pickle 里的任意代码）。加载旧 checkpoint 遇到 `UnpicklingError` 就是这个原因。分布式 checkpoint 见第 22 章。
]

== train() 与 eval()：只影响两类层

`model.train()` / `model.eval()` 只做一件事：递归地把每个子 module 的 `self.training` 置成 `True` / `False`。行为改变发生在那些*读了这个标志*的层里：

#table(
  columns: (auto, 1.4fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*层*], [*`train()` vs `eval()` 的差别*],
  [`Dropout` / `Dropout1d/2d/3d` / `AlphaDropout`], [train 时按 `p` 置零并放大 $1/(1-p)$；eval 时恒等映射],
  [`BatchNorm1d/2d/3d`], [train 用当前 batch 统计量并更新 running stats；eval 用 running stats],
  [`InstanceNorm`（`track_running_stats=True`）], [同 BN],
  [`LayerNorm` / `GroupNorm` / `RMSNorm`], [*没有区别*，永远用当前样本统计量],
)

所以 `eval()` 之后指标突变，八成是 BN 的 running stats 没喂够（batch 太小、`momentum` 太大），或者数据分布与训练时不一致。

=== `eval()` 与 `torch.no_grad()` 是两件不同的事

- `eval()` 改的是*层的计算行为*，不影响建图，梯度照算、显存照吃。
- `no_grad()` 改的是*是否建 autograd 图*，完全不影响 Dropout / BN 的行为。

推理两个都要（`model.eval()` 加 `with torch.inference_mode():`）。也有只要一个的场景：BN 的 warmup 校准需要 `train()` + `no_grad()`——要更新 running stats，但不要梯度。

#warn[
  只写 `torch.no_grad()` 不写 `model.eval()`，是验证集指标偏低最常见的原因：Dropout 仍在随机置零，BN 仍在用 batch 统计量并且*污染 running stats*。反过来只写 `eval()` 不写 `no_grad()`，会白建一整张 autograd 图、显存翻倍甚至 OOM，但结果是对的。
]

`inference_mode()` 比 `no_grad()` 更快（还免掉 version counter 和 view 追踪），代价是里面产生的张量不能在外面参与 autograd。RL rollout 这种要把结果拿回来继续训练的场景只能用 `no_grad()`。

== 参数初始化

`nn.Linear` 的默认初始化是 `kaiming_uniform_(weight, a=math.sqrt(5))`。代入 kaiming uniform 的公式，`a=sqrt(5)` 让 gain $= sqrt(2/(1+5)) = sqrt(1/3)$、边界 $= "gain" dot sqrt(3 / "fan_in") = sqrt(1 / "fan_in")$，也就是 $U(-1/sqrt("fan_in"), 1/sqrt("fan_in"))$。bias 用同样的边界。

#note[
  `a=sqrt(5)` 是对齐旧版 Torch 的历史遗留，*不是*"为 ReLU 调好的 Kaiming"。要按 ReLU 推荐值初始化得自己写 `kaiming_normal_(w, nonlinearity="relu")`。
]

Xavier（Glorot）与 Kaiming（He）的选择标准是*激活函数是否关于 0 对称*：

#table(
  columns: (auto, auto, 1.2fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*方法*], [*方差*], [*适用激活*],
  [Xavier / Glorot], [$2 / ("fan_in" + "fan_out")$], [tanh、sigmoid、恒等——正负对称，前后向方差都想守住],
  [Kaiming / He], [$2 / "fan_in"$（`fan_in` 模式）], [ReLU、LeakyReLU——砍掉一半输出，方差要补 2 倍回来],
)

Kaiming 的 2 倍就是"ReLU 让一半神经元输出 0、输出方差减半"的补偿。transformer 常用的 GELU / SiLU 更接近 ReLU 族，一般按 Kaiming 或直接用固定小 std（GPT-2 系用 `normal_(std=0.02)`）。

常用 `nn.init`：`xavier_uniform_` / `xavier_normal_` / `kaiming_uniform_` / `kaiming_normal_` / `normal_` / `zeros_` / `ones_` / `trunc_normal_` / `orthogonal_`，全部 in-place。`apply()` 会*后序*递归遍历所有子 module（先子后父）并逐个调用给定函数，是自定义初始化的标准姿势：

```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# 在 __init__ 末尾：
self.apply(self._init_weights)
# 残差分支的输出投影额外缩小，防止深层激活方差随层数线性增长（GPT-2 的做法）
for name, p in self.named_parameters():
    if name.endswith("proj.weight"):
        nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
```

== 权重共享（tied embedding）

把输入 embedding 与输出投影绑成同一个 `Parameter`，省一份 `vocab * dim` 的参数，同时是一种正则。写法就是直接赋值：

```python
class LM(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.emb.weight    # 同一个 Parameter 对象

m = LM(100, 8)
m.lm_head.weight is m.emb.weight               # True
len(list(m.parameters()))                      # 1 —— named_parameters 去重
sorted(m.state_dict().keys())                  # ['emb.weight', 'lm_head.weight'] —— 不去重
```

*关键的不对称：`named_parameters()` / `parameters()` 默认去重（`remove_duplicate=True`），`state_dict()` 不去重。* 前者保证 optimizer 不会对同一个参数更新两次；后者保证 checkpoint 自描述、能被没有 tying 的模型加载（两个 key 指向同一块内存，`torch.save` 正确处理引用共享，磁盘上不存两份）。反向时两处用法的梯度会*自动累加*到同一个 `.grad`，这正是 tying 想要的语义。

#warn[
  绑定必须在构造时用*赋值*完成。写成 `self.lm_head.weight.data = self.emb.weight.data` 只是让两个 Parameter 对象共享 storage：`parameters()` 会返回两个，optimizer 更新两次，momentum 各存一份，行为与真 tying 不同且难查。
]

== hooks 全表

#table(
  columns: (auto, 1fr, 1.2fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*hook*], [*触发时机与签名*], [*真实用途*],
  [`register_forward_pre_hook`],
    [`forward` 之前，`fn(mod, args)`；返回值替换输入],
    [FSDP 在这里 all-gather 出完整参数],
  [`register_forward_hook`],
    [`forward` 之后，`fn(mod, args, out)`；返回值替换输出],
    [抓中间激活做特征提取、量化时统计 activation 分布],
  [`register_full_backward_pre_hook`],
    [该 module 的 `grad_output` 到手时，`fn(mod, grad_out)`],
    [反向前把参数从 CPU 换回 GPU（activation offload）],
  [`register_full_backward_hook`],
    [`grad_input` 算完后，`fn(mod, grad_in, grad_out)`],
    [逐层梯度范数监控、定位哪一层先出 `nan`],
  [`Tensor.register_hook`],
    [该*张量*的梯度算好时，`fn(grad)`；返回值替换梯度],
    [DDP 绑在参数上触发 bucket ready、梯度裁剪 / 反向 mask],
  [`register_state_dict_pre_hook`],
    [`state_dict()` 组装之前],
    [FSDP 在存之前把分片参数聚合成完整张量],
  [`register_load_state_dict_post_hook`],
    [`load_state_dict()` 完成之后],
    [加载后重新切分参数、校验 `missing_keys`],
)

```python
grad_norms = {}
def watch(name):
    def hook(mod, grad_in, grad_out):
        grad_norms[name] = grad_out[0].norm().item()
    return hook

handles = [m.register_full_backward_hook(watch(n))
           for n, m in model.named_modules() if isinstance(m, nn.Linear)]
for h in handles:
    h.remove()          # 用完必须 remove，否则闭包持有张量引用 -> 显存泄漏
```

#warn[
  三个坑：（1）hook 必须 `remove()`，闭包里捕获的张量会阻止显存回收；（2）`register_backward_hook`（无 `full_`）是废弃接口，在多输入 / in-place 时给错的 `grad_input`，一律用 `full_backward` 版；（3）直接调 `model.forward(x)` 绕过 `__call__`，所有 hook 都不触发——这也是 DDP / FSDP 包过的模型必须用 `model(x)` 调用的原因。
]

`state_dict` 系列 hook 的公开接口在 torch 2.3–2.5 之间调整过（早期只有带下划线的私有版本），用之前先确认版本上的签名。全局 hook 在 `torch.nn.modules.module.register_module_forward_hook`。

== 遍历参数：named_parameters 与 weight decay 分组

```python
for name, p in model.named_parameters(): ...  # 'layers.0.attn.qkv.weight'
for name, m in model.named_modules(): ...     # '' 是模型自己，然后逐层展开
for name, b in model.named_buffers(): ...
model.parameters(recurse=False)               # 只要本层直属参数，不含子 module
model.named_parameters(prefix="student")      # 给 key 加前缀，多模型合并时有用
```

最常考的实用代码是 *weight decay 分组*：decay 只应作用在矩阵型权重上。bias 和 norm 的 `weight` / `bias` 都是逐通道的一维参数，对它们做 L2 会把 scale 往 0 拉，损害表达能力且没有正则收益。判据用 `p.ndim` 最稳，不依赖命名习惯：

```python
def param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue                       # 冻结的参数别放进 optimizer
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)             # bias、LayerNorm/RMSNorm 的 weight、可学 scale
        else:
            decay.append(p)                # Linear / Embedding / Conv 的权重矩阵
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

opt = torch.optim.AdamW(param_groups(model), lr=3e-4, betas=(0.9, 0.95))
```

两个细节：`named_parameters()` 默认去重，所以 tied embedding 不会被放进两个组；`nn.Embedding` 是二维的、会落进 decay 组——是否给 embedding 加 decay 各家做法不一。

== 冻结参数：`requires_grad_(False)` vs 从 optimizer 排除

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [`p.requires_grad_(False)`], [不放进 optimizer],
  [算梯度], [不算], [算],
  [`.grad` 显存], [不占], [占（与参数同大小）],
  [反向计算量], [该子图被剪掉，省时间], [照跑],
  [weight decay / momentum], [不适用], [不更新，但可能残留 state],
  [被 `state_dict` 保存], [是], [是],
)

结论：*要冻结就用 `requires_grad_(False)`*，它同时省显存和算力；"从 optimizer 排除"只在你想手动接管更新（自定义 EMA、手写 LoRA 合并）时才用。实践中两个一起做：

```python
model.backbone.requires_grad_(False)     # Module 上也有这个方法，递归生效
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
```

#warn[
  两个坑。（1）只做 `requires_grad_(False)` 但*仍把参数留在 optimizer 里*，且它们之前已经有过 `.grad`（冻结发生在若干 step 之后），Adam 会继续用残留的 `.grad` 和 momentum 更新它们——参数在"冻结"后还在动。冻结时顺手 `p.grad = None` 并重建 optimizer。（2）冻了参数没冻 BN：BN 在 `train()` 下仍会更新 running stats，backbone 的行为还在变，要配 `backbone.eval()`。DDP 下参数完全不需要梯度时还要注意 `find_unused_parameters`，见第 18 章。
]

== `to()` / `cuda()` 的不一致：module 原地改，tensor 返回新对象

```python
t = torch.randn(4)
t.to("cuda")            # 返回新张量，t 本身还在 CPU！必须写 t = t.to("cuda")

m = nn.Linear(4, 4)
m.to("cuda")            # 原地改 module（也返回 self，所以 m = m.to(...) 也对）
next(m.parameters()).device      # cuda:0
```

这个不一致是有原因的：`Tensor` 语义上不可变，改 dtype / device 必须换内存；而 `Module` 是容器，`_apply()` 遍历所有 parameter / buffer，把 `param.data` 换成搬过去的新张量，*`Parameter` 对象本身的身份保持不变*。身份不变很关键——optimizer 按对象引用持有参数，如果 `.cuda()` 换掉了 Parameter 对象，已建好的 optimizer 就全指向 CPU 上的旧对象。

#warn[
  *先 `.cuda()`，再建 optimizer。* optimizer 的 state（Adam 的 `exp_avg` / `exp_avg_sq`）是第一次 `step()` 时按当时的 device 惰性创建的。更实际的问题是恢复训练：`model.cuda()` 之后再 `optimizer.load_state_dict(...)` 才能让 state 落在正确 device；反过来先加载 CPU state 再 `model.cuda()`，optimizer state 留在 CPU，`step()` 时 device mismatch。标准顺序是 *build model，`.cuda()`，build optimizer，load 两个 state_dict*。
]

还有一条：`module.to()` *只搬 `_parameters` 和 `_buffers`*，放在 `self.__dict__` 里的普通张量搬不走——这是 `register_buffer` 存在的第二个理由。`.to(dtype)` 在 module 上也只影响浮点参数与 buffer，整型 buffer（比如 `num_batches_tracked`）会被跳过。

== 面试考点

#interview[
  *Q1*：`nn.Module` 有哪几张注册表？`nn.Parameter` 为什么不能直接用 Tensor 代替？

  A：`_parameters`、`_buffers`、`_modules`，加一组 hook 字典，`__setattr__` 按类型分流。`nn.Parameter` 的作用是*类型标记*，让 `__setattr__` 认出"这是要被 optimizer 更新的参数"。不能靠 `requires_grad` 判断，因为冻结的参数 `requires_grad=False` 但依然要进 `state_dict`、依然要被 `.cuda()` 搬走——类型标记和梯度开关必须解耦。
]

#interview[
  *Q2*：为什么 BN 的 `running_mean` 是 buffer 不是 parameter？`persistent=False` 什么时候用？

  A：它是靠 EMA 累积的数据统计量、不是梯度学出来的，所以不该进 optimizer；但推理完全依赖它，必须存进 checkpoint。"要持久化 + 不要梯度"正好是 buffer 的定义（BN 里靠梯度学的是 `weight` / `bias`）。`persistent=False` 给能从超参重算的常量用：causal mask、RoPE 的 sin/cos 表——checkpoint 更小，改 `max_seq_len` 时也不会 shape 不匹配。
]

#interview[
  *Q3*：用 python `list` 装 module 会怎样？

  A：`__setattr__` 只识别 `nn.Parameter` 和 `nn.Module`，普通 list 落到 `__dict__`，里面的 module 一个都不注册。后果是 `parameters()` 返回空（optimizer 什么都不更新）、`state_dict()` 存不下、`.cuda()` 搬不走。症状是 loss 不动或 device mismatch，*不报错*。换 `nn.ModuleList` / `nn.ParameterList`。
]

#interview[
  *Q4*：`state_dict()` 返回的是引用还是拷贝？`load_state_dict` 是 copy 还是重绑定？

  A：返回 `detach()` 后的张量，*与参数共享 storage*，`optimizer.step()` 之后值会跟着变，不能当快照用——要快照得 `{k: v.detach().clone() for ...}`。`load_state_dict` 是在 `no_grad()` 下做 `param.copy_(...)` 的*原地拷贝*、不重绑对象，所以 optimizer 持有的引用 load 后依然有效，不用重建。
]

#interview[
  *Q5*：`model.eval()` 和 `torch.no_grad()` 的区别？推理要写哪个？

  A：两件不同的事。`eval()` 只是递归置 `self.training = False`，改变 Dropout（变恒等映射）和 BatchNorm（改用 running stats 且不再更新）的*计算行为*，但照样建图、照样吃显存。`no_grad()` 只是不建图，*完全不影响* Dropout / BN。推理两个都要：只写 `no_grad` 会让 Dropout 还在丢、BN 还在污染 running stats（验证指标偏低）；只写 `eval` 结果对但显存翻倍。纯推理用 `inference_mode()`。
]

#interview[
  *Q6*：Xavier 和 Kaiming 怎么选？`nn.Linear` 默认用的是哪个？

  A：看激活是否关于 0 对称。tanh / sigmoid / 恒等用 Xavier，方差 $2/("fan_in" + "fan_out")$；ReLU 族用 Kaiming，方差 $2/"fan_in"$，那个 2 是"ReLU 砍掉一半输出、方差减半"的补偿。`nn.Linear` 默认 `kaiming_uniform_(a=sqrt(5))`，代入公式恰好等于 $U(-1/sqrt("fan_in"), 1/sqrt("fan_in"))$，这是对齐旧版 Torch 的历史遗留、*不是*为 ReLU 调好的值。
]

#interview[
  *Q7*：tied embedding 怎么写？`state_dict` 和 `parameters()` 里各是什么表现？

  A：`self.lm_head.weight = self.emb.weight`，赋的是同一个 `Parameter` 对象。`named_parameters()` 默认 `remove_duplicate=True` 只返回一个，所以 optimizer 不会更新两次；`state_dict()` *不去重*，两个 key 都在但指向同一块内存，保证 checkpoint 自描述。反向时梯度自动累加。不能写 `lm_head.weight.data = emb.weight.data`——那是两个 Parameter 共享 storage，会被更新两次。
]

#interview[
  *Q8*：为什么 `t.to("cuda")` 要接返回值，`model.to("cuda")` 不用？

  A：`Tensor` 语义不可变，换 device 必须换内存，所以 `.to()` 返回新对象、原张量不动。`Module` 是容器，`_apply()` 遍历 parameter / buffer 把 `param.data` 替换成新张量，*Parameter 对象的身份保持不变*——这是必须的，因为 optimizer 按对象引用持有参数。两个推论：`module.to()` 只搬 `_parameters` 和 `_buffers`，塞在 `self.__dict__` 里的张量搬不走；标准顺序是 model 先 `.cuda()` 再建 optimizer / 加载 optimizer state，否则 `step()` 会 device mismatch。
]
