#import "../template.typ": *

= 数据管线：Dataset / DataLoader / Sampler

数据管线是面试里最容易问出真实经验的地方：`nn.Module` 写错了 loss 会炸，数据管线写错了训练*照样收敛*，只是慢一半、或者悄悄把同一批增强样本喂了 4 遍。面试官问 `num_workers`、`pin_memory`、`set_epoch`，考的不是 API 名字，而是你有没有真的看过 GPU 利用率的锯齿波。

这一章讲三件事：`Dataset` 怎么定义"一条样本"，`DataLoader` 怎么把它变成 batch 并与训练重叠，`Sampler` 怎么决定"看哪些样本、以什么顺序"。GPU 侧的拷贝机制见第 9 章，profiler 用法见第 11 章，`DistributedSampler` 之外的分布式数据切分见第 18 章。

== 两种 Dataset：map-style 与 iterable-style

PyTorch 只有两种数据集协议，选哪种取决于*你能不能随机寻址*。

*map-style*：实现 `__getitem__(idx)` 和 `__len__()`。DataLoader 先用 Sampler 生成一串 index，再让 worker 去取。

```python
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, path):
        self.offsets = load_index(path)     # 只存偏移表，不 load 全量数据
        self.path, self.fh = path, None     # 句柄延迟到 worker 里再开

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self.fh is None:                 # 每个 worker 各开一份
            self.fh = open(self.path, "rb")
        self.fh.seek(self.offsets[idx])
        return decode(self.fh.read(...))    # 返回 tensor / dict / tuple
```

*iterable-style*：继承 `IterableDataset`，实现 `__iter__()`。适合无法预知长度、或者只能顺序读的源：Kafka / 消息队列、无限流、S3 上的 shard 文件、需要边读边解码的 webdataset。

```python
from torch.utils.data import IterableDataset, get_worker_info

class ShardStream(IterableDataset):
    def __init__(self, shards):
        self.shards = shards

    def __iter__(self):
        info = get_worker_info()
        shards = self.shards
        if info is not None:                          # 多 worker：必须自己分片
            shards = shards[info.id :: info.num_workers]
        for s in shards:
            for rec in read_records(s):
                yield rec
```

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [], [*map-style*], [*iterable-style*],
  [必须实现], [`__getitem__` + `__len__`], [`__iter__`],
  [`shuffle=True`], [可用（走 `RandomSampler`）], [不可用，DataLoader 直接报错],
  [`sampler` / `batch_sampler`], [可用], [不可用],
  [多 worker 分片], [DataLoader 按 index 自动分], [*你自己分*],
  [`len(dataloader)`], [准确], [取决于你是否实现 `__len__`（可能不准）],
  [典型场景], [本地文件、内存数据集、图像目录], [流式、无限数据、超大 shard],
)

#warn[
  `IterableDataset` + `num_workers>0` 且*没有*按 `get_worker_info()` 分片，是最高频的静默数据 bug：每个 worker 都会把 `__iter__` 从头跑一遍，于是每条样本被喂 `num_workers` 遍，epoch 长度也变成 `num_workers` 倍。loss 曲线照样下降，只是你在做 4 倍重复数据的训练。

  ```python
  # 错：4 个 worker 各自遍历全部 shards → 数据 ×4
  def __iter__(self):
      for s in self.shards:
          yield from read_records(s)
  ```

  修法有两种：*shard 级切分*（上面的 `shards[id::num_workers]`，要求 shard 数 $>=$ worker 数，否则有 worker 空转）；或*样本级切分*（`if i % num_workers == info.id`，负载均衡但每个 worker 仍要解码全部记录，白做 IO）。生产上优先 shard 级，并让 shard 数是 worker 数的整数倍。

  分布式下还要做*两级*切分：先按 `dist.get_rank()` / `world_size` 分给各进程，再按 `worker_info` 分给进程内的 worker，总切分数 `world_size * num_workers`。`DistributedSampler` 对 iterable-style 无效。
]

== DataLoader 的参数，逐个讲清

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4,
                    collate_fn=my_collate, worker_init_fn=my_init)
```

/ `batch_size`: 每批样本数。设成 `None` 表示"不自动 batch"，此时 DataLoader 直接把单条样本交给 `collate_fn`（流式场景里 dataset 自己已经 yield 好 batch 时用）。
/ `shuffle`: 只是 `sampler=RandomSampler(dataset)` 的语法糖。与 `sampler` 互斥，二者同时给会报错。分布式下不要用 `shuffle=True`，把打乱交给 `DistributedSampler`。
/ `num_workers`: 加载子进程数。`0` 表示在主进程里同步加载。
/ `pin_memory`: 把 batch 拷进 page-locked（pinned）内存，为异步 H2D 拷贝铺路，见下面 `pin_memory` 一节。
/ `drop_last`: 丢掉最后一个不足 `batch_size` 的残批。开它的三个理由：BatchNorm 遇到 batch=1 会报错；`torch.compile` 下变化的 batch size 会触发重编译；DDP 各 rank 残批大小不同会让梯度权重失衡。
/ `persistent_workers`: `True` 时 epoch 结束不杀 worker。默认 `False`，意味着*每个 epoch 都要重新 fork + 重建 dataset 状态*，小数据集上这个开销能占 epoch 时间的可观比例。
/ `prefetch_factor`: 每个 worker 最多预备几个 batch（`num_workers>0` 时默认 2）。在途 batch 总量 = `num_workers * prefetch_factor` 个 batch，这直接决定了 DataLoader 的常驻内存。
/ `collate_fn`: 把 `list[样本]` 拼成 batch，跑在 worker 进程里，见下面 collate 一节。
/ `worker_init_fn`: 在每个 worker 进程里、开始取数据*之前*调用一次，签名 `fn(worker_id)`。用来设 RNG、绑核、打开句柄。
/ `timeout`: 从 worker 队列取 batch 的超时秒数，`0` 为不超时。设个正值能把"worker 卡死"从 hang 变成异常。

#note[
  较新版本加了 `in_order` 参数，默认 `True`。设 `False` 允许 worker 乱序返回 batch，慢 worker 不再拖住整条流水线，代价是 batch 顺序不可复现。样本解码耗时方差大（比如混了超长视频）时值得试。
]

== `num_workers` 的工作原理

`num_workers>0` 时 DataLoader 的结构是一条多进程流水线：

#figure(
  align(center, flow-boxes(boxes: (
    "Sampler",
    "index_queue",
    "worker",
    "data_queue",
    "pin 线程",
  ), box-w: 2.1, box-h: 0.8)),
  caption: [
    `num_workers>0` 时的数据流。Sampler 和队列在主进程，`__getitem__` + `collate_fn` 在子进程，
    pin 线程回到主进程，最后交给训练循环。
  ],
) <fig-dl-pipeline>

+ 主进程的 Sampler 生成 index（或 index 列表），round-robin 塞进每个 worker 的 `index_queue`。
+ 每个 worker 循环：取 index → 调 `dataset[idx]` → 调 `collate_fn` 拼 batch → 把结果*写进共享内存*、把句柄放进公共的 `data_queue`。
+ 主进程（若 `pin_memory=True` 则由专门的 pin 线程）从 `data_queue` 拿到结果，按原始顺序重排后交给训练循环。
+ 主进程每消费掉一个 batch，就补发一个新 index，维持"在途 `num_workers * prefetch_factor` 个 batch"的水位。

关键点是*预取*：worker 在你还在做 `loss.backward()` 的时候就已经在准备后面的 batch 了。所以理想情况下 `next(iter(loader))` 是一次队列 pop，几乎不耗时——数据加载被完全隐藏在 GPU 计算后面。

`num_workers=0` 就没有这条流水线：`for batch in loader` 会在主进程里同步跑 `__getitem__` 和 `collate_fn`。CPU 解码和 GPU 计算严格串行，step time = 加载时间 + 计算时间。

#figure(
  align(center, timeline(streams: (
    ("CPU (nw=0)", (("compute", 6), ("wait", 8), ("compute", 6), ("wait", 8))),
    ("GPU (nw=0)", (("wait", 6), ("compute", 8), ("wait", 6), ("compute", 8))),
    ("CPU (nw=4)", (("compute", 6), ("compute", 6), ("compute", 6), ("compute", 6))),
    ("GPU (nw=4)", (("wait", 6), ("compute", 8), ("compute", 8), ("compute", 8))),
  ), title: "num_workers=0 串行 vs num_workers=4 + pin_memory 重叠")),
  caption: [
    上两行 `num_workers=0`：CPU 解码（compute）与 GPU 计算交替，谁在干活另一边就在 wait。
    下两行 4 个 worker 预取：只有第一个 batch 有冷启动的 wait，之后 GPU 背靠背跑，解码被隐藏。
  ],
) <fig-nw-timeline>

#insight[
  DataLoader 的目标不是"加载得快"，而是"加载得*不比 GPU 慢*"。只要 `解码一个 batch 的时间 / num_workers < GPU 算一个 batch 的时间`，再多加 worker 也不会让 step 变快。判断有没有到这个点，见后面"判断 dataloader 是不是瓶颈"一节。
]

== `pin_memory` 为什么能加速

普通 `torch.Tensor` 在 pageable（可换页）内存里。CUDA 的 DMA 引擎不能直接读 pageable 内存——操作系统随时可能把那些页换出去。所以 `cudaMemcpyAsync` 从 pageable 内存出发时，driver 只能先把数据*同步*拷到一块内部的 pinned 中转缓冲区，再从中转区发 DMA。结果：多一次 CPU 侧 memcpy，而且这个"异步"拷贝实际上会阻塞调用线程。

pinned（page-locked）内存被锁定在物理页上，DMA 引擎可以直接读，`cudaMemcpyAsync` 才真正是异步的：立即返回，拷贝在 copy engine 上与 compute kernel 并行。

```python
loader = DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True)

for x, y in loader:
    x = x.to("cuda", non_blocking=True)     # 必须配 non_blocking
    y = y.to("cuda", non_blocking=True)
    out = model(x)                          # 排在同一 stream，自动等拷贝完
```

#warn[
  `pin_memory=True` 和 `non_blocking=True` *必须成对出现*，单独用哪一个都没有意义：

  - 只开 `pin_memory`，`.to(device)` 默认 `non_blocking=False` → 仍然同步等待，你只是白花了 pin 的开销。
  - 只开 `non_blocking`，源是 pageable 内存 → driver 走中转缓冲，实际仍然同步。这是最容易自欺欺人的一种"优化"。

  另外：pinned 内存是稀缺的物理资源。`num_workers * prefetch_factor * batch 大小` 全都常驻 pinned，几十 GB 的 pin 会拖慢整个系统的内存分配，甚至触发 OOM killer。数据量大时先降 `prefetch_factor`。
]

pin 操作由主进程里*一个专门的线程*做，不在 worker 里做（worker 里 pin 的内存没法跨进程传）。所以这一步是单线程的，超大 batch 下 pin 本身也可能成为瓶颈——此时让 dataset 返回 uint8 而不是 float32、把归一化搬到 GPU 上做，能直接把要 pin 的字节数降到 1/4。

== worker 的四类坑

*进程启动开销。* 默认 `persistent_workers=False`，每个 epoch 的第一个 `iter()` 都要 fork 出 `num_workers` 个进程、重跑 `worker_init_fn`、重建 dataset 内部状态（重开文件、重连数据库）。数据集小、epoch 短时这个固定开销很显眼。开 `persistent_workers=True` 即可，代价是 worker 常驻内存。

*shared memory 不足。* worker 通过 `/dev/shm` 把 tensor 传回主进程。容器里 `/dev/shm` 默认常常只有 64 MB，一旦装不下就报：

```text
RuntimeError: DataLoader worker (pid 12345) is killed by signal: Bus error.
It is possible that dataloader's workers are out of shared memory.
```

修法：`docker run --shm-size=16g`（或 `--ipc=host`）；或降 `num_workers` / `prefetch_factor` / `batch_size`。用 `df -h /dev/shm` 确认实际大小。

*`num_workers` 过大。* worker 数超过物理核数后开始 CPU 争抢：context switch 变多、每个 worker 变慢、总吞吐反而下降。而且每个 worker 都是完整的 Python 进程，dataset 里如果 hold 了大对象（比如整个 `list` 的 Python str），内存是 `num_workers` 倍。经验起点：`num_workers = min(物理核数 / 每卡进程数, 8)`，然后实测调整。

#warn[
  *fork 与 CUDA 不兼容。* Linux 上 DataLoader 默认用 `fork` 起 worker。CUDA context 无法跨 fork 继承，所以 worker 里一旦碰 CUDA 就报：

  ```text
  RuntimeError: Cannot re-initialize CUDA in forked subprocess.
  To use CUDA with multiprocessing, you must use the 'spawn' start method
  ```

  实践规则：`__getitem__` 里*只返回 CPU tensor*，`.cuda()` 一律在训练循环里做。dataset 也不要在 `__init__` 里 hold CUDA tensor（fork 后子进程拿到的是无效句柄）。真要在 worker 里用 GPU（比如 GPU 解码），得传 `multiprocessing_context="spawn"`，代价是启动慢很多、dataset 必须可 pickle。
]

#note[
  `torch.set_num_threads(1)` 已经在 worker 里自动设好了。但如果 dataset 用了 OpenCV / numpy 的多线程后端，还要手动 `cv2.setNumThreads(0)`、设 `OMP_NUM_THREADS=1`，否则 8 个 worker 各开 8 个线程会把机器打满。
]

== 随机性：多 worker 下的 RNG

DataLoader 每次建迭代器时，从主进程的 `generator` 抽一个 `base_seed`，然后在 worker $i$ 里执行：

```python
seed = base_seed + worker_id
random.seed(seed)                                  # Python 内置 random
torch.manual_seed(seed)                            # torch CPU RNG
np.random.seed(_generate_state(base_seed, worker_id))   # numpy 全局 RNG
```

所以 `random`、`torch`、`np.random` 这三个*全局* RNG 在各 worker 里是不同的。`torch.initial_seed()` 在 worker 里返回的就是上面那个 `seed`，`get_worker_info().seed` 也是它。

#warn[
  "多 worker 下 numpy 增强重复" 这个经典 bug，在现代 PyTorch 里*不是*由 `np.random` 全局 RNG 引起的（它已经被逐 worker 播种了）。真正还会踩的是 *fork 前就创建好的 RNG 对象*：

  ```python
  class BadAug(Dataset):
      def __init__(self):
          self.rng = np.random.default_rng(0)     # ← fork 前建好

      def __getitem__(self, i):
          # 每个 worker 继承同一份 rng 状态 → 各 worker 产出完全相同的抖动序列
          return self.rng.random(3)
  ```

  同理还有 `__init__` 里建的 `torch.Generator()`、`random.Random(0)` 实例，以及某些第三方增强库在 import 时快照的 RNG。判断标准很简单：*RNG 状态是不是在 fork 之前就固定了*。

  修法一：在 `worker_init_fn` 里按 worker 重新播种。

  ```python
  def worker_init_fn(worker_id):
      info = torch.utils.data.get_worker_info()
      info.dataset.rng = np.random.default_rng(info.seed)   # info.dataset 是本 worker 的副本
  ```

  修法二：不持有 RNG 对象，`__getitem__` 里直接用 `np.random` / `torch.rand` 全局接口。

  修法三（最干净）：让随机性只依赖 index，`np.random.default_rng((epoch, idx))`，增强结果可复现、与 worker 数无关。
]

要让整条管线可复现，需要固定三处：DataLoader 的 `generator`（决定 shuffle 顺序和 `base_seed`）、`worker_init_fn`（决定 worker 内 RNG）、以及全局的 `torch.manual_seed`。完整的确定性讨论见第 10 章。

```python
g = torch.Generator()
g.manual_seed(1234)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
                    generator=g, worker_init_fn=worker_init_fn)
```

== `collate_fn`：从样本列表到 batch

默认的 `default_collate` 按类型递归处理一个 `list[样本]`：

- `Tensor` → `torch.stack(batch, 0)`，*要求所有样本 shape 完全一致*
- numpy array → 先转 tensor 再 stack；`int` / `float` → 转成 1-D tensor
- `str` / `bytes` → 原样保留成 list（不转 tensor）
- `Mapping`（dict）→ 逐 key 递归，输出同样 key 的 dict
- `Sequence`（tuple / list）→ 先转置再逐位置递归：`[(x1,y1),(x2,y2)]` 变成 `(stack([x1,x2]), stack([y1,y2]))`
- namedtuple → 保留类型，逐字段递归

`__getitem__` 返回 dict 时特别顺手，因为 `default_collate` 会自动把 `{"input": t, "label": i}` 拼成 `{"input": (B,...), "label": (B,)}`。

需要自定义 collate 的最常见理由是*变长序列*——shape 不一致，`stack` 直接报错。

#figure(
  align(center, shape-pipeline(stages: (
    ("样本列表", "B × (L_i,)", "每条长度不同"),
    ("pad-sequence", "(B, L_max)", "右侧补 pad_id"),
    ("attention mask", "(B, L_max)", "真实 token 处为 True"),
  ))),
  caption: [变长序列 collate 的 shape 变化。`L_max` 是这个 batch 内的最大长度，不是全局最大长度。],
) <fig-collate>

```python
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch, pad_id: int = 0):
    """batch: list of dict(input_ids=LongTensor(L_i,), label=int)"""
    seqs = [b["input_ids"] for b in batch]
    lengths = torch.tensor([s.numel() for s in seqs])

    # batch_first=True → (B, L_max)；只 pad 到本 batch 的最大长度
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn_mask = torch.arange(input_ids.size(1))[None, :] < lengths[:, None]

    return {
        "input_ids": input_ids,              # (B, L_max) long
        "attention_mask": attn_mask,         # (B, L_max) bool
        "lengths": lengths,                  # (B,) long，给 pack_padded_sequence 用
        "labels": torch.tensor([b["label"] for b in batch]),
    }

loader = DataLoader(ds, batch_size=32, collate_fn=pad_collate, num_workers=4)
```

三个细节值得在面试里主动说：

+ *只 pad 到本 batch 最大长度*，不是全局最大长度。配合下面的 bucketing，padding 浪费能降一个数量级。
+ `collate_fn` 是在 *worker 进程*里执行的（`num_workers>0` 时），所以把 padding、tokenize 这类 CPU 活放这里是免费的并行；反过来说，collate 里绝对不能碰 CUDA。
+ mask 的 dtype 用 `bool`，别用 float——`scaled_dot_product_attention` 和 `masked_fill_` 都吃 bool mask，float mask 语义还容易和 additive mask 搞混。

想在 collate 里丢掉坏样本（解码失败时 `__getitem__` 返回 `None`）：`batch = [b for b in batch if b is not None]`，空 batch 返回 `None` 并在训练循环里 `continue`。代价是 batch size 会变，与 `torch.compile` 的静态 shape 假设冲突。

== Sampler：决定看哪些样本、什么顺序

Sampler 产出的是 *index*，不是数据。DataLoader 里的层次是：`Sampler` 出单个 index → `BatchSampler` 攒成 index 列表 → worker 按列表取样本 → `collate_fn` 拼 batch。

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*Sampler*], [*语义与用途*],
  [`SequentialSampler`], [`range(len(ds))`。`shuffle=False` 时的默认值，验证集用它],
  [`RandomSampler`], [`shuffle=True` 时的默认值。`replacement=False` 是无放回全排列；开 `replacement=True` 可配 `num_samples` 做有放回过采样],
  [`WeightedRandomSampler`], [按*逐样本*权重有放回采样，类别不均衡的标准解法],
  [`SubsetRandomSampler`], [在给定 index 子集里随机采，做 train/val split 很方便],
  [`BatchSampler`], [把上面任一个包一层，一次吐一个 index list。传给 `batch_sampler=` 参数],
  [`DistributedSampler`], [按 rank 取交错分片，DDP 必用],
)

*类别不均衡：* `WeightedRandomSampler` 的 `weights` 是长度等于数据集大小的*逐样本*权重，不是逐类别权重。最常见的写法是"权重 = 1 / 该样本所属类别的样本数"：

```python
from torch.utils.data import WeightedRandomSampler

labels = torch.tensor([ds[i]["label"] for i in range(len(ds))])   # 或从元信息直接读
counts = torch.bincount(labels)                     # (num_classes,)
weights = 1.0 / counts[labels].float()              # (N,) 逐样本权重

sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
loader = DataLoader(ds, batch_size=64, sampler=sampler)   # 注意：不能再传 shuffle=True
```

`replacement=True` 意味着一个 epoch 内少数类样本会被重复看到、多数类样本可能一次都没看到。它和"给 loss 加 class weight"是两种思路：前者改数据分布（每个 batch 内类别均衡，BatchNorm 统计更稳），后者改梯度权重（不重复数据，但 batch 内仍不均衡）。面试里能说出这个对比就够了。

*`DistributedSampler` 的机制*，源码只有几行但每一行都是考点：

```python
# rank r, world_size W, epoch e
g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
indices = torch.randperm(len(dataset), generator=g).tolist()   # 各 rank 结果相同

if not drop_last:
    indices += indices[: total_size - len(indices)]   # 重复开头几条，padding 到整除
else:
    indices = indices[:total_size]                    # 截掉尾巴

indices = indices[rank : total_size : W]              # 交错取，步长 W
```

两个要点：

+ *各 rank 用同一个 seed 做同一个 `randperm`*，再按 `[rank::W]` 交错切片。这保证了不重不漏，且不需要任何通信。
+ 各 rank 的分片必须*等长*，否则先跑完的 rank 会让其他 rank 在 AllReduce 上永久 hang。所以 `drop_last=False`（默认）会重复开头的若干条样本补到 `total_size`，`drop_last=True` 则丢掉尾部不足整除的部分。

#warn[
  *忘记 `sampler.set_epoch(epoch)` 是最高频的 DDP 数据 bug。* seed 是 `self.seed + self.epoch`，而 `self.epoch` 初始是 0 且*不会自己变*。不调 `set_epoch`，每个 epoch 的 `randperm` 完全一样 → 你的模型在按*固定顺序*反复看同一串数据，shuffle 形同虚设。

  ```python
  sampler = DistributedSampler(ds, shuffle=True)
  loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=8)

  for epoch in range(epochs):
      sampler.set_epoch(epoch)      # ← 必须在建迭代器之前调
      for batch in loader:
          ...
  ```

  它没有任何报错或警告，只表现为"收敛比单卡差一点"。面试里被问到"DDP 训练效果不如单卡，你怎么排查"，`set_epoch` 应该是你前三个怀疑对象之一。
]

#warn[
  *验证集上 `drop_last=False` 会让 metric 偏。* padding 出来的重复样本会被算进 accuracy / loss 的分母，各 rank AllReduce 求平均后结果比真值略偏。严谨做法：验证时每个 rank 记录 `(sum, count)` 而不是均值，AllReduce 后再相除；或者手动截掉 `rank` 分片里超出 `len(dataset)` 的那几条。
]

== 判断 dataloader 是不是瓶颈

按代价从低到高排查：

+ *看 GPU 利用率波形。* `nvidia-smi dmon` 或 `nvitop`，如果 util 是"100% → 0% → 100%"的锯齿而不是稳定平台，GPU 在等数据。
+ *把 dataset 换成常量。* 最直接的判据：用一个 `__getitem__` 返回预生成 tensor 的假 dataset 跑同样的循环。step time 明显变快 → 瓶颈在数据；几乎不变 → 瓶颈在模型。这个实验 5 行代码，比读 profiler 快得多。
+ *`torch.profiler` 看 gap。* trace 里两个 step 之间有一大段 GPU 空白，且 CPU 侧停在 `DataLoader.__next__` / `_get_data` 上，就是在等 worker 出货。
+ *拉 `prefetch_factor` 看变化。* 如果只有前几个 step 慢、后面就跟上了，说明只是冷启动，不是持续瓶颈。

确认是数据瓶颈后，按*收益 / 改动成本*排序去修：

#ladder(
  ("加 num_workers", "多进程并行解码", "改一个数字，先做这个"),
  ("persistent_workers + prefetch_factor", "消除 epoch 冷启动、加深队列", "改参数，代价是内存"),
  ("pin_memory + non_blocking", "H2D 真异步，与 compute 重叠", "两行代码"),
  ("轻量化返回值", "返回 uint8，归一化搬到 GPU", "改 dataset + 训练循环"),
  ("预处理成二进制格式", "webdataset / tfrecord / npy memmap，去掉逐文件 IO 和解码", "要跑一次离线转换"),
  ("GPU 解码", "nvJPEG / DALI / torchcodec，解码搬上 GPU", "引入依赖，吃 GPU 算力"),
)

#insight[
  绝大多数"dataloader 慢"最后都归结到两件事之一：*小文件随机 IO*（几百万个 jpg，每次 open + seek）或 *Python 层解码*（PIL、json.loads）。前者的正解是打包成大 shard 顺序读，后者的正解是离线预处理成能 `memmap` 直接读的二进制。加 worker 只是把这两个问题并行化，治不了根。
]

== padding 浪费：bucketing 与 packing

变长序列 batch 里，`pad_sequence` 把所有样本补到本 batch 最长。如果一个 batch 里混了长度 8 和长度 512 的样本，绝大部分算力花在 pad token 上——它们的输出会被 mask 掉，纯属白算。

*bucketing（长度分桶）*：让长度相近的样本进同一个 batch。实现方式是自定义 `BatchSampler`：先按长度粗排序（加一点随机扰动以免每个 epoch batch 组成完全固定），再切成 batch，最后把 batch 的顺序打乱。

```python
class LengthBucketSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, batch_size, pool=100):
        self.lengths, self.bs = lengths, batch_size
        self.pool = pool * batch_size                     # 局部排序窗口

    def __iter__(self):
        idx = torch.randperm(len(self.lengths)).tolist()   # 先全局打乱
        batches = []
        for i in range(0, len(idx), self.pool):
            chunk = sorted(idx[i : i + self.pool], key=lambda j: self.lengths[j])
            batches += [chunk[k : k + self.bs] for k in range(0, len(chunk), self.bs)]
        yield from (batches[i] for i in torch.randperm(len(batches)).tolist())

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs

loader = DataLoader(ds, batch_sampler=LengthBucketSampler(lens, 32),
                    collate_fn=pad_collate, num_workers=4)
```

*packing（序列拼接）*：更彻底——把多条短序列首尾拼进一个固定长度 `L` 的序列，padding 几乎为零，每个 batch 的 token 数恒定（对 `torch.compile` 的静态 shape 也友好）。代价是需要防止 attention 跨越样本边界：要么传 block-diagonal mask，要么用 FlashAttention 的 varlen 接口（`cu_seqlens` 累积长度）。LLM 预训练几乎都用 packing，SFT 阶段则要格外小心 loss mask 别把上一条样本的 label 算进来。

#note[
  bucketing 让每个 batch 的 token 数不固定，配固定 `batch_size` 会导致显存忽高忽低（长 batch OOM）。生产上常按 *token 预算* 组 batch：`batch_size = max_tokens // 本 batch 最长长度`。这需要 `batch_sampler` 而不是 `batch_size`。
]

== 面试考点

#interview[
  *Q1*：`Dataset` 和 `IterableDataset` 的区别？各自什么时候用？

  A：map-style 实现 `__getitem__` + `__len__`，支持随机寻址，DataLoader 用 Sampler 生成 index 并自动在 worker 间分配；iterable-style 实现 `__iter__`，只能顺序读，不支持 `shuffle` / `sampler`。本地文件、内存数据集用前者；Kafka、无限流、超大 shard 用后者。后者在多 worker 下必须自己按 `get_worker_info()` 分片。
]

#interview[
  *Q2*：`IterableDataset` 配 `num_workers=4` 会发生什么？

  A：如果不分片，4 个 worker 各自把 `__iter__` 从头跑一遍，每条样本被喂 4 遍，epoch 变成 4 倍长，而且*完全不报错*。正确写法是在 `__iter__` 里读 `get_worker_info()`，按 `shards[info.id::info.num_workers]` 切分。分布式下还要再叠一层按 rank 的切分，总切分数是 `world_size * num_workers`。
]

#interview[
  *Q3*：`num_workers` 设多少合适？设太大有什么问题？

  A：起点是 `min(物理核数 / 每卡进程数, 8)`，然后实测。太大有三个问题：worker 数超过核数后 CPU 争抢导致每个 worker 都变慢；每个 worker 是完整 Python 进程，内存是 `num_workers` 倍；在途 batch 数 `num_workers * prefetch_factor` 变大，`/dev/shm` 和 pinned 内存吃紧。判断"够不够"的标准是 GPU util 是否稳定，不是 worker 越多越好。
]

#interview[
  *Q4*：`pin_memory=True` 为什么能加速？单独开它有用吗？

  A：DMA 引擎不能直接读 pageable 内存，driver 得先同步拷到内部 pinned 中转区，所谓的异步拷贝实际会阻塞。pinned 内存锁定物理页，`cudaMemcpyAsync` 才真异步，能与 compute overlap。必须配 `.to(device, non_blocking=True)` 才生效——只开 `pin_memory` 而 `.to()` 是同步的，等于白花 pin 的开销；只开 `non_blocking` 而源是 pageable，driver 仍走同步路径。
]

#interview[
  *Q5*：报 `DataLoader worker is killed by signal: Bus error` 怎么排查？

  A：几乎总是 `/dev/shm` 不够——worker 通过共享内存把 tensor 传回主进程，容器里 `/dev/shm` 默认可能只有 64 MB。修法：`docker run --shm-size=16g` 或 `--ipc=host`；或降 `num_workers` / `prefetch_factor` / `batch_size`。另一类可能是 worker 里真的 OOM 或 segfault，用 `num_workers=0` 复现就能看到真实 traceback。
]

#interview[
  *Q6*：多 worker 下数据增强重复是怎么回事？

  A：现代 PyTorch 已经给每个 worker 分别播种了 `random`、`torch` 和 `np.random` 三个全局 RNG（`seed = base_seed + worker_id`），所以用全局接口不会重复。还会踩坑的是 *fork 之前就创建好的 RNG 对象*——比如在 `Dataset.__init__` 里 `self.rng = np.random.default_rng(0)`，fork 后每个 worker 继承同一份状态，产出完全相同的随机序列。修法是在 `worker_init_fn` 里按 `get_worker_info().seed` 重新播种，或者让随机性只依赖 `(epoch, idx)`。
]

#interview[
  *Q7*：写一个变长序列的 `collate_fn`。

  A：用 `pad_sequence(seqs, batch_first=True, padding_value=pad_id)` 补到*本 batch* 最大长度，同时用 `arange(L_max)[None,:] < lengths[:,None]` 造 bool attention mask，并把 `lengths` 一起返回。关键点：只 pad 到 batch 内最大长度而不是全局最大；`collate_fn` 跑在 worker 进程里所以 CPU 开销是并行的；mask 用 bool dtype。
]

#interview[
  *Q8*：`DistributedSampler` 是怎么保证各 rank 不重不漏的？为什么必须 `set_epoch`？

  A：所有 rank 用同一个 `seed + epoch` 做同一个 `randperm`，然后各自取 `indices[rank::world_size]` 交错分片，不需要通信就能保证不重不漏。`set_epoch(epoch)` 是唯一改变 seed 的途径，不调它 `self.epoch` 永远是 0，每个 epoch 的顺序完全相同，shuffle 失效。这个 bug 不报错，只表现为收敛变差。
]

#interview[
  *Q9*：`DistributedSampler` 的 `drop_last` 有什么讲究？

  A：`drop_last=False`（默认）会重复开头的若干样本，把总数 padding 到 `world_size` 的整数倍，因为 DDP 要求各 rank 步数完全相同——否则先跑完的 rank 会让其他 rank 在 AllReduce 上 hang。训练时这几条重复样本无所谓；*验证时会污染 metric*，应该改成各 rank 上报 `(sum, count)` 再 AllReduce 相除，或手动截掉多出来的样本。
]

#interview[
  *Q10*：怎么确认瓶颈在 dataloader，确认后按什么顺序优化？

  A：最快的判据是把 dataset 换成返回常量 tensor 的假 dataset，step time 明显变快就是数据瓶颈；辅助证据是 GPU util 呈锯齿、profiler trace 里 step 之间有 GPU 空白且 CPU 卡在 `DataLoader.__next__`。优化顺序：加 `num_workers` → `persistent_workers` + `prefetch_factor` → `pin_memory` + `non_blocking` → 返回 uint8 把归一化搬 GPU → 离线预处理成二进制 shard → GPU 解码（DALI / nvJPEG）。前三步改参数，后三步改架构。
]