#import "../template.typ": *

= Data Loader, Packing 与 Sample Balancing

Data loader 常常被面试忽略，但它决定了训练吞吐的上限——GPU 再快，data 供不上就白算。这一章讲 IterableDataset、streaming、packing 算法 (BFD/FFD)、多源混合、shard 均衡、以及多模态特有的问题。

== 一个隐藏的 bottleneck

大规模训练里最常见的 "step time 突然变慢 10%" 原因不是通信，而是 dataloader 供不上：

+ *磁盘 IO 慢*：checkpoint / 数据在同一 NFS，同时读时被拖
+ *CPU 预处理慢*：tokenize + pack + augment 单进程算不过来
+ *DDP sampler 不均*：某几个 rank 拿到超大 batch (可变长 seq)
+ *网络 shuffle 抖动*：streaming 从 S3 拉数据抖

诊断：nsys 看是否有 gap，或加时间戳 `time.time()` 在 `next(loader)` 前后测。生产训练常见 loader wait > 5% step time。目标应该 < 1%。

== 数据集抽象

*Map-style Dataset*：`__getitem__(idx)` 随机访问，`__len__` 已知。适合小数据集（几十 GB fit in memory or fast SSD）。

*IterableDataset*：`__iter__` 返回 iterator，无 `__len__`。适合流式数据（TB+ 无法 index）、多源混合、动态生成。生产大模型训练都用 IterableDataset。

*难点*：DistributedSampler 只能用于 map-style。IterableDataset 需要自己实现 shard 划分 + rank-aware iteration:

```python
class MyStreamingDataset(IterableDataset):
    def __init__(self, urls):
        self.urls = urls   # list of shard files

    def __iter__(self):
        rank = dist.get_rank(); world = dist.get_world_size()
        worker = torch.utils.data.get_worker_info()
        n_workers = worker.num_workers if worker else 1
        wid = worker.id if worker else 0

        my_urls = self.urls[rank::world]         # 按 rank 分 shard
        my_urls = my_urls[wid::n_workers]        # 再按 worker 分
        for url in my_urls:
            for sample in read_shard(url):
                yield sample
```

*坑*：如果 shard 数不能被 world × workers 整除，某些 rank 分到少 shard → epoch 提早结束 → 训练 hang（其他 rank 等）。修：设 `drop_last` 语义或补 pad。

== Packing：变长序列的显存/算力救星

Transformer 训练里 seq 长度往往不等长（一句话 128 tokens，一个 code file 8K tokens）。naive batching：

```
batch = [seq1 (128 tok), seq2 (8000 tok), seq3 (400 tok)]
padded to (3, 8000) — 大部分是 pad token, 浪费 90%+ 的 compute
```

*Packing*：把多个短 seq *拼接*成一个长 seq，attention mask 保证不跨 doc attention：

```
packed = [seq1, seq2, seq3, seq4, ...]  concatenated to length 8192
mask sequence: [1,1,...,1, 2,2,...,2, 3,3,...,3, ...]   # doc id
```

Attention 里用 `attention_mask` 或 FlashAttention 的 `varlen` 接口只让同 doc 内的 tokens attend。

*收益*：训练时 padding ratio 从 40% 降到 \< 5%，吞吐 +50-100%。

=== Packing 算法

*Best Fit Decreasing (BFD)*：按 seq 长度降序，每 seq 找当前"最能装下且剩余空间最小"的 bin (packed sequence)。经典 bin-packing 算法。$O(n log n)$。

*First Fit Decreasing (FFD)*：按 seq 长度降序，放到第一个能装下的 bin。$O(n)$。

*Online / streaming packing*：不知道全部 seq 长度分布，边读边 pack。做法：维护一个 "open bins" 池，读到新 seq 时找最合适 bin；bin 满就 flush。DeepSeek / Mixtral 训练用这个。

*Multi-turn conversation packing*：对话数据里保持 turn 顺序，但可以跨 conversation pack。要小心 loss mask（只对 assistant token 算 loss，user token mask 掉）。

=== Attention mask 的处理

*Naive*：$B S times B S$ mask matrix，$S=8192$ 时 128 MB per sample —— OOM。

*FlashAttention varlen*：

```python
from flash_attn import flash_attn_varlen_func
# packed: 1D concatenated tokens
# cu_seqlens: cumulative seq lens, e.g. [0, 128, 8128, 8528, 8929, ...]
out = flash_attn_varlen_func(
    q=q_packed, k=k_packed, v=v_packed,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len, max_seqlen_k=max_len,
    causal=True,
)
```

无需 mask，kernel 内部按 `cu_seqlens` block 分别做 attention。$O(B S H)$ memory instead of $O(B S^2)$.

*Cross-doc contamination*：如果 pack 错了让 doc 之间 attend，模型会学到"下一个 doc 的信息 predict 当前 doc" → 训练 val loss 偏低但 test 崩。生产必须验证 packing 正确性。

== Padding vs Packing vs No-pad (Bucket)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, auto, left),
    [*策略*], [*方法*], [*Padding 比*], [*说明*],
    [Naive pad], [batch 内 pad 到 max len], [30-70%], [最简单，浪费多],
    [Bucketing], [按长度分桶，桶内 pad], [10-30%], [每桶 pad 少，但 batch shape 不定],
    [Packing (BFD)], [拼接短 seq 到固定长], [\< 5%], [生产标配, 需 varlen attention],
    [Packing (dynamic)], [长度平衡的 pack], [\< 5%], [进一步减少每 batch 长度方差],
  ),
  kind: table,
  caption: [变长处理策略。生产 LLM 训练几乎全用 packing + FA varlen。],
)

== 多源数据混合

预训练常混多个 corpus：Web (Common Crawl) + Code (GitHub) + Books + arXiv + Wiki + Math。每类比例（domain weight）影响下游性能。

*采样策略*：

+ *Uniform*：每 source 相同概率
+ *Weighted*：预设 weight (Llama-2: 67% Web, 15% code, 4.5% arXiv, ...)
+ *Temperature-based*：$p_i prop n_i^tau$，$tau in [0, 1]$；$tau=0$ 均匀，$tau=1$ 按 size 采样。多语言场景常用。
+ *DoReMi / Data Mixing Laws*：online 学习最优 weight

实现：多个 IterableDataset，用 `torch.utils.data.WeightedRandomSampler` 或自己写 iterator：

```python
def mixed_iterator(sources, weights):
    iters = [iter(s) for s in sources]
    while True:
        idx = np.random.choice(len(sources), p=weights)
        try:
            yield next(iters[idx])
        except StopIteration:
            iters[idx] = iter(sources[idx])   # infinite loop
            yield next(iters[idx])
```

*Reproducibility*：seed 每 rank 不同 (`rank + epoch`), 保证不同 rank 数据不重复但全局按 weight 分布。

== Streaming: 从 S3 / HDFS 拉数据

10T tokens 数据集不可能全下载到本地。生产用 streaming：

+ *mosaicml Streaming* (MDS format)：sharded, seek-able, prefetch
+ *WebDataset*：tar-based, streaming friendly
+ *HuggingFace datasets `streaming=True`*：Arrow-based, 简单但性能弱

关键点：

+ *Prefetch*：`DataLoader(num_workers=N, prefetch_factor=M)`，N=8-16, M=4，让 CPU 一直提前准备下 batch
+ *S3 并发*：多 worker 各自 open connection，注意 S3 rate limit
+ *断点续训*：streaming 要能 resume。MDS 支持 `.set_epoch(epoch)` + `.state_dict()` 保存 iterator 状态

*Shuffle*：全局 shuffle 不可能 (TB 数据)。做 shard-level shuffle + shard 内 buffer shuffle:

```python
shuffle_buffer = 10000
buf = []
for sample in shard_iter:
    buf.append(sample)
    if len(buf) >= shuffle_buffer:
        idx = random.randint(0, len(buf)-1)
        yield buf.pop(idx)
```

Buffer 越大越接近真 shuffle 但 memory 越多。生产用 10K-100K。

== 长文档 vs 短文档：DDP 里的 sample-level imbalance

如果 batch 里长文档过多，某个 rank 拿到超长 seq → 该 rank compute 时间 2×，DP AllReduce 里其他 rank 都等它。

*Sequence-length balancing*：给每 rank 分*近似等 length total* 的 samples。

方法：
+ *Fixed-length after packing*：packing 输出恒定 seq_len，天然均衡（首选）
+ *Length-based grouping*：把长度类似的 sample 分给同 rank
+ *Dynamic batching*：`max_tokens_per_batch=8192` 而非 `batch_size=8`（HF trainer 支持）

不做 sequence balancing 的后果：DP AR 里 straggler 主导 step time，MFU 掉 20-30%。

== 开了 SP/CP 之后的 DataLoader

一旦启用 sequence sharding（Megatron SP 或 CP），DataLoader 的契约变了：*一个 CP 组合起来才持有一条序列*。三条硬性要求：

+ *组内同样本*：sampler 索引和 shuffle 种子都必须用 `dp_rank`，不能用 `global_rank`。写错不报错——attention 会把两个半截文档当一条序列算，同时 global batch size 悄悄变成配置值的 CP 倍
+ *长度对齐*：`seq_len` 要能被 CP 整除；zigzag（负载均衡切分）需要被 $2 times "CP"$ 整除，padding 要按这个粒度补
+ *位置与文档信息随 token 走*：`position_ids`、`doc_ids`（packing 用）、以及 zigzag 的置换索引都必须作为 batch 的显式字段产出并按同样方式切分。模型里任何地方都不要用 `arange` 现场生成本地位置——varlen kernel 的 `cu_seqlens` 也要按本地 shard 重算

完整的规则、失败征状与逐层对拍方法见 §7，代码见 `20_cp_dataloader_halo.py`。

== Multimodal Data Loader 的特殊难度

见下章多模态细节，这里只列关键差异：

+ *样本 size 方差大*：一段 3B tokens 的 audio + 一张 32×32 图片，size 差 $10^6$
+ *异构类型*：需要 tokenizer 之外的 preprocessing (image resize, video frame extract, audio spectrogram)
+ *缓存友好性*：图像 decoded 是巨大 tensor，pre-cache 到 SSD
+ *Multi-worker safe*：图像/视频解码库有些不 thread-safe，多 worker 会 crash

== 一个可用的 DataLoader 骨架

```python
import torch
from torch.utils.data import IterableDataset, DataLoader

class PackedTextDataset(IterableDataset):
    def __init__(self, tokenizer, urls, max_len=8192, mix_weights=None):
        self.tokenizer = tokenizer
        self.urls = urls
        self.max_len = max_len
        self.mix_weights = mix_weights

    def __iter__(self):
        rank  = dist.get_rank(); world = dist.get_world_size()
        wi    = torch.utils.data.get_worker_info()
        w_id  = wi.id if wi else 0
        w_num = wi.num_workers if wi else 1

        my_urls = self.urls[rank::world][w_id::w_num]
        buf = []
        cur_len = 0
        for url in my_urls:
            for text in read_url(url):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if cur_len + len(tokens) + 1 > self.max_len:
                    # yield current packed batch
                    yield self._pack(buf)
                    buf = []; cur_len = 0
                buf.append(tokens)
                cur_len += len(tokens) + 1   # +1 for EOS

    def _pack(self, seqs):
        # concat with EOS, build cu_seqlens for FA varlen
        input_ids  = []
        cu_seqlens = [0]
        for s in seqs:
            input_ids += s + [self.tokenizer.eos_token_id]
            cu_seqlens.append(len(input_ids))
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        }

def collate(batch):
    # batch of dicts, stack input_ids, keep cu_seqlens per-sample
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "cu_seqlens": [b["cu_seqlens"] for b in batch],
    }

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate,
)
```

要点：
- `persistent_workers=True` 避免每 epoch 重启 worker
- `pin_memory=True` D2H 快
- `prefetch_factor` 让 worker 提前准备

== 面试考点

#interview[
  *Q1*: 什么是 sequence packing？和 batch padding 差在哪？

  A: Padding 是每 batch 内 pad 到 max seq length，浪费很多 compute (padding tokens 都被算但 loss=0)。Packing 是把多个短 seq 拼接成固定长的 packed sequence，用 attention mask (或 FA varlen 的 cu_seqlens) 保证 doc 间不 attend。padding ratio 从 30-70% 降到 \< 5%，训练吞吐 +50-100%。
]

#interview[
  *Q2*: Packing 里怎么保证 doc 间不 cross-attend？

  A: 两种：(a) 显式 attention_mask matrix — 简单但 $O(S^2)$ 显存；(b) FlashAttention varlen — 传 cu_seqlens，kernel 内部按 seq 边界 block-diagonal attention，$O(S)$ 显存。生产都用 (b)。RoPE 的 position id 也要按 doc 重置（每 doc 从 0 开始），不然 pos embedding 跨 doc 混。
]

#interview[
  *Q3*: BFD 和 FFD 算法的差别？

  A: 都是 bin-packing。BFD (Best Fit Decreasing)：按 seq 长度降序，每 seq 放到"最能装下且剩余空间最小"的 bin—— 平均更 tight 但 $O(n log n)$。FFD (First Fit Decreasing)：放到第一个能装下的—— $O(n)$，实际差别 < 3% padding。生产多用 online / streaming 版：不知全部长度，边读边 pack。
]

#interview[
  *Q4*: IterableDataset 里 rank 分 shard 常见的 bug？

  A: (1) shard 数不能被 world × workers 整除，某些 rank 少数据 → epoch 早停 → 训练 hang；(2) 忘了 `worker_init_fn`，多 worker 拉同一 shard；(3) shard 内没 shuffle → 样本顺序固定；(4) resume 时 iterator 状态未保存 → 重新训相同 sample；(5) 长 seq 分给某 rank 造成负载不均。
]

#interview[
  *Q5*: DDP 里如果每 rank 的 batch 长度差异大会怎样？

  A: DP AllReduce 是同步的，慢的 rank 拖慢整个 step。长 seq rank compute 2×，其他 rank 等 → MFU 掉 20-30%。解决：packing 让每 batch 长度恒定；或 length-based grouping 让相似长度 sample 分给同 rank；或 dynamic batching 按 `max_tokens_per_batch` 决定 batch size。
]

#interview[
  *Q6*: 一个 dataloader 供不上 GPU 怎么排查？

  A: 加时间戳测 `time.time()` 在 `next(loader)` 前后，判断 loader wait 占 step 比例。> 5% 就要优化。检查：(a) `num_workers` 太少 (< 8)；(b) `prefetch_factor=2` 默认不够，调 4-8；(c) tokenizer 单进程慢——用 fast tokenizer；(d) 数据在慢盘（NFS/S3 抖）；(e) preprocess 太重（大图 decode），预先缓存。
]

#interview[
  *Q7*: 数据混合的 weight 怎么设？

  A: 生产做法：(a) 按 domain 大小 + 质量启发式（Llama-2 report 里有）；(b) DoReMi (Xie 2023) — 用小 proxy 模型学 optimal weight；(c) Data Mixing Laws (Ye 2024) — 拟合 weight 与 loss 的函数关系。学术复现建议直接抄 Llama-3 或 Nemotron 的 mix。domain 数少 (< 10) 时直接 grid search 也行。
]

#interview[
  *Q8*: streaming dataset 断点续训怎么做？

  A: 保存 iterator 状态：当前处理到哪个 shard 的哪个位置，per-rank 保存。mosaicml Streaming 有 `state_dict()` / `load_state_dict()` API。自己实现 IterableDataset 时要暴露：`shard_idx`, `sample_offset_in_shard`, `epoch`, `rng_state`。resume 时跳到那个位置继续。断点保存频率与 checkpoint 一致。
]
