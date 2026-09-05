#import "../template.typ": *

= 多模态训练的特殊优化

多模态大模型（VLM: Vision-Language；VLA: Vision-Language-Action；Audio-LM 等）在分布式训练上遇到与纯文本 LLM 不同的挑战。这一章讲清楚：变长模态、跨模态负载不均、encoder/decoder 隔离、packing 的特殊难度、以及公开的 LLaVA/Qwen-VL/InternVL/Gemini/Sora 类做法。

== 多模态的结构

典型 VLM 架构（LLaVA-style）：

```
image ──► vision encoder (ViT / CLIP)  ──► image tokens  ┐
                                                          ├─► projection ──► LLM decoder ──► loss
text  ──► tokenizer                    ──► text tokens ──┘
```

多模态训练里"模态"变成了 sample-level 属性：
- Text-only sample: image tokens 数 = 0
- Image caption: image tokens ~576 (ViT-L/14 @ 336px)，text tokens ~50
- Long-form OCR: text tokens 2000+
- Video: image tokens $times$ frame 数，可能 $10^4+$

*两个新问题*：
+ *Modality-heterogeneous cost*：不同 sample 的 encoder / decoder 计算完全不同——一张图 encoder 5 ms，1000 frame video encoder 5 s
+ *Modality-imbalanced batch*：如果 batch 里全是 text-only, vision encoder 空转；反之 LLM 空转

== 变长图像/视频的处理

=== Native resolution 训练

传统 CLIP/ViT 训练用固定 $224 times 224$ or $336 times 336$，简单但丢信息（OCR 需要高分辨率）。

*Native resolution*（Qwen-VL, InternVL, Fuyu, Molmo）：保留原始 aspect ratio + resolution。做法：

+ *NaViT (Dehghani 2023)*：把不同尺寸图像 patch 化后 pack 到一个 sequence 里，attention 里 mask 保证图内 attend。这就是"图像版 packing"。
+ *Dynamic patch grid*：图像 resize 到最近的 patch grid (e.g., 14×14 tokens)，再 pack。Qwen-VL v2/v3 用这个。
+ *Any-resolution* (LLaVA-Next)：把大图切成多个 sub-image，分别编码，concat 为 tokens。$4 times 336 -> 4 times 576 = 2304$ tokens 就正常了。

*后果*：一个 batch 里图像 tokens 数从 100 到 10000+ 波动 → LLM decoder 输入长度剧烈变化。necessity of *padding-free packing*。

=== 视频的时空处理

视频 = frame 序列 + audio。frame 数从 8 到几千。做法：

+ *Sparse frame sampling*：均匀采 8-32 frames（VideoLLaMA, Video-LLaVA）
+ *Dense sampling + token compression*：256+ frames，通过 temporal pooling / Q-Former / Perceiver 压缩到固定 tokens (100-200)
+ *3D VAE (Sora / CogVideoX)*：视频编码到 4D latent，再展平

计算成本从几十 ms 到几秒/sample，比图像高 10-100×。

== Vision Encoder 与 LLM Decoder 的分工

=== 冻结 encoder vs full training

*Stage-1 pretrain* (LLaVA)：冻结 ViT，只训 projection + LLM。ViT forward 一次即可，无需 grad → 显存小
*Stage-2 fine-tune*：full training，vision 也训。需要 vision encoder grad → 大 activation

冻结 ViT 时，可以*预计算 vision features 缓存*，训练时直接读——大幅省 compute。适合数据 stable 的场景。

=== Encoder / Decoder 的并行策略隔离

Vision encoder (0.3-1B params) 和 LLM decoder (7-70B params) 大小差 10-100×。用同一 TP/PP setup 不合适：

+ *ViT 小，TP=1 就够*，用 DDP/FSDP 复制多份
+ *LLM 大，需要 TP+PP+FSDP 组合*

*方案*：ViT 与 LLM *用不同 process group*。

```python
# Vision encoder: DP only, no model parallel
vision_encoder = FSDP(vit, sharding_strategy=SHARD_GRAD_OP,
                      device_mesh=dp_mesh)

# LLM decoder: TP+PP+FSDP
llm = FSDP(llm_with_tp_pp, sharding_strategy=FULL_SHARD,
           device_mesh=llm_dp_mesh)
```

Data flow：ViT forward on all ranks → image features → send to LLM's first stage (via P2P if PP) → LLM forward+backward → grad back → ViT backward。

Kimi VLM / Qwen-VL 类系统这么做。工程复杂度显著上升，但避免了"整个模型都 TP=8 但 ViT 用不上 TP"的浪费。

=== 分离式 Vision Encoding

更激进：把 ViT 放独立的 GPU 组（"encoder ranks"），LLM 放另一组 (`decoder ranks`)，之间只传 image features（低带宽）。

```
encoder ranks: [ ViT (0.3B) ] × 32 GPU   (纯 DP)
decoder ranks: [ LLM (70B) ] × 32 GPU    (TP+PP+FSDP)
Encoder → Decoder: send image_features (few MB) per sample
```

好处：encoder 与 decoder 可以异步、独立 batch size、独立并行。坏处：backward 从 decoder 传 grad 回 encoder 需要 P2P + 同步。工业界的 Sora / Veo 类大视频 model 用这种。

== Packing 在多模态里的困难

文本 packing：拼接 tokens，加 cu_seqlens，varlen FA。

多模态 packing 挑战：

+ *Image tokens 是连续 block*（一张图 = 576 tokens，不能被 pack 拆开）
+ *Text-image interleaved*（LLaVA-Interleaved 类）：`<img1><text1><img2><text2>...` 顺序要保
+ *不同 sample 的 image 数不同*（有的 0，有的 5）
+ *Modality mask*：loss 只在 text tokens 上算，image tokens loss=0（LLaVA 做法）

*Packing 策略*：

+ *固定 seq_len packing*：与文本一样，pack 到 8K/16K。image tokens 视为"不可分块"
+ *Modality-aware batching*：先按图像数分桶，每桶内 pack。避免"一个 batch 全是 image-heavy sample"
+ *Multi-image packing*：一个 packed seq 里可以有 3-5 images + 相关 text，用 special token 分隔

*loss mask 实现*：

```python
input_ids       # (B, S)
labels          # (B, S), image position 设 -100 (ignore)
attention_mask  # varlen cu_seqlens
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

== 图像预处理的 dataloader 特殊坑

+ *Decode 慢*：JPEG/PNG decode CPU 单核 100 ms/image；一个 worker 拉 batch 时会成为 bottleneck
  - 用 nvJPEG (GPU decode) 或 turbo-jpeg
  - 或预 tokenize + 缓存 to WebDataset shard
+ *Resize / augment*：torchvision 慢。用 kornia (GPU-side) 或 pillow-simd
+ *Multi-worker 与 GPU 内存*：num_workers=32 × prefetch=8 × per-sample-4MB image = 4 GB CPU memory 常见，注意 OOM
+ *Persistent workers 与 forking*：CUDA-aware library 里 fork 会 crash。用 `spawn` context 或不 pin CUDA in worker

*一个 vision dataloader 骨架*：

```python
class VLMDataset(IterableDataset):
    def __iter__(self):
        for sample in stream_samples():
            # sample: {"image_paths": [...], "text": "..."}
            # 1. decode images (可放 num_workers)
            imgs = [decode_and_resize(p) for p in sample["image_paths"]]
            # 2. tokenize text
            text_tokens = tokenizer.encode(sample["text"])
            # 3. build interleaved sequence: <img1_placeholder> ... text ...
            input_ids, labels, modality_ids = interleave(imgs, text_tokens)
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": torch.stack(imgs),      # (n_imgs, C, H, W)
                "modality_ids": modality_ids,           # 0=text, 1=image
            }
```

collate 时按最大 image 数 pad 或用 nested tensor。

== 跨模态 sample balancing

DDP 里如果不做 balancing：

- rank 0 拿到 5 张图 + 短文本 → ViT 5 forward + short LLM
- rank 1 拿到 0 张图 + 长文本 → 无 ViT, long LLM
- rank 2 拿到 1 张图 + 中长文本 → normal

同 step 内三 rank compute time 差 3-5×，DP AR sync 时慢 rank 拖整体。

*解决*：

+ *"Modality-uniform" batch*：每 batch 每 rank 有相同图像数（预先 sort by n_images）
+ *"Total-cost balancing"*：估算每 sample 的 ViT + LLM cost，把总 cost 均分到 rank
+ *Bucketing by cost*：按 (n_images, text_len) 二维分桶，同桶 sample 分给同 rank

Qwen-VL v3 tech report 提到用 cost-based balancing 让 MFU +12%。

== Sequence length variance 在 attention 里

多模态一个 batch 里 seq 长度从 500 到 16K 变化 → FlashAttention varlen 里 tail sample 主导 kernel time。

方法：*Dynamic batching + max_tokens*：
- `max_tokens_per_batch = 32768`
- batch size = max_tokens / max(seq_len)
- 长 seq 时 batch=2, 短 seq 时 batch=32

HF Trainer + `group_by_length=True` 能做。或自己写 dataset iterator。

== 一个 VLM 分布式训练配置样例

*Qwen2.5-VL-72B on 256 H100*：

```bash
torchrun --nnodes=32 --nproc-per-node=8 ...  \
  # LLM part
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --data-parallel-size 8 \                  # = 256 / (8×4)
  --sequence-parallel \
  \
  # Vision encoder part (独立 group)
  --vision-tp 1 \
  --vision-dp 256 \                         # ViT 每卡 replica
  \
  # Multimodal loader
  --dataset-type interleaved-vlm \
  --max-seq-length 16384 \
  --max-image-tokens 2048 \
  --pack-multimodal \
  --group-by-cost \
  \
  # Standard
  --bf16 --use-flash-attn --swiglu ...
```

== 多模态特有的稳定性问题

+ *Image token vs text token loss scale*：如果 image tokens 参与 loss，image loss 数值范围可能与 text 差 10×。用 loss weighting
+ *Long-video training 会 spike*：某 batch 突然 5000 image tokens 让显存暴涨。做 hard cap 或 outlier filter
+ *Vision encoder 与 LLM 学习率不同*：ViT 已 pretrained，LR 小 (1e-5)；LLM projection 层新，LR 大 (1e-4)。分组 LR
+ *Modal collapse*：训练早期 LLM 忽略 image (直接从 text 预测)。做 vision-conditioned loss up-weight

== 面试考点

#interview[
  *Q1*: 多模态训练里 vision encoder 与 LLM 的并行策略为什么要分开？

  A: ViT 通常 300M-1B 参数，一张卡装得下 → 用 DDP/FSDP 复制。LLM 70B+ 需要 TP+PP+FSDP。用同一 TP=8 setup 会让 ViT 上 TP AllReduce 白开销（本来不需要）。分离让 ViT DP=world, LLM TP+PP+FSDP，各自最优。工程复杂但 MFU +10-20%。
]

#interview[
  *Q2*: NaViT 的 packing 与文本 packing 有什么不同？

  A: 文本 packing 沿 seq 拼接；NaViT 把不同尺寸图像 patch 化后 pack 到一个 sequence（每图 = block of patches），attention 内部 mask 图间不 attend。核心相同（varlen packing），差别在"unit"是 image patch block 而非 sub-sequence。让 batch 里可以混合不同分辨率图像。
]

#interview[
  *Q3*: LLaVA-Next 的"any-resolution"是怎么实现的？

  A: 高分辨率图切成多个 sub-image (e.g., 4 patches of 336×336)，每个独立编码得 576 tokens，concat 成 4×576=2304 tokens 拼到 LLM 输入。加上一张 thumbnail (整图 resize 到 336×336) 作 global context。让 LLM 看到 high-res detail 又不训练新架构。代价：一个高分辨率图变 2000+ tokens。
]

#interview[
  *Q4*: 视频训练里最大的显存瓶颈？

  A: 视频 tokens 数量爆炸——8 frames × 576 tokens/frame = 4608 tokens/video，dense sampling 更多。attention 里 seq_len 从 1K 涨到 16K+。做法：(a) Q-Former / Perceiver 压缩 (每 frame 8 tokens)；(b) temporal pooling；(c) 3D VAE encoder 直接编码 spatio-temporal。CogVideoX 用 3D VAE 把 (T=49, H=480, W=720) 压缩到 latent (13, 60, 90)。
]

#interview[
  *Q5*: DDP 里跨 rank 图像数不均衡会怎样？

  A: rank 0 十张图，rank 1 零张图 → ViT forward 时间差 10×。DP AR 时同步，快 rank 空等。MFU -20-30%。解决：cost-based batch balancing（按 total ViT+LLM cost 均分）；或 group_by_length + group_by_n_images 二维桶。Qwen-VL v3 报告 balancing 后 +12% throughput。
]

#interview[
  *Q6*: 多模态 packing 里 loss 怎么 mask？

  A: LLaVA 传统做法：只在 text tokens 上算 loss。image tokens 位置 label 设 -100 (`ignore_index`)。cross_entropy 自动跳过。有些多模态模型（如 Chameleon）也让 image tokens 参与 loss（image tokens 是可预测的 discrete codes）——这时 loss 里所有位置都算，但 image / text loss 分开监控。
]

#interview[
  *Q7*: 训练视频模型时 sequence length 从 1K 到 16K+ 变化，attention kernel 怎么处理？

  A: FA varlen (`cu_seqlens`) 自然支持。但 kernel launch overhead 随 seq_len 分布方差增大。做法：(a) dynamic batching by `max_tokens_per_batch` 而非固定 batch size；(b) sort by length，减少每 batch 内方差；(c) `torch.cuda.CUDAGraph` capture 每个 seq_len bucket 一个 graph，避免重复 launch。
]

#interview[
  *Q8*: 一个 32B VLM 训练 MFU 只有 25%，你会先查什么？

  A: 按概率：(1) dataloader 供不上 (vision decode 慢) —— 加 time.time 测；(2) modality-imbalance —— nsys 看 ViT stream 是否有大 gap；(3) sequence length variance —— log 每 batch seq_len 分布；(4) TP=8 但 ViT 用不上 —— 分离 encoder/decoder group；(5) FA varlen 内部小 seq 效率低 —— 用 CUDA graph 或调 batch 结构。这几个查完通常能从 25% 提到 40%+。
]
