#import "../template.typ": *

= 附录 A：符号表 & 延伸阅读

== 符号表

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (center, left),
    [*符号*], [*含义*],
    [$B$], [batch size],
    [$S$], [sequence length],
    [$N = B S$], [flatten 后的 token 总数],
    [$H$], [hidden dim (Transformer 主维度)],
    [$I$], [FFN intermediate dim ($I approx 4 H$ 常见, SwiGLU 常 $8/3 H$)],
    [$E$], [专家总数 num_experts],
    [$K$], [top-k, 每 token 激活的专家数],
    [$M_e$], [路由到专家 $e$ 的 token 数 ($sum_e M_e = N K$)],
    [$C$], [capacity, 每个专家在 batch 内允许接收的最大 token 数],
    [$c_"factor"$], [capacity factor, $C = "ceil"(N K / E times c_"factor")$],
    [$f_e$], [expert $e$ 被 top-1 (或 top-K) 选中的 batch 内比例],
    [$P_e$], [gate_probs 对 expert $e$ 的 batch 均值],
    [$L_"aux"$], [auxiliary load-balancing loss, $= E dot (f dot P)$],
    [$L_z$], [router z-loss, $= "mean"("lse"(ell)^2)$],
    [$"EP"$], [expert parallel world size],
    [$"TP"$], [tensor parallel world size],
    [$"DP"$], [data parallel world size],
    [$"PP"$], [pipeline parallel world size],
    [$W_g$], [router (gate) weight, shape $(E, H)$],
    [$W_"up", W_"gate", W_"down"$], [SwiGLU expert 三个 Linear 的 weight],
    [$ell$ / gate_logits], [router pre-softmax 输出 $(N, E)$],
    [$p$ / gate_probs], [router 后 softmax 输出 $(N, E)$],
    [$"exp_ids"$], [expert_indices $(N, K)$],
    [$"exp_wts"$], [expert_weights $(N, K)$],
    [packed_input], [permute 后按 expert 排序打包的 input, $(N K, H)$],
    [group_sizes], [每 expert 的 token 数, $(E,)$],
    [a2a], [all-to-all],
  ),
  kind: table,
)

== 延伸阅读

=== 经典论文

- Shazeer et al. 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
  #link("https://arxiv.org/abs/1701.06538")[arxiv:1701.06538] — 开山之作
- Lepikhin et al. 2020, "GShard: Scaling Giant Models with Conditional Computation",
  #link("https://arxiv.org/abs/2006.16668")[arxiv:2006.16668] — 生产级 MoE, capacity+drop
- Fedus et al. 2021, "Switch Transformer",
  #link("https://arxiv.org/abs/2101.03961")[arxiv:2101.03961] — $K=1$ 极简路由
- Du et al. 2022, "GLaM: Efficient Scaling of Language Models with MoE",
  #link("https://arxiv.org/abs/2112.06905")[arxiv:2112.06905] — 1.2T 参数 MoE
- Zoph et al. 2022, "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
  #link("https://arxiv.org/abs/2202.08906")[arxiv:2202.08906] — router z-loss, 稳定性
- Zhou et al. 2022, "Mixture-of-Experts with Expert Choice Routing",
  #link("https://arxiv.org/abs/2202.09368")[arxiv:2202.09368] — 反向路由

=== 生产模型

- Jiang et al. 2024, "Mixtral of Experts",
  #link("https://arxiv.org/abs/2401.04088")[arxiv:2401.04088]
- Databricks 2024, "DBRX: Introducing a New State-of-the-Art LLM",
  #link("https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm")[blog]
  — 使用 Megablocks
- DeepSeek-AI 2024, "DeepSeek-V3 Technical Report",
  #link("https://github.com/deepseek-ai/DeepSeek-V3")[github/DeepSeek-V3]
  — 671B 参数, aux-loss-free, fine-grained overlap
- DeepSeek-AI 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization",
  #link("https://arxiv.org/abs/2401.06066")[arxiv:2401.06066]
  — 细粒度 + shared expert
- Qwen 2024, "Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters",
  #link("https://qwenlm.github.io/blog/qwen-moe/")[blog]

=== Kernel / 系统

- Gale et al. 2022, "MegaBlocks: Efficient Sparse Training with MoE",
  #link("https://arxiv.org/abs/2211.15841")[arxiv:2211.15841]
- Hwang et al. 2022, "Tutel: Adaptive Mixture-of-Experts at Scale",
  #link("https://arxiv.org/abs/2206.03382")[arxiv:2206.03382]
- Rajbhandari et al. 2022, "DeepSpeed-MoE",
  #link("https://arxiv.org/abs/2201.05596")[arxiv:2201.05596]
- Cai et al. 2024, "Shortcut-connected Expert Parallelism for Accelerating MoE",
  #link("https://arxiv.org/abs/2404.05019")[arxiv:2404.05019] — expert parallel overlap

=== 代码仓库

- #link("https://github.com/deepseek-ai/DeepEP")[deepseek-ai/DeepEP] — DeepSeek expert parallel 通信库
- #link("https://github.com/NVIDIA/Megatron-LM")[NVIDIA/Megatron-LM] — Megatron-LM MoE
- #link("https://github.com/microsoft/tutel")[microsoft/tutel] — Tutel
- #link("https://github.com/databricks/megablocks")[databricks/megablocks] — MegaBlocks
- #link("https://github.com/vllm-project/vllm")[vllm-project/vllm] — 推理时 fused MoE kernel
- #link("https://github.com/sgl-project/sglang")[sgl-project/sglang] — SGLang MoE 推理

=== 博客

- HuggingFace, "Mixture of Experts Explained",
  #link("https://huggingface.co/blog/moe")[hf.co/blog/moe]
- Yao Fu, "MoE Paradigm: Sparse Activations for Efficient LLMs",
  #link("https://yaofu.notion.site/")[notion]
- Cameron Wolfe, "Mixture-of-Experts (MoE) LLMs",
  #link("https://cameronrwolfe.substack.com/p/moe-llms")[substack]

== 本仓库文件

- 最小 PyTorch 实现: `python/pytorch/test_moe.py`
- 运行 smoke test:

  ```bash
  python python/pytorch/test_moe.py
  ```

- 本书源码: `books/moe/`
- 编译: `cd books/moe && typst compile book.typ`
