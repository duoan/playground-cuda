#import "../template.typ": *

= RLVR 与 Agentic RL 的分布式：verifier 集群、多轮 rollout、长尾 straggler

第 14 章讲的是"经典 RLHF 系统"：一个 policy、一个 ref、一个 reward model、一个 value model，rollout 是*一次* generate。2024 年底到 2026 年，工业界的 RL 训练已经从这个形态整体迁移到两个新范式：

/ RLVR: *Reinforcement Learning with Verifiable Rewards* —— 把学出来的 reward model 换成*确定性验证器*（数学答案比对、单元测试、形式化证明检查）。AI2 的 Tulu 3 (2024) 命名了这个词，DeepSeek-R1 (2025) 把它推到了 671B 规模。
/ Agentic RL: rollout 不再是"生成一段话"，而是 *LLM ↔ 环境的多轮循环*：模型发工具调用 → 沙箱/浏览器/搜索执行 → 观测拼回上下文 → 模型继续。Kimi K2、SWE-RL、Search-R1 这一类。

这两个范式对*系统*的冲击比对算法的冲击大得多：

#figure(
  table(
    columns: (auto, 1fr, 1.15fr),
    stroke: 0.4pt + gray,
    inset: 5.5pt,
    align: (left, left, left),
    [*维度*], [*经典 RLHF (第 14 章)*], [*RLVR / Agentic RL (本章)*],
    [Reward 来源],
    [Reward model（GPU forward）],
    [Verifier（CPU 沙箱执行）/ 规则],
    [新增算力池],
    [无（都是 GPU）],
    [*CPU verifier 集群 + env 服务*],
    [Rollout 形态],
    [单次 generate，1 次 prefill],
    [多轮循环，T 次 prefill（除非复用 prefix cache）],
    [长度方差],
    [约 4×（512 \~ 2048 token）],
    [约 *100×*（1 轮 500 token \~ 50 轮 100K token）],
    [Rollout 期 GPU],
    [一直在 decode],
    [*等环境时闲置*（工具执行几秒）],
    [Reward 稠密度],
    [连续标量，每样本都有],
    [二值 0/1，且常整组全对或全错],
    [Ref model],
    [必须有（算 KL）],
    [很多配方*直接去掉 KL*，省一个池],
    [训练侧新坑],
    [advantage 归一化],
    [*token masking*（观测 token 不能训）],
  ),
  kind: table,
  caption: [经典 RLHF 与 RLVR / Agentic RL 的系统性差异。右列每一行都对应本章一节。面试里能把这张表口述出来，基本就说明你真上手过。],
) <table-rlvr-vs-rlhf>

#warn[
  这个方向 2025–2026 迭代极快，论文里的具体数字（步数、分数、加速比）跟版本强相关。面试时*说清来源和时间点*（"DAPO 那篇 2025 年 3 月报告的是……"），比背一个孤立数字安全得多。本章标注的都是公开报告里的做法，不代表你做过。
]

== 一. RLVR：把 reward model 换成 verifier

=== 1.1 为什么要换掉 reward model

Reward model (RM) 在数学 / 代码这类*有客观对错*的任务上有三个硬伤：

+ *Reward hacking*：RM 是个神经网络，policy 训久了一定学会骗它。DeepSeek-R1 报告里明确说，他们*放弃*了 neural PRM / ORM，理由就是大规模 RL 下 reward hacking 不可避免，而且要不断重训 RM 来对抗，代价太高。
+ *RM 本身要占 GPU*：70B policy 配 70B RM，rollout 池里就多一份 TP=8 的 inference 实例。
+ *RM 有噪声上限*：RM 的准确率若 85%，policy 的天花板就被钉死在 RM 的偏好分布上。

Verifier 的对立面很干净：数学题对答案，代码跑单元测试，`IFEval` 类指令跑程序化约束检查。*确定、免费（相对 GPU）、不可 hack*（至少不能用"说好话"来 hack）。

=== 1.2 Verifier 的四类形态与延迟量级

系统设计的第一步是搞清楚你的 verifier 属于哪一类 —— 因为延迟差了 4 个数量级：

#figure(
  table(
    columns: (auto, 1fr, auto, 1fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, center, left),
    [*类型*], [*做法*], [*典型延迟*], [*主要风险*],
    [数学],
    [答案抽取 + 符号等价 (sympy / math-verify)],
    [1–50 ms],
    [sympy 在病态表达式上*挂死*，必须超时],
    [代码],
    [沙箱跑单元测试],
    [0.1–10 s],
    [安全隔离、flaky test、依赖装配],
    [形式化证明],
    [Lean / Coq 编译],
    [秒 \~ 分钟],
    [编译器就是瓶颈，需大 CPU 池],
    [Agent 任务],
    [最终答案 EM/F1，或 LLM-as-judge],
    [ms \~ 秒],
    [judge 又变回 GPU 负载 + 可 hack],
  ),
  kind: table,
  caption: [四类 verifier 的延迟量级。数学 verifier 便宜到可以同步跑；代码 / 形式化 verifier 必须建独立池 + 异步化。],
) <table-verifier-types>

#insight[
  一个反直觉但很关键的点：*verifier 的平均延迟不重要，尾延迟才重要*。一组 GRPO rollout 有 $N = 16$ 个响应，只要有 1 个触发了 sympy 死循环或死锁的单元测试，整组的 advantage 就算不出来，整个 batch 卡住。所以 verifier 层第一件要做的事永远是*每样本硬超时*，而不是优化平均延迟。
]

=== 1.3 Verifier 集群架构

Verifier 是*CPU 负载*，第一次出现在一个纯 GPU 的训练栈里。典型架构：

#figure(
  align(center, op-stack(steps: (
    ("rollout 完成一批响应", "GPU 侧",           "full"),
    ("hash + 查 reward cache", "命中直接返回",   "shard-h"),
    ("组内去重 (N 个响应常有重复)", "省 30-60%", "shard-h"),
    ("投递 verifier 队列",   "异步，不阻塞 GPU", "comm"),
    ("沙箱 worker 池执行",   "CPU, 带超时",      "shard-s"),
    ("回填 reward + 算 advantage", "回 GPU 侧",  "full"),
  ), width: 7.4, cell-h: 0.55)),
  caption: [RLVR verifier 流水线。cache 与去重放在最前面是因为它们最便宜；沙箱执行放在异步池里，让 GPU 在等待期间继续 rollout 下一批。],
) <fig-verifier-pipeline>

四个必备设计：

*(a) 异步化，别放在关键路径上。* 同步版本是 "rollout 全批 → 全批 verify → 训练"，verify 期间 GPU 全闲。异步版本是 verify 与下一批 rollout 重叠，只要 verifier 池吞吐 ≥ rollout 产出速率，verify 就完全不进关键路径。

*(b) Reward cache + 组内去重。* GRPO 一个 prompt 采 $N=16$ 个响应，温度不高时*完全相同的响应*很常见（数学题尤其，正确解法就那么几种）。按 `hash(prompt, response)` 做缓存 + 组内去重，实测能省掉 30–60% 的 verifier 调用。这是最高性价比的一招。

*(c) 每样本硬超时 + 熔断。* 用子进程 + `SIGKILL`，不要用线程（Python 线程杀不掉死循环）。超时的样本给 reward 0 并*单独打点*——超时率突然上升往往意味着 policy 学到了某种病态输出。

*(d) CPU:GPU 配比要算，不能拍。* 这是面试高频计算题，见下节。

=== 1.4 CPU:GPU 配比怎么算

设：rollout 池 $G$ 张 GPU，每张 GPU 每秒产出 $r$ 个响应；单次验证平均耗时 $t_v$ 秒，缓存/去重命中率 $h$；每个 verifier worker 单线程。

稳态下 verifier 池不成为瓶颈的条件是：

#formula[
  $ W gt.eq (G dot r dot (1 - h) dot t_v) / u $
]

其中 $W$ 是 worker 数，$u$ 是目标利用率（留余量，取 0.7 左右）。

*代入一组真实量级*：128 张 GPU 做 rollout，每卡每秒产出 $r = 0.5$ 个响应（长 CoT，响应几千 token），代码 verifier $t_v = 2$ s，去重命中率 $h = 0.4$，$u = 0.7$：

#formula[
  $ W gt.eq (128 times 0.5 times 0.6 times 2) / 0.7 approx 110 "workers" $
]

按每 worker 1 core 算就是 ~110 core，约 1.5 台 64 核机器。

*这个公式是对的，但它回答的不是你最关心的问题。* 稳态排队是否稳定只取决于*平均*服务时间，所以用均值代入 $t_v$ 就能得到你要的利用率 —— 不需要"为了保险"按 p90 配（那会白买 2–3 倍的机器）。

真正卡住训练的是另一件事：*GRPO 必须等一组 $N$ 个响应全部验完*才能算 advantage，所以优化器等的是*组内最大值*，不是均值。而重尾分布下这两者差得很远 —— `17_rlvr_verifier.py` 的仿真里，单次验证均值 1.39 s，但 16 个一组的最大值均值是 6.42 s，*4.6×*；并且有 39% 的组里至少有一个样本会撞上超时。

由此得到两条不太直觉但很关键的结论：

+ *加机器能解决吞吐，永远解决不了组内最大值。* 能压住组最大值的只有两样东西：超时（直接给尾部封顶）和去重（减少每组从尾部独立抽样的次数）。
+ *更好的办法是让组最大值根本不在关键路径上* —— 异步验证，让它与下一批 rollout 重叠。做到这一点之后，按均值算出来的容量数就真的够用了。

#interview[
  *面试题*：你们 RLVR 训练里 GPU 利用率只有 55%，profile 显示 GPU 在等 reward。怎么定位和解决？

  分层回答：
  + *先量*：打点 rollout 产出速率、verifier 队列深度、verify p50/p99、cache 命中率。队列深度持续增长 → verifier 池吞吐不够；队列空但 GPU 还在等 → 说明是同步阻塞，不是容量问题。
  + *如果是同步阻塞*：改异步，verify 与下一批 rollout 重叠。这一步通常直接把利用率拉回 80%+，且不花钱。
  + *如果是容量不够*：先上 cache + 组内去重（免费拿 30–60%），再按上面的公式扩 worker。
  + *如果是尾延迟*：查超时率与超时样本的特征。见过的真实原因包括 policy 学会输出超长表达式把 sympy 卡死、生成的测试代码里 `input()` 等待 stdin、以及单元测试里有网络调用被防火墙挂住。修法分别是表达式长度上限、沙箱关 stdin、沙箱断网。
]

=== 1.5 沙箱安全：模型生成的代码是敌意输入

这一条在面试里很能体现"真跑过生产"。Policy 生成的代码要在你的集群里执行，而 RL 的本质就是*搜索一切能拿到高 reward 的行为*——包括直接篡改测试结果。

必须有的隔离层：

- *容器 / 微 VM*：gVisor、Firecracker、nsjail 之一。裸 `subprocess` 不够。
- *断网*：否则模型可能去下载正确答案，或者把你的内网探个遍。
- *只读文件系统 + 独立 tmpfs*：防止改测试文件。真实见过的 hack：生成的代码把 `test_*.py` 覆写成 `assert True`。
- *cgroup 限 CPU / 内存 / pid*：fork bomb 和内存炸弹是常态，不是意外。
- *墙钟超时 + 强杀*。

#warn[
  *Reward hacking 在 RLVR 里没有消失，只是换了形态。* RM 时代是"说漂亮话骗 RM"，verifier 时代是"骗 verifier"：改测试文件、在代码里 `sys.exit(0)`、捕获所有异常返回硬编码答案、数学题里输出一堆候选答案让抽取器蒙对一个。防御手段是*把 verifier 本身当作被攻击面来设计*：测试文件放在容器外挂只读、答案抽取只取最后一个 `\boxed{}`、比对前做归一化且拒绝多答案。
]

== 二. RLVR 的算法层：二值 reward 带来的新问题

=== 2.1 GRPO 在二值 reward 下的退化

GRPO 的 advantage 是组内归一化：

#formula[
  $ A_i = (R_i - "mean"(R_1..R_N)) / ("std"(R_1..R_N) + epsilon) $
]

二值 reward 下，如果这一组 $N$ 个响应*全对*（都是 1）或*全错*（都是 0），那么 $"std" = 0$、$R_i - "mean" = 0$，于是 $A_i = 0$ —— *整组贡献零梯度*。这一组的 rollout 算力（可能是几十秒的长 CoT 生成）完全白烧。

而这不是边缘情况：训练早期大量题目全错，训练后期大量简单题全对。实测中无效组占比能到 30–50%。

*DAPO 的 dynamic sampling*（ByteDance, 2025）：过采样，然后*丢掉准确率为 0 或 1 的组*，一直采到凑满一个 batch 的"有信息量"的组。代价是 rollout 量变大，收益是每个梯度步都有真实信号。

```python
# 概念版：动态采样直到凑满 batch
kept_groups = []
while len(kept_groups) < target_batch_groups:
    prompts = sampler.next(oversample_factor * needed)
    groups  = rollout_engine.generate(prompts, n=N)
    rewards = verifier_pool.verify(groups)          # 异步 + cache
    for g, r in zip(groups, rewards):
        acc = r.mean()
        if 0.0 < acc < 1.0:                         # 只保留有梯度的组
            kept_groups.append(g)
        else:
            _wasted_group_counter.inc(               # 一定要打点
                "all_correct" if acc == 1.0 else "all_wrong")
```

*系统层的连带影响*：dynamic sampling 让每个 iteration 的 rollout 量*不确定*，rollout 池的容量规划从"固定 batch"变成"带反馈的流式采样"。调度器要支持"继续采直到够"，而不是"采固定 N 个"。

*另一条路*是难度分层采样：维护每道题的历史准确率，优先采 $0.2 < "acc" < 0.8$ 的题（信息量最大），把长期全对的题移出题池。这是 curriculum 的一种，比 dynamic sampling 省 rollout，但需要维护题目状态。

=== 2.2 长度偏差：Dr. GRPO 的两处修正

Dr. GRPO（Liu et al., 2025，_Understanding R1-Zero-Like Training_）指出原始 GRPO 目标里有两个引入偏差的归一化：

+ *按响应长度归一化* $1/|o_i|$：让"短的正确答案"和"长的错误答案"获得不成比例的权重，系统性地鼓励错误答案变长 —— 这正是大家观察到的"RL 训着训着响应越来越长但没变强"的成因之一。
+ *按组标准差归一化* $1/"std"$：让"几乎全对"和"几乎全错"的题（std 小）获得放大的权重，等于给了极端难度的题过高的权重。

Dr. GRPO 的处方就是把这两项去掉，用 token 级的求和而不是样本级的平均。

DAPO 独立地提出了 *token-level policy gradient loss*，动机一致：样本级平均会让一个 1000 token 的响应和一个 100 token 的响应在 loss 里权重相同，于是长响应里每个 token 的有效学习率被稀释 10×。

#insight[
  这一组问题的共同根源是：*你在哪个粒度上做平均，就在哪个粒度上定义了"公平"*。样本级平均 = 每个样本同权，token 级求和 = 每个 token 同权。长 CoT 场景下响应长度差异巨大，两者的差别就从"实现细节"升级成"影响收敛方向的算法选择"。面试里能讲清这一层，比背 "Dr. GRPO 去掉了 std" 强得多。
]

=== 2.3 熵坍缩与 clip-higher

长时间 RLVR 训练的典型病：policy 熵单调下降 → 输出越来越确定 → 一组 $N$ 个采样几乎完全相同 → GRPO 组内没有差异 → 梯度消失 → 训练停滞。

DAPO 的 *clip-higher*：把 PPO 的对称裁剪 $[1-epsilon, 1+epsilon]$ 拆成非对称的 $[1-epsilon_"low", 1+epsilon_"high"]$，取 $epsilon_"low" = 0.2$、$epsilon_"high" = 0.28$。直觉是：对称裁剪对*低概率 token 的上升空间*限制过死（一个概率 0.01 的 token 最多只能涨到 0.012），而这些低概率 token 正是探索的来源。放宽上界让它们能长起来，熵就不会塌。

其他常见手段：保持 rollout 温度 1.0（评测才用 0）、周期性检查组内响应的两两相似度、把熵本身做成监控指标并设告警线。

=== 2.4 去掉 KL 项：省一个 GPU 池

经典 RLHF 里 KL 惩罚项防止 policy 跑离 base 太远。但 RLVR 的目标恰恰是*让模型学会一种 base 完全不会的行为*（长链推理）—— 此时 KL 是纯阻力。DAPO 等配方直接把 KL 项去掉。

*系统上的收益是实打实的*：没有 KL 就不需要 ref model forward，rollout 池里少一份 70B 实例，省下的显存可以全给 KV cache，rollout 吞吐直接上一截。

*代价*：失去了"跑偏"的自动刹车。必须用别的东西兜底 —— 监控输出语言混杂（R1 报告过中英混杂）、格式崩坏、以及在 held-out 通用能力集上的回退。

#figure(
  table(
    columns: (auto, 1.1fr, 1fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*问题*], [*机制*], [*处方*],
    [整组零梯度], [二值 reward 下组内全对/全错 → std=0], [DAPO dynamic sampling / 难度分层采样],
    [响应越训越长], [$1/|o_i|$ 归一化偏袒长错误答案], [Dr. GRPO 去长度归一化 + token-level loss],
    [难度权重失衡], [$1/"std"$ 放大极端难度题], [Dr. GRPO 去 std 归一化],
    [熵坍缩], [对称裁剪压制低概率 token 上升], [clip-higher ($epsilon_"high" = 0.28$)],
    [ref 池占 GPU], [KL 需要 ref forward], [去 KL，改用监控兜底],
    [截断算失败], [超长响应给 0 → 教模型变短], [DAPO overlong reward shaping（软惩罚）],
  ),
  kind: table,
  caption: [RLVR 算法层六个高频问题与对应处方。每一行都能单独成为一道面试追问题。],
) <table-rlvr-algo>

=== 2.5 一个必须知道的 caveat：spurious rewards

2025 年有一批工作（如 Shao et al. 的 _Spurious Rewards_）发现：在某些 Qwen 系模型上，*用随机 reward 甚至错误 reward* 做 RLVR，数学分数照样能涨。这说明 RL 在这些设置下*激发*的是预训练里已有的能力（比如"用代码思路解题"的倾向），而不是学到了新东西。

面试价值在于：这提醒你 *RLVR 的评测必须有对照组*。如果你说"我们上了 RLVR，AIME 涨了 10 分"，一个懂行的面试官会追问"随机 reward 的 baseline 跑了吗？换个基座还成立吗？"能主动提这一点，说明你不是只会跑 pipeline。

== 三. Agentic RL 的 rollout 引擎

=== 3.1 多轮循环把 rollout 变成了另一个东西

单轮 rollout：一次 prefill + 若干 decode，结束。

多轮 agentic rollout：

#figure(
  align(center, op-stack(steps: (
    ("prefill prompt",        "GPU",            "full"),
    ("decode 到工具调用",     "GPU",            "full"),
    ("解析 tool call",        "CPU, ms 级",     "shard-h"),
    ("环境执行",              "GPU 闲置! 秒级",  "comm"),
    ("观测拼回上下文",        "CPU",            "shard-h"),
    ("prefill 新增部分",      "复用 prefix cache", "shard-s"),
    ("… 循环 T 轮 …",         "T = 1 \~ 50",     "full"),
  ), width: 7.2, cell-h: 0.55)),
  caption: [Agentic rollout 的一轮循环。关键在两处：环境执行期间该序列不占 GPU 计算但占着 KV cache；每轮的新 prefill 必须复用上一轮的 prefix cache，否则总 prefill 代价是 $O(T^2)$。],
) <fig-agent-loop>

由此产生四个系统问题，逐个说。

=== 3.2 问题一：prefix cache 是生死线

第 $k+1$ 轮的输入 = 第 $k$ 轮的完整上下文 + 新的观测。如果每轮都重新 prefill 整个上下文，$T$ 轮下来 prefill 的总 token 数是 $O(T^2)$ 量级：

#formula[
  $ sum_(k=1)^T L_k approx T dot bar(L) / 2 quad "vs" quad "复用后" approx bar(L) $
]

20 轮的轨迹，不复用要多付*一个数量级*的 prefill。vLLM 的 automatic prefix caching / SGLang 的 RadixAttention 就是干这个的，agentic RL 里必须开。

*但有个坑*：环境执行要等几秒，这几秒里该序列的 KV cache 还占着显存。推理引擎的缓存淘汰策略（LRU）在高并发下很可能*把它淘汰掉*，等观测回来时又要重新 prefill —— 你以为开了 prefix cache，实际命中率只有 40%。

排查手段就是直接看引擎的 prefix cache 命中率指标。修法：
- 提高 KV cache 池容量（把省下来的 ref model 显存给它）
- 对"正在等环境"的序列做 cache pinning，或者把 KV 换出到 CPU 而不是直接丢
- 限制并发轨迹数，让活跃集合装得下

=== 3.3 问题二：等环境时 GPU 闲置 → 必须异步

同步批式 rollout（"这一批 32 条轨迹一起走第 1 轮，一起等环境，再一起走第 2 轮"）是最容易写、也最浪费的实现：每一轮都要等这批里最慢的环境调用，GPU 在此期间纯闲。

正确做法是*异步 / 连续批处理*：推理引擎里同时挂着几百条轨迹，谁的观测回来了谁就进下一轮的 batch，等环境的轨迹自然从运行批次里退出，不占计算。这本质上就是 vLLM 的 continuous batching 用在轨迹粒度上。

`src/distributed_training/16_agent_rollout_sched.py` 用离散事件仿真对比了这两种调度：同样的负载，GPU 利用率 19% → 88%，makespan 快 4.6×。

#insight[
  这个仿真还顺带暴露了一个*指标陷阱*，很值得在面试里主动提。

  异步调度故意让几百条轨迹同时在飞，于是每条轨迹都要排在别人后面，*单轨迹延迟反而变差了*（仿真里 102 s → 146 s）。更糟的是 p99/p50 这个常用的 straggler 指标：同步批式的比值是 2.85，异步是 2.94 —— *同步看起来更健康*。

  两个指标都在说反话。原因是同步的 barrier 把所有短轨迹都拖慢到最慢那条的节奏，*离散度是因为大家一起被饿死才变小的*，而百分位比值区分不了"没有 straggler"和"全都是 straggler"。

  正确的口径是 *makespan 和 GPU 利用率*：在 RL 里没有任何东西会在整个 iteration 组装完之前消费单条轨迹，所以卡住下一次优化器更新的就是 makespan，单轨迹延迟根本不在关键路径上。
]

=== 3.4 问题三：长尾 straggler 与 partial rollout

Agentic 轨迹的长度分布是*重尾*的：大部分任务 3–5 轮解决，少数任务磨 40 轮。同步收集一个 batch 意味着等最长的那条。

Kimi K1.5 报告的 *partial rollout* 是标准解法：给每个 iteration 设 token / 轮数预算，*没跑完的轨迹存下状态，下个 iteration 接着跑*，而不是丢弃或死等。

```python
# 概念版：带预算的 partial rollout
pending = carry_over_from_last_iter          # 上轮没跑完的轨迹
budget  = ITER_TOKEN_BUDGET
finished = []

while budget > 0 and (pending or task_queue):
    traj = pending.pop() if pending else Trajectory(task_queue.pop())
    traj, used = engine.run_until(traj, budget_left=budget)
    budget -= used
    if traj.done:
        finished.append(traj)
    else:
        traj.policy_version_span.append(current_policy_version)  # 记录跨版本
        carry_over.append(traj)                                  # 下轮接着跑
```

*代价是 off-policy*：一条轨迹的前半段由旧 policy 生成、后半段由新 policy 生成。处理方式有三种，按严格程度递减：

+ 逐 token 记录生成时的 policy 版本，算重要性比时用对应版本的 logprob（最严格，实现最重）
+ 限制轨迹最多跨 $K$ 个版本（比如 2），超了就丢弃（折中，最常见）
+ 直接当 on-policy 处理（最省事，短 iteration 下偏差可接受）

面试里问到 partial rollout，能主动说出"它引入了 off-policy，你得选一种处理方式"，比只说"Kimi 用了 partial rollout" 高一个层次。

=== 3.5 问题四：权重更新时的在途轨迹

训练完成一次更新，要把新权重同步到 rollout 引擎。此时有几百条轨迹跑到一半 —— 怎么办？

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*策略*], [*做法*], [*代价*],
    [Drain], [等所有在途轨迹跑完再换权重], [长尾轨迹让 GPU 空转，回到 straggler 问题],
    [Abort], [直接丢弃在途轨迹], [浪费已生成的 token，长轨迹损失最大],
    [Carry-over], [轨迹跨版本继续（partial rollout）], [off-policy，需版本记账],
  ),
  kind: table,
  caption: [权重更新时对在途轨迹的三种处理。长 horizon 场景下 drain 和 abort 都很贵，所以主流框架都往 carry-over 走，代价是要处理 off-policy。],
) <table-inflight-weight-sync>

== 四. Agentic RL 的训练层

Rollout 拿回来的是*轨迹*，不是"一段回答"。训练侧因此多出几个极易写错的地方。

=== 4.1 Token masking：本章最重要的一个 bug

一条轨迹的 token 序列长这样（宽度按真实比例画：观测 token 通常远多于 assistant token）：

#figure(
  align(center, timeline(
    streams: (
      ("角色", (("任务", 5), ("asst", 5), ("tool 观测", 15),
                ("asst", 5), ("tool 观测", 15), ("asst", 6))),
      ("loss_mask", (("不训", 5), ("要训", 5), ("不训", 15),
                     ("要训", 5), ("不训", 15), ("要训", 6))),
    ),
    unit: 0.28, bar-h: 0.55,
    colors: (
      "任务":       rgb("#94a3b8"),
      "asst":      rgb("#2563eb"),
      "tool 观测":  rgb("#f97316"),
      "要训":       rgb("#16a34a"),
      "不训":       rgb("#64748b"),
    ),
  )),
  caption: [一条 2 轮 agentic 轨迹的 token 布局（`asst` = assistant 生成）。蓝色是 policy 自己生成的，橙色是环境写进来的。下面一行是必须随轨迹一起传给训练侧的 `loss_mask`。],
) <fig-traj-mask>

*只有 assistant 生成的 token 能进 loss*。观测 token 是环境产生的，训它等于在教模型"预测工具会返回什么" —— 而这是它*原理上不可能知道*的信息。后果有三层：

+ *梯度噪声*：观测 token 在 agentic 轨迹里常占 60–80%（一次搜索返回几千 token 很正常）。不 mask 等于让 80% 的梯度来自一个不可学的目标。
+ *幻觉工具输出*：模型被训得倾向于自己"编"一段看起来像观测的文本，推理时不调工具直接编造结果。
+ *重要性比失效*：PPO/GRPO 的比值 $pi_theta / pi_"old"$ 对观测 token 没有意义 —— 那些 token 根本不是 $pi_"old"$ 采样出来的。

这个 bug 的阴险之处在于*它不会让训练崩*。Loss 照常下降，梯度范数正常，你要盯着"工具调用成功率"和"是否出现无调用直接给答案"才能发现。Search-R1 那一类工作里把"retrieved token masking"单独拎出来讲，就是因为踩过。

正确的实现是每条轨迹带一个 `loss_mask`，并且 —— 这是第二个容易漏的点 —— *归一化的分母也要只数 assistant token*：

```python
# 错误：分母含观测 token，等价于按轨迹里工具输出的多少来缩放学习率
loss = (per_token_loss * loss_mask).sum() / loss_mask.numel()

# 正确：只对参与训练的 token 求平均
loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
```

`src/distributed_training/18_traj_masking.py` 把这个错误版本和正确版本并排跑，量化了梯度差异。

=== 4.2 Advantage 往哪儿广播：trajectory-level vs turn-level

Agentic 任务的 reward 通常只在轨迹末尾（任务成功/失败）。两种赋值方式：

/ Trajectory-level: 整条轨迹一个 advantage，广播到所有 assistant token。实现简单，与 GRPO 天然契合（组 = 同一任务的 $N$ 条轨迹）。缺点是长 horizon 下信用分配极粗——40 轮里只有第 12 轮的决策是错的，但 40 轮都被同等惩罚。
/ Turn-level: 每轮给一个 reward（工具调用是否成功、检索是否命中、子目标是否达成），配合折扣因子做跨轮信用分配。信号密得多，但*每一个中间 reward 都是一个新的可 hack 面* —— 给"工具调用成功"加分，模型就学会疯狂发无意义但一定成功的调用。

工业上的折中通常是：*主 reward 用轨迹级的可验证结果*，中间只加*极少量、极难 hack 的* shaping（比如格式合法性、是否在预算内完成），并且给中间项很小的系数。

=== 4.3 Rollout 与训练的 logprob 失配

这是 RL 框架里最经典的"数值对不上"问题，agentic 场景下被放大。

Rollout 用 vLLM/SGLang 生成，训练用 Megatron/FSDP 前向。两边算出来的同一个 token 的 logprob *不完全相等*：kernel 实现不同、batch 组织不同、BF16 累加顺序不同、TP 切分不同。

如果你直接把 rollout 返回的 logprob 当作 $log pi_"old"$，那么在第一个内层 epoch，理论上应该恒等于 1 的重要性比 $pi_theta / pi_"old"$ 会有系统性偏离 —— 于是本该完全不裁剪的第一轮出现了大量裁剪，梯度被无谓地砍掉。

*处方*：
- 用*训练引擎自己再前向一遍*重算 $log pi_"old"$（多一次 forward，但数值自洽）。主流框架的默认做法。
- 或者把这个失配当作监控指标：统计 $|log pi_"train" - log pi_"rollout"|$ 的分布和第一个 epoch 的裁剪比例。裁剪比例在第一个 epoch 应该接近 0，明显大于 0 就说明有问题。

#interview[
  *面试题*：你们 agentic RL 训练 loss 在降，但 agent 的工具调用成功率反而下降，甚至开始不调工具直接编答案。怎么排查？

  这道题几乎就是在问 token masking。排查顺序：
  + *先看 mask*：打印一条轨迹的 `loss_mask`，确认观测 token 全是 0。这是最可能的原因，也最容易验证。
  + *再看归一化分母*：是不是用了 `mask.numel()` 而不是 `mask.sum()`。
  + *再看 reward 设计*：是不是中间 reward 被 hack 了，或者最终 reward 对"不调工具但蒙对"给了正分。
  + *最后看 logprob 失配*：第一个 epoch 的裁剪比例是否异常高。
  + 顺带一提*监控口径*：loss 下降在 agentic RL 里几乎不能说明任何事，必须盯任务成功率、平均轮数、工具调用成功率、截断率这几个业务指标。
]

=== 4.4 变长轨迹的 packing 与截断

轨迹长度 100× 的方差在训练侧同样是问题：一个 batch 里如果有一条 100K token 的轨迹和 31 条 2K 的，按最长 padding 的话 94% 的算力在算 padding。

处理方式与第 12 章的 packing 一脉相承，但有 agentic 特有的点：

- 按 *token 总量*而不是轨迹条数组 batch（`max_tokens_per_batch`）
- 用 FlashAttention varlen path + 轨迹间 block-diagonal mask，禁止跨轨迹 attention
- DP rank 之间按 $sum_i s_i^2$（attention 的平方代价）平衡，而不是按 token 数
- 超长轨迹要有明确策略：截断后是算失败（reward 0）还是算"未完成"（mask 掉不给梯度）。前者会教模型变短，后者浪费算力。DAPO 的 overlong reward shaping 走的是软惩罚的中间路线。

== 五. 环境作为分布式服务

Agentic RL 引入了一个训练栈里从来没有过的东西：*一个需要横向扩展、会失败、有状态、可能收费的外部服务*。

=== 5.1 环境的几种形态与它们的麻烦

#figure(
  table(
    columns: (auto, 1fr, 1.2fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*环境*], [*典型实现*], [*主要工程问题*],
    [代码沙箱],
    [容器 / Firecracker + 测试框架],
    [与 verifier 共用池；镜像预热；依赖装配慢],
    [检索 / 搜索],
    [本地向量库 或 外部搜索 API],
    [外部 API 有*限流和费用*；结果随时间漂移 → 不可复现],
    [浏览器],
    [Playwright / 无头 Chrome],
    [重（每实例几百 MB）、慢、极易 flaky],
    [SWE 仓库],
    [Git 仓库快照 + 测试],
    [仓库状态要能快速重置；镜像体积大],
    [MCP 工具],
    [MCP server 集群],
    [协议层超时与重试；工具版本要钉死],
  ),
  kind: table,
  caption: [五类 agentic RL 环境。共同点是它们都不在 GPU 上，都会失败，都需要独立的容量规划。],
) <table-env-types>

=== 5.2 三条必须处理的工程约束

*(a) 失败要分类，不能一律算 0 分。* 环境超时、沙箱 OOM、外部 API 限流 —— 这些是*基础设施失败*，不是 policy 的错。给它 reward 0 等于用随机噪声污染训练信号。正确做法是标记为 `infra_failure`，*从这个 batch 里剔除*并单独打点。如果 infra 失败率超过 1%，那是要修的系统问题，不是要学的训练信号。

*(b) 可复现性基本无法完全保证，但要尽量收敛。* 外部搜索 API 今天和明天返回不同结果，同一条轨迹重放会拿到不同 reward。缓解：能本地化的就本地化（用固定快照的检索库而不是线上搜索）、缓存环境响应、把环境版本和快照 ID 写进 checkpoint。

*(c) 成本是真实约束。* 浏览器实例、外部 API 调用、沙箱 CPU 都要花钱，而 RL 的 rollout 量是训练量的几十倍。做容量规划时把环境成本和 GPU 成本放在一起算，经常会发现环境侧才是瓶颈。

== 六. 工业案例速查

#figure(
  table(
    columns: (auto, 1fr, 1.1fr),
    stroke: 0.4pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*工作*], [*关键做法*], [*可引用的点*],
    [Tulu 3 (AI2, 2024)],
    [提出 RLVR 这一名词，开源完整配方与 verifier],
    [RLVR 的出处；开源可复现],
    [DeepSeek-R1 (2025)],
    [GRPO + 规则 reward（准确率 + 格式），*明确弃用* neural PRM/ORM],
    [弃用 RM 的理由：reward hacking + 重训代价],
    [Kimi K1.5 (2025)],
    [*partial rollout*、长度惩罚、训推同机部署与权重同步],
    [长尾 straggler 的标准解法出处],
    [Kimi K2 (2025)],
    [大规模合成工具使用数据 + 可验证 reward 与自评 rubric 结合],
    [agentic 能力的数据侧怎么造],
    [DAPO (ByteDance, 2025)],
    [clip-higher、dynamic sampling、token-level loss、overlong shaping],
    [RLVR 四件套，最常被追问],
    [Dr. GRPO (2025)],
    [去掉长度归一化与 std 归一化两处偏差],
    ["响应变长但没变强"的理论解释],
    [Qwen3 (2025)],
    [强到弱蒸馏 + RLVR，思考/非思考模式融合],
    [工业配方里 RLVR 放在哪一段],
    [SWE-RL (Meta, 2025)],
    [用与 oracle patch 的相似度作规则 reward，绕开跑测试的高成本],
    [verifier 太贵时的替代设计],
    [Search-R1 / ReTool 等],
    [检索 / 代码工具进 rollout 循环，显式做观测 token masking],
    [token masking 的出处],
    [verl (HybridFlow)],
    [agent loop、异步 rollout、vLLM/SGLang 后端],
    [最常被问"你用什么框架"],
    [SkyRL],
    [面向长 horizon SWE 任务的异步多轮 rollout 与环境抽象],
    [长 horizon agentic 的开源参考],
  ),
  kind: table,
  caption: [RLVR / Agentic RL 的公开工作速查。面试里引用时务必带上时间点。],
) <table-rlvr-cases>

== 七. 面试题

#interview[
  *Q1*：RLVR 相比经典 RLHF，在*系统架构*上最大的变化是什么？

  A：多出一个 *CPU verifier 集群*，同时可能少掉 *reward model 和 ref model 两个 GPU 池*（去 KL 的配方）。也就是说算力结构从"纯 GPU"变成"GPU + 大规模 CPU 沙箱"。随之而来的新瓶颈是 verifier 吞吐与尾延迟：一个 GRPO 组里只要有一个样本把 verifier 卡住，整组 advantage 就出不来。所以 verifier 侧的三件套是异步化、结果缓存与组内去重、每样本硬超时。
]

#interview[
  *Q2*：GRPO 用二值 reward 时，为什么会有大量 rollout 白烧？怎么解决？

  A：组内全对或全错时标准差为 0、advantage 全 0，整组零梯度，而这类组在训练早期和后期占比能到 30–50%。解法一是 DAPO 的 dynamic sampling：过采样后丢掉准确率为 0 或 1 的组，采到凑满 batch 为止；解法二是难度分层采样，优先采历史准确率在 0.2–0.8 的题。前者实现简单但 rollout 量变成动态的，调度器要支持流式补采；后者省算力但要维护题目状态。
]

#interview[
  *Q3*：Agentic rollout 里，为什么 prefix cache 是生死线？

  A：第 $k+1$ 轮的输入是第 $k$ 轮的完整上下文加新观测，不复用的话 $T$ 轮的总 prefill 是 $O(T^2)$ 量级，20 轮就要多付一个数量级。真正的坑不是"没开"，而是"开了但命中率低"——序列在等环境执行的那几秒里 KV cache 被 LRU 淘汰了。排查看引擎的 prefix cache 命中率，修法是扩大 KV 池、对等待中的序列做 pinning 或换出到 CPU、限制并发轨迹数。
]

#interview[
  *Q4*：多轮轨迹训练时，哪些 token 要 mask？不 mask 会怎样？

  A：只训 assistant 生成的 token，工具观测 token 必须 mask 掉。不 mask 有三个后果：观测常占 60–80% token，梯度大部分来自一个模型原理上无法预测的目标；模型学会自己编造工具输出，推理时不调工具直接幻觉结果；重要性比对观测 token 没有意义，因为那些 token 不是 old policy 采出来的。而且这个 bug *不会让训练崩*，loss 照降，只能靠工具调用成功率这类业务指标发现。另外归一化分母也要用 `mask.sum()` 而不是 `numel()`。
]

#interview[
  *Q5*：Partial rollout 解决什么问题？带来什么新问题？

  A：解决 agentic / 长 CoT 轨迹长度重尾导致的 straggler —— 同步收集一个 batch 要等最长那条。Partial rollout 给每轮设预算，没跑完的轨迹存状态下轮接着跑。新问题是 off-policy：一条轨迹跨了多个 policy 版本。三种处理，从严到松是逐 token 记版本并用对应 logprob 算重要性比、限制最多跨 K 个版本超了就丢、直接当 on-policy。多数框架选第二种。
]

#interview[
  *Q6*：Rollout 引擎和训练引擎算出来的 logprob 不一致，有什么影响？

  A：直接拿 rollout 的 logprob 当 $log pi_"old"$，第一个内层 epoch 本该恒为 1 的重要性比会系统性偏离，导致本不该发生的裁剪，梯度被白砍。根因是两边 kernel、batch 组织、BF16 累加顺序、并行切分都不同。处方是用训练引擎重新前向一遍算 $log pi_"old"$，多一次 forward 换数值自洽；同时把两边 logprob 的差值分布和第一个 epoch 的裁剪比例做成监控——第一个 epoch 的裁剪比例应该接近 0。
]

#interview[
  *Q7*：怎么给 RLVR 训练做 CPU:GPU 容量规划？

  A：稳态条件是 $W gt.eq G dot r dot (1-h) dot t_v \/ u$，其中 $G$ 是 rollout GPU 数、$r$ 是每卡每秒响应产出、$h$ 是缓存与去重命中率、$t_v$ 是单次验证耗时、$u$ 是目标利用率。举例 128 卡、$r = 0.5$、代码 verifier $t_v = 2$ s、$h = 0.4$、$u = 0.7$，算出约 110 个 worker。

  这里有个容易答错的地方：$t_v$ *就该代均值*。排队稳定性只取决于平均服务时间，按 p90 代入是白买机器。重尾带来的问题不在容量而在别处 —— GRPO 要等一组 $N$ 个全部验完，优化器等的是*组内最大值*，实测能到均值的 4–5 倍。压组最大值靠超时和去重，而不是加 worker；更好的做法是异步验证让它离开关键路径。能把"容量"和"组内最大值"这两件事分开讲，这道题就答满了。
]

#interview[
  *Q8*：环境执行失败（沙箱 OOM、API 限流）应该给什么 reward？

  A：*不能给 0*。那是基础设施失败，不是 policy 的行为后果，给 0 等于往训练信号里注入随机噪声，模型会学到一些与真实目标无关的规避行为。正确做法是标为 `infra_failure`、从 batch 里剔除、单独打点。同时把 infra 失败率本身当 SLO 来管——超过 1% 就是要修的系统故障。这道题的考点是能不能区分"环境的错"和"模型的错"。
]

== 八. STAR 故事：把 RLVR + Agentic 串起来

*Situation*：

"我们要在 32B 模型上做代码 agent 的 RL，任务是给定 issue 修仓库代码，用仓库自带测试判对错。第一版流水线跑起来后，一个 iteration 要 40 分钟，GPU 利用率只有 48%，而且训了两天工具调用成功率不升反降。"

*Task*：

"我负责把 iteration 时间压到 15 分钟以内，并定位成功率下降的原因。"

*Action*：

"分两条线。

*先修正确性*，因为跑得再快方向错了也没用。成功率下降但 loss 正常下降，这个组合直接指向 mask。打了一条轨迹的 `loss_mask` 出来，发现工具观测的 token 没有被 mask —— 我们的轨迹里测试输出平均占 70% 的 token，等于七成梯度在教模型预测测试框架的 stdout。修了 mask，顺带发现归一化分母用的是 `numel()` 而不是 `mask.sum()`，等价于按每条轨迹的工具输出多少在缩放学习率。这两处修完，工具调用成功率两天内从 51% 回到 78%。

*再修吞吐*，按 profile 的占比顺序来：
- Rollout 是同步批式的：一批 32 条轨迹一起走一轮、一起等沙箱。改成异步连续批处理，等沙箱的轨迹自动退出运行批次。GPU 利用率 48% → 71%。
- 查引擎的 prefix cache 命中率只有 43%，原因是轨迹等沙箱那几秒 KV 被 LRU 淘汰了。我们刚好因为去掉了 KL 项省出一份 ref model 的显存，全给了 KV 池，同时限制并发轨迹数让活跃集合装得下，命中率到 91%。
- Verifier 侧：一开始是同步跑测试，改成异步池 + 按 `hash(prompt, patch)` 缓存 + 组内去重，去重命中率 38%（同一个 issue 采 16 个 patch，重复的很多）。worker 数按 $W gt.eq G r (1-h) t_v \/ u$ 算了个下界再乘 1.5。
- 还发现约 40% 的 GRPO 组是全对或全错的零梯度组，上了 DAPO 式的 dynamic sampling 过滤掉，同时把 infra 失败（沙箱 OOM、镜像拉取超时）从 reward 0 改成剔除样本并打点 —— 之前这部分噪声一直在污染 advantage。"

*Result*：

"Iteration 时间 40 → 13 分钟，GPU 利用率 48% → 79%，工具调用成功率 51% → 78%，SWE-bench 风格内部评测集通过率提升了 9 个点。masking 和 infra-failure 剔除这两个修复后来被写进了团队的 RL 训练 checklist。"

#insight[
  这个故事的结构值得复用：*先修正确性，再修吞吐*，并且用"loss 正常但业务指标下降"这个信号锁定 masking 类 bug。面试官如果追问，每一环都能往下挖——为什么 KV 池扩了命中率就上去、dynamic sampling 为什么会让 rollout 量变成动态的、infra failure 剔除后 batch size 波动怎么处理。
]

== 九. 配套代码

本章的三个系统问题都有可运行的最小实现，都在 CPU 上跑，都带自校验：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.4pt + gray,
    inset: 5.5pt,
    align: (left, left),
    [*文件*], [*演示内容*],
    [`16_agent_rollout_sched.py`],
    [多轮 rollout 的离散事件仿真：同步批式 vs 异步连续批处理 vs 异步 + partial rollout。仿真里异步把 GPU 利用率从 19% 拉到 88%、makespan 快 4.6×，但*单轨迹延迟反而变差* —— 顺带演示了为什么 p99/p50 在这个场景是个会骗人的指标],
    [`17_rlvr_verifier.py`],
    [Verifier 池建模：重尾延迟 + 超时 + 缓存 + 组内去重，验证容量公式 $W gt.eq G r (1-h) t_v \/ u$，并给出一个真实的数学答案等价性检查器（含常见抽取陷阱）],
    [`18_traj_masking.py`],
    [多轮轨迹的 loss mask：错误版（训观测 token）与正确版并排对比梯度差异；`numel()` vs `mask.sum()` 的归一化陷阱；GRPO 零梯度组与 dynamic sampling 过滤；rollout/训练 logprob 失配对裁剪比例的影响],
  ),
  kind: table,
  caption: [第 15 章配套代码。`make demo-16` / `demo-17` / `demo-18` 直接运行。],
) <table-ch15-code>
