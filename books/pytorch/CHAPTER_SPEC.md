# PyTorch 面试书 — 章节写作规范（子代理必读）

这本书的目标读者：正在准备 PyTorch / 训练框架 / AI Infra 岗位面试的工程师。
定位：**常见高频面试题**，不是冷门八股。凡是"面试官几乎不会问"的边角 API，一律不写。

## 0. 硬性约束

- **只写自己被分配的 `chapters/*.typ` 文件。** 不要动 `template.typ`、`book.typ`、其他章节。
- 语言：**中文正文 + 英文技术术语原样保留**（"autograd 图"、"graph break"、"caching allocator"）。
- 语气：直接、密度高、无套话。一句话讲一个点。不要"综上所述"、"值得注意的是"这类填充。
- **代码必须是真能跑的 PyTorch 2.10**（环境：torch 2.10.0+cu128，CUDA 12.8，2× A100-SXM4-80GB）。
  写完自己在脑子里过一遍 shape 和 API 签名。不确定的 API 不要编。
- **不许编造数字。** 要给性能数字时，要么写清是"数量级估算"，要么写清测量条件（GPU 型号、shape、dtype）。
  宁可写"A100 上这类 kernel 通常是 memory-bound，收益主要来自省一次 HBM 往返"，也不要写一个假的 "1.73×"。
- 每章长度：**250–450 行 Typst**。宁可密不可松。

## 1. 章节骨架

每个文件第一行必须是：

```typ
#import "../template.typ": *
```

然后是一个 level-1 标题（`= 标题`），之后用 `==` / `===` 分节。标题不要自己编号，
`template.typ` 里已经 `set heading(numbering: "1.1.1")`。

推荐结构（按章节内容灵活调整，但**必须有开头一段"这章讲什么"和结尾的面试考点**）：

```typ
#import "../template.typ": *

= <章标题>

<一段话：这章解决什么问题、为什么面试爱问、与前后章的关系。不要写"本章将介绍"。>

== <第一个主题>
...

== 面试考点

#interview[
  *Q1*：<问题>

  A：<答案，2-5 句，给出关键机制或数字>
]
```

结尾的面试考点：**每章 5–10 题**，每题一个 `#interview[...]` block。
题目必须是真高频题（"`view` 和 `reshape` 的区别"、"DDP 为什么要分 bucket"），
答案要能直接背下来说给面试官听。

## 2. 可用的 callout（来自 template.typ）

| helper | 渲染 | 用途 |
|---|---|---|
| `#note[...]` | 蓝色 *Note.* | 补充说明、版本差异 |
| `#warn[...]` | 橙色 *Warning.* | 坑、会踩的错、不兼容 |
| `#insight[...]` | 绿色 *Key insight.* | 一节的核心结论，每节最多一个 |
| `#interview[...]` | 紫色 *面试考点.* | 面试题 Q/A |
| `#story[...]` | 灰色 | 真实工程故事（可选，少用） |
| `#formula[...]` | 居中方框 | 关键公式 |

## 3. 可用的图元（CetZ，来自 template.typ）

**图不是装饰。** 每章 1–3 个，只在图能替代一段解释时才画。

下面每个调用都**已实测编译 + 渲染通过**，照抄参数形状即可（数据换成你自己的）。
**不在这个列表里的图元不要用** —— 换成 `#table(...)` 或代码块。

图统一包在 `#figure` 里加 caption（全书惯例）：

```typ
#figure(
  align(center, stride-view(shape: (3, 4), stride: (4, 1), n-storage: 12)),
  caption: [`reshape(3,4)` 后 stride=(4,1)，逐行走 storage。],
) <fig-stride>
```

### 3.1 PyTorch 专用

```typ
// 逻辑视图 vs 一维 storage —— 讲 stride / view / transpose / slice 的神器
#stride-view(shape: (3, 4), stride: (4, 1), offset: 0, n-storage: 12,
             title: "x = arange(12).reshape(3,4)")

// 竖向 shape 流水线：stages = ((名字, shape 字符串, 右侧注释), ...)
#shape-pipeline(stages: (
  ("input", "(B, S, H)", "hidden states"),
  ("qkv",   "(B, S, 3H)", "一次 fused Linear"),
))

// autograd 图：forward tensor 链，左侧 op，右侧 grad_fn，反向红色虚线
// tensors = ((名字, 备注), ...)；ops = ((forward 表达式, grad_fn 名), ...)
// 约束：ops 的长度必须 = tensors 的长度 - 1
#autograd-graph(
  tensors: (("x", "(B, H) leaf"), ("h", "(B, O)"), ("loss", "() scalar")),
  ops: (("h = x @ W", "MmBackward0"), ("loss = h.sum()", "SumBackward0")),
)
```

### 3.2 通用

```typ
// 横向流程框
#flow-boxes(boxes: ("Python bytecode", "FX Graph", "Triton kernel"))

// 多 stream 时间线。streams = ((行名, ((kind, 时长), ...)), ...)
// 每行的块是【顺序排列】的，不是 (起点, 长度)；想留空白就插 ("wait", n)
// kind 建议用 "compute" / "comm" / "wait"（有配色）
#timeline(streams: (
  ("compute", (("compute", 10), ("wait", 4))),
  ("comm",    (("wait", 10), ("comm", 4))),
), title: "AllReduce 与 backward overlap")

// 多配置显存堆叠对比。configs = ((配置名, ((段名, GB), ...)), ...)
#mem-stack(configs: (
  ("DDP",    (("params", 14), ("grads", 14), ("optim", 56))),
  ("ZeRO-3", (("params", 2),  ("grads", 2),  ("optim", 7))),
))

// 单配置显存分解（横向条，每条一段，标注 GB）。位置参数，无 entries: 前缀
#mem-bar((("params", 14), ("grads", 14), ("optim", 56)))

// 单条横向堆叠百分比条（step time 拆解）
#stacked-bar(entries: (("compute", 62), ("comm", 21), ("dataloader", 17)))

// 横向条形图（延迟/吞吐对比）。位置参数
#hbar-chart((("eager", 100), ("compile", 68)), unit: "ms")

// 折线图。series = ((名字, ((x, y), ...)), ...)
#line-plot(series: (("throughput", ((1, 100), (2, 190), (4, 360))),),
           x-label: "GPUs", y-label: "samples/s")

// 阶梯表：每行是位置参数 (版本, 核心思路, 成本/收益)
#ladder(("v0 eager", "无优化", "baseline"),
        ("v1 compile", "Inductor fusion", "省 kernel launch"))

// 居中公式框
#formula[$ "mem" = 2 P + 2 P + 12 P $]
```

### 3.3 分布式专用

```typ
// 拓扑网格。groups 是【嵌套】列表，每行一个 list；group-labels = ((组 id, 名字), ...)
#topology-grid(rows: 2, cols: 4,
               groups: ((0, 0, 0, 0), (1, 1, 1, 1)),
               group-labels: ((0, "dp0"), (1, "dp1")))

// TP 权重切分示意。mode 取 "column" / "row"
#tp-partition(mode: "column", tp: 4)

// PP 调度时间线。schedule = 每个 stage 一行，行内是 ((kind, 宽度), ...)
// kind: "F" forward / "B" input-grad / "W" weight-grad / "R" recompute / "_" bubble
#pipeline-schedule(stages: 2, schedule: (
  (("F", 1), ("F", 1), ("B", 1), ("_", 1)),
  (("_", 1), ("F", 1), ("F", 1), ("B", 1)),
))

// 通信成本表：header 默认 (策略, 每层通信量, 同步次数, 备注)，每行 4 个位置参数
#cost-table(("DDP", "2 |P|", "1 / step", "梯度 AllReduce"))

// Ring 拓扑
#ring-diagram(n: 4)

// MoE token → expert 路由
#moe-dispatch(n-tokens: 6, n-experts: 4)
```

写完必须编译通过（见 §6），编译错误必须自己修完。

## 4. Typst 语法坑（最容易翻车的地方）

1. **数学是 Typst 语法，不是 LaTeX。**
   - 行内：`$x_i$`；块级：`$ y = W x $`（块级要求 `$` 两侧有空格）
   - 多字符标识符要引号或用 `#`：`$"mean"(x)$`、`$hat(y)$`、`$alpha$`、`$times$`、`$approx$`
   - 分数：`$a / b$`；求和：`$sum_(i=1)^n x_i$`
   - **不要写** `\frac{}{}`、`\times`、`\alpha`、`$$...$$`
2. **代码块用三个反引号 + `python` / `bash` / `text`**，内容原样写，不需要转义。
3. 正文里的 `_`、`*`、`@` 有 markup 含义：
   - 文件名/标识符放进行内代码：`` `find_unused_parameters` ``，不要裸写 `find_unused_parameters`（会被当斜体）
   - 需要字面量星号写 `\*`
4. `*粗体*` 和 `_斜体_` 是 Typst 的强调语法（不是 markdown 的 `**`）。**一律用 `*...*`。**
5. 表格：
   ```typ
   #table(
     columns: (auto, 1fr, 1fr),
     stroke: 0.4pt + gray,
     inset: 5pt,
     align: (left, left, left),
     [*列 1*], [*列 2*], [*列 3*],
     [a], [b], [c],
   )
   ```
   单元格是 content block `[...]`，逗号分隔，**行末也要逗号**。
6. callout 里放代码块/表格是允许的，但 `#interview[...]` 内部尽量只放文字和行内代码。
7. 中文标点直接写。引号用 `"..."` 或中文引号都可以。

## 5. 内容规则

- **不要重复其他章。** 分配给你的章节范围外的内容，用一句"见第 N 章"带过。
- 讲机制时给出"为什么这样设计"，不要只列 API。面试官问的是原理。
- 涉及"常见 bug"的地方写成 `#warn[...]`，并给出**能复现的最小代码**和正确写法。
- 版本敏感的内容标明版本：`torch 2.4+`、`FSDP2 (torch 2.6+)`。不确定具体版本就写"较新版本"。
- Part 5（coding problems，第 23–29 章）的代码：
  - 必须是**面试白板能写出来的规模**（20–60 行），不是生产级封装
  - 给出参考实现 + 与 `torch.nn` 官方实现的等价性校验方式
  - 落地文件在 `python/pytorch/interview/`（由另一批任务负责），章节里引用路径即可

## 6. 交付前自检（必须做）

```bash
cd /home/duo.an/workspaces/playground-cuda/books/pytorch
# --root . 是必须的：章节里 `#import "../template.typ"` 会被判为逃出 project root
~/.local/bin/typst compile --root . chapters/<你的文件>.typ /tmp/chk_<名字>.pdf

# 整本编译（在 books/pytorch/ 下不需要 --root）
~/.local/bin/typst compile book.typ book.pdf
```

- **必须零 error**（warning 可以留）。单独编译时标题不会有书里的样式，这是正常的。
- 编译通过后，再检查一遍：有没有 `\frac`、`$$`、裸下划线标识符、表格漏逗号。

## 7. 参考章节（风格标杆）

读这两个文件感受语气和密度：
- `/home/duo.an/workspaces/playground-cuda/books/distributed-training/chapters/P0_pytorch_basics.typ`
- `/home/duo.an/workspaces/playground-cuda/books/moe/chapters/03_dispatch.typ`
