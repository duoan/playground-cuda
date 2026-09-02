#import "../template.typ": *

= 附录 B：Nsight Compute 使用指南

这本书里所有的性能数字都来自 `ncu` (Nsight Compute)。这个附录告诉你怎么用它。

== 为什么需要 ncu

Wall-time（用 CUDA event / host timer 测的 kernel 时间）能告诉你"这个 kernel 花了多久"，但*不能*告诉你：

- 它慢的原因是访存慢、计算慢、还是 stall 在 barrier？
- 它把 HBM 用到多少、SM 用到多少？
- 它有没有 bank conflict、warp divergence、register spill？
- 上一版和这一版的性能差距是从哪里来的？

`ncu` 是 NVIDIA 官方的 CUDA kernel profiler，能通过硬件性能计数器 (hardware performance counters) 抓到几百个 metric。它有两种模式：

- *交互式 GUI*：`ncu-ui`，能看火焰图、SASS 高亮、metric roofline。适合分析单个 kernel。
- *命令行*：`ncu`，能被 shell / Makefile 自动化。适合 CI / regression。

本书所有的实测数字都是命令行模式抓的。

== 权限

在很多云 / 共享环境上，`ncu` 会报：

```
ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

原因是驱动里 `RmProfilingAdminOnly=1`（默认值）。解决办法：

1. *sudo 直接跑*（如果你有 sudo 权限）：
   ```bash
   sudo -n ncu -k regex:my_kernel ./my_binary
   ```
2. *全局放开*（需要重启 nvidia-persistenced 或 reboot）：
   ```
   sudo nvidia-smi -q -d SUPPORTED_CLOCKS   # 确认驱动版本
   # /etc/modprobe.d/nvidia.conf 里加:
   options nvidia NVreg_RestrictProfilingToAdminUsers=0
   ```
3. *WSL / Docker*：需要以 `--privileged` 或加 `--cap-add=SYS_ADMIN` 运行容器。

Google Colab / 大部分云 notebook 没有这个权限，`ncu` 完全用不了。

== 命令行基本用法

=== 最简：抓所有 kernel 的默认 metric

```bash
ncu ./my_binary
```

会打印每个 kernel 的 "SpeedOfLight" section（HBM 用率、SM 用率、TC 用率），大约每 kernel 输出 30 行。

=== 过滤到特定 kernel

```bash
ncu -k regex:my_kernel_v2 ./my_binary
```

`regex:` 前缀让 `-k` 接受正则；不加就要 exact match，很不方便。

=== 只跑前 N 次 launch

如果一个 kernel 被调用很多次（比如 batch loop），profile 每次都会拖慢很多：

```bash
ncu --launch-count 3 -k regex:my_kernel ./my_binary
```

=== 抓指定 metric，输出 CSV

```bash
ncu -k regex:my_kernel \
    --csv \
    --log-file out.csv \
    --metrics gpu__time_duration.sum,dram__bytes.sum.pct_of_peak_sustained_elapsed \
    ./my_binary
```

这是我们 book 里 `bench/run_bench.py` 用的方式。

=== 抓完整 section（比命令行 metric 全）

```bash
ncu --set full -k regex:my_kernel --export report ./my_binary
# 然后:
ncu-ui report.ncu-rep
```

`full` set 会抓所有 section（约 30 个）+ SASS 采样。慢，但一次搞定。

== Section vs. Metric

`ncu` 的 metric 分两个层级：

- *Section*：一组相关 metric 的集合，对应 GUI 里一个可折叠 panel。例如 `SpeedOfLight` = HBM/SM/TC 用率概览；`Occupancy` = warp 数量 / 寄存器 / smem 三个维度的 occupancy 分析。
- *Metric*：单个数值指标，例如 `dram__bytes.sum.pct_of_peak_sustained_elapsed`。

命令行 `--section` 加 section 名一次抓整组，`--metrics` 加逗号分隔的 metric 名精确控制。

常用 section：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray, inset: 5pt, align: (left, left),
    [*Section 名*], [*用来看什么*],
    [`SpeedOfLight`], [第一眼：memory-bound 还是 compute-bound],
    [`LaunchStats`], [grid/block/register/smem 配置],
    [`Occupancy`], [warp 数是被谁限制的（reg / smem / block dim）],
    [`SchedulerStats`], [warp 有多少 issue slot 空转],
    [`WarpStateStats`], [warp stall 的原因分类（LG throttle / MIO / Barrier ...）],
    [`SourceCounters`], [热点行 + 每行的 stall 原因（需 `-lineinfo`）],
    [`InstructionStats`], [SASS 指令 mix（FMA / LDG / STG / SHFL ...）],
    [`MemoryWorkloadAnalysis`], [L1/L2/HBM 的字节数、hit rate、bank conflict],
    [`ComputeWorkloadAnalysis`], [FMA / ALU / SFU / MUFU / tensor pipe 各自的忙度],
  ),
  caption: [*Table:* Nsight Compute 常用 section 名与用途。*Section 名* 对应 `ncu --section` 或 GUI panel 名；*用来看什么* 列说明第一眼 diagnostic 目标（SpeedOfLight、LaunchStats、Occupancy 等）。],
  kind: table,
)

*Observation*：*先用 `SpeedOfLight` 定性*——HBM % vs TC % vs SM % 决定往下挖 Occupancy 还是 MemoryWorkloadAnalysis；本书 bench 表里的 TC %/HBM % 多来自 SpeedOfLight 或等价 metric。

== Metric 命名规范

`ncu` 的 metric 名看起来像天书，其实是有语法的：

```
<pipeline>__<subject>.<statistic>[.<qualifier>]
```

拆解 `dram__bytes.sum.pct_of_peak_sustained_elapsed`：

- `dram` = pipeline，指 HBM 控制器
- `bytes` = subject，指字节数
- `.sum` = statistic，把所有 SM 上的量加起来（也可以是 `.avg`, `.max`, `.per_second`）
- `.pct_of_peak_sustained_elapsed` = qualifier，占硬件持续峰值的百分比（分母是 elapsed wall time）

其它常见 pipeline 前缀：

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray, inset: 5pt, align: (left, left),
    [*前缀*], [*含义*],
    [`gpu__`], [整个 GPU 层面（例如 `gpu__time_duration.sum` kernel 时间）],
    [`sm__`], [SM 层面聚合（多个 SM 的和 / 均值）],
    [`smsp__`], [SM sub-partition 层面（一个 SM 有 4 个 sub-partition，各自有 warp scheduler）],
    [`l1tex__`], [L1 texture cache（also 处理 shared memory）],
    [`lts__`], [L2 cache（"large text slice"，历史命名）],
    [`dram__`], [HBM DRAM],
    [`launch__`], [launch config（grid, block, regs, smem）],
  ),
  caption: [*Table:* ncu metric 名的 pipeline 前缀含义。前缀 `gpu__`/`sm__`/`smsp__`/`l1tex__`/`lts__`/`dram__`/`launch__` 对应聚合层级；读 metric 时需先看前缀再读 subject 与 statistic。],
  kind: table,
)

*Observation*：*同一次 kernel profile 要串三层*——`launch__` 看配置是否合理，`dram__`/`l1tex__` 看访存，`sm__`/`smsp__` 看 compute 与 warp stall；本书 ch4/ch10 的 `issued/32` 来自 `smsp__` 层。

== 内存-bound kernel 该抓什么

```
--metrics \
  gpu__time_duration.sum,\
  dram__bytes.sum,\
  dram__bytes.sum.pct_of_peak_sustained_elapsed,\
  sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
```

看：

- `dram__bytes.sum` / time = 实测 HBM GB/s
- `dram__bytes.sum.pct_of_peak_sustained_elapsed` = HBM %
- 如果 HBM % 低但 kernel 慢，`long_scoreboard` stall 高说明 memory latency 掩盖不住（warp 不够多）
- `l1tex bytes` 高但 `dram bytes` 低说明 L2 命中好——micro-benchmark 上常见（本书 vector_add 章里就踩过）

== Compute-bound kernel 该抓什么

```
--metrics \
  gpu__time_duration.sum,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
```

看：

- `sm__pipe_tensor_cycles_active` = tensor core 忙度（GEMM 该 > 60% 才算健康）
- `smsp__sass_thread_inst_executed_op_ffma` = CUDA core FMA 计数（tensor core kernel 上会低）
- `sm__warps_active` = warp 占用率
- `bank_conflicts_pipe_lsu_mem_shared_op_ld` = shared memory bank conflict 次数
- `stalled_barrier` 高 = `__syncthreads()` 等太久

== Roofline 视图

`ncu-ui` 里有一个 "Roofline" tab，能自动画：

- 硬件 peak（HBM 斜线 + Compute 平线）
- 你的 kernel 落在哪个点上

规则：

- 落在 memory line 上 → memory-bound，优化访存
- 落在 compute line 上 → compute-bound，优化 FLOP 效率
- 两者中间 → mixed，两边都能优化

命令行 `ncu --set roofline` 抓所需 metric，`--export` 出报告后在 GUI 里看。

== 常见坑

1. *ncu 让 kernel 变慢很多*。抓 metric 会让 kernel 慢 2-100x（因为要多次 replay 采样）。别用 wall-time 看进程整体，用 `gpu__time_duration.sum` 单个 kernel 的时间。
2. *L2 cache 让 GB/s 虚高*。数据集比 40MB 小时，逻辑 GB/s 远超实测 HBM GB/s。看 `dram__bytes.sum` 才准。
3. *小 kernel 的 pct_of_peak 偏低*。kernel 短于 100 μs 时，启动 / 结束窗口会把 pct 拉低，但 effective GB/s 是准的。
4. *`--kernel-name my_k` 不匹配*。默认是 exact，要加 `regex:` 前缀。
5. *Volta 以下不支持*。ncu 只支持 Volta（sm_70）及以上。Kepler / Pascal 得用旧的 `nvprof`。
6. *多 kernel 同名会各 profile 一次*。想只抓一次，加 `--launch-count 1`。

== 本书 bench 目录布局

```
book/bench/
  Makefile           # 每章一个 target
  run_bench.py       # 通用 ncu 包装
  common.sh          # sudo ncu wrapper
  01_vector_add.csv  # ncu 原始输出
  01_vector_add.md   # 人可读表格
  01_vector_add.typ  # typst 表格（章节里 include）
  ...
```

要重跑某一章的 bench：

```bash
cd book/bench && make 01   # 或 02, 03, ...
```

会自动重新编译 kernel、跑 ncu、更新表。然后 `typst compile book.typ book.pdf` 会拿到新数字。
