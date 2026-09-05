// Main entry: `typst compile book.typ`

#import "template.typ": *

#show: book.with(
  title: "大模型分布式训练面试通关手册",
  subtitle: "从 AllReduce 到 DualPipe，从 ZeRO 到 RLHF Rollout",
  author: "duo.an",
)

#include "chapters/00_preface.typ"

// ==== 第零部分：训练算法基础 ====
#include "chapters/P0_pytorch_basics.typ"
#include "chapters/P1_llm_arch.typ"
#include "chapters/P2_optimizer.typ"
#include "chapters/P3_lr_schedule.typ"
#include "chapters/P4_numerical_stability.typ"
#include "chapters/P5_training_recipe.typ"

// ==== 主体：并行策略 ====
#include "chapters/01_basics.typ"
#include "chapters/02_scaling_laws.typ"
#include "chapters/03_dp_ddp.typ"
#include "chapters/04_zero_fsdp.typ"
#include "chapters/05_tp_sp.typ"
#include "chapters/06_pp.typ"
#include "chapters/07_cp_long_ctx.typ"
#include "chapters/08_ep_moe.typ"
#include "chapters/09_precision.typ"
#include "chapters/10_recompute_offload.typ"
#include "chapters/11_overlap.typ"
#include "chapters/12_dataloader.typ"
#include "chapters/13_multimodal.typ"
#include "chapters/14_rl_training.typ"
#include "chapters/15_agentic_rl_rlvr.typ"
#include "chapters/16_infra_stability.typ"
#include "chapters/M_interview_math.typ"
#include "chapters/A_appendix.typ"
#include "chapters/B_papers.typ"
#include "chapters/D_interview_problems.typ"
#include "chapters/E_stories.typ"
#include "chapters/F_framework_delta.typ"
