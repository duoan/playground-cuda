// Main entry: `typst compile book.typ`

#import "template.typ": *

#show: book.with(
  title: "Sparse MoE 训练实战",
  subtitle: "从 top-k gating 到 EP + All-to-All 的完整指南",
  author: "duo.an",
)

#include "chapters/00_preface.typ"
#include "chapters/01_intro.typ"
#include "chapters/02_router.typ"
#include "chapters/03_dispatch.typ"
#include "chapters/04_walkthrough.typ"
#include "chapters/05_load_balancing.typ"
#include "chapters/06_single_gpu_perf.typ"
#include "chapters/07_distributed.typ"
#include "chapters/08_model_layer.typ"
#include "chapters/09_pitfalls.typ"
#include "chapters/A_appendix.typ"
#include "chapters/D_interview_problems.typ"
