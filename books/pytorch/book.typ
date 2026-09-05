// Main entry: `typst compile book.typ`

#import "template.typ": *

#show: book.with(
  title: "PyTorch 面试通关手册",
  subtitle: "从 Tensor 与 autograd，到 torch.compile、分布式，与从零手写模型",
  author: "duo.an",
)

#include "chapters/00_preface.typ"

// ==== 第一部分：基础 ====
#include "chapters/01_tensor.typ"
#include "chapters/02_ops_shapes.typ"
#include "chapters/03_module.typ"
#include "chapters/04_data.typ"
#include "chapters/05_train_loop.typ"

// ==== 第二部分：原理 ====
#include "chapters/06_autograd.typ"
#include "chapters/07_dispatcher.typ"
#include "chapters/08_memory.typ"
#include "chapters/09_cuda_exec.typ"
#include "chapters/10_determinism.typ"
#include "chapters/11_profiling.typ"

// ==== 第三部分：torch.compile ====
#include "chapters/12_dynamo.typ"
#include "chapters/13_aotautograd.typ"
#include "chapters/14_inductor.typ"
#include "chapters/15_compile_practice.typ"
#include "chapters/16_export_deploy.typ"

// ==== 第四部分：分布式 ====
#include "chapters/17_dist_basics.typ"
#include "chapters/18_ddp.typ"
#include "chapters/19_zero_fsdp.typ"
#include "chapters/20_tp_pp.typ"
#include "chapters/21_dtensor_mesh.typ"
#include "chapters/22_dist_debug_ckpt.typ"

// ==== 第五部分：Coding Problems ====
#include "chapters/23_layers_from_scratch.typ"
#include "chapters/24_custom_autograd.typ"
#include "chapters/25_classic_models.typ"
#include "chapters/26_transformer.typ"
#include "chapters/27_advanced_models.typ"
#include "chapters/28_training_tricks.typ"
#include "chapters/29_whiteboard_drills.typ"

// ==== 附录 ====
#include "chapters/A_errors.typ"
#include "chapters/B_checklist.typ"
#include "chapters/C_version_delta.typ"
