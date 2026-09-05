# Makefile for distributed-training demos.
#
# Usage:
#   make demo-01                 # run collectives demo on 4 CPU procs (gloo)
#   NPROC=8 make demo-01         # override world size
#   BACKEND=nccl make demo-04    # force NCCL (auto-picked when CUDA is available)
#   make all                     # run every demo in sequence
#
# All demos self-check with `assert torch.allclose(...)`; a run that
# reaches the "✓" print at the end of stdout has passed.

NPROC ?= 4
PYTHON ?= python3
TORCHRUN ?= torchrun
RUN = $(TORCHRUN) --standalone --nproc-per-node=$(NPROC)

.PHONY: all clean demo-01 demo-02 demo-03 demo-04 demo-05 demo-06 demo-07 demo-08 demo-09 demo-10 demo-11 demo-12 demo-13 demo-14 demo-15 demo-16 demo-17 demo-18 demo-19 demo-20 demo-21 demo-22 demo-23

demo-01:  ## Collectives + hand-written Ring AllReduce
	$(RUN) 01_collectives.py

demo-02:  ## DDP from scratch (bucket + hook + async AR)
	$(RUN) 02_ddp_from_scratch.py

demo-03:  ## ZeRO-1 / ZeRO-2 / ZeRO-3
	$(RUN) 03_zero.py

demo-04:  ## TP: ColumnParallelLinear + RowParallelLinear + FFN + Attention
	$(RUN) 04_tp_column_row.py

demo-05:  ## Sequence Parallel + TP
	$(RUN) 05_sp_with_tp.py

demo-06:  ## Pipeline Parallel: GPipe + 1F1B
	$(RUN) 06_pp_1f1b.py

demo-07:  ## Ring Attention with Flash-style LSE merge
	$(RUN) 07_ring_attention.py

demo-08:  ## Ulysses (a2a on head dim). Requires GPU/NCCL.
	$(RUN) 08_ulysses_attention.py

demo-09:  ## Expert Parallel MoE. Requires GPU/NCCL.
	$(RUN) 09_ep_moe.py

demo-10:  ## Sequence packing (BFD/FFD) + varlen attention. Single process.
	$(PYTHON) 10_seq_packing.py

demo-11:  ## Compute/comm overlap (two CUDA streams). Requires GPU.
	$(RUN) 11_overlap_2stream.py

demo-12:  ## AdamW / Lion / LAMB / Muon reference impl + torch.optim parity
	$(PYTHON) 12_optimizers.py

demo-13:  ## LR schedules: warmup+cosine, WSD, WSD-rewarm (with ASCII plot)
	$(PYTHON) 13_lr_schedules.py

demo-14:  ## Reproduce 6 canonical NaN pathologies + fixes
	$(PYTHON) 14_nan_reproducer.py

demo-15:  ## TrainingMonitor: grad_norm / activation / weight-norm alerts
	$(PYTHON) 15_training_monitor.py

demo-16:  ## Agentic rollout scheduling: sync vs async vs partial rollout
	$(PYTHON) 16_agent_rollout_sched.py

demo-17:  ## RLVR verifier pool: timeouts, cache/dedup, capacity planning
	$(PYTHON) 17_rlvr_verifier.py

demo-18:  ## Trajectory loss masking, GRPO degeneracy, logprob mismatch
	$(PYTHON) 18_traj_masking.py

demo-19:  ## CP cross-token reductions: loss denominator, MoE aux, pooling, grad-norm
	$(RUN) 19_cp_loss_and_metrics.py

demo-20:  ## CP data path: rank mapping, zigzag, RoPE/mask, label shift, halo
	$(RUN) 20_cp_dataloader_halo.py

demo-21:  ## PP from scratch: autograd boundary, deadlock, 4 schedules verified
	$(RUN) 21_pp_from_scratch.py

demo-22:  ## PP schedule cost: measured bubble vs formula, Gantt, activation peak
	$(PYTHON) 22_pp_schedule_sim.py

demo-23:  ## PP integration: rank layout, partitioning, tied weights, DP sync
	$(TORCHRUN) --standalone --nproc-per-node=4 23_pp_integration.py

all: demo-01 demo-02 demo-03 demo-04 demo-05 demo-06 demo-07 demo-10 demo-12 demo-13 demo-14 demo-15 demo-16 demo-17 demo-18 demo-19 demo-20 demo-21 demo-22 demo-23
	@echo ""
	@echo "GPU-only demos (run separately with GPUs):"
	@echo "  make demo-08   # Ulysses"
	@echo "  make demo-09   # EP MoE"
	@echo "  make demo-11   # overlap benchmark"

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
