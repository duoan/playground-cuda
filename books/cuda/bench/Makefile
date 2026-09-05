# One target per chapter.  Each target:
#   1. rebuilds the underlying binary
#   2. calls run_bench.py with the right kernel regex + bytes-per-launch
#   3. emits <chapter>.csv (raw), <chapter>.md (human), <chapter>.typ (embed)
#
# All chapters together: `make -j1 all`  (serialize — profiling doesn't like
# multiple ncu processes racing on the same GPU).

REPO := $(abspath ../..)
BUILD := $(REPO)/build
RUN := ./run_bench.py

.PHONY: all
all: 01 02 03 04 05 06 07 08 09 10

# ---------------------------------------------------------------------------
# 01 vector_add: use log2N=27 (3 * 512 MB = 1.5 GB, ≫ L2 40 MB) so we hit real
# HBM and cache-hit artifacts disappear.  Bytes = 3 * 4 * (1<<27) = 1.5 GB.
# ---------------------------------------------------------------------------
.PHONY: 01
01: $(BUILD)/01_vector_add/bench
	$(RUN) $< vector_add 01_vector_add mem \
	    --bytes-per-launch $$((3 * 4 * (1 << 27))) \
	    --binary-args "27"

# ---------------------------------------------------------------------------
# 02 reduce_sum: log2N=27 → ~128M elements = 512 MB input.  Stage-1 kernels
# read the whole thing; subsequent stages get much smaller.
# ---------------------------------------------------------------------------
.PHONY: 02
02: $(BUILD)/02_reduce_sum/bench
	$(RUN) $< reduce_sum 02_reduce_sum mem \
	    --bytes-per-launch $$((4 * ((1 << 27) + 37))) \
	    --binary-args "27"

# ---------------------------------------------------------------------------
# 03 softmax: 2 reads + 1 write of the same (B, S, H) tensor.  We probe the
# main app with its default problem size; users can override with SOFTMAX_N.
# ---------------------------------------------------------------------------
SOFTMAX_BYTES ?= $(shell echo $$((3 * 4 * 1024 * 1024)))
.PHONY: 03
03: $(BUILD)/03_softmax/bench
	$(RUN) $< softmax 03_softmax mem --bytes-per-launch $(SOFTMAX_BYTES)

# ---------------------------------------------------------------------------
# 04 matmul: compute-bound; GB/s not the right yardstick, we look at TC%.
# ---------------------------------------------------------------------------
.PHONY: 04
04: $(BUILD)/04_matmul/bench
	$(RUN) $< matmul 04_matmul compute

# ---------------------------------------------------------------------------
# 05 layernorm: 1 read + 1 write of (B, S, H)
# ---------------------------------------------------------------------------
LN_BYTES ?= $(shell echo $$((2 * 4 * 1024 * 1024)))
.PHONY: 05
05: $(BUILD)/05_layernorm/bench
	$(RUN) $< layernorm 05_layernorm mem --bytes-per-launch $(LN_BYTES)

.PHONY: 06
06: $(BUILD)/06_mlp/bench
	$(RUN) $< mlp 06_mlp compute

.PHONY: 07
07: $(BUILD)/07_attention/bench
	$(RUN) $< attention 07_attention compute

.PHONY: 08
08: $(BUILD)/08_flash_attention_v1/bench
	$(RUN) $< flash_attention 08_flash_attention_v1 compute

.PHONY: 09
09: $(BUILD)/09_flash_attention_v2/bench
	$(RUN) $< flash_attention 09_flash_attention_v2 compute

.PHONY: 10
10: $(BUILD)/10_flash_attention_v3/bench
	$(RUN) $< flash_attention 10_flash_attention_v3 compute

# rebuild-on-demand delegating to top-level Makefile
$(BUILD)/%:
	$(MAKE) -C $(REPO) $@
