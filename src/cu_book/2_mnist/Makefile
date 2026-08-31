# MNIST MLP - Chapter 02
# Supports mixed .c (gcc) and .cu (nvcc) sources side-by-side.
# Add a new vN.c or vN.cu and it will be picked up automatically.

CUDA_HOME  ?= /usr/local/cuda
NVCC       ?= $(CUDA_HOME)/bin/nvcc
CC         ?= gcc

OUT_DIR     = out

CFLAGS     ?= -O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -fopenmp
NVCCFLAGS  ?= -O2 -std=c++17
LDLIBS     ?= -lm -fopenmp

C_SRCS     := $(wildcard v*.c)
CU_SRCS    := $(wildcard v*.cu)
C_BINS     := $(patsubst %.c,$(OUT_DIR)/%,$(C_SRCS))
CU_BINS    := $(patsubst %.cu,$(OUT_DIR)/%,$(CU_SRCS))
TARGETS    := $(C_BINS) $(CU_BINS)

NAMES      := $(basename $(notdir $(TARGETS)))

.PHONY: all clean run help $(NAMES)
.DEFAULT_GOAL := all

all: $(TARGETS)

# Aliases: `make v3` -> builds out/v3
$(NAMES): %: $(OUT_DIR)/%

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%: %.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(OUT_DIR)/%: %.cu | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@if [ -z "$(APP)" ]; then echo "Usage: make run APP=<name>  (e.g. make run APP=v3)"; exit 1; fi
	./$(OUT_DIR)/$(APP)

clean:
	rm -rf $(OUT_DIR)

help:
	@echo "MNIST MLP - Chapter 02"
	@echo ""
	@echo "Auto-detected targets:"
	@for t in $(TARGETS); do echo "  $$t"; done
	@echo ""
	@echo "Usage:"
	@echo "  make            build everything"
	@echo "  make v3         build one target (e.g. v3, v4)"
	@echo "  make run APP=v3 build all, then run out/v3"
	@echo "  make clean      remove out/"
