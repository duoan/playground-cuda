CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
BUILD_DIR ?= build
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_80
CUDA_SRC_DIR ?= src/cuda
CUDA_PRACTICE_DIR ?= src/cuda/practice

# ---- flat kernels: src/cuda/*.cu -> build/* ----
SRC_FILES := $(wildcard $(CUDA_SRC_DIR)/*.cu)
TARGETS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SRC_FILES))

# ---- versioned kernels: src/cuda/<kernel>/<version>.cu -> build/<kernel>/<version> ----
# Wildcard is one level deep; skip common/ (headers only, no .cu) and practice/ (built separately).
VERSIONED_FILES := $(wildcard $(CUDA_SRC_DIR)/*/*.cu)
VERSIONED_FILES := $(filter-out $(CUDA_SRC_DIR)/common/%,$(VERSIONED_FILES))
VERSIONED_FILES := $(filter-out $(CUDA_PRACTICE_DIR)/%,$(VERSIONED_FILES))
VERSIONED_TARGETS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(VERSIONED_FILES))

# ---- practice (flat + versioned) ----
PRACTICE_FILES := $(wildcard $(CUDA_PRACTICE_DIR)/*.cu)
PRACTICE_TARGETS := $(patsubst $(CUDA_PRACTICE_DIR)/%.cu,$(BUILD_DIR)/practice/%,$(PRACTICE_FILES))
PRACTICE_VERSIONED_FILES := $(wildcard $(CUDA_PRACTICE_DIR)/*/*.cu)
PRACTICE_VERSIONED_TARGETS := $(patsubst $(CUDA_PRACTICE_DIR)/%.cu,$(BUILD_DIR)/practice/%,$(PRACTICE_VERSIONED_FILES))

.PHONY: all clean run practice

all: $(TARGETS) $(VERSIONED_TARGETS)

practice: $(PRACTICE_TARGETS) $(PRACTICE_VERSIONED_TARGETS)

# Per-file link flags. Sources that need cuBLAS either include
# `cublas` in their filename, or live under 04_matmul/ (whose bench.cu
# calls cuBLAS as the K0 baseline). We also bake in an rpath so the
# binary finds libcublas.so at runtime even when launched via sudo/ncu
# (which strips LD_LIBRARY_PATH).
CUBLAS_LDFLAGS = -lcublas -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
EXTRA_LDFLAGS = $(if $(or $(findstring cublas,$@),$(findstring 04_matmul,$@)),$(CUBLAS_LDFLAGS),)

# One rule for both flat and versioned targets under build/, since the
# `%` pattern happily matches a nested path (e.g. 01_vector_add/01_naive).
# mkdir -p ensures the nested output directory exists.
$(BUILD_DIR)/%: $(CUDA_SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(EXTRA_LDFLAGS)

# Same idea for practice/, but note the source path prefix differs.
$(BUILD_DIR)/practice/%: $(CUDA_PRACTICE_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(EXTRA_LDFLAGS)

run: all
	@if [ -z "$(APP)" ]; then echo "Usage: make run APP=<name>  (e.g. APP=01_vector_add/01_naive)"; exit 1; fi
	./$(BUILD_DIR)/$(APP)

clean:
	rm -rf $(BUILD_DIR)
