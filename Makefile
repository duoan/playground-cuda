CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
BUILD_DIR ?= build
NVCCFLAGS ?= -std=c++17 -O2
CUDA_SRC_DIR ?= src/cuda
CUDA_PRACTICE_DIR ?= src/cuda/practice

SRC_FILES := $(wildcard $(CUDA_SRC_DIR)/*.cu)
TARGETS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SRC_FILES))
PRACTICE_FILES := $(wildcard $(CUDA_PRACTICE_DIR)/*.cu)
PRACTICE_TARGETS := $(patsubst $(CUDA_PRACTICE_DIR)/%.cu,$(BUILD_DIR)/practice/%,$(PRACTICE_FILES))

.PHONY: all clean run practice

all: $(TARGETS)

practice: $(PRACTICE_TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(CUDA_SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/practice:
	mkdir -p $(BUILD_DIR)/practice

$(BUILD_DIR)/practice/%: $(CUDA_PRACTICE_DIR)/%.cu | $(BUILD_DIR) $(BUILD_DIR)/practice
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@if [ -z "$(APP)" ]; then echo "Usage: make run APP=<name>"; exit 1; fi
	./$(BUILD_DIR)/$(APP)

clean:
	rm -rf $(BUILD_DIR)
