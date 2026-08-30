# CUDA Vector Addition Examples - Makefile
# Chapter 02: GPU Memory Management and Kernel Launch

# Compiler
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Compiler flags
NVCCFLAGS ?= -std=c++17 -O2

# Targets
TARGETS = vec_add tensor_add

# Default target
all: $(TARGETS)

# Basic vector addition example
vec_add: vec_add.cu
	$(NVCC) $(NVCC_FLAGS) vec_add.cu -o vec_add

# 3D tensor addition example
tensor_add: tensor_add.cu
	$(NVCC) $(NVCC_FLAGS) tensor_add.cu -o tensor_add

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Run all examples
run: all
	@echo "=== Running Basic Vector Addition ==="
	./vecadd
	@echo ""
	@echo "=== Running Scalable Vector Addition ==="
	./vecadd_scalable
	@echo ""
	@echo "=== Running 3D Tensor Addition ==="
	./tensor_add_3d

# Show help
help:
	@echo "CUDA Vector Addition Examples - Chapter 02"
	@echo ""
	@echo "Available targets:"
	@echo "  all             - Build all examples"
	@echo "  vec_add         - Vector addition example"
	@echo "  tensor_add      - Tensor addition example"
	@echo "  clean           - Remove all compiled executables"
	@echo "  run             - Build and run all examples"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Usage: make [target]"
