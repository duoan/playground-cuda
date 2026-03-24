# CUDA Practice Track

This directory is for hands-on practice after reading the teaching kernels in `src/cuda/`.

Recommended order:

1. `01_vector_add_practice.cu`
2. `02_reduce_sum_practice.cu`
3. `03_softmax_practice.cu`
4. `04_matmul_practice.cu`
5. `05_layernorm_practice.cu`
6. `06_mlp_practice.cu`
7. `07_attention_practice.cu`
8. `08_flash_attention_v1_practice.cu`
9. `09_flash_attention_v2_practice.cu`
10. `10_flash_attention_v3_practice.cu`

How to use these files:

- First read the matching teaching file in `src/cuda/`.
- Then open the matching practice file here.
- Fill in the `TODO` kernels and helper functions yourself.
- Keep the host-side checks so you can verify correctness after each step.

Suggested workflow:

```bash
make build/01_vector_add
make build/02_reduce_sum
make build/03_softmax
make build/04_matmul
```
