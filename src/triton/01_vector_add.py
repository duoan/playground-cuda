import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr, # output
    n_elements, # total size of the vector
    BLOCK_SIZE: tl.constexpr, # Number of elements each program should process
):
    # load the program data
    # 1. get the start index
    pid = tl.program_id(axis=0) # We use a 1D lanuch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    c = a + b # vectorized operation on a whole block wise
    # write the data back to DRAM
    tl.store(c_ptr + offsets, c, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # allocate memory for output c
    c = torch.empty_like(a, device=DEVICE)

    assert a.device == DEVICE and b.device == DEVICE and c.device == DEVICE
    n_elements = c.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)

    return c

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals =[2**i for i in range(12, 32, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green','-')],
        ylabel='GB/s',
        plot_name='01_vector_add_performance',
        args={},
    )
)
def benchmark(size, provider):
    a = torch.rand(size, device=DEVICE, dtype=torch.float32)
    b = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a + b, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(a, b), quantiles=quantiles)

    gbps = lambda ms: 3 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    a = torch.rand(size, device=DEVICE)
    b = torch.rand(size, device=DEVICE)
    expected = a + b
    actual = add(a, b)
    torch.testing.assert_close(actual, expected)

    benchmark.run(print_data=True, show_plots=False, save_path=".")
