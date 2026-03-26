import triton
import triton.language as tl
import torch

def _get_num_sms() -> int:
    """动态获取当前 GPU 的 SM 数（H100 = 132）。"""
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

# ═══════════════════════════════════════════════════════
#  Pass 1 — persistent kernel：每个 SM 一个 program，
#           grid-stride 跑完整个输入，产出 N_SMS 个 partial sum
# ═══════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        # BLOCK_SIZE × N_SMS = 每轮覆盖的元素数
        # 越大 → 循环次数越少，但寄存器压力越大；autotune 自动选最优
        triton.Config({"BLOCK_SIZE":  2048}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_SIZE":  4096}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_SIZE":  4096}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_SIZE":  8192}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_SIZE":  8192}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_SIZE":  8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def _sum_pass1(
    inp_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    N_SMS:      tl.constexpr,
):
    pid = tl.program_id(0)

    # ── fp32 accumulator（即使输入是 fp16/bf16 也不丢精度）──
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    stride = N_SMS * BLOCK_SIZE        # 编译期常量，乘法免费
    block_start = pid * BLOCK_SIZE

    # ── grid-stride loop ──
    # 132 个 program 交错读整块显存，天然 coalesced
    while block_start < n_elements:
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(
            inp_ptr + offs,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",   # 只读一次，不污染 L2
        )
        acc += x.to(tl.float32)
        block_start += stride

    # ── 块内归约 → 1 个标量 ──
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))

# ═══════════════════════════════════════════════════════
#  Pass 2 — 单 block 把 N_SMS 个 partial sum 汇总成 1 个标量
# ═══════════════════════════════════════════════════════

@triton.jit
def _sum_pass2(
    partial_ptr,
    out_ptr,
    N:          tl.constexpr,     # = N_SMS
    BLOCK_SIZE: tl.constexpr,     # 最小的 2^k >= N
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(x, axis=0))

# ═══════════════════════════════════════════════════════
#  Host API
# ═══════════════════════════════════════════════════════

def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Persistent two-pass sum reduction.
    • 任何大小的输入，永远只 launch 2 个 kernel。
    • 结果始终以 fp32 返回（数值稳定）。
    """
    assert x.is_cuda
    n = x.numel()
    num_sms = _get_num_sms()                            # H100 → 132

    # power-of-2 >= num_sms，用于 pass2 的 BLOCK_SIZE
    final_block = triton.next_power_of_2(num_sms)       # 132 → 256

    partials = torch.empty(num_sms, device=x.device, dtype=torch.float32)
    result   = torch.empty(1,       device=x.device, dtype=torch.float32)

    # Pass 1: 132 programs, grid-stride
    _sum_pass1[(num_sms,)](x, partials, n, N_SMS=num_sms)

    # Pass 2: 1 program, 132 → 1
    _sum_pass2[(1,)](partials, result, N=num_sms, BLOCK_SIZE=final_block)

    return result

# ═══════════════════════════════════════════════════════
#  验证 + 带宽基准测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SMs: {_get_num_sms()}\n")

    # ── correctness ──
    print("── Correctness ──")
    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        x   = torch.randn(n, device="cuda")
        ref = x.sum().float()
        out = reduce_sum(x).squeeze()
        ok  = torch.allclose(ref, out, rtol=1e-3, atol=1e-3)
        print(f"  n={n:>12,}  torch={ref.item():>12.4f}  "
              f"triton={out.item():>12.4f}  {'✓' if ok else '✗'}")

    # ── bandwidth ──
    print("\n── Bandwidth (fp32) ──")
    for n in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
        x = torch.randn(n, device="cuda")
        nbytes = x.numel() * x.element_size()

        ms_t = triton.testing.do_bench(lambda: x.sum())
        ms_r = triton.testing.do_bench(lambda: reduce_sum(x))

        bw_t = nbytes / (ms_t * 1e-3) / 1e12
        bw_r = nbytes / (ms_r * 1e-3) / 1e12

        tag = "faster ✓" if ms_r < ms_t else "slower"
        print(f"  n={n:>13,} │ Torch {ms_t:.3f} ms ({bw_t:.2f} TB/s) │ "
              f"Triton {ms_r:.3f} ms ({bw_r:.2f} TB/s) │ {tag}")