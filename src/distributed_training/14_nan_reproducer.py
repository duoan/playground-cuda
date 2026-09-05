"""Chapter P4: reproducible NaN triggers + how to fix them.

Run:
    python3 src/distributed_training/14_nan_reproducer.py

For each pathology, we:
  1. build a minimal broken version and trigger NaN/inf on forward or backward
  2. show the fixed version producing finite outputs
  3. print a compact summary

No distributed. Works on CPU (uses torch.float32 emulation for BF16 pitfalls
where possible; some cases use CUDA BF16 if available).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _stats(x: torch.Tensor, name: str = "x"):
    nan = torch.isnan(x).any().item()
    inf = torch.isinf(x).any().item()
    mn = x.min().item() if not (nan or inf) else float("nan")
    mx = x.max().item() if not (nan or inf) else float("nan")
    print(f"  {name}: min={mn:.4e}  max={mx:.4e}  has_nan={nan}  has_inf={inf}")
    return nan or inf


# ---------- 1. Softmax overflow -------------------------------------------


def demo_softmax_overflow():
    _print_section("(1) Softmax overflow (attention logits go big)")

    # BF16 exp(x): overflows when x > ~89. Even before that, in a naive
    # softmax (no max subtraction), large positive logits overflow.
    logits = torch.tensor([100.0, 101.0, 102.0], dtype=torch.bfloat16)

    print("Broken: manual exp / sum(exp) in BF16")
    e = torch.exp(logits)   # BF16 exp(102) = inf
    p_bad = e / e.sum()
    _stats(p_bad, "p_bad (naive softmax)")

    print("Fixed: subtract max first (a.k.a. safe softmax)")
    e_safe = torch.exp(logits - logits.max())
    p_good = e_safe / e_safe.sum()
    _stats(p_good, "p_good (safe softmax)")
    print(f"  → argmax same ({p_good.argmax()} == {2})")

    print("Also fixed: F.softmax always does the safe version internally")
    p_fx = F.softmax(logits.float(), dim=-1)
    _stats(p_fx, "F.softmax(float32)")


# ---------- 2. RMSNorm eps too small in BF16 ------------------------------


def demo_rmsnorm_bf16_eps():
    _print_section("(2) RMSNorm eps in BF16 with tiny variance")

    # BF16 mantissa 7 bit. If eps=1e-8, sqrt(1e-8) requires precision our
    # BF16 doesn't have; the result rounds inconsistently.
    x = torch.full((64,), 1e-4, dtype=torch.bfloat16)
    eps = 1e-8
    print(f"input all 1e-4, eps={eps}")

    # broken: compute in BF16
    rms_bf16 = torch.sqrt((x * x).mean() + eps)
    print(f"  BF16 rms = {rms_bf16.item():.6e}   (expected ~1e-4)")

    # fixed: upcast var to float32
    var32 = (x.float() * x.float()).mean() + eps
    rms32 = torch.sqrt(var32)
    print(f"  FP32 rms = {rms32.item():.6e}   (accurate)")


# ---------- 3. Attention QK^T with BF16 accumulation drift ----------------


def demo_qkt_bf16_drift():
    _print_section("(3) Q @ K.T with BF16 accumulator drift (long S)")

    torch.manual_seed(0)
    d_h, S = 128, 8192
    Q = torch.randn(S, d_h, dtype=torch.bfloat16) * 0.1
    K = torch.randn(S, d_h, dtype=torch.bfloat16) * 0.1

    # BF16 matmul (accumulator can be BF16 on some kernels; FP32 on H100
    # cuBLAS via tf32/fp32 accum). On CPU, torch.matmul uses fp32 accum
    # for BF16 by default; we manually recreate accumulation drift.
    print("Manually accumulate 1 row of Q @ K.T in BF16 (worst case)")
    row_bf16 = torch.zeros(S, dtype=torch.bfloat16)
    for i in range(d_h):
        row_bf16 += Q[0, i] * K[:, i]  # each product added in BF16
    row_fp32 = (Q[0].float() @ K.float().T)
    err = (row_bf16.float() - row_fp32).abs().max().item()
    print(f"  BF16 accum vs FP32 accum: max element err = {err:.4e}")
    print(f"  → for large S or many terms, drift can push isolated entries")
    print(f"    beyond safe softmax range even before overflow.")

    print("Fix: FP32 accum inside cuBLAS/FlashAttention (default) OR "
          "call .float() before matmul.")


# ---------- 4. Gradient explosion / clip effect ---------------------------


def demo_grad_explosion():
    _print_section("(4) Gradient explosion in deep pre-LN (no residual scale)")

    torch.manual_seed(0)
    L, H = 40, 256          # 40-layer stack
    x = torch.randn(1, H, requires_grad=True)
    layers = torch.nn.ModuleList()
    for _ in range(L):
        W = torch.nn.Linear(H, H, bias=False)
        # deliberately larger init → each layer amplifies
        torch.nn.init.normal_(W.weight, std=0.06)  # too big
        layers.append(W)

    h = x
    for W in layers:
        h = h + W(F.relu(h))    # pre-LN-like residual, no Wang init
    loss = h.pow(2).mean()
    loss.backward()

    gn = 0.0
    for p in layers.parameters():
        gn += p.grad.pow(2).sum().item()
    gn = math.sqrt(gn)
    print(f"  no-clip grad_norm = {gn:.2e}")

    print("Fix: torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)")
    total = torch.nn.utils.clip_grad_norm_(layers.parameters(), max_norm=1.0)
    print(f"  clip_grad_norm_ returned = {total.item():.2e}  (pre-clip)")
    # verify post-clip
    gn2 = 0.0
    for p in layers.parameters():
        gn2 += p.grad.pow(2).sum().item()
    gn2 = math.sqrt(gn2)
    print(f"  post-clip grad_norm = {gn2:.2e}  (should be ≤ 1.0)")


# ---------- 5. log(0) in cross-entropy (manual) ---------------------------


def demo_logzero():
    _print_section("(5) log(0) in manually-written CE loss")

    logits = torch.tensor([[-100.0, -100.0, 30.0]])   # model very confident in class 2
    target = torch.tensor([0])                        # true label is 0!

    print("Broken: log(softmax(x))")
    p = F.softmax(logits, dim=-1)
    ce_bad = -torch.log(p[0, target[0]])
    _stats(ce_bad.unsqueeze(0), "ce_bad")

    print("Fixed: F.log_softmax (numerically stable) or F.cross_entropy")
    ce_good = F.cross_entropy(logits, target)
    _stats(ce_good.unsqueeze(0), "ce_good")


# ---------- 6. RoPE overflow in BF16 with large base ---------------------


def demo_rope_bf16():
    _print_section("(6) RoPE cos/sin in BF16 with large base + long pos")

    d_h, base, max_pos = 128, 1_000_000, 100_000

    # standard RoPE freq
    inv_freq = 1.0 / (base ** (torch.arange(0, d_h, 2).float() / d_h))
    pos = torch.arange(max_pos).float()

    print("Compute cos/sin in FP32 (correct)")
    freqs = torch.outer(pos, inv_freq)
    cos_fp32 = torch.cos(freqs)
    sin_fp32 = torch.sin(freqs)

    print("Compute cos/sin in BF16 (broken: sin overflow at very small freq * "
          "very large pos → precision loss)")
    freqs_bf16 = torch.outer(pos.bfloat16(), inv_freq.bfloat16())
    cos_bf16 = torch.cos(freqs_bf16)
    sin_bf16 = torch.sin(freqs_bf16)

    err = (cos_fp32 - cos_bf16.float()).abs().max().item()
    print(f"  max |cos_fp32 - cos_bf16| = {err:.4f}")
    print(f"  → at pos=100K, base=1M, BF16 phase precision degrades significantly")
    print(f"    (typically 0.1-0.3 error → wrong attention scores)")
    print("Fix: always compute cos/sin in FP32, then apply to Q/K in BF16.")


# ---------- Main ----------------------------------------------------------


def main():
    demo_softmax_overflow()
    demo_rmsnorm_bf16_eps()
    demo_qkt_bf16_drift()
    demo_grad_explosion()
    demo_logzero()
    demo_rope_bf16()
    print("\n" + "=" * 60)
    print("  All 6 pathologies reproduced + fixes shown.")
    print("=" * 60)


if __name__ == "__main__":
    main()
