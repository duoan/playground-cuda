"""Chapter P3: LR schedules — warmup+cosine, WSD, WSD-rewarm.

Run:
    python3 src/distributed_training/13_lr_schedules.py

Prints ASCII plot of each schedule; also prints (step, lr) sequences that
can be pasted into Typst `line-plot(series: (...))` for the book.

No torch dependency (pure math). Optionally uses torch.optim.lr_scheduler if
available to sanity-check `warmup + cosine`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, List


# ---------- Individual schedules -------------------------------------------


def linear_warmup(step: int, warmup: int, peak_lr: float) -> float:
    if warmup <= 0:
        return peak_lr
    return peak_lr * min(1.0, step / warmup)


def warmup_cosine(step: int, warmup: int, total: int,
                  peak_lr: float, min_lr: float = 0.0) -> float:
    """Standard warmup + cosine annealing (GPT-3, Llama)."""
    if step < warmup:
        return linear_warmup(step, warmup, peak_lr)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def wsd(step: int, warmup: int, total: int, peak_lr: float,
        decay_frac: float = 0.2, min_lr: float = 0.0,
        decay_shape: str = "linear") -> float:
    """Warmup-Stable-Decay (MiniCPM 2024, DeepSeek-V3, Kimi K2).

    decay_frac: fraction of total that is decay (e.g., 0.2 = last 20%).
    decay_shape: "linear" or "sqrt".
    """
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return linear_warmup(step, warmup, peak_lr)
    if step < decay_start:
        return peak_lr
    if step >= total:
        return min_lr
    x = (step - decay_start) / max(1, total - decay_start)   # 0..1 in decay
    if decay_shape == "linear":
        return peak_lr - (peak_lr - min_lr) * x
    elif decay_shape == "sqrt":
        return peak_lr - (peak_lr - min_lr) * math.sqrt(x)
    else:
        raise ValueError(decay_shape)


def wsd_rewarm(step: int, warmup: int, total: int, peak_lr: float,
               rewarm_step: int, rewarm_warmup: int, rewarm_peak_ratio: float = 0.5,
               decay_frac: float = 0.15, min_lr: float = 0.0) -> float:
    """WSD with a mini-rewarm at `rewarm_step` (Kimi K2 style, 60% of training).

    After rewarm_step, LR drops to some fraction, then re-warms up to
    rewarm_peak = peak_lr * rewarm_peak_ratio, stays, and decays.
    """
    if step < rewarm_step:
        return wsd(step, warmup, rewarm_step, peak_lr,
                   decay_frac=0.0, min_lr=peak_lr)  # no decay in first phase
    # Phase 2: rewarm + stable + decay
    step2 = step - rewarm_step
    total2 = total - rewarm_step
    peak2 = peak_lr * rewarm_peak_ratio
    return wsd(step2, rewarm_warmup, total2, peak2, decay_frac, min_lr)


# ---------- Utility --------------------------------------------------------


@dataclass
class ScheduleSpec:
    name: str
    fn: Callable[[int], float]


def ascii_plot(specs: List[ScheduleSpec], total: int,
               width: int = 60, height: int = 14) -> None:
    """Multi-series ASCII plot. Overlays with different chars."""
    xs = [int(i * total / (width - 1)) for i in range(width)]
    charset = "*#o+@x"
    grid = [[" " for _ in range(width)] for _ in range(height)]
    all_ys = []
    for spec in specs:
        ys = [spec.fn(x) for x in xs]
        all_ys.append(ys)
    lo = min(min(y for y in ys) for ys in all_ys)
    hi = max(max(y for y in ys) for ys in all_ys)
    span = max(hi - lo, 1e-12)
    for i, (spec, ys) in enumerate(zip(specs, all_ys)):
        ch = charset[i % len(charset)]
        for x, y in enumerate(ys):
            row = height - 1 - int((y - lo) / span * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][x] = ch
    print()
    # y-axis labels
    for r, row in enumerate(grid):
        y = hi - (hi - lo) * r / (height - 1)
        print(f"  {y:8.2e} | {''.join(row)}")
    print("           " + "-" * width)
    print(f"           0{'step'.rjust(width // 2 - 2)}{str(total).rjust(width // 2)}")
    print()
    for i, spec in enumerate(specs):
        print(f"    [{charset[i % len(charset)]}] {spec.name}")


def to_typst_series(fn: Callable[[int], float], total: int,
                    n_points: int = 40, peak_lr: float = 1.0) -> str:
    """Emit a `(x, y/peak_lr)` sequence suitable to paste into template
    `line-plot(series: ((label, ...), ...))`."""
    xs = [int(i * total / (n_points - 1)) for i in range(n_points)]
    pts = ", ".join(f"({x}, {fn(x) / peak_lr:.3f})" for x in xs)
    return f"({pts})"


# ---------- Main -----------------------------------------------------------


def main():
    peak_lr = 3e-4
    total = 10000
    warmup = 1000

    schedules = [
        ScheduleSpec("warmup+cosine",
                     lambda s: warmup_cosine(s, warmup, total, peak_lr,
                                             min_lr=0.1 * peak_lr)),
        ScheduleSpec("WSD (linear decay 20%)",
                     lambda s: wsd(s, warmup, total, peak_lr,
                                    decay_frac=0.2, min_lr=0.0,
                                    decay_shape="linear")),
        ScheduleSpec("WSD (sqrt decay 15%)",
                     lambda s: wsd(s, warmup, total, peak_lr,
                                    decay_frac=0.15, min_lr=0.0,
                                    decay_shape="sqrt")),
        ScheduleSpec("WSD + rewarm at 60%",
                     lambda s: wsd_rewarm(s, warmup, total, peak_lr,
                                          rewarm_step=6000, rewarm_warmup=500,
                                          rewarm_peak_ratio=0.5,
                                          decay_frac=0.15)),
    ]

    print(f"peak_lr={peak_lr}, total_steps={total}, warmup={warmup}")
    ascii_plot(schedules, total, width=70, height=16)

    print("\n===== Typst series (paste into template line-plot) =====")
    for spec in schedules:
        print(f'\n  ("{spec.name}", {to_typst_series(spec.fn, total, peak_lr=peak_lr)}),')

    # Sanity: torch.optim.lr_scheduler.CosineAnnealingLR should match ours (approx)
    try:
        import torch
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=peak_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total - warmup, eta_min=0.1 * peak_lr)
        print("\n===== Cross-check with torch CosineAnnealingLR (post-warmup) =====")
        for s in [0, 100, 1000, 5000, 9999]:
            ours = warmup_cosine(s, warmup, total, peak_lr, min_lr=0.1 * peak_lr)
            # Advance torch scheduler to same step (post-warmup)
            if s >= warmup:
                # rebuild for each query — cheap
                opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)],
                                        lr=peak_lr)
                sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt2, T_max=total - warmup, eta_min=0.1 * peak_lr)
                for _ in range(s - warmup):
                    sched2.step()
                theirs = opt2.param_groups[0]["lr"]
                diff = abs(ours - theirs)
                print(f"  step={s:5d}  ours={ours:.6e}  torch={theirs:.6e}  diff={diff:.1e}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
