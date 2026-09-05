"""Chapter P4: training monitor — grad_norm, activation stats, weight tracker.

Run:
    python3 src/distributed_training/15_training_monitor.py

Drop-in monitor wrapper that:
  - hooks every Module to record activation min/max/nan
  - records grad_norm (global L2) each step
  - records per-parameter weight norm growth
  - prints a short "training health report"

Usage in your own training loop:

    monitor = TrainingMonitor(model)
    for step in ...:
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        monitor.record_step(step, loss)
        # inspect anomalies:
        if monitor.grad_norm[-1] > 100:
            monitor.print_snapshot()
        optim.step(); optim.zero_grad()
    monitor.print_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class TrainingMonitor:
    model: nn.Module
    max_activation_records: int = 4    # how many recent forward passes to keep per layer
    _act: Dict[str, List[dict]] = field(default_factory=dict)
    grad_norm: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    weight_norm: Dict[str, List[float]] = field(default_factory=dict)
    _hooks: list = field(default_factory=list)

    def __post_init__(self):
        for name, mod in self.model.named_modules():
            # skip container-only modules (they don't do actual compute)
            if len(list(mod.children())) > 0:
                continue
            def hook(m, inp, out, nm=name):
                self._act.setdefault(nm, [])
                if isinstance(out, torch.Tensor):
                    stat = {
                        "min": out.min().item(),
                        "max": out.max().item(),
                        "mean": out.mean().item(),
                        "std": out.std().item() if out.numel() > 1 else 0.0,
                        "has_nan": torch.isnan(out).any().item(),
                        "has_inf": torch.isinf(out).any().item(),
                    }
                    self._act[nm].append(stat)
                    if len(self._act[nm]) > self.max_activation_records:
                        self._act[nm].pop(0)
            self._hooks.append(mod.register_forward_hook(hook))

    def _global_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return math.sqrt(total)

    def _snapshot_weight_norms(self):
        for name, p in self.model.named_parameters():
            self.weight_norm.setdefault(name, [])
            self.weight_norm[name].append(p.detach().norm().item())

    def record_step(self, step: int, loss: torch.Tensor):
        self.loss.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))
        self.grad_norm.append(self._global_grad_norm())
        self._snapshot_weight_norms()

    def alert_this_step(self) -> List[str]:
        """Return list of alerts for the most recent step."""
        alerts = []
        if not self.grad_norm:
            return alerts
        gn = self.grad_norm[-1]
        if math.isnan(gn) or math.isinf(gn):
            alerts.append(f"grad_norm={gn} (NaN/Inf!)")
        elif len(self.grad_norm) >= 20:
            recent = sorted(self.grad_norm[-20:-1])
            median = recent[len(recent) // 2]
            if gn > 10 * median:
                alerts.append(f"grad_norm={gn:.2e} vs 20-step median {median:.2e} "
                              f"({gn / median:.1f}× jump)")
        for nm, records in self._act.items():
            if records and (records[-1]["has_nan"] or records[-1]["has_inf"]):
                alerts.append(f"activation NaN/Inf at {nm}")
        return alerts

    def print_snapshot(self):
        step = len(self.loss) - 1
        print(f"\n--- snapshot @ step {step} ---")
        print(f"  loss = {self.loss[-1]:.4e}")
        print(f"  grad_norm = {self.grad_norm[-1]:.4e}")
        # top 3 layers by activation |max|
        rank = sorted(
            [(nm, recs[-1]["max"], recs[-1]["min"])
             for nm, recs in self._act.items() if recs],
            key=lambda x: -max(abs(x[1]), abs(x[2])))
        print("  top-3 activation-magnitude layers:")
        for nm, mx, mn in rank[:3]:
            print(f"    {nm:>30s}  min={mn:.2e}  max={mx:.2e}")

    def print_report(self):
        n = len(self.loss)
        if n == 0:
            print("no steps recorded")
            return
        print("\n" + "=" * 60)
        print(f"  Training report ({n} steps)")
        print("=" * 60)
        loss_start, loss_end = self.loss[0], self.loss[-1]
        gn_med = sorted(self.grad_norm)[n // 2]
        gn_max = max(self.grad_norm)
        gn_spikes = sum(1 for g in self.grad_norm if g > 10 * gn_med)
        print(f"  loss:      {loss_start:.4e} -> {loss_end:.4e}"
              f"  ({(loss_start - loss_end):+.4e})")
        print(f"  grad_norm: median={gn_med:.2e}  max={gn_max:.2e}"
              f"  spikes(>10×med)={gn_spikes}")
        # weight norm growth per param (top 3 growth rates)
        growth = []
        for nm, series in self.weight_norm.items():
            if len(series) < 2 or series[0] == 0:
                continue
            growth.append((nm, series[-1] / series[0]))
        growth.sort(key=lambda x: -x[1])
        print("  top 3 weight-norm growth (final / initial):")
        for nm, g in growth[:3]:
            print(f"    {nm:>40s}  ×{g:.3f}")

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ------------- Demo -------------------------------------------------------


class ToyMLP(nn.Module):
    def __init__(self, dim=64, layers=6):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.fcs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for norm, fc in zip(self.norms, self.fcs):
            x = x + fc(torch.relu(norm(x)))
        return self.head(x)


def demo():
    torch.manual_seed(0)
    model = ToyMLP(dim=64, layers=6)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monitor = TrainingMonitor(model)

    steps = 60
    for step in range(steps):
        x = torch.randn(32, 64)
        y = torch.randint(0, 10, (32,))
        # inject a poisoned batch at step 30 → creates a grad_norm spike
        if step == 30:
            x = x * 50
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optim.zero_grad()
        loss.backward()
        monitor.record_step(step, loss)
        alerts = monitor.alert_this_step()
        if alerts:
            print(f"\n[step {step}] ALERT:")
            for a in alerts:
                print(f"  - {a}")
            monitor.print_snapshot()
        optim.step()

    monitor.print_report()
    monitor.close()


if __name__ == "__main__":
    demo()
