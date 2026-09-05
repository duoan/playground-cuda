"""Chapter P2: AdamW / Lion / LAMB / Muon from scratch.

Run:
    python3 src/distributed_training/12_optimizers.py

What this demo shows:
    - Minimal reference implementation of 4 optimizers relevant to modern LLMs.
    - Numerical parity with `torch.optim.AdamW` where a canonical exists.
    - Prints per-optimizer footprint (state bytes / param).

No distributed setup — this file is single-process, single-GPU (or CPU).
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch


# ---------- AdamW -----------------------------------------------------------


class AdamW:
    """Reference AdamW that matches torch.optim.AdamW.

    Update rule:
        m = b1 m + (1-b1) g
        v = b2 v + (1-b2) g^2
        m_hat = m / (1 - b1^t)
        v_hat = v / (1 - b2^t)
        theta -= lr * m_hat / (sqrt(v_hat) + eps)
        theta -= lr * wd * theta      # decoupled
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        self.params: List[torch.Tensor] = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.step_num = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        self.step_num += 1
        b1c = 1 - self.b1 ** self.step_num
        b2c = 1 - self.b2 ** self.step_num
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            update = (m / b1c) / ((v / b2c).sqrt() + self.eps)
            # decoupled weight decay: applied to p directly, then Adam step
            p.mul_(1 - self.lr * self.wd)
            p.add_(update, alpha=-self.lr)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------- Lion (Chen 2023) -----------------------------------------------


class Lion:
    """Lion optimizer. Signed momentum, no second moment.

    update = sign(b1 * m_{t-1} + (1-b1) * g)
    theta -= lr * (update + wd * theta)
    m_t = b2 * m_{t-1} + (1-b2) * g          # separate slower momentum

    Typical lr = AdamW lr / 3 to 10 (because sign is unit-norm).
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.1):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, m in zip(self.params, self.m):
            if p.grad is None:
                continue
            g = p.grad
            update = (self.b1 * m + (1 - self.b1) * g).sign_()
            p.mul_(1 - self.lr * self.wd)
            p.add_(update, alpha=-self.lr)
            # update momentum for next step (with slower beta)
            m.mul_(self.b2).add_(g, alpha=1 - self.b2)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------- LAMB (You 2020) ------------------------------------------------


class LAMB:
    """Layer-wise adaptive AdamW. Trust ratio scales the update to weight norm.

    m, v = AdamW's
    r = m_hat / (sqrt(v_hat) + eps) + wd * theta
    phi = ||theta|| / ||r||       # trust ratio, per-param-tensor
    theta -= lr * phi * r
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.95),
                 eps: float = 1e-6, weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.step_num = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        self.step_num += 1
        b1c = 1 - self.b1 ** self.step_num
        b2c = 1 - self.b2 ** self.step_num
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            r = (m / b1c) / ((v / b2c).sqrt() + self.eps) + self.wd * p
            w_norm = p.norm()
            r_norm = r.norm()
            phi = (w_norm / r_norm) if (w_norm > 0 and r_norm > 0) else 1.0
            p.add_(r, alpha=-self.lr * phi)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------- Muon (Jordan 2024) ---------------------------------------------


def newton_schulz_orthogonalize(G: torch.Tensor, iters: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Iteratively map matrix G to nearest orthogonal matrix (SVD singular
    values -> 1). Used by Muon on 2D+ weights.

    Standard NS5 coefficients from Jordan et al. 2024.
    G must be at least 2D. For higher-D (conv weight etc), flatten trailing dims.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    if G.ndim < 2:
        raise ValueError("Muon Newton-Schulz needs 2D+ tensor")
    orig_shape = G.shape
    G2 = G.reshape(G.shape[0], -1)  # (rows, cols)
    # spectral-normalize so the iteration converges from within radius of stability
    X = G2 / (G2.norm() + eps)
    # If rows > cols, iterate on X^T (cheaper); flip back at end.
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(iters):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.reshape(orig_shape)


class Muon:
    """Muon: Momentum + Newton-Schulz orthogonalization for 2D+ weights.

    For 1D params (LN gain/bias), falls back to plain AdamW-like step
    (kept minimal here). Kimi K2 wraps two separate optimizers instead.

    Update:
        m = beta * m + g
        U = newton_schulz(m)
        theta -= lr * U
        theta -= lr * wd * theta        # decoupled
    """

    def __init__(self, params, lr: float = 2e-3, beta: float = 0.95,
                 weight_decay: float = 0.1, ns_iters: int = 5):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.wd = weight_decay
        self.ns_iters = ns_iters
        self.m = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, m in zip(self.params, self.m):
            if p.grad is None:
                continue
            m.mul_(self.beta).add_(p.grad)
            if p.ndim >= 2:
                U = newton_schulz_orthogonalize(m, iters=self.ns_iters)
            else:
                # 1D: no orthogonalization possible, use sign momentum
                U = m.sign()
            p.mul_(1 - self.lr * self.wd)
            p.add_(U, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------- Verification ---------------------------------------------------


def _mk_toy(seed: int = 0, dim: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple regression: theta * x = y."""
    torch.manual_seed(seed)
    theta_star = torch.randn(dim)
    X = torch.randn(64, dim)
    y = X @ theta_star + 0.01 * torch.randn(64)
    return X, y


def _run(optimizer_ctor, params, X, y, steps: int = 50):
    opt = optimizer_ctor(params)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        pred = X @ params[0]
        loss = (pred - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def verify_adamw_matches_torch():
    """Our AdamW should match torch.optim.AdamW numerically."""
    X, y = _mk_toy()
    dim = X.shape[1]
    torch.manual_seed(1)
    p_ours = torch.zeros(dim, requires_grad=True)
    torch.manual_seed(1)
    p_ref = torch.zeros(dim, requires_grad=True)

    ours = AdamW([p_ours], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW([p_ref], lr=1e-2, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    for _ in range(30):
        ours.zero_grad()
        ref.zero_grad()
        (X @ p_ours - y).pow(2).mean().backward()
        (X @ p_ref - y).pow(2).mean().backward()
        ours.step()
        ref.step()
    max_diff = (p_ours - p_ref).abs().max().item()
    assert max_diff < 1e-5, f"AdamW mismatch max_diff={max_diff}"
    print(f"AdamW vs torch.optim.AdamW: max param diff = {max_diff:.2e} ✓")


def demo_all():
    """Fit toy regression with 4 optimizers, print final loss + state size."""
    X, y = _mk_toy()
    dim = X.shape[1]

    def theta0():
        torch.manual_seed(2)
        return [torch.zeros(dim, requires_grad=True)]

    def m_theta0():
        torch.manual_seed(3)
        return [torch.randn(16, dim, requires_grad=True)]  # 2D for Muon

    configs = [
        ("SGD",   lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9), theta0),
        ("AdamW", lambda ps: AdamW(ps, lr=1e-2), theta0),
        ("Lion",  lambda ps: Lion(ps, lr=1e-3, weight_decay=0.1), theta0),
        ("LAMB",  lambda ps: LAMB(ps, lr=5e-3), theta0),
    ]
    print()
    print(f"{'Optim':<8}  {'final_loss':<12}  {'state/param':<14}")
    print("-" * 40)
    for name, ctor, init in configs:
        ps = init()
        losses = _run(ctor, ps, X, y, steps=60)
        # state footprint
        opt = ctor(init())
        n_bufs = 0
        if isinstance(opt, (AdamW, LAMB)):
            n_bufs = 2  # m, v
        elif isinstance(opt, Lion):
            n_bufs = 1  # m only
        elif isinstance(opt, torch.optim.SGD):
            n_bufs = 1  # momentum
        print(f"{name:<8}  {losses[-1]:<12.6f}  {n_bufs} buffer(s)")

    # Muon on 2D param (Muon requires ndim >= 2)
    torch.manual_seed(3)
    W_true = torch.randn(16, dim)
    Y_mat = X @ W_true.T + 0.01 * torch.randn(64, 16)

    torch.manual_seed(4)
    W = torch.randn(16, dim, requires_grad=True)
    opt = Muon([W], lr=1e-2, beta=0.95, weight_decay=0.0)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (X @ W.T - Y_mat).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"{'Muon':<8}  {losses[-1]:<12.6f}  1 buffer(s) (2D-only)")


def demo_newton_schulz():
    """Show NS5 makes a matrix near-orthogonal (SVDs → 1)."""
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    print("\nNewton-Schulz orthogonalization:")
    print(f"  before: singular values range = "
          f"[{torch.linalg.svdvals(G).min():.3f}, {torch.linalg.svdvals(G).max():.3f}]")
    U = newton_schulz_orthogonalize(G, iters=5)
    svd = torch.linalg.svdvals(U)
    print(f"  after 5 iters: [{svd.min():.3f}, {svd.max():.3f}] (should be ≈ 1)")


def main():
    verify_adamw_matches_torch()
    demo_newton_schulz()
    demo_all()


if __name__ == "__main__":
    main()
