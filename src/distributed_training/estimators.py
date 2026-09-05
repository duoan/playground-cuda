"""Analytical estimators for distributed training: memory, FLOPs, comm,
and end-to-end step time.

Companion to `books/distributed-training/chapters/M_interview_math.typ`.

Everything is a plain function of dataclasses so you can:
    python3 src/distributed_training/estimators.py --preset llama3-70b-tp4-dp2-h100x8

And get a step-time breakdown you can quote in interviews.

Design constraints (Karpathy §2 simplicity):
    - No external dependencies.
    - No frameworks. Every number derived from the formulas in Chapter M.
    - Every function under 30 lines.
    - Presets at bottom encode the 8 canonical scenarios from the book.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------- data classes ----------------------------------------------------


@dataclass
class ModelConfig:
    L: int              # layers
    H: int              # hidden
    I: int              # ffn inter (already effective for GLU)
    A: int              # heads
    V: int              # vocab
    S: int              # sequence length
    tokens_per_step: int  # global batch tokens
    glu: bool = True    # SwiGLU adds 1 extra proj → 3 * H * I instead of 2

    @property
    def d_h(self) -> int:
        return self.H // self.A

    @property
    def params(self) -> int:
        """Total parameters (embedding tied)."""
        p_attn = 4 * self.H * self.H
        p_ffn = (3 if self.glu else 2) * self.H * self.I
        p_block = p_attn + p_ffn
        p_emb = self.V * self.H  # tied
        return self.L * p_block + p_emb


@dataclass
class ParallelConfig:
    P: int = 1     # pipeline parallel
    T: int = 1     # tensor parallel
    D: int = 1     # data parallel
    C: int = 1     # context parallel
    E: int = 1     # expert parallel (for MoE)
    m: int = 1     # micro batches per PP step (also grad-accum steps if P=1)
    B_micro: int = 1  # sequences per micro batch per DP rank
    zero_stage: int = 0  # 0/1/2/3; applies within DP dim
    sp: bool = False     # sequence parallel with TP
    recompute: str = "selective"  # 'none' | 'selective' | 'full'
    tp_overlap: float = 0.0  # fraction of TP AR overlapped with compute
    dp_overlap: float = 1.0  # DP grad AR usually 100% overlapped in bwd

    @property
    def world(self) -> int:
        return self.P * self.T * self.D * self.C * max(1, self.E)


@dataclass
class HWConfig:
    name: str = "H100-SXM"
    peak_flops: float = 989e12       # BF16 dense TFLOPS
    hbm_bw: float = 3.35e12          # bytes/s
    nvlink_bw: float = 300e9         # bytes/s effective (peak 450)
    ib_bw: float = 50e9              # bytes/s per NIC (400G IB ≈ 50 GB/s)
    mfu: float = 0.45                # 0.4-0.55 typical
    bytes_per_elem: int = 2          # BF16
    alpha_us: float = 5.0            # NCCL launch overhead per call


# ---------- memory ---------------------------------------------------------


def mem_estimate(cfg: ModelConfig, para: ParallelConfig,
                 hw: HWConfig) -> Dict[str, float]:
    """Return GB per GPU, broken down by category.

    Correct for TP (params/opt sharded on T dim), ZeRO (further shard on D),
    and CP (activation only, along S dim). Selective vs full recompute
    controls the activation constant.
    """
    P = cfg.params
    b = hw.bytes_per_elem
    T, D, C, E = para.T, para.D, para.C, max(1, para.E)

    # EP shards the *expert* parameters across E; other params replicated.
    # Rough split: for MoE models the FFN (2H*I or 3H*I) is expert; attn+emb are not.
    if E > 1:
        expert_p = cfg.L * (3 if cfg.glu else 2) * cfg.H * cfg.I
        non_expert = P - expert_p
        p_effective = non_expert + expert_p / E
    else:
        p_effective = P

    # PP also shards params: each stage owns 1/P of the layers, so its
    # per-GPU param/grad/opt footprint is divided by P as well.
    P_ = para.P

    # params: BF16, sharded by TP/PP always; by DP if ZeRO-3
    p_denom = T * P_ * (D if para.zero_stage >= 3 else 1)
    params_gb = p_effective * b / p_denom / 1e9

    # grads: BF16, TP/PP-sharded; DP-sharded if ZeRO-2+
    g_denom = T * P_ * (D if para.zero_stage >= 2 else 1)
    grads_gb = p_effective * b / g_denom / 1e9

    # optimizer (Adam): master weight FP32 (4B) + m FP32 + v FP32 = 12B
    o_denom = T * P_ * (D if para.zero_stage >= 1 else 1)
    opt_gb = 12 * p_effective / o_denom / 1e9

    # activation: Megatron "34 / 12 / 2" per (B, S, H, L, b)
    # KEY: activation only needs to hold B_MICRO batches live at any time,
    # not the whole grad-accum'd global batch. Grad accum saves activation
    # linearly — this is a common interview trap.
    factor = {"none": 34, "selective": 12, "full": 2}[para.recompute]
    # SP reduces LN/dropout activation by T; CP reduces along S by C.
    act_denom = C * (T if para.sp else 1)
    # With PP 1F1B, ≤ P live micro-batches per stage.
    live_micros = para.P if para.P > 1 else 1
    B_live = para.B_micro * live_micros
    act_gb = factor * B_live * cfg.S * cfg.H * cfg.L * b / act_denom / 1e9

    return {
        "params": params_gb,
        "grads": grads_gb,
        "opt": opt_gb,
        "act": act_gb,
        "total": params_gb + grads_gb + opt_gb + act_gb,
    }


# ---------- FLOPs ---------------------------------------------------------


def flops_step(cfg: ModelConfig, para: ParallelConfig) -> float:
    """Total FLOPs for one training step (all GPUs summed).

    Base formula: 6 * P * N_tok (fwd+bwd = 3× fwd, fwd = 2× params).
    Recompute adds one extra forward → 8 * P * N_tok.
    Attention S² term added when S > 4k (not negligible for long ctx).
    """
    base = 6 * cfg.params * cfg.tokens_per_step
    if para.recompute == "full":
        base = base * 8 / 6

    # Attention quadratic term: 4 * A * S * d_h * S * L * B  (per fwd)
    # fwd only counts; bwd doubles.
    B = cfg.tokens_per_step / cfg.S
    attn_quad = 3 * 4 * cfg.A * cfg.S * cfg.d_h * cfg.S * cfg.L * B
    return base + attn_quad


# ---------- Communication -------------------------------------------------


@dataclass
class Collective:
    kind: str          # 'AR' 'AG' 'RS' 'A2A' 'P2P'
    vol_per_gpu: int   # bytes moved out of this GPU (single direction)
    count: int         # how many times per step
    scope: str         # 'intra' (NVLink) or 'inter' (IB)
    overlap: float     # 0..1 fraction hidden behind compute


def _factor(w: int) -> float:
    if w <= 1:
        return 0.0
    return 2 * (w - 1) / w   # ring AR


def comm_volumes(cfg: ModelConfig, para: ParallelConfig,
                 hw: HWConfig) -> List[Collective]:
    b = hw.bytes_per_elem
    T, D, P_, C, E, m = para.T, para.D, para.P, para.C, max(1, para.E), para.m
    L, H, S = cfg.L, cfg.H, cfg.S
    B_micro = para.B_micro
    # micro batches per DP rank per step (= grad accum × PP micros)
    micros_per_step = cfg.tokens_per_step // (para.B_micro * cfg.S * D)
    micros_per_step = max(1, micros_per_step)

    colls: List[Collective] = []

    # ---- TP: every block has 2 AR (fwd/bwd × 2 sites = 4 per block per micro)
    if T > 1:
        # per-micro AR of size (B_micro, S, H) BF16; ×4 sites/block/micro × L × micros
        act = B_micro * S * H * b
        colls.append(Collective(
            kind="AR" if not para.sp else "AG+RS",
            vol_per_gpu=int(_factor(T) * act),
            count=4 * L * micros_per_step,
            scope="intra",
            overlap=para.tp_overlap,
        ))

    # ---- DP: grad AR once per step (or per bucket), vol = 2 * (params/T/(D_shard if zero))
    if D > 1:
        param_shard = cfg.params / T
        if para.zero_stage >= 2:
            # ReduceScatter instead of AR (halves volume per direction)
            colls.append(Collective(
                kind="RS",
                vol_per_gpu=int((D - 1) / D * param_shard * b),
                count=1,
                scope="inter" if D > 8 else "intra",
                overlap=para.dp_overlap,
            ))
        else:
            colls.append(Collective(
                kind="AR",
                vol_per_gpu=int(_factor(D) * param_shard * b),
                count=1,
                scope="inter" if D > 8 else "intra",
                overlap=para.dp_overlap,
            ))
        if para.zero_stage >= 3:
            # AllGather params per block per fwd+bwd
            colls.append(Collective(
                kind="AG-param",
                vol_per_gpu=int((D - 1) / D * (param_shard / L) * b),
                count=2 * L,  # fwd + bwd needs re-gather
                scope="inter" if D > 8 else "intra",
                overlap=0.5,   # partial overlap
            ))

    # ---- PP: P2P per micro at each boundary
    if P_ > 1:
        vol = int(B_micro * S * H * b)
        colls.append(Collective(
            kind="P2P",
            vol_per_gpu=vol,
            count=2 * (P_ - 1) * micros_per_step,
            scope="intra" if P_ <= 8 else "inter",
            overlap=0.8,
        ))

    # ---- CP: ring-attention K/V rotate
    if C > 1:
        vol = int(2 * B_micro * S * H * b / C)  # K+V per step, per micro
        colls.append(Collective(
            kind="P2P-ring",
            vol_per_gpu=vol,
            count=(C - 1) * L * 2 * micros_per_step,
            scope="intra",
            overlap=0.7,
        ))

    # ---- EP: dispatch + combine a2a per MoE layer per micro
    if E > 1:
        vol = int(2 * B_micro * S * H * b * (E - 1) / E)
        colls.append(Collective(
            kind="A2A",
            vol_per_gpu=vol,
            count=L * micros_per_step,
            scope="inter" if E > 8 else "intra",
            overlap=0.3,
        ))
    return colls


def comm_time_seconds(coll: Collective, hw: HWConfig) -> float:
    bw = hw.nvlink_bw if coll.scope == "intra" else hw.ib_bw
    per_call = hw.alpha_us * 1e-6 + coll.vol_per_gpu / bw
    total = per_call * coll.count
    return total * (1 - coll.overlap)


# ---------- Roofline ------------------------------------------------------


def roofline(ai: float, hw: HWConfig) -> Tuple[str, float]:
    """Return ('compute-bound' or 'memory-bound', achievable FLOPS)."""
    ridge = hw.peak_flops / hw.hbm_bw
    if ai > ridge:
        return "compute-bound", hw.peak_flops
    return "memory-bound", ai * hw.hbm_bw


# ---------- step time -----------------------------------------------------


def step_time(cfg: ModelConfig, para: ParallelConfig,
              hw: HWConfig) -> Dict[str, float]:
    F = flops_step(cfg, para)
    # per-GPU compute: TP and CP shard tokens; PP shards layers.
    # DP shards batch. Total F / world already accounts for all four.
    t_compute = F / (para.world * hw.peak_flops * hw.mfu)

    colls = comm_volumes(cfg, para, hw)
    t_comm = sum(comm_time_seconds(c, hw) for c in colls)

    # PP bubble: (P-1)/(m+P-1) * ideal_iter
    if para.P > 1:
        ideal = (t_compute + t_comm) / para.P  # per-stage; but t_compute is already per-GPU
        t_bubble = (para.P - 1) / (para.m + para.P - 1) * (t_compute + t_comm)
    else:
        t_bubble = 0.0

    t_overhead = 0.02 * (t_compute + t_comm)  # ~2% dataloader/host

    return {
        "compute": t_compute,
        "comm": t_comm,
        "bubble": t_bubble,
        "overhead": t_overhead,
        "total": t_compute + t_comm + t_bubble + t_overhead,
    }


# ---------- Presets --------------------------------------------------------


def _llama_family(L, H, I, A, V=32000, S=8192, tokens=1_000_000):
    return ModelConfig(L=L, H=H, I=I, A=A, V=V, S=S,
                       tokens_per_step=tokens, glu=True)


PRESETS: Dict[str, Tuple[ModelConfig, ParallelConfig, HWConfig]] = {
    "llama-7b-dp8-h100x8": (
        _llama_family(32, 4096, 11008, 32),
        ParallelConfig(P=1, T=1, D=8, B_micro=1, zero_stage=2),
        HWConfig(),
    ),
    "llama-13b-tp2-dp4-h100x8": (
        _llama_family(40, 5120, 13824, 40),
        ParallelConfig(P=1, T=2, D=4, B_micro=1, zero_stage=2, sp=True),
        HWConfig(),
    ),
    "llama-70b-tp8-dp4-h100x32": (
        _llama_family(80, 8192, 28672, 64),
        ParallelConfig(P=1, T=8, D=4, B_micro=1, zero_stage=2, sp=True),
        HWConfig(),
    ),
    "llama-70b-tp4-pp4-dp4-h100x64": (
        _llama_family(80, 8192, 28672, 64),
        ParallelConfig(P=4, T=4, D=4, m=32, B_micro=1, zero_stage=2, sp=True,
                       recompute="full"),
        HWConfig(),
    ),
    "llama-70b-cp4-tp4-dp4-h100x64-32k": (
        _llama_family(80, 8192, 28672, 64, S=32768),
        ParallelConfig(P=1, T=4, D=4, C=4, B_micro=1, zero_stage=2, sp=True),
        HWConfig(),
    ),
    "llama3-405b-pp16-tp8-dp4-h100x512": (
        _llama_family(126, 16384, 53248, 128),
        ParallelConfig(P=16, T=8, D=4, m=64, B_micro=1, zero_stage=3, sp=True,
                       recompute="full"),
        HWConfig(),
    ),
    "deepseek-v3-moe-dp8-ep8-h100x128": (
        # DeepSeek-V3 671B (37B active). Simplified — real config uses TP=1,
        # PP=2 DualPipe, EP=8 within node, ZeRO-1 + full activation ckpt.
        _llama_family(61, 7168, 18432, 128, V=100000),
        ParallelConfig(P=2, T=1, D=8, E=8, m=8, B_micro=1, zero_stage=1,
                       recompute="full"),
        HWConfig(),
    ),
    "mixtral-47b-tp2-dp8-ep2-h100x32": (
        _llama_family(32, 4096, 14336, 32),
        ParallelConfig(P=1, T=2, D=8, E=2, B_micro=1, zero_stage=2),
        HWConfig(),
    ),
}


# ---------- CLI -----------------------------------------------------------


def _fmt_gb(x): return f"{x:>7.1f} GB"
def _fmt_s(x): return f"{x:>7.2f} s"


def print_report(cfg, para, hw, name=""):
    mem = mem_estimate(cfg, para, hw)
    st = step_time(cfg, para, hw)
    total = st["total"]
    print()
    print(f"=== {name or 'custom'} ===")
    print(f"model: L={cfg.L} H={cfg.H} I={cfg.I} A={cfg.A} S={cfg.S} "
          f"P_params={cfg.params/1e9:.1f}B")
    print(f"para:  P={para.P} T={para.T} D={para.D} C={para.C} E={para.E} "
          f"m={para.m} zero{para.zero_stage} sp={para.sp} "
          f"recomp={para.recompute} world={para.world}")
    print(f"hw:    {hw.name} peak={hw.peak_flops/1e12:.0f}TF "
          f"nvlink={hw.nvlink_bw/1e9:.0f}GB/s ib={hw.ib_bw/1e9:.0f}GB/s "
          f"mfu={hw.mfu*100:.0f}%")
    print(f"tokens/step: {cfg.tokens_per_step:,}")
    print()
    print(f"Memory per GPU:")
    print(f"  params    {_fmt_gb(mem['params'])}")
    print(f"  grads     {_fmt_gb(mem['grads'])}")
    print(f"  opt state {_fmt_gb(mem['opt'])}")
    print(f"  activation{_fmt_gb(mem['act'])}")
    print(f"  TOTAL     {_fmt_gb(mem['total'])}")
    print()
    print("Step time breakdown:")
    print(f"  compute   {_fmt_s(st['compute'])}  ({st['compute']/total*100:4.1f}%)")
    print(f"  comm      {_fmt_s(st['comm'])}  ({st['comm']/total*100:4.1f}%)")
    print(f"  bubble    {_fmt_s(st['bubble'])}  ({st['bubble']/total*100:4.1f}%)")
    print(f"  overhead  {_fmt_s(st['overhead'])}  ({st['overhead']/total*100:4.1f}%)")
    print(f"  TOTAL     {_fmt_s(st['total'])}")
    print()
    print("Per-collective breakdown:")
    for c in comm_volumes(cfg, para, hw):
        t = comm_time_seconds(c, hw)
        print(f"  {c.kind:10s} vol={c.vol_per_gpu/1e6:7.1f}MB × {c.count:5d} "
              f"scope={c.scope:5s} overlap={c.overlap*100:3.0f}%  "
              f"→ {t:6.3f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None,
                    choices=list(PRESETS.keys()) + ["all"])
    args = ap.parse_args()

    if args.preset in (None, "all"):
        for name, (cfg, para, hw) in PRESETS.items():
            print_report(cfg, para, hw, name=name)
    else:
        cfg, para, hw = PRESETS[args.preset]
        print_report(cfg, para, hw, name=args.preset)


if __name__ == "__main__":
    main()
