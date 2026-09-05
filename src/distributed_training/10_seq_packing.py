"""Chapter 12: Sequence packing with First-Fit-Decreasing (FFD) + optional
Flash-Attention varlen call.

Run:
    python src/distributed_training/10_seq_packing.py

What this demo shows:
    - Take a list of variable-length sequences.
    - FFD-pack them into `n_packs` fixed-length bins (each of size seq_len).
    - Build cu_seqlens / max_seqlen tensors compatible with the
      `flash-attn` varlen API and with PyTorch's SDPA block-diagonal mask
      trick.
    - Verify equivalence: attention with block-diagonal mask over the
      packed sequence == running attention on each original sequence
      separately then concatenating.

We do NOT require flash-attn to be installed: we use SDPA with an
attn_mask built from cu_seqlens as a stand-in. Real training would call
`flash_attn_varlen_func(qkv, cu_seqlens, max_seqlen)` which does the
same thing without materializing the mask.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def ffd_pack(lengths: List[int], bin_size: int) -> List[List[int]]:
    """First-Fit-Decreasing packing. Returns list of bins, each is a list
    of original-sequence indices."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: List[List[int]] = []
    used: List[int] = []
    for i in order:
        L = lengths[i]
        if L > bin_size:
            raise ValueError(f"seq {i} (len {L}) > bin_size {bin_size}")
        placed = False
        for b_idx in range(len(bins)):
            if used[b_idx] + L <= bin_size:
                bins[b_idx].append(i)
                used[b_idx] += L
                placed = True
                break
        if not placed:
            bins.append([i])
            used.append(L)
    return bins


def build_cu_seqlens(bin_indices: List[int], all_lengths: List[int]) -> torch.Tensor:
    """cu_seqlens[0..n] such that seq k covers [cu[k], cu[k+1])."""
    cu = [0]
    for i in bin_indices:
        cu.append(cu[-1] + all_lengths[i])
    return torch.tensor(cu, dtype=torch.int32)


def build_block_diag_mask(cu_seqlens: torch.Tensor, total_len: int, causal: bool,
                          device) -> torch.Tensor:
    """Return an additive mask (total_len, total_len) with 0 inside each
    block and -inf across blocks. If causal, also mask the upper triangle
    within each block."""
    mask = torch.full((total_len, total_len), float("-inf"), device=device)
    for k in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[k]), int(cu_seqlens[k + 1])
        block = mask[s:e, s:e]
        block.fill_(0)
        if causal:
            up = torch.triu(torch.ones_like(block), diagonal=1).bool()
            block.masked_fill_(up, float("-inf"))
    return mask


def attention(q, k, v, mask=None):
    """q,k,v: (1, A, T, d). mask: (T, T) additive."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Fake dataset of variable-length sequences.
    lengths = [11, 5, 7, 3, 9, 4, 6, 2, 8, 5]
    bin_size = 16
    n_seq = len(lengths)
    d = 8
    A = 2  # heads

    # ---- pack ----
    bins = ffd_pack(lengths, bin_size)
    total_used = sum(sum(lengths[i] for i in b) for b in bins)
    theoretical_min_bins = (sum(lengths) + bin_size - 1) // bin_size
    packing_ratio = total_used / (len(bins) * bin_size)
    print(f"packed {n_seq} sequences into {len(bins)} bins of size {bin_size} "
          f"(min possible = {theoretical_min_bins})")
    print(f"packing ratio = {packing_ratio:.2%}")

    # ---- build features for each original sequence (random tokens) ----
    feats = [torch.randn(L, d * A, device=device) for L in lengths]  # (L, H)

    causal = True

    # ---- reference: run attention on each original sequence separately ----
    def per_seq_attn(x):
        L, H = x.shape
        qkv = x.view(1, L, A, d).transpose(1, 2)  # (1,A,L,d) — reuse for q=k=v
        mask = None
        if causal:
            mask = torch.zeros(L, L, device=device)
            up = torch.triu(torch.ones_like(mask), diagonal=1).bool()
            mask.masked_fill_(up, float("-inf"))
        return attention(qkv, qkv, qkv, mask).transpose(1, 2).reshape(L, H)

    ref_outs = [per_seq_attn(f) for f in feats]

    # ---- packed: for each bin, concat -> single attention with block mask ----
    for b_idx, bin_indices in enumerate(bins):
        seqs = [feats[i] for i in bin_indices]
        packed_x = torch.cat(seqs, dim=0)                    # (T, H)
        T, H = packed_x.shape
        cu = build_cu_seqlens(bin_indices, lengths).to(device)
        mask = build_block_diag_mask(cu, T, causal, device)
        qkv = packed_x.view(1, T, A, d).transpose(1, 2)      # (1, A, T, d)
        packed_out = attention(qkv, qkv, qkv, mask).transpose(1, 2).reshape(T, H)

        # Compare each sub-sequence output
        for k, i in enumerate(bin_indices):
            s, e = int(cu[k]), int(cu[k + 1])
            our = packed_out[s:e]
            ref = ref_outs[i]
            assert torch.allclose(our, ref, rtol=1e-4, atol=1e-4), \
                f"bin {b_idx} sub {k} (orig seq {i}) mismatch: " \
                f"max diff {(our - ref).abs().max().item():.2e}"

    print("packed attention matches per-sequence attention ✓")
    print("cu_seqlens example (first bin):",
          build_cu_seqlens(bins[0], lengths).tolist())
    print("max_seqlen (first bin):",
          max(lengths[i] for i in bins[0]))


if __name__ == "__main__":
    main()
