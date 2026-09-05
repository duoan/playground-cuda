"""A tiny Transformer used across demos.

Deliberately not fused / not optimized — we want every op to be a clear
call so parallelism edits are obvious. No GQA / no RoPE / no dropout, so
that verification tests are as tight as possible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, H)
        return self.w2(F.gelu(self.w1(x)))


class ToyAttention(nn.Module):
    """Vanilla MHA, no GQA. Uses SDPA so it stays fast enough for tests."""

    def __init__(self, hidden: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_h = hidden // n_heads
        self.causal = causal
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, H)
        B, S, H = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.d_h).transpose(1, 2)
        # (B, A, S, d_h) each
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.wo(out)


class ToyBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, inter: int, causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = ToyAttention(hidden, n_heads, causal=causal)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = ToyMLP(hidden, inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, n_layers: int = 4, hidden: int = 128, n_heads: int = 4,
                 inter: int = 512, vocab: int = 1024, seq: int = 64,
                 causal: bool = True, tie_embed: bool = True):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(seq, hidden)
        self.blocks = nn.ModuleList([
            ToyBlock(hidden, n_heads, inter, causal=causal)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        if tie_embed:
            self.head.weight = self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # idx: (B, S)
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.emb(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, S, V)


def make_random_batch(B: int, S: int, V: int, device: torch.device,
                      seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for verification tests."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, V, (B, S), generator=g)
    tgt = torch.randint(0, V, (B, S), generator=g)
    return ids.to(device), tgt.to(device)
