"""Chapter 7: making the data path and local operators CP-correct.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/20_cp_dataloader_halo.py

Companion to 19_cp_loss_and_metrics.py. That file covers cross-token
*reductions*; this one covers everything upstream of the model plus the
operators that need a few tokens from their neighbour.

Parts:
    1. rank mapping     - a CP group must be fed the SAME sample
    2. zigzag split     - contiguous sharding halves your throughput under
                          a causal mask; the fix permutes the sequence
    3. position + mask  - RoPE needs GLOBAL positions, and after a zigzag
                          permutation the causal mask is no longer triangular
    4. label shift      - next-token labels cross the shard boundary
    5. halo exchange    - causal conv / sliding-window / SSM scan need the
                          k-1 tokens that live on the previous rank

Every part reconstructs the single-rank result and asserts equality.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import setup, cleanup, rprint, assert_close

A, D = 4, 16           # heads, head dim
CHUNKS_PER_RANK = 2    # zigzag needs 2 chunks per rank
ROPE_BASE = 10000.0


# ---- sequence layouts -----------------------------------------------------


def contiguous_perm(S: int, cp: int) -> torch.Tensor:
    """perm[r*L + i] = global position of local slot i on rank r."""
    return torch.arange(S)


def zigzag_perm(S: int, cp: int) -> torch.Tensor:
    """Load-balanced split: rank r takes chunk r and chunk (2*cp-1-r).

    With 2*cp chunks, pairing an early chunk with a late one gives every rank
    the same number of unmasked (query, key) pairs under a causal mask.
    """
    n_chunks = CHUNKS_PER_RANK * cp
    c = S // n_chunks
    chunks = [torch.arange(m * c, (m + 1) * c) for m in range(n_chunks)]
    return torch.cat([torch.cat([chunks[r], chunks[n_chunks - 1 - r]])
                      for r in range(cp)])


def causal_work(perm: torch.Tensor, S: int, cp: int) -> list[int]:
    """Unmasked (q,k) pairs per rank under a global causal mask."""
    L = S // cp
    return [int((perm[r * L:(r + 1) * L] + 1).sum().item()) for r in range(cp)]


# ---- RoPE -----------------------------------------------------------------


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: (S, A, D) -- rotate using the GLOBAL positions in `pos` (S,)."""
    half = x.shape[-1] // 2
    inv = 1.0 / (ROPE_BASE ** (torch.arange(half, device=x.device).float() * 2 / x.shape[-1]))
    ang = pos.float().unsqueeze(-1) * inv.unsqueeze(0)        # (S, half)
    cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)  # (S, 1, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attend(q, k, v, mask):
    """q: (Sq,A,D), k/v: (Sk,A,D), mask: (Sq,Sk) bool (True = keep)."""
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))           # (A, S, D)
    scores = q @ k.transpose(-1, -2) / D ** 0.5
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1)            # (Sq, A, D)


# ---- part 1: rank mapping -------------------------------------------------


def part1_rank_mapping(rank, world, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 1  which sample does each rank load?", rank=0)
    rprint("=" * 74, rank=0)

    cp, n_dp = 2, world // 2
    cp_rank, dp_rank = rank % cp, rank // cp     # global = dp_rank * cp + cp_rank

    cp_groups = [dist.new_group(list(range(d * cp, (d + 1) * cp))) for d in range(n_dp)]
    cp_group = cp_groups[dp_rank]

    rprint(f"  mesh: world={world} = DP{n_dp} x CP{cp}; this rank is "
           f"dp={dp_rank} cp={cp_rank}", rank=0)

    def gather_in_cp(x: int):
        t = torch.tensor([x], device=device)
        out = [torch.zeros_like(t) for _ in range(cp)]
        dist.all_gather(out, t, group=cp_group)
        return [int(o.item()) for o in out]

    step = 7
    wrong = gather_in_cp(step * world + rank)          # indexed by global rank
    good = gather_in_cp(step * n_dp + dp_rank)         # indexed by dp rank

    rprint(f"  WRONG   sampler keyed on global_rank -> CP group sees samples {wrong}",
           rank=0)
    rprint(f"  CORRECT sampler keyed on dp_rank     -> CP group sees samples {good}",
           rank=0)
    assert len(set(wrong)) == cp and len(set(good)) == 1

    # The shuffle RNG has the same requirement as the index.
    def head_of_shuffle(seed: int):
        g = torch.Generator().manual_seed(seed)
        return gather_in_cp(int(torch.randperm(1024, generator=g)[0]))

    rprint(f"  WRONG   shuffle seed = base+global_rank -> {head_of_shuffle(100 + rank)}",
           rank=0)
    rprint(f"  CORRECT shuffle seed = base+dp_rank     -> {head_of_shuffle(100 + dp_rank)}",
           rank=0)

    rprint("\n  A CP group holds ONE sequence between them. Feed its ranks", rank=0)
    rprint("  different samples and rank 0 holds the first half of document A", rank=0)
    rprint("  while rank 1 holds the second half of document B -- attention", rank=0)
    rprint("  happily mixes them, no error is raised, and the effective batch", rank=0)
    rprint("  size silently becomes CP x larger than you configured.", rank=0)
    rprint("  Same rule for TP and for PP stages: only the DP dim varies data.", rank=0)


# ---- part 2: contiguous vs zigzag ----------------------------------------


def part2_zigzag(rank, cp, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 2  contiguous vs zigzag sequence split", rank=0)
    rprint("=" * 74, rank=0)

    S = 8 * cp
    w_cont = causal_work(contiguous_perm(S, cp), S, cp)
    w_zig = causal_work(zigzag_perm(S, cp), S, cp)

    def report(name, w):
        mean = sum(w) / len(w)
        rprint(f"  {name:11s} {w}  max/min={max(w)/min(w):.2f}x  "
               f"max/mean={max(w)/mean:.2f}x", rank=0)

    rprint(f"  S={S}, CP={cp}; unmasked (q,k) pairs per rank under a causal mask", rank=0)
    report("contiguous", w_cont)
    report("zigzag", w_zig)

    assert max(w_zig) == min(w_zig)
    assert max(w_cont) / (sum(w_cont) / cp) > 1.5
    rprint("\n  Contiguous sharding gives rank 0 almost nothing to do and the last", rank=0)
    rprint("  rank almost everything. A collective runs at the speed of the", rank=0)
    rprint("  slowest rank, so the cost is max/mean -- approaching 2x as CP", rank=0)
    rprint("  grows, not the larger max/min figure. Zigzag pairs chunk r with", rank=0)
    rprint(f"  chunk {CHUNKS_PER_RANK * cp - 1}-r, making the count identical everywhere.",
           rank=0)
    rprint("  The cost: the shard is no longer a contiguous span, which is what", rank=0)
    rprint("  breaks positions (part 3), masks (part 3) and halos (part 5).", rank=0)


# ---- part 3: global positions and the permuted mask ----------------------


def part3_position_and_mask(rank, cp, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 3  RoPE positions and mask reconstruction under zigzag", rank=0)
    rprint("=" * 74, rank=0)

    S = 8 * cp
    L = S // cp
    g = torch.Generator().manual_seed(7)
    q = torch.randn(S, A, D, generator=g).to(device)
    k = torch.randn(S, A, D, generator=g).to(device)
    v = torch.randn(S, A, D, generator=g).to(device)

    # Two packed documents in one sequence -> attention must not cross docs.
    doc = torch.zeros(S, dtype=torch.long, device=device)
    doc[S // 2 + 3:] = 1
    pos_global = torch.arange(S, device=device)

    # Reference: full sequence, global positions, causal AND same-document.
    causal = pos_global.unsqueeze(1) >= pos_global.unsqueeze(0)
    same_doc = doc.unsqueeze(1) == doc.unsqueeze(0)
    ref = attend(rope(q, pos_global), rope(k, pos_global), v, causal & same_doc)

    perm = zigzag_perm(S, cp).to(device)
    mine = perm[rank * L:(rank + 1) * L]
    rprint(f"  rank {rank} owns global positions {mine.tolist()}", rank=0)

    q_l, k_l, v_l = q[mine], k[mine], v[mine]
    doc_l = doc[mine]

    # K/V from every rank. A real implementation rotates them (Ring) or
    # exchanges heads (Ulysses); all-gather keeps this demo about positions.
    def gather_rows(x):
        parts = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=0)

    k_all, v_all = gather_rows(k_l), gather_rows(v_l)
    pos_all, doc_all = perm, doc[perm]

    # WRONG: local positions 0..L-1, and a mask built from local indices.
    pos_bad = torch.arange(L, device=device)
    bad_mask = pos_bad.unsqueeze(1) >= torch.arange(k_all.shape[0], device=device).unsqueeze(0)
    out_bad = attend(rope(q_l, pos_bad), rope(k_all, torch.arange(k_all.shape[0], device=device)),
                     v_all, bad_mask)

    # CORRECT: global positions everywhere; mask from positions and doc ids.
    mask = ((mine.unsqueeze(1) >= pos_all.unsqueeze(0))
            & (doc_l.unsqueeze(1) == doc_all.unsqueeze(0)))
    out_good = attend(rope(q_l, mine), rope(k_all, pos_all), v_all, mask)

    def unpermute(local):
        gathered = gather_rows(local)
        full = torch.empty_like(gathered)
        full[perm] = gathered
        return full

    full_bad, full_good = unpermute(out_bad), unpermute(out_good)
    rprint(f"  WRONG   local positions + index mask: max|err| = "
           f"{(full_bad - ref).abs().max().item():.3f}", rank=0)
    rprint(f"  CORRECT global positions + doc mask:  max|err| = "
           f"{(full_good - ref).abs().max().item():.2e}", rank=0)

    assert (full_bad - ref).abs().max().item() > 0.1
    assert_close(full_good, ref, rtol=1e-4, atol=1e-5, name="zigzag CP attention")

    rprint("\n  Three things had to travel with the tokens: the global position", rank=0)
    rprint("  ids (RoPE is relative, so wrong offsets change every score), the", rank=0)
    rprint("  document ids (packing), and the permutation itself (needed to put", rank=0)
    rprint("  the output back in order). Ship them as fields of the batch --", rank=0)
    rprint("  never recompute them from a local arange. cu_seqlens for varlen", rank=0)
    rprint("  kernels has to be rebuilt per rank from the sharded doc ids.", rank=0)


# ---- part 4: label shift across the boundary -----------------------------


def part4_label_shift(rank, cp, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 4  next-token labels cross the shard boundary", rank=0)
    rprint("=" * 74, rank=0)

    S = 8 * cp
    L = S // cp
    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 1000, (S,), generator=g).to(device)

    # A local roll breaks the last pair of every contiguous run a rank owns.
    # Contiguous: cp runs, minus the one ending at global S-1 (no label) = cp-1.
    # Zigzag: 2*cp runs, minus that same one, minus one more because rank cp-1
    # happens to own the adjacent pair (chunk cp-1, chunk cp) so its internal
    # seam is accidentally correct.
    layouts = (("contiguous", contiguous_perm(S, cp).to(device), cp - 1),
               ("zigzag", zigzag_perm(S, cp).to(device), CHUNKS_PER_RANK * cp - 2))
    for name, perm, expect_bad in layouts:
        mine = perm[rank * L:(rank + 1) * L]
        tok_l = tokens[mine]

        # Reference pairing for the positions this rank owns: label at global
        # position p is token p+1 (the last position has no label).
        valid = mine < S - 1
        ref_lab = torch.where(valid, tokens[mine.clamp(max=S - 2) + 1], tokens[mine])

        # WRONG: roll inside the local shard.
        bad_lab = torch.roll(tok_l, shifts=-1, dims=0)
        bad = int(((bad_lab != ref_lab) & valid).sum().item())

        # CORRECT: shift the whole sequence BEFORE sharding, then apply perm.
        labels_global = torch.cat([tokens[1:], tokens[-1:]])
        good_lab = labels_global[mine]
        good = int(((good_lab != ref_lab) & valid).sum().item())

        n_bad = torch.tensor([bad], device=device)
        dist.all_reduce(n_bad)
        rprint(f"  {name:11s} WRONG local roll: {int(n_bad.item())}/{S} label pairs "
               f"corrupted (= {expect_bad}); CORRECT pre-shift: {good}", rank=0)
        assert good == 0
        assert int(n_bad.item()) == expect_bad

    rprint("\n  Under a contiguous split only the CP-1 boundary tokens are wrong,", rank=0)
    rprint("  which is a rounding error on the loss and therefore invisible.", rank=0)
    rprint("  Under zigzag each rank holds two disjoint chunks, so a local roll", rank=0)
    rprint("  mispairs a whole chunk boundary too.", rank=0)
    rprint("  Shift labels once, globally, before the split -- it costs nothing", rank=0)
    rprint("  and needs no communication. If a head must look k tokens ahead", rank=0)
    rprint("  (MTP, speculative decoding) the same trick handles all k at once.", rank=0)


# ---- part 5: halo exchange for local operators ---------------------------


def part5_halo(rank, cp, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 5  halo exchange: causal conv / sliding window / SSM scan", rank=0)
    rprint("=" * 74, rank=0)

    S, K = 8 * cp, 4          # sequence, conv kernel width
    L = S // cp
    g = torch.Generator().manual_seed(13)
    x = torch.randn(S, D, generator=g).to(device)
    w = torch.randn(K, D, generator=g).to(device)

    def causal_conv(inp, left):
        """y[i] = sum_j w[j] * inp[i-j], with `left` extra rows of history."""
        padded = torch.cat([left, inp], dim=0)
        n = inp.shape[0]
        return sum(w[j] * padded[K - 1 - j: K - 1 - j + n] for j in range(K))

    ref = causal_conv(x, torch.zeros(K - 1, D, device=device))

    # Contiguous split: the history a rank needs is on rank-1. (Zigzag would
    # need two halos from two different ranks -- which is exactly why frameworks
    # keep the split contiguous when the model has local operators.)
    lo = rank * L
    x_l = x[lo:lo + L]

    # WRONG: zero-pad the shard, as a stock conv1d would.
    bad = causal_conv(x_l, torch.zeros(K - 1, D, device=device))

    # CORRECT: receive K-1 rows of history from the left neighbour.
    halo = torch.zeros(K - 1, D, device=device)
    reqs = []
    if rank + 1 < cp:
        reqs.append(dist.P2POp(dist.isend, x_l[-(K - 1):].contiguous(), rank + 1))
    if rank > 0:
        reqs.append(dist.P2POp(dist.irecv, halo, rank - 1))
    for r in dist.batch_isend_irecv(reqs) if reqs else []:
        r.wait()
    good = causal_conv(x_l, halo)

    def gather_rows(t):
        parts = [torch.empty_like(t) for _ in range(cp)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=0)

    full_bad, full_good = gather_rows(bad), gather_rows(good)
    n_wrong = int((full_bad - ref).abs().max(dim=-1).values.gt(1e-5).sum().item())
    rprint(f"  S={S}, kernel K={K}, CP={cp}", rank=0)
    rprint(f"  WRONG   zero-padded shards: {n_wrong} of {S} positions wrong "
           f"(= (K-1) x (CP-1) at the seams)", rank=0)
    rprint(f"  CORRECT halo of K-1={K-1} rows:  max|err| = "
           f"{(full_good - ref).abs().max().item():.2e}", rank=0)

    assert n_wrong == (K - 1) * (cp - 1)
    assert_close(full_good, ref, rtol=1e-4, atol=1e-5, name="CP causal conv + halo")

    rprint("\n  The error is confined to K-1 rows per seam, so it shrinks as the", rank=0)
    rprint("  shard grows -- the loss barely moves and the bug survives review.", rank=0)
    rprint("  Halo width per operator: causal conv K-1, sliding-window attention", rank=0)
    rprint("  window-1, and a Mamba/SSM scan needs the recurrent STATE from the", rank=0)
    rprint("  previous rank, which serialises the ranks unless you use a chunked", rank=0)
    rprint("  scan. Any depthwise conv in a ViT/audio front-end is the same.", rank=0)


def main():
    rank, world, device = setup()
    if world < 4 or world % 2:
        rprint("This demo needs an even world >= 4: torchrun --nproc-per-node=4 ...")
        cleanup()
        return

    part1_rank_mapping(rank, world, device)
    # Parts 2-5 treat the whole world as one CP group (DP=1).
    part2_zigzag(rank, world, device)
    part3_position_and_mask(rank, world, device)
    part4_label_shift(rank, world, device)
    part5_halo(rank, world, device)

    rprint("\n" + "=" * 74, rank=0)
    rprint("data path and local operators match the single-rank reference", rank=0)
    rprint("=" * 74, rank=0)
    cleanup()


if __name__ == "__main__":
    main()
