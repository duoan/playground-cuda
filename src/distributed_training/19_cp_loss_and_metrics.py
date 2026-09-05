"""Chapter 7: making cross-token *reductions* CP-correct.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/19_cp_loss_and_metrics.py

CP shards the sequence. Attention is the loud part of the port -- if you get
Ring/Ulysses wrong the numbers explode and you notice. Everything that
*reduces across tokens* is the quiet part: nothing crashes, the loss curve
looks plausible, and the model is simply trained on the wrong objective.

Every part below computes the same quantity three ways -- a single-rank
reference over the full sequence, a tempting-but-wrong sharded version, and
the correct sharded version -- and asserts which one matches.

Parts:
    1. cross-entropy denominator      - a mean of local means is not the mean
    2. loss scale vs gradient reduce  - off by exactly CP, in the gradient only
    3. MoE router aux loss            - needs a GLOBAL expert histogram
    4. pooling heads (last / mean)    - the last token lives on ONE rank
    5. metrics (ppl / accuracy)       - weight by token count, not by rank
    6. grad-norm clipping             - never sum over a replica group
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import setup, cleanup, get_rank, get_world_size, rprint, assert_close

H, V = 32, 64          # hidden size, vocab size
LOCAL_S = 16           # tokens per CP rank -> global S = LOCAL_S * CP


# ---- differentiable collectives -------------------------------------------
# Megatron calls these `f` and `g`. A plain dist.all_reduce mutates the tensor
# in place and autograd never learns about the other ranks, so any loss that
# has to *combine* contributions from several CP ranks needs this wrapper.


class AllReduceSum(torch.autograd.Function):
    """Sum across the CP group in forward; identity in backward.

    Backward is identity because d(sum_r x_r)/d(x_i) = 1: each rank only ever
    needs the gradient of its own contribution.
    """

    @staticmethod
    def forward(ctx, x, group):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def all_reduce_sum(x, group=None):
    return AllReduceSum.apply(x, group)


def reduce_scalar(value: float, device, group=None) -> float:
    """Sum a python scalar across the group. No autograd: counts only."""
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t.item()


# ---- shared batch ---------------------------------------------------------


def make_batch(cp: int, device):
    """Build one global batch, bit-identical on every rank.

    Two properties matter for the bugs below, and both are true of real SFT
    batches:

    1. The loss mask is a long prompt (masked out) + an answer + right
       padding, so the number of *valid* tokens differs wildly per CP rank.
       With cp=4 rank 0 ends up with zero valid tokens.
    2. Per-token loss falls along the sequence -- later tokens have more
       context and are easier. So the per-rank mean loss differs too, and
       re-weighting the ranks actually changes the answer.
    """
    S = LOCAL_S * cp
    g = torch.Generator().manual_seed(1234)
    h = torch.randn(S, H, generator=g)
    # Scale W so the logits actually have spread -- otherwise an "easy" label
    # (the argmax) is no cheaper than a random one and the loss ramp vanishes.
    W = torch.randn(H, V, generator=g) * 0.6

    # Predictability ramps from 0 to 1 along the sequence: an easy token gets
    # the model's own argmax as its label (low loss), a hard one gets noise.
    easy = (h @ W).argmax(-1)
    noise = torch.randint(0, V, (S,), generator=g)
    take_easy = torch.rand(S, generator=g) < torch.linspace(0.0, 1.0, S)
    labels = torch.where(take_easy, easy, noise)

    prompt_len = LOCAL_S + 4           # covers rank 0, eats 4 tokens of rank 1
    pad_from = S - LOCAL_S + 2         # leaves only 2 tokens on the last rank
    mask = torch.zeros(S)
    mask[prompt_len:pad_from] = 1.0

    return (h.to(device), W.to(device), labels.to(device), mask.to(device))


def shard(x, cp: int, rank: int):
    """Contiguous CP shard along dim 0."""
    return x.chunk(cp, dim=0)[rank].contiguous()


def token_nll(h, W, labels):
    """Per-token negative log-likelihood. No reduction."""
    return F.cross_entropy(h @ W, labels, reduction="none")


# ---- part 1: the cross-entropy denominator --------------------------------


def part1_loss_denominator(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 1  cross-entropy denominator", rank=0)
    rprint("=" * 74, rank=0)

    h, W, labels, mask = make_batch(cp, device)
    nll = token_nll(h, W, labels)
    ref = ((nll * mask).sum() / mask.sum()).item()

    h_l = shard(h, cp, rank)
    lab_l = shard(labels, cp, rank)
    mask_l = shard(mask, cp, rank)
    nll_l = token_nll(h_l, W, lab_l)

    counts = [int(shard(mask, cp, r).sum().item()) for r in range(cp)]
    rprint(f"valid tokens per CP rank: {counts}  total={sum(counts)} "
           f"of {LOCAL_S * cp} positions", rank=0)

    local_sum = (nll_l * mask_l).sum()
    local_cnt = mask_l.sum()

    # WRONG 1: local masked mean, then average the means over CP.
    local_mean = local_sum / local_cnt                     # 0/0 on rank 0
    wrong_mean_of_means = reduce_scalar(local_mean.item(), device) / cp

    # WRONG 2: same, but guard the empty shard away.
    contrib = local_mean.item() if local_cnt > 0 else 0.0
    n_nonempty = reduce_scalar(1.0 if local_cnt > 0 else 0.0, device)
    wrong_guarded = reduce_scalar(contrib, device) / n_nonempty

    # WRONG 3: denominator = all local positions (the numel() bug, CP edition).
    wrong_numel = reduce_scalar((nll_l * mask_l).sum().item(), device) / (LOCAL_S * cp)

    # CORRECT: reduce numerator and denominator separately, then divide.
    good = reduce_scalar(local_sum.item(), device) / reduce_scalar(local_cnt.item(), device)

    per_rank = [round((shard(nll * mask, cp, r).sum()
                       / shard(mask, cp, r).sum().clamp(min=1)).item(), 2)
                for r in range(cp)]
    rprint(f"mean loss per CP rank:     {per_rank}  "
           f"(falls along the sequence)", rank=0)

    rprint(f"  reference (single rank)      {ref:.6f}", rank=0)
    rprint(f"  WRONG mean-of-local-means    {wrong_mean_of_means:.6f}  "
           f"<- NaN: rank 0 divided 0/0", rank=0)
    rprint(f"  WRONG guarded mean-of-means  {wrong_guarded:.6f}  "
           f"({wrong_guarded / ref:.3f}x ref)", rank=0)
    rprint(f"  WRONG numel() denominator    {wrong_numel:.6f}  "
           f"({wrong_numel / ref:.3f}x ref)", rank=0)
    rprint(f"  CORRECT sum/sum then divide  {good:.6f}", rank=0)

    assert wrong_mean_of_means != wrong_mean_of_means, "expected NaN"
    assert abs(wrong_guarded / ref - 1) > 0.05
    assert abs(wrong_numel / ref - 1) > 0.05
    assert abs(good - ref) < 1e-5

    rprint("\n  The guarded variant is the dangerous one: finite, stable, and", rank=0)
    rprint("  wrong. It reweights every CP rank to 1/CP regardless of how many", rank=0)
    rprint("  real tokens it holds, so short-answer samples get up-weighted.", rank=0)


# ---- part 2: loss scale vs the gradient reduce -----------------------------


def part2_gradient_scale(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 2  loss scale vs the gradient all-reduce", rank=0)
    rprint("=" * 74, rank=0)

    h, W, labels, mask = make_batch(cp, device)

    W_ref = W.clone().requires_grad_(True)
    ((token_nll(h, W_ref, labels) * mask).sum() / mask.sum()).backward()
    g_ref = W_ref.grad

    # Sharded: each rank computes local_sum / GLOBAL_count.
    W_l = W.clone().requires_grad_(True)
    h_l, lab_l, mask_l = shard(h, cp, rank), shard(labels, cp, rank), shard(mask, cp, rank)
    global_cnt = reduce_scalar(mask_l.sum().item(), device)
    ((token_nll(h_l, W_l, lab_l) * mask_l).sum() / global_cnt).backward()
    g_local = W_l.grad

    # The gradient reduce that follows. CP ranks hold *replicas* of W, and in
    # Megatron the CP dim is folded into the data-parallel group -- which
    # AVERAGES. Averaging partial sums loses a factor of CP.
    g_avg = g_local.clone()
    dist.all_reduce(g_avg, op=dist.ReduceOp.SUM)
    g_avg /= cp
    g_sum = g_local.clone()
    dist.all_reduce(g_sum, op=dist.ReduceOp.SUM)

    r_avg = (g_avg.norm() / g_ref.norm()).item()
    r_sum = (g_sum.norm() / g_ref.norm()).item()
    rprint(f"  |g| ratio to reference,  AVG over CP: {r_avg:.4f}  (= 1/CP = {1/cp:.4f})",
           rank=0)
    rprint(f"  |g| ratio to reference,  SUM over CP: {r_sum:.4f}", rank=0)

    assert abs(r_avg - 1.0 / cp) < 1e-3
    assert_close(g_sum, g_ref, rtol=1e-4, atol=1e-6, name="CP grad (SUM)")

    rprint("\n  Both losses PRINT the same number -- only the gradient is wrong,", rank=0)
    rprint("  so you cannot see this on the loss curve. It looks like the LR is", rank=0)
    rprint(f"  {cp}x too small. Two valid fixes:", rank=0)
    rprint(f"    a) keep the AVG reduce, multiply the loss by CP={cp}", rank=0)
    rprint("    b) reduce SUM over the CP dim and AVG only over the true DP dim", rank=0)
    rprint("  Pick one and assert it in a test; frameworks differ on convention.", rank=0)


# ---- part 3: MoE router auxiliary loss ------------------------------------


def switch_aux_loss(gate_probs, assign, n_exp, *, group=None, cp_global=False):
    """Switch-Transformer load-balancing loss: E * sum_e f_e * P_e.

    f_e = fraction of tokens routed to e, P_e = mean router prob of e.
    Both are means *over tokens*, so under CP both need a global reduction.
    """
    n_tok = gate_probs.shape[0]
    counts = torch.zeros(n_exp, device=gate_probs.device).index_add_(
        0, assign, torch.ones(n_tok, device=gate_probs.device))
    prob_sum = gate_probs.sum(dim=0)

    if cp_global:
        counts = all_reduce_sum(counts, group)
        prob_sum = all_reduce_sum(prob_sum, group)
        n_tok = reduce_scalar(float(n_tok), gate_probs.device, group)

    return n_exp * torch.dot(counts / n_tok, prob_sum / n_tok)


def part3_moe_aux_loss(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 3  MoE router auxiliary loss", rank=0)
    rprint("=" * 74, rank=0)

    n_exp = cp
    S = LOCAL_S * cp

    # Router logits that send every token on rank r to expert r. Globally this
    # is a PERFECTLY balanced router: each expert gets exactly 1/E of tokens.
    logits = torch.full((S, n_exp), 0.0, device=device)
    for r in range(cp):
        logits[r * LOCAL_S:(r + 1) * LOCAL_S, r] = 3.0
    logits.requires_grad_(True)

    probs = logits.softmax(dim=-1)
    assign = probs.argmax(dim=-1)
    ref = switch_aux_loss(probs, assign, n_exp)

    lo, hi = rank * LOCAL_S, (rank + 1) * LOCAL_S
    probs_l, assign_l = probs[lo:hi], assign[lo:hi]

    local_only = switch_aux_loss(probs_l, assign_l, n_exp)
    cp_global = switch_aux_loss(probs_l, assign_l, n_exp, cp_global=True)

    rprint(f"  E={n_exp}, global routing is perfectly balanced "
           f"(each expert gets 1/{n_exp} of tokens)", rank=0)
    rprint(f"  reference aux loss (full seq)   {ref.item():.4f}   "
           f"<- minimum for E={n_exp}", rank=0)
    rprint(f"  WRONG local-only histogram      {local_only.item():.4f}   "
           f"({local_only.item()/ref.item():.2f}x)", rank=0)
    rprint(f"  CORRECT global histogram        {cp_global.item():.4f}", rank=0)

    assert abs(cp_global.item() - ref.item()) < 1e-4
    assert local_only.item() > 1.5 * ref.item()

    # The wrong value is bad; the spurious gradient is worse. With a uniform
    # global histogram f_e = 1/E the aux loss collapses to sum_e P_e = 1, a
    # constant -- so the correct gradient is exactly zero. Nothing to fix.
    g_ref = torch.autograd.grad(ref, logits, retain_graph=True)[0][lo:hi]
    g_bad = torch.autograd.grad(local_only, logits, retain_graph=True)[0][lo:hi]
    own = g_bad[:, rank].mean().item()
    rprint(f"\n  |grad| w.r.t. router logits: reference {g_ref.norm():.2e}  "
           f"local-only {g_bad.norm():.3f}", rank=0)
    rprint(f"  d(local aux)/d(logit for my own expert {rank}) = {own:+.4f} > 0", rank=0)

    assert g_ref.norm().item() < 1e-6 < g_bad.norm().item()
    assert own > 0

    rprint("\n  The reference gradient is zero because the router is already", rank=0)
    rprint("  balanced. The local-only version instead sees 'all my tokens went", rank=0)
    rprint("  to one expert', and its gradient is POSITIVE on that expert's", rank=0)
    rprint("  logit -- i.e. it pushes each rank away from the assignment that is", rank=0)
    rprint("  globally optimal. The aux loss fights the thing it exists to", rank=0)
    rprint("  encourage, and you see it as a load-balance metric that never", rank=0)
    rprint("  converges. Same fix for the z-loss and for any capacity or", rank=0)
    rprint("  drop-rate computed from a per-rank token count.", rank=0)


# ---- part 4: pooling heads ------------------------------------------------


def part4_pooling(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 4  pooling heads (reward model / classifier / embedding)", rank=0)
    rprint("=" * 74, rank=0)

    h, _, _, mask = make_batch(cp, device)
    S = h.shape[0]
    last_global = int(mask.nonzero()[-1].item())
    ref_last = h[last_global]
    ref_mean = (h * mask.unsqueeze(-1)).sum(0) / mask.sum()

    h_l, mask_l = shard(h, cp, rank), shard(mask, cp, rank)
    owner = last_global // LOCAL_S
    rprint(f"  S={S}, last valid token is global index {last_global} "
           f"-> lives only on rank {owner}", rank=0)

    # WRONG: every rank pools its own shard and trains on that.
    if mask_l.sum() > 0:
        wrong_last = h_l[int(mask_l.nonzero()[-1].item())]
        err = (wrong_last - ref_last).abs().max().item()
    else:
        err = float("inf")   # rank 0: no valid token at all -> empty index
    errs = [torch.zeros(1, device=device) for _ in range(cp)]
    dist.all_gather(errs, torch.tensor([err], device=device))
    rprint(f"  WRONG per-shard last token, max|err| per rank: "
           f"{[round(e.item(), 3) for e in errs]}", rank=0)

    # CORRECT: one-hot select locally, then a differentiable sum-all-reduce.
    # Only the owning rank contributes; the others contribute exact zeros.
    sel = torch.zeros(LOCAL_S, device=device)
    if rank == owner:
        sel[last_global - rank * LOCAL_S] = 1.0
    good_last = all_reduce_sum(sel @ h_l)

    # Mean pooling: reduce numerator and denominator, exactly as in part 1.
    num = all_reduce_sum((h_l * mask_l.unsqueeze(-1)).sum(0))
    den = reduce_scalar(mask_l.sum().item(), device)
    good_mean = num / den

    assert_close(good_last, ref_last, name="CP last-token pool")
    assert_close(good_mean, ref_mean, name="CP mean pool")
    assert errs[owner].item() < 1e-6 and max(e.item() for e in errs) > 1e-3
    rprint("  CORRECT one-hot + all-reduce last-token pool  matches reference", rank=0)
    rprint("  CORRECT masked-sum / count mean pool          matches reference", rank=0)

    rprint("\n  This is the RLHF reward-model bug (see ch.15): the scalar score", rank=0)
    rprint("  comes from the final token, so under CP every non-owning rank", rank=0)
    rprint("  scores a token from the middle of the sequence -- or crashes on an", rank=0)
    rprint("  empty index when its whole shard is padding.", rank=0)


# ---- part 5: metrics ------------------------------------------------------


def part5_metrics(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 5  metrics: perplexity and token accuracy", rank=0)
    rprint("=" * 74, rank=0)

    h, W, labels, mask = make_batch(cp, device)
    nll = token_nll(h, W, labels)
    correct = ((h @ W).argmax(-1) == labels).float()
    ppl_ref = torch.exp((nll * mask).sum() / mask.sum()).item()
    acc_ref = ((correct * mask).sum() / mask.sum()).item()

    h_l, lab_l, mask_l = shard(h, cp, rank), shard(labels, cp, rank), shard(mask, cp, rank)
    nll_l = token_nll(h_l, W, lab_l)
    corr_l = ((h_l @ W).argmax(-1) == lab_l).float()

    # WRONG: mean of per-rank means, skipping empty shards (and for ppl,
    # exponentiating on each rank *before* the reduction).
    live = mask_l.sum() > 0
    n = mask_l.sum().clamp(min=1)
    n_live = reduce_scalar(1.0 if live else 0.0, device)
    ppl_bad = reduce_scalar(
        torch.exp((nll_l * mask_l).sum() / n).item() if live else 0.0, device) / n_live
    acc_bad = reduce_scalar(
        ((corr_l * mask_l).sum() / n).item() if live else 0.0, device) / n_live

    # CORRECT: token-weighted, and exponentiate only after reducing.
    den = reduce_scalar(mask_l.sum().item(), device)
    ppl_good = torch.exp(torch.tensor(
        reduce_scalar((nll_l * mask_l).sum().item(), device) / den)).item()
    acc_good = reduce_scalar((corr_l * mask_l).sum().item(), device) / den

    rprint(f"  perplexity     ref {ppl_ref:8.3f}   WRONG mean-of-exp "
           f"{ppl_bad:8.3f}   CORRECT {ppl_good:8.3f}", rank=0)
    rprint(f"  token accuracy ref {acc_ref:8.3f}   WRONG mean-of-means "
           f"{acc_bad:8.3f}   CORRECT {acc_good:8.3f}", rank=0)

    assert abs(ppl_good - ppl_ref) < 1e-2 and abs(acc_good - acc_ref) < 1e-5
    assert abs(ppl_bad / ppl_ref - 1) > 0.05 and abs(acc_bad / acc_ref - 1) > 0.05
    rprint("\n  Two errors stack here. The rank weighting is wrong (part 1), and", rank=0)
    rprint("  exp() is convex so averaging local perplexities is biased upward by", rank=0)
    rprint("  Jensen. Reduce the log-loss first, exponentiate exactly once.", rank=0)
    rprint("  The same holds for every per-domain / per-task loss you log.", rank=0)


# ---- part 6: grad-norm clipping ------------------------------------------


def part6_grad_norm(cp, rank, device):
    rprint("\n" + "=" * 74, rank=0)
    rprint("PART 6  grad-norm clipping across replica groups", rank=0)
    rprint("=" * 74, rank=0)

    h, W, labels, mask = make_batch(cp, device)
    W_l = W.clone().requires_grad_(True)
    h_l, lab_l, mask_l = shard(h, cp, rank), shard(labels, cp, rank), shard(mask, cp, rank)
    ((token_nll(h_l, W_l, lab_l) * mask_l).sum()
     / reduce_scalar(mask_l.sum().item(), device)).backward()

    g = W_l.grad.clone()
    dist.all_reduce(g, op=dist.ReduceOp.SUM)     # after this, every rank holds
    true_norm = g.norm().item()                  # the SAME full gradient

    # WRONG: the usual sharded-parameter recipe, applied to a replicated param.
    bad_norm = reduce_scalar(g.pow(2).sum().item(), device) ** 0.5

    clip = 1.0
    rprint(f"  W is REPLICATED across the CP group -- after the grad reduce every", rank=0)
    rprint(f"  rank already holds the complete gradient.", rank=0)
    rprint(f"  correct |g|                      {true_norm:.4f}", rank=0)
    rprint(f"  WRONG sum |g|^2 over CP then sqrt {bad_norm:.4f}  "
           f"(= sqrt(CP) x = {cp ** 0.5:.3f}x)", rank=0)
    rprint(f"  clip coefficient at max_norm={clip}: correct "
           f"{min(1.0, clip / true_norm):.4f}  vs wrong "
           f"{min(1.0, clip / bad_norm):.4f}", rank=0)

    assert abs(bad_norm / true_norm - cp ** 0.5) < 1e-3
    rprint("\n  Inflating the norm by sqrt(CP) makes the clip fire earlier and", rank=0)
    rprint("  scales every update down -- a silent LR cut that grad_norm logging", rank=0)
    rprint("  will not reveal, because the logged norm is the inflated one.", rank=0)
    rprint("  Rule: reduce |g|^2 over groups the parameter is SHARDED across", rank=0)
    rprint("  (ZeRO/FSDP shards, TP column/row shards, EP experts) and never over", rank=0)
    rprint("  groups it is REPLICATED across (CP, DP-replica, TP-duplicated LN).", rank=0)


def main():
    rank, world, device = setup()
    if world < 4:
        rprint("This demo needs >= 4 ranks to build a skewed mask: "
               "torchrun --nproc-per-node=4 ...")
        cleanup()
        return
    cp = world

    part1_loss_denominator(cp, rank, device)
    part2_gradient_scale(cp, rank, device)
    part3_moe_aux_loss(cp, rank, device)
    part4_pooling(cp, rank, device)
    part5_metrics(cp, rank, device)
    part6_grad_norm(cp, rank, device)

    rprint("\n" + "=" * 74, rank=0)
    rprint("all cross-token reductions match the single-rank reference", rank=0)
    rprint("=" * 74, rank=0)
    cleanup()


if __name__ == "__main__":
    main()
