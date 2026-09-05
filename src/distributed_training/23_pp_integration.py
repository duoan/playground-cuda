"""Chapter 6: wiring a pipeline into a real training step.

Run:
    torchrun --nproc-per-node=4 src/distributed_training/23_pp_integration.py

21_pp_from_scratch.py proves the schedules are correct in isolation. Everything
that breaks when you put one into an actual trainer lives here, and none of it
appears in a schedule diagram:

    1. rank layout        - who are my pipeline neighbours, and why is TP innermost
    2. partitioning       - equal layer counts give unequal stage times
    3. weight tying       - embedding and lm_head land on different stages
    4. loss scale + DP    - one gradient all-reduce per step, not per micro-batch
    5. data routing       - stage 0 needs tokens, the last stage needs labels

Verified end to end: PP x DP with tied weights must reproduce a single-process
run over the concatenated global batch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F_

from common import assert_close, cleanup, rprint, setup
from common.pipeline import PipelineEngine, sched_1f1b

HID, VOCAB, MICRO_B = 16, 32, 4
N_MICRO = 4
LAYERS_PER_STAGE = 2


# ---- part 1: rank layout --------------------------------------------------


def coords(rank: int, tp: int, dp: int, pp: int) -> tuple[int, int, int]:
    """Megatron's default ordering: TP varies fastest, PP slowest.

        global_rank = pp_rank * (dp * tp) + dp_rank * tp + tp_rank

    The ordering is not arbitrary. TP exchanges a full activation twice per
    layer, so its group must sit inside one NVLink domain -- which means
    consecutive global ranks. PP only ships one activation per stage boundary,
    so it gets the widest stride and is the one allowed to cross nodes.
    """
    return rank % tp, (rank // tp) % dp, rank // (tp * dp)


def part1_layout(rank, world):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 1  rank layout: finding your neighbours", rank=0)
    rprint("=" * 76, rank=0)

    tp, dp, pp = 2, 2, 2
    demo_world = tp * dp * pp
    rprint(f"  world={demo_world}, TP={tp} DP={dp} PP={pp}, 4 GPUs per node\n", rank=0)
    rprint(f"  {'rank':>5}{'tp':>4}{'dp':>4}{'pp':>4}{'node':>6}   groups it belongs to",
           rank=0)
    for r in range(demo_world):
        t, d, p = coords(r, tp, dp, pp)
        tp_grp = [x for x in range(demo_world) if coords(x, tp, dp, pp)[1:] == (d, p)]
        pp_grp = [x for x in range(demo_world) if coords(x, tp, dp, pp)[:2] == (t, d)]
        dp_grp = [x for x in range(demo_world)
                  if (coords(x, tp, dp, pp)[0], coords(x, tp, dp, pp)[2]) == (t, p)]
        rprint(f"  {r:>5}{t:>4}{d:>4}{p:>4}{r // 4:>6}   "
               f"TP{tp_grp} PP{pp_grp} DP{dp_grp}", rank=0)

    rprint("\n  Read the TP column: TP groups are always CONSECUTIVE ranks, so they", rank=0)
    rprint("  never straddle a node. PP groups have stride DP*TP = 4, so they are", rank=0)
    rprint("  the ones that cross the node boundary -- which is what you want,", rank=0)
    rprint("  since a stage boundary ships one activation per micro-batch while a", rank=0)
    rprint("  TP layer ships two per layer.", rank=0)
    rprint("\n  Your pipeline neighbours are rank -+ (DP*TP), not rank -+ 1. Getting", rank=0)
    rprint("  this wrong is the single most common bug when hand-rolling PP: with", rank=0)
    rprint("  DP*TP > 1 you end up sending activations to a rank that holds the", rank=0)
    rprint("  same layers you do, and it hangs rather than erroring.", rank=0)

    # Build the real groups for THIS world and confirm membership.
    tp_r, dp_r, pp_r = 1, world // 2, 2
    my = coords(rank, tp_r, dp_r, pp_r)
    pp_groups = {}
    for d in range(dp_r):
        members = [x for x in range(world) if coords(x, tp_r, dp_r, pp_r)[:2] == (0, d)]
        pp_groups[d] = (dist.new_group(members), members)
    grp, members = pp_groups[my[1]]
    rprint(f"\n  this job: world={world} TP=1 DP={dp_r} PP={pp_r}; rank {rank} is "
           f"pp={my[2]} dp={my[1]}, pipeline group {members}", rank=0)
    assert dist.get_world_size(grp) == pp_r
    return grp, my


# ---- part 2: partitioning -------------------------------------------------


def part2_partition(rank, world):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 2  equal layer counts are not equal work", rank=0)
    rprint("=" * 76, rank=0)

    # Relative cost per module, one transformer block = 1.0.
    # 32 layers, h=4096, vocab=128K: the head GEMM alone is h*V / 12h^2 = 2.6
    # blocks, and the cross-entropy over a (B, S, 128K) logit tensor adds more.
    # The embedding is a cheap lookup by comparison.
    costs = [0.5] + [1.0] * 32 + [5.0]        # embedding, blocks, lm_head
    names = ["embed"] + [f"blk{i}" for i in range(32)] + ["lm_head"]
    P = 4

    def report(label, bounds):
        loads = [sum(costs[bounds[i]:bounds[i + 1]]) for i in range(P)]
        mean = sum(loads) / P
        rprint(f"  {label:<22}{[round(x, 1) for x in loads]}  "
               f"max/mean = {max(loads)/mean:.2f}x", rank=0)
        return max(loads) / mean

    n = len(costs)
    uniform = [round(i * n / P) for i in range(P)] + [n]
    slow = report("by layer count", uniform)

    # Greedy balance on cumulative cost -- what --custom-pipeline-partitioning
    # or Megatron's uneven-stage flags let you express by hand.
    total, target = sum(costs), sum(costs) / P
    bounds, acc = [0], 0.0
    for i, c in enumerate(costs):
        acc += c
        if acc >= target * len(bounds) and len(bounds) < P:
            bounds.append(i + 1)
    bounds.append(n)
    fast = report("by measured cost", bounds)

    rprint(f"\n  Stage layout by cost: " +
           ", ".join(f"{names[bounds[i]]}..{names[bounds[i+1]-1]}" for i in range(P)),
           rank=0)
    rprint(f"\n  A pipeline runs at the speed of its slowest stage, so the {slow:.2f}x", rank=0)
    rprint(f"  imbalance is a {slow:.2f}x slowdown on EVERY micro-batch -- it multiplies", rank=0)
    rprint("  with the bubble rather than hiding inside it.", rank=0)
    rprint("  The embedding and the lm_head are the culprits: a 128K-vocab head", rank=0)
    rprint("  produces a (B, S, V) logit tensor, which dwarfs a transformer block", rank=0)
    rprint("  in both compute and activation memory. Standard fixes are to give", rank=0)
    rprint("  the first and last stages fewer blocks, or to put the loss on its", rank=0)
    rprint("  own virtual chunk.", rank=0)
    assert slow > fast
    rprint("\n  Note you cannot build the full model and then slice it: at 70B+ no", rank=0)
    rprint("  single rank can hold it. Each rank constructs only its own layers,", rank=0)
    rprint("  which means the partition has to be decided BEFORE any weights are", rank=0)
    rprint("  allocated -- from a cost table, not from a profile of the real run.", rank=0)


# ---- part 3-5: tied weights, loss scaling, DP sync ------------------------


class Embed(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight          # (VOCAB, HID), shared with the head

    def forward(self, tok):
        return F_.embedding(tok, self.weight)


class Blocks(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(HID, HID, bias=False) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            x = F_.gelu(l(x))
        return x


class Head(nn.Module):
    """lm_head tied to the embedding table."""

    def __init__(self, weight, n_blocks):
        super().__init__()
        self.blocks = Blocks(n_blocks)
        self.weight = weight

    def forward(self, x):
        return self.blocks(x) @ self.weight.t()


class FirstStage(nn.Module):
    def __init__(self, weight, n_blocks):
        super().__init__()
        self.embed = Embed(weight)
        self.blocks = Blocks(n_blocks)

    def forward(self, tok):
        return self.blocks(self.embed(tok.long()))


def build_weights(device):
    torch.manual_seed(4242)
    emb = nn.Parameter(torch.randn(VOCAB, HID, device=device) * 0.05)
    blocks = [nn.Parameter(torch.randn(HID, HID, device=device) * 0.1)
              for _ in range(2 * LAYERS_PER_STAGE)]
    return emb, blocks


def make_batch(dp_rank: int, device):
    """Every DP replica gets DIFFERENT data; a PP group shares it."""
    g = torch.Generator().manual_seed(1000 + dp_rank)
    toks = [torch.randint(0, VOCAB, (MICRO_B,), generator=g) for _ in range(N_MICRO)]
    labs = [torch.randint(0, VOCAB, (MICRO_B,), generator=g) for _ in range(N_MICRO)]
    return [t.to(device) for t in toks], [l.to(device) for l in labs]


def part345(rank, world, pp_group, my, device):
    rprint("\n" + "=" * 76, rank=0)
    rprint("PART 3-5  tied weights, loss scaling, and one DP all-reduce", rank=0)
    rprint("=" * 76, rank=0)

    tp_r, dp_r, pp_r = 1, world // 2, 2
    _, dp_rank, pp_rank = my

    dp_groups = {p: dist.new_group([x for x in range(world)
                                    if coords(x, tp_r, dp_r, pp_r)[2] == p])
                 for p in range(pp_r)}
    # The tied embedding lives on the first AND last stage, so those two ranks
    # form their own group. Megatron calls it the embedding group.
    emb_groups = {d: dist.new_group([x for x in range(world)
                                     if coords(x, tp_r, dp_r, pp_r)[1] == d
                                     and coords(x, tp_r, dp_r, pp_r)[2] in (0, pp_r - 1)])
                  for d in range(dp_r)}

    emb_w, blk_w = build_weights(device)
    toks, labs = make_batch(dp_rank, device)

    # -- build only this rank's stage --
    emb_local = nn.Parameter(emb_w.detach().clone())
    if pp_rank == 0:
        stage = FirstStage(emb_local, LAYERS_PER_STAGE).to(device)
        for i, l in enumerate(stage.blocks.layers):
            with torch.no_grad():
                l.weight.copy_(blk_w[i])
    else:
        stage = Head(emb_local, LAYERS_PER_STAGE).to(device)
        for i, l in enumerate(stage.blocks.layers):
            with torch.no_grad():
                l.weight.copy_(blk_w[LAYERS_PER_STAGE + i])

    def loss_fn(logits, target):
        return F_.cross_entropy(logits, target.long())

    # Part 5: route the data. Stage 0 consumes tokens and never sees a label;
    # the last stage consumes labels and never sees a token. Handing both to
    # every stage is harmless here but wastes host-to-device bandwidth at scale,
    # and hides bugs where a stage silently uses the wrong tensor.
    inputs = toks if pp_rank == 0 else [None] * N_MICRO
    labels = labs if pp_rank == pp_r - 1 else [None] * N_MICRO

    engine = PipelineEngine([stage], pp_rank, pp_r, N_MICRO, loss_fn=loss_fn,
                            group=pp_group, device=device,
                            act_shape=(MICRO_B, HID))
    engine.run(sched_1f1b(pp_rank, pp_r, N_MICRO), inputs, labels)

    # Part 3: the tied gradient. Each end computed only its own half.
    local_emb_grad = emb_local.grad.clone()
    dist.all_reduce(emb_local.grad, op=dist.ReduceOp.SUM, group=emb_groups[dp_rank])
    tied_after_emb_group = emb_local.grad.clone()

    # Part 4: ONE data-parallel all-reduce, after all m micro-batches. The
    # engine already divided each micro-batch loss by m, so averaging over the
    # DP group turns the accumulated gradient into the global-batch mean.
    params = [emb_local] + [l.weight for l in stage.blocks.layers]
    for p in params:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_groups[pp_rank])
        p.grad /= dp_r

    # -- reference: one process, tied parameter, the whole global batch --
    ref_emb = nn.Parameter(emb_w.detach().clone())
    ref_blocks = [nn.Parameter(w.detach().clone()) for w in blk_w]
    for d in range(dp_r):
        t_d, l_d = make_batch(d, device)
        for i in range(N_MICRO):
            h = F_.embedding(t_d[i].long(), ref_emb)
            for w in ref_blocks:
                h = F_.gelu(h @ w.t())
            logits = h @ ref_emb.t()
            (F_.cross_entropy(logits, l_d[i].long()) / (N_MICRO * dp_r)).backward()

    assert_close(emb_local.grad, ref_emb.grad, rtol=1e-4, atol=1e-6,
                 name="tied embedding grad")
    off = 0 if pp_rank == 0 else LAYERS_PER_STAGE
    for i, l in enumerate(stage.blocks.layers):
        assert_close(l.weight.grad, ref_blocks[off + i].grad, rtol=1e-4, atol=1e-6,
                     name=f"block grad pp={pp_rank} i={i}")

    # Grad-norm clipping counts the tied parameter twice unless you tell it not
    # to: both ends hold an identical, fully-reduced copy, so summing local
    # squared norms over the pipeline inflates the total.
    def global_sq(include_emb_here: bool) -> float:
        local = sum((l.weight.grad ** 2).sum() for l in stage.blocks.layers)
        if include_emb_here:
            local = local + (emb_local.grad ** 2).sum()
        t = local.clone().detach()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pp_group)
        return t.item()

    sq_naive = global_sq(True)                       # every end counts it
    sq_right = global_sq(pp_rank == 0)               # only one end counts it
    inflation = (sq_naive / sq_right) ** 0.5

    share = (local_emb_grad.norm() / tied_after_emb_group.norm()).item()
    rprint(f"  tied embedding: this stage produced {share:.0%} of the tied gradient", rank=0)
    rprint("  (stage 0 gets it from the lookup, the last stage from the lm_head)", rank=0)
    rprint("  after the embedding-group all-reduce, both ends hold 100% of it", rank=0)
    rprint(f"\n  all-reduce count per step: 1 embedding-group + 1 DP per tensor,", rank=0)
    rprint(f"  NOT one per micro-batch (that would be {N_MICRO}x the traffic)", rank=0)
    rprint("\n  tied embedding grad and every block grad match a single-process run", rank=0)
    rprint(f"  over the full global batch of DP x m = {dp_r} x {N_MICRO} micro-batches",
           rank=0)

    rprint("\n  Why the embedding group is mandatory: the two ends each see half the", rank=0)
    rprint("  gradient (lookup vs projection). Skip the all-reduce and the two", rank=0)
    rprint("  copies drift apart within a few steps, so the model is silently", rank=0)
    rprint("  using one table to encode and a different one to decode. Nothing", rank=0)
    rprint("  crashes; the loss just stops improving.", rank=0)
    rprint("\n  Order of the two reductions does NOT matter: a SUM over the", rank=0)
    rprint("  embedding group and an AVERAGE over the DP group act on disjoint", rank=0)
    rprint("  rank sets and both are linear, so they commute. What does matter is", rank=0)
    rprint("  that both finish before clipping.", rank=0)
    rprint(f"\n  And clipping has its own trap: grad-norm over the pipeline group", rank=0)
    rprint(f"  double-counts the tied table, because after the all-reduce BOTH ends", rank=0)
    rprint(f"  hold the same full copy. Measured inflation here: {inflation:.3f}x", rank=0)
    rprint("  (sqrt(2) in the worst case, when the embedding dominates the norm).", rank=0)
    rprint("  An inflated norm means clipping engages earlier than you asked, so", rank=0)
    rprint("  the effective learning rate quietly drops. Count each replicated", rank=0)
    rprint("  parameter on exactly one rank -- the same rule as for CP in ch. 7.", rank=0)
    assert inflation > 1.0


def main():
    rank, world, device = setup(backend="gloo")
    if world != 4:
        rprint("this demo is written for exactly 4 ranks: torchrun --nproc-per-node=4 ...")
        cleanup()
        return

    pp_group, my = part1_layout(rank, world)
    part2_partition(rank, world)
    part345(rank, world, pp_group, my, device)

    rprint("\n" + "=" * 76, rank=0)
    rprint("PP x DP with tied weights matches the single-process global batch", rank=0)
    rprint("=" * 76, rank=0)
    cleanup()


if __name__ == "__main__":
    main()
