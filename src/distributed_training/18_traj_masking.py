"""Chapter 15: multi-turn trajectory loss masking, advantages, and ratio drift.

Run:
    python3 src/distributed_training/18_traj_masking.py

Four things that are easy to get wrong when you move from single-turn RLHF to
multi-turn agentic RL, each demonstrated with real tensors:

  1. loss masking       -- training on tool-observation tokens
  2. normalisation      -- dividing by mask.numel() instead of mask.sum()
  3. GRPO degeneracy    -- all-correct / all-wrong groups give zero gradient
  4. logprob mismatch   -- rollout engine vs training engine disagree, which
                           silently clips the first inner epoch

Bug 1 is the important one: it does not crash, it does not spike the loss, and
the gradient norm looks perfectly healthy. The only way to see it is that the
agent gets worse at calling tools.

CPU-only, deterministic, self-checking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB = 256
HIDDEN = 64

# Role ids for every position in a trajectory.
PROMPT, ASSISTANT, OBSERVATION = 0, 1, 2
ROLE_NAME = {PROMPT: "prompt", ASSISTANT: "assistant", OBSERVATION: "observation"}


# ----------------------------------------------------------------------------
# A trajectory: prompt, then alternating assistant / observation turns
# ----------------------------------------------------------------------------


# How many tokens a tool typically returns. The spread across real tools is
# enormous, and that spread is what makes the normalisation bug (demo 2) bite.
TOOL_PROFILES = {
    "calculator": (5, 30),        # a number and a unit
    "code_search": (150, 600),    # a handful of matching hunks
    "web_page": (1200, 3500),     # a scraped page or a full pytest trace
}


def make_trajectory(n_turns: int, gen: torch.Generator, tool: str = "code_search",
                    prompt_len: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens, roles) for one agentic episode using a given tool."""
    lo, hi = TOOL_PROFILES[tool]
    tokens, roles = [], []

    def emit(n: int, role: int) -> None:
        tokens.extend(torch.randint(0, VOCAB, (n,), generator=gen).tolist())
        roles.extend([role] * n)

    emit(prompt_len, PROMPT)
    for _ in range(n_turns):
        emit(int(torch.randint(15, 40, (1,), generator=gen).item()), ASSISTANT)
        emit(int(torch.randint(lo, hi, (1,), generator=gen).item()), OBSERVATION)
    emit(int(torch.randint(15, 40, (1,), generator=gen).item()), ASSISTANT)

    return torch.tensor(tokens), torch.tensor(roles)


def loss_mask_from_roles(roles: torch.Tensor) -> torch.Tensor:
    """1 on tokens the policy generated, 0 everywhere else.

    This is the whole fix for bug #1. Prompt tokens were written by the user
    and observation tokens were written by the environment; neither was
    sampled from the policy, so neither belongs in a policy-gradient loss.
    """
    return (roles == ASSISTANT).float()


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.body = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.head = nn.Linear(HIDDEN, VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.emb(tokens)))


def token_logps(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """log pi(x_t | x_<t) aligned so index t is the logprob of tokens[t]."""
    logits = model(tokens[:-1])
    return torch.log_softmax(logits, dim=-1).gather(
        -1, tokens[1:].unsqueeze(-1)).squeeze(-1)


def flat_grad(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                      if p.grad is not None])


# ----------------------------------------------------------------------------
# 1. Loss masking
# ----------------------------------------------------------------------------


def demo_masking() -> None:
    print("=" * 100)
    print("1. Loss masking — the bug that does not crash")
    print("=" * 100)

    gen = torch.Generator().manual_seed(7)
    tokens, roles = make_trajectory(n_turns=6, gen=gen)
    mask = loss_mask_from_roles(roles)[1:]  # align with next-token targets

    counts = {r: int((roles == r).sum()) for r in (PROMPT, ASSISTANT, OBSERVATION)}
    total = sum(counts.values())
    print(f"  one 6-turn trajectory, {total} tokens:")
    for r, c in counts.items():
        print(f"    {ROLE_NAME[r]:<12} {c:>5}  ({c / total:>5.1%})")
    obs_frac = counts[OBSERVATION] / total
    print(f"\n  -> {obs_frac:.0%} of the tokens were written by the environment.")
    print("     Training on them means most of the gradient is spent teaching the")
    print("     model to predict tool output it cannot possibly know.")

    model = TinyPolicy()
    advantage = 1.0  # pretend this trajectory succeeded

    # WRONG: every token contributes, observations included.
    model.zero_grad()
    lp = token_logps(model, tokens)
    (-(advantage * lp).mean()).backward()
    g_wrong = flat_grad(model).clone()

    # RIGHT: only assistant tokens contribute.
    model.zero_grad()
    lp = token_logps(model, tokens)
    (-(advantage * lp * mask).sum() / mask.sum()).backward()
    g_right = flat_grad(model).clone()

    cos = F.cosine_similarity(g_wrong, g_right, dim=0).item()
    print(f"\n  gradient norm  unmasked={g_wrong.norm():.4f}   masked={g_right.norm():.4f}")
    print(f"  cosine similarity between the two gradients: {cos:.3f}")
    print("\n  Both norms are perfectly ordinary -- nothing here would trip a")
    print("  gradient-norm alarm -- yet the two updates point in substantially")
    print("  different directions. This is why the bug survives to production:")
    print("  loss falls, grad_norm is healthy, and only the tool-call success")
    print("  rate quietly degrades.")

    assert cos < 0.9, "masked and unmasked gradients should differ materially"
    assert obs_frac > 0.5, "observations should dominate an agentic trajectory"


# ----------------------------------------------------------------------------
# 2. Normalisation denominator
# ----------------------------------------------------------------------------


def demo_normalisation() -> None:
    print("\n" + "=" * 100)
    print("2. Normalisation — mask.sum() vs mask.numel()")
    print("=" * 100)

    gen = torch.Generator().manual_seed(11)
    model = TinyPolicy()

    print("  Same 5-turn task, same assistant behaviour, three different tools:\n")
    print(f"  {'tool':>12} {'tokens':>8} {'assistant':>10} {'asst %':>8}"
          f" {'/numel':>9} {'/sum':>9} {'effective LR':>13}")
    print("  " + "-" * 76)

    by_numel_vals, by_sum_vals = [], []
    for tool in TOOL_PROFILES:
        tokens, roles = make_trajectory(5, gen, tool=tool)
        mask = loss_mask_from_roles(roles)[1:]
        lp = token_logps(model, tokens)

        by_numel = ((-lp * mask).sum() / mask.numel()).item()
        by_sum = ((-lp * mask).sum() / mask.sum()).item()
        by_numel_vals.append(by_numel)
        by_sum_vals.append(by_sum)
        print(f"  {tool:>12} {len(tokens):>8} {int(mask.sum()):>10}"
              f" {mask.mean():>7.1%} {by_numel:>9.4f} {by_sum:>9.4f}"
              f" {by_numel / max(by_numel_vals):>12.2f}x")

    spread_numel = max(by_numel_vals) / min(by_numel_vals)
    spread_sum = max(by_sum_vals) / min(by_sum_vals)
    print(f"\n  spread across the three: /numel = {spread_numel:.1f}x,"
          f"  /sum = {spread_sum:.2f}x")
    print("\n  The assistant did the same amount of work in all three cases, so the")
    print("  loss should be about the same -- and with /sum it is. With /numel it")
    print(f"  varies {spread_numel:.0f}x purely because the web_page tool is chatty.")
    print("  You have handed the environment a hidden per-trajectory multiplier on")
    print("  your learning rate: verbose tools get down-weighted into irrelevance,")
    print("  terse ones dominate the batch. Nothing in the loss curve reveals this.")

    assert spread_numel > 3.0, "the numel bug should swing wildly with tool verbosity"
    assert spread_sum < 1.5, "the correct denominator should be roughly stable"


# ----------------------------------------------------------------------------
# 3. GRPO group degeneracy
# ----------------------------------------------------------------------------


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def demo_grpo_degeneracy() -> None:
    print("\n" + "=" * 100)
    print("3. GRPO with binary rewards — groups that produce no gradient")
    print("=" * 100)

    n, group_n = 400, 16
    gen = torch.Generator().manual_seed(3)
    # Per-task success probability: many tasks are hopeless, many are trivial.
    p = torch.rand(n, generator=gen) ** 2
    p = torch.where(torch.rand(n, generator=gen) < 0.35, torch.ones_like(p) * 0.97, p)
    rewards = (torch.rand(n, group_n, generator=gen) < p[:, None]).float()

    acc = rewards.mean(dim=1)
    all_wrong = int((acc == 0).sum())
    all_right = int((acc == 1).sum())
    dead = all_wrong + all_right

    print(f"  {n} groups x {group_n} responses, binary reward\n")
    print(f"    all wrong (acc=0) : {all_wrong:>4}  ({all_wrong / n:>5.1%})")
    print(f"    all right (acc=1) : {all_right:>4}  ({all_right / n:>5.1%})")
    print(f"    informative       : {n - dead:>4}  ({(n - dead) / n:>5.1%})")

    adv = torch.stack([grpo_advantages(r) for r in rewards])
    per_group = adv.abs().sum(dim=1)
    zero_groups = int((per_group < 1e-3).sum())

    print(f"\n  groups whose advantages are all ~0: {zero_groups} "
          f"(= the {dead} degenerate ones)")
    print(f"  -> {dead / n:.0%} of rollout compute produced no gradient at all.")

    # DAPO-style dynamic sampling: keep only groups with 0 < acc < 1.
    keep = (acc > 0) & (acc < 1)
    print(f"\n  DAPO dynamic sampling keeps {int(keep.sum())}/{n} groups. To fill a")
    print(f"  batch of B informative groups you must roll out about "
          f"B/{keep.float().mean():.2f} = {1 / keep.float().mean():.2f}B groups.")
    print("\n  The systems consequence is the part people miss: the rollout volume")
    print("  per iteration is no longer a constant. The scheduler has to keep")
    print("  sampling until the batch is full, so capacity planning shifts from")
    print("  'generate N groups' to 'generate until N survive the filter'.")

    assert zero_groups == dead
    assert 0 < dead < n, "expected a mix of degenerate and informative groups"


# ----------------------------------------------------------------------------
# 4. Rollout / training logprob mismatch
# ----------------------------------------------------------------------------


def demo_logprob_mismatch() -> None:
    print("\n" + "=" * 100)
    print("4. Rollout vs training logprobs — phantom clipping in epoch 0")
    print("=" * 100)

    gen = torch.Generator().manual_seed(5)
    tokens, roles = make_trajectory(n_turns=60, gen=gen, tool="calculator")
    mask = loss_mask_from_roles(roles)[1:].bool()

    model = TinyPolicy()
    with torch.no_grad():
        lp_train = token_logps(model, tokens)[mask]

    clip_eps = 0.2
    print(f"  {int(mask.sum())} assistant tokens, PPO clip range +-{clip_eps}")
    print("\n  The disagreement between two engines is not Gaussian. Almost every")
    print("  token matches closely; a few -- typically low-probability ones, where")
    print("  a small logit difference moves the log-softmax a lot -- are far off.")
    print("  Modelled here as 95% bulk + 5% tail at 10x the scale.\n")

    print(f"  {'engine skew':>13} {'mean|dlogp|':>12} {'p99|dlogp|':>11}"
          f" {'clipped frac':>13} {'max ratio':>10}")
    print("  " + "-" * 64)

    # Draw the disagreement pattern ONCE at unit scale, then scale it. Each row
    # is then the same tokens disagreeing by the same relative amount, so the
    # rows are directly comparable instead of being three separate samples.
    unit = torch.randn(lp_train.shape, generator=gen)
    is_tail = torch.rand(lp_train.shape, generator=gen) < 0.05
    unit = torch.where(is_tail, unit * 10, unit)

    fracs = {}
    for skew in (0.0, 0.01, 0.03, 0.08):
        noise = unit * skew
        lp_rollout = lp_train + noise
        ratio = torch.exp(lp_train - lp_rollout)          # epoch 0: should be 1
        clipped = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
        fracs[skew] = clipped.item()

        d = noise.abs()
        p99 = d.sort().values[int(0.99 * len(d))]
        label = "exact" if skew == 0 else f"sigma={skew}"
        print(f"  {label:>13} {d.mean():>12.4f} {p99:>11.4f} {clipped:>12.1%}"
              f" {ratio.max():>10.3f}")

    print("\n  At the first inner epoch the policy has not been updated yet, so")
    print("  every ratio should be exactly 1 and nothing should clip. Whatever")
    print("  clipping you see is pure numerical disagreement, and it silently")
    print("  deletes the gradient of precisely the tokens the two engines")
    print("  disagree about -- the rare, low-probability, exploratory ones.")
    print("\n  Note how small the mean disagreement is when clipping starts: the")
    print("  bulk statistic stays tiny while the tail does all the damage. If you")
    print("  monitor only mean |dlogp| you will conclude everything is fine.")
    print("\n  Two things to do about it:")
    print("    - recompute log pi_old with the TRAINING engine (one extra")
    print("      forward) so the ratio is self-consistent by construction;")
    print("    - monitor the epoch-0 clipped fraction, which should be ~0. It is")
    print("      the only number that catches this directly.")

    assert fracs[0.0] == 0.0, "identical logprobs must never clip"
    assert fracs[0.01] <= fracs[0.03] <= fracs[0.08], \
        "phantom clipping must grow monotonically with engine disagreement"
    assert fracs[0.08] > 0.02, "a realistic skew should produce visible clipping"


def main() -> None:
    demo_masking()
    demo_normalisation()
    demo_grpo_degeneracy()
    demo_logprob_mismatch()
    print("\n" + "=" * 100)
    print("all checks passed")
    print("=" * 100)


if __name__ == "__main__":
    main()
