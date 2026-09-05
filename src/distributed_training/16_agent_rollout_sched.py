"""Chapter 15: agentic RL rollout scheduling — sync vs async vs partial rollout.

Run:
    python3 src/distributed_training/16_agent_rollout_sched.py

A multi-turn agentic rollout is a loop:

    prefill -> decode -> tool call -> ENV EXECUTES (GPU idle!) -> observation
            -> decode -> ... -> done

Two properties make this hard to schedule:

  1. While the environment runs (sandbox / browser / search: 0.5-20 s), that
     trajectory needs no GPU compute. A scheduler that blocks on it burns GPU.
  2. Turn counts are heavy-tailed: most tasks finish in 3-5 turns, a few grind
     for 40+. Collecting a batch synchronously means waiting for the worst one.

This file is a discrete-event simulation comparing three schedulers:

    A. sync-batch   : lockstep rounds with a barrier at every turn
    B. async        : continuous batching at trajectory granularity
    C. async+partial: B, plus a per-iteration token budget; unfinished
                      trajectories are carried over instead of discarded

The GPU is modelled as a single work-conserving server with a fixed token
throughput; the environment is modelled as a pool with `env_slots` capacity.
Everything is deterministic given the seed, and the invariants at the bottom
are asserted, so this doubles as a regression test.
"""

from __future__ import annotations

import heapq
import random
import statistics
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Workload model
# ----------------------------------------------------------------------------

GPU_TOK_PER_S = 4000.0  # aggregate decode throughput of the rollout engine
ENV_SLOTS = 64          # how many sandbox / browser instances we can run at once


@dataclass
class Trajectory:
    """One agentic episode: alternating (generate, env-execute) turns."""

    tid: int
    turns: list[tuple[int, float]]  # (tokens_to_generate, env_latency_s)
    cursor: int = 0                 # which turn we are on
    tokens_done: int = 0            # tokens generated so far (across turns)
    started_at: float = 0.0
    finished_at: float = 0.0
    policy_versions: set[int] = field(default_factory=set)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t for t, _ in self.turns)


def make_workload(n_traj: int, seed: int = 0) -> list[Trajectory]:
    """Heavy-tailed turn counts, which is the whole point of the exercise."""
    rng = random.Random(seed)
    trajs = []
    for tid in range(n_traj):
        # 80% short (2-6 turns), 15% medium (7-15), 5% long tail (16-45).
        u = rng.random()
        if u < 0.80:
            n_turns = rng.randint(2, 6)
        elif u < 0.95:
            n_turns = rng.randint(7, 15)
        else:
            n_turns = rng.randint(16, 45)

        turns = []
        for _ in range(n_turns):
            tokens = rng.randint(120, 900)          # assistant tokens this turn
            # Env latency is itself heavy-tailed: most tool calls are fast,
            # some (test suites, page loads) take many seconds.
            v = rng.random()
            if v < 0.70:
                env_s = rng.uniform(0.2, 1.5)
            elif v < 0.95:
                env_s = rng.uniform(1.5, 6.0)
            else:
                env_s = rng.uniform(6.0, 20.0)
            turns.append((tokens, env_s))
        trajs.append(Trajectory(tid=tid, turns=turns))
    return trajs


# ----------------------------------------------------------------------------
# Result bookkeeping
# ----------------------------------------------------------------------------


@dataclass
class SimResult:
    name: str
    wall_s: float
    gpu_busy_s: float
    finished: int
    carried_over: int
    wasted_tokens: int
    completion_times: list[float]

    @property
    def util(self) -> float:
        return self.gpu_busy_s / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.completion_times) if self.completion_times else 0.0

    @property
    def p99(self) -> float:
        if not self.completion_times:
            return 0.0
        s = sorted(self.completion_times)
        return s[min(len(s) - 1, int(0.99 * len(s)))]

    @property
    def straggler_ratio(self) -> float:
        return self.p99 / self.p50 if self.p50 > 0 else 0.0


# ----------------------------------------------------------------------------
# A. Synchronous batch scheduler (the naive implementation)
# ----------------------------------------------------------------------------


def sim_sync_batch(trajs: list[Trajectory], batch: int = 64) -> SimResult:
    """Lockstep rounds: the whole batch generates, then the whole batch waits.

    The barrier at every turn is what kills throughput: each round costs
    (round generation time) + (SLOWEST env latency in the round).
    """
    now = 0.0
    gpu_busy = 0.0
    completion = []

    for start in range(0, len(trajs), batch):
        group = trajs[start : start + batch]
        for t in group:
            t.started_at = now

        active = list(group)
        while active:
            # --- generation phase: GPU does all of this batch's tokens ---
            round_tokens = sum(t.turns[t.cursor][0] for t in active)
            gen_s = round_tokens / GPU_TOK_PER_S
            gpu_busy += gen_s
            now += gen_s
            for t in active:
                t.tokens_done += t.turns[t.cursor][0]

            # --- env phase: everyone blocks on the slowest tool call ---
            slowest_env = max(t.turns[t.cursor][1] for t in active)
            now += slowest_env  # GPU is idle for this entire stretch

            still = []
            for t in active:
                t.cursor += 1
                if t.done:
                    t.finished_at = now
                    completion.append(t.finished_at - t.started_at)
                else:
                    still.append(t)
            active = still

    return SimResult("A. sync-batch", now, gpu_busy, len(trajs), 0, 0, completion)


# ----------------------------------------------------------------------------
# B / C. Async continuous batching, optionally with a carry-over budget
# ----------------------------------------------------------------------------


def sim_async(
    trajs: list[Trajectory],
    max_concurrent: int = 256,
    token_budget: int | None = None,
    carry_over: bool = True,
) -> SimResult:
    """Event-driven scheduler.

    The GPU is work-conserving: it always serves some trajectory that is ready
    to generate, so a trajectory blocked on its environment costs nothing.

    `token_budget` caps how many tokens this iteration may generate. When the
    budget runs out, unfinished trajectories are either carried over to the
    next iteration (partial rollout) or discarded (their tokens are wasted).
    """
    name = "C. async+partial" if token_budget is not None and carry_over else (
        "B. async" if token_budget is None else "B'. async+abort"
    )

    pending = list(trajs)                 # not yet admitted
    ready: list[Trajectory] = []          # ready to generate now
    env_heap: list[tuple[float, int, Trajectory]] = []  # (t_free, tid, traj)
    in_flight = 0

    now = 0.0
    gpu_busy = 0.0
    tokens_spent = 0
    completion = []
    finished = 0
    budget_hit = False

    def admit() -> None:
        nonlocal in_flight
        while pending and in_flight < max_concurrent and not budget_hit:
            t = pending.pop(0)
            t.started_at = now
            ready.append(t)
            in_flight += 1

    admit()

    while ready or env_heap:
        if not ready:
            # GPU has nothing to do: fast-forward to the next env completion.
            # (This idle time is exactly what async scheduling minimises.)
            t_free, _, traj = heapq.heappop(env_heap)
            now = max(now, t_free)
            ready.append(traj)
            # Drain everything else that finished at the same instant.
            while env_heap and env_heap[0][0] <= now:
                _, _, other = heapq.heappop(env_heap)
                ready.append(other)
            continue

        traj = ready.pop(0)
        tokens, env_s = traj.turns[traj.cursor]

        if token_budget is not None and tokens_spent + tokens > token_budget:
            budget_hit = True
            # Put it back; it is unfinished when the loop ends.
            ready.insert(0, traj)
            break

        gen_s = tokens / GPU_TOK_PER_S
        gpu_busy += gen_s
        now += gen_s
        tokens_spent += tokens
        traj.tokens_done += tokens
        traj.cursor += 1

        if traj.done:
            traj.finished_at = now
            completion.append(traj.finished_at - traj.started_at)
            finished += 1
            in_flight -= 1
            admit()
        else:
            # Hand off to the environment. With ENV_SLOTS capacity, a call may
            # queue; approximate that by pushing the free time out.
            queue_delay = 0.0
            if len(env_heap) >= ENV_SLOTS:
                queue_delay = max(0.0, env_heap[0][0] - now)
            heapq.heappush(env_heap, (now + queue_delay + env_s, traj.tid, traj))

        # Anything whose env finished by now becomes ready again.
        while env_heap and env_heap[0][0] <= now:
            _, _, other = heapq.heappop(env_heap)
            ready.append(other)

    unfinished = [t for t in trajs if not t.done]
    wasted = 0 if carry_over else sum(t.tokens_done for t in unfinished)

    return SimResult(name, now, gpu_busy, finished, len(unfinished), wasted, completion)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def bar(frac: float, width: int = 28) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def report(results: list[SimResult]) -> None:
    print(f"{'scheduler':<18} {'wall(s)':>9} {'GPU util':>9}  {'utilisation':<30}"
          f" {'done':>5} {'carry':>6} {'wasted tok':>11} {'p99/p50':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r.name:<18} {r.wall_s:>9.1f} {r.util:>8.1%}  [{bar(r.util)}]"
              f" {r.finished:>5} {r.carried_over:>6} {r.wasted_tokens:>11,}"
              f" {r.straggler_ratio:>8.2f}")


# ----------------------------------------------------------------------------
# Demos
# ----------------------------------------------------------------------------


def demo_scheduling(n_traj: int = 512) -> tuple[SimResult, SimResult]:
    print("=" * 110)
    print("1. Scheduler comparison — same workload, three schedulers")
    print("=" * 110)
    print(f"workload: {n_traj} trajectories, heavy-tailed turns, "
          f"GPU={GPU_TOK_PER_S:.0f} tok/s, env_slots={ENV_SLOTS}\n")

    a = sim_sync_batch(make_workload(n_traj), batch=64)
    b = sim_async(make_workload(n_traj), max_concurrent=256)
    report([a, b])

    print(f"\n  async wall-clock speedup : {a.wall_s / b.wall_s:.2f}x")
    print(f"  GPU utilisation          : {a.util:.1%} -> {b.util:.1%}")
    print("\n  Why: sync-batch pays `max(env_latency)` at EVERY turn with the GPU")
    print("  parked. Async lets trajectories blocked on their environment drop")
    print("  out of the running batch, so the GPU always has work.")
    print("\n  Async still does not reach 100%: at the end of the iteration only a")
    print("  few long trajectories remain, too few to keep the engine fed. That")
    print("  drain phase is exactly what partial rollout removes (demo 2).")
    return a, b


def demo_partial_rollout(n_traj: int = 512) -> tuple[SimResult, SimResult]:
    print("\n" + "=" * 110)
    print("2. Iteration budget — partial rollout (carry over) vs abort (discard)")
    print("=" * 110)

    # A budget that deliberately cannot fit the long tail.
    total = sum(t.total_tokens for t in make_workload(n_traj))
    budget = int(total * 0.60)
    print(f"total workload = {total:,} tokens, iteration budget = {budget:,} "
          f"({budget / total:.0%})\n")

    abort = sim_async(make_workload(n_traj), token_budget=budget, carry_over=False)
    part = sim_async(make_workload(n_traj), token_budget=budget, carry_over=True)
    report([abort, part])

    print(f"\n  tokens thrown away: abort={abort.wasted_tokens:,}  "
          f"partial={part.wasted_tokens:,}")
    print("\n  Partial rollout keeps the half-finished long-tail trajectories and")
    print("  resumes them next iteration. The price is off-policy: those")
    print("  trajectories span more than one policy version, so you must either")
    print("  track the version per token, cap the span, or accept the bias.")
    return abort, part


def demo_straggler(n_traj: int = 512) -> None:
    print("\n" + "=" * 110)
    print("3. Where the long tail actually is")
    print("=" * 110)
    trajs = make_workload(n_traj)
    turns = sorted(len(t.turns) for t in trajs)
    toks = sorted(t.total_tokens for t in trajs)

    def pct(xs: list[int], p: float) -> int:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    print(f"  turns  per trajectory: p50={pct(turns, .5):>3}  p90={pct(turns, .9):>3}"
          f"  p99={pct(turns, .99):>3}  max={turns[-1]:>3}"
          f"   (p99/p50 = {pct(turns, .99) / pct(turns, .5):.1f}x)")
    print(f"  tokens per trajectory: p50={pct(toks, .5):>6,}  p90={pct(toks, .9):>6,}"
          f"  p99={pct(toks, .99):>6,}  max={toks[-1]:>6,}"
          f"   (p99/p50 = {pct(toks, .99) / pct(toks, .5):.1f}x)")

    tail = [t for t in trajs if len(t.turns) >= 16]
    tail_tokens = sum(t.total_tokens for t in tail)
    all_tokens = sum(t.total_tokens for t in trajs)
    print(f"\n  the {len(tail)}/{n_traj} ({len(tail) / n_traj:.0%}) longest trajectories"
          f" carry {tail_tokens / all_tokens:.0%} of all tokens")
    print("  -> this is why a synchronous barrier is so expensive, and why the")
    print("     long tail must be carried over rather than waited on.")


def intrinsic_times(trajs: list[Trajectory]) -> list[float]:
    """How long each trajectory would take if it had the GPU to itself.

    This is the trajectory's own critical path: its generation time plus the
    environment latency it genuinely has to wait for. Comparing measured
    completion against this is the honest way to price a scheduler.
    """
    return [
        sum(tok for tok, _ in t.turns) / GPU_TOK_PER_S + sum(e for _, e in t.turns)
        for t in trajs
    ]


def demo_metrics(a: SimResult, b: SimResult, n_traj: int = 512) -> tuple[float, float]:
    """Two per-trajectory latency metrics both say async is worse. Both are wrong."""
    print("\n" + "=" * 110)
    print("4. Which metric should you actually optimise?")
    print("=" * 110)

    ideal = statistics.mean(intrinsic_times(make_workload(n_traj)))
    lat_a = statistics.mean(a.completion_times)
    lat_b = statistics.mean(b.completion_times)

    print(f"  mean intrinsic time (a trajectory alone on the GPU): {ideal:.1f} s\n")
    print(f"  {'metric':<38} {'A. sync-batch':>15} {'B. async':>13}   verdict")
    print("  " + "-" * 92)
    rows = [
        ("makespan / iteration wall-clock (s)", a.wall_s, b.wall_s, "async 4.6x better", True),
        ("GPU utilisation", a.util, b.util, "async 4.6x better", True),
        ("mean per-trajectory latency (s)", lat_a, lat_b, "async looks WORSE", False),
        ("per-trajectory p99/p50", a.straggler_ratio, b.straggler_ratio,
         "async looks WORSE", False),
    ]
    for label, va, vb, verdict, matters in rows:
        fa = f"{va:.1%}" if "utilisation" in label else f"{va:.2f}"
        fb = f"{vb:.1%}" if "utilisation" in label else f"{vb:.2f}"
        flag = "<-- optimise this" if matters else "<-- red herring"
        print(f"  {label:<38} {fa:>15} {fb:>13}   {verdict:<18} {flag}")

    print("\n  Async deliberately keeps 256 trajectories in flight, so each one")
    print("  queues behind many others and its individual latency gets worse.")
    print("  That is a real effect, and it does not matter: nothing consumes a")
    print("  trajectory until the whole iteration is assembled for the training")
    print("  step. The quantity that gates the next optimiser step is makespan.")
    print("\n  The p99/p50 trap is worse still. Sync-batch scores BETTER on it")
    print("  because the barrier drags every short trajectory down to the pace")
    print("  of the slowest one -- the spread narrows precisely because")
    print("  everyone is equally starved. A percentile ratio cannot distinguish")
    print("  'no stragglers' from 'everything is a straggler'.")
    return lat_a, lat_b


def verify(a: SimResult, b: SimResult, abort: SimResult, part: SimResult,
           lat_a: float, lat_b: float) -> None:
    print("\n" + "=" * 110)
    print("5. Invariants")
    print("=" * 110)

    checks = [
        ("async GPU utilisation beats sync-batch", b.util > a.util),
        ("async wall-clock beats sync-batch", b.wall_s < a.wall_s),
        ("async is near work-conserving (util > 85%)", b.util > 0.85),
        ("async lifts utilisation by >3x", b.util > 3 * a.util),
        ("sync-batch wastes most of the GPU (util < 50%)", a.util < 0.50),
        ("both schedulers finish every trajectory", a.finished == b.finished),
        ("both do identical GPU work (same tokens)",
         abs(a.gpu_busy_s - b.gpu_busy_s) < 1e-6),
        ("partial rollout wastes no tokens", part.wasted_tokens == 0),
        ("abort throws away real work", abort.wasted_tokens > 0),
        ("partial rollout carries the unfinished tail",
         part.carried_over > 0 and part.carried_over == abort.carried_over),
        # The two counter-intuitive ones: async wins the metric that matters
        # (makespan) while losing both per-trajectory latency metrics.
        ("async wins on makespan, the metric that gates the training step",
         b.wall_s < a.wall_s / 3),
        ("async loses on mean per-trajectory latency (and that is fine)",
         lat_b > lat_a),
        ("async loses on p99/p50 too -- the ratio cannot see starvation",
         b.straggler_ratio >= a.straggler_ratio),
    ]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        assert ok, label


def main() -> None:
    a, b = demo_scheduling()
    abort, part = demo_partial_rollout()
    demo_straggler()
    lat_a, lat_b = demo_metrics(a, b)
    verify(a, b, abort, part, lat_a, lat_b)
    print("\n" + "=" * 110)
    print("all checks passed")
    print("=" * 110)


if __name__ == "__main__":
    main()
