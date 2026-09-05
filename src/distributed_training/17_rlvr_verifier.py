"""Chapter 15: the RLVR verifier pool — latency, timeouts, caching, capacity.

Run:
    python3 src/distributed_training/17_rlvr_verifier.py

RLVR replaces the learned reward model with a deterministic verifier. That
moves reward computation off the GPU and onto a CPU fleet, which introduces a
capacity-planning problem that did not exist in classic RLHF.

Three parts:

  1. A real math-answer verifier, to show that "just compare the strings" is
     nowhere near enough, and that the extraction step is an attack surface.
  2. A discrete-event model of the verifier pool: heavy-tailed latency, hangs,
     per-sample timeouts, result caching and intra-group dedup.
  3. A check of the capacity rule used in the book,

         W >= G * r * (1 - h) * t_v / u

     against the simulated queue, including why plugging in the MEAN latency
     under-provisions a heavy-tailed verifier.

Everything is stdlib-only and deterministic given the seed.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from fractions import Fraction

# ============================================================================
# Part 1 — a real (small) math verifier
# ============================================================================

NAIVE_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def all_boxed(response: str) -> list[str]:
    """Every \\boxed{...} payload, with correct brace matching.

    A regex cannot do this: \\boxed{\\frac{1}{2}} nests braces, and the
    obvious [^{}]* pattern silently fails to match it at all. Silently, as in
    the verifier returns "no answer found" and scores a correct solution as
    wrong -- which teaches the model to stop using \\frac.
    """
    out = []
    for m in re.finditer(r"\\boxed\{", response):
        depth, i = 1, m.end()
        while i < len(response) and depth:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(response[m.end() : i - 1].strip())
    return out


def extract_answer(response: str) -> str | None:
    """Take the LAST \\boxed{...}: the model must commit to one final answer."""
    boxed = all_boxed(response)
    return boxed[-1] if boxed else None


def extract_answer_permissive(response: str) -> list[str]:
    """The tempting-but-exploitable variant: accept ANY \\boxed{...}."""
    return all_boxed(response)


def normalise(ans: str) -> str:
    """Canonicalise an answer so that equivalent spellings compare equal."""
    s = ans.strip()
    # Strip common LaTeX decoration and units that carry no mathematical content.
    for junk in (r"\left", r"\right", r"\!", r"\,", r"\;", "$", " ", "^\\circ"):
        s = s.replace(junk, "")
    s = s.rstrip(".")
    # "x = 5" / "answer: 5" -> "5"
    if "=" in s:
        s = s.split("=")[-1]
    s = s.removeprefix("answer:").removeprefix("Answer:")
    # 1,024 -> 1024
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    # \frac{a}{b} -> a/b
    m = re.fullmatch(r"\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    # \dfrac and \tfrac
    m = re.fullmatch(r"\\[dt]frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    return s


def numeric_value(s: str) -> Fraction | None:
    """Best-effort exact numeric value, so 1/2, 0.5 and 2/4 all agree."""
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return Fraction(Fraction(num), Fraction(den))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equal(got: str, gold: str) -> bool:
    a, b = normalise(got), normalise(gold)
    if a == b:
        return True
    va, vb = numeric_value(a), numeric_value(b)
    return va is not None and vb is not None and va == vb


def verify_math(response: str, gold: str) -> bool:
    """Return True iff the response's final answer equals the gold answer."""
    got = extract_answer(response)
    return got is not None and answers_equal(got, gold)


def verify_math_permissive(response: str, gold: str) -> bool:
    """Exploitable: rewards the response if ANY boxed candidate is right."""
    return any(answers_equal(c, gold) for c in extract_answer_permissive(response))


def demo_math_verifier() -> None:
    print("=" * 100)
    print("1. A real math verifier — why naive string equality is not enough")
    print("=" * 100)

    gold = "1/2"
    cases = [
        (r"... so the answer is \boxed{1/2}.", True, "exact match"),
        (r"... thus \boxed{0.5}", True, "decimal form of the same number"),
        (r"... hence \boxed{2/4}", True, "unreduced fraction"),
        (r"... giving \boxed{\frac{1}{2}}", True, r"\frac -> needs brace matching"),
        (r"... so \boxed{x = 1/2}", True, "answer carries 'x =' prefix"),
        (r"... therefore \boxed{ 1/2 }.", True, "whitespace + trailing period"),
        (r"... the answer is 1/2", False, r"no \boxed -> cannot extract"),
        (r"... \boxed{1/3}", False, "genuinely wrong"),
    ]

    ok = 0
    for resp, want, why in cases:
        got = verify_math(resp, gold)
        mark = "ok " if got == want else "BAD"
        ok += got == want
        shown = resp if len(resp) < 46 else resp[:43] + "..."
        print(f"  [{mark}] {str(got):<5} {shown:<48} {why}")
    assert ok == len(cases), "math verifier behaved unexpectedly"
    print(f"\n  {ok}/{len(cases)} behaved as specified.\n")

    # --- the extractor itself is an attack surface -------------------------
    print("  Now the same verifier against a hedging response, two extractors:\n")
    hedge = r"maybe \boxed{1/3}, or \boxed{1/2}, or possibly \boxed{2/3}"
    strict = verify_math(hedge, gold)
    loose = verify_math_permissive(hedge, gold)
    print(f"    response: {hedge}")
    print(f"    last-boxed  (correct)    -> {strict}   commits to 2/3, scored wrong")
    print(f"    any-boxed   (exploitable)-> {loose}   one guess matched, scored right")
    assert strict is False and loose is True

    print("\n  If the extractor accepts any candidate, spraying ten guesses is a")
    print("  strictly winning strategy and the policy finds it within a few")
    print("  hundred steps. Nothing about the reward model is involved -- this")
    print("  is reward hacking against the PARSER. In RLVR the verifier and its")
    print("  extractor are part of the attack surface and should be reviewed")
    print("  the way you would review a sandbox escape.")

    print("\n  Note also the \\frac case: the obvious regex \\\\boxed\\{([^{}]*)\\}")
    print("  cannot match nested braces, so it reports 'no answer' on a correct")
    print("  solution. That failure is silent and one-directional -- it only")
    print("  ever marks correct answers wrong -- so it shows up as a mysteriously")
    print("  low pass rate on exactly the problems whose answers are fractions.")
    naive_hit = NAIVE_BOXED.findall(r"\boxed{\frac{1}{2}}")
    print(f"    naive regex on \\boxed{{\\frac{{1}}{{2}}}} -> {naive_hit}  (empty!)")
    assert naive_hit == []

    print("\n  Every normalisation rule above is also a decision about what the")
    print("  model is allowed to learn. Accepting '0.5' for '1/2' is usually")
    print("  right; accepting it for a question that asks for a reduced")
    print("  fraction is a silent grading bug.")


# ============================================================================
# Part 2 — verifier pool simulation
# ============================================================================

TIMEOUT_S = 10.0


@dataclass
class VerifyJob:
    job_id: int
    group_id: int
    payload_hash: str
    true_latency: float   # what it would cost if allowed to run to completion
    hangs: bool           # pathological input: would spin forever


def make_jobs(n_groups: int, group_n: int, dup_rate: float,
              seed: int = 0) -> list[VerifyJob]:
    """Code-style verifier workload: heavy-tailed, with a few hangs.

    `dup_rate` is the probability that a response inside a group is an exact
    duplicate of an earlier one. At low temperature on math/code this is very
    common: there are only so many ways to write the correct solution.
    """
    rng = random.Random(seed)
    jobs: list[VerifyJob] = []
    jid = 0
    for g in range(n_groups):
        seen: list[str] = []
        for _ in range(group_n):
            if seen and rng.random() < dup_rate:
                h = rng.choice(seen)  # exact duplicate response
            else:
                h = hashlib.sha1(f"{g}:{jid}:{rng.random()}".encode()).hexdigest()[:12]
                seen.append(h)

            u = rng.random()
            if u < 0.75:
                lat = rng.uniform(0.05, 1.0)     # fast unit tests
            elif u < 0.97:
                lat = rng.uniform(1.0, 5.0)      # slower suites
            else:
                lat = rng.uniform(5.0, 30.0)     # pathological but finite
            hangs = rng.random() < 0.004          # infinite loop / stdin wait

            jobs.append(VerifyJob(jid, g, h, lat, hangs))
            jid += 1
    return jobs


@dataclass
class PoolResult:
    name: str
    wall_s: float
    executed: int          # jobs that actually ran on a worker
    served: int            # jobs that got a reward (executed + cache + dedup)
    cache_hits: int
    dedup_hits: int
    timeouts: int
    worker_busy_s: float
    waits: list[float] = field(default_factory=list)

    @property
    def util(self) -> float:
        return self.worker_busy_s / (self.wall_s * self.workers) if self.wall_s else 0.0

    workers: int = 1

    @property
    def p50(self) -> float:
        return statistics.median(self.waits) if self.waits else 0.0

    @property
    def p99(self) -> float:
        s = sorted(self.waits)
        return s[min(len(s) - 1, int(0.99 * len(s)))] if s else 0.0


def run_pool(jobs: list[VerifyJob], workers: int, arrival_rate: float,
             use_cache: bool, use_timeout: bool, name: str) -> PoolResult:
    """Model `workers` single-threaded verifier processes.

    Jobs arrive at `arrival_rate` per second (this is the rollout engine's
    output rate). Each is dispatched to the first free worker; if none is free
    it queues. A cached or deduped result returns instantly and never occupies
    a worker -- that is the whole point of caching.
    """
    free: list[float] = [0.0] * workers      # min-heap of worker free times
    heapq.heapify(free)

    cache: dict[str, bool] = {}
    cache_hits = dedup_hits = executed = timeouts = 0
    worker_busy = 0.0
    waits: list[float] = []
    now = 0.0
    last_done = 0.0

    for i, job in enumerate(jobs):
        now = i / arrival_rate  # arrival time

        if use_cache and job.payload_hash in cache:
            # Same (prompt, response) seen before -> reuse. Free, no worker.
            if cache[job.payload_hash]:
                dedup_hits += 1
            else:
                cache_hits += 1
            waits.append(0.0)
            last_done = max(last_done, now)
            continue

        worker_free_at = heapq.heappop(free)
        start = max(now, worker_free_at)
        cost = TIMEOUT_S if job.hangs else job.true_latency
        if use_timeout:
            cost = min(cost, TIMEOUT_S)
            if job.hangs or job.true_latency > TIMEOUT_S:
                timeouts += 1
        elif job.hangs:
            # No timeout: the worker is gone for the rest of the run.
            cost = 1e9

        finish = start + cost
        heapq.heappush(free, finish)
        worker_busy += cost
        executed += 1
        waits.append(finish - now)
        last_done = max(last_done, finish)

        if use_cache:
            cache[job.payload_hash] = True

    res = PoolResult(name, last_done, executed, len(jobs), cache_hits, dedup_hits,
                     timeouts, worker_busy, waits)
    res.workers = workers
    return res


def demo_pool() -> tuple[PoolResult, PoolResult, PoolResult]:
    print("\n" + "=" * 100)
    print("2. Verifier pool — timeouts and caching")
    print("=" * 100)

    jobs = make_jobs(n_groups=200, group_n=16, dup_rate=0.35, seed=1)
    arrival = 20.0   # responses/s coming out of the rollout engine
    workers = 40

    print(f"  {len(jobs)} verify jobs ({200} groups x {16} responses), "
          f"arrival {arrival:.0f}/s, {workers} workers, timeout {TIMEOUT_S:.0f}s")
    hangs = sum(j.hangs for j in jobs)
    print(f"  {hangs} of them ({hangs / len(jobs):.1%}) are pathological "
          f"(infinite loop / waiting on stdin)\n")

    naive = run_pool(jobs, workers, arrival, use_cache=False, use_timeout=False,
                     name="no timeout, no cache")
    timed = run_pool(jobs, workers, arrival, use_cache=False, use_timeout=True,
                     name="timeout, no cache")
    cached = run_pool(jobs, workers, arrival, use_cache=True, use_timeout=True,
                      name="timeout + cache/dedup")

    print(f"  {'configuration':<24} {'makespan(s)':>12} {'ran':>7} {'cached':>7}"
          f" {'timeouts':>9} {'p50 wait':>9} {'p99 wait':>9}")
    print("  " + "-" * 84)
    for r in (naive, timed, cached):
        span = f"{r.wall_s:,.0f}" if r.wall_s < 1e8 else "collapsed"
        print(f"  {r.name:<24} {span:>12} {r.executed:>7}"
              f" {r.cache_hits + r.dedup_hits:>7} {r.timeouts:>9}"
              f" {r.p50:>8.2f}s {r.p99:>8.2f}s")

    saved = cached.cache_hits + cached.dedup_hits
    print(f"\n  Without a timeout, {hangs} hung jobs each consume a worker forever.")
    print(f"  The pool loses {hangs}/{workers} of its capacity permanently, arrivals")
    print("  keep coming, and the queue grows without bound. One pathological")
    print("  sample is enough to stall the run.")
    print(f"\n  Cache + dedup served {saved}/{len(jobs)} ({saved / len(jobs):.0%}) of jobs")
    print(f"  for free and cut the p50 wait {timed.p50:.2f}s -> {cached.p50:.2f}s")
    print(f"  ({1 - cached.p50 / timed.p50:.0%}). Duplicate responses within a GRPO")
    print("  group are the bulk of it: the same prompt sampled N times produces")
    print("  the same solution repeatedly, and each copy would re-run the tests.")
    print(f"\n  Both p99 figures sit at exactly {TIMEOUT_S:.0f}s, which is the point:")
    print("  once a timeout exists it DEFINES the tail. Caching buys throughput")
    print("  and median latency; only the timeout bounds the worst case. Those")
    print("  are two different jobs and you need both.")
    return naive, timed, cached


# ============================================================================
# Part 3 — capacity planning
# ============================================================================


def required_workers(gpus: int, resp_per_gpu_s: float, hit_rate: float,
                     t_v: float, util: float) -> float:
    """W >= G * r * (1 - h) * t_v / u  (the rule quoted in chapter 15)."""
    return gpus * resp_per_gpu_s * (1 - hit_rate) * t_v / util


def demo_capacity() -> None:
    print("\n" + "=" * 100)
    print("3. Capacity planning — the formula is right, but it answers the")
    print("   wrong question")
    print("=" * 100)

    gpus, r, h, u = 128, 0.5, 0.4, 0.7
    group_n = 16
    jobs = make_jobs(n_groups=400, group_n=group_n, dup_rate=h, seed=2)
    lats = sorted(TIMEOUT_S if j.hangs else min(j.true_latency, TIMEOUT_S)
                  for j in jobs)
    t_mean = statistics.mean(lats)

    print(f"  fleet: G={gpus} rollout GPUs, r={r}/s per GPU, "
          f"cache+dedup h={h:.0%}, target utilisation u={u:.0%}")
    print(f"  verify latency: mean={t_mean:.2f}s  p90={lats[int(.90 * len(lats))]:.2f}s"
          f"  p99={lats[int(.99 * len(lats))]:.2f}s   (heavy-tailed)\n")

    w_mean = int(required_workers(gpus, r, h, t_mean, u))
    arrival = gpus * r
    res = run_pool(jobs, w_mean, arrival, use_cache=True, use_timeout=True,
                   name="mean-sized")

    print(f"  W = G*r*(1-h)*t_v/u  with t_v = mean  ->  {w_mean} workers")
    print(f"  simulated at arrival = G*r = {arrival:.0f}/s:")
    print(f"    worker utilisation {res.util:.1%} (target {u:.0%}), "
          f"p50 wait {res.p50:.2f}s, p99 wait {res.p99:.2f}s -> queue is stable")
    print("\n  So the formula does its job. Steady-state stability depends on the")
    print("  MEAN service time, and sizing on the mean lands on the utilisation")
    print("  you asked for. Do not let anyone talk you into sizing on p90 'for")
    print("  safety' -- here that would be 2.8x the machines for no throughput.")

    # --- the question the formula does NOT answer --------------------------
    print("\n  But throughput is not what gates the training step. GRPO cannot")
    print(f"  compute advantages for a group until ALL {group_n} of its responses are")
    print("  verified, so what the optimiser waits on is the MAXIMUM latency in")
    print("  the group, not the mean.\n")

    by_group: dict[int, list[float]] = {}
    for j in jobs:
        cost = TIMEOUT_S if j.hangs else min(j.true_latency, TIMEOUT_S)
        by_group.setdefault(j.group_id, []).append(cost)
    group_max = [max(v) for v in by_group.values()]

    print(f"    E[single verify]          = {t_mean:6.2f} s")
    print(f"    E[max over {group_n} in a group] = {statistics.mean(group_max):6.2f} s"
          f"   ({statistics.mean(group_max) / t_mean:.1f}x the mean)")
    print(f"    p99 of the group maximum  = "
          f"{sorted(group_max)[int(.99 * len(group_max))]:6.2f} s")

    hit_timeout = sum(1 for m in group_max if math.isclose(m, TIMEOUT_S))
    print(f"\n    {hit_timeout}/{len(group_max)} groups ({hit_timeout / len(group_max):.0%})"
          f" contain at least one sample that hits the timeout,")
    print(f"    so for those groups the wait is the full {TIMEOUT_S:.0f}s no matter how")
    print("    many workers you buy.")

    print("\n  Two consequences worth stating in an interview:")
    print("    1. Adding workers fixes throughput, never the group-max wait.")
    print("       The lever for the group max is the TIMEOUT, plus dedup (fewer")
    print("       independent draws from the tail per group).")
    print("    2. Do not block the training step on the group max at all: verify")
    print("       asynchronously and let the next rollout batch overlap it. Then")
    print("       the group max stops being on the critical path and the")
    print("       mean-based capacity number is genuinely all you need.")

    assert res.util > 0.5, "mean-sized pool should be reasonably utilised"
    assert res.p99 <= TIMEOUT_S + 1e-6, "timeout must bound the per-job wait"
    assert statistics.mean(group_max) > 2 * t_mean, \
        "group max should dominate the mean for a heavy-tailed verifier"


def verify_all(naive: PoolResult, timed: PoolResult, cached: PoolResult) -> None:
    print("\n" + "=" * 100)
    print("4. Invariants")
    print("=" * 100)
    checks = [
        ("every job gets a reward in every configuration",
         naive.served == timed.served == cached.served),
        ("without a timeout the pool collapses", naive.wall_s > 1e8),
        ("the timeout bounds the tail", timed.p99 < 1e6),
        ("timeouts fire only on pathological jobs", timed.timeouts > 0),
        ("cache + dedup removes real work", cached.executed < timed.executed),
        ("cache + dedup halves the median wait", cached.p50 < timed.p50),
        ("the timeout, not the cache, is what pins the tail",
         math.isclose(cached.p99, TIMEOUT_S) and math.isclose(timed.p99, TIMEOUT_S)),
        ("cache accounting is consistent",
         cached.executed + cached.cache_hits + cached.dedup_hits == cached.served),
    ]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        assert ok, label


def main() -> None:
    demo_math_verifier()
    naive, timed, cached = demo_pool()
    demo_capacity()
    verify_all(naive, timed, cached)
    print("\n" + "=" * 100)
    print("all checks passed")
    print("=" * 100)


if __name__ == "__main__":
    main()
