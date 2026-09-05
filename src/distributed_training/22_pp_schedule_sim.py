"""Chapter 6: what a pipeline schedule actually costs.

Run (single process, no torchrun):
    python3 src/distributed_training/22_pp_schedule_sim.py

Companion to 21_pp_from_scratch.py. That file proves the four schedules are
*correct*; this one shows what they *cost*, by simulating the instruction
lists from common/pipeline.py against a dependency graph.

Simulating instead of benchmarking is deliberate. A schedule's bubble is a
property of its dependency structure, not of your GPU, so it is reproducible
to the digit and can be compared against the closed-form results people quote
in interviews. The simulator also catches a malformed schedule as a deadlock
rather than as a hang.

Parts:
    1. schedules are data          - print them and read them
    2. dependency simulation       - measured bubble vs the textbook formula
    3. ASCII Gantt charts          - see where the bubble actually is
    4. activation memory           - the real reason 1F1B beat GPipe
    5. sweeping m and P            - which knob actually removes the bubble
"""

from __future__ import annotations

from dataclasses import dataclass

from common.pipeline import (
    B, BI, BW, F, Instr, sched_1f1b, sched_gpipe, sched_interleaved_1f1b,
    sched_zero_bubble,
)

# Cost model in arbitrary units. A fused backward costs about twice a forward
# because it computes two GEMMs (dL/dx and dL/dW); splitting it gives one each.
COST = {F: 1.0, B: 2.0, BI: 1.0, BW: 1.0}


def build(name: str, P: int, m: int, V: int) -> dict[int, list[Instr]]:
    """Instruction list per rank, plus the per-chunk cost scaling for V."""
    if name == "gpipe":
        return {r: sched_gpipe(r, P, m) for r in range(P)}
    if name == "1f1b":
        return {r: sched_1f1b(r, P, m) for r in range(P)}
    if name == "zero-bubble":
        return {r: sched_zero_bubble(r, P, m) for r in range(P)}
    if name == "interleaved":
        return {r: sched_interleaved_1f1b(r, P, m, V) for r in range(P)}
    raise ValueError(name)


@dataclass
class SimResult:
    makespan: float
    busy: list[float]
    trace: list[tuple[int, Instr, float, float]]   # rank, instr, start, end

    @property
    def bubble(self) -> float:
        mean_busy = sum(self.busy) / len(self.busy)
        return 1.0 - mean_busy / self.makespan


def simulate(scheds: dict[int, list[Instr]], P: int, V: int,
             comm: float = 0.0) -> SimResult:
    """List-schedule the instruction lists against their data dependencies.

    Two dependency families, and that is all pipeline parallelism is:
      forward   F(mb, g)  needs F(mb, g-1)
      backward  B(mb, g)  needs B(mb, g+1)
    plus the rule that each rank runs its own list strictly in order.

    Costs are divided by V so that the V=1 and V>1 configurations do the same
    total work -- otherwise interleaving would look better just by counting
    smaller chunks.
    """
    n_global = V * P
    done: dict[tuple[str, int, int], float] = {}   # (kind, mb, global chunk) -> end
    ptr = {r: 0 for r in range(P)}
    now = {r: 0.0 for r in range(P)}
    trace: list[tuple[int, Instr, float, float]] = []
    busy = [0.0] * P

    def dep_time(r: int, ins: Instr) -> float | None:
        """Earliest this instruction may start, or None if deps not yet known."""
        g = ins.chunk * P + r
        if ins.op == F:
            if g == 0:
                return 0.0
            t = done.get(("F", ins.mb, g - 1))
            return None if t is None else t + comm
        if ins.op in (B, BI):
            if g == n_global - 1:
                return 0.0            # needs only this rank's own forward
            t = done.get(("B", ins.mb, g + 1))
            return None if t is None else t + comm
        return 0.0                    # BW blocks on nothing but this rank

    while any(ptr[r] < len(scheds[r]) for r in range(P)):
        progressed = False
        for r in range(P):
            if ptr[r] >= len(scheds[r]):
                continue
            ins = scheds[r][ptr[r]]
            dt = dep_time(r, ins)
            if dt is None:
                continue
            cost = COST[ins.op] / V
            start = max(now[r], dt)
            end = start + cost
            now[r] = end
            busy[r] += cost
            trace.append((r, ins, start, end))
            g = ins.chunk * P + r
            if ins.op == F:
                done[("F", ins.mb, g)] = end
            elif ins.op in (B, BI):
                done[("B", ins.mb, g)] = end
            ptr[r] += 1
            progressed = True
        if not progressed:
            stuck = {r: str(scheds[r][ptr[r]]) for r in range(P)
                     if ptr[r] < len(scheds[r])}
            raise RuntimeError(f"schedule deadlock, each rank waiting on: {stuck}")

    return SimResult(max(now.values()), busy, trace)


def live_activations(sched: list[Instr]) -> tuple[int, int]:
    """(peak live activations, peak deferred weight-grad holdings) for one rank.

    Both are read straight off the instruction list -- no execution needed.
    A forward stores an activation; BI releases it but keeps the tensors that
    the deferred W still needs, so zero-bubble trades memory for bubble.
    """
    live = peak = defer = peak_defer = 0
    for ins in sched:
        if ins.op == F:
            live += 1
            peak = max(peak, live)
        elif ins.op == B:
            live -= 1
        elif ins.op == BI:
            live -= 1
            defer += 1
            peak_defer = max(peak_defer, defer)
        elif ins.op == BW:
            defer -= 1
    return peak, peak_defer


def gantt(res: SimResult, P: int, unit: float = 1.0) -> list[str]:
    """One row per rank; '.' is a bubble.

    `unit` is time per character. Pass the smallest op cost (1.0 when V=1, 1/V
    otherwise) so every op lands on a whole number of cells -- otherwise
    rounding invents single-character gaps that look like bubbles and are not.
    """
    cols = int(round(res.makespan / unit))
    rows = []
    for r in range(P):
        cells = ["."] * cols
        for rr, ins, s, e in res.trace:
            if rr != r:
                continue
            lo = int(round(s / unit))
            hi = max(lo + 1, int(round(e / unit)))
            for c in range(lo, min(hi, cols)):
                cells[c] = {F: "F", B: "B", BI: "b", BW: "w"}[ins.op]
        rows.append(f"  stage {r} |" + "".join(cells) + "|")
    return rows


# ---- parts ----------------------------------------------------------------


def part1_schedules_are_data(P: int, m: int, V: int):
    print("\n" + "=" * 78)
    print("PART 1  a schedule is just a list of instructions")
    print("=" * 78)
    print(f"  P={P} stages, m={m} micro-batches, V={V} virtual chunks\n")
    print("  F=forward  B=backward(fused)  b=backward-input  w=backward-weight")
    print("  'B3.1' means micro-batch 3 on this rank's virtual chunk 1\n")
    for name in ("gpipe", "1f1b", "zero-bubble", "interleaved"):
        scheds = build(name, P, m, V)
        print(f"  {name}:")
        for r in range(P):
            print(f"    stage {r}: " + " ".join(str(i) for i in scheds[r]))
        print()
    print("  Everything below is computed from these lists alone. Being able to")
    print("  print, diff and simulate a schedule before running it is the whole")
    print("  reason to represent it as data instead of nested for-loops.")


def part2_bubble(P: int, m: int, V: int):
    print("\n" + "=" * 78)
    print("PART 2  measured bubble vs the closed-form formula")
    print("=" * 78)

    theory = {
        "gpipe": (P - 1) / (m + P - 1),
        "1f1b": (P - 1) / (m + P - 1),
        "interleaved": (P - 1) / (V * m + P - 1),
    }
    print(f"  {'schedule':<14}{'bubble':>9}{'formula':>10}{'makespan':>11}"
          f"{'F/B units':>11}")
    results = {}
    for name in ("gpipe", "1f1b", "zero-bubble", "interleaved"):
        res = simulate(build(name, P, m, V), P, V if name == "interleaved" else 1)
        results[name] = res
        th = theory.get(name)
        th_s = f"{th:>9.1%}" if th is not None else f"{'--':>9}"
        print(f"  {name:<14}{res.bubble:>8.1%}{th_s}{res.makespan:>11.1f}"
              f"{sum(res.busy) / P:>11.1f}")
        if th is not None:
            assert abs(res.bubble - th) < 1e-6, f"{name}: {res.bubble} vs {th}"

    print("\n  GPipe and 1F1B have the SAME bubble -- a point worth making in an")
    print("  interview, because the usual assumption is that 1F1B is faster. It")
    print("  is not; it uses less memory (part 4) at identical throughput.")
    zb, f1 = results["zero-bubble"].bubble, results["1f1b"].bubble
    print(f"\n  zero-bubble cuts the bubble {f1:.1%} -> {zb:.1%}, not to zero. The W")
    print("  work fills the cool-down, but during warm-up no gradient has arrived")
    print("  yet, so there is no W to do. Removing that half needs ZB-2P, which")
    print("  holds more activations.")
    assert zb < f1
    return results


def part3_gantt(P: int, m: int, V: int, results):
    print("\n" + "=" * 78)
    print("PART 3  where the bubble actually sits")
    print("=" * 78)
    for name in ("gpipe", "1f1b", "zero-bubble"):
        res = results[name]
        print(f"\n  {name}  (bubble {res.bubble:.1%})")
        for row in gantt(res, P):
            print(row)
    print("\n  Read the shape, not just the number.")
    print("  GPipe: one contiguous idle block per stage, in the middle -- every")
    print("  stage finishes all m forwards, then waits for the backward wave.")
    print("  1F1B: identical total idle, but it sits in the warm-up wait plus")
    print("  small cool-down gaps. Note what did NOT change: the bubble. 1F1B's")
    print("  win is that backward starts after only P-1-rank forwards instead of")
    print("  m, which is a memory property, not a time property.")
    print("  Zero-bubble: fills the right-hand ramp with 'w' and leaves the")
    print("  left-hand one, exactly as the warm-up argument predicts.")


def part4_memory(P: int, m: int, V: int):
    print("\n" + "=" * 78)
    print("PART 4  activation memory: the actual reason 1F1B won")
    print("=" * 78)
    print("  One rule covers every schedule: a stage's activation peak equals the")
    print("  number of forwards it runs before its first backward. The last column")
    print("  is that count for stage 0 -- it matches the peak exactly.\n")
    print(f"  {'schedule':<14}{'stage 0':>9}{'stage ' + str(P-1):>9}"
          f"{'deferred W':>13}   forwards before first backward")
    for name in ("gpipe", "1f1b", "zero-bubble", "interleaved"):
        scheds = build(name, P, m, V)
        first, defer0 = live_activations(scheds[0])
        last, _ = live_activations(scheds[P - 1])
        note = {
            "gpipe": f"m = {m}  (no steady phase at all)",
            "1f1b": f"P - rank = {P}",
            "zero-bubble": f"P - rank = {P}, but W holds its tensors longer",
            "interleaved": f"2(P-1-rank) + (V-1)P + 1 = {2*(P-1) + (V-1)*P + 1}",
        }[name]
        print(f"  {name:<14}{first:>9}{last:>9}{defer0:>13}   {note}")

    gp, _ = live_activations(build("gpipe", P, m, V)[0])
    f1, _ = live_activations(build("1f1b", P, m, V)[0])
    print(f"\n  With m={m}, P={P}: GPipe holds {gp} activations on stage 0, 1F1B holds"
          f" {f1}.")
    print(f"  The ratio is m/P = {m/P:.1f}x, and it grows with m -- so GPipe caps how")
    print("  large a global batch you can use, which is precisely the knob you need")
    print("  to turn to shrink the bubble. That coupling is what kills GPipe.")
    assert gp == m and f1 == P


def part5_sweep(P: int, m: int, V: int):
    print("\n" + "=" * 78)
    print("PART 5  which knob removes the bubble?")
    print("=" * 78)
    print(f"  bubble for 1F1B, P={P}:")
    print(f"    {'m':>6}" + "".join(f"{x:>9}" for x in (2, 4, 8, 16, 32, 64)))
    row = ""
    for mm in (2, 4, 8, 16, 32, 64):
        res = simulate(build("1f1b", P, mm, V), P, 1)
        row += f"{res.bubble:>9.1%}"
    print(f"    {'':>6}" + row)

    print(f"\n  bubble for interleaved (m={m}, P={P}) as V grows:")
    print(f"    {'V':>6}" + "".join(f"{v:>9}" for v in (1, 2, 4, 8)))
    row = ""
    for vv in (1, 2, 4, 8):
        res = simulate(build("interleaved", P, m, vv), P, vv)
        row += f"{res.bubble:>9.1%}"
    print(f"    {'':>6}" + row)

    print("\n  Both knobs work, and they cost different things. Raising m costs")
    print("  activation memory under GPipe but nothing under 1F1B -- so with 1F1B")
    print("  the first move is always more micro-batches. Raising V costs P2P")
    print("  round-trips and activation memory, so it is what you reach for when m")
    print("  is already capped by the global batch size you want.")


def main():
    P, m, V = 4, 8, 2
    part1_schedules_are_data(P, m, V)
    results = part2_bubble(P, m, V)
    part3_gantt(P, m, V, results)
    part4_memory(P, m, V)
    part5_sweep(P, m, V)
    print("\n" + "=" * 78)
    print("all measured bubbles match their closed-form formulas")
    print("=" * 78)


if __name__ == "__main__":
    main()
