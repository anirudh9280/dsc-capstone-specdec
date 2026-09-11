"""M4: how much headroom does adaptive draft-size scheduling actually have?

The quarter's candidate project is scheduling gamma per position. Before building
a scheduler it is worth knowing the ceiling -- the speedup a PERFECT scheduler
would reach, one that knows in advance exactly how many tokens will be accepted.
If the gap between the best static gamma and that oracle is small, adaptive
scheduling is not worth doing on this workload, and establishing that is itself a
result worth presenting.

Where the oracle's advantage comes from
---------------------------------------
A static-gamma round ALWAYS pays gamma draft steps, because all gamma proposals
are generated before the target verifies any of them. If only 2 of 8 are accepted,
6 draft steps were wasted. An oracle that knew the run would end at 2 would draft
exactly 2. Both emit the same tokens -- speculative decoding is lossless, so the
output is the target's greedy sequence either way. The oracle wins purely by not
wasting draft compute.

    static gamma:  emit mean(r)+1 tokens   for   gamma*c + 1
    oracle:        emit mean(r)+1 tokens   for   mean(r)*c + 1

Why one large-gamma trace suffices
----------------------------------
Under greedy decoding the emitted sequence is identical for every gamma (that is
what losslessness means), so "does the draft agree at position j given the true
prefix" is a property of the position, not of gamma. A single gamma=8 run
therefore contains the acceptance structure for all smaller gamma.

The honest caveat: runs that reach gamma are RIGHT-CENSORED -- they might have
continued further. So the oracle computed here is a LOWER BOUND, and the censoring
rate is reported alongside it. If censoring is high, rerun with a larger gamma.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter


def load_rounds(path: str) -> tuple[list[dict], int]:
    rounds, gamma = [], 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            gamma = rec["gamma"]
            rounds.extend(rec["rounds"])
    return rounds, gamma


def run_length_stats(rounds: list[dict], gamma: int) -> dict:
    lengths = [r["n_accepted"] for r in rounds]
    censored = sum(1 for r in rounds if r["n_accepted"] >= gamma)
    return {
        "n_rounds": len(rounds),
        "mean_run": statistics.mean(lengths) if lengths else 0.0,
        "median_run": statistics.median(lengths) if lengths else 0.0,
        "censored": censored,
        "censored_frac": censored / len(rounds) if rounds else 0.0,
        "histogram": dict(sorted(Counter(lengths).items())),
    }


def speedups(rounds: list[dict], gamma: int, c: float,
             sweep: list[dict] | None = None) -> dict:
    """Static-gamma vs oracle, both emitting the same tokens.

    Two different comparisons, and the distinction matters:

    * vs static at THIS gamma -- inflated whenever the trace's gamma is a poor
      choice, because a large static gamma wastes many draft steps. Reporting only
      this number would overstate what adaptive scheduling buys.
    * vs the BEST static gamma -- the honest baseline. Nobody would deploy a gamma
      they had not tuned, so an adaptive scheduler has to beat the tuned constant,
      not an arbitrary one.
    """
    lengths = [r["n_accepted"] for r in rounds]
    mean_emit = statistics.mean([n + 1 for n in lengths])

    static_cost = gamma * c + 1.0          # always pays gamma drafts
    oracle_cost = statistics.mean([n * c + 1.0 for n in lengths])  # drafts exactly n
    oracle_speedup = mean_emit / oracle_cost

    out = {
        "mean_emitted_per_round": mean_emit,
        "static_cost_per_round": static_cost,
        "oracle_cost_per_round": oracle_cost,
        "static_speedup": mean_emit / static_cost,
        "oracle_speedup": oracle_speedup,
        "headroom_vs_same_gamma": oracle_speedup / (mean_emit / static_cost),
        "wasted_draft_steps_per_round": gamma - statistics.mean(lengths),
    }

    # Best tuned static gamma, from MEASURED per-gamma traces (see measured_curve()).
    #
    # Do not try to re-derive the static curve from a single large-gamma trace by
    # averaging min(r, gamma')+1 over its rounds. That is biased LOW, and the bias
    # is not small -- it gave 3.692 where the measured gamma=4 run gave 4.187.
    # The reason: a long agreeing run becomes SEVERAL rounds at a smaller gamma'
    # (a run of 10 is two rounds at gamma'=4), so long runs must be weighted by how
    # many gamma'-rounds they produce. Averaging over the large-gamma round
    # boundaries under-weights exactly the rounds that matter most.
    #
    # A fully exact re-derivation is not available at all: when every draft in a
    # round is accepted, the bonus token comes from the target and was never tested
    # against the draft, so the per-position agreement sequence has holes precisely
    # at those positions. Measure each gamma instead; it is cheap.
    if sweep:
        best_g, best_sp = None, 0.0
        for row in sweep:
            g = row["gamma"]
            sp_g = row["acceptance_length"] / (g * c + 1.0)
            if sp_g > best_sp:
                best_g, best_sp = g, sp_g
        out["best_static_gamma"] = best_g
        out["best_static_speedup"] = best_sp
        out["headroom_vs_best_static"] = oracle_speedup / best_sp if best_sp else None
    return out


def measured_curve(rdir: str, c: float) -> list[dict]:
    """Measured acceptance length per gamma, read from each gamma's own trace.

    Read from the JSONL files rather than summary.json, which a later
    single-gamma run overwrites -- that silently made gamma=16 look like the best
    static choice when the full sweep had already shown gamma=3.
    """
    rows = []
    for fn in sorted(os.listdir(rdir)):
        if not (fn.startswith("gamma") and fn.endswith(".jsonl")):
            continue
        g = int(fn[5:-6])
        emitted = rounds = 0
        with open(os.path.join(rdir, fn)) as fh:
            for line in fh:
                rec = json.loads(line)
                emitted += rec["n_generated"]
                rounds += rec["n_rounds"]
        if rounds:
            acc = emitted / rounds
            rows.append({"gamma": g, "acceptance_length": acc,
                         "speedup": acc / (g * c + 1.0)})
    return sorted(rows, key=lambda r: r["gamma"])


def burstiness(rounds: list[dict]) -> dict:
    """Are rejections clustered or independent?

    A scheduler can only exploit structure that persists. If run lengths are
    i.i.d., knowing the last round tells you nothing about the next and an
    adaptive policy has nothing to condition on.
    """
    lengths = [r["n_accepted"] for r in rounds]
    if len(lengths) < 3:
        return {"lag1_autocorr": None}
    mean = statistics.mean(lengths)
    var = statistics.pvariance(lengths)
    if var == 0:
        return {"lag1_autocorr": 0.0, "variance": 0.0}
    cov = statistics.mean(
        [(lengths[i] - mean) * (lengths[i + 1] - mean) for i in range(len(lengths) - 1)]
    )
    return {"lag1_autocorr": cov / var, "variance": var}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--gamma", type=int, default=None,
                    help="which trace to analyze (default: largest available)")
    ap.add_argument("--cost-ratio", type=float, default=None,
                    help="override c; default reads it from summary.json")
    args = ap.parse_args()

    rdir = args.results_dir or os.path.join(
        os.environ.get("SPECDEC_RESULTS", os.path.expanduser("~/specdec-results")), "math500"
    )
    summary_path = os.path.join(rdir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"no summary.json in {rdir} -- run bench/run_math500.py first", file=sys.stderr)
        return 1
    summary = json.load(open(summary_path))

    available = sorted(
        int(f[5:-6]) for f in os.listdir(rdir)
        if f.startswith("gamma") and f.endswith(".jsonl")
    )
    gamma = args.gamma or max(available)
    path = os.path.join(rdir, f"gamma{gamma}.jsonl")
    rounds, gamma = load_rounds(path)

    row = next((r for r in summary["sweep"] if r["gamma"] == gamma), None)
    c = args.cost_ratio if args.cost_ratio is not None else (row["cost_ratio"] if row else 1.0)

    rl = run_length_stats(rounds, gamma)
    curve = measured_curve(rdir, c)
    sp = speedups(rounds, gamma, c, curve)
    bz = burstiness(rounds)

    print("=" * 68)
    print(f"ORACLE-GAMMA CEILING   (trace gamma={gamma}, c={c:.3f}, {rl['n_rounds']} rounds)")
    print("=" * 68)
    print(f"  mean accepted run       {rl['mean_run']:.3f}  of gamma={gamma}")
    print(f"  median run              {rl['median_run']:.1f}")
    print(f"  right-censored rounds   {rl['censored']} ({rl['censored_frac'] * 100:.1f}%)")
    if rl["censored_frac"] > 0.25:
        print("    [WARN] heavy censoring -- oracle below is a loose lower bound.")
        print("           Rerun the sweep with a larger gamma to tighten it.")
    print()
    print("  run-length histogram (accepted tokens per round):")
    total = max(1, rl["n_rounds"])
    for k, v in rl["histogram"].items():
        bar = "#" * max(1, round(40 * v / total))
        tag = "  <- censored" if k >= gamma else ""
        print(f"    {k:>2}: {v:>5} {bar}{tag}")
    print()
    print("-" * 68)
    print(f"  wasted draft steps/round   {sp['wasted_draft_steps_per_round']:.3f}")
    print(f"  emitted tokens/round       {sp['mean_emitted_per_round']:.3f}")
    print(f"  static cost/round          {sp['static_cost_per_round']:.3f}")
    print(f"  oracle cost/round          {sp['oracle_cost_per_round']:.3f}")
    print("-" * 68)
    print(f"  measured static-gamma curve at c={c:.3f} (one run per gamma):")
    for row in curve:
        star = "  <- best" if row["gamma"] == sp.get("best_static_gamma") else ""
        print(f"    gamma={row['gamma']:>2}: accept_len {row['acceptance_length']:>6.3f}   "
              f"speedup {row['speedup']:.3f}x{star}")
    print("-" * 68)
    print(f"  static gamma={gamma} speedup     {sp['static_speedup']:.3f}x")
    print(f"  best static (gamma={sp['best_static_gamma']})       "
          f"{sp['best_static_speedup']:.3f}x")
    print(f"  ORACLE speedup             {sp['oracle_speedup']:.3f}x")
    print("-" * 68)
    print(f"  headroom vs same gamma     {sp['headroom_vs_same_gamma']:.3f}x   "
          f"(inflated if gamma={gamma} is a poor choice)")
    if sp.get("headroom_vs_best_static") is not None:
        print(f"  HEADROOM vs BEST STATIC    {sp['headroom_vs_best_static']:.3f}x   "
              f"<- the honest number")
    print("=" * 68)
    print()
    print(f"  lag-1 autocorrelation of run length: {bz['lag1_autocorr']}")
    if bz["lag1_autocorr"] is not None:
        if abs(bz["lag1_autocorr"]) < 0.1:
            print("    Run lengths look close to independent. A scheduler conditioned")
            print("    on recent history has little to work with; per-position signals")
            print("    (draft entropy, top-1 margin) are the better bet -- see signals.py.")
        else:
            print("    Run lengths are autocorrelated: recent history carries signal,")
            print("    so a cheap history-based scheduler is worth trying.")
    print()
    hr = sp.get("headroom_vs_best_static") or sp["headroom_vs_same_gamma"]
    against = (f"the best TUNED static gamma={sp['best_static_gamma']}"
               if sp.get("best_static_gamma") is not None else f"static gamma={gamma}")
    print(f"  Interpretation: a perfect scheduler wins at most {hr:.2f}x over")
    print(f"  {against}, by not wasting "
          f"{sp['wasted_draft_steps_per_round']:.1f} draft steps per round.")
    print("  Quote this against the tuned baseline, not against the trace's own")
    print("  gamma -- nobody deploys an untuned constant, so beating one is not")
    print("  evidence that adaptive scheduling is worthwhile.")

    out = os.path.join(rdir, f"oracle_gamma{gamma}.json")
    with open(out, "w") as f:
        json.dump({"gamma": gamma, "cost_ratio": c,
                   "run_lengths": rl, "speedups": sp, "burstiness": bz}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
