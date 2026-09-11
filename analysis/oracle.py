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


def speedups(rounds: list[dict], gamma: int, c: float) -> dict:
    """Static-gamma vs oracle, both emitting the same tokens."""
    lengths = [r["n_accepted"] for r in rounds]
    mean_emit = statistics.mean([n + 1 for n in lengths])

    static_cost = gamma * c + 1.0          # always pays gamma drafts
    oracle_cost = statistics.mean([n * c + 1.0 for n in lengths])  # drafts exactly n

    return {
        "mean_emitted_per_round": mean_emit,
        "static_cost_per_round": static_cost,
        "oracle_cost_per_round": oracle_cost,
        "static_speedup": mean_emit / static_cost,
        "oracle_speedup": mean_emit / oracle_cost,
        "headroom_x": (mean_emit / oracle_cost) / (mean_emit / static_cost),
        "wasted_draft_steps_per_round": gamma - statistics.mean(lengths),
    }


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
    sp = speedups(rounds, gamma, c)
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
    print(f"  static gamma={gamma} speedup     {sp['static_speedup']:.3f}x")
    print(f"  ORACLE speedup             {sp['oracle_speedup']:.3f}x")
    print(f"  HEADROOM                   {sp['headroom_x']:.3f}x")
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
    print(f"  Interpretation: an adaptive scheduler can win at most {sp['headroom_x']:.2f}x")
    print(f"  over static gamma={gamma} here, by not wasting "
          f"{sp['wasted_draft_steps_per_round']:.1f} draft steps per round.")

    out = os.path.join(rdir, f"oracle_gamma{gamma}.json")
    with open(out, "w") as f:
        json.dump({"gamma": gamma, "cost_ratio": c,
                   "run_lengths": rl, "speedups": sp, "burstiness": bz}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
