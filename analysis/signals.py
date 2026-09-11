"""M4: what could an adaptive-gamma scheduler actually condition on?

A scheduler has to decide how far to draft BEFORE the target verifies anything, so
it may only use information available at draft time: the draft model's own output
distribution, the position within the current round, and the token text. It cannot
use the target's probabilities -- those are precisely what it is trying to avoid
paying for.

For each drafted position we hold out the binary label "was this accepted" and ask
how well each cheap signal predicts it, by AUC. AUC 0.5 is worthless; the further
from 0.5, the more a scheduler could exploit it. Direction matters and is reported:
a signal with AUC 0.2 is as useful as one with 0.8, just inverted.

Labels come from the round structure: within a round the loop stops at the first
rejection, so positions 0..n_accepted-1 were accepted and position n_accepted (when
it exists) was rejected. Positions after that were never scored and are excluded.

  python analysis/signals.py
  python analysis/signals.py --tokenizer Qwen/Qwen3-1.7B   # adds token-class breakdown
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged. No sklearn dependency."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    rank_sum = sum(r for r, l in zip(ranks, labels) if l)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def load_examples(path: str):
    """One example per SCORED drafted position."""
    ents, margins, positions, tokens, labels = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            for r in rec["rounds"]:
                n_acc = r["n_accepted"]
                n_scored = len(r["draft_entropy"])
                for i in range(n_scored):
                    ents.append(r["draft_entropy"][i])
                    margins.append(r["draft_top1_margin"][i])
                    positions.append(i)
                    tokens.append(r["draft_token_ids"][i] if i < len(r["draft_token_ids"]) else -1)
                    labels.append(1 if i < n_acc else 0)
    return ents, margins, positions, tokens, labels


def classify(text: str) -> str:
    s = text.strip()
    if not s:
        return "whitespace"
    if s.startswith("\\"):
        return "latex_cmd"
    if any(ch.isdigit() for ch in s):
        return "numeric"
    if s.isalpha():
        return "word"
    return "symbol"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--gamma", type=int, default=None)
    ap.add_argument("--tokenizer", default=None, help="enables the token-class breakdown")
    args = ap.parse_args()

    rdir = args.results_dir or os.path.join(
        os.environ.get("SPECDEC_RESULTS", os.path.expanduser("~/specdec-results")), "math500"
    )
    available = sorted(
        int(f[5:-6]) for f in os.listdir(rdir)
        if f.startswith("gamma") and f.endswith(".jsonl")
    )
    if not available:
        print(f"no gamma*.jsonl in {rdir} -- run bench/run_math500.py first", file=sys.stderr)
        return 1
    gamma = args.gamma or max(available)
    path = os.path.join(rdir, f"gamma{gamma}.jsonl")

    ents, margins, positions, tokens, labels = load_examples(path)
    n, pos = len(labels), sum(labels)
    print("=" * 70)
    print(f"ACCEPTANCE PREDICTORS   (gamma={gamma}, {n} scored drafted positions)")
    print("=" * 70)
    print(f"  accepted {pos} ({pos / n * 100:.1f}%)   rejected {n - pos}")
    print()

    feats = {
        "draft entropy": ents,
        "draft top-1 margin": margins,
        "position within round": [float(p) for p in positions],
    }
    print(f"  {'signal':<26}{'AUC':>8}{'mean|acc':>12}{'mean|rej':>12}   usable?")
    print("  " + "-" * 66)
    results = {}
    for name, vals in feats.items():
        a = auc(vals, labels)
        m_acc = statistics.mean([v for v, l in zip(vals, labels) if l])
        m_rej = statistics.mean([v for v, l in zip(vals, labels) if not l])
        strength = abs(a - 0.5)
        verdict = "strong" if strength > 0.20 else "moderate" if strength > 0.10 else "weak"
        results[name] = {"auc": a, "mean_accepted": m_acc, "mean_rejected": m_rej}
        print(f"  {name:<26}{a:>8.3f}{m_acc:>12.4f}{m_rej:>12.4f}   {verdict}")

    print()
    print("  AUC < 0.5 means LOWER values predict acceptance (expected for entropy:")
    print("  a confident draft is more likely to be right).")

    # Does acceptance decay with depth into a round? This is what makes a fixed
    # gamma wasteful -- later proposals are conditioned on earlier guesses.
    by_pos = defaultdict(list)
    for p, l in zip(positions, labels):
        by_pos[p].append(l)
    print()
    print("  acceptance rate by depth within a round:")
    for p in sorted(by_pos):
        rate = statistics.mean(by_pos[p])
        bar = "#" * max(1, round(40 * rate))
        print(f"    pos {p}: {rate * 100:>5.1f}%  n={len(by_pos[p]):<6} {bar}")

    tok_stats = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(args.tokenizer)
        buckets = defaultdict(list)
        for t, l in zip(tokens, labels):
            if t >= 0:
                buckets[classify(tk.decode([t]))].append(l)
        print()
        print("  acceptance rate by token class:")
        tok_stats = {}
        for cls in sorted(buckets, key=lambda c: -len(buckets[c])):
            rate = statistics.mean(buckets[cls])
            tok_stats[cls] = {"rate": rate, "n": len(buckets[cls])}
            bar = "#" * max(1, round(40 * rate))
            print(f"    {cls:<12} {rate * 100:>5.1f}%  n={len(buckets[cls]):<6} {bar}")
        print()
        print("  If LaTeX/whitespace accept far more often than numeric tokens, a")
        print("  scheduler keyed on token class alone would already capture much of")
        print("  the available headroom -- and costs nothing to evaluate.")

    out = os.path.join(rdir, f"signals_gamma{gamma}.json")
    with open(out, "w") as f:
        json.dump({
            "gamma": gamma, "n_examples": n, "n_accepted": pos,
            "auc": results,
            "acceptance_by_position": {p: statistics.mean(v) for p, v in sorted(by_pos.items())},
            "acceptance_by_token_class": tok_stats,
        }, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
