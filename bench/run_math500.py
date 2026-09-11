"""M2: acceptance length on MATH-500 -- the baseline Prof. Liu asked for.

Why MATH-500 specifically: solutions are long (decode dominates, so acceptance
matters) and structurally mixed. LaTeX scaffolding is near-deterministic and should
accept at very high rates; the actual numeric and algebraic choices should not.
That heterogeneity is the entire premise of adaptive draft scheduling, so the
dataset is a testbed for the quarter's question, not just a benchmark.

Sweeps gamma in one process so both models load once. Writes one JSONL per gamma
with FULL per-round telemetry -- M4's oracle and signal analysis run offline on
these files and must never require a new GPU run.

  python bench/run_math500.py --n-problems 25 --gammas 1 2 3 4 5 6 8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.loop import SpecDecConfig, autoregressive_generate, speculative_generate  # noqa: E402

DATASET_ID = "HuggingFaceH4/MATH-500"


def build_prompt(tok, problem: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": problem + "\n\nReason step by step, then give the final answer in \\boxed{}."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # bounded generations; thinking mode is a separate study
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--n-problems", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="also time plain autoregressive decoding for a real speedup denominator")
    ap.add_argument("--assume-cost-ratio", type=float, default=None,
                    help="recompute speedup with an externally measured c. Acceptance length "
                         "is a property of the model pair and is identical eager or compiled, "
                         "but c is not -- so measure c under the config you intend to ship "
                         "(bench/roofline.py --compile) and pass it here.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.environ.get("SPECDEC_RESULTS", os.path.expanduser("~/specdec-results")), "math500"
    )
    os.makedirs(out_dir, exist_ok=True)

    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"free VRAM: {free_gib:.2f} GiB")
    if free_gib < 8.0:
        print("  [WARN] low -- close Overwatch / Ollama / Chrome before trusting timings.")

    from datasets import load_dataset

    print(f"loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="test")
    problems = [ds[i]["problem"] for i in range(min(args.n_problems, len(ds)))]
    print(f"  {len(problems)} problems")

    print(f"loading target {args.target} / draft {args.draft} (bf16)...")
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda").eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=torch.bfloat16, device_map="cuda").eval()

    prompts = [build_prompt(tok, p) for p in problems]
    id_batches = [tok(p, return_tensors="pt").input_ids.to("cuda") for p in prompts]

    # Optional plain-decode denominator, so speedup is measured rather than inferred.
    baseline_ms: list[float] = []
    if args.baseline:
        print("\ntiming plain autoregressive decoding...")
        for i, ids in enumerate(id_batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            autoregressive_generate(target, ids, args.max_new_tokens,
                                    temperature=args.temperature, seed=args.seed,
                                    eos_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            baseline_ms.append((time.perf_counter() - t0) * 1000.0)
            print(f"  [{i + 1}/{len(id_batches)}] {baseline_ms[-1]:.0f} ms", end="\r")
        print(f"\n  median {statistics.median(baseline_ms):.0f} ms per problem")

    summary = []
    for gamma in args.gammas:
        path = os.path.join(out_dir, f"gamma{gamma}.jsonl")
        acc_lens, speedups, ratios, walls = [], [], [], []
        print(f"\n=== gamma = {gamma} ===")

        with open(path, "w") as fh:
            for i, ids in enumerate(id_batches):
                res = speculative_generate(
                    target, draft, ids,
                    SpecDecConfig(gamma=gamma, temperature=args.temperature,
                                  max_new_tokens=args.max_new_tokens, seed=args.seed),
                    eos_token_id=tok.eos_token_id,
                )
                acc_lens.append(res.acceptance_length)
                ratios.append(res.cost_ratio)
                speedups.append(res.analytic_speedup)
                walls.append(res.wall_ms)

                fh.write(json.dumps({
                    "problem_idx": i,
                    "gamma": gamma,
                    "target": args.target,
                    "draft": args.draft,
                    "temperature": args.temperature,
                    "prompt_len": int(ids.shape[1]),
                    "n_generated": res.n_generated,
                    "n_rounds": len(res.rounds),
                    "acceptance_length": res.acceptance_length,
                    "mean_accepted": res.mean_accepted,
                    "cost_ratio": res.cost_ratio,
                    "analytic_speedup": res.analytic_speedup,
                    "wall_ms": res.wall_ms,
                    "baseline_wall_ms": baseline_ms[i] if baseline_ms else None,
                    "rounds": [asdict(r) for r in res.rounds],
                }) + "\n")
                print(f"  [{i + 1}/{len(id_batches)}] accept_len={res.acceptance_length:.3f}  "
                      f"c={res.cost_ratio:.3f}  {res.wall_ms:.0f} ms", end="\r")

        med_acc = statistics.median(acc_lens)
        med_c = statistics.median(ratios)
        med_sp = statistics.median(speedups)
        measured_sp = (statistics.median(baseline_ms) / statistics.median(walls)
                       if baseline_ms else None)
        print(f"\n  acceptance length : {med_acc:.3f}  (max possible {gamma + 1})")
        print(f"  cost ratio c      : {med_c:.3f}")
        print(f"  analytic speedup  : {med_sp:.3f}x")
        if measured_sp is not None:
            print(f"  measured speedup  : {measured_sp:.3f}x")
        print(f"  -> {path}")

        projected = (med_acc / (gamma * args.assume_cost_ratio + 1)
                     if args.assume_cost_ratio is not None else None)
        if projected is not None:
            print(f"  projected @ c={args.assume_cost_ratio:.3f} : {projected:.3f}x")

        summary.append({
            "gamma": gamma,
            "acceptance_length": med_acc,
            "max_possible": gamma + 1,
            "cost_ratio": med_c,
            "analytic_speedup": med_sp,
            "measured_speedup": measured_sp,
            "assumed_cost_ratio": args.assume_cost_ratio,
            "projected_speedup": projected,
            "n_problems": len(id_batches),
        })

    # Merge rather than overwrite. A later single-gamma run used to clobber the
    # whole sweep, which silently made gamma=16 look like the best static choice
    # when the full sweep had already shown gamma=3.
    spath = os.path.join(out_dir, "summary.json")
    merged = {row["gamma"]: row for row in summary}
    if os.path.exists(spath):
        try:
            prev = json.load(open(spath))
            for row in prev.get("sweep", []):
                merged.setdefault(row["gamma"], row)
        except (json.JSONDecodeError, OSError):
            pass  # unreadable previous summary: just write the fresh one
    with open(spath, "w") as f:
        json.dump({"target": args.target, "draft": args.draft,
                   "temperature": args.temperature,
                   "max_new_tokens": args.max_new_tokens,
                   "sweep": [merged[g] for g in sorted(merged)]}, f, indent=2)

    print()
    print("=" * 82)
    print(f"{'gamma':>6}{'accept_len':>13}{'of max':>9}{'c':>8}"
          f"{'analytic':>11}{'measured':>11}{'projected':>12}")
    print("=" * 82)
    for row in summary:
        meas = f"{row['measured_speedup']:.3f}x" if row["measured_speedup"] else "--"
        proj = f"{row['projected_speedup']:.3f}x" if row["projected_speedup"] else "--"
        print(f"{row['gamma']:>6}{row['acceptance_length']:>13.3f}"
              f"{row['max_possible']:>9}{row['cost_ratio']:>8.3f}"
              f"{row['analytic_speedup']:>10.3f}x{meas:>11}{proj:>12}")
    print("=" * 82)
    print()
    print("Read the two trends against each other: acceptance length rises")
    print("monotonically with gamma, but speedup is unimodal and peaks earlier.")
    print("That gap is the argument for scheduling gamma per position (M4).")
    print(f"\nwrote {spath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
