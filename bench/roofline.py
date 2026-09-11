"""M0: what actually limits autoregressive decode on this machine?

The speculative-decoding cost model assumes decode is memory-bandwidth bound: one
token costs a full read of the weights, arithmetic is free, and therefore a small
draft model is proportionally cheaper than a large target. That last clause is the
load-bearing one -- speedup is accept_len / (gamma*c + 1), and it only pays off if
c = draft_cost / target_cost is genuinely small.

Three measurements:

  1. per-token latency (CUDA events) vs the bandwidth ceiling.
  2. CPU ENQUEUE TIME: the same loop re-run with NO synchronization. The CPU runs
     ahead as far as the queue allows, so this is pure host-side cost. If the CPU
     needs nearly as long to SUBMIT the work as the GPU needs to DO it, the loop is
     host/launch bound -- a smaller draft then cannot be proportionally cheaper,
     c approaches 1, and speculative decoding cannot pay off until CUDA graphs fix
     it. Note: a "GPU busy fraction" built from CUDA events CANNOT detect this,
     because the event pair brackets host gaps too and reads ~100% either way.
  3. whether latency actually scales with weight bytes across the model ladder.

  python bench/roofline.py --model Qwen/Qwen3-0.6B
  python bench/roofline.py --model Qwen/Qwen3-0.6B --compile   # CUDA graphs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.timing import cuda_timer, model_weight_bytes, summarize, wall_timer  # noqa: E402

SPEC_BANDWIDTH_GBPS = 960.0       # RTX 5080 (GB203), 256-bit GDDR7 @ 30 Gbps
MEASURED_BANDWIDTH_GBPS = 749.0   # achieved copy bw, scripts/02_verify_gpu.py
DEFAULT_TFLOPS = 225.0            # dense BF16 tensor core, conservative


def decode_step(model, nxt, past):
    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
    return out.logits[:, -1, :].argmax(-1, keepdim=True), out.past_key_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--prompt", default="Explain why memory bandwidth limits LLM decoding.")
    ap.add_argument("--bandwidth", type=float, default=SPEC_BANDWIDTH_GBPS)
    ap.add_argument("--measured-bandwidth", type=float, default=MEASURED_BANDWIDTH_GBPS)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"free VRAM before load: {free_gib:.2f} GiB")
    if free_gib < 8.0:
        print("  [WARN] low -- close Overwatch / Ollama / Chrome first.")

    print(f"loading {args.model} (bf16)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if args.compile:
        print("  compiling (reduce-overhead -> CUDA graphs); warmup will be slow...")
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    w_bytes = model_weight_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e9:.3f} B    weights: {w_bytes / 1e9:.2f} GB")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")
    per_token_ms: list[float] = []
    loop_wall_ms: list[float] = []

    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)   # prefill: untimed, different regime
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

        for _ in range(args.warmup):
            nxt, past = decode_step(model, nxt, past)

        with wall_timer(loop_wall_ms):
            for _ in range(args.tokens):
                with cuda_timer(per_token_ms):
                    nxt, past = decode_step(model, nxt, past)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            nxt, past = decode_step(model, nxt, past)
        cpu_enqueue_ms = (time.perf_counter() - t0) * 1000.0
        torch.cuda.synchronize()
        submit_plus_exec_ms = (time.perf_counter() - t0) * 1000.0

    s = summarize(per_token_ms)
    measured_tps = 1000.0 / s["median"]
    cpu_frac = cpu_enqueue_ms / submit_plus_exec_ms

    bw = args.bandwidth * 1e9
    mem_ms = w_bytes / bw * 1000.0
    ceil_spec = bw / w_bytes
    ceil_meas = args.measured_bandwidth * 1e9 / w_bytes
    compute_ms = (2 * n_params) / (DEFAULT_TFLOPS * 1e12) * 1000.0
    pct_ceiling = measured_tps / ceil_meas * 100

    iqr_str = f"{s['p25']:.2f} - {s['p75']:.2f} ms"
    bw_label = f"weight read @ {args.bandwidth:.0f} GB/s (spec)"
    fl_label = f"arithmetic @ {DEFAULT_TFLOPS:.0f} TFLOP/s"

    print()
    print("=" * 66)
    print(f"{'metric':<36}{'value':>30}")
    print("=" * 66)
    print(f"{'median latency / token':<36}{s['median']:>26.3f} ms")
    print(f"{'IQR':<36}{iqr_str:>30}")
    print(f"{'throughput':<36}{measured_tps:>24.1f} tok/s")
    print("-" * 66)
    print(f"{'synchronized wall, N tokens':<36}{submit_plus_exec_ms:>26.1f} ms")
    print(f"{'CPU-only enqueue, same N':<36}{cpu_enqueue_ms:>26.1f} ms")
    print(f"{'>> CPU SHARE OF WALL':<36}{cpu_frac * 100:>26.1f} %")
    print("-" * 66)
    print(f"{bw_label:<36}{mem_ms:>26.3f} ms")
    print(f"{'=> ceiling (spec BW)':<36}{ceil_spec:>24.1f} tok/s")
    print(f"{'=> ceiling (measured BW)':<36}{ceil_meas:>24.1f} tok/s")
    print(f"{fl_label:<36}{compute_ms:>26.3f} ms")
    print("-" * 66)
    print(f"{'% of measured-BW ceiling':<36}{pct_ceiling:>26.1f} %")
    print("=" * 66)
    print()

    if cpu_frac > 0.85:
        print(f"VERDICT: the CPU needs {cpu_frac * 100:.0f}% of the wall time just to SUBMIT")
        print("         this work -- HOST/LAUNCH BOUND. A smaller draft cannot be")
        print("         proportionally cheaper, so c approaches 1 and speculative")
        print("         decoding cannot pay off until CUDA graphs remove it. Try --compile.")
    elif pct_ceiling < 25:
        print(f"VERDICT: GPU-bound but only {pct_ceiling:.0f}% of the bandwidth ceiling.")
        print("         At batch 1 these are GEMV-shaped kernels that cannot saturate")
        print("         HBM. Bandwidth is the wrong roofline; per-kernel efficiency is")
        print("         the real limit. Confirm the kernel mix in M3.")
    else:
        print("VERDICT: near the bandwidth ceiling. Memory-bound premise holds and the")
        print("         speculative-decoding cost model applies as written.")

    if args.json_out:
        payload = {
            "model": args.model, "compiled": args.compile,
            "n_params": n_params, "weight_bytes": w_bytes,
            "latency_ms": s, "measured_tok_s": measured_tps,
            "cpu_enqueue_ms": cpu_enqueue_ms,
            "submit_plus_exec_ms": submit_plus_exec_ms,
            "cpu_share_of_wall": cpu_frac,
            "ceiling_spec_tok_s": ceil_spec,
            "ceiling_measured_tok_s": ceil_meas,
            "pct_of_measured_ceiling": pct_ceiling,
            "free_vram_gib_before_load": free_gib,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
