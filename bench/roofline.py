"""M0: what actually limits autoregressive decode on this machine?

The speculative-decoding cost model assumes decode is memory-bandwidth bound: one
token costs a full read of the weights, arithmetic is free, and therefore a small
draft model is proportionally cheaper than a large target. That last clause is the
load-bearing one -- speedup is accept_len / (gamma*c + 1), and it only pays off if
c = draft_cost / target_cost is genuinely small.

The falsifiable prediction is that latency should scale with weight bytes. If a
0.6B and a 1.7B model decode at the same speed, the bottleneck is NOT bandwidth;
it is fixed per-step host cost (Python dispatch + kernel launch, ~500 launches per
token for a 28-layer model), and the cost model does not apply as written.

  python bench/roofline.py --model Qwen/Qwen3-0.6B
  python bench/roofline.py --model Qwen/Qwen3-0.6B --compile

--compile implies --static-cache. CUDA graphs need static shapes AND buffers they
own: DynamicCache stores tensors produced inside the graph, and graph trees reuse
those buffers, so a replay silently overwrites the KV cache ("accessing tensor
output of CUDAGraphs that has been overwritten"). StaticCache is preallocated
outside the graph and written in place, which is what makes replay safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.timing import cuda_timer, model_weight_bytes, summarize, wall_timer  # noqa: E402

SPEC_BANDWIDTH_GBPS = 960.0       # RTX 5080 (GB203), 256-bit GDDR7 @ 30 Gbps
MEASURED_BANDWIDTH_GBPS = 749.0   # achieved copy bw, scripts/02_verify_gpu.py
DEFAULT_TFLOPS = 225.0            # dense BF16 tensor core, conservative


def decode_step(model, nxt, past, pos, compiled: bool):
    """One decode step. `pos` is the cache_position StaticCache requires; None for
    DynamicCache, which infers it."""
    if compiled:
        # Tells cudagraph trees a new step began, so it may reclaim last step's
        # output buffers. Anything we carry across the boundary must be cloned.
        torch.compiler.cudagraph_mark_step_begin()
    kw = {} if pos is None else {"cache_position": pos}
    out = model(input_ids=nxt, past_key_values=past, use_cache=True, **kw)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    return (nxt.clone() if compiled else nxt), out.past_key_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--prompt", default="Explain why memory bandwidth limits LLM decoding.")
    ap.add_argument("--bandwidth", type=float, default=SPEC_BANDWIDTH_GBPS)
    ap.add_argument("--measured-bandwidth", type=float, default=MEASURED_BANDWIDTH_GBPS)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="'reduce-overhead' adds CUDA graphs; 'default' is inductor fusion only")
    ap.add_argument("--static-cache", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    use_static = args.static_cache or args.compile
    # Only the cudagraph-backed mode needs step markers / cloned carry-overs.
    uses_cudagraphs = args.compile and args.compile_mode == "reduce-overhead"

    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"free VRAM: {free_gib:.2f} GiB")
    if free_gib < 8.0:
        print("  [WARN] low -- close Overwatch / Ollama / Chrome first.")

    print(f"loading {args.model} (bf16)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if args.compile:
        print(f"  compiling (mode={args.compile_mode}); warmup will be slow...")
        model.forward = torch.compile(model.forward, mode=args.compile_mode)

    w_bytes = model_weight_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e9:.3f} B    weights: {w_bytes / 1e9:.2f} GB")
    print(f"  cache: {'StaticCache' if use_static else 'DynamicCache'}")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")
    plen = ids.shape[1]
    total_steps = args.warmup + 2 * args.tokens + 8
    max_len = plen + total_steps + 8

    per_token_ms: list[float] = []
    loop_wall_ms: list[float] = []

    with torch.inference_mode():
        if use_static:
            cache = StaticCache(config=model.config, max_cache_len=max_len,
                                device="cuda", dtype=torch.bfloat16)
            pos = torch.arange(plen, device="cuda")
            out = model(input_ids=ids, past_key_values=cache,
                        cache_position=pos, use_cache=True)
        else:
            cache = None
            out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        if args.compile:
            nxt = nxt.clone()
        cur = plen

        def step():
            nonlocal nxt, past, cur
            p = torch.tensor([cur], device="cuda") if use_static else None
            nxt, past = decode_step(model, nxt, past, p, uses_cudagraphs)
            cur += 1

        for _ in range(args.warmup):   # warmup outside the measured region
            step()

        with wall_timer(loop_wall_ms):
            for _ in range(args.tokens):
                with cuda_timer(per_token_ms):
                    step()

        # CPU enqueue probe: same work, no synchronization. Reported for context
        # but NOT used as the conclusion -- CUDA's finite launch queue makes a
        # genuinely GPU-bound loop read ~100% here too. The model-size sweep is
        # what actually discriminates.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            step()
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
    print(f"{'CPU share of wall (see note)':<36}{cpu_frac * 100:>26.1f} %")
    print("-" * 66)
    print(f"{bw_label:<36}{mem_ms:>26.3f} ms")
    print(f"{'=> ceiling (spec BW)':<36}{ceil_spec:>24.1f} tok/s")
    print(f"{'=> ceiling (measured BW)':<36}{ceil_meas:>24.1f} tok/s")
    print(f"{fl_label:<36}{compute_ms:>26.3f} ms")
    print("-" * 66)
    print(f"{'% of measured-BW ceiling':<36}{pct_ceiling:>26.1f} %")
    print("=" * 66)
    print()

    if pct_ceiling < 25:
        print(f"VERDICT: {pct_ceiling:.0f}% of the bandwidth ceiling. Compare against the")
        print("         other model sizes: if latency barely moves with weight bytes,")
        print("         this is host/launch bound and c will approach 1.")
    elif pct_ceiling < 60:
        print(f"VERDICT: {pct_ceiling:.0f}% of ceiling -- partly freed. Some host overhead")
        print("         remains, or kernels are GEMV-shaped and cannot saturate HBM.")
    else:
        print(f"VERDICT: {pct_ceiling:.0f}% of ceiling. Memory-bound premise holds; the")
        print("         speculative-decoding cost model applies as written.")

    if args.json_out:
        payload = {
            "model": args.model, "compiled": args.compile, "static_cache": use_static,
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
