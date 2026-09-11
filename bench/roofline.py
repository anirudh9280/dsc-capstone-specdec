"""M0: is autoregressive decode actually memory-bandwidth bound on this machine?

The premise of the whole speculative decoding track is that generating one token
costs a full read of the model's weights from HBM, while the arithmetic is nearly
free. If that is true, measured tok/s should land close to

    ceiling = memory_bandwidth / weight_bytes

and far below what the card's FLOP/s would allow. This script measures both and
prints them side by side. Run it before trusting any later speedup number.

  python bench/roofline.py --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.timing import cuda_timer, model_weight_bytes, summarize  # noqa: E402

# RTX 5080 (GB203): 256-bit GDDR7 @ 30 Gbps.
DEFAULT_BANDWIDTH_GBPS = 960.0
# Dense BF16 tensor-core throughput, conservative.
DEFAULT_TFLOPS = 225.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128, help="tokens to decode (timed)")
    ap.add_argument("--warmup", type=int, default=16, help="untimed tokens first")
    ap.add_argument("--prompt", default="Explain why memory bandwidth limits LLM decoding.")
    ap.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_GBPS)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"free VRAM before load: {free_gib:.2f} GiB")
    if free_gib < 4.0:
        print("  [WARN] very low. Close Ollama / Overwatch / Chrome first.")

    print(f"loading {args.model} (bf16)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    w_bytes = model_weight_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e9:.3f} B    weights: {w_bytes / 1e9:.2f} GB")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")

    per_token_ms: list[float] = []
    with torch.inference_mode():
        # Prefill. Not timed: it is compute bound and a different regime entirely.
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

        for i in range(args.warmup + args.tokens):
            timed = i >= args.warmup
            if timed:
                with cuda_timer(per_token_ms):
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            else:
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

    s = summarize(per_token_ms)
    measured_tps = 1000.0 / s["median"]

    # Ceilings.
    bw_bytes_s = args.bandwidth * 1e9
    mem_ms = w_bytes / bw_bytes_s * 1000.0
    mem_ceiling_tps = bw_bytes_s / w_bytes
    flops_per_token = 2 * n_params
    compute_ms = flops_per_token / (DEFAULT_TFLOPS * 1e12) * 1000.0

    print()
    print("=" * 64)
    print(f"{'metric':<34}{'value':>28}")
    print("=" * 64)
    print(f"{'measured median latency/token':<34}{s['median']:>24.3f} ms")
    iqr_str = f"{s['p25']:.3f} - {s['p75']:.3f} ms"
    print(f"{'measured IQR':<34}{iqr_str:>28}")
    print(f"{'measured throughput':<34}{measured_tps:>22.1f} tok/s")
    print("-" * 64)
    print(f"{'weight-read time @ ' + str(args.bandwidth) + ' GB/s':<34}{mem_ms:>24.3f} ms")
    print(f"{'=> bandwidth ceiling':<34}{mem_ceiling_tps:>22.1f} tok/s")
    print(f"{'arithmetic time @ ' + str(DEFAULT_TFLOPS) + ' TFLOP/s':<34}{compute_ms:>24.3f} ms")
    print("-" * 64)
    print(f"{'% of bandwidth ceiling':<34}{measured_tps / mem_ceiling_tps * 100:>24.1f} %")
    print(f"{'memory : compute ratio':<34}{mem_ms / compute_ms:>24.0f} x")
    print("=" * 64)
    print()
    print(f"Interpretation: the GPU spends {mem_ms / compute_ms:.0f}x longer moving weights than")
    print("doing math. That idle headroom is exactly what speculative decoding spends.")

    if args.json_out:
        payload = {
            "model": args.model,
            "n_params": n_params,
            "weight_bytes": w_bytes,
            "latency_ms": s,
            "measured_tok_s": measured_tps,
            "bandwidth_ceiling_tok_s": mem_ceiling_tps,
            "pct_of_ceiling": measured_tps / mem_ceiling_tps * 100,
            "mem_compute_ratio": mem_ms / compute_ms,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
