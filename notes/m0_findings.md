# M0 findings — decode is host/launch bound, not bandwidth bound

Date: 2026-09-10. RTX 5080 (sm_120), WSL2 Ubuntu 24.04, torch 2.11.0+cu128,
transformers 5.17.0. Clean GPU: Overwatch/Chrome closed, 14.5 GiB free, 2% util.

## The measurement

| model | weights | median ms/tok | tok/s | % of measured-BW ceiling |
|---|---|---|---|---|
| Qwen3-0.6B | 1.19 GB | 20.93 | 47.8 | 7.6 |
| Qwen3-1.7B | 3.44 GB | 19.04 | 52.5 | 24.1 |

Achieved copy bandwidth on this card: **749 GB/s** (78% of the 960 GB/s spec).
That is the honest denominator for every roofline number here.

## The finding

**Latency is invariant to model size.** 2.9x the weight bytes costs *less* wall
time. Under a bandwidth-bound roofline that is impossible — reading 3.44 GB
cannot be faster than reading 1.19 GB.

Supporting evidence: after the host finished enqueuing 128 decode steps
(2843 ms), the GPU needed only 7.6 ms more to drain. Host cost is ~22 ms/token.
For a 28-layer model that is roughly 500 kernel launches per token, each paying
Python dispatch + launch latency, amplified by WSL2's higher launch cost.

### Method caveat, stated honestly

`roofline.py` also reports "CPU share of wall". That statistic is **weaker than
it appears** and should not be quoted on its own: CUDA's launch queue is finite
(~1024 pending launches), so at ~500 kernels/token the queue fills after ~2
tokens and `cudaLaunchKernel` blocks. A genuinely GPU-bound loop would therefore
*also* read ~100%. The conclusion rests on the model-size sweep, not on it.

## Why this matters more than the number itself

Speculative decoding speedup is

    speedup = accept_len / (gamma * c + 1),    c = draft_cost / target_cost

The whole method assumes `c` is small — that a small draft is proportionally
cheaper because decode is bandwidth bound. Here the measured ratio is
20.93 / 19.04 = **c ~ 1.1**: the draft is not cheaper at all.

At gamma=4 with c=1.1, even *perfect* acceptance (accept_len = 5) gives
5 / 5.4 = **0.93x — a slowdown.** No draft model, however good, can rescue this.

## What is and is not blocked

- **Not blocked: acceptance length.** It is an algorithmic property — how often
  the draft's proposal matches the target's distribution — and is entirely
  independent of launch overhead. Prof. Liu's requested MATH-500 deliverable is
  measurable right now, in eager mode, and will be valid.
- **Blocked: wall-clock speedup.** Any speedup measured on this harness would be
  an artifact of Python overhead, not of the algorithm.

## Fixing it

`torch.compile(mode="reduce-overhead")` (CUDA graphs) is the intended fix. It
currently fails:

    InductorError: RuntimeError: Failed to find C compiler.

Triton JIT-compiles a C launcher stub; WSL has no `gcc`. Two paths:

1. **Install the toolchain** (needs sudo password, one time):
   `sudo apt update && sudo apt install -y build-essential`
   Then `torch.compile` works, but CUDA graphs need *static shapes*, so the
   growing `DynamicCache` must be replaced with `StaticCache`
   (`StaticCache(config, max_cache_len, ...)` in transformers 5.17).

2. **No-sudo fallback: capture the graph manually** with `torch.cuda.CUDAGraph()`
   and `torch.cuda.graph(...)`. This is pure PyTorch/CUDA runtime with no Triton
   codegen and therefore **no C compiler required**. Still needs `StaticCache`
   for fixed shapes.

## Consequences for the plan

- M1's from-scratch loop should target `StaticCache` from the start, not
  `DynamicCache` — the rollback then becomes an index/position reset rather than
  a `crop()`, and the loop stays CUDA-graph-capturable.
- Profiling (was M3) is no longer a late verification step. It has already
  changed the project's direction, and the launch-overhead fix should land
  before any wall-clock claim is made.
- `DynamicCache.crop()` does exist in 5.17, so the simpler dynamic-cache
  implementation remains available as a correctness reference to check the
  static-cache version against.
