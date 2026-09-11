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

---

# Addendum — the launch-overhead fix (same day, after `build-essential`)

With `gcc` installed, `torch.compile` runs. Two modes, very different outcomes.

## `mode="reduce-overhead"` (CUDA graphs) — does not work here

Two failures in sequence:

1. With `DynamicCache`: `RuntimeError: accessing tensor output of CUDAGraphs that
   has been overwritten by a subsequent run`. Graph trees reuse output buffers, and
   `DynamicCache` *stores* tensors produced inside the graph, so a replay silently
   clobbers the KV cache. This is the structural reason CUDA graphs need a
   preallocated, in-place cache.
2. With `StaticCache`: inductor logs `skipping cudagraphs due to mutated inputs
   (84 instances)`, traced to `self.cumulative_length.add_()` in
   `cache_utils.py:478` — StaticCache mutates its own buffers by design, which the
   cudagraph pass refuses to capture. The process then **segfaults (exit 139)**
   with no Python traceback.

Treat cudagraphs as unavailable on this stack (torch 2.11 + transformers 5.17 +
Qwen3 + sm_120) until proven otherwise. Not worth more time right now.

## `mode="default"` (inductor fusion only) — works, and is a large win

| config | ms/token | tok/s | % of measured-BW ceiling |
|---|---|---|---|
| Qwen3-0.6B eager | 20.93 | 47.8 | 7.6 |
| Qwen3-0.6B compiled | **6.185** | **161.7** | **25.7** |
| Qwen3-1.7B eager | 19.04 | 52.5 | 24.1 |
| Qwen3-1.7B compiled | **7.354** | **136.0** | **62.5** |

**3.4x on the draft, 2.6x on the target**, from kernel fusion alone — no CUDA
graphs. Fusion cuts the number of launches, which is what the bottleneck was.

The qualitative change matters more than the ratios: **latency now scales with
model size again** (6.19 -> 7.35 ms), and the 1.7B sits at 62.5% of its bandwidth
ceiling. The memory-bound premise is starting to hold for the target. The 0.6B at
25.7% is still overhead-dominated, which is expected — fixed per-step cost is a
larger share of a smaller model.

## What this does to the cost model

    c = 6.185 / 7.354 = 0.841      (was ~1.1 in eager)

    gamma=4:  4.194 / (4*0.841 + 1) = 0.96x   still losing, barely
    gamma=2:  2.783 / (2*0.841 + 1) = 1.04x   profitable

So the systems fix moves speculative decoding from "cannot win at any gamma" to
"wins at small gamma", and the unimodal speedup-vs-gamma curve predicted by theory
starts to appear. Worth re-running the MATH-500 sweep compiled to see the whole
curve rather than two points.

## The remaining problem is the model pair, not the systems

c = 0.841 is still far above the ~0.35 the weight ratio implies. The cause is now
visible and is a *choice*, not a bug: **Qwen3-0.6B against Qwen3-1.7B is only a
2.9x size ratio.** Production speculative decoding uses 10-30x (e.g. a 0.5B draft
against a 7B-32B target). A draft that is a third the size of its target cannot be
cheap enough, however fast the kernels are.

Next: measure Qwen3-4B compiled and recompute c for a 4B/0.6B pair (6.7x ratio).
That is the M3 model-pair decision, made from measurement rather than estimate.

---

# M3 model-pair decision (measured, not estimated)

Compiled (`--compile --compile-mode default`), clean GPU:

| model | weights | ms/token | tok/s | % of measured-BW ceiling |
|---|---|---|---|---|
| Qwen3-0.6B | 1.19 GB | 6.185 | 161.7 | 25.7 |
| Qwen3-1.7B | 3.44 GB | 7.354 | 136.0 | 62.5 |
| Qwen3-4B | 8.04 GB | 14.026 | 71.3 | **76.6** |

Latency now scales with weight bytes, and the 4B sits at **76.6% of its bandwidth
ceiling** -- for the target model, the memory-bound premise finally holds. The
0.6B at 25.7% is still overhead-dominated, which is exactly what you expect: a
fixed per-step cost is a larger fraction of a smaller model.

## Cost ratio across candidate pairs

    target        draft        c = draft_ms / target_ms
    Qwen3-1.7B    Qwen3-0.6B   6.185 / 7.354  = 0.841
    Qwen3-4B      Qwen3-0.6B   6.185 / 14.026 = 0.441

**Decision: Qwen3-4B target / Qwen3-0.6B draft.** VRAM 8.04 + 1.19 = 9.23 GB
against 14.6 GB free, comfortable with both KV caches.

Projected with the acceptance lengths measured on the 1.7B pair (which will be
somewhat optimistic -- a 0.6B and a 4B diverge more than a 0.6B and a 1.7B, so
acceptance should drop):

    gamma=2:  2.783 / (2*0.441 + 1) = 1.48x
    gamma=4:  4.194 / (4*0.441 + 1) = 1.52x

A real speedup, pending the measured acceptance length for this pair.

## Why this ordering of findings matters

Each fix exposed the next bottleneck, which is the method Prof. Liu described:

1. eager, 1.7B/0.6B  -> c ~ 1.1   -- host/launch bound; cannot win at any gamma
2. compiled, 1.7B/0.6B -> c = 0.841 -- launch cost cut 3.4x; wins at small gamma
3. compiled, 4B/0.6B  -> c = 0.441 -- pair ratio fixed; ~1.5x projected

Note that steps 2 and 3 are different KINDS of intervention. Step 2 is a systems
fix (kernel fusion). Step 3 is a configuration choice that was invisible until the
systems fix removed the overhead masking it. In eager mode every pair looked
equally bad, because fixed host cost dominated the weight-ratio signal entirely.

## Methodological note on combining these numbers

Acceptance length is a property of the model pair and the data -- it is identical
eager or compiled, because it only asks how often the draft's token matches the
target's. The cost ratio c is NOT config-independent. So the defensible procedure,
and the one `--assume-cost-ratio` implements, is:

  measure acceptance length wherever convenient; measure c under the configuration
  you intend to ship; compute speedup from both.

Reporting an eager-measured speedup would understate the method; reporting a
compiled speedup against eager-measured acceptance without saying so would be
sloppy. Both inputs are recorded in summary.json.
