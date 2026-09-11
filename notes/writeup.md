# Speculative decoding: summer baseline

**Anirudh Annabathula** — DSC Capstone D38, Efficient AI (Prof. Zhijian Liu)
Track: inference-time acceleration via speculative decoding
Hardware: RTX 5080 (16 GB, Blackwell sm_120), WSL2 Ubuntu 24.04, torch 2.11.0+cu128,
transformers 5.17.0

Repo: `github.com/anirudh9280/dsc-capstone-specdec` (private)

---

## Summary

I set out to measure acceptance length on MATH-500 and found that I could not
translate it into a speedup, because decode on my setup was **host/launch bound
rather than memory-bandwidth bound**. The speculative-decoding cost model,
`speedup = accept_len / (γ·c + 1)`, assumes `c = draft_cost/target_cost` is small
because a smaller model reads fewer weight bytes. In eager HuggingFace I measured
`c ≈ 1.1`: one draft step on Qwen3-0.6B cost *more* than a full verify pass on
Qwen3-1.7B. At that cost ratio no acceptance length can produce a speedup —
even perfect acceptance at γ=4 gives 0.92×.

Fixing that, then re-profiling, exposed a second and different bottleneck:

| stage | intervention | c | speedup @ γ=4 |
|---|---|---|---|
| eager, 1.7B / 0.6B | — | ~1.10 | 0.92× (loses) |
| compiled, 1.7B / 0.6B | `torch.compile` fusion | 0.841 | 0.96× (loses) |
| compiled, 4B / 0.6B | pair ratio | **0.441** | **~1.5×** |

Acceptance length is high throughout — the *algorithm* was never the problem.
Everything above is systems and configuration.

---

## 1. Decode was not bandwidth bound

The cost model's premise makes a falsifiable prediction: per-token latency should
scale with weight bytes. It did not.

| model | weights | ms/token | tok/s | % of measured-BW ceiling |
|---|---|---|---|---|
| Qwen3-0.6B | 1.19 GB | 20.93 | 47.8 | 7.6 |
| Qwen3-1.7B | 3.44 GB | 19.04 | 52.5 | 24.1 |

2.9× the weight bytes cost *less* wall time. Reading 3.44 GB cannot be faster than
reading 1.19 GB, so bandwidth was not the limit. Host cost was ~22 ms/token —
roughly 500 kernel launches for a 28-layer model, each paying Python dispatch and
launch latency, amplified by WSL2.

Achieved copy bandwidth on this card is **749 GB/s**, 78% of the 960 GB/s spec;
that is the denominator used throughout rather than the spec figure.

The spec-dec loop confirmed this independently, from a different measurement path:
its per-round draft and verify timers gave `c = 1.08–1.15`.

## 2. Kernel fusion fixed it

`torch.compile(mode="default")` — inductor fusion, no CUDA graphs:

| model | eager | compiled | speedup | % of ceiling |
|---|---|---|---|---|
| Qwen3-0.6B | 20.93 | 6.185 | 3.4× | 7.6 → 25.7 |
| Qwen3-1.7B | 19.04 | 7.354 | 2.6× | 24.1 → 62.5 |
| Qwen3-4B | — | 14.026 | — | 76.6 |

Fusion reduces launch count, which was the actual bottleneck. The qualitative
change matters more than the ratios: **latency scales with weight bytes again**,
and the 4B reaches 76.6% of its bandwidth ceiling. For the target model the
memory-bound premise now holds.

`mode="reduce-overhead"` (CUDA graphs) does **not** work on this stack — see §5.

## 3. The pair ratio was the next bottleneck

With host overhead removed, a configuration problem became visible that had been
masked: Qwen3-0.6B against Qwen3-1.7B is only a **2.9× size ratio**, where
production speculative decoding uses 10–30×.

    target       draft        c
    Qwen3-1.7B   Qwen3-0.6B   0.841
    Qwen3-4B     Qwen3-0.6B   0.441   <- chosen

In eager mode every pair looked equally bad, because fixed per-step cost swamped
the weight-ratio signal entirely. The configuration error was undiscoverable until
the systems fix removed what was hiding it.

## 4. Acceptance length on MATH-500

*[TABLE PENDING — full sweep running: Qwen3-4B / Qwen3-0.6B, 25 problems,
γ ∈ {1,2,3,4,5,6,8}, 512 max new tokens, greedy]*

Preliminary, on the 1.7B / 0.6B pair (3 problems, 128 tokens):

| γ | acceptance length | of max | 
|---|---|---|
| 2 | 2.783 | 3 |
| 4 | 4.194 | 5 |

93% of theoretical maximum at γ=2. Draft and target share a family and tokenizer,
and MATH-500 contains long stretches of near-deterministic LaTeX, so the draft
agrees often.

**Method note.** Acceptance length is a property of the model pair and the data —
it only asks whether the draft's token matches the target's, so it is identical
eager or compiled. `c` is *not* config-independent. I therefore measure acceptance
wherever convenient, measure `c` under the configuration I intend to ship, and
combine. Both inputs are recorded rather than folded into a single number.

## 5. What does not work here

**CUDA graphs.** Two distinct failures:

- With `DynamicCache`: `accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run`. Graph trees reuse output buffers while
  `DynamicCache` stores tensors produced inside the graph, so replay clobbers the
  KV cache. This is the structural reason graphs need a preallocated cache.
- With `StaticCache`: inductor skips graphs (`mutated inputs (84 instances)`, from
  `cumulative_length.add_()` in `cache_utils.py:478` — StaticCache mutates its own
  buffers by design), then the process **segfaults, exit 139**, no traceback.

I stopped after two attempts rather than spend the week on it. This is the main
thing I would like guidance on (§7).

## 6. Correctness

Speculative decoding is exactly lossless, which gives a binary test rather than a
tolerance. `tests/test_lossless.py`:

1. **draft == target** — acceptance length exactly 5.000 = γ+1 at γ=4, zero
   rejections. Catches misalignment between verification logits and drafted
   positions.
2. **draft != target** — output token-identical to plain greedy decoding from the
   target.

This mattered. KV-cache desync does not crash; it silently corrupts attention and
yields acceptance numbers that look plausible and are wrong.

Two of my own measurement errors, both caught and both recorded:

- A "GPU busy fraction" built from CUDA events is **tautological** — the event pair
  brackets host gaps too, so it reads ~100% whether the GPU is saturated or
  starved. CUDA's finite launch queue also makes a genuinely GPU-bound loop report
  ~100% host share. The model-size sweep is what actually discriminates, and the
  conclusions rest on it.
- Deriving draft entropy and top-1 margin from the *sampling* distribution made
  them constant under greedy decoding (`_distribution` returns a one-hot, so
  entropy ≡ 0, margin ≡ 1). Those are the exact features an adaptive-γ scheduler
  would use, so M4 would have concluded — wrongly — that draft confidence carries
  no information. Fixed to use the raw softmax, which is also what a scheduler
  observes at decision time. With the fix: mean entropy 0.171 for accepted vs
  0.796 for rejected tokens, a 4.6× separation.

## 7. Questions

1. **CUDA graphs on Blackwell.** Does the group run `reduce-overhead` / cudagraphs
   on sm_120, and on what torch + transformers combination? The draft is still at
   only 25.7% of its bandwidth ceiling while the 4B target is at 76.6%, so the
   remaining headroom is concentrated exactly where graphs would help most — a
   small model whose per-step cost is dominated by fixed overhead.

2. **Which regime is adaptive γ really for?** The oracle's advantage comes from not
   wasting draft steps, so its headroom scales with `c`. At `c = 0.441` wasted
   drafts are comparatively cheap and the ceiling may be modest, whereas at
   `c ≈ 1` it was large but the method lost outright. Is the group's interest in
   adaptive scheduling aimed at the high-`c` regime — EAGLE-style heads with tree
   drafting, where many candidates are proposed per step — rather than a
   small-independent-draft setup like mine? That changes what I should be
   optimizing, and whether I should move to a draft head before studying γ.

3. **Which lever has more room?** At 4B/0.6B, `c = 0.441` is still 3× the weight
   ratio (0.148), because the draft is overhead-bound rather than bandwidth-bound.
   Three options: a smaller draft (0.6B is already the smallest Qwen3), an
   EAGLE-style single-layer head (far cheaper per step, and trainable on my
   hardware), or accept `c` and push acceptance length instead. I lean toward the
   draft head, since it attacks `c` structurally rather than incrementally — but
   that is a guess about where the group's work is headed.

---

## Reproducing

```bash
bash scripts/00_install_miniconda.sh
bash scripts/01_create_env.sh          # conda-forge only, avoids Anaconda ToS
conda activate specdec
python scripts/02_verify_gpu.py        # gates on sm_120 + real work, not is_available()

python bench/roofline.py --model Qwen/Qwen3-4B --compile --compile-mode default
python tests/test_lossless.py
python bench/run_math500.py --target Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B \
    --n-problems 25 --gammas 1 2 3 4 5 6 8 --baseline --assume-cost-ratio 0.441
python analysis/oracle.py
python analysis/signals.py --tokenizer Qwen/Qwen3-4B
```

Two environment notes that cost real time: sm_120 needs cu128+ wheels (a wheel
without Blackwell kernels imports fine and `torch.cuda.is_available()` returns
True, then dies at launch), and the HF cache and conda env must live on WSL ext4,
never `/mnt/c`.
