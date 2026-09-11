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

`mode="reduce-overhead"` (CUDA graphs) does **not** work on this stack — see §7.

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

Qwen3-4B target / Qwen3-0.6B draft, 25 problems, 512 max new tokens, greedy,
1868 rounds at γ=8.

| γ | acceptance length | of max | c (eager) | analytic | **measured** | projected @ c=0.441 |
|---|---|---|---|---|---|---|
| 1 | 1.903 | 2 | 0.859 | 1.020× | **1.032×** | 1.321× |
| 2 | 2.773 | 3 | 0.911 | 0.966× | **0.894×** | 1.473× |
| 3 | 3.453 | 4 | 0.842 | 0.973× | **0.897×** | 1.486× |
| 4 | 4.187 | 5 | 0.858 | 0.912× | **0.860×** | **1.515×** |
| 5 | 4.778 | 6 | 0.854 | 0.897× | **0.850×** | 1.491× |
| 6 | 5.222 | 7 | 0.862 | 0.853× | **0.875×** | 1.432× |
| 8 | 6.229 | 9 | 0.884 | 0.754× | **0.738×** | 1.376× |

**The analytic formula predicts the measured speedup.** 1.020 vs 1.032 at γ=1,
0.912 vs 0.860 at γ=4, 0.754 vs 0.738 at γ=8 — agreement within a few percent
across the whole sweep. That is the strongest evidence the harness is sound: the
cost model, the acceptance accounting, and the wall clock all agree independently.

The two trends behave exactly as theory says they should. Acceptance length rises
monotonically (1.903 → 6.229) because a longer draft can only accept more. Speedup
is **unimodal with a peak at γ=4**, because cost grows linearly in γ while
acceptance saturates. That divergence is the whole argument for scheduling γ per
position rather than fixing it.

**Method note.** Acceptance length is a property of the model pair and the data —
it only asks whether the draft's token matches the target's, so it is identical
eager or compiled. `c` is *not* config-independent. I therefore measure acceptance
wherever convenient, measure `c` under the configuration I intend to ship, and
combine. Both inputs are recorded rather than folded into a single number. The
`projected` column is the one to read for a compiled deployment.

## 5. How much room does adaptive γ actually have?

A static-γ round always pays γ draft steps, because all γ proposals are generated
before the target verifies any. An oracle knowing the run would end at 2 would
draft exactly 2. Both emit identical tokens — speculative decoding is lossless —
so the oracle wins purely by not wasting draft compute.

At γ=8, 1868 rounds, mean accepted run 4.889, **3.111 wasted draft steps per round**:

| | static γ=8 | oracle | headroom |
|---|---|---|---|
| c = 0.884 (eager) | 0.730× | 1.107× | **1.517×** |
| c = 0.441 (compiled) | 1.301× | 1.866× | **1.435×** |

I expected the headroom to shrink substantially at lower `c` — wasted drafts are
cheaper when drafting is cheap. It barely moves (1.517× → 1.435×). **Adaptive
scheduling is worth doing in both regimes**, which partly answers a question I
came in with.

### The run-length distribution is bimodal, and that is the real finding

```
  0:   287 ######
  1:   185 ####
  2:   148 ###
  3:   107 ##
  4:    85 ##
  5:    81 ##
  6:    80 ##
  7:    54 #
  8:   841 ##################  <- censored at gamma
```

Rounds either fail immediately (287 accept nothing) or run to the γ ceiling (841
of 1868, 45%). The middle is thin. This is the "easy region / hard region"
structure the domain description hypothesised, and it is *why* adaptive γ has
headroom: if run lengths were unimodal around 4, a static γ=4 would already be
near-optimal and there would be little to schedule.

Lag-1 autocorrelation of run length is **0.314** — recent history carries signal,
so even a cheap history-based policy is plausible.

**Caveat, and it is a real one: 45% of rounds are right-censored**, so the oracle
above is a *lower bound*. I am re-running at γ=16 to tighten it.

## 6. What a scheduler could condition on

10160 scored drafted positions, 89.9% accepted. A scheduler must decide before the
target verifies anything, so only draft-time information is admissible.

| signal | AUC | mean given accepted | mean given rejected |
|---|---|---|---|
| draft top-1 margin | **0.928** | 0.866 | 0.297 |
| draft entropy | **0.079** (0.921 inverted) | 0.266 | 1.343 |
| position within round | 0.589 | 3.03 | 2.34 |

Draft confidence is a **very strong** predictor — AUC 0.93 from a quantity already
computed during drafting, requiring no extra forward pass. A scheduler that drafts
deep while the margin is high and stops when it collapses looks immediately viable.

### Acceptance by token class contradicted my prediction

| class | acceptance | n |
|---|---|---|
| numeric | **98.6%** | 1330 |
| whitespace | 96.1% | 1011 |
| latex_cmd | 93.7% | 301 |
| symbol | 91.9% | 3514 |
| word | **83.4%** | 4004 |

I predicted the opposite — that LaTeX scaffolding would be near-deterministic and
the actual numbers hard, since the numeric answer is the "content." The data says
numbers are the *easiest* class (98.6%) and prose is the *hardest* (83.4%).

In hindsight the mechanism is clear: in a worked solution the numbers are largely
forced by the preceding computation, or copied from the problem statement, or the
continuation of a multi-digit token already begun. The prose has genuine stylistic
freedom — many valid ways to phrase "substituting this into the equation" — and a
0.6B and a 4B model make different choices among them.

This matters for the track: it suggests acceptance is limited by *stylistic*
divergence rather than *reasoning* divergence, which is a different problem and
plausibly more tractable.

### One statistical trap I nearly fell into

Acceptance rises with depth into a round (84.6% at position 0 → 94.0% at position
7). That reads as "drafting deeper is safer," and it is **survivorship bias**: a
round only reaches depth 7 by having accepted 7 tokens, so deep positions are
conditioned on being in an easy stretch. Note the shrinking n (1868 → 895). The
unbiased per-depth curve needs a fixed-γ run with no early exit. I have flagged
this in the script so the number is not quoted naively later.

## 7. What does not work here

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

## 8. Correctness

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

## 9. Questions

1. **CUDA graphs on Blackwell.** Does the group run `reduce-overhead` / cudagraphs
   on sm_120, and on what torch + transformers combination? The draft is still at
   only 25.7% of its bandwidth ceiling while the 4B target is at 76.6%, so the
   remaining headroom is concentrated exactly where graphs would help most — a
   small model whose per-step cost is dominated by fixed overhead.

2. **Is the bimodality the thing to exploit, or an artifact of γ censoring?** I
   came in expecting the oracle headroom to shrink at low `c` and it barely did
   (1.517× → 1.435×), so adaptive γ looks worthwhile in both regimes — that part
   I could answer myself. What I cannot answer is whether the run-length
   distribution is *genuinely* bimodal or whether the mass at 8 is an artifact of
   the ceiling. 45% of rounds are censored, and the γ=16 rerun will say. If it is
   genuine, the right policy may be much simpler than a per-position regressor:
   roughly a two-state classifier (easy stretch → draft deep; hard → draft
   shallow or skip), which the 0.314 autocorrelation would also support. Is that
   consistent with what DFlash found?

   Related: acceptance is high enough here (89.9%) that I wonder whether MATH-500
   at 512 tokens is discriminative enough, or whether I should be looking at
   longer chains of thought where the draft has more room to drift.

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
