"""Speculative decoding, written from scratch so every round is instrumentable.

Deliberately NOT `model.generate(assistant_model=...)`. The whole project depends
on per-round telemetry -- which tokens were proposed, which were accepted, what the
draft and target probabilities were at each position -- and on being able to swap
the draft-length policy. None of that is reachable through the library path.

Algorithm: Leviathan et al. (arXiv:2211.17192), Algorithm 1. Also derived
independently in Chen et al. (arXiv:2302.01318), which has the cleaner proof.

The correctness property that makes this worth doing: the accepted output is
distributed EXACTLY as if sampled from the target alone. Not approximately.
`tests/test_lossless.py` gates on that.

---------------------------------------------------------------------------
KV cache bookkeeping (the part that is actually hard)
---------------------------------------------------------------------------
`tokens` is the running sequence. `t_cached` / `d_cached` count how many tokens'
keys and values currently sit in the target / draft caches.

Round invariant, checked on entry:   t_cached == len(tokens) - 1

That "one behind" is not an off-by-one; it is load-bearing. Verification needs
gamma+1 distributions from ONE target forward pass:

    p_1     ... the distribution over the 1st drafted token
    p_gamma ... the distribution over the last drafted token
    p_{gamma+1} ... the bonus distribution, used when every draft is accepted

A transformer emits the distribution for position j+1 from the hidden state at
position j. So to get p_1 we must re-feed the last already-accepted token, and to
get p_{gamma+1} we must feed the last drafted token. That is gamma+1 inputs, which
is exactly `tokens[t_cached:]` when the cache sits one token behind.

On rejection at position k, positions k..gamma-1 in both caches are garbage: they
were computed conditioned on tokens that are now discarded. `crop()` truncates
them. Forgetting this does not crash -- it silently corrupts attention for the
rest of the sequence, which is why the greedy identity test exists.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class SpecDecConfig:
    gamma: int = 4
    temperature: float = 0.0  # 0.0 => greedy (and the losslessness test is exact)
    max_new_tokens: int = 256
    seed: int | None = None


@dataclass
class RoundTrace:
    """One draft-then-verify round. M4's offline analysis runs entirely on these."""

    gamma: int
    n_accepted: int
    draft_token_ids: list[int]
    emitted_token_ids: list[int]
    all_accepted: bool
    draft_ms: float
    verify_ms: float
    # Per drafted position: the draft's and target's probability of that token,
    # plus the draft's entropy and top-1 margin. These are the candidate features
    # for an adaptive-gamma scheduler, logged now so M4 needs no new GPU runs.
    q_of_drafted: list[float] = field(default_factory=list)
    p_of_drafted: list[float] = field(default_factory=list)
    draft_entropy: list[float] = field(default_factory=list)
    draft_top1_margin: list[float] = field(default_factory=list)


@dataclass
class GenerationResult:
    token_ids: list[int]
    rounds: list[RoundTrace]
    wall_ms: float

    @property
    def n_generated(self) -> int:
        return sum(len(r.emitted_token_ids) for r in self.rounds)

    @property
    def acceptance_length(self) -> float:
        """Mean tokens emitted per target forward pass -- THE metric for this track.

        Hardware independent by construction, which is why it is the number the
        professor asked for rather than tokens/sec.
        """
        return self.n_generated / len(self.rounds) if self.rounds else 0.0

    @property
    def mean_accepted(self) -> float:
        return sum(r.n_accepted for r in self.rounds) / len(self.rounds) if self.rounds else 0.0

    @property
    def cost_ratio(self) -> float:
        """c = cost of ONE draft step / cost of one target verify pass.

        The parameter the whole method lives or dies on. Speculative decoding
        assumes c is small because decode is bandwidth bound and the draft is
        smaller. When the loop is launch bound instead, c approaches (or exceeds)
        1 and no acceptance length can produce a speedup.
        """
        verify = sum(r.verify_ms for r in self.rounds)
        draft = sum(r.draft_ms for r in self.rounds)
        gamma = self.rounds[0].gamma if self.rounds else 1
        return (draft / gamma) / verify if verify > 0 else float("inf")

    @property
    def analytic_speedup(self) -> float:
        """accept_len / (gamma * c + 1) -- expected speedup vs plain decoding.

        Below 1.0 means speculative decoding is actively losing, regardless of how
        well the draft model predicts.
        """
        gamma = self.rounds[0].gamma if self.rounds else 1
        return self.acceptance_length / (gamma * self.cost_ratio + 1)


def _distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Row of logits -> probability vector. Greedy is represented as a one-hot,
    so accept/reject can share one code path with sampling."""
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out[logits.argmax(-1)] = 1.0
        return out
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    if probs.dtype != torch.float32:
        probs = probs.float()
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _entropy(p: torch.Tensor) -> float:
    p = p.float()
    nz = p > 0
    return float(-(p[nz] * p[nz].log()).sum())


def _top1_margin(p: torch.Tensor) -> float:
    if p.numel() < 2:
        return 1.0
    top2 = torch.topk(p.float(), 2).values
    return float(top2[0] - top2[1])


def _crop_to(cache: DynamicCache, keep: int) -> None:
    """Truncate `cache` to `keep` tokens.

    transformers 5.17 deprecates positive `crop()` (removed in 5.18) in favour of
    a negative argument meaning "drop this many from the end", so pass the delta.
    """
    current = cache.get_seq_length()
    if current > keep:
        cache.crop(keep - current)


@torch.inference_mode()
def speculative_generate(
    target,
    draft,
    input_ids: torch.Tensor,
    cfg: SpecDecConfig,
    eos_token_id: int | None = None,
) -> GenerationResult:
    """Generate from `target`, using `draft` to propose. Output is distributionally
    identical to sampling from `target` alone.

    `input_ids` is [1, prompt_len]; batch size 1 only, which is the regime the
    whole speculative-decoding cost model is about.
    """
    device = input_ids.device
    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device=device).manual_seed(cfg.seed)

    tokens: list[int] = input_ids[0].tolist()
    prompt_len = len(tokens)

    target_cache, draft_cache = DynamicCache(), DynamicCache()

    # Prefill both models on everything except the final prompt token, establishing
    # the "one behind" invariant that verification depends on.
    prefill = input_ids[:, :-1]
    if prefill.shape[1] > 0:
        target(input_ids=prefill, past_key_values=target_cache, use_cache=True)
        draft(input_ids=prefill, past_key_values=draft_cache, use_cache=True)
    t_cached = d_cached = prompt_len - 1

    rounds: list[RoundTrace] = []
    wall_t0 = time.perf_counter()
    hit_eos = False

    while len(tokens) - prompt_len < cfg.max_new_tokens and not hit_eos:
        round_start_len = len(tokens)
        assert t_cached == round_start_len - 1, (
            f"target cache desync: {t_cached} vs {round_start_len - 1}"
        )

        # ---------------- draft: gamma cheap autoregressive steps -------------
        q_dists: list[torch.Tensor] = []
        torch.cuda.synchronize()
        d_t0 = time.perf_counter()
        for _ in range(cfg.gamma):
            inp = torch.tensor([tokens[d_cached:]], device=device)
            out = draft(input_ids=inp, past_key_values=draft_cache, use_cache=True)
            d_cached = len(tokens)
            q = _distribution(out.logits[0, -1], cfg.temperature)
            q_dists.append(q)
            tokens.append(_sample(q, gen))
        torch.cuda.synchronize()
        draft_ms = (time.perf_counter() - d_t0) * 1000.0

        drafted = tokens[round_start_len:]

        # ---------------- verify: ONE target pass over gamma+1 positions ------
        torch.cuda.synchronize()
        v_t0 = time.perf_counter()
        inp = torch.tensor([tokens[t_cached:]], device=device)
        out = target(input_ids=inp, past_key_values=target_cache, use_cache=True)
        torch.cuda.synchronize()
        verify_ms = (time.perf_counter() - v_t0) * 1000.0
        t_cached = len(tokens)

        # out.logits[0, j] is the distribution over tokens[t_cached_old + 1 + j].
        p_dists = [_distribution(out.logits[0, j], cfg.temperature) for j in range(cfg.gamma + 1)]

        # ---------------- accept / reject (Leviathan Algorithm 1) -------------
        n_accepted = 0
        correction: int | None = None
        q_of, p_of, ents, margins = [], [], [], []

        for i in range(cfg.gamma):
            x = drafted[i]
            q_x = float(q_dists[i][x])
            p_x = float(p_dists[i][x])
            q_of.append(q_x)
            p_of.append(p_x)
            ents.append(_entropy(q_dists[i]))
            margins.append(_top1_margin(q_dists[i]))

            if cfg.temperature <= 0.0:
                accept = x == int(p_dists[i].argmax())
            else:
                r = torch.rand(1, device=device, generator=gen).item()
                accept = r < min(1.0, p_x / q_x) if q_x > 0 else False

            if accept:
                n_accepted += 1
                continue

            # Rejected. Resample from the normalized residual max(0, p - q) so the
            # round's output still matches the target distribution exactly.
            if cfg.temperature <= 0.0:
                correction = int(p_dists[i].argmax())
            else:
                resid = torch.clamp(p_dists[i] - q_dists[i], min=0.0)
                total = float(resid.sum())
                correction = _sample(resid / total, gen) if total > 0 else _sample(p_dists[i], gen)
            break

        # ---------------- commit, then roll the caches back -------------------
        if correction is None:  # every draft accepted -> free bonus token
            bonus = _sample(p_dists[cfg.gamma], gen)
            del tokens[round_start_len + n_accepted:]
            tokens.append(bonus)
        else:
            del tokens[round_start_len + n_accepted:]
            tokens.append(correction)

        emitted = tokens[round_start_len:]

        # Everything past the accepted prefix was conditioned on discarded tokens.
        keep = len(tokens) - 1
        _crop_to(target_cache, keep)
        _crop_to(draft_cache, keep)
        t_cached = min(t_cached, keep)
        d_cached = min(d_cached, keep)

        rounds.append(
            RoundTrace(
                gamma=cfg.gamma,
                n_accepted=n_accepted,
                draft_token_ids=drafted,
                emitted_token_ids=emitted,
                all_accepted=correction is None,
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                q_of_drafted=q_of,
                p_of_drafted=p_of,
                draft_entropy=ents,
                draft_top1_margin=margins,
            )
        )

        if eos_token_id is not None and eos_token_id in emitted:
            hit_eos = True

    return GenerationResult(
        token_ids=tokens,
        rounds=rounds,
        wall_ms=(time.perf_counter() - wall_t0) * 1000.0,
    )


@torch.inference_mode()
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Plain one-token-at-a-time decoding. The reference for the losslessness test."""
    device = input_ids.device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    tokens: list[int] = input_ids[0].tolist()
    cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    for _ in range(max_new_tokens):
        probs = _distribution(out.logits[0, -1], temperature)
        nxt = _sample(probs, gen)
        tokens.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            use_cache=True,
        )
    return tokens
