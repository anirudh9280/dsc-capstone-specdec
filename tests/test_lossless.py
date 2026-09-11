"""The gate that blocks M2: speculative decoding must be exactly lossless.

Under greedy decoding the speculative loop must emit a token sequence IDENTICAL to
plain autoregressive decoding from the target. Not similar -- identical. The
algorithm guarantees it, so any divergence is a bug in the KV cache rollback, not
a tolerance to be tuned. This is the single most valuable test in the project:
cache desync does not crash, it silently corrupts attention, and without an exact
oracle the resulting acceptance numbers would look plausible and be wrong.

Two cases:

  1. draft IS target. Degenerate but sharp: q == p at every position, so every
     draft must be accepted, acceptance length must be exactly gamma+1, and any
     rejection means the verification indexing is misaligned.
  2. draft != target. The real configuration. Output must still match exactly.

  python tests/test_lossless.py          # readable report
  pytest tests/test_lossless.py          # same assertions
"""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.loop import (  # noqa: E402
    SpecDecConfig,
    autoregressive_generate,
    speculative_generate,
)

TARGET_ID = os.environ.get("SPECDEC_TARGET", "Qwen/Qwen3-1.7B")
DRAFT_ID = os.environ.get("SPECDEC_DRAFT", "Qwen/Qwen3-0.6B")
PROMPT = "Let f(x) = 3x + 2. Compute f(5), showing each step."
MAX_NEW = 48

_cache: dict[str, object] = {}


def _load(model_id: str):
    if model_id not in _cache:
        _cache[model_id] = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
    return _cache[model_id]


def _prompt_ids(tok):
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(text, return_tensors="pt").input_ids.to("cuda")


def test_identity_draft_equals_target():
    """draft == target: every proposal must be accepted."""
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    model = _load(TARGET_ID)
    ids = _prompt_ids(tok)

    gamma = 4
    baseline = autoregressive_generate(model, ids, MAX_NEW, temperature=0.0,
                                       eos_token_id=tok.eos_token_id)
    spec = speculative_generate(
        model, model, ids,
        SpecDecConfig(gamma=gamma, temperature=0.0, max_new_tokens=MAX_NEW),
        eos_token_id=tok.eos_token_id,
    )

    n = min(len(baseline), len(spec.token_ids))
    assert baseline[:n] == spec.token_ids[:n], "self-speculation diverged from greedy"

    rejects = [r.n_accepted for r in spec.rounds if r.n_accepted < gamma]
    assert not rejects, (
        f"draft==target must accept everything, saw rejections at {rejects}. "
        "Verification logits are misaligned with the drafted positions."
    )
    assert abs(spec.acceptance_length - (gamma + 1)) < 1e-6, (
        f"acceptance length {spec.acceptance_length:.3f}, expected exactly {gamma + 1}"
    )
    return spec


def test_lossless_distinct_models():
    """The real configuration: different draft, identical output."""
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    target, draft = _load(TARGET_ID), _load(DRAFT_ID)
    ids = _prompt_ids(tok)

    baseline = autoregressive_generate(target, ids, MAX_NEW, temperature=0.0,
                                       eos_token_id=tok.eos_token_id)
    spec = speculative_generate(
        target, draft, ids,
        SpecDecConfig(gamma=4, temperature=0.0, max_new_tokens=MAX_NEW),
        eos_token_id=tok.eos_token_id,
    )

    n = min(len(baseline), len(spec.token_ids))
    if baseline[:n] != spec.token_ids[:n]:
        for i, (a, b) in enumerate(zip(baseline[:n], spec.token_ids[:n])):
            if a != b:
                raise AssertionError(
                    f"diverged at token {i} (prompt was {ids.shape[1]} tokens): "
                    f"autoregressive={a} ({tok.decode([a])!r}) "
                    f"speculative={b} ({tok.decode([b])!r})"
                )
    return spec


def _tok_check(label, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main() -> int:
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"target {TARGET_ID}\ndraft  {DRAFT_ID}\n")

    print("=" * 62)
    print("1. draft == target (every proposal must be accepted)")
    print("=" * 62)
    ok = True
    try:
        spec = test_identity_draft_equals_target()
        print(f"  acceptance length : {spec.acceptance_length:.3f}  (expected 5.000)")
        print(f"  rounds            : {len(spec.rounds)}")
        ok &= _tok_check("identical output and zero rejections", True)
    except AssertionError as e:
        ok &= _tok_check(f"{e}", False)

    print()
    print("=" * 62)
    print("2. distinct draft and target (must still be exactly lossless)")
    print("=" * 62)
    try:
        spec = test_lossless_distinct_models()
        acc = spec.acceptance_length
        print(f"  acceptance length : {acc:.3f}  tokens per target forward pass")
        print(f"  mean accepted     : {spec.mean_accepted:.3f} of gamma=4")
        print(f"  rounds            : {len(spec.rounds)}")
        draft_ms = sum(r.draft_ms for r in spec.rounds)
        verify_ms = sum(r.verify_ms for r in spec.rounds)
        print(f"  draft / verify    : {draft_ms:.0f} ms / {verify_ms:.0f} ms "
              f"(c = {draft_ms / verify_ms / spec.rounds[0].gamma:.3f} per draft step)")
        ok &= _tok_check("token-identical to greedy autoregressive", True)
    except AssertionError as e:
        ok &= _tok_check(f"{e}", False)

    print()
    print("GATE PASSED -- M2 may proceed." if ok else "GATE FAILED -- fix before M2.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
