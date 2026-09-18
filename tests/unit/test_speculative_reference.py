"""CPU-only: exact generated-token save/load/compare -- speculative
decoding's correctness claim (docs/design/2026-09-17-phase-5b-speculative-decoding.md
section 5) is byte-exact token match, not a logit tolerance, so this is a
separate, stricter mechanism than benchmark/reference.py's
compare_top_k_agreement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.speculative.reference import (
    compare_generated_tokens,
    load_generated_tokens,
    save_generated_tokens,
)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    tokens = {"prompt_000_tokens": (1, 2, 3), "prompt_001_tokens": (4, 5)}
    path = tmp_path / "generated-tokens.json"

    save_generated_tokens(tokens, path)
    loaded = load_generated_tokens(path)

    assert loaded == tokens


def test_compare_reports_true_when_identical() -> None:
    tokens = {"prompt_000_tokens": (1, 2, 3)}

    assert compare_generated_tokens(tokens, tokens) == {"prompt_000_tokens": True}


def test_compare_reports_false_on_any_divergence() -> None:
    actual = {"prompt_000_tokens": (1, 2, 9)}
    reference = {"prompt_000_tokens": (1, 2, 3)}

    assert compare_generated_tokens(actual, reference) == {"prompt_000_tokens": False}


def test_compare_rejects_a_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        compare_generated_tokens({"a": (1,)}, {"b": (1,)})
