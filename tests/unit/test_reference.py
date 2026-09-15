"""Reference-logit capture is Phase 1's correctness oracle -- get the
comparison semantics right here, before any kernel exists to check."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dispatch.benchmark.harness import load_model
from dispatch.benchmark.reference import (
    capture_reference_logits,
    compare_within_tolerance,
    load_reference,
    save_reference,
)


def test_compare_within_tolerance_true_for_identical_tensors() -> None:
    tensors = {"a": torch.tensor([1.0, 2.0, 3.0])}

    assert compare_within_tolerance(tensors, tensors) == {"a": True}


def test_compare_within_tolerance_false_beyond_tolerance() -> None:
    actual = {"a": torch.tensor([1.0, 2.0, 3.0])}
    reference = {"a": torch.tensor([1.0, 2.0, 30.0])}

    assert compare_within_tolerance(actual, reference, rtol=1e-3, atol=1e-5) == {"a": False}


def test_compare_within_tolerance_raises_on_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        compare_within_tolerance({"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])})


def test_save_and_load_reference_round_trips(tmp_path: Path) -> None:
    tensors = {"prompt_000_logits": torch.randn(4, 10)}
    path = tmp_path / "reference.safetensors"

    save_reference(tensors, path)
    loaded = load_reference(path)

    assert loaded.keys() == tensors.keys()
    assert torch.equal(loaded["prompt_000_logits"], tensors["prompt_000_logits"])


@pytest.mark.slow
def test_capture_reference_logits_is_deterministic_for_fixed_prompts() -> None:
    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")

    first = capture_reference_logits(model, tokenizer, ["hello"])
    second = capture_reference_logits(model, tokenizer, ["hello"])

    assert compare_within_tolerance(first, second, rtol=0, atol=0) == {"prompt_000_logits": True}
