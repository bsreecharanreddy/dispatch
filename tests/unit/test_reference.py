"""Reference-logit capture is Phase 1's correctness oracle -- get the
comparison semantics right here, before any kernel exists to check."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dispatch.benchmark.harness import load_model
from dispatch.benchmark.reference import (
    TopKAgreement,
    capture_reference_logits,
    compare_top_k_agreement,
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


def test_compare_top_k_agreement_is_perfect_for_identical_logits() -> None:
    logits = {"p": torch.randn(4, 10)}

    assert compare_top_k_agreement(logits, logits) == {
        "p": TopKAgreement(positions=4, top1_agreement=1.0, mutual_top_k=True, max_abs_diff=0.0)
    }


def test_compare_top_k_agreement_tolerates_a_near_tie_flip() -> None:
    reference = {"p": torch.tensor([[5.0, 4.99, 3.0, 2.0, 1.0, 0.0]])}
    actual = {"p": torch.tensor([[4.99, 5.0, 3.0, 2.0, 1.0, 0.0]])}

    result = compare_top_k_agreement(actual, reference)["p"]

    assert result.top1_agreement == 0.0
    assert result.mutual_top_k


def test_compare_top_k_agreement_flags_an_argmax_outside_the_top_k() -> None:
    reference = {"p": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]])}
    actual = {"p": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 9.0]])}

    assert not compare_top_k_agreement(actual, reference)["p"].mutual_top_k


def test_compare_top_k_agreement_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_top_k_agreement({"p": torch.zeros(2, 6)}, {"p": torch.zeros(3, 6)})


@pytest.mark.slow
def test_capture_reference_logits_is_deterministic_for_fixed_prompts() -> None:
    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")

    first = capture_reference_logits(model, tokenizer, ["hello"])
    second = capture_reference_logits(model, tokenizer, ["hello"])

    assert compare_within_tolerance(first, second, rtol=0, atol=0) == {"prompt_000_logits": True}
