from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.agreement import (
    LARGE_GAP_THRESHOLD,
    MIN_GATE_POSITIONS,
    GapSplitAgreement,
    aggregate_gap_split,
    compare_gap_split,
)


def _row(top1: float, top2: float, *, winner: int) -> torch.Tensor:
    """One 5-token-vocab position whose two largest logits are top1 and top2;
    `winner` is which of columns 0/1 holds the larger value."""
    row = torch.zeros(5)
    row[winner], row[1 - winner] = top1, top2
    return row


def _logits(*rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"prompt_000_logits": torch.stack(rows)}


def test_threshold_and_minimum_positions_are_the_preregistered_values() -> None:
    # Pre-registered before any Phase 6 GPU time; changing either needs a
    # written plan amendment first, not an edit that makes a red run green.
    assert LARGE_GAP_THRESHOLD == 1.0
    assert MIN_GATE_POSITIONS == 500


def test_identical_logits_have_no_disagreements() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0), _row(9.0, 8.5, winner=1))

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert result.positions == 2
    assert result.disagreements == 0
    assert result.top1_agreement == 1.0
    assert result.max_disagreement_gap == 0.0


def test_a_flip_at_a_tiny_gap_is_a_near_tie() -> None:
    reference = _logits(_row(20.0, 19.875, winner=0))  # one bf16 spacing at ~20
    actual = _logits(_row(20.0, 19.875, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (1, 0)
    assert result.max_disagreement_gap == pytest.approx(0.125)


def test_a_flip_at_a_large_gap_is_reported_separately() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0))
    actual = _logits(_row(10.0, 4.0, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (0, 1)
    assert result.max_disagreement_gap == pytest.approx(6.0)


def test_a_gap_exactly_at_the_threshold_is_a_near_tie() -> None:
    reference = _logits(_row(5.0 + LARGE_GAP_THRESHOLD, 5.0, winner=0))
    actual = _logits(_row(5.0 + LARGE_GAP_THRESHOLD, 5.0, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (1, 0)


def test_agreements_do_not_count_as_disagreements_whatever_their_gap() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0), _row(10.0, 9.9, winner=0))

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert (result.disagreements, result.large_gap_disagreements) == (0, 0)


def test_half_precision_inputs_are_compared_in_float32() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0)).copy()
    reference = {key: value.to(torch.bfloat16) for key, value in reference.items()}

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert result.disagreements == 0


def test_aggregate_sums_counts_and_keeps_the_widest_gap() -> None:
    first = GapSplitAgreement(
        positions=10,
        disagreements=2,
        near_tie_disagreements=2,
        large_gap_disagreements=0,
        max_disagreement_gap=0.3,
    )
    second = GapSplitAgreement(
        positions=30,
        disagreements=1,
        near_tie_disagreements=0,
        large_gap_disagreements=1,
        max_disagreement_gap=4.0,
    )

    total = aggregate_gap_split([first, second])

    assert (total.positions, total.disagreements) == (40, 3)
    assert (total.near_tie_disagreements, total.large_gap_disagreements) == (2, 1)
    assert total.max_disagreement_gap == 4.0
    assert total.top1_agreement == pytest.approx(37 / 40)


def test_to_dict_carries_the_derived_agreement_rate() -> None:
    result = GapSplitAgreement(
        positions=4,
        disagreements=1,
        near_tie_disagreements=1,
        large_gap_disagreements=0,
        max_disagreement_gap=0.1,
    )

    assert result.to_dict()["top1_agreement"] == 0.75


def test_aggregate_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="zero prompts"):
        aggregate_gap_split([])


def test_mismatched_keys_and_shapes_raise() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0))
    with pytest.raises(ValueError, match="key mismatch"):
        compare_gap_split({"other": reference["prompt_000_logits"]}, reference)
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_gap_split({"prompt_000_logits": torch.zeros(2, 5)}, reference)
