"""Gap-split agreement between two logit tensors: every top-1 disagreement is
classified by how close the *reference's* own top-1 and top-2 logits were.

Phase 5b found genuine near-tied logits (a top-2 gap of 0.1-0.4 out of
~20-magnitude logits; bf16's own spacing at that magnitude is 0.125) that
flip under the int8 kernel's floating-point precision and even between two
processes running the same config. A kernel swap may flip a near-tie. It may
not flip a position the reference was confident about: that is a bug, not
noise. `top1_agreement` alone cannot tell the two apart, and demanding 100%
agreement would fail a correct kernel, so the gate splits them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch

# Pre-registered 2026-09-19, before any Phase 6 GPU time (docs/plans/
# 2026-09-19-phase-6-final-benchmark-plan.md, "Pre-registered gate rule").
# 1.0 logit is 2.5x the widest near-tie gap Phase 5b measured (0.4) and 8
# bf16 spacings at ~20 magnitude. It is NOT to be loosened after seeing a
# run's numbers: a run that trips it is a finding to investigate, and any
# change needs a written amendment to the plan first. A test pins this value.
LARGE_GAP_THRESHOLD = 1.0

# Phase 5a's "perfect agreement" covered 29 positions; "at scale" here means
# at least this many compared positions per config.
MIN_GATE_POSITIONS = 500


@dataclass(frozen=True)
class GapSplitAgreement:
    positions: int
    disagreements: int
    near_tie_disagreements: int  # reference top1-top2 gap <= threshold
    large_gap_disagreements: int  # reference top1-top2 gap > threshold
    max_disagreement_gap: float  # widest reference gap among disagreements; 0.0 if none

    @property
    def top1_agreement(self) -> float:
        if self.positions == 0:
            raise ValueError("cannot compute top1_agreement over zero positions")
        return (self.positions - self.disagreements) / self.positions

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "top1_agreement": self.top1_agreement}


def compare_gap_split(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    threshold: float = LARGE_GAP_THRESHOLD,
) -> dict[str, GapSplitAgreement]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )
    return {key: _gap_split(actual[key], reference[key], threshold) for key in reference}


def aggregate_gap_split(per_prompt: Iterable[GapSplitAgreement]) -> GapSplitAgreement:
    items = list(per_prompt)
    if not items:
        raise ValueError("cannot aggregate zero prompts")
    return GapSplitAgreement(
        positions=sum(item.positions for item in items),
        disagreements=sum(item.disagreements for item in items),
        near_tie_disagreements=sum(item.near_tie_disagreements for item in items),
        large_gap_disagreements=sum(item.large_gap_disagreements for item in items),
        max_disagreement_gap=max(item.max_disagreement_gap for item in items),
    )


def _gap_split(
    actual: torch.Tensor, reference: torch.Tensor, threshold: float
) -> GapSplitAgreement:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: actual {tuple(actual.shape)}, reference {tuple(reference.shape)}"
        )
    if reference.shape[-1] < 2:  # noqa: PLR2004 -- a top-2 gap needs two logits
        raise ValueError("need at least two logits per position to measure a top-2 gap")
    reference_f = reference.float()
    top2 = reference_f.topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]
    disagree = actual.float().argmax(dim=-1) != reference_f.argmax(dim=-1)
    large = disagree & (gap > threshold)
    near_tie = disagree & ~large
    return GapSplitAgreement(
        positions=int(disagree.numel()),
        disagreements=int(disagree.sum()),
        near_tie_disagreements=int(near_tie.sum()),
        large_gap_disagreements=int(large.sum()),
        max_disagreement_gap=float(gap[disagree].max()) if bool(disagree.any()) else 0.0,
    )
