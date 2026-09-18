"""Exact generated-token reference capture/compare for speculative
decoding's correctness gate (docs/design/2026-09-17-phase-5b-speculative-decoding.md
section 5). Unlike benchmark/reference.py's compare_top_k_agreement (built
for kernel-swap comparisons that may legitimately diverge a little),
speculative decoding's claim is that its *generated token ids* are
byte-identical to plain greedy decoding of the same target -- so this
compares exact integer sequences, not logits within a tolerance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def save_generated_tokens(tokens: Mapping[str, Sequence[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: list(value) for key, value in tokens.items()}, indent=2))


def load_generated_tokens(path: Path) -> dict[str, tuple[int, ...]]:
    raw: dict[str, list[int]] = json.loads(path.read_text())
    return {key: tuple(value) for key, value in raw.items()}


def compare_generated_tokens(
    actual: Mapping[str, Sequence[int]], reference: Mapping[str, Sequence[int]]
) -> dict[str, bool]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )
    # Normalize to tuple before comparing: Mapping/Sequence-typed inputs mean a
    # caller could pass a list on one side and a tuple on the other, which
    # Python's == would treat as unequal by container type alone, not content.
    return {key: tuple(actual[key]) == tuple(reference[key]) for key in reference}
