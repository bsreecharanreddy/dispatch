"""CPU-only, no model: PromptLookupDrafter's longest-suffix-match logic,
tested directly against constructed token sequences. Per
docs/design/2026-09-17-phase-5b-speculative-decoding.md section 3.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.speculative.drafters import PromptLookupDrafter


def test_proposes_whatever_followed_the_most_recent_matching_ngram() -> None:
    # [A, B, C, A, B] with ngram_size=2: the last two tokens are [A, B],
    # which also occurred at positions 0-1, followed there by [C].
    token_ids = torch.tensor([[10, 20, 30, 10, 20]])
    drafter = PromptLookupDrafter(ngram_size=2)

    proposed = drafter.propose(token_ids, num_tokens=1)

    assert proposed.tolist() == [[30]]


def test_proposes_up_to_num_tokens_from_the_matched_continuation() -> None:
    # [A, B, C, D, A, B] with ngram_size=2: [A, B] matched at positions
    # 0-1, followed there by [C, D] -- only 2 tokens available.
    token_ids = torch.tensor([[10, 20, 30, 40, 10, 20]])
    drafter = PromptLookupDrafter(ngram_size=2)

    proposed_all = drafter.propose(token_ids, num_tokens=2)
    proposed_truncated = drafter.propose(token_ids, num_tokens=1)

    assert proposed_all.tolist() == [[30, 40]]
    assert proposed_truncated.tolist() == [[30]]


def test_returns_fewer_than_num_tokens_when_the_match_is_near_the_end() -> None:
    # ngram_size=1: the last token (7) also occurred at position 2, where
    # it was followed by [3, 7] -- only 2 tokens, fewer than the 5 requested.
    token_ids = torch.tensor([[1, 2, 7, 3, 7]])
    drafter = PromptLookupDrafter(ngram_size=1)

    proposed = drafter.propose(token_ids, num_tokens=5)

    assert proposed.tolist() == [[3, 7]]


def test_returns_empty_when_no_earlier_match_exists() -> None:
    token_ids = torch.tensor([[10, 20, 30, 40, 50]])
    drafter = PromptLookupDrafter(ngram_size=2)

    proposed = drafter.propose(token_ids, num_tokens=3)

    assert proposed.shape == (1, 0)


def test_returns_empty_when_the_sequence_is_too_short_for_the_ngram() -> None:
    token_ids = torch.tensor([[10, 20]])
    drafter = PromptLookupDrafter(ngram_size=3)

    proposed = drafter.propose(token_ids, num_tokens=3)

    assert proposed.shape == (1, 0)


def test_returns_empty_for_a_non_positive_num_tokens() -> None:
    token_ids = torch.tensor([[10, 20, 30, 10, 20, 30]])
    drafter = PromptLookupDrafter(ngram_size=2)

    assert drafter.propose(token_ids, num_tokens=0).shape == (1, 0)


def test_rejects_an_ngram_size_below_one() -> None:
    with pytest.raises(ValueError, match="ngram_size"):
        PromptLookupDrafter(ngram_size=0)


def test_on_accepted_is_a_no_op() -> None:
    PromptLookupDrafter().on_accepted(accepted_len=2, rejected_len=1)
