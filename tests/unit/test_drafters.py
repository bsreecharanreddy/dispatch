"""CPU-only, no model: PromptLookupDrafter's longest-suffix-match logic,
tested directly against constructed token sequences. Per
docs/design/2026-09-17-phase-5b-speculative-decoding.md section 3.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.speculative.drafters import DraftModelDrafter, PromptLookupDrafter


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


class _FakeCache:
    """Tracks only a length -- the toy model below never reads cache
    *contents*, only its length, so a cache-rollback bug shows up as a
    wrong length, exactly the invariant this task's tests check."""

    def __init__(self, length: int) -> None:
        self.length = length

    def crop(self, tokens_to_remove: int) -> None:
        self.length -= tokens_to_remove


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor, past_key_values: _FakeCache) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeIncrementModel:
    """A toy causal LM: predicts (input_token + 1) % vocab_size at every
    position, deliberately ignoring any actual cache contents -- only
    call_count and cache length are used for assertions, not model
    quality."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size
        self.call_count = 0

    def __call__(
        self, *, input_ids: torch.Tensor, past_key_values: _FakeCache | None, use_cache: bool
    ) -> _FakeOutputs:
        self.call_count += 1
        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.nn.functional.one_hot(next_ids, self.vocab_size).float() * 10.0
        prior_length = 0 if past_key_values is None else past_key_values.length
        return _FakeOutputs(logits, _FakeCache(prior_length + input_ids.shape[1]))


def test_first_propose_call_feeds_the_whole_prompt() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]
    prompt = torch.tensor([[3, 4, 5]])

    proposed = drafter.propose(prompt, num_tokens=2)

    assert proposed.tolist() == [[6, 7]]
    assert model.call_count == 2
    assert drafter.past_key_values is not None
    # Iteration 1 feeds the whole 3-token prompt (cache was empty);
    # iteration 2 feeds only the 1 token iteration 1 just produced.
    assert drafter.past_key_values.length == 3 + 1  # type: ignore[attr-defined]


def test_later_propose_call_feeds_only_the_newest_token() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]
    drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=2)  # seeds the cache, length 3+1=4

    proposed = drafter.propose(torch.tensor([[3, 4, 5, 6, 7]]), num_tokens=2)

    assert proposed.tolist() == [[8, 9]]
    assert drafter.past_key_values is not None
    # Cache already holds a real KV entry (length 4), so both iterations of
    # this call feed exactly 1 new token each: 4 + 1 + 1 = 6.
    assert drafter.past_key_values.length == 4 + 1 + 1  # type: ignore[attr-defined]


def test_on_accepted_crops_the_cache_by_the_rejected_length() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]
    # Iteration 1 feeds the whole 3-token prompt; iterations 2-4 each feed
    # 1 token: cache length = 3 + 1 + 1 + 1 = 6.
    drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=4)

    drafter.on_accepted(accepted_len=1, rejected_len=3)

    assert drafter.past_key_values is not None
    assert drafter.past_key_values.length == 6 - 3  # type: ignore[attr-defined]


def test_on_accepted_before_any_propose_call_is_a_no_op() -> None:
    DraftModelDrafter(_FakeIncrementModel()).on_accepted(accepted_len=0, rejected_len=0)  # type: ignore[arg-type]


def test_propose_with_non_positive_num_tokens_returns_empty_without_calling_the_model() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]

    proposed = drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=0)

    assert proposed.shape == (1, 0)
    assert model.call_count == 0
