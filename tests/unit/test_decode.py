"""CPU-only correctness of the shared speculative-decoding loop, against
a toy causal LM whose predictions are a pure, memoryless function of the
current input token -- deliberately so that a cache-rollback bug is
caught only by the explicit cache-length assertions below, never by a
lucky output-correctness coincidence. Per
docs/design/2026-09-17-phase-5b-speculative-decoding.md sections 3-5:
speculative decoding's output must be byte-identical to plain sequential
greedy decoding, regardless of whether the drafter is ever right.
"""

from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dispatch.benchmark.harness import load_model
from dispatch.speculative.decode import (
    plain_greedy_generate,
    run_speculative_rounds,
    speculative_generate,
)
from dispatch.speculative.drafters import DraftModelDrafter, PromptLookupDrafter

VOCAB_SIZE = 16


class _FakeBatchEncoding(dict):  # type: ignore[type-arg]
    def to(self, device: str) -> _FakeBatchEncoding:
        return self


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def crop(self, tokens_to_remove: int) -> None:
        # transformers>=5.17.0 uses negative integers to remove tokens
        # (e.g., crop(-2) removes the last 2 tokens)
        self.length += tokens_to_remove

    def get_seq_length(self) -> int:
        # Real transformers.Cache objects implement this; exercising it here
        # (rather than relying on the hasattr guard to skip it) is what
        # makes run_speculative_rounds's cache-lag assertion (finding I3,
        # final review) actually run against these CPU-only tests.
        return self.length


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor, past_key_values: _FakeCache) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeIncrementModel:
    """logits at every position spike at (that position's input token + 1)
    % VOCAB_SIZE, regardless of past_key_values' contents -- so the
    expected greedy continuation of any prompt is the closed-form
    +1-mod-VOCAB_SIZE sequence computed by _expected_continuation below."""

    def __call__(
        self, *, input_ids: torch.Tensor, past_key_values: _FakeCache | None, use_cache: bool
    ) -> _FakeOutputs:
        next_ids = (input_ids + 1) % VOCAB_SIZE
        logits = torch.nn.functional.one_hot(next_ids, VOCAB_SIZE).float() * 10.0
        prior_length = 0 if past_key_values is None else past_key_values.length
        return _FakeOutputs(logits, _FakeCache(prior_length + input_ids.shape[1]))


class _AlwaysCorrectDrafter:
    """Proposes exactly what _FakeIncrementModel will predict."""

    def propose(self, token_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        last = int(token_ids[0, -1])
        return torch.tensor([[(last + 1 + i) % VOCAB_SIZE for i in range(num_tokens)]])

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        pass


class _AlwaysWrongDrafter:
    """Proposes a constant offset that can never match _FakeIncrementModel's
    +1 rule (VOCAB_SIZE > 2, so +2 != +1 mod VOCAB_SIZE)."""

    def propose(self, token_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        last = int(token_ids[0, -1])
        return torch.tensor([[(last + 2) % VOCAB_SIZE for _ in range(num_tokens)]])

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        pass


def _expected_continuation(prompt: list[int], num_tokens: int) -> list[int]:
    sequence = list(prompt)
    for _ in range(num_tokens):
        sequence.append((sequence[-1] + 1) % VOCAB_SIZE)
    return sequence[len(prompt) :]


def test_always_correct_drafter_matches_the_closed_form_continuation() -> None:
    prompt = [3, 4, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _AlwaysCorrectDrafter(),
        torch.tensor([prompt]),
        num_speculative_tokens=4,
        max_new_tokens=10,
        eos_token_id=None,
    )

    assert list(result.generated_token_ids) == _expected_continuation(prompt, 10)
    # 2 rounds of k=4, each fully accepted (5 tokens emitted per round: 4
    # candidates + 1 bonus), summing to the 10 requested.
    assert result.accepted_lengths == (4, 4)


def test_always_wrong_drafter_still_matches_the_closed_form_continuation() -> None:
    """Correctness must hold even when the drafter is never right -- the
    target's own bonus/correction token is what actually advances the
    sequence every round."""
    prompt = [3, 4, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _AlwaysWrongDrafter(),
        torch.tensor([prompt]),
        num_speculative_tokens=4,
        max_new_tokens=10,
        eos_token_id=None,
    )

    assert list(result.generated_token_ids) == _expected_continuation(prompt, 10)
    assert all(length == 0 for length in result.accepted_lengths)


def test_none_drafter_matches_the_closed_form_continuation() -> None:
    prompt = [3, 4, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        None,
        torch.tensor([prompt]),
        num_speculative_tokens=4,
        max_new_tokens=10,
        eos_token_id=None,
    )

    assert list(result.generated_token_ids) == _expected_continuation(prompt, 10)
    # drafter=None means candidates are always empty, so every round emits
    # exactly 1 (bonus) token -- 10 rounds for 10 requested tokens.
    assert result.accepted_lengths == (0,) * 10


def test_prompt_lookup_drafter_matches_the_continuation_and_accepts_something() -> None:
    # [5, 6, 5] with ngram_size=1: the last token (5) also occurred at
    # position 0, where it was followed by 6 -- exactly what
    # _FakeIncrementModel's own +1 rule predicts after the current 5 too
    # (its rule is a pure function of the token's value, memoryless), so
    # the drafter's guess here is genuinely correct, not just harmless.
    prompt = [5, 6, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        PromptLookupDrafter(ngram_size=1),
        torch.tensor([prompt]),
        num_speculative_tokens=1,
        max_new_tokens=4,
        eos_token_id=None,
    )

    assert list(result.generated_token_ids) == _expected_continuation(prompt, 4)
    assert sum(result.accepted_lengths) > 0


def test_cache_length_invariant_holds_after_multiple_rejecting_rounds() -> None:
    """The target's cache always lags the full sequence (prompt +
    generated) by exactly one token -- the most recent bonus/correction
    token is never fed back in until the *next* round. A cache-rollback
    bug (e.g. forgetting to crop) breaks this even though
    _FakeIncrementModel's predictions don't depend on cache contents, so
    output correctness alone would not catch it."""
    prompt = [3, 4, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _AlwaysWrongDrafter(),  # every round rejects, exercising crop(rejected_len > 0)
        torch.tensor([prompt]),
        num_speculative_tokens=2,
        max_new_tokens=6,
        eos_token_id=None,
    )

    total_sequence_length = len(prompt) + len(result.generated_token_ids)
    assert result.past_key_values.length == total_sequence_length - 1  # type: ignore[attr-defined]


def test_speculative_generate_matches_plain_decode_via_a_fake_tokenizer() -> None:
    class _FakeTokenizer:
        eos_token_id = 999

        def __call__(self, prompt: str, return_tensors: str) -> _FakeBatchEncoding:
            return _FakeBatchEncoding(input_ids=torch.tensor([[3, 4, 5]]))

    timing, accepted_lengths, generated = speculative_generate(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _FakeTokenizer(),  # type: ignore[arg-type]
        _AlwaysCorrectDrafter(),
        "prompt",
        num_speculative_tokens=2,
        max_new_tokens=4,
    )

    assert list(generated) == _expected_continuation([3, 4, 5], 4)
    assert timing.generated_token_count == 4
    assert timing.prompt_token_count == 3
    # Round 1 (k=2): both candidates accepted, +1 bonus = 3 tokens emitted.
    # Round 2 (room=1, k=1): the 1 candidate accepted, no room left for a
    # bonus = 1 token emitted. accepted_lengths = (2, 1), summing to 3.
    assert sum(accepted_lengths) == 3


def test_plain_greedy_generate_matches_the_closed_form_continuation() -> None:
    class _FakeTokenizer:
        eos_token_id = 999

        def __call__(self, prompt: str, return_tensors: str) -> _FakeBatchEncoding:
            return _FakeBatchEncoding(input_ids=torch.tensor([[3, 4, 5]]))

    timing, generated = plain_greedy_generate(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _FakeTokenizer(),  # type: ignore[arg-type]
        "prompt",
        max_new_tokens=4,
    )

    assert list(generated) == _expected_continuation([3, 4, 5], 4)
    assert timing.generated_token_count == 4
    assert timing.prompt_token_count == 3


@pytest.mark.slow
def test_speculative_generate_matches_plain_greedy_decode_on_a_real_tiny_model() -> None:
    """The strongest correctness proof: a real model, real tokenizer, real
    DynamicCache.crop() (transformers>=5.17.0) -- not the toy model above.
    Uses a genuinely repetitive prompt so PromptLookupDrafter has real
    matches to find. Plain greedy decoding is run manually, token-for-token
    (generate_with_timings only returns timings, not the generated ids),
    to get real ids to check token-for-token equality against."""
    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")
    prompt = "the cat sat on the mat the cat sat on the mat"
    max_new_tokens = 8

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    plain_ids: list[int] = []
    past_key_values = None
    next_input = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            plain_ids.append(int(next_token.item()))
            next_input = next_token

    _, _, speculative_ids = speculative_generate(
        model,
        tokenizer,
        PromptLookupDrafter(ngram_size=3),
        prompt,
        num_speculative_tokens=4,
        max_new_tokens=max_new_tokens,
    )

    assert list(speculative_ids) == plain_ids


def _plain_greedy_reference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
) -> list[int]:
    """Factors out the same manual plain-greedy loop
    test_speculative_generate_matches_plain_greedy_decode_on_a_real_tiny_model
    above hand-rolls, so the new DraftModelDrafter end-to-end test below can
    reuse it rather than duplicating the loop body a third time (the other
    two being here and decode.py's own plain_greedy_generate -- deliberately
    not reused directly, since that would make this test's reference no
    longer independent of the code under test)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    plain_ids: list[int] = []
    past_key_values = None
    next_input = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            plain_ids.append(int(next_token.item()))
            next_input = next_token
    return plain_ids


@pytest.mark.slow
def test_speculative_generate_with_draft_model_drafter_matches_plain_greedy_decode() -> None:
    """DraftModelDrafter has never been tested end-to-end inside
    run_speculative_rounds against a real model, real tokenizer, and a real
    transformers.Cache -- only in isolation against fake caches
    (test_drafters.py). Given this branch's own history (DraftModelDrafter
    carried two real Critical bugs, both caught only by hand-tracing, not
    by its own given tests), and given the GPU session's own correctness
    gate -- the only thing that WAS exercising this exact combination for
    real -- turned out to be structurally non-discriminating (finding C2,
    final review: the gate compared against a baseline generated by this
    same shared loop), this closes a real, now-urgent coverage gap. Uses a
    second, independently-loaded copy of the same tiny model as the draft
    model, satisfying the design doc's shared-vocab requirement trivially."""
    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")
    draft_model, _ = load_model("hf-internal-testing/tiny-random-gpt2")
    prompt = "the quick brown fox jumps over the lazy dog"
    max_new_tokens = 8

    plain_ids = _plain_greedy_reference(model, tokenizer, prompt, max_new_tokens)

    _, _, speculative_ids = speculative_generate(
        model,
        tokenizer,
        DraftModelDrafter(draft_model),
        prompt,
        num_speculative_tokens=4,
        max_new_tokens=max_new_tokens,
    )

    assert list(speculative_ids) == plain_ids


def test_stops_exactly_at_eos_even_when_a_full_round_of_room_remains() -> None:
    """A plain greedy decode stops the instant it emits eos_token_id,
    never emitting anything after it -- even a single-token round's own
    bonus token. Without the fix, a round that accepts an eos-valued
    candidate would still tack on its bonus token before the eos check
    fires, overshooting what plain greedy decoding would have produced."""
    prompt = [3, 4, 5]
    result = run_speculative_rounds(
        _FakeIncrementModel(),  # type: ignore[arg-type]
        _AlwaysCorrectDrafter(),
        torch.tensor([prompt]),
        num_speculative_tokens=1,
        max_new_tokens=5,
        eos_token_id=6,
    )

    assert list(result.generated_token_ids) == [6]
    assert result.accepted_lengths == (1,)
