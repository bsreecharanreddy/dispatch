"""The shared propose/verify/accept/rollback speculative-decoding loop
(docs/design/2026-09-17-phase-5b-speculative-decoding.md sections 3-4),
parameterized by a Drafter so neither drafter implementation duplicates
verification or KV-cache-rollback logic. Because both the target and (for
DraftModelDrafter) the drafter decode strictly greedily, and a candidate
is accepted iff it equals the target's own greedy argmax, this loop's
output is a deterministic function of the target model alone -- it must
produce byte-identical tokens to plain sequential greedy decoding of the
same target (design doc section 5).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dispatch.benchmark.metrics import TokenTimings
from dispatch.speculative.drafters import Drafter


def plain_greedy_generate(  # noqa: PLR0913 -- device/clock_fn are what make this testable
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    device: str = "cpu",
    clock_fn: Callable[[], float] = time.perf_counter,
) -> tuple[TokenTimings, tuple[int, ...]]:
    """Independent correctness oracle for speculative decoding's
    --drafter none baseline: a plain, token-by-token greedy decode loop
    that shares NO code with run_speculative_rounds, so it can actually
    catch a bug in the shared propose/verify/accept/rollback loop rather
    than silently agreeing with it (a real gap found by this phase's
    final whole-branch review -- the baseline was previously generated
    by run_speculative_rounds itself with drafter=None, making the
    correctness gate unable in principle to catch a bug in that shared
    code). Deliberately duplicates harness.generate_with_timings's own
    loop shape (not reused directly, since that file is reused
    unmodified elsewhere in this phase and doesn't return token ids)
    plus token-id capture."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_token_count = int(input_ids.shape[-1])
    eos_token_id = tokenizer.eos_token_id

    start_time = clock_fn()
    token_times: list[float] = []
    generated: list[int] = []
    past_key_values = None
    next_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            token_times.append(clock_fn())
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            next_input = next_token
            if eos_token_id is not None and token_id == eos_token_id:
                break

    timing = TokenTimings(
        start_time=start_time,
        token_times=tuple(token_times),
        prompt_token_count=prompt_token_count,
    )
    return timing, tuple(generated)


@dataclass(frozen=True)
class SpeculativeRoundsResult:
    """past_key_values is exposed only so tests can assert the
    cache-length invariant (design doc section 5) directly -- callers of
    speculative_generate don't need it."""

    token_times: tuple[float, ...]
    accepted_lengths: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    past_key_values: object


def run_speculative_rounds(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model: PreTrainedModel,
    drafter: Drafter | None,
    token_ids: torch.Tensor,
    *,
    num_speculative_tokens: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> SpeculativeRoundsResult:
    token_times: list[float] = []
    accepted_lengths: list[int] = []
    generated: list[int] = []
    past_key_values = None
    total_new = 0

    with torch.no_grad():
        while total_new < max_new_tokens:
            room = max_new_tokens - total_new
            k = min(num_speculative_tokens, room)
            candidates = (
                drafter.propose(token_ids, k)
                if drafter is not None
                else token_ids.new_empty((1, 0))
            )
            num_candidates = int(candidates.shape[-1])

            step_input = token_ids if past_key_values is None else token_ids[:, -1:]
            forward_input = torch.cat([step_input, candidates], dim=1)
            outputs = model(
                input_ids=forward_input, past_key_values=past_key_values, use_cache=True
            )
            past_key_values = outputs.past_key_values
            assert past_key_values is not None  # use_cache=True guarantees this

            if hasattr(past_key_values, "get_seq_length"):
                # Immediately after this round's forward call, the cache must
                # hold exactly the known-token history (token_ids, which
                # still holds its pre-round value here) plus this round's
                # num_candidates freshly-proposed-but-not-yet-verified
                # tokens -- no more, no less. A future change to the
                # catch-up/cache-update logic above that silently drifts
                # from this would otherwise produce plausible-looking
                # garbage instead of a crash (finding I3, final review).
                cached_length = past_key_values.get_seq_length()
                expected_length = token_ids.shape[1] + num_candidates
                assert cached_length == expected_length, (
                    f"cache-lag invariant violated: cache holds {cached_length} tokens, "
                    f"expected {expected_length} (this would silently corrupt position "
                    "encoding on the next round -- see the phase's final review, finding I3)"
                )

            offset = step_input.shape[1] - 1
            target_predictions = outputs.logits[0, offset:, :].argmax(dim=-1)

            accepted_len = 0
            while accepted_len < num_candidates and int(target_predictions[accepted_len]) == int(
                candidates[0, accepted_len]
            ):
                accepted_len += 1
            accepted_lengths.append(accepted_len)

            rejected_len = num_candidates - accepted_len
            past_key_values.crop(-rejected_len)
            if drafter is not None:
                drafter.on_accepted(accepted_len, rejected_len)

            emitted = candidates[:, :accepted_len]
            if accepted_len < room:
                bonus_token = target_predictions[accepted_len].view(1, 1)
                emitted = torch.cat([emitted, bonus_token], dim=1)

            emitted_list = emitted[0].tolist()
            if eos_token_id is not None and eos_token_id in emitted_list:
                # Plain sequential greedy decoding stops the instant it
                # emits eos_token_id, never emitting anything after it --
                # so truncate this round's emission there too, even if an
                # earlier-accepted candidate (not the bonus token) is what
                # actually matched, and even if room would have allowed
                # more.
                eos_position = emitted_list.index(eos_token_id)
                emitted = emitted[:, : eos_position + 1]
                emitted_list = emitted_list[: eos_position + 1]

            token_ids = torch.cat([token_ids, emitted], dim=1)
            for _ in emitted_list:
                token_times.append(clock_fn())
            generated.extend(int(t) for t in emitted_list)
            total_new += emitted.shape[1]

            if eos_token_id is not None and eos_token_id in emitted_list:
                break

    return SpeculativeRoundsResult(
        token_times=tuple(token_times),
        accepted_lengths=tuple(accepted_lengths),
        generated_token_ids=tuple(generated),
        past_key_values=past_key_values,
    )


def speculative_generate(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    drafter: Drafter | None,
    prompt: str,
    *,
    num_speculative_tokens: int,
    max_new_tokens: int = 32,
    device: str = "cpu",
    clock_fn: Callable[[], float] = time.perf_counter,
) -> tuple[TokenTimings, tuple[int, ...], tuple[int, ...]]:
    """Returns per-token timings (matching harness.generate_with_timings's
    shape, so benchmark/metrics.py's summarize() works unchanged), each
    round's accepted-candidate count (the raw data behind the
    acceptance-rate metric), and the generated token ids."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"]
    prompt_token_count = int(token_ids.shape[-1])
    start_time = clock_fn()

    result = run_speculative_rounds(
        model,
        drafter,
        token_ids,
        num_speculative_tokens=num_speculative_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        clock_fn=clock_fn,
    )

    timing = TokenTimings(
        start_time=start_time,
        token_times=result.token_times,
        prompt_token_count=prompt_token_count,
    )
    return timing, result.accepted_lengths, result.generated_token_ids
