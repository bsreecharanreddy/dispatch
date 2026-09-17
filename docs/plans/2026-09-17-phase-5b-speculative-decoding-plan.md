# Phase 5b: Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement speculative decoding on top of dispatch's Phase 5a
int8-quantized target kernel -- a real draft-model drafter
(`deepseek-llm-7b-base`) and a model-free prompt-lookup drafter -- wire
both through a new CLI script, and measure real throughput/acceptance-rate
against a plain-greedy quantized-target baseline on the same hardware
class this project has used throughout.

**Architecture:** A new `src/dispatch/speculative/` package. `drafters.py`
defines a `Drafter` protocol plus two implementations (`DraftModelDrafter`,
`PromptLookupDrafter`). `decode.py` holds the one shared
propose/verify/accept/rollback loop (`run_speculative_rounds`, wrapped by
the tokenizer-facing `speculative_generate`), parameterized by a `Drafter`
so neither implementation duplicates verification or cache-rollback logic.
`reference.py` captures/compares the exact generated token ids (not
logits -- speculative decoding's correctness claim is byte-exact token
match against plain greedy decoding, a tighter bar than the tolerance-based
logit agreement `benchmark/reference.py` already provides for kernel
swaps). `scripts/run_speculative_bench.py` wires it all together, always
running the target through Phase 5a's `patch_moe_infer_quantized` per the
design doc's memory-budget decision.

**Tech Stack:** PyTorch, `transformers>=5.17.0` (its `Cache.crop()`,
confirmed live to exist in the pinned version -- design doc section "Data
flow"). No new third-party dependency; `deepseek-llm-7b-base` is loaded
through the existing `load_model` exactly like the target.

**Spec:** `docs/design/2026-09-17-phase-5b-speculative-decoding.md` (all
sections). Also reused, unmodified: `src/dispatch/benchmark/harness.py`
(`load_model`, `generate_with_timings`), `src/dispatch/benchmark/metrics.py`
(`TokenTimings`, `summarize`), `src/dispatch/kernels/backends.py`
(`resolve_quantized_backend`), `src/dispatch/kernels/integration.py`
(`patch_moe_infer_quantized`), `scripts/run_baseline.py`
(`DEFAULT_PROMPTS`), `scripts/gpu/provision.py` (`write_cost_record`).

## Global Constraints

- **Two drafters, both implemented, not one** -- `DraftModelDrafter`
  (wraps `deepseek-llm-7b-base`) and `PromptLookupDrafter` (no model),
  per design doc section 2.
- **Verification is exact-match against the target's own greedy argmax,
  never sampling-based.** Speculative decoding's output must be
  byte-identical to plain sequential greedy decoding of the same target
  (design doc section 5) -- every task's tests hold this bar literally,
  not approximately.
- **The target always runs through Phase 5a's int8-quantized kernel**
  (`patch_moe_infer_quantized` + `resolve_quantized_backend()`, both
  reused unmodified) -- never bf16, per the design doc's memory-budget
  decision (section 2).
- **`deepseek-llm-7b-base` is the draft model**, loaded via the existing
  `load_model`, no `trust_remote_code` (it's a plain `LlamaForCausalLM`,
  confirmed live -- design doc section 2). Not configurable to a smaller
  same-vocab model because none exists standalone in the `deepseek-ai`
  namespace (design doc section 2, section 7).
- **Four prompts for this phase's own runs**: the existing
  `DEFAULT_PROMPTS` (`scripts/run_baseline.py`) plus one new
  repetition-heavy prompt, so `PromptLookupDrafter` has a genuine chance
  to find n-gram matches (design doc section 2).
- **Budget cap: $10, a ceiling not a target**, single GPU (L40 class,
  Phase 5a's tier), one combined session, checked live against real-time
  marketplace/spot availability at rental time. This is a live session
  with the user, not something to run unattended -- get explicit
  go-ahead before renting (Task 6).
- **Never quote a benchmark number that wasn't measured** on this exact
  run (repo-wide rule).
- **Out of scope for this plan** (design doc section 7): sampling-based
  speculative decoding, combining with multi-GPU EP, a third drafter
  type, beating vLLM/SGLang's own speculative-decoding numbers, a
  same-vocab draft model smaller than 7B.

---

### Task 1: `Drafter` protocol and `PromptLookupDrafter`

**Files:**
- Create: `src/dispatch/speculative/__init__.py`
- Create: `src/dispatch/speculative/drafters.py`
- Test: `tests/unit/test_drafters.py`

**Interfaces:**
- Produces: `Drafter` (`Protocol`: `propose(token_ids: torch.Tensor,
  num_tokens: int) -> torch.Tensor`; `on_accepted(accepted_len: int,
  rejected_len: int) -> None`). `PromptLookupDrafter(ngram_size: int = 3)`
  implementing `Drafter`.

- [ ] **Step 1: Write the failing tests**

Create `src/dispatch/speculative/__init__.py` (empty, matching
`src/dispatch/kernels/__init__.py` and `src/dispatch/serving/__init__.py`).

Create `tests/unit/test_drafters.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_drafters.py -v`
Expected: FAIL (collection error) -- `dispatch.speculative.drafters` does
not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `src/dispatch/speculative/drafters.py`:

```python
"""Pluggable draft-token proposers for speculative decoding
(docs/design/2026-09-17-phase-5b-speculative-decoding.md section 3): a
Drafter proposes candidate tokens; decode.py's run_speculative_rounds
verifies them against the target model's own greedy output and tells the
drafter how many were accepted so it can roll back any state it holds (a
draft model's own KV cache; a no-op for stateless drafters like this
module's PromptLookupDrafter).
"""

from __future__ import annotations

from typing import Protocol

import torch


class Drafter(Protocol):
    def propose(self, token_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """token_ids: (1, seq_len), the full sequence so far. Returns up to
        num_tokens proposed token ids, shape (1, <=num_tokens) -- a shorter
        (including empty, shape (1, 0)) return is valid and the caller
        must treat it as a smaller round, not an error."""
        ...

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        """Called once per verification round so a drafter holding its
        own cache can roll it back. A no-op for stateless drafters."""
        ...


class PromptLookupDrafter:
    """No model, no cache: proposes whatever tokens followed the most
    recent earlier occurrence of the last `ngram_size` tokens, up to
    num_tokens of them. Returns an empty proposal if no match exists, or
    if fewer than num_tokens tokens followed the match (never padded) --
    both real, expected outcomes on non-repetitive prompts, not errors."""

    def __init__(self, ngram_size: int = 3) -> None:
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
        self.ngram_size = ngram_size

    def propose(self, token_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        sequence = token_ids[0]
        seq_len = int(sequence.shape[0])
        if num_tokens <= 0 or seq_len <= self.ngram_size:
            return sequence.new_empty((1, 0))
        needle = sequence[-self.ngram_size :]
        windows = sequence.unfold(0, self.ngram_size, 1)[:-1]  # excludes needle's own window
        matches = (windows == needle).all(dim=1).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            return sequence.new_empty((1, 0))
        match_end = int(matches[-1]) + self.ngram_size  # latest match wins
        take = min(num_tokens, seq_len - match_end)
        return sequence[match_end : match_end + take].clone().unsqueeze(0)

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_drafters.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/speculative/__init__.py src/dispatch/speculative/drafters.py \
  tests/unit/test_drafters.py
git commit -m "feat: add Drafter protocol and PromptLookupDrafter"
```

---

### Task 2: `DraftModelDrafter`

**Files:**
- Modify: `src/dispatch/speculative/drafters.py`
- Modify: `tests/unit/test_drafters.py`

**Interfaces:**
- Consumes: `transformers.PreTrainedModel`, `transformers.Cache`.
- Produces: `DraftModelDrafter(model: PreTrainedModel)` implementing
  `Drafter`; `self.past_key_values: Cache | None` (exposed so
  `decode.py`'s tests, Task 3, can assert cache bookkeeping directly if
  needed).

- [ ] **Step 1: Write the failing tests**

At the top of `tests/unit/test_drafters.py`, change the import line to:

```python
from dispatch.speculative.drafters import DraftModelDrafter, PromptLookupDrafter
```

Then append the following fake model and tests to the end of the file:

```python
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
    assert drafter.past_key_values.length == 3 + 1


def test_later_propose_call_feeds_only_the_newest_token() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]
    drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=2)  # seeds the cache, length 3+1=4

    proposed = drafter.propose(torch.tensor([[3, 4, 5, 6, 7]]), num_tokens=2)

    assert proposed.tolist() == [[8, 9]]
    assert drafter.past_key_values is not None
    # Cache already holds a real KV entry (length 4), so both iterations of
    # this call feed exactly 1 new token each: 4 + 1 + 1 = 6.
    assert drafter.past_key_values.length == 4 + 1 + 1


def test_on_accepted_crops_the_cache_by_the_rejected_length() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]
    # Iteration 1 feeds the whole 3-token prompt; iterations 2-4 each feed
    # 1 token: cache length = 3 + 1 + 1 + 1 = 6.
    drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=4)

    drafter.on_accepted(accepted_len=1, rejected_len=3)

    assert drafter.past_key_values is not None
    assert drafter.past_key_values.length == 6 - 3


def test_on_accepted_before_any_propose_call_is_a_no_op() -> None:
    DraftModelDrafter(_FakeIncrementModel()).on_accepted(accepted_len=0, rejected_len=0)  # type: ignore[arg-type]


def test_propose_with_non_positive_num_tokens_returns_empty_without_calling_the_model() -> None:
    model = _FakeIncrementModel()
    drafter = DraftModelDrafter(model)  # type: ignore[arg-type]

    proposed = drafter.propose(torch.tensor([[3, 4, 5]]), num_tokens=0)

    assert proposed.shape == (1, 0)
    assert model.call_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_drafters.py -v`
Expected: FAIL -- `DraftModelDrafter` doesn't exist yet.

- [ ] **Step 3: Write the minimal implementation**

At the top of `src/dispatch/speculative/drafters.py`, replace the
`from typing import Protocol` line with:

```python
from typing import Protocol

from transformers import Cache, PreTrainedModel
```

Then append the following below `PromptLookupDrafter`:

```python
class DraftModelDrafter:
    """Wraps a loaded causal LM and its own KV cache, greedily decoding up
    to num_tokens candidates per round. Mirrors harness.py's plain decode
    loop's own catch-up pattern (feed the whole sequence when the cache is
    empty, otherwise just the newest token) applied across repeated
    propose() calls: the drafter's cache always lags the true sequence by
    exactly one token (the target's own most recent bonus/correction
    token, which the drafter hasn't seen yet) -- the first forward call of
    each round after the first both catches that token up and produces
    this round's first candidate."""

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.past_key_values: Cache | None = None

    def propose(self, token_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if num_tokens <= 0:
            return token_ids.new_empty((1, 0))
        next_input = token_ids if self.past_key_values is None else token_ids[:, -1:]
        proposed: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.model(
                    input_ids=next_input, past_key_values=self.past_key_values, use_cache=True
                )
                self.past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                proposed.append(next_token)
                next_input = next_token
        return torch.cat(proposed, dim=1)

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        if self.past_key_values is not None:
            self.past_key_values.crop(rejected_len)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_drafters.py -v`
Expected: 13 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/speculative/drafters.py tests/unit/test_drafters.py
git commit -m "feat: add DraftModelDrafter"
```

---

### Task 3: The shared speculative decode loop

**Files:**
- Create: `src/dispatch/speculative/decode.py`
- Test: `tests/unit/test_decode.py`

**Interfaces:**
- Consumes: `Drafter`, `PromptLookupDrafter` (Task 1). `TokenTimings`
  (`benchmark/metrics.py`, unmodified). `transformers.PreTrainedModel`,
  `transformers.PreTrainedTokenizerBase`.
- Produces: `SpeculativeRoundsResult` (frozen dataclass: `token_times:
  tuple[float, ...]`, `accepted_lengths: tuple[int, ...]`,
  `generated_token_ids: tuple[int, ...]`, `past_key_values: object`).
  `run_speculative_rounds(model, drafter: Drafter | None, token_ids:
  torch.Tensor, *, num_speculative_tokens: int, max_new_tokens: int,
  eos_token_id: int | None, clock_fn=time.perf_counter) ->
  SpeculativeRoundsResult`. `speculative_generate(model, tokenizer,
  drafter: Drafter | None, prompt: str, *, num_speculative_tokens: int,
  max_new_tokens: int = 32, device: str = "cpu", clock_fn=time.perf_counter)
  -> tuple[TokenTimings, tuple[int, ...], tuple[int, ...]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_decode.py`:

```python
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

from dispatch.speculative.decode import run_speculative_rounds, speculative_generate
from dispatch.speculative.drafters import PromptLookupDrafter

VOCAB_SIZE = 16


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def crop(self, tokens_to_remove: int) -> None:
        self.length -= tokens_to_remove


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
    class _FakeBatchEncoding(dict):  # type: ignore[type-arg]
        def to(self, device: str) -> _FakeBatchEncoding:
            return self

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


@pytest.mark.slow
def test_speculative_generate_matches_plain_greedy_decode_on_a_real_tiny_model() -> None:
    """The strongest correctness proof: a real model, real tokenizer, real
    DynamicCache.crop() (transformers>=5.17.0) -- not the toy model above.
    Uses a genuinely repetitive prompt so PromptLookupDrafter has real
    matches to find. Plain greedy decoding is run manually, token-for-token
    (generate_with_timings only returns timings, not the generated ids),
    to get real ids to check token-for-token equality against."""
    from dispatch.benchmark.harness import load_model

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_decode.py -v -m "not slow"`
Expected: FAIL (collection error) -- `dispatch.speculative.decode` does
not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `src/dispatch/speculative/decode.py`:

```python
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

            offset = step_input.shape[1] - 1
            target_predictions = outputs.logits[0, offset:, :].argmax(dim=-1)

            accepted_len = 0
            while (
                accepted_len < num_candidates
                and int(target_predictions[accepted_len]) == int(candidates[0, accepted_len])
            ):
                accepted_len += 1
            accepted_lengths.append(accepted_len)

            rejected_len = num_candidates - accepted_len
            past_key_values.crop(rejected_len)
            if drafter is not None:
                drafter.on_accepted(accepted_len, rejected_len)

            emitted = candidates[:, :accepted_len]
            if accepted_len < room:
                bonus_token = target_predictions[accepted_len].view(1, 1)
                emitted = torch.cat([emitted, bonus_token], dim=1)

            token_ids = torch.cat([token_ids, emitted], dim=1)
            emitted_list = emitted[0].tolist()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_decode.py -v -m "not slow"`
Expected: 6 passed.

Run (network access required, no GPU): `uv run pytest tests/unit/test_decode.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/speculative/decode.py tests/unit/test_decode.py
git commit -m "feat: add the shared speculative decode loop"
```

---

### Task 4: Exact generated-token reference capture/compare

**Files:**
- Create: `src/dispatch/speculative/reference.py`
- Test: `tests/unit/test_speculative_reference.py`

**Interfaces:**
- Produces: `save_generated_tokens(tokens: dict[str, tuple[int, ...]],
  path: Path) -> None`. `load_generated_tokens(path: Path) -> dict[str,
  tuple[int, ...]]`. `compare_generated_tokens(actual: dict[str,
  tuple[int, ...]], reference: dict[str, tuple[int, ...]]) -> dict[str,
  bool]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_speculative_reference.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_speculative_reference.py -v`
Expected: FAIL (collection error) -- `dispatch.speculative.reference`
does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `src/dispatch/speculative/reference.py`:

```python
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
from pathlib import Path


def save_generated_tokens(tokens: dict[str, tuple[int, ...]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: list(value) for key, value in tokens.items()}, indent=2))


def load_generated_tokens(path: Path) -> dict[str, tuple[int, ...]]:
    raw: dict[str, list[int]] = json.loads(path.read_text())
    return {key: tuple(value) for key, value in raw.items()}


def compare_generated_tokens(
    actual: dict[str, tuple[int, ...]], reference: dict[str, tuple[int, ...]]
) -> dict[str, bool]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )
    return {key: actual[key] == reference[key] for key in reference}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_speculative_reference.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/speculative/reference.py tests/unit/test_speculative_reference.py
git commit -m "feat: add exact generated-token reference capture/compare"
```

---

### Task 5: CLI -- `scripts/run_speculative_bench.py`

**Files:**
- Create: `scripts/run_speculative_bench.py`
- Test: `tests/unit/test_run_speculative_bench.py`

**Interfaces:**
- Consumes: `load_model` (`benchmark/harness.py`). `summarize`
  (`benchmark/metrics.py`). `resolve_quantized_backend`
  (`kernels/backends.py`). `patch_moe_infer_quantized`
  (`kernels/integration.py`). `speculative_generate` (Task 3). `Drafter`,
  `DraftModelDrafter`, `PromptLookupDrafter` (Tasks 1-2).
  `save_generated_tokens`, `load_generated_tokens`,
  `compare_generated_tokens` (Task 4). `DEFAULT_PROMPTS`
  (`scripts/run_baseline.py`).
- Produces: `SPECULATIVE_PROMPTS: list[str]`. `DRAFTERS: tuple[str, ...]`.
  `build_drafter(drafter_name, *, draft_model_name, device, dtype,
  prompt_lookup_ngram_size) -> Drafter | None`.
  `run_speculative_bench(model_name, *, device, dtype,
  trust_remote_code, prompts, repetitions, max_new_tokens, drafter_name,
  draft_model_name, num_speculative_tokens, prompt_lookup_ngram_size) ->
  tuple[list[tuple[TokenTimings, tuple[int, ...]]], dict[str, tuple[int,
  ...]], int]`. `main(argv)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_run_speculative_bench.py`:

```python
"""main()'s plumbing (args -> files) is tested fast with
run_speculative_bench and summarize monkeypatched out, matching
test_run_baseline.py's pattern; the real end-to-end guard test is `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_speculative_bench as run_speculative_bench_module
import torch
from scripts.run_speculative_bench import build_drafter, main

from dispatch.benchmark.metrics import BenchmarkSummary
from dispatch.speculative.drafters import DraftModelDrafter, PromptLookupDrafter
from dispatch.speculative.reference import save_generated_tokens

FAKE_SUMMARY = BenchmarkSummary(
    run_count=1,
    mean_ttft=0.1,
    p50_ttft=0.1,
    p99_ttft=0.1,
    mean_inter_token_latency=0.05,
    mean_tokens_per_second=20.0,
)
FAKE_TOKENS = {"prompt_000_tokens": (1, 2, 3)}


def _fake_run_speculative_bench(
    tokens: dict[str, tuple[int, ...]], moe_layers_patched: int
) -> object:
    def fake(model_name: str, **kwargs: object) -> tuple[list[object], dict, int]:  # type: ignore[type-arg]
        return [(object(), (2, 1))], tokens, moe_layers_patched

    return fake


def test_build_drafter_none_returns_none() -> None:
    assert (
        build_drafter(
            "none", draft_model_name="x", device="cpu", dtype=torch.float32, prompt_lookup_ngram_size=3
        )
        is None
    )


def test_build_drafter_prompt_lookup_returns_a_prompt_lookup_drafter() -> None:
    drafter = build_drafter(
        "prompt-lookup",
        draft_model_name="x",
        device="cpu",
        dtype=torch.float32,
        prompt_lookup_ngram_size=5,
    )

    assert isinstance(drafter, PromptLookupDrafter)
    assert drafter.ngram_size == 5


def test_build_drafter_draft_model_loads_and_wraps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_speculative_bench_module, "load_model", lambda *a, **k: ("the-model", "the-tokenizer")
    )

    drafter = build_drafter(
        "draft-model",
        draft_model_name="deepseek-ai/deepseek-llm-7b-base",
        device="cpu",
        dtype=torch.float32,
        prompt_lookup_ngram_size=3,
    )

    assert isinstance(drafter, DraftModelDrafter)
    assert drafter.model == "the-model"


def test_build_drafter_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown drafter"):
        build_drafter(
            "bogus", draft_model_name="x", device="cpu", dtype=torch.float32, prompt_lookup_ngram_size=3
        )


def test_main_writes_results_and_generated_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(FAKE_TOKENS, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(
        [
            "--model-name",
            "tiny/test-model",
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "test-run",
        ]
    )

    results = json.loads((tmp_path / "test-run-results.json").read_text())
    assert results["model"] == "tiny/test-model"
    assert results["mean_tokens_per_second"] == 20.0
    assert results["moe_layers_patched"] == 27
    assert results["acceptance_rate"] == pytest.approx(1.5 / 4)  # mean(2, 1) / num_speculative_tokens
    assert results["token_match"] == {}
    assert (tmp_path / "test-run-generated-tokens.json").exists()


def test_main_records_agreement_with_a_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "baseline-generated-tokens.json"
    save_generated_tokens(FAKE_TOKENS, reference_path)
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(FAKE_TOKENS, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(
        [
            "--drafter",
            "prompt-lookup",
            "--compare-generated-tokens",
            str(reference_path),
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "spec-run",
        ]
    )

    results = json.loads((tmp_path / "spec-run-results.json").read_text())
    assert results["token_match"] == {"prompt_000_tokens": True}


def test_main_exits_nonzero_on_token_divergence_but_keeps_the_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "baseline-generated-tokens.json"
    save_generated_tokens(FAKE_TOKENS, reference_path)
    wrong_tokens = {"prompt_000_tokens": (1, 2, 9)}
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(wrong_tokens, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

    with pytest.raises(SystemExit, match="disagree"):
        main(
            [
                "--drafter",
                "draft-model",
                "--compare-generated-tokens",
                str(reference_path),
                "--output-dir",
                str(tmp_path),
                "--run-label",
                "spec-run",
            ]
        )

    results = json.loads((tmp_path / "spec-run-results.json").read_text())
    assert results["token_match"] == {"prompt_000_tokens": False}
    assert (tmp_path / "spec-run-generated-tokens.json").exists()


@pytest.mark.slow
def test_run_speculative_bench_refuses_a_target_that_patches_nothing() -> None:
    import scripts.run_speculative_bench as module

    with pytest.raises(RuntimeError, match="patched no MoE layers"):
        module.run_speculative_bench(
            "hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            prompts=["hello"],
            repetitions=1,
            max_new_tokens=3,
            drafter_name="none",
            draft_model_name="unused",
            num_speculative_tokens=4,
            prompt_lookup_ngram_size=3,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_run_speculative_bench.py -v -m "not slow"`
Expected: FAIL (collection error) -- `scripts.run_speculative_bench` does
not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/run_speculative_bench.py`:

```python
"""CLI: speculative decoding on top of dispatch's own int8-quantized
target kernel (Phase 5a) -- either a real draft model
(deepseek-llm-7b-base) or model-free prompt-lookup decoding. Writes
latency/throughput and acceptance rate, and with
--compare-generated-tokens, checks the run's exact generated token ids
against a same-session baseline and exits non-zero on any divergence.
docs/design/2026-09-17-phase-5b-speculative-decoding.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dispatch.benchmark.harness import load_model
from dispatch.benchmark.metrics import TokenTimings, summarize
from dispatch.kernels.backends import resolve_quantized_backend
from dispatch.kernels.integration import patch_moe_infer_quantized
from dispatch.speculative.decode import speculative_generate
from dispatch.speculative.drafters import DraftModelDrafter, Drafter, PromptLookupDrafter
from dispatch.speculative.reference import (
    compare_generated_tokens,
    load_generated_tokens,
    save_generated_tokens,
)
from scripts.run_baseline import DEFAULT_PROMPTS

REPETITION_PROMPT = (
    "Please restate the following sentence exactly twice, back to back: "
    "'The system will now process each incoming request in the order it was received.'"
)
SPECULATIVE_PROMPTS = [*DEFAULT_PROMPTS, REPETITION_PROMPT]
DRAFTERS = ("none", "draft-model", "prompt-lookup")


def build_drafter(
    drafter_name: str,
    *,
    draft_model_name: str,
    device: str,
    dtype: torch.dtype,
    prompt_lookup_ngram_size: int,
) -> Drafter | None:
    if drafter_name == "none":
        return None
    if drafter_name == "prompt-lookup":
        return PromptLookupDrafter(ngram_size=prompt_lookup_ngram_size)
    if drafter_name == "draft-model":
        draft_model, _ = load_model(draft_model_name, device=device, dtype=dtype)
        return DraftModelDrafter(draft_model)
    raise ValueError(f"unknown drafter {drafter_name!r}; expected one of {DRAFTERS}")


def run_speculative_bench(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    prompts: list[str],
    repetitions: int,
    max_new_tokens: int,
    drafter_name: str,
    draft_model_name: str,
    num_speculative_tokens: int,
    prompt_lookup_ngram_size: int,
) -> tuple[list[tuple[TokenTimings, tuple[int, ...]]], dict[str, tuple[int, ...]], int]:
    """Returns the timed runs paired with each run's per-round accepted
    lengths, the first repetition's generated tokens per prompt (greedy
    decoding is deterministic, so later repetitions would be identical),
    and how many MoE layers were patched."""
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )
    moe_layers_patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
    if moe_layers_patched == 0:
        raise RuntimeError(
            f"quantized target patched no MoE layers: {model_name} has no moe_infer to "
            "replace, so this run would time the stock model under the quantized kernel's name"
        )

    drafter = build_drafter(
        drafter_name,
        draft_model_name=draft_model_name,
        device=device,
        dtype=dtype,
        prompt_lookup_ngram_size=prompt_lookup_ngram_size,
    )

    runs: list[tuple[TokenTimings, tuple[int, ...]]] = []
    generated_tokens: dict[str, tuple[int, ...]] = {}
    for prompt_index, prompt in enumerate(prompts):
        for repetition in range(repetitions):
            timing, accepted_lengths, tokens = speculative_generate(
                model,
                tokenizer,
                drafter,
                prompt,
                num_speculative_tokens=num_speculative_tokens,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            runs.append((timing, accepted_lengths))
            if repetition == 0:
                generated_tokens[f"prompt_{prompt_index:03d}_tokens"] = tokens

    return runs, generated_tokens, moe_layers_patched


def _acceptance_rate(
    runs: list[tuple[TokenTimings, tuple[int, ...]]], num_speculative_tokens: int
) -> float | None:
    lengths = [length for _, accepted_lengths in runs for length in accepted_lengths]
    if not lengths or num_speculative_tokens <= 0:
        return None
    return statistics.mean(lengths) / num_speculative_tokens


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run dispatch's speculative-decoding benchmark (Phase 5b)"
    )
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--draft-model-name", default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-speculative"))
    parser.add_argument("--drafter", default="none", choices=DRAFTERS)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--prompt-lookup-ngram-size", type=int, default=3)
    parser.add_argument(
        "--compare-generated-tokens",
        type=Path,
        default=None,
        help="generated-tokens JSON from a same-session baseline run of the same model and prompts",
    )
    args = parser.parse_args(argv)

    dtype: torch.dtype = getattr(torch, args.dtype)
    runs, generated_tokens, moe_layers_patched = run_speculative_bench(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=SPECULATIVE_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        drafter_name=args.drafter,
        draft_model_name=args.draft_model_name,
        num_speculative_tokens=args.num_speculative_tokens,
        prompt_lookup_ngram_size=args.prompt_lookup_ngram_size,
    )
    timings = [timing for timing, _ in runs]
    summary = summarize(timings)
    acceptance_rate = _acceptance_rate(runs, args.num_speculative_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output_dir / f"{args.run_label}-generated-tokens.json"
    save_generated_tokens(generated_tokens, tokens_path)
    print(f"wrote {tokens_path}")

    comparison = (
        compare_generated_tokens(generated_tokens, load_generated_tokens(args.compare_generated_tokens))
        if args.compare_generated_tokens is not None
        else {}
    )

    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "draft_model": args.draft_model_name if args.drafter == "draft-model" else None,
                "device": args.device,
                "dtype": args.dtype,
                "drafter": args.drafter,
                "num_speculative_tokens": args.num_speculative_tokens,
                "prompt_lookup_ngram_size": args.prompt_lookup_ngram_size,
                "moe_layers_patched": moe_layers_patched,
                "acceptance_rate": acceptance_rate,
                **asdict(summary),
                "token_match": comparison,
            },
            indent=2,
        )
    )
    print(f"wrote {results_path}")

    if comparison and not all(comparison.values()):
        raise SystemExit(
            f"{args.drafter} generated tokens disagree with "
            f"{args.compare_generated_tokens} -- see {results_path}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_speculative_bench.py -v -m "not slow"`
Expected: 8 passed.

Run (network access required, no GPU): `uv run pytest tests/unit/test_run_speculative_bench.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Run the full CPU-only suite**

Run: `make test`
Expected: all pass, nothing else broken by this task's edits.

- [ ] **Step 6: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_speculative_bench.py tests/unit/test_run_speculative_bench.py
git commit -m "feat: add scripts/run_speculative_bench.py"
```

---

### Task 6: GPU rental runbook -- correctness gate, measured run, k-sweep

**Files:**
- Create: `docs/runbooks/phase-5b-speculative-decoding.md`
- Creates live, on the pod / written by the CLI (not pre-committed as
  code, committed as evidence after the session):
  `docs/findings/<date>-phase-5b-baseline-generated-tokens.json`,
  `docs/findings/<date>-phase-5b-baseline-results.json`,
  `docs/findings/<date>-phase-5b-draft-model-results.json`,
  `docs/findings/<date>-phase-5b-prompt-lookup-results.json`,
  `docs/findings/<date>-phase-5b-k-sweep-*-results.json`,
  `docs/findings/<date>-phase-5b-memory-checkpoints.json`
- Create (via `write_cost_record`, reused unmodified):
  `docs/findings/<date>-phase-5b-speculative-decoding-cost.md`

**Budget cap: $10, a ceiling not a target -- be surgical (Global
Constraints). This is a live session with the user, not something to run
unattended -- get explicit go-ahead and confirm the cap before renting.**

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/phase-5b-speculative-decoding.md`:

````markdown
# Runbook: Phase 5b speculative decoding (rented GPU)

One combined session -- correctness gate, then the measured run and a
targeted k-sweep -- per
docs/design/2026-09-17-phase-5b-speculative-decoding.md section 8.
Budget cap: $10, a ceiling not a target.

1. Pick the cheapest L40-class card (Phase 5a's tier) with enough memory
   for the quantized target plus the bf16 draft model resident together
   (design doc section 2: quantizing the target frees ~13.9GB, per Phase
   5a's own measurement, specifically to make room for this), checked
   live against real-time availability.
2. Create and wait, with a 60GB+ volume (Phase 0's real trap: the target
   model alone is 32.8GB and this session also downloads a 7B draft
   model):

       uv run python -m scripts.gpu.provision create --name dispatch-phase-5b \
         --gpu-type "<id from step 1>" --image "<current runpod/pytorch tag>" \
         --cloud SECURE --disk-gb 70
       uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

3. Confirm the card:

       nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

4. Transfer and set up, all of Phase 0/1/5a's known environment traps in
   one place -- **confirm live whether DeepSeek's `modeling_deepseek.py`
   still needs these** (check
   https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/commits/main
   since Phase 5a's session) before assuming either is still required:

       git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"
       ssh <pod>
       cd dispatch
       export HF_HOME=/workspace/hf_cache   # NOT the container disk
       export HF_HUB_ENABLE_HF_TRANSFER=1
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv pip install hf_transfer   # Phase 5a's session needed this explicitly
       uv pip install transformers==4.57.6  # only if step 4's live check still shows the break

   If the `get_usable_length` break (Phase 0/1/5a) is still present, use
   the same pod-local, never-committed wrapper prior sessions used:

       cat > _patch_and_run.py <<'EOF'
       import sys
       from transformers.cache_utils import DynamicCache


       def _get_usable_length(self, new_seq_length=None, layer_idx=0):
           return self.get_seq_length(layer_idx)


       DynamicCache.get_usable_length = _get_usable_length

       from scripts.run_speculative_bench import main

       main(sys.argv[1:])
       EOF

   From here on invoke `.venv/bin/python` directly (`uv run` re-syncs to
   `uv.lock` on every call and silently undoes the `transformers`
   override). If the wrapper isn't needed, invoke
   `scripts/run_speculative_bench.py` directly instead of
   `_patch_and_run.py` in every command below.

5. Full CPU-testable suite, one more time, on the pod itself:

       .venv/bin/python -m pytest -m "not gpu" -v

   Expected: all pass (confirms the pod's environment doesn't disagree
   with what already passed locally, including the `slow` real-tiny-model
   tests from Tasks 3 and 5).

6. **Memory checkpoint after loading the target alone** (design doc
   section 8's named risk: two real models resident on one GPU is new
   territory for this project):

       DATE=$(date +%Y-%m-%d)
       .venv/bin/python <<'EOF'
       import json
       import torch

       from dispatch.benchmark.harness import load_model
       from dispatch.kernels.backends import resolve_quantized_backend
       from dispatch.kernels.integration import patch_moe_infer_quantized

       model, _ = load_model(
           "deepseek-ai/deepseek-moe-16b-base",
           device="cuda", dtype=torch.bfloat16, trust_remote_code=True,
       )
       patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
       checkpoint = {
           "after": "quantized target only",
           "moe_layers_patched": patched,
           "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
           "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
       }
       print(json.dumps(checkpoint, indent=2))
       with open("target_checkpoint.json", "w") as f:
           json.dump(checkpoint, f)
       EOF

   If `allocated_gb` already leaves no plausible room for a 7B bf16 draft
   (~14GB) under the card's total memory (from step 3), stop and report
   this as a named finding -- do not proceed to load the draft model on a
   card that clearly can't hold both; downsize the target's dtype further
   or move to a card with more memory instead of discovering an OOM live.

7. **Memory checkpoint after also loading the draft model:**

       .venv/bin/python <<'EOF'
       import json
       import torch

       from dispatch.benchmark.harness import load_model

       # Continues from step 6's session if run in the same process (a
       # notebook/REPL); otherwise re-run step 6's loading first, then:
       draft_model, _ = load_model(
           "deepseek-ai/deepseek-llm-7b-base", device="cuda", dtype=torch.bfloat16,
       )
       checkpoint = {
           "after": "quantized target + bf16 draft model",
           "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
           "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
       }
       print(json.dumps(checkpoint, indent=2))
       with open("both_checkpoint.json", "w") as f:
           json.dump(checkpoint, f)
       EOF

       cat target_checkpoint.json both_checkpoint.json > \
         docs/findings/$DATE-phase-5b-memory-checkpoints.json

   Anything that OOMs here: stop, report exactly what was measured before
   the failure (same discipline as Phase 5a's own mid-session OOM
   finding), and decide live whether to move to a larger card within the
   $10 cap before continuing.

8. **The baseline run** (`--drafter none`), this session's own reference:

       .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
         --drafter none \
         --run-label $DATE-phase-5b-baseline

   Must show `"moe_layers_patched": 27`.

9. **Correctness gate -- draft-model drafter, exact match against the
   baseline. Must pass before any timing is trusted.**

       .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
         --drafter draft-model \
         --compare-generated-tokens docs/findings/$DATE-phase-5b-baseline-generated-tokens.json \
         --run-label $DATE-phase-5b-draft-model

   Must show `"moe_layers_patched": 27` and every prompt's `token_match`
   `true`. Any `false` is a real bug -- stop and debug before proceeding
   to step 11's throughput numbers; the exact-match bar (design doc
   section 5) is not a suggestion.

10. **Correctness gate -- prompt-lookup drafter, same check:**

        .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
          --drafter prompt-lookup \
          --compare-generated-tokens docs/findings/$DATE-phase-5b-baseline-generated-tokens.json \
          --run-label $DATE-phase-5b-prompt-lookup

    Must show `"moe_layers_patched": 27` and every prompt's `token_match`
    `true`.

11. **Throughput and acceptance-rate comparison.** Read
    `mean_tokens_per_second` and `acceptance_rate` out of the three
    results JSONs from steps 8-10 -- no separate benchmark tool needed,
    `run_speculative_bench`'s own harness already measured both
    identically for all three configurations, at the default
    `--num-speculative-tokens 4`.

12. **Targeted k-sweep**, draft-model and prompt-lookup only, on the new
    repetition-heavy prompt alone (the 4th of `SPECULATIVE_PROMPTS`) --
    smaller and more targeted than re-running the full 4-prompt suite at
    every k, matching Phase 1's own token-count sweep's shape:

        for K in 1 2 4 8; do
          .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
            --drafter draft-model --num-speculative-tokens $K \
            --run-label $DATE-phase-5b-k-sweep-draft-model-$K
          .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
            --drafter prompt-lookup --num-speculative-tokens $K \
            --run-label $DATE-phase-5b-k-sweep-prompt-lookup-$K
        done

    (This sweep intentionally omits `--compare-generated-tokens`: steps
    9-10 already proved exact-match correctness at k=4 for both drafters
    across all 4 prompts; correctness does not depend on k, since the
    verification rule is the same at every k, so the sweep only needs to
    measure throughput/acceptance-rate, not re-verify correctness 8 more
    times.)

13. **Cost record and teardown:**

        .venv/bin/python -c "
        from pathlib import Path
        from scripts.gpu.provision import write_cost_record
        write_cost_record(
            Path('docs/findings'),
            pod_id='<pod_id>',
            gpu_type_id='<id from step 1>',
            cost_per_hour=<rate>,
            duration_s=<elapsed>,
            note='Phase 5b speculative decoding: correctness gates + measured run + k-sweep',
            run_label='phase-5b-speculative-decoding',
        )
        "

    Then tear the pod down immediately:

        uv run python -m scripts.gpu.provision delete --pod-id <pod_id>
        uv run python -m scripts.gpu.provision get --pod-id <pod_id>  # confirm TERMINATED
````

- [ ] **Step 2: Get explicit go-ahead and run the session**

Confirm with the user: budget cap ($10), GPU class/price at real-time
availability, and that this is a live, watched session -- then execute
the runbook above. Copy back every result JSON, the memory-checkpoint
file, and the cost file into `docs/findings/` on the local machine before
tearing the pod down.

- [ ] **Step 3: Commit the evidence**

```bash
git add docs/runbooks/phase-5b-speculative-decoding.md docs/findings/*phase-5b*
git commit -m "docs: run Phase 5b GPU correctness gates, measured session, and k-sweep"
```

---

### Task 7: Findings doc and STATUS.md

**Files:**
- Create: `docs/findings/<date>-phase-5b-speculative-decoding-run.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write the findings doc**

Cover, in `docs/findings/<date>-phase-5b-speculative-decoding-run.md`:
whether both drafters' correctness gates passed (Task 6 steps 9-10), and
on which GPU; the baseline/draft-model/prompt-lookup throughput and
acceptance-rate numbers from the three results JSONs (Task 6 step 11),
each with its full config (model, draft model where applicable, dtype,
hardware, prompts, repetitions, k); the k-sweep results (Task 6 step 12),
reported as measured across {1, 2, 4, 8} for both drafters on the
repetition-heavy prompt; the memory-checkpoint numbers (Task 6 steps 6-7)
showing how much headroom the quantized target left for the draft model;
cost per 1M generated tokens for baseline/draft-model/prompt-lookup (same
`$/hr / (tokens/sec * 3600) * 1e6` computation every prior phase's
findings doc has used); the actual GPU type, duration, and cost from Task
6's cost record, **explicitly compared against the $10 cap**; and an
honest verdict on this phase's two named risks from the design doc
(whether the 7B draft's net speedup held up once its own cost is counted;
whether prompt-lookup found real matches on the repetition-heavy prompt
and how that showed up in its acceptance rate vs. the original 3
prompts). If either correctness gate failed, or the budget ran out before
all three configurations and the sweep completed, say so plainly and
report exactly what was measured, matching this project's practice of
writing down a null or partial result rather than a flattering guess.

- [ ] **Step 2: Update STATUS.md**

Add a "## Phase 5b progress" section following the Phase 0-5a pattern:
design and plan links, task checklist, the correctness-gate results for
both drafters, the throughput/acceptance-rate numbers, the k-sweep
summary, and total GPU cost against the $10 cap. Update "## Next step" to
note Phase 5b is complete (or exactly how far it got) and that Phase 6
(final benchmark vs. vLLM/SGLang) is next.

- [ ] **Step 3: Commit**

```bash
git add docs/findings/<date>-phase-5b-speculative-decoding-run.md docs/STATUS.md
git commit -m "docs: record Phase 5b speculative decoding outcome"
```

Then open the PR for the whole `phase-5b-speculative-decoding` branch,
per this repo's one-branch-per-phase convention (README/CLAUDE.md refresh
first, per the standing instruction to update them proactively at natural
stopping points).

---

## Self-Review Notes

- **Spec coverage:** design doc section 2 (two drafters, quantized
  target, exact-match verification, single GPU, $10 cap, 4 prompts) is a
  Global Constraint and concretely implemented across Tasks 1-2 (both
  drafters), Task 3 (exact-match loop), Task 5 (quantized target wiring,
  the 4th prompt). Section 3 (architecture: `Drafter` protocol,
  `DraftModelDrafter`, `PromptLookupDrafter`, `speculative_generate`, new
  CLI script) maps to Tasks 1-3 and 5 one component per task, in
  dependency order. Section 4 (data flow: propose, one forward pass,
  accept longest prefix, crop, repeat) is Task 3's `run_speculative_rounds`
  line for line. Section 5 (exact-match correctness, both CPU and GPU
  tiers) is implemented in Task 3's tests (stub drafters, real
  `PromptLookupDrafter`, the cache-length invariant, and the `slow`
  real-tiny-model test) and Task 6 steps 9-10 (the GPU gate, against a
  real target). Section 6's testing table maps one-to-one onto Tasks
  1-5 (CPU/`slow` rows) and Task 6 (the `gpu`/paid row, plus the k-sweep).
  Section 7 (non-goals) is listed in Global Constraints so no task
  reaches for sampling-based decoding, multi-GPU EP, or a third drafter.
  Section 8 (risk/cost) is the $10 cap (Global Constraints, Task 6) and
  the named memory/speedup/acceptance risks (Task 6 steps 6-7's explicit
  checkpoint instruction and stop condition, Task 7's explicit
  honest-verdict requirement -- not silently smoothed over).
- **Placeholder scan:** no task step describes an action without showing
  the code; every test has real assertions; the GPU runbook's "confirm
  live" instructions and stop conditions are concrete, tied to specific,
  already-documented prior findings (Phase 0/1/5a's real bugs, Phase 5a's
  own OOM), not vague "handle appropriately" language.
- **Type consistency:** `Drafter`'s two methods (`propose`, `on_accepted`)
  are implemented identically by `PromptLookupDrafter` (Task 1) and
  `DraftModelDrafter` (Task 2), and consumed identically by
  `run_speculative_rounds` (Task 3) -- verified directly in Task 3's tests,
  which pass a real `PromptLookupDrafter` and two stub doubles into the
  exact same call sites `scripts/run_speculative_bench.py` (Task 5) later
  uses for the real drafters. `SpeculativeRoundsResult`'s four fields
  (Task 3) are consumed unchanged by `speculative_generate`'s return
  tuple, which `run_speculative_bench` (Task 5) unpacks with the same
  three-item shape its tests assert. `save_generated_tokens`/
  `load_generated_tokens`/`compare_generated_tokens` (Task 4) round-trip
  the exact `dict[str, tuple[int, ...]]` shape `run_speculative_bench`
  produces -- verified in Task 5's tests, which save real reference files
  with Task 4's own `save_generated_tokens` and read them back through
  `main()`'s `--compare-generated-tokens` path.
- **A decision the design doc left to this plan, resolved here:** the
  design doc didn't specify how the "none" (plain baseline) path should
  be implemented. This plan makes `run_speculative_rounds` accept
  `drafter: Drafter | None` and treat `None` explicitly as "always
  propose nothing," rather than requiring a trivial always-empty
  `Drafter` implementation just to satisfy the type -- one fewer public
  type, and the baseline path is provably identical to the speculative
  path's own degenerate case (verified directly by
  `test_none_drafter_matches_the_closed_form_continuation` in Task 3),
  not a separately-maintained code path that could silently drift from
  it.
- **A second decision resolved here:** the design doc's correctness bar
  (section 5) is byte-exact *generated tokens*, which is a different
  question from the tolerance-based *logit* agreement every prior phase's
  `--compare-reference` checks. Task 4 adds a parallel, deliberately
  separate mechanism (`reference.py`'s exact-token compare) rather than
  overloading `benchmark/reference.py`'s existing logit-tolerance
  machinery for a claim it wasn't built to make.
