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
from transformers import Cache, PreTrainedModel


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
            # on_accepted's crop(rejected_len) assumes every proposed
            # candidate is cache-resident (matching the target's own
            # crop formula in decode.py exactly) -- without this call the
            # loop above would leave the very last candidate un-cached,
            # silently evicting an accepted token on the next crop.
            outputs = self.model(
                input_ids=next_input, past_key_values=self.past_key_values, use_cache=True
            )
            self.past_key_values = outputs.past_key_values
        return torch.cat(proposed, dim=1)

    def on_accepted(self, accepted_len: int, rejected_len: int) -> None:
        if self.past_key_values is not None:
            self.past_key_values.crop(-rejected_len)
