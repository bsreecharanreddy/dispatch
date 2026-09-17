"""CPU-only: proves the continuous-batching scheduler's admission,
padding, and completion bookkeeping against a fake prefill/decode
forward pass -- no GPU, no DeepEP, no real model. Task 5 wires the same
PrefillFn/DecodeFn shapes to the real DeepSeekMoE-16B model and DeepEP EP
pools; nothing here changes at that point.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from dispatch.serving.disaggregated import DecodeWorker, PrefillResult, PrefillWorker, Request

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4


def _fake_cache(batch_size: int, seq_len: int) -> DynamicCache:
    per_layer = [
        (
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def _fake_prefill_fn(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask
    batch_size, seq_len = input_ids.shape
    first_tokens = torch.arange(batch_size) + 100  # distinctive, deterministic per row
    return first_tokens, _fake_cache(batch_size, seq_len)


def _fake_decode_fn(
    next_input_ids: torch.Tensor,
    cache: DynamicCache,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask, position_ids
    batch_size = next_input_ids.shape[0]
    new_seq_len = cache.get_seq_length() + 1
    next_tokens = next_input_ids.squeeze(1) + 1  # deterministic increment, easy to assert on
    return next_tokens, _fake_cache(batch_size, new_seq_len)


def test_prefill_worker_drains_up_to_batch_size_and_produces_per_request_caches() -> None:
    worker = PrefillWorker(_fake_prefill_fn, batch_size=2)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=4))
    worker.submit(Request("b", torch.tensor([[1, 2]]), max_new_tokens=4))
    worker.submit(Request("c", torch.tensor([[1]]), max_new_tokens=4))

    results = worker.step()

    assert [r.request_id for r in results] == ["a", "b"]
    assert results[0].cache.get_seq_length() == 3  # unpadded back to its own prompt length
    assert results[1].cache.get_seq_length() == 2
    assert results[0].first_token_id == 100
    assert results[1].first_token_id == 101

    remaining = worker.step()
    assert [r.request_id for r in remaining] == ["c"]


def test_prefill_worker_step_with_nothing_waiting_returns_empty() -> None:
    worker = PrefillWorker(_fake_prefill_fn, batch_size=2)

    assert worker.step() == []


def test_decode_worker_admits_and_completes_requests_after_max_new_tokens() -> None:
    """max_new_tokens=3, not 2: admission and one decode step happen in
    the same step() call (continuous batching), so a max_new_tokens=2
    request would already be complete at admission time -- too degenerate
    to show the "still needs another step()" case this test is for.
    """
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    cache = _fake_cache(1, 3)
    worker.admit(
        PrefillResult("a", cache, first_token_id=5, ttft=0.1, max_new_tokens=3, eos_token_id=None)
    )

    first_step = worker.step()
    assert first_step == []  # admitted and decoded once this call: 2 of 3 tokens so far

    second_step = worker.step()
    assert len(second_step) == 1
    assert second_step[0].request_id == "a"
    assert second_step[0].generated_token_ids == [5, 6, 7]


def test_decode_worker_completes_a_request_early_on_eos() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    cache = _fake_cache(1, 3)
    worker.admit(
        PrefillResult("a", cache, first_token_id=5, ttft=0.1, max_new_tokens=10, eos_token_id=6)
    )

    results = worker.step()

    assert len(results) == 1
    assert results[0].generated_token_ids == [5, 6]


def test_decode_worker_batches_requests_with_different_cache_lengths() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    worker.admit(
        PrefillResult(
            "short",
            _fake_cache(1, 2),
            first_token_id=1,
            ttft=0.1,
            max_new_tokens=2,
            eos_token_id=None,
        )
    )
    worker.admit(
        PrefillResult(
            "long",
            _fake_cache(1, 5),
            first_token_id=1,
            ttft=0.1,
            max_new_tokens=2,
            eos_token_id=None,
        )
    )

    results = worker.step()

    assert {r.request_id for r in results} == {"short", "long"}


def test_decode_worker_completes_immediately_when_admission_alone_hits_max_new_tokens() -> None:
    """max_new_tokens=1: prefill's own first token already satisfies it,
    so admission must not trigger an extra decode_fn call -- regression
    test for a real bug found while tracing this scheduler's logic:
    admission and decode share one step() call, so a naive
    always-decode-after-admitting implementation would over-generate by
    one token for any request this degenerate.
    """
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    worker.admit(
        PrefillResult(
            "a", _fake_cache(1, 2), first_token_id=1, ttft=0.1, max_new_tokens=1, eos_token_id=None
        )
    )

    results = worker.step()

    assert len(results) == 1
    assert results[0].generated_token_ids == [1]


def test_decode_worker_respects_batch_size_admitting_only_free_slots() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=1)
    worker.admit(
        PrefillResult(
            "a", _fake_cache(1, 2), first_token_id=1, ttft=0.1, max_new_tokens=5, eos_token_id=None
        )
    )
    worker.admit(
        PrefillResult(
            "b", _fake_cache(1, 2), first_token_id=1, ttft=0.1, max_new_tokens=2, eos_token_id=None
        )
    )

    first_step = worker.step()
    assert first_step == []  # "a" admitted (only free slot); "b" still waiting

    for _ in range(2):
        worker.step()
    a_result = worker.step()
    assert [r.request_id for r in a_result] == [
        "a"
    ]  # 5 tokens total: 1 from prefill + 4 decode steps

    b_result = (
        worker.step()
    )  # "b" now admitted, needs only 1 more token -> completes this same step
    assert [r.request_id for r in b_result] == ["b"]
