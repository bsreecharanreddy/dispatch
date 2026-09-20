"""CPU-only: proves ColocatedWorker interleaves inline prefill and
decode on the same loop -- new requests get their first token the same
step old ones advance, which is exactly what creates the contention
Phase 4 measures disaggregation against. Reuses test_disaggregated.py's
fake prefill/decode functions unchanged: the forward-pass contract is
identical, only which worker calls it differs.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from dispatch.serving.colocated import ColocatedWorker
from dispatch.serving.disaggregated import Request

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
    first_tokens = torch.arange(batch_size) + 100
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
    next_tokens = next_input_ids.squeeze(1) + 1
    return next_tokens, _fake_cache(batch_size, new_seq_len)


def test_colocated_worker_admits_via_inline_prefill_and_completes_after_max_new_tokens() -> None:
    """max_new_tokens=3, not 2: admission (inline prefill) and one decode
    step happen in the same step() call here too, same as DecodeWorker --
    a max_new_tokens=2 request would already be done at the first step().
    """
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=3))

    first_step = worker.step()
    assert first_step == []  # prefilled and decoded once this call: 2 of 3 tokens so far

    second_step = worker.step()
    assert len(second_step) == 1
    assert second_step[0].request_id == "a"
    assert second_step[0].generated_token_ids == [100, 101, 102]


def test_colocated_worker_interleaves_new_prefill_with_existing_decode_in_one_step() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("old", torch.tensor([[1, 2]]), max_new_tokens=3))
    worker.step()  # "old" admitted and decoded once: 2 of 3 tokens so far

    worker.submit(Request("new", torch.tensor([[9]]), max_new_tokens=1))
    step_result = worker.step()

    ids = {r.request_id for r in step_result}
    assert ids == {"new", "old"}  # "new" done from its own prefill token; "old" reaches 3 of 3 here


def test_colocated_worker_respects_batch_size_across_prefill_and_decode() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=1)
    worker.submit(Request("a", torch.tensor([[1]]), max_new_tokens=5))
    worker.submit(Request("b", torch.tensor([[2]]), max_new_tokens=1))

    for _ in range(3):
        worker.step()  # "a" holds the only slot; "b" stays waiting
    a_done = worker.step()
    assert [r.request_id for r in a_done] == ["a"]  # 5 tokens: 1 from prefill + 4 decode steps

    b_done = worker.step()  # "b" now gets its prefill turn, already done at max_new_tokens=1
    assert [r.request_id for r in b_done] == ["b"]


def test_snapshot_active_tokens_reports_in_flight_generated_ids_so_far() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=3))

    worker.step()  # prefilled and decoded once: 2 of 3 tokens so far, still active

    assert worker.snapshot_active_tokens() == {"a": [100, 101]}

    worker.step()  # completes; no longer active
    assert worker.snapshot_active_tokens() == {}
