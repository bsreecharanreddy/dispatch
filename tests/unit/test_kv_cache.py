"""CPU-only: proves the dense-padding slice/rebatch bookkeeping a
continuous-batching scheduler needs, built entirely on transformers v5's
surviving public DynamicCache API (cache.layers[i].keys/.values, the
ddp_cache_data constructor path) -- to_legacy_cache/from_legacy_cache and
the old key_cache/value_cache attributes were removed in v5 (confirmed
live 2026-09-16 against transformers' own v5 migration guide).
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from dispatch.benchmark.harness import load_model
from dispatch.serving.kv_cache import layer_kv, pad_and_batch_caches, slice_cache

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4


def _make_cache(batch_size: int, seq_len: int, *, seed: int = 0) -> DynamicCache:
    generator = torch.Generator().manual_seed(seed)
    per_layer = [
        (
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM, generator=generator),
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM, generator=generator),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def test_slice_cache_extracts_one_request_without_mutating_the_original() -> None:
    cache = _make_cache(batch_size=2, seq_len=3)
    cache_keys, _ = layer_kv(cache.layers[0])
    original_row_0 = cache_keys[0].clone()

    sliced = slice_cache(cache, index=1)
    sliced_keys, _ = layer_kv(sliced.layers[0])

    assert sliced.get_seq_length() == 3
    assert sliced_keys.shape[0] == 1
    torch.testing.assert_close(sliced_keys[0], cache_keys[1])
    torch.testing.assert_close(cache_keys[0], original_row_0)


def test_slice_cache_with_keep_last_drops_left_padding() -> None:
    cache = _make_cache(batch_size=1, seq_len=5)
    cache_keys, _ = layer_kv(cache.layers[0])
    real_tail = cache_keys[:, :, -2:, :].clone()

    sliced = slice_cache(cache, index=0, keep_last=2)
    sliced_keys, _ = layer_kv(sliced.layers[0])

    assert sliced.get_seq_length() == 2
    torch.testing.assert_close(sliced_keys, real_tail)


def test_pad_and_batch_caches_left_pads_to_a_common_length_and_masks_correctly() -> None:
    short = _make_cache(batch_size=1, seq_len=2, seed=1)
    long_ = _make_cache(batch_size=1, seq_len=5, seed=2)

    batched, attention_mask = pad_and_batch_caches([short, long_], pad_to=5)
    batched_keys, _ = layer_kv(batched.layers[0])
    short_keys, _ = layer_kv(short.layers[0])
    long_keys, _ = layer_kv(long_.layers[0])

    assert batched.get_seq_length() == 5
    assert batched_keys.shape[0] == 2
    assert attention_mask.tolist() == [[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]
    torch.testing.assert_close(batched_keys[0, :, -2:, :], short_keys[0])
    torch.testing.assert_close(batched_keys[1], long_keys[0])


def test_pad_and_batch_caches_rejects_a_cache_longer_than_pad_to() -> None:
    long_ = _make_cache(batch_size=1, seq_len=5)

    with pytest.raises(ValueError, match="exceeds"):
        pad_and_batch_caches([long_], pad_to=3)


def test_pad_and_batch_caches_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        pad_and_batch_caches([], pad_to=3)


@pytest.mark.slow
def test_slice_and_rebatch_round_trip_through_a_real_tiny_model() -> None:
    """Proves the helpers produce a cache the real model accepts back and
    continues generating from correctly -- against hf-internal-testing/
    tiny-random-gpt2, this repo's established tiny-model pattern
    (test_harness.py).
    """
    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")
    input_ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = outputs.past_key_values
    assert isinstance(cache, DynamicCache)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sliced = slice_cache(cache, index=0)
    assert sliced.get_seq_length() == cache.get_seq_length()

    with torch.no_grad():
        continued = model(input_ids=next_token, past_key_values=sliced, use_cache=True)

    assert continued.logits.shape == (1, 1, continued.logits.shape[-1])
