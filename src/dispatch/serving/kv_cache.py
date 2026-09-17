"""KV-cache slicing and batch (re)assembly for Phase 4's continuous-
batching scheduler. Built entirely on transformers v5's surviving public
DynamicCache API -- to_legacy_cache/from_legacy_cache and the old
key_cache/value_cache attributes were removed in v5 (confirmed live
2026-09-16 against transformers' own v5 migration guide and
src/transformers/cache_utils.py). Dense, padded batching only -- not a
paged/block cache -- the known simplification this project's Phase 4
design doc discloses (section 6): real wasted compute next to a
production engine, applied identically to every configuration this
project measures, so it should not bias their relative comparison.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.cache_utils import CacheLayerMixin

# DeepSeekMoE-16B (this project's only model) uses standard multi-head
# attention throughout -- no linear/SSM (Mamba) layers -- so every layer
# in its cache is a CacheLayerMixin (.keys/.values), never the separate
# LinearAttentionCacheLayerMixin hierarchy DynamicCache.layers' type also
# allows, and (since every cache this module handles already has at
# least one token) its keys/values are always populated, never None.
# Asserted, not just assumed, so mypy --strict can narrow both away and
# a real regression (an uninitialized or linear-attention layer) fails
# loudly here instead of silently returning wrong tensors.


def layer_kv(layer: object) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(layer, CacheLayerMixin), f"expected an attention layer, got {type(layer)}"
    assert layer.keys is not None and layer.values is not None, "expected an initialized layer"
    return layer.keys, layer.values


def slice_cache(cache: DynamicCache, index: int, *, keep_last: int | None = None) -> DynamicCache:
    """Extract one request's KV cache (by batch index) out of a batched
    cache, without mutating the original -- unlike Cache.batch_select_indices,
    which selects in place and offers no way to also drop padding. If
    keep_last is given, keeps only the last keep_last positions along the
    sequence dim, dropping this request's left-padding after a batched
    forward call where its real content is shorter than the batch's
    padded length.
    """
    per_layer = []
    for raw_layer in cache.layers:
        layer_keys, layer_values = layer_kv(raw_layer)
        key = layer_keys[index : index + 1]
        value = layer_values[index : index + 1]
        if keep_last is not None:
            key = key[:, :, -keep_last:, :]
            value = value[:, :, -keep_last:, :]
        per_layer.append((key.clone(), value.clone()))
    return DynamicCache(ddp_cache_data=per_layer)


def pad_and_batch_caches(
    caches: list[DynamicCache], pad_to: int
) -> tuple[DynamicCache, torch.Tensor]:
    """Left-pads each cache's key/value tensors to pad_to positions and
    concatenates them along the batch dim, returning the batched cache
    and a matching (batch, pad_to) attention_mask (0 = pad, 1 = real).
    """
    if not caches:
        raise ValueError("caches must not be empty")
    num_layers = len(caches[0].layers)
    per_layer_keys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    per_layer_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    attention_mask = torch.zeros(len(caches), pad_to, dtype=torch.long)

    for row, cache in enumerate(caches):
        seq_len = cache.get_seq_length()
        if seq_len > pad_to:
            raise ValueError(f"cache seq_len={seq_len} exceeds pad_to={pad_to}")
        attention_mask[row, pad_to - seq_len :] = 1
        pad_len = pad_to - seq_len
        for layer_idx, raw_layer in enumerate(cache.layers):
            key, value = layer_kv(raw_layer)
            if pad_len > 0:
                pad_shape = (1, key.shape[1], pad_len, key.shape[3])
                key = torch.cat([torch.zeros(pad_shape, dtype=key.dtype), key], dim=2)
                value = torch.cat([torch.zeros(pad_shape, dtype=value.dtype), value], dim=2)
            per_layer_keys[layer_idx].append(key)
            per_layer_values[layer_idx].append(value)

    per_layer = [
        (torch.cat(per_layer_keys[i], dim=0), torch.cat(per_layer_values[i], dim=0))
        for i in range(num_layers)
    ]
    return DynamicCache(ddp_cache_data=per_layer), attention_mask
