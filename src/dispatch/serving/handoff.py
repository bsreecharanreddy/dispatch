"""Cross-rank KV-cache handoff between the prefill and decode EP pools,
built on plain torch.distributed point-to-point send/recv so the exact
same code runs over gloo (CPU, this file's own tests) and NCCL (real
GPUs, Task 5) -- the transport logic gets proven once, for $0, before it
is ever run against real hardware (design doc section 8's "maximize
what's proven before any rental").
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from transformers import DynamicCache

from dispatch.serving.kv_cache import layer_kv


def send_kv_cache(cache: DynamicCache, dst: int, group: dist.ProcessGroup) -> None:
    # NCCL (unlike gloo) requires every send/recv tensor to be on the
    # correct CUDA device -- matching the cache's own device keeps this
    # gloo-safe too (the existing CPU test's caches default to "cpu").
    device = layer_kv(cache.layers[0])[0].device
    seq_len = torch.tensor([cache.get_seq_length()], dtype=torch.int64, device=device)
    dist.send(seq_len, dst=dst, group=group)
    for layer in cache.layers:
        key, value = layer_kv(layer)
        dist.send(key.contiguous(), dst=dst, group=group)
        dist.send(value.contiguous(), dst=dst, group=group)


def recv_kv_cache(  # noqa: PLR0913 -- the receiver can't infer shape/dtype/device from the wire, they must be passed
    src: int,
    group: dist.ProcessGroup,
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> DynamicCache:
    seq_len_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    dist.recv(seq_len_tensor, src=src, group=group)
    seq_len = int(seq_len_tensor.item())

    per_layer = []
    for _ in range(num_layers):
        key = torch.zeros(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        value = torch.zeros(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        dist.recv(key, src=src, group=group)
        dist.recv(value, src=src, group=group)
        per_layer.append((key, value))
    return DynamicCache(ddp_cache_data=per_layer)
