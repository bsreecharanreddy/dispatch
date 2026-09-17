"""CPU-only, multi-process: proves send_kv_cache/recv_kv_cache round-trip
correctly using torch.distributed's gloo backend -- no GPU, no DeepEP, no
NCCL. Task 5 reuses these exact functions unchanged with the NCCL
backend on the real rented 4-GPU node; only init_process_group's backend
argument and the real ranks differ.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import DynamicCache

from dispatch.serving.handoff import recv_kv_cache, send_kv_cache
from dispatch.serving.kv_cache import layer_kv

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4
SEQ_LEN = 3
PORT = 29513


def _make_cache(seed: int) -> DynamicCache:
    generator = torch.Generator().manual_seed(seed)
    per_layer = [
        (
            torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=generator),
            torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=generator),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def _worker(rank: int, world_size: int, result_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    assert group is not None

    if rank == 0:
        send_kv_cache(_make_cache(seed=42), dst=1, group=group)
    else:
        received = recv_kv_cache(
            src=0,
            group=group,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
        )
        expected = _make_cache(seed=42)
        ok = all(
            torch.equal(layer_kv(received.layers[i])[0], layer_kv(expected.layers[i])[0])
            and torch.equal(layer_kv(received.layers[i])[1], layer_kv(expected.layers[i])[1])
            for i in range(NUM_LAYERS)
        )
        torch.save({"ok": ok, "seq_len": received.get_seq_length()}, result_path)

    dist.destroy_process_group()


@pytest.mark.slow
def test_send_and_recv_kv_cache_round_trip_over_gloo(tmp_path: object) -> None:
    result_path = str(tmp_path / "result.pt")  # type: ignore[operator]

    mp.spawn(  # type: ignore[attr-defined,no-untyped-call]
        _worker, args=(2, result_path), nprocs=2, join=True
    )

    result = torch.load(result_path, weights_only=True)
    assert result["ok"] is True
    assert result["seq_len"] == SEQ_LEN
