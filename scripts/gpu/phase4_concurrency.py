"""Phase 4 concurrency/contention measurement: co-located (one 4-rank EP
pool doing both prefill and decode) vs disaggregated (2-rank prefill pool
+ 2-rank decode pool, connected by handoff.py's real cross-rank cache
transfer), same 4 GPUs, at a few concurrency levels, staggered arrivals.
Only run after both correctness gates in phase4_correctness_gate.py have
passed -- this script does not re-verify correctness, per this project's
"correctness gates the benchmark" rule.

Co-located:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_concurrency.py \
        --mode colocated --concurrency 4
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_concurrency.py \
        --mode colocated --concurrency 8

Disaggregated:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_concurrency.py \
        --mode disaggregated --concurrency 4
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_concurrency.py \
        --mode disaggregated --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from deep_ep import Buffer
from transformers import DynamicCache, PreTrainedTokenizerBase

from dispatch.benchmark.harness import load_model
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.expert_parallel import patch_moe_infer_ep
from dispatch.serving.colocated import ColocatedWorker
from dispatch.serving.disaggregated import (
    DecodeFn,
    DecodeWorker,
    PrefillFn,
    PrefillResult,
    PrefillWorker,
    Request,
    RequestResult,
)
from dispatch.serving.kv_cache import layer_kv

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]
MAX_NEW_TOKENS = 8
ARRIVAL_STAGGER_S = 0.05
OUTPUT_DIR = Path("docs/findings")


# See phase4_correctness_gate.py's own module docstring/comments for why
# this patch and the from_legacy_cache/to_legacy_cache conversions below
# are needed -- same DeepSeek remote-code quirk, same transformers 4.57.6
# pin, not repeated here.
def _get_usable_length(
    self: DynamicCache, new_seq_length: int | None = None, layer_idx: int = 0
) -> int:
    return self.get_seq_length(layer_idx)


DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]


def make_prefill_fn(model: torch.nn.Module, device: str) -> PrefillFn:
    def prefill_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=True,
            )
        first_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        cache = DynamicCache.from_legacy_cache(outputs.past_key_values)  # type: ignore[attr-defined]
        return first_tokens.cpu(), cache

    return prefill_fn


def make_decode_fn(model: torch.nn.Module, device: str) -> DecodeFn:
    def decode_fn(
        next_input_ids: torch.Tensor,
        cache: DynamicCache,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        with torch.no_grad():
            outputs = model(
                input_ids=next_input_ids.to(device),
                past_key_values=cache.to_legacy_cache(),  # type: ignore[attr-defined]
                attention_mask=attention_mask.to(device),
                position_ids=position_ids.to(device),
                use_cache=True,
            )
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        updated_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)  # type: ignore[attr-defined]
        return next_tokens.cpu(), updated_cache

    return decode_fn


def _make_requests(concurrency: int, tokenizer: PreTrainedTokenizerBase) -> list[Request]:
    requests = []
    for i in range(concurrency):
        prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        requests.append(Request(str(i), input_ids, max_new_tokens=MAX_NEW_TOKENS))
    return requests


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_colocated(concurrency: int) -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    dist.barrier()
    world_group = dist.group.WORLD
    assert world_group is not None

    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    hidden_bytes = model.config.hidden_size * 2
    Buffer.set_num_sms(24)
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
    buffer = Buffer(world_group, num_nvl_bytes, 0)
    patch_moe_infer_ep(model, resolve_backend("naive"), buffer, rank, world_size)

    requests = _make_requests(concurrency, tokenizer)
    arrival_times = [i * ARRIVAL_STAGGER_S for i in range(concurrency)]
    colocated_worker = ColocatedWorker(
        make_prefill_fn(model, "cuda"), make_decode_fn(model, "cuda"), batch_size=4
    )

    dist.barrier()
    t0 = time.perf_counter()
    pending = list(zip(arrival_times, requests, strict=True))
    completed: list[RequestResult] = []
    while len(completed) < concurrency:
        now = time.perf_counter() - t0
        while pending and pending[0][0] <= now:
            _, req = pending.pop(0)
            colocated_worker.submit(req)
        completed.extend(colocated_worker.step())
    total_wall_s = time.perf_counter() - t0

    if rank == 0:
        ttfts = [r.ttft for r in completed]
        total_tokens = sum(
            len(r.generated_token_ids) - 1 for r in completed
        )  # decode-generated only
        result = {
            "topology": "colocated",
            "concurrency": concurrency,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_wall_s": total_wall_s,
            "mean_ttft_s": statistics.mean(ttfts),
            "p50_ttft_s": _percentile(ttfts, 0.5),
            "p99_ttft_s": _percentile(ttfts, 0.99),
            "decode_tokens_per_s": total_tokens / total_wall_s,
        }
        print(f"[rank 0] colocated concurrency={concurrency}: {result}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"2026-09-16-phase-4-colocated-c{concurrency}.json").write_text(
            json.dumps(result, indent=2)
        )

    dist.destroy_process_group()


def _run_prefill_side(  # noqa: PLR0913, PLR0917 -- one call site (run_disaggregated), each arg independently needed
    concurrency: int,
    rank: int,
    world_group: dist.ProcessGroup,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    t0: float,
) -> None:
    requests = _make_requests(concurrency, tokenizer)
    arrival_times = [i * ARRIVAL_STAGGER_S for i in range(concurrency)]
    prefill_worker = PrefillWorker(make_prefill_fn(model, "cuda"), batch_size=4)
    dst = rank + 2
    pending = list(zip(arrival_times, requests, strict=True))
    sent = 0
    ttfts: list[float] = []
    while sent < concurrency:
        now = time.perf_counter() - t0
        while pending and pending[0][0] <= now:
            _, req = pending.pop(0)
            prefill_worker.submit(req)
        for result in prefill_worker.step():
            send_kv_cache_inline(result.cache, dst, world_group)
            meta = torch.tensor(
                [int(result.request_id), result.first_token_id], dtype=torch.int64, device="cuda"
            )
            dist.send(meta, dst=dst, group=world_group)
            ttfts.append(result.ttft)
            sent += 1
    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"2026-09-16-phase-4-disaggregated-prefill-c{concurrency}.json").write_text(
            json.dumps(
                {
                    "mean_ttft_s": statistics.mean(ttfts),
                    "p50_ttft_s": _percentile(ttfts, 0.5),
                    "p99_ttft_s": _percentile(ttfts, 0.99),
                },
                indent=2,
            )
        )
        print(f"[rank 0] disaggregated prefill side done, mean_ttft={statistics.mean(ttfts):.4f}s")


def _run_decode_side(  # noqa: PLR0913, PLR0917 -- one call site (run_disaggregated), each arg independently needed
    concurrency: int,
    rank: int,
    world_group: dist.ProcessGroup,
    model: torch.nn.Module,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    t0: float,
) -> None:
    decode_worker = DecodeWorker(make_decode_fn(model, "cuda"), batch_size=8)
    src = rank - 2
    received = 0
    completed: list[RequestResult] = []
    seq_len_buf = torch.zeros(1, dtype=torch.int64, device="cuda")
    pending_recv: Any = dist.irecv(seq_len_buf, src=src, group=world_group)
    while received < concurrency or len(completed) < concurrency:
        if pending_recv is not None and pending_recv.is_completed():
            seq_len = int(seq_len_buf.item())
            per_layer = []
            for _ in range(num_layers):
                key = torch.zeros(
                    1, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"
                )
                value = torch.zeros(
                    1, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"
                )
                dist.recv(key, src=src, group=world_group)
                dist.recv(value, src=src, group=world_group)
                per_layer.append((key, value))
            cache = DynamicCache(ddp_cache_data=per_layer)
            meta = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(meta, src=src, group=world_group)
            request_idx, first_token_id = int(meta[0].item()), int(meta[1].item())
            decode_worker.admit(
                PrefillResult(
                    request_id=str(request_idx),
                    cache=cache,
                    first_token_id=first_token_id,
                    ttft=0.0,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=None,
                )
            )
            received += 1
            if received < concurrency:
                seq_len_buf = torch.zeros(1, dtype=torch.int64, device="cuda")
                pending_recv = dist.irecv(seq_len_buf, src=src, group=world_group)
            else:
                pending_recv = None
        completed.extend(decode_worker.step())
    total_wall_s = time.perf_counter() - t0
    decode_pool_local_rank_0 = 2  # rank 2 is the decode pool's local rank 0, the protocol
    if rank == decode_pool_local_rank_0:
        total_tokens = sum(len(r.generated_token_ids) - 1 for r in completed)
        result = {
            "topology": "disaggregated",
            "concurrency": concurrency,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_wall_s": total_wall_s,
            "decode_tokens_per_s": total_tokens / total_wall_s,
        }
        print(f"[rank 2] disaggregated decode side concurrency={concurrency}: {result}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"2026-09-16-phase-4-disaggregated-decode-c{concurrency}.json").write_text(
            json.dumps(result, indent=2)
        )


def run_disaggregated(concurrency: int) -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_group = dist.group.WORLD
    prefill_group = dist.new_group([0, 1])
    decode_group = dist.new_group([2, 3])
    assert world_group is not None
    dist.barrier()

    is_prefill = rank in (0, 1)
    pool_group = prefill_group if is_prefill else decode_group
    assert isinstance(pool_group, dist.ProcessGroup)
    pool_rank = dist.get_rank(group=pool_group)

    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    hidden_bytes = model.config.hidden_size * 2
    Buffer.set_num_sms(24)
    dispatch_config = Buffer.get_dispatch_config(2)
    combine_config = Buffer.get_combine_config(2)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, 2),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, 2),
    )
    buffer = Buffer(pool_group, num_nvl_bytes, 0)
    patch_moe_infer_ep(model, resolve_backend("naive"), buffer, pool_rank, 2)

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers

    dist.barrier()
    t0 = time.perf_counter()

    if is_prefill:
        _run_prefill_side(concurrency, rank, world_group, model, tokenizer, t0)
    else:
        _run_decode_side(concurrency, rank, world_group, model, num_layers, num_heads, head_dim, t0)

    dist.destroy_process_group()


def send_kv_cache_inline(cache: DynamicCache, dst: int, group: dist.ProcessGroup) -> None:
    seq_len = torch.tensor([cache.get_seq_length()], dtype=torch.int64, device="cuda")
    dist.send(seq_len, dst=dst, group=group)
    for layer in cache.layers:
        key, value = layer_kv(layer)
        dist.send(key.contiguous(), dst=dst, group=group)
        dist.send(value.contiguous(), dst=dst, group=group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["colocated", "disaggregated"])
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "colocated":
        run_colocated(args.concurrency)
    else:
        run_disaggregated(args.concurrency)


if __name__ == "__main__":
    main()
