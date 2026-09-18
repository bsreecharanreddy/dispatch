"""Phase 4 correctness gate: does the real DeepSeekMoE-16B model produce
the exact same greedily-generated tokens under (a) a 4-rank co-located
EP pool and (b) two 2-rank disaggregated EP pools (prefill + decode,
connected by handoff.py's real cross-rank cache transfer) as a plain
single-GPU reference? Every rank in a pool runs the identical scheduler
logic against the identical prompt list -- PrefillWorker/DecodeWorker/
ColocatedWorker are pure and deterministic, so this keeps every rank's
model(...) calls naturally synchronized without a driver/follower split,
the same pattern Phase 3's own correctness gate already used (every EP
rank calling capture_reference_logits on the identical prompt).

Single-GPU reference:
    uv run python phase4_correctness_gate.py --mode single

Co-located (4 ranks, one EP pool):
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_correctness_gate.py --mode colocated

Disaggregated (4 ranks, prefill=[0,1] decode=[2,3]):
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 phase4_correctness_gate.py --mode disaggregated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from deep_ep import Buffer
from transformers import DynamicCache

from dispatch.benchmark.harness import load_model
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.expert_parallel import patch_moe_infer_ep
from dispatch.kernels.integration import patch_moe_infer
from dispatch.serving.colocated import ColocatedWorker
from dispatch.serving.disaggregated import (
    DecodeFn,
    DecodeWorker,
    PrefillFn,
    PrefillResult,
    PrefillWorker,
    Request,
)
from dispatch.serving.handoff import recv_kv_cache, send_kv_cache


# Pod-local, never committed to the repo -- DeepSeek's own remote-code
# modeling file (unchanged since Phase 0) still calls the pre-v5
# DynamicCache.get_usable_length, renamed to get_seq_length even in
# transformers 4.57.6 (the pin Phase 0/1/3 all needed for the *other*
# direction: transformers 5.17.0 removed is_torch_fx_available, which
# the same modeling file also imports). Confirmed equivalent for this
# model (no sliding-window attention) by Phase 0's own investigation.
# transformers' own type stubs don't declare this pre-v5 method (or
# from_legacy_cache/to_legacy_cache below), hence the type: ignore
# comments throughout this file -- real, version-specific runtime
# behavior the stubs don't (and can't) capture.
def _get_usable_length(
    self: DynamicCache, new_seq_length: int | None = None, layer_idx: int = 0
) -> int:
    return self.get_seq_length(layer_idx)


DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]
MAX_NEW_TOKENS = 16
OUTPUT_DIR = Path("docs/findings/phase-4")
REFERENCE_PATH = OUTPUT_DIR / "2026-09-16-phase-4-single-gpu-reference.json"


# DeepSeek's remote-code modeling file (unchanged since Phase 0) predates
# the Cache-class refactor: it accepts and returns past_key_values as the
# legacy tuple-of-(key, value)-per-layer format, not a DynamicCache
# object, confirmed live by inspecting a real forward call's output type
# (plain tuple, len == num_layers). Converting at this boundary only --
# via from_legacy_cache/to_legacy_cache, still present in transformers
# 4.57.6 though removed in v5 -- keeps kv_cache.py/disaggregated.py's own
# DynamicCache-based contract, proven by Tasks 1-4's CPU tests, unchanged.
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


def run_single_gpu_reference() -> None:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    patched = patch_moe_infer(model, resolve_backend("naive"))
    if patched == 0:
        raise RuntimeError("patched no MoE layers -- naive kernel never ran")
    print(f"single-GPU reference: patched {patched} MoE layers")

    results = {}
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated: list[int] = []
        past = None
        next_input = input_ids
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(int(next_token.item()))
                next_input = next_token
        results[prompt] = generated
        print(f"  {prompt!r} -> {generated}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {REFERENCE_PATH}")


def _run_worker_to_completion(worker: ColocatedWorker | DecodeWorker, request_id: str) -> list[int]:
    while True:
        for result in worker.step():
            if result.request_id == request_id:
                return result.generated_token_ids


def run_colocated() -> None:
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
    patched = patch_moe_infer_ep(model, resolve_backend("naive"), buffer, rank, world_size)
    if patched == 0:
        raise RuntimeError(f"[rank {rank}] patched no MoE layers")
    print(f"[rank {rank}] colocated: patched {patched} MoE layers, world_size={world_size}")

    colocated_worker = ColocatedWorker(
        make_prefill_fn(model, "cuda"), make_decode_fn(model, "cuda"), batch_size=1
    )
    results: dict[str, list[int]] = {}
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        colocated_worker.submit(Request(prompt, input_ids, max_new_tokens=MAX_NEW_TOKENS))
        results[prompt] = _run_worker_to_completion(colocated_worker, prompt)

    if rank == 0:
        print("[rank 0] colocated results:")
        for prompt, tokens in results.items():
            print(f"  {prompt!r} -> {tokens}")
        reference = json.loads(REFERENCE_PATH.read_text())
        match = all(results[p] == reference[p] for p in DEFAULT_PROMPTS)
        print(f"[rank 0] MATCH: {match}")
        (OUTPUT_DIR / "2026-09-16-phase-4-colocated-gate-results.json").write_text(
            json.dumps({"results": results, "match": match}, indent=2)
        )
        if not match:
            dist.destroy_process_group()
            raise SystemExit("colocated EP disagrees with the single-GPU reference")

    dist.destroy_process_group()


def run_disaggregated() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_group = dist.group.WORLD
    prefill_group = dist.new_group([0, 1])
    decode_group = dist.new_group([2, 3])
    assert world_group is not None
    assert prefill_group is not None
    assert decode_group is not None
    dist.barrier()

    is_prefill = rank in (0, 1)
    pool_group = prefill_group if is_prefill else decode_group
    # Every rank here is a member of exactly one of the two sub-groups, so
    # this never actually hits GroupMember.NON_GROUP_MEMBER -- but that
    # sentinel is part of new_group()'s declared return type, so `is not
    # None` alone doesn't narrow it away for mypy.
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
    patched = patch_moe_infer_ep(model, resolve_backend("naive"), buffer, pool_rank, 2)
    if patched == 0:
        raise RuntimeError(f"[rank {rank}] patched no MoE layers")
    role = "prefill" if is_prefill else "decode"
    print(
        f"[rank {rank}] disaggregated {role}: patched {patched} MoE layers, pool_rank={pool_rank}"
    )

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers

    results: dict[str, list[int]] = {}

    if is_prefill:
        prefill_worker = PrefillWorker(make_prefill_fn(model, "cuda"), batch_size=1)
        dst = rank + 2
        for prompt in DEFAULT_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prefill_worker.submit(Request(prompt, input_ids, max_new_tokens=MAX_NEW_TOKENS))
            prefill_results = prefill_worker.step()
            assert len(prefill_results) == 1
            result = prefill_results[0]
            send_kv_cache(result.cache, dst=dst, group=world_group)
            first_token_tensor = torch.tensor(
                [result.first_token_id], dtype=torch.int64, device="cuda"
            )
            dist.send(first_token_tensor, dst=dst, group=world_group)
    else:
        decode_worker = DecodeWorker(make_decode_fn(model, "cuda"), batch_size=1)
        src = rank - 2
        for prompt in DEFAULT_PROMPTS:
            cache = recv_kv_cache(
                src=src,
                group=world_group,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            first_token_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.recv(first_token_tensor, src=src, group=world_group)
            first_token_id = int(first_token_tensor.item())

            prefill_result = PrefillResult(
                request_id=prompt,
                cache=cache,
                first_token_id=first_token_id,
                ttft=0.0,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=None,
            )
            decode_worker.admit(prefill_result)
            results[prompt] = _run_worker_to_completion(decode_worker, prompt)

    if not is_prefill and rank == 2:
        print("[rank 2] disaggregated results:")
        for prompt, tokens in results.items():
            print(f"  {prompt!r} -> {tokens}")
        reference = json.loads(REFERENCE_PATH.read_text())
        match = all(results.get(p) == reference[p] for p in DEFAULT_PROMPTS)
        print(f"[rank 2] MATCH: {match}")
        (OUTPUT_DIR / "2026-09-16-phase-4-disaggregated-gate-results.json").write_text(
            json.dumps({"results": results, "match": match}, indent=2)
        )
        if not match:
            dist.destroy_process_group()
            raise SystemExit("disaggregated EP disagrees with the single-GPU reference")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["single", "colocated", "disaggregated"])
    args = parser.parse_args()
    if args.mode == "single":
        run_single_gpu_reference()
    elif args.mode == "colocated":
        run_colocated()
    else:
        run_disaggregated()


if __name__ == "__main__":
    main()
