"""Ad hoc measurement (not a committed pytest test): what per-local-expert
token-count distribution does DeepEP's real dispatch() actually produce on
the real DeepSeekMoE-16B model, across its real 27 MoE layers and
DEFAULT_PROMPTS? Directly answers Phase 3's thesis question -- does this
land in Phase 1's 16-128-token persistent-kernel win region, the
512+-token loss region, or somewhere else -- from measured data, not a
guess. Wraps local_expert_contribution to record each call's per-local-
expert bincount as a side effect; no change to the committed EP layer.

Run: NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=2 \
    scripts/gpu/measure_real_expert_token_counts.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from deep_ep import Buffer

from dispatch.benchmark.harness import load_model
from dispatch.kernels import expert_parallel
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.expert_parallel import patch_moe_infer_ep
from dispatch.kernels.moe_forward import GroupedMatmul, StackedExpertWeights

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]

_observed_counts: list[int] = []
_real_local_expert_contribution = expert_parallel.local_expert_contribution


def _recording_local_expert_contribution(  # noqa: PLR0913, PLR0917 -- mirrors the wrapped function's own signature
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    local_expert_ids: torch.Tensor,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    counts = torch.bincount(
        topk_idx[torch.isin(topk_idx, local_expert_ids)], minlength=local_expert_ids.numel()
    )
    _observed_counts.extend(int(c) for c in counts.tolist())
    return _real_local_expert_contribution(
        x, topk_idx, topk_weight, local_weights, matmul, local_expert_ids, block_m=block_m
    )


def main() -> None:
    expert_parallel.local_expert_contribution = _recording_local_expert_contribution

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    dist.barrier()

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
    buffer = Buffer(group, num_nvl_bytes, 0)
    patch_moe_infer_ep(model, resolve_backend("naive"), buffer, rank, world_size)

    with torch.no_grad():
        for prompt in DEFAULT_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            model(input_ids=input_ids)

    counts = torch.tensor(_observed_counts, dtype=torch.float32)
    nonzero = counts[counts > 0]
    print(
        f"[rank {rank}] {counts.numel()} (layer, local-expert) observations, "
        f"{int((counts > 0).sum())} received >=1 token; "
        f"over those: min={int(nonzero.min())} max={int(nonzero.max())} "
        f"mean={float(nonzero.mean()):.2f} median={float(nonzero.median()):.1f}",
        flush=True,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
