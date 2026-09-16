"""Expert-to-rank sharding for multi-GPU expert-parallel MoE (Phase 3).
assign_experts_to_ranks is a plain contiguous split, matching DeepEP's own
required convention exactly (confirmed live 2026-09-15 against its test
code: num_local_experts = num_experts // world_size, dst_rank = topk_idx
// num_local_experts) -- not an independent design choice.
simulate_ep_moe_routed proves the dispatch/local-compute/combine
bookkeeping is correct in a single process, with no GPU and no DeepEP
import, before the real library is ever involved.
"""

from __future__ import annotations

import torch

from dispatch.kernels.moe_forward import GroupedMatmul, StackedExpertWeights, grouped_moe_routed


def assign_experts_to_ranks(n_experts: int, n_ranks: int) -> torch.Tensor:
    """expert_id -> rank_id, contiguous blocks (e.g. 64 experts over 2
    ranks -> experts 0-31 on rank 0, 32-63 on rank 1). Every rank computes
    this identically and independently -- nothing to communicate."""
    if n_experts <= 0 or n_ranks <= 0:
        raise ValueError(f"n_experts={n_experts} and n_ranks={n_ranks} must both be positive")
    if n_experts < n_ranks:
        raise ValueError(f"n_experts={n_experts} is fewer than n_ranks={n_ranks}")
    return torch.arange(n_experts) * n_ranks // n_experts


def local_expert_contribution(  # routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    local_expert_ids: torch.Tensor,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    """The rows a real dispatch + local-compute + combine round trip would
    contribute for one rank: zero for any (token, slot) whose expert isn't
    in local_expert_ids, the weighted expert output otherwise. Feeds
    straight into the existing grouped_moe_routed by remapping global
    expert ids to this rank's local 0..num_local-1 indexing and zeroing
    the weight (not dropping the row) for non-local slots, so shapes stay
    valid without changing grouped_moe_routed itself."""
    is_local = torch.isin(topk_idx, local_expert_ids)
    local_index_of = torch.zeros(int(topk_idx.max().item()) + 1, dtype=torch.int64)
    local_index_of[local_expert_ids] = torch.arange(local_expert_ids.numel())
    local_topk_idx = local_index_of[topk_idx.clamp(min=0)]
    local_topk_weight = torch.where(is_local, topk_weight, torch.zeros_like(topk_weight))
    return grouped_moe_routed(
        x, local_topk_idx, local_topk_weight, local_weights, matmul, block_m=block_m
    )


def simulate_ep_moe_routed(  # routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    rank_of_expert: torch.Tensor,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    """Single-process stand-in for real cross-GPU dispatch + per-rank
    local compute + combine: sums each rank's local_expert_contribution
    over only the experts rank_of_expert assigns it. Proves the
    sharding/combine arithmetic in isolation from DeepEP's real
    transport -- no process group, no GPU, no DeepEP import."""
    n_ranks = int(rank_of_expert.max().item()) + 1
    combined = torch.zeros_like(x)
    for rank in range(n_ranks):
        local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0]
        local_weights = StackedExpertWeights(
            gate=weights.gate[local_expert_ids],
            up=weights.up[local_expert_ids],
            down=weights.down[local_expert_ids],
        )
        combined = combined + local_expert_contribution(
            x, topk_idx, topk_weight, local_weights, matmul, local_expert_ids, block_m=block_m
        )
    return combined
