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

from typing import TYPE_CHECKING

import torch

from dispatch.kernels.integration import MoEInfer
from dispatch.kernels.moe_forward import (
    GroupedMatmul,
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
)

if TYPE_CHECKING:
    # deep_ep needs 2 real GPUs to even build -- not installed on this dev
    # box or in CI (same boundary as triton). `from __future__ import
    # annotations` above means this is never evaluated at runtime, only by
    # mypy (which has its own ignore_missing_imports override for deep_ep).
    from deep_ep import Buffer


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
    # Sized to cover both topk_idx's observed max AND local_expert_ids' own
    # max -- not topk_idx's alone. A real batch's tokens don't always route
    # to every local expert (confirmed live 2026-09-16 against the real
    # 64-expert model: local_expert_ids can reach higher than whatever this
    # particular batch's topk_idx happens to touch), so sizing on topk_idx
    # alone left local_index_of[local_expert_ids] indexing out of bounds.
    # topk_idx can also be entirely empty -- confirmed live 2026-09-17 with
    # 4-way EP (finer sharding than Phase 3's 2-way) on short prompts, where
    # a rank can receive zero tokens for any of its local experts at all --
    # and .max() on an empty tensor raises, so that case sizes off
    # local_expert_ids alone.
    index_span = (
        int(local_expert_ids.max().item()) + 1
        if topk_idx.numel() == 0
        else max(int(topk_idx.max().item()), int(local_expert_ids.max().item())) + 1
    )
    local_index_of = torch.zeros(index_span, dtype=torch.int64, device=topk_idx.device)
    local_index_of[local_expert_ids] = torch.arange(
        local_expert_ids.numel(), device=topk_idx.device
    )
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


def make_ep_moe_infer(
    local_weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    buffer: Buffer,
    num_experts: int,
    *,
    block_m: int = 16,
) -> MoEInfer:
    """DeepEP's real dispatch()/combine() round trip wired into the
    grouped-GEMM path. Uses DeepEP's V1 (legacy) `Buffer`, not V2's
    `ElasticBuffer`: V2's NCCL Gin backend requires NVSwitch-level
    multicast (GPU Fabric Manager), unavailable on this project's rented
    pod (confirmed live 2026-09-16 -- see deepep_smoke_test.py's
    docstring). V1's recv_topk_idx comes back already remapped to this
    rank's LOCAL 0..num_local_experts-1 indexing (confirmed against
    DeepEP's own tests/legacy/test_intranode.py), so local_expert_ids
    here is the identity range -- local_expert_contribution's own remap
    becomes a no-op that still correctly zeros the weight for DeepEP's
    -1 padding sentinel (rows this rank received but doesn't own)."""
    num_experts_per_rank = local_weights.num_experts
    identity_local_ids = torch.arange(num_experts_per_rank)

    @torch.no_grad()
    def moe_infer(
        x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        top_k = flat_expert_indices.numel() // x.shape[0]
        topk_idx = flat_expert_indices.view(-1, top_k)
        # V1's dispatch requires topk_weights as float32 (its own C++
        # assertion, confirmed live 2026-09-16) -- cast back to x's dtype
        # after receiving so grouped_moe_routed's output dtype matches
        # the rest of the model.
        topk_weight = flat_expert_weights.view(-1, top_k).to(torch.float32)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)
        recv_x, recv_topk_idx, recv_topk_weight, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weight,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        local_out = local_expert_contribution(
            recv_x,
            recv_topk_idx,
            recv_topk_weight.to(x.dtype),
            local_weights,
            matmul,
            identity_local_ids.to(recv_x.device),
            block_m=block_m,
        )

        # No topk_weights on combine: DeepEP's own documented forward-pass
        # example (docs/legacy.md's combine_forward) omits it too -- the
        # weighting already happened above, in local_expert_contribution's
        # call into grouped_moe_routed. Passing topk_weights here would be
        # for the backward pass (gradient w.r.t. the gate weights), not
        # this forward-inference path.
        combined_x: torch.Tensor
        combined_x, _, _ = buffer.combine(local_out, handle)
        return combined_x

    return moe_infer


def patch_moe_infer_ep(
    model: torch.nn.Module,
    matmul: GroupedMatmul,
    buffer: Buffer,
    rank: int,
    n_ranks: int,
    *,
    block_m: int = 16,
) -> int:
    """Same swap-in contract as moe_forward.py's patch_moe_infer, but each
    layer's moe_infer only computes this rank's expert shard, dispatching/
    combining the rest via DeepEP's real Buffer."""
    model.eval()
    patched = 0
    for module in model.modules():
        if not hasattr(module, "moe_infer"):
            continue
        experts = module.experts
        if not isinstance(experts, torch.nn.ModuleList):
            raise TypeError(
                f"expected {type(module).__name__}.experts to be nn.ModuleList, "
                f"got {type(experts).__name__}"
            )
        n_experts = len(experts)
        rank_of_expert = assign_experts_to_ranks(n_experts, n_ranks)
        local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0]
        all_weights = stack_expert_weights(experts)
        local_weights = StackedExpertWeights(
            gate=all_weights.gate[local_expert_ids],
            up=all_weights.up[local_expert_ids],
            down=all_weights.down[local_expert_ids],
        )
        module.moe_infer = make_ep_moe_infer(  # type: ignore[assignment]
            local_weights, matmul, buffer, n_experts, block_m=block_m
        )
        patched += 1
    return patched
