"""Token-to-expert grouping: sorts (token, top-k slot) pairs into
per-expert-contiguous rows so a grouped GEMM can treat each expert's tokens
as one dense matmul -- the same sort-by-expert step DeepseekMoE.moe_infer
performs inline, pulled out as its own tested unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenGrouping:
    """Row i of the grouped layout is token sorted_token_idx[i], weighted by
    sorted_weight[i]; the first group_sizes[0] rows belong to expert 0, and
    so on (a group may be empty)."""

    sorted_token_idx: torch.Tensor  # (num_tokens * top_k,) int64
    sorted_weight: torch.Tensor  # (num_tokens * top_k,)
    group_sizes: torch.Tensor  # (n_routed_experts,) int64


def group_tokens_by_expert(
    topk_idx: torch.Tensor, topk_weight: torch.Tensor, n_routed_experts: int
) -> TokenGrouping:
    if topk_idx.shape != topk_weight.shape:
        raise ValueError(
            f"topk_idx.shape {tuple(topk_idx.shape)} != "
            f"topk_weight.shape {tuple(topk_weight.shape)}"
        )
    num_tokens, top_k = topk_idx.shape
    flat_expert_idx = topk_idx.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=topk_idx.device).repeat_interleave(top_k)
    sort_order = torch.argsort(flat_expert_idx, stable=True)
    return TokenGrouping(
        sorted_token_idx=flat_token_idx[sort_order],
        sorted_weight=topk_weight.reshape(-1)[sort_order],
        group_sizes=torch.bincount(flat_expert_idx, minlength=n_routed_experts),
    )


def ungroup_and_combine(
    grouped_output: torch.Tensor, grouping: TokenGrouping, num_tokens: int
) -> torch.Tensor:
    """Weights each row by its gate weight and sums the top_k rows that
    share a token back into that token's output row."""
    if grouped_output.shape[0] != grouping.sorted_token_idx.shape[0]:
        raise ValueError(
            f"grouped_output has {grouped_output.shape[0]} rows, "
            f"grouping describes {grouping.sorted_token_idx.shape[0]}"
        )
    weighted = grouped_output * grouping.sorted_weight.unsqueeze(-1)
    combined = torch.zeros(
        num_tokens,
        grouped_output.shape[-1],
        dtype=grouped_output.dtype,
        device=grouped_output.device,
    )
    return combined.index_add_(0, grouping.sorted_token_idx, weighted)
