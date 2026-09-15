"""The routed-expert half of a DeepSeek-style MoE layer, as three grouped
GEMMs over the per-expert-contiguous layout. The GEMM is pluggable:
`torch_grouped_matmul` is the eager-PyTorch implementation of the contract
(one matmul per expert, the pattern DeepseekMoE.moe_infer uses), and the
Triton kernels in grouped_gemm.py are drop-in replacements that must meet
`assert_matches_reference` against it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine
from dispatch.kernels.tile_schedule import TileSchedule, build_tile_schedule

GroupedMatmul = Callable[[torch.Tensor, torch.Tensor, TileSchedule], torch.Tensor]

CORRECTNESS_RTOL = 1.6e-2  # torch.testing.assert_close's own bf16 default


@dataclass(frozen=True)
class StackedExpertWeights:
    """Each projection's weights for every expert, as (n_experts, N, K)."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.gate.shape[0])


def torch_grouped_matmul(
    x: torch.Tensor, expert_weight: torch.Tensor, schedule: TileSchedule
) -> torch.Tensor:
    """out[rows of expert e] = x[rows of expert e] @ expert_weight[e].T"""
    out = x.new_empty((x.shape[0], expert_weight.shape[1]))
    for expert_id, (start, end) in enumerate(itertools.pairwise(schedule.group_offsets)):
        if end > start:
            out[start:end] = x[start:end] @ expert_weight[expert_id].T
    return out


def stack_expert_weights(experts: Iterable[torch.nn.Module]) -> StackedExpertWeights:
    """Stacks each projection into one tensor and re-points every expert's
    Linear at a view of it, so the stack costs no extra memory."""
    modules = list(experts)
    return StackedExpertWeights(
        gate=_stack_and_share([_linear(module, "gate_proj") for module in modules]),
        up=_stack_and_share([_linear(module, "up_proj") for module in modules]),
        down=_stack_and_share([_linear(module, "down_proj") for module in modules]),
    )


def grouped_moe_routed(  # noqa: PLR0913 -- routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    grouping = group_tokens_by_expert(topk_idx, topk_weight, weights.num_experts)
    schedule = build_tile_schedule(grouping.group_sizes, block_m)
    gathered = x[grouping.sorted_token_idx]
    gate_out = matmul(gathered, weights.gate, schedule)
    up_out = matmul(gathered, weights.up, schedule)
    expert_out = matmul(F.silu(gate_out) * up_out, weights.down, schedule)
    return ungroup_and_combine(expert_out, grouping, num_tokens=x.shape[0])


def assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """assert_close with atol scaled to the reference's magnitude, so an
    output whose values are all tiny cannot pass on absolute tolerance alone."""
    atol = CORRECTNESS_RTOL * float(expected.detach().abs().max())
    torch.testing.assert_close(
        actual.detach().float(), expected.detach().float(), rtol=CORRECTNESS_RTOL, atol=atol
    )


def _linear(module: torch.nn.Module, name: str) -> torch.nn.Linear:
    projection = getattr(module, name)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError(f"expected {name} to be nn.Linear, got {type(projection).__name__}")
    return projection


def _stack_and_share(linears: list[torch.nn.Linear]) -> torch.Tensor:
    stacked = torch.stack([linear.weight.detach() for linear in linears])
    for index, linear in enumerate(linears):
        linear.weight = torch.nn.Parameter(stacked[index], requires_grad=False)
    return stacked
