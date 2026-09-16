"""CPU-only: proves the expert-to-rank sharding and dispatch/combine
bookkeeping is correct before Phase 3's real DeepEP integration ever
touches a GPU. simulate_ep_moe_routed must reproduce grouped_moe_routed's
existing single-process output exactly, since it's the same computation
partitioned by expert-owning rank and summed back.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.expert_parallel import (
    assign_experts_to_ranks,
    local_expert_contribution,
    simulate_ep_moe_routed,
)
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=3,
)


def test_assign_experts_to_ranks_splits_64_experts_evenly_over_2_ranks() -> None:
    ranks = assign_experts_to_ranks(64, 2)

    assert ranks[:32].eq(0).all()
    assert ranks[32:].eq(1).all()


def test_assign_experts_to_ranks_rejects_more_ranks_than_experts() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        assign_experts_to_ranks(1, 2)


def test_assign_experts_to_ranks_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        assign_experts_to_ranks(0, 2)


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_simulate_ep_moe_routed_matches_the_non_ep_reference(n_ranks: int) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(11, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    expected = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul
    )

    rank_of_expert = assign_experts_to_ranks(TOY_CONFIG.n_routed_experts, n_ranks)
    actual = simulate_ep_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul, rank_of_expert
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_local_expert_contribution_matches_cpu_when_inputs_are_on_cuda() -> None:
    """Regression test for a real bug (found 2026-09-16 while wiring Task 4's
    real EP layer): local_expert_contribution's remap tensors defaulted to
    CPU regardless of topk_idx's device, so the first real CUDA call would
    have raised a device-mismatch error. CPU-only tests can't catch this --
    CPU tensors trivially satisfy the same-device check -- so this needs an
    actual CUDA device."""
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(11, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    local_expert_ids = torch.arange(4)
    local_weights = StackedExpertWeights(
        gate=weights.gate[:4], up=weights.up[:4], down=weights.down[:4]
    )

    expected = local_expert_contribution(
        hidden_states, topk_idx, topk_weight, local_weights, torch_grouped_matmul, local_expert_ids
    )
    actual = local_expert_contribution(
        hidden_states.cuda(),
        topk_idx.cuda(),
        topk_weight.cuda(),
        StackedExpertWeights(
            gate=local_weights.gate.cuda(),
            up=local_weights.up.cuda(),
            down=local_weights.down.cuda(),
        ),
        torch_grouped_matmul,
        local_expert_ids.cuda(),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
