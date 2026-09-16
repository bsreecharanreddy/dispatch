"""CPU-only: proves the expert-to-rank sharding and dispatch/combine
bookkeeping is correct before Phase 3's real DeepEP integration ever
touches a GPU. simulate_ep_moe_routed must reproduce grouped_moe_routed's
existing single-process output exactly, since it's the same computation
partitioned by expert-owning rank and summed back.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.expert_parallel import assign_experts_to_ranks, simulate_ep_moe_routed
from dispatch.kernels.moe_forward import (
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
