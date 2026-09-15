"""CPU-only proof of the grouped-GEMM contract: the eager backend over the
grouped layout reproduces ReferenceMoE. The Triton kernels only have to
match torch_grouped_matmul -- everything around the GEMM is proven here."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import (
    assert_matches_reference,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import ExpertMLP, MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


def test_torch_grouped_matmul_matches_a_per_row_computation() -> None:
    torch.manual_seed(0)
    group_sizes = torch.tensor([3, 0, 2, 4])
    schedule = build_tile_schedule(group_sizes, block_m=2)
    x = torch.randn(int(group_sizes.sum()), 5)
    weight = torch.randn(4, 6, 5)
    row_expert = torch.repeat_interleave(torch.arange(4), group_sizes).tolist()

    actual = torch_grouped_matmul(x, weight, schedule)

    expected = torch.stack([x[row] @ weight[expert].T for row, expert in enumerate(row_expert)])
    torch.testing.assert_close(actual, expected)


def test_stack_expert_weights_shares_storage_instead_of_copying() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(3, TOY_CONFIG.hidden_size)
    before = [expert(hidden_states) for expert in moe.experts]

    weights = stack_expert_weights(moe.experts)

    assert weights.gate.shape == (4, 16, 8)
    assert weights.down.shape == (4, 8, 16)
    for expert, output_before in zip(moe.experts, before, strict=True):
        assert isinstance(expert, ExpertMLP)
        assert (
            expert.gate_proj.weight.untyped_storage().data_ptr()
            == weights.gate.untyped_storage().data_ptr()
        )
        assert torch.equal(expert(hidden_states), output_before)


@pytest.mark.parametrize("block_m", [1, 4, 16])
def test_grouped_moe_routed_matches_reference_routed_output(block_m: int) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)
    expected = moe.routed(hidden_states)

    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    actual = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul, block_m=block_m
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_assert_matches_reference_scales_atol_to_the_output() -> None:
    expected = torch.full((4,), 1e-3)

    assert_matches_reference(expected * 1.01, expected)
    with pytest.raises(AssertionError):
        # a flat atol=1e-2 would pass this: every value is off by 100%
        assert_matches_reference(expected * 2, expected)


def test_stack_expert_weights_rejects_a_non_linear_projection() -> None:
    class NotAnExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Identity()

    with pytest.raises(TypeError, match="gate_proj"):
        stack_expert_weights([NotAnExpert()])
