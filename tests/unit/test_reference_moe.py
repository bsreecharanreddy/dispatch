"""CPU-only, toy dims. Covers the shapes, the zero-token expert (the edge
case a grouped-GEMM kernel mishandles first), and the one routing detail
that is easy to get wrong from memory: this model does not renormalize its
top-k weights."""

from __future__ import annotations

import torch

from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


def test_forward_preserves_shape() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)

    assert moe(hidden_states).shape == hidden_states.shape


def test_route_selects_top_k_distinct_experts_per_token() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)

    topk_idx, topk_weight = moe.route(torch.randn(7, TOY_CONFIG.hidden_size))

    assert topk_idx.shape == (7, 2)
    assert topk_weight.shape == (7, 2)
    assert all(len(set(row.tolist())) == 2 for row in topk_idx)


def test_topk_weights_are_not_renormalized() -> None:
    """deepseek-moe-16b-base's config.json has norm_topk_prob=false."""
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)

    _, topk_weight = moe.route(torch.randn(20, TOY_CONFIG.hidden_size))

    row_sums = topk_weight.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums))


def test_forward_is_routed_plus_shared_experts() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(5, TOY_CONFIG.hidden_size)

    assert moe.shared_experts is not None
    torch.testing.assert_close(
        moe(hidden_states), moe.routed(hidden_states) + moe.shared_experts(hidden_states)
    )


def test_routed_handles_experts_that_receive_zero_tokens() -> None:
    torch.manual_seed(0)
    config = MoEConfig(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=1,
    )
    moe = ReferenceMoE(config)
    hidden_states = torch.randn(2, config.hidden_size)

    topk_idx, _ = moe.route(hidden_states)
    assert len(set(topk_idx.reshape(-1).tolist())) < config.n_routed_experts

    assert moe.routed(hidden_states).shape == hidden_states.shape
