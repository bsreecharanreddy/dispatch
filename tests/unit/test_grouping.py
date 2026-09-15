"""CPU-only. Hand-computed layouts, so a reader can check each expectation
against the sort-by-expert rule without running anything."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine


def test_group_sizes_count_every_slot_including_an_empty_expert() -> None:
    topk_idx = torch.tensor([[0, 2], [1, 0], [2, 2]])

    grouping = group_tokens_by_expert(topk_idx, torch.ones(3, 2), n_routed_experts=4)

    assert grouping.group_sizes.tolist() == [2, 1, 3, 0]


def test_rows_are_sorted_expert_major_and_stable_within_an_expert() -> None:
    # flat (token, expert) pairs: (0,2) (0,0) (1,0) (1,1) (2,2) (2,1)
    topk_idx = torch.tensor([[2, 0], [0, 1], [2, 1]])
    topk_weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    grouping = group_tokens_by_expert(topk_idx, topk_weight, n_routed_experts=3)

    assert grouping.sorted_token_idx.tolist() == [0, 1, 1, 2, 0, 2]
    torch.testing.assert_close(grouping.sorted_weight, torch.tensor([0.2, 0.3, 0.4, 0.6, 0.1, 0.5]))


def test_ungroup_and_combine_weights_and_sums_rows_per_token() -> None:
    # token 0 -> experts 0 and 1 (weights 2, 3); token 1 -> experts 0 and 1
    # (weights 4, 5).
    topk_idx = torch.tensor([[0, 1], [0, 1]])
    topk_weight = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    grouping = group_tokens_by_expert(topk_idx, topk_weight, n_routed_experts=2)
    # grouped rows: expert 0 = [token 0, token 1], expert 1 = [token 0, token 1]
    grouped_output = torch.tensor([[1.0], [10.0], [100.0], [1000.0]])

    combined = ungroup_and_combine(grouped_output, grouping, num_tokens=2)

    assert combined.tolist() == [[2.0 * 1 + 3.0 * 100], [4.0 * 10 + 5.0 * 1000]]


def test_group_tokens_by_expert_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        group_tokens_by_expert(
            torch.zeros(3, 2, dtype=torch.long), torch.zeros(3, 3), n_routed_experts=4
        )


def test_ungroup_and_combine_rejects_row_count_mismatch() -> None:
    grouping = group_tokens_by_expert(torch.tensor([[0]]), torch.ones(1, 1), n_routed_experts=1)

    with pytest.raises(ValueError, match="rows"):
        ungroup_and_combine(torch.zeros(2, 4), grouping, num_tokens=1)
