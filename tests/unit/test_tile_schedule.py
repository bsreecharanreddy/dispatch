"""The index arithmetic most likely to hide an off-by-one, tested on CPU
before any kernel reads it."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.tile_schedule import build_tile_schedule


def test_tiles_cover_each_expert_including_an_empty_one() -> None:
    schedule = build_tile_schedule(torch.tensor([5, 0, 3, 9]), block_m=4)

    assert schedule.block_m == 4
    assert schedule.num_tiles == 6
    assert schedule.tile_expert.tolist() == [0, 0, 2, 3, 3, 3]
    assert schedule.tile_valid_rows.tolist() == [4, 1, 3, 4, 4, 1]
    assert schedule.group_offsets == (0, 5, 5, 8, 17)


def test_row_starts_are_globally_contiguous() -> None:
    schedule = build_tile_schedule(torch.tensor([6, 4]), block_m=4)

    assert schedule.tile_row_start.tolist() == [0, 4, 6]
    assert schedule.tile_valid_rows.tolist() == [4, 2, 4]


def test_every_row_is_covered_once_and_no_tile_crosses_an_expert_boundary() -> None:
    group_sizes = torch.tensor([7, 0, 16, 1, 33])
    schedule = build_tile_schedule(group_sizes, block_m=16)

    covered = torch.zeros(int(group_sizes.sum()), dtype=torch.int64)
    for expert, start, valid in zip(
        schedule.tile_expert.tolist(),
        schedule.tile_row_start.tolist(),
        schedule.tile_valid_rows.tolist(),
        strict=True,
    ):
        assert schedule.group_offsets[expert] <= start
        assert start + valid <= schedule.group_offsets[expert + 1]
        covered[start : start + valid] += 1

    assert torch.equal(covered, torch.ones_like(covered))


def test_all_experts_empty_gives_no_tiles() -> None:
    schedule = build_tile_schedule(torch.tensor([0, 0, 0]), block_m=4)

    assert schedule.num_tiles == 0
    assert schedule.group_offsets == (0, 0, 0, 0)


def test_schedule_tensors_are_int32() -> None:
    schedule = build_tile_schedule(torch.tensor([3]), block_m=4)

    assert schedule.tile_expert.dtype == torch.int32
    assert schedule.tile_row_start.dtype == torch.int32
    assert schedule.tile_valid_rows.dtype == torch.int32


def test_rejects_non_positive_block_m() -> None:
    with pytest.raises(ValueError, match="block_m"):
        build_tile_schedule(torch.tensor([4]), block_m=0)
