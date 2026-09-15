"""Host-side tile schedule for the grouped-GEMM kernels: per-expert group
sizes become a flat list of block_m-row tiles, each tagged with its expert,
its first row in the sorted-by-expert input, and how many of its rows are
real (a group's last tile is usually partial). Pure index arithmetic --
testable on CPU with no Triton involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileSchedule:
    block_m: int
    group_offsets: tuple[int, ...]  # host-side row offsets, len n_experts + 1
    tile_expert: torch.Tensor  # (num_tiles,) int32
    tile_row_start: torch.Tensor  # (num_tiles,) int32
    tile_valid_rows: torch.Tensor  # (num_tiles,) int32, each <= block_m

    @property
    def num_tiles(self) -> int:
        return int(self.tile_expert.shape[0])


def build_tile_schedule(group_sizes: torch.Tensor, block_m: int) -> TileSchedule:
    """Tensors land on group_sizes' device, where the kernel will read them."""
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    sizes: list[int] = group_sizes.tolist()
    offsets = [0]
    tile_expert: list[int] = []
    tile_row_start: list[int] = []
    tile_valid_rows: list[int] = []
    for expert_id, size in enumerate(sizes):
        for local_start in range(0, size, block_m):
            tile_expert.append(expert_id)
            tile_row_start.append(offsets[-1] + local_start)
            tile_valid_rows.append(min(block_m, size - local_start))
        offsets.append(offsets[-1] + size)

    return TileSchedule(
        block_m=block_m,
        group_offsets=tuple(offsets),
        tile_expert=_int32(tile_expert, group_sizes.device),
        tile_row_start=_int32(tile_row_start, group_sizes.device),
        tile_valid_rows=_int32(tile_valid_rows, group_sizes.device),
    )


def _int32(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)
