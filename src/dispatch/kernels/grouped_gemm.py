"""Triton grouped-GEMM kernels for MoE expert computation: drop-in
GroupedMatmul backends for dispatch.kernels.moe_forward, each held to
assert_matches_reference against its eager torch_grouped_matmul.

Adapted from Triton's official Group GEMM tutorial (08-grouped-gemm) for
MoE's shape of the problem: every group is a contiguous slice of one
sorted-by-expert tensor and every expert's weight shares one (N, K) shape,
so a host-built TileSchedule stands in for the tutorial's per-group
pointer arrays.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dispatch.kernels.tile_schedule import TileSchedule

MIN_BLOCK_M = 16  # tl.dot's smallest tile dimension
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64
DEFAULT_GROUP_SIZE_M = 8


@triton.jit  # type: ignore[untyped-decorator]
def _matmul_tile(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    m_tile,
    n_tile,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.load(tile_expert_ptr + m_tile)
    row_start = tl.load(tile_row_start_ptr + m_tile)
    valid_rows = tl.load(tile_valid_rows_ptr + m_tile)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m[:, None] < valid_rows
    col_mask = offs_n[None, :] < n

    x_ptrs = x_ptr + (row_start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        x_tile = tl.load(x_ptrs, mask=row_mask & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + (row_start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask & col_mask)


@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _matmul_tile(
        x_ptr,
        w_ptr,
        out_ptr,
        tile_expert_ptr,
        tile_row_start_ptr,
        tile_valid_rows_ptr,
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        n,
        k,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_persistent_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    num_m_tiles,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_n_tiles = tl.cdiv(n, BLOCK_N)
    tiles_per_group = GROUP_SIZE_M * num_n_tiles
    for tile_id in tl.range(tl.program_id(axis=0), num_m_tiles * num_n_tiles, NUM_SMS):
        first_m_tile = (tile_id // tiles_per_group) * GROUP_SIZE_M
        group_rows = tl.minimum(num_m_tiles - first_m_tile, GROUP_SIZE_M)
        m_tile = first_m_tile + (tile_id % tiles_per_group) % group_rows
        n_tile = (tile_id % tiles_per_group) // group_rows
        _matmul_tile(
            x_ptr,
            w_ptr,
            out_ptr,
            tile_expert_ptr,
            tile_row_start_ptr,
            tile_valid_rows_ptr,
            m_tile,
            n_tile,
            n,
            k,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_om,
            stride_on,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )


def grouped_matmul(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> torch.Tensor:
    """One CTA per (m_tile, n_tile) -- the simple, correctness-first launch."""
    out = _validated_output(x, expert_weight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = expert_weight.shape[1], expert_weight.shape[2]
    grid = (schedule.num_tiles, triton.cdiv(n, block_n))
    _grouped_matmul_kernel[grid](
        x,
        expert_weight,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        n,
        k,
        *x.stride(),
        *expert_weight.stride(),
        *out.stride(),
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def grouped_matmul_persistent(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    group_size_m: int = DEFAULT_GROUP_SIZE_M,
) -> torch.Tensor:
    """One long-lived CTA per SM, each looping over (m_tile, n_tile) work in
    L2-friendly grouped order instead of one launch-slot per tile."""
    out = _validated_output(x, expert_weight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = expert_weight.shape[1], expert_weight.shape[2]
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    _grouped_matmul_persistent_kernel[(num_sms,)](
        x,
        expert_weight,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        schedule.num_tiles,
        n,
        k,
        *x.stride(),
        *expert_weight.stride(),
        *out.stride(),
        NUM_SMS=num_sms,
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
    )
    return out


def _validated_output(
    x: torch.Tensor, expert_weight: torch.Tensor, schedule: TileSchedule
) -> torch.Tensor:
    if not (x.is_cuda and expert_weight.is_cuda):
        raise ValueError("the Triton grouped-GEMM kernels need CUDA tensors")
    if x.dtype != expert_weight.dtype:
        raise ValueError(f"dtype mismatch: x is {x.dtype}, expert_weight is {expert_weight.dtype}")
    if x.shape[1] != expert_weight.shape[2]:
        raise ValueError(
            f"K mismatch: x has {x.shape[1]}, expert_weight has {expert_weight.shape[2]}"
        )
    if schedule.block_m < MIN_BLOCK_M:
        raise ValueError(
            f"schedule.block_m={schedule.block_m} is below tl.dot's minimum of {MIN_BLOCK_M}"
        )
    return torch.empty((x.shape[0], expert_weight.shape[1]), device=x.device, dtype=x.dtype)
