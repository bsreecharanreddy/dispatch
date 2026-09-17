"""Triton int8 weight-only grouped-GEMM kernel: accumulates the raw
int8-times-activation dot product on native bf16/fp16 tensor cores --
casting the int8 weight tile up to the activation's dtype, exact since
int8's range is exactly representable in both -- then applies the
per-output-channel scale once to the finished accumulator, exact since
that scale is invariant across K. This keeps tl.dot's precision profile
identical to grouped_gemm.py's bf16 naive kernel instead of falling back
to the slower fp32/TF32 path a per-K-step dequant would force. Activations
(x) stay bf16/fp16 throughout -- only the weight side is ever int8. Held
to assert_matches_reference against torch_grouped_matmul_dequant fed the
same QuantizedTensor, per docs/design/2026-09-16-phase-5a-quantization.md
section 5.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dispatch.kernels.grouped_gemm import DEFAULT_BLOCK_K, DEFAULT_BLOCK_N, MIN_BLOCK_M
from dispatch.kernels.quantization import QuantizedTensor
from dispatch.kernels.tile_schedule import TileSchedule


@triton.jit  # type: ignore[untyped-decorator]
def _matmul_tile_int8(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    scale_ptr,
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
    stride_se,
    stride_sn,
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
    col_mask_2d = offs_n[None, :] < n
    col_mask_1d = offs_n < n

    x_ptrs = x_ptr + (row_start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )
    scale_ptrs = scale_ptr + expert_id * stride_se + offs_n * stride_sn
    scale_tile = tl.load(scale_ptrs, mask=col_mask_1d, other=1.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        x_tile = tl.load(x_ptrs, mask=row_mask & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask_2d, other=0)
        # int8 -> x_tile.dtype is exact, so casting the weight tile (rather
        # than dequantizing per-K-step) keeps tl.dot on native bf16/fp16
        # tensor cores instead of the ~2x-slower fp32/TF32 path -- the same
        # precision profile as grouped_gemm.py's bf16 kernel. The
        # per-output-channel scale is invariant across K, so it's exact and
        # mathematically equivalent to hoist it outside the loop below.
        acc += tl.dot(x_tile, w_tile.to(x_tile.dtype))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    acc = acc * scale_tile[None, :]

    out_ptrs = out_ptr + (row_start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask & col_mask_2d)


@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_int8_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    scale_ptr,
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
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _matmul_tile_int8(
        x_ptr,
        w_ptr,
        scale_ptr,
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
        stride_se,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def grouped_matmul_int8(
    x: torch.Tensor,
    qweight: QuantizedTensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> torch.Tensor:
    """One CTA per (m_tile, n_tile), dequantizing qweight's int8 tile
    against its per-output-channel scale before accumulating -- the int8
    analog of grouped_gemm.grouped_matmul."""
    out = _validated_quantized_output(x, qweight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = qweight.data.shape[1], qweight.data.shape[2]
    grid = (schedule.num_tiles, triton.cdiv(n, block_n))
    _grouped_matmul_int8_kernel[grid](
        x,
        qweight.data,
        qweight.scale,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        n,
        k,
        *x.stride(),
        *qweight.data.stride(),
        *qweight.scale.stride(),
        *out.stride(),
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _validated_quantized_output(
    x: torch.Tensor, qweight: QuantizedTensor, schedule: TileSchedule
) -> torch.Tensor:
    if not (x.is_cuda and qweight.data.is_cuda and qweight.scale.is_cuda):
        raise ValueError("the Triton int8 grouped-GEMM kernel needs CUDA tensors")
    if qweight.data.dtype != torch.int8:
        raise ValueError(f"expected int8 weight data, got {qweight.data.dtype}")
    if qweight.scale.dtype != torch.float32:
        raise ValueError(f"expected float32 scales, got {qweight.scale.dtype}")
    if qweight.scale.shape != qweight.data.shape[:2]:
        raise ValueError(f"scale shape {tuple(qweight.scale.shape)} does not match weight (E, N)")
    if x.shape[1] != qweight.data.shape[2]:
        raise ValueError(f"K mismatch: x has {x.shape[1]}, qweight has {qweight.data.shape[2]}")
    if schedule.block_m < MIN_BLOCK_M:
        raise ValueError(
            f"schedule.block_m={schedule.block_m} is below tl.dot's minimum of {MIN_BLOCK_M}"
        )
    return torch.empty((x.shape[0], qweight.data.shape[1]), device=x.device, dtype=x.dtype)
