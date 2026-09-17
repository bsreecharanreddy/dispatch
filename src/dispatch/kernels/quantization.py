"""Weight-only int8 quantization for MoE expert weights: per-output-channel,
symmetric, round-to-nearest scales computed directly in dispatch's own
code -- not sourced from bitsandbytes/AWQ, per
docs/design/2026-09-16-phase-5a-quantization.md section 2. Activations are
never quantized; only expert weight tensors are."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine
from dispatch.kernels.moe_forward import StackedExpertWeights, torch_grouped_matmul
from dispatch.kernels.tile_schedule import TileSchedule, build_tile_schedule

INT8_MAX = 127


@dataclass(frozen=True)
class QuantizedTensor:
    """An (E, N, K) weight tensor quantized to int8, one scale per expert
    per output channel (E, N)."""

    data: torch.Tensor
    scale: torch.Tensor


def quantize_per_channel_int8(weight: torch.Tensor) -> QuantizedTensor:
    """weight: (..., N, K). scale[..., n] = max(abs(weight[..., n, :])) / 127.
    An all-zero channel's scale is set to 1.0 (any nonzero placeholder
    works: dequantizing a channel that quantized to all-zero data just
    multiplies 0 by it) rather than a NaN-producing divide-by-zero.

    absmax, scale, and the quotient are all computed in float32
    regardless of weight's own dtype. Doing this arithmetic directly in
    bf16/fp16 measurably widens round-trip error (bf16's 8-bit mantissa
    pushes the worst case to ~1.5x the ideal 0.5-quantization-step
    bound), and for fp16 specifically, clamping the scale floor at
    fp16's own `tiny()` (~6e-5) -- an earlier version of this guard --
    silently clamped small-but-nonzero channels' scales *upward*,
    throwing away several bits of int8 range for no reason tied to that
    channel's actual magnitude. Computing in fp32 sidesteps both."""
    weight_fp32 = weight.detach().float()
    absmax = weight_fp32.abs().amax(dim=-1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / INT8_MAX)
    quantized = (weight_fp32 / scale.unsqueeze(-1)).round().clamp(-INT8_MAX, INT8_MAX)
    return QuantizedTensor(data=quantized.to(torch.int8), scale=scale.to(torch.float32))


def dequantize_int8(qtensor: QuantizedTensor) -> torch.Tensor:
    return qtensor.data.to(torch.float32) * qtensor.scale.unsqueeze(-1)


@dataclass(frozen=True)
class QuantizedStackedExpertWeights:
    """Each projection's int8-quantized weights for every expert."""

    gate: QuantizedTensor
    up: QuantizedTensor
    down: QuantizedTensor

    @property
    def num_experts(self) -> int:
        return int(self.gate.data.shape[0])


QuantizedGroupedMatmul = Callable[[torch.Tensor, QuantizedTensor, TileSchedule], torch.Tensor]


def quantize_stacked_weights(weights: StackedExpertWeights) -> QuantizedStackedExpertWeights:
    return QuantizedStackedExpertWeights(
        gate=quantize_per_channel_int8(weights.gate),
        up=quantize_per_channel_int8(weights.up),
        down=quantize_per_channel_int8(weights.down),
    )


def torch_grouped_matmul_dequant(
    x: torch.Tensor, qweight: QuantizedTensor, schedule: TileSchedule
) -> torch.Tensor:
    """The Triton int8 kernel's correctness oracle: dequantizes qweight --
    the *same* already-quantized weights the kernel sees -- and runs the
    same per-expert-slice matmul torch_grouped_matmul does. Not a
    model-quality reference: both sides here use identical quantized
    weights, so nothing should diverge beyond float precision."""
    return torch_grouped_matmul(x, dequantize_int8(qweight).to(x.dtype), schedule)


def grouped_moe_routed_quantized(  # noqa: PLR0913 -- routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: QuantizedStackedExpertWeights,
    matmul: QuantizedGroupedMatmul,
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


def stacked_weights_nbytes(weights: StackedExpertWeights) -> int:
    return sum(t.element_size() * t.nelement() for t in (weights.gate, weights.up, weights.down))


def quantized_stacked_weights_nbytes(weights: QuantizedStackedExpertWeights) -> int:
    total = 0
    for qtensor in (weights.gate, weights.up, weights.down):
        total += qtensor.data.element_size() * qtensor.data.nelement()
        total += qtensor.scale.element_size() * qtensor.scale.nelement()
    return total
