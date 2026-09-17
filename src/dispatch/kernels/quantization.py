"""Weight-only int8 quantization for MoE expert weights: per-output-channel,
symmetric, round-to-nearest scales computed directly in dispatch's own
code -- not sourced from bitsandbytes/AWQ, per
docs/design/2026-09-16-phase-5a-quantization.md section 2. Activations are
never quantized; only expert weight tensors are."""

from __future__ import annotations

from dataclasses import dataclass

import torch

INT8_MAX = 127


@dataclass(frozen=True)
class QuantizedTensor:
    """An (E, N, K) weight tensor quantized to int8, one scale per expert
    per output channel (E, N)."""

    data: torch.Tensor
    scale: torch.Tensor


def quantize_per_channel_int8(weight: torch.Tensor) -> QuantizedTensor:
    """weight: (..., N, K). scale[..., n] = max(abs(weight[..., n, :])) / 127.
    An all-zero channel's scale is clamped away from zero so dividing by it
    is a no-op (result: 0) rather than a NaN-producing divide-by-zero."""
    absmax = weight.detach().abs().amax(dim=-1)
    scale = (absmax / INT8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    quantized = (weight.detach() / scale.unsqueeze(-1)).round().clamp(-INT8_MAX, INT8_MAX)
    return QuantizedTensor(data=quantized.to(torch.int8), scale=scale.to(torch.float32))


def dequantize_int8(qtensor: QuantizedTensor) -> torch.Tensor:
    return qtensor.data.to(torch.float32) * qtensor.scale.unsqueeze(-1)
