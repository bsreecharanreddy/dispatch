"""CPU-only proof of the int8 quantization math: per-channel scale
computation, round-trip error bounds, and the zero-channel edge case.
Per docs/design/2026-09-16-phase-5a-quantization.md section 2."""

from __future__ import annotations

import torch

from dispatch.kernels.quantization import dequantize_int8, quantize_per_channel_int8


def test_quantize_dequantize_round_trip_is_within_one_quantization_step() -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 5, 7) * 10

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    max_step = weight.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((dequantized - weight).abs() <= max_step * 1.01)


def test_quantized_data_is_int8_and_within_range() -> None:
    torch.manual_seed(0)
    weight = torch.randn(2, 4, 6) * 100

    quantized = quantize_per_channel_int8(weight)

    assert quantized.data.dtype == torch.int8
    assert quantized.scale.dtype == torch.float32
    assert quantized.data.abs().max() <= 127


def test_quantized_shapes_match_weight() -> None:
    weight = torch.randn(6, 8, 10)

    quantized = quantize_per_channel_int8(weight)

    assert quantized.data.shape == (6, 8, 10)
    assert quantized.scale.shape == (6, 8)


def test_an_all_zero_channel_does_not_produce_nan() -> None:
    weight = torch.zeros(1, 2, 4)
    weight[0, 1] = torch.tensor([1.0, -2.0, 3.0, -4.0])

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    assert torch.isfinite(dequantized).all()
    assert torch.equal(dequantized[0, 0], torch.zeros(4))
    assert quantized.data[0, 0].abs().sum() == 0


def test_an_all_zero_channel_with_float16_does_not_produce_nan() -> None:
    weight = torch.zeros(1, 2, 4, dtype=torch.float16)
    weight[0, 1] = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float16)

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    assert torch.isfinite(dequantized).all()
    assert torch.equal(dequantized[0, 0], torch.zeros(4))
    assert quantized.data[0, 0].abs().sum() == 0


def test_scale_is_per_expert_per_output_channel() -> None:
    weight = torch.zeros(2, 2, 3)
    weight[0, 0] = torch.tensor([1.0, -1.0, 0.5])
    weight[0, 1] = torch.tensor([10.0, -10.0, 5.0])

    quantized = quantize_per_channel_int8(weight)

    assert quantized.scale.shape == (2, 2)
    assert quantized.scale[0, 1] > quantized.scale[0, 0]
