"""CPU-only proof of the int8 quantization math: per-channel scale
computation, round-trip error bounds, and the zero-channel edge case.
Per docs/design/2026-09-16-phase-5a-quantization.md section 2."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.quantization import (
    QuantizedStackedExpertWeights,
    QuantizedTensor,
    dequantize_int8,
    grouped_moe_routed_quantized,
    quantize_per_channel_int8,
    quantize_stacked_weights,
    quantized_stacked_weights_nbytes,
    stacked_weights_nbytes,
    torch_grouped_matmul_dequant,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_quantize_dequantize_round_trip_is_within_one_quantization_step(
    dtype: torch.dtype,
) -> None:
    """The quantize/dequantize arithmetic itself runs in float32 regardless
    of weight's own dtype, so this bound must hold identically for all
    three -- it would fail for bf16 (up to ~1.5x this bound) if the
    arithmetic were ever done in the input dtype again."""
    torch.manual_seed(0)
    weight = (torch.randn(3, 5, 7) * 10).to(dtype)

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    max_step = weight.float().abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((dequantized - weight.float()).abs() <= max_step * 1.01)


def test_a_small_nonzero_float16_channel_still_uses_the_full_int8_range() -> None:
    """A channel whose absmax (here ~0.001) is well below float16's
    `tiny()` floor (~6.1e-5) must not have its scale clamped upward --
    that would throw away int8 range for a reason unrelated to the
    channel's own magnitude. Regression test for a real bug: an earlier
    version of the all-zero-channel guard clamped every channel's scale
    at fp16's tiny floor, not just the all-zero one -- ideal scale here
    is ~7.9e-6, which the old clamp forced up to ~6.1e-5, cutting the
    largest value's int8 code from 127 down to 16."""
    weight = torch.zeros(1, 1, 4, dtype=torch.float16)
    weight[0, 0] = torch.tensor([0.001, -0.0009, 0.0008, -0.0007], dtype=torch.float16)

    quantized = quantize_per_channel_int8(weight)

    assert quantized.data[0, 0].abs().max() >= 120


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


TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


def test_grouped_moe_routed_quantized_matches_dequantized_weights_reference() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    quantized_weights = quantize_stacked_weights(weights)

    dequantized = StackedExpertWeights(
        gate=dequantize_int8(quantized_weights.gate),
        up=dequantize_int8(quantized_weights.up),
        down=dequantize_int8(quantized_weights.down),
    )
    expected = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, dequantized, torch_grouped_matmul
    )

    actual = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, quantized_weights, torch_grouped_matmul_dequant
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_quantize_stacked_weights_shapes() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    weights = stack_expert_weights(moe.experts)

    quantized = quantize_stacked_weights(weights)

    assert quantized.num_experts == 4
    assert quantized.gate.data.shape == weights.gate.shape
    assert quantized.down.data.shape == weights.down.shape


def test_quantized_stacked_weights_are_smaller_than_bf16() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    weights = stack_expert_weights(moe.experts)
    bf16_weights = StackedExpertWeights(
        gate=weights.gate.to(torch.bfloat16),
        up=weights.up.to(torch.bfloat16),
        down=weights.down.to(torch.bfloat16),
    )
    quantized = quantize_stacked_weights(weights)

    bf16_bytes = stacked_weights_nbytes(bf16_weights)
    int8_bytes = quantized_stacked_weights_nbytes(quantized)

    assert int8_bytes < bf16_bytes


def test_stacked_weights_nbytes_counts_every_projection() -> None:
    weights = StackedExpertWeights(
        gate=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        up=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        down=torch.zeros(2, 4, 3, dtype=torch.bfloat16),
    )

    assert stacked_weights_nbytes(weights) == 3 * (2 * 3 * 4 * 2)


def test_quantized_stacked_weights_nbytes_counts_data_and_scale() -> None:
    tensor = QuantizedTensor(
        data=torch.zeros(2, 3, 4, dtype=torch.int8), scale=torch.zeros(2, 3, dtype=torch.float32)
    )
    weights = QuantizedStackedExpertWeights(gate=tensor, up=tensor, down=tensor)

    expected = 3 * (2 * 3 * 4 * 1 + 2 * 3 * 4)
    assert quantized_stacked_weights_nbytes(weights) == expected
