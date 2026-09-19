from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.benchmark.engines.base import (
    dequantized_weights,
    fuse_gate_up,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def test_make_weights_is_seeded_and_shaped_from_the_dims() -> None:
    first = make_weights(torch.float32, seed=3, device="cpu", **DIMS)
    again = make_weights(torch.float32, seed=3, device="cpu", **DIMS)
    other = make_weights(torch.float32, seed=4, device="cpu", **DIMS)

    assert first.gate.shape == (6, 8, 16)
    assert first.up.shape == (6, 8, 16)
    assert first.down.shape == (6, 16, 8)
    assert torch.equal(first.gate, again.gate)
    assert not torch.equal(first.gate, other.gate)


def test_uniform_and_zipf_cases_share_inputs_and_differ_only_in_routing() -> None:
    uniform = make_case(64, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    zipf = make_case(64, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    assert torch.equal(uniform.x, zipf.x)
    assert torch.equal(uniform.topk_weight, zipf.topk_weight)
    assert not torch.equal(uniform.topk_idx, zipf.topk_idx)
    assert uniform.topk_weight.dtype == torch.float32
    assert uniform.topk_idx.dtype == torch.int64


def test_zipf_routing_concentrates_load_on_low_experts() -> None:
    uniform = make_case(512, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    zipf = make_case(512, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    uniform_load = torch.bincount(uniform.topk_idx.reshape(-1), minlength=6)
    zipf_load = torch.bincount(zipf.topk_idx.reshape(-1), minlength=6)
    assert zipf_load[0] > zipf_load[-1] * 2
    assert uniform_load.max() < uniform_load.min() * 2


def test_fuse_gate_up_puts_gate_first_and_is_the_layout_silu_and_mul_reads() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    fused = fuse_gate_up(weights.gate, weights.up)
    assert fused.shape == (6, 16, 16)
    assert torch.equal(fused[:, :8], weights.gate)
    assert torch.equal(fused[:, 8:], weights.up)

    x = torch.randn(3, 16)
    expert = 2
    hidden = x @ fused[expert].T
    fused_activation = F.silu(hidden[:, :8]) * hidden[:, 8:]
    separate_activation = F.silu(x @ weights.gate[expert].T) * (x @ weights.up[expert].T)
    torch.testing.assert_close(fused_activation, separate_activation)


def test_fuse_gate_up_also_fuses_per_channel_scales() -> None:
    gate_scale, up_scale = torch.ones(6, 8), torch.full((6, 8), 2.0)
    fused = fuse_gate_up(gate_scale, up_scale)
    assert fused.shape == (6, 16)
    assert torch.equal(fused[:, :8], gate_scale)
    assert torch.equal(fused[:, 8:], up_scale)


def test_reference_output_is_float32_whatever_the_case_dtype() -> None:
    weights = make_weights(torch.bfloat16, seed=0, device="cpu", **DIMS)
    case = make_case(5, "uniform", dtype=torch.bfloat16, seed=0, device="cpu", **CASE_DIMS)

    assert reference_output(case, weights).dtype == torch.float32


def test_int8_reference_uses_the_dequantized_weights_not_the_originals() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(5, "uniform", dtype=torch.float32, seed=0, device="cpu", **CASE_DIMS)

    int8_reference = reference_output(case, dequantized_weights(qweights))
    original_reference = reference_output(case, weights)

    assert not torch.equal(int8_reference, original_reference)
    assert_matches_reference(int8_reference, original_reference)  # close, not identical
    with pytest.raises(AssertionError):
        torch.testing.assert_close(int8_reference, original_reference, rtol=0, atol=0)
