from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.engines.base import (
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.dispatch_kernels import DispatchEngine
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def test_name_carries_backend_and_tile_size() -> None:
    assert DispatchEngine("naive", block_m=32).name == "dispatch-naive-bm32"


def test_bf16_layer_matches_the_fp32_reference() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = DispatchEngine("torch").prepare_bf16(weights)(case)

    assert_matches_reference(layer(), reference_output(case, weights))


def test_bound_layer_is_repeatable_so_it_can_be_timed() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    layer = DispatchEngine("torch").prepare_bf16(weights)(case)

    assert torch.equal(layer(), layer())


def test_int8_layer_matches_the_dequantized_reference() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = DispatchEngine("torch").prepare_int8(qweights)(case)

    assert_matches_reference(layer(), reference_output(case, dequantized_weights(qweights)))


def test_persistent_backend_has_no_int8_kernel() -> None:
    qweights = quantize_stacked_weights(make_weights(torch.float32, seed=0, device="cpu", **DIMS))

    with pytest.raises(ValueError, match="no int8 kernel"):
        DispatchEngine("persistent").prepare_int8(qweights)


def test_a_layer_that_disagrees_with_the_reference_is_detectable() -> None:
    """The race driver's refuse-not-time gate depends on assert_matches_reference
    rejecting a wrong layer; prove it does for this adapter's output shape."""
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    wrong = DispatchEngine("torch").prepare_bf16(weights)(case)() * 1.5

    with pytest.raises(AssertionError):
        assert_matches_reference(wrong, reference_output(case, weights))
