"""vLLM's and SGLang's adapters, driven through fakes that reproduce each
engine's documented fused-layout contract (gate rows first in w1, per-channel
int8 scales). The fakes are eager PyTorch, so a wrong layout, a wrong scale
shape, or a mutated input in the adapter changes the output and fails here;
the real engines are checked by the `gpu`-marked test on the pod."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.benchmark.engines import registry
from dispatch.benchmark.engines.base import (
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.sglang_moe import SglangEngine
from dispatch.benchmark.engines.vllm_moe import VllmEngine
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def _eager_fused_moe(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """What vLLM's and SGLang's fused-MoE compute, in eager float32: w1 holds
    gate rows then up rows, activation is silu(gate) * up, int8 weights are
    scaled per output channel."""
    w1_f, w2_f = w1.float(), w2.float()
    if w1_scale is not None and w2_scale is not None:
        w1_f, w2_f = w1_f * w1_scale.unsqueeze(-1), w2_f * w2_scale.unsqueeze(-1)
    n = w2.shape[2]
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            hidden = x[token].float() @ w1_f[expert].T
            activated = F.silu(hidden[:n]) * hidden[n:]
            out[token] += topk_weights[token, slot].float() * (activated @ w2_f[expert].T)
    return out


def _fake_vllm_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_config: Any = None,
) -> torch.Tensor:
    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    scales = (
        (None, None) if quant_config is None else (quant_config.w1_scale, quant_config.w2_scale)
    )
    return _eager_fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, *scales)


def _fake_vllm_int8_quant_config(*, w1_scale: torch.Tensor, w2_scale: torch.Tensor) -> Any:
    return SimpleNamespace(w1_scale=w1_scale, w2_scale=w2_scale)


class _FakeTopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


class _FakeRunnerConfig:
    inplace: bool

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _fake_sglang_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: _FakeTopKOutput,
    moe_runner_config: _FakeRunnerConfig,
    use_int8_w8a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert moe_runner_config.inplace is False  # the default would overwrite case.x
    assert topk_output.topk_ids.dtype == torch.int32
    if use_int8_w8a16:
        assert per_channel_quant
    return _eager_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_output.topk_weights,
        topk_output.topk_ids,
        w1_scale,
        w2_scale,
    )


def _vllm() -> VllmEngine:
    return VllmEngine(
        fused_experts=_fake_vllm_fused_experts, int8_quant_config=_fake_vllm_int8_quant_config
    )


def _sglang() -> SglangEngine:
    return SglangEngine(
        fused_experts=_fake_sglang_fused_experts,
        topk_output_cls=_FakeTopKOutput,
        runner_config_cls=_FakeRunnerConfig,
    )


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_bf16_layout_conversion_reproduces_the_reference(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_bf16(weights)(case)

    assert_matches_reference(layer(), reference_output(case, weights))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_int8_scale_layout_reproduces_the_dequantized_reference(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_int8(qweights)(case)

    assert_matches_reference(layer(), reference_output(case, dequantized_weights(qweights)))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_a_swapped_gate_and_up_layout_is_caught(make_engine: Any) -> None:
    """The failure this whole adapter layer exists to prevent: fusing up
    before gate silently computes silu(up) * gate. The reference check must
    reject it."""
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    swapped = type(weights)(gate=weights.up, up=weights.gate, down=weights.down)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_bf16(swapped)(case)

    with pytest.raises(AssertionError):
        assert_matches_reference(layer(), reference_output(case, weights))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_bound_layer_does_not_mutate_the_case_and_is_repeatable(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    x_before = case.x.clone()
    layer = make_engine().prepare_bf16(weights)(case)

    assert torch.equal(layer(), layer())
    assert torch.equal(case.x, x_before)


def test_registry_builds_a_tile_size_sweep_for_dispatch_and_one_engine_otherwise() -> None:
    naive = registry.build_engines("dispatch-naive", (16, 64))
    assert [engine.name for engine in naive] == ["dispatch-naive-bm16", "dispatch-naive-bm64"]
    assert len(registry.build_engines("vllm")) == 1
    assert len(registry.build_engines("sglang")) == 1


def test_registry_rejects_unknown_engines() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        registry.build_engines("tensorrt")
    with pytest.raises(ValueError, match="unknown engine"):
        registry.build_engines("dispatch-cuda")
