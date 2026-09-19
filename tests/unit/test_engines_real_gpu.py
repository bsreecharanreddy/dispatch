"""Phase 6's real-engine correctness gate for the kernel race: every
contestant -- dispatch's Triton kernels, vLLM's fused_experts, SGLang's
fused_experts -- must meet assert_matches_reference against the fp32
reference on DeepSeek-MoE-16B's real routed-expert dims, before any latency
number means anything. The CPU tests (test_engine_adapters.py) prove the
adapters against eager fakes; this proves them against the real engines and
the real kernels.

Skipped, with the reason shown, where CUDA, triton, vllm or sglang is
missing. Excluded from CI by the `gpu` marker; runs in the Phase 6 pod
session's engines venv.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.engines.base import (
    MoEEngine,
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.registry import build_engines
from dispatch.kernels.moe_forward import StackedExpertWeights, assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

pytestmark = pytest.mark.gpu
pytest.importorskip("triton", reason="triton ships Linux wheels only")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

ENGINES = ["dispatch-naive", "dispatch-persistent", "vllm", "sglang"]
TOKEN_COUNTS = [1, 64, 512]  # decode-sized through a prefill-sized batch


@pytest.fixture(scope="module")
def weights() -> StackedExpertWeights:
    return make_weights(torch.bfloat16, seed=0, device="cuda")


def _engine(name: str) -> MoEEngine:
    if name == "vllm":
        pytest.importorskip("vllm", reason="vllm is installed only in the Phase 6 engines venv")
    if name == "sglang":
        pytest.importorskip("sglang", reason="sglang is installed only in the Phase 6 engines venv")
        from dispatch.benchmark.engines.sglang_moe import init_distributed  # noqa: PLC0415

        init_distributed()
    return build_engines(name, [16])[0]


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("name", ENGINES)
def test_bf16_engine_matches_the_fp32_reference(
    name: str, tokens: int, distribution: str, weights: StackedExpertWeights
) -> None:
    case = make_case(tokens, distribution, dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_bf16(weights)(case)()

    assert_matches_reference(output, reference_output(case, weights))


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("name", ["dispatch-naive", "vllm", "sglang"])
def test_int8_engine_matches_the_dequantized_reference(
    name: str, tokens: int, distribution: str, weights: StackedExpertWeights
) -> None:
    qweights = quantize_stacked_weights(weights)
    case = make_case(tokens, distribution, dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_int8(qweights)(case)()

    assert_matches_reference(output, reference_output(case, dequantized_weights(qweights)))


@pytest.mark.parametrize("name", ENGINES)
def test_a_mutated_weight_layout_turns_the_gate_red(
    name: str, weights: StackedExpertWeights
) -> None:
    """The suite's own proof that it can fail: hand each engine gate and up
    swapped and require the reference check to reject the output."""
    swapped = StackedExpertWeights(gate=weights.up, up=weights.gate, down=weights.down)
    case = make_case(64, "uniform", dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_bf16(swapped)(case)()

    with pytest.raises(AssertionError):
        assert_matches_reference(output, reference_output(case, weights))


@pytest.mark.parametrize("name", ENGINES)
def test_bound_layers_leave_the_input_untouched_and_repeat_within_rounding(
    name: str, weights: StackedExpertWeights
) -> None:
    """SGLang's inplace default would overwrite x, and timing calls the layer
    hundreds of times, so any input mutation would corrupt every later call.

    Repeatability is checked within the reference tolerance, not bitwise:
    dispatch's combine step sums each token's top-k rows with index_add_,
    which is atomicAdd on CUDA, so two identical calls differ by bf16 rounding
    (measured on the first real run: one ulp, 0.03125, on 24% of elements).
    That is a property of dispatch's kernel path, not a bug in the check."""
    case = make_case(64, "zipf", dtype=torch.bfloat16, seed=1, device="cuda")
    x_before = case.x.clone()
    layer = _engine(name).prepare_bf16(weights)(case)

    first = layer().clone()
    second = layer()

    assert_matches_reference(second, first)
    assert torch.equal(case.x, x_before)
