"""GPU-only correctness gate for the int8 kernel: it must match
torch_grouped_matmul_dequant fed the *same* already-quantized weights --
the kernel-level bar in docs/design/2026-09-16-phase-5a-quantization.md
section 5. Excluded from CI by the `gpu` marker."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import assert_matches_reference, stack_expert_weights
from dispatch.kernels.quantization import (
    QuantizedTensor,
    grouped_moe_routed_quantized,
    quantize_per_channel_int8,
    quantize_stacked_weights,
    torch_grouped_matmul_dequant,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

pytest.importorskip("triton", reason="triton ships Linux wheels only")

from dispatch.kernels import grouped_gemm_int8

pytestmark = pytest.mark.gpu

SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="bf16 tensor-core matmul needs compute capability 8.0+ (Ampere or newer)",
)
DTYPES = [torch.float16, pytest.param(torch.bfloat16, marks=SM80)]

TOY_DIMS = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)
REAL_DIMS = MoEConfig(
    hidden_size=2048,
    moe_intermediate_size=1408,
    n_routed_experts=64,
    n_shared_experts=2,
    num_experts_per_tok=6,
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("group_sizes", "n", "k"),
    [
        ([5, 0, 3, 40], 24, 16),  # toy dims, an empty expert, a multi-tile expert
        ([1, 1, 1, 1, 1, 1], 1408, 2048),  # decode-shaped, gate/up_proj dims
        ([70, 0, 33, 129], 2048, 1408),  # prefill-shaped, down_proj dims, partial tiles
    ],
)
def test_int8_kernel_matches_dequant_reference(
    dtype: torch.dtype, group_sizes: list[int], n: int, k: int
) -> None:
    torch.manual_seed(0)
    sizes = torch.tensor(group_sizes, device="cuda")
    schedule = build_tile_schedule(sizes, block_m=16)
    x = torch.randn(int(sizes.sum()), k, device="cuda", dtype=dtype)
    weight = torch.randn(len(group_sizes), n, k, device="cuda", dtype=torch.float32) / k**0.5
    qweight = quantize_per_channel_int8(weight)

    actual = grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)

    assert_matches_reference(actual, torch_grouped_matmul_dequant(x.float(), qweight, schedule))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("config", "num_tokens"), [(TOY_DIMS, 37), (REAL_DIMS, 1), (REAL_DIMS, 256)]
)
def test_int8_kernel_backed_moe_layer_matches_dequant_reference(
    dtype: torch.dtype, config: MoEConfig, num_tokens: int
) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(config).to(device="cuda", dtype=dtype)
    hidden_states = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=dtype)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    qweights = quantize_stacked_weights(weights)

    expected = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, qweights, torch_grouped_matmul_dequant
    )
    actual = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, qweights, grouped_gemm_int8.grouped_matmul_int8
    )

    assert_matches_reference(actual, expected)


def test_int8_kernel_rejects_a_schedule_below_tl_dots_minimum_block_m() -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 16, 16, device="cuda", dtype=torch.float32)
    qweight = quantize_per_channel_int8(weight)
    schedule = build_tile_schedule(torch.tensor([4], device="cuda"), block_m=4)

    with pytest.raises(ValueError, match="block_m"):
        grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)


def test_int8_kernel_rejects_non_cuda_tensors() -> None:
    x = torch.randn(4, 16, dtype=torch.float16)
    weight = torch.randn(1, 16, 16, dtype=torch.float32)
    qweight = quantize_per_channel_int8(weight)
    schedule = build_tile_schedule(torch.tensor([4]), block_m=16)

    with pytest.raises(ValueError, match="CUDA"):
        grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)


def test_int8_kernel_rejects_a_scale_shape_that_does_not_match_weight_e_n() -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 16, 16, device="cuda", dtype=torch.float32)
    qweight = quantize_per_channel_int8(weight)
    mismatched = QuantizedTensor(data=qweight.data, scale=qweight.scale.unsqueeze(-1))
    schedule = build_tile_schedule(torch.tensor([4], device="cuda"), block_m=16)

    with pytest.raises(ValueError, match="scale shape"):
        grouped_gemm_int8.grouped_matmul_int8(x, mismatched, schedule)
