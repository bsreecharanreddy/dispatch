"""GPU-only correctness gate for the Triton kernels: each must meet
assert_matches_reference against torch_grouped_matmul (itself proven
against ReferenceMoE on CPU in test_moe_forward.py) before any benchmark
number means anything.

Skipped, with the reason shown, where triton can't be imported (it ships
Linux wheels only); where triton imports, a missing or broken grouped_gemm
module errors loudly instead of skipping. Excluded from CI by the `gpu`
marker.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import (
    assert_matches_reference,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

pytest.importorskip("triton", reason="triton ships Linux wheels only")

from dispatch.kernels import grouped_gemm

pytestmark = pytest.mark.gpu

SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="bf16 tensor-core matmul needs compute capability 8.0+ (Ampere or newer)",
)
DTYPES = [torch.float16, pytest.param(torch.bfloat16, marks=SM80)]
KERNELS = ["grouped_matmul"]

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


@pytest.mark.parametrize("kernel_name", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("group_sizes", "n", "k"),
    [
        ([5, 0, 3, 40], 24, 16),  # toy dims, an empty expert, a multi-tile expert
        ([1, 1, 1, 1, 1, 1], 1408, 2048),  # decode-shaped, gate/up_proj dims
        ([70, 0, 33, 129], 2048, 1408),  # prefill-shaped, down_proj dims, partial tiles
    ],
)
def test_kernel_matches_fp32_reference(
    kernel_name: str, dtype: torch.dtype, group_sizes: list[int], n: int, k: int
) -> None:
    torch.manual_seed(0)
    sizes = torch.tensor(group_sizes, device="cuda")
    schedule = build_tile_schedule(sizes, block_m=16)
    x = torch.randn(int(sizes.sum()), k, device="cuda", dtype=dtype)
    weight = torch.randn(len(group_sizes), n, k, device="cuda", dtype=dtype) / k**0.5

    actual = getattr(grouped_gemm, kernel_name)(x, weight, schedule)

    assert_matches_reference(actual, torch_grouped_matmul(x.float(), weight.float(), schedule))


@pytest.mark.parametrize("kernel_name", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("config", "num_tokens"), [(TOY_DIMS, 37), (REAL_DIMS, 1), (REAL_DIMS, 256)]
)
def test_kernel_backed_moe_layer_matches_reference_moe(
    kernel_name: str, dtype: torch.dtype, config: MoEConfig, num_tokens: int
) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(config).to(device="cuda", dtype=dtype)
    hidden_states = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=dtype)
    expected = moe.routed(hidden_states)

    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    actual = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, getattr(grouped_gemm, kernel_name)
    )

    assert_matches_reference(actual, expected)


def test_rejects_a_schedule_below_tl_dots_minimum_block_m() -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 16, 16, device="cuda", dtype=torch.float16)
    schedule = build_tile_schedule(torch.tensor([4], device="cuda"), block_m=4)

    with pytest.raises(ValueError, match="block_m"):
        grouped_gemm.grouped_matmul(x, weight, schedule)
