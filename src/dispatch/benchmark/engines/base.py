"""The contract every Phase 6 race contestant implements, plus the seeded
inputs and fp32 reference they are all checked against.

An engine is anything that computes DeepSeek's routed-expert MoE forward from
`(x, topk_idx, topk_weight)` and a set of expert weights. `prepare_*` does
the one-time weight conversion (fusing gate+up, uploading scales), and
`bind` does the per-case input conversion (dtype casts) -- both untimed --
so the closure a benchmark times is only the engine's own routed-MoE call,
never a cast the engine's real serving path wouldn't pay.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from dispatch.kernels.bench import sample_topk_idx
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    grouped_moe_routed,
    torch_grouped_matmul,
)
from dispatch.kernels.quantization import QuantizedStackedExpertWeights, dequantize_int8

# deepseek-ai/deepseek-moe-16b-base config.json, checked live 2026-09-15
# (same constants as scripts/run_kernel_bench.py).
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 1408
N_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 6

# Phase 1's default sweep (1, 16, 128, 512, 2048) plus 4 and 64, so results
# extend Phase 1's tables and decode-sized batches are better covered.
TOKEN_COUNTS = (1, 4, 16, 64, 128, 512, 2048)


@dataclass(frozen=True)
class RaceCase:
    """One benchmark input. `topk_weight` is float32 by contract; an adapter
    casts it to whatever its engine wants inside `bind`."""

    x: torch.Tensor  # (num_tokens, hidden)
    topk_idx: torch.Tensor  # (num_tokens, top_k) int64
    topk_weight: torch.Tensor  # (num_tokens, top_k) float32


BoundLayer = Callable[[], torch.Tensor]
LayerFactory = Callable[[RaceCase], BoundLayer]


class MoEEngine(Protocol):
    name: str

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory: ...

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory: ...


def make_weights(  # noqa: PLR0913 -- the real DeepSeek dims are the defaults; tests shrink them
    dtype: torch.dtype,
    *,
    seed: int,
    device: str,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = MOE_INTERMEDIATE_SIZE,
    n_experts: int = N_ROUTED_EXPERTS,
) -> StackedExpertWeights:
    generator = torch.Generator(device=device).manual_seed(seed)

    def stack(n: int, k: int) -> torch.Tensor:
        weights = torch.randn(n_experts, n, k, device=device, dtype=dtype, generator=generator)
        return weights / math.sqrt(k)

    return StackedExpertWeights(
        gate=stack(intermediate_size, hidden_size),
        up=stack(intermediate_size, hidden_size),
        down=stack(hidden_size, intermediate_size),
    )


def make_case(  # noqa: PLR0913 -- the real DeepSeek dims are the defaults; tests shrink them
    num_tokens: int,
    distribution: str,
    *,
    dtype: torch.dtype,
    seed: int,
    device: str,
    hidden_size: int = HIDDEN_SIZE,
    n_experts: int = N_ROUTED_EXPERTS,
    top_k: int = NUM_EXPERTS_PER_TOK,
) -> RaceCase:
    """Uniform and zipf cases at the same `seed` and `num_tokens` share x and
    topk_weight exactly; only the routing differs, so the two distributions
    are a controlled comparison."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, generator=generator)
    topk_idx = sample_topk_idx(
        num_tokens,
        n_experts,
        top_k,
        distribution=distribution,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    topk_weight = torch.rand(
        num_tokens, top_k, device=device, dtype=torch.float32, generator=generator
    )
    return RaceCase(x=x, topk_idx=topk_idx, topk_weight=topk_weight)


def dequantized_weights(qweights: QuantizedStackedExpertWeights) -> StackedExpertWeights:
    """float32 weights equal to what an int8 kernel effectively multiplies by:
    the int8 race's reference is the *same* quantized weights, dequantized,
    so nothing should diverge beyond float precision."""
    return StackedExpertWeights(
        gate=dequantize_int8(qweights.gate),
        up=dequantize_int8(qweights.up),
        down=dequantize_int8(qweights.down),
    )


def reference_output(case: RaceCase, weights: StackedExpertWeights) -> torch.Tensor:
    """The fp32 eager reference every engine is checked against: the same
    per-expert-loop contract DeepseekMoE.moe_infer implements, in float32."""
    weights_fp32 = StackedExpertWeights(
        gate=weights.gate.float(), up=weights.up.float(), down=weights.down.float()
    )
    return grouped_moe_routed(
        case.x.float(), case.topk_idx, case.topk_weight.float(), weights_fp32, torch_grouped_matmul
    )


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(E, N, K) gate and up -> (E, 2N, K) with gate's rows first. Works for
    (E, N) per-channel scales too (-> (E, 2N)). This is the layout vLLM's and
    SGLang's fused-MoE kernels read: silu(first half) * second half."""
    return torch.cat([gate, up], dim=1)
