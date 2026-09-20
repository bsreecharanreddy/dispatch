"""dispatch's own kernels as a race contestant: the naive or persistent
Triton grouped-GEMM (bf16), or the int8 weight-only kernel, behind the
common engine contract. `torch` is the eager backend, which runs on CPU and
is what the unit tests use."""

from __future__ import annotations

import functools

import torch

from dispatch.benchmark.engines.base import DEFAULT_BLOCK_M, LayerFactory, RaceCase
from dispatch.kernels.backends import QUANTIZED_BACKENDS, resolve_backend, resolve_quantized_backend
from dispatch.kernels.moe_forward import StackedExpertWeights, grouped_moe_routed
from dispatch.kernels.quantization import (
    QuantizedGroupedMatmul,
    QuantizedStackedExpertWeights,
    grouped_moe_routed_quantized,
    torch_grouped_matmul_dequant,
)


class DispatchEngine:
    def __init__(self, backend: str, *, block_m: int = DEFAULT_BLOCK_M) -> None:
        self.name = f"dispatch-{backend}-bm{block_m}"
        self._backend = backend
        self._block_m = block_m

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory:
        matmul = resolve_backend(self._backend)
        block_m = self._block_m

        def bind(case: RaceCase) -> functools.partial[torch.Tensor]:
            return functools.partial(
                grouped_moe_routed,
                case.x,
                case.topk_idx,
                case.topk_weight.to(case.x.dtype),
                weights,
                matmul,
                block_m=block_m,
            )

        return bind

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory:
        matmul = self._quantized_matmul()
        block_m = self._block_m

        def bind(case: RaceCase) -> functools.partial[torch.Tensor]:
            return functools.partial(
                grouped_moe_routed_quantized,
                case.x,
                case.topk_idx,
                case.topk_weight.to(case.x.dtype),
                qweights,
                matmul,
                block_m=block_m,
            )

        return bind

    def _quantized_matmul(self) -> QuantizedGroupedMatmul:
        if self._backend not in QUANTIZED_BACKENDS:
            raise ValueError(
                f"no int8 kernel for backend {self._backend!r}; expected one of "
                f"{QUANTIZED_BACKENDS} (Phase 5a's int8 kernel is built on the naive "
                "launch order only)"
            )
        if self._backend == "torch":
            return torch_grouped_matmul_dequant
        return resolve_quantized_backend()
