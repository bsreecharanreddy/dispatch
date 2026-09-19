"""vLLM's fused-MoE as a race contestant. Calls `fused_experts` directly with
the race's fixed topk_ids/topk_weights (no gating, so routing is identical to
every other contestant's). vLLM is imported lazily, on first use, so this
module -- and its tests, which inject fakes -- import on a machine without it.

API checked live 2026-09-19 against vllm-project/vllm main and the 0.29.0
release: `fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...,
quant_config=)`, with weight-only int8 built by `int8_w8a16_moe_quant_config`
(per-output-channel scales: w1_scale (E, 2N), w2_scale (E, K)).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from dispatch.benchmark.engines.base import BoundLayer, LayerFactory, RaceCase, fuse_gate_up
from dispatch.kernels.moe_forward import StackedExpertWeights
from dispatch.kernels.quantization import QuantizedStackedExpertWeights


class VllmEngine:
    name = "vllm"

    def __init__(
        self,
        *,
        fused_experts: Callable[..., torch.Tensor] | None = None,
        int8_quant_config: Callable[..., Any] | None = None,
    ) -> None:
        self._fused_experts = fused_experts
        self._int8_quant_config = int8_quant_config

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory:
        return self._factory(fuse_gate_up(weights.gate, weights.up), weights.down, None)

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory:
        quant_config = self._resolve_int8_quant_config()(
            w1_scale=fuse_gate_up(qweights.gate.scale, qweights.up.scale),
            w2_scale=qweights.down.scale,
        )
        w1 = fuse_gate_up(qweights.gate.data, qweights.up.data)
        return self._factory(w1, qweights.down.data, quant_config)

    def _factory(self, w1: torch.Tensor, w2: torch.Tensor, quant_config: Any) -> LayerFactory:
        fused_experts = self._resolve_fused_experts()

        def bind(case: RaceCase) -> BoundLayer:
            return functools.partial(
                fused_experts,
                case.x,
                w1,
                w2,
                case.topk_weight,
                case.topk_idx.to(torch.int32),
                quant_config=quant_config,
            )

        return bind

    def _resolve_fused_experts(self) -> Callable[..., torch.Tensor]:
        if self._fused_experts is None:
            from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: PLC0415
                fused_experts,
            )

            self._fused_experts = fused_experts
        return self._fused_experts

    def _resolve_int8_quant_config(self) -> Callable[..., Any]:
        if self._int8_quant_config is None:
            from vllm.model_executor.layers.fused_moe.config import (  # noqa: PLC0415
                int8_w8a16_moe_quant_config,
            )

            self._int8_quant_config = int8_w8a16_moe_quant_config
        return self._int8_quant_config
