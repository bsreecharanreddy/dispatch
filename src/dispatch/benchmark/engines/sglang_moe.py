"""SGLang's fused-MoE as a race contestant, called through `fused_experts`
with the race's fixed topk_ids/topk_weights. SGLang is imported lazily.

API checked live 2026-09-19 against sgl-project/sglang main and the 0.5.20
release, whose fused-MoE code was recently reorganized: `fused_experts` lives
in `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe` and takes a
`StandardTopKOutput` plus a `MoeRunnerConfig`. The pinned version is recorded
in every output JSON; if the pinned version's API differs, the import fails
loudly rather than falling back to another code path.

`inplace=False` is essential: the default (`True`) writes the output into
`hidden_states`, which would corrupt `case.x` across repeated timed calls.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from dispatch.benchmark.engines.base import (
    BoundLayer,
    LayerFactory,
    RaceCase,
    fuse_gate_up,
)
from dispatch.kernels.moe_forward import StackedExpertWeights
from dispatch.kernels.quantization import QuantizedStackedExpertWeights

_INIT_METHOD = "tcp://127.0.0.1:23456"


def init_distributed() -> None:
    """SGLang's fused-MoE reads its tensor-parallel group even on one GPU, so
    a world-size-1 group must exist first. Mirrors SGLang's own
    benchmark/kernels/fused_moe_triton/ scripts. Call once, before `bind`."""
    from sglang.srt.distributed.parallel_state import (  # noqa: PLC0415
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method=_INIT_METHOD, world_size=1, rank=0
        )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=_INIT_METHOD,
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


class SglangEngine:
    name = "sglang"

    def __init__(
        self,
        *,
        fused_experts: Callable[..., torch.Tensor] | None = None,
        topk_output_cls: Callable[..., Any] | None = None,
        runner_config_cls: Callable[..., Any] | None = None,
    ) -> None:
        self._fused_experts = fused_experts
        self._topk_output_cls = topk_output_cls
        self._runner_config_cls = runner_config_cls

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory:
        w1 = fuse_gate_up(weights.gate, weights.up)
        return self._factory(w1, weights.down, {})

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory:
        w1 = fuse_gate_up(qweights.gate.data, qweights.up.data)
        quant_kwargs = {
            "use_int8_w8a16": True,
            "per_channel_quant": True,
            "w1_scale": fuse_gate_up(qweights.gate.scale, qweights.up.scale),
            "w2_scale": qweights.down.scale,
        }
        return self._factory(w1, qweights.down.data, quant_kwargs)

    def _factory(
        self, w1: torch.Tensor, w2: torch.Tensor, quant_kwargs: dict[str, Any]
    ) -> LayerFactory:
        fused_experts, topk_output_cls, runner_config_cls = self._resolve()
        n_experts = int(w1.shape[0])

        def bind(case: RaceCase) -> BoundLayer:
            topk_output = topk_output_cls(
                topk_weights=case.topk_weight,
                topk_ids=case.topk_idx.to(torch.int32),
                router_logits=case.x.new_empty(0),  # unused: fused_experts drops it
            )
            runner_config = runner_config_cls(
                num_experts=n_experts,
                num_local_experts=n_experts,
                top_k=int(case.topk_idx.shape[1]),
                inplace=False,
            )
            return functools.partial(
                fused_experts, case.x, w1, w2, topk_output, runner_config, **quant_kwargs
            )

        return bind

    def _resolve(
        self,
    ) -> tuple[Callable[..., torch.Tensor], Callable[..., Any], Callable[..., Any]]:
        fused_experts = self._fused_experts
        topk_output_cls = self._topk_output_cls
        runner_config_cls = self._runner_config_cls
        if fused_experts is None or topk_output_cls is None or runner_config_cls is None:
            from sglang.srt.layers.moe.moe_runner.base import (  # noqa: PLC0415
                MoeRunnerConfig,
            )
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: PLC0415
                fused_experts as sglang_fused_experts,
            )
            from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: PLC0415

            fused_experts = fused_experts or sglang_fused_experts
            topk_output_cls = topk_output_cls or StandardTopKOutput
            runner_config_cls = runner_config_cls or MoeRunnerConfig
        return fused_experts, topk_output_cls, runner_config_cls
