"""Swaps a GroupedMatmul backend into a loaded DeepSeekMoE model by replacing
each MoE layer's `moe_infer` -- the inference-time routed-expert method of
DeepSeek's remote-code DeepseekMoE (modeling_deepseek.py). The gate,
attention, and shared experts stay DeepSeek's own code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

from dispatch.kernels.moe_forward import (
    GroupedMatmul,
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
)
from dispatch.kernels.quantization import (
    QuantizedGroupedMatmul,
    QuantizedStackedExpertWeights,
    grouped_moe_routed_quantized,
    quantize_stacked_weights,
)

MoEInfer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _iter_validated_moe_layers(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.nn.Module, torch.nn.ModuleList]]:
    for module in model.modules():
        if not hasattr(module, "moe_infer"):
            continue
        experts = module.experts
        if not isinstance(experts, torch.nn.ModuleList):
            raise TypeError(
                f"expected {type(module).__name__}.experts to be nn.ModuleList, "
                f"got {type(experts).__name__}"
            )
        yield module, experts


def patch_moe_infer(model: torch.nn.Module, matmul: GroupedMatmul, *, block_m: int = 16) -> int:
    """Returns how many MoE layers were patched, for the caller to check.

    Forces the model into eval mode first: DeepSeek's own DeepseekMoE.forward
    only calls moe_infer on the not-self.training branch, so a model left in
    training mode would report a nonzero patched count while never actually
    executing the patched path -- silently comparing the stock model against
    itself rather than against the kernel.
    """
    model.eval()
    patched = 0
    for module, experts in _iter_validated_moe_layers(model):
        # An instance attribute shadowing the remote-code method; nn.Module's
        # __setattr__ stub only admits Tensor | Module values.
        module.moe_infer = _grouped_moe_infer(  # type: ignore[assignment]
            stack_expert_weights(experts), matmul, block_m
        )
        patched += 1
    return patched


def patch_moe_infer_quantized(
    model: torch.nn.Module, matmul: QuantizedGroupedMatmul, *, block_m: int = 16
) -> int:
    """Same contract as patch_moe_infer, but quantizes each layer's stacked
    weights to int8 (docs/design/2026-09-16-phase-5a-quantization.md) once
    at patch time, before building the per-layer closure -- not per
    forward pass."""
    model.eval()
    patched = 0
    for module, experts in _iter_validated_moe_layers(model):
        quantized_weights = quantize_stacked_weights(stack_expert_weights(experts))
        _free_expert_weights(experts)
        module.moe_infer = _grouped_moe_infer_quantized(  # type: ignore[assignment]
            quantized_weights, matmul, block_m
        )
        patched += 1
    return patched


def _free_expert_weights(experts: torch.nn.ModuleList) -> None:
    """Releases each expert's original bf16 weight now that moe_infer's
    quantized closure never reads `experts` again. stack_expert_weights
    re-points every expert's Linear at a *view* into one shared bf16
    tensor rather than copying it (so building that stack costs no extra
    memory) -- but that means those views keep the whole bf16 tensor
    resident for the model's lifetime unless something drops them. Left
    alone, a quantized model holds its bf16 weights AND their int8 copies
    at once, defeating the point of quantizing (and OOMing on real
    hardware at DeepSeekMoE-16B's scale -- found running Phase 5a's
    measured session)."""
    for expert in experts:
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(expert, name)
            linear.weight = torch.nn.Parameter(linear.weight.new_empty(0), requires_grad=False)


def _grouped_moe_infer(
    weights: StackedExpertWeights, matmul: GroupedMatmul, block_m: int
) -> MoEInfer:
    @torch.no_grad()
    def moe_infer(
        x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        top_k = flat_expert_indices.numel() // x.shape[0]
        return grouped_moe_routed(
            x,
            flat_expert_indices.view(-1, top_k),
            flat_expert_weights.view(-1, top_k),
            weights,
            matmul,
            block_m=block_m,
        )

    return moe_infer


def _grouped_moe_infer_quantized(
    weights: QuantizedStackedExpertWeights, matmul: QuantizedGroupedMatmul, block_m: int
) -> MoEInfer:
    @torch.no_grad()
    def moe_infer(
        x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        top_k = flat_expert_indices.numel() // x.shape[0]
        return grouped_moe_routed_quantized(
            x,
            flat_expert_indices.view(-1, top_k),
            flat_expert_weights.view(-1, top_k),
            weights,
            matmul,
            block_m=block_m,
        )

    return moe_infer
