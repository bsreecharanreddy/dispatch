"""Swaps a GroupedMatmul backend into a loaded DeepSeekMoE model by replacing
each MoE layer's `moe_infer` -- the inference-time routed-expert method of
DeepSeek's remote-code DeepseekMoE (modeling_deepseek.py). The gate,
attention, and shared experts stay DeepSeek's own code.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from dispatch.kernels.moe_forward import (
    GroupedMatmul,
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
)

MoEInfer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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
    for module in model.modules():
        if not hasattr(module, "moe_infer"):
            continue
        experts = module.experts
        if not isinstance(experts, torch.nn.ModuleList):
            raise TypeError(
                f"expected {type(module).__name__}.experts to be nn.ModuleList, "
                f"got {type(experts).__name__}"
            )
        # An instance attribute shadowing the remote-code method; nn.Module's
        # __setattr__ stub only admits Tensor | Module values.
        module.moe_infer = _grouped_moe_infer(  # type: ignore[assignment]
            stack_expert_weights(experts), matmul, block_m
        )
        patched += 1
    return patched


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
