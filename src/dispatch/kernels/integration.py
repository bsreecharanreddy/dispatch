"""Swaps a GroupedMatmul backend into a loaded DeepSeekMoE model by replacing
each MoE layer's `moe_infer` -- the inference-time routed-expert method of
DeepSeek's remote-code DeepseekMoE (modeling_deepseek.py). The gate,
attention, and shared experts stay DeepSeek's own code.

Also carries `fix_rope_inv_freq`, an unrelated but similarly-shaped
after-load repair: transformers>=5.17.0's model-loading path leaves
DeepSeek's remote code's rotary-embedding `inv_freq` buffer as
uninitialized memory rather than the value its own __init__ computes (see
that function's docstring for the full mechanism and how it was found).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

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
        # An instance attribute shadowing the remote-code method; nn.Module's
        # __setattr__ stub only admits Tensor | Module values.
        module.moe_infer = _grouped_moe_infer_quantized(  # type: ignore[assignment]
            quantized_weights, matmul, block_m
        )
        patched += 1
    return patched


def fix_rope_inv_freq(model: torch.nn.Module) -> int:
    """Returns how many rotary-embedding buffers were fixed, for the
    caller to check.

    transformers==5.17.0's `from_pretrained` leaves DeepSeek's remote
    code's `inv_freq` buffer -- computed fresh in
    `DeepseekRotaryEmbedding.__init__` from `1/base**(i/dim)`, and marked
    `persistent=False` because it is never meant to be part of a
    checkpoint's state_dict -- as uninitialized memory instead of that
    computed value. Found live on a real GPU session (Phase 5b): the
    corruption is already present immediately after `from_pretrained`
    returns, before any `.to(device)` call, so it is not a GPU/CUDA
    numerics issue despite only manifesting as NaN once a forward pass
    actually runs on GPU (a CPU forward pass reads the same garbage
    `inv_freq` and still applies rotary embeddings with it, but generates
    real, non-NaN, wrong-content output instead -- something about the
    GPU path additionally propagates it to NaN). Every attention layer's
    rotary embedding gets its own `inv_freq`, so this must run once per
    loaded model, after `load_model` returns, before any forward pass.

    Discovered and matched via duck typing (`inv_freq`/`dim`/`base`
    attributes) rather than a hardcoded module path, matching this file's
    `moe_infer` discovery -- the same fix applies to any transformers
    remote-code model built on the same Llama-derived rotary embedding
    class (DeepSeek's is a direct, `# Copied from` derivative)."""
    fixed = 0
    for module in model.modules():
        if not (hasattr(module, "inv_freq") and hasattr(module, "dim") and hasattr(module, "base")):
            continue
        dim = cast(int, module.dim)
        base = cast(int, module.base)
        inv_freq = cast(torch.Tensor, module.inv_freq)
        correct_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        module.inv_freq = correct_inv_freq.to(inv_freq.device)
        if hasattr(module, "max_seq_len_cached"):
            # Forces the next forward call to rebuild cos_cached/sin_cached
            # from the now-correct inv_freq -- they were derived from the
            # broken buffer and are just as wrong. nn.Module's __setattr__
            # stub types this plain int attribute as Tensor | Module.
            module.max_seq_len_cached = None  # type: ignore[assignment]
        fixed += 1
    return fixed


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
