"""CPU-only: patches a stand-in whose moe_infer is transcribed from
deepseek-ai/deepseek-moe-16b-base's own modeling_deepseek.py (numpy's
cumsum swapped for torch's), so the swap is proven against DeepSeek's
actual inference code, not against a paraphrase of it."""

from __future__ import annotations

import copy
from typing import cast

import pytest
import torch

from dispatch.kernels.integration import patch_moe_infer, patch_moe_infer_quantized
from dispatch.kernels.moe_forward import torch_grouped_matmul
from dispatch.kernels.quantization import (
    dequantize_int8,
    quantize_per_channel_int8,
    torch_grouped_matmul_dequant,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


class FakeDeepseekMoE(ReferenceMoE):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_weight = self.route(hidden_states)
        y = self.moe_infer(hidden_states, topk_idx.view(-1), topk_weight.view(-1, 1))
        if self.shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y

    @torch.no_grad()
    def moe_infer(
        self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0).tolist()
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_out = self.experts[i](x[exp_token_idx])
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum"
            )
        return expert_cache


class FakeModel(torch.nn.Module):
    def __init__(self, num_moe_layers: int) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(TOY_CONFIG.hidden_size, TOY_CONFIG.hidden_size)
        self.moe_layers = torch.nn.ModuleList(
            FakeDeepseekMoE(TOY_CONFIG) for _ in range(num_moe_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        for layer in self.moe_layers:
            x = x + layer(x)
        return x


def test_patch_counts_only_moe_layers() -> None:
    torch.manual_seed(0)

    assert patch_moe_infer(FakeModel(num_moe_layers=3), torch_grouped_matmul) == 3


def test_patched_model_matches_deepseeks_own_moe_infer() -> None:
    torch.manual_seed(0)
    model = FakeModel(num_moe_layers=3)
    hidden_states = torch.randn(9, TOY_CONFIG.hidden_size)
    with torch.no_grad():
        expected = model(hidden_states)

    patch_moe_infer(model, torch_grouped_matmul)
    with torch.no_grad():
        actual = model(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_patch_rejects_experts_that_are_not_a_module_list() -> None:
    class Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.Linear(2, 2)

        def moe_infer(self) -> None:
            raise NotImplementedError

    with pytest.raises(TypeError, match="ModuleList"):
        patch_moe_infer(Odd(), torch_grouped_matmul)


def test_patch_quantized_counts_only_moe_layers() -> None:
    torch.manual_seed(0)

    assert patch_moe_infer_quantized(FakeModel(num_moe_layers=3), torch_grouped_matmul_dequant) == 3


def test_patched_quantized_model_matches_weights_quantized_in_place() -> None:
    """Proves patch_moe_infer_quantized's full wiring (quantize at patch
    time, route through grouped_moe_routed_quantized) is mathematically
    equivalent to independently replacing every expert Linear's weight
    with its own quantize-then-dequantize round trip and running
    DeepSeek's stock moe_infer -- an independent computation path, not a
    call to any of the same helpers."""
    torch.manual_seed(0)
    model = FakeModel(num_moe_layers=3)
    hidden_states = torch.randn(9, TOY_CONFIG.hidden_size)

    expected_model = copy.deepcopy(model)
    for layer in expected_model.moe_layers:
        for expert in cast(torch.nn.ModuleList, layer.experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(expert, name)
                requantized = dequantize_int8(
                    quantize_per_channel_int8(linear.weight.detach().unsqueeze(0))
                ).squeeze(0)
                linear.weight = torch.nn.Parameter(requantized, requires_grad=False)
    with torch.no_grad():
        expected = expected_model(hidden_states)

    patch_moe_infer_quantized(model, torch_grouped_matmul_dequant)
    with torch.no_grad():
        actual = model(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_patch_quantized_frees_the_original_bf16_expert_weights() -> None:
    """A quantized model must not keep both the bf16 originals and their
    int8 copies resident -- that defeats quantizing at all, and OOMed a
    real rented-GPU run at DeepSeekMoE-16B's scale (Phase 5a)."""
    torch.manual_seed(0)
    model = FakeModel(num_moe_layers=2)

    patch_moe_infer_quantized(model, torch_grouped_matmul_dequant)

    for layer in model.moe_layers:
        for expert in cast(torch.nn.ModuleList, layer.experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                assert getattr(expert, name).weight.numel() == 0


def test_patch_quantized_rejects_experts_that_are_not_a_module_list() -> None:
    class Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.Linear(2, 2)

        def moe_infer(self) -> None:
            raise NotImplementedError

    with pytest.raises(TypeError, match="ModuleList"):
        patch_moe_infer_quantized(Odd(), torch_grouped_matmul_dequant)
