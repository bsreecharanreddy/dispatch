"""CPU-only: patches a stand-in whose moe_infer is transcribed from
deepseek-ai/deepseek-moe-16b-base's own modeling_deepseek.py (numpy's
cumsum swapped for torch's), so the swap is proven against DeepSeek's
actual inference code, not against a paraphrase of it."""

from __future__ import annotations

import copy
import gc
import weakref
from typing import cast

import pytest
import torch

from dispatch.kernels import integration as integration_module
from dispatch.kernels.integration import (
    fix_rope_inv_freq,
    patch_moe_infer,
    patch_moe_infer_quantized,
)
from dispatch.kernels.moe_forward import stack_expert_weights, torch_grouped_matmul
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


def test_patch_quantized_actually_releases_the_shared_bf16_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The previous test proves the *views* into stack_expert_weights'
    shared bf16 tensor are dropped, but a view being replaced doesn't by
    itself prove the tensor those views pointed at is actually collected
    -- something else could still be holding it. Pin that directly with
    a weakref to the tensor stack_expert_weights actually builds."""
    torch.manual_seed(0)
    model = FakeModel(num_moe_layers=1)
    refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def spying_stack_expert_weights(experts: object) -> object:
        weights = stack_expert_weights(experts)  # type: ignore[arg-type]
        refs.extend(weakref.ref(tensor) for tensor in (weights.gate, weights.up, weights.down))
        return weights

    monkeypatch.setattr(integration_module, "stack_expert_weights", spying_stack_expert_weights)

    patch_moe_infer_quantized(model, torch_grouped_matmul_dequant)
    gc.collect()

    assert refs, "spy never ran -- test is broken, not proving anything"
    assert all(ref() is None for ref in refs)


def test_patch_quantized_rejects_experts_that_are_not_a_module_list() -> None:
    class Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.Linear(2, 2)

        def moe_infer(self) -> None:
            raise NotImplementedError

    with pytest.raises(TypeError, match="ModuleList"):
        patch_moe_infer_quantized(Odd(), torch_grouped_matmul_dequant)


class _FakeRotaryEmbedding(torch.nn.Module):
    """Duck-types DeepseekRotaryEmbedding's shape (inv_freq/dim/base/
    max_seq_len_cached) AND its real forward()/_set_cos_sin_cache()
    rebuild guard (`max_seq_len_cached is None or seq_len > ...`) -- so
    a test can exercise the fix's second half (forcing forward() to
    rebuild cos/sin from the *corrected* inv_freq) on CPU, not just
    assert the precondition for it."""

    def __init__(self, dim: int, base: int, corrupted_inv_freq: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", corrupted_inv_freq, persistent=False)
        self.max_seq_len_cached = 11  # a real forward call would have set this
        # Garbage caches, standing in for what a real model's stale
        # cos_cached/sin_cached (derived from the corrupted inv_freq)
        # would look like right after loading, before any fix runs.
        self.register_buffer("cos_cached", torch.zeros(11, dim), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(11, dim), persistent=False)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, cast(torch.Tensor, self.inv_freq))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class _FakeAttention(torch.nn.Module):
    def __init__(self, rotary_emb: _FakeRotaryEmbedding) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb


class _NativeLlamaStyleRotaryEmbedding(torch.nn.Module):
    """Shaped like transformers' own (non-remote-code) 5.x
    LlamaRotaryEmbedding: has inv_freq and max_seq_len_cached, but not
    dim/base (it takes a config object in its __init__ instead) -- the
    exact shape a native, non-DeepSeek draft model's rotary embedding
    has today, per the final review. Must NOT match: a real Llama-native
    model's inv_freq is not corrupted by this bug, so fixing it here
    would rewrite a buffer this function has no evidence is wrong."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.1, 0.01, 0.001]), persistent=False)
        self.max_seq_len_cached = 2048


def _expected_inv_freq(dim: int, base: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def test_fix_rope_inv_freq_recomputes_a_corrupted_buffer() -> None:
    garbage = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    rotary = _FakeRotaryEmbedding(dim=8, base=10000, corrupted_inv_freq=garbage)
    model = _FakeAttention(rotary)

    fixed = fix_rope_inv_freq(model)

    assert fixed == 1
    # Literal expected values, not a second call to the same formula the
    # implementation uses -- a shared misunderstanding of DeepSeek's own
    # formula would otherwise pass silently (dim=8, base=10000: exponents
    # 0/8, 2/8, 4/8, 6/8 -> 10000**0, 10000**0.25, 10000**0.5, 10000**0.75
    # -> 1, 10, 100, 1000, inverted).
    assert torch.allclose(
        cast(torch.Tensor, rotary.inv_freq), torch.tensor([1.0, 0.1, 0.01, 0.001])
    )
    assert rotary.max_seq_len_cached is None


def test_fix_rope_inv_freq_forces_forward_to_rebuild_from_the_corrected_buffer() -> None:
    """The fix has two halves: recomputing inv_freq, and resetting
    max_seq_len_cached so the next forward() call actually rebuilds
    cos_cached/sin_cached from it (they were derived from the corrupted
    buffer and are just as wrong). This test fails if either half is
    missing: skip the inv_freq fix and the expected values below are
    wrong; skip the max_seq_len_cached reset and forward() returns the
    stale garbage cache untouched (seq_len=5 <= the fake's initial
    max_seq_len_cached=11, so the guard's `seq_len > ...` branch alone
    would never trigger a rebuild)."""
    dim, base = 8, 10000
    garbage = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    rotary = _FakeRotaryEmbedding(dim, base, garbage)
    model = _FakeAttention(rotary)

    fix_rope_inv_freq(model)
    cos, sin = rotary(seq_len=5)

    t = torch.arange(5, dtype=torch.float32)
    expected_freqs = torch.einsum("i,j->ij", t, _expected_inv_freq(dim, base))
    expected_emb = torch.cat((expected_freqs, expected_freqs), dim=-1)
    assert torch.allclose(cos, expected_emb.cos())
    assert torch.allclose(sin, expected_emb.sin())


def test_fix_rope_inv_freq_fixes_every_layer_found() -> None:
    class ManyLayers(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [_FakeAttention(_FakeRotaryEmbedding(8, 10000, torch.zeros(4))) for _ in range(3)]
            )

    model = ManyLayers()

    fixed = fix_rope_inv_freq(model)

    assert fixed == 3
    for module in model.layers:
        layer = cast(_FakeAttention, module)
        assert torch.allclose(
            cast(torch.Tensor, layer.rotary_emb.inv_freq), _expected_inv_freq(8, 10000)
        )


def test_fix_rope_inv_freq_ignores_modules_without_the_rope_shape() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    fixed = fix_rope_inv_freq(model)

    assert fixed == 0


def test_fix_rope_inv_freq_ignores_a_native_llama_shaped_rotary_embedding() -> None:
    model = _NativeLlamaStyleRotaryEmbedding()

    fixed = fix_rope_inv_freq(model)

    assert fixed == 0
