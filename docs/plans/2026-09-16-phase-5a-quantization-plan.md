# Phase 5a: int8 Quantized Grouped-GEMM Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend dispatch's custom Triton grouped-GEMM kernel to consume
int8 weight-only quantized expert weights, wire it through the existing
patch/CLI plumbing as a new path alongside the bf16 backends, and measure
its real memory/throughput/quality tradeoff against Phase 0's stock model
and Phase 1's bf16 naive kernel on the same hardware.

**Architecture:** A fully parallel set of quantized types and functions
(`QuantizedTensor`, `QuantizedStackedExpertWeights`,
`grouped_moe_routed_quantized`, all new, in a new
`src/dispatch/kernels/quantization.py`) sits alongside the existing bf16
contract (`StackedExpertWeights`, `grouped_moe_routed` in `moe_forward.py`)
without modifying it -- the same pattern Phase 3 used for
`patch_moe_infer_ep` next to `patch_moe_infer`. A new Triton kernel
(`grouped_matmul_int8`, in a new `grouped_gemm_int8.py`, built on the
naive kernel's grid/launch structure) dequantizes each int8 weight tile
against its per-output-channel scale inside the tile loop. `integration.py`
gains `patch_moe_infer_quantized`, sharing a small extracted
`_iter_validated_moe_layers` helper with the unmodified `patch_moe_infer`.
`scripts/run_baseline.py --moe-kernel quantized` routes through this new
path instead of the existing `backends.BACKENDS` registry.

**Tech Stack:** Triton (Linux-only, lazily imported everywhere it's
touched, matching the existing convention). PyTorch. No new third-party
dependency -- quantization scales are computed in dispatch's own code,
not sourced from bitsandbytes/AWQ (design doc §2).

**Spec:** `docs/design/2026-09-16-phase-5a-quantization.md` (all
sections). Also reused, unmodified: `src/dispatch/kernels/moe_forward.py`
(`StackedExpertWeights`, `stack_expert_weights`, `torch_grouped_matmul`,
`assert_matches_reference`), `src/dispatch/kernels/grouping.py`
(`group_tokens_by_expert`, `ungroup_and_combine`),
`src/dispatch/kernels/tile_schedule.py` (`TileSchedule`,
`build_tile_schedule`), `src/dispatch/kernels/reference_moe.py`
(`MoEConfig`, `ReferenceMoE`), `src/dispatch/benchmark/reference.py`
(`compare_top_k_agreement`, `capture_reference_logits`, `save_reference`,
`load_reference`), `scripts/gpu/provision.py` (`write_cost_record`).

## Global Constraints

- **Weight-only int8 (W8A16).** Activations never quantized. Only routed
  expert weights already flowing through Phase 1's kernel (`gate_proj`,
  `up_proj`, `down_proj`) are quantized -- shared experts, attention, the
  gate, and `lm_head` stay bf16 (design doc §2).
- **Per-output-channel symmetric round-to-nearest scales, self-computed**
  (`scale = max(abs(weight_row)) / 127`) -- no bitsandbytes/AWQ dependency
  (design doc §2).
- **Built on the naive kernel's structure, not persistent** -- naive has
  won every benchmark since Phase 1 (design doc §2).
- **The existing bf16 contract is not modified.**
  `GroupedMatmul`/`StackedExpertWeights`/`grouped_moe_routed` in
  `moe_forward.py`, and `BACKENDS`/`resolve_backend` in `backends.py`,
  keep their exact current signatures and behavior. Every new quantized
  type/function is additive, in a new module (design doc §3).
- **Two distinct correctness bars, never conflated (design doc §5):**
  - *Kernel-level* (tight, same `assert_matches_reference` discipline as
    every prior phase): the Triton int8 kernel vs. `torch_grouped_matmul_dequant`
    fed the *same* already-quantized weights. A mutation must turn this red.
  - *Model-level* (stated, expected to show real divergence): mutual
    top-5/top-1 agreement (`compare_top_k_agreement`, unmodified) between
    the int8-patched real model and Phase 1's bf16-naive-patched real
    model. Reported as measured, never forced toward 100%.
- **Budget cap: $5, a ceiling not a target**, single GPU (RTX 3090/L40
  class), one combined session (correctness gate, then the measured run),
  checked live against real-time marketplace/spot availability at rental
  time. This is a live session with the user, not something to run
  unattended -- get explicit go-ahead before renting (Task 6).
- **Never quote a benchmark number that wasn't measured** on this exact
  run (repo-wide rule).
- **Out of scope for this plan** (design doc §7): int4, activation
  quantization (W8A8), bitsandbytes/AWQ-sourced scales, a quantized
  persistent kernel, combining with multi-GPU EP, speculative decoding
  (Phase 5b, separate).

---

### Task 1: Int8 per-channel weight quantization primitives

**Files:**
- Create: `src/dispatch/kernels/quantization.py`
- Test: `tests/unit/test_quantization.py`

**Interfaces:**
- Produces: `QuantizedTensor` (frozen dataclass: `data: torch.Tensor`
  int8 shape `(E, N, K)`; `scale: torch.Tensor` float32 shape `(E, N)`).
  `quantize_per_channel_int8(weight: torch.Tensor) -> QuantizedTensor`.
  `dequantize_int8(qtensor: QuantizedTensor) -> torch.Tensor` (returns
  float32).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_quantization.py`:

```python
"""CPU-only proof of the int8 quantization math: per-channel scale
computation, round-trip error bounds, and the zero-channel edge case.
Per docs/design/2026-09-16-phase-5a-quantization.md section 2."""

from __future__ import annotations

import torch

from dispatch.kernels.quantization import dequantize_int8, quantize_per_channel_int8


def test_quantize_dequantize_round_trip_is_within_one_quantization_step() -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 5, 7) * 10

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    max_step = weight.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((dequantized - weight).abs() <= max_step * 1.01)


def test_quantized_data_is_int8_and_within_range() -> None:
    torch.manual_seed(0)
    weight = torch.randn(2, 4, 6) * 100

    quantized = quantize_per_channel_int8(weight)

    assert quantized.data.dtype == torch.int8
    assert quantized.scale.dtype == torch.float32
    assert quantized.data.abs().max() <= 127


def test_quantized_shapes_match_weight() -> None:
    weight = torch.randn(6, 8, 10)

    quantized = quantize_per_channel_int8(weight)

    assert quantized.data.shape == (6, 8, 10)
    assert quantized.scale.shape == (6, 8)


def test_an_all_zero_channel_does_not_produce_nan() -> None:
    weight = torch.zeros(1, 2, 4)
    weight[0, 1] = torch.tensor([1.0, -2.0, 3.0, -4.0])

    quantized = quantize_per_channel_int8(weight)
    dequantized = dequantize_int8(quantized)

    assert torch.isfinite(dequantized).all()
    assert torch.equal(dequantized[0, 0], torch.zeros(4))
    assert quantized.data[0, 0].abs().sum() == 0


def test_scale_is_per_expert_per_output_channel() -> None:
    weight = torch.zeros(2, 2, 3)
    weight[0, 0] = torch.tensor([1.0, -1.0, 0.5])
    weight[0, 1] = torch.tensor([10.0, -10.0, 5.0])

    quantized = quantize_per_channel_int8(weight)

    assert quantized.scale.shape == (2, 2)
    assert quantized.scale[0, 1] > quantized.scale[0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_quantization.py -v`
Expected: FAIL (collection error) -- `dispatch.kernels.quantization` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `src/dispatch/kernels/quantization.py`:

```python
"""Weight-only int8 quantization for MoE expert weights: per-output-channel,
symmetric, round-to-nearest scales computed directly in dispatch's own
code -- not sourced from bitsandbytes/AWQ, per
docs/design/2026-09-16-phase-5a-quantization.md section 2. Activations are
never quantized; only expert weight tensors are."""

from __future__ import annotations

from dataclasses import dataclass

import torch

INT8_MAX = 127


@dataclass(frozen=True)
class QuantizedTensor:
    """An (E, N, K) weight tensor quantized to int8, one scale per expert
    per output channel (E, N)."""

    data: torch.Tensor
    scale: torch.Tensor


def quantize_per_channel_int8(weight: torch.Tensor) -> QuantizedTensor:
    """weight: (..., N, K). scale[..., n] = max(abs(weight[..., n, :])) / 127.
    An all-zero channel's scale is clamped away from zero so dividing by it
    is a no-op (result: 0) rather than a NaN-producing divide-by-zero."""
    absmax = weight.detach().abs().amax(dim=-1)
    scale = (absmax / INT8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    quantized = (weight.detach() / scale.unsqueeze(-1)).round().clamp(-INT8_MAX, INT8_MAX)
    return QuantizedTensor(data=quantized.to(torch.int8), scale=scale.to(torch.float32))


def dequantize_int8(qtensor: QuantizedTensor) -> torch.Tensor:
    return qtensor.data.to(torch.float32) * qtensor.scale.unsqueeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_quantization.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git checkout -b phase-5a-quantization
git add src/dispatch/kernels/quantization.py tests/unit/test_quantization.py
git commit -m "feat: add int8 per-channel weight quantization primitives"
```

---

### Task 2: Quantized MoE forward path and memory-footprint measurement

**Files:**
- Modify: `src/dispatch/kernels/quantization.py`
- Test: Modify `tests/unit/test_quantization.py`

**Interfaces:**
- Consumes: `StackedExpertWeights`, `stack_expert_weights`,
  `torch_grouped_matmul` (`moe_forward.py`); `group_tokens_by_expert`,
  `ungroup_and_combine` (`grouping.py`); `TileSchedule`,
  `build_tile_schedule` (`tile_schedule.py`); `MoEConfig`, `ReferenceMoE`
  (`reference_moe.py`, tests only). `QuantizedTensor`,
  `quantize_per_channel_int8`, `dequantize_int8` (Task 1).
- Produces: `QuantizedStackedExpertWeights` (dataclass: `gate, up, down:
  QuantizedTensor`; `num_experts: int` property).
  `QuantizedGroupedMatmul = Callable[[torch.Tensor, QuantizedTensor,
  TileSchedule], torch.Tensor]`.
  `quantize_stacked_weights(weights: StackedExpertWeights) ->
  QuantizedStackedExpertWeights`.
  `torch_grouped_matmul_dequant(x: torch.Tensor, qweight: QuantizedTensor,
  schedule: TileSchedule) -> torch.Tensor`.
  `grouped_moe_routed_quantized(x, topk_idx, topk_weight, weights:
  QuantizedStackedExpertWeights, matmul: QuantizedGroupedMatmul, *,
  block_m: int = 16) -> torch.Tensor`.
  `stacked_weights_nbytes(weights: StackedExpertWeights) -> int`.
  `quantized_stacked_weights_nbytes(weights: QuantizedStackedExpertWeights)
  -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_quantization.py`:

```python
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.quantization import (
    grouped_moe_routed_quantized,
    quantize_stacked_weights,
    quantized_stacked_weights_nbytes,
    stacked_weights_nbytes,
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


def test_grouped_moe_routed_quantized_matches_dequantized_weights_reference() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    quantized_weights = quantize_stacked_weights(weights)

    dequantized = StackedExpertWeights(
        gate=dequantize_int8(quantized_weights.gate),
        up=dequantize_int8(quantized_weights.up),
        down=dequantize_int8(quantized_weights.down),
    )
    expected = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, dequantized, torch_grouped_matmul
    )

    actual = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, quantized_weights, torch_grouped_matmul_dequant
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_quantize_stacked_weights_shapes() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    weights = stack_expert_weights(moe.experts)

    quantized = quantize_stacked_weights(weights)

    assert quantized.num_experts == 4
    assert quantized.gate.data.shape == weights.gate.shape
    assert quantized.down.data.shape == weights.down.shape


def test_quantized_stacked_weights_are_smaller_than_bf16() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    weights = stack_expert_weights(moe.experts)
    bf16_weights = StackedExpertWeights(
        gate=weights.gate.to(torch.bfloat16),
        up=weights.up.to(torch.bfloat16),
        down=weights.down.to(torch.bfloat16),
    )
    quantized = quantize_stacked_weights(weights)

    bf16_bytes = stacked_weights_nbytes(bf16_weights)
    int8_bytes = quantized_stacked_weights_nbytes(quantized)

    assert int8_bytes < bf16_bytes


def test_stacked_weights_nbytes_counts_every_projection() -> None:
    weights = StackedExpertWeights(
        gate=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        up=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        down=torch.zeros(2, 4, 3, dtype=torch.bfloat16),
    )

    assert stacked_weights_nbytes(weights) == 3 * (2 * 3 * 4 * 2)


def test_quantized_stacked_weights_nbytes_counts_data_and_scale() -> None:
    from dispatch.kernels.quantization import QuantizedStackedExpertWeights, QuantizedTensor

    tensor = QuantizedTensor(
        data=torch.zeros(2, 3, 4, dtype=torch.int8), scale=torch.zeros(2, 3, dtype=torch.float32)
    )
    weights = QuantizedStackedExpertWeights(gate=tensor, up=tensor, down=tensor)

    expected = 3 * (2 * 3 * 4 * 1 + 2 * 3 * 4)
    assert quantized_stacked_weights_nbytes(weights) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_quantization.py -v`
Expected: FAIL -- `grouped_moe_routed_quantized`, `quantize_stacked_weights`,
etc. don't exist yet.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/dispatch/kernels/quantization.py`:

```python
from collections.abc import Callable

import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine
from dispatch.kernels.moe_forward import StackedExpertWeights, torch_grouped_matmul
from dispatch.kernels.tile_schedule import TileSchedule, build_tile_schedule


@dataclass(frozen=True)
class QuantizedStackedExpertWeights:
    """Each projection's int8-quantized weights for every expert."""

    gate: QuantizedTensor
    up: QuantizedTensor
    down: QuantizedTensor

    @property
    def num_experts(self) -> int:
        return int(self.gate.data.shape[0])


QuantizedGroupedMatmul = Callable[[torch.Tensor, QuantizedTensor, TileSchedule], torch.Tensor]


def quantize_stacked_weights(weights: StackedExpertWeights) -> QuantizedStackedExpertWeights:
    return QuantizedStackedExpertWeights(
        gate=quantize_per_channel_int8(weights.gate),
        up=quantize_per_channel_int8(weights.up),
        down=quantize_per_channel_int8(weights.down),
    )


def torch_grouped_matmul_dequant(
    x: torch.Tensor, qweight: QuantizedTensor, schedule: TileSchedule
) -> torch.Tensor:
    """The Triton int8 kernel's correctness oracle: dequantizes qweight --
    the *same* already-quantized weights the kernel sees -- and runs the
    same per-expert-slice matmul torch_grouped_matmul does. Not a
    model-quality reference: both sides here use identical quantized
    weights, so nothing should diverge beyond float precision."""
    return torch_grouped_matmul(x, dequantize_int8(qweight).to(x.dtype), schedule)


def grouped_moe_routed_quantized(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: QuantizedStackedExpertWeights,
    matmul: QuantizedGroupedMatmul,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    grouping = group_tokens_by_expert(topk_idx, topk_weight, weights.num_experts)
    schedule = build_tile_schedule(grouping.group_sizes, block_m)
    gathered = x[grouping.sorted_token_idx]
    gate_out = matmul(gathered, weights.gate, schedule)
    up_out = matmul(gathered, weights.up, schedule)
    expert_out = matmul(F.silu(gate_out) * up_out, weights.down, schedule)
    return ungroup_and_combine(expert_out, grouping, num_tokens=x.shape[0])


def stacked_weights_nbytes(weights: StackedExpertWeights) -> int:
    return sum(t.element_size() * t.nelement() for t in (weights.gate, weights.up, weights.down))


def quantized_stacked_weights_nbytes(weights: QuantizedStackedExpertWeights) -> int:
    total = 0
    for qtensor in (weights.gate, weights.up, weights.down):
        total += qtensor.data.element_size() * qtensor.data.nelement()
        total += qtensor.scale.element_size() * qtensor.scale.nelement()
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_quantization.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/kernels/quantization.py tests/unit/test_quantization.py
git commit -m "feat: add quantized MoE forward path and memory-footprint measurement"
```

---

### Task 3: Triton int8 grouped-GEMM kernel

**Files:**
- Create: `src/dispatch/kernels/grouped_gemm_int8.py`
- Modify: `src/dispatch/kernels/backends.py`
- Test: Create `tests/unit/test_grouped_gemm_int8_kernel.py`
- Test: Modify `tests/unit/test_backends.py`

**Interfaces:**
- Consumes: `QuantizedTensor`, `quantize_per_channel_int8`,
  `quantize_stacked_weights`, `torch_grouped_matmul_dequant`,
  `grouped_moe_routed_quantized`, `QuantizedGroupedMatmul` (Tasks 1-2).
  `TileSchedule`, `build_tile_schedule` (`tile_schedule.py`).
  `DEFAULT_BLOCK_N`, `DEFAULT_BLOCK_K`, `MIN_BLOCK_M`
  (`grouped_gemm.py`, reused unchanged). `assert_matches_reference`
  (`moe_forward.py`). `MoEConfig`, `ReferenceMoE` (`reference_moe.py`,
  tests only). `stack_expert_weights` (`moe_forward.py`, tests only).
- Produces: `grouped_matmul_int8(x: torch.Tensor, qweight: QuantizedTensor,
  schedule: TileSchedule, *, block_n: int = DEFAULT_BLOCK_N, block_k: int
  = DEFAULT_BLOCK_K) -> torch.Tensor`.
  `resolve_quantized_backend() -> QuantizedGroupedMatmul` (`backends.py`).

- [ ] **Step 1: Write the failing GPU-marked kernel tests**

Create `tests/unit/test_grouped_gemm_int8_kernel.py`:

```python
"""GPU-only correctness gate for the int8 kernel: it must match
torch_grouped_matmul_dequant fed the *same* already-quantized weights --
the kernel-level bar in docs/design/2026-09-16-phase-5a-quantization.md
section 5. Excluded from CI by the `gpu` marker."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import assert_matches_reference, stack_expert_weights
from dispatch.kernels.quantization import (
    grouped_moe_routed_quantized,
    quantize_per_channel_int8,
    quantize_stacked_weights,
    torch_grouped_matmul_dequant,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

pytest.importorskip("triton", reason="triton ships Linux wheels only")

from dispatch.kernels import grouped_gemm_int8

pytestmark = pytest.mark.gpu

SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="bf16 tensor-core matmul needs compute capability 8.0+ (Ampere or newer)",
)
DTYPES = [torch.float16, pytest.param(torch.bfloat16, marks=SM80)]

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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("group_sizes", "n", "k"),
    [
        ([5, 0, 3, 40], 24, 16),  # toy dims, an empty expert, a multi-tile expert
        ([1, 1, 1, 1, 1, 1], 1408, 2048),  # decode-shaped, gate/up_proj dims
        ([70, 0, 33, 129], 2048, 1408),  # prefill-shaped, down_proj dims, partial tiles
    ],
)
def test_int8_kernel_matches_dequant_reference(
    dtype: torch.dtype, group_sizes: list[int], n: int, k: int
) -> None:
    torch.manual_seed(0)
    sizes = torch.tensor(group_sizes, device="cuda")
    schedule = build_tile_schedule(sizes, block_m=16)
    x = torch.randn(int(sizes.sum()), k, device="cuda", dtype=dtype)
    weight = torch.randn(len(group_sizes), n, k, device="cuda", dtype=torch.float32) / k**0.5
    qweight = quantize_per_channel_int8(weight)

    actual = grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)

    assert_matches_reference(actual, torch_grouped_matmul_dequant(x.float(), qweight, schedule))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("config", "num_tokens"), [(TOY_DIMS, 37), (REAL_DIMS, 1), (REAL_DIMS, 256)]
)
def test_int8_kernel_backed_moe_layer_matches_dequant_reference(
    dtype: torch.dtype, config: MoEConfig, num_tokens: int
) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(config).to(device="cuda", dtype=dtype)
    hidden_states = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=dtype)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    qweights = quantize_stacked_weights(weights)

    expected = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, qweights, torch_grouped_matmul_dequant
    )
    actual = grouped_moe_routed_quantized(
        hidden_states, topk_idx, topk_weight, qweights, grouped_gemm_int8.grouped_matmul_int8
    )

    assert_matches_reference(actual, expected)


def test_int8_kernel_rejects_a_schedule_below_tl_dots_minimum_block_m() -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 16, 16, device="cuda", dtype=torch.float32)
    qweight = quantize_per_channel_int8(weight)
    schedule = build_tile_schedule(torch.tensor([4], device="cuda"), block_m=4)

    with pytest.raises(ValueError, match="block_m"):
        grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)


def test_int8_kernel_rejects_non_cuda_tensors() -> None:
    x = torch.randn(4, 16, dtype=torch.float16)
    weight = torch.randn(1, 16, 16, dtype=torch.float32)
    qweight = quantize_per_channel_int8(weight)
    schedule = build_tile_schedule(torch.tensor([4]), block_m=16)

    with pytest.raises(ValueError, match="CUDA"):
        grouped_gemm_int8.grouped_matmul_int8(x, qweight, schedule)
```

- [ ] **Step 2: Write the failing CPU-only backend-registry test**

Append to `tests/unit/test_backends.py`:

```python
from dispatch.kernels.backends import resolve_quantized_backend


def test_quantized_backend_maps_to_grouped_matmul_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in = types.ModuleType("dispatch.kernels.grouped_gemm_int8")
    stand_in.grouped_matmul_int8 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dispatch.kernels.grouped_gemm_int8", stand_in)
    monkeypatch.setattr(dispatch.kernels, "grouped_gemm_int8", stand_in, raising=False)

    assert resolve_quantized_backend() is stand_in.grouped_matmul_int8
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_backends.py -v`
Expected: FAIL -- `resolve_quantized_backend` doesn't exist yet.

Run (only if on a machine with triton and a CUDA device; otherwise this
suite is `gpu`-marked and skipped until Task 6's rental):
`uv run pytest tests/unit/test_grouped_gemm_int8_kernel.py -v -m gpu`
Expected: FAIL -- `dispatch.kernels.grouped_gemm_int8` does not exist yet.

- [ ] **Step 4: Write the minimal implementation**

Create `src/dispatch/kernels/grouped_gemm_int8.py`:

```python
"""Triton int8 weight-only grouped-GEMM kernel: dequantizes each weight
tile against its per-output-channel scale inside the tile loop, then
accumulates exactly like grouped_gemm.py's naive kernel. Activations (x)
stay bf16/fp16 throughout -- only the weight side is ever int8. Held to
assert_matches_reference against torch_grouped_matmul_dequant fed the same
QuantizedTensor, per docs/design/2026-09-16-phase-5a-quantization.md
section 5.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dispatch.kernels.grouped_gemm import DEFAULT_BLOCK_K, DEFAULT_BLOCK_N, MIN_BLOCK_M
from dispatch.kernels.quantization import QuantizedTensor
from dispatch.kernels.tile_schedule import TileSchedule


@triton.jit  # type: ignore[untyped-decorator]
def _matmul_tile_int8(  # type: ignore[no-untyped-def]  # noqa: PLR0913 -- one pointer/stride per tensor, matching grouped_gemm.py's own kernel
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    m_tile,
    n_tile,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.load(tile_expert_ptr + m_tile)
    row_start = tl.load(tile_row_start_ptr + m_tile)
    valid_rows = tl.load(tile_valid_rows_ptr + m_tile)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m[:, None] < valid_rows
    col_mask_2d = offs_n[None, :] < n
    col_mask_1d = offs_n < n

    x_ptrs = x_ptr + (row_start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )
    scale_ptrs = scale_ptr + expert_id * stride_se + offs_n * stride_sn
    scale_tile = tl.load(scale_ptrs, mask=col_mask_1d, other=1.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        x_tile = tl.load(x_ptrs, mask=row_mask & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask_2d, other=0)
        w_dequant = w_tile.to(tl.float32) * scale_tile[None, :]
        acc += tl.dot(x_tile.to(tl.float32), w_dequant)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + (row_start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask & col_mask_2d)


@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_int8_kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0913
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _matmul_tile_int8(
        x_ptr,
        w_ptr,
        scale_ptr,
        out_ptr,
        tile_expert_ptr,
        tile_row_start_ptr,
        tile_valid_rows_ptr,
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        n,
        k,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def grouped_matmul_int8(
    x: torch.Tensor,
    qweight: QuantizedTensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> torch.Tensor:
    """One CTA per (m_tile, n_tile), dequantizing qweight's int8 tile
    against its per-output-channel scale before accumulating -- the int8
    analog of grouped_gemm.grouped_matmul."""
    out = _validated_quantized_output(x, qweight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = qweight.data.shape[1], qweight.data.shape[2]
    grid = (schedule.num_tiles, triton.cdiv(n, block_n))
    _grouped_matmul_int8_kernel[grid](
        x,
        qweight.data,
        qweight.scale,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        n,
        k,
        *x.stride(),
        *qweight.data.stride(),
        *qweight.scale.stride(),
        *out.stride(),
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _validated_quantized_output(
    x: torch.Tensor, qweight: QuantizedTensor, schedule: TileSchedule
) -> torch.Tensor:
    if not (x.is_cuda and qweight.data.is_cuda and qweight.scale.is_cuda):
        raise ValueError("the Triton int8 grouped-GEMM kernel needs CUDA tensors")
    if qweight.data.dtype != torch.int8:
        raise ValueError(f"expected int8 weight data, got {qweight.data.dtype}")
    if qweight.scale.dtype != torch.float32:
        raise ValueError(f"expected float32 scales, got {qweight.scale.dtype}")
    if x.shape[1] != qweight.data.shape[2]:
        raise ValueError(f"K mismatch: x has {x.shape[1]}, qweight has {qweight.data.shape[2]}")
    if schedule.block_m < MIN_BLOCK_M:
        raise ValueError(
            f"schedule.block_m={schedule.block_m} is below tl.dot's minimum of {MIN_BLOCK_M}"
        )
    return torch.empty((x.shape[0], qweight.data.shape[1]), device=x.device, dtype=x.dtype)
```

Add to `src/dispatch/kernels/backends.py`, after the existing imports:

```python
from dispatch.kernels.quantization import QuantizedGroupedMatmul
```

And append at the end of the file:

```python
def resolve_quantized_backend() -> QuantizedGroupedMatmul:
    from dispatch.kernels import grouped_gemm_int8  # noqa: PLC0415 -- triton is Linux-only

    return grouped_gemm_int8.grouped_matmul_int8
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_backends.py -v`
Expected: all passed.

Run (GPU host only -- this is the `gpu`-marked suite this task's real
verification happens against; on a non-Linux/no-triton dev machine this
step is deferred to Task 6's rental):
`uv run pytest tests/unit/test_grouped_gemm_int8_kernel.py -v -m gpu -rs`
Expected: all passed, 0 skipped (on a compute-capability-8.0+ card). If
`assert_matches_reference`'s default tolerance does not hold here, do not
silently loosen it -- investigate why first (see design doc section 8's
named risk); any justified tolerance change must be a new, explicitly
named constant with a comment explaining the numerical reason, not a
tweaked parameter.

- [ ] **Step 6: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean. (`mypy` runs on `grouped_gemm_int8.py` even without
triton installed locally, same as it already does for `grouped_gemm.py` --
triton ships type stubs.)

- [ ] **Step 7: Commit**

```bash
git add src/dispatch/kernels/grouped_gemm_int8.py src/dispatch/kernels/backends.py \
  tests/unit/test_grouped_gemm_int8_kernel.py tests/unit/test_backends.py
git commit -m "feat: add Triton int8 grouped-GEMM kernel and its backend resolver"
```

---

### Task 4: `patch_moe_infer_quantized`

**Files:**
- Modify: `src/dispatch/kernels/integration.py`
- Test: Modify `tests/unit/test_integration.py`

**Interfaces:**
- Consumes: `QuantizedStackedExpertWeights`, `QuantizedGroupedMatmul`,
  `quantize_stacked_weights`, `grouped_moe_routed_quantized`,
  `torch_grouped_matmul_dequant`, `dequantize_int8`,
  `quantize_per_channel_int8` (Tasks 1-2). `stack_expert_weights`
  (`moe_forward.py`, unchanged).
- Produces: `patch_moe_infer_quantized(model: torch.nn.Module, matmul:
  QuantizedGroupedMatmul, *, block_m: int = 16) -> int` -- same contract
  as the existing `patch_moe_infer`.
- Refactors (no behavior change): `_iter_validated_moe_layers(model:
  torch.nn.Module) -> Iterator[tuple[torch.nn.Module, torch.nn.ModuleList]]`,
  extracted from `patch_moe_infer`'s body and shared by both patch
  functions.

- [ ] **Step 1: Confirm the existing tests are green before refactoring**

Run: `uv run pytest tests/unit/test_integration.py -v`
Expected: 3 passed (this is the baseline the refactor in Step 3 must not break).

- [ ] **Step 2: Write the failing tests for the new function**

Add to `tests/unit/test_integration.py` (add `import copy` to the top):

```python
import copy

from dispatch.kernels.integration import patch_moe_infer, patch_moe_infer_quantized
from dispatch.kernels.quantization import dequantize_int8, quantize_per_channel_int8, torch_grouped_matmul_dequant


def test_patch_quantized_counts_only_moe_layers() -> None:
    torch.manual_seed(0)

    assert (
        patch_moe_infer_quantized(FakeModel(num_moe_layers=3), torch_grouped_matmul_dequant) == 3
    )


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
        for expert in layer.experts:
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


def test_patch_quantized_rejects_experts_that_are_not_a_module_list() -> None:
    class Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.Linear(2, 2)

        def moe_infer(self) -> None:
            raise NotImplementedError

    with pytest.raises(TypeError, match="ModuleList"):
        patch_moe_infer_quantized(Odd(), torch_grouped_matmul_dequant)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_integration.py -v`
Expected: FAIL -- `patch_moe_infer_quantized` doesn't exist yet.

- [ ] **Step 4: Refactor `patch_moe_infer` and add `patch_moe_infer_quantized`**

Replace `src/dispatch/kernels/integration.py`'s `patch_moe_infer` function
and everything below it with:

```python
from collections.abc import Callable, Iterator

from dispatch.kernels.quantization import (
    QuantizedGroupedMatmul,
    QuantizedStackedExpertWeights,
    grouped_moe_routed_quantized,
    quantize_stacked_weights,
)


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
        module.moe_infer = _grouped_moe_infer_quantized(  # type: ignore[assignment]
            quantized_weights, matmul, block_m
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
```

(`Callable` was already imported at the top of the file for `MoEInfer`;
add `Iterator` to that same `from collections.abc import` line rather than
duplicating the import.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_integration.py -v`
Expected: 6 passed (the original 3, still green after the refactor, plus
the 3 new ones).

- [ ] **Step 6: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add src/dispatch/kernels/integration.py tests/unit/test_integration.py
git commit -m "feat: add patch_moe_infer_quantized, sharing layer-iteration with patch_moe_infer"
```

---

### Task 5: CLI wiring -- `--moe-kernel quantized`

**Files:**
- Modify: `scripts/run_baseline.py`
- Test: Modify `tests/unit/test_run_baseline.py`

**Interfaces:**
- Consumes: `resolve_quantized_backend` (`backends.py`, Task 3).
  `patch_moe_infer_quantized` (`integration.py`, Task 4).
- Produces: `run_baseline(..., moe_kernel="quantized")` routes through
  `patch_moe_infer_quantized`/`resolve_quantized_backend` instead of
  `patch_moe_infer`/`resolve_backend`. `--moe-kernel` CLI choices gain
  `"quantized"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_run_baseline.py`:

```python
def test_run_baseline_routes_the_quantized_kernel_through_its_own_patch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_resolve_quantized_backend() -> object:
        calls["resolved"] = True
        return "the-quantized-matmul"

    def fake_patch_moe_infer_quantized(model: object, matmul: object) -> int:
        calls["patched_model"] = model
        calls["patched_matmul"] = matmul
        return 27

    monkeypatch.setattr(
        run_baseline_module, "load_model", lambda *a, **k: ("the-model", "the-tokenizer")
    )
    monkeypatch.setattr(
        run_baseline_module, "resolve_quantized_backend", fake_resolve_quantized_backend
    )
    monkeypatch.setattr(
        run_baseline_module, "patch_moe_infer_quantized", fake_patch_moe_infer_quantized
    )
    monkeypatch.setattr(run_baseline_module, "generate_with_timings", lambda *a, **k: object())
    monkeypatch.setattr(run_baseline_module, "capture_reference_logits", lambda *a, **k: {})

    _, _, moe_layers_patched = run_baseline_module.run_baseline(
        "some/model",
        device="cpu",
        dtype=torch.float32,
        trust_remote_code=False,
        prompts=["hi"],
        repetitions=1,
        max_new_tokens=1,
        moe_kernel="quantized",
    )

    assert moe_layers_patched == 27
    assert calls == {
        "resolved": True,
        "patched_model": "the-model",
        "patched_matmul": "the-quantized-matmul",
    }


def test_run_baseline_refuses_a_quantized_run_that_patches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_baseline_module, "load_model", lambda *a, **k: ("the-model", "the-tokenizer")
    )
    monkeypatch.setattr(run_baseline_module, "resolve_quantized_backend", lambda: object())
    monkeypatch.setattr(run_baseline_module, "patch_moe_infer_quantized", lambda model, matmul: 0)

    with pytest.raises(RuntimeError, match="patched no MoE layers"):
        run_baseline_module.run_baseline(
            "some/model",
            device="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            prompts=["hi"],
            repetitions=1,
            max_new_tokens=1,
            moe_kernel="quantized",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_run_baseline.py -v`
Expected: FAIL -- `run_baseline_module` has no `resolve_quantized_backend`/
`patch_moe_infer_quantized` attribute to monkeypatch, and `"quantized"`
isn't a recognized `moe_kernel` value yet.

- [ ] **Step 3: Write the minimal implementation**

In `scripts/run_baseline.py`, change the imports:

```python
from dispatch.kernels.backends import BACKENDS, resolve_backend, resolve_quantized_backend
from dispatch.kernels.integration import patch_moe_infer, patch_moe_infer_quantized
```

Replace the patch block inside `run_baseline`:

```python
    moe_layers_patched = 0
    if moe_kernel == "quantized":
        moe_layers_patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
    elif moe_kernel != "none":
        moe_layers_patched = patch_moe_infer(model, resolve_backend(moe_kernel))
    if moe_kernel != "none" and moe_layers_patched == 0:
        raise RuntimeError(
            f"--moe-kernel {moe_kernel} patched no MoE layers: {model_name} has no "
            "moe_infer to replace, so this run would time the stock model under a kernel's name"
        )
```

Change the `--moe-kernel` argument's choices:

```python
    parser.add_argument("--moe-kernel", default="none", choices=["none", *BACKENDS, "quantized"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_baseline.py -v`
Expected: all passed.

- [ ] **Step 5: Run the full CPU-only suite**

Run: `make test`
Expected: all pass, nothing else broken by this task's edits.

- [ ] **Step 6: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_baseline.py tests/unit/test_run_baseline.py
git commit -m "feat: wire --moe-kernel quantized through patch_moe_infer_quantized"
```

---

### Task 6: GPU rental runbook -- kernel correctness gate, three-way measured run

**Files:**
- Create: `docs/runbooks/phase-5a-quantization.md`
- Creates live, on the pod / written by the CLIs (not pre-committed as
  code, committed as evidence after the session):
  `docs/findings/<date>-phase-5a-stock-reference.safetensors`,
  `docs/findings/<date>-phase-5a-stock-results.json`,
  `docs/findings/<date>-phase-5a-naive-results.json`,
  `docs/findings/<date>-phase-5a-quantized-results.json`,
  `docs/findings/<date>-phase-5a-memory-footprint.json`
- Create (via `write_cost_record`, reused unmodified):
  `docs/findings/<date>-phase-5a-quantization-cost.md`

**Budget cap: $5, a ceiling not a target -- be surgical (Global
Constraints). This is a live session with the user, not something to run
unattended -- get explicit go-ahead and confirm the cap before renting.**

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/phase-5a-quantization.md`:

````markdown
# Runbook: Phase 5a int8 quantized kernel (rented GPU)

One combined session -- kernel correctness, then the measured run --
matching Phase 3/4's single-session shape rather than Phase 1's two
separate sessions, per docs/design/2026-09-16-phase-5a-quantization.md
section 8. Budget cap: $5, a ceiling not a target.

1. Pick the cheapest Community Cloud card with compute capability 8.0+
   (Triton's own floor) -- RTX 3090/4090, L4, L40, A40, A100, checked
   live against real-time availability, same practice every prior phase
   used.
2. Create and wait, with a 50GB+ volume (Phase 0's real trap: the model
   is 32.8GB and the container disk alone is often only 30GB):

       uv run python -m scripts.gpu.provision create --name dispatch-phase-5a \
         --gpu-type "<id from step 1>" --image "<current runpod/pytorch tag>" \
         --cloud COMMUNITY --disk-gb 60
       uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

3. Confirm the card:

       nvidia-smi --query-gpu=name,compute_cap --format=csv

4. Transfer and set up, all of Phase 0/1's known environment traps in one
   place -- **confirm live whether DeepSeek's `modeling_deepseek.py` still
   needs these** (check
   https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/commits/main
   since Phase 1's session) before assuming either is still required:

       git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"
       ssh <pod>
       cd dispatch
       export HF_HOME=/workspace/hf_cache   # NOT the container disk
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv pip install transformers==4.57.6  # only if step 4's live check still shows the break

   If the `get_usable_length` break (Phase 0/1) is still present, use the
   same pod-local, never-committed wrapper Phase 1's Task 9 used:

       cat > _patch_and_run.py <<'EOF'
       import sys
       from transformers.cache_utils import DynamicCache


       def _get_usable_length(self, new_seq_length=None, layer_idx=0):
           return self.get_seq_length(layer_idx)


       DynamicCache.get_usable_length = _get_usable_length

       from scripts.run_baseline import main

       main(sys.argv[1:])
       EOF

   From here on invoke `.venv/bin/python` directly (`uv run` re-syncs to
   `uv.lock` on every call and silently undoes the `transformers` override).
   If the wrapper isn't needed, invoke `scripts/run_baseline.py` directly
   instead of `_patch_and_run.py` in every command below.

5. Full CPU-testable suite, one more time, on the pod itself:

       .venv/bin/python -m pytest -m "not gpu" -v

   Expected: all pass (confirms the pod's environment doesn't disagree
   with what already passed locally).

6. **Kernel correctness gate -- must pass before any timing.**

       .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_int8_kernel.py -v -rs

   Expected: all passed, 0 skipped (on a compute-capability-8.0+ card).
   Also re-verify the existing bf16 kernels still pass on this card
   (cheap, and rules out a hardware-specific surprise unrelated to this
   phase's own new code):

       .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs

   Anything red: stop. Nothing below is meaningful until this is green.

7. **The stock run** -- this session's own reference, same
   `DEFAULT_PROMPTS` every prior phase has used:

       DATE=$(date +%Y-%m-%d)
       .venv/bin/python -m scripts.run_baseline --trust-remote-code \
         --run-label $DATE-phase-5a-stock

8. **The bf16 naive-kernel run** -- Phase 1's own "before" number,
   re-verified on this card, and the model-level agreement bar's
   comparison point:

       .venv/bin/python -m scripts.run_baseline --trust-remote-code \
         --moe-kernel naive \
         --compare-reference docs/findings/$DATE-phase-5a-stock-reference.safetensors \
         --run-label $DATE-phase-5a-naive

   Must show `"moe_layers_patched": 27` and `mutual_top_k: true` against
   the stock reference.

9. **The int8 quantized run** -- the model-level correctness bar from
   design doc section 5, compared against the bf16 naive run (not the
   stock run: quantization-induced divergence is expected, and the naive
   kernel is this project's own closest apples-to-apples bf16 comparison
   point):

       .venv/bin/python -m scripts.run_baseline --trust-remote-code \
         --moe-kernel quantized \
         --compare-reference docs/findings/$DATE-phase-5a-naive-reference.safetensors \
         --run-label $DATE-phase-5a-quantized

   Must show `"moe_layers_patched": 27`. Record the measured
   `mutual_top_k`/`top1_agreement` values from the results JSON exactly as
   they come back -- this is expected to show some real divergence; do
   not treat anything short of perfect agreement as a failure unless
   `mutual_top_k` is false (an argmax landing outside the other side's
   top-k, the same bar every prior phase has used to distinguish a real
   bug from an acceptable near-tie flip).

10. **Memory footprint.** From the repo root on the pod:

        .venv/bin/python <<'EOF'
        import json
        import torch

        from dispatch.benchmark.harness import load_model
        from dispatch.kernels.moe_forward import stack_expert_weights
        from dispatch.kernels.quantization import (
            quantize_stacked_weights,
            quantized_stacked_weights_nbytes,
            stacked_weights_nbytes,
        )

        model, _ = load_model(
            "deepseek-ai/deepseek-moe-16b-base",
            device="cuda",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        total_bf16 = 0
        total_int8 = 0
        layers = 0
        for module in model.modules():
            if not hasattr(module, "moe_infer"):
                continue
            weights = stack_expert_weights(module.experts)
            total_bf16 += stacked_weights_nbytes(weights)
            total_int8 += quantized_stacked_weights_nbytes(quantize_stacked_weights(weights))
            layers += 1

        report = {
            "layers": layers,
            "bf16_bytes": total_bf16,
            "int8_bytes": total_int8,
            "reduction_pct": round(100 * (1 - total_int8 / total_bf16), 2),
        }
        print(json.dumps(report, indent=2))
        with open("docs/findings/memory-footprint.json", "w") as f:
            json.dump(report, f, indent=2)
        EOF

        mv docs/findings/memory-footprint.json docs/findings/$DATE-phase-5a-memory-footprint.json

11. **Throughput comparison.** Read `mean_tokens_per_second` out of the
    three results JSONs from steps 7-9 (`$DATE-phase-5a-{stock,naive,quantized}-results.json`)
    -- no separate benchmark tool needed, `run_baseline`'s own harness
    already measured it identically for all three.

12. **Cost record and teardown:**

        .venv/bin/python -c "
        from pathlib import Path
        from scripts.gpu.provision import write_cost_record
        write_cost_record(
            Path('docs/findings'),
            pod_id='<pod_id>',
            gpu_type_id='<id from step 1>',
            cost_per_hour=<rate>,
            duration_s=<elapsed>,
            note='Phase 5a int8 quantized kernel: correctness gate + three-way measured run',
            run_label='phase-5a-quantization',
        )
        "

    Then tear the pod down immediately:

        uv run python -m scripts.gpu.provision delete --pod-id <pod_id>
        uv run python -m scripts.gpu.provision get --pod-id <pod_id>  # confirm TERMINATED
````

- [ ] **Step 2: Get explicit go-ahead and run the session**

Confirm with the user: budget cap ($5), GPU class/price at real-time
availability, and that this is a live, watched session -- then execute
the runbook above. Copy back every result JSON and the cost/memory-footprint
files into `docs/findings/` on the local machine before tearing the pod down.

- [ ] **Step 3: Commit the evidence**

```bash
git add docs/runbooks/phase-5a-quantization.md docs/findings/*phase-5a*
git commit -m "docs: run Phase 5a GPU correctness gate and measured session"
```

---

### Task 7: Findings doc and STATUS.md

**Files:**
- Create: `docs/findings/<date>-phase-5a-quantization-run.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write the findings doc**

Cover, in `docs/findings/<date>-phase-5a-quantization-run.md`: whether the
kernel-level correctness gate passed (Task 6 step 6), and on which GPU;
the stock/naive/quantized throughput numbers from the three results
JSONs, each with its full config (model, dtype, hardware, prompts,
repetitions); the memory-footprint numbers (bytes and percent reduction)
from Task 6 step 10; the model-level agreement result (quantized vs.
naive) exactly as measured, including `top1_agreement` and whether
`mutual_top_k` held -- reported honestly even if agreement is well below
100%; cost per 1M generated tokens for all three configurations (same
`$/hr / (tokens/sec * 3600) * 1e6` computation Phase 1's findings doc
used); the actual GPU type, duration, and cost from Task 6's cost record,
**explicitly compared against the $5 cap**; and whether the
`transformers==4.57.6`/`get_usable_length` workarounds were still needed
or had been resolved upstream since Phase 1. If the kernel correctness
gate failed, or the budget ran out before all three runs completed, say
so plainly and report exactly what was measured, matching this project's
practice of writing down a null or partial result rather than a
flattering guess.

- [ ] **Step 2: Update STATUS.md**

Add a "## Phase 5a progress" section following the Phase 0-4 pattern:
design and plan links, task checklist, the kernel-level and model-level
correctness results, the throughput/memory numbers, and total GPU cost
against the $5 cap. Update "## Next step" to note Phase 5a is complete
and Phase 5b (speculative decoding) is next, not yet planned.

- [ ] **Step 3: Commit**

```bash
git add docs/findings/<date>-phase-5a-quantization-run.md docs/STATUS.md
git commit -m "docs: record Phase 5a int8 quantized kernel outcome"
```

Then open the PR for the whole `phase-5a-quantization` branch, per this
repo's one-branch-per-phase convention (README/CLAUDE.md refresh first,
per the standing instruction to update them proactively at natural
stopping points).

---

## Self-Review Notes

- **Spec coverage:** design doc §2 (scope: weight-only int8, self-computed
  scales, naive-kernel-based, single GPU, $5 cap) is a Global Constraint,
  restated in Task 6's runbook. §3 (architecture: parallel quantized
  types, new kernel file, `patch_moe_infer_quantized`, CLI routing) maps
  to Tasks 1-5 one component per task, in dependency order (quantization
  math before the forward path that uses it, before the kernel that's
  pluggable into it, before the integration wiring, before the CLI).
  §4 (data flow: quantize once at patch time, forward pass reuses
  unchanged grouping/tiling, dequant inside the kernel) is implemented in
  Task 4's `patch_moe_infer_quantized` (quantize once) and Task 3's kernel
  (dequant inside the tile loop) respectively. §5 (two correctness bars)
  is a Global Constraint and is concretely implemented: kernel-level in
  Task 3's tests (`torch_grouped_matmul_dequant` reference), model-level
  in Task 6 step 9 (naive-vs-quantized agreement, reported as measured).
  §6 (testing table) maps one-to-one onto Tasks 1-5 (CPU rows) and Task 6
  (the two `gpu`/paid rows, plus the new memory-footprint measurement).
  §7 (non-goals) is listed in Global Constraints so no task accidentally
  reaches for bitsandbytes, int4, or the persistent kernel. §8 (risk/cost)
  is the $5 cap (Global Constraints, Task 6) and the named tolerance/scale
  risks (Task 3 step 5's explicit contingency instruction, not a silent
  workaround).
- **Placeholder scan:** no task step describes an action without showing
  the code; every test has real assertions; the GPU runbook's "confirm
  live" instructions (transformers workaround, tolerance) are concrete
  contingencies tied to a specific, already-documented prior finding
  (Phase 0/1's real bugs), not vague "handle appropriately" language.
- **Type consistency:** `QuantizedTensor` (Task 1) is used identically by
  every later task -- `.data` (int8, `(E,N,K)`), `.scale` (float32,
  `(E,N)`) -- never renamed. `QuantizedGroupedMatmul`'s signature
  (`Callable[[Tensor, QuantizedTensor, TileSchedule], Tensor]`, Task 2) is
  what `grouped_matmul_int8` (Task 3), `torch_grouped_matmul_dequant`
  (Task 2), and `patch_moe_infer_quantized`'s `matmul` parameter (Task 4)
  all satisfy -- verified directly in Task 3's and Task 4's tests, which
  pass `torch_grouped_matmul_dequant` and (once rented)
  `grouped_matmul_int8` into the exact same call sites. `patch_moe_infer`'s
  post-refactor behavior is pinned by Task 4 step 1 running its existing
  tests *before* the refactor and step 5 confirming they're still green
  *after* -- the same green-refactor-green discipline Phase 1's
  `_require_same_keys` extraction used.
- **A decision the design doc left to this plan, resolved here:** where
  the new Triton kernel lives (design doc §3 said "grouped_gemm.py or a
  new sibling module, decided in the implementation plan"). This plan
  puts it in a new `grouped_gemm_int8.py` (Task 3) rather than growing
  `grouped_gemm.py` further, importing its tuning constants
  (`DEFAULT_BLOCK_N`/`DEFAULT_BLOCK_K`/`MIN_BLOCK_M`) rather than
  duplicating them -- keeps the existing file's two bf16 kernels and this
  phase's one int8 kernel each in a file with one clear dtype contract,
  per the design-for-isolation guidance every prior phase's plan has
  followed.
