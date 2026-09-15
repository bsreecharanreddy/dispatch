# Phase 1: Grouped-GEMM Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom Triton grouped-GEMM kernel for DeepSeekMoE-16B's
routed experts -- naive first, then persistent and cache-aware -- prove each
version numerically correct before timing it, swap it into the real model,
and measure it against a stock run on the same GPU class as Phase 0.

**Architecture:** Three layers, each proven before the next leans on it.
(1) Pure PyTorch, CPU-only, testable on a Mac with no GPU: a reference MoE
block, the token-to-expert grouping, a tile schedule, and an eager
`torch_grouped_matmul` that *defines* the grouped-GEMM contract. (2) Two
Triton kernels implementing that same `GroupedMatmul` signature, each held
to `assert_matches_reference` against the eager version on a rented GPU.
(3) The same pluggable backend swapped into DeepSeek's real model by
replacing each MoE layer's `moe_infer`, judged end to end by mutual top-5
logit agreement against a stock run in the same session, plus a
`triton.testing.do_bench` micro-benchmark that refuses to time any backend
that disagrees with the eager one.

**Tech Stack:** Python 3.12, `uv`, `ruff`, `mypy --strict`, `pytest`,
PyTorch >=2.14.0 (already here), Triton >=3.8.0 (new, Linux-only),
`transformers` (already here -- the real-model run still needs Phase 0's
pod-local 4.57.6 override), RunPod.

**Spec:** `docs/design/2026-09-14-dispatch-system-design.md` §4 (kernel
scope and build order), §6 (benchmark methodology, the same-hardware rule),
§7 (the Phase 1 row), §9 (cost plan), §10 (what "done" looks like);
`docs/adr/0001` (why grouped-GEMM at all); and
`docs/findings/2026-09-14-phase-0-baseline-run.md` -- the three
pod-environment bugs there recur the moment this model is loaded again.

## Global Constraints

- Python 3.12+, `uv`, `ruff`, `mypy --strict`, `pytest`. `make check`
  (lint, typecheck, test) is green before every push.
- **Correctness before speed** (CLAUDE.md's governing principle): no kernel
  is benchmarked until it meets `assert_matches_reference` against
  `torch_grouped_matmul` *on the hardware being measured*. The benchmark
  CLI enforces this itself and refuses to time a disagreeing backend.
- **Tolerances are declared here, before the first run, and are not tuned
  afterwards.** Operator level: `CORRECTNESS_RTOL = 1.6e-2` (the bf16 rtol
  `torch.testing.assert_close` itself defaults to) with
  `atol = CORRECTNESS_RTOL * max|expected|`, so an output whose values are
  all tiny cannot pass on absolute tolerance alone. End to end: mutual
  top-5 agreement at every position of every prompt. Loosening either one
  requires a written justification in `docs/findings/`, never a quiet edit
  to turn a red test green. Known limit, stated up front: this tolerance
  catches indexing, masking and routing bugs by a wide margin; it is not
  tight enough to catch a pure precision downgrade (accumulating in bf16
  instead of fp32, say). The end-to-end logit comparison in Task 9 is the
  check for that.
- **Never quote a number that wasn't measured**, and every benchmark claim
  states its config alongside it: GPU, dtype, token count, routing
  distribution, tile sizes, library versions.
- **Same-hardware rule** (design doc §6): the numbers that get quoted come
  from one session on the same GPU class as Phase 0 (NVIDIA L40), with a
  stock run re-measured *in that same session*. Phase 0's 12.75 tokens/sec
  is context, not the comparison.
- **Triton is Linux-only.** Declared as `triton>=3.8.0; sys_platform ==
  'linux'`. Checked live against pypi.org 2026-09-15: 3.8.0 is current,
  publishes manylinux wheels (x86_64 and aarch64) and nothing else, and
  `torch==2.14.0` already requires `triton~=3.8.0` on Linux, so the floor
  cannot conflict with the pinned torch.
- **Triton's README lists "NVIDIA GPUs (Compute Capability 8.0+)"** as
  supported hardware (fetched 2026-09-15). The cards free tiers typically
  hand out -- T4 (7.5), P100 (6.0) -- are below that line, so kernel
  debugging happens on a cheap *rented* Ampere-or-newer card, not a free
  tier. bf16 test cases additionally skip below 8.0, with the reason
  visible in the test.
- **GPU-dependent tests are marked `gpu`** and excluded from CI, which has
  no GPU runner. The GPU test file opens with
  `pytest.importorskip("triton", ...)` so it *skips with a stated reason*
  on macOS instead of erroring at collection -- but it imports
  `dispatch.kernels.grouped_gemm` normally, so a missing or broken kernel
  module fails loudly rather than skipping quietly.
- **`src/dispatch/kernels/grouped_gemm.py` is the only module that imports
  `triton` at module scope.** Everything else imports it lazily, so the Mac
  dev loop, the CPU test suite and CI's mypy all work without it.
- **mypy:** a `triton.*` override with `ignore_missing_imports` *and*
  `follow_imports = "skip"`, so mypy treats Triton as `Any` whether or not
  it is installed and gives the same answer on macOS and Linux CI. The
  kernels themselves carry narrow `# type: ignore[untyped-decorator]` and
  `# type: ignore[no-untyped-def]` comments: Triton reads kernel parameter
  annotations itself (`tl.constexpr`), so those parameters stay unannotated
  by design.
- **Model facts**, checked live against `deepseek-ai/deepseek-moe-16b-base`'s
  `config.json` on 2026-09-15, not assumed: `hidden_size` 2048,
  `moe_intermediate_size` 1408, `n_routed_experts` 64, `n_shared_experts` 2,
  `num_experts_per_tok` 6, **`norm_topk_prob` false**, `num_hidden_layers`
  28, `first_k_dense_replace` 1, `moe_layer_freq` 1 -- so 27 of 28 layers
  are MoE layers. The top-k gate weights are **not** renormalized; a
  reference that renormalizes them is wrong for this model.
- **Integration point**, checked live against the model's own
  `modeling_deepseek.py` on 2026-09-15: `DeepseekDecoderLayer.mlp` holds a
  `DeepseekMoE`; at inference `DeepseekMoE.forward` calls
  `self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1))`
  and adds `shared_experts` itself; `experts` is an `nn.ModuleList` of
  `DeepseekMLP` (`gate_proj` / `up_proj` / `down_proj`). This plan replaces
  `moe_infer` and nothing else -- gate, attention and shared experts stay
  DeepSeek's code.
- **TMA is out of scope.** The PyTorch post's third optimization needs
  Hopper (SM90+); Phase 0's L40 is Ada Lovelace (SM89). Phase 1's
  "persistent, cache-aware" means the persistent-CTA loop plus grouped
  launch ordering, both of which work on Ampere and newer.
- **Cost discipline:** a budget cap is stated before each rental; every
  rental is a runbook step taken with the user's explicit go-ahead, never
  unattended; teardown is verified independently; cost is measured (never
  estimated) and written to `docs/findings/`.
- **One branch for the phase**, `phase-1-grouped-gemm`, with this plan as
  its first commit. One commit per completed task, `docs/STATUS.md` in the
  same commit, conventional prefixes, **plain-ASCII commit messages (`--`,
  never an em-dash)**. Pushed once at the end as a single PR.

## What was already verified, and what wasn't

Everything in this plan that can run without a GPU was written into a
scratch copy of this repo on 2026-09-15 and run through this repo's own
tooling. Measured there, not predicted:

- `ruff check`, `ruff format --check`, and `mypy --strict src tests
  scripts`: clean.
- `pytest -m "not gpu"` (CI's own selection): **71 passed, 1 skipped** --
  the 26 tests already in the repo plus 45 new ones. The skip is the GPU
  file, reporting `triton ships Linux wheels only`.
- A mutation check: deleting the gate weighting from `ungroup_and_combine`
  turned **5 tests red** across three files; restoring it returned all 71
  to green. The CPU tests can fail.
- The persistent kernel's grouped tile ordering was mirrored in plain
  Python and checked to be a bijection onto every `(m_tile, n_tile)` pair
  across 6,084 grid shapes (1..39 x 1..39 x GROUP_SIZE_M in {1,2,3,8}):
  no gaps, no repeats.
- `uv lock` resolves `triton 3.8.0` behind `sys_platform == 'linux'`.

**Not verified, and the plan does not pretend otherwise:** the two Triton
kernels have never been compiled or executed. The authoring machine has no
CUDA device. Lint and mypy passing on them proves they are valid Python,
nothing more. Task 6 exists precisely to find out what is wrong with them,
and this project's own history says to expect something: Phase 0's one real
run surfaced three environment bugs that no static review had caught.

**Correction found while checking sources:** the design doc (§4) and this
repo's notes cite the PyTorch grouped-GEMM post as "2026". The post,
"Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM
Kernel", is dated **2025-08-19**. Not fixed here -- the design doc is
someone else's document to amend -- but it should be.

## Decisions this plan makes

1. **Contiguous groups, host-built tile schedule.** Triton's tutorial
   handles fully general groups via per-group pointer arrays. MoE's case is
   narrower: one sorted-by-expert tensor, and one `(N, K)` weight shape
   shared by every expert. So the host precomputes a tile schedule (which
   expert each tile belongs to, its first row, how many of its rows are
   real) and the kernel reads that. Every off-by-one risk lands in pure
   index arithmetic that is tested on CPU.
2. **Three grouped GEMMs per layer** (`gate_proj`, `up_proj`, `down_proj`),
   all through one kernel. Fusing `gate` and `up` into a single wider GEMM
   is a real and well-known optimization, and is deliberately **deferred**:
   Phase 1's stated scope is naive then persistent, not operator fusion.
3. **The eager backend is the contract.** `torch_grouped_matmul` is a
   `GroupedMatmul` like the kernels are, proven equal to `ReferenceMoE` on
   CPU. It doubles as the control run on the GPU, isolating "reordered
   arithmetic" from "kernel bug".
4. **Patch `moe_infer`, not `forward`.** The smallest possible surface: the
   routing, the shared experts and everything else stay DeepSeek's own
   code, so a logit difference can only come from the expert GEMMs.
5. **Stacked expert weights share storage.** The kernel needs each
   projection as one `(n_experts, N, K)` tensor. Copying them would add
   roughly 29.9GB (27 layers x 64 experts x 3 projections x 1408 x 2048 x 2
   bytes -- arithmetic from the config, not a measurement) next to a 32.8GB
   model on a 48GB card. So each expert's `Linear` is re-pointed at a view
   of the stack instead.
6. **Kernel-correctness first, on a cheap card** (Task 6, synthetic weights,
   no model download), then the measured run on an L40 (Task 9). Debugging
   a kernel at L40 prices is avoidable; producing comparable numbers on
   anything other than an L40 is not.
7. **`triton.testing.do_bench` for timing**, not a hand-rolled
   `perf_counter` loop -- warmup, repetition and GPU synchronization are
   exactly what it exists to get right, and §6 of the design doc already
   prefers standard tools. Signature checked live 2026-09-15:
   `do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None,
   return_mode='mean')`.
8. **The per-layer host sync stays.** Building the schedule calls
   `.tolist()` on the group sizes, which synchronizes. DeepSeek's own
   `moe_infer` does the same thing (`.bincount().cpu().numpy()`), so the
   comparison is fair; a device-side schedule is a later optimization, not
   a Phase 1 requirement.
9. **vLLM's `benchmark_serving.py` is not used here.** Design doc §6 points
   at it for the final comparison, and it drives an OpenAI-compatible
   server this project does not have until Phase 7. Phase 1 measures the
   kernel with `do_bench` and the model with Phase 0's own harness, which
   is what makes it comparable to Phase 0's number.
10. **A null result is a finding.** Phase 0's baseline is unbatched,
    token-by-token decode: 6 routed rows per MoE layer per step. Grouped
    GEMM earns its keep when groups are large. If end-to-end decode
    throughput barely moves, that gets written down plainly, and the
    micro-benchmark's token-count sweep is what shows where the kernel
    does pay off.

## File Structure

```
src/dispatch/kernels/
  __init__.py              # Task 1 (empty)
  reference_moe.py         # Task 1: MoEConfig, ExpertMLP, ReferenceMoE -- the CPU oracle
  grouping.py              # Task 2: routing -> per-expert-contiguous rows, and back
  tile_schedule.py         # Task 2: group sizes -> the kernel's tile metadata
  moe_forward.py           # Task 3: the GroupedMatmul contract + eager backend
  grouped_gemm.py          # Task 4 (naive), Task 5 (persistent) -- the only triton import
  backends.py              # Task 7: name -> GroupedMatmul, triton imported lazily
  bench.py                 # Task 7: do_bench wrapper, FLOP math, synthetic routing
  integration.py           # Task 8: swap a backend into a real DeepseekMoE
src/dispatch/benchmark/
  reference.py             # Task 8: + TopKAgreement, compare_top_k_agreement
scripts/
  gpu/provision.py         # Task 6: write_cost_record gains run_label
  run_kernel_bench.py      # Task 7: the micro-benchmark CLI
  run_baseline.py          # Task 8: --moe-kernel, --compare-reference
tests/unit/
  test_reference_moe.py    # Task 1
  test_grouping.py         # Task 2
  test_tile_schedule.py    # Task 2
  test_moe_forward.py      # Task 3
  test_grouped_gemm_kernel.py  # Tasks 4-5, marked gpu
  test_provision.py        # Task 6 (modified)
  test_backends.py         # Task 7
  test_kernel_bench.py     # Task 7
  test_run_kernel_bench.py # Task 7
  test_integration.py      # Task 8
  test_reference.py        # Task 8 (modified)
  test_run_baseline.py     # Task 8 (modified)
docs/
  runbooks/phase-1-grouped-gemm.md   # Task 6 (session A), Task 9 (session B)
  findings/                          # Task 6 cost + note, Task 9 results
pyproject.toml, uv.lock              # Task 4
```

---

### Task 1: Pure-PyTorch MoE reference

**Files:**
- Create: `src/dispatch/kernels/__init__.py` (empty)
- Create: `src/dispatch/kernels/reference_moe.py`
- Test: `tests/unit/test_reference_moe.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `MoEConfig(hidden_size: int, moe_intermediate_size: int,
  n_routed_experts: int, n_shared_experts: int, num_experts_per_tok: int)`
  (frozen dataclass); `ExpertMLP(hidden_size: int, intermediate_size: int)`
  with `.gate_proj` / `.up_proj` / `.down_proj` as bias-free `nn.Linear`;
  `ReferenceMoE(config)` with `.config`, `.experts` (`nn.ModuleList`),
  `.shared_experts` (`ExpertMLP | None`), `route(hidden_states) ->
  (topk_idx, topk_weight)`, `routed(hidden_states) -> Tensor`,
  `forward(hidden_states) -> Tensor`. Tasks 2, 3, 4, 5 and 8 all use these.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_reference_moe.py`:

```python
"""CPU-only, toy dims. Covers the shapes, the zero-token expert (the edge
case a grouped-GEMM kernel mishandles first), and the one routing detail
that is easy to get wrong from memory: this model does not renormalize its
top-k weights."""

from __future__ import annotations

import torch

from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


def test_forward_preserves_shape() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)

    assert moe(hidden_states).shape == hidden_states.shape


def test_route_selects_top_k_distinct_experts_per_token() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)

    topk_idx, topk_weight = moe.route(torch.randn(7, TOY_CONFIG.hidden_size))

    assert topk_idx.shape == (7, 2)
    assert topk_weight.shape == (7, 2)
    assert all(len(set(row.tolist())) == 2 for row in topk_idx)


def test_topk_weights_are_not_renormalized() -> None:
    """deepseek-moe-16b-base's config.json has norm_topk_prob=false."""
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)

    _, topk_weight = moe.route(torch.randn(20, TOY_CONFIG.hidden_size))

    row_sums = topk_weight.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums))


def test_forward_is_routed_plus_shared_experts() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(5, TOY_CONFIG.hidden_size)

    assert moe.shared_experts is not None
    torch.testing.assert_close(
        moe(hidden_states), moe.routed(hidden_states) + moe.shared_experts(hidden_states)
    )


def test_routed_handles_experts_that_receive_zero_tokens() -> None:
    torch.manual_seed(0)
    config = MoEConfig(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=1,
    )
    moe = ReferenceMoE(config)
    hidden_states = torch.randn(2, config.hidden_size)

    topk_idx, _ = moe.route(hidden_states)
    assert len(set(topk_idx.reshape(-1).tolist())) < config.n_routed_experts

    assert moe.routed(hidden_states).shape == hidden_states.shape
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_reference_moe.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dispatch.kernels'`.

- [ ] **Step 3: Write the implementation**

Create empty `src/dispatch/kernels/__init__.py`, then
`src/dispatch/kernels/reference_moe.py`:

```python
"""Pure-PyTorch reference for a DeepSeek-style MoE block -- the CPU-testable
ground truth Phase 1's grouped-GEMM path is checked against. Mirrors the
inference path of deepseek-ai/deepseek-moe-16b-base's DeepseekMoE: softmax
gate, top-k routing, per-expert SiLU-gated MLP, weighted combine, plus
shared experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int


class ExpertMLP(torch.nn.Module):
    """down(silu(gate(x)) * up(x)) -- the per-expert MLP shape DeepseekMLP uses."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class ReferenceMoE(torch.nn.Module):
    """A masked loop over experts: slow, obviously correct, and the oracle."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = torch.nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            ExpertMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_experts = (
            ExpertMLP(config.hidden_size, config.moe_intermediate_size * config.n_shared_experts)
            if config.n_shared_experts > 0
            else None
        )

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k expert ids and their softmax weights, deliberately not
        renormalized: the real model's config.json sets norm_topk_prob=false."""
        scores = F.softmax(self.gate(hidden_states), dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)
        return topk_idx, topk_weight.to(hidden_states.dtype)

    def routed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_weight = self.route(hidden_states)
        combined = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = topk_weight[token_idx, slot_idx].unsqueeze(-1)
            combined.index_add_(0, token_idx, expert(hidden_states[token_idx]) * weight)
        return combined

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.routed(hidden_states)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_reference_moe.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make check-fast`
Expected: clean. `mypy --strict` needs the annotated local in
`ExpertMLP.forward` (`out: torch.Tensor = ...`) because `nn.Linear.__call__`
is typed as returning `Any`; without it, `no-any-return` fires.

- [ ] **Step 6: Update STATUS.md**

In `docs/STATUS.md`, add after the "## Phase 0 progress" section:

```markdown
## Phase 1 progress

Plan: `docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md`. Branch
`phase-1-grouped-gemm`, pushed once as a single PR when the phase is done.

- [x] Task 1: Pure-PyTorch MoE reference (`src/dispatch/kernels/reference_moe.py`)
- [ ] Task 2: Token grouping + tile schedule (`grouping.py`, `tile_schedule.py`)
- [ ] Task 3: Eager grouped MoE path -- the grouped-GEMM contract (`moe_forward.py`)
- [ ] Task 4: Naive Triton grouped-GEMM kernel -- written; GPU-verified in Task 6
- [ ] Task 5: Persistent, cache-aware kernel -- written; GPU-verified in Task 6
- [ ] Task 6: Kernel correctness session on a rented GPU (runbook session A)
- [ ] Task 7: Backend registry + kernel micro-benchmark CLI
- [ ] Task 8: Real-model integration (`--moe-kernel`, `--compare-reference`)
- [ ] Task 9: Measured run on an L40 (runbook session B)
```

Then set "## Next step" to: `Task 2 of
docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md.`

- [ ] **Step 7: Commit**

```bash
git add src/dispatch/kernels/__init__.py src/dispatch/kernels/reference_moe.py \
  tests/unit/test_reference_moe.py docs/STATUS.md
git commit -m "feat: add pure-PyTorch MoE reference for kernel correctness"
```

---

### Task 2: Token grouping and tile scheduling

**Files:**
- Create: `src/dispatch/kernels/grouping.py`
- Create: `src/dispatch/kernels/tile_schedule.py`
- Test: `tests/unit/test_grouping.py`
- Test: `tests/unit/test_tile_schedule.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `TokenGrouping(sorted_token_idx: Tensor, sorted_weight: Tensor,
  group_sizes: Tensor)` (frozen dataclass);
  `group_tokens_by_expert(topk_idx, topk_weight, n_routed_experts) ->
  TokenGrouping`; `ungroup_and_combine(grouped_output, grouping, num_tokens)
  -> Tensor`; `TileSchedule(block_m: int, group_offsets: tuple[int, ...],
  tile_expert: Tensor, tile_row_start: Tensor, tile_valid_rows: Tensor)` with
  a `num_tiles` property; `build_tile_schedule(group_sizes, block_m) ->
  TileSchedule`. Tasks 3, 4, 5, 7 and 8 use these.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_grouping.py`:

```python
"""CPU-only. Hand-computed layouts, so a reader can check each expectation
against the sort-by-expert rule without running anything."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine


def test_group_sizes_count_every_slot_including_an_empty_expert() -> None:
    topk_idx = torch.tensor([[0, 2], [1, 0], [2, 2]])

    grouping = group_tokens_by_expert(topk_idx, torch.ones(3, 2), n_routed_experts=4)

    assert grouping.group_sizes.tolist() == [2, 1, 3, 0]


def test_rows_are_sorted_expert_major_and_stable_within_an_expert() -> None:
    # flat (token, expert) pairs: (0,2) (0,0) (1,0) (1,1) (2,2) (2,1)
    topk_idx = torch.tensor([[2, 0], [0, 1], [2, 1]])
    topk_weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    grouping = group_tokens_by_expert(topk_idx, topk_weight, n_routed_experts=3)

    assert grouping.sorted_token_idx.tolist() == [0, 1, 1, 2, 0, 2]
    torch.testing.assert_close(grouping.sorted_weight, torch.tensor([0.2, 0.3, 0.4, 0.6, 0.1, 0.5]))


def test_ungroup_and_combine_weights_and_sums_rows_per_token() -> None:
    # token 0 -> experts 0 and 1 (weights 2, 3); token 1 -> experts 0 and 1
    # (weights 4, 5).
    topk_idx = torch.tensor([[0, 1], [0, 1]])
    topk_weight = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    grouping = group_tokens_by_expert(topk_idx, topk_weight, n_routed_experts=2)
    # grouped rows: expert 0 = [token 0, token 1], expert 1 = [token 0, token 1]
    grouped_output = torch.tensor([[1.0], [10.0], [100.0], [1000.0]])

    combined = ungroup_and_combine(grouped_output, grouping, num_tokens=2)

    assert combined.tolist() == [[2.0 * 1 + 3.0 * 100], [4.0 * 10 + 5.0 * 1000]]


def test_group_tokens_by_expert_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        group_tokens_by_expert(
            torch.zeros(3, 2, dtype=torch.long), torch.zeros(3, 3), n_routed_experts=4
        )


def test_ungroup_and_combine_rejects_row_count_mismatch() -> None:
    grouping = group_tokens_by_expert(torch.tensor([[0]]), torch.ones(1, 1), n_routed_experts=1)

    with pytest.raises(ValueError, match="rows"):
        ungroup_and_combine(torch.zeros(2, 4), grouping, num_tokens=1)
```

Create `tests/unit/test_tile_schedule.py`:

```python
"""The index arithmetic most likely to hide an off-by-one, tested on CPU
before any kernel reads it."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.tile_schedule import build_tile_schedule


def test_tiles_cover_each_expert_including_an_empty_one() -> None:
    schedule = build_tile_schedule(torch.tensor([5, 0, 3, 9]), block_m=4)

    assert schedule.block_m == 4
    assert schedule.num_tiles == 6
    assert schedule.tile_expert.tolist() == [0, 0, 2, 3, 3, 3]
    assert schedule.tile_valid_rows.tolist() == [4, 1, 3, 4, 4, 1]
    assert schedule.group_offsets == (0, 5, 5, 8, 17)


def test_row_starts_are_globally_contiguous() -> None:
    schedule = build_tile_schedule(torch.tensor([6, 4]), block_m=4)

    assert schedule.tile_row_start.tolist() == [0, 4, 6]
    assert schedule.tile_valid_rows.tolist() == [4, 2, 4]


def test_every_row_is_covered_once_and_no_tile_crosses_an_expert_boundary() -> None:
    group_sizes = torch.tensor([7, 0, 16, 1, 33])
    schedule = build_tile_schedule(group_sizes, block_m=16)

    covered = torch.zeros(int(group_sizes.sum()), dtype=torch.int64)
    for expert, start, valid in zip(
        schedule.tile_expert.tolist(),
        schedule.tile_row_start.tolist(),
        schedule.tile_valid_rows.tolist(),
        strict=True,
    ):
        assert schedule.group_offsets[expert] <= start
        assert start + valid <= schedule.group_offsets[expert + 1]
        covered[start : start + valid] += 1

    assert torch.equal(covered, torch.ones_like(covered))


def test_all_experts_empty_gives_no_tiles() -> None:
    schedule = build_tile_schedule(torch.tensor([0, 0, 0]), block_m=4)

    assert schedule.num_tiles == 0
    assert schedule.group_offsets == (0, 0, 0, 0)


def test_schedule_tensors_are_int32() -> None:
    schedule = build_tile_schedule(torch.tensor([3]), block_m=4)

    assert schedule.tile_expert.dtype == torch.int32
    assert schedule.tile_row_start.dtype == torch.int32
    assert schedule.tile_valid_rows.dtype == torch.int32


def test_rejects_non_positive_block_m() -> None:
    with pytest.raises(ValueError, match="block_m"):
        build_tile_schedule(torch.tensor([4]), block_m=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_grouping.py tests/unit/test_tile_schedule.py -v`
Expected: FAIL, `ModuleNotFoundError` for `dispatch.kernels.grouping`.

- [ ] **Step 3: Write `grouping.py`**

```python
"""Token-to-expert grouping: sorts (token, top-k slot) pairs into
per-expert-contiguous rows so a grouped GEMM can treat each expert's tokens
as one dense matmul -- the same sort-by-expert step DeepseekMoE.moe_infer
performs inline, pulled out as its own tested unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenGrouping:
    """Row i of the grouped layout is token sorted_token_idx[i], weighted by
    sorted_weight[i]; the first group_sizes[0] rows belong to expert 0, and
    so on (a group may be empty)."""

    sorted_token_idx: torch.Tensor  # (num_tokens * top_k,) int64
    sorted_weight: torch.Tensor  # (num_tokens * top_k,)
    group_sizes: torch.Tensor  # (n_routed_experts,) int64


def group_tokens_by_expert(
    topk_idx: torch.Tensor, topk_weight: torch.Tensor, n_routed_experts: int
) -> TokenGrouping:
    if topk_idx.shape != topk_weight.shape:
        raise ValueError(
            f"topk_idx.shape {tuple(topk_idx.shape)} != "
            f"topk_weight.shape {tuple(topk_weight.shape)}"
        )
    num_tokens, top_k = topk_idx.shape
    flat_expert_idx = topk_idx.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=topk_idx.device).repeat_interleave(top_k)
    sort_order = torch.argsort(flat_expert_idx, stable=True)
    return TokenGrouping(
        sorted_token_idx=flat_token_idx[sort_order],
        sorted_weight=topk_weight.reshape(-1)[sort_order],
        group_sizes=torch.bincount(flat_expert_idx, minlength=n_routed_experts),
    )


def ungroup_and_combine(
    grouped_output: torch.Tensor, grouping: TokenGrouping, num_tokens: int
) -> torch.Tensor:
    """Weights each row by its gate weight and sums the top_k rows that
    share a token back into that token's output row."""
    if grouped_output.shape[0] != grouping.sorted_token_idx.shape[0]:
        raise ValueError(
            f"grouped_output has {grouped_output.shape[0]} rows, "
            f"grouping describes {grouping.sorted_token_idx.shape[0]}"
        )
    weighted = grouped_output * grouping.sorted_weight.unsqueeze(-1)
    combined = torch.zeros(
        num_tokens,
        grouped_output.shape[-1],
        dtype=grouped_output.dtype,
        device=grouped_output.device,
    )
    return combined.index_add_(0, grouping.sorted_token_idx, weighted)
```

- [ ] **Step 4: Write `tile_schedule.py`**

```python
"""Host-side tile schedule for the grouped-GEMM kernels: per-expert group
sizes become a flat list of block_m-row tiles, each tagged with its expert,
its first row in the sorted-by-expert input, and how many of its rows are
real (a group's last tile is usually partial). Pure index arithmetic --
testable on CPU with no Triton involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileSchedule:
    block_m: int
    group_offsets: tuple[int, ...]  # host-side row offsets, len n_experts + 1
    tile_expert: torch.Tensor  # (num_tiles,) int32
    tile_row_start: torch.Tensor  # (num_tiles,) int32
    tile_valid_rows: torch.Tensor  # (num_tiles,) int32, each <= block_m

    @property
    def num_tiles(self) -> int:
        return int(self.tile_expert.shape[0])


def build_tile_schedule(group_sizes: torch.Tensor, block_m: int) -> TileSchedule:
    """Tensors land on group_sizes' device, where the kernel will read them."""
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    sizes: list[int] = group_sizes.tolist()
    offsets = [0]
    tile_expert: list[int] = []
    tile_row_start: list[int] = []
    tile_valid_rows: list[int] = []
    for expert_id, size in enumerate(sizes):
        for local_start in range(0, size, block_m):
            tile_expert.append(expert_id)
            tile_row_start.append(offsets[-1] + local_start)
            tile_valid_rows.append(min(block_m, size - local_start))
        offsets.append(offsets[-1] + size)

    return TileSchedule(
        block_m=block_m,
        group_offsets=tuple(offsets),
        tile_expert=_int32(tile_expert, group_sizes.device),
        tile_row_start=_int32(tile_row_start, group_sizes.device),
        tile_valid_rows=_int32(tile_valid_rows, group_sizes.device),
    )


def _int32(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_grouping.py tests/unit/test_tile_schedule.py -v`
Expected: 11 passed.

- [ ] **Step 6: Lint, typecheck, update STATUS.md, commit**

Run: `make check-fast` -- clean. Flip Task 2's box, point "Next step" at
Task 3, then:

```bash
git add src/dispatch/kernels/grouping.py src/dispatch/kernels/tile_schedule.py \
  tests/unit/test_grouping.py tests/unit/test_tile_schedule.py docs/STATUS.md
git commit -m "feat: add token-to-expert grouping and grouped-GEMM tile schedule"
```

---

### Task 3: The grouped-GEMM contract and its eager backend

**Files:**
- Create: `src/dispatch/kernels/moe_forward.py`
- Test: `tests/unit/test_moe_forward.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `group_tokens_by_expert`, `ungroup_and_combine` (Task 2);
  `build_tile_schedule`, `TileSchedule` (Task 2); `ReferenceMoE`,
  `ExpertMLP` (Task 1, tests only).
- Produces: `GroupedMatmul = Callable[[Tensor, Tensor, TileSchedule],
  Tensor]`; `CORRECTNESS_RTOL = 1.6e-2`; `StackedExpertWeights(gate, up,
  down)` with `.num_experts`; `torch_grouped_matmul(x, expert_weight,
  schedule) -> Tensor`; `stack_expert_weights(experts: Iterable[nn.Module])
  -> StackedExpertWeights`; `grouped_moe_routed(x, topk_idx, topk_weight,
  weights, matmul, *, block_m=16) -> Tensor`;
  `assert_matches_reference(actual, expected) -> None`. Tasks 4, 5, 7 and 8
  all use these.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_moe_forward.py`:

```python
"""CPU-only proof of the grouped-GEMM contract: the eager backend over the
grouped layout reproduces ReferenceMoE. The Triton kernels only have to
match torch_grouped_matmul -- everything around the GEMM is proven here."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import (
    assert_matches_reference,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import ExpertMLP, MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
)


def test_torch_grouped_matmul_matches_a_per_row_computation() -> None:
    torch.manual_seed(0)
    group_sizes = torch.tensor([3, 0, 2, 4])
    schedule = build_tile_schedule(group_sizes, block_m=2)
    x = torch.randn(int(group_sizes.sum()), 5)
    weight = torch.randn(4, 6, 5)
    row_expert = torch.repeat_interleave(torch.arange(4), group_sizes).tolist()

    actual = torch_grouped_matmul(x, weight, schedule)

    expected = torch.stack([x[row] @ weight[expert].T for row, expert in enumerate(row_expert)])
    torch.testing.assert_close(actual, expected)


def test_stack_expert_weights_shares_storage_instead_of_copying() -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(3, TOY_CONFIG.hidden_size)
    before = [expert(hidden_states) for expert in moe.experts]

    weights = stack_expert_weights(moe.experts)

    assert weights.gate.shape == (4, 16, 8)
    assert weights.down.shape == (4, 8, 16)
    for expert, output_before in zip(moe.experts, before, strict=True):
        assert isinstance(expert, ExpertMLP)
        assert (
            expert.gate_proj.weight.untyped_storage().data_ptr()
            == weights.gate.untyped_storage().data_ptr()
        )
        assert torch.equal(expert(hidden_states), output_before)


@pytest.mark.parametrize("block_m", [1, 4, 16])
def test_grouped_moe_routed_matches_reference_routed_output(block_m: int) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(7, TOY_CONFIG.hidden_size)
    expected = moe.routed(hidden_states)

    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    actual = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul, block_m=block_m
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_assert_matches_reference_scales_atol_to_the_output() -> None:
    expected = torch.full((4,), 1e-3)

    assert_matches_reference(expected * 1.01, expected)
    with pytest.raises(AssertionError):
        # a flat atol=1e-2 would pass this: every value is off by 100%
        assert_matches_reference(expected * 2, expected)


def test_stack_expert_weights_rejects_a_non_linear_projection() -> None:
    class NotAnExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Identity()

    with pytest.raises(TypeError, match="gate_proj"):
        stack_expert_weights([NotAnExpert()])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_moe_forward.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named
'dispatch.kernels.moe_forward'`.

- [ ] **Step 3: Write the implementation**

Create `src/dispatch/kernels/moe_forward.py`:

```python
"""The routed-expert half of a DeepSeek-style MoE layer, as three grouped
GEMMs over the per-expert-contiguous layout. The GEMM is pluggable:
`torch_grouped_matmul` is the eager-PyTorch implementation of the contract
(one matmul per expert, the pattern DeepseekMoE.moe_infer uses), and the
Triton kernels in grouped_gemm.py are drop-in replacements that must meet
`assert_matches_reference` against it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.kernels.grouping import group_tokens_by_expert, ungroup_and_combine
from dispatch.kernels.tile_schedule import TileSchedule, build_tile_schedule

GroupedMatmul = Callable[[torch.Tensor, torch.Tensor, TileSchedule], torch.Tensor]

CORRECTNESS_RTOL = 1.6e-2  # torch.testing.assert_close's own bf16 default


@dataclass(frozen=True)
class StackedExpertWeights:
    """Each projection's weights for every expert, as (n_experts, N, K)."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.gate.shape[0])


def torch_grouped_matmul(
    x: torch.Tensor, expert_weight: torch.Tensor, schedule: TileSchedule
) -> torch.Tensor:
    """out[rows of expert e] = x[rows of expert e] @ expert_weight[e].T"""
    out = x.new_empty((x.shape[0], expert_weight.shape[1]))
    for expert_id, (start, end) in enumerate(itertools.pairwise(schedule.group_offsets)):
        if end > start:
            out[start:end] = x[start:end] @ expert_weight[expert_id].T
    return out


def stack_expert_weights(experts: Iterable[torch.nn.Module]) -> StackedExpertWeights:
    """Stacks each projection into one tensor and re-points every expert's
    Linear at a view of it, so the stack costs no extra memory."""
    modules = list(experts)
    return StackedExpertWeights(
        gate=_stack_and_share([_linear(module, "gate_proj") for module in modules]),
        up=_stack_and_share([_linear(module, "up_proj") for module in modules]),
        down=_stack_and_share([_linear(module, "down_proj") for module in modules]),
    )


def grouped_moe_routed(  # noqa: PLR0913 -- routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: StackedExpertWeights,
    matmul: GroupedMatmul,
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


def assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """assert_close with atol scaled to the reference's magnitude, so an
    output whose values are all tiny cannot pass on absolute tolerance alone."""
    atol = CORRECTNESS_RTOL * float(expected.abs().max())
    torch.testing.assert_close(actual.float(), expected.float(), rtol=CORRECTNESS_RTOL, atol=atol)


def _linear(module: torch.nn.Module, name: str) -> torch.nn.Linear:
    projection = getattr(module, name)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError(f"expected {name} to be nn.Linear, got {type(projection).__name__}")
    return projection


def _stack_and_share(linears: list[torch.nn.Linear]) -> torch.Tensor:
    stacked = torch.stack([linear.weight.detach() for linear in linears])
    for index, linear in enumerate(linears):
        linear.weight = torch.nn.Parameter(stacked[index], requires_grad=False)
    return stacked
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_moe_forward.py -v`
Expected: 7 passed.

- [ ] **Step 5: Prove the CPU gate can fail**

Temporarily change `ungroup_and_combine` in
`src/dispatch/kernels/grouping.py` to skip the gate weighting:

```python
    weighted = grouped_output  # MUTATION -- revert immediately after this step
```

Run: `uv run pytest -m "not gpu and not slow" -q`
Expected: **5 failed** -- one in `test_grouping.py`, three parametrized
cases in `test_moe_forward.py`, one in `test_integration.py` once Task 8
exists (4 failures before then). Revert the line, rerun, confirm green
again. A test suite that cannot fail is not a gate.

- [ ] **Step 6: Lint, typecheck, update STATUS.md, commit**

```bash
git add src/dispatch/kernels/moe_forward.py tests/unit/test_moe_forward.py docs/STATUS.md
git commit -m "feat: add eager grouped MoE path as the grouped-GEMM contract"
```

---

### Task 4: Naive Triton grouped-GEMM kernel

**Files:**
- Modify: `pyproject.toml` (triton dependency, ruff per-file ignores, mypy override)
- Modify: `uv.lock` (regenerated)
- Create: `src/dispatch/kernels/grouped_gemm.py`
- Test: `tests/unit/test_grouped_gemm_kernel.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `TileSchedule` (Task 2); `assert_matches_reference`,
  `torch_grouped_matmul`, `grouped_moe_routed`, `stack_expert_weights`
  (Task 3, tests).
- Produces: `MIN_BLOCK_M = 16`, `DEFAULT_BLOCK_N = 64`, `DEFAULT_BLOCK_K =
  64`; `grouped_matmul(x, expert_weight, schedule, *, block_n=DEFAULT_BLOCK_N,
  block_k=DEFAULT_BLOCK_K) -> Tensor`, which satisfies `GroupedMatmul`.
  Tasks 5, 7 and 8 use it.

**This task's tests cannot go green on the authoring machine.** Triton has
no macOS wheels, so the GPU file skips locally; the red-to-green cycle for
the kernel itself happens on rented hardware in Task 6. What *must* be
green here: lint, mypy, and the rest of the suite.

- [ ] **Step 1: Add the dependency and the tool config**

In `pyproject.toml`, extend `dependencies`:

```toml
dependencies = [
    "requests>=2.34.2",
    "torch>=2.14.0",
    "transformers>=5.17.0",
    "accelerate>=1.15.0",
    "huggingface_hub>=1.31.0",
    "safetensors>=0.8.0",
    # Linux-only: triton publishes manylinux wheels and nothing else.
    "triton>=3.8.0; sys_platform == 'linux'",
]
```

Extend the ruff per-file ignores:

```toml
[tool.ruff.lint.per-file-ignores]
# Comparing against expected literals is normal and readable in tests.
"tests/**/*.py" = ["PLR2004", "PLR0913"]
# Triton kernel signatures are pointer/stride/tile-size lists by construction,
# and Triton's convention is UPPER_CASE for tl.constexpr arguments.
"src/dispatch/kernels/grouped_gemm.py" = ["N803", "PLR0913", "PLR0917"]
```

And add the mypy override after the `[tool.mypy]` block:

```toml
# Triton isn't installed at all on macOS. Skipping it makes mypy treat it as
# Any everywhere, so the Mac and Linux CI give the same answer.
[[tool.mypy.overrides]]
module = ["triton", "triton.*"]
ignore_missing_imports = true
follow_imports = "skip"
```

Run: `uv lock && uv sync --all-extras --dev`
Expected: resolves; `uv.lock` gains `triton 3.8.0` with the marker
`sys_platform == 'linux'`, and nothing is installed on macOS. Confirm with
`uv run python -c "import importlib.util;
print(importlib.util.find_spec('triton'))"` -- prints `None` on a Mac.

- [ ] **Step 2: Write the failing GPU test**

Create `tests/unit/test_grouped_gemm_kernel.py`:

```python
"""GPU-only correctness gate for the Triton kernels: each must meet
assert_matches_reference against torch_grouped_matmul (itself proven
against ReferenceMoE on CPU in test_moe_forward.py) before any benchmark
number means anything.

Skipped, with the reason shown, where triton can't be imported (it ships
Linux wheels only); where triton imports, a missing or broken grouped_gemm
module errors loudly instead of skipping. Excluded from CI by the `gpu`
marker.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.moe_forward import (
    assert_matches_reference,
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE
from dispatch.kernels.tile_schedule import build_tile_schedule

pytest.importorskip("triton", reason="triton ships Linux wheels only")

from dispatch.kernels import grouped_gemm

pytestmark = pytest.mark.gpu

SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="bf16 tensor-core matmul needs compute capability 8.0+ (Ampere or newer)",
)
DTYPES = [torch.float16, pytest.param(torch.bfloat16, marks=SM80)]
KERNELS = ["grouped_matmul"]

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


@pytest.mark.parametrize("kernel_name", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("group_sizes", "n", "k"),
    [
        ([5, 0, 3, 40], 24, 16),  # toy dims, an empty expert, a multi-tile expert
        ([1, 1, 1, 1, 1, 1], 1408, 2048),  # decode-shaped, gate/up_proj dims
        ([70, 0, 33, 129], 2048, 1408),  # prefill-shaped, down_proj dims, partial tiles
    ],
)
def test_kernel_matches_fp32_reference(
    kernel_name: str, dtype: torch.dtype, group_sizes: list[int], n: int, k: int
) -> None:
    torch.manual_seed(0)
    sizes = torch.tensor(group_sizes, device="cuda")
    schedule = build_tile_schedule(sizes, block_m=16)
    x = torch.randn(int(sizes.sum()), k, device="cuda", dtype=dtype)
    weight = torch.randn(len(group_sizes), n, k, device="cuda", dtype=dtype) / k**0.5

    actual = getattr(grouped_gemm, kernel_name)(x, weight, schedule)

    assert_matches_reference(actual, torch_grouped_matmul(x.float(), weight.float(), schedule))


@pytest.mark.parametrize("kernel_name", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("config", "num_tokens"), [(TOY_DIMS, 37), (REAL_DIMS, 1), (REAL_DIMS, 256)]
)
def test_kernel_backed_moe_layer_matches_reference_moe(
    kernel_name: str, dtype: torch.dtype, config: MoEConfig, num_tokens: int
) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(config).to(device="cuda", dtype=dtype)
    hidden_states = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=dtype)
    expected = moe.routed(hidden_states)

    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    actual = grouped_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, getattr(grouped_gemm, kernel_name)
    )

    assert_matches_reference(actual, expected)


def test_rejects_a_schedule_below_tl_dots_minimum_block_m() -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 16, 16, device="cuda", dtype=torch.float16)
    schedule = build_tile_schedule(torch.tensor([4], device="cuda"), block_m=4)

    with pytest.raises(ValueError, match="block_m"):
        grouped_gemm.grouped_matmul(x, weight, schedule)
```

Note the output buffer is `torch.empty`, not `torch.zeros` (Step 3): any
tile the schedule fails to cover shows up as garbage rather than a
plausible zero, which is what makes these tests sensitive to coverage bugs.

- [ ] **Step 3: Run the test to see what it does on this machine**

Run: `uv run pytest tests/unit/test_grouped_gemm_kernel.py -rs`
Expected on macOS: `1 skipped`, reason `triton ships Linux wheels only`.
On a Linux GPU host, the same command at this point errors at collection
with `ModuleNotFoundError: No module named
'dispatch.kernels.grouped_gemm'` -- that is this task's red half, and
Task 6 is where it gets observed for real.

- [ ] **Step 4: Write the kernel**

Create `src/dispatch/kernels/grouped_gemm.py`:

```python
"""Triton grouped-GEMM kernels for MoE expert computation: drop-in
GroupedMatmul backends for dispatch.kernels.moe_forward, each held to
assert_matches_reference against its eager torch_grouped_matmul.

Adapted from Triton's official Group GEMM tutorial (08-grouped-gemm) for
MoE's shape of the problem: every group is a contiguous slice of one
sorted-by-expert tensor and every expert's weight shares one (N, K) shape,
so a host-built TileSchedule stands in for the tutorial's per-group
pointer arrays.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dispatch.kernels.tile_schedule import TileSchedule

MIN_BLOCK_M = 16  # tl.dot's smallest tile dimension
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64


@triton.jit  # type: ignore[untyped-decorator]
def _matmul_tile(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
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
    col_mask = offs_n[None, :] < n

    x_ptrs = x_ptr + (row_start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        x_tile = tl.load(x_ptrs, mask=row_mask & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + (row_start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_mask & col_mask)


@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
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
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _matmul_tile(
        x_ptr,
        w_ptr,
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
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def grouped_matmul(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> torch.Tensor:
    """One CTA per (m_tile, n_tile) -- the simple, correctness-first launch."""
    out = _validated_output(x, expert_weight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = expert_weight.shape[1], expert_weight.shape[2]
    grid = (schedule.num_tiles, triton.cdiv(n, block_n))
    _grouped_matmul_kernel[grid](
        x,
        expert_weight,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        n,
        k,
        *x.stride(),
        *expert_weight.stride(),
        *out.stride(),
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _validated_output(
    x: torch.Tensor, expert_weight: torch.Tensor, schedule: TileSchedule
) -> torch.Tensor:
    if not (x.is_cuda and expert_weight.is_cuda):
        raise ValueError("the Triton grouped-GEMM kernels need CUDA tensors")
    if x.dtype != expert_weight.dtype:
        raise ValueError(f"dtype mismatch: x is {x.dtype}, expert_weight is {expert_weight.dtype}")
    if x.shape[1] != expert_weight.shape[2]:
        raise ValueError(
            f"K mismatch: x has {x.shape[1]}, expert_weight has {expert_weight.shape[2]}"
        )
    if schedule.block_m < MIN_BLOCK_M:
        raise ValueError(
            f"schedule.block_m={schedule.block_m} is below tl.dot's minimum of {MIN_BLOCK_M}"
        )
    return torch.empty((x.shape[0], expert_weight.shape[1]), device=x.device, dtype=x.dtype)
```

- [ ] **Step 5: Lint, typecheck, run everything that can run here**

Run: `make check`
Expected: ruff clean, `mypy --strict` clean, and the non-GPU suite green
with `test_grouped_gemm_kernel.py` skipped.

- [ ] **Step 6: Update STATUS.md and commit**

Flip Task 4's box to `[x]` and leave its "GPU-verified in Task 6" note.

```bash
git add pyproject.toml uv.lock src/dispatch/kernels/grouped_gemm.py \
  tests/unit/test_grouped_gemm_kernel.py docs/STATUS.md
git commit -m "feat: add naive Triton grouped-GEMM kernel"
```

---

### Task 5: Persistent, cache-aware kernel

**Files:**
- Modify: `src/dispatch/kernels/grouped_gemm.py`
- Modify: `tests/unit/test_grouped_gemm_kernel.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `DEFAULT_GROUP_SIZE_M = 8`; `grouped_matmul_persistent(x,
  expert_weight, schedule, *, block_n=DEFAULT_BLOCK_N,
  block_k=DEFAULT_BLOCK_K, group_size_m=DEFAULT_GROUP_SIZE_M) -> Tensor`,
  satisfying `GroupedMatmul`. Tasks 7 and 8 use it.

Two changes from the naive kernel, both from PyTorch's engineering post
"Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM
Kernel" (2025-08-19): launch exactly one CTA per SM and let each loop over
its share of tiles (`for tile_id in tl.range(start_pid, num_tiles,
NUM_SMS)`), and walk the tiles in grouped order rather than row-major, so
consecutive programs reuse the same weight tile while a band of input rows
stays in L2. The tile arithmetic is shared with the naive kernel through
`_matmul_tile`, so this task changes scheduling only -- which is exactly
what the already-written correctness tests re-check.

- [ ] **Step 1: Extend the test matrix**

In `tests/unit/test_grouped_gemm_kernel.py`, change one line:

```python
KERNELS = ["grouped_matmul", "grouped_matmul_persistent"]
```

- [ ] **Step 2: Run to confirm the new cases fail for the right reason**

Run: `uv run pytest tests/unit/test_grouped_gemm_kernel.py -rs`
Expected on macOS: still `1 skipped`. On a GPU host, the new parametrized
cases fail with `AttributeError: module 'dispatch.kernels.grouped_gemm' has
no attribute 'grouped_matmul_persistent'`.

- [ ] **Step 3: Add the constant**

In `src/dispatch/kernels/grouped_gemm.py`, after `DEFAULT_BLOCK_K = 64`:

```python
DEFAULT_GROUP_SIZE_M = 8
```

- [ ] **Step 4: Add the kernel**

Insert after `_grouped_matmul_kernel`:

```python
@triton.jit  # type: ignore[untyped-decorator]
def _grouped_matmul_persistent_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    out_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_valid_rows_ptr,
    num_m_tiles,
    n,
    k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_n_tiles = tl.cdiv(n, BLOCK_N)
    tiles_per_group = GROUP_SIZE_M * num_n_tiles
    for tile_id in tl.range(tl.program_id(axis=0), num_m_tiles * num_n_tiles, NUM_SMS):
        first_m_tile = (tile_id // tiles_per_group) * GROUP_SIZE_M
        group_rows = tl.minimum(num_m_tiles - first_m_tile, GROUP_SIZE_M)
        m_tile = first_m_tile + (tile_id % tiles_per_group) % group_rows
        n_tile = (tile_id % tiles_per_group) // group_rows
        _matmul_tile(
            x_ptr,
            w_ptr,
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
            stride_om,
            stride_on,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
```

- [ ] **Step 5: Add the wrapper**

Insert after `grouped_matmul`:

```python
def grouped_matmul_persistent(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    schedule: TileSchedule,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    group_size_m: int = DEFAULT_GROUP_SIZE_M,
) -> torch.Tensor:
    """One long-lived CTA per SM, each looping over (m_tile, n_tile) work in
    L2-friendly grouped order instead of one launch-slot per tile."""
    out = _validated_output(x, expert_weight, schedule)
    if schedule.num_tiles == 0:
        return out
    n, k = expert_weight.shape[1], expert_weight.shape[2]
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    _grouped_matmul_persistent_kernel[(num_sms,)](
        x,
        expert_weight,
        out,
        schedule.tile_expert,
        schedule.tile_row_start,
        schedule.tile_valid_rows,
        schedule.num_tiles,
        n,
        k,
        *x.stride(),
        *expert_weight.stride(),
        *out.stride(),
        NUM_SMS=num_sms,
        BLOCK_M=schedule.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
    )
    return out
```

The grid is always `NUM_SMS` programs, even when there is less work than
that: programs with nothing to do exit the loop immediately, and keeping
the launch shape constant keeps `NUM_SMS` a stable `tl.constexpr` instead
of forcing a recompile for every new tile count at decode time.

- [ ] **Step 6: Lint, typecheck, run what can run, update STATUS.md, commit**

Run: `make check` -- clean, GPU file still skipped locally.

```bash
git add src/dispatch/kernels/grouped_gemm.py tests/unit/test_grouped_gemm_kernel.py docs/STATUS.md
git commit -m "feat: add persistent cache-aware Triton grouped-GEMM kernel"
```

---

### Task 6: Kernel correctness session on a rented GPU

**Files:**
- Modify: `scripts/gpu/provision.py` (`write_cost_record` gains `run_label`)
- Modify: `tests/unit/test_provision.py`
- Create: `docs/runbooks/phase-1-grouped-gemm.md` (session A)
- Create: `docs/findings/<date>-phase-1-kernel-correctness.md`
- Create: `docs/findings/<date>-phase-1-kernel-correctness-cost.md` (written by the tool)
- Modify: `src/dispatch/kernels/grouped_gemm.py` (only if the session finds bugs)
- Modify: `docs/STATUS.md`

**Interfaces:**
- Modifies: `write_cost_record(findings_dir, *, pod_id, gpu_type_id,
  cost_per_hour, duration_s, note, run_label="phase-0-baseline", now=None)
  -> Path` -- the filename becomes `<date>-<run_label>-cost.md`. Task 9
  uses it too.

**This task spends real money.** It needs the user's explicit go-ahead and
a stated budget cap before any pod exists. Synthetic weights only -- no
model download -- so it wants the cheapest card Triton supports, not an L40.

- [ ] **Step 1: Let cost records name their run (local, TDD)**

Add to `tests/unit/test_provision.py`, after the existing
`test_write_cost_record_computes_cost_and_writes_file`:

```python
def test_write_cost_record_names_the_file_for_its_run(tmp_path: Path) -> None:
    path = write_cost_record(
        tmp_path,
        pod_id="pod_9",
        gpu_type_id="NVIDIA L40",
        cost_per_hour=0.82,
        duration_s=1800.0,
        note="phase 1 kernel run",
        run_label="phase-1-grouped-gemm",
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )

    assert path.name == "2026-09-20-phase-1-grouped-gemm-cost.md"
    assert path.read_text().startswith("# phase-1-grouped-gemm -- GPU rental cost")
```

Run it (expect FAIL: unexpected keyword argument `run_label`), then in
`scripts/gpu/provision.py` add the parameter and use it:

```python
    note: str,
    run_label: str = "phase-0-baseline",
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    cost_usd = cost_per_hour * (duration_s / 3600)
    findings_dir.mkdir(parents=True, exist_ok=True)
    path = findings_dir / f"{now:%Y-%m-%d}-{run_label}-cost.md"
    path.write_text(
        f"# {run_label} -- GPU rental cost\n\n"
```

Run: `uv run pytest tests/unit/test_provision.py -v`
Expected: 8 passed.

- [ ] **Step 2: Write the runbook's session A**

Create `docs/runbooks/phase-1-grouped-gemm.md`:

```markdown
# Runbook: Phase 1 grouped-GEMM (rented GPU)

Two paid sessions. Both need the user's explicit go-ahead and a budget cap
stated before a pod exists. Tasks 1-5, 7 and 8 cost nothing.

## Session A -- kernel correctness (Task 6)

Synthetic weights only; no model download; any GPU Triton supports.

1. Pick the cheapest Community Cloud card with compute capability 8.0+ --
   Triton's README supports "NVIDIA GPUs (Compute Capability 8.0+)", so an
   Ampere-or-newer part (RTX A4000/A5000/A6000, RTX 3090/4090, A40, L4,
   L40, A100). A T4 is 7.5 and a P100 is 6.0: neither qualifies. Query the
   live catalog the same way `docs/runbooks/phase-0-baseline.md` step 1
   does, or use the RunPod MCP plugin, which is what Phase 0 actually used.
2. Create and wait:

       uv run python -m scripts.gpu.provision create --name dispatch-phase-1-kernels \
         --gpu-type "<id from step 1>" --image "<current runpod/pytorch tag>" --cloud COMMUNITY
       uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

3. Confirm the card before spending time on it:

       nvidia-smi --query-gpu=name,compute_cap --format=csv

   Below 8.0: terminate now and pick another.
4. Transfer the committed branch without pushing it (this repo pushes a
   phase once, at the end -- Phase 0 used the same trick):

       git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"

5. On the pod:

       cd dispatch
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv run python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
       uv run pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs

6. Iterate: edit locally (the Mac stays the source of truth), re-sync, rerun:

       command -v rsync || apt-get install -y rsync   # on the pod, once
       rsync -az --exclude .venv --exclude .git --exclude '*.safetensors' ./ <pod>:dispatch/

7. Mutation check, once green -- prove the suite can fail. In
   `_matmul_tile`, temporarily make every tile read expert 0's weights:

       expert_id = tl.load(tile_expert_ptr + m_tile) * 0

   Re-sync, rerun: every multi-expert case must fail. Revert, re-sync,
   rerun: green again.
8. Record the cost, terminate, and verify termination independently:

       uv run python -c "
       from pathlib import Path
       from scripts.gpu.provision import write_cost_record
       from scripts.gpu.runpod_client import get_pod
       pod = get_pod('<pod_id>')
       write_cost_record(
           Path('docs/findings'),
           pod_id=pod.id,
           gpu_type_id='<id from step 1>',
           cost_per_hour=pod.cost_per_hour,
           duration_s=<measured wall-clock seconds RUNNING>,
           note='Phase 1 kernel correctness, synthetic weights',
           run_label='phase-1-kernel-correctness',
       )
       "
       uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>

   Then re-read `get_pod('<pod_id>').status` and confirm `TERMINATED`
   rather than trusting the terminate command's exit code.
```

- [ ] **Step 3: Get the go-ahead**

State the intended card, its hourly rate, and a cap. Do not create a pod
without an explicit yes.

- [ ] **Step 4: Run the session**

Follow session A, steps 1-6.
Expected when the kernels are right: **25 passed** (2 kernels x 2 dtypes x
3 shapes, twice, plus the block_m guard), 0 skipped on a card with compute
capability 8.0+.

- [ ] **Step 5: When something fails -- and something probably will**

Discipline, in order:
1. Re-run the single smallest failing parametrization (`-k` the toy-dims
   case). Toy dims fail fast and print small tensors.
2. Compare against `torch_grouped_matmul` on that exact input in a REPL on
   the pod; look at *which* rows and columns differ. All rows of one
   expert wrong points at expert indexing; the tail rows of a group wrong
   points at `tile_valid_rows` masking; the last columns wrong points at
   `col_mask`; a whole tile of garbage points at schedule coverage.
3. Fix the kernel. **Never** widen a tolerance to make this go green --
   see Global Constraints.
4. Each fix is its own commit, with its own STATUS.md line, once green:
   `git commit -m "fix: <what was wrong in the kernel>"`.

- [ ] **Step 6: Mutation check on real hardware**

Session A step 7. Record what failed and what passed after reverting.

- [ ] **Step 7: Tear down and record cost**

Session A step 8. Verify `TERMINATED` twice, as Phase 0's runbook requires.

- [ ] **Step 8: Write the finding**

Create `docs/findings/<date>-phase-1-kernel-correctness.md`: the card and
its compute capability, torch/triton versions, how many gpu tests ran and
passed, every bug found and its fix, the mutation-check result, the
measured cost, and confirmation the pod is terminated. Numbers measured,
never estimated.

- [ ] **Step 9: Update STATUS.md and commit**

Flip Tasks 4, 5 and 6 to fully done (Tasks 4-5 lose their "GPU-verified in
Task 6" caveat).

```bash
git add scripts/gpu/provision.py tests/unit/test_provision.py \
  docs/runbooks/phase-1-grouped-gemm.md docs/findings/ docs/STATUS.md
git commit -m "test: verify both Triton kernels on a rented GPU"
```

---

### Task 7: Backend registry and kernel micro-benchmark

**Files:**
- Create: `src/dispatch/kernels/backends.py`
- Create: `src/dispatch/kernels/bench.py`
- Create: `scripts/run_kernel_bench.py`
- Test: `tests/unit/test_backends.py`
- Test: `tests/unit/test_kernel_bench.py`
- Test: `tests/unit/test_run_kernel_bench.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `BACKENDS = ("torch", "naive", "persistent")`;
  `resolve_backend(name) -> GroupedMatmul`; `KernelBenchmarkSummary(label,
  mean_latency_ms, p50_latency_ms, p99_latency_ms, tflops)`;
  `grouped_gemm_flops(total_rows, n, k) -> float`;
  `summarize_kernel_latencies(label, latencies_ms, *, flops) ->
  KernelBenchmarkSummary`; `sample_topk_idx(num_tokens, n_experts, top_k, *,
  distribution, generator) -> Tensor`; `ROUTING_DISTRIBUTIONS`;
  `DO_BENCH_WARMUP_MS = 25`, `DO_BENCH_REP_MS = 100`;
  `time_grouped_gemm(fn, *, label, flops) -> KernelBenchmarkSummary`;
  `run_kernel_bench(...)`, `describe_run_environment()`, `main(argv)`.
  Task 8 imports `BACKENDS` and `resolve_backend`.

- [ ] **Step 1: Write `tests/unit/test_backends.py` (failing)**

```python
"""CPU-only: the Triton names resolve against a stand-in module, so the
mapping itself is tested everywhere, not only on a GPU host."""

from __future__ import annotations

import sys
import types

import pytest

import dispatch.kernels
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.moe_forward import torch_grouped_matmul


def test_torch_backend_is_the_eager_contract() -> None:
    assert resolve_backend("torch") is torch_grouped_matmul


def test_triton_backend_names_map_to_their_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in = types.ModuleType("dispatch.kernels.grouped_gemm")
    stand_in.grouped_matmul = object()  # type: ignore[attr-defined]
    stand_in.grouped_matmul_persistent = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dispatch.kernels.grouped_gemm", stand_in)
    monkeypatch.setattr(dispatch.kernels, "grouped_gemm", stand_in, raising=False)

    assert resolve_backend("naive") is stand_in.grouped_matmul
    assert resolve_backend("persistent") is stand_in.grouped_matmul_persistent


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("cutlass")
```

Both the `sys.modules` entry and the package attribute are patched on
purpose: on Linux CI the real module may already be imported and attached
to the package by the GPU test file's collection, and only the attribute
patch shadows that.

- [ ] **Step 2: Write `src/dispatch/kernels/backends.py`**

```python
"""Backend name -> GroupedMatmul. The Triton backends import lazily, so this
module -- and everything that only needs the eager backend -- stays
importable where triton isn't installed."""

from __future__ import annotations

from dispatch.kernels.moe_forward import GroupedMatmul, torch_grouped_matmul

BACKENDS = ("torch", "naive", "persistent")


def resolve_backend(name: str) -> GroupedMatmul:
    if name == "torch":
        return torch_grouped_matmul
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    from dispatch.kernels import grouped_gemm  # noqa: PLC0415 -- triton is Linux-only

    return (
        grouped_gemm.grouped_matmul if name == "naive" else grouped_gemm.grouped_matmul_persistent
    )
```

- [ ] **Step 3: Write `tests/unit/test_kernel_bench.py` (failing)**

```python
"""Pure math and synthetic routing -- no GPU, and no triton import (the one
function that needs triton imports it locally)."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.bench import grouped_gemm_flops, sample_topk_idx, summarize_kernel_latencies


def test_grouped_gemm_flops_is_2mnk() -> None:
    assert grouped_gemm_flops(total_rows=10, n=20, k=30) == 2 * 10 * 20 * 30


def test_grouped_gemm_flops_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="invalid dims"):
        grouped_gemm_flops(total_rows=-1, n=1, k=1)


def test_summarize_kernel_latencies_computes_tflops_from_the_mean() -> None:
    summary = summarize_kernel_latencies("naive/gemm", (1.0, 0.9, 1.5), flops=2e9)

    assert summary.label == "naive/gemm"
    assert (summary.mean_latency_ms, summary.p50_latency_ms, summary.p99_latency_ms) == (
        1.0,
        0.9,
        1.5,
    )
    assert summary.tflops == pytest.approx(2.0)


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
def test_sample_topk_idx_gives_distinct_in_range_experts(distribution: str) -> None:
    topk_idx = sample_topk_idx(
        50, 64, 6, distribution=distribution, generator=torch.Generator().manual_seed(0)
    )

    assert topk_idx.shape == (50, 6)
    assert int(topk_idx.min()) >= 0
    assert int(topk_idx.max()) < 64
    assert all(len(set(row.tolist())) == 6 for row in topk_idx)


def test_zipf_routing_is_skewed_toward_low_index_experts() -> None:
    topk_idx = sample_topk_idx(
        2000, 64, 6, distribution="zipf", generator=torch.Generator().manual_seed(0)
    )

    counts = torch.bincount(topk_idx.reshape(-1), minlength=64)
    assert int(counts[:8].sum()) > int(counts[-8:].sum())


def test_sample_topk_idx_rejects_unknown_distribution() -> None:
    with pytest.raises(ValueError, match="unknown distribution"):
        sample_topk_idx(1, 4, 1, distribution="pareto", generator=torch.Generator())
```

- [ ] **Step 4: Write `src/dispatch/kernels/bench.py`**

```python
"""Kernel micro-benchmark helpers. Timing goes through
triton.testing.do_bench -- warmup, repetition, and GPU synchronization are
Triton's own tool's job, not a bespoke perf_counter loop -- and everything
else here is pure math, testable without a GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

DO_BENCH_WARMUP_MS = 25
DO_BENCH_REP_MS = 100

_ROUTING_WEIGHTS: dict[str, Callable[[int], torch.Tensor]] = {
    "uniform": torch.ones,
    "zipf": lambda n_experts: 1.0 / torch.arange(1, n_experts + 1, dtype=torch.float32),
}
ROUTING_DISTRIBUTIONS = tuple(_ROUTING_WEIGHTS)


@dataclass(frozen=True)
class KernelBenchmarkSummary:
    label: str
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    tflops: float


def grouped_gemm_flops(total_rows: int, n: int, k: int) -> float:
    """2*M*N*K; every group shares N and K, so only the total row count matters."""
    if total_rows < 0 or n <= 0 or k <= 0:
        raise ValueError(f"invalid dims: total_rows={total_rows}, n={n}, k={k}")
    return 2.0 * total_rows * n * k


def summarize_kernel_latencies(
    label: str, latencies_ms: tuple[float, float, float], *, flops: float
) -> KernelBenchmarkSummary:
    """latencies_ms is (mean, p50, p99), as do_bench reports them."""
    mean_ms, p50_ms, p99_ms = latencies_ms
    return KernelBenchmarkSummary(
        label=label,
        mean_latency_ms=mean_ms,
        p50_latency_ms=p50_ms,
        p99_latency_ms=p99_ms,
        tflops=flops / (mean_ms * 1e-3) / 1e12,
    )


def sample_topk_idx(
    num_tokens: int,
    n_experts: int,
    top_k: int,
    *,
    distribution: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Synthetic routing: `zipf` skews load toward low-index experts,
    `uniform` balances it. Each row holds top_k distinct experts."""
    if distribution not in _ROUTING_WEIGHTS:
        raise ValueError(
            f"unknown distribution {distribution!r}; expected one of {ROUTING_DISTRIBUTIONS}"
        )
    probs = _ROUTING_WEIGHTS[distribution](n_experts).expand(num_tokens, n_experts)
    return torch.multinomial(probs, top_k, replacement=False, generator=generator)


def time_grouped_gemm(
    fn: Callable[[], torch.Tensor], *, label: str, flops: float
) -> KernelBenchmarkSummary:
    import triton.testing  # noqa: PLC0415 -- keeps this module importable without triton

    mean_ms = float(
        triton.testing.do_bench(
            fn, warmup=DO_BENCH_WARMUP_MS, rep=DO_BENCH_REP_MS, return_mode="mean"
        )
    )
    p50_ms, p99_ms = (
        float(value)
        for value in triton.testing.do_bench(
            fn, warmup=DO_BENCH_WARMUP_MS, rep=DO_BENCH_REP_MS, quantiles=[0.5, 0.99]
        )
    )
    return summarize_kernel_latencies(label, (mean_ms, p50_ms, p99_ms), flops=flops)
```

- [ ] **Step 5: Write `tests/unit/test_run_kernel_bench.py` (failing)**

```python
"""main()'s plumbing (args -> JSON) with the GPU-only pieces monkeypatched
out; the real timing runs in the Task 9 runbook, on a GPU host."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_kernel_bench as bench_module
from scripts.run_kernel_bench import main

from dispatch.kernels.bench import KernelBenchmarkSummary


def test_main_writes_config_and_one_result_per_token_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = KernelBenchmarkSummary(
        label="naive/layer", mean_latency_ms=1.0, p50_latency_ms=1.0, p99_latency_ms=1.2, tflops=3.0
    )
    calls: list[int] = []

    def fake_run_kernel_bench(*, num_tokens: int, **kwargs: object) -> list[KernelBenchmarkSummary]:
        calls.append(num_tokens)
        return [summary]

    monkeypatch.setattr(bench_module, "run_kernel_bench", fake_run_kernel_bench)
    monkeypatch.setattr(bench_module, "describe_run_environment", lambda: {"gpu": "fake-gpu"})

    main(
        [
            "--num-tokens",
            "1",
            "16",
            "--dtype",
            "float16",
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "bench-test",
        ]
    )

    record = json.loads((tmp_path / "bench-test.json").read_text())
    assert calls == [1, 16]
    assert record["config"]["gpu"] == "fake-gpu"
    assert record["config"]["dtype"] == "float16"
    assert record["config"]["hidden_size"] == 2048
    assert [result["num_tokens"] for result in record["results"]] == [1, 16]
    assert record["results"][0]["summaries"][0]["label"] == "naive/layer"
```

- [ ] **Step 6: Write `scripts/run_kernel_bench.py`**

```python
"""CLI: time the grouped-GEMM backends -- eager torch loop, naive Triton,
persistent Triton -- on synthetic DeepSeekMoE-16B-shaped routed-expert work,
and write every summary with its full config to docs/findings/. A backend
that disagrees with the eager torch backend on the benchmark's own input is
refused, not timed.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dispatch.kernels.backends import BACKENDS, resolve_backend
from dispatch.kernels.bench import (
    DO_BENCH_REP_MS,
    DO_BENCH_WARMUP_MS,
    ROUTING_DISTRIBUTIONS,
    KernelBenchmarkSummary,
    grouped_gemm_flops,
    sample_topk_idx,
    time_grouped_gemm,
)
from dispatch.kernels.grouping import group_tokens_by_expert
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    assert_matches_reference,
    grouped_moe_routed,
)
from dispatch.kernels.tile_schedule import build_tile_schedule

# deepseek-ai/deepseek-moe-16b-base config.json, checked live 2026-09-15.
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 1408
N_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 6


def run_kernel_bench(
    *, num_tokens: int, distribution: str, dtype: torch.dtype, block_m: int, seed: int
) -> list[KernelBenchmarkSummary]:
    """Per backend: the whole routed MoE layer (grouping and scheduling
    overhead included), and one gate_proj-shaped grouped GEMM on its own."""
    cuda = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=dtype, generator=cuda)
    weights = StackedExpertWeights(
        gate=_expert_weights(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype, cuda),
        up=_expert_weights(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype, cuda),
        down=_expert_weights(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype, cuda),
    )
    topk_idx = sample_topk_idx(
        num_tokens,
        N_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        distribution=distribution,
        generator=torch.Generator().manual_seed(seed),
    ).cuda()
    topk_weight = torch.rand(
        num_tokens, NUM_EXPERTS_PER_TOK, device="cuda", dtype=dtype, generator=cuda
    )
    grouping = group_tokens_by_expert(topk_idx, topk_weight, N_ROUTED_EXPERTS)
    schedule = build_tile_schedule(grouping.group_sizes, block_m)
    gathered = x[grouping.sorted_token_idx]
    gemm_flops = grouped_gemm_flops(
        num_tokens * NUM_EXPERTS_PER_TOK, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE
    )

    expected: torch.Tensor | None = None
    summaries: list[KernelBenchmarkSummary] = []
    for name in BACKENDS:
        matmul = resolve_backend(name)
        layer = functools.partial(
            grouped_moe_routed, x, topk_idx, topk_weight, weights, matmul, block_m=block_m
        )
        if expected is None:
            expected = layer()
        else:
            _require_agreement(layer(), expected, name)
        gemm = functools.partial(matmul, gathered, weights.gate, schedule)
        summaries.append(time_grouped_gemm(layer, label=f"{name}/layer", flops=3 * gemm_flops))
        summaries.append(time_grouped_gemm(gemm, label=f"{name}/gemm", flops=gemm_flops))
    return summaries


def describe_run_environment() -> dict[str, Any]:
    """Hardware, library versions, and the kernels' fixed tile sizes -- the
    config a benchmark number means nothing without."""
    import triton  # noqa: PLC0415 -- Linux-only; this CLI only runs on a GPU host

    from dispatch.kernels import grouped_gemm  # noqa: PLC0415

    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "block_n": grouped_gemm.DEFAULT_BLOCK_N,
        "block_k": grouped_gemm.DEFAULT_BLOCK_K,
        "group_size_m": grouped_gemm.DEFAULT_GROUP_SIZE_M,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Time dispatch's grouped-GEMM backends")
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 16, 128, 512, 2048])
    parser.add_argument("--distribution", choices=ROUTING_DISTRIBUTIONS, default="zipf")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-1-kernel-bench"))
    args = parser.parse_args(argv)

    results = [
        {
            "num_tokens": num_tokens,
            "summaries": [
                asdict(summary)
                for summary in run_kernel_bench(
                    num_tokens=num_tokens,
                    distribution=args.distribution,
                    dtype=getattr(torch, args.dtype),
                    block_m=args.block_m,
                    seed=args.seed,
                )
            ],
        }
        for num_tokens in args.num_tokens
    ]
    record = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "distribution": args.distribution,
            "dtype": args.dtype,
            "block_m": args.block_m,
            "seed": args.seed,
            "do_bench_warmup_ms": DO_BENCH_WARMUP_MS,
            "do_bench_rep_ms": DO_BENCH_REP_MS,
            **describe_run_environment(),
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.run_label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


def _expert_weights(n: int, k: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    weights = torch.randn(N_ROUTED_EXPERTS, n, k, device="cuda", dtype=dtype, generator=generator)
    return weights / math.sqrt(k)


def _require_agreement(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    try:
        assert_matches_reference(actual, expected)
    except AssertionError as exc:
        raise RuntimeError(
            f"backend {name!r} disagrees with the eager torch backend -- refusing to time it"
        ) from exc


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests, lint, typecheck, update STATUS.md, commit**

Run: `uv run pytest tests/unit/test_backends.py tests/unit/test_kernel_bench.py tests/unit/test_run_kernel_bench.py -v`
Expected: 11 passed. Then `make check` -- clean.

```bash
git add src/dispatch/kernels/backends.py src/dispatch/kernels/bench.py \
  scripts/run_kernel_bench.py tests/unit/test_backends.py \
  tests/unit/test_kernel_bench.py tests/unit/test_run_kernel_bench.py docs/STATUS.md
git commit -m "feat: add backend registry and kernel micro-benchmark CLI"
```

---

### Task 8: Real-model integration

**Files:**
- Create: `src/dispatch/kernels/integration.py`
- Test: `tests/unit/test_integration.py`
- Modify: `src/dispatch/benchmark/reference.py`
- Modify: `tests/unit/test_reference.py`
- Modify: `scripts/run_baseline.py`
- Modify: `tests/unit/test_run_baseline.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `patch_moe_infer(model, matmul, *, block_m=16) -> int` (returns
  how many MoE layers were patched); `TopKAgreement(positions,
  top1_agreement, mutual_top_k, max_abs_diff)`;
  `compare_top_k_agreement(actual, reference, *, k=5) -> dict[str,
  TopKAgreement]`; `run_baseline(..., moe_kernel="none") ->
  tuple[list[TokenTimings], dict[str, Tensor], int]` (now a 3-tuple);
  `--moe-kernel` and `--compare-reference` on `scripts/run_baseline.py`.

- [ ] **Step 1: Write `tests/unit/test_integration.py` (failing)**

```python
"""CPU-only: patches a stand-in whose moe_infer is transcribed from
deepseek-ai/deepseek-moe-16b-base's own modeling_deepseek.py (numpy's
cumsum swapped for torch's), so the swap is proven against DeepSeek's
actual inference code, not against a paraphrase of it."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.integration import patch_moe_infer
from dispatch.kernels.moe_forward import torch_grouped_matmul
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
```

- [ ] **Step 2: Write `src/dispatch/kernels/integration.py`**

```python
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
    """Returns how many MoE layers were patched, for the caller to check."""
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
```

- [ ] **Step 3: Add the logit comparison (failing test first)**

Add to `tests/unit/test_reference.py` -- extend the import list with
`TopKAgreement` and `compare_top_k_agreement`, then add before the `slow`
test:

```python
def test_compare_top_k_agreement_is_perfect_for_identical_logits() -> None:
    logits = {"p": torch.randn(4, 10)}

    assert compare_top_k_agreement(logits, logits) == {
        "p": TopKAgreement(positions=4, top1_agreement=1.0, mutual_top_k=True, max_abs_diff=0.0)
    }


def test_compare_top_k_agreement_tolerates_a_near_tie_flip() -> None:
    reference = {"p": torch.tensor([[5.0, 4.99, 3.0, 2.0, 1.0, 0.0]])}
    actual = {"p": torch.tensor([[4.99, 5.0, 3.0, 2.0, 1.0, 0.0]])}

    result = compare_top_k_agreement(actual, reference)["p"]

    assert result.top1_agreement == 0.0
    assert result.mutual_top_k


def test_compare_top_k_agreement_flags_an_argmax_outside_the_top_k() -> None:
    reference = {"p": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]])}
    actual = {"p": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 9.0]])}

    assert not compare_top_k_agreement(actual, reference)["p"].mutual_top_k


def test_compare_top_k_agreement_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_top_k_agreement({"p": torch.zeros(2, 6)}, {"p": torch.zeros(3, 6)})
```

Then in `src/dispatch/benchmark/reference.py`: add `from dataclasses import
dataclass` to the imports, the `TopKAgreement` dataclass after them, factor
the existing key check out of `compare_within_tolerance` into
`_require_same_keys`, and add the comparison:

```python
@dataclass(frozen=True)
class TopKAgreement:
    """Position-by-position agreement between two logit tensors, judged the
    way a kernel swap should be: a near-tie may flip, a real bug may not."""

    positions: int
    top1_agreement: float
    mutual_top_k: bool  # at every position, each side's argmax is in the other's top-k
    max_abs_diff: float


def compare_top_k_agreement(
    actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], *, k: int = 5
) -> dict[str, TopKAgreement]:
    _require_same_keys(actual, reference)
    return {key: _top_k_agreement(actual[key], reference[key], k) for key in reference}


def _require_same_keys(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> None:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )


def _top_k_agreement(actual: torch.Tensor, reference: torch.Tensor, k: int) -> TopKAgreement:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: actual {tuple(actual.shape)}, reference {tuple(reference.shape)}"
        )
    actual_f, reference_f = actual.float(), reference.float()
    actual_top1 = actual_f.argmax(dim=-1)
    reference_top1 = reference_f.argmax(dim=-1)
    actual_in_reference = (reference_f.topk(k, dim=-1).indices == actual_top1.unsqueeze(-1)).any(
        dim=-1
    )
    reference_in_actual = (actual_f.topk(k, dim=-1).indices == reference_top1.unsqueeze(-1)).any(
        dim=-1
    )
    return TopKAgreement(
        positions=int(actual_top1.numel()),
        top1_agreement=float((actual_top1 == reference_top1).float().mean()),
        mutual_top_k=bool((actual_in_reference & reference_in_actual).all()),
        max_abs_diff=float((actual_f - reference_f).abs().max()),
    )
```

`compare_within_tolerance` keeps its behavior; its first two lines become a
call to `_require_same_keys(actual, reference)`.

- [ ] **Step 4: Teach the baseline CLI to run a kernel (failing tests first)**

Replace `tests/unit/test_run_baseline.py` with:

```python
"""main()'s plumbing (args -> files) is tested fast with run_baseline and
summarize monkeypatched out; the real end-to-end passes against a tiny
model are `slow`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import scripts.run_baseline as run_baseline_module
import torch
from scripts.run_baseline import main

from dispatch.benchmark.metrics import BenchmarkSummary
from dispatch.benchmark.reference import save_reference

FAKE_SUMMARY = BenchmarkSummary(
    run_count=1,
    mean_ttft=0.1,
    p50_ttft=0.1,
    p99_ttft=0.1,
    mean_inter_token_latency=0.05,
    mean_tokens_per_second=20.0,
)
STOCK_LOGITS = {"prompt_000_logits": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]])}

FakeRunBaseline = Callable[..., tuple[list[object], dict[str, torch.Tensor], int]]


def _fake_run_baseline(logits: dict[str, torch.Tensor], moe_layers_patched: int) -> FakeRunBaseline:
    def fake(
        model_name: str, **kwargs: object
    ) -> tuple[list[object], dict[str, torch.Tensor], int]:
        return [object()], logits, moe_layers_patched

    return fake


def test_main_writes_results_and_reference_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_baseline_module,
        "run_baseline",
        _fake_run_baseline({"prompt_000_logits": torch.zeros(1)}, 0),
    )
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(
        [
            "--model-name",
            "tiny/test-model",
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "test-run",
        ]
    )

    results = json.loads((tmp_path / "test-run-results.json").read_text())
    assert results["model"] == "tiny/test-model"
    assert results["mean_tokens_per_second"] == 20.0
    assert results["moe_kernel"] == "none"
    assert results["reference_comparison"] == {}
    assert (tmp_path / "test-run-reference.safetensors").exists()


def test_main_records_agreement_with_a_stock_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "stock-reference.safetensors"
    save_reference(STOCK_LOGITS, reference_path)
    monkeypatch.setattr(run_baseline_module, "run_baseline", _fake_run_baseline(STOCK_LOGITS, 27))
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(
        [
            "--moe-kernel",
            "persistent",
            "--compare-reference",
            str(reference_path),
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "kernel-run",
        ]
    )

    results = json.loads((tmp_path / "kernel-run-results.json").read_text())
    assert results["moe_kernel"] == "persistent"
    assert results["moe_layers_patched"] == 27
    assert results["reference_comparison"]["prompt_000_logits"]["mutual_top_k"] is True


def test_main_exits_nonzero_on_disagreement_but_keeps_the_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "stock-reference.safetensors"
    save_reference(STOCK_LOGITS, reference_path)
    wrong = {"prompt_000_logits": torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 9.0]])}
    monkeypatch.setattr(run_baseline_module, "run_baseline", _fake_run_baseline(wrong, 27))
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)

    with pytest.raises(SystemExit, match="disagree"):
        main(
            [
                "--moe-kernel",
                "naive",
                "--compare-reference",
                str(reference_path),
                "--output-dir",
                str(tmp_path),
                "--run-label",
                "kernel-run",
            ]
        )

    results = json.loads((tmp_path / "kernel-run-results.json").read_text())
    assert results["reference_comparison"]["prompt_000_logits"]["mutual_top_k"] is False
    assert (tmp_path / "kernel-run-reference.safetensors").exists()


@pytest.mark.slow
def test_run_baseline_end_to_end_with_a_real_tiny_model() -> None:
    runs, logits, moe_layers_patched = run_baseline_module.run_baseline(
        "hf-internal-testing/tiny-random-gpt2",
        device="cpu",
        dtype=torch.float32,
        trust_remote_code=False,
        prompts=["hello"],
        repetitions=1,
        max_new_tokens=3,
    )

    assert len(runs) == 1
    assert "prompt_000_logits" in logits
    assert moe_layers_patched == 0


@pytest.mark.slow
def test_run_baseline_refuses_a_kernel_run_that_patches_nothing() -> None:
    with pytest.raises(RuntimeError, match="patched no MoE layers"):
        run_baseline_module.run_baseline(
            "hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            prompts=["hello"],
            repetitions=1,
            max_new_tokens=3,
            moe_kernel="torch",
        )
```

- [ ] **Step 5: Update `scripts/run_baseline.py`**

Replace the module docstring and imports, add `moe_kernel` to
`run_baseline`, and extend `main`:

```python
"""CLI: run DeepSeekMoE-16B on one rented GPU -- the stock HF forward pass by
default, or with a dispatch grouped-GEMM backend patched into every MoE
layer (--moe-kernel). Writes latency/throughput and the run's logits; with
--compare-reference, also checks those logits against a stock run's and
exits non-zero if they disagree.
"""
```

```python
from dispatch.benchmark.reference import (
    capture_reference_logits,
    compare_top_k_agreement,
    load_reference,
    save_reference,
)
from dispatch.kernels.backends import BACKENDS, resolve_backend
from dispatch.kernels.integration import patch_moe_infer
```

```python
def run_baseline(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    prompts: list[str],
    repetitions: int,
    max_new_tokens: int,
    moe_kernel: str = "none",
) -> tuple[list[TokenTimings], dict[str, torch.Tensor], int]:
    """Returns the timed runs, the logits, and how many MoE layers were patched."""
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )
    moe_layers_patched = 0
    if moe_kernel != "none":
        moe_layers_patched = patch_moe_infer(model, resolve_backend(moe_kernel))
        if moe_layers_patched == 0:
            raise RuntimeError(
                f"--moe-kernel {moe_kernel} patched no MoE layers: {model_name} has no "
                "moe_infer to replace, so this run would time the stock model under a kernel's name"
            )

    runs = [
        generate_with_timings(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        for prompt in prompts
        for _ in range(repetitions)
    ]
    logits = capture_reference_logits(model, tokenizer, prompts, device=device)
    return runs, logits, moe_layers_patched
```

In `main`, add the two arguments after `--run-label`:

```python
    parser.add_argument("--moe-kernel", default="none", choices=["none", *BACKENDS])
    parser.add_argument(
        "--compare-reference",
        type=Path,
        default=None,
        help="logits file from a stock run of the same model and prompts",
    )
```

then unpack three values, compare, record, and fail loudly *after* the
evidence is written:

```python
    runs, logits, moe_layers_patched = run_baseline(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=DEFAULT_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        moe_kernel=args.moe_kernel,
    )
    summary = summarize(runs)
    comparison = (
        compare_top_k_agreement(logits, load_reference(args.compare_reference))
        if args.compare_reference is not None
        else {}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "device": args.device,
                "dtype": args.dtype,
                "moe_kernel": args.moe_kernel,
                "moe_layers_patched": moe_layers_patched,
                **asdict(summary),
                "reference_comparison": {key: asdict(value) for key, value in comparison.items()},
            },
            indent=2,
        )
    )

    reference_path = args.output_dir / f"{args.run_label}-reference.safetensors"
    save_reference(logits, reference_path)

    print(f"wrote {results_path}")
    print(f"wrote {reference_path}")

    if not all(value.mutual_top_k for value in comparison.values()):
        raise SystemExit(
            f"{args.moe_kernel} logits disagree with {args.compare_reference} -- see {results_path}"
        )
```

- [ ] **Step 6: Run the tests, lint, typecheck, commit**

Run: `uv run pytest tests/unit/test_integration.py tests/unit/test_reference.py tests/unit/test_run_baseline.py -v`
Expected: 17 passed (3 of them `slow`, needing network for the tiny model).
Then `make check` -- clean; the whole non-GPU suite is 71 passed, 1 skipped.

```bash
git add src/dispatch/kernels/integration.py tests/unit/test_integration.py \
  src/dispatch/benchmark/reference.py tests/unit/test_reference.py \
  scripts/run_baseline.py tests/unit/test_run_baseline.py docs/STATUS.md
git commit -m "feat: swap grouped-GEMM backends into DeepSeekMoE and compare logits"
```

---

### Task 9: The measured run

**Files:**
- Modify: `docs/runbooks/phase-1-grouped-gemm.md` (session B)
- Create: `docs/findings/<date>-phase-1-grouped-gemm-run.md`
- Create: `docs/findings/<date>-phase-1-*.json` (written by the CLIs)
- Create: `docs/findings/<date>-phase-1-grouped-gemm-cost.md` (written by the tool)
- Modify: `docs/STATUS.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`

**This is the second and last task that spends money.** Explicit go-ahead
and a stated cap first, same as Phase 0's Task 8.

- [ ] **Step 1: Append session B to the runbook**

```markdown
## Session B -- the measured run (Task 9)

Same GPU class as Phase 0 -- NVIDIA L40 -- so the comparison holds. If no
L40 is available, stop and ask: switching class silently breaks the
same-hardware rule that makes these numbers mean anything.

1. Locally: `make check` green, `git status` clean, Tasks 1-8 committed,
   budget cap stated out loud.
2. Check whether DeepSeek has changed `modeling_deepseek.py` since Phase 0
   (https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/commits/main).
   If it has, the two workarounds below may be unnecessary or wrong.
3. Create an L40 pod with a 60GB+ volume, wait, and confirm the card:

       nvidia-smi --query-gpu=name,compute_cap --format=csv

4. Transfer: `git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"`
5. Environment, on the pod -- all three of Phase 0's traps at once:

       cd dispatch
       export HF_HOME=/workspace/hf_cache        # NOT the 30GB container disk
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv pip install transformers==4.57.6       # Phase 0 bug 1

   From here on invoke `.venv/bin/python` directly. `uv run` re-syncs to
   `uv.lock` on every call and silently undoes that override.

   For Phase 0 bug 2, create a pod-local wrapper -- never committed:

       cat > _patch_and_run.py <<'EOF'
       import sys
       from transformers.cache_utils import DynamicCache


       def _get_usable_length(self, new_seq_length=None, layer_idx=0):
           return self.get_seq_length(layer_idx)


       DynamicCache.get_usable_length = _get_usable_length

       from scripts.run_baseline import main

       main(sys.argv[1:])
       EOF

6. Kernel correctness on the measurement hardware, before any timing:

       .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs

   Expected: 25 passed, 0 skipped (an L40 is compute capability 8.9, so the
   bf16 cases run). Anything red: stop. Nothing below is meaningful.
7. The stock run -- this session's own "before" number and the reference
   every kernel run is compared against:

       DATE=$(date +%Y-%m-%d)
       .venv/bin/python _patch_and_run.py --trust-remote-code \
         --run-label $DATE-phase-1-stock

8. The three backend runs. The CLI exits non-zero on disagreement, after
   writing its evidence:

       for kernel in torch naive persistent; do
         .venv/bin/python _patch_and_run.py --trust-remote-code \
           --moe-kernel $kernel \
           --compare-reference docs/findings/$DATE-phase-1-stock-reference.safetensors \
           --run-label $DATE-phase-1-$kernel || break
       done

   Every results JSON must show `"moe_layers_patched": 27`.
9. The micro-benchmark, both routing shapes:

       .venv/bin/python -m scripts.run_kernel_bench --distribution zipf \
         --run-label $DATE-phase-1-kernel-bench-zipf
       .venv/bin/python -m scripts.run_kernel_bench --distribution uniform \
         --run-label $DATE-phase-1-kernel-bench-uniform

10. Copy the JSON back (the .safetensors stay on the pod -- `.gitignore`
    excludes them repo-wide by design):

        scp '<pod>:dispatch/docs/findings/'"$DATE"'-phase-1-*.json' docs/findings/

11. Cost, teardown, and an independent verification that the pod reports
    `TERMINATED` -- same as session A step 8, with
    `run_label='phase-1-grouped-gemm'`.
```

- [ ] **Step 2: Get the go-ahead, then run session B**

- [ ] **Step 3: Write the finding**

Create `docs/findings/<date>-phase-1-grouped-gemm-run.md`. Every cell comes
from the JSON files the run produced; nothing is filled in from anywhere
else, and nothing is estimated.

```markdown
# Phase 1 grouped-GEMM run -- findings

## Infrastructure
Pod id, card and compute capability, hourly rate, cloud tier, image,
torch/triton/CUDA versions (from the kernel-bench JSON's `config`),
wall-clock duration, measured cost.

## Correctness
- `pytest -m gpu` on this card: N passed, N skipped.
- End to end, from each results JSON's `reference_comparison`:

| Run | moe_layers_patched | mutual top-5 | top-1 agreement | max abs logit diff |
|---|---|---|---|---|
| torch (control) | | | | |
| naive | | | | |
| persistent | | | | |

The control row is the noise floor: identical math, eager PyTorch, same
grouped layout. A Triton row materially worse than it is a kernel
precision problem to investigate, not a tolerance to loosen.

## End-to-end throughput
Same config as Phase 0 -- bf16, single L40, unbatched eager decode, 3
prompts x 5 repetitions, 64 new tokens -- all four runs in one session.

| Run | mean tokens/sec | mean TTFT | p50 TTFT | p99 TTFT | mean ITL | $ per 1M tokens |
|---|---|---|---|---|---|---|

Cost per 1M generated tokens = rate ($/hr) / (mean tokens/sec x 3600) x 1e6,
computed from two measured inputs.

## Kernel micro-benchmark
Per token count, per backend, for both `zipf` and `uniform` routing: mean /
p50 / p99 latency and TFLOP/s, for the whole routed layer and for the
gate_proj-shaped GEMM alone.

## What it shows
Written once the numbers exist. Unbatched decode gives each MoE layer six
routed rows per step, so a flat or negative end-to-end result is a real
finding about where grouped GEMM pays off -- not something to bury -- and
the token-count sweep is where the kernel's own effect is visible.
```

- [ ] **Step 4: Refresh the reader-facing artifacts**

Per CLAUDE.md's proactive-refresh convention, ask of each: what changed
today that a reader would want to know?

- `docs/STATUS.md`: Phase 1 complete, quoting measured numbers; all boxes.
- `README.md`: the status line, with the measured kernel result.
- `CHANGELOG.md`: a Phase 1 entry in the same shape as Phase 0's.
- `CLAUDE.md`: the Current status block, and the Testing policy section --
  which still says the kernel specifics "get written once there's a real
  kernel". There is one now:

| Layer | What must be covered |
|---|---|
| Kernel contract (CPU) | eager backend == `ReferenceMoE`; every row covered by exactly one tile; no tile crosses an expert boundary; experts that receive zero tokens |
| Kernels (`gpu`) | each kernel meets `assert_matches_reference` against an fp32 reference at toy, decode- and prefill-shaped dims, fp16 and bf16; a mutation must turn the suite red |
| End to end (`gpu`, paid) | mutual top-5 logit agreement with a same-session stock run; a kernel run that patches no layers refuses to run |
| Benchmarks | `triton.testing.do_bench`; a backend that disagrees with the eager one is refused, not timed; every JSON carries its full config |

- [ ] **Step 5: Story bank**

Check gist-worthiness -- the trigger is documented as unreliable, so check
explicitly. Likely candidates: whatever the kernel got wrong in Task 6, the
Triton-is-Linux-only constraint, and the weight-stacking memory trap.

- [ ] **Step 6: Commit, push, PR**

```bash
git add docs/runbooks/phase-1-grouped-gemm.md docs/findings/ docs/STATUS.md \
  README.md CHANGELOG.md CLAUDE.md
git commit -m "docs: record Phase 1 measured run"
```

Then, with the user's go-ahead, push the branch and open the phase's single
PR -- title `Phase 1: custom Triton grouped-GEMM kernel`, body summarizing
what was built, what was measured, and what was found.

---

## Risks this plan knows about

- **The kernels have never run.** Task 6 is budgeted for finding that out.
  Expect at least one real bug; Phase 0's precedent is three.
- **Block sizes are fixed, not tuned.** `BLOCK_N=64`, `BLOCK_K=64`,
  `GROUP_SIZE_M=8`, `block_m=16` are defaults, recorded with every number
  rather than optimized. `triton.autotune` is a deliberate non-goal here:
  it would make Phase 1 a tuning exercise rather than a correctness-first
  kernel build.
- **Grouped launch ordering may show nothing at decode sizes.** With one
  tile per expert there is no band of rows to keep in L2. That is why the
  benchmark sweeps token counts instead of reporting a single number.
- **DeepSeek's remote code may have moved** since Phase 0. Session B step 2
  checks before assuming the old workarounds still apply.
- **L40 availability** is checked live, not assumed; Phase 0 already had to
  take Secure Cloud at $0.82/hr when Community's $0.69/hr L40 was out of
  stock.
