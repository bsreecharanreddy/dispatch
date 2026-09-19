# Phase 6 final benchmark vs. vLLM and SGLang -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Tasks 10-14 rent a GPU and spend real money: never start one without the user's explicit go-ahead for that task, and read "Cost and stop rules" first.**

**Goal:** Measure, on one rented L40, how dispatch's Triton grouped-GEMM kernels compare with vLLM's and SGLang's fused-MoE on DeepSeek-MoE-16B's real routed-expert shape, gate every config with an at-scale correctness check first, and report where a full production engine lands on the same model as a labeled reference.

**Architecture:** Three stages on one pod. Stage 1 is a per-config at-scale logit-agreement gate reusing `scripts/run_baseline.py`, extended with a gap-split classifier. Stage 2 is a kernel race: identical seeded inputs and fp32 references are written once to disk, then each engine behind a thin adapter is checked against the reference (refused, not timed, on disagreement) and timed with `triton.testing.do_bench`. Stage 3 launches vLLM and SGLang servers and drives both with `vllm bench serve`; dispatch's own harness supplies the concurrency-1 row. Everything CPU-testable (classifier, adapters, driver, summarizer, serving helpers) is built and verified locally first with fakes.

**Tech Stack:** Python 3.12+, uv, ruff, `mypy --strict`, pytest, PyTorch, Triton, safetensors; on the pod only: vllm 0.29.0, sglang 0.5.20, torch 2.13.0 (all checked live 2026-09-19).

**Spec:** `docs/design/2026-09-18-phase-6-final-benchmark.md` (amended 2026-09-19 alongside this plan; the amendments are recorded in its §3 and in system design §6).

## Global Constraints

Every task's requirements implicitly include this section.

- Python 3.12+; `uv`, `ruff` (line length 100), `mypy --strict`, `pytest`. `make check` (lint, typecheck, test) is green before every commit and before any push.
- **Correctness before speed.** No latency number is reported for an engine that has not first matched the fp32 reference (`assert_matches_reference`, rtol 1.6e-2, atol scaled to the reference magnitude).
- **Never quote a benchmark number that wasn't measured** on this hardware, at this config, by this repo. Every reported number states its config: GPU, dtype, precision, token count, routing distribution, tuning label, engine versions.
- `gpu`-marked tests are excluded from CI, with the skip reason visible in the test itself.
- One commit per task. `docs/STATUS.md` is updated in the same commit as the work it describes.
- **Commit messages are plain ASCII: `--`, never an em-dash. No attribution lines in commit messages or PR descriptions** (standing instruction for this session).
- One branch for the phase: `phase-6-final-benchmark` (already created). One PR at the end. Do not push or open the PR without asking the user.
- Model: `deepseek-ai/deepseek-moe-16b-base`. Routed-expert dims (checked live 2026-09-15): hidden 2048, moe_intermediate 1408, 64 routed experts, top-6.
- Hardware: **one L40, RunPod Secure Cloud**. Budget cap **$10**, a ceiling not a target, set before the first rental.
- Token counts (per layer call): `(1, 4, 16, 64, 128, 512, 2048)`; routing `uniform` and `zipf`; precisions `bf16` and weight-only `int8`; seed 0.
- Engine pins: `torch==2.13.0`, `vllm==0.29.0`, `sglang==0.5.20`. A different version is a recorded deviation, never a silent one.
- Findings go to `docs/findings/phase-6/`; the new scripts default their output there.

## Pre-registered gate rule and decision rules

Fixed here, **before any Phase 6 GPU time**. Changing any of them after seeing a run's numbers is not allowed; a change needs a written amendment to this section first, with the new evidence that motivates it.

**The at-scale gate (Stage 1, Task 11).** For each config C in {naive bf16, persistent bf16, int8 (`quantized`)}, run `scripts/run_baseline.py --prompt-set gate` against a same-session stock reference. The gate compares the logits at every prompt position (1,036 positions across 16 prompts with the real tokenizer, measured 2026-09-19; a run under 500 fails).

- `LARGE_GAP_THRESHOLD = 1.0` logits. A top-1 disagreement where the *reference's* top-1/top-2 gap exceeds this is a **failure**; at or below it is a **near-tie flip**, reported and counted, not failed.
- **Why 1.0:** Phase 5b measured near-tie gaps of 0.1-0.4 on logits of about 20 magnitude (`docs/findings/phase-5b/2026-09-18-phase-5b-speculative-decoding-run.md`), and bf16's own spacing at that magnitude is 0.125. 1.0 is 2.5x the widest measured near-tie and 8 bf16 spacings: loose enough not to fail a correct kernel on floating-point noise, tight enough that a flip at a gap of 1.0+ is not explained by anything measured so far. Phase 5a's `max_abs_diff` (1.3-2.1, taken over every position and every vocabulary entry) is a looser bound on a different quantity and is *not* used: it would classify almost any flip as noise.
- `MIN_GATE_POSITIONS = 500`. Phase 5a's claim rested on 29 positions.
- **Stock-vs-stock control:** one extra run of the unpatched model compared against the first. If the control itself shows a large-gap disagreement, the gate's noise floor is broken (nondeterminism beyond near-ties): **stop the session and investigate before spending on Stages 2-3.**
- **Per-config consequence.** A config that trips the gate is excluded from the race and reported as "failed its correctness gate" with the evidence; the other configs proceed. A trip is a finding to investigate, never a threshold to loosen.

**The race (Stage 2, Task 12).**

- Every contestant is tuned under **uniform** routing only (the distribution the engines' own tuners use), then timed under both distributions. dispatch's "tuned" row is the tile size (`block_m` in 16/32/64/128) fastest on uniform at that token count, reused for zipf. Its "default" row is `block_m` 16. vLLM and SGLang get one `default` run (whatever ships for this GPU, recorded) and one `tuned` run.
- An engine that fails the fp32-reference check is refused, not timed. An adapter/API mismatch on the pod gets a **30-minute timebox** to fix; past that, that engine is reported "not measurable at the pinned version" with the failing output kept.
- Tuner timeboxes: **60 minutes per engine per precision.** Token counts a tuner did not finish are labeled `default`, not `tuned`.

**The engine reference (Stage 3, Task 13).**

- 30-minute timebox per engine to get a server answering `/health`; past it, the failure log is kept and the engine is reported as unable to serve this checkpoint at the pinned version.
- A bench result with any failed request, or any request that generated other than 64 tokens, is refused (`summarize_bench_result`).
- The dispatch concurrency-1 row is labeled with its non-comparabilities (in-process vs. HTTP, no CUDA graphs); concurrency 4/16/64 read "n/a: no dispatch server".

## Cost and stop rules

- **State the live hourly price and get the user's go-ahead before creating the pod.** Estimated cost is *not* a measurement: at the ~$0.7/hr an L40 cost in Phase 5a, the $10 cap is roughly 14 pod-hours; the plan's stages are budgeted at 6-8 hours. Both numbers are planning estimates only.
- **Checkpoints:** at $5 spent, pause and report to the user; at $8, stop starting new work and go to teardown. Never let the total pass $10 without the user raising the cap in writing.
- **Never leave the pod running across an unbounded wait** (a question to the user, a background task with no deadline). Stop it first, restart on the answer. (Phase 5b's $10 cap was exceeded on exactly this.)
- **Pull every evidence file off the pod before `stop` or terminate.** Nothing outside `/workspace` survives a stop, and a stopped pod may be unrestartable ("not enough free GPUs on the host machine", hit on three Phase 5b pods).
- Everything expensive to redo (repo, venvs, model cache, race inputs, tuned configs) lives under `/workspace`.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/dispatch/benchmark/agreement.py` | gap-split classifier, the pre-registered threshold and position floor | 1 |
| `src/dispatch/benchmark/gate_prompts.py` | the 16 fixed prompts (gate set and engine-reference trace) | 2 |
| `scripts/run_baseline.py` (modify) | `--prompt-set gate`, gap-split output and enforcement, `--ignore-eos` | 2, 3 |
| `src/dispatch/benchmark/harness.py`, `metrics.py` (modify) | `ignore_eos`; public `percentile` | 3 |
| `src/dispatch/benchmark/engines/base.py` | engine contract, seeded case/weight builders, fp32 reference, fused layout | 4 |
| `src/dispatch/benchmark/engines/dispatch_kernels.py` | dispatch's kernels as a contestant | 4 |
| `src/dispatch/benchmark/engines/vllm_moe.py`, `sglang_moe.py`, `registry.py` | the two production engines behind adapters; name -> contestants | 5 |
| `src/dispatch/benchmark/race_summary.py` | tuning rule, rows, markdown table | 6 |
| `scripts/run_engine_race.py` | `prepare` / `run` / `merge` driver | 7 |
| `src/dispatch/benchmark/serving_bench.py`, `scripts/gpu/phase6_engine_reference.py` | serving-benchmark commands, refuse rules, orchestration | 8 |
| `tests/unit/test_engines_real_gpu.py` | real-engine gate (`gpu`), incl. the mutation check | 9 |
| `docs/findings/phase-6/`, `docs/runbooks/phase-6-final-benchmark.md` | evidence and the runbook written from what actually worked | 14, 15 |

Tasks 1-9 are local, free, and TDD. Tasks 10-14 are the paid pod session. Task 15 is findings, docs and the PR.

---

### Task 1: Gap-split agreement classifier

**Files:**
- Create: `src/dispatch/benchmark/agreement.py`
- Test: `tests/unit/test_agreement.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (torch only).
- Produces: `LARGE_GAP_THRESHOLD: float`, `MIN_GATE_POSITIONS: int`, `GapSplitAgreement` (frozen dataclass: `positions`, `disagreements`, `near_tie_disagreements`, `large_gap_disagreements`, `max_disagreement_gap`; property `top1_agreement`; `to_dict()`), `compare_gap_split(actual, reference, *, threshold=LARGE_GAP_THRESHOLD) -> dict[str, GapSplitAgreement]`, `aggregate_gap_split(Iterable[GapSplitAgreement]) -> GapSplitAgreement`. Task 2 consumes all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_agreement.py`:

````python
from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.agreement import (
    LARGE_GAP_THRESHOLD,
    MIN_GATE_POSITIONS,
    GapSplitAgreement,
    aggregate_gap_split,
    compare_gap_split,
)


def _row(top1: float, top2: float, *, winner: int) -> torch.Tensor:
    """One 5-token-vocab position whose two largest logits are top1 and top2;
    `winner` is which of columns 0/1 holds the larger value."""
    row = torch.zeros(5)
    row[winner], row[1 - winner] = top1, top2
    return row


def _logits(*rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"prompt_000_logits": torch.stack(rows)}


def test_threshold_and_minimum_positions_are_the_preregistered_values() -> None:
    # Pre-registered before any Phase 6 GPU time; changing either needs a
    # written plan amendment first, not an edit that makes a red run green.
    assert LARGE_GAP_THRESHOLD == 1.0
    assert MIN_GATE_POSITIONS == 500


def test_identical_logits_have_no_disagreements() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0), _row(9.0, 8.5, winner=1))

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert result.positions == 2
    assert result.disagreements == 0
    assert result.top1_agreement == 1.0
    assert result.max_disagreement_gap == 0.0


def test_a_flip_at_a_tiny_gap_is_a_near_tie() -> None:
    reference = _logits(_row(20.0, 19.875, winner=0))  # one bf16 spacing at ~20
    actual = _logits(_row(20.0, 19.875, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (1, 0)
    assert result.max_disagreement_gap == pytest.approx(0.125)


def test_a_flip_at_a_large_gap_is_reported_separately() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0))
    actual = _logits(_row(10.0, 4.0, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (0, 1)
    assert result.max_disagreement_gap == pytest.approx(6.0)


def test_a_gap_exactly_at_the_threshold_is_a_near_tie() -> None:
    reference = _logits(_row(5.0 + LARGE_GAP_THRESHOLD, 5.0, winner=0))
    actual = _logits(_row(5.0 + LARGE_GAP_THRESHOLD, 5.0, winner=1))

    result = compare_gap_split(actual, reference)["prompt_000_logits"]

    assert (result.near_tie_disagreements, result.large_gap_disagreements) == (1, 0)


def test_agreements_do_not_count_as_disagreements_whatever_their_gap() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0), _row(10.0, 9.9, winner=0))

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert (result.disagreements, result.large_gap_disagreements) == (0, 0)


def test_half_precision_inputs_are_compared_in_float32() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0)).copy()
    reference = {key: value.to(torch.bfloat16) for key, value in reference.items()}

    result = compare_gap_split(reference, reference)["prompt_000_logits"]

    assert result.disagreements == 0


def test_aggregate_sums_counts_and_keeps_the_widest_gap() -> None:
    first = GapSplitAgreement(
        positions=10,
        disagreements=2,
        near_tie_disagreements=2,
        large_gap_disagreements=0,
        max_disagreement_gap=0.3,
    )
    second = GapSplitAgreement(
        positions=30,
        disagreements=1,
        near_tie_disagreements=0,
        large_gap_disagreements=1,
        max_disagreement_gap=4.0,
    )

    total = aggregate_gap_split([first, second])

    assert (total.positions, total.disagreements) == (40, 3)
    assert (total.near_tie_disagreements, total.large_gap_disagreements) == (2, 1)
    assert total.max_disagreement_gap == 4.0
    assert total.top1_agreement == pytest.approx(37 / 40)


def test_to_dict_carries_the_derived_agreement_rate() -> None:
    result = GapSplitAgreement(
        positions=4,
        disagreements=1,
        near_tie_disagreements=1,
        large_gap_disagreements=0,
        max_disagreement_gap=0.1,
    )

    assert result.to_dict()["top1_agreement"] == 0.75


def test_aggregate_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="zero prompts"):
        aggregate_gap_split([])


def test_mismatched_keys_and_shapes_raise() -> None:
    reference = _logits(_row(10.0, 4.0, winner=0))
    with pytest.raises(ValueError, match="key mismatch"):
        compare_gap_split({"other": reference["prompt_000_logits"]}, reference)
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_gap_split({"prompt_000_logits": torch.zeros(2, 5)}, reference)
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_agreement.py -v`
Expected: collection FAIL with `ModuleNotFoundError: No module named 'dispatch.benchmark.agreement'`.

- [ ] **Step 3: Write the implementation**

Create `src/dispatch/benchmark/agreement.py`:

````python
"""Gap-split agreement between two logit tensors: every top-1 disagreement is
classified by how close the *reference's* own top-1 and top-2 logits were.

Phase 5b found genuine near-tied logits (a top-2 gap of 0.1-0.4 out of
~20-magnitude logits; bf16's own spacing at that magnitude is 0.125) that
flip under the int8 kernel's floating-point precision and even between two
processes running the same config. A kernel swap may flip a near-tie. It may
not flip a position the reference was confident about: that is a bug, not
noise. `top1_agreement` alone cannot tell the two apart, and demanding 100%
agreement would fail a correct kernel, so the gate splits them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch

# Pre-registered 2026-09-19, before any Phase 6 GPU time (docs/plans/
# 2026-09-19-phase-6-final-benchmark-plan.md, "Pre-registered gate rule").
# 1.0 logit is 2.5x the widest near-tie gap Phase 5b measured (0.4) and 8
# bf16 spacings at ~20 magnitude. It is NOT to be loosened after seeing a
# run's numbers: a run that trips it is a finding to investigate, and any
# change needs a written amendment to the plan first. A test pins this value.
LARGE_GAP_THRESHOLD = 1.0

# Phase 5a's "perfect agreement" covered 29 positions; "at scale" here means
# at least this many compared positions per config.
MIN_GATE_POSITIONS = 500


@dataclass(frozen=True)
class GapSplitAgreement:
    positions: int
    disagreements: int
    near_tie_disagreements: int  # reference top1-top2 gap <= threshold
    large_gap_disagreements: int  # reference top1-top2 gap > threshold
    max_disagreement_gap: float  # widest reference gap among disagreements; 0.0 if none

    @property
    def top1_agreement(self) -> float:
        return (self.positions - self.disagreements) / self.positions

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "top1_agreement": self.top1_agreement}


def compare_gap_split(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    threshold: float = LARGE_GAP_THRESHOLD,
) -> dict[str, GapSplitAgreement]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )
    return {key: _gap_split(actual[key], reference[key], threshold) for key in reference}


def aggregate_gap_split(per_prompt: Iterable[GapSplitAgreement]) -> GapSplitAgreement:
    items = list(per_prompt)
    if not items:
        raise ValueError("cannot aggregate zero prompts")
    return GapSplitAgreement(
        positions=sum(item.positions for item in items),
        disagreements=sum(item.disagreements for item in items),
        near_tie_disagreements=sum(item.near_tie_disagreements for item in items),
        large_gap_disagreements=sum(item.large_gap_disagreements for item in items),
        max_disagreement_gap=max(item.max_disagreement_gap for item in items),
    )


def _gap_split(
    actual: torch.Tensor, reference: torch.Tensor, threshold: float
) -> GapSplitAgreement:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: actual {tuple(actual.shape)}, reference {tuple(reference.shape)}"
        )
    if reference.shape[-1] < 2:  # noqa: PLR2004 -- a top-2 gap needs two logits
        raise ValueError("need at least two logits per position to measure a top-2 gap")
    reference_f = reference.float()
    top2 = reference_f.topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]
    disagree = actual.float().argmax(dim=-1) != reference_f.argmax(dim=-1)
    large = disagree & (gap > threshold)
    near_tie = disagree & ~large
    return GapSplitAgreement(
        positions=int(disagree.numel()),
        disagreements=int(disagree.sum()),
        near_tie_disagreements=int(near_tie.sum()),
        large_gap_disagreements=int(large.sum()),
        max_disagreement_gap=float(gap[disagree].max()) if bool(disagree.any()) else 0.0,
    )
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_agreement.py -v`
Expected: 11 passed.

- [ ] **Step 5: Mutation check**

Change `LARGE_GAP_THRESHOLD` to `2.0` in `agreement.py` and re-run. Expected: `test_threshold_and_minimum_positions_are_the_preregistered_values` FAILS. Change it back to `1.0`. (This is the guard against post-hoc loosening.)

- [ ] **Step 6: Commit**

Add a bullet at the end of the "Phase 6 progress" section of `docs/STATUS.md`: `- **Task 1 (gap-split classifier)**: dispatch.benchmark.agreement splits every top-1 disagreement by the reference's top1-top2 logit gap; threshold 1.0 and 500-position floor pre-registered in the plan and pinned by a test.`

```bash
make check
git add src/dispatch/benchmark/agreement.py tests/unit/test_agreement.py docs/STATUS.md
git commit -m "feat: gap-split agreement classifier with pre-registered threshold"
```

---

### Task 2: Gate prompt set and `run_baseline --prompt-set gate`

**Files:**
- Create: `src/dispatch/benchmark/gate_prompts.py`, `tests/unit/test_gate_prompts.py`
- Modify: `scripts/run_baseline.py`
- Modify (append tests): `tests/unit/test_run_baseline.py`

**Interfaces:**
- Consumes: `compare_gap_split`, `aggregate_gap_split`, `MIN_GATE_POSITIONS` from Task 1.
- Produces: `GATE_PROMPTS: tuple[str, ...]` (16 prompts; Task 8 reuses it as the engine-reference trace). `run_baseline.py` gains `--prompt-set {default,gate}`; results JSON gains `prompt_set` and `gap_split` (`None` when no `--compare-reference`, else `GapSplitAgreement.to_dict()`); in gate mode with a reference the run exits non-zero on fewer than 500 positions or on any large-gap disagreement, *after* writing the results and reference files.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_gate_prompts.py`:

````python
from __future__ import annotations

from dispatch.benchmark.gate_prompts import GATE_PROMPTS


def test_gate_prompts_are_sixteen_distinct_nonempty_strings() -> None:
    assert len(GATE_PROMPTS) == 16
    assert len(set(GATE_PROMPTS)) == 16
    assert all(prompt.strip() for prompt in GATE_PROMPTS)


def test_gate_prompts_are_immutable_so_the_trace_cannot_drift_between_runs() -> None:
    assert isinstance(GATE_PROMPTS, tuple)


def test_gate_prompts_are_long_enough_to_reach_the_position_floor() -> None:
    # A cheap lower bound on the real-tokenizer count (measured 1036 positions
    # on 2026-09-19 with deepseek-moe-16b-base's tokenizer; the gate's runtime
    # MIN_GATE_POSITIONS check is the authoritative guard). Each prompt is
    # several dozen characters per expected token even at 3 chars/token.
    assert sum(len(prompt) for prompt in GATE_PROMPTS) / 6 > 500
````

Append to `tests/unit/test_run_baseline.py`, immediately above the first `@pytest.mark.slow` test (it needs no new imports):

````python
def _gate_logits(rows: int) -> torch.Tensor:
    """`rows` positions over a 7-token vocab whose top-1 is column 0 at a 3.0 gap."""
    logits = torch.zeros(rows, 7)
    logits[:, 0], logits[:, 1] = 5.0, 2.0
    return logits


def _gate_pair(
    rows: int, flips: dict[int, float]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(reference, actual): identical except that at each row in `flips` the
    reference's top-1/top-2 gap is that value and actual has them swapped."""
    reference = _gate_logits(rows)
    actual = _gate_logits(rows)
    for row, gap in flips.items():
        reference[row, 0], reference[row, 1] = 5.0, 5.0 - gap
        actual[row, 0], actual[row, 1] = 5.0 - gap, 5.0
    return {"prompt_000_logits": reference}, {"prompt_000_logits": actual}


def _run_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    reference_path = tmp_path / "stock-reference.safetensors"
    save_reference(reference, reference_path)
    monkeypatch.setattr(run_baseline_module, "run_baseline", _fake_run_baseline(actual, 27))
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)
    main(
        [
            "--prompt-set",
            "gate",
            "--moe-kernel",
            "quantized",
            "--compare-reference",
            str(reference_path),
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "gate-run",
        ]
    )


def test_gate_records_the_gap_split_and_passes_a_clean_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference, actual = _gate_pair(600, {})

    _run_gate(monkeypatch, tmp_path, reference, actual)

    results = json.loads((tmp_path / "gate-run-results.json").read_text())
    assert results["prompt_set"] == "gate"
    assert results["gap_split"]["positions"] == 600
    assert results["gap_split"]["disagreements"] == 0
    assert results["gap_split"]["top1_agreement"] == 1.0


def test_gate_passes_near_tie_flips_but_counts_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference, actual = _gate_pair(600, {3: 0.125, 40: 0.25, 99: 0.4})

    _run_gate(monkeypatch, tmp_path, reference, actual)

    split = json.loads((tmp_path / "gate-run-results.json").read_text())["gap_split"]
    assert (split["near_tie_disagreements"], split["large_gap_disagreements"]) == (3, 0)
    assert split["max_disagreement_gap"] == pytest.approx(0.4)


def test_gate_fails_a_large_gap_flip_but_keeps_the_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference, actual = _gate_pair(600, {3: 0.125, 7: 3.0})

    with pytest.raises(SystemExit, match="a bug, not a near-tie"):
        _run_gate(monkeypatch, tmp_path, reference, actual)

    split = json.loads((tmp_path / "gate-run-results.json").read_text())["gap_split"]
    assert split["large_gap_disagreements"] == 1
    assert (tmp_path / "gate-run-reference.safetensors").exists()


def test_gate_refuses_to_claim_agreement_over_too_few_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference, actual = _gate_pair(29, {})  # Phase 5a's sample size

    with pytest.raises(SystemExit, match="fewer than the 500"):
        _run_gate(monkeypatch, tmp_path, reference, actual)


def test_a_run_without_a_reference_records_no_gap_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_baseline_module, "run_baseline", _fake_run_baseline(_gate_pair(600, {})[1], 0)
    )
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(["--prompt-set", "gate", "--output-dir", str(tmp_path), "--run-label", "ref-run"])

    results = json.loads((tmp_path / "ref-run-results.json").read_text())
    assert results["gap_split"] is None
    assert (tmp_path / "ref-run-reference.safetensors").exists()
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gate_prompts.py tests/unit/test_run_baseline.py -v -m "not slow"`
Expected: `test_gate_prompts.py` fails to import; the new `test_run_baseline.py` tests fail with `SystemExit: 2` (argparse: unrecognized `--prompt-set`).

- [ ] **Step 3: Write the prompt set**

Create `src/dispatch/benchmark/gate_prompts.py`:

````python
"""The fixed prompt set for Phase 6's at-scale correctness gate. Real prose,
code, and arithmetic text of mixed style, so the routed experts see varied
token distributions rather than one register. Compared teacher-forced (the
logits at every prompt position), which is why each is a few sentences long:
the gate needs hundreds of positions per config, and the runtime check
against MIN_GATE_POSITIONS enforces that with the real tokenizer.

Also the request trace for the engine reference (docs/design/
2026-09-18-phase-6-final-benchmark.md section 3.3), so every engine, and
dispatch's own harness, sees identical text.
"""

from __future__ import annotations

GATE_PROMPTS: tuple[str, ...] = (
    "The committee reviewed the proposal for the new harbor bridge over three long "
    "sessions. Engineers argued about the cost of steel, while residents worried about "
    "traffic noise and the loss of the old fishing pier. In the end they voted to "
    "approve a smaller design with a wider footpath.",
    "def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) "
    "and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n"
    "            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n",
    "To compute the area of a triangle with base 12 and height 7, multiply the two "
    "numbers and divide by two. So 12 times 7 is 84, and half of 84 is 42. The area is "
    "therefore 42 square units, which we can check by drawing the triangle on grid paper.",
    "When the storm finally reached the coast, the lighthouse keeper lit the lamp an hour "
    "early. The waves climbed the rocks below and threw white spray against the glass. He "
    "wrote in his logbook that the night was the worst he had seen in thirty years.",
    "SELECT customer_id, SUM(amount) AS total_spent FROM orders WHERE order_date >= "
    "'2025-01-01' GROUP BY customer_id HAVING SUM(amount) > 1000 ORDER BY total_spent "
    "DESC LIMIT 20; This query lists the twenty biggest customers of the year.",
    "Photosynthesis converts light energy into chemical energy stored in glucose. Inside "
    "the chloroplast, chlorophyll absorbs red and blue light and uses it to split water, "
    "releasing oxygen. The resulting energy carriers then drive the fixation of carbon "
    "dioxide in the Calvin cycle.",
    "Dear Ms. Alvarez, thank you for your patience while we investigated the delayed "
    "shipment. Your order left our warehouse on Tuesday and should arrive by Friday. We "
    "have refunded the shipping fee and added a discount code to your account.",
    "The Roman Republic expanded across the Mediterranean through a mix of alliances and "
    "conquest. After the Punic Wars, Carthage was destroyed and Rome controlled the "
    "western sea. Wealth flowed into the city, and with it came deep arguments about land "
    "and power.",
    "fn fibonacci(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n "
    "{\n        let next = a + b;\n        a = b;\n        b = next;\n    }\n    a\n}\n"
    "// Runs in linear time and constant space.",
    "A good sourdough starter needs regular feeding, warmth, and a little patience. Mix "
    "equal weights of flour and water each day, discard most of the old starter, and keep "
    "the jar somewhere around twenty-four degrees. After a week it should smell pleasantly "
    "sour and double in size within hours.",
    "In 1969 the first humans walked on the surface of the Moon. The lunar module landed "
    "in a flat region called the Sea of Tranquility, and the crew spent about two and a "
    "half hours outside collecting rock samples. Millions of people watched the broadcast "
    "on television.",
    "Question: If a train leaves the station at 3 pm travelling at 80 kilometres per hour, "
    "and a second train leaves at 4 pm at 100 kilometres per hour on the same track, when "
    "does the second train catch up? Answer: the first train has an 80 kilometre head "
    "start, closing at 20 kilometres per hour.",
    "The city council announced that the downtown library will stay open until midnight "
    "during exam week. Volunteers will serve free coffee, and extra study rooms can be "
    "reserved online. Officials hope the pilot programme will become permanent next year.",
    "Neural networks learn by adjusting their weights to reduce a loss function. Each "
    "training step computes the gradient of the loss with respect to every weight and "
    "moves the weights a small distance in the opposite direction. Repeating this over "
    "millions of examples gradually produces useful behaviour.",
    "She opened the old wooden chest and found a bundle of letters tied with blue string. "
    "The ink had faded, but the handwriting was still clear, looping and quick. The first "
    "line read simply: I hope this reaches you before the winter does.",
    "import argparse\n\nparser = argparse.ArgumentParser(description='Resize images')\n"
    "parser.add_argument('--width', type=int, default=640)\nparser.add_argument('--height', "
    "type=int, default=480)\nargs = parser.parse_args()\nprint(f'Resizing to "
    "{args.width}x{args.height}')\n",
)
````

- [ ] **Step 4: Modify `scripts/run_baseline.py`**

Apply these five edits.

(a) Add imports. Replace
```python
from dispatch.benchmark.harness import generate_with_timings, load_model
```
with
```python
from dispatch.benchmark.agreement import (
    MIN_GATE_POSITIONS,
    aggregate_gap_split,
    compare_gap_split,
)
from dispatch.benchmark.gate_prompts import GATE_PROMPTS
from dispatch.benchmark.harness import generate_with_timings, load_model
```

(b) Add the flag. Replace
```python
    parser.add_argument(
        "--compare-reference",
```
with
```python
    parser.add_argument(
        "--prompt-set",
        choices=["default", "gate"],
        default="default",
        help="'gate' runs Phase 6's larger fixed prompt set and enforces its gap-split rule",
    )
    parser.add_argument(
        "--compare-reference",
```

(c) Select the prompts. Replace
```python
        prompts=DEFAULT_PROMPTS,
```
with
```python
        prompts=list(GATE_PROMPTS) if args.prompt_set == "gate" else DEFAULT_PROMPTS,
```

(d) Compute the gap split once, from one loaded reference. Replace
```python
    comparison = (
        compare_top_k_agreement(logits, load_reference(args.compare_reference))
        if args.compare_reference is not None
        else {}
    )
```
with
```python
    reference = load_reference(args.compare_reference) if args.compare_reference else None
    comparison = compare_top_k_agreement(logits, reference) if reference is not None else {}
    gap_split = (
        aggregate_gap_split(compare_gap_split(logits, reference).values())
        if reference is not None
        else None
    )
```
and in the results dict, after `"moe_layers_patched": moe_layers_patched,` add
```python
                "prompt_set": args.prompt_set,
                "gap_split": None if gap_split is None else gap_split.to_dict(),
```

(e) Enforce the gate after the existing `mutual_top_k` check. Replace
```python
    if not all(value.mutual_top_k for value in comparison.values()):
        raise SystemExit(
            f"{args.moe_kernel} logits disagree with {args.compare_reference} -- see {results_path}"
        )
```
with
```python
    if not all(value.mutual_top_k for value in comparison.values()):
        raise SystemExit(
            f"{args.moe_kernel} logits disagree with {args.compare_reference} -- see {results_path}"
        )
    if args.prompt_set == "gate" and gap_split is not None:
        if gap_split.positions < MIN_GATE_POSITIONS:
            raise SystemExit(
                f"gate compared only {gap_split.positions} positions, fewer than the "
                f"{MIN_GATE_POSITIONS} an at-scale claim needs -- see {results_path}"
            )
        if gap_split.large_gap_disagreements > 0:
            raise SystemExit(
                f"{args.moe_kernel} flips {gap_split.large_gap_disagreements} position(s) the "
                f"reference was confident about (widest gap {gap_split.max_disagreement_gap:.3f}) "
                f"-- a bug, not a near-tie; see {results_path}"
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gate_prompts.py tests/unit/test_run_baseline.py tests/unit/test_agreement.py -v -m "not slow"`
Expected: all pass (the existing `test_run_baseline.py` tests still pass unmodified).

- [ ] **Step 6: Commit**

STATUS bullet: `- **Task 2 (gate prompt set + run_baseline gate mode)**: --prompt-set gate runs 16 fixed prompts (1,036 positions with the real tokenizer) and fails on any large-gap flip or fewer than 500 positions.`

```bash
make check
git add src/dispatch/benchmark/gate_prompts.py tests/unit/test_gate_prompts.py scripts/run_baseline.py tests/unit/test_run_baseline.py docs/STATUS.md
git commit -m "feat: at-scale gate mode for run_baseline (16 prompts, gap-split rule)"
```

---

### Task 3: `ignore_eos` and a public `percentile`

Dispatch's concurrency-1 reference row must generate exactly 64 tokens per request, as `vllm bench serve --ignore-eos` does, or the throughputs are not comparable. `metrics._percentile` becomes public because Task 8's serving summary reuses it.

**Files:**
- Modify: `src/dispatch/benchmark/harness.py`, `src/dispatch/benchmark/metrics.py`, `scripts/run_baseline.py`
- Modify (insert tests): `tests/unit/test_harness.py`

**Interfaces:**
- Consumes: Task 2's `run_baseline.py`.
- Produces: `generate_with_timings(..., ignore_eos: bool = False)`; `run_baseline(..., ignore_eos: bool = False)`; CLI `--ignore-eos`; results JSON gains `max_new_tokens` and `ignore_eos`; `dispatch.benchmark.metrics.percentile(sorted_values, fraction)`.

- [ ] **Step 1: Write the failing tests**

Insert into `tests/unit/test_harness.py`, immediately above `def test_load_model_omits_attn_implementation_by_default(`:

````python
class _EosAtSecondTokenTokenizer(_FakeTokenizer):
    eos_token_id = 2  # _FakeModel emits token (call_count % vocab): 1, 2, 3, ...


def _run_with_eos_tokenizer(*, ignore_eos: bool) -> int:
    counter = iter(range(100))
    timing = generate_with_timings(
        _FakeModel(),  # type: ignore[arg-type]
        _EosAtSecondTokenTokenizer(),  # type: ignore[arg-type]
        "prompt",
        max_new_tokens=5,
        clock_fn=lambda: float(next(counter)),
        ignore_eos=ignore_eos,
    )
    return timing.generated_token_count


def test_generate_with_timings_stops_at_eos_by_default() -> None:
    assert _run_with_eos_tokenizer(ignore_eos=False) == 2


def test_generate_with_timings_ignore_eos_generates_exactly_max_new_tokens() -> None:
    assert _run_with_eos_tokenizer(ignore_eos=True) == 5
````

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_harness.py -v -m "not slow"`
Expected: the two new tests FAIL with `TypeError: generate_with_timings() got an unexpected keyword argument 'ignore_eos'`.

- [ ] **Step 3: Implement**

`src/dispatch/benchmark/harness.py`: in `generate_with_timings`' signature, add `ignore_eos: bool = False,` after `clock_fn: Callable[[], float] = time.perf_counter,`; and replace
```python
            if eos_token_id is not None and next_token.item() == eos_token_id:
```
with
```python
            if not ignore_eos and eos_token_id is not None and next_token.item() == eos_token_id:
```

`src/dispatch/benchmark/metrics.py`: rename `_percentile` to `percentile` (its definition and its two call sites inside `summarize`). No other module imports it (`scripts/gpu/phase4_concurrency.py` has its own private copy; leave it).

`scripts/run_baseline.py`: (a) add `ignore_eos: bool = False,` after `moe_kernel: str = "none",` in `run_baseline`'s signature; (b) replace
```python
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
```
with
```python
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            ignore_eos=ignore_eos,
        )
```
(c) after the `--max-new-tokens` argument add
```python
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="generate exactly --max-new-tokens even past an EOS (matches vllm bench serve)",
    )
```
(d) pass `ignore_eos=args.ignore_eos,` after `moe_kernel=args.moe_kernel,` in the `run_baseline(...)` call in `main`; (e) in the results dict, after `"prompt_set": args.prompt_set,` add `"max_new_tokens": args.max_new_tokens,` and `"ignore_eos": args.ignore_eos,`.

- [ ] **Step 3b: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_harness.py tests/unit/test_metrics.py tests/unit/test_run_baseline.py -v -m "not slow"`
Expected: all pass.

- [ ] **Step 4: Commit**

STATUS bullet: `- **Task 3 (ignore_eos, public percentile)**: the harness can generate exactly N tokens past EOS, matching vllm bench serve --ignore-eos, so dispatch's concurrency-1 reference row is comparable.`

```bash
make check
git add src/dispatch/benchmark/harness.py src/dispatch/benchmark/metrics.py scripts/run_baseline.py tests/unit/test_harness.py docs/STATUS.md
git commit -m "feat: ignore_eos in the harness so reference rows match vllm bench serve"
```

---
### Task 4: Engine contract, seeded inputs, fp32 reference, dispatch adapter

**Files:**
- Create: `src/dispatch/benchmark/engines/__init__.py` (empty), `src/dispatch/benchmark/engines/base.py`, `src/dispatch/benchmark/engines/dispatch_kernels.py`
- Test: `tests/unit/test_engine_base.py`, `tests/unit/test_dispatch_engine.py`

**Interfaces:**
- Consumes: `dispatch.kernels.bench.sample_topk_idx`, `dispatch.kernels.moe_forward` (`StackedExpertWeights`, `grouped_moe_routed`, `torch_grouped_matmul`, `assert_matches_reference`), `dispatch.kernels.quantization` (`QuantizedStackedExpertWeights`, `dequantize_int8`, `quantize_stacked_weights`, `grouped_moe_routed_quantized`, `torch_grouped_matmul_dequant`), `dispatch.kernels.backends` (`resolve_backend`, `resolve_quantized_backend`).
- Produces (Tasks 5-9 rely on these exact names):
  - `RaceCase(x, topk_idx, topk_weight)`; `BoundLayer = Callable[[], Tensor]`; `LayerFactory = Callable[[RaceCase], BoundLayer]`.
  - `MoEEngine` Protocol: `name: str`, `prepare_bf16(StackedExpertWeights) -> LayerFactory`, `prepare_int8(QuantizedStackedExpertWeights) -> LayerFactory`. `prepare_*` does one-time weight conversion; `bind` (the factory call) does per-case dtype casts; only the returned closure is ever timed.
  - `make_weights(dtype, *, seed, device, hidden_size=..., intermediate_size=..., n_experts=...)`, `make_case(num_tokens, distribution, *, dtype, seed, device, hidden_size=..., n_experts=..., top_k=...)`, `reference_output(case, weights)`, `dequantized_weights(qweights)`, `fuse_gate_up(gate, up)`; constants `HIDDEN_SIZE`, `MOE_INTERMEDIATE_SIZE`, `N_ROUTED_EXPERTS`, `NUM_EXPERTS_PER_TOK`, `TOKEN_COUNTS`.
  - `DispatchEngine(backend, *, block_m=16)` with `name == f"dispatch-{backend}-bm{block_m}"`; backends `torch` (CPU, for tests), `naive`, `persistent`; int8 supported for `torch` and `naive` only.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_engine_base.py`:

````python
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.benchmark.engines.base import (
    dequantized_weights,
    fuse_gate_up,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def test_make_weights_is_seeded_and_shaped_from_the_dims() -> None:
    first = make_weights(torch.float32, seed=3, device="cpu", **DIMS)
    again = make_weights(torch.float32, seed=3, device="cpu", **DIMS)
    other = make_weights(torch.float32, seed=4, device="cpu", **DIMS)

    assert first.gate.shape == (6, 8, 16)
    assert first.up.shape == (6, 8, 16)
    assert first.down.shape == (6, 16, 8)
    assert torch.equal(first.gate, again.gate)
    assert not torch.equal(first.gate, other.gate)


def test_uniform_and_zipf_cases_share_inputs_and_differ_only_in_routing() -> None:
    uniform = make_case(64, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    zipf = make_case(64, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    assert torch.equal(uniform.x, zipf.x)
    assert torch.equal(uniform.topk_weight, zipf.topk_weight)
    assert not torch.equal(uniform.topk_idx, zipf.topk_idx)
    assert uniform.topk_weight.dtype == torch.float32
    assert uniform.topk_idx.dtype == torch.int64


def test_zipf_routing_concentrates_load_on_low_experts() -> None:
    uniform = make_case(512, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    zipf = make_case(512, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    uniform_load = torch.bincount(uniform.topk_idx.reshape(-1), minlength=6)
    zipf_load = torch.bincount(zipf.topk_idx.reshape(-1), minlength=6)
    assert zipf_load[0] > zipf_load[-1] * 2
    assert uniform_load.max() < uniform_load.min() * 2


def test_fuse_gate_up_puts_gate_first_and_is_the_layout_silu_and_mul_reads() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    fused = fuse_gate_up(weights.gate, weights.up)
    assert fused.shape == (6, 16, 16)
    assert torch.equal(fused[:, :8], weights.gate)
    assert torch.equal(fused[:, 8:], weights.up)

    x = torch.randn(3, 16)
    expert = 2
    hidden = x @ fused[expert].T
    fused_activation = F.silu(hidden[:, :8]) * hidden[:, 8:]
    separate_activation = F.silu(x @ weights.gate[expert].T) * (x @ weights.up[expert].T)
    torch.testing.assert_close(fused_activation, separate_activation)


def test_fuse_gate_up_also_fuses_per_channel_scales() -> None:
    gate_scale, up_scale = torch.ones(6, 8), torch.full((6, 8), 2.0)
    fused = fuse_gate_up(gate_scale, up_scale)
    assert fused.shape == (6, 16)
    assert torch.equal(fused[:, :8], gate_scale)
    assert torch.equal(fused[:, 8:], up_scale)


def test_reference_output_is_float32_whatever_the_case_dtype() -> None:
    weights = make_weights(torch.bfloat16, seed=0, device="cpu", **DIMS)
    case = make_case(5, "uniform", dtype=torch.bfloat16, seed=0, device="cpu", **CASE_DIMS)

    assert reference_output(case, weights).dtype == torch.float32


def test_int8_reference_uses_the_dequantized_weights_not_the_originals() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(5, "uniform", dtype=torch.float32, seed=0, device="cpu", **CASE_DIMS)

    int8_reference = reference_output(case, dequantized_weights(qweights))
    original_reference = reference_output(case, weights)

    assert not torch.equal(int8_reference, original_reference)
    assert_matches_reference(int8_reference, original_reference)  # close, not identical
    with pytest.raises(AssertionError):
        torch.testing.assert_close(int8_reference, original_reference, rtol=0, atol=0)
````

Create `tests/unit/test_dispatch_engine.py`:

````python
from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.engines.base import (
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.dispatch_kernels import DispatchEngine
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def test_name_carries_backend_and_tile_size() -> None:
    assert DispatchEngine("naive", block_m=32).name == "dispatch-naive-bm32"


def test_bf16_layer_matches_the_fp32_reference() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = DispatchEngine("torch").prepare_bf16(weights)(case)

    assert_matches_reference(layer(), reference_output(case, weights))


def test_bound_layer_is_repeatable_so_it_can_be_timed() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    layer = DispatchEngine("torch").prepare_bf16(weights)(case)

    assert torch.equal(layer(), layer())


def test_int8_layer_matches_the_dequantized_reference() -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = DispatchEngine("torch").prepare_int8(qweights)(case)

    assert_matches_reference(layer(), reference_output(case, dequantized_weights(qweights)))


def test_persistent_backend_has_no_int8_kernel() -> None:
    qweights = quantize_stacked_weights(make_weights(torch.float32, seed=0, device="cpu", **DIMS))

    with pytest.raises(ValueError, match="no int8 kernel"):
        DispatchEngine("persistent").prepare_int8(qweights)


def test_a_layer_that_disagrees_with_the_reference_is_detectable() -> None:
    """The race driver's refuse-not-time gate depends on assert_matches_reference
    rejecting a wrong layer; prove it does for this adapter's output shape."""
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    wrong = DispatchEngine("torch").prepare_bf16(weights)(case)() * 1.5

    with pytest.raises(AssertionError):
        assert_matches_reference(wrong, reference_output(case, weights))
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_engine_base.py tests/unit/test_dispatch_engine.py -v`
Expected: collection FAIL with `ModuleNotFoundError: No module named 'dispatch.benchmark.engines'`.

- [ ] **Step 3: Write the implementation**

Create the empty package marker, then the two modules:

```bash
mkdir -p src/dispatch/benchmark/engines && touch src/dispatch/benchmark/engines/__init__.py
```

`src/dispatch/benchmark/engines/base.py`:

````python
"""The contract every Phase 6 race contestant implements, plus the seeded
inputs and fp32 reference they are all checked against.

An engine is anything that computes DeepSeek's routed-expert MoE forward from
`(x, topk_idx, topk_weight)` and a set of expert weights. `prepare_*` does
the one-time weight conversion (fusing gate+up, uploading scales), and
`bind` does the per-case input conversion (dtype casts) -- both untimed --
so the closure a benchmark times is only the engine's own routed-MoE call,
never a cast the engine's real serving path wouldn't pay.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from dispatch.kernels.bench import sample_topk_idx
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    grouped_moe_routed,
    torch_grouped_matmul,
)
from dispatch.kernels.quantization import QuantizedStackedExpertWeights, dequantize_int8

# deepseek-ai/deepseek-moe-16b-base config.json, checked live 2026-09-15
# (same constants as scripts/run_kernel_bench.py).
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 1408
N_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 6

# Phase 1's default sweep (1, 16, 128, 512, 2048) plus 4 and 64, so results
# extend Phase 1's tables and decode-sized batches are better covered.
TOKEN_COUNTS = (1, 4, 16, 64, 128, 512, 2048)


@dataclass(frozen=True)
class RaceCase:
    """One benchmark input. `topk_weight` is float32 by contract; an adapter
    casts it to whatever its engine wants inside `bind`."""

    x: torch.Tensor  # (num_tokens, hidden)
    topk_idx: torch.Tensor  # (num_tokens, top_k) int64
    topk_weight: torch.Tensor  # (num_tokens, top_k) float32


BoundLayer = Callable[[], torch.Tensor]
LayerFactory = Callable[[RaceCase], BoundLayer]


class MoEEngine(Protocol):
    name: str

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory: ...

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory: ...


def make_weights(  # noqa: PLR0913 -- the real DeepSeek dims are the defaults; tests shrink them
    dtype: torch.dtype,
    *,
    seed: int,
    device: str,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = MOE_INTERMEDIATE_SIZE,
    n_experts: int = N_ROUTED_EXPERTS,
) -> StackedExpertWeights:
    generator = torch.Generator(device=device).manual_seed(seed)

    def stack(n: int, k: int) -> torch.Tensor:
        weights = torch.randn(n_experts, n, k, device=device, dtype=dtype, generator=generator)
        return weights / math.sqrt(k)

    return StackedExpertWeights(
        gate=stack(intermediate_size, hidden_size),
        up=stack(intermediate_size, hidden_size),
        down=stack(hidden_size, intermediate_size),
    )


def make_case(  # noqa: PLR0913 -- the real DeepSeek dims are the defaults; tests shrink them
    num_tokens: int,
    distribution: str,
    *,
    dtype: torch.dtype,
    seed: int,
    device: str,
    hidden_size: int = HIDDEN_SIZE,
    n_experts: int = N_ROUTED_EXPERTS,
    top_k: int = NUM_EXPERTS_PER_TOK,
) -> RaceCase:
    """Uniform and zipf cases at the same `seed` and `num_tokens` share x and
    topk_weight exactly; only the routing differs, so the two distributions
    are a controlled comparison."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, generator=generator)
    topk_idx = sample_topk_idx(
        num_tokens,
        n_experts,
        top_k,
        distribution=distribution,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    topk_weight = torch.rand(
        num_tokens, top_k, device=device, dtype=torch.float32, generator=generator
    )
    return RaceCase(x=x, topk_idx=topk_idx, topk_weight=topk_weight)


def dequantized_weights(qweights: QuantizedStackedExpertWeights) -> StackedExpertWeights:
    """float32 weights equal to what an int8 kernel effectively multiplies by:
    the int8 race's reference is the *same* quantized weights, dequantized,
    so nothing should diverge beyond float precision."""
    return StackedExpertWeights(
        gate=dequantize_int8(qweights.gate),
        up=dequantize_int8(qweights.up),
        down=dequantize_int8(qweights.down),
    )


def reference_output(case: RaceCase, weights: StackedExpertWeights) -> torch.Tensor:
    """The fp32 eager reference every engine is checked against: the same
    per-expert-loop contract DeepseekMoE.moe_infer implements, in float32."""
    weights_fp32 = StackedExpertWeights(
        gate=weights.gate.float(), up=weights.up.float(), down=weights.down.float()
    )
    return grouped_moe_routed(
        case.x.float(), case.topk_idx, case.topk_weight.float(), weights_fp32, torch_grouped_matmul
    )


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(E, N, K) gate and up -> (E, 2N, K) with gate's rows first. Works for
    (E, N) per-channel scales too (-> (E, 2N)). This is the layout vLLM's and
    SGLang's fused-MoE kernels read: silu(first half) * second half."""
    return torch.cat([gate, up], dim=1)
````

`src/dispatch/benchmark/engines/dispatch_kernels.py`:

````python
"""dispatch's own kernels as a race contestant: the naive or persistent
Triton grouped-GEMM (bf16), or the int8 weight-only kernel, behind the
common engine contract. `torch` is the eager backend, which runs on CPU and
is what the unit tests use."""

from __future__ import annotations

import functools

import torch

from dispatch.benchmark.engines.base import LayerFactory, RaceCase
from dispatch.kernels.backends import resolve_backend, resolve_quantized_backend
from dispatch.kernels.moe_forward import StackedExpertWeights, grouped_moe_routed
from dispatch.kernels.quantization import (
    QuantizedGroupedMatmul,
    QuantizedStackedExpertWeights,
    grouped_moe_routed_quantized,
    torch_grouped_matmul_dequant,
)


class DispatchEngine:
    def __init__(self, backend: str, *, block_m: int = 16) -> None:
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
        if self._backend == "torch":
            return torch_grouped_matmul_dequant
        if self._backend == "naive":
            return resolve_quantized_backend()
        raise ValueError(
            f"no int8 kernel for backend {self._backend!r}: Phase 5a's int8 kernel is "
            "built on the naive launch order only"
        )
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_engine_base.py tests/unit/test_dispatch_engine.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

STATUS bullet: `- **Task 4 (engine contract + dispatch adapter)**: seeded inputs, fp32 reference, fused gate+up layout, and dispatch's kernels behind one MoEEngine contract; uniform and zipf cases share x and weights so only routing differs.`

```bash
make check
git add src/dispatch/benchmark/engines tests/unit/test_engine_base.py tests/unit/test_dispatch_engine.py docs/STATUS.md
git commit -m "feat: race engine contract, seeded inputs, fp32 reference, dispatch adapter"
```

---

### Task 5: vLLM and SGLang adapters, and the engine registry

The adapters call each engine's own `fused_experts` directly with the race's fixed `topk_ids`/`topk_weights` (no gating), so routing is identical for every contestant. APIs checked live 2026-09-19 against vllm-project/vllm `main` and 0.29.0, and sgl-project/sglang `main` and 0.5.20:

- vLLM: `vllm.model_executor.layers.fused_moe.fused_moe.fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ..., quant_config=)`; int8 via `vllm.model_executor.layers.fused_moe.config.int8_w8a16_moe_quant_config(w1_scale, w2_scale, ...)`.
- SGLang: `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts(hidden_states, w1, w2, topk_output, moe_runner_config, ..., use_int8_w8a16, per_channel_quant, w1_scale, w2_scale)` with `topk_output = sglang.srt.layers.moe.topk.StandardTopKOutput(topk_weights, topk_ids, router_logits)` and `moe_runner_config = sglang.srt.layers.moe.moe_runner.base.MoeRunnerConfig(...)`. **`inplace=False` is essential** (the default overwrites `x`). SGLang's fused-MoE reads its tensor-parallel group even on one GPU, so `init_distributed()` (mirroring SGLang's own benchmark scripts) must run first.

These are the API shapes on `main`; the pinned release may differ. That is what Task 10's real-engine test exists to find. On this dev machine neither engine is installed, so the CPU tests inject fakes that reproduce each engine's documented layout contract in eager PyTorch.

**Files:**
- Create: `src/dispatch/benchmark/engines/vllm_moe.py`, `src/dispatch/benchmark/engines/sglang_moe.py`, `src/dispatch/benchmark/engines/registry.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_engine_adapters.py`

**Interfaces:**
- Consumes: Task 4's `MoEEngine`, `RaceCase`, `BoundLayer`, `LayerFactory`, `fuse_gate_up`, `DispatchEngine`.
- Produces: `VllmEngine(*, fused_experts=None, int8_quant_config=None)` and `SglangEngine(*, fused_experts=None, topk_output_cls=None, runner_config_cls=None)` (each injectable for tests, lazily importing the real engine otherwise); `sglang_moe.init_distributed() -> None`; `registry.ENGINE_NAMES = ("dispatch-naive", "dispatch-persistent", "vllm", "sglang")`, `registry.DEFAULT_BLOCK_MS = (16, 32, 64, 128)`, `registry.build_engines(engine: str, block_ms: Sequence[int] = DEFAULT_BLOCK_MS) -> list[MoEEngine]` (dispatch returns one contestant per tile size; vllm and sglang exactly one).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_engine_adapters.py`:

````python
"""vLLM's and SGLang's adapters, driven through fakes that reproduce each
engine's documented fused-layout contract (gate rows first in w1, per-channel
int8 scales). The fakes are eager PyTorch, so a wrong layout, a wrong scale
shape, or a mutated input in the adapter changes the output and fails here;
the real engines are checked by the `gpu`-marked test on the pod."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention

from dispatch.benchmark.engines import registry
from dispatch.benchmark.engines.base import (
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.sglang_moe import SglangEngine
from dispatch.benchmark.engines.vllm_moe import VllmEngine
from dispatch.kernels.moe_forward import assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

DIMS = {"hidden_size": 16, "intermediate_size": 8, "n_experts": 6}
CASE_DIMS = {"hidden_size": 16, "n_experts": 6, "top_k": 2}


def _eager_fused_moe(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """What vLLM's and SGLang's fused-MoE compute, in eager float32: w1 holds
    gate rows then up rows, activation is silu(gate) * up, int8 weights are
    scaled per output channel."""
    w1_f, w2_f = w1.float(), w2.float()
    if w1_scale is not None and w2_scale is not None:
        w1_f, w2_f = w1_f * w1_scale.unsqueeze(-1), w2_f * w2_scale.unsqueeze(-1)
    n = w2.shape[2]
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            hidden = x[token].float() @ w1_f[expert].T
            activated = F.silu(hidden[:n]) * hidden[n:]
            out[token] += topk_weights[token, slot].float() * (activated @ w2_f[expert].T)
    return out


def _fake_vllm_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_config: Any = None,
) -> torch.Tensor:
    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    scales = (
        (None, None) if quant_config is None else (quant_config.w1_scale, quant_config.w2_scale)
    )
    return _eager_fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, *scales)


def _fake_vllm_int8_quant_config(*, w1_scale: torch.Tensor, w2_scale: torch.Tensor) -> Any:
    return SimpleNamespace(w1_scale=w1_scale, w2_scale=w2_scale)


class _FakeTopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


class _FakeRunnerConfig:
    inplace: bool

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _fake_sglang_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: _FakeTopKOutput,
    moe_runner_config: _FakeRunnerConfig,
    use_int8_w8a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert moe_runner_config.inplace is False  # the default would overwrite case.x
    assert topk_output.topk_ids.dtype == torch.int32
    if use_int8_w8a16:
        assert per_channel_quant
    return _eager_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_output.topk_weights,
        topk_output.topk_ids,
        w1_scale,
        w2_scale,
    )


def _vllm() -> VllmEngine:
    return VllmEngine(
        fused_experts=_fake_vllm_fused_experts, int8_quant_config=_fake_vllm_int8_quant_config
    )


def _sglang() -> SglangEngine:
    return SglangEngine(
        fused_experts=_fake_sglang_fused_experts,
        topk_output_cls=_FakeTopKOutput,
        runner_config_cls=_FakeRunnerConfig,
    )


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_bf16_layout_conversion_reproduces_the_reference(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "zipf", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_bf16(weights)(case)

    assert_matches_reference(layer(), reference_output(case, weights))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_int8_scale_layout_reproduces_the_dequantized_reference(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    qweights = quantize_stacked_weights(weights)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_int8(qweights)(case)

    assert_matches_reference(layer(), reference_output(case, dequantized_weights(qweights)))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_a_swapped_gate_and_up_layout_is_caught(make_engine: Any) -> None:
    """The failure this whole adapter layer exists to prevent: fusing up
    before gate silently computes silu(up) * gate. The reference check must
    reject it."""
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    swapped = type(weights)(gate=weights.up, up=weights.gate, down=weights.down)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)

    layer = make_engine().prepare_bf16(swapped)(case)

    with pytest.raises(AssertionError):
        assert_matches_reference(layer(), reference_output(case, weights))


@pytest.mark.parametrize("make_engine", [_vllm, _sglang], ids=["vllm", "sglang"])
def test_bound_layer_does_not_mutate_the_case_and_is_repeatable(make_engine: Any) -> None:
    weights = make_weights(torch.float32, seed=0, device="cpu", **DIMS)
    case = make_case(9, "uniform", dtype=torch.float32, seed=1, device="cpu", **CASE_DIMS)
    x_before = case.x.clone()
    layer = make_engine().prepare_bf16(weights)(case)

    assert torch.equal(layer(), layer())
    assert torch.equal(case.x, x_before)


def test_registry_builds_a_tile_size_sweep_for_dispatch_and_one_engine_otherwise() -> None:
    naive = registry.build_engines("dispatch-naive", (16, 64))
    assert [engine.name for engine in naive] == ["dispatch-naive-bm16", "dispatch-naive-bm64"]
    assert len(registry.build_engines("vllm")) == 1
    assert len(registry.build_engines("sglang")) == 1


def test_registry_rejects_unknown_engines() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        registry.build_engines("tensorrt")
    with pytest.raises(ValueError, match="unknown engine"):
        registry.build_engines("dispatch-cuda")
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_engine_adapters.py -v`
Expected: collection FAIL with `ModuleNotFoundError: No module named 'dispatch.benchmark.engines.vllm_moe'`.

- [ ] **Step 3: Write the implementation**

`src/dispatch/benchmark/engines/vllm_moe.py`:

````python
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
````

`src/dispatch/benchmark/engines/sglang_moe.py`:

````python
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
````

`src/dispatch/benchmark/engines/registry.py`:

````python
"""Engine name -> contestants. vLLM and SGLang import lazily, so this module
is importable (and the driver's tests run) without either installed."""

from __future__ import annotations

from collections.abc import Sequence

from dispatch.benchmark.engines.base import MoEEngine
from dispatch.benchmark.engines.dispatch_kernels import DispatchEngine

ENGINE_NAMES = ("dispatch-naive", "dispatch-persistent", "vllm", "sglang")
DEFAULT_BLOCK_MS = (16, 32, 64, 128)


def build_engines(engine: str, block_ms: Sequence[int] = DEFAULT_BLOCK_MS) -> list[MoEEngine]:
    """dispatch-* returns one contestant per tile size in `block_ms` (its own
    tuning sweep); vllm and sglang return exactly one, tuned by their own
    tuners through their own config folders, not by this call."""
    if engine.startswith("dispatch-"):
        backend = engine.removeprefix("dispatch-")
        if f"dispatch-{backend}" not in ENGINE_NAMES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
        return [DispatchEngine(backend, block_m=block_m) for block_m in block_ms]
    if engine == "vllm":
        from dispatch.benchmark.engines.vllm_moe import VllmEngine  # noqa: PLC0415

        return [VllmEngine()]
    if engine == "sglang":
        from dispatch.benchmark.engines.sglang_moe import SglangEngine  # noqa: PLC0415

        return [SglangEngine()]
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
````

- [ ] **Step 4: Update `pyproject.toml`**

vLLM and SGLang exist only in the pod's engines venv, so mypy must treat them as `Any`, the same boundary already drawn for `triton` and `deep_ep`. Add this block immediately above `[tool.coverage.run]`:

```toml
# vLLM and SGLang are only installed in Phase 6's rented-pod engine venv, never
# on this dev box or CI (they pin torch==2.13.0 and need a CUDA build), so their
# adapters import them lazily and mypy treats them as Any, same boundary as
# triton and deep_ep above.
[[tool.mypy.overrides]]
module = ["vllm", "vllm.*", "sglang", "sglang.*"]
ignore_missing_imports = true
follow_imports = "skip"
```

The fake-engine tests take many positional arguments by construction, so extend the existing test ignore (`PLR0917` is "too many positional arguments"; `PLR0913` is already there). Replace
```toml
"tests/**/*.py" = ["PLR2004", "PLR0913"]
```
with
```toml
"tests/**/*.py" = ["PLR2004", "PLR0913", "PLR0917"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_engine_adapters.py -v`
Expected: 10 passed, including `test_a_swapped_gate_and_up_layout_is_caught` for both engines (the silent-wrong-activation failure this layer exists to prevent).

- [ ] **Step 6: Commit**

STATUS bullet: `- **Task 5 (vLLM/SGLang adapters + registry)**: each engine behind one adapter calling its own fused_experts with fixed routing; verified against eager fakes only -- the real engines are not exercised until the pod (Task 10).`

```bash
make check
git add src/dispatch/benchmark/engines pyproject.toml tests/unit/test_engine_adapters.py docs/STATUS.md
git commit -m "feat: vLLM and SGLang fused-MoE adapters behind the race contract"
```

---

### Task 6: Race summarizer (tuning rule, rows, markdown)

**Files:**
- Create: `src/dispatch/benchmark/race_summary.py`
- Test: `tests/unit/test_race_summary.py`

**Interfaces:**
- Consumes: `TOKEN_COUNTS` from Task 4. Reads the per-engine record JSON that Task 7's `run` writes: `{"config": {"engine", "precision", "tuning_label", ...}, "results": [{"num_tokens", "distribution", "variant", "status", "mean_ms", "p50_ms", "p99_ms", "tflops"}]}`.
- Produces: `RaceRow` (frozen dataclass), `summarize_race(records) -> list[RaceRow]`, `render_markdown(rows) -> str`, `DEFAULT_BLOCK_M = 16`, `TUNING_DISTRIBUTION = "uniform"`. Encodes the pre-registered tuning rule: dispatch's "tuned" variant is chosen on uniform results only and reused for zipf.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_race_summary.py`:

````python
from __future__ import annotations

from typing import Any

from dispatch.benchmark.race_summary import render_markdown, summarize_race


def _result(
    variant: str, tokens: int, distribution: str, mean_ms: float, status: str = "ok"
) -> dict[str, Any]:
    return {
        "num_tokens": tokens,
        "distribution": distribution,
        "variant": variant,
        "status": status,
        "mean_ms": mean_ms,
        "p50_ms": mean_ms,
        "p99_ms": mean_ms * 1.2,
        "tflops": 1.0,
    }


def _record(
    engine: str, results: list[dict[str, Any]], *, tuning_label: str = "sweep"
) -> dict[str, Any]:
    return {
        "config": {"engine": engine, "precision": "bf16", "tuning_label": tuning_label},
        "results": results,
    }


def _dispatch_record() -> dict[str, Any]:
    return _record(
        "dispatch-naive",
        [
            _result("dispatch-naive-bm16", 16, "uniform", 1.0),
            _result("dispatch-naive-bm64", 16, "uniform", 0.6),
            _result("dispatch-naive-bm16", 16, "zipf", 1.1),
            _result("dispatch-naive-bm64", 16, "zipf", 0.9),
        ],
    )


def test_dispatch_default_is_the_kernels_own_tile_size() -> None:
    rows = summarize_race([_dispatch_record()])

    default = [row for row in rows if row.tuning == "default"]
    assert {row.variant for row in default} == {"dispatch-naive-bm16"}
    assert {row.distribution for row in default} == {"uniform", "zipf"}


def test_dispatch_tuned_is_picked_on_uniform_and_reused_for_zipf() -> None:
    rows = summarize_race([_dispatch_record()])

    tuned = {row.distribution: row for row in rows if row.tuning == "tuned"}
    assert tuned["uniform"].variant == "dispatch-naive-bm64"
    # Not re-tuned on zipf, even though bm16's zipf number (1.1) is worse
    # than bm64's (0.9) only by coincidence here: the rule is uniform-only.
    assert tuned["zipf"].variant == "dispatch-naive-bm64"
    assert tuned["zipf"].mean_ms == 0.9


def test_the_tuning_choice_never_looks_at_zipf_results() -> None:
    record = _record(
        "dispatch-naive",
        [
            _result("dispatch-naive-bm16", 16, "uniform", 1.0),
            _result("dispatch-naive-bm64", 16, "uniform", 1.1),
            _result("dispatch-naive-bm16", 16, "zipf", 5.0),
            _result("dispatch-naive-bm64", 16, "zipf", 0.1),  # far better on zipf
        ],
    )

    tuned = {row.distribution: row for row in summarize_race([record]) if row.tuning == "tuned"}

    assert tuned["zipf"].variant == "dispatch-naive-bm16"


def test_engines_with_their_own_tuner_keep_the_label_they_ran_under() -> None:
    rows = summarize_race(
        [
            _record("vllm", [_result("vllm", 16, "uniform", 0.5)], tuning_label="tuned"),
            _record("vllm", [_result("vllm", 16, "uniform", 0.8)], tuning_label="default"),
        ]
    )

    assert {(row.tuning, row.mean_ms) for row in rows} == {("tuned", 0.5), ("default", 0.8)}


def test_refused_results_never_reach_a_table() -> None:
    record = _record("vllm", [_result("vllm", 16, "uniform", 0.5, status="refused")])

    assert summarize_race([record]) == []


def test_markdown_has_one_table_per_precision_and_tuning_with_a_column_per_engine() -> None:
    rows = summarize_race(
        [
            _dispatch_record(),
            _record("vllm", [_result("vllm", 16, "uniform", 0.5)], tuning_label="tuned"),
        ]
    )

    text = render_markdown(rows)

    assert "### bf16, tuned" in text
    assert "### bf16, default" in text
    assert "| tokens | routing | dispatch-naive | vllm |" in text
    assert "| 16 | uniform | 0.600 / 0.720 | 0.500 / 0.600 |" in text
    assert "| 16 | zipf | 0.900 / 1.080 | n/a |" in text
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_race_summary.py -v`
Expected: collection FAIL, `ModuleNotFoundError: No module named 'dispatch.benchmark.race_summary'`.

- [ ] **Step 3: Write the implementation**

`src/dispatch/benchmark/race_summary.py`:

````python
"""Turns the race driver's per-engine JSON records into comparable rows and a
markdown table. Pure, so the selection rule is unit-tested without a GPU.

Tuning rule, fixed before any run (docs/plans/2026-09-19-phase-6-final-
benchmark-plan.md): every contestant is tuned under *uniform* routing, the
distribution the engines' own tuners use, and then timed under both
distributions. vLLM and SGLang are tuned by their own tuners (one run per
tuning label); dispatch sweeps its tile size `block_m`, and its "tuned" row
is the variant fastest on uniform routing at that token count, applied to the
zipf row too. Its "default" row is `block_m` 16, the kernel's default.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dispatch.benchmark.engines.base import TOKEN_COUNTS

DEFAULT_BLOCK_M = 16
TUNING_DISTRIBUTION = "uniform"


@dataclass(frozen=True)
class RaceRow:
    precision: str
    num_tokens: int
    distribution: str
    engine: str
    tuning: str  # "default" | "tuned"
    variant: str
    mean_ms: float
    p50_ms: float
    p99_ms: float
    tflops: float


def summarize_race(records: Iterable[dict[str, Any]]) -> list[RaceRow]:
    """Refused results are excluded (the driver already exits non-zero on
    them); only measured rows reach a table."""
    rows: list[RaceRow] = []
    for record in records:
        config = record["config"]
        ok = [result for result in record["results"] if result["status"] == "ok"]
        if config["engine"].startswith("dispatch-"):
            rows.extend(_dispatch_rows(config, ok))
        else:
            rows.extend(
                _row(config, result, tuning=config["tuning_label"], variant=result["variant"])
                for result in ok
            )
    return rows


def render_markdown(rows: list[RaceRow]) -> str:
    """One table per (precision, tuning): a row per (tokens, distribution), a
    column per engine, each cell 'mean / p99' in milliseconds."""
    blocks: list[str] = []
    for precision, tuning in sorted({(row.precision, row.tuning) for row in rows}):
        subset = [row for row in rows if (row.precision, row.tuning) == (precision, tuning)]
        engines = sorted({row.engine for row in subset})
        lookup = {(row.num_tokens, row.distribution, row.engine): row for row in subset}
        keys = sorted(
            {(row.num_tokens, row.distribution) for row in subset},
            key=lambda key: (_token_rank(key[0]), key[1]),
        )
        lines = [
            f"### {precision}, {tuning} (mean / p99 ms per routed-MoE layer call)",
            "",
            "| tokens | routing | " + " | ".join(engines) + " |",
            "|---|---|" + "---|" * len(engines),
        ]
        for tokens, distribution in keys:
            cells = [
                f"{found.mean_ms:.3f} / {found.p99_ms:.3f}"
                if (found := lookup.get((tokens, distribution, engine)))
                else "n/a"
                for engine in engines
            ]
            lines.append(f"| {tokens} | {distribution} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _dispatch_rows(config: dict[str, Any], ok: list[dict[str, Any]]) -> list[RaceRow]:
    rows: list[RaceRow] = []
    for num_tokens in sorted({result["num_tokens"] for result in ok}):
        at_tokens = [result for result in ok if result["num_tokens"] == num_tokens]
        default = [r for r in at_tokens if _block_m(r["variant"]) == DEFAULT_BLOCK_M]
        rows.extend(
            _row(config, result, tuning="default", variant=result["variant"]) for result in default
        )
        tuning_pool = [r for r in at_tokens if r["distribution"] == TUNING_DISTRIBUTION]
        if not tuning_pool:
            continue
        best_variant = min(tuning_pool, key=lambda r: r["mean_ms"])["variant"]
        rows.extend(
            _row(config, result, tuning="tuned", variant=best_variant)
            for result in at_tokens
            if result["variant"] == best_variant
        )
    return rows


def _row(config: dict[str, Any], result: dict[str, Any], *, tuning: str, variant: str) -> RaceRow:
    return RaceRow(
        precision=config["precision"],
        num_tokens=result["num_tokens"],
        distribution=result["distribution"],
        engine=config["engine"],
        tuning=tuning,
        variant=variant,
        mean_ms=result["mean_ms"],
        p50_ms=result["p50_ms"],
        p99_ms=result["p99_ms"],
        tflops=result["tflops"],
    )


def _block_m(variant: str) -> int:
    return int(variant.rsplit("-bm", 1)[1])


def _token_rank(num_tokens: int) -> int:
    return TOKEN_COUNTS.index(num_tokens) if num_tokens in TOKEN_COUNTS else len(TOKEN_COUNTS)
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_race_summary.py -v`
Expected: 6 passed, including `test_the_tuning_choice_never_looks_at_zipf_results`.

- [ ] **Step 5: Commit**

STATUS bullet: `- **Task 6 (race summarizer)**: the pre-registered tuning rule as code -- dispatch's tuned tile size is picked on uniform routing only and reused for zipf; refused results never reach a table.`

```bash
make check
git add src/dispatch/benchmark/race_summary.py tests/unit/test_race_summary.py docs/STATUS.md
git commit -m "feat: race summarizer encoding the uniform-only tuning rule"
```

---

### Task 7: Race driver (`prepare` / `run` / `merge`)

**Files:**
- Create: `scripts/run_engine_race.py`
- Test: `tests/unit/test_run_engine_race.py`

**Interfaces:**
- Consumes: Tasks 4-6 (`make_weights`, `make_case`, `reference_output`, `dequantized_weights`, `registry`, `summarize_race`, `render_markdown`), `dispatch.kernels.bench` (`time_grouped_gemm`, `grouped_gemm_flops`, `DO_BENCH_*`, `ROUTING_DISTRIBUTIONS`, `KernelBenchmarkSummary`).
- Produces: CLI `python -m scripts.run_engine_race {prepare,run,merge}`; `prepare_inputs(out_dir, *, num_tokens, distributions, dtype, seed, device)`; `run_engines(inputs_dir, engines, *, precision, device, time_fn=None) -> list[dict]` (each result `status` is `ok` or `refused`); `describe_environment(device) -> dict`. `run` writes `<output-dir>/<label>.json` and then **exits non-zero if any result was refused**, so the evidence survives. Inputs and references are written once by `prepare` and loaded by every `run`, so all contestants see byte-identical tensors.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_run_engine_race.py`:

````python
"""The driver's plumbing and its refuse-not-time rule, on CPU with a fake
timer and the eager `torch` dispatch backend (the real timing and the real
engines run in the Phase 6 pod session)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_engine_race as race_module
import torch
from scripts.run_engine_race import main, prepare_inputs, run_engines

from dispatch.benchmark.engines import registry
from dispatch.benchmark.engines.base import LayerFactory, RaceCase, make_case, make_weights
from dispatch.benchmark.engines.dispatch_kernels import DispatchEngine
from dispatch.kernels.bench import KernelBenchmarkSummary
from dispatch.kernels.moe_forward import StackedExpertWeights
from dispatch.kernels.quantization import QuantizedStackedExpertWeights


def _fake_timer(bound: object, label: str, flops: float) -> KernelBenchmarkSummary:
    return KernelBenchmarkSummary(
        label=label, mean_latency_ms=2.0, p50_latency_ms=1.9, p99_latency_ms=2.5, tflops=1.0
    )


@pytest.fixture
def small_dims(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink DeepSeek's real dims (2048/1408/64 experts) so prepare_inputs
    runs in milliseconds; the manifest and FLOP counts follow the shrunk dims."""
    real_weights, real_case = make_weights, make_case

    def small_weights(dtype: torch.dtype, *, seed: int, device: str) -> StackedExpertWeights:
        return real_weights(
            dtype, seed=seed, device=device, hidden_size=16, intermediate_size=8, n_experts=6
        )

    def small_case(
        tokens: int, distribution: str, *, dtype: torch.dtype, seed: int, device: str
    ) -> RaceCase:
        return real_case(
            tokens,
            distribution,
            dtype=dtype,
            seed=seed,
            device=device,
            hidden_size=16,
            n_experts=6,
            top_k=2,
        )

    for name, value in (
        ("HIDDEN_SIZE", 16),
        ("MOE_INTERMEDIATE_SIZE", 8),
        ("N_ROUTED_EXPERTS", 6),
        ("NUM_EXPERTS_PER_TOK", 2),
    ):
        monkeypatch.setattr(race_module, name, value)
    monkeypatch.setattr(race_module, "make_weights", small_weights)
    monkeypatch.setattr(race_module, "make_case", small_case)


class _WrongEngine:
    """An engine that returns plausible but wrong output."""

    name = "wrong-engine"

    def prepare_bf16(self, weights: StackedExpertWeights) -> LayerFactory:
        return lambda case: lambda: torch.ones_like(case.x, dtype=torch.float32)

    def prepare_int8(self, qweights: QuantizedStackedExpertWeights) -> LayerFactory:
        return self.prepare_bf16(None)  # type: ignore[arg-type]


def test_prepare_writes_weights_manifest_and_one_file_per_case(
    small_dims: None, tmp_path: Path
) -> None:
    prepare_inputs(
        tmp_path,
        num_tokens=[3, 5],
        distributions=["uniform", "zipf"],
        dtype=torch.float32,
        seed=0,
        device="cpu",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["num_tokens"] == [3, 5]
    assert manifest["seed"] == 0
    assert (tmp_path / "weights.safetensors").exists()
    for tokens in (3, 5):
        for distribution in ("uniform", "zipf"):
            assert (tmp_path / f"case_{tokens}_{distribution}.safetensors").exists()


def test_run_times_every_variant_on_every_case(small_dims: None, tmp_path: Path) -> None:
    prepare_inputs(
        tmp_path,
        num_tokens=[3, 5],
        distributions=["uniform", "zipf"],
        dtype=torch.float32,
        seed=0,
        device="cpu",
    )
    engines = [DispatchEngine("torch", block_m=16), DispatchEngine("torch", block_m=32)]

    results = run_engines(tmp_path, engines, precision="bf16", device="cpu", time_fn=_fake_timer)

    assert len(results) == 2 * 2 * 2  # variants x token counts x distributions
    assert {result["status"] for result in results} == {"ok"}
    assert {result["variant"] for result in results} == {
        "dispatch-torch-bm16",
        "dispatch-torch-bm32",
    }
    assert results[0]["mean_ms"] == 2.0


def test_int8_precision_checks_against_the_dequantized_reference(
    small_dims: None, tmp_path: Path
) -> None:
    prepare_inputs(
        tmp_path,
        num_tokens=[4],
        distributions=["zipf"],
        dtype=torch.float32,
        seed=0,
        device="cpu",
    )

    results = run_engines(
        tmp_path,
        [DispatchEngine("torch")],
        precision="int8",
        device="cpu",
        time_fn=_fake_timer,
    )

    assert [result["status"] for result in results] == ["ok"]


def test_an_engine_that_disagrees_with_the_reference_is_refused_not_timed(
    small_dims: None, tmp_path: Path
) -> None:
    prepare_inputs(
        tmp_path,
        num_tokens=[3],
        distributions=["uniform"],
        dtype=torch.float32,
        seed=0,
        device="cpu",
    )
    timed: list[str] = []

    def recording_timer(bound: object, label: str, flops: float) -> KernelBenchmarkSummary:
        timed.append(label)
        return _fake_timer(bound, label, flops)

    results = run_engines(
        tmp_path, [_WrongEngine()], precision="bf16", device="cpu", time_fn=recording_timer
    )

    assert results[0]["status"] == "refused"
    assert "mean_ms" not in results[0]
    assert timed == []


def test_main_run_writes_the_record_then_exits_nonzero_on_a_refusal(
    small_dims: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    prepare_inputs(
        inputs,
        num_tokens=[3],
        distributions=["uniform"],
        dtype=torch.float32,
        seed=0,
        device="cpu",
    )
    monkeypatch.setattr(registry, "build_engines", lambda engine, block_ms: [_WrongEngine()])
    monkeypatch.setattr(race_module, "_do_bench_timer", _fake_timer)

    with pytest.raises(SystemExit, match="refused, not timed"):
        main(
            [
                "run",
                "--inputs-dir",
                str(inputs),
                "--engine",
                "vllm",
                "--precision",
                "bf16",
                "--tuning-label",
                "tuned",
                "--device",
                "cpu",
                "--output-dir",
                str(out),
                "--run-label",
                "vllm-run",
            ]
        )

    record = json.loads((out / "vllm-run.json").read_text())
    assert record["config"]["engine"] == "vllm"
    assert record["config"]["tuning_label"] == "tuned"
    assert record["results"][0]["status"] == "refused"
    assert "environment" in record["config"]


def test_environment_records_tuned_config_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "configs" / "triton_3_8_0").mkdir(parents=True)
    (tmp_path / "configs" / "triton_3_8_0" / "E=64,N=1408.json").write_text("{}")
    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)

    environment = race_module.describe_environment("cpu")

    assert environment["SGLANG_MOE_CONFIG_DIR_files"] == ["configs/triton_3_8_0/E=64,N=1408.json"]
    assert environment["VLLM_TUNED_CONFIG_FOLDER"] is None
    assert environment["VLLM_TUNED_CONFIG_FOLDER_files"] == []


def test_merge_writes_a_json_and_a_markdown_table(tmp_path: Path) -> None:
    record = {
        "config": {"engine": "vllm", "precision": "bf16", "tuning_label": "tuned"},
        "results": [
            {
                "num_tokens": 16,
                "distribution": "uniform",
                "variant": "vllm",
                "status": "ok",
                "mean_ms": 0.5,
                "p50_ms": 0.5,
                "p99_ms": 0.6,
                "tflops": 2.0,
            }
        ],
    }
    (tmp_path / "x-race-vllm.json").write_text(json.dumps(record))

    main(
        ["merge", "--results-dir", str(tmp_path), "--output-dir", str(tmp_path), "--run-label", "s"]
    )

    assert "| 16 | uniform | 0.500 / 0.600 |" in (tmp_path / "s.md").read_text()
    assert json.loads((tmp_path / "s.json").read_text())[0]["engine"] == "vllm"
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_run_engine_race.py -v`
Expected: collection FAIL, `ModuleNotFoundError: No module named 'scripts.run_engine_race'`.

- [ ] **Step 3: Write the implementation**

`scripts/run_engine_race.py`:

````python
"""CLI for Phase 6's kernel race: dispatch's grouped-GEMM vs. vLLM's and
SGLang's fused-MoE on identical seeded inputs.

  prepare  generate the seeded weights, cases and fp32 references once, to disk
  run      time one engine (dispatch's tile-size sweep, vllm, or sglang) on them
  merge    combine the per-engine JSONs into one table

Inputs and references are written once and *loaded* by every `run`, so all
contestants see byte-identical tensors and are checked against the same
reference. An engine that disagrees with the reference is refused, not timed:
its result is recorded as refused, and `run` exits non-zero after writing the
JSON, so the evidence survives.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from dispatch.benchmark.engines import registry
from dispatch.benchmark.engines.base import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    N_ROUTED_EXPERTS,
    NUM_EXPERTS_PER_TOK,
    TOKEN_COUNTS,
    BoundLayer,
    MoEEngine,
    RaceCase,
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.race_summary import render_markdown, summarize_race
from dispatch.kernels.bench import (
    DO_BENCH_REP_MS,
    DO_BENCH_WARMUP_MS,
    ROUTING_DISTRIBUTIONS,
    KernelBenchmarkSummary,
    grouped_gemm_flops,
    time_grouped_gemm,
)
from dispatch.kernels.moe_forward import StackedExpertWeights, assert_matches_reference
from dispatch.kernels.quantization import (
    QuantizedStackedExpertWeights,
    QuantizedTensor,
    quantize_stacked_weights,
)

TimeFn = Callable[[BoundLayer, str, float], KernelBenchmarkSummary]

MANIFEST = "manifest.json"
WEIGHTS = "weights.safetensors"


def prepare_inputs(  # noqa: PLR0913 -- each is an independent, user-facing knob
    out_dir: Path,
    *,
    num_tokens: Sequence[int],
    distributions: Sequence[str],
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = make_weights(dtype, seed=seed, device=device)
    qweights = quantize_stacked_weights(weights)
    save_file(
        _to_cpu(
            {
                "gate": weights.gate,
                "up": weights.up,
                "down": weights.down,
                "q_gate_data": qweights.gate.data,
                "q_gate_scale": qweights.gate.scale,
                "q_up_data": qweights.up.data,
                "q_up_scale": qweights.up.scale,
                "q_down_data": qweights.down.data,
                "q_down_scale": qweights.down.scale,
            }
        ),
        str(out_dir / WEIGHTS),
    )
    int8_oracle = dequantized_weights(qweights)
    for tokens in num_tokens:
        for distribution in distributions:
            case = make_case(tokens, distribution, dtype=dtype, seed=seed, device=device)
            save_file(
                _to_cpu(
                    {
                        "x": case.x,
                        "topk_idx": case.topk_idx,
                        "topk_weight": case.topk_weight,
                        "ref_bf16": reference_output(case, weights),
                        "ref_int8": reference_output(case, int8_oracle),
                    }
                ),
                str(out_dir / _case_file(tokens, distribution)),
            )
    manifest = {
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "n_routed_experts": N_ROUTED_EXPERTS,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "num_tokens": list(num_tokens),
        "distributions": list(distributions),
        "prepared_on": describe_environment(device),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))


def run_engines(
    inputs_dir: Path,
    engines: Sequence[MoEEngine],
    *,
    precision: str,
    device: str,
    time_fn: TimeFn | None = None,
) -> list[dict[str, Any]]:
    """One result per (engine variant, case). A variant whose output disagrees
    with the fp32 reference is recorded as refused and never timed."""
    manifest = json.loads((inputs_dir / MANIFEST).read_text())
    weights, qweights = _load_weights(inputs_dir, device)
    timer = time_fn or _do_bench_timer
    results: list[dict[str, Any]] = []
    for engine in engines:
        factory = (
            engine.prepare_bf16(weights) if precision == "bf16" else engine.prepare_int8(qweights)
        )
        for tokens in manifest["num_tokens"]:
            for distribution in manifest["distributions"]:
                case, reference = _load_case(inputs_dir, tokens, distribution, precision, device)
                bound = factory(case)
                entry: dict[str, Any] = {
                    "num_tokens": tokens,
                    "distribution": distribution,
                    "variant": engine.name,
                }
                try:
                    assert_matches_reference(bound(), reference)
                except AssertionError as exc:
                    entry.update(status="refused", reason=str(exc).splitlines()[0])
                    results.append(entry)
                    continue
                flops = 3 * grouped_gemm_flops(
                    tokens * manifest["num_experts_per_tok"],
                    manifest["moe_intermediate_size"],
                    manifest["hidden_size"],
                )
                summary = timer(bound, f"{engine.name}/{tokens}/{distribution}", flops)
                entry.update(
                    status="ok",
                    mean_ms=summary.mean_latency_ms,
                    p50_ms=summary.p50_latency_ms,
                    p99_ms=summary.p99_latency_ms,
                    tflops=summary.tflops,
                )
                results.append(entry)
    return results


def describe_environment(device: str) -> dict[str, Any]:
    """Hardware, library versions, and tuned-config provenance: the config a
    benchmark number means nothing without."""
    environment: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
    }
    for package in ("triton", "vllm", "sglang"):
        environment[package] = _installed_version(package)
    for variable in ("VLLM_TUNED_CONFIG_FOLDER", "SGLANG_MOE_CONFIG_DIR"):
        folder = os.environ.get(variable)
        environment[variable] = folder
        environment[f"{variable}_files"] = (
            sorted(str(path.relative_to(folder)) for path in Path(folder).rglob("*.json"))
            if folder and Path(folder).is_dir()
            else []
        )
    return environment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6 kernel race")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--num-tokens", type=int, nargs="+", default=list(TOKEN_COUNTS))
    prepare.add_argument(
        "--distributions", nargs="+", choices=ROUTING_DISTRIBUTIONS, default=ROUTING_DISTRIBUTIONS
    )
    prepare.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--device", default="cuda")

    run = sub.add_parser("run")
    run.add_argument("--inputs-dir", type=Path, required=True)
    run.add_argument("--engine", choices=registry.ENGINE_NAMES, required=True)
    run.add_argument("--block-ms", type=int, nargs="+", default=list(registry.DEFAULT_BLOCK_MS))
    run.add_argument("--precision", choices=["bf16", "int8"], required=True)
    run.add_argument(
        "--tuning-label",
        choices=["sweep", "default", "tuned"],
        required=True,
        help="'sweep' for dispatch's tile-size sweep; 'default'/'tuned' for vllm and sglang",
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    run.add_argument("--run-label", default=None)

    merge = sub.add_parser("merge")
    merge.add_argument("--results-dir", type=Path, required=True)
    merge.add_argument("--glob", default="*-race-*.json")
    merge.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    merge.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-6-race-summary"))

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_inputs(
            args.out_dir,
            num_tokens=args.num_tokens,
            distributions=args.distributions,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
            device=args.device,
        )
        print(f"wrote inputs to {args.out_dir}")
    elif args.command == "run":
        _command_run(args)
    else:
        _command_merge(args)


def _command_run(args: argparse.Namespace) -> None:
    if args.engine == "sglang":
        from dispatch.benchmark.engines.sglang_moe import init_distributed  # noqa: PLC0415

        init_distributed()
    engines = registry.build_engines(args.engine, args.block_ms)
    results = run_engines(args.inputs_dir, engines, precision=args.precision, device=args.device)
    manifest = json.loads((args.inputs_dir / MANIFEST).read_text())
    record = {
        "config": {
            "engine": args.engine,
            "precision": args.precision,
            "tuning_label": args.tuning_label,
            "block_ms": list(args.block_ms) if args.engine.startswith("dispatch-") else None,
            "inputs": manifest,
            "do_bench_warmup_ms": DO_BENCH_WARMUP_MS,
            "do_bench_rep_ms": DO_BENCH_REP_MS,
            "environment": describe_environment(args.device),
        },
        "results": results,
    }
    label = args.run_label or time.strftime(
        f"%Y-%m-%d-phase-6-race-{args.engine}-{args.precision}-{args.tuning_label}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")
    refused = [result for result in results if result["status"] == "refused"]
    if refused:
        raise SystemExit(
            f"{len(refused)} result(s) from {args.engine} disagree with the fp32 reference and "
            f"were refused, not timed -- see {path}"
        )


def _command_merge(args: argparse.Namespace) -> None:
    records = [json.loads(path.read_text()) for path in sorted(args.results_dir.glob(args.glob))]
    if not records:
        raise SystemExit(f"no files matching {args.glob} in {args.results_dir}")
    rows = summarize_race(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.run_label}.json").write_text(
        json.dumps([row.__dict__ for row in rows], indent=2)
    )
    markdown_path = args.output_dir / f"{args.run_label}.md"
    markdown_path.write_text(render_markdown(rows))
    print(f"wrote {markdown_path}")


def _do_bench_timer(bound: BoundLayer, label: str, flops: float) -> KernelBenchmarkSummary:
    return time_grouped_gemm(bound, label=label, flops=flops)


def _case_file(num_tokens: int, distribution: str) -> str:
    return f"case_{num_tokens}_{distribution}.safetensors"


def _to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in tensors.items()}


def _load_weights(
    inputs_dir: Path, device: str
) -> tuple[StackedExpertWeights, QuantizedStackedExpertWeights]:
    loaded = load_file(str(inputs_dir / WEIGHTS), device=device)
    weights = StackedExpertWeights(gate=loaded["gate"], up=loaded["up"], down=loaded["down"])
    qweights = QuantizedStackedExpertWeights(
        gate=QuantizedTensor(data=loaded["q_gate_data"], scale=loaded["q_gate_scale"]),
        up=QuantizedTensor(data=loaded["q_up_data"], scale=loaded["q_up_scale"]),
        down=QuantizedTensor(data=loaded["q_down_data"], scale=loaded["q_down_scale"]),
    )
    return weights, qweights


def _load_case(
    inputs_dir: Path, num_tokens: int, distribution: str, precision: str, device: str
) -> tuple[RaceCase, torch.Tensor]:
    loaded = load_file(str(inputs_dir / _case_file(num_tokens, distribution)), device=device)
    case = RaceCase(x=loaded["x"], topk_idx=loaded["topk_idx"], topk_weight=loaded["topk_weight"])
    return case, loaded["ref_bf16" if precision == "bf16" else "ref_int8"]


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_engine_race.py -v`
Expected: 7 passed, including `test_an_engine_that_disagrees_with_the_reference_is_refused_not_timed`.

- [ ] **Step 5: Commit**

STATUS bullet: `- **Task 7 (race driver)**: prepare/run/merge; seeded inputs and fp32 references written once and loaded by every contestant; a disagreeing engine is refused, not timed, and its record is written before the non-zero exit.`

```bash
make check
git add scripts/run_engine_race.py tests/unit/test_run_engine_race.py docs/STATUS.md
git commit -m "feat: engine race driver with refuse-not-time and one-time seeded inputs"
```

---

### Task 8: Serving-benchmark helpers and the pod reference driver

Flags below are the documented `vllm bench serve` ones (checked live 2026-09-19 against vllm-project/vllm `docs/benchmarking/cli.md` and `vllm/benchmarks/serve.py`); Task 10 re-verifies them against the installed 0.29.0 with `--help` before any paid run depends on them.

**Files:**
- Create: `src/dispatch/benchmark/serving_bench.py`, `scripts/gpu/phase6_engine_reference.py`
- Test: `tests/unit/test_serving_bench.py`, `tests/unit/test_phase6_engine_reference.py`

**Interfaces:**
- Consumes: `metrics.percentile` (Task 3), `GATE_PROMPTS` (Task 2).
- Produces: `serving_bench`: `ENGINES`, `CONCURRENCIES = (1, 4, 16, 64)`, `OUTPUT_LEN = 64`, `NUM_WARMUPS = 8`, `num_prompts_for(concurrency)`, `write_trace(path, prompts, count)`, `build_serve_command(engine, model, port, *, max_model_len=2048)`, `build_bench_command(model, port, *, concurrency, trace_path, result_dir, result_filename)`, `ServingSummary`, `summarize_bench_result(raw, *, concurrency, expected_output_len, gpu_cost_per_hour)` (raises `ValueError` to refuse), `wait_until_healthy(...)`. `run_engine_reference(engine, *, model, port, concurrencies, output_dir, run_label, gpu_cost_per_hour, ...)` in the pod script always terminates the server and writes its evidence JSON (with an `error` field) even when a later concurrency fails.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_serving_bench.py`:

````python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dispatch.benchmark.serving_bench import (
    OUTPUT_LEN,
    build_bench_command,
    build_serve_command,
    num_prompts_for,
    summarize_bench_result,
    wait_until_healthy,
    write_trace,
)


def _raw(**overrides: Any) -> dict[str, Any]:
    """A vLLM bench-serve --save-detailed result for 4 requests of 64 tokens."""
    raw: dict[str, Any] = {
        "completed": 4,
        "failed": 0,
        "duration": 8.0,
        "output_throughput": 32.0,  # 4 * 64 tokens / 8 s
        "output_lens": [64, 64, 64, 64],
        "ttfts": [0.1, 0.2, 0.3, 0.4],
        "itls": [[0.02, 0.02], [0.03, 0.03], [0.02, 0.04], [0.02, 0.02]],
        "latencies": [2.0, 2.0, 2.0, 2.0],
        "errors": ["", "", "", ""],
    }
    raw.update(overrides)
    return raw


def _summarize(raw: dict[str, Any]) -> Any:
    return summarize_bench_result(
        raw, concurrency=1, expected_output_len=OUTPUT_LEN, gpu_cost_per_hour=0.72
    )


def test_summary_converts_to_milliseconds_and_computes_request_throughput() -> None:
    summary = _summarize(_raw())

    assert summary.mean_ttft_ms == pytest.approx(250.0)
    assert summary.p50_ttft_ms == pytest.approx(250.0)
    assert summary.mean_request_tokens_per_s == pytest.approx(32.0)  # 64 tokens / 2.0 s
    assert summary.p99_e2e_ms == pytest.approx(2000.0)
    assert summary.mean_itl_ms == pytest.approx(25.0)


def test_cost_per_million_tokens_comes_from_the_rate_and_measured_throughput() -> None:
    summary = _summarize(_raw())

    # $0.72/hr at 32 output tokens/s: 0.72 / (32 * 3600) * 1e6
    assert summary.cost_per_million_output_tokens_usd == pytest.approx(6.25)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"completed": 0}, "zero requests"),
        ({"failed": 1}, "failed request"),
        ({"errors": ["", "", "", "boom"]}, "failed request"),
        ({"output_lens": [64, 64, 64, 10]}, "ignore-eos was not honored"),
    ],
)
def test_a_run_that_cannot_be_trusted_is_refused(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _summarize(_raw(**overrides))


def test_bench_command_carries_the_verified_flags_and_per_concurrency_prompt_count() -> None:
    command = build_bench_command(
        "m",
        8000,
        concurrency=16,
        trace_path=Path("t.jsonl"),
        result_dir=Path("out"),
        result_filename="r.json",
    )

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert command[:3] == ["vllm", "bench", "serve"]
    assert value("--dataset-name") == "custom"
    assert value("--max-concurrency") == "16"
    assert value("--num-prompts") == str(num_prompts_for(16)) == "128"
    assert value("--custom-output-len") == "64"
    for flag in ("--ignore-eos", "--skip-chat-template", "--save-detailed"):
        assert flag in command


def test_minimum_prompt_count_holds_at_low_concurrency() -> None:
    assert num_prompts_for(1) == 32
    assert num_prompts_for(4) == 32
    assert num_prompts_for(64) == 512


def test_serve_commands_pin_dtype_and_context_and_reject_unknown_engines() -> None:
    vllm = build_serve_command("vllm", "m", 8000)
    sglang = build_serve_command("sglang", "m", 8000)

    assert vllm[:3] == ["vllm", "serve", "m"]
    assert sglang[:3] == ["python", "-m", "sglang.launch_server"]
    assert "bfloat16" in vllm
    assert "bfloat16" in sglang
    with pytest.raises(ValueError, match="unknown engine"):
        build_serve_command("tgi", "m", 8000)


def test_write_trace_cycles_prompts_into_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"

    write_trace(path, ["a", "b", "c"], count=7)

    prompts = [json.loads(line)["prompt"] for line in path.read_text().splitlines()]
    assert prompts == ["a", "b", "c", "a", "b", "c", "a"]


def test_wait_until_healthy_returns_once_the_endpoint_answers_200() -> None:
    statuses = iter([503, 503, 200])
    clock = iter(range(1000))

    wait_until_healthy(
        "http://x/health",
        get_fn=lambda url: next(statuses),
        is_alive=lambda: True,
        sleep_fn=lambda seconds: None,
        clock_fn=lambda: float(next(clock)),
        timeout_s=100.0,
    )


def test_wait_until_healthy_treats_a_refused_connection_as_not_yet() -> None:
    calls: Iterator[Exception | int] = iter([ConnectionRefusedError(), 200])

    def get(url: str) -> int:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    clock = iter(range(1000))
    wait_until_healthy(
        "http://x/health",
        get_fn=get,
        is_alive=lambda: True,
        sleep_fn=lambda seconds: None,
        clock_fn=lambda: float(next(clock)),
        timeout_s=100.0,
    )


def test_wait_until_healthy_fails_fast_when_the_server_dies() -> None:
    with pytest.raises(RuntimeError, match="exited"):
        wait_until_healthy(
            "http://x/health",
            get_fn=lambda url: 503,
            is_alive=lambda: False,
            sleep_fn=lambda seconds: None,
            clock_fn=lambda: 0.0,
            timeout_s=100.0,
        )


def test_wait_until_healthy_times_out() -> None:
    clock = iter(range(0, 10_000, 40))

    with pytest.raises(TimeoutError, match="not healthy after 100s"):
        wait_until_healthy(
            "http://x/health",
            get_fn=lambda url: 503,
            is_alive=lambda: True,
            sleep_fn=lambda seconds: None,
            clock_fn=lambda: float(next(clock)),
            timeout_s=100.0,
        )
````

Create `tests/unit/test_phase6_engine_reference.py`:

````python
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts.gpu.phase6_engine_reference import run_engine_reference


class _FakeServer:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class _StuckServer(_FakeServer):
    def wait(self, timeout: float) -> int:
        raise subprocess.TimeoutExpired("server", timeout)


def _good_raw() -> dict[str, Any]:
    return {
        "completed": 2,
        "failed": 0,
        "duration": 4.0,
        "output_throughput": 32.0,
        "output_lens": [64, 64],
        "ttfts": [0.1, 0.1],
        "itls": [[0.02], [0.02]],
        "latencies": [2.0, 2.0],
        "errors": ["", ""],
    }


def _run(
    tmp_path: Path, server: _FakeServer, raw_by_concurrency: dict[int, dict[str, Any]]
) -> Path:
    def fake_run(command: list[str], check: bool) -> None:
        concurrency = int(command[command.index("--max-concurrency") + 1])
        filename = command[command.index("--result-filename") + 1]
        (tmp_path / filename).write_text(json.dumps(raw_by_concurrency[concurrency]))

    return run_engine_reference(
        "vllm",
        model="m",
        port=8000,
        concurrencies=list(raw_by_concurrency),
        output_dir=tmp_path,
        run_label="ref",
        gpu_cost_per_hour=0.72,
        popen_fn=lambda command, **kwargs: server,
        run_fn=fake_run,
        get_fn=lambda url: 200,
        sleep_fn=lambda seconds: None,
        clock_fn=lambda: 0.0,
    )


def test_a_clean_run_writes_one_summary_per_concurrency_and_stops_the_server(
    tmp_path: Path,
) -> None:
    server = _FakeServer()

    path = _run(tmp_path, server, {1: _good_raw(), 4: _good_raw()})

    record = json.loads(path.read_text())
    assert [result["concurrency"] for result in record["results"]] == [1, 4]
    assert record["error"] is None
    assert record["config"]["output_len"] == 64
    assert record["config"]["gpu_cost_per_hour"] == 0.72
    assert server.terminated
    assert (tmp_path / "ref-trace.jsonl").exists()


def test_a_bad_run_still_writes_the_earlier_evidence_and_stops_the_server(
    tmp_path: Path,
) -> None:
    server = _FakeServer()
    bad = {**_good_raw(), "output_lens": [64, 3]}

    with pytest.raises(ValueError, match="ignore-eos was not honored"):
        _run(tmp_path, server, {1: _good_raw(), 4: bad})

    record = json.loads((tmp_path / "ref.json").read_text())
    assert [result["concurrency"] for result in record["results"]] == [1]
    assert "ignore-eos" in record["error"]
    assert server.terminated


def test_a_server_that_ignores_terminate_is_killed(tmp_path: Path) -> None:
    server = _StuckServer()

    _run(tmp_path, server, {1: _good_raw()})

    assert server.killed
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_serving_bench.py tests/unit/test_phase6_engine_reference.py -v`
Expected: collection FAIL, `ModuleNotFoundError: No module named 'dispatch.benchmark.serving_bench'`.

- [ ] **Step 3: Write the implementation**

`src/dispatch/benchmark/serving_bench.py`:

````python
"""Pure helpers for Phase 6's engine reference: build the server and
`vllm bench serve` commands, summarize a bench result, and refuse one that
cannot be trusted. No subprocess, network or GPU here -- the orchestration
that uses these lives in scripts/gpu/phase6_engine_reference.py.

`vllm bench serve` flags checked live 2026-09-19 against vllm-project/vllm
docs/benchmarking/cli.md (custom dataset) and vllm/benchmarks/serve.py. The
same client drives both engines, so client-side behavior is never a
difference between them.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatch.benchmark.metrics import percentile

ENGINES = ("vllm", "sglang")
CONCURRENCIES = (1, 4, 16, 64)
OUTPUT_LEN = 64  # matches every earlier phase's --max-new-tokens
NUM_WARMUPS = 8  # discarded by the client and recorded in the output config
MIN_PROMPTS = 32
PROMPTS_PER_CONCURRENCY = 8
TRACE_LINES = 512  # >= num_prompts_for(max(CONCURRENCIES))
MS = 1000.0


def num_prompts_for(concurrency: int) -> int:
    return max(MIN_PROMPTS, PROMPTS_PER_CONCURRENCY * concurrency)


def write_trace(path: Path, prompts: Sequence[str], count: int = TRACE_LINES) -> None:
    """One {"prompt": ...} JSONL line per request, cycling `prompts`: the
    format `vllm bench serve --dataset-name custom` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"prompt": prompts[i % len(prompts)]}) for i in range(count))
    path.write_text("\n".join(lines) + "\n")


def build_serve_command(
    engine: str, model: str, port: int, *, max_model_len: int = 2048
) -> list[str]:
    if engine == "vllm":
        return [
            "vllm", "serve", model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(max_model_len),
            "--trust-remote-code",
        ]  # fmt: skip
    if engine == "sglang":
        return [
            "python", "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--context-length", str(max_model_len),
            "--trust-remote-code",
        ]  # fmt: skip
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


def build_bench_command(  # noqa: PLR0913 -- each is an independent, user-facing knob
    model: str,
    port: int,
    *,
    concurrency: int,
    trace_path: Path,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    return [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model,
        "--endpoint", "/v1/completions",
        "--dataset-name", "custom",
        "--dataset-path", str(trace_path),
        "--skip-chat-template",
        "--custom-output-len", str(OUTPUT_LEN),
        "--ignore-eos",
        "--num-prompts", str(num_prompts_for(concurrency)),
        "--max-concurrency", str(concurrency),
        "--num-warmups", str(NUM_WARMUPS),
        "--seed", "0",
        "--save-result", "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]  # fmt: skip


@dataclass(frozen=True)
class ServingSummary:
    concurrency: int
    completed: int
    duration_s: float
    output_tokens_per_s: float
    mean_request_tokens_per_s: float  # per request: output tokens / end-to-end latency
    mean_ttft_ms: float
    p50_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    p99_itl_ms: float
    p50_e2e_ms: float
    p99_e2e_ms: float
    cost_per_million_output_tokens_usd: float


def summarize_bench_result(
    raw: dict[str, Any],
    *,
    concurrency: int,
    expected_output_len: int,
    gpu_cost_per_hour: float,
) -> ServingSummary:
    """Refuses -- raises ValueError -- rather than report a number from a run
    that failed requests, returned no completions, or did not generate exactly
    `expected_output_len` tokens per request (an engine that ignored
    --ignore-eos would report a throughput no other engine's is comparable to)."""
    completed = int(raw["completed"])
    if completed == 0:
        raise ValueError("bench completed zero requests -- refusing to report numbers")
    failed = int(raw.get("failed", 0))
    errors = [error for error in raw["errors"] if error]
    if failed or errors:
        raise ValueError(f"{failed} failed request(s), first error: {errors[:1]} -- refusing")
    wrong = [n for n in raw["output_lens"] if n != expected_output_len]
    if wrong:
        raise ValueError(
            f"{len(wrong)} request(s) generated {sorted(set(wrong))} tokens, not "
            f"{expected_output_len} -- ignore-eos was not honored; refusing"
        )
    ttfts_ms = sorted(t * MS for t in raw["ttfts"])
    itls_ms = sorted(gap * MS for request in raw["itls"] for gap in request)
    e2e_ms = sorted(latency * MS for latency in raw["latencies"])
    tokens_per_s = [
        n / latency for n, latency in zip(raw["output_lens"], raw["latencies"], strict=True)
    ]
    output_tokens_per_s = float(raw["output_throughput"])
    return ServingSummary(
        concurrency=concurrency,
        completed=completed,
        duration_s=float(raw["duration"]),
        output_tokens_per_s=output_tokens_per_s,
        mean_request_tokens_per_s=statistics.mean(tokens_per_s),
        mean_ttft_ms=statistics.mean(ttfts_ms),
        p50_ttft_ms=percentile(ttfts_ms, 0.50),
        p99_ttft_ms=percentile(ttfts_ms, 0.99),
        mean_itl_ms=statistics.mean(itls_ms) if itls_ms else 0.0,
        p99_itl_ms=percentile(itls_ms, 0.99) if itls_ms else 0.0,
        p50_e2e_ms=percentile(e2e_ms, 0.50),
        p99_e2e_ms=percentile(e2e_ms, 0.99),
        cost_per_million_output_tokens_usd=(
            gpu_cost_per_hour / (output_tokens_per_s * 3600) * 1_000_000
        ),
    )


def wait_until_healthy(  # noqa: PLR0913 -- injectable clock/sleep/get is what makes this testable
    health_url: str,
    *,
    get_fn: Callable[[str], int],
    is_alive: Callable[[], bool],
    sleep_fn: Callable[[float], None],
    clock_fn: Callable[[], float],
    timeout_s: float,
    interval_s: float = 5.0,
) -> None:
    """`get_fn` returns the HTTP status (raising on a refused connection).
    A server that dies while loading the model fails fast instead of waiting
    out the timeout on a metered pod."""
    deadline = clock_fn() + timeout_s
    while clock_fn() < deadline:
        if not is_alive():
            raise RuntimeError(f"server exited before {health_url} became healthy")
        try:
            if get_fn(health_url) == 200:  # noqa: PLR2004 -- HTTP OK
                return
        except OSError:
            pass
        sleep_fn(interval_s)
    raise TimeoutError(f"{health_url} not healthy after {timeout_s:.0f}s")
````

`scripts/gpu/phase6_engine_reference.py`:

````python
"""Pod-side driver for Phase 6's engine reference: launch one engine's
server, run `vllm bench serve` against it at each concurrency, summarize with
serving_bench's refuse rules, and always tear the server down. Run once per
engine, in the shared engines venv:

  python -m scripts.gpu.phase6_engine_reference --engine vllm --gpu-cost-per-hour 0.69

The evidence JSON is written even when a later concurrency fails, so a
partial run survives; the failure is then re-raised (non-zero exit).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from dispatch.benchmark.gate_prompts import GATE_PROMPTS
from dispatch.benchmark.serving_bench import (
    CONCURRENCIES,
    ENGINES,
    NUM_WARMUPS,
    OUTPUT_LEN,
    TRACE_LINES,
    ServingSummary,
    build_bench_command,
    build_serve_command,
    num_prompts_for,
    summarize_bench_result,
    wait_until_healthy,
    write_trace,
)

DEFAULT_MODEL = "deepseek-ai/deepseek-moe-16b-base"
TERMINATE_GRACE_S = 60.0


def run_engine_reference(  # noqa: PLR0913 -- injectable process/network hooks make this testable
    engine: str,
    *,
    model: str,
    port: int,
    concurrencies: Sequence[int],
    output_dir: Path,
    run_label: str,
    gpu_cost_per_hour: float,
    health_timeout_s: float = 1800.0,
    popen_fn: Callable[..., Any] = subprocess.Popen,
    run_fn: Callable[..., Any] = subprocess.run,
    get_fn: Callable[[str], int] = lambda url: requests.get(url, timeout=5).status_code,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{run_label}-trace.jsonl"
    write_trace(trace_path, GATE_PROMPTS, TRACE_LINES)
    summaries: list[ServingSummary] = []
    error: str | None = None

    server_log = (output_dir / f"{run_label}-server.log").open("w")
    server = popen_fn(
        build_serve_command(engine, model, port), stdout=server_log, stderr=server_log
    )
    try:
        wait_until_healthy(
            f"http://127.0.0.1:{port}/health",
            get_fn=get_fn,
            is_alive=lambda: server.poll() is None,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            timeout_s=health_timeout_s,
        )
        for concurrency in concurrencies:
            filename = f"{run_label}-c{concurrency}-raw.json"
            run_fn(
                build_bench_command(
                    model,
                    port,
                    concurrency=concurrency,
                    trace_path=trace_path,
                    result_dir=output_dir,
                    result_filename=filename,
                ),
                check=True,
            )
            raw = json.loads((output_dir / filename).read_text())
            summaries.append(
                summarize_bench_result(
                    raw,
                    concurrency=concurrency,
                    expected_output_len=OUTPUT_LEN,
                    gpu_cost_per_hour=gpu_cost_per_hour,
                )
            )
    except Exception as exc:  # recorded in the evidence JSON, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        (output_dir / f"{run_label}.json").write_text(
            json.dumps(
                {
                    "config": {
                        "engine": engine,
                        "engine_version": _version(engine),
                        "model": model,
                        "dtype": "bfloat16",
                        "output_len": OUTPUT_LEN,
                        "num_warmups": NUM_WARMUPS,
                        "concurrencies": list(concurrencies),
                        "num_prompts": {c: num_prompts_for(c) for c in concurrencies},
                        "gpu_cost_per_hour": gpu_cost_per_hour,
                        "trace": trace_path.name,
                    },
                    "results": [asdict(summary) for summary in summaries],
                    "error": error,
                },
                indent=2,
            )
        )
    return output_dir / f"{run_label}.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6 engine reference (vLLM or SGLang)")
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES))
    parser.add_argument("--gpu-cost-per-hour", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args(argv)
    label = args.run_label or time.strftime(f"%Y-%m-%d-phase-6-reference-{args.engine}")
    path = run_engine_reference(
        args.engine,
        model=args.model,
        port=args.port,
        concurrencies=args.concurrencies,
        output_dir=args.output_dir,
        run_label=label,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
    )
    print(f"wrote {path}")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_serving_bench.py tests/unit/test_phase6_engine_reference.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

STATUS bullet: `- **Task 8 (serving-benchmark helpers)**: vllm bench serve / server commands, a summarizer that refuses failed requests and non-64-token outputs, cost per million tokens from the measured GPU rate, and a driver that always tears its server down.`

```bash
make check
git add src/dispatch/benchmark/serving_bench.py scripts/gpu/phase6_engine_reference.py tests/unit/test_serving_bench.py tests/unit/test_phase6_engine_reference.py docs/STATUS.md
git commit -m "feat: serving-benchmark helpers and pod reference driver with refuse rules"
```

---

### Task 9: Real-engine correctness gate for the race (`gpu`)

The CPU adapter tests prove the layouts against eager fakes. This file proves the same adapters against the real engines and the real Triton kernels, on DeepSeek's real dims, before any latency number means anything. It is skipped here (no CUDA, no triton on macOS) and runs in Task 10.

**Files:**
- Create: `tests/unit/test_engines_real_gpu.py`

**Interfaces:**
- Consumes: `registry.build_engines`, `sglang_moe.init_distributed`, Task 4's builders and `reference_output`, `dequantized_weights`.
- Produces: `gpu`-marked tests: every engine matches the fp32 (bf16) or dequantized (int8) reference at token counts 1/64/512 under both routing distributions; a **mutation** test hands every engine swapped gate/up and requires the check to fail; a repeatability test proves no engine mutates its input.

- [ ] **Step 1: Write the test file**

Create `tests/unit/test_engines_real_gpu.py`:

````python
"""Phase 6's real-engine correctness gate for the kernel race: every
contestant -- dispatch's Triton kernels, vLLM's fused_experts, SGLang's
fused_experts -- must meet assert_matches_reference against the fp32
reference on DeepSeek-MoE-16B's real routed-expert dims, before any latency
number means anything. The CPU tests (test_engine_adapters.py) prove the
adapters against eager fakes; this proves them against the real engines and
the real kernels.

Skipped, with the reason shown, where CUDA, triton, vllm or sglang is
missing. Excluded from CI by the `gpu` marker; runs in the Phase 6 pod
session's engines venv.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.engines.base import (
    MoEEngine,
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.engines.registry import build_engines
from dispatch.kernels.moe_forward import StackedExpertWeights, assert_matches_reference
from dispatch.kernels.quantization import quantize_stacked_weights

pytestmark = pytest.mark.gpu
pytest.importorskip("triton", reason="triton ships Linux wheels only")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

ENGINES = ["dispatch-naive", "dispatch-persistent", "vllm", "sglang"]
TOKEN_COUNTS = [1, 64, 512]  # decode-sized through a prefill-sized batch


@pytest.fixture(scope="module")
def weights() -> StackedExpertWeights:
    return make_weights(torch.bfloat16, seed=0, device="cuda")


def _engine(name: str) -> MoEEngine:
    if name == "vllm":
        pytest.importorskip("vllm", reason="vllm is installed only in the Phase 6 engines venv")
    if name == "sglang":
        pytest.importorskip("sglang", reason="sglang is installed only in the Phase 6 engines venv")
        from dispatch.benchmark.engines.sglang_moe import init_distributed  # noqa: PLC0415

        init_distributed()
    return build_engines(name, [16])[0]


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("name", ENGINES)
def test_bf16_engine_matches_the_fp32_reference(
    name: str, tokens: int, distribution: str, weights: StackedExpertWeights
) -> None:
    case = make_case(tokens, distribution, dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_bf16(weights)(case)()

    assert_matches_reference(output, reference_output(case, weights))


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("name", ["dispatch-naive", "vllm", "sglang"])
def test_int8_engine_matches_the_dequantized_reference(
    name: str, tokens: int, distribution: str, weights: StackedExpertWeights
) -> None:
    qweights = quantize_stacked_weights(weights)
    case = make_case(tokens, distribution, dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_int8(qweights)(case)()

    assert_matches_reference(output, reference_output(case, dequantized_weights(qweights)))


@pytest.mark.parametrize("name", ENGINES)
def test_a_mutated_weight_layout_turns_the_gate_red(
    name: str, weights: StackedExpertWeights
) -> None:
    """The suite's own proof that it can fail: hand each engine gate and up
    swapped and require the reference check to reject the output."""
    swapped = StackedExpertWeights(gate=weights.up, up=weights.gate, down=weights.down)
    case = make_case(64, "uniform", dtype=torch.bfloat16, seed=1, device="cuda")

    output = _engine(name).prepare_bf16(swapped)(case)()

    with pytest.raises(AssertionError):
        assert_matches_reference(output, reference_output(case, weights))


def test_bound_layers_are_repeatable_and_leave_the_input_untouched(
    weights: StackedExpertWeights,
) -> None:
    """SGLang's inplace default would overwrite x; timing calls the layer
    hundreds of times, so any input mutation would corrupt every later call."""
    for name in ENGINES:
        case = make_case(64, "zipf", dtype=torch.bfloat16, seed=1, device="cuda")
        x_before = case.x.clone()
        layer = _engine(name).prepare_bf16(weights)(case)

        first = layer().clone()
        second = layer()

        torch.testing.assert_close(first.float(), second.float())
        assert torch.equal(case.x, x_before)
````

- [ ] **Step 2: Verify it collects and skips cleanly here**

Run: `uv run pytest tests/unit/test_engines_real_gpu.py -v -rs`
Expected: 1 skipped, reason "triton ships Linux wheels only" (on a Linux box without CUDA: "needs a CUDA device"). It must not error at collection.

- [ ] **Step 3: Verify the whole gate is still green**

Run: `make check`
Expected: lint, mypy strict, and the full non-`gpu` suite pass.

- [ ] **Step 4: Commit**

STATUS bullet: `- **Task 9 (real-engine gate)**: gpu-marked tests check every contestant against the fp32/dequantized reference on real dims, plus a mutation test (swapped gate/up must fail); skipped off-GPU, run on the pod in Task 10.`

```bash
git add tests/unit/test_engines_real_gpu.py docs/STATUS.md
git commit -m "test: real-engine correctness gate for the race, with a mutation check"
```

---
## Paid pod session (Tasks 10-14)

These tasks cannot be verified on the dev machine; their steps are commands with expected outputs and the pre-registered stop rules above. **Read "Pre-registered gate rule and decision rules" and "Cost and stop rules" again before starting Task 10.** Shell snippets assume the variables defined in Task 10.

The pod-access conventions are inherited from `docs/runbooks/phase-5b-speculative-decoding.md`, step 4, which records what actually worked on RunPod's `ssh.runpod.io` proxy (PTY required, no SFTP/`scp`, base64 transfer in 76-character lines) and the five DeepSeek remote-code breaks under `transformers==5.17.0`. Do not re-derive them; follow that runbook and note any drift.

---

### Task 10: Pre-flight, provisioning, environment, and real-engine verification

**Files:**
- Modify (only if the real engines demand it, each via its own failing test first): `src/dispatch/benchmark/engines/vllm_moe.py`, `src/dispatch/benchmark/engines/sglang_moe.py`, `src/dispatch/benchmark/serving_bench.py`, their tests
- Create (pod-local, never committed): `/workspace/dispatch/_patch_and_run_baseline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-9, on a clean checkout of `phase-6-final-benchmark`.
- Produces: a running L40 pod with `/workspace/dispatch` (repo), `/workspace/dispatch/.venv` (dispatch venv), `/workspace/engines-venv` (vllm 0.29.0 + sglang 0.5.20 + torch 2.13.0 + dispatch installed `--no-deps`), `/workspace/hf_cache` (model), and the variables `POD_ID`, `POD_SSH`, `RATE` (the live $/hr), `D` (today's date) that Tasks 11-14 use.

- [ ] **Step 1: Confirm the live price and get the go-ahead**

Query the live L40 Secure Cloud price and availability (RunPod MCP `list-gpu-types` / `get-capacity`, or the console). Tell the user: the GPU id, the hourly rate, the $10 cap, the $5 and $8 checkpoints, and that Tasks 10-14 will run for roughly 6-8 hours (an estimate, not a measurement). **Do not create the pod until the user says yes.** Record the rate as `RATE`.

- [ ] **Step 2: Create the pod with a disk that fits everything**

Phase 5b's default of 60GB is too small: the model is 32.8GB and the engines venv (torch + vllm + sglang, CUDA wheels) is large. Use 150GB, and put everything under `/workspace`.

```bash
uv run python -m scripts.gpu.provision create --name dispatch-phase-6 \
  --gpu-type "<the L40 id from step 1>" --image "<current runpod/pytorch tag>" \
  --cloud SECURE --disk-gb 150
uv run python -m scripts.gpu.provision wait --pod-id <pod_id>
```

Note the pod id (`POD_ID`) and the SSH proxy user (`POD_SSH=<user>@ssh.runpod.io`). Start a wall-clock note (`date -u`): duration is billed, so it is also the cost measurement.

- [ ] **Step 3: Confirm the card and the driver**

```bash
ssh -tt $POD_SSH   # then on the pod:
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
df -h /workspace
```
Expected: an L40 (`8.9`, ~46-48GB), and `/workspace` with 100GB+ free. vllm 0.29.0 and sglang 0.5.20 ship CUDA 13 wheels (`flashinfer_python[cu13]`); if the driver is too old for CUDA 13 (driver below 580), stop, record it, and ask the user before renting a different pod.

- [ ] **Step 4: Ship the repo into `/workspace`**

Follow runbook 5b step 4 (base64 through `ssh -tt`), extracting to `/workspace/dispatch`, **not** `/root/dispatch`: Phase 5b's clone lived outside the persistent mount and did not survive a stop.

- [ ] **Step 5: Build the dispatch venv and download the model**

```bash
cd /workspace/dispatch
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
command -v uv || pip install uv
uv sync --all-extras --dev
uv pip install hf_transfer
.venv/bin/huggingface-cli download deepseek-ai/deepseek-moe-16b-base
.venv/bin/python -m pytest -m "not gpu" -q
```
Expected: the model downloads (~32.8GB) and the full non-`gpu` suite passes on the pod. Create `_patch_and_run_baseline.py` from runbook 5b step 4's `_patch_and_run.py`, changing the last two lines to `from scripts.run_baseline import main` / `main(sys.argv[1:])`, after confirming live which of the five remote-code breaks still reproduce (check the model repo's recent commits). Invoke `.venv/bin/python` directly from here on, never `uv run` (it re-syncs to `uv.lock` and undoes overrides).

- [ ] **Step 6: Build the engines venv**

```bash
uv venv /workspace/engines-venv --python 3.12
E=/workspace/engines-venv/bin/python
uv pip install --python $E torch==2.13.0 vllm==0.29.0 sglang==0.5.20 safetensors requests pytest
uv pip install --python $E --no-deps -e /workspace/dispatch
$E -c "import torch, triton, vllm, sglang; print(torch.__version__, triton.__version__)"
```
Expected: one venv resolves. (Checked live 2026-09-19: both engines pin `torch==2.13.0`; transformers pins `>=5.10.4` and `==5.12.1` are compatible.) **If the joint install fails to resolve:** fall back to `/workspace/vllm-venv` and `/workspace/sglang-venv`, install dispatch (`--no-deps -e`) into each, run each engine's race, dispatch's own runs and the serving reference in its own venv, and **record each venv's Triton version** in the findings (the shared-compiler fairness argument no longer holds and the write-up must say so). SGLang's base install may lack serving extras; if `python -m sglang.launch_server --help` errors on a missing module, install the named package and record it.

- [ ] **Step 7: Verify `vllm bench serve`'s flags against the installed version**

```bash
$E -m vllm.entrypoints.cli.main bench serve --help 2>&1 | grep -E "dataset-name|skip-chat-template|custom-output-len|ignore-eos|save-detailed|num-warmups|max-concurrency|result-filename|--backend"
```
(or `vllm bench serve --help` if the entry point is on `PATH`). Expected: every flag `build_bench_command` uses is listed. If one is missing or renamed, fix `build_bench_command` in `serving_bench.py` with a failing test first (`tests/unit/test_serving_bench.py` asserts each flag), then re-run Task 8's tests locally.

- [ ] **Step 8: Run the real-engine correctness gate on the pod**

```bash
cd /workspace/dispatch
$E -m pytest tests/unit/test_engines_real_gpu.py -m gpu -v -x
```
Expected: every engine passes `assert_matches_reference` at token counts 1/64/512, both routing distributions, bf16 and int8, and every mutation test raises. This is the first time the real engines run; expect API drift. Known failure modes and their fixes, each applied test-first (write the failing CPU test that captures the requirement, then the change; `make check` locally before re-shipping):

- **vLLM: `Current vLLM config is not set` (or similar).** vLLM's fused-MoE reads a global config. Give `VllmEngine` an injectable `config_context: Callable[[], ContextManager[Any]]` (default: lazily `lambda: set_current_vllm_config(VllmConfig())` from `vllm.config`; tests pass `contextlib.nullcontext`) and call `fused_experts` inside it in the bound closure. The context-manager overhead is then inside the timed region: note it in the findings.
- **vLLM/SGLang int8: a scale dtype or shape error.** Cast `w1_scale`/`w2_scale` to the activation dtype in the adapter's `prepare_int8`, and add a CPU test that the fake engine receives that dtype.
- **SGLang: a missing global (server args, a forward context, a CUDA-graph flag).** Read the traceback, initialize exactly what it names inside `init_distributed()`, and add nothing speculative.
- **Timebox: 30 minutes per engine.** Past it, mark that engine "not measurable at the pinned version", keep the failing output, and continue with the others.

Any adapter change is its own commit (`fix: ...`) with `make check` green and a STATUS bullet. If nothing needed fixing, there is nothing to commit.

- [ ] **Step 9: Record the checkpoint**

Note elapsed pod time and running cost. Setup typically consumes the first hour or two; if elapsed cost already exceeds $2, tell the user before Task 11.

---

### Task 11: Stage 1 -- at-scale correctness gate

**Files:**
- Evidence (pod): `/workspace/evidence/phase-6/*-gate-*-results.json` (and `*-reference.safetensors`, which stay on the pod: ~210MB each and gitignored)
- Later committed: `docs/findings/phase-6/*-gate-*-results.json`

**Interfaces:**
- Consumes: Task 2's `--prompt-set gate`, Task 10's dispatch venv and wrapper.
- Produces: five results JSONs: `gate-stock` (the reference), `gate-stock-control`, `gate-naive`, `gate-persistent`, `gate-quantized`, each carrying `gap_split`. Consumed by Task 12's per-config decision and Task 15's findings.

- [ ] **Step 1: Run the stock reference and the stock-vs-stock control**

```bash
cd /workspace/dispatch
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
D=$(date +%Y-%m-%d); OUT=/workspace/evidence/phase-6; mkdir -p $OUT
GATE="--prompt-set gate --repetitions 1 --max-new-tokens 1 --trust-remote-code --output-dir $OUT"
REF=$OUT/${D}-phase-6-gate-stock-reference.safetensors

.venv/bin/python _patch_and_run_baseline.py $GATE --run-label ${D}-phase-6-gate-stock
.venv/bin/python _patch_and_run_baseline.py $GATE --run-label ${D}-phase-6-gate-stock-control --compare-reference $REF
```
Expected: the first run writes the reference; the control exits 0 and its `gap_split` shows `positions` of about 1036 and `large_gap_disagreements` of 0.

**Decision (pre-registered):** if the control reports any `large_gap_disagreements`, the gate's own noise floor is broken. **Stop.** Do not spend on Tasks 12-13. Write up the finding, pull the evidence (Task 14), and tell the user. Near-tie disagreements in the control are expected-possible (Phase 5b saw the same config differ between processes) and are recorded, not failed.

- [ ] **Step 2: Gate each config**

Each is a separate invocation: a failing config exits non-zero and must not stop the others.

```bash
for KERNEL in naive persistent quantized; do
  .venv/bin/python _patch_and_run_baseline.py $GATE --moe-kernel $KERNEL \
    --run-label ${D}-phase-6-gate-$KERNEL --compare-reference $REF \
    || echo "GATE TRIPPED: $KERNEL"
done
```
Expected per config: exit 0, `gap_split.positions` about 1036, `large_gap_disagreements` 0, `near_tie_disagreements` a small count (bf16 kernels likely 0-few; int8 more).

- [ ] **Step 3: Read the results and apply the per-config rule**

```bash
for f in $OUT/*-phase-6-gate-*-results.json; do echo "== $f"; python3 -c "import json,sys; d=json.load(open('$f')); print(d['moe_kernel'], d['gap_split'])"; done
```
A config whose `large_gap_disagreements > 0` **failed its gate**: exclude it from Task 12's race (its rows read "failed correctness gate"), keep the evidence, and continue with the others. Do not change `LARGE_GAP_THRESHOLD`. If every kernel config trips, stop and ask the user (something systematic, such as the RoPE bug class from Phase 5b, is likelier than three independent bugs).

- [ ] **Step 4: Record the checkpoint**

Elapsed cost so far; at $5 or more, pause and report before Task 12.

---

### Task 12: Stage 2 -- the kernel race

**Files:**
- Evidence (pod): `/workspace/evidence/phase-6/*-race-*.json`, `*-race-summary.{json,md}`
- Pod-local: `/workspace/race-inputs/` (seeded inputs and references), `/workspace/vllm-tuned/`, `/workspace/sglang-moe-config/`, `/workspace/vllm-src`, `/workspace/sglang-src`, `/workspace/deepseek-config-shim/`

**Interfaces:**
- Consumes: Tasks 5-7, Task 11's per-config gate outcome, Task 10's engines venv.
- Produces: per-engine race records (`config.tuning_label` of `sweep`, `default` or `tuned`) and the merged summary table. Consumed by Task 15.

- [ ] **Step 1: Generate the seeded inputs and references once**

```bash
cd /workspace/dispatch
E=/workspace/engines-venv/bin/python
$E -m scripts.run_engine_race prepare --out-dir /workspace/race-inputs
```
Expected: `wrote inputs to /workspace/race-inputs`; the directory holds `weights.safetensors`, `manifest.json`, and 14 `case_<tokens>_<distribution>.safetensors` files (7 token counts x 2 distributions).

- [ ] **Step 2: dispatch's tile-size sweeps**

Skip any dispatch config that failed its Task 11 gate (record it as "failed correctness gate").

```bash
R="$E -m scripts.run_engine_race run --inputs-dir /workspace/race-inputs --output-dir $OUT"
$R --engine dispatch-naive      --precision bf16 --tuning-label sweep
$R --engine dispatch-persistent --precision bf16 --tuning-label sweep
$R --engine dispatch-naive      --precision int8 --tuning-label sweep
```
Expected: each exits 0 and writes a record with `status: ok` for 4 tile sizes x 7 token counts x 2 distributions (56 results). (The persistent kernel has no int8 variant, by design.) A non-zero exit means an engine variant disagreed with the reference: read the record's `reason`.

- [ ] **Step 3: vLLM and SGLang, default configs**

```bash
unset VLLM_TUNED_CONFIG_FOLDER SGLANG_MOE_CONFIG_DIR
for ENG in vllm sglang; do
  for PREC in bf16 int8; do
    $R --engine $ENG --precision $PREC --tuning-label default
  done
done
```
Expected: four records. Check each record's `config.environment`: the `*_files` lists show whether any tuned or shipped config for this GPU was present. An engine failing here follows Task 10 step 8's timebox rule.

- [ ] **Step 4: Tune vLLM (60-minute timebox per precision)**

```bash
git ls-remote --tags https://github.com/vllm-project/vllm | grep -E "refs/tags/v0.29.0$"   # the tag exists
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm /workspace/vllm-src
$E /workspace/vllm-src/benchmarks/kernels/benchmark_moe.py --help | head -40   # confirm flag names first
mkdir -p /workspace/vllm-tuned && cd /workspace/vllm-tuned
timeout 3600 $E /workspace/vllm-src/benchmarks/kernels/benchmark_moe.py \
  --model deepseek-ai/deepseek-moe-16b-base --tp-size 1 --trust-remote-code --tune \
  --batch-size 1 4 16 64 128 512 2048
timeout 3600 $E /workspace/vllm-src/benchmarks/kernels/benchmark_moe.py \
  --model deepseek-ai/deepseek-moe-16b-base --tp-size 1 --trust-remote-code --tune \
  --dtype int8_w8a16 --batch-size 1 4 16 64 128 512 2048
ls /workspace/vllm-tuned
```
(This is the same tool Phase 2 ran on this exact model, with `--tp-size 1` for a real single-GPU shape.) Adjust flag names to what `--help` shows; `--dtype` values and where the tuner saves its JSON are the two things to confirm. Expected: files like `E=64,N=1408,device_name=NVIDIA_L40.json` and `...,dtype=int8_w8a16.json`. If the tag does not exist, use `main` and record the commit SHA. **Token counts the tuner did not finish are labeled `default` in the findings, not `tuned`.**

- [ ] **Step 5: Tune SGLang (60-minute timebox per precision)**

SGLang's tuner keys its model shapes off an explicit architecture list, and V1's `DeepseekForCausalLM` is not in it (checked live 2026-09-19); V2's branch reads `n_routed_experts`, `num_experts_per_tok` and `moe_intermediate_size`, exactly V1's config fields. A config-only shim relabeling the architecture makes the tuner read the right shapes.

```bash
git ls-remote --tags https://github.com/sgl-project/sglang | grep -E "refs/tags/v0.5.20$"
git clone --depth 1 --branch v0.5.20 https://github.com/sgl-project/sglang /workspace/sglang-src
/workspace/dispatch/.venv/bin/huggingface-cli download deepseek-ai/deepseek-moe-16b-base \
  --include "*.json" "*.py" --local-dir /workspace/deepseek-config-shim
python3 - <<'EOF'
import json, pathlib
path = pathlib.Path("/workspace/deepseek-config-shim/config.json")
config = json.loads(path.read_text())
config["architectures"] = ["DeepseekV2ForCausalLM"]  # shape-reading shim only; no weights involved
path.write_text(json.dumps(config, indent=2))
EOF
cd /workspace/sglang-src/benchmark/kernels/fused_moe_triton
$E tuning_fused_moe_triton.py --help | head -40   # confirm flag names first
timeout 3600 $E tuning_fused_moe_triton.py --model /workspace/deepseek-config-shim \
  --tp-size 1 --tune --batch-size 1 4 16 64 128 512 2048
timeout 3600 $E tuning_fused_moe_triton.py --model /workspace/deepseek-config-shim \
  --tp-size 1 --dtype int8_w8a16 --tune --batch-size 1 4 16 64 128 512 2048
```
SGLang reads tuned configs from `$SGLANG_MOE_CONFIG_DIR/configs/triton_<version>/`, where `<version>` is Triton's version with dots as underscores (checked live in `fused_moe_triton_config.py`). Install what the tuner wrote:

```bash
TV=$($E -c "import triton; print('triton_' + triton.__version__.replace('.', '_'))")
mkdir -p /workspace/sglang-moe-config/configs/$TV
cp E=64,N=1408*.json /workspace/sglang-moe-config/configs/$TV/
```
**If the tuner cannot be made to run within the timebox** (unrecognized flags, an architecture error the shim does not cure), stop, keep the error output, and report SGLang with its default config only, labeled as untuned. That is an honest outcome, not a failure of the phase.

- [ ] **Step 6: vLLM and SGLang, tuned configs**

```bash
export VLLM_TUNED_CONFIG_FOLDER=/workspace/vllm-tuned
export SGLANG_MOE_CONFIG_DIR=/workspace/sglang-moe-config
cd /workspace/dispatch
for ENG in vllm sglang; do
  for PREC in bf16 int8; do
    $R --engine $ENG --precision $PREC --tuning-label tuned
  done
done
```
**Verify the tuned configs were actually loaded, twice:** (1) each tuned record's `config.environment.*_files` lists the tuned JSON files; (2) its results differ from the matching `default` record's at some token count. If a tuned record's numbers are identical to default at every token count, the config was not picked up (a filename-suffix mismatch such as `dtype=` or `per_channel_quant=True` is the usual cause): fix the placement, do not report it as tuned.

- [ ] **Step 7: Merge into the table**

```bash
$E -m scripts.run_engine_race merge --results-dir $OUT --output-dir $OUT \
  --run-label ${D}-phase-6-race-summary
```
Expected: `${D}-phase-6-race-summary.md` with one table per (precision, tuning) and a column per engine. Read it against the pre-registered rules; note every `n/a`.

- [ ] **Step 8: Record the checkpoint**

Elapsed cost; at $5 or more pause and report; at $8 skip Task 13's remaining work and go to Task 14.

---

### Task 13: Stage 3 -- the engine reference

**Files:**
- Evidence (pod): `/workspace/evidence/phase-6/*-phase-6-reference-{vllm,sglang}.json`, `*-c<N>-raw.json`, `*-server.log`, `*-trace.jsonl`, `*-phase-6-ref-dispatch-{stock,naive}-results.json`

**Interfaces:**
- Consumes: Task 8's `phase6_engine_reference`, Task 3's `--ignore-eos`, Task 2's `GATE_PROMPTS`.
- Produces: per-engine serving summaries at concurrency 1/4/16/64 (with `$/Mtok` from `RATE`), and dispatch's concurrency-1 rows. Consumed by Task 15.

- [ ] **Step 1: dispatch's concurrency-1 rows (GPU must be free)**

```bash
cd /workspace/dispatch
nvidia-smi --query-gpu=memory.used --format=csv   # near 0 before each model load
REFRUN="--prompt-set gate --repetitions 2 --max-new-tokens 64 --ignore-eos --trust-remote-code --output-dir $OUT"
.venv/bin/python _patch_and_run_baseline.py $REFRUN --run-label ${D}-phase-6-ref-dispatch-stock
.venv/bin/python _patch_and_run_baseline.py $REFRUN --moe-kernel naive --run-label ${D}-phase-6-ref-dispatch-naive
```
Expected: two results JSONs, `run_count` 32 (16 prompts x 2), `ignore_eos: true`, `max_new_tokens: 64`. Their `mean_tokens_per_second` is per-request output tokens divided by end-to-end latency, the same definition Task 8's summary uses for the engines' `mean_request_tokens_per_s`.

- [ ] **Step 2: vLLM's server and benchmark (30-minute timebox to a healthy server)**

```bash
E=/workspace/engines-venv/bin/python
$E -m scripts.gpu.phase6_engine_reference --engine vllm --gpu-cost-per-hour $RATE \
  --output-dir $OUT --run-label ${D}-phase-6-reference-vllm
```
Expected: the script starts `vllm serve`, waits for `/health`, runs `vllm bench serve` at concurrency 1, 4, 16, 64, prints `wrote <path>`, and terminates the server. Then `nvidia-smi` shows memory near 0 again. A refused run (failed requests, or any output not exactly 64 tokens) exits non-zero after writing `error` into the JSON: read it. If the server never becomes healthy, the server log in `$OUT` says why (a DeepSeek remote-code or config-class break under vLLM's own pinned transformers is the likely one): keep it, apply the 30-minute rule, and report "could not serve this checkpoint at the pinned version".

- [ ] **Step 3: SGLang's server and benchmark (same rules)**

```bash
$E -m scripts.gpu.phase6_engine_reference --engine sglang --gpu-cost-per-hour $RATE \
  --output-dir $OUT --run-label ${D}-phase-6-reference-sglang
```
Same expectations and failure handling.

- [ ] **Step 4: Sanity-check the numbers before believing them**

For each engine's JSON: `completed` equals `num_prompts` at every concurrency; `output_tokens_per_s` rises with concurrency (if it does not rise at all, the client or server is the bottleneck, not the engine: say so); concurrency-1 `mean_request_tokens_per_s` is in the same order of magnitude as dispatch's stock row (10-25 tok/s in Phases 1/5a on an L40); `cost_per_million_output_tokens_usd` matches `RATE / (output_tokens_per_s * 3600) * 1e6`. A number that fails a sanity check is investigated, not reported.

---

### Task 14: Evidence, teardown, and cost record

**Files:**
- Create: `docs/findings/phase-6/*` (the pulled evidence, minus `*.safetensors`), `docs/findings/phase-6/<date>-phase-6-final-benchmark-cost.md`

**Interfaces:**
- Consumes: everything on the pod.
- Produces: all evidence committed to the repo, the pod terminated, the cost measured and recorded.

- [ ] **Step 1: Pull every evidence file off the pod before anything stops it**

The `ssh.runpod.io` proxy rejects SFTP, so stream a tarball through the PTY as wrapped base64 with explicit markers, then decode locally. Exclude the large gitignored reference tensors.

```bash
ssh -tt $POD_SSH "cd /workspace/evidence && echo __BEGIN__ && tar -czf - --exclude='*.safetensors' phase-6 | base64 -w 76 && echo __END__ && exit" > /tmp/phase6-evidence.raw
tr -d '\r' < /tmp/phase6-evidence.raw | awk '/^__BEGIN__$/{f=1;next} /^__END__$/{f=0} f' | base64 -d | tar -xzf - -C docs/findings/
ls -l docs/findings/phase-6 | head -60
```
Verify by count and size against the pod (`ls -l /workspace/evidence/phase-6` over ssh): every JSON present, none zero-length, every gate/race/reference record and every server log accounted for. Also copy the tuned config JSONs (`/workspace/vllm-tuned/*.json`, `/workspace/sglang-moe-config/configs/*/*.json`) the same way into `docs/findings/phase-6/tuned-configs/`: they are the tuning provenance.

**Do not stop or terminate the pod until this step's verification passes.**

- [ ] **Step 2: Terminate the pod**

```bash
uv run python -m scripts.gpu.provision terminate --pod-id $POD_ID
```
Everything needed is now in the repo. (Terminate, not stop: a stopped pod may be unrestartable and still bills for its disk.) Note the end time (`date -u`).

- [ ] **Step 3: Record the measured cost**

Read the pod's billed cost from RunPod (MCP `list-pod-billing`, or the console) and compare it with `RATE x duration`. Then:

```bash
uv run python - <<'EOF'
from pathlib import Path

from scripts.gpu.provision import write_cost_record

write_cost_record(
    Path("docs/findings/phase-6"),
    pod_id="<POD_ID>",
    gpu_type_id="<the L40 id>",
    cost_per_hour=<RATE>,
    duration_s=<measured seconds>,
    note="Phase 6 single session: at-scale gate, kernel race, engine reference. Billed cost from RunPod: $<billed>.",
    run_label="phase-6-final-benchmark",
)
EOF
```
Replace the angle-bracket values with the measured ones; the cost file states measured numbers only.

- [ ] **Step 4: Commit the evidence**

STATUS bullet: `- **Task 14 (evidence + teardown)**: evidence pulled from the pod and verified before termination; pod terminated; total measured cost $<X> of the $10 cap.`

```bash
git add docs/findings/phase-6 docs/STATUS.md
git commit -m "docs: Phase 6 raw evidence (gate, race, engine reference) and measured cost"
```

---

### Task 15: Findings, docs, review, and PR

**Files:**
- Create: `docs/findings/phase-6/<date>-phase-6-final-benchmark-run.md`, `docs/runbooks/phase-6-final-benchmark.md`
- Modify: `docs/STATUS.md`, `README.md`, `CLAUDE.md`, `CHANGELOG.md` (if it exists; Phase 5b's missing entry was a real omission)

**Interfaces:**
- Consumes: all committed evidence from Task 14.
- Produces: the written result, an accurate STATUS, refreshed README and CLAUDE.md, a runbook of what actually worked, and one PR.

- [ ] **Step 1: Write the findings doc, from the evidence only**

`docs/findings/phase-6/<date>-phase-6-final-benchmark-run.md`. Required, in this order:

1. **Outcome first**, in plain words, including whatever the data says even if it is unflattering (a null or negative result is a result). State the claim the way this repo's rules require: config alongside every number.
2. **The gate:** a table of the five runs (positions, top-1 agreement, near-tie and large-gap counts, widest gap), the stock-vs-stock control's own noise floor, and the per-config consequence. State plainly that this replaces Phase 5a's 29-position claim for the benchmarked config.
3. **The race:** the merged tables, verbatim from `*-race-summary.md`, one per (precision, tuning), with the tuning provenance (which tuner, how long it ran, which token counts got a tuned config), engine versions, torch and Triton versions, GPU. Then per-shape observations that the table directly supports, and no others. Do not write "beats" or "loses to" except where a specific row says so.
4. **The engine reference:** vLLM and SGLang at concurrency 1/4/16/64 (TTFT, ITL, end-to-end, tokens/s, $/Mtok from the measured rate), and dispatch's concurrency-1 rows with the non-comparabilities stated in the table itself (in-process vs. HTTP, no CUDA graphs, no batched server; concurrency 4/16/64 are "n/a: no dispatch server").
5. **What went wrong and what it cost:** every adapter fix from Task 10, every timebox that fired, every engine that could not be measured and why.
6. **Explicit non-claims:** what this does not show (single GPU class, one model, synthetic weights in the race, one seed, kernel-level rather than end-to-end).
7. **Cost:** measured, per the cost record, against the $10 cap.

- [ ] **Step 2: Write the runbook from what actually worked**

`docs/runbooks/phase-6-final-benchmark.md`, in the style of the earlier runbooks (dated "actual finding" notes, the exact commands that worked, every place the plan's command needed changing). Record the corrected flag names, the SGLang shim, the tuned-config placement, and any adapter fix.

- [ ] **Step 3: Update STATUS, README, CLAUDE.md, CHANGELOG**

STATUS: rewrite "Phase 6 progress" as a completed-phase record (outcome, config, numbers, cost) and update "Next step" (Phase 7 is next). README and CLAUDE.md: ask what changed today that a reader would want to know, not whether the file is still technically accurate: the Phase 6 headline result and its honest limits, the gate's outcome, the two new cost-discipline lessons if any, and the corrected "Current status" block. CHANGELOG: add the Phase 6 entry if the file exists.

- [ ] **Step 4: Run the story-bank check**

The `story-bank-reminder.sh` hook fires on the STATUS.md commit. Decide whether this phase earned a STAR entry (the near-tie gate design, the design assumption a live check disproved, any real incident) and, if so, update the gist named in `.claude/story-bank-gist-id`.

- [ ] **Step 5: Whole-branch review**

Run `/code-review` (or the `pr-review-toolkit` agents) over the branch diff against `main`. Phase 5b's whole-branch review caught defects that per-task tests did not; expect and fix findings, each as its own commit.

- [ ] **Step 6: Final gate**

```bash
make check
git log --oneline main..HEAD
```
Expected: green, and one commit per task plus any review-fix commits, all ASCII, none with an attribution line.

- [ ] **Step 7: Commit the docs**

```bash
git add docs/findings/phase-6 docs/runbooks/phase-6-final-benchmark.md docs/STATUS.md README.md CLAUDE.md CHANGELOG.md
git commit -m "docs: Phase 6 findings, runbook, and refreshed STATUS/README/CLAUDE.md"
```

- [ ] **Step 8: Ask before pushing**

Show the user the branch summary and ask whether to push and open the PR. Do not push or open it unasked. When they say yes: push `phase-6-final-benchmark`, open one PR whose body states the headline result with its config, the gate outcome, the measured cost, and the non-claims. After it merges, offer the `refresh-gh-profile` skill (this phase changes what belongs at the top of the GitHub profile), and record the merge in STATUS in a follow-up.
