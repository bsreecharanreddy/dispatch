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

    main(["--prompt-set", "default", "--output-dir", str(tmp_path), "--run-label", "ref-run"])

    results = json.loads((tmp_path / "ref-run-results.json").read_text())
    assert results["gap_split"] is None
    assert (tmp_path / "ref-run-reference.safetensors").exists()


def test_gate_without_a_reference_is_refused_before_spending_any_gpu_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--prompt-set gate promises to enforce the gap-split rule; without
    --compare-reference it had nothing to enforce against and silently
    passed. It must refuse instead of pretending to have gated anything."""
    run_baseline_mock = _fake_run_baseline(_gate_pair(600, {})[1], 0)
    calls: list[object] = []

    def recording_run_baseline(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return run_baseline_mock(*args, **kwargs)

    monkeypatch.setattr(run_baseline_module, "run_baseline", recording_run_baseline)
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: FAKE_SUMMARY)

    with pytest.raises(SystemExit, match="requires --compare-reference"):
        main(["--prompt-set", "gate", "--output-dir", str(tmp_path), "--run-label", "ref-run"])

    assert calls == []  # refused before the model was even loaded


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
    def fake_resolve_quantized_backend() -> object:
        return object()

    def fake_patch_moe_infer_quantized(model: object, matmul: object) -> int:
        return 0

    monkeypatch.setattr(
        run_baseline_module, "load_model", lambda *a, **k: ("the-model", "the-tokenizer")
    )
    monkeypatch.setattr(
        run_baseline_module, "resolve_quantized_backend", fake_resolve_quantized_backend
    )
    monkeypatch.setattr(
        run_baseline_module, "patch_moe_infer_quantized", fake_patch_moe_infer_quantized
    )

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
