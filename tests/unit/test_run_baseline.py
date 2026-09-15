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
