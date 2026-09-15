"""main()'s plumbing (args -> files) is tested fast with run_baseline and
summarize monkeypatched out; the one real end-to-end pass against a tiny
model is `slow`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_baseline as run_baseline_module
import torch
from scripts.run_baseline import main

from dispatch.benchmark.metrics import BenchmarkSummary


def test_main_writes_results_and_reference_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_summary = BenchmarkSummary(
        run_count=1,
        mean_ttft=0.1,
        p50_ttft=0.1,
        p99_ttft=0.1,
        mean_inter_token_latency=0.05,
        mean_tokens_per_second=20.0,
    )

    def fake_run_baseline(
        model_name: str, **kwargs: object
    ) -> tuple[list[object], dict[str, torch.Tensor]]:
        return [object()], {"prompt_000_logits": torch.zeros(1)}

    monkeypatch.setattr(run_baseline_module, "run_baseline", fake_run_baseline)
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: fake_summary)

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
    assert (tmp_path / "test-run-reference.safetensors").exists()


@pytest.mark.slow
def test_run_baseline_end_to_end_with_a_real_tiny_model() -> None:
    runs, reference = run_baseline_module.run_baseline(
        "hf-internal-testing/tiny-random-gpt2",
        device="cpu",
        dtype=torch.float32,
        trust_remote_code=False,
        prompts=["hello"],
        repetitions=1,
        max_new_tokens=3,
    )

    assert len(runs) == 1
    assert "prompt_000_logits" in reference
