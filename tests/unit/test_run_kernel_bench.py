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
