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
