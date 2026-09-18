"""main()'s plumbing (args -> files) is tested fast with
run_speculative_bench and summarize monkeypatched out, matching
test_run_baseline.py's pattern; the real end-to-end guard test is `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_speculative_bench as run_speculative_bench_module
import torch
from scripts.run_speculative_bench import build_drafter, main

from dispatch.benchmark.metrics import BenchmarkSummary
from dispatch.speculative.drafters import DraftModelDrafter, PromptLookupDrafter
from dispatch.speculative.reference import save_generated_tokens

FAKE_SUMMARY = BenchmarkSummary(
    run_count=1,
    mean_ttft=0.1,
    p50_ttft=0.1,
    p99_ttft=0.1,
    mean_inter_token_latency=0.05,
    mean_tokens_per_second=20.0,
)
FAKE_TOKENS: dict[str, tuple[int, ...]] = {"prompt_000_tokens": (1, 2, 3)}


def _fake_run_speculative_bench(
    tokens: dict[str, tuple[int, ...]], moe_layers_patched: int, rope_buffers_fixed: int = 27
) -> object:
    def fake(model_name: str, **kwargs: object) -> tuple[list[object], dict, int, int]:  # type: ignore[type-arg]
        return [(object(), (2, 1))], tokens, moe_layers_patched, rope_buffers_fixed

    return fake


def test_build_drafter_none_returns_none() -> None:
    assert (
        build_drafter(
            "none",
            draft_model_name="x",
            device="cpu",
            dtype=torch.float32,
            prompt_lookup_ngram_size=3,
        )
        is None
    )


def test_build_drafter_prompt_lookup_returns_a_prompt_lookup_drafter() -> None:
    drafter = build_drafter(
        "prompt-lookup",
        draft_model_name="x",
        device="cpu",
        dtype=torch.float32,
        prompt_lookup_ngram_size=5,
    )

    assert isinstance(drafter, PromptLookupDrafter)
    assert drafter.ngram_size == 5


def test_build_drafter_draft_model_loads_and_wraps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_load_model(*args: object, **kwargs: object) -> tuple[str, str]:
        captured_kwargs.update(kwargs)
        return "the-model", "the-tokenizer"

    monkeypatch.setattr(run_speculative_bench_module, "load_model", fake_load_model)

    drafter = build_drafter(
        "draft-model",
        draft_model_name="deepseek-ai/deepseek-llm-7b-base",
        device="cpu",
        dtype=torch.float32,
        prompt_lookup_ngram_size=3,
    )

    assert isinstance(drafter, DraftModelDrafter)
    model: object = drafter.model  # load_model is monkeypatched to return a plain str here
    assert model == "the-model"
    assert captured_kwargs["attn_implementation"] == "sdpa"
    # fix_rope_inv_freq itself now runs inside load_model, for every
    # caller (see test_harness.py) -- not spied on here, since
    # load_model is mocked out in this test and build_drafter no longer
    # calls it a second time.


def test_build_drafter_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown drafter"):
        build_drafter(
            "bogus",
            draft_model_name="x",
            device="cpu",
            dtype=torch.float32,
            prompt_lookup_ngram_size=3,
        )


def test_main_writes_results_and_generated_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(FAKE_TOKENS, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

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
    assert results["moe_layers_patched"] == 27
    assert results["rope_buffers_fixed"] == 27
    assert results["attn_implementation"] == "sdpa"
    assert results["acceptance_rate"] == pytest.approx(
        1.5 / 4
    )  # mean(2, 1) / num_speculative_tokens
    assert results["token_match"] == {}
    assert (tmp_path / "test-run-generated-tokens.json").exists()


def test_main_records_agreement_with_a_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "baseline-generated-tokens.json"
    save_generated_tokens(FAKE_TOKENS, reference_path)
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(FAKE_TOKENS, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

    main(
        [
            "--drafter",
            "prompt-lookup",
            "--compare-generated-tokens",
            str(reference_path),
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "spec-run",
        ]
    )

    results = json.loads((tmp_path / "spec-run-results.json").read_text())
    assert results["token_match"] == {"prompt_000_tokens": True}


def test_main_exits_nonzero_on_token_divergence_but_keeps_the_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_path = tmp_path / "baseline-generated-tokens.json"
    save_generated_tokens(FAKE_TOKENS, reference_path)
    wrong_tokens: dict[str, tuple[int, ...]] = {"prompt_000_tokens": (1, 2, 9)}
    monkeypatch.setattr(
        run_speculative_bench_module,
        "run_speculative_bench",
        _fake_run_speculative_bench(wrong_tokens, 27),
    )
    monkeypatch.setattr(run_speculative_bench_module, "summarize", lambda runs: FAKE_SUMMARY)

    with pytest.raises(SystemExit, match="disagree"):
        main(
            [
                "--drafter",
                "draft-model",
                "--compare-generated-tokens",
                str(reference_path),
                "--output-dir",
                str(tmp_path),
                "--run-label",
                "spec-run",
            ]
        )

    results = json.loads((tmp_path / "spec-run-results.json").read_text())
    assert results["token_match"] == {"prompt_000_tokens": False}
    assert (tmp_path / "spec-run-generated-tokens.json").exists()


@pytest.mark.gpu
@pytest.mark.slow
def test_run_speculative_bench_refuses_a_target_that_patches_nothing() -> None:
    with pytest.raises(RuntimeError, match="patched no MoE layers"):
        run_speculative_bench_module.run_speculative_bench(
            "hf-internal-testing/tiny-random-gpt2",
            device="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            prompts=["hello"],
            repetitions=1,
            max_new_tokens=3,
            drafter_name="none",
            draft_model_name="unused",
            num_speculative_tokens=4,
            prompt_lookup_ngram_size=3,
        )
