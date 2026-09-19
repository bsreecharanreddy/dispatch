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
