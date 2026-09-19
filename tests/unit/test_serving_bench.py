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
    """A vLLM 0.29.0 bench-serve --save-detailed result for 4 requests of 64
    tokens. Every request's ttft is 0.1s and its 63 remaining tokens arrive at
    a steady 0.03s/token, so end-to-end latency (derived, not a raw key --
    0.29.0 carries no per-request "latencies" field) is 0.1 + 63*0.03 = 1.99s
    for every request, and tokens_per_s is 64 / 1.99."""
    raw: dict[str, Any] = {
        "completed": 4,
        "failed": 0,
        "duration": 8.0,
        "output_throughput": 32.0,  # 4 * 64 tokens / 8 s
        "output_lens": [64, 64, 64, 64],
        "ttfts": [0.1, 0.1, 0.1, 0.1],
        "itls": [[0.03] * 63, [0.03] * 63, [0.03] * 63, [0.03] * 63],
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

    assert summary.mean_ttft_ms == pytest.approx(100.0)
    assert summary.p50_ttft_ms == pytest.approx(100.0)
    assert summary.mean_request_tokens_per_s == pytest.approx(64 / 1.99)
    assert summary.p99_e2e_ms == pytest.approx(1990.0)
    assert summary.mean_itl_ms == pytest.approx(30.0)


def test_end_to_end_latency_is_derived_from_ttft_plus_the_inter_token_gaps() -> None:
    """The real vLLM 0.29.0 schema carries no per-request "latencies" key at
    all -- summarize_bench_result must not depend on one being present."""
    raw = _raw()
    assert "latencies" not in raw

    summary = _summarize(raw)

    assert summary.p50_e2e_ms == pytest.approx(1990.0)


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
