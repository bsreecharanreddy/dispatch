"""Pure math -- no model, no I/O, fabricated timestamps throughout."""

from __future__ import annotations

import pytest

from dispatch.benchmark.metrics import TokenTimings, summarize


def test_token_timings_computes_ttft_and_latency() -> None:
    timing = TokenTimings(start_time=10.0, token_times=(10.1, 10.15, 10.25), prompt_token_count=5)

    assert timing.time_to_first_token == pytest.approx(0.1)
    assert timing.generated_token_count == 3
    assert timing.total_latency == pytest.approx(0.25)
    assert timing.inter_token_latencies == pytest.approx((0.05, 0.10))
    assert timing.tokens_per_second == pytest.approx(3 / 0.25)


def test_token_timings_rejects_empty_token_times() -> None:
    with pytest.raises(ValueError, match="token_times"):
        TokenTimings(start_time=0.0, token_times=(), prompt_token_count=1)


def test_summarize_aggregates_across_runs() -> None:
    runs = [
        TokenTimings(start_time=0.0, token_times=(0.1, 0.2), prompt_token_count=1),
        TokenTimings(start_time=0.0, token_times=(0.2, 0.4), prompt_token_count=1),
    ]

    summary = summarize(runs)

    assert summary.run_count == 2
    assert summary.mean_ttft == pytest.approx((0.1 + 0.2) / 2)
    assert summary.p50_ttft == pytest.approx(0.15)
    assert summary.mean_inter_token_latency == pytest.approx((0.1 + 0.2) / 2)


def test_summarize_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="runs"):
        summarize([])
