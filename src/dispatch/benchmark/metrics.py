"""Pure latency/throughput metrics computed from per-token timestamps.

Kept free of torch/transformers imports on purpose: this is the one part
of the benchmark stack that needs no model, no GPU, and no network to test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenTimings:
    """Timestamps (seconds, from a monotonic clock) captured during one generation run."""

    start_time: float
    token_times: tuple[float, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        if not self.token_times:
            raise ValueError("token_times must have at least one entry")

    @property
    def generated_token_count(self) -> int:
        return len(self.token_times)

    @property
    def time_to_first_token(self) -> float:
        return self.token_times[0] - self.start_time

    @property
    def inter_token_latencies(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False))

    @property
    def total_latency(self) -> float:
        return self.token_times[-1] - self.start_time

    @property
    def tokens_per_second(self) -> float:
        return self.generated_token_count / self.total_latency


@dataclass(frozen=True)
class BenchmarkSummary:
    run_count: int
    mean_ttft: float
    p50_ttft: float
    p99_ttft: float
    mean_inter_token_latency: float
    mean_tokens_per_second: float


def summarize(runs: list[TokenTimings]) -> BenchmarkSummary:
    if not runs:
        raise ValueError("runs must not be empty")
    ttfts = sorted(run.time_to_first_token for run in runs)
    all_itls = [latency for run in runs for latency in run.inter_token_latencies]
    return BenchmarkSummary(
        run_count=len(runs),
        mean_ttft=statistics.mean(ttfts),
        p50_ttft=_percentile(ttfts, 0.50),
        p99_ttft=_percentile(ttfts, 0.99),
        mean_inter_token_latency=statistics.mean(all_itls) if all_itls else 0.0,
        mean_tokens_per_second=statistics.mean(run.tokens_per_second for run in runs),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
