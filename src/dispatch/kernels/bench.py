"""Kernel micro-benchmark helpers. Timing goes through
triton.testing.do_bench -- warmup, repetition, and GPU synchronization are
Triton's own tool's job, not a bespoke perf_counter loop -- and everything
else here is pure math, testable without a GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

DO_BENCH_WARMUP_MS = 25
DO_BENCH_REP_MS = 100

_ROUTING_WEIGHTS: dict[str, Callable[[int], torch.Tensor]] = {
    "uniform": torch.ones,
    "zipf": lambda n_experts: 1.0 / torch.arange(1, n_experts + 1, dtype=torch.float32),
}
ROUTING_DISTRIBUTIONS = tuple(_ROUTING_WEIGHTS)


@dataclass(frozen=True)
class KernelBenchmarkSummary:
    label: str
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    tflops: float


def grouped_gemm_flops(total_rows: int, n: int, k: int) -> float:
    """2*M*N*K; every group shares N and K, so only the total row count matters."""
    if total_rows < 0 or n <= 0 or k <= 0:
        raise ValueError(f"invalid dims: total_rows={total_rows}, n={n}, k={k}")
    return 2.0 * total_rows * n * k


def summarize_kernel_latencies(
    label: str, latencies_ms: tuple[float, float, float], *, flops: float
) -> KernelBenchmarkSummary:
    """latencies_ms is (mean, p50, p99), as do_bench reports them."""
    mean_ms, p50_ms, p99_ms = latencies_ms
    return KernelBenchmarkSummary(
        label=label,
        mean_latency_ms=mean_ms,
        p50_latency_ms=p50_ms,
        p99_latency_ms=p99_ms,
        tflops=flops / (mean_ms * 1e-3) / 1e12,
    )


def sample_topk_idx(
    num_tokens: int,
    n_experts: int,
    top_k: int,
    *,
    distribution: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Synthetic routing: `zipf` skews load toward low-index experts,
    `uniform` balances it. Each row holds top_k distinct experts."""
    if distribution not in _ROUTING_WEIGHTS:
        raise ValueError(
            f"unknown distribution {distribution!r}; expected one of {ROUTING_DISTRIBUTIONS}"
        )
    probs = _ROUTING_WEIGHTS[distribution](n_experts).expand(num_tokens, n_experts)
    return torch.multinomial(probs, top_k, replacement=False, generator=generator)


def time_grouped_gemm(
    fn: Callable[[], torch.Tensor], *, label: str, flops: float
) -> KernelBenchmarkSummary:
    import triton.testing  # noqa: PLC0415 -- keeps this module importable without triton

    mean_ms = float(
        triton.testing.do_bench(
            fn, warmup=DO_BENCH_WARMUP_MS, rep=DO_BENCH_REP_MS, return_mode="mean"
        )
    )
    p50_ms, p99_ms = (
        float(value)
        for value in triton.testing.do_bench(
            fn, warmup=DO_BENCH_WARMUP_MS, rep=DO_BENCH_REP_MS, quantiles=[0.5, 0.99]
        )
    )
    return summarize_kernel_latencies(label, (mean_ms, p50_ms, p99_ms), flops=flops)
