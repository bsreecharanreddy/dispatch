"""Pure math and synthetic routing -- no GPU, and no triton import (the one
function that needs triton imports it locally)."""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.bench import grouped_gemm_flops, sample_topk_idx, summarize_kernel_latencies


def test_grouped_gemm_flops_is_2mnk() -> None:
    assert grouped_gemm_flops(total_rows=10, n=20, k=30) == 2 * 10 * 20 * 30


def test_grouped_gemm_flops_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="invalid dims"):
        grouped_gemm_flops(total_rows=-1, n=1, k=1)


def test_summarize_kernel_latencies_computes_tflops_from_the_mean() -> None:
    summary = summarize_kernel_latencies("naive/gemm", (1.0, 0.9, 1.5), flops=2e9)

    assert summary.label == "naive/gemm"
    assert (summary.mean_latency_ms, summary.p50_latency_ms, summary.p99_latency_ms) == (
        1.0,
        0.9,
        1.5,
    )
    assert summary.tflops == pytest.approx(2.0)


@pytest.mark.parametrize("distribution", ["uniform", "zipf"])
def test_sample_topk_idx_gives_distinct_in_range_experts(distribution: str) -> None:
    topk_idx = sample_topk_idx(
        50, 64, 6, distribution=distribution, generator=torch.Generator().manual_seed(0)
    )

    assert topk_idx.shape == (50, 6)
    assert int(topk_idx.min()) >= 0
    assert int(topk_idx.max()) < 64
    assert all(len(set(row.tolist())) == 6 for row in topk_idx)


def test_zipf_routing_is_skewed_toward_low_index_experts() -> None:
    topk_idx = sample_topk_idx(
        2000, 64, 6, distribution="zipf", generator=torch.Generator().manual_seed(0)
    )

    counts = torch.bincount(topk_idx.reshape(-1), minlength=64)
    assert int(counts[:8].sum()) > int(counts[-8:].sum())


def test_sample_topk_idx_rejects_unknown_distribution() -> None:
    with pytest.raises(ValueError, match="unknown distribution"):
        sample_topk_idx(1, 4, 1, distribution="pareto", generator=torch.Generator())
