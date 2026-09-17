"""CLI: time the grouped-GEMM backends -- eager torch loop, naive Triton,
persistent Triton -- on synthetic DeepSeekMoE-16B-shaped routed-expert work,
and write every summary with its full config to docs/findings/. A backend
that disagrees with the eager torch backend on the benchmark's own input is
refused, not timed.

The int8 quantized kernel (Phase 5a) has no backend here by design: its
own correctness/throughput claims come from run_baseline.py's real-model
--compare-reference gate instead, since a synthetic-weight micro-benchmark
would need to fabricate a quantization to time in the first place.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dispatch.kernels.backends import BACKENDS, resolve_backend
from dispatch.kernels.bench import (
    DO_BENCH_REP_MS,
    DO_BENCH_WARMUP_MS,
    ROUTING_DISTRIBUTIONS,
    KernelBenchmarkSummary,
    grouped_gemm_flops,
    sample_topk_idx,
    time_grouped_gemm,
)
from dispatch.kernels.grouping import group_tokens_by_expert
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    assert_matches_reference,
    grouped_moe_routed,
)
from dispatch.kernels.tile_schedule import build_tile_schedule

# deepseek-ai/deepseek-moe-16b-base config.json, checked live 2026-09-15.
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 1408
N_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 6


def run_kernel_bench(
    *, num_tokens: int, distribution: str, dtype: torch.dtype, block_m: int, seed: int
) -> list[KernelBenchmarkSummary]:
    """Per backend: the whole routed MoE layer (grouping and scheduling
    overhead included), and one gate_proj-shaped grouped GEMM on its own."""
    cuda = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=dtype, generator=cuda)
    weights = StackedExpertWeights(
        gate=_expert_weights(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype, cuda),
        up=_expert_weights(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype, cuda),
        down=_expert_weights(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype, cuda),
    )
    topk_idx = sample_topk_idx(
        num_tokens,
        N_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        distribution=distribution,
        generator=torch.Generator().manual_seed(seed),
    ).cuda()
    topk_weight = torch.rand(
        num_tokens, NUM_EXPERTS_PER_TOK, device="cuda", dtype=dtype, generator=cuda
    )
    grouping = group_tokens_by_expert(topk_idx, topk_weight, N_ROUTED_EXPERTS)
    schedule = build_tile_schedule(grouping.group_sizes, block_m)
    gathered = x[grouping.sorted_token_idx]
    gemm_flops = grouped_gemm_flops(
        num_tokens * NUM_EXPERTS_PER_TOK, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE
    )

    expected: torch.Tensor | None = None
    summaries: list[KernelBenchmarkSummary] = []
    for name in BACKENDS:
        matmul = resolve_backend(name)
        layer = functools.partial(
            grouped_moe_routed, x, topk_idx, topk_weight, weights, matmul, block_m=block_m
        )
        if expected is None:
            expected = layer()
        else:
            _require_agreement(layer(), expected, name)
        gemm = functools.partial(matmul, gathered, weights.gate, schedule)
        summaries.append(time_grouped_gemm(layer, label=f"{name}/layer", flops=3 * gemm_flops))
        summaries.append(time_grouped_gemm(gemm, label=f"{name}/gemm", flops=gemm_flops))
    return summaries


def describe_run_environment() -> dict[str, Any]:
    """Hardware, library versions, and the kernels' fixed tile sizes -- the
    config a benchmark number means nothing without."""
    import triton  # noqa: PLC0415 -- Linux-only; this CLI only runs on a GPU host

    from dispatch.kernels import grouped_gemm  # noqa: PLC0415

    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "block_n": grouped_gemm.DEFAULT_BLOCK_N,
        "block_k": grouped_gemm.DEFAULT_BLOCK_K,
        "group_size_m": grouped_gemm.DEFAULT_GROUP_SIZE_M,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Time dispatch's grouped-GEMM backends")
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 16, 128, 512, 2048])
    parser.add_argument("--distribution", choices=ROUTING_DISTRIBUTIONS, default="zipf")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-1-kernel-bench"))
    args = parser.parse_args(argv)

    results = [
        {
            "num_tokens": num_tokens,
            "summaries": [
                asdict(summary)
                for summary in run_kernel_bench(
                    num_tokens=num_tokens,
                    distribution=args.distribution,
                    dtype=getattr(torch, args.dtype),
                    block_m=args.block_m,
                    seed=args.seed,
                )
            ],
        }
        for num_tokens in args.num_tokens
    ]
    record = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "distribution": args.distribution,
            "dtype": args.dtype,
            "block_m": args.block_m,
            "seed": args.seed,
            "do_bench_warmup_ms": DO_BENCH_WARMUP_MS,
            "do_bench_rep_ms": DO_BENCH_REP_MS,
            **describe_run_environment(),
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.run_label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


def _expert_weights(n: int, k: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    weights = torch.randn(N_ROUTED_EXPERTS, n, k, device="cuda", dtype=dtype, generator=generator)
    return weights / math.sqrt(k)


def _require_agreement(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    try:
        assert_matches_reference(actual, expected)
    except AssertionError as exc:
        raise RuntimeError(
            f"backend {name!r} disagrees with the eager torch backend -- refusing to time it"
        ) from exc


if __name__ == "__main__":
    main()
