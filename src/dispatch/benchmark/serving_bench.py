"""Pure helpers for Phase 6's engine reference: build the server and
`vllm bench serve` commands, summarize a bench result, and refuse one that
cannot be trusted. No subprocess, network or GPU here -- the orchestration
that uses these lives in scripts/gpu/phase6_engine_reference.py.

`vllm bench serve` flags checked live 2026-09-19 against vllm-project/vllm
docs/benchmarking/cli.md (custom dataset) and vllm/benchmarks/serve.py. The
same client drives both engines, so client-side behavior is never a
difference between them.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatch.benchmark.metrics import percentile

ENGINES = ("vllm", "sglang")
CONCURRENCIES = (1, 4, 16, 64)
OUTPUT_LEN = 64  # matches every earlier phase's --max-new-tokens
NUM_WARMUPS = 8  # discarded by the client and recorded in the output config
MIN_PROMPTS = 32
PROMPTS_PER_CONCURRENCY = 8
TRACE_LINES = 512  # >= num_prompts_for(max(CONCURRENCIES))
MS = 1000.0


def num_prompts_for(concurrency: int) -> int:
    return max(MIN_PROMPTS, PROMPTS_PER_CONCURRENCY * concurrency)


def write_trace(path: Path, prompts: Sequence[str], count: int = TRACE_LINES) -> None:
    """One {"prompt": ...} JSONL line per request, cycling `prompts`: the
    format `vllm bench serve --dataset-name custom` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"prompt": prompts[i % len(prompts)]}) for i in range(count))
    path.write_text("\n".join(lines) + "\n")


def build_serve_command(
    engine: str, model: str, port: int, *, max_model_len: int = 2048
) -> list[str]:
    if engine == "vllm":
        return [
            "vllm", "serve", model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(max_model_len),
            "--trust-remote-code",
        ]  # fmt: skip
    if engine == "sglang":
        return [
            "python", "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--context-length", str(max_model_len),
            "--trust-remote-code",
        ]  # fmt: skip
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


def build_bench_command(  # noqa: PLR0913 -- each is an independent, user-facing knob
    model: str,
    port: int,
    *,
    concurrency: int,
    trace_path: Path,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    return [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model,
        "--endpoint", "/v1/completions",
        "--dataset-name", "custom",
        "--dataset-path", str(trace_path),
        "--skip-chat-template",
        "--custom-output-len", str(OUTPUT_LEN),
        "--ignore-eos",
        "--num-prompts", str(num_prompts_for(concurrency)),
        "--max-concurrency", str(concurrency),
        "--num-warmups", str(NUM_WARMUPS),
        "--seed", "0",
        "--save-result", "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]  # fmt: skip


@dataclass(frozen=True)
class ServingSummary:
    concurrency: int
    completed: int
    duration_s: float
    output_tokens_per_s: float
    mean_request_tokens_per_s: float  # per request: output tokens / end-to-end latency
    mean_ttft_ms: float
    p50_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    p99_itl_ms: float
    p50_e2e_ms: float
    p99_e2e_ms: float
    cost_per_million_output_tokens_usd: float


def summarize_bench_result(
    raw: dict[str, Any],
    *,
    concurrency: int,
    expected_output_len: int,
    gpu_cost_per_hour: float,
) -> ServingSummary:
    """Refuses -- raises ValueError -- rather than report a number from a run
    that failed requests, returned no completions, did not generate exactly
    `expected_output_len` tokens per request (an engine that ignored
    --ignore-eos would report a throughput no other engine's is comparable
    to), or whose per-request arrays don't actually match its own reported
    `completed` count (a malformed or schema-drifted result JSON)."""
    completed = int(raw["completed"])
    if completed == 0:
        raise ValueError("bench completed zero requests -- refusing to report numbers")
    for key in ("errors", "output_lens", "ttfts", "itls"):
        values = raw.get(key)
        if values is None or len(values) != completed:
            got = "missing" if values is None else f"{len(values)} entries"
            raise ValueError(
                f"bench reported completed={completed} but {key!r} is {got} -- "
                "refusing to report numbers"
            )
    failed = int(raw.get("failed", 0))
    errors = [error for error in raw["errors"] if error]
    if failed or errors:
        raise ValueError(f"{failed} failed request(s), first error: {errors[:1]} -- refusing")
    wrong = [n for n in raw["output_lens"] if n != expected_output_len]
    if wrong:
        raise ValueError(
            f"{len(wrong)} request(s) generated {sorted(set(wrong))} tokens, not "
            f"{expected_output_len} -- ignore-eos was not honored; refusing"
        )
    ttfts_ms = sorted(t * MS for t in raw["ttfts"])
    itls_ms = sorted(gap * MS for request in raw["itls"] for gap in request)
    # vLLM's own --save-detailed output carries no per-request end-to-end
    # "latencies" key (confirmed live on 0.29.0) -- only per-request ttft and
    # the inter-token gaps that follow it, so end-to-end is their sum.
    latencies_s = [ttft + sum(gaps) for ttft, gaps in zip(raw["ttfts"], raw["itls"], strict=True)]
    e2e_ms = sorted(latency * MS for latency in latencies_s)
    tokens_per_s = [n / latency for n, latency in zip(raw["output_lens"], latencies_s, strict=True)]
    output_tokens_per_s = float(raw["output_throughput"])
    if output_tokens_per_s <= 0:
        raise ValueError(
            f"bench reported output_throughput={output_tokens_per_s} -- refusing to report numbers"
        )
    return ServingSummary(
        concurrency=concurrency,
        completed=completed,
        duration_s=float(raw["duration"]),
        output_tokens_per_s=output_tokens_per_s,
        mean_request_tokens_per_s=statistics.mean(tokens_per_s),
        mean_ttft_ms=statistics.mean(ttfts_ms),
        p50_ttft_ms=percentile(ttfts_ms, 0.50),
        p99_ttft_ms=percentile(ttfts_ms, 0.99),
        mean_itl_ms=statistics.mean(itls_ms) if itls_ms else 0.0,
        p99_itl_ms=percentile(itls_ms, 0.99) if itls_ms else 0.0,
        p50_e2e_ms=percentile(e2e_ms, 0.50),
        p99_e2e_ms=percentile(e2e_ms, 0.99),
        cost_per_million_output_tokens_usd=(
            gpu_cost_per_hour / (output_tokens_per_s * 3600) * 1_000_000
        ),
    )


def wait_until_healthy(  # noqa: PLR0913 -- injectable clock/sleep/get is what makes this testable
    health_url: str,
    *,
    get_fn: Callable[[str], int],
    is_alive: Callable[[], bool],
    sleep_fn: Callable[[float], None],
    clock_fn: Callable[[], float],
    timeout_s: float,
    interval_s: float = 5.0,
) -> None:
    """`get_fn` returns the HTTP status (raising on a refused connection).
    A server that dies while loading the model fails fast instead of waiting
    out the timeout on a metered pod."""
    deadline = clock_fn() + timeout_s
    while clock_fn() < deadline:
        if not is_alive():
            raise RuntimeError(f"server exited before {health_url} became healthy")
        try:
            if get_fn(health_url) == 200:  # noqa: PLR2004 -- HTTP OK
                return
        except OSError:
            pass
        sleep_fn(interval_s)
    raise TimeoutError(f"{health_url} not healthy after {timeout_s:.0f}s")
