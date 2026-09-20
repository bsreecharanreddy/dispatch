"""Pod-side driver for Phase 6's engine reference: launch one engine's
server, run `vllm bench serve` against it at each concurrency, summarize with
serving_bench's refuse rules, and always tear the server down. Run once per
engine, in the shared engines venv:

  python -m scripts.gpu.phase6_engine_reference --engine vllm --gpu-cost-per-hour 0.69

The evidence JSON is written even when a later concurrency fails, or the
server itself never launches, so a partial run survives; the failure is
then re-raised (non-zero exit).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from dispatch.benchmark.engines.base import installed_version
from dispatch.benchmark.gate_prompts import GATE_PROMPTS
from dispatch.benchmark.serving_bench import (
    CONCURRENCIES,
    ENGINES,
    NUM_WARMUPS,
    OUTPUT_LEN,
    TRACE_LINES,
    ServingSummary,
    build_bench_command,
    build_serve_command,
    num_prompts_for,
    summarize_bench_result,
    wait_until_healthy,
    write_trace,
)

DEFAULT_MODEL = "deepseek-ai/deepseek-moe-16b-base"
TERMINATE_GRACE_S = 60.0


def run_engine_reference(  # noqa: PLR0913 -- injectable process/network hooks make this testable
    engine: str,
    *,
    model: str,
    port: int,
    concurrencies: Sequence[int],
    output_dir: Path,
    run_label: str,
    gpu_cost_per_hour: float,
    health_timeout_s: float = 1800.0,
    popen_fn: Callable[..., Any] = subprocess.Popen,
    run_fn: Callable[..., Any] = subprocess.run,
    get_fn: Callable[[str], int] = lambda url: requests.get(url, timeout=5).status_code,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{run_label}-trace.jsonl"
    write_trace(trace_path, GATE_PROMPTS, TRACE_LINES)
    summaries: list[ServingSummary] = []
    error: str | None = None
    server: Any = None
    server_log: Any = None

    try:
        server_log = (output_dir / f"{run_label}-server.log").open("w")
        server = popen_fn(
            build_serve_command(engine, model, port), stdout=server_log, stderr=server_log
        )
        wait_until_healthy(
            f"http://127.0.0.1:{port}/health",
            get_fn=get_fn,
            is_alive=lambda: server.poll() is None,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            timeout_s=health_timeout_s,
        )
        for concurrency in concurrencies:
            filename = f"{run_label}-c{concurrency}-raw.json"
            run_fn(
                build_bench_command(
                    model,
                    port,
                    concurrency=concurrency,
                    trace_path=trace_path,
                    result_dir=output_dir,
                    result_filename=filename,
                ),
                check=True,
            )
            raw = json.loads((output_dir / filename).read_text())
            summaries.append(
                summarize_bench_result(
                    raw,
                    concurrency=concurrency,
                    expected_output_len=OUTPUT_LEN,
                    gpu_cost_per_hour=gpu_cost_per_hour,
                )
            )
    except Exception as exc:  # recorded in the evidence JSON, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=TERMINATE_GRACE_S)
        if server_log is not None:
            server_log.close()
        (output_dir / f"{run_label}.json").write_text(
            json.dumps(
                {
                    "config": {
                        "engine": engine,
                        "engine_version": installed_version(engine),
                        "model": model,
                        "dtype": "bfloat16",
                        "output_len": OUTPUT_LEN,
                        "num_warmups": NUM_WARMUPS,
                        "concurrencies": list(concurrencies),
                        "num_prompts": {c: num_prompts_for(c) for c in concurrencies},
                        "gpu_cost_per_hour": gpu_cost_per_hour,
                        "trace": trace_path.name,
                    },
                    "results": [asdict(summary) for summary in summaries],
                    "error": error,
                },
                indent=2,
            )
        )
    return output_dir / f"{run_label}.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6 engine reference (vLLM or SGLang)")
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES))
    parser.add_argument("--gpu-cost-per-hour", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args(argv)
    label = args.run_label or time.strftime(f"%Y-%m-%d-phase-6-reference-{args.engine}")
    path = run_engine_reference(
        args.engine,
        model=args.model,
        port=args.port,
        concurrencies=args.concurrencies,
        output_dir=args.output_dir,
        run_label=label,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
