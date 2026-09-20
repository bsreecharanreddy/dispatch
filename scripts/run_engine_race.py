"""CLI for Phase 6's kernel race: dispatch's grouped-GEMM vs. vLLM's and
SGLang's fused-MoE on identical seeded inputs.

  prepare  generate the seeded weights, cases and fp32 references once, to disk
  run      time one engine (dispatch's tile-size sweep, vllm, or sglang) on them
  merge    combine the per-engine JSONs into one table

Inputs and references are written once and *loaded* by every `run`, so all
contestants see byte-identical tensors and are checked against the same
reference. An engine that disagrees with the reference is refused, not timed:
its result is recorded as refused, and `run` exits non-zero after writing the
JSON, so the evidence survives.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from dispatch.benchmark.engines import registry
from dispatch.benchmark.engines.base import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    N_ROUTED_EXPERTS,
    NUM_EXPERTS_PER_TOK,
    TOKEN_COUNTS,
    BoundLayer,
    MoEEngine,
    RaceCase,
    dequantized_weights,
    make_case,
    make_weights,
    reference_output,
)
from dispatch.benchmark.race_summary import render_markdown, summarize_race
from dispatch.kernels.bench import (
    DO_BENCH_REP_MS,
    DO_BENCH_WARMUP_MS,
    ROUTING_DISTRIBUTIONS,
    KernelBenchmarkSummary,
    grouped_gemm_flops,
    time_grouped_gemm,
)
from dispatch.kernels.moe_forward import StackedExpertWeights, assert_matches_reference
from dispatch.kernels.quantization import (
    QuantizedStackedExpertWeights,
    QuantizedTensor,
    quantize_stacked_weights,
)

TimeFn = Callable[[BoundLayer, str, float], KernelBenchmarkSummary]

MANIFEST = "manifest.json"
WEIGHTS = "weights.safetensors"


def prepare_inputs(  # noqa: PLR0913 -- each is an independent, user-facing knob
    out_dir: Path,
    *,
    num_tokens: Sequence[int],
    distributions: Sequence[str],
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = make_weights(dtype, seed=seed, device=device)
    qweights = quantize_stacked_weights(weights)
    save_file(
        _to_cpu(
            {
                "gate": weights.gate,
                "up": weights.up,
                "down": weights.down,
                "q_gate_data": qweights.gate.data,
                "q_gate_scale": qweights.gate.scale,
                "q_up_data": qweights.up.data,
                "q_up_scale": qweights.up.scale,
                "q_down_data": qweights.down.data,
                "q_down_scale": qweights.down.scale,
            }
        ),
        str(out_dir / WEIGHTS),
    )
    int8_oracle = dequantized_weights(qweights)
    for tokens in num_tokens:
        for distribution in distributions:
            case = make_case(tokens, distribution, dtype=dtype, seed=seed, device=device)
            save_file(
                _to_cpu(
                    {
                        "x": case.x,
                        "topk_idx": case.topk_idx,
                        "topk_weight": case.topk_weight,
                        "ref_bf16": reference_output(case, weights),
                        "ref_int8": reference_output(case, int8_oracle),
                    }
                ),
                str(out_dir / _case_file(tokens, distribution)),
            )
    manifest = {
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "n_routed_experts": N_ROUTED_EXPERTS,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "num_tokens": list(num_tokens),
        "distributions": list(distributions),
        "prepared_on": describe_environment(device),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))


def run_engines(
    inputs_dir: Path,
    engines: Sequence[MoEEngine],
    *,
    precision: str,
    device: str,
    time_fn: TimeFn | None = None,
) -> list[dict[str, Any]]:
    """One result per (engine variant, case). A variant whose output disagrees
    with the fp32 reference is recorded as refused and never timed; so is a
    variant that cannot even be prepared for this precision (e.g. dispatch's
    persistent kernel has no int8 path) -- either way the evidence for every
    other variant already run survives."""
    manifest = json.loads((inputs_dir / MANIFEST).read_text())
    weights, qweights = _load_weights(inputs_dir, device)
    timer = time_fn or _do_bench_timer
    results: list[dict[str, Any]] = []
    for engine in engines:
        try:
            factory = (
                engine.prepare_bf16(weights)
                if precision == "bf16"
                else engine.prepare_int8(qweights)
            )
        except Exception as exc:  # any prep failure is a refusal, not a crash
            results.append(
                {
                    "num_tokens": None,
                    "distribution": None,
                    "variant": engine.name,
                    "status": "refused",
                    "reason": _first_line(exc),
                }
            )
            continue
        for tokens in manifest["num_tokens"]:
            for distribution in manifest["distributions"]:
                case, reference = _load_case(inputs_dir, tokens, distribution, precision, device)
                bound = factory(case)
                entry: dict[str, Any] = {
                    "num_tokens": tokens,
                    "distribution": distribution,
                    "variant": engine.name,
                }
                try:
                    assert_matches_reference(bound(), reference)
                except AssertionError as exc:
                    entry.update(status="refused", reason=_first_line(exc))
                    results.append(entry)
                    continue
                flops = 3 * grouped_gemm_flops(
                    tokens * manifest["num_experts_per_tok"],
                    manifest["moe_intermediate_size"],
                    manifest["hidden_size"],
                )
                summary = timer(bound, f"{engine.name}/{tokens}/{distribution}", flops)
                entry.update(
                    status="ok",
                    mean_ms=summary.mean_latency_ms,
                    p50_ms=summary.p50_latency_ms,
                    p99_ms=summary.p99_latency_ms,
                    tflops=summary.tflops,
                )
                results.append(entry)
    return results


def describe_environment(device: str) -> dict[str, Any]:
    """Hardware, library versions, and tuned-config provenance: the config a
    benchmark number means nothing without."""
    environment: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
    }
    for package in ("triton", "vllm", "sglang"):
        environment[package] = _installed_version(package)
    for variable in ("VLLM_TUNED_CONFIG_FOLDER", "SGLANG_MOE_CONFIG_DIR"):
        folder = os.environ.get(variable)
        environment[variable] = folder
        environment[f"{variable}_files"] = (
            sorted(str(path.relative_to(folder)) for path in Path(folder).rglob("*.json"))
            if folder and Path(folder).is_dir()
            else []
        )
    return environment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6 kernel race")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--num-tokens", type=int, nargs="+", default=list(TOKEN_COUNTS))
    prepare.add_argument(
        "--distributions", nargs="+", choices=ROUTING_DISTRIBUTIONS, default=ROUTING_DISTRIBUTIONS
    )
    prepare.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--device", default="cuda")

    run = sub.add_parser("run")
    run.add_argument("--inputs-dir", type=Path, required=True)
    run.add_argument("--engine", choices=registry.ENGINE_NAMES, required=True)
    run.add_argument("--block-ms", type=int, nargs="+", default=list(registry.DEFAULT_BLOCK_MS))
    run.add_argument("--precision", choices=["bf16", "int8"], required=True)
    run.add_argument(
        "--tuning-label",
        choices=["sweep", "default", "tuned"],
        required=True,
        help="'sweep' for dispatch's tile-size sweep; 'default'/'tuned' for vllm and sglang",
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    run.add_argument("--run-label", default=None)

    merge = sub.add_parser("merge")
    merge.add_argument("--results-dir", type=Path, required=True)
    merge.add_argument("--glob", default="*-race-*.json")
    merge.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-6"))
    # Deliberately does not contain "-race-": the default --glob ("*-race-*.json")
    # would otherwise match this command's own output on a second run in the
    # same directory, feeding a prior summary back in as if it were a fresh
    # per-engine record.
    merge.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-6-summary"))

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_inputs(
            args.out_dir,
            num_tokens=args.num_tokens,
            distributions=args.distributions,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
            device=args.device,
        )
        print(f"wrote inputs to {args.out_dir}")
    elif args.command == "run":
        _command_run(args)
    else:
        _command_merge(args)


def _command_run(args: argparse.Namespace) -> None:
    if args.engine.startswith("dispatch-") and args.tuning_label != "sweep":
        # race_summary.summarize_race derives dispatch's "default"/"tuned"
        # rows entirely from the --block-ms sweep; it never reads this flag
        # for a dispatch-* engine, so any other value is silently a no-op.
        raise SystemExit(
            f"--tuning-label {args.tuning_label!r} has no effect for {args.engine!r} -- "
            "dispatch's default/tuned rows are derived from the --block-ms sweep, not "
            "chosen by this flag; pass --tuning-label sweep"
        )
    manifest = json.loads((args.inputs_dir / MANIFEST).read_text())
    if args.engine == "sglang":
        from dispatch.benchmark.engines.sglang_moe import init_distributed  # noqa: PLC0415

        init_distributed(dtype=manifest["dtype"])
    engines = registry.build_engines(args.engine, args.block_ms)
    results = run_engines(args.inputs_dir, engines, precision=args.precision, device=args.device)
    record = {
        "config": {
            "engine": args.engine,
            "precision": args.precision,
            "tuning_label": args.tuning_label,
            "block_ms": list(args.block_ms) if args.engine.startswith("dispatch-") else None,
            "inputs": manifest,
            "do_bench_warmup_ms": DO_BENCH_WARMUP_MS,
            "do_bench_rep_ms": DO_BENCH_REP_MS,
            "environment": describe_environment(args.device),
        },
        "results": results,
    }
    label = args.run_label or time.strftime(
        f"%Y-%m-%d-phase-6-race-{args.engine}-{args.precision}-{args.tuning_label}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")
    refused = [result for result in results if result["status"] == "refused"]
    if refused:
        raise SystemExit(
            f"{len(refused)} result(s) from {args.engine} disagree with the fp32 reference and "
            f"were refused, not timed -- see {path}"
        )


def _command_merge(args: argparse.Namespace) -> None:
    paths = sorted(args.results_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob} in {args.results_dir}")
    records = [_load_race_record(path) for path in paths]
    rows = summarize_race(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.run_label}.json").write_text(
        json.dumps([row.__dict__ for row in rows], indent=2)
    )
    markdown_path = args.output_dir / f"{args.run_label}.md"
    markdown_path.write_text(render_markdown(rows))
    print(f"wrote {markdown_path}")


def _do_bench_timer(bound: BoundLayer, label: str, flops: float) -> KernelBenchmarkSummary:
    return time_grouped_gemm(bound, label=label, flops=flops)


def _case_file(num_tokens: int, distribution: str) -> str:
    return f"case_{num_tokens}_{distribution}.safetensors"


def _to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in tensors.items()}


def _load_weights(
    inputs_dir: Path, device: str
) -> tuple[StackedExpertWeights, QuantizedStackedExpertWeights]:
    loaded = load_file(str(inputs_dir / WEIGHTS), device=device)
    weights = StackedExpertWeights(gate=loaded["gate"], up=loaded["up"], down=loaded["down"])
    qweights = QuantizedStackedExpertWeights(
        gate=QuantizedTensor(data=loaded["q_gate_data"], scale=loaded["q_gate_scale"]),
        up=QuantizedTensor(data=loaded["q_up_data"], scale=loaded["q_up_scale"]),
        down=QuantizedTensor(data=loaded["q_down_data"], scale=loaded["q_down_scale"]),
    )
    return weights, qweights


def _load_case(
    inputs_dir: Path, num_tokens: int, distribution: str, precision: str, device: str
) -> tuple[RaceCase, torch.Tensor]:
    loaded = load_file(str(inputs_dir / _case_file(num_tokens, distribution)), device=device)
    case = RaceCase(x=loaded["x"], topk_idx=loaded["topk_idx"], topk_weight=loaded["topk_weight"])
    return case, loaded["ref_bf16" if precision == "bf16" else "ref_int8"]


def _first_line(exc: BaseException) -> str:
    """A short, single-line reason for a results entry. `str(exc)` is empty
    for some exception types (bare `raise ValueError`), and `"".splitlines()`
    is `[]` -- indexing that unconditionally is its own crash."""
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


def _load_race_record(path: Path) -> dict[str, Any]:
    """A per-engine race JSON, not -- e.g. -- a previous merge's own summary
    output, which `--glob` can otherwise match right back in."""
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or "config" not in record or "results" not in record:
        raise SystemExit(
            f"{path} does not look like a per-engine race record (needs 'config' and "
            "'results' keys) -- does --glob also match a previous merge's own output?"
        )
    return record


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
