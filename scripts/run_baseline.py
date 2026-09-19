"""CLI: run DeepSeekMoE-16B on one rented GPU -- the stock HF forward pass by
default, or with a dispatch grouped-GEMM backend patched into every MoE
layer (--moe-kernel). Writes latency/throughput and the run's logits; with
--compare-reference, also checks those logits against a stock run's and
exits non-zero if they disagree.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dispatch.benchmark.agreement import (
    MIN_GATE_POSITIONS,
    aggregate_gap_split,
    compare_gap_split,
)
from dispatch.benchmark.gate_prompts import GATE_PROMPTS
from dispatch.benchmark.harness import generate_with_timings, load_model
from dispatch.benchmark.metrics import TokenTimings, summarize
from dispatch.benchmark.reference import (
    capture_reference_logits,
    compare_top_k_agreement,
    load_reference,
    save_reference,
)
from dispatch.kernels.backends import (
    BACKENDS,
    QUANTIZED_BACKEND,
    resolve_backend,
    resolve_quantized_backend,
)
from dispatch.kernels.integration import patch_moe_infer, patch_moe_infer_quantized

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]


def run_baseline(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    prompts: list[str],
    repetitions: int,
    max_new_tokens: int,
    moe_kernel: str = "none",
) -> tuple[list[TokenTimings], dict[str, torch.Tensor], int]:
    """Returns the timed runs, the logits, and how many MoE layers were patched."""
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )
    moe_layers_patched = 0
    if moe_kernel == QUANTIZED_BACKEND:
        moe_layers_patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
    elif moe_kernel != "none":
        moe_layers_patched = patch_moe_infer(model, resolve_backend(moe_kernel))
    if moe_kernel != "none" and moe_layers_patched == 0:
        raise RuntimeError(
            f"--moe-kernel {moe_kernel} patched no MoE layers: {model_name} has no "
            "moe_infer to replace, so this run would time the stock model under a kernel's name"
        )

    runs = [
        generate_with_timings(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        for prompt in prompts
        for _ in range(repetitions)
    ]
    logits = capture_reference_logits(model, tokenizer, prompts, device=device)
    return runs, logits, moe_layers_patched


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dispatch's latency/throughput benchmark")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings/phase-0"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-baseline"))
    parser.add_argument(
        "--moe-kernel", default="none", choices=["none", *BACKENDS, QUANTIZED_BACKEND]
    )
    parser.add_argument(
        "--prompt-set",
        choices=["default", "gate"],
        default="default",
        help="'gate' runs Phase 6's larger fixed prompt set and enforces its gap-split rule",
    )
    parser.add_argument(
        "--compare-reference",
        type=Path,
        default=None,
        help="logits file from a stock run of the same model and prompts",
    )
    args = parser.parse_args(argv)

    dtype: torch.dtype = getattr(torch, args.dtype)
    runs, logits, moe_layers_patched = run_baseline(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=list(GATE_PROMPTS) if args.prompt_set == "gate" else DEFAULT_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        moe_kernel=args.moe_kernel,
    )
    summary = summarize(runs)

    # The logits are written before comparing against --compare-reference: a
    # mistyped path or a shape/key mismatch raises out of load_reference or
    # compare_top_k_agreement, and this run's own (expensive to reproduce)
    # evidence must survive that rather than being lost with it.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / f"{args.run_label}-reference.safetensors"
    save_reference(logits, reference_path)
    print(f"wrote {reference_path}")

    reference = load_reference(args.compare_reference) if args.compare_reference else None
    comparison = compare_top_k_agreement(logits, reference) if reference is not None else {}
    gap_split = (
        aggregate_gap_split(compare_gap_split(logits, reference).values())
        if reference is not None
        else None
    )

    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "device": args.device,
                "dtype": args.dtype,
                "moe_kernel": args.moe_kernel,
                "moe_layers_patched": moe_layers_patched,
                "prompt_set": args.prompt_set,
                "gap_split": None if gap_split is None else gap_split.to_dict(),
                **asdict(summary),
                "reference_comparison": {key: asdict(value) for key, value in comparison.items()},
            },
            indent=2,
        )
    )
    print(f"wrote {results_path}")

    if not all(value.mutual_top_k for value in comparison.values()):
        raise SystemExit(
            f"{args.moe_kernel} logits disagree with {args.compare_reference} -- see {results_path}"
        )
    if args.prompt_set == "gate" and gap_split is not None:
        if gap_split.positions < MIN_GATE_POSITIONS:
            raise SystemExit(
                f"gate compared only {gap_split.positions} positions, fewer than the "
                f"{MIN_GATE_POSITIONS} an at-scale claim needs -- see {results_path}"
            )
        if gap_split.large_gap_disagreements > 0:
            raise SystemExit(
                f"{args.moe_kernel} flips {gap_split.large_gap_disagreements} position(s) the "
                f"reference was confident about (widest gap {gap_split.max_disagreement_gap:.3f}) "
                f"-- a bug, not a near-tie; see {results_path}"
            )


if __name__ == "__main__":
    main()
