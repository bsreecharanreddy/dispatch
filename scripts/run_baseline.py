"""CLI: run dispatch's Phase 0 baseline -- a plain HF forward pass on one
rented GPU. Produces the honest "before" latency/throughput numbers and
the reference logits Phase 1's kernel gets checked against.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dispatch.benchmark.harness import generate_with_timings, load_model
from dispatch.benchmark.metrics import TokenTimings, summarize
from dispatch.benchmark.reference import capture_reference_logits, save_reference

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
) -> tuple[list[TokenTimings], dict[str, torch.Tensor]]:
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )

    runs = [
        generate_with_timings(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        for prompt in prompts
        for _ in range(repetitions)
    ]
    reference = capture_reference_logits(model, tokenizer, prompts, device=device)
    return runs, reference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dispatch's Phase 0 baseline benchmark")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-0-baseline"))
    args = parser.parse_args(argv)

    dtype: torch.dtype = getattr(torch, args.dtype)
    runs, reference = run_baseline(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=DEFAULT_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "device": args.device,
                "dtype": args.dtype,
                **asdict(summary),
            },
            indent=2,
        )
    )

    reference_path = args.output_dir / f"{args.run_label}-reference.safetensors"
    save_reference(reference, reference_path)

    print(f"wrote {results_path}")
    print(f"wrote {reference_path}")


if __name__ == "__main__":
    main()
