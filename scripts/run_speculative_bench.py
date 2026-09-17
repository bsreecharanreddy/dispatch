"""CLI: speculative decoding on top of dispatch's own int8-quantized
target kernel (Phase 5a) -- either a real draft model
(deepseek-llm-7b-base) or model-free prompt-lookup decoding. Writes
latency/throughput and acceptance rate, and with
--compare-generated-tokens, checks the run's exact generated token ids
against a same-session baseline and exits non-zero on any divergence.
docs/design/2026-09-17-phase-5b-speculative-decoding.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dispatch.benchmark.harness import load_model
from dispatch.benchmark.metrics import TokenTimings, summarize
from dispatch.kernels.backends import resolve_quantized_backend
from dispatch.kernels.integration import patch_moe_infer_quantized
from dispatch.speculative.decode import speculative_generate
from dispatch.speculative.drafters import Drafter, DraftModelDrafter, PromptLookupDrafter
from dispatch.speculative.reference import (
    compare_generated_tokens,
    load_generated_tokens,
    save_generated_tokens,
)
from scripts.run_baseline import DEFAULT_PROMPTS

REPETITION_PROMPT = (
    "Please restate the following sentence exactly twice, back to back: "
    "'The system will now process each incoming request in the order it was received.'"
)
SPECULATIVE_PROMPTS = [*DEFAULT_PROMPTS, REPETITION_PROMPT]
DRAFTERS = ("none", "draft-model", "prompt-lookup")


def build_drafter(
    drafter_name: str,
    *,
    draft_model_name: str,
    device: str,
    dtype: torch.dtype,
    prompt_lookup_ngram_size: int,
) -> Drafter | None:
    if drafter_name == "none":
        return None
    if drafter_name == "prompt-lookup":
        return PromptLookupDrafter(ngram_size=prompt_lookup_ngram_size)
    if drafter_name == "draft-model":
        draft_model, _ = load_model(draft_model_name, device=device, dtype=dtype)
        return DraftModelDrafter(draft_model)
    raise ValueError(f"unknown drafter {drafter_name!r}; expected one of {DRAFTERS}")


def run_speculative_bench(  # noqa: PLR0913 -- each of these is an independent, user-facing knob
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    prompts: list[str],
    repetitions: int,
    max_new_tokens: int,
    drafter_name: str,
    draft_model_name: str,
    num_speculative_tokens: int,
    prompt_lookup_ngram_size: int,
) -> tuple[list[tuple[TokenTimings, tuple[int, ...]]], dict[str, tuple[int, ...]], int]:
    """Returns the timed runs paired with each run's per-round accepted
    lengths, the first repetition's generated tokens per prompt (greedy
    decoding is deterministic, so later repetitions would be identical),
    and how many MoE layers were patched."""
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )
    moe_layers_patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
    if moe_layers_patched == 0:
        raise RuntimeError(
            f"quantized target patched no MoE layers: {model_name} has no moe_infer to "
            "replace, so this run would time the stock model under the quantized kernel's name"
        )

    drafter = build_drafter(
        drafter_name,
        draft_model_name=draft_model_name,
        device=device,
        dtype=dtype,
        prompt_lookup_ngram_size=prompt_lookup_ngram_size,
    )

    runs: list[tuple[TokenTimings, tuple[int, ...]]] = []
    generated_tokens: dict[str, tuple[int, ...]] = {}
    for prompt_index, prompt in enumerate(prompts):
        for repetition in range(repetitions):
            timing, accepted_lengths, tokens = speculative_generate(
                model,
                tokenizer,
                drafter,
                prompt,
                num_speculative_tokens=num_speculative_tokens,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            runs.append((timing, accepted_lengths))
            if repetition == 0:
                generated_tokens[f"prompt_{prompt_index:03d}_tokens"] = tokens

    return runs, generated_tokens, moe_layers_patched


def _acceptance_rate(
    runs: list[tuple[TokenTimings, tuple[int, ...]]], num_speculative_tokens: int
) -> float | None:
    lengths = [length for _, accepted_lengths in runs for length in accepted_lengths]
    if not lengths or num_speculative_tokens <= 0:
        return None
    return statistics.mean(lengths) / num_speculative_tokens


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run dispatch's speculative-decoding benchmark (Phase 5b)"
    )
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--draft-model-name", default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-speculative"))
    parser.add_argument("--drafter", default="none", choices=DRAFTERS)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--prompt-lookup-ngram-size", type=int, default=3)
    parser.add_argument(
        "--compare-generated-tokens",
        type=Path,
        default=None,
        help="generated-tokens JSON from a same-session baseline run of the same model and prompts",
    )
    args = parser.parse_args(argv)

    dtype: torch.dtype = getattr(torch, args.dtype)
    runs, generated_tokens, moe_layers_patched = run_speculative_bench(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=SPECULATIVE_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        drafter_name=args.drafter,
        draft_model_name=args.draft_model_name,
        num_speculative_tokens=args.num_speculative_tokens,
        prompt_lookup_ngram_size=args.prompt_lookup_ngram_size,
    )
    timings = [timing for timing, _ in runs]
    summary = summarize(timings)
    acceptance_rate = _acceptance_rate(runs, args.num_speculative_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output_dir / f"{args.run_label}-generated-tokens.json"
    save_generated_tokens(generated_tokens, tokens_path)
    print(f"wrote {tokens_path}")

    comparison = (
        compare_generated_tokens(
            generated_tokens, load_generated_tokens(args.compare_generated_tokens)
        )
        if args.compare_generated_tokens is not None
        else {}
    )

    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "draft_model": args.draft_model_name if args.drafter == "draft-model" else None,
                "device": args.device,
                "dtype": args.dtype,
                "drafter": args.drafter,
                "num_speculative_tokens": args.num_speculative_tokens,
                "prompt_lookup_ngram_size": args.prompt_lookup_ngram_size,
                "moe_layers_patched": moe_layers_patched,
                "acceptance_rate": acceptance_rate,
                **asdict(summary),
                "token_match": comparison,
            },
            indent=2,
        )
    )
    print(f"wrote {results_path}")

    if comparison and not all(comparison.values()):
        raise SystemExit(
            f"{args.drafter} generated tokens disagree with "
            f"{args.compare_generated_tokens} -- see {results_path}"
        )


if __name__ == "__main__":
    main()
