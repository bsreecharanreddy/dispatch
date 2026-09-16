"""Phase 3 correctness gate: does DeepEP-backed 2-GPU expert-parallel
inference on the real DeepSeekMoE-16B model agree with a single-GPU
reference, at the same top-5/top-1 bar Phase 1 used
(docs/findings/2026-09-15-phase-1-grouped-gemm-run.md)? Must pass before
Task 4's benchmark sweep runs -- an EP run that disagrees with the
single-GPU reference is not benchmarked, per this project's standing
"correctness before speed" rule.

Single-GPU reference (run first -- also warms the shared HF cache so the
EP run below never races a concurrent first-time download):
    uv run python scripts/gpu/run_phase3_correctness_gate.py --mode single

2-GPU EP (run second, same prompts):
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=2 \
        scripts/gpu/run_phase3_correctness_gate.py --mode ep
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from deep_ep import Buffer

from dispatch.benchmark.harness import load_model
from dispatch.benchmark.reference import (
    capture_reference_logits,
    compare_top_k_agreement,
    load_reference,
    save_reference,
)
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.expert_parallel import patch_moe_infer_ep
from dispatch.kernels.integration import patch_moe_infer

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]
MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
OUTPUT_DIR = Path("docs/findings")
SINGLE_GPU_REFERENCE_PATH = OUTPUT_DIR / "2026-09-16-phase-3-single-gpu-reference.safetensors"


def run_single_gpu_reference() -> None:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    patched = patch_moe_infer(model, resolve_backend("naive"))
    if patched == 0:
        raise RuntimeError(f"patched no MoE layers in {MODEL_NAME} -- naive kernel never ran")
    print(f"single-GPU reference: patched {patched} MoE layers with the naive kernel")

    logits = capture_reference_logits(model, tokenizer, DEFAULT_PROMPTS, device="cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_reference(logits, SINGLE_GPU_REFERENCE_PATH)
    print(f"wrote {SINGLE_GPU_REFERENCE_PATH}")


def run_ep() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    dist.barrier()

    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )

    hidden_bytes = model.config.hidden_size * 2  # bf16 element size
    Buffer.set_num_sms(24)
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
    buffer = Buffer(group, num_nvl_bytes, 0)  # num_rdma_bytes=0: pure intranode

    patched = patch_moe_infer_ep(model, resolve_backend("naive"), buffer, rank, world_size)
    if patched == 0:
        raise RuntimeError(f"patched no MoE layers in {MODEL_NAME} -- EP path never ran")
    print(f"[rank {rank}] EP: patched {patched} MoE layers")

    logits = capture_reference_logits(model, tokenizer, DEFAULT_PROMPTS, device="cuda")

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ep_path = OUTPUT_DIR / "2026-09-16-phase-3-ep-logits.safetensors"
        save_reference(logits, ep_path)
        print(f"[rank 0] wrote {ep_path}")

        reference = load_reference(SINGLE_GPU_REFERENCE_PATH)
        comparison = compare_top_k_agreement(logits, reference)
        results_path = OUTPUT_DIR / "2026-09-16-phase-3-correctness-gate-results.json"
        results_path.write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "prompts": DEFAULT_PROMPTS,
                    "moe_layers_patched": patched,
                    "world_size": world_size,
                    "comparison": {key: asdict(value) for key, value in comparison.items()},
                },
                indent=2,
            )
        )
        print(f"[rank 0] wrote {results_path}")

        for key, agreement in comparison.items():
            print(
                f"[rank 0] {key}: top1_agreement={agreement.top1_agreement:.4f} "
                f"mutual_top_k={agreement.mutual_top_k} "
                f"max_abs_diff={agreement.max_abs_diff:.4f}"
            )

        if not all(value.mutual_top_k for value in comparison.values()):
            dist.destroy_process_group()
            raise SystemExit(
                f"EP logits disagree with the single-GPU reference -- see {results_path}"
            )

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 EP correctness gate")
    parser.add_argument("--mode", required=True, choices=["single", "ep"])
    args = parser.parse_args()

    if args.mode == "single":
        run_single_gpu_reference()
    else:
        run_ep()


if __name__ == "__main__":
    main()
