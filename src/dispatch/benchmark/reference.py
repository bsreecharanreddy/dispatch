"""Captures reference logits for a fixed prompt set -- the numerical
ground truth Phase 1's kernel gets checked against within a stated
tolerance (CLAUDE.md's "correctness before speed").
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def capture_reference_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            logits = model(input_ids=input_ids).logits
            tensors[f"prompt_{i:03d}_logits"] = logits.squeeze(0).cpu().contiguous()
    return tensors


def save_reference(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def load_reference(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def compare_within_tolerance(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict[str, bool]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )
    return {
        key: bool(torch.allclose(actual[key], reference[key], rtol=rtol, atol=atol))
        for key in reference
    }
