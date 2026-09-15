"""Captures reference logits for a fixed prompt set -- the numerical
ground truth Phase 1's kernel gets checked against within a stated
tolerance (CLAUDE.md's "correctness before speed").
"""

from __future__ import annotations

from dataclasses import dataclass
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
    _require_same_keys(actual, reference)
    return {
        key: bool(torch.allclose(actual[key], reference[key], rtol=rtol, atol=atol))
        for key in reference
    }


@dataclass(frozen=True)
class TopKAgreement:
    """Position-by-position agreement between two logit tensors, judged the
    way a kernel swap should be: a near-tie may flip, a real bug may not."""

    positions: int
    top1_agreement: float
    mutual_top_k: bool  # at every position, each side's argmax is in the other's top-k
    max_abs_diff: float


def compare_top_k_agreement(
    actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], *, k: int = 5
) -> dict[str, TopKAgreement]:
    _require_same_keys(actual, reference)
    return {key: _top_k_agreement(actual[key], reference[key], k) for key in reference}


def _require_same_keys(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> None:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, "
            f"reference has {sorted(reference.keys())}"
        )


def _top_k_agreement(actual: torch.Tensor, reference: torch.Tensor, k: int) -> TopKAgreement:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: actual {tuple(actual.shape)}, reference {tuple(reference.shape)}"
        )
    actual_f, reference_f = actual.float(), reference.float()
    actual_top1 = actual_f.argmax(dim=-1)
    reference_top1 = reference_f.argmax(dim=-1)
    actual_in_reference = (reference_f.topk(k, dim=-1).indices == actual_top1.unsqueeze(-1)).any(
        dim=-1
    )
    reference_in_actual = (actual_f.topk(k, dim=-1).indices == reference_top1.unsqueeze(-1)).any(
        dim=-1
    )
    return TopKAgreement(
        positions=int(actual_top1.numel()),
        top1_agreement=float((actual_top1 == reference_top1).float().mean()),
        mutual_top_k=bool((actual_in_reference & reference_in_actual).all()),
        max_abs_diff=float((actual_f - reference_f).abs().max()),
    )
