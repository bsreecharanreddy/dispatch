"""Pure-PyTorch reference for a DeepSeek-style MoE block -- the CPU-testable
ground truth Phase 1's grouped-GEMM path is checked against. Mirrors the
inference path of deepseek-ai/deepseek-moe-16b-base's DeepseekMoE: softmax
gate, top-k routing, per-expert SiLU-gated MLP, weighted combine, plus
shared experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- F is the universal PyTorch convention


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int


class ExpertMLP(torch.nn.Module):
    """down(silu(gate(x)) * up(x)) -- the per-expert MLP shape DeepseekMLP uses."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class ReferenceMoE(torch.nn.Module):
    """A masked loop over experts: slow, obviously correct, and the oracle."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = torch.nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            ExpertMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_experts = (
            ExpertMLP(config.hidden_size, config.moe_intermediate_size * config.n_shared_experts)
            if config.n_shared_experts > 0
            else None
        )

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k expert ids and their softmax weights, deliberately not
        renormalized: the real model's config.json sets norm_topk_prob=false."""
        scores = F.softmax(self.gate(hidden_states), dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)
        return topk_idx, topk_weight.to(hidden_states.dtype)

    def routed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_weight = self.route(hidden_states)
        combined = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = topk_weight[token_idx, slot_idx].unsqueeze(-1)
            combined.index_add_(0, token_idx, expert(hidden_states[token_idx]) * weight)
        return combined

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.routed(hidden_states)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
