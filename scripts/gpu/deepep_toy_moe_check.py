"""Ad hoc validation (not a committed pytest test): proves make_ep_moe_infer
produces the same result as the CPU-proven simulate_ep_moe_routed reference
(Task 2), driven through DeepEP's real V1 Buffer across 2 real GPUs. Run
once before wiring the real 16B model, so a bug here doesn't get chased
through a much slower and more expensive real-model correctness gate.

Run: NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=2 scripts/gpu/deepep_toy_moe_check.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from deep_ep import Buffer

from dispatch.kernels.expert_parallel import (
    assign_experts_to_ranks,
    make_ep_moe_infer,
    simulate_ep_moe_routed,
)
from dispatch.kernels.moe_forward import (
    StackedExpertWeights,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    # hidden_size=16, not Task 2's CPU-only 8: DeepEP's real kernel asserts
    # hidden_int4 % 2 == 0 (hidden_size * bf16_bytes must be a multiple of
    # 32), which 8 fails and 16 satisfies -- confirmed live 2026-09-16.
    hidden_size=16,
    moe_intermediate_size=16,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=3,
)
NUM_TOKENS = 11


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD
    dist.barrier()

    # Same seed on every rank: every rank must compute the identical
    # reference and route the identical tokens, since this toy check
    # isn't sharding tokens across ranks, only experts.
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(NUM_TOKENS, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    rank_of_expert = assign_experts_to_ranks(TOY_CONFIG.n_routed_experts, world_size)
    expected = simulate_ep_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul, rank_of_expert
    )

    local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0]
    # ReferenceMoE's nn.Linear weights default to float32; the real
    # DeepSeekMoE-16B model loads in bf16 (Phase 0/1 precedent), so match
    # that here rather than the reference's own float32 default.
    local_weights = StackedExpertWeights(
        gate=weights.gate[local_expert_ids].to(torch.bfloat16).cuda(),
        up=weights.up[local_expert_ids].to(torch.bfloat16).cuda(),
        down=weights.down[local_expert_ids].to(torch.bfloat16).cuda(),
    )

    Buffer.set_num_sms(24)
    hidden_bytes = TOY_CONFIG.hidden_size * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
    buffer = Buffer(group, num_nvl_bytes, 0)

    moe_infer = make_ep_moe_infer(
        local_weights, torch_grouped_matmul, buffer, TOY_CONFIG.n_routed_experts
    )

    x = hidden_states.to(torch.bfloat16).cuda()
    flat_expert_indices = topk_idx.reshape(-1).cuda()
    flat_expert_weights = topk_weight.reshape(-1).to(torch.bfloat16).cuda()
    actual = moe_infer(x, flat_expert_indices, flat_expert_weights)

    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    print(
        f"[rank {rank}] toy EP MoE check passed: actual matches simulate_ep_moe_routed", flush=True
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
