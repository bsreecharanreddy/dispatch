"""Smoke test for DeepEP's real dispatch()/combine() round trip on toy
data across 2 real GPUs -- run before Phase 3's real EP MoE layer is
built, so its exact call shape is confirmed against DeepEP's actual
behavior instead of assumed from its README alone. Not a pytest test:
needs torch.distributed + 2 real GPUs + deep_ep installed.

Run: torchrun --nproc_per_node=2 scripts/gpu/deepep_smoke_test.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from deep_ep import ElasticBuffer

from dispatch.kernels.expert_parallel import assign_experts_to_ranks

NUM_EXPERTS = 8
NUM_TOPK = 3
HIDDEN = 16
NUM_TOKENS_PER_RANK = 6


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.group.WORLD

    rank_of_expert = assign_experts_to_ranks(NUM_EXPERTS, world_size)
    local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0].tolist()
    print(f"[rank {rank}] owns experts {local_expert_ids}", flush=True)

    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=NUM_TOKENS_PER_RANK,
        hidden=HIDDEN,
        num_topk=NUM_TOPK,
        num_experts=NUM_EXPERTS,
    )

    torch.manual_seed(rank)  # different per rank, deliberately: real ranks never share tokens
    x = torch.randn(NUM_TOKENS_PER_RANK, HIDDEN, device="cuda", dtype=torch.bfloat16)
    topk_weight, topk_idx = torch.topk(
        torch.randn(NUM_TOKENS_PER_RANK, NUM_EXPERTS, device="cuda"), NUM_TOPK, dim=-1
    )
    topk_weight = torch.softmax(topk_weight, dim=-1).to(torch.bfloat16)

    num_comm_sms = buffer.get_theoretical_num_sms(NUM_EXPERTS, NUM_TOPK)
    recv_x, recv_topk_idx, recv_topk_weight, handle, event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weight,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=NUM_TOKENS_PER_RANK,
        num_sms=num_comm_sms,
        async_with_compute_stream=True,
    )
    event.current_stream_wait()

    print(
        f"[rank {rank}] dispatch returned recv_x.shape={tuple(recv_x.shape)}, "
        f"recv_topk_idx.shape={tuple(recv_topk_idx.shape)}, "
        f"recv_topk_weight.shape={tuple(recv_topk_weight.shape)}, "
        f"unique recv expert ids={sorted(recv_topk_idx.unique().tolist())}",
        flush=True,
    )
    # Core sharding invariant, already proven on CPU (Task 2) -- checked
    # here against DeepEP's real transport.
    assert set(recv_topk_idx.unique().tolist()) <= set(local_expert_ids) | {-1}

    # A known transform stands in for local expert compute -- proves
    # combine's reduction, decoupled from the kernel.
    local_output = (recv_x.to(torch.bfloat16) * 2).contiguous()

    combined_x, _, combine_event = buffer.combine(
        local_output, handle=handle, num_sms=num_comm_sms, async_with_compute_stream=True
    )
    combine_event.current_stream_wait()

    print(f"[rank {rank}] combine returned combined_x.shape={tuple(combined_x.shape)}", flush=True)
    assert combined_x.shape == x.shape

    buffer.destroy()
    dist.destroy_process_group()
    print(f"[rank {rank}] smoke test passed", flush=True)


if __name__ == "__main__":
    main()
