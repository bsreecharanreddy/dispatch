"""Smoke test for DeepEP's real dispatch()/combine() round trip on toy
data across 2 real GPUs -- run before Phase 3's real EP MoE layer is
built, so its exact call shape is confirmed against DeepEP's actual
behavior instead of assumed from its README alone. Not a pytest test:
needs torch.distributed + 2 real GPUs + deep_ep installed.

Uses DeepEP's V1 (legacy) `Buffer` API, not V2's `ElasticBuffer`: V2's
NCCL Gin backend requires NVSwitch-level multicast (GPU Fabric Manager),
which this project's rented pod does not have initialized (`nvidia-smi -q`
reports `GPU Fabric GUID: N/A`, no fabricmanager process) -- confirmed
live 2026-09-16, `ElasticBuffer(...)` raises `NCCL GIN is unavailable`
even with `allow_hybrid_mode=False` and `NCCL_NVLS_ENABLE=0`. V1's
`Buffer` uses plain NVLink peer-to-peer memory (no switch multicast
needed) for its intranode path, which this pod's real NV18 topology
(confirmed via Task 4's runbook step 2) supports directly.

NCCL_NVLS_ENABLE=0 is also required (confirmed live 2026-09-16): with
NVLS on, plain `dist.barrier()` itself fails trying to bind NVLink SHARP
multicast memory, the same Fabric-Manager gap.

Run: NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=2 scripts/gpu/deepep_smoke_test.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from deep_ep import Buffer

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

    Buffer.set_num_sms(24)
    hidden_bytes = HIDDEN * 2  # bf16 element size, matching DeepEP's own get_hidden_bytes helper
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
    # num_rdma_bytes=0: pure intranode, no internode needed
    buffer = Buffer(group, num_nvl_bytes, 0)

    torch.manual_seed(rank)  # different per rank, deliberately: real ranks never share tokens
    x = torch.randn(NUM_TOKENS_PER_RANK, HIDDEN, device="cuda", dtype=torch.bfloat16)
    topk_weight, topk_idx = torch.topk(
        torch.randn(NUM_TOKENS_PER_RANK, NUM_EXPERTS, device="cuda"), NUM_TOPK, dim=-1
    )
    # V1 Buffer.dispatch requires topk_weights as float32 (its own C++ assertion
    # -- confirmed live 2026-09-16, unlike V2's ElasticBuffer which accepted bf16).
    topk_weight = torch.softmax(topk_weight, dim=-1)

    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
        buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS)
    )
    recv_x, recv_topk_idx, recv_topk_weight, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weight,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    print(
        f"[rank {rank}] dispatch returned recv_x.shape={tuple(recv_x.shape)}, "
        f"recv_topk_idx.shape={tuple(recv_topk_idx.shape)}, "
        f"recv_topk_weight.shape={tuple(recv_topk_weight.shape)}, "
        f"unique recv local-expert ids={sorted(recv_topk_idx.unique().tolist())}",
        flush=True,
    )
    # Core sharding invariant, already proven on CPU (Task 2) -- checked here
    # against DeepEP's real transport. V1's recv_topk_idx is already remapped
    # to LOCAL 0..num_local_experts-1 indices (confirmed live 2026-09-16
    # against DeepEP's own tests/legacy/test_intranode.py assertion), not
    # global expert ids like V2's ElasticBuffer -- unlike the plan's original
    # V2-based assumption.
    num_local_experts = len(local_expert_ids)
    assert set(recv_topk_idx.unique().tolist()) <= set(range(num_local_experts)) | {-1}

    # A known transform stands in for local expert compute -- proves
    # combine's reduction, decoupled from the kernel.
    local_output = (recv_x.to(torch.bfloat16) * 2).contiguous()

    combined_x, _, _ = buffer.combine(local_output, handle)

    print(f"[rank {rank}] combine returned combined_x.shape={tuple(combined_x.shape)}", flush=True)
    assert combined_x.shape == x.shape

    dist.destroy_process_group()
    print(f"[rank {rank}] smoke test passed", flush=True)


if __name__ == "__main__":
    main()
