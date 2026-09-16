# Phase 3: Multi-GPU Expert-Parallel Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put DeepSeekMoE-16B's routed experts on two real, NVLink-connected
Hopper-class GPUs, using DeepEP for cross-GPU dispatch/combine and Phase 1's
own Triton grouped-GEMM kernel for local compute, and measure whether Phase
1's naive-vs-persistent kernel crossover (persistent wins 16-128
tokens/expert, loses at 512-2048) holds under DeepEP's real per-expert
token-count distribution or shifts.

**Architecture:** `grouped_moe_routed` and its pluggable naive/persistent
backend (Phase 1, unchanged) become the *local* half of each rank's MoE
layer. A new `expert_parallel.py` supplies the expert-to-rank sharding and
the DeepEP dispatch/combine wrapper around it; `integration.py` gets a
second `moe_infer` patcher (`patch_moe_infer_ep`) alongside the existing
single-GPU one. The sharding/combine bookkeeping is proven correct on CPU,
in a single process, before DeepEP or a second GPU is ever involved.

**Tech Stack:** DeepEP V2 (NCCL Gin backend), `torch.distributed` (NCCL),
`torchrun --nproc_per_node=2`, this repo's existing `torch`/`triton`
dependencies. RunPod's REST API (`scripts/gpu/runpod_client.py`, extended
here for multi-GPU rentals).

**Spec:** `docs/design/2026-09-15-phase-3-multi-gpu-expert-parallel-serving.md`
(all sections); `docs/adr/0002-deepep-over-hand-rolled-communication.md`
(the Resolution section -- DeepEP confirmed, Hopper-class hardware
required); `src/dispatch/kernels/moe_forward.py`, `integration.py`,
`backends.py` (Phase 1's existing kernel-swap machinery, reused unchanged).

## Global Constraints

- **Budget cap: $25**, set in the design doc, before any rental. Spin up,
  run, capture evidence, tear down immediately.
- **Hardware: Hopper-class (H100/H200, SM90) specifically** -- DeepEP V2's
  own requirement (confirmed live 2026-09-15 against its current README).
  Ampere SXM (A100) does not qualify despite having NVLink. Exact
  type/cloud/data-center chosen live at rental time against RunPod's real
  catalog, same practice as Phase 2. Real pricing checked 2026-09-15: 2x
  H100 NVL on Community cloud (~$5.18/hr combined) or 2x H100 SXM on
  Secure cloud (~$6.98/hr combined) both fit the cap for a multi-hour
  session.
- **Verify real NVLink before spending budget on anything else**:
  `nvidia-smi topo -m` on the rented pod must show `NV#` between the two
  GPUs, not `PHB`/`PXB`/`SYS`. If it doesn't, stop and re-provision --
  requesting `count: 2` of a GPU type does not itself prove the two
  instances are NVLink-connected.
- **DeepEP's version floors** (its own README, confirmed live
  2026-09-15): PyTorch >=2.10 (this repo's own floor, `torch>=2.14.0`,
  already clears it), NCCL >=2.30.4, CUDA >=12.3. Confirm the rented
  image's CUDA/NCCL against these before installing.
- **DeepEP's own required expert-to-rank sharding is a contiguous split**
  (confirmed live in its test code: `num_local_experts = num_experts //
  world_size`, `dst_rank = topk_idx // num_local_experts`) -- Task 2's
  `assign_experts_to_ranks` matches this exactly, not an independent
  design choice. Do not shard experts any other way.
- **Correctness gates the benchmark.** An EP run that fails mutual top-k
  logit agreement against the single-GPU reference refuses to benchmark,
  same discipline as Phase 1's kernel work.
- **Single-GPU baseline re-measured on the same GPU class** as one node
  of the rented pair (i.e. one Hopper-class GPU), not reused from Phase
  0/1's L40/3090 numbers -- CLAUDE.md's benchmark rules require
  same-hardware comparisons.
- **DeepEP's exact dispatch/combine call shape is confirmed live, not
  assumed from its README alone.** The README's worked example and its
  own test code (`tests/elastic/test_ep.py`, fetched 2026-09-15) agree on
  the kwargs and return-tuple shapes this plan's code uses below, but
  Task 3's smoke test runs and prints those shapes on real hardware
  *before* Task 4 wires the same call into the full model -- if the
  printed shapes contradict what's written below, fix the code to match
  what's actually observed, don't force the observation to match the plan.
- **Never quote a benchmark number that wasn't measured** on this exact
  run (repo-wide rule).

## File Structure

```text
scripts/gpu/runpod_client.py            # Task 1: gpu_count param
scripts/gpu/provision.py                # Task 1: --gpu-count CLI flag
tests/unit/test_runpod_client.py        # Task 1
tests/unit/test_provision.py            # Task 1

src/dispatch/kernels/expert_parallel.py # Task 2 (assign_experts_to_ranks,
                                         # local_expert_contribution,
                                         # simulate_ep_moe_routed), extended
                                         # in Task 4 (make_ep_moe_infer)
tests/unit/test_expert_parallel.py      # Task 2

scripts/gpu/deepep_smoke_test.py        # Task 3

docs/runbooks/phase-3-multi-gpu-ep.md              # Task 4
docs/findings/2026-09-15-phase-3-multi-gpu-ep-cost.md  # Task 4 (write_cost_record)
docs/findings/2026-09-15-phase-3-multi-gpu-ep-run.md   # Task 5
docs/STATUS.md                                          # Task 5
```

---

### Task 1: Multi-GPU RunPod provisioning

**Files:**

- Modify: `scripts/gpu/runpod_client.py` (`create_pod`)
- Modify: `scripts/gpu/provision.py` (`_cmd_create`, `build_parser`)
- Test: `tests/unit/test_runpod_client.py`, `tests/unit/test_provision.py`

**Interfaces:**

- Produces: `create_pod(..., gpu_count: int = 1, ...)`; `--gpu-count`
  CLI flag on `provision.py create`, default `1` (existing single-GPU
  behavior unchanged when omitted).

`create_pod` currently hardcodes `"gpu": {"id": gpu_type_id, "count": 1}`
-- Phase 3 is the first phase that needs more than one GPU per pod.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_runpod_client.py`, add:

```python
def test_create_pod_requests_multiple_gpus_when_asked() -> None:
    session = FakeSession(FakeResponse(200, {"id": "pod_1", "status": "PROVISIONING", "cost": 5.18}))

    create_pod(
        "phase-3-ep",
        "NVIDIA H100 NVL",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        gpu_count=2,
        session=session,  # type: ignore[arg-type]
    )

    assert session.calls[0]["json"]["gpu"] == {"id": "NVIDIA H100 NVL", "count": 2}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_runpod_client.py::test_create_pod_requests_multiple_gpus_when_asked -v`
Expected: FAIL with `TypeError: create_pod() got an unexpected keyword argument 'gpu_count'`.

- [ ] **Step 3: Add the parameter**

In `scripts/gpu/runpod_client.py`:

```python
def create_pod(  # noqa: PLR0913 -- a pod-create request has this many independent knobs
    name: str,
    gpu_type_id: str,
    image: str,
    *,
    disk_gb: int = 60,
    gpu_count: int = 1,
    cloud: str = "COMMUNITY",
    ports: tuple[str, ...] = ("22/tcp",),
    start_ssh: bool = True,
    session: requests.Session | None = None,
) -> PodHandle:
    body = {
        "name": name,
        "image": image,
        "gpu": {"id": gpu_type_id, "count": gpu_count},
        "disk": disk_gb,
        "ports": list(ports),
        "cloud": cloud,
        "startSsh": start_ssh,
    }
    data = _request(session or requests.Session(), "POST", "/pods", json=body)
    return _pod_from_response(data)
```

(Only `gpu_count: int = 1,` and `"count": gpu_count` are new.)

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/test_runpod_client.py -v`
Expected: all pass, including the existing
`test_create_pod_sends_expected_body_and_parses_response` (still gets
`count: 1` by default).

- [ ] **Step 5: Thread it through `provision.py`'s CLI**

In `scripts/gpu/provision.py`, update `_cmd_create`:

```python
def _cmd_create(args: argparse.Namespace) -> None:
    pod = create_pod(
        args.name, args.gpu_type, args.image,
        cloud=args.cloud, disk_gb=args.disk_gb, gpu_count=args.gpu_count,
    )
    print(f"created pod {pod.id} status={pod.status} rate=${pod.cost_per_hour:.4f}/hr")
```

and `build_parser`, after the existing `create.add_argument("--disk-gb", ...)`:

```python
    create.add_argument("--gpu-count", type=int, default=1, dest="gpu_count")
```

- [ ] **Step 6: Update the existing fake in `test_provision.py` to accept the new kwarg**

`test_main_create_invokes_create_pod_and_prints_result`'s `fake_create_pod`
must accept `gpu_count` now that `_cmd_create` always passes it:

```python
def test_main_create_invokes_create_pod_and_prints_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_create_pod(
        name: str,
        gpu_type_id: str,
        image: str,
        *,
        cloud: str = "COMMUNITY",
        disk_gb: int = 60,
        gpu_count: int = 1,
    ) -> PodHandle:
        return PodHandle(id="pod_999", status="PROVISIONING", cost_per_hour=0.4)

    monkeypatch.setattr("scripts.gpu.provision.create_pod", fake_create_pod)

    main(["create", "--name", "x", "--gpu-type", "NVIDIA A40", "--image", "img"])

    assert "pod_999" in capsys.readouterr().out
```

- [ ] **Step 7: Add the new provision.py tests**

```python
def test_build_parser_create_defaults_to_a_single_gpu() -> None:
    parser = build_parser()

    args = parser.parse_args(["create", "--name", "x", "--gpu-type", "NVIDIA A40", "--image", "img"])

    assert args.gpu_count == 1


def test_main_create_passes_gpu_count_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_pod(
        name: str,
        gpu_type_id: str,
        image: str,
        *,
        cloud: str = "COMMUNITY",
        disk_gb: int = 60,
        gpu_count: int = 1,
    ) -> PodHandle:
        captured["gpu_count"] = gpu_count
        return PodHandle(id="pod_999", status="PROVISIONING", cost_per_hour=5.18)

    monkeypatch.setattr("scripts.gpu.provision.create_pod", fake_create_pod)

    main(
        [
            "create",
            "--name",
            "x",
            "--gpu-type",
            "NVIDIA H100 NVL",
            "--image",
            "img",
            "--gpu-count",
            "2",
        ]
    )

    assert captured["gpu_count"] == 2
```

- [ ] **Step 8: Run the full suite and typecheck**

Run: `uv run pytest tests/unit/test_runpod_client.py tests/unit/test_provision.py -v && uv run mypy src scripts tests`
Expected: all pass, mypy clean.

- [ ] **Step 9: Commit**

```bash
git add scripts/gpu/runpod_client.py scripts/gpu/provision.py tests/unit/test_runpod_client.py tests/unit/test_provision.py
git commit -m "feat: support multi-GPU pod rentals in RunPod provisioning"
```

---

### Task 2: Expert-to-rank sharding, proven correct on CPU before any GPU is involved

**Files:**

- Create: `src/dispatch/kernels/expert_parallel.py`
- Test: `tests/unit/test_expert_parallel.py`

**Interfaces:**

- Consumes: `GroupedMatmul`, `StackedExpertWeights`, `grouped_moe_routed`
  (`moe_forward.py`, Phase 1, unchanged).
- Produces: `assign_experts_to_ranks(n_experts, n_ranks) -> torch.Tensor`;
  `local_expert_contribution(x, topk_idx, topk_weight, local_weights,
  matmul, local_expert_ids, *, block_m=16) -> torch.Tensor`;
  `simulate_ep_moe_routed(x, topk_idx, topk_weight, weights, matmul,
  rank_of_expert, *, block_m=16) -> torch.Tensor`. Task 4 extends this
  same file with `make_ep_moe_infer`, built on
  `local_expert_contribution` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_expert_parallel.py`:

```python
"""CPU-only: proves the expert-to-rank sharding and dispatch/combine
bookkeeping is correct before Phase 3's real DeepEP integration ever
touches a GPU. simulate_ep_moe_routed must reproduce grouped_moe_routed's
existing single-process output exactly, since it's the same computation
partitioned by expert-owning rank and summed back.
"""

from __future__ import annotations

import pytest
import torch

from dispatch.kernels.expert_parallel import assign_experts_to_ranks, simulate_ep_moe_routed
from dispatch.kernels.moe_forward import (
    grouped_moe_routed,
    stack_expert_weights,
    torch_grouped_matmul,
)
from dispatch.kernels.reference_moe import MoEConfig, ReferenceMoE

TOY_CONFIG = MoEConfig(
    hidden_size=8,
    moe_intermediate_size=16,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=3,
)


def test_assign_experts_to_ranks_splits_64_experts_evenly_over_2_ranks() -> None:
    ranks = assign_experts_to_ranks(64, 2)

    assert ranks[:32].eq(0).all()
    assert ranks[32:].eq(1).all()


def test_assign_experts_to_ranks_rejects_more_ranks_than_experts() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        assign_experts_to_ranks(1, 2)


def test_assign_experts_to_ranks_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        assign_experts_to_ranks(0, 2)


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_simulate_ep_moe_routed_matches_the_non_ep_reference(n_ranks: int) -> None:
    torch.manual_seed(0)
    moe = ReferenceMoE(TOY_CONFIG)
    hidden_states = torch.randn(11, TOY_CONFIG.hidden_size)
    topk_idx, topk_weight = moe.route(hidden_states)
    weights = stack_expert_weights(moe.experts)
    expected = grouped_moe_routed(hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul)

    rank_of_expert = assign_experts_to_ranks(TOY_CONFIG.n_routed_experts, n_ranks)
    actual = simulate_ep_moe_routed(
        hidden_states, topk_idx, topk_weight, weights, torch_grouped_matmul, rank_of_expert
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_expert_parallel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch.kernels.expert_parallel'`.

- [ ] **Step 3: Write the implementation**

Create `src/dispatch/kernels/expert_parallel.py`:

```python
"""Expert-to-rank sharding for multi-GPU expert-parallel MoE (Phase 3).
assign_experts_to_ranks is a plain contiguous split, matching DeepEP's own
required convention exactly (confirmed live 2026-09-15 against its test
code: num_local_experts = num_experts // world_size, dst_rank = topk_idx
// num_local_experts) -- not an independent design choice.
simulate_ep_moe_routed proves the dispatch/local-compute/combine
bookkeeping is correct in a single process, with no GPU and no DeepEP
import, before the real library is ever involved.
"""

from __future__ import annotations

import torch

from dispatch.kernels.moe_forward import GroupedMatmul, StackedExpertWeights, grouped_moe_routed


def assign_experts_to_ranks(n_experts: int, n_ranks: int) -> torch.Tensor:
    """expert_id -> rank_id, contiguous blocks (e.g. 64 experts over 2
    ranks -> experts 0-31 on rank 0, 32-63 on rank 1). Every rank computes
    this identically and independently -- nothing to communicate."""
    if n_experts <= 0 or n_ranks <= 0:
        raise ValueError(f"n_experts={n_experts} and n_ranks={n_ranks} must both be positive")
    if n_experts < n_ranks:
        raise ValueError(f"n_experts={n_experts} is fewer than n_ranks={n_ranks}")
    return torch.arange(n_experts) * n_ranks // n_experts


def local_expert_contribution(  # noqa: PLR0913 -- routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    local_expert_ids: torch.Tensor,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    """The rows a real dispatch + local-compute + combine round trip would
    contribute for one rank: zero for any (token, slot) whose expert isn't
    in local_expert_ids, the weighted expert output otherwise. Feeds
    straight into the existing grouped_moe_routed by remapping global
    expert ids to this rank's local 0..num_local-1 indexing and zeroing
    the weight (not dropping the row) for non-local slots, so shapes stay
    valid without changing grouped_moe_routed itself."""
    is_local = torch.isin(topk_idx, local_expert_ids)
    local_index_of = torch.zeros(int(topk_idx.max().item()) + 1, dtype=torch.int64)
    local_index_of[local_expert_ids] = torch.arange(local_expert_ids.numel())
    local_topk_idx = local_index_of[topk_idx.clamp(min=0)]
    local_topk_weight = torch.where(is_local, topk_weight, torch.zeros_like(topk_weight))
    return grouped_moe_routed(
        x, local_topk_idx, local_topk_weight, local_weights, matmul, block_m=block_m
    )


def simulate_ep_moe_routed(  # noqa: PLR0913 -- routing inputs plus a pluggable GEMM and tile size
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: StackedExpertWeights,
    matmul: GroupedMatmul,
    rank_of_expert: torch.Tensor,
    *,
    block_m: int = 16,
) -> torch.Tensor:
    """Single-process stand-in for real cross-GPU dispatch + per-rank
    local compute + combine: sums each rank's local_expert_contribution
    over only the experts rank_of_expert assigns it. Proves the
    sharding/combine arithmetic in isolation from DeepEP's real
    transport -- no process group, no GPU, no DeepEP import."""
    n_ranks = int(rank_of_expert.max().item()) + 1
    combined = torch.zeros_like(x)
    for rank in range(n_ranks):
        local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0]
        local_weights = StackedExpertWeights(
            gate=weights.gate[local_expert_ids],
            up=weights.up[local_expert_ids],
            down=weights.down[local_expert_ids],
        )
        combined = combined + local_expert_contribution(
            x, topk_idx, topk_weight, local_weights, matmul, local_expert_ids, block_m=block_m
        )
    return combined
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_expert_parallel.py -v`
Expected: all pass.

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/dispatch/kernels/expert_parallel.py tests/unit/test_expert_parallel.py && uv run ruff format --check src/dispatch/kernels/expert_parallel.py tests/unit/test_expert_parallel.py && uv run mypy src tests`
Expected: clean. (Add `"src/dispatch/kernels/expert_parallel.py" = ["PLR0913"]` to `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` only if the inline `# noqa: PLR0913` comments above don't satisfy ruff -- try the inline comments first, they match this file's own existing convention in `moe_forward.py`.)

- [ ] **Step 6: Commit**

```bash
git add src/dispatch/kernels/expert_parallel.py tests/unit/test_expert_parallel.py
git commit -m "feat: expert-to-rank sharding for multi-GPU EP, proven correct on CPU"
```

---

### Task 3: DeepEP dispatch/combine smoke test on toy data

**Files:**

- Create: `scripts/gpu/deepep_smoke_test.py`

**Interfaces:**

- Consumes: `assign_experts_to_ranks` (Task 2).
- Produces: nothing consumed by later *code* -- its printed shapes and
  pass/fail result are what Task 4's real `make_ep_moe_infer` is built
  and (if needed) corrected against.

Not a pytest test: needs `torch.distributed`, 2 real GPUs, and `deep_ep`
installed, none of which CI or a single-GPU dev box has. Runs once, live,
on the rented pod in Task 4, before the full model integration is written
against unconfirmed assumptions.

- [ ] **Step 1: Write the smoke test**

Create `scripts/gpu/deepep_smoke_test.py`:

```python
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
```

- [ ] **Step 2: Syntax-check without GPU/deep_ep (this repo's dev box has neither)**

Run: `python3 -c "import ast; ast.parse(open('scripts/gpu/deepep_smoke_test.py').read()); print('parses cleanly')"`
Expected: `parses cleanly`. This is a syntax-level check only -- its actual
behavior is proven live in Task 4, on the rented pod.

- [ ] **Step 3: Commit**

```bash
git add scripts/gpu/deepep_smoke_test.py
git commit -m "feat: DeepEP dispatch/combine smoke test for Phase 3's real integration"
```

---

### Task 4: GPU rental runbook -- verify, build the real EP layer live, correctness gate, benchmark

**Files:**

- Create: `docs/runbooks/phase-3-multi-gpu-ep.md`
- Create (via `write_cost_record`, reused unmodified):
  `docs/findings/2026-09-15-phase-3-multi-gpu-ep-cost.md`
- Extends `src/dispatch/kernels/expert_parallel.py` live, on the pod, with
  `make_ep_moe_infer` -- written here rather than pre-committed, because
  its exact shape depends on Task 3's smoke-test findings (Global
  Constraints).

**Budget cap: $25, set in the design doc, before this task runs.** This is
a live session with the user, not something to run unattended -- get
explicit go-ahead before renting.

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/phase-3-multi-gpu-ep.md`:

````markdown
# Phase 3 runbook: multi-GPU expert-parallel serving

Budget cap: $25. Hopper-class (H100/H200, SM90) required -- DeepEP V2's
own constraint, not a generic NVLink requirement. Check RunPod's real
catalog at rental time; as of 2026-09-15, 2x H100 NVL on Community cloud
(~$5.18/hr combined) or 2x H100 SXM on Secure cloud (~$6.98/hr combined)
both fit the cap for a multi-hour session.

1. Create and wait for the 2-GPU pod:

   ```bash
   uv run python scripts/gpu/provision.py create --name phase-3-multi-gpu-ep \
     --gpu-type <Hopper-class type available at rental time> --gpu-count 2 \
     --image <RunPod CUDA/PyTorch template, CUDA>=12.3> --cloud <community|secure> --disk-gb 40
   uv run python scripts/gpu/provision.py wait <pod-id>
   ```

2. **Verify real NVLink before doing anything else** -- this is the whole
   premise of the phase:

   ```bash
   nvidia-smi topo -m
   ```

   Expected: an `NV#` entry between GPU 0 and GPU 1. If it shows
   `PHB`/`PXB`/`SYS` instead, stop, terminate the pod, and re-provision --
   do not spend any more of the budget against a non-NVLink pair.

3. Confirm the version floors DeepEP's README states (Global Constraints):

   ```bash
   python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
   nvcc --version
   python3 -c "import torch.cuda.nccl as nccl; print(nccl.version())"
   ```

   Expected: PyTorch >=2.10, CUDA >=12.3, NCCL >=2.30.4. If any floor
   isn't met by the base image, install/upgrade before proceeding (e.g.
   `pip install "nvidia-nccl-cu13>=2.30.4" --no-deps`, per DeepEP's own
   documented install step).

4. Install DeepEP (NCCL Gin backend, no NVSHMEM build needed for this
   phase's pure-intranode topology):

   ```bash
   git clone https://github.com/deepseek-ai/DeepEP.git
   cd DeepEP
   pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
   python setup.py install
   cd ..
   python3 -c "import deep_ep; print('deep_ep imported OK')"
   ```

   If this fails, record exactly why in the findings doc and stop --
   do not fall back to a from-source NVSHMEM build without first checking
   with the user, since that risks the time budget on a build rather than
   a measurement.

5. Clone this repo onto the pod and run Task 3's smoke test:

   ```bash
   git clone <this repo's URL> dispatch
   cd dispatch
   uv sync
   torchrun --nproc_per_node=2 scripts/gpu/deepep_smoke_test.py
   ```

   Expected: both ranks print `smoke test passed`. Record the printed
   `recv_x.shape`/`recv_topk_idx.shape`/`recv_topk_weight.shape` values --
   step 6 depends on them matching what `expert_parallel.py`'s
   `local_expert_contribution` (Task 2) expects: `recv_topk_idx` and
   `recv_topk_weight` shaped `(num_recv_tokens, NUM_TOPK)`, matching the
   *sent* `topk_idx`/`topk_weights` shape, not flattened to one row per
   slot. If the printed shapes don't match this, adjust step 6's code to
   fit what was actually observed before writing more of it.

6. **Write `make_ep_moe_infer` into `expert_parallel.py` on the pod**,
   directly against step 5's confirmed shapes:

   ```python
   # Appended to src/dispatch/kernels/expert_parallel.py on the pod
   from collections.abc import Callable

   from deep_ep import ElasticBuffer

   MoEInfer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


   def make_ep_moe_infer(  # noqa: PLR0913
       local_weights: StackedExpertWeights,
       local_expert_ids: torch.Tensor,
       matmul: GroupedMatmul,
       buffer: ElasticBuffer,
       num_experts: int,
       num_max_tokens_per_rank: int,
       *,
       block_m: int = 16,
   ) -> MoEInfer:
       num_comm_sms = buffer.get_theoretical_num_sms(num_experts, local_expert_ids.numel())

       @torch.no_grad()
       def moe_infer(
           x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
       ) -> torch.Tensor:
           top_k = flat_expert_indices.numel() // x.shape[0]
           topk_idx = flat_expert_indices.view(-1, top_k)
           topk_weight = flat_expert_weights.view(-1, top_k)

           recv_x, recv_topk_idx, recv_topk_weight, handle, event = buffer.dispatch(
               x,
               topk_idx=topk_idx,
               topk_weights=topk_weight,
               num_experts=num_experts,
               num_max_tokens_per_rank=num_max_tokens_per_rank,
               num_sms=num_comm_sms,
               async_with_compute_stream=True,
           )
           event.current_stream_wait()

           local_out = local_expert_contribution(
               recv_x, recv_topk_idx, recv_topk_weight, local_weights, matmul,
               local_expert_ids, block_m=block_m,
           )

           combined_x, _, combine_event = buffer.combine(
               local_out, handle=handle, num_sms=num_comm_sms, async_with_compute_stream=True
           )
           combine_event.current_stream_wait()
           return combined_x

       return moe_infer


   def patch_moe_infer_ep(
       model: torch.nn.Module,
       matmul: GroupedMatmul,
       buffer: ElasticBuffer,
       rank: int,
       n_ranks: int,
       *,
       num_max_tokens_per_rank: int,
       block_m: int = 16,
   ) -> int:
       """Same swap-in contract as moe_forward.py's patch_moe_infer, but
       each layer's moe_infer only computes this rank's expert shard,
       dispatching/combining the rest via DeepEP."""
       model.eval()
       patched = 0
       for module in model.modules():
           if not hasattr(module, "moe_infer"):
               continue
           experts = module.experts
           if not isinstance(experts, torch.nn.ModuleList):
               raise TypeError(
                   f"expected {type(module).__name__}.experts to be nn.ModuleList, "
                   f"got {type(experts).__name__}"
               )
           n_experts = len(experts)
           rank_of_expert = assign_experts_to_ranks(n_experts, n_ranks)
           local_expert_ids = (rank_of_expert == rank).nonzero(as_tuple=True)[0]
           all_weights = stack_expert_weights(experts)
           local_weights = StackedExpertWeights(
               gate=all_weights.gate[local_expert_ids],
               up=all_weights.up[local_expert_ids],
               down=all_weights.down[local_expert_ids],
           )
           module.moe_infer = make_ep_moe_infer(  # type: ignore[assignment]
               local_weights, local_expert_ids, matmul, buffer, n_experts,
               num_max_tokens_per_rank, block_m=block_m,
           )
           patched += 1
       return patched
   ```

   (Add the matching `from dispatch.kernels.moe_forward import
   stack_expert_weights` import if not already present after Task 2.)

7. **Correctness gate -- must pass before any benchmark runs.** On rank 0
   only, load the model once with the existing single-GPU
   `patch_moe_infer` (naive kernel) and run a fixed prompt/seed to get a
   reference logit tensor. Then, across both ranks via `torchrun
   --nproc_per_node=2`, load the same model with `patch_moe_infer_ep` and
   run the same prompt/seed; gather rank 0's output logits. Compare
   top-5 and top-1 argmax agreement at every position against the
   single-GPU reference, the same bar Phase 1 used
   (`docs/findings/2026-09-15-phase-1-grouped-gemm-run.md`). If agreement
   fails, stop -- do not benchmark a wrong result. Record the exact
   agreement measured (perfect, or the first position/token where it
   diverges) in the findings doc either way.

8. **Only if step 7 passes**, run the benchmark sweep -- decode and
   prefill, a small batch-size sweep (e.g. 1, 8, 32 tokens/expert-ish,
   bracketing Phase 1's own 16-128-token win region and 512+-token loss
   region), {naive, persistent} kernel, {single-GPU, 2-GPU EP} topology.
   Reuse `dispatch.kernels.bench.time_grouped_gemm` and
   `summarize_kernel_latencies` unchanged for the timing -- the only new
   thing being timed is which forward function runs (single-GPU
   `grouped_moe_routed` call vs. the EP-patched layer's `moe_infer`).
   Watch running cost; if pace projects past the $25 cap before the full
   grid finishes, stop after the last fully-completed batch size and
   report exactly what was measured, per this project's standing cost
   discipline.

9. Copy all console output and any saved JSON back to this repo's
   `docs/findings/` before doing anything else.

10. Tear down immediately:

    ```bash
    uv run python scripts/gpu/provision.py terminate <pod-id>
    ```

11. From this repo's root, record the measured cost:

    ```python
    from pathlib import Path
    from scripts.gpu.provision import write_cost_record

    write_cost_record(
        Path("docs/findings"),
        pod_id="<pod-id>",
        gpu_type_id="<GPU type actually used>",
        cost_per_hour=<combined rate for both GPUs>,
        duration_s=<measured seconds>,
        note=(
            "Phase 3: DeepEP install, NVLink verification, EP correctness "
            "gate, decode/prefill benchmark sweep vs single-GPU baseline"
        ),
        run_label="phase-3-multi-gpu-ep",
    )
    ```
````

- [ ] **Step 2: Get the user's explicit go-ahead, then execute the runbook**

Confirm the budget cap and GPU choice with the user before the first
`create` call -- this is a paid action, never taken unattended.

- [ ] **Step 3: Commit the runbook, the pod-live `expert_parallel.py`
  addition, and the cost record**

```bash
git add docs/runbooks/phase-3-multi-gpu-ep.md docs/findings/2026-09-15-phase-3-multi-gpu-ep-cost.md src/dispatch/kernels/expert_parallel.py
git commit -m "docs: record Phase 3 GPU runbook and cost; add DeepEP-backed EP MoE layer"
```

If step 6 needed adjustment to match the smoke test's real shapes, note
the adjustment and why in the commit body -- not silently, per this
project's practice of recording what actually happened rather than what
was assumed.

---

### Task 5: Findings doc and STATUS.md

**Files:**

- Create: `docs/findings/2026-09-15-phase-3-multi-gpu-ep-run.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write the findings doc**

Cover, in `docs/findings/2026-09-15-phase-3-multi-gpu-ep-run.md`: whether
real NVLink was confirmed and on what hardware; whether DeepEP installed
cleanly or needed a documented workaround; the correctness-gate result
(pass, with the measured agreement, or the honest reason it didn't); the
full measured decode/prefill x batch-size x kernel x topology grid, if the
gate passed; **the direct answer to Phase 3's thesis** (does Phase 1's
naive-vs-persistent crossover hold, shift, or disappear under DeepEP's
real per-expert token counts) stated at exactly the strength the data
supports, not amplified; and the actual GPU type, duration, and cost from
Task 4's cost record. If the correctness gate failed or the budget ran out
before the full grid, say so plainly and report exactly what was
measured -- matching this project's practice of writing down a null or
partial result rather than a flattering guess.

- [ ] **Step 2: Update STATUS.md**

Add a "## Phase 3 progress" section following the Phase 0/1/2 pattern:
plan link, hardware actually used, the correctness-gate result, the
crossover-thesis answer, and total GPU cost. Set "## Next step" to
reflect what's actually next (Phase 4 planning, or follow-up on Phase 3 if
something didn't land cleanly).

- [ ] **Step 3: Commit**

```bash
git add docs/findings/2026-09-15-phase-3-multi-gpu-ep-run.md docs/STATUS.md
git commit -m "docs: record Phase 3 multi-GPU expert-parallel serving outcome"
```

---

## Self-Review Notes

- **Spec coverage:** design doc §3-5 (architecture, components, data
  flow) map to Tasks 2-4 (sharding + DeepEP wrapper reuse
  `grouped_moe_routed`/`moe_forward.py` exactly as specified, unchanged).
  §6 (testing table's new EP-bookkeeping row, correctness gate, benchmark
  row) maps to Task 2 (CPU test) and Task 4 steps 7-8 (GPU-paid gate and
  sweep). §7 (risk/cost/rollout: hardware, NVLink verification, budget,
  session order) maps to Task 4's Global Constraints and runbook steps
  1-4. §1's thesis (crossover-shift question) is answered directly in
  Task 5's findings doc, not left implicit.
- **New scope beyond the design doc:** Task 1 (multi-GPU provisioning
  support) wasn't named in the design doc -- it surfaced while reading the
  actual `runpod_client.py` source (`create_pod` hardcoded `count: 1`),
  the same "read the live code before assuming" discipline Phase 2's plan
  already established. Necessary infrastructure, not scope creep: Phase 3
  cannot rent 2 GPUs without it. Task 3 (the DeepEP smoke test) similarly
  wasn't in the design doc by name, but follows directly from the design
  doc's own §7 admission that DeepEP's exact call shape needs live
  verification -- made concrete here as its own reviewable step rather
  than folded silently into Task 4.
- **Ambiguity check:** "a few batch sizes bracketing Phase 1's win/loss
  regions" (Task 4 step 8) is deliberately not pinned to exact numbers
  here, unlike Phase 2's plan which did pin `--batch-size 1 2 4 8 16` --
  the right EP-relevant token counts depend on DeepEP's real dispatch
  behavior (how many tokens actually land on each rank's each expert),
  which isn't known until Task 3's smoke test and Task 4's correctness
  gate have run on real hardware. Pinning exact numbers now, before that
  evidence exists, would be guessing dressed up as precision -- the
  runbook instead names the *goal* (bracket the 16-128 win region and
  512+ loss region) and leaves the exact grid to be set live, the same
  choice Phase 2 made for GPU type and Phase 0 made for model precision.
- **Type consistency:** `local_expert_contribution`'s signature (Task 2)
  and `make_ep_moe_infer`'s call to it (Task 4) match exactly:
  `(x, topk_idx, topk_weight, local_weights, matmul, local_expert_ids,
  *, block_m=16)`. `patch_moe_infer_ep`'s use of `stack_expert_weights`
  and `StackedExpertWeights` matches `moe_forward.py`'s existing
  definitions verbatim, no renaming.
