# Phase 3 runbook: multi-GPU expert-parallel serving

Budget cap: $25. Hopper-class (H100/H200, SM90) required -- DeepEP V2's
own constraint, not a generic NVLink requirement. Check RunPod's real
catalog at rental time; as of 2026-09-15, 2x H100 NVL on Community cloud
(~$5.18/hr combined) or 2x H100 SXM on Secure cloud (~$6.98/hr combined)
both fit the cap for a multi-hour session.

**What actually happened (2026-09-16), read this before following the
steps below as written:** H100 stock had vanished on both clouds by
rental time (RunPod inventory churns in real time); rented 2x H200 SXM
on Secure cloud instead (~$9.18/hr combined, still well inside the cap).
More importantly, **DeepEP V2's `ElasticBuffer` never worked on this
rental** -- its NCCL Gin backend requires NVSwitch-level multicast (GPU
Fabric Manager), and this pod's `nvidia-smi -q` reports `GPU Fabric
GUID: N/A` with no `fabricmanager` process running (likely a
container-tenancy limitation: Fabric Manager is normally a host-level
daemon a rented container can't start itself). Every one of step 6's
`ElasticBuffer`-based code below was replaced with DeepEP's older V1
(legacy) `Buffer` API, which uses plain NVLink peer-to-peer memory and
worked immediately. See `scripts/gpu/deepep_smoke_test.py`'s docstring
and `src/dispatch/kernels/expert_parallel.py`'s `make_ep_moe_infer` for
the real, working code -- step 6 below is kept as the historical record
of what was planned before real hardware corrected it, not as
instructions to follow literally. Total cost: $13.03 (docs/findings/
2026-09-16-phase-3-multi-gpu-ep-cost.md). Full account, including the
correctness gate result and the measured answer to Phase 3's thesis:
docs/findings/phase-3/2026-09-16-phase-3-multi-gpu-ep-run.md.

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
               recv_x,
               recv_topk_idx,
               recv_topk_weight,
               local_weights,
               matmul,
               local_expert_ids,
               block_m=block_m,
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
               local_weights,
               local_expert_ids,
               matmul,
               buffer,
               n_experts,
               num_max_tokens_per_rank,
               block_m=block_m,
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
   (`docs/findings/phase-1/2026-09-15-phase-1-grouped-gemm-run.md`). If agreement
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
