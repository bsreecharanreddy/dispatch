# Phase 3 multi-GPU expert-parallel run -- findings

## Infrastructure

- Pod `3i210zz3qvgvo1`, RunPod Secure Cloud, data center US-CA-2, 2x
  NVIDIA H200 SXM (141GB VRAM each, real NVLink confirmed -- see below).
  **Not** the H100 NVL the design doc originally quoted: H100 stock
  (SXM, NVL, and PCIe) had vanished on both RunPod clouds by rental time,
  ~1 minute after a live catalog check showed `LOW` availability --
  real-time marketplace churn, not a planning error. H200 SXM was the
  only Hopper-class 2-GPU option showing real stock at that moment.
  Rate: **$9.18/hr** combined.
- `nvidia-smi topo -m` confirmed real NVLink: `GPU0 <-> GPU1` shows
  `NV18` (18 bonded links), not `PHB`/`PXB`/`SYS` -- this phase's whole
  premise held.
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8
  toolkit preinstalled); `uv sync` pulled this repo's own `torch==2.14.0`
  floor, which defaults to a `+cu130` build.
- Same `transformers==4.57.6` pin + `DynamicCache.get_usable_length`
  monkeypatch Phase 0/1 needed, applied identically via a pod-local,
  never-committed `_patch_and_run.py` wrapper. DeepSeek's model repo is
  still unchanged since Phase 0 (`lastModified` `2024-01-12`).
- Pod terminated via `pod-action terminate` after all findings were
  copied off; a follow-up `list-pods` returned an empty list, confirming
  full removal.

## DeepEP: V2 (`ElasticBuffer`) does not work on this rental; V1 (`Buffer`) does

The design doc and plan assumed DeepEP V2's `ElasticBuffer` (its current,
NCCL-Gin-backed API). That assumption did not survive contact with real
hardware:

- `ElasticBuffer(...)` first segfaulted inside its constructor
  (`get_nccl_comm_handle`'s reuse of PyTorch's NCCL communicator via
  `backend._comm_ptr()`, called before any real collective had forced
  NCCL to actually create that communicator -- fixed by adding
  `dist.barrier()` right after `init_process_group()`).
- With that fixed, plain `dist.barrier()` itself then failed: NCCL tried
  to bind NVLink SHARP (NVLS) multicast memory and got `CUDA error 401`,
  with NCCL's own error text pointing at "a system or configuration
  error in the Fabric Manager or NVSwitches." `nvidia-smi -q` confirmed
  it: `GPU Fabric GUID: N/A`, and no `fabricmanager` process anywhere on
  the pod. Fabric Manager is normally a host-level daemon; a rented
  container tenant on a shared multi-GPU host apparently can't start it
  itself. Working around this with `NCCL_NVLS_ENABLE=0` unblocked
  `barrier()`.
- Even past that, `ElasticBuffer`'s own constructor asserted: `NCCL GIN
  is unavailable`, with `allow_hybrid_mode=False` making no difference
  (the assertion checks `props.ginType`, not the hybrid-mode variant,
  when hybrid mode is off). GIN -- DeepEP V2's whole transport --
  appears to depend on the same NVSwitch-level multicast capability that
  NVLS needed and this rental doesn't have initialized.

This looks like a rental-tier/container-tenancy limitation of this
specific pod, not a bug in DeepEP or in this project's code -- but it
meant Task 3's plan (confirm V2's exact call shape live, then build Task
4 against it) had no V2 to confirm against. Switched instead to DeepEP's
older **V1 (legacy) `Buffer` API**, which uses plain NVLink peer-to-peer
memory for its intranode path (no switch-level multicast) and worked
immediately once its own two real quirks were handled: `dispatch()`
requires `topk_weights` as `float32` (not bf16, unlike V2), and its
`recv_topk_idx` comes back already remapped to this rank's **local**
`0..num_local_experts-1` indexing rather than global expert ids (matches
what `local_expert_contribution`'s existing global-to-local remap was
built to produce anyway -- calling it with an identity `local_expert_ids
= arange(num_local_experts)` reuses that logic as a no-op remap that
still correctly zeros DeepEP's `-1` padding sentinel).

Two real correctness-relevant bugs were found and fixed along the way,
neither exercised by the CPU-only tests that preceded real hardware:

1. **Device placement**: `local_expert_contribution`'s remap tensors
   (`torch.zeros`/`torch.arange`) defaulted to CPU regardless of the
   surrounding tensors' device -- the first real CUDA call raised a
   device-mismatch error. CPU tensors trivially satisfy the same-device
   check, so this needed an actual GPU to surface; a `gpu`-marked
   regression test now covers it.
2. **Undersized remap table**: `local_index_of` was sized off
   `topk_idx`'s own *observed* max in the current batch, not
   `local_expert_ids`' own max. With 32 local experts per rank and a
   short real prompt, plenty of forward passes never route any token to
   this rank's highest-numbered local expert, so `local_index_of[local_
   expert_ids]` indexed out of bounds -- surfaced as an async CUDA
   device-side assert; `CUDA_LAUNCH_BLOCKING=1` pinned the real faulting
   line. The toy tests never caught this because their randomly-routed
   batches were large relative to their (small) expert counts, hitting
   every local expert by chance every time.

Both fixes are covered by CPU (and, for the first, GPU) regression
tests; see `src/dispatch/kernels/expert_parallel.py` and
`tests/unit/test_expert_parallel.py` for the exact commits.

`make_ep_moe_infer`/`patch_moe_infer_ep` (`expert_parallel.py`) were
validated first against a toy 8-expert model across both real GPUs,
matching the CPU-proven `simulate_ep_moe_routed` reference exactly
(`scripts/gpu/deepep_toy_moe_check.py`), before ever touching the real
16B-parameter model.

## Correctness gate: passed

Single-GPU reference (naive kernel, `patch_moe_infer`) vs. 2-GPU EP
(naive kernel via DeepEP's real V1 `Buffer`, `patch_moe_infer_ep`), same
3 prompts as Phase 0/1, on the real `deepseek-ai/deepseek-moe-16b-base`
(27 MoE layers patched both sides):

| prompt | positions | top-1 agreement | mutual top-5 | max abs logit diff |
|---|---|---|---|---|
| `prompt_000` | 11 | **1.0000** | **true** | 0.680 |
| `prompt_001` | 11 | **1.0000** | **true** | 0.875 |
| `prompt_002` | 7 | **1.0000** | **true** | 1.219 |

Perfect top-1 agreement and mutual top-5 agreement at every tested
position of every prompt -- the same bar Phase 1 used, met the same way.
The `max_abs_diff` values (0.68-1.22) are the same order of magnitude as
Phase 1's own control-row numbers (0.66-1.90, from an eager reformulation
with no kernel or cross-GPU transport involved at all), i.e. ordinary
bf16 summation-order noise, not evidence of a correctness problem.
Full data: `docs/findings/2026-09-16-phase-3-correctness-gate-results.json`.

## The measured answer to Phase 3's thesis

**Does Phase 1's naive-vs-persistent crossover (persistent wins
16-128 tokens/expert, loses at 512-2048, ties at 1-token decode) hold
under DeepEP's real per-expert token-count distribution, or shift?**

Two measurements, in order:

**1. The crossover does not reproduce on H200 at all**, independent of
EP. Re-running Phase 1's own kernel-bench methodology
(`scripts/run_kernel_bench.py`, unmodified) on a single H200 at Phase
1's original token counts:

| tokens | torch/layer | naive/layer | persistent/layer | winner |
|---|---|---|---|---|
| 16 | 7.069ms | 0.560ms | 1.061ms | naive, by 89.7% |
| 32 | 5.121ms | 0.704ms | 0.724ms | naive, by 2.8% |
| 64 | 6.145ms | 0.627ms | 0.861ms | naive, by 37.4% |
| 128 | 6.596ms | 0.647ms | 1.003ms | naive, by 55.1% |
| 256 | 6.617ms | 0.703ms | 1.192ms | naive, by 69.7% |
| 512 | 7.767ms | 0.902ms | 1.749ms | naive, by 93.8% |
| 1024 | 8.904ms | 1.425ms | 3.437ms | naive, by 141.2% |
| 2048 | 8.003ms | 2.396ms | 4.919ms | naive, by 105.3% |

Naive wins at **every** token count tested, including the 16-128 range
where Phase 1 measured persistent winning 3-8% on RTX 3090/L40. This is
a hardware-generation finding, not an EP one: H200 (Hopper, SM90, much
larger/faster L2 and more SMs than Ampere/Ada) apparently changes the
balance enough that persistent's grouped-launch-ordering optimization no
longer pays for itself anywhere in this range, on a single GPU, before
DeepEP or a second rank is ever involved.

**2. Real DeepEP dispatch produces far smaller per-expert token counts
than Phase 1 ever tested**, at this workload's actual scale. Measured
directly (`scripts/gpu/measure_real_expert_token_counts.py`, wrapping
`local_expert_contribution` to record each layer's real per-local-expert
`bincount`) across the real model's 27 layers, both ranks, the same 3
prompts as the correctness gate (single request each, no batching --
2592 (layer, local-expert) observations per rank):

| | rank 0 | rank 1 |
|---|---|---|
| local experts that received >=1 token | 1289 / 2592 | 1339 / 2592 |
| min tokens (over experts that received any) | 1 | 1 |
| max tokens | 18 | 20 |
| mean tokens | 3.54 | 3.61 |
| median tokens | 2.0 | 2.0 |

Real per-expert counts under this workload sit almost entirely **below**
Phase 1's smallest tested point (16 tokens) -- closer to Phase 1's
single-token decode regime (measured tying) than to its 16-128-token
win region. A follow-up kernel-bench sweep bracketing this real range
directly:

| tokens | naive/layer | persistent/layer | winner |
|---|---|---|---|
| 1 | 0.5514ms | 0.5536ms | naive, by 0.4% |
| 2 | 0.5192ms | 0.5407ms | naive, by 4.1% |
| 4 | 0.5028ms | 0.5235ms | naive, by 4.1% |
| 8 | 0.5271ms | 0.5418ms | naive, by 2.8% |
| 16 | 0.5438ms | 0.6148ms | naive, by 13.1% |
| 20 | 0.5183ms | 0.6160ms | naive, by 18.8% |

(The two independent 16-token measurements above -- 0.560/1.061ms in the
first sweep vs. 0.544/0.615ms here -- disagree by roughly 2x on absolute
latency while agreeing on the ordering; both are single `do_bench` runs
minutes apart on a shared rented card, and this run-to-run spread is
itself part of the honest picture, not smoothed over.)

**Direct answer**: the crossover does not so much "hold" or "shift" as
**not apply** at this project's actual single-request-EP workload scale.
Real per-expert token counts under 2-way EP with short, unbatched
prompts are almost always in the 1-4 range (median 2), an order of
magnitude below where Phase 1 found persistent's 3-8% win on older
hardware -- and on H200 specifically, naive already wins across the
entire range Phase 1 tested, before EP is even in the picture. Both
kernels sit near a shared ~0.5ms latency floor at real EP scale, where
launch/synchronization overhead dominates actual compute -- kernel
choice is close to moot for this workload on this hardware, a stronger
and more specific finding than Phase 3 set out to test, reached from two
independent measurements (a hardware-swap re-run of Phase 1's own
methodology, and a direct measurement of DeepEP's real dispatch shape)
rather than assumed from either alone.

## What was *not* measured

The runbook's plan called for a broader decode/prefill x batch-size x
{naive, persistent} x {single-GPU, 2-GPU EP} grid. That full grid was not
run: once the correctness gate passed and the two measurements above
gave a clear, mutually-reinforcing answer to Phase 3's actual thesis
question well within budget and time, running a larger synthetic grid on
top would have added volume without changing the conclusion -- the real
per-expert distribution is the load-bearing fact here, and it was
measured directly rather than approximated by a synthetic sweep. A
genuine multi-request batched-serving workload (larger, more varied
per-expert counts than a single short prompt) is the natural follow-up
if this question matters again; this run answers it for the
single-request scale this project has scoped to throughout.

## Cost

**Total: $13.03** of the $25 cap, 2x H200 SXM at $9.18/hr, 85 minutes
(pod creation to termination). Full record:
`docs/findings/2026-09-16-phase-3-multi-gpu-ep-cost.md`.
