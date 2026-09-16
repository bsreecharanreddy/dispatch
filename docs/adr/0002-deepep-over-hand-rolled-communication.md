# ADR-0002: DeepEP over hand-rolled cross-GPU expert-parallel communication

**Status:** Accepted 2026-09-14. Open question resolved 2026-09-15 during
Phase 3 design -- see "Resolution" below.

## Context

Phase 3 (`docs/design/2026-09-14-dispatch-system-design.md` §5) needs a way
to move tokens between GPUs so each expert only runs on the GPU(s) that
hold it. This is a real, hard, separate problem from the grouped-GEMM
kernel work (ADR-0001) — it's low-level GPU-to-GPU networking, not expert
computation.

## What was checked first

DeepSeek open-sourced DeepEP (MIT license) specifically for this: a
high-throughput, low-latency all-to-all communication library for MoE
dispatch/combine, with FP8 support, used in DeepSeek's own production
stack. It is the standard tool for this exact problem as of 2026, not a
toy or abandoned project — actively maintained, at v2 with a NCCL Gin
backend.

The same research pass found a real constraint, not just a recommendation:
DeepEP's fast NVSHMEM path needs NVLink (SXM-form-factor GPUs) within a
node, or InfiniBand across nodes. A plain PCIe multi-GPU rental — the kind
the cheap marketplace tier (Vast.ai, RunPod Community Cloud) usually
offers — does not get DeepEP's real performance.

## Decision

Use DeepEP for cross-GPU dispatch/combine rather than hand-rolling
NVSHMEM/RDMA-style all-to-all communication. Accept that Phase 3 needs a
short, deliberate, higher-cost burst on dedicated NVLink/SXM multi-GPU
hardware (RunPod Secure Cloud or Lambda, not marketplace spot) rather than
running on the same cheap instances the rest of the project uses — priced
and time-boxed as its own line item in the cost plan.

## Alternatives considered

**Hand-rolling all-to-all communication.** Rejected: this is a separate,
extremely hard networking problem (the kind DeepSeek needed a dedicated
library and paper for), not the skill this project is trying to
demonstrate, and attempting it would consume time and budget better spent
on the grouped-GEMM kernel itself.

**Running expert-parallel dispatch on plain PCIe GPUs without DeepEP's fast
path.** Not rejected outright — DeepEP still functions without NVLink, just
without its NVSHMEM fast path. Left as a possible cheaper fallback if the
NVLink/SXM rental line item proves impractical when Phase 3 actually
starts, rather than committed to now.

**UCCL-EP.** A newer expert-parallelism library surfaced in the same
research pass, potentially more cloud/PCIe-flexible than DeepEP. Not
adopted now because it wasn't evaluated in depth — noted as the first
thing to check against real rented hardware when Phase 3 starts, per this
project's design doc §11.

## Open question

Whether DeepEP's NVLink requirement or its cost forces a switch to UCCL-EP
or the no-fast-path fallback is a Phase 3 decision, made against real
rented hardware and real prices at that time — not resolved here.

## Resolution (2026-09-15)

Checked live (`deepseek-ai/DeepEP`'s current README, `uccl-project/uccl`'s
`ep/README.md`, and RunPod's real GPU catalog) during Phase 3 design.
**DeepEP confirmed, UCCL-EP rejected** — and a real constraint neither
this ADR nor the system design doc had: DeepEP's now-current V2 release
requires **Hopper (SM90) GPUs specifically** (or newer, e.g. SM100), not
just "NVLink" generally — Ampere SXM (A100) no longer qualifies, V2
dropped it. V2 also switched its primary backend from NVSHMEM to a
lighter-weight **NCCL Gin** backend ("header-only... reuse existing NCCL
communicators"); NVSHMEM is now legacy-only. Net effect: DeepEP's install
risk is lower than this ADR originally assumed, not higher.

UCCL-EP, by contrast, is built for heterogeneous multi-node clusters with
RDMA NICs (EFA/InfiniBand) — its build needs NIC-specific kernel modules
(`nvidia_peermem`/`efa_nv_peermem`), and every one of its documented
benchmarks is an 8-GPU node or larger. It solves a real problem (running
EP across mixed Nvidia/AMD hardware and NICs), just not this project's
problem: a single node, two homogeneous GPUs, no RDMA fabric at all. It
would add build complexity here without buying anything back.

Real RunPod pricing, checked live: 2x Hopper-class GPU (H100 NVL on
Community cloud, ~$5.18/hr combined; H100 SXM on Secure cloud, ~$6.98/hr
combined) still fits comfortably inside Phase 3's $25 cap for a
multi-hour session — the cost concern this ADR raised does not force a
fallback. The design doc's hardware target is corrected from generic
"NVLink-connected pair" to Hopper-class (H100/H200) specifically, with a
live `nvidia-smi topo -m` check that the two rented GPUs are actually
NVLink-connected (not just co-located) before proceeding — see
`docs/design/2026-09-15-phase-3-multi-gpu-expert-parallel-serving.md` §7.
