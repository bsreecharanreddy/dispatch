# ADR-0002: DeepEP over hand-rolled cross-GPU expert-parallel communication

**Status:** Accepted 2026-09-14. Revisit at Phase 3 against actually-rented
hardware — see "Open question" below.

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
