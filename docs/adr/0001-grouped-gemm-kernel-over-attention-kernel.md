# ADR-0001: Grouped-GEMM expert kernel over a from-scratch attention kernel

**Status:** Accepted 2026-09-14.

## Context

The project needs one genuinely hard, hand-written GPU kernel as its
central differentiator (see `docs/design/2026-09-14-dispatch-system-design.md`
§1). The obvious first instinct — "write Flash Attention from scratch" —
was checked against what already exists before being adopted.

## What was checked first

A web search for existing "flash attention from scratch" repositories
turned up a dozen-plus GitHub projects, several achieving 99%+ of official
Flash Attention 2 performance through iterative optimization, plus multiple
tutorial repos aimed explicitly at CUDA beginners. This is a saturated,
well-trodden portfolio pattern, not a differentiator.

The same research pass found the opposite for MoE's grouped-GEMM: current
(2026) arXiv papers and a PyTorch engineering blog post actively describing
it as an open, current problem — skewed per-expert token counts make naive
per-expert GEMM launches starve small GEMMs of GPU occupancy, and the fix
(persistent, cache-aware kernels) is recent, documented work rather than a
settled tutorial topic.

## Decision

The custom kernel targets the MoE grouped-GEMM expert computation, not
attention. Attention stays whatever the reference implementation (HF
`transformers`) already provides.

## Alternatives considered

**Flash Attention from scratch.** Rejected specifically because it's
saturated (see above) — building it well would prove CUDA/Triton
competence but not differentiate this project from the many existing
"from scratch" repos already doing exactly that, some at near-production
performance.

**A fused attention + MoE kernel.** Considered and deferred rather than
rejected outright — combining both would be more ambitious, but grouped-
GEMM alone is already the harder, less-covered problem and the scope this
project's timeline (6+ weeks) and budget can support without diluting
effort across two hard kernels at once.
