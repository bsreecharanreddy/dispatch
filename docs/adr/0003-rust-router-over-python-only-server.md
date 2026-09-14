# ADR-0003: Rust router (TGI-style) over an all-Python server

**Status:** Accepted 2026-09-14.

## Context

Phase 7 (`docs/design/2026-09-14-dispatch-system-design.md` §8) adds a real
serving layer with Docker/K8s/observability, prompted by checking actual
inference-engineer job postings (national and Atlanta-metro) against the
original design and finding a real gap: containerized, production-shaped
deployment. That raised the question directly: should the serving layer be
Python (matching the rest of the project) or Rust, given how much current
attention Rust gets in ML-infra discussion.

## What was checked first

Not just "is Rust popular" — specifically, what production inference
systems actually do. Hugging Face's own `text-generation-inference` (TGI),
which powers HuggingChat and their Inference Endpoints in production, is
documented as a three-tier architecture: a Rust Router (HTTP-facing,
request validation, queuing, continuous batching) talking gRPC to a Python
Server (model loading and execution). This is not a hypothetical pattern —
it's what a real company runs today, checked against HF's own architecture
docs rather than a blog post's opinion.

The reasoning behind that split, also checked rather than assumed: Python's
GIL caps true request-level parallelism in a server process; Rust has none,
plus no interpreter overhead and no GC pauses, which matters most for
request-handling latency (especially at low batch sizes) — the "last mile"
of serving, not the GPU-bound model computation itself. The research was
explicit that the win is not universal: it shows up most in request/serving
overhead, much less once GPU compute dominates at large batch sizes. Job
postings list Rust as one of Python/C++/Rust/Go, not a hard requirement.

## Decision

Split the serving layer the way TGI does: a Rust router (HTTP/gRPC,
request queuing/batching, Prometheus metrics) in front of the existing
Python model server (unchanged from phases 0-6 — router/gating, the custom
Triton kernel, DeepEP). Rust is scoped to the router only.

## Alternatives considered

**All-Python server (e.g., FastAPI + Uvicorn).** Simpler, zero new
language to learn, and legitimate — Python is explicitly an accepted
language in the postings this decision was checked against. Rejected in
favor of Rust specifically because Phase 7 exists to close a job-posting
gap, and a second real, production-precedented skill (Rust, in exactly the
place real companies use it) closes that gap further than staying in
Python would, for one additional phase's worth of scope.

**Rewriting the model server itself in Rust (e.g., on Candle).** Rejected:
disproportionate scope for the signal gained — the differentiated work in
this project is the grouped-GEMM kernel (ADR-0001) and the multi-GPU
serving (ADR-0002), not reimplementing PyTorch's ecosystem. Candle is real
and used in production (`mistral.rs`), but adopting it here would dilute
focus across three hard problems instead of two, for a project already
carrying real scope.

## Consequences

Phase 7 now has a real, non-trivial skill-acquisition cost if Rust is new
(see design doc §12, "open risks") — budgeted as calendar time, not assumed
free. In exchange, the finished project demonstrates a second,
complementary, production-precedented skill (systems-level Rust) in
addition to GPU kernel authorship, rather than only the latter.
