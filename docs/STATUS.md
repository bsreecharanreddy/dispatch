# dispatch — Status

Authoritative record of where implementation stands against
`docs/design/2026-09-14-dispatch-system-design.md`. Updated **in the same
commit as the work it describes**, never as a follow-up.

## Current position

**System design at its second pass, 2026-09-14. Still no code.** Design doc
covers: model choice (deepseek-ai/deepseek-moe-16b-base), architecture
(custom Triton grouped-GEMM kernel + DeepEP for cross-GPU dispatch +
standard benchmark tooling against vLLM/SGLang), an 8-phase plan (0-7), cost
plan, and explicit scope boundaries. Three ADRs recorded:

- `docs/adr/0001` — grouped-GEMM kernel over another flash-attention
  reimplementation (that pattern is saturated on GitHub; checked before
  deciding).
- `docs/adr/0002` — DeepEP over hand-rolled cross-GPU communication, and
  the real NVLink/SXM hardware constraint that carries into Phase 3's cost.
- `docs/adr/0003` — a Rust router in front of the existing Python model
  server for Phase 7 (mirrors Hugging Face's own `text-generation-inference`
  three-tier architecture), added after checking real inference-engineer
  postings (national and Atlanta-metro) against the original design and
  finding real gaps: containerization, K8s deployment, live observability.
  Rust is scoped to the router only — the model server, kernel, and
  multi-GPU work all stay Python/Triton.

Phase 7 (productionization: Rust router, Docker, Prometheus/Grafana
observability, a K8s deployment demoed once) was added specifically because
it closes gaps found by checking the design against real job postings
rather than assuming coverage.

## Next step

Write the Phase 0 implementation plan in `docs/plans/` (baseline:
DeepSeekMoE-16B on one rented GPU, measured latency/throughput as the
correctness reference), then GPU provisioning scripts in `scripts/gpu/`,
before any model-serving code lands.
