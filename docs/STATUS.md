# dispatch — Status

Authoritative record of where implementation stands against
`docs/design/2026-09-14-dispatch-system-design.md`. Updated **in the same
commit as the work it describes**, never as a follow-up.

## Current position

**System design written, 2026-09-14. Still no code.** Design doc at
`docs/design/2026-09-14-dispatch-system-design.md`: model choice
(deepseek-ai/deepseek-moe-16b-base), architecture (custom Triton
grouped-GEMM kernel + DeepEP for cross-GPU dispatch + standard benchmark
tooling against vLLM/SGLang), a 7-phase plan, cost plan, and explicit scope
boundaries. Two ADRs recorded alongside it:

- `docs/adr/0001-grouped-gemm-kernel-over-attention-kernel.md` — why the
  kernel targets MoE grouped-GEMM and not another flash-attention
  reimplementation (that pattern is saturated; checked against actual
  GitHub search results before deciding, not assumed).
- `docs/adr/0002-deepep-over-hand-rolled-communication.md` — why cross-GPU
  expert dispatch uses DeepSeek's DeepEP rather than hand-rolled
  communication, and the real NVLink/SXM hardware constraint that decision
  carries into Phase 3's cost plan.

Repo scaffolding (tooling, conventions, licensing) landed in the previous
commit — see CLAUDE.md for what's carried over from
[almanac](https://github.com/bsreecharanreddy/almanac)/
[canopica](https://github.com/bsreecharanreddy/canopica) and what's
deliberately not (project-specific incident-derived skills).

**Next step**: write the Phase 0 implementation plan in `docs/plans/`
(baseline: DeepSeekMoE-16B on one rented GPU, measured latency/throughput
as the correctness reference), then GPU provisioning scripts in
`scripts/gpu/`, before any model-serving code lands.
