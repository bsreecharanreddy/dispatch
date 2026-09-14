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

## Phase 0 progress

Implementation plan written and reviewed:
`docs/plans/2026-09-14-phase-0-baseline-plan.md` (8 tasks: RunPod API
client, pod-wait orchestration + cost logging, provisioning CLI, pure
benchmark metrics, generation harness, reference-logit capture, baseline
CLI, then the one real rented-GPU run). Work is happening on the
`phase-0-baseline` branch per this repo's one-branch-per-phase convention
-- pushed as a single PR once the phase is done, not before.

- [ ] Task 1: RunPod API client (`scripts/gpu/runpod_client.py`)
- [ ] Task 2: Pod-wait orchestration + cost-record logger (`scripts/gpu/provision.py`)
- [ ] Task 3: Provisioning CLI (`scripts/gpu/provision.py` `main()`)
- [ ] Task 4: Pure benchmark metrics (`src/dispatch/benchmark/metrics.py`)
- [ ] Task 5: Generation harness (`src/dispatch/benchmark/harness.py`)
- [ ] Task 6: Reference-logit capture + tolerance compare (`src/dispatch/benchmark/reference.py`)
- [ ] Task 7: Baseline CLI (`scripts/run_baseline.py`)
- [ ] Task 8: Real rented-GPU run -- runbook executed, results + reference + cost recorded

## Next step

Execute the plan task by task (TDD, one commit per task, this checklist
updated in the same commit as each). Task 8 is gated on the user's
explicit go-ahead and a stated budget cap -- it is the only task that
spends money.
