# dispatch — Status

Authoritative record of where implementation stands against
`docs/design/2026-09-14-dispatch-system-design.md`. Updated **in the same
commit as the work it describes**, never as a follow-up.

## Current position

**System design at its second pass, 2026-09-14.** Design doc
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

- [x] Task 1: RunPod API client (`scripts/gpu/runpod_client.py`)
- [x] Task 2: Pod-wait orchestration + cost-record logger (`scripts/gpu/provision.py`)
- [x] Task 3: Provisioning CLI (`scripts/gpu/provision.py` `main()`) -- landed in the same commit as Task 2 (both touch `provision.py` and were written/tested together before the first commit of either)
- [x] Task 4: Pure benchmark metrics (`src/dispatch/benchmark/metrics.py`)
- [x] Task 5: Generation harness (`src/dispatch/benchmark/harness.py`)
- [x] Task 6: Reference-logit capture + tolerance compare (`src/dispatch/benchmark/reference.py`)
- [x] Task 7: Baseline CLI (`scripts/run_baseline.py`)
- [ ] Task 8: Real rented-GPU run -- runbook executed, results + reference + cost recorded

All of Phase 0's tooling (Tasks 1-7) is built and tested: RunPod
provisioning, pod-wait orchestration, cost logging, the pure metrics math,
the model-loading/timed-generation harness, reference-logit capture, and
the baseline CLI that ties them together. `make check` is green -- 26
tests, lint and `mypy --strict` clean, including a real (network, CPU,
no GPU) pass against `hf-internal-testing/tiny-random-gpt2` to prove the
harness and reference capture actually work end to end. One real bug was
caught by TDD along the way: `TokenTimings.inter_token_latencies` used
`zip(..., strict=True)` over two sequences of different length by
construction (`token_times` and `token_times[1:]`), which raises rather
than pairwise-zips -- fixed to `strict=False` before the first commit
touching it.

Only Task 8 remains: the one real run against the full
deepseek-ai/deepseek-moe-16b-base model on a rented GPU, per
`docs/runbooks/phase-0-baseline.md` (not yet written).

## Next step

Write Task 8's runbook (`docs/runbooks/phase-0-baseline.md`), then execute
it with the user's explicit go-ahead and a stated budget cap -- it is the
only task in this phase that spends money.
