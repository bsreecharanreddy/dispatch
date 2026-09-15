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
- [x] Task 8: real rented-GPU run executed -- results, reference logits, and
      cost recorded in `docs/findings/`

**Phase 0 is complete.** All 8 tasks done, `make check` green throughout
(26 tests, lint and `mypy --strict` clean). One real bug was caught by TDD
along the way: `TokenTimings.inter_token_latencies` used `zip(...,
strict=True)` over two sequences of different length by construction
(`token_times` and `token_times[1:]`), which raises rather than
pairwise-zips -- fixed to `strict=False` before the first commit touching
it.

**Task 8's real run found three more bugs, all in the environment rather
than in `dispatch`'s own code** -- see
`docs/findings/2026-09-14-phase-0-baseline-run.md` for the full account:
DeepSeek's `trust_remote_code` modeling file calling a `transformers`
utility (`is_torch_fx_available`) removed entirely by transformers 5.17.0
(this repo's pinned floor); the same file calling a `Cache` method
(`get_usable_length`) already renamed to `get_seq_length` even one release
before that; and the 32.8GB model download defaulting into the pod's 30GB
ephemeral container disk rather than the 50GB+ persistent volume. Fixed
with a transformers 4.57.6 override plus a narrow, pod-local (never
committed) monkeypatch, and `HF_HOME` redirected to `/workspace`.

**Measured baseline** for `deepseek-ai/deepseek-moe-16b-base`, bf16,
single NVIDIA L40 (46GB usable), unbatched eager-mode token-by-token
decode, 15 runs (3 prompts x 5 repetitions, 64 max new tokens):
**12.75 tokens/sec mean throughput, 0.355s mean time-to-first-token**
(p50 0.258s, p99 1.588s). Cost: **$0.35** for 25.7 minutes of L40 rental
(RunPod Secure Cloud, $0.82/hr -- Community's $0.69/hr L40 was out of
stock at deploy time), all three failed attempts included since the model
was already cached locally by the second one. Pod verified `TERMINATED`
independently after deletion. Full record:
`docs/findings/2026-09-14-phase-0-baseline-results.json` and
`2026-09-15-phase-0-baseline-cost.md` (both committed). The reference
logits (`-reference.safetensors`, Phase 1's correctness oracle) are
**not** committed -- `.gitignore` excludes `*.safetensors` repo-wide by
design; the file exists locally but Phase 1 regenerates it from
`capture_reference_logits` rather than relying on a checked-in blob.
See `docs/findings/2026-09-14-phase-0-baseline-run.md` for the full
account.

## Phase 1 progress

Plan: `docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md`. Branch
`phase-1-grouped-gemm`, pushed once as a single PR when the phase is done.

- [x] Task 1: Pure-PyTorch MoE reference (`src/dispatch/kernels/reference_moe.py`)
- [x] Task 2: Token grouping + tile schedule (`grouping.py`, `tile_schedule.py`)
- [x] Task 3: Eager grouped MoE path -- the grouped-GEMM contract (`moe_forward.py`)
- [x] Task 4: Naive Triton grouped-GEMM kernel -- GPU-verified in Task 6
- [x] Task 5: Persistent, cache-aware kernel -- GPU-verified in Task 6
- [x] Task 6: Kernel correctness session on a rented GPU (runbook session A)
- [ ] Task 7: Backend registry + kernel micro-benchmark CLI
- [ ] Task 8: Real-model integration (`--moe-kernel`, `--compare-reference`)
- [ ] Task 9: Measured run on an L40 (runbook session B)

Task 6's rented-GPU session (2026-09-15, RTX 3090 substituted for the
originally-quoted RTX A4000, which sold out at deploy time): both the
naive and persistent Triton kernels passed **25/25** GPU correctness
tests on the first real execution -- no kernel bugs found, an unusually
clean outcome. One real, non-kernel bug found and fixed: a
`requires_grad`-related warning in `assert_matches_reference`
(`moe_forward.py`), first surfaced by a real gradient-carrying tensor on
a real device. A mutation check (forcing every tile to read expert 0's
weights) turned 24/25 tests red, confirmed the revert was clean, and
reran green -- the suite can fail. Cost: **$0.0606** for 991s of RTX 3090
rental. Full account: `docs/findings/2026-09-15-phase-1-kernel-correctness.md`.

## Next step

Task 7 of `docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md`.
