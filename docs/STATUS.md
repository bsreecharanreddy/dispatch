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
- [x] Task 7: Backend registry + kernel micro-benchmark CLI (`backends.py`, `bench.py`, `scripts/run_kernel_bench.py`)
- [x] Task 8: Real-model integration (`--moe-kernel`, `--compare-reference`)
- [x] Task 9: Measured run on an L40 (runbook session B)

**Phase 1 is complete.** All 9 tasks done, `make check` green throughout
(73 tests, lint and `mypy --strict` clean).

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

Task 7 adds `dispatch.kernels.backends` (`BACKENDS = ("torch", "naive",
"persistent")` and `resolve_backend`, which imports the Triton kernel
module lazily so the registry itself stays importable without triton) and
`dispatch.kernels.bench` (pure-math helpers -- FLOPs, latency-summary
dataclass, synthetic zipf/uniform routing -- plus `time_grouped_gemm`,
the one function that imports `triton.testing` locally). `scripts/run_kernel_bench.py`
is the CLI: times every backend's whole routed-MoE layer and its
gate_proj-shaped grouped GEMM alone on synthetic DeepSeekMoE-16B-shaped
work, refuses to time a backend that disagrees with the eager torch
backend on the benchmark's own input, and writes config + results JSON to
`docs/findings/`. All CPU-testable (11 new tests, 62 passed + 1 skipped
total); the real timed run happens in Task 9's runbook, on a GPU host.

Task 8 adds `dispatch.kernels.integration` (`patch_moe_infer`, which walks
a loaded model's modules and replaces each MoE layer's `moe_infer` --
DeepSeek's own remote-code inference method -- with a closure over
`grouped_moe_routed` and a stacked-weights view of that layer's experts;
rejects an `experts` attribute that isn't an `nn.ModuleList`) and, in
`dispatch.benchmark.reference`, `TopKAgreement`/`compare_top_k_agreement`
-- a kernel-swap-shaped comparison (top-1 agreement, mutual top-k
membership, max abs diff) that tolerates a near-tie flip but flags an
argmax that lands outside the other side's top-k. `compare_within_tolerance`
was refactored to share the key-mismatch check via a new `_require_same_keys`
helper, with no behavior change. `scripts/run_baseline.py` gained
`--moe-kernel` (patches the resolved backend into the model before timing;
raises if it patches zero layers, so a kernel run can never silently time
the stock model) and `--compare-reference` (compares the run's logits
against a stock run's file and exits non-zero on disagreement, but only
*after* the results JSON and reference safetensors are written to disk).
The `FakeDeepseekMoE` test harness in `test_integration.py` transcribes
DeepSeek's actual `moe_infer` body (only numpy's cumsum swapped for
torch's) so the patched path is proven against DeepSeek's real inference
code, not a paraphrase. 17 new/changed tests across
`test_integration.py`, `test_reference.py`, and `test_run_baseline.py` (3
of them `slow`, downloading `hf-internal-testing/tiny-random-gpt2` from
Hugging Face Hub); full non-GPU suite was 72 passed + 1 skipped at
Task 8's initial commit (the skip is the pre-existing Triton-is-Linux-only
guard in `test_grouped_gemm_kernel.py`, unrelated to this task; the
72 -> 73 in the count above is one more test added during Task 8's own
review-fix rounds, below). Task 8's review found four Important
hardening gaps in this exact code path (fixed
across two rounds before Task 9 ran on it): evidence written before a
bad `--compare-reference` could raise and lose it; `patch_moe_infer`
forcing eval mode so a training-mode model can't report a nonzero
patched count while never executing the patched path; a corrected
bidirectional test for `mutual_top_k`; and stale "Phase 0" defaults on
what is now a multi-phase CLI.

**Task 9's measured run** (2026-09-15, NVIDIA L40, RunPod Secure Cloud,
$0.82/hr, $0.59 total for 43.3 minutes -- same GPU class as Phase 0, so
the comparison holds): both kernels re-verified correct on this card
(25/25, 0 skipped), then swapped into `deepseek-ai/deepseek-moe-16b-base`'s
real 27 MoE layers. Stock throughput **12.55 tokens/sec** (consistent
with Phase 0's separately-measured 12.75 tokens/sec on the same config --
a cross-session sanity check). Kernel-backed: **naive 20.98 tokens/sec
(+67.2%)**, **persistent 20.80 tokens/sec (+65.7%)**, both at perfect
mutual top-5 and top-1 logit agreement across every tested position (3
prompts x 5 repetitions x 64 new tokens, bf16). Cost per 1M generated
tokens: $18.15 (stock) -> $10.86 (naive). A token-count micro-benchmark
(1-2048 tokens, zipf and uniform routing) found the persistent kernel
**ties naive at the 1-token/step granularity that drives decode
throughput** -- a null result the plan's own risk section predicted
before the run happened (unbatched decode gives each expert too few rows
for L2 tile reuse to matter) -- but wins 3-8% at 16-128 tokens and loses
by up to 14% at 512-2048, a workload-specific result recorded rather than
buried. Full account:
`docs/findings/2026-09-15-phase-1-grouped-gemm-run.md`.

**Total Phase 1 GPU cost: $0.65** across both paid sessions ($0.0606
kernel correctness + $0.5916 the measured run) -- both well under their
stated caps ($3 and $5 respectively).

**Phase 1 merged to `main`, 2026-09-15** (PR #2).

## Phase 2 progress

Plan: `docs/plans/2026-09-15-phase-2-vllm-benchmark-contribution-plan.md`.
Design: `docs/design/2026-09-15-phase-2-vllm-benchmark-contribution.md`.
Research found neither vLLM's nor SGLang's official MoE benchmark/tuning
harness models skewed (zipf) expert load -- both generate gating logits
near-uniformly only. Ported Phase 1's zipf-vs-uniform load idea into
vLLM's `benchmarks/kernels/benchmark_moe.py` as an opt-in
`--expert-load-distribution {uniform,zipf}` flag (default `uniform`,
existing behavior unchanged).

While opening what was originally planned as a separate prerequisite PR
(a `get_model_params` fix so the script recognizes
`deepseek-ai/deepseek-moe-16b-base`'s `DeepseekForCausalLM` architecture
string, which it didn't), discovered `vllm-project/vllm`'s own `AGENTS.md`
governing AI-assisted contributions -- closed that standalone PR
(non-compliant: no AI-assistance disclosure, and their own policy
discourages one-off tiny-edit PRs) and re-opened as a single PR with the
fix folded in as its first commit, fully AGENTS.md-compliant this time
(AI-assistance disclosure, duplicate-work check, test results, all in
the PR body).

**Open PR: `vllm-project/vllm#57100`** ("[Benchmark] Add skewed (zipf)
expert-load coverage to benchmark_moe.py"). Measured on one rented RTX
3090: `--tune` under uniform vs. zipf routing picks a **different winning
Triton config at 4 of 5 tested batch sizes** (1/2/4/8/16, E=64, topk=6) --
real, moderate divergence, not dramatic, reported at that strength.

**Total Phase 2 GPU cost: $0.77**, against a $3 cap. Full account,
including three real environment issues found and fixed (a failed
precompiled-wheel install, a local-package-shadowing bug affecting Ray
workers, and RunPod's proxy-only SSH access for this pod) and every
deviation from the plan and why:
`docs/findings/2026-09-15-phase-2-vllm-benchmark-run.md`.

## Next step

Phase 2's PR is open and awaiting maintainer review. No further work is
planned on it beyond responding to review feedback. Phase 3 is not yet
planned.
