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
`docs/findings/phase-0/2026-09-14-phase-0-baseline-run.md` for the full account:
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
`docs/findings/phase-0/2026-09-14-phase-0-baseline-results.json` and
`2026-09-15-phase-0-baseline-cost.md` (both committed). The reference
logits (`-reference.safetensors`, Phase 1's correctness oracle) are
**not** committed -- `.gitignore` excludes `*.safetensors` repo-wide by
design; the file exists locally but Phase 1 regenerates it from
`capture_reference_logits` rather than relying on a checked-in blob.
See `docs/findings/phase-0/2026-09-14-phase-0-baseline-run.md` for the full
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
rental. Full account: `docs/findings/phase-1/2026-09-15-phase-1-kernel-correctness.md`.

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
`docs/findings/phase-1/2026-09-15-phase-1-grouped-gemm-run.md`.

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
`docs/findings/phase-2/2026-09-15-phase-2-vllm-benchmark-run.md`.

## Phase 3 progress

Plan: `docs/plans/2026-09-15-phase-3-multi-gpu-expert-parallel-serving-plan.md`.
Design: `docs/design/2026-09-15-phase-3-multi-gpu-expert-parallel-serving.md`.
ADR: `docs/adr/0002-deepep-over-hand-rolled-communication.md`. Branch
`phase-3-multi-gpu-expert-parallel-serving`, pushed as PR #3.

- [x] Task 1: Multi-GPU RunPod provisioning (`gpu_count` on `create_pod`/`provision.py`)
- [x] Task 2: Expert-to-rank sharding, proven correct on CPU (`expert_parallel.py`)
- [x] Task 3: DeepEP dispatch/combine smoke test (`scripts/gpu/deepep_smoke_test.py`) --
      switched from the plan's assumed V2 `ElasticBuffer` to V1 `Buffer` live, see below
- [x] Task 4: GPU rental runbook, real EP MoE layer, correctness gate, benchmark
- [x] Task 5: Findings doc and this update

**Phase 3 is complete.** All 5 tasks done, `make check` green throughout.

**Real hardware corrected the plan on two fronts before any measurement
happened.** RunPod's H100 stock (the design doc's quoted GPU) vanished
on both clouds within a minute of a live catalog check showing it
available -- rented 2x H200 SXM instead (still Hopper-class, still
within the $25 cap). More significantly, DeepEP V2's `ElasticBuffer`
(the plan's assumed API) never worked on this rental: its NCCL Gin
backend needs NVSwitch-level multicast (GPU Fabric Manager), and this
pod's `nvidia-smi -q` reports `GPU Fabric GUID: N/A` with no
`fabricmanager` process -- a rented-container tenancy limitation, not a
code bug. Switched to DeepEP's older V1 (legacy) `Buffer` API, which
uses plain NVLink peer-to-peer memory and worked immediately once its
own two real API differences were handled (float32-only dispatch
weights; `recv_topk_idx` already remapped to local, not global, expert
indices). Two more real bugs surfaced wiring the real EP layer against
CUDA tensors and the real 64-expert model for the first time (a
device-placement bug and an undersized remap table in
`local_expert_contribution`), both fixed with regression tests, neither
caught by the CPU-only tests that preceded real hardware.

**Correctness gate passed**: perfect top-1 agreement and mutual top-5
agreement between the real 2-GPU DeepEP-backed EP path and a single-GPU
reference, across all 3 prompts, on the real
`deepseek-ai/deepseek-moe-16b-base` (27 MoE layers patched both sides).

**The measured answer to Phase 3's thesis** (does Phase 1's
naive-vs-persistent crossover hold, shift, or disappear under DeepEP's
real per-expert token-count distribution): it doesn't apply at this
project's real workload scale. The crossover doesn't reproduce on H200
at all -- naive wins at every token count Phase 1 tested (16-2048),
independent of EP. And DeepEP's real dispatch, measured directly across
the real model's 27 layers, produces per-local-expert token counts far
below Phase 1's tested range (median 2, max ~20, vs. Phase 1's smallest
tested point of 16) -- real single-request EP traffic lands close to
Phase 1's single-token-decode tie, not its 16-128-token win region.
Both kernels sit near a shared ~0.5ms latency floor at that real scale.
Full account, including the exact bugs, fixes, and every number:
`docs/findings/phase-3/2026-09-16-phase-3-multi-gpu-ep-run.md`.

**Total Phase 3 GPU cost: $13.03** of the $25 cap, 2x H200 SXM, 85
minutes.

## Phase 4 progress

Design: `docs/design/2026-09-16-phase-4-disaggregated-prefill-decode.md`.
Plan: `docs/plans/2026-09-16-phase-4-disaggregated-prefill-decode-plan.md`.
Branch `phase-4-disaggregated-prefill-decode`.

- [x] Task 1: KV-cache slice/pad-batch helpers (`src/dispatch/serving/kv_cache.py`),
      grounded in transformers v5's real `DynamicCache` API (confirmed live
      against transformers' own migration guide and source -- `to_legacy_cache`
      etc. were removed in v5)
- [x] Task 2: Cross-rank KV-cache handoff (`src/dispatch/serving/handoff.py`),
      proven over CPU-only `gloo` before any GPU
- [x] Task 3: Continuous-batching prefill/decode workers
      (`src/dispatch/serving/disaggregated.py`) -- found and fixed a real
      scheduler bug (admission and decode sharing one `step()` call could
      over-generate by one token) before ever running on GPU
- [x] Task 4: Co-located baseline worker (`src/dispatch/serving/colocated.py`)
- [x] Task 5: 4x H100 SXM rental, real EP wiring, correctness gate, concurrency measurement
- [x] Task 6: Findings doc and this update

**Phase 4 is complete.** All 6 tasks done, `make check` green throughout
(102 tests total, including the new `test_kv_cache.py`/`test_handoff.py`/
`test_disaggregated.py`/`test_colocated.py`).

**Three real bugs found and fixed on real hardware**, none caught by
CPU-only tests: DeepSeek's remote-code model returns the legacy
tuple-of-tensors cache format (not `DynamicCache`), handled at the
pod-local script boundary; `local_expert_contribution` (Phase 3 code)
crashed when a rank received zero tokens for any local expert at all --
a case 4-way EP's finer sharding made reachable where Phase 3's 2-way EP
never hit it; and `handoff.py`/`kv_cache.py` had two device-placement
bugs (NCCL needs CUDA tensors; `gloo`-only tests never caught it) found
by re-reading the code before running it, not by a live crash. All three
fixed with regression tests.

**Correctness gate passed for both topologies**: exact greedy-token-sequence
match (the strict bar this scheduler's token-only interface actually
admits) against a single-GPU reference, all 3 prompts, for both a
co-located 4-rank EP pool and a disaggregated 2+2-rank EP pool connected
by a real cross-rank KV-cache handoff.

**The measured answer to Phase 4's thesis** (does disaggregation relieve
prefill/decode contention, holding GPU count fixed at 4): mixed, not a
clean result. Disaggregated TTFT beats co-located's at concurrency 4
(0.46s vs 1.12s mean) but loses at concurrency 8 (0.98s vs 0.70s) --
most likely a kernel-warmup confound between independent process
launches rather than a real topology effect (co-located's own wall time
*dropped* going from concurrency 4 to 8, the signature of warmup, not
contention). Reported as genuinely inconclusive rather than forced into
either direction; a warmed-up measurement protocol is the identified
follow-up, not attempted this session per the standing cost-discipline
instruction once both correctness gates had passed. Full account:
`docs/findings/phase-4/2026-09-17-phase-4-disaggregated-prefill-decode-run.md`.

**Total Phase 4 GPU cost: $10.94** of the $40 cap, 4x H100 SXM, 47
minutes.

## Phase 5a progress

Design: `docs/design/2026-09-16-phase-5a-quantization.md`. Plan:
`docs/plans/2026-09-16-phase-5a-quantization-plan.md`. Branch
`phase-5a-quantization`. Split from Phase 5 (quantization +
speculative decoding) into two sub-phases; 5b (speculative decoding) is
separate, later work.

- [x] Task 1: `QuantizedTensor`, per-channel int8 quantize/dequantize (`quantization.py`)
- [x] Task 2: Quantized MoE forward path + memory-footprint helpers
- [x] Task 3: Triton int8 grouped-GEMM kernel + backend resolver (`grouped_gemm_int8.py`, `backends.py`)
- [x] Task 4: `patch_moe_infer_quantized`, sharing layer-iteration with `patch_moe_infer`
- [x] Task 5: `--moe-kernel quantized` wired through `run_baseline.py`
- [x] Task 6: GPU rental runbook -- correctness gate, three-way measured run, memory footprint, cost
- [x] Task 7: this update

**Phase 5a is complete.** `make check` green throughout (119 tests,
lint and `mypy --strict` clean).

**Kernel-level correctness gate passed** on a real NVIDIA L40 (RunPod
Secure Cloud): the int8 kernel's first real-hardware run, 15/15 GPU
tests passed against `torch_grouped_matmul_dequant` (an independent
quantize-then-dequantize-then-eager-matmul reference); the existing bf16
kernels re-verified at 25/25.

**A real bug found and fixed on real hardware, none caught by CPU-only
tests:** `patch_moe_infer_quantized` quantized each layer's stacked bf16
weights into int8 but never released the bf16 originals --
`stack_expert_weights` re-points every expert's `Linear.weight` at a
*view* into one shared bf16 tensor (so building that stack costs no
extra memory in the existing bf16 path), and those views kept the whole
bf16 tensor resident even after quantizing, holding both copies at once
and OOMing a 44GB L40 at DeepSeekMoE-16B's real scale. Fixed with
`_free_expert_weights` (frees each expert's original weight right after
quantizing, since the quantized closure never reads `experts` again) and
a new CPU regression test asserting those weights are actually freed.

**Measured three-way run** (bf16, single L40, unbatched eager decode, 15
runs per config, 27/27 MoE layers patched): stock **12.13 tok/s** ($18.78
per 1M tokens), naive bf16 kernel **20.97 tok/s** (**+72.9%**, $10.86 per
1M tokens -- consistent with Phase 1's separately-measured +67.2% on the
same GPU class), int8 quantized kernel **20.72 tok/s** (**+70.9%** over
stock, but **-1.2%** vs. the naive kernel it's built on -- essentially a
throughput tie, expected: this is a weight-only quantization meant to
save memory bandwidth/footprint, not FLOPs, and the naive kernel already
runs `tl.dot` at native tensor-core precision).

**Memory footprint**, computed directly from the real model's stacked
expert weights across all 27 MoE layers: bf16 **27.84 GiB** -> int8
**13.95 GiB**, a **49.89%** reduction (just under a clean 50% from the
per-channel fp32 scale overhead).

**Model-level agreement** (quantized vs. the naive bf16 run, per the
design doc's two-tier bar): **perfect top-1 agreement and mutual top-k
membership at every tested position**, all 3 prompts -- stronger than
the bar required, which explicitly allowed for real divergence here.

**Two RunPod Community Cloud pods hit the same real host-level bug this
session** (`cuInit()` returning `CUDA_ERROR_UNKNOWN` even against the
base image's own stock torch, confirmed via a raw ctypes test) before a
Secure Cloud L40 worked immediately -- a new environment finding, not
previously documented in this project. `transformers==4.57.6` +
`DynamicCache.get_usable_length` monkeypatch (Phase 0/1/3/4's fix) was
still needed; new this session, `HF_HUB_ENABLE_HF_TRANSFER=1` was set on
the pod with `hf_transfer` not installed.

**Total Phase 5a GPU cost: $1.9543** of the $5 cap (39.1%), across all
three pods including both abandoned Community Cloud attempts. Full
account: `docs/findings/phase-5a/2026-09-17-phase-5a-quantization-run.md`.

## Phase 5b progress

Design: `docs/design/2026-09-17-phase-5b-speculative-decoding.md`. Plan:
`docs/plans/2026-09-17-phase-5b-speculative-decoding-plan.md`. Branch
`phase-5b-speculative-decoding`.

- [x] Task 1: `Drafter` protocol and `PromptLookupDrafter` (`src/dispatch/speculative/drafters.py`)
- [x] Task 2: `DraftModelDrafter`
- [x] Task 3: The shared speculative decode loop (`src/dispatch/speculative/decode.py`)
- [x] Task 4: Exact generated-token reference capture/compare (`src/dispatch/speculative/reference.py`)
- [x] Task 5: CLI (`scripts/run_speculative_bench.py`)
- [x] Task 6: GPU rental runbook -- correctness gates, measured run, k-sweep
- [x] Task 7: this update
- [x] Final whole-branch review fix wave, 2026-09-17: independent
      correctness oracle, runtime cache-lag assertions, a real
      end-to-end draft-model test -- and withdrawal of the GPU session's
      throughput/correctness-gate numbers below (see below)
- [x] Task 8, 2026-09-18: root-cause `fix_rope_inv_freq()` (commit
      `c5b59df`) for the withdrawn baseline's degenerate output; real
      GPU re-run (baseline, both gates, full k-sweep, every point
      checked against the baseline) superseding the withdrawal

**Phase 5b is complete, 2026-09-18.** The prior session's degenerate
baseline (finding C1) is root-caused and fixed: `transformers==5.17.0`
leaves DeepSeek's remote code's RoPE `inv_freq` buffer uninitialized
after `from_pretrained`, poisoning every attention layer with NaN on
GPU. Fixed permanently via `fix_rope_inv_freq()`
(`src/dispatch/kernels/integration.py`, commit `c5b59df`), wired into
`scripts/run_speculative_bench.py` for both the target and any draft
model. `make check` green throughout; lint and `mypy --strict` clean.

**What the prior review found, and what this session's real GPU run
confirmed as root-caused rather than a shared-code artifact**: the
original baseline (`--drafter none`) generated a degenerate output --
64 repetitions of token id `0` -- caused by the uninitialized-`inv_freq`
bug above, not by the shared propose/verify/accept/rollback loop
(`run_speculative_rounds`) itself, which was re-derived by hand this
session and found correct for every candidate-count case. Full account
of the fix, the real re-measured baseline, both correctness gates, and
the full k-sweep (checked at every k this time, closing finding I5):
`docs/findings/phase-5b/2026-09-18-phase-5b-speculative-decoding-run.md`.

**Real, non-degenerate results, 2026-09-18** (A40, bf16, int8 target
kernel, unbatched decode, 64 new tokens, 4 prompts x 5 reps): baseline
25.18 tok/s. draft-model k=4 gate: byte-exact on all 4 prompts, 23.03
tok/s (-8.5% vs. baseline), 74.5% acceptance. prompt-lookup k=4 gate:
byte-exact on 2 of 4 prompts, 47.21 tok/s, 17.7% acceptance; the same
config re-run in the k-sweep matched 3 of 4 (44.74 tok/s, +77.7%) --
same code and flags, different process, different outcome. Across the
8-config k-sweep, draft-model matched the baseline in 13 of 16
(prompt, k) combinations and prompt-lookup in 9 of 16; prompt_000
diverged in 6 of 8 configs (all but k=4), including draft-model at
k=1/2/8. Root-caused live on the rented GPU to a genuine near-tied
logit position under the int8-quantized kernel's actual floating-point
precision -- confirmed by two independent probes (a batch-width sweep
and a 10-trial same-process determinism check) -- not a defect in
`dispatch.speculative`. **Only prompt-lookup beats the baseline**
(34.4-50.9 tok/s across k=1-8); draft-model is below it at every k
(19.6-23.0), despite 2-7x higher acceptance.

**Memory checkpoints** (from the original 2026-09-17 session, not
re-measured this session -- allocator behavior is unrelated to the
token-correctness bug this session fixed): quantized target alone
allocates 16.75GB, target plus bf16 draft model together 29.62GB --
both well within a single GPU's capacity, no OOM at either checkpoint.

**Total Phase 5b GPU cost across both sessions: $12.26 of the $20 cap**
(61.3% -- cap raised from the original $10 mid-session, 2026-09-18,
after a disclosed cost overrun on one pod left running through an
unbounded wait; user ruling: raise to $20 and continue). Per-pod:
`2t6wh9l3okl3gj` (L40, 2026-09-17) $0.6833; `oby7hbkzx8o8yt` (L40)
$10.8848; `5ufi854zxkfwbz` (A40) $0.2022; `d2h1sgmr5wjssv` (A40, where
the real baseline/gates/k-sweep ran) $0.49. All four pods stopped or
terminated.

**Scope of Phase 5a's agreement claim (resolved as an input to Phase 6,
2026-09-18)**: this session's root-cause work found concrete evidence of
near-tie floating-point sensitivity in `grouped_matmul_int8`, the same
quantized kernel Phase 5a's "perfect top-1/mutual-top-k agreement" claim
was measured against. Checked against the record, without new GPU spend:
Phase 5a's GPU session ran on `transformers==4.57.6` (a pod-side
override of this repo's `>=5.17.0` floor), the version the RoPE
`inv_freq` bug was never observed on, and its `max_abs_diff` values
(2.125 / 1.906 / 1.344, one per prompt) are finite and differ per
prompt, so it was not a degenerate-output pass. (That 4.57.6 is
unaffected was inferred from those numbers, not tested directly.) What
does stand: the check covered 29 positions in a single run, quantized
vs. naive bf16 -- too small a sample to rule out the 0.1-0.4 logit-gap
near-ties 5b found. Decision: do not re-rent a GPU just to re-audit 5a;
Phase 6's own correctness gate measures agreement at scale on the exact
config being benchmarked, and 5a's agreement figure is not to be quoted
in the head-to-head as more than "true on a small sample".

## Phase 6 progress

**Design approved, 2026-09-18:**
`docs/design/2026-09-18-phase-6-final-benchmark.md`, on branch
`phase-6-final-benchmark`. No implementation plan or code yet, no GPU
spent. Decided in brainstorming:

- **Kernel-level race plus a labeled engine reference, not an end-to-end
  race.** Design doc §6 assumed a dispatch server for the serving
  benchmark to target; none exists (the engine is an in-process HF eager
  loop), so an end-to-end race would measure that gap, not the kernel.
  §6 is amended to say so. A real server belongs to Phase 7.
- **One L40, ~$10 cap** (same GPU class as Phase 1 and 5a's kernel
  numbers). Kernel race: dispatch naive/persistent vs. vLLM and SGLang
  fused-MoE, uniform and zipf routing, bf16 and weight-only int8. Engine
  reference: vLLM and SGLang serving the full model, concurrency 1/4/16/64,
  dispatch at concurrency 1 only.
- **Checked live 2026-09-18:** SGLang serves `DeepseekForCausalLM` (V1);
  both engines' fused-MoE accept `use_int8_w8a16` + `per_channel_quant`;
  SGLang's fused-MoE code was recently reorganized, so engine versions get
  pinned and each engine sits behind one adapter.
- **Stage 1 blocks the rest:** the at-scale correctness gate STATUS owes
  from 5a/5b, with a large-gap disagreement threshold fixed in the plan
  from 5b's near-tie data before any GPU time.

**Implementation plan written, 2026-09-19:**
`docs/plans/2026-09-19-phase-6-final-benchmark-plan.md`, 15 tasks. Its CPU-testable
code (gap-split classifier, gate prompts, engine adapters, race driver,
summarizer, serving-benchmark helpers) was built and verified in a scratch copy
of the repo before being written into the plan: 236 tests, ruff and
`mypy --strict` clean. Only what the plan says is verified was verified; the real
vLLM and SGLang engines have **not** been exercised (neither installs on this
machine) -- that is Task 10's first job on the pod.

- **Gate rule pre-registered in the plan, before any GPU time:** a top-1
  disagreement fails only where the reference's top-1/top-2 logit gap exceeds
  1.0 (2.5x Phase 5b's widest measured near-tie of 0.4); minimum 500 compared
  positions (the 16 fixed prompts give 1,036 with the real tokenizer); a
  stock-vs-stock control measures the gate's own noise floor; a failing config
  is excluded from the race rather than halting the session. A test pins the
  threshold so it cannot be loosened after seeing a run.
- **Design assumptions the plan's live checks corrected (2026-09-19):** vllm
  0.29.0 and sglang 0.5.20 both pin `torch==2.13.0` with compatible transformers
  pins, so one shared engines venv is possible (the design assumed conflicting
  pins) and all contestants can share one Triton compiler; SGLang's own tuner
  does not list V1's `DeepseekForCausalLM`, so the plan tunes it through a
  config-only architecture shim, with a bounded fallback to untuned; the
  per-engine bare-GEMM diagnostic is dropped (the fused paths expose no separable
  single-GEMM entry point). Design amended accordingly.
- **Tuning rule pre-registered:** every contestant is tuned under uniform routing
  only, then timed under both distributions; dispatch's tuned tile size is picked
  on uniform results and reused for zipf (a test asserts the choice never looks
  at zipf).

- **Task 1 (gap-split classifier)**: dispatch.benchmark.agreement splits every top-1 disagreement by the reference's top1-top2 logit gap; threshold 1.0 and 500-position floor pre-registered in the plan and pinned by a test.

- **Task 2 (gate prompt set + run_baseline gate mode)**: --prompt-set gate runs 16 fixed prompts (1,036 positions with the real tokenizer) and fails on any large-gap flip or fewer than 500 positions.

- **Task 3 (ignore_eos, public percentile)**: the harness can generate exactly N tokens past EOS, matching vllm bench serve --ignore-eos, so dispatch's concurrency-1 reference row is comparable.

- **Task 4 (engine contract + dispatch adapter)**: seeded inputs, fp32 reference, fused gate+up layout, and dispatch's kernels behind one MoEEngine contract; uniform and zipf cases share x and weights so only routing differs.

- **Task 5 (vLLM/SGLang adapters + registry)**: each engine behind one adapter calling its own fused_experts with fixed routing; verified against eager fakes only -- the real engines are not exercised until the pod (Task 10).

## Next step

Phase 5b is done: root cause fixed, real numbers measured, both
correctness gates and the full k-sweep checked and explained. Phase 4 is
merged (PR #4), as are Phase 5a (PR #5) and Phase 5b (PR #6), each on its
own branch per the one-branch-per-phase convention. Phase 3's PR (#3) also
merged.
Phase 2's PR is still open and awaiting maintainer review; no further
work planned on it beyond responding to review feedback.

Phase 6's design is approved and its implementation plan is written
(see "Phase 6 progress"). Next: execute
`docs/plans/2026-09-19-phase-6-final-benchmark-plan.md` task by task.
Tasks 1-9 are local, free and TDD; Tasks 10-14 are the paid L40 session and
need the user's explicit go-ahead (live price stated first, $10 cap).
