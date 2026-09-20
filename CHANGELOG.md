# Changelog

Phase-level history. `docs/STATUS.md` carries the task-level verification
log; this is the summary a reader wants first.

Every number here is measured. Where one is later found wrong, it gets
corrected **in place with the original kept**, because a retraction that
deletes its own evidence is not a retraction.

## Unreleased

No phase shipped yet (`v0.1.0` is still the scaffold version; a tag lands
once a phase's exit criteria are actually met).

- **Phase 6 (final benchmark vs. vLLM and SGLang) complete**, 2026-09-20,
  branch `phase-6-final-benchmark` (PR pending)
  (`docs/plans/2026-09-19-phase-6-final-benchmark-plan.md`). Ran the
  at-scale correctness gate Phase 5b flagged as owed: naive, persistent
  and int8 all agree with the stock reference at 1,036 positions
  (97.1-97.3% top-1 agreement, zero large-gap disagreements against a
  pre-registered 1.0-logit threshold), superseding Phase 5a's 29-position
  claim for these configs — `make check` green throughout (262 tests,
  after a whole-branch review found and fixed 20 real issues in this
  session's own new code, none affecting the measured GPU results;
  `docs/STATUS.md`, "Task 15 step 5").
  **The kernel race is a real, disclosed loss**: vLLM 0.29.0's and SGLang
  0.5.20's own fused-MoE beat dispatch's kernels at their shipped default
  config at nearly every tested shape, bf16 and int8 alike, on the
  identical GPU, model, and `torch==2.13.0`/`triton==3.7.1` build every
  contestant shared — dispatch's naive kernel wins only at int8, 128 and
  512 tokens. vLLM and SGLang both served the real model correctly across
  concurrency 1/4/16/64 (vLLM ~6-9% ahead of SGLang throughout, e.g.
  1269.2 vs. 1180.4 output tok/s at concurrency 64); dispatch has no
  served, batched engine of its own to place in that same ranked table
  (design doc §6 amended), so its concurrency-1 numbers (naive: 21.04
  tok/s, ~1.8x stock's 11.87, consistent with Phase 1) are reported
  alongside it, labeled non-comparable. Full tuning of vLLM's and
  SGLang's own MoE tuners was not run: a live check found each takes
  ~18 minutes per token count and saves its config only after every
  requested count finishes (confirmed by starting a real run, watching it
  reach 29% of the first of 7 batch sizes after 5 minutes, and killing it
  for zero usable output), so the full matrix would have cost several
  times the phase's budget; both engines are reported at default
  configuration only, a disclosed limitation decided with the user at a
  $4.22 checkpoint. Six real fixes shipped mid-session: a
  `DeepseekForCausalLM`-to-`DeepseekV2ForCausalLM` config shim both
  tuners needed (neither recognizes the model's real V1 architecture), a
  published `ServerArgs` SGLang's `fused_experts` requires
  (`config namespace 'exec' not published`), a repeatability check
  widened for `index_add_`'s non-bitwise-deterministic `atomicAdd` on
  CUDA, a derived end-to-end latency for vLLM 0.29.0's `--save-detailed`
  schema (which carries no per-request latency field at all), a
  subprocess `PATH`-order fix for launching SGLang's server, and a
  self-matching `pkill -f` pattern that had been silently killing its own
  SSH session (fixed with the `[b]enchmark_moe.py` bracket trick). Cost:
  **$4.5448** of a $10 cap, measured from RunPod's billing API rather
  than rate x duration. Full account:
  `docs/findings/phase-6/2026-09-20-phase-6-final-benchmark-run.md`;
  runbook: `docs/runbooks/phase-6-final-benchmark.md`.
- **Phase 5b (speculative decoding) complete**, 2026-09-18, merged via PR #6
  (`docs/plans/2026-09-17-phase-5b-speculative-decoding-plan.md`). A shared
  propose/verify/accept/rollback loop with two drafters (a 7B draft model,
  and model-free prompt-lookup) on Phase 5a's int8 target, with an
  independent plain-greedy oracle for the baseline — `make check` green
  throughout (170 tests). **The first GPU session's numbers (+21.8% /
  +76.9%) were withdrawn** by the final whole-branch review: its baseline
  was a degenerate 64 x token-id-0 on every prompt. Root cause, found on a
  second session across four pods: `transformers==5.17.0` leaves
  DeepSeek's remote-code RoPE `inv_freq` buffer uninitialized after
  `from_pretrained`, poisoning attention with NaN on GPU;
  `fix_rope_inv_freq()` now runs inside `load_model` for every caller.
  Re-measured on an A40 (bf16, unbatched, 64 new tokens, 4 prompts x 5
  reps), checked against the baseline at every k: baseline **25.18
  tok/s**; **only prompt-lookup beats it** (34.4-50.9 tok/s across k=1-8,
  +77.7% at k=4 in the sweep), while the 7B draft model is below baseline
  at every k (19.6-23.0). **Output is not always byte-exact against plain
  greedy**: draft-model matched in 13 of 16 (prompt, k) combinations,
  prompt-lookup in 9 of 16, and the same k=4 prompt-lookup config matched
  2/4 prompts in one process and 3/4 in another. Root-caused by a
  batch-width probe and a 10-trial determinism probe to a near-tied logit
  under the int8 kernel's floating-point precision, not a logic error.
  Phase 5a's "perfect top-1/top-k agreement" used the same kernel but is a
  small sample (29 positions, one run, `transformers==4.57.6`, so not
  affected by the RoPE bug; its logit differences were finite and
  non-zero, so it was not a degenerate-output pass) -- too few positions
  to rule this sensitivity out, so Phase 6's correctness gate re-checks
  agreement at scale rather than leaning on 5a's number. Two findings-doc figures were also corrected after re-verifying against
  the results JSONs (see the git history of the findings doc). Cost:
  **$12.26** across both sessions against a $10 cap that one pod exceeded
  and that was raised to $20 with disclosure. Also: `docs/findings/` split
  into one subfolder per phase. Full account:
  `docs/findings/phase-5b/2026-09-18-phase-5b-speculative-decoding-run.md`.
- **Phase 5a (int8 weight-only quantization) complete**, 2026-09-17
  (`docs/plans/2026-09-16-phase-5a-quantization-plan.md`). Self-computed,
  per-output-channel int8 quantization extending the Triton kernel
  itself (not sourced from bitsandbytes/AWQ) — `make check` green
  throughout (123 tests). Kernel-level correctness gate passed on a real
  L40 (15/15 int8, 25/25 bf16). Measured on the real
  `deepseek-ai/deepseek-moe-16b-base` model: int8 kernel **+70.9%** over
  stock (a throughput tie with the bf16 naive kernel it's built on,
  expected since weight-only quantization saves memory bandwidth, not
  FLOPs), **49.89%** expert-weight memory reduction, perfect model-level
  top-1/mutual-top-k agreement at every tested position. Two real bugs
  found on real hardware: quantizing without freeing the original bf16
  expert weights (holding both copies at once, OOMing a 44GB L40; found
  live during the rental) and quantization arithmetic computed in the
  weight's own bf16/fp16 dtype instead of float32 (widened round-trip
  error, and an fp16-specific scale-clamp cliff; found by the final
  whole-branch review), both fixed before merge. Cost: **$1.95** of a $5
  cap, including two RunPod Community Cloud pods that hit a real
  host-level GPU passthrough bug. Full account:
  `docs/findings/phase-5a/2026-09-17-phase-5a-quantization-run.md`.
- **Phase 4 (disaggregated prefill/decode) complete**, 2026-09-17
  (`docs/plans/2026-09-16-phase-4-disaggregated-prefill-decode-plan.md`).
  Continuous-batching prefill/decode workers across two real 4-GPU H100
  SXM topologies (co-located 4-rank EP, disaggregated 2+2-rank EP with a
  real cross-rank KV-cache handoff), both byte-exact against a
  single-GPU reference — `make check` green throughout (102 tests).
  Three real bugs found and fixed on real hardware, none caught by
  CPU-only tests. Measured result was mixed, not clean: disaggregated
  TTFT beats co-located at concurrency 4 (0.46s vs 1.12s) but loses at
  concurrency 8 (0.98s vs 0.70s) — most likely a kernel-warmup confound,
  reported as genuinely inconclusive. Cost: **$10.94** of a $40 cap.
  Full account:
  `docs/findings/phase-4/2026-09-17-phase-4-disaggregated-prefill-decode-run.md`.
- **Phase 3 (multi-GPU expert-parallel serving) complete**, 2026-09-16
  (`docs/plans/2026-09-15-phase-3-multi-gpu-expert-parallel-serving-plan.md`).
  Real 2x H200 SXM expert-parallel serving over DeepSeek's own DeepEP
  (V1 `Buffer`, after V2's `ElasticBuffer` proved unavailable on this
  rental's Fabric-Manager-less hardware) — `make check` green
  throughout. Correctness gate passed: perfect top-1 and mutual top-5
  agreement vs. a single-GPU reference. The measured run disproved this
  project's own hypothesis: Phase 1's naive-vs-persistent crossover
  doesn't reproduce under DeepEP's real per-expert token counts (naive
  wins at every tested count on H200; real per-local-expert counts,
  median 2 max ~20, sit below Phase 1's smallest tested point of 16).
  Cost: **$13.03** of a $25 cap. Full account:
  `docs/findings/phase-3/2026-09-16-phase-3-multi-gpu-ep-run.md`.
- **Phase 2 (vLLM benchmark contribution) complete**, 2026-09-15
  (`docs/plans/2026-09-15-phase-2-vllm-benchmark-contribution-plan.md`).
  Neither vLLM's nor SGLang's official MoE benchmark modeled skewed
  (zipf) expert load; upstreamed an opt-in `--expert-load-distribution`
  flag to vLLM's `benchmarks/kernels/benchmark_moe.py`. Measured on one
  rented RTX 3090: `--tune` under zipf vs. uniform routing picks a
  different winning Triton config at 4 of 5 tested batch sizes. Open
  PR: [vllm-project/vllm#57100](https://github.com/vllm-project/vllm/pull/57100).
  Cost: **$0.77**. This phase landed no code in `dispatch` itself, only
  docs — its actual contribution is the upstreamed PR. Full account:
  `docs/findings/phase-2/2026-09-15-phase-2-vllm-benchmark-run.md`.
- **Phase 1 (custom Triton grouped-GEMM kernel) complete**, 2026-09-15
  (`docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md`). Built and
  tested: a CPU-only MoE reference and grouped-GEMM contract, a naive and
  a persistent cache-aware Triton kernel, a backend registry and
  micro-benchmark CLI, and real-model integration (`--moe-kernel`,
  `--compare-reference`) — `make check` green throughout (73 tests, lint
  and `mypy --strict` clean). Two real rented-GPU sessions:
  - **Kernel correctness** (RTX 3090, $0.06): both kernels passed 25/25
    correctness tests on the first real execution — no kernel bugs
    found. A mutation check (forcing every tile to read expert 0's
    weights) turned 24/25 red, confirming the suite can fail.
  - **The measured run** (L40, $0.59, same GPU class as Phase 0): both
    kernels re-verified correct (25/25) on this card, then swapped into
    `deepseek-ai/deepseek-moe-16b-base`'s real 27 MoE layers. Measured
    **12.55 -> 20.98 tokens/sec (naive kernel, +67.2%)**, **12.55 ->
    20.80 tokens/sec (persistent kernel, +65.7%)** against DeepSeek's
    own stock `moe_infer`, at perfect mutual top-5 and top-1 logit
    agreement across every tested position (bf16, single L40, unbatched
    eager-mode decode, 3 prompts x 5 repetitions, 64 new tokens — same
    config as Phase 0's baseline). Cost per 1M generated tokens: $18.15
    (stock) -> $10.86 (naive). A token-count micro-benchmark (1 to 2048
    tokens, zipf and uniform routing) found the persistent kernel ties
    naive at the 1-token/step granularity that drives decode throughput
    (unbatched decode gives each expert too few rows for L2 reuse to
    matter, exactly as the plan's own risk section predicted before the
    run happened), wins 3-8% at 16-128 tokens, then loses by up to 14% at
    512-2048 — a workload-specific result recorded rather than buried.
    Total GPU cost across both sessions: **$0.65**.
    Full account: `docs/findings/phase-1/2026-09-15-phase-1-grouped-gemm-run.md`.
- **Phase 0 (baseline) complete**, 2026-09-14
  (`docs/plans/2026-09-14-phase-0-baseline-plan.md`). Built and tested: a
  RunPod REST client and provisioning CLI, a token-by-token-timed
  generation harness, reference-logit capture, and the baseline CLI tying
  them together (`make check` green, 26 tests). The real rented-GPU run
  measured `deepseek-ai/deepseek-moe-16b-base` (bf16, single NVIDIA L40,
  unbatched eager-mode decode, 15 runs): **12.75 tokens/sec mean
  throughput, 0.355s mean time-to-first-token** (p50 0.258s, p99 1.588s).
  Cost: **$0.35** for 25.7 minutes, three failed attempts included — all
  three were real environment bugs (a `transformers` version break in
  DeepSeek's own remote code, the same break's cousin one release
  earlier, and a model-cache-on-the-wrong-disk trap), none in this
  repo's own code. Full account:
  `docs/findings/phase-0/2026-09-14-phase-0-baseline-run.md`.
- Repo scaffolded: conventions, tooling, and doc structure carried over
  from [almanac](https://github.com/bsreecharanreddy/almanac) and
  [canopica](https://github.com/bsreecharanreddy/canopica) where
  applicable.
- System design written
  (`docs/design/2026-09-14-dispatch-system-design.md`): DeepSeekMoE-16B as
  the target model, a custom Triton grouped-GEMM kernel as the
  differentiator, DeepSeek's DeepEP for cross-GPU expert dispatch, standard
  benchmark tooling (vLLM's `benchmark_serving.py` / NVIDIA GenAI-Perf)
  against vLLM/SGLang. Two ADRs recorded alongside it (grouped-GEMM over
  another attention-kernel reimplementation; DeepEP over hand-rolled
  communication, and the NVLink/SXM hardware constraint that decision
  carries into cost planning).
- Design revised to 8 phases (0-7) after checking the plan against real
  inference-engineer job postings, national and Atlanta-metro: Phase 7 adds
  a Rust router (HF `text-generation-inference`'s own documented
  architecture) in front of the unchanged Python model server, Docker, a
  Prometheus/Grafana observability layer, and a K8s deployment demoed once.
  ADR-0003 records the Rust-scoped-to-the-router decision and what was
  deliberately not rewritten.
