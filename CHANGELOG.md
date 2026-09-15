# Changelog

Phase-level history. `docs/STATUS.md` carries the task-level verification
log; this is the summary a reader wants first.

Every number here is measured. Where one is later found wrong, it gets
corrected **in place with the original kept**, because a retraction that
deletes its own evidence is not a retraction.

## Unreleased

No phase shipped yet (`v0.1.0` is still the scaffold version; a tag lands
once a phase's exit criteria are actually met).

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
  `docs/findings/2026-09-14-phase-0-baseline-run.md`.
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
