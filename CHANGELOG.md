# Changelog

Phase-level history. `docs/STATUS.md` carries the task-level verification
log; this is the summary a reader wants first.

Every number here is measured. Where one is later found wrong, it gets
corrected **in place with the original kept**, because a retraction that
deletes its own evidence is not a retraction.

## Unreleased

No phase shipped yet.

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
