# dispatch

An inference engine for the token-routing layer of Mixture-of-Experts (MoE)
language models — a custom GPU kernel for expert dispatch and grouped-GEMM
execution, multi-GPU expert-parallel serving, and disaggregated
prefill/decode, benchmarked against production serving engines (vLLM,
SGLang) on real, measured throughput/latency/cost numbers, not estimates.

**Status: Phase 1 (custom Triton grouped-GEMM kernel) complete.**
A naive and a persistent, cache-aware Triton grouped-GEMM kernel, swapped
into `deepseek-ai/deepseek-moe-16b-base`'s real 27 MoE layers, measured
**~65-67% faster decode throughput than DeepSeek's own stock `moe_infer`**
(12.55 → 20.98 tokens/sec, bf16, single NVIDIA L40, unbatched eager-mode
decode) at perfect mutual top-5 and top-1 logit agreement across every
tested position. One honest null result: the persistent kernel's grouped
launch ordering showed no measurable benefit over the naive kernel at
this project's decode-shaped workload — recorded rather than buried. See
[`docs/findings/2026-09-15-phase-1-grouped-gemm-run.md`](docs/findings/2026-09-15-phase-1-grouped-gemm-run.md)
for the full measured account, and
[`docs/design/2026-09-14-dispatch-system-design.md`](docs/design/2026-09-14-dispatch-system-design.md)
for the architecture — model choice, what's custom vs. reused, the 7-phase
plan, and explicit scope boundaries — and `docs/STATUS.md` for the
authoritative, same-commit-updated record of what's actually built.
Conventions and tooling carried over from
[almanac](https://github.com/bsreecharanreddy/almanac) and
[canopica](https://github.com/bsreecharanreddy/canopica) where applicable.

## Development

```bash
uv sync --all-extras --dev
make check       # lint + typecheck + test — the full gate, same as CI
make check-fast  # inner loop
```

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
