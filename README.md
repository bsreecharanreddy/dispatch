# dispatch

An inference engine for the token-routing layer of Mixture-of-Experts (MoE)
language models — a custom GPU kernel for expert dispatch and grouped-GEMM
execution, multi-GPU expert-parallel serving, and disaggregated
prefill/decode, benchmarked against production serving engines (vLLM,
SGLang) on real, measured throughput/latency/cost numbers, not estimates.

**Status: design written, no code yet.** See
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
