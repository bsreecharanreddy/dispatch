# dispatch

An inference engine for the token-routing layer of Mixture-of-Experts (MoE)
language models — a custom GPU kernel for expert dispatch and grouped-GEMM
execution, multi-GPU expert-parallel serving, and disaggregated
prefill/decode, benchmarked against production serving engines (vLLM,
SGLang) on real, measured throughput/latency/cost numbers, not estimates.

**Status: not started.** This repo currently holds only its own scaffolding
— conventions, tooling, and doc structure, carried over from
[almanac](https://github.com/bsreecharanreddy/almanac) and
[canopica](https://github.com/bsreecharanreddy/canopica) where applicable.
Nothing below is built yet. See `docs/STATUS.md` for the authoritative,
same-commit-updated record.

The system design is still being worked out and will land in `docs/design/`
before any code does — read this section again once it's there rather than
trusting this paragraph.

## Development

```bash
uv sync --all-extras --dev
make check       # lint + typecheck + test — the full gate, same as CI
make check-fast  # inner loop
```

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
