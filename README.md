# dispatch

An inference engine for the token-routing layer of Mixture-of-Experts (MoE)
language models — a custom GPU kernel for expert dispatch and grouped-GEMM
execution, multi-GPU expert-parallel serving, and disaggregated
prefill/decode, benchmarked against production serving engines (vLLM,
SGLang) on real, measured throughput/latency/cost numbers, not estimates.

**Status: Phase 4 (disaggregated prefill/decode) complete, open as a PR.**
Phases 0-3 are merged: a plain-HF baseline, a naive and a persistent
Triton grouped-GEMM kernel (**~65-67% faster decode throughput** than
DeepSeek's own stock `moe_infer`, perfect mutual top-5/top-1 logit
agreement), a skewed-load benchmark contribution upstreamed to vLLM, and
real 2-GPU expert-parallel serving via DeepSeek's DeepEP. Phase 4 adds a
continuous-batching prefill/decode scheduler and tests it across two real
4-GPU topologies — a co-located 4-rank EP pool and a disaggregated
2+2-rank EP pool connected by a real cross-rank KV-cache handoff — both
proven byte-exact correct against a single-GPU reference on real
hardware. Three real bugs found and fixed along the way, all now covered
by regression tests. See
[`docs/findings/2026-09-17-phase-4-disaggregated-prefill-decode-run.md`](docs/findings/2026-09-17-phase-4-disaggregated-prefill-decode-run.md)
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
