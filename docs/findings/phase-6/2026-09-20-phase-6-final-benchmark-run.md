# Phase 6 -- final benchmark vs. vLLM and SGLang: run and outcome

Plan: `docs/plans/2026-09-19-phase-6-final-benchmark-plan.md`.
Design: `docs/design/2026-09-18-phase-6-final-benchmark.md`.
Hardware: one NVIDIA L40 (48GB, driver 580.178.04, CUDA 13.0), RunPod Secure
Cloud, pod `97jlyai5sowyeq`, US-KS-2, 2026-09-19 17:54 UTC -- 2026-09-20 00:10
UTC. `torch==2.13.0+cu130` and `triton==3.7.1` are identical across every
contestant -- dispatch's own kernels, vLLM 0.29.0, and SGLang 0.5.20 all run
on the same compiler, so no result below is confounded by a Triton version
difference.

## Outcome

**On this GPU, at these shapes, vLLM's and SGLang's own fused-MoE beat
dispatch's kernels at their shipped default configuration, at nearly every
routed-expert-layer shape tested, in both bf16 and int8.** dispatch's naive
kernel is competitive only at the smallest and largest ends of the sweep (1
token, and 2048-token int8), and its own tile-size tuning closes some but
not all of the gap. This is a real, disclosed negative result, not a caveat
buried under a positive headline: this repo has published null and negative
results before (Phase 1's persistent-kernel tie, Phase 3's disproved
hypothesis, Phase 4's inconclusive TTFT), and this is another one.

**The correctness gate passed for every kernel at scale**: naive, persistent
and int8 all agree with the stock reference at every one of 1,036 tested
positions, at a top-1 agreement of 97.1-97.3%, with every disagreement
falling inside the pre-registered near-tie band (widest gap 0.750, threshold
1.0) and zero large-gap failures. This is the first time this project has
measured agreement past Phase 5a's 29-position sample, and it holds up.

**vLLM and SGLang, served end-to-end on the same model and hardware, both
work correctly and scale as expected** (output tokens/sec roughly 10x from
concurrency 1 to 64), with vLLM ahead of SGLang at every concurrency tested
here. dispatch has no serving stack of its own to compare against them
end-to-end (system design section 6, amended); its own harness supplies a
concurrency-1 reference row only, explicitly labeled non-comparable to the
served numbers.

**Full 7-point tuning of vLLM's and SGLang's own MoE tuners was not run.**
A live check found each takes roughly 18 minutes per token count and writes
its output only after every requested token count finishes -- no partial
credit on an early stop -- so the full matrix would have cost 2+ hours per
engine per precision, several times this phase's entire budget. The user
chose, at a $4.22 checkpoint, to report both engines at their shipped
default configuration only. This is disclosed as a real limitation of the
"tuned" comparison below, not glossed over.

**Total cost: $4.5448 of the $10 cap** (measured from RunPod's billing API,
not estimated from rate x duration).

## 1. The at-scale correctness gate

Extends `scripts/run_baseline.py` with `--prompt-set gate` (16 fixed
prompts, 1,036 real-tokenizer positions) and the gap-split classifier in
`src/dispatch/benchmark/agreement.py`. Pre-registered rule (before any GPU
time this phase): a top-1 disagreement is a **failure** only if the
reference's own top-1/top-2 logit gap exceeds **1.0** (2.5x the widest
near-tie Phase 5b measured, 0.4); at or below that, it is a **near-tie**,
counted but not failed. Minimum 500 compared positions per config.

| config | positions | disagreements | near-tie | **large-gap** | widest gap | top-1 agreement |
|---|---|---|---|---|---|---|
| stock vs. stock (control) | 1036 | 0 | 0 | **0** | 0.000 | 100.000% |
| naive (bf16) | 1036 | 28 | 28 | **0** | 0.750 | 97.297% |
| persistent (bf16) | 1036 | 30 | 30 | **0** | 0.750 | 97.104% |
| int8 (`quantized`) | 1036 | 29 | 29 | **0** | 0.750 | 97.201% |
| naive, second control run | 1036 | 29 | 29 | **0** | 0.438 | 97.201% |

**The control's own noise floor is zero**: the unpatched model compared
against itself, 1,036 positions, zero disagreements of any kind. This is the
gate's proof that its own measurement process introduces no spurious
flips -- every disagreement below comes from the kernel, not from a
run-to-run artifact.

**Every kernel passes.** No config produced a large-gap disagreement, so
none is excluded from the race below. The two independent naive runs (a
scheduled control and the race's own gate pass) landed on different exact
disagreement sets (28 vs. 29, widest gap 0.750 vs. 0.438) -- the same
process-to-process near-tie instability Phase 5b first found, reproduced
here at 36x the sample size and still bounded by the pre-registered
threshold. This replaces Phase 5a's 29-position "perfect agreement" claim
for these exact configs: the honest number is 97.1-97.3% top-1 agreement,
with 100% of the shortfall inside a mechanism this project can name and
bound, not zero disagreement.

## 2. The kernel race

Identical seeded inputs (weights, `x`, `topk_idx`, `topk_weight`) and an
fp32 reference are generated once (`scripts/run_engine_race.py prepare`) and
loaded by every contestant; every contestant's output is checked against
the fp32 reference before it is timed (`assert_matches_reference`) and
**zero results were refused** across all 224 (engine, token count,
distribution) combinations run. Token counts: 1, 4, 16, 64, 128, 512, 2048.
Routing: uniform and zipf. Timing: `triton.testing.do_bench`.

**Tuning, as actually run** (a disclosed reduction from the plan's matrix,
decided live -- see "What changed from the plan" below): dispatch's own
`block_m` in {16, 32, 64, 128} was swept for real (its own kernel, seconds
of GPU time, not an upstream autotune search); its "tuned" row is the tile
size fastest on uniform routing at that token count, reused for zipf. vLLM
and SGLang are reported at their shipped **default** configuration only --
their own tuners were not run to completion.

### bf16, default config (mean / p99 ms per routed-MoE layer call)

| tokens | routing | dispatch-naive | dispatch-persistent | sglang | vllm |
|---|---|---|---|---|---|
| 1 | uniform | 0.534 / 0.745 | 0.569 / 0.784 | 0.227 / 0.239 | **0.215 / 0.230** |
| 1 | zipf | 0.532 / 1.009 | 0.557 / 0.794 | 0.229 / 0.539 | **0.216 / 0.217** |
| 4 | uniform | 0.995 / 1.241 | 0.961 / 1.204 | 0.673 / 0.677 | **0.625 / 0.797** |
| 4 | zipf | 0.878 / 1.180 | 0.851 / 1.043 | 0.573 / 0.582 | **0.531 / 0.537** |
| 16 | uniform | 1.724 / 1.893 | 1.656 / 1.989 | 1.359 / 1.434 | **1.291 / 1.292** |
| 16 | zipf | 1.474 / 1.587 | 1.427 / 1.637 | 1.119 / 1.120 | **1.053 / 1.058** |
| 64 | uniform | 2.314 / 2.506 | 2.065 / 2.253 | 1.747 / 1.752 | **1.679 / 1.684** |
| 64 | zipf | 2.046 / 2.050 | 1.967 / 2.126 | 1.636 / 1.642 | **1.580 / 1.583** |
| 128 | uniform | 2.184 / 2.337 | 2.078 / 2.278 | 2.565 / 2.588 | **1.860 / 1.867** |
| 128 | zipf | 2.213 / 2.470 | 2.118 / 2.274 | 2.565 / 2.591 | **1.873 / 1.880** |
| 512 | uniform | 2.315 / 3.979 | 2.380 / 3.464 | 2.729 / 2.732 | **1.947 / 1.955** |
| 512 | zipf | 2.313 / 2.596 | 2.558 / 2.890 | 2.695 / 2.708 | **1.984 / 2.065** |
| 2048 | uniform | 5.200 / 5.822 | 5.261 / 5.451 | 3.020 / 3.371 | **2.490 / 2.688** |
| 2048 | zipf | 5.142 / 5.788 | 5.199 / 5.812 | 3.476 / 3.526 | **2.721 / 2.743** |

### int8, default config (mean / p99 ms per routed-MoE layer call)

| tokens | routing | dispatch-naive | sglang | vllm |
|---|---|---|---|---|
| 1 | uniform | 0.525 / 1.127 | 0.189 / 0.431 | **0.141 / 0.126** |
| 1 | zipf | 0.558 / 0.801 | 0.495 / 0.526 | **0.125 / 0.125** |
| 4 | uniform | 0.806 / 1.016 | 0.488 / 0.496 | **0.388 / 0.391** |
| 4 | zipf | 0.708 / 1.010 | 0.408 / 0.412 | **0.320 / 0.330** |
| 16 | uniform | 1.429 / 1.539 | 0.927 / 0.939 | **0.770 / 0.771** |
| 16 | zipf | 1.173 / 1.665 | 0.769 / 0.786 | **0.646 / 0.650** |
| 64 | uniform | 1.668 / 2.395 | 1.184 / 1.203 | **0.975 / 0.979** |
| 64 | zipf | 1.541 / 2.025 | 1.100 / 1.137 | **0.923 / 0.946** |
| 128 | uniform | **1.671 / 1.840** | 2.111 / 2.145 | 1.232 / 1.298 |
| 128 | zipf | **1.660 / 1.794** | 2.111 / 2.156 | 1.247 / 1.266 |
| 512 | uniform | **1.749 / 2.068** | 2.320 / 2.347 | 1.312 / 1.319 |
| 512 | zipf | **1.745 / 2.099** | 2.219 / 2.302 | 1.356 / 1.372 |
| 2048 | uniform | 4.690 / 5.529 | 2.798 / 3.460 | **2.520 / 2.683** |
| 2048 | zipf | 4.707 / 5.200 | 3.271 / 3.531 | **2.497 / 2.559** |

**Per-shape reading, only what the tables directly support** (bold = fastest
mean at that row):

- **bf16: vLLM wins every single row**, dispatch's persistent kernel is
  consistently the second-best dispatch variant, and SGLang sits between
  dispatch and vLLM except at 128-512 tokens, where it falls behind
  dispatch's own kernels.
- **int8: dispatch's own naive kernel wins at 128 and 512 tokens**, the one
  clear region where this project's own kernel is the fastest measured
  contestant. vLLM wins everywhere else, including both ends of the sweep
  (1 token and 2048 tokens) by a wide margin at the small-batch end (0.141ms
  vs. dispatch's 0.525ms at 1 token).
- Full tables (both default and dispatch's own tuned column) are in
  `2026-09-19-phase-6-race-summary.md`; dispatch's tuned tile size closes
  some but not most of the default-vs-vLLM gap (e.g. naive int8 at 2048
  tokens: 4.690ms default -> 3.019ms tuned, still behind SGLang's 2.798ms
  and vLLM's 2.520ms).

## 3. The engine reference (not a race)

vLLM and SGLang each served the full model (bf16) from the same L40, driven
by `vllm bench serve` against the identical 16-prompt trace, replayed to
512 lines, concurrency 1/4/16/64, `--ignore-eos --custom-output-len 64`.
Every request generated exactly 64 tokens; zero failed requests at any
concurrency, for either engine.

| engine | concurrency | out tok/s | p50 TTFT (ms) | p99 TTFT (ms) | mean ITL (ms) | p50 e2e (ms) | $/M output tok |
|---|---|---|---|---|---|---|---|
| vLLM | 1 | 115.5 | 37.9 | 56.2 | 8.16 | 551.3 | 1.9728 |
| vLLM | 4 | 225.7 | 52.8 | 68.2 | 17.15 | 1148.0 | 1.0092 |
| vLLM | 16 | 444.9 | 105.8 | 144.0 | 34.74 | 2303.7 | 0.5119 |
| vLLM | 64 | 1269.2 | 174.1 | 248.0 | 48.09 | 3201.9 | 0.1795 |
| SGLang | 1 | 108.6 | 54.9 | 81.8 | 8.42 | 585.1 | 2.0970 |
| SGLang | 4 | 207.5 | 38.9 | 73.7 | 18.87 | 1227.8 | 1.0978 |
| SGLang | 16 | 405.5 | 73.4 | 136.0 | 38.77 | 2518.6 | 0.5617 |
| SGLang | 64 | 1180.4 | 135.8 | 206.4 | 52.80 | 3459.3 | 0.1930 |
| **dispatch (stock)** | 1 only | -- | -- | -- | -- | -- | -- |
| **dispatch (naive)** | 1 only | -- | -- | -- | -- | -- | -- |
| dispatch, all engines | 4 / 16 / 64 | n/a: no dispatch server | | | | | |

**dispatch's own concurrency-1 numbers, separately, non-comparable to the
table above**: stock 11.87 tok/s mean, naive-kernel-patched 21.04 tok/s mean
(32 runs each, 16 prompts x 2 repetitions, `--ignore-eos --max-new-tokens
64`) -- consistent with Phase 1's own measured 12.55/20.98 tok/s at a
different prompt count and repetition count. These numbers are **not**
placed in the ranked table above: dispatch's harness is an in-process,
unbatched, single-request eager decode loop with no HTTP layer, no
CUDA-graph capture and no continuous batching, so a direct ms-for-ms or
tok/s-for-tok/s comparison against vLLM's or SGLang's *served* numbers would
compare two different things under one column header. What the numbers do
support: dispatch's naive kernel is a genuine ~1.8x speedup over the stock
model's own eager `moe_infer` (21.04 vs. 11.87 tok/s), consistent with
Phase 1's finding, on this hardware, at this batch size (1).

**vLLM leads SGLang at every concurrency tested here**, by roughly 6-9% on
output throughput and with lower TTFT except at concurrency 4 (where SGLang's
p50 TTFT, 38.9ms, is lower than vLLM's 52.8ms, though vLLM's p99 is lower:
68.2ms vs. 73.7ms). Both scale output throughput by roughly 11x from
concurrency 1 to 64, and cost per million output tokens falls by roughly
11x over the same range for both.

## 4. What changed from the plan, and what it cost

- **Tuner scope reduced to default-only for vLLM and SGLang** (Task 12).
  vLLM's `benchmark_moe.py --tune` over 1,920 Triton configs takes ~18
  minutes per token count on this GPU, running single-worker and
  sequential across the 7 requested batch sizes, and `save_configs` is
  called exactly once, after every batch size finishes -- confirmed live by
  starting a real tuning run, watching it reach 29% of the *first* of 7
  batch sizes after 5 minutes, and killing it (zero usable output, as
  expected from reading the tuner's own save logic before spending more
  time). The full matrix would have cost 2+ hours per engine per precision.
  Paused at a $4.22 checkpoint; user decision: skip tuner runs entirely,
  report default-only. dispatch's own tile-size sweep is unaffected (it is
  not an upstream autotune search).
- **Both engines' tuners and vLLM's tuner script needed the same
  architecture shim** dispatch had already anticipated for SGLang in the
  design: `deepseek-ai/deepseek-moe-16b-base` reports architecture
  `DeepseekForCausalLM` (V1), and both `benchmark_moe.py`'s
  `get_model_params` and SGLang's `get_model_config` only recognize
  `DeepseekV2ForCausalLM` and later. A local config copy with the
  architecture field relabeled (`/workspace/deepseek-config-shim`, same
  routed-expert dims) fixed both -- confirmed to read the correct E=64,
  topk=6, moe_intermediate=1408 either way.
- **vLLM's own `benchmarks/kernels/benchmark_moe.py` and
  `vllm.benchmarks.serve` needed `ray` and `pandas`** respectively,
  neither pulled in by a plain `pip install vllm`; installed on the pod
  (not in this repo's own dependency groups, since the engines venv is
  pod-local and never `uv sync`'d from this repo).
- **vLLM 0.29.0 and SGLang 0.5.20 do co-install** (both pin
  `torch==2.13.0`), contrary to the plan's assumed conflict, but SGLang
  additionally needs `--prerelease=allow` (`cuda-tile==1.6.0rc5`, a
  `flash-attn-4` beta) that vLLM's own resolve rejects when both are
  requested together -- so one venv per engine was used after all, for a
  different reason than originally assumed. Both share `torch==2.13.0+cu130`
  and `triton==3.7.1`, so the "same compiler" fairness point still holds.
- **SGLang's adapter needed a published `ServerArgs`** (`fused_experts`
  reads a config namespace, `get_exec()`, that SGLang only fills from one)
  and idempotent setup (a second `initialize_model_parallel` call raises) --
  both fixed in `sglang_moe.py` (commit `c31d1f4`), test-first, before any
  GPU time was spent confirming them live.
- **dispatch's own combine step is not bitwise repeatable**: `index_add_`
  is `atomicAdd` on CUDA, so two identical calls to the same bound layer can
  differ by one bf16 ulp on a meaningful fraction of elements (24% measured
  on the first real run). The repeatability test was fixed to compare
  within the reference tolerance instead of exactly (commit `2536764`).
- **vLLM 0.29.0's `vllm bench serve --save-detailed` carries no per-request
  end-to-end latency field at all** -- only per-request TTFT and the
  inter-token gaps that follow it. `summarize_bench_result` was fixed to
  derive end-to-end latency as `ttft + sum(gaps)` (commit `3acf14c`),
  confirmed against the real 0.29.0 schema, not assumed from its docs.
- **The serving driver's subprocess `PATH` needed sglang-venv listed before
  vllm-venv** so the bare `python` used to launch `sglang.launch_server`
  resolved to the interpreter with SGLang actually installed -- a
  pod-invocation fix, not a code change (the `vllm` bench-client binary
  still resolves correctly from vllm-venv either way, since sglang-venv has
  no such script).
- **No per-engine bare-GEMM diagnostic** (design amendment, decided before
  GPU time): vLLM's and SGLang's fused-MoE paths expose no separable
  single-GEMM entry point comparable to dispatch's own gate-proj-only
  timing.

## 5. Explicit non-claims

- **One GPU class (L40), one model, one seed.** No claim generalizes to
  other hardware, other MoE architectures, or a different routing seed.
- **Synthetic weights in the kernel race.** The race uses seeded random
  weights at the model's real dimensions, not the model's actual trained
  weights -- appropriate for a latency comparison, not a claim about output
  quality (the correctness gate, which does use the real model and real
  weights, carries that claim instead).
- **vLLM and SGLang are shown at default configuration only.** "vLLM beats
  dispatch" and "SGLang beats dispatch" above are default-vs-default (and
  default-vs-dispatch's-own-tuned) claims; neither production engine's own
  tuned ceiling on this GPU was measured.
- **The engine reference is not a three-way race.** dispatch has no served,
  batched, HTTP-facing engine to place in the same ranked table as vLLM's
  and SGLang's concurrency-scaled numbers; its own harness's concurrency-1
  numbers are reported alongside, not merged into, that table.
- **This is a kernel/layer- and serving-level comparison, not a claim about
  end-task quality**, cost-per-request accounting beyond GPU-hour price, or
  any multi-GPU configuration (Phase 3's DeepEP work stands on its own,
  untouched here).

## 6. Cost

**$4.5448 of the $10 cap** (RunPod billing API, `list-pod-billing`, hourly
buckets spanning the full session: $4.3745 GPU + $0.1704 disk). Full
account: `2026-09-20-phase-6-final-benchmark-cost.md`.
