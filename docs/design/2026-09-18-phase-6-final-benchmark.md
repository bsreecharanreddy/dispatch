# Phase 6 design: final benchmark vs. vLLM and SGLang

Per `docs/design/2026-09-14-dispatch-system-design.md` §7 (phase plan, row
6: "Final benchmark vs. vLLM/SGLang, standard tool, same hardware/model").
Phases 0-5b are complete and merged (Phase 2's upstream PR is open). Every
earlier phase deferred the head-to-head comparison to this one -- see the
"Phase 6 owns that comparison" non-goals in the Phase 4, 5a and 5b designs.

## 1. Purpose and thesis

Every number this project has published so far was measured against its
own baseline: stock `moe_infer`, the naive kernel, the bf16 target. None
was measured against a production engine. This phase does that.

**Thesis: on one L40, at DeepSeek-MoE-16B's real routed-expert shape, how
do dispatch's Triton grouped-GEMM kernels compare with vLLM's and SGLang's
fused-MoE on latency -- across token counts, uniform vs. zipf routing, and
bf16 vs. weight-only int8 -- and where does a full production engine land
on the same model and hardware?**

The result is a table, not a headline. Nothing is summarized as "beats
vLLM" or "loses to vLLM" unless the per-shape table shows it shape by
shape. This repo has published null and negative results before (Phase 1's
persistent kernel, Phase 3's disproved hypothesis, Phase 4's inconclusive
TTFT) and this phase is designed to tolerate the same: production
fused-MoE kernels are heavily tuned and may simply win at some or all
shapes.

## 2. Scope, decided during brainstorming

- **Kernel-level race plus an engine reference, not an end-to-end race.**
  Design doc §6 assumed "this project's own server" would exist to point
  the standard serving benchmark at. It does not: dispatch's engine is a
  Hugging Face eager decode loop with the MoE layer patched -- unbatched,
  single-request, no CUDA graphs, no paged KV cache. An end-to-end race
  against vLLM/SGLang would measure that gap, not the kernel, which is
  the part dispatch actually built. So the like-for-like comparison is at
  the kernel/layer level, and the full-engine numbers are a labeled
  reference. (Options considered and not chosen: building an
  OpenAI-compatible dispatch server for a literal three-way end-to-end
  race -- expected to lose on HF-eager overhead rather than kernel
  quality; swapping dispatch's kernel into vLLM as a custom MoE backend --
  strongest claim if it works, highest risk, plausibly its own
  multi-day integration.)
- **Hardware: one L40, ~$10 cap.** Same GPU class as Phase 1's and Phase
  5a's measured kernel numbers, so this phase extends those tables. 48GB
  fits the 16B bf16 model for the engine reference. Rejected: H100 (breaks
  continuity, several times the hourly cost) and an L40-plus-H100 split
  (the reference would no longer share hardware with the race, which §6
  of the system design forbids).
- **Weight-only int8 gets a like-for-like row.** Checked live 2026-09-18:
  both vLLM's `fused_experts_impl` and SGLang's `fused_experts` accept
  `use_int8_w8a16` and `per_channel_quant`, the same scheme Phase 5a's
  kernel implements (per-output-channel int8 weights, bf16 activations).
- **SGLang serves this model.** Checked live 2026-09-18: SGLang's
  `python/sglang/srt/models/deepseek.py` defines `DeepseekForCausalLM`,
  the V1 architecture `deepseek-ai/deepseek-moe-16b-base` uses, and routes
  its experts through `fused_moe`.
- **Routed experts only.** The timed region matches Phase 1's: the routed
  experts' work. DeepSeek's shared experts are a dense MLP separate from
  the routed path and are not part of any engine's fused-MoE call.

## 3. Deliverables, in run order

Stage 1 gates stage 2; stage 2 gates nothing but is run before stage 3 so
that the cheapest, highest-value evidence is on disk first.

### 3.1 At-scale correctness gate (gates each config, per the plan)

`docs/STATUS.md` owes this from Phase 5a and 5b: Phase 5a's "perfect
top-1/mutual-top-k agreement" covered 29 positions in one run, and Phase
5b then found genuine near-tied logits (gaps of 0.1-0.4) under the int8
kernel's floating-point precision. Phase 5a's figure is not to be quoted
as more than "true on a small sample".

- Real model (`deepseek-ai/deepseek-moe-16b-base`), the exact configs
  being benchmarked: stock `moe_infer` (reference) vs. naive bf16,
  persistent bf16, and int8, over hundreds of positions.
- Reports top-1 agreement and mutual top-5 agreement as **rates with
  counts**, and splits every disagreement by the reference's top-1/top-2
  logit gap.
- **Gate rule:** any disagreement where the reference's logit gap exceeds
  a threshold is a failure (it would mean a bug, not a near-tie).
  Disagreements below the threshold are reported, not failed -- they are
  the known, explained floating-point sensitivity.
- **The threshold is fixed in the implementation plan, from Phase 5b's
  recorded near-tie data, before any GPU time is spent -- not chosen after
  seeing results.** (Fixed at 1.0 logit; see the plan's "Pre-registered
  gate rule".)
- **Gate scope, refined in the plan (2026-09-19):** the gate is per config.
  A config that trips it is excluded from the race and reported as failing
  its gate, rather than halting the whole paid session; the other configs
  proceed. A stock-vs-stock control run measures the gate's own noise floor.
- Reuses `scripts/run_baseline.py`'s reference-capture and comparison
  path rather than a new one; extends it to a larger, seeded prompt set.

### 3.2 Kernel race (`scripts/run_engine_race.py`)

**Compared:** dispatch naive, dispatch persistent, vLLM `fused_experts`,
SGLang `fused_experts`, each at bf16 and (where the engine supports it)
weight-only int8.

**Matrix:** total token counts per layer call, spanning decode-sized
batches (1-16) through prefill-sized ones (2048); uniform and
zipf routing (reusing `dispatch.kernels.bench.sample_topk_idx`); bf16 and
int8. The exact grid is fixed in the plan; Phase 1's sweep points are the
default so results extend Phase 1's tables.

**Same inputs, engine-specific adapters.** All contestants, dispatch's own
kernels included, run inside the one engines venv so they share a torch and
a Triton compiler version (dispatch's `torch>=2.14.0` floor is a
model-loading constraint, not a kernel one). One seeded input set is
generated once -- `x`, `topk_ids`, `topk_weights`, and the expert weights
-- and a thin adapter per engine converts it to that engine's layout
(vLLM and SGLang take fused gate+up in `w1` and down in `w2`; dispatch
takes gate, up and down stacked separately). Adapters are the only place
that knows an engine's API, because SGLang's fused-MoE code was recently
reorganized (now under `moe_runner/triton_utils/` and
`sglang/kernels/ops/moe/`) and the vLLM signature has already moved from
per-flag arguments to a `FusedMoEQuantConfig` object at the top level.
Exact engine versions are pinned and recorded in every output JSON.

**Correctness before timing.** Every engine's output is checked against
the fp32 reference (`assert_matches_reference`, as in Phase 1) before it is
timed. An engine that disagrees with the reference on the benchmark's own
input is refused, not timed -- the same rule `run_kernel_bench.py` already
enforces.

**Timed region.** The whole routed layer from `(x, topk_ids,
topk_weights)` to output, each engine's own path (grouping/alignment, the
GEMMs, activation, weighted combine). Phase 1's bare-GEMM diagnostic
stays with dispatch's own kernels only (run_kernel_bench.py already has it):
vLLM's and SGLang's fused-MoE paths expose no separable single-GEMM entry
point, so a per-engine diagnostic isn't measurable like-for-like. Timing is `triton.testing.do_bench`
(warmup, repetition and synchronization are its job), reporting mean, p50
and p99 as Phase 1 does.

**Tuning fairness.** vLLM and SGLang each get their own tuner run on
this exact L40 -- Phase 2's `--tune` path for vLLM, SGLang's
`tuning_fused_moe_triton.py` -- so a baseline is never handicapped by a
missing config for this GPU. Dispatch gets a block-size sweep of
comparable effort (its `block_m`, plus persistent-vs-naive, which is
already a comparison). **The tuned column is primary.** A stock-default
column (whatever each engine picks out of the box, with no tuned config)
is reported as a secondary sensitivity check, since that is what a user
who never runs the tuner gets. Both columns state exactly what tuning was
done and how long it ran. Tuning under zipf vs. uniform routing is itself
a Phase 2 finding (different winning config at 4 of 5 tested sizes), so
each engine is timed under both distributions, and the table says which
distribution each engine's config was tuned under.

### 3.3 Engine reference (not a race)

vLLM and SGLang each serve the full model (bf16) on the same L40, from
one shared engines virtual environment (checked live 2026-09-19: vllm 0.29.0
and sglang 0.5.20 both pin `torch==2.13.0` and their transformers pins are
compatible, so the conflict this design first assumed does not exist at
these versions; if a joint install fails to resolve, fall back to one venv
per engine and record each Triton version in the output), driven by
the standard serving benchmark (vLLM's `vllm bench serve` /
`benchmark_serving.py`, which works against any OpenAI-compatible server),
on one seeded trace, at concurrency 1, 4, 16 and 64. Standard metrics:
TTFT, inter-token latency, end-to-end latency, tokens/sec, requests/sec,
and **cost per million tokens computed from the measured GPU-hour rate**
(design §6), never estimated.

**Dispatch appears at concurrency 1 only**, replaying the identical
prompts (the gate's 16-prompt set, written out as the JSONL trace the
engines read; dispatch runs each prompt twice, one at a time) through `scripts/run_baseline.py`'s in-process path. That row
carries an explicit caveat in the table itself: the vLLM/SGLang numbers
include HTTP overhead, CUDA-graph capture and torch.compile warmup
effects that dispatch's in-process loop doesn't, and dispatch has no
batched server. Concurrency 4/16/64 rows for dispatch read "n/a: no
dispatch server", not a blank and not a number. Warmup requests are
discarded and the count is recorded (Phase 4's kernel-warmup confound is
the precedent for why).

## 4. Architecture

```text
scripts/run_engine_race.py          stage 3.2 driver: build inputs, call adapters,
                                    gate on correctness, time, write JSON
src/dispatch/benchmark/engines/     adapter per engine, one common signature:
    base.py                           (x, topk_ids, topk_weights, weights) -> out
    dispatch_kernels.py               wraps grouped_moe_routed (naive/persistent,
                                      bf16/int8)
    vllm_moe.py                       wraps vLLM fused_experts, layout conversion
    sglang_moe.py                     wraps SGLang fused_experts, layout conversion
scripts/run_baseline.py             stage 3.1 (extended): larger seeded prompt
                                    set, gap-split disagreement report
scripts/gpu/phase6_engine_reference.py
                                    stage 3.3 driver: launch server, run the
                                    standard benchmark, tear down, per engine
docs/findings/phase-6/              outputs (default output dir for the new scripts)
```

Adapters import their engine lazily, inside the call, so the module stays
importable -- and unit-testable -- on a machine with neither engine
installed (the same pattern `kernels/bench.py` uses for Triton).

## 5. Testing

Follows the repo's existing table (CLAUDE.md, "Testing policy"):

| Layer | What must be covered |
|---|---|
| Adapters (CPU) | layout conversion is lossless and round-trips (gate+up fused == separate); every adapter accepts identical inputs; a fake engine exercises the refuse-not-time path |
| Race driver (CPU) | an engine that disagrees with the reference is refused, not timed; every output JSON carries full config, engine versions, tuning provenance and hardware; the distribution and token-count grid is what the plan says |
| Gate (CPU) | the gap-split classifier puts near-ties and large-gap disagreements in the right buckets on synthetic logits; a planted large-gap disagreement turns the gate red |
| Race and gate on real hardware (`gpu`) | each adapter meets `assert_matches_reference` at toy, decode- and prefill-shaped dims; a mutation must turn the suite red (Phase 1's rule) |
| Engine reference (`gpu`, paid) | a server that fails to come up, or returns empty completions, refuses to report numbers rather than reporting zeros |

`gpu`-marked tests stay excluded from CI, with the reason visible in the
test, as elsewhere in the repo. `make check` stays green throughout.

## 6. Non-goals

- **A dispatch server.** Phase 7's Rust-router-plus-Python-model-server
  architecture (system design §8, ADR-0003) is where a dispatch server
  belongs; building a throwaway one here just to have an HTTP endpoint
  would measure HF-eager overhead, not the kernel.
- **Swapping dispatch's kernel into vLLM** as a custom MoE backend.
- **H100 or any second hardware class**, and any multi-GPU comparison
  (Phase 3's DeepEP result stands on its own; vLLM's and SGLang's own
  expert-parallel paths are not benchmarked here).
- **Quantized end-to-end serving comparison.** The engine reference is
  bf16 only; int8 is compared at the kernel level, where both engines
  expose the same scheme.
- **Beating the engines.** The goal is a correctly measured, honestly
  reported table.

## 7. Risk, cost, and rollout

**Cost.** One L40 pod, RunPod Secure Cloud, about $0.69/hr as of Phase
5a's rental (checked live at rental time, not assumed). **Budget cap:
$10**, set before the first rental. Three timeboxed stages in the order
above. Cost recorded with `scripts/gpu/provision.py`'s `write_cost_record`.

**Cost discipline, from CLAUDE.md, applied to this phase:**

- Pull every evidence file off the pod before `stop`, not after.
- Never leave the pod running across an unbounded wait -- stop first,
  restart on the answer.
- Treat `stop` as potentially final for that host's disk (a stopped pod
  may be unrestartable); the model download and both engine venvs are
  expensive to redo, so the stage order keeps the most valuable evidence
  first.
- Develop and debug adapters on CPU against fakes first; rent only for
  runs that need the GPU.

**Risks:**

- **Production fused-MoE may win at most or all shapes.** A reportable
  result, not a failure. What would be a failure is an unfair one: a
  baseline running on an untuned or mismatched config, which the tuning
  section above exists to prevent.
- **SGLang API drift.** Its fused-MoE code moved recently. Mitigated by
  pinned versions and one adapter per engine; if the pinned SGLang version's
  API doesn't match what the adapter expects, the adapter fails loudly
  rather than falling back to another code path.
- **Disk and install weight.** A 32GB model plus two engine venvs with
  conflicting torch pins on one pod. Checked against the pod's disk at
  provisioning; models cached on the persistent `/workspace` mount (the
  Phase 0 model-cache-on-the-wrong-disk trap).
- **vLLM/SGLang warmup and CUDA-graph confounds** in the serving numbers.
  Mitigated by discarded, counted warmup requests, and by stating the
  confound in the table where it applies rather than only in prose.
- **Near-tie sensitivity in the correctness gate** (Phase 5b's finding).
  Mitigated by the gap-split gate rule and a threshold fixed before the
  run.
- **Upstream MoE tuner cost.** vLLM's and SGLang's tuners are slow (Phase
  2's search space was ~1,920 configs per batch size). The plan bounds
  tuning to the token counts in the matrix and records how long it ran.

**Rollout.** Branch `phase-6-final-benchmark` (already created), one commit
per plan task, `docs/STATUS.md` updated in the same commit as each piece
of work, one PR at the end. Findings in `docs/findings/phase-6/`. README,
CLAUDE.md and the story-bank gist refreshed at completion, and design §6
amended in the same PR to note that the "this project's own server"
assumption was stale and why it wasn't built.
