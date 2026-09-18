# dispatch (CLAUDE.md)

Read this before doing anything in this repo.

## What this project is

**dispatch** is an inference engine for the token-routing layer of
Mixture-of-Experts (MoE) language models — the architecture nearly every
frontier model now uses (DeepSeek, Llama, Mixtral, Grok, Qwen), because it
gets a model's full capacity at a fraction of the compute per token. The
core mechanic is a *router* that sends each token to a handful of
specialist sub-networks ("experts") out of many, and the hard engineering
problem is making that dispatch — routing, grouped-GEMM execution across
skewed expert loads, multi-GPU coordination, memory management — fast in
practice, not just correct on paper.

This project writes a real custom GPU kernel for that dispatch step
(Triton, possibly CUDA), gets it running across multiple GPUs, and
benchmarks the result against production serving engines (vLLM, SGLang) on
real hardware — throughput, p50/p99 latency, and cost per million tokens,
never estimated.

Positioned for **inference engineering** roles specifically (the kind
OpenAI, Anthropic, and NVIDIA hire for under that title) — closer to
GPU/HPC systems engineering than to the data-platform/ML-platform territory
`almanac` and `canopica` cover. Sibling projects, deliberately different
specialization: almanac is a data + ML platform, canopica is a governed
decision system with an AI capability layer, dispatch is the GPU-systems
piece neither of them touches.

## Read this first

**`docs/design/2026-09-14-dispatch-system-design.md`** is the authoritative
architecture doc — model choice (deepseek-ai/deepseek-moe-16b-base), what's
custom vs. reused, kernel scope, the multi-GPU/DeepEP plan and its real
NVLink hardware constraint, benchmark methodology, the 7-phase plan, cost
plan, and explicit scope boundaries. Read it before making any structural
decision. `docs/adr/` carries the specific "why this over that" calls made
along the way (0001: grouped-GEMM over another attention-kernel
reimplementation; 0002: DeepEP over hand-rolled cross-GPU communication).

`docs/STATUS.md` is the authoritative record of implementation state —
updates **in the same commit as the work it describes**.

`docs/plans/` holds per-phase implementation plans, starting with
`docs/plans/2026-09-14-phase-0-baseline-plan.md`.

`docs/findings/` holds measured results, **one subfolder per phase**
(`docs/findings/phase-0/` ... `phase-5b/`), filenames still date-prefixed.
GPU scripts default their output dir to their own phase's folder; a new
phase's scripts should do the same rather than write into the top level.

## Current status

**Phase 0 (baseline) is complete, 2026-09-14**, and merged to `main` via
PR #1. RunPod
provisioning, the benchmark harness, and reference-logit capture are built
and tested (`make check` green, 26 tests). The one real rented-GPU run
(Task 8) measured **12.75 tokens/sec, 0.355s mean TTFT** for
`deepseek-ai/deepseek-moe-16b-base` on a single L40, for **$0.35** total —
after finding and fixing three real environment bugs along the way (a
`transformers` version break in DeepSeek's own remote code, a second break
in the same file one release earlier, and a model-cache-on-the-wrong-disk
trap). Full account: `docs/findings/phase-0/2026-09-14-phase-0-baseline-run.md`.

**Phase 1 (custom Triton grouped-GEMM kernel) is complete, 2026-09-15**
(`make check` green throughout, 73 tests, lint and `mypy --strict`
clean):
`docs/plans/2026-09-15-phase-1-grouped-gemm-plan.md`. A naive and a
persistent, cache-aware Triton grouped-GEMM kernel both passed 25/25
correctness tests on two real GPUs (RTX 3090, then the L40 the measured
run used) with zero kernel bugs found. Swapped into
`deepseek-ai/deepseek-moe-16b-base`'s real 27 MoE layers, both kernels
measured **~65-67% faster decode throughput than DeepSeek's own stock
`moe_infer`** (12.55 -> 20.98 tokens/sec, naive kernel; bf16, single L40,
unbatched eager decode, 3 prompts x 5 repetitions, 64 new tokens) at
perfect mutual top-5 and top-1 logit agreement across every tested
position -- zero measured correctness cost for that speedup. One honest
null result: the persistent kernel's grouped launch ordering (built for
L2 cache reuse) showed no benefit over the naive kernel at the token
count that actually drives decode throughput (a tie at 1 token/step);
a token-count sweep (1-2048) found it does win 3-8% at 16-128 tokens but
loses by up to 14% at 512-2048, a result too workload-specific to
generalize past this project's own single-request decode scope -- exactly
the kind of risk the plan's own risk section flagged before the run
happened. Total GPU
cost across both paid sessions: **$0.65** ($0.06 kernel correctness +
$0.59 the measured run). Full account:
`docs/findings/phase-1/2026-09-15-phase-1-grouped-gemm-run.md`.

**Phase 2 (vLLM benchmark contribution) is complete, 2026-09-15.**
Neither vLLM's nor SGLang's official MoE benchmark modeled skewed (zipf)
expert load; upstreamed an opt-in `--expert-load-distribution` flag to
vLLM's `benchmarks/kernels/benchmark_moe.py`. Measured on one rented RTX
3090: `--tune` under zipf vs. uniform routing picks a **different
winning Triton config at 4 of 5 tested batch sizes**. Open PR:
[vllm-project/vllm#57100](https://github.com/vllm-project/vllm/pull/57100),
awaiting maintainer review. Cost: **$0.77**. This phase landed no code
in `dispatch` itself, only docs — its actual contribution is the
upstreamed PR. Full account:
`docs/findings/phase-2/2026-09-15-phase-2-vllm-benchmark-run.md`.

**Phase 3 (multi-GPU expert-parallel serving) is complete, 2026-09-16**,
merged via PR #3 alongside Phase 2's docs. Real 2x H200 SXM EP over
DeepSeek's own DeepEP (V1 `Buffer`, after V2's `ElasticBuffer` proved
unavailable -- this rental had no GPU Fabric Manager). Correctness gate
passed: perfect top-1 and mutual top-5 agreement vs. a single-GPU
reference. **The measured answer disproved this project's own
hypothesis**: Phase 1's naive-vs-persistent crossover doesn't reproduce
under DeepEP's real per-expert token counts (naive wins at every tested
count on H200; real per-local-expert counts, median 2 max ~20, sit
below Phase 1's smallest tested point of 16). Cost: **$13.03** of a $25
cap. Full account:
`docs/findings/phase-3/2026-09-16-phase-3-multi-gpu-ep-run.md`.

**Phase 4 (disaggregated prefill/decode) is complete, 2026-09-17**,
merged via PR #4. Continuous-batching prefill/decode workers across two
real 4-GPU H100 SXM topologies (co-located 4-rank EP, disaggregated
2+2-rank EP with a real cross-rank KV-cache handoff), both proven
byte-exact against a single-GPU reference. Three real bugs found and
fixed on real hardware, none caught by CPU-only tests. **The measured
result was mixed, not clean**: disaggregated TTFT beats co-located at
concurrency 4 (0.46s vs 1.12s) but loses at concurrency 8 (0.98s vs
0.70s) -- most likely a kernel-warmup confound, reported as genuinely
inconclusive. Cost: **$10.94** of a $40 cap. Full account:
`docs/findings/phase-4/2026-09-17-phase-4-disaggregated-prefill-decode-run.md`.

**Phase 5a (int8 weight-only quantization) is complete, 2026-09-17**, merged
to `main` via PR #5. Self-computed,
per-output-channel int8 quantization extending the Triton kernel itself
(not sourced from bitsandbytes/AWQ). Kernel-level correctness gate
passed on a real L40 (15/15 int8, 25/25 bf16, both first-time on this
hardware). A real bug found live during the GPU session -- quantizing
without freeing the original bf16 expert weights, holding both copies
at once and OOMing a 44GB L40 -- was found and fixed mid-session, and
the final whole-branch review caught two more real precision defects
(quantization arithmetic done in the input's own bf16/fp16 dtype
instead of float32, and an fp16-specific scale-clamp cliff), both fixed
before merge. Measured: int8 kernel **+70.9%** over stock (a throughput
tie with the bf16 naive kernel it's built on, as expected -- weight-only
quantization saves memory, not FLOPs), **49.89%** expert-weight memory
reduction, perfect model-level top-1/mutual-top-k agreement at every
tested position. Cost: **$1.95** of a $5 cap, including two RunPod
Community Cloud pods that hit a real host-level GPU passthrough bug
before a Secure Cloud pod worked. Full account:
`docs/findings/phase-5a/2026-09-17-phase-5a-quantization-run.md`.

**Phase 5b (speculative decoding) is complete, 2026-09-18**, merged to
`main` via PR #6. A shared propose/verify/accept/rollback
loop with two drafters (a 7B draft model, and model-free prompt-lookup) on
top of Phase 5a's int8 target, with an independent plain-greedy oracle for
the baseline. The first GPU session's numbers were withdrawn by the final
review (degenerate all-token-0 baseline); this phase's second session
root-caused it: `transformers==5.17.0` leaves DeepSeek's remote-code RoPE
`inv_freq` buffer uninitialized after `from_pretrained`, poisoning
attention with NaN on GPU. `fix_rope_inv_freq()` now runs inside
`load_model` for every caller. Real re-run on an A40, checked against the
baseline at every k: baseline 25.18 tok/s; **only prompt-lookup beats it**
(34.4-50.9 tok/s across k=1-8; draft-model is below baseline at every k,
19.6-23.0). **Not every config is byte-exact against the baseline** -- in
the 8-config sweep draft-model matched in 13 of 16 (prompt, k)
combinations and prompt-lookup in 9 of 16, and the same k=4 prompt-lookup
config matched 2/4 prompts in one process and 3/4 in another. Root cause,
confirmed by two GPU probes: a genuine near-tied logit under the int8
kernel's floating-point precision, not a logic bug in the loop. **Open
question, deliberately not audited:** Phase 5a's own "perfect top-1/top-k
agreement" claim used this same kernel and a single-run check that could
not see this. Cost: **$12.26** across both sessions, against a $10 cap
that was exceeded on one pod and raised to $20 with disclosure. Full
account: `docs/findings/phase-5b/2026-09-18-phase-5b-speculative-decoding-run.md`.

## One governing principle

**Correctness before speed.** A kernel is not "fast" until it has been
proven numerically correct against a reference implementation within a
stated tolerance — a kernel that's fast because it's silently wrong is a
bug, not a result, and it's invisible unless the correctness check is
explicit. This is the kernel-work equivalent of almanac's point-in-time
correctness: the specific failure mode that's easy to produce by accident,
invisible in a casual read, and the entire reason this project's testing
policy leads with correctness rather than benchmarks.

## Engineering patterns — non-negotiable

These hold regardless of what the design doc ends up choosing:

- **Correctness before speed.** A kernel is not "fast" until it has first
  been proven numerically correct against a reference implementation
  (PyTorch eager, or the relevant paper's reference code) within a stated
  tolerance. A kernel that's fast because it's silently wrong is a bug, not
  a result — this is the kernel-work equivalent of a model that beats its
  baseline by leaking the label.
- **Never quote a benchmark number that wasn't measured**, on this exact
  hardware, at this exact config (batch size, sequence length, quantization
  level, warm-up handled), by this repo. Same rule almanac and canopica
  both carry, applied to throughput/latency/cost instead of row counts and
  file sizes.
- **A benchmark claim states its config alongside the number.** "40%
  faster" is not a result; "40% faster at batch size 32, seq len 2048, on a
  single A100 80GB, vLLM 0.x.y as baseline" is.
- **GPU-dependent tests are marked and excluded from CI**, never silently
  skipped without a reason visible in the test itself — CI has no GPU
  runner, so `gpu`-marked tests are the explicit boundary of what CI can
  verify.

## Testing policy

No implementation code is committed without tests, and `make check` (the
full gate: lint, typecheck, test) runs before every push. This project's
testing table, settled by Phase 1's actual kernel work:

| Layer | What must be covered |
|---|---|
| Kernel contract (CPU) | eager backend == `ReferenceMoE`; every row covered by exactly one tile; no tile crosses an expert boundary; experts that receive zero tokens |
| Kernels (`gpu`) | each kernel meets `assert_matches_reference` against an fp32 reference at toy, decode- and prefill-shaped dims, fp16 and bf16; a mutation must turn the suite red |
| End to end (`gpu`, paid) | mutual top-5 logit agreement with a same-session stock run; a kernel run that patches no layers refuses to run |
| Benchmarks | `triton.testing.do_bench`; a backend that disagrees with the eager one is refused, not timed; every JSON carries its full config |

## Cost discipline

GPU rental is real money, same discipline as almanac's Azure spend, adapted
to how GPU rental actually works:

- **Marketplace/spot instances only** (RunPod Community Cloud, Vast.ai) —
  meaningfully cheaper than dedicated on-demand, and this project's runs
  are short and checkpointable.
- **Develop and debug on a free tier first** (Colab, Kaggle's free GPU
  hours) — rent only for the runs that actually need multiple GPUs or a
  specific accelerator class.
- **Pick the smallest model that proves the architecture.** A frontier-scale
  MoE model needs a full node of GPUs before anything runs; a small real
  MoE model (10-20B total params, fits on one 40-80GB GPU) proves the same
  expert-routing/dispatch mechanics at a fraction of the cost, and can
  still be deliberately split across cheaper GPUs to prove the multi-GPU
  story.
- **Budget cap set before the first rental, not after.**
- **Spin up, run the measured session, capture the evidence, tear down
  immediately.** Never leave a rented GPU idle between sessions.
- **Cost per run is measured and reported**, same as every other number.
- **Never leave a pod running across an unbounded wait** (a question to the
  user, a background task with no deadline). Phase 5b's original $10 cap
  was exceeded on exactly this: one pod left running while waiting on a
  reply. Stop it first, restart on the answer.
- **Pull every evidence file off the pod before `stop`, not after.** The
  repo clone lived outside the persistent mount (`/workspace`) and did
  not survive a stop/start; only the JSONs whose contents had already
  been printed to captured stdout could be recovered.
- **A stopped pod may be unrestartable** ("not enough free GPUs on the
  host machine") -- hit on three separate pods in Phase 5b (one started
  on a third attempt, two never did). Treat `stop` as potentially final
  for that host's disk.

GPU provisioning lives in `scripts/gpu/` as lightweight scripts calling
provider APIs directly — not Terraform. RunPod/Vast.ai's Terraform
providers are much less mature than AWS/Azure's, and the actual workflow
(rent, run a benchmark, tear down) is a better fit for a script than for
Terraform state management.

## Language and tooling

Python 3.12+. `uv`, `ruff`, `mypy --strict`, `pytest`. PyTorch and Triton
(and CUDA directly, if the design doc calls for it) get added once the
design doc lands — each version floor checked live against pypi.org's JSON
API when it's added, not guessed or carried over from anywhere else.

## Development workflow

```text
new subsystem?
├── yes → brainstorm → dated doc in docs/design/ → approval
│         → implementation plan in docs/plans/ → then code
└── no  → is there an approved plan task for it?
          ├── yes → TDD: failing test → implement → full suite green
          │         → STATUS.md row in the SAME commit → one commit per task
          └── no  → stop and ask; don't freelance scope
```

## Conventions

- **One branch per phase**, carrying the whole phase, pushed once as a
  single PR — same convention almanac settled on (written down there after
  getting it wrong twice; starting dispatch already following it rather
  than re-learning it).
- **One commit per completed task**, not one bundled commit per phase.
- **`docs/STATUS.md` updates in the same commit as the work.**
- Conventional commit prefixes (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`).
- **Commit messages are plain ASCII: `--`, never an em-dash.** Docs use `—`
  freely; the git log does not.
- **README.md, CLAUDE.md, and the story-bank gist get refreshed
  proactively, not on request** — same standing instruction as almanac and
  canopica, carried over rather than re-learned: at every natural stopping
  point, ask what changed today that a reader would want to know, not
  whether the file is still technically accurate (a stale file passes that
  check trivially — see almanac's CLAUDE.md for the incident history behind
  this rule).

## Interview story bank

A **secret** GitHub Gist holds the STAR-format story bank for this project
— a dedicated gist, separate from almanac's, so each project's incident
history stays separable. Its id lives in `.claude/story-bank-gist-id`
(gitignored).

Same trigger-discipline note as almanac: this reminder has a documented
history of not firing on its own when left to memory, which is why
`.claude/hooks/story-bank-reminder.sh` exists as a mechanical hook rather
than being trusted to happen unprompted.

## `.claude/` tooling — carried over deliberately, not automatically

almanac's CLAUDE.md is explicit that its skills exist because specific
incidents already happened on almanac (a leakage bug, a paid-window
mistake, two design-decision errors) — "nothing here is anticipatory."
Copying those skills into this repo verbatim would violate the exact
principle they demonstrate: dispatch hasn't earned any of them yet.

What *is* carried over here, because it's mechanical rather than a
judgment call:

- **`hooks/check-status-md-commit.sh`** — warns before a commit that
  changes non-docs files without `docs/STATUS.md` staged alongside it.
- **`hooks/story-bank-reminder.sh`** — reminds, on a `docs/STATUS.md`
  commit, to check whether the task earned a story-bank entry.

`.claude/skills/` starts empty. It gets its first skill the same way
almanac's did — a real incident happens here, costs something, and the
skill is written to prevent it recurring, not in anticipation of one.

## graphify

Same tool, same setup as almanac and canopica: a local knowledge graph
under `graphify-out/`, gitignored, rebuilt on every commit once
`graphify hook install` has been run once for this clone. Run `/graphify`
once there's real code to graph.
