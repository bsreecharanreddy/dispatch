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

`docs/plans/` will hold per-phase implementation plans, starting with
Phase 0, once written.

## Current status

System design written, 2026-09-14 (see `docs/STATUS.md`). Phase 0's
implementation plan is written
(`docs/plans/2026-09-14-phase-0-baseline-plan.md`) and its tasks are being
executed on the `phase-0-baseline` branch. `docs/STATUS.md` carries the
task-by-task checklist.

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
full gate: lint, typecheck, test) runs before every push. Specifics for
this project's testing table — what a kernel-correctness test looks like,
how benchmark reproducibility gets asserted — get written once the design
doc exists and there's a real kernel/architecture to write them against.

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
