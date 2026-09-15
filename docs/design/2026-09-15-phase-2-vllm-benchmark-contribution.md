# Phase 2 — vLLM benchmark contribution: skewed expert-load coverage

Status: approved, 2026-09-15. Written after a brainstorming session that
researched real (not assumed) gaps in `vllm-project/vllm`'s and
`sgl-project/sglang`'s current MoE kernel code and issue trackers, live via
`gh api`/`gh pr list`/GitHub source reads rather than training-data memory.

## 1. What Phase 2 actually is

The system design doc (`docs/design/2026-09-14-dispatch-system-design.md`,
phase table) scopes Phase 2 as: "Attempt an upstream PR (vLLM or SGLang) —
starts by finding a real gap in what they already ship, not duplicating
it." It did not specify the gap; finding one honestly was this phase's
first task, done via two rounds of live research before any design was
proposed.

## 2. Research findings (why this scope, not a kernel PR)

Both `vllm-project/vllm` and `sgl-project/sglang` have MoE kernel backend
spaces far more saturated than the original design line implied — vLLM's
`fused_moe/` alone has 30+ backend files (cutlass, deepgemm, flashinfer,
marlin, aiter, trtllm, and multiple quantized variants). A new grouped-GEMM
kernel PR duplicating compute that already exists three or four times over
is not a realistic accepted contribution, and pretending otherwise would
violate this repo's "never quote a benchmark/claim that wasn't measured
and verified" discipline applied to the claim "there is a gap here" itself.

The one real, currently unclaimed gap found: **neither project's official
MoE benchmark/tuning harness models skewed (zipf) expert load** — both
generate router gating via `torch.randn(...)`, i.e. near-uniform only.
Verified by grep (`zipf|skew|imbalance|power.law` — zero hits) in both
`benchmarks/kernels/benchmark_moe.py` (vLLM) and
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` (SGLang),
and confirmed no open issue requests it in either tracker.

Phase 1 already built exactly this — a zipf-vs-uniform expert-load
generator, used for the token-count sweep in
`docs/findings/2026-09-15-phase-1-grouped-gemm-run.md` — making this a
direct, reusable contribution rather than new work invented for the
occasion.

One finding was explicitly *ruled out* as a gap rather than confirmed as
one: vLLM's own Triton `fused_moe.py` already implements the same
L2-cache-aware grouped-launch idea Phase 1's persistent kernel built
(`GROUP_SIZE_M`), and its own tuning already disables grouping at
decode-shaped `M <= 16` — independently corroborating Phase 1's null
result rather than exposing something unaddressed. This is cited in the
PR as honest motivating context, not pitched as a bug or a gap.

## 3. Target: vLLM, not SGLang

Chosen on evidence, not project size. Verified live:

- **Named maintainer.** vLLM's `CODEOWNERS` explicitly assigns
  `/vllm/model_executor/layers/fused_moe` to @mgoin, @pavanimajety,
  @zyongye. @mgoin personally authored and merged PR #46642, "\[Kernel]\[MoE]
  Tune block-FP8 fused MoE for low-batch decode" — touching both
  `benchmark_moe.py` and `fused_moe.py`, almost exactly this PR's
  territory. SGLang has no CODEOWNERS entry for the equivalent path.
- **Contribution bar.** vLLM requires DCO sign-off only (no CLA); thin but
  clear `CONTRIBUTING.md`; no requirement to file an issue before a PR
  (observed on real merged config/bugfix PRs with no prior linked issue).
  SGLang has no DCO/CLA check found and a sprawling multi-hundred-job CI
  matrix (NPU/ROCm/MUSA/XPU tiers) with non-obvious gating — one real
  merged PR showed "-finish" aggregator jobs reporting FAILURE alongside
  the merge, confusing for a first-time external contributor to read.
- **Review speed.** Config-scale PRs on vLLM: under 2 hours to first
  review, same-day to next-day merge. SGLang's raw merge speed is
  comparable but the review trail is noisier and harder to attribute to a
  specific reviewer.
- **Rejections.** No maintainer-pushback rejections found in either
  project's benchmark/tuning-config history in the last 3-6 months; the
  two closed-unmerged PRs found in this area were both self-closed by
  their own authors for unrelated reasons (a duplicate PR; a config
  targeting a kernel path the model doesn't actually select at serve
  time — a real methodology lesson folded into Section 5's validation
  step).

## 4. Scope

**In scope:**

- Port Phase 1's zipf-vs-uniform expert-load generator into
  `benchmark_moe.py`'s gating-logit generation, as a new opt-in CLI option
  (uniform stays the default — no existing behavior changes), written in
  vLLM's own code style rather than pasted from this repo.
- Run `benchmark_moe.py --tune` under both distributions at
  `deepseek-ai/deepseek-moe-16b-base`'s shape (E=64 routed experts) across
  decode-relevant token counts, on a rented GPU, and record whether/how the
  winning tuned config diverges between uniform and zipf load.
- Open the PR with that measured divergence stated plainly, citing
  vLLM's existing `GROUP_SIZE_M` decode threshold (Section 2) as honest
  corroborating context.

**Explicitly out of scope for this PR** (each is a separate, later,
contingent decision — not promised up front):

- Changing `fused_moe.py`'s kernel-selection logic itself (stretch-goal
  "Approach C" from the brainstorm — only pursued as a distinct follow-on
  PR if this PR's own data clearly justifies it).
- Submitting tuned configs for GPU/shape combos this project has data for
  ("Approach B" — weaker, closer to a token-gesture contribution;
  deferred, not committed).
- Any new kernel code. This is a benchmarking-tool contribution, not a
  kernel contribution — consistent with Section 2's finding that the
  kernel-backend space is already saturated.

## 5. Validation

- Run vLLM's own existing lint/tests for the touched file, if any exist,
  before opening the PR.
- Before opening the PR: verify which kernel backend
  `deepseek-ai/deepseek-moe-16b-base` actually selects at serve time in
  vLLM (the exact mistake that sank one of the two rejected PRs found in
  Section 3's research) — tune against the backend that's actually live,
  not an assumed one.
- One short GPU rental, spot/marketplace, **budget cap $3** (set here,
  before rental, matching Phase 1's kernel-correctness session — this task
  is one existing script run under two distributions, no kernel
  iteration, so it should cost less): spin up -> run -> capture evidence
  -> tear down immediately, same cost-discipline rules as Phases 0-1 — to
  prove the new flag produces real, sane, divergent data before the PR
  claims it does. No benchmark number in the PR body that wasn't measured
  on real hardware by this exact run.

## 6. Workspace and process

The actual code change happens in a sibling clone of
`bsreecharanreddy/vllm` (forked from `vllm-project/vllm`), **outside this
repository** — vLLM has its own massive codebase, git history, and
conventions that don't belong nested inside dispatch's own git tree.
dispatch's `docs/` records the plan, the PR link, the measured data, and
the outcome, the same way `docs/findings/` already records GPU run
results — it does not contain vLLM's code.

Inside the vLLM fork: DCO-signed commits and vLLM's own commit/branch
conventions apply, not dispatch's (e.g. dispatch's ASCII-only commit
message rule is a dispatch convention, not a vLLM one). Let vLLM's
CODEOWNERS auto-request @mgoin rather than hand-tagging.

## 7. Deliverables in this repo

- This design doc.
- An implementation plan in `docs/plans/` (next step).
- A findings doc in `docs/findings/` once the GPU run and PR are done,
  recording the measured config divergence, the GPU cost, and the PR's
  outcome (opened / under review / merged / rejected, honestly reported
  whichever it is) — same pattern as
  `docs/findings/2026-09-15-phase-1-grouped-gemm-run.md`.

## 8. Explicit non-goals (project-level)

- This phase is not a claim that vLLM's or SGLang's MoE kernels are
  deficient. Section 2's research found the opposite: both are mature and
  well-covered. The contribution is a real but modest gap in benchmark
  *coverage*, not kernel *quality*.
- Not a vehicle for landing dispatch's own kernel code upstream. No code
  from `src/dispatch/kernels/` is contributed as-is; only the
  load-distribution *idea*, reimplemented idiomatically for vLLM's script.
