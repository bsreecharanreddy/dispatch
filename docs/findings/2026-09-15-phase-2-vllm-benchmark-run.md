# Phase 2 — vLLM skewed-load benchmark contribution: run and outcome

Plan: `docs/plans/2026-09-15-phase-2-vllm-benchmark-contribution-plan.md`.
Design: `docs/design/2026-09-15-phase-2-vllm-benchmark-contribution.md`.

## Outcome

**One PR open, not two.** The plan and its approved design both called
for two separate PRs against `vllm-project/vllm` -- a small prerequisite
fix, then the feature. While opening the first one, reading its
automated welcome comment surfaced `vllm-project/vllm`'s own `AGENTS.md`,
which explicitly governs AI-assisted contributions and states breaching
it "can result in automatic banning." Checked honestly against it, the
standalone fix PR failed on two counts: it never disclosed AI assistance
(mandatory, not optional) and its own "no low-value busywork PRs" policy
explicitly discourages a standalone PR for an isolated one-tuple-entry
change -- their own listed example, "one mutable default," is the same
size and shape. Its own stated exception is "mechanical cleanups...
bundled with substantive work," which is exactly what commit 1 of a
combined PR is.

Fixed by closing the first PR
(`vllm-project/vllm#57072`, closed with an explanation) and re-opening as
a single PR with the fix as its first commit:
**`vllm-project/vllm#57100`**, open as of this writing. All three commits
carry an `Assisted-by: Claude (Anthropic)` trailer, the PR body states
plainly that AI assistance was used, includes the duplicate-work search
commands and their (clean) results, and states the model-evaluation
question is not applicable since no serving/inference code path is
touched.

**This is the real lesson worth recording**: a project's own contribution
policy is itself a fact to check live, the same discipline this project
already applies to model configs and library versions -- not something to
assume from general open-source norms. It was checked here only after a
PR was already open, which is later than it should have been; the
duplicate-work search specifically should run *before* opening any PR,
not after.

## What was measured

`benchmarks/kernels/benchmark_moe.py --tune`, `deepseek-ai/deepseek-moe-16b-base`
(E=64 routed experts, topk=6, `--tp-size 1` to model a real single-GPU
serving shape rather than the script's `tp_size=2` default), batch sizes
1/2/4/8/16, on one rented NVIDIA GeForce RTX 3090 (RunPod community
cloud). Both the `uniform` (existing default) and `zipf` (new flag)
distributions were run against the identical ~1,920-config Triton search
space per batch size.

**Winning config per batch size** (`BLOCK_SIZE_M/BLOCK_SIZE_N/BLOCK_SIZE_K/GROUP_SIZE_M/num_warps/num_stages`):

| batch size | uniform | zipf | diverges? |
|---|---|---|---|
| 1  | 16/32/128/64/4/2 | 16/32/128/64/4/2 | no |
| 2  | 16/32/64/16/4/3  | 16/32/64/32/4/3  | GROUP_SIZE_M |
| 4  | 16/64/64/32/8/4  | 16/32/64/64/8/4  | BLOCK_SIZE_N, GROUP_SIZE_M |
| 8  | 32/64/128/1/4/2  | 16/32/256/1/4/3  | BLOCK_SIZE_M, BLOCK_SIZE_K, num_stages |
| 16 | 16/32/64/1/4/3   | 16/32/64/16/4/3  | GROUP_SIZE_M |

**4 of 5 batch sizes land on a different winning config once routing is
skewed instead of uniform.** Only batch size 1 -- the smallest, where
`GROUP_SIZE_M`'s L2-reuse effect already has the least room to matter --
matched exactly. This directly supports the PR's claim: tuning
`benchmark_moe.py` under its current uniform-only default can select a
config that is not actually the fastest one for skewed production
traffic. It is a real, moderate result, not a dramatic one -- most of the
divergence is in `GROUP_SIZE_M` and `num_stages`, not a wholesale
different tile shape, and is reported at exactly that strength rather
than oversold.

Raw output: `docs/findings/2026-09-15-phase-2-tuned-uniform.json`,
`docs/findings/2026-09-15-phase-2-tuned-zipf.json`.

## Cost and session account

Full record: `docs/findings/2026-09-16-phase-2-vllm-benchmark-cost.md`
(via `scripts/gpu/provision.py`'s `write_cost_record`, reused unmodified
from Phase 1).

- Pod: RTX 3090, RunPod community cloud, $0.22/hr.
- Duration: ~3.50 hours (setup, the `get_model_params`-fix verification,
  both `--tune` sweeps). Uniform sweep alone: 8838.73s (~2h27m). Zipf
  sweep: 1985.18s (~33m) -- faster than uniform, most likely a warm
  Triton JIT/kernel cache carried over from the first sweep rather than
  anything about the distribution itself.
- **Total cost: $0.77**, against a **$3 cap** set before rental.

## Real environment issues found and fixed, not predicted in the plan

- **vLLM's precompiled-wheel install path failed** for this exact commit
  (`VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto`
  returned HTTP 404 fetching wheel metadata for the merge-base commit +
  `cu129`). Fell back to the plan's own documented alternative: `uv pip
  install vllm` (latest released wheel, 0.29.0) plus running the
  locally-edited `benchmark_moe.py` from the checked-out fork.
- **Local-package shadowing.** With the released wheel installed but the
  fork's own `vllm/` source tree still sitting in the working directory,
  both the main process and Ray's worker subprocesses (which inherit the
  driver's working directory) resolved `import vllm` to the incomplete
  local source tree instead of the installed wheel, failing on the
  missing compiled extension `vllm._C_stable_libtorch`. Fixed by moving
  the local `vllm/` package directory out of the way
  (`mv vllm vllm_src_unused`) once it was clear the released wheel, not
  the local source, was what would actually run -- not a plan-anticipated
  step, but the direct fix once diagnosed.
- **`ray` was not pulled in** by the released `vllm` wheel's own
  dependencies; needed an explicit `uv pip install ray`.
- **RunPod's SSH access for this pod was proxy-only** (`ssh.direct` was
  `null` on creation; only `ssh.direct` variant supports plain remote-exec
  `ssh host "command"`). The proxy accepts only an interactive PTY
  session, so every remote command in this session was driven by piping
  a heredoc into `ssh -tt ... <<'EOF' ... EOF` rather than one-shot
  `ssh host cmd`, and `scp` did not work through the proxy at all --
  files were pulled back by having the remote side `base64 -w0` them and
  decoding the captured PTY transcript locally.
- **The plan's disk-size estimate (60GB) was wrong, caught before
  renting, not after.** `benchmark_moe.py` never loads real model
  weights -- only the model's small HF `config.json` plus synthetic
  random tensors sized to match it (~1GB total for this shape) -- so the
  60GB figure, carried over from Phase 0/1's very different
  real-model-download workflow, did not apply. Rented at 20GB instead.

## Deviations from the plan, and why

- **One PR, not two** -- covered above (AGENTS.md).
- **GPU choice**: an RTX 3090 at $0.22/hr, not an L40, chosen live against
  RunPod's actual catalog at rental time -- no cross-phase throughput
  number is being quoted in this phase, so the cheapest available
  Ampere+ card was the right call, consistent with the plan's own stated
  reasoning.
- **`--tp-size 1`** added to every benchmark invocation, not present in
  the plan's drafted commands -- the script defaults to `tp_size=2`,
  which shapes the synthetic weights for a 2-way-sharded deployment. A
  single rented GPU models a real single-GPU serving shape more
  faithfully with `tp_size=1`, so this was corrected at execution time.

## Non-goals held

No claim beyond what was measured: the PR states the divergence exactly
as found (moderate, concentrated in `GROUP_SIZE_M`/`num_stages`, 4 of 5
batch sizes), not amplified. No kernel code, tuned-config submission
(Approach B), or `fused_moe.py` kernel-selection change (Approach C) was
attempted -- consistent with the design doc's explicit scope.
