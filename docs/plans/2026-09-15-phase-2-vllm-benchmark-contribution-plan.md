# Phase 2: vLLM Benchmark Contribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Phase 1's zipf-vs-uniform expert-load generator into vLLM's
own `benchmarks/kernels/benchmark_moe.py` as an opt-in flag, measure how the
`--tune`-selected Triton config diverges between the two distributions at
`deepseek-ai/deepseek-moe-16b-base`'s shape on a rented GPU, and open a real
upstream PR with that measured data -- plus a small prerequisite PR fixing a
real compatibility gap this work exposed (vLLM's benchmark harness cannot
currently resolve this project's own reference model's shape at all).

**Architecture:** All code changes happen in a sibling fork of
`vllm-project/vllm`, outside this repository -- this repo's `docs/` records
the plan, both PR links, the measured data, and the outcome. Two PRs, in
order: PR 1 is a three-line fix so `benchmark_moe.py`'s `get_model_params()`
recognizes `deepseek-moe-16b-base`'s actual architecture string
(`DeepseekForCausalLM`), which it currently does not -- discovered by
reading the live file, not assumed. PR 2 is the zipf/uniform
`--expert-load-distribution` flag itself, built on top of PR 1, verified
end to end on one rented GPU before it claims any measured divergence.

**Tech Stack:** vLLM's own `main` branch (Python, Triton, Ray, CUDA), a
`uv`-managed venv inside the fork, `VLLM_USE_PRECOMPILED=1` editable
install (no from-source CUDA build). Verification of the pure gating-logit
math happens first in dispatch's own `uv run python` (this repo already
depends on `torch`), before the same function is copied into the fork.

**Spec:** `docs/design/2026-09-15-phase-2-vllm-benchmark-contribution.md`
(all sections); `docs/design/2026-09-14-dispatch-system-design.md` §7 (the
Phase 2 row); `src/dispatch/kernels/bench.py` (the zipf-weighting idea this
phase ports, `_ROUTING_WEIGHTS["zipf"] = 1/i`).

## Global Constraints

- **Two PRs against `vllm-project/vllm`, in order**, decided with the user
  after this plan's research surfaced the compatibility gap: PR 1 (the
  `get_model_params` fix) opens first as a small, obviously-correct,
  independently-reviewable change; PR 2 (the distribution flag) is built on
  top of it and references it, opened once PR 2's own GPU-measured data
  exists. Do not combine them into one PR.
- **Sibling workspace, outside this repo:** clone to
  `/Users/sree/Documents/Projects/vllm-fork` (confirmed empty before this
  plan; a sibling of `dispatch`, per the user's explicit choice during
  brainstorming). vLLM's own git history, branches, and conventions apply
  there -- not dispatch's `make check`, not dispatch's commit-message
  ASCII rule.
- **DCO sign-off on every commit in the fork** (`git commit -s`) -- vLLM
  requires DCO, not a CLA (confirmed live in the design doc's research).
  vLLM's own `pre-commit` config (`.pre-commit-config.yaml`, confirmed live
  2026-09-15: `ruff-check --fix`, `ruff-format`, `typos`, `markdownlint-cli2`)
  governs style on touched files; run it before every commit in the fork.
- **Budget cap $3** for the one GPU rental (Task 5), set here, before
  rental, spot/marketplace only (RunPod Community Cloud), matching Phase
  0-1's cost discipline: spin up, run, capture the evidence, tear down
  immediately. A compute-capability-8.0+ card is required (this repo's own
  established Triton floor) but it does not need to match Phase 0/1's L40 --
  no cross-phase throughput number is being quoted here, only a same-session
  comparison between two distributions, so the cheapest available Ampere+
  spot card is the right choice (Phase 1's own kernel-correctness session
  found an RTX 3090 spot instance at a fraction of the L40's rate).
- **Tuning is restricted to `--batch-size 1 2 4 8 16`**, not vLLM's 18-value
  default sweep. Read live from `get_configs_compute_bound()`
  (`benchmark_moe.py`): the non-ROCm Triton search space is `5 * 4 * 3 * 2 *
  4 * 5 = 2400` configs, tried per batch size. A full 18-batch-size x
  2-distribution x 2400-config sweep would spend the $3 cap on tuning time
  alone, not on producing a usable result. Five batch sizes bracketing
  Phase 1's own finding (grouped-GEMM tile reuse stops mattering above
  `M<=16` at decode) is enough to show whether the winning config diverges
  by distribution, which is what the PR claims.
- **If cost approaches the cap before both distributions finish all five
  batch sizes**, stop after the last fully-completed batch size on the
  current distribution and report exactly what was measured. A partial,
  honest result is acceptable; an estimated one is not.
- **Install via precompiled wheel, not a source build:**
  `VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto`,
  vLLM's own documented contributor path
  (`docs/contributing/incremental_build.md`, confirmed live 2026-09-15).
  This project's two changes are pure-Python edits to
  `benchmarks/kernels/benchmark_moe.py`, a script outside the installed
  package's compiled surface -- a from-source CUDA build is unnecessary
  and would itself risk the cap on compile time alone.
- **Never quote a benchmark number that wasn't measured on this exact run**
  (repo-wide rule) -- applies to both PR bodies and this repo's findings
  doc equally.
- **vLLM code anchors below (line numbers, function names) were read from
  `vllm-project/vllm`'s `main` branch via the GitHub API on 2026-09-15.**
  `main` moves fast; match edits by function/variable name and re-read the
  surrounding code before patching if it doesn't look like what's quoted
  here, don't patch blind against a stale line number.
- **dispatch's own repo receives only docs in this phase**: this plan (already
  written), and `docs/findings/` + `docs/STATUS.md` once the GPU run and
  both PRs exist (Task 7). No `src/dispatch/` changes.

## File Structure

```text
dispatch (this repo):
  docs/plans/2026-09-15-phase-2-vllm-benchmark-contribution-plan.md   # this file
  docs/findings/2026-09-15-phase-2-vllm-benchmark-run.md              # Task 7
  docs/findings/2026-09-15-phase-2-vllm-benchmark-cost.md             # Task 5 (write_cost_record)
  docs/runbooks/phase-2-vllm-benchmark.md                             # Task 5
  docs/STATUS.md                                                      # Task 7

vllm-fork (sibling clone, /Users/sree/Documents/Projects/vllm-fork,
bsreecharanreddy/vllm, forked from vllm-project/vllm):
  benchmarks/kernels/benchmark_moe.py   # Tasks 2 (PR 1), 3+4 (PR 2)
```

Both PRs touch the same file, so they are two branches off vLLM's `main`,
not two branches off each other: `fix/deepseek-v1-get-model-params` (PR 1)
and `feat/moe-benchmark-skewed-load` (PR 2, branched from `main` *after* PR
1's branch is pushed, so PR 2's diff doesn't re-show PR 1's lines; rebase
PR 2 onto `main` once PR 1 merges, before it's opened, if PR 1 lands first).

---

### Task 1: Fork and clone the vLLM sibling workspace

**Files:** none in this repo; creates `/Users/sree/Documents/Projects/vllm-fork`.

- [ ] **Step 1: Fork `vllm-project/vllm`**

```bash
gh repo fork vllm-project/vllm --clone=false
```

Expected: creates `bsreecharanreddy/vllm` (confirmed not to already exist,
checked live 2026-09-15).

- [ ] **Step 2: Clone the fork to the sibling directory**

```bash
git clone git@github.com:bsreecharanreddy/vllm.git /Users/sree/Documents/Projects/vllm-fork
cd /Users/sree/Documents/Projects/vllm-fork
git remote add upstream https://github.com/vllm-project/vllm.git
git fetch upstream
```

- [ ] **Step 3: Create the two branches**

```bash
git checkout -b fix/deepseek-v1-get-model-params upstream/main
git push -u origin fix/deepseek-v1-get-model-params
git checkout upstream/main
git checkout -b feat/moe-benchmark-skewed-load upstream/main
git push -u origin feat/moe-benchmark-skewed-load
git checkout fix/deepseek-v1-get-model-params
```

- [ ] **Step 4: Confirm DCO sign-off is configured**

```bash
git log -1 --format='%an <%ae>'
```

Expected: matches the GitHub account's registered email (`gh api user
--jq .email`, or the noreply address GitHub assigns) -- DCO checks that
the commit author matches the signer, so mismatched `user.email` in this
clone fails vLLM's DCO check on the first push.

- [ ] **Step 5: Install pre-commit**

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install pre-commit
pre-commit install --install-hooks
```

Expected: hooks install cleanly (`ruff-check`, `ruff-format`, `typos`,
`markdownlint-cli2`, `actionlint`, `shellcheck` per
`.pre-commit-config.yaml`, confirmed live 2026-09-15).

---

### Task 2: PR 1 -- recognize `DeepseekForCausalLM` in `get_model_params`

**Files:**


- Modify (fork): `benchmarks/kernels/benchmark_moe.py`

**Verified live, 2026-09-15:** `get_model_params()` (defined in this same
file) branches on `config.architectures[0]`. Its `DeepseekV2ForCausalLM` /
`V3` / `V32` / `V4` branch reads `config.n_routed_experts`,
`config.num_experts_per_tok`, `config.moe_intermediate_size`,
`config.hidden_size` -- but the branch's tuple does not include
`DeepseekForCausalLM`, which is `deepseek-ai/deepseek-moe-16b-base`'s
actual `architectures[0]` (confirmed live against the model's real
`config.json`: `architectures: ["DeepseekForCausalLM"]`, `n_routed_experts:
64`, `num_experts_per_tok: 6`, `moe_intermediate_size: 1408`,
`hidden_size: 2048` -- every attribute the existing branch already reads is
present under the same name). Anything not matching an explicit branch
falls to the `else` clause, which reads `config.num_local_experts` (the
Mixtral/llama4 field name) -- an attribute this model's config does not
have, so `benchmark_moe.py --model deepseek-ai/deepseek-moe-16b-base`
currently raises `AttributeError` before it ever reaches gating-logit
generation.

- [ ] **Step 1: Verify the live HF config from dispatch's own venv (CPU-only, no vLLM needed)**

Run from this repo (`transformers>=5.17.0` is already a dispatch
dependency, so no new install is needed for this check):

```bash
cd /Users/sree/Documents/Projects/dispatch
uv run python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('deepseek-ai/deepseek-moe-16b-base', trust_remote_code=True)
print('architectures:', config.architectures)
print('n_routed_experts:', config.n_routed_experts)
print('num_experts_per_tok:', config.num_experts_per_tok)
print('moe_intermediate_size:', config.moe_intermediate_size)
print('hidden_size:', config.hidden_size)
"
```

Expected: `architectures: ['DeepseekForCausalLM']`, `n_routed_experts: 64`,
`num_experts_per_tok: 6`, `moe_intermediate_size: 1408`, `hidden_size:
2048` -- confirms the *live config object* (not just the raw JSON) exposes
every attribute the fix's target branch reads, under the same names. This
step is the reason the fix below needs no fallback logic: the fields
already match exactly.

- [ ] **Step 2: Write the fix**

In `benchmarks/kernels/benchmark_moe.py`, `get_model_params()`, extend the
existing `DeepseekV2ForCausalLM` branch's tuple:

```python
    elif architecture in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "DeepseekForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "NemotronHForCausalLM",
        "MistralLarge3ForCausalLM",
    ):
```

(Only the `"DeepseekForCausalLM",` line is new -- everything else in the
tuple and the branch body it guards is unchanged.)

- [ ] **Step 3: Lint**

```bash
cd /Users/sree/Documents/Projects/vllm-fork
pre-commit run --files benchmarks/kernels/benchmark_moe.py
```

Expected: clean (`ruff-check`, `ruff-format`, `typos` on this one file).

- [ ] **Step 4: Commit, DCO-signed**

```bash
git add benchmarks/kernels/benchmark_moe.py
git commit -s -m "[Benchmark] Recognize DeepseekForCausalLM in benchmark_moe get_model_params

deepseek-ai/deepseek-moe-16b-base's architectures[0] is
DeepseekForCausalLM (the original DeepSeek MoE, predating V2/V3), which
get_model_params does not recognize -- it falls through to the
Mixtral-shaped default branch and raises AttributeError on
config.num_local_experts before benchmark_moe.py can run at all.

DeepseekForCausalLM's config already exposes n_routed_experts,
num_experts_per_tok, moe_intermediate_size, and hidden_size under the
same names the existing DeepseekV2ForCausalLM/V3/V32/V4 branch already
reads, so recognizing it needs no new branch, just this one addition to
the existing tuple."
```

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin fix/deepseek-v1-get-model-params
gh pr create --repo vllm-project/vllm \
  --base main --head bsreecharanreddy:fix/deepseek-v1-get-model-params \
  --title "[Benchmark] Recognize DeepseekForCausalLM in benchmark_moe get_model_params" \
  --body "$(cat <<'EOF'
## Summary
\`benchmark_moe.py\`'s \`get_model_params()\` recognizes
\`DeepseekV2ForCausalLM\`/\`V3\`/\`V32\`/\`V4\` but not the original
\`DeepseekForCausalLM\` (e.g. \`deepseek-ai/deepseek-moe-16b-base\`), which
falls through to the Mixtral-shaped default branch and raises
\`AttributeError: 'DeepseekConfig' object has no attribute
'num_local_experts'\` -- the script can't run against this model at all
right now.

\`DeepseekForCausalLM\`'s config already exposes \`n_routed_experts\`,
\`num_experts_per_tok\`, \`moe_intermediate_size\`, and \`hidden_size\`
under the same names the existing V2/V3/V32/V4 branch reads (verified
against the model's real config.json), so this is a one-line addition to
the existing tuple, not a new branch.

## Test plan
- \`pre-commit run --files benchmarks/kernels/benchmark_moe.py\`: clean.
- Verified \`deepseek-ai/deepseek-moe-16b-base\`'s live HF config exposes
  every attribute this branch reads, under the same names.
EOF
)"
```

Record the PR URL -- Task 6 and Task 7 both reference it.

---

### Task 3: The zipf/uniform gating-logit generator -- proved standalone first

**Files:**


- Verify only (dispatch repo, not committed): a throwaway script using this
  repo's own `torch` dependency.
- Modify (fork): `benchmarks/kernels/benchmark_moe.py`

**The idea being ported**, from `src/dispatch/kernels/bench.py`
(`_ROUTING_WEIGHTS["zipf"] = 1.0 / arange(1, n+1)`): Phase 1 skewed expert
load by sampling top-k expert indices directly from a multinomial with
weight `1/i` per expert `i`. `benchmark_moe.py` doesn't select experts that
way -- it generates raw gating *logits*
(`gating_output = torch.randn(num_iters, num_tokens, num_experts,
dtype=torch.float32)`, line 165, read live 2026-09-15) which
`fused_topk(x, input_gating, topk, renormalize=...)` (line 277) then runs
through softmax and top-k. The idiomatic port is a per-expert **logit
bias** of `-log(i)`: since `softmax(logit_i) ∝ exp(logit_i)`, adding
`-log(i)` to expert `i`'s logit scales its selection mass by `exp(-log(i))
= 1/i` on top of the existing per-token noise -- the same `1/i` profile
Phase 1 used, expressed in the space this harness already operates in,
not pasted from dispatch's multinomial-sampling code.

- [ ] **Step 1: Prove the bias-in-logit-space math standalone (throwaway, not committed)**

Run from dispatch's own venv (proves the pure-tensor math before it goes
anywhere near vLLM):

```bash
cd /Users/sree/Documents/Projects/dispatch
uv run python -c "
import torch

def generate_gating_output(num_iters, num_tokens, num_experts, *, distribution, seed=0):
    bias = {
        'uniform': lambda n: torch.zeros(n),
        'zipf': lambda n: -torch.log(torch.arange(1, n + 1, dtype=torch.float32)),
    }[distribution](num_experts)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(num_iters, num_tokens, num_experts, generator=g)
    return noise + bias

torch.manual_seed(0)
E, topk, tokens, iters = 64, 6, 256, 20

uniform = generate_gating_output(iters, tokens, E, distribution='uniform')
zipf = generate_gating_output(iters, tokens, E, distribution='zipf')

u_counts = torch.bincount(uniform.topk(topk, dim=-1).indices.reshape(-1), minlength=E)
z_counts = torch.bincount(zipf.topk(topk, dim=-1).indices.reshape(-1), minlength=E)

print('uniform: min/max selection count across experts', u_counts.min().item(), u_counts.max().item())
print('zipf: expert 0 selected', z_counts[0].item(), 'times; expert 63 selected', z_counts[63].item(), 'times')
print('zipf: top-8 lowest-index experts share of all selections:', (z_counts[:8].sum() / z_counts.sum()).item())
"
```

Expected: `uniform`'s min/max are close (near-balanced load, same as the
existing default behavior this PR must not change). `zipf`'s expert 0 count
is far larger than expert 63's, and the lowest 8 of 64 experts (12.5% of
experts) capture well over 12.5% of total selections -- confirms the skew
is real and directionally correct before this function goes anywhere near
vLLM. This script is not committed anywhere; it exists to de-risk Step 2.

- [ ] **Step 2: Add the function to the fork**

On branch `feat/moe-benchmark-skewed-load`, in
`benchmarks/kernels/benchmark_moe.py`, add near the top (after the existing
imports, before `class BenchmarkConfig`):

```python
_EXPERT_LOAD_LOGIT_BIAS: dict[str, Callable[[int], torch.Tensor]] = {
    "uniform": lambda n_experts: torch.zeros(n_experts),
    "zipf": lambda n_experts: -torch.log(torch.arange(1, n_experts + 1, dtype=torch.float32)),
}
EXPERT_LOAD_DISTRIBUTIONS = tuple(_EXPERT_LOAD_LOGIT_BIAS)


def generate_gating_output(
    num_iters: int, num_tokens: int, num_experts: int, *, distribution: str
) -> torch.Tensor:
    """Per-iteration gating logits fed to fused_topk. 'uniform' (this
    script's existing default) draws iid gating logits, so every expert is
    equally likely to land in an unbatched token's top-k. 'zipf' adds a
    per-expert bias of -log(rank) on top of the same noise: since
    softmax(logit) is scaled by exp(bias), this puts roughly 1/rank of the
    selection mass on the rank'th expert, modeling the skewed expert load
    real serving traffic produces instead of this benchmark's current
    near-uniform-only routing.
    """
    if distribution not in _EXPERT_LOAD_LOGIT_BIAS:
        raise ValueError(
            f"unknown --expert-load-distribution {distribution!r}; "
            f"expected one of {EXPERT_LOAD_DISTRIBUTIONS}"
        )
    bias = _EXPERT_LOAD_LOGIT_BIAS[distribution](num_experts)
    return torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32) + bias
```

Add `from collections.abc import Callable` to the existing import block if
not already present (it is not, as of the 2026-09-15 read).

- [ ] **Step 3: Lint**

```bash
cd /Users/sree/Documents/Projects/vllm-fork
pre-commit run --files benchmarks/kernels/benchmark_moe.py
```

Expected: clean.

- [ ] **Step 4: Commit, DCO-signed (plumbing comes in Task 4, same branch, separate commit)**

```bash
git add benchmarks/kernels/benchmark_moe.py
git commit -s -m "[Benchmark] Add zipf/uniform gating-logit generator for MoE benchmarks

Adds generate_gating_output(), an opt-in alternative to this script's
existing torch.randn(...) gating-logit generation. 'uniform' preserves
current behavior exactly; 'zipf' biases lower-index experts toward
disproportionately more selection mass, modeling skewed expert load.
Not yet wired to a CLI flag or the benchmark/tune paths -- that's the
next commit."
```

---

### Task 4: Thread `--expert-load-distribution` through the CLI and both run paths

**Files:**


- Modify (fork): `benchmarks/kernels/benchmark_moe.py`

**Interfaces:**


- Consumes: `generate_gating_output`, `EXPERT_LOAD_DISTRIBUTIONS` (Task 3).
- Produces: `benchmark_config(..., distribution: str = "uniform")`;
  `BenchmarkWorker.benchmark(..., distribution: str)`;
  `BenchmarkWorker.tune(..., distribution: str)`; `--expert-load-distribution`
  CLI flag, default `"uniform"` (existing behavior unchanged when omitted).

Five call sites, all read live 2026-09-15 -- edit by matching the code
shown, not the line number, if `main` has moved:

- [ ] **Step 1: `benchmark_config()` -- add the parameter and use it**

Signature (currently ends `block_quant_shape: list[int] = None,
use_deep_gemm: bool = False,`):

```python
def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool = False,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
    distribution: str = "uniform",
) -> float:
```

Replace line 165's body:

```python
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
```

with:

```python
    gating_output = generate_gating_output(
        num_iters, num_tokens, num_experts, distribution=distribution
    )
```

- [ ] **Step 2: `BenchmarkWorker.benchmark()` -- add the parameter, pass it through**

Add `distribution: str,` to the method signature (after `use_deep_gemm:
bool = False,`), and add `distribution=distribution,` to its
`benchmark_config(...)` call (after `use_deep_gemm=use_deep_gemm,`).

- [ ] **Step 3: `BenchmarkWorker.tune()` -- same, for the tuning loop's call**

Add `distribution: str,` to the method signature (after `use_deep_gemm:
bool,`), and add `distribution=distribution,` to the `benchmark_config(...)`
call inside the `for idx, config in enumerate(tqdm(search_space)):` loop
(after `use_deep_gemm=use_deep_gemm,`).

- [ ] **Step 4: `main()` -- pass `args.expert_load_distribution` into both `_distribute(...)` calls**

In the `if args.tune:` branch, the `_distribute("tune", [...])` tuple
currently ends `search_space, block_quant_shape, use_deep_gemm,` -- add
`args.expert_load_distribution,` after it.

In the `else:` branch, the `_distribute("benchmark", [...])` tuple
currently ends `block_quant_shape, use_deep_gemm,` -- add
`args.expert_load_distribution,` after it.

- [ ] **Step 5: argparse -- add the flag**

After the existing `parser.add_argument("--tune", action="store_true")`
line:

```python
    parser.add_argument(
        "--expert-load-distribution",
        type=str,
        choices=list(EXPERT_LOAD_DISTRIBUTIONS),
        default="uniform",
        help=(
            "Synthetic router gating-logit distribution used to generate "
            "benchmark tokens. 'uniform' (default) matches this script's "
            "existing behavior. 'zipf' biases lower-index experts toward "
            "disproportionately more selection mass, modeling the skewed "
            "expert load real serving traffic produces."
        ),
    )
```

- [ ] **Step 6: Lint**

```bash
cd /Users/sree/Documents/Projects/vllm-fork
pre-commit run --files benchmarks/kernels/benchmark_moe.py
```

Expected: clean.

- [ ] **Step 7: Smoke-check the plumbing without a GPU**

```bash
cd /Users/sree/Documents/Projects/vllm-fork
python3 -c "
import ast
tree = ast.parse(open('benchmarks/kernels/benchmark_moe.py').read())
print('parses cleanly')
"
```

Expected: `parses cleanly` -- a syntax-level check only; the plumbing's
actual behavior isn't provable without vLLM installed, which happens in
Task 5. Do not claim this step proves correctness beyond parsing.

- [ ] **Step 8: Commit, DCO-signed**

```bash
git add benchmarks/kernels/benchmark_moe.py
git commit -s -m "[Benchmark] Wire --expert-load-distribution through benchmark_moe

Threads the distribution choice from argparse through both the --tune
and plain-benchmark paths into benchmark_config's gating-logit
generation. Default is 'uniform', matching current behavior exactly
when the flag is omitted."
```

Do not open PR 2 yet -- Task 6 opens it, after Task 5 has real measured
data to put in the PR body.

---

### Task 5: GPU rental runbook -- verify, then measure the divergence

**Files:**


- Create (this repo): `docs/runbooks/phase-2-vllm-benchmark.md`
- Create (this repo, via `scripts/gpu/provision.py`'s `write_cost_record`,
  reused as-is): `docs/findings/2026-09-15-phase-2-vllm-benchmark-cost.md`

**Budget cap: $3, set above, before this task runs.** Spot/marketplace
only. This task is a live session with the user, not something to run
unattended -- get explicit go-ahead before renting.

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/phase-2-vllm-benchmark.md`:

````markdown
# Phase 2 runbook: vLLM skewed-load benchmark contribution

Budget cap: $3. Spot/marketplace only. Compute capability 8.0+ required
(Triton floor). No cross-phase throughput claim is being made here, so
the cheapest available Ampere+ spot card is the right choice -- check
`list-gpu-types`/RunPod marketplace pricing at rental time rather than
assuming Phase 0/1's L40 rate.

1. Create and wait for the pod (60GB disk: deepseek-moe-16b-base is
   ~32.8GB in bf16, per Phase 0/1's own findings, plus vLLM's own
   footprint):

   ```bash
   uv run python scripts/gpu/provision.py create --name phase-2-vllm-bench \
     --gpu-type <cheapest 8.0+ spot card available at rental time> \
     --image <RunPod CUDA/PyTorch template> --cloud community --disk-gb 60
   uv run python scripts/gpu/provision.py wait <pod-id>
   ```

2. SSH in. Clone the fork and check out `feat/moe-benchmark-skewed-load`
   (Task 1-4's commits; rebase onto `fix/deepseek-v1-get-model-params` if
   PR 1 hasn't merged yet -- see Task 6 Step 1):

   ```bash
   git clone git@github.com:bsreecharanreddy/vllm.git
   cd vllm
   git checkout feat/moe-benchmark-skewed-load
   ```

3. Install, using vLLM's own precompiled-wheel editable-install path (not
   a from-source build):

   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
   ```

   If this fails (precompiled wheel unavailable for the pod's CUDA/torch
   combination), fall back to `uv pip install vllm` (latest released
   wheel) and copy just the locally-edited
   `benchmarks/kernels/benchmark_moe.py` over it -- the two changes touch
   only that script, not the installed package, so a version-mismatched
   released wheel still works as long as its
   `vllm.model_executor.layers.fused_moe` exports the names this script's
   top-of-file imports list. If neither install path works against this
   script's current imports, stop, record why in the findings doc, and do
   not claim a measurement was taken.

4. **Verify PR 1's fix actually unblocks the model** before spending any
   more GPU time -- no `--tune`, one batch size, fast:

   ```bash
   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base \
     --trust-remote-code --batch-size 1 --seed 0
   ```

   Expected: it runs and prints a kernel time, instead of the pre-fix
   `AttributeError`. If it still errors, stop and diagnose before step 6
   -- do not proceed to a paid `--tune` sweep against a broken model
   resolution path.

5. Confirm the model is genuinely unquantized (already known from Phase
   0/1: bf16, no `quantization_config` in `config.json`), so
   `benchmark_config`'s unquantized branch (`args.dtype auto`, the
   default) is what actually runs -- the same branch that would execute
   at real vLLM serve time for this model on this hardware, since none of
   the quantized backends (fp8/int8/int4/deep_gemm) apply without a
   `quantization_config`. This is the design doc's "verify the backend
   that's actually live" check, satisfied by the model's own config
   rather than needing to trace vLLM's kernel-selection code further.

6. Run the actual comparison, uniform first, then zipf:

   ```bash
   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base --trust-remote-code \
     --tune --batch-size 1 2 4 8 16 \
     --expert-load-distribution uniform --save-dir ./tuned-uniform/ --seed 0

   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base --trust-remote-code \
     --tune --batch-size 1 2 4 8 16 \
     --expert-load-distribution zipf --save-dir ./tuned-zipf/ --seed 0
   ```

   Watch elapsed time and the pod's running cost after batch size 1
   completes on each distribution; if the pace projects past the $3 cap
   before both distributions finish all five sizes, stop after the
   current distribution's last completed batch size (per this plan's
   Global Constraints) rather than letting it run to a budget overrun.

7. Compare and capture the evidence:

   ```bash
   diff -u ./tuned-uniform/*.json ./tuned-zipf/*.json
   ```

   Record, per batch size, whether the winning `BLOCK_SIZE_M/N/K`,
   `GROUP_SIZE_M`, `num_warps`, or `num_stages` differs between the two
   distributions. Copy both JSON files and the full console output back
   to this repo's `docs/findings/` before doing anything else.

8. Tear down immediately, right after step 7's evidence is copied out:

   ```bash
   uv run python scripts/gpu/provision.py terminate <pod-id>
   ```

9. From dispatch's own repo root, after the pod's evidence is copied out,
   record the measured cost (reuses `write_cost_record` from Phase 1's
   `scripts/gpu/provision.py` as-is -- no new cost-recording code):

   ```python
   from pathlib import Path
   from scripts.gpu.provision import write_cost_record

   write_cost_record(
       Path("docs/findings"),
       pod_id="<pod-id>",
       gpu_type_id="<gpu-type>",
       cost_per_hour=<rate>,
       duration_s=<measured seconds>,
       note=(
           "Phase 2: verify get_model_params fix + --tune under uniform "
           "and zipf expert-load distributions, batch sizes 1/2/4/8/16"
       ),
       run_label="phase-2-vllm-benchmark",
   )
   ```
````

- [ ] **Step 2: Get the user's explicit go-ahead, then execute the runbook**

Confirm the budget cap and GPU choice with the user before the first
`create` call -- this is a paid action, never taken unattended.

- [ ] **Step 3: Commit the runbook and the cost record**

```bash
cd /Users/sree/Documents/Projects/dispatch
git add docs/runbooks/phase-2-vllm-benchmark.md docs/findings/2026-09-15-phase-2-vllm-benchmark-cost.md
git commit -m "docs: record Phase 2 GPU runbook and cost"
```

---

### Task 6: Open PR 2 with the measured divergence

**Files:**


- Modify (fork): none beyond Task 3-4's existing commits on
  `feat/moe-benchmark-skewed-load` (rebase onto `main` first if PR 1 has
  merged in the meantime).

- [ ] **Step 1: Rebase if PR 1 has merged**

```bash
cd /Users/sree/Documents/Projects/vllm-fork
git fetch upstream
git checkout feat/moe-benchmark-skewed-load
git rebase upstream/main   # only if PR 1 (fix/deepseek-v1-get-model-params) is already merged into upstream/main
git push --force-with-lease
```

If PR 1 hasn't merged yet, leave `feat/moe-benchmark-skewed-load` based on
`fix/deepseek-v1-get-model-params` and note the dependency in the PR body
instead (see below).

- [ ] **Step 2: Open the PR with Task 5's real numbers filled in**

```bash
gh pr create --repo vllm-project/vllm \
  --base main --head bsreecharanreddy:feat/moe-benchmark-skewed-load \
  --title "[Benchmark] Add skewed (zipf) expert-load coverage to benchmark_moe.py" \
  --body "$(cat <<'EOF'
## Summary
Neither this script's --tune search nor SGLang's equivalent currently
models skewed expert load -- gating logits are drawn i.i.d.
(torch.randn), so every tuned config is chosen under near-uniform routing
even though production MoE traffic is not uniform.

Adds an opt-in --expert-load-distribution {uniform,zipf} flag (default
uniform, matching current behavior exactly when omitted). 'zipf' biases
gating logits by -log(rank) per expert, so lower-index experts receive
disproportionately more routed tokens -- modeling skewed load without
changing anything about how tokens are dispatched once routed.

Depends on #<PR 1 number> (DeepseekForCausalLM wasn't recognized by
get_model_params, so this script couldn't run against
deepseek-moe-16b-base at all before that fix).

## Measured data
--tune, deepseek-ai/deepseek-moe-16b-base (E=64, topk=6), batch sizes
<batch sizes actually completed>, single <GPU type actually used>:

<one row per completed batch size: uniform's winning config vs zipf's,
and whether they differ -- filled in from Task 5's actual JSON output,
never estimated>

## Test plan
- pre-commit run --files benchmarks/kernels/benchmark_moe.py: clean.
- Ran --tune under both distributions on a rented <GPU type>; verified
  the flag produces the measured divergence above (or: "produced
  identical configs at these batch sizes" if that's what was measured --
  reported either way).
EOF
)"
```

Record the PR URL and the exact measured-divergence text used -- both go
into Task 7's findings doc verbatim, not re-derived from memory.

---

### Task 7: Findings doc and STATUS.md

**Files:**


- Create: `docs/findings/2026-09-15-phase-2-vllm-benchmark-run.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write the findings doc**

Cover, in `docs/findings/2026-09-15-phase-2-vllm-benchmark-run.md`: both PR
URLs and their state (open / under review / merged / rejected -- whichever
is true at write time, not assumed); the measured per-batch-size config
divergence table from Task 6; the actual GPU type, duration, and cost from
Task 5's cost record; which batch sizes were actually completed if the
runbook's cost-cap stop condition triggered; and one honest sentence on
whether the result supports the PR's claim or not, matching this repo's
practice of writing down a null result plainly if that's what happened
(Phase 1's persistent-kernel tie is the precedent).

- [ ] **Step 2: Update STATUS.md**

Add a "## Phase 2 progress" section following the Phase 0/1 pattern: plan
link, both PR links and their state, the measured divergence summary, and
total GPU cost. Set "## Next step" to reflect whatever is actually next
(PR review follow-up, or Phase 3 planning if both PRs are resolved).

- [ ] **Step 3: Commit**

```bash
cd /Users/sree/Documents/Projects/dispatch
git add docs/findings/2026-09-15-phase-2-vllm-benchmark-run.md docs/STATUS.md
git commit -m "docs: record Phase 2 vLLM benchmark contribution outcome"
```

---

## Self-Review Notes

- **Spec coverage:** design doc §4's in-scope items (port the generator,
  run --tune under both distributions at E=64 decode-relevant tokens, open
  the PR with measured divergence) map to Tasks 3-6. §5's validation items
  (lint, verify the live backend, budget-capped rental, no unmeasured
  numbers) map to Task 5's runbook steps 4-6 and this plan's Global
  Constraints. §6 (sibling workspace, DCO, vLLM's own conventions) maps to
  Task 1 and the Global Constraints. §7 (design doc + plan + findings doc)
  maps to this file plus Task 7. §3's explicitly-out-of-scope items
  (fused_moe.py kernel-selection changes, tuned-config submission, new
  kernel code) are not touched by any task above -- confirmed by re-reading
  the task list against that list.
- **New scope beyond the design doc:** PR 1 (the `get_model_params` fix)
  wasn't in the approved design doc -- it surfaced during this plan's own
  research (reading the live file rather than assuming its structure) and
  was confirmed with the user as a two-PR split before this plan was
  written. Documented here rather than silently added.
- **Ambiguity check:** "decode-relevant token counts" (design doc §4) is
  made concrete as `--batch-size 1 2 4 8 16`, justified against Phase 1's
  own `M<=16` grouping threshold and the search-space-size math in Global
  Constraints -- not left for the executor to guess at rental time.
