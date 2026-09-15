# Phase 0 baseline run — findings

Real run of `scripts/run_baseline.py` (Task 8 of
`docs/plans/2026-09-14-phase-0-baseline-plan.md`) against the full
`deepseek-ai/deepseek-moe-16b-base` model on a rented GPU. Three attempts
failed before the fourth succeeded; none of the failures were in
`dispatch`'s own code (Tasks 1-7 stayed green throughout).

## Infrastructure

- Pod `ku9l6dh8izji0n`, RunPod Secure Cloud, single NVIDIA L40 (48GB),
  `$0.82/hr` (Community Cloud's `$0.69/hr` L40 was out of stock at
  deploy time). Provisioned via the RunPod MCP plugin/OAuth rather than
  the REST client built in Tasks 1-3 — the user's choice at the time
  (see conversation); the REST client itself remains unit-tested against
  fakes but was not live-fire tested this run.
- Repo transferred to the pod via `git archive HEAD | ssh ... tar -x`,
  not `git clone` — the `phase-0-baseline` branch hadn't been pushed yet
  (this repo's convention: a phase branch pushes once, at the end, as
  one PR), and archiving the committed worktree respects that without
  needing a premature push.
- Cost: **$0.35** for 1543s (25.7 min) of pod runtime, all three failed
  attempts included — cheap because the model was already cached locally
  by the second attempt. See `2026-09-15-phase-0-baseline-cost.md` (dated
  a day after the other artifacts here — `write_cost_record` timestamps
  in UTC, and this run crossed UTC midnight while everything else in this
  doc uses the run's local-time date).

## Three bugs, in order

**1. `ImportError: cannot import name 'is_torch_fx_available'`**
DeepSeek's own `trust_remote_code` modeling file
(`modeling_deepseek.py`, last touched ~2024) imports
`transformers.utils.import_utils.is_torch_fx_available`. This repo's
pinned floor, `transformers>=5.17.0` (checked live against pypi.org when
Task 5 added it), has removed the symbol entirely — confirmed by
grepping the installed package (`grep -rn 'def is_torch_fx_available'`
found nothing anywhere in the installed `transformers` tree).

**2. `AttributeError: 'DynamicCache' object has no attribute
'get_usable_length'`**
Installed `transformers==4.57.6` (the last pre-5.0 release — chosen over
an older 4.3x/4.4x version specifically to stay compatible with the
already-locked modern `huggingface_hub==1.31.0`/`tokenizers==0.23.2`/
`safetensors==0.8.0`) to dodge bug 1. That surfaced a second, different
break one line later in the same model file:
`past_key_values.get_usable_length(seq_length)` — `DynamicCache` no
longer has that method even one release before the major bump; it was
renamed to `get_seq_length(layer_idx=0)`. Confirmed the two are
equivalent for this model (DeepSeekMoE uses no sliding-window attention,
so "usable length" and "current sequence length" are the same value)
before patching.

Fix, applied narrowly and **never committed to the repo** — this is a
DeepSeek-repo problem, not a `dispatch` one, and would be wrong to bake
into the general-purpose harness:

```python
from transformers.cache_utils import DynamicCache


def _get_usable_length(self, new_seq_length=None, layer_idx=0):
    return self.get_seq_length(layer_idx)


DynamicCache.get_usable_length = _get_usable_length
```

Applied via a pod-local `_patch_and_run.py` that monkeypatches then
calls `scripts.run_baseline.main()` — not part of the committed
codebase. **Phase 1 will need this same patch** (or a newer model
revision, or a different transformers pin) to load this model again;
worth checking whether DeepSeek has updated `modeling_deepseek.py`
before re-deriving this from scratch.

**3. `OSError: I/O error: ... No space left on device`**
The model download (32.8GB) defaults into `~/.cache/huggingface`, i.e.
`/root/.cache` — the pod's 30GB **ephemeral container disk** — not
`/workspace`, the 50GB+ **persistent volume** the template actually
provisions generously for. `df -h /workspace` alone, checked before
launching, looked fine and was the wrong filesystem to have checked.
Fixed with `HF_HOME=/workspace/hf_cache` on subsequent runs.

**One more snag, not a run failure but worth recording:** `uv run`
re-syncs the venv to `uv.lock` on *every* invocation. The first `uv pip
install transformers==4.57.6` override (meant to fix bug 1) was silently
reverted the moment the next `uv run python3 -c '...'` check ran,
because `uv run`'s auto-sync restored the locked `5.17.0`. Fixed by
calling `.venv/bin/python3` directly for anything meant to bypass the
lockfile temporarily.

## Measured baseline

`deepseek-ai/deepseek-moe-16b-base`, bf16, single NVIDIA L40, unbatched
eager-mode token-by-token decode (`scripts/run_baseline.py` defaults: 3
prompts x 5 repetitions, 64 max new tokens = 15 runs):

| Metric | Value |
|---|---|
| Mean time-to-first-token | 0.355s |
| p50 time-to-first-token | 0.258s |
| p99 time-to-first-token | 1.588s (first run's CUDA warmup) |
| Mean inter-token latency | 0.0746s |
| Mean tokens/sec | 12.75 |

Full machine-readable record:
`docs/findings/2026-09-14-phase-0-baseline-results.json` (committed).

**The reference logits are not committed.** `.gitignore` excludes
`*.safetensors` repo-wide, on purpose, from scaffold time: "GPU
rental / benchmark run artifacts -- raw output belongs in
`docs/findings/` as a written-up, measured result, not as a committed
blob." An 11.9MB tensor blob is exactly what that rule means to keep
out, and this finding doesn't override it. The file exists locally
(`docs/findings/2026-09-14-phase-0-baseline-reference.safetensors`,
produced by this run) but isn't tracked. Phase 1 gets it back by
re-running `capture_reference_logits` against the same model and the
same `DEFAULT_PROMPTS` in `scripts/run_baseline.py` -- reproducible by
construction, not by keeping a blob around -- unless Phase 1 decides
committed reference tensors are worth a deliberate policy change (Git
LFS, a release asset, object storage), which is its call to make, not
an autopilot workaround for Task 8.

This is the "before" number every later kernel optimization (Phase 1's
Triton grouped-GEMM, onward) gets measured against, per this project's
non-negotiable: a benchmark claim states its config alongside the
number, and none of it is estimated.
