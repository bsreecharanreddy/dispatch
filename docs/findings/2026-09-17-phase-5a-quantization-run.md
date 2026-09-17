# Phase 5a int8 quantized kernel run -- findings

Design: `docs/design/2026-09-16-phase-5a-quantization.md`. Plan:
`docs/plans/2026-09-16-phase-5a-quantization-plan.md`. Branch
`phase-5a-quantization`.

## Infrastructure

Three RunPod pods rented this session, in order:

1. `4u8rfiy5dz5vj6` -- L40, Community Cloud. Out of stock the instant it
   came up; unusable, abandoned.
2. `wmv06rf1l2sm8s` -- L40S, Community Cloud. Booted, but `cuInit()` (the
   raw CUDA driver API, called directly via ctypes to rule out anything
   about this project's own torch/venv choice) returned
   `CUDA_ERROR_UNKNOWN` (999) -- even against the base image's own
   preinstalled torch. A pod restart changed which `/dev/nvidiaN` device
   node appeared but did not fix it. **A genuine RunPod Community Cloud
   host-level GPU passthrough bug**, not a code or environment issue on
   this project's side -- new finding this session, distinct from every
   environment issue Phases 0/1/3/4 documented.
3. `924l6eft4d8251` -- L40, **Secure Cloud**. Same raw `cuInit()` test
   returned `0` immediately. Everything below ran here.

Abandoned Community Cloud entirely for this session after two pods hit
the same bug. Whether it's specific to this account, this data center, or
Community Cloud generally is unknown -- worth a quick stock/host check
before assuming Community Cloud is usable again in a later phase.

## Kernel-level correctness gate: passed

On the working Secure Cloud L40 (driver 580.178.04, CUDA 13.0):

- `pytest -m gpu tests/unit/test_grouped_gemm_int8_kernel.py`: **15/15
  passed** -- the int8 kernel's first execution on real hardware.
- `pytest -m gpu tests/unit/test_grouped_gemm_kernel.py`: **25/25
  passed** -- the existing bf16 naive/persistent kernels, re-verified on
  this card.
- `pytest -m "not gpu"` (full CPU suite): **118 passed** (later 119, then
  122, after the fixes below).

**Not run this session, and worth saying plainly rather than leaving
implicit:** the design doc's own testing table calls for a mutation
check on the kernel (e.g. forcing every tile to read the wrong expert's
scale) turning the suite red, the same bar Phase 1 met and recorded
(`docs/STATUS.md`'s Phase 1 section). Phase 5a's GPU session ran out of
scope before this was attempted -- the 15/15 gate proves the kernel
matches its reference on real, varied per-channel-scale data, but it
does not independently prove the suite is capable of catching a broken
per-channel scale index the way Phase 1's explicit mutation run did.
Flagged by the final whole-branch review; deferred to a future GPU
session rather than invented after the fact.

## A kernel-precision finding from Task 3's fix round, recorded here

`grouped_matmul_int8`'s first implementation dequantized each int8
weight tile to fp32 and cast the activation tile to fp32 on every
K-iteration before calling `tl.dot` -- which silently drops `tl.dot`
off the native bf16/fp16 tensor-core path onto the much slower fp32/TF32
one. Caught by task review before this ever ran on real hardware: **a
kernel comparison that timed this would have measured "quantization is
slower," and the true cause would have been the dot's precision, not
quantization itself.** The design doc named exactly this risk in
advance ("if this proves harder than expected inside the tile loop,
that is itself a reportable finding").

Fixed by casting the int8 tile to the activation's own native dtype
before `tl.dot` and hoisting the per-output-channel scale multiply to
run once *after* the K-loop rather than once per iteration -- exact,
because the scale is K-invariant, and strictly more accurate than the
original (one rounding instead of `K / BLOCK_K` of them). No before/after
timing was measured on real hardware for this specific change in
isolation -- the fix landed before Task 6's rental, so the 20.72 tok/s
measured for the quantized kernel already reflects the corrected,
native-precision version; there is no "slow" data point to compare it
against without deliberately re-introducing the bug.

## A real bug found mid-session, fixed before the measured run

The first quantized-model run OOMed inside `patch_moe_infer_quantized`
itself (`stack_expert_weights` -> `torch.stack`), before any generation
happened -- a pure static-weight memory problem, not an activation or
batch-size one.

**Root cause:** `stack_expert_weights` re-points every expert's
`nn.Linear.weight` at a *view* into one shared bf16 tensor rather than
copying it, specifically so building that stack costs no extra memory
(this is exactly what makes the existing bf16 `patch_moe_infer` path
memory-neutral). `patch_moe_infer_quantized` quantized that stack into
int8 but never dropped those views -- so the model held the bf16
originals *and* their int8 copies simultaneously, for every layer, for
the rest of the process. Quantizing was making memory usage strictly
worse, not better, and it OOMed a 44GB L40 at DeepSeekMoE-16B's real
scale.

This is a real defect in already-task-reviewed code (Task 4), found only
by a real hardware run at real model scale -- the same pattern Phases 3
and 4 both hit (see their findings docs). Ruled and fixed directly rather
than re-dispatched through the SDD task loop, per that skill's own
authority for judgment calls mid-session:

- Added `_free_expert_weights` (`src/dispatch/kernels/integration.py`):
  replaces each expert's `gate_proj`/`up_proj`/`down_proj` `.weight` with
  a 0-element `Parameter` immediately after quantizing, since the
  quantized closure never reads `experts` again after patch time.
- Added `test_patch_quantized_frees_the_original_bf16_expert_weights`
  (CPU-only): asserts `.weight.numel() == 0` on every expert/projection
  after patching.
- Verified the existing independent-oracle test
  (`test_patched_quantized_model_matches_weights_quantized_in_place`,
  which exercises a full patched forward pass) still passes -- proof that
  `experts` is genuinely unused after patching, not just unused in the
  common case.
- `make check` green (119 passed, 2 skipped, 1 deselected; lint and
  `mypy --strict` clean). Commit `c75da79`.

Re-run after the fix: one transient `CUDACachingAllocator` OOM warning
(the allocator retrying, not failing) instead of a crash, and the run
completed cleanly.

## Two more real bugs, found by the final whole-branch review

The final review (broader and more architectural than the per-task
reviews above, dispatched after all seven tasks were otherwise
complete) found two real precision defects in `quantize_per_channel_int8`
that no per-task review or CPU test had caught, because the existing
round-trip test only ran fp32 and the zero-channel tests only checked
for NaN, never range utilization:

1. **Quantization arithmetic ran in the weight's own dtype.** `absmax`,
   `scale`, and the quotient were all computed in bf16/fp16 rather than
   float32. Measured: bf16 widens round-trip error to up to **1.5x** the
   ideal 0.5-quantization-step bound (bf16's 8-bit mantissa is the
   cause). The measured run's own perfect model-level agreement shows
   this was empirically tolerable at this model's real scale -- but
   nothing in the suite would have noticed if it regressed further.
2. **The all-zero-channel guard clamped every small channel, not just
   zero ones.** For fp16 specifically, the guard floored `scale` at
   `torch.finfo(torch.float16).tiny` (~6.1e-5). Any channel with
   `absmax < 127 x 6.1e-5 ≈ 0.00775` got its scale clamped *upward*,
   silently under-using the int8 range -- demonstrated on a channel with
   absmax ~0.001: ideal scale ~7.9e-6, clamped to ~6.1e-5, cutting the
   largest value's int8 code from 127 down to 16 (a 2.4% relative error
   where <0.4% was achievable). bf16 was immune (its own `tiny` is
   ~1.18e-38), which is why the measured run above was unaffected --
   this is CLI-reachable today via `--dtype float16 --moe-kernel
   quantized`.

Fixed by computing `absmax`/`scale`/the quotient in float32 regardless
of the weight's own dtype, and changing the zero-channel guard to set
`scale = 1.0` only when `absmax == 0` exactly, rather than clamping
toward a dtype-dependent floor. Added a round-trip test parametrized
over float32/bfloat16/float16 (one bound that must now hold for all
three) and a regression test proving a small-but-nonzero fp16 channel
still uses the full int8 range. `make check` green (122 passed, 2
skipped, 1 deselected). Commit `589834c`.

**One caveat this doesn't resolve:** computing in float32 means peak
GPU memory during `patch_moe_infer_quantized` is not actually reduced
by quantizing -- `load_model` loads the full bf16 model, then each
layer transiently doubles to a float32 copy for the quantization math
before `_free_expert_weights` releases the bf16 original. The **49.89%**
figure below is the *steady-state* expert-weight footprint after
quantizing, not a claim about peak memory during the process that gets
there; a genuinely lower-peak path would need to load pre-quantized
weights directly rather than quantizing a fully-materialized bf16 model
in place. Out of scope for this phase; worth naming for whoever picks
up quantize-on-load next.

## Three-way measured run

`deepseek-ai/deepseek-moe-16b-base`, bf16, single NVIDIA L40 (Secure
Cloud), unbatched eager-mode decode, 15 runs per configuration (3 prompts
x 5 repetitions, 64 max new tokens), `transformers==4.57.6`, 27/27 MoE
layers patched for both kernel configurations:

| Config | Tokens/sec | Mean TTFT | Cost / 1M tokens |
|---|---|---|---|
| Stock (unpatched) | 12.126 | 0.363s | $18.78 |
| Naive bf16 kernel | 20.966 | 0.391s | $10.86 |
| int8 quantized kernel | 20.718 | 0.392s | $10.99 |

Naive bf16 kernel: **+72.9%** over stock -- consistent with Phase 1's
separately-measured +67.2% on the same GPU class, a cross-phase sanity
check. int8 quantized kernel: **+70.9%** over stock, but **-1.2%** vs. the
naive bf16 kernel it's built on top of -- essentially a throughput tie,
not a speedup from quantizing. This int8 kernel dequantizes per-channel
in-kernel and was built to match the naive kernel's own `tl.dot`
precision profile exactly (see Task 3's fix round); it was never expected
to be *faster* than bf16 on compute-bound decode -- weight-only int8
saves memory bandwidth and footprint, not FLOPs, and this project's own
naive kernel already keeps every `tl.dot` at native tensor-core rate. Any
speedup from int8 specifically would need to come from a wider
batch/sequence-length regime where memory bandwidth, not compute, is the
bottleneck -- out of scope for this phase's single-request decode
benchmark.

## Memory footprint

Computed directly from the loaded model's real stacked expert weights
(`stack_expert_weights` + `quantize_stacked_weights` against all 27 MoE
layers, `stacked_weights_nbytes`/`quantized_stacked_weights_nbytes` --
not derived from GPU memory-usage sampling):

| | Bytes | |
|---|---|---|
| bf16 (all 27 layers' routed-expert weights) | 29,896,998,912 (27.84 GiB) | |
| int8 (data + per-channel fp32 scale) | 14,982,119,424 (13.95 GiB) | |
| **Reduction** | | **49.89%** |

Just under a clean 50% because the int8 data itself *is* exactly half of
bf16's bytes, plus a small per-output-channel fp32 scale on top (one
`float32` per row rather than per element) -- expected, and consistent
with the design doc's stated shape.

## Model-level agreement: quantized vs. naive

Per the design doc's two-tier correctness bar, the quantized run was
compared against the **naive bf16 run's** reference logits (not stock's:
quantization-induced divergence vs. bf16 is expected, and the naive
kernel is this project's own closest apples-to-apples bf16 comparison
point), across all 3 prompts:

| Prompt | Positions | `top1_agreement` | `mutual_top_k` | `max_abs_diff` |
|---|---|---|---|---|
| 0 | 11 | 1.0 | true | 2.125 |
| 1 | 11 | 1.0 | true | 1.90625 |
| 2 | 7 | 1.0 | true | 1.34375 |

**Perfect top-1 agreement and mutual top-k membership at every tested
position** -- int8 weight-only quantization did not change which token
would be selected anywhere in this test, on this model, at this prompt
set. `max_abs_diff` is larger than naive-vs-stock's (0.64-1.43 in Phase
5a's own stock/naive comparison) as expected from quantization error, but
never large enough to move an argmax outside the top-k. This is a
stronger result than the design doc's own bar required (it explicitly
allowed for real divergence here) -- reported as measured, not
downgraded to match the expectation.

## Environment

- `transformers==5.17.0` (this repo's pinned floor) still breaks
  DeepSeek's own remote-code model file the same two ways Phases 0/1/3/4
  documented: `is_torch_fx_available` removed from
  `transformers.utils.import_utils`, and `DynamicCache.get_usable_length`
  renamed to `get_seq_length`. **Still unresolved upstream** -- fourth
  phase in a row needing the same fix: `uv pip install --reinstall
  transformers==4.57.6` (not `uv run`, which re-syncs to the lockfile)
  plus the same narrow, pod-local, never-committed `_patch_and_run.py`
  monkeypatch.
- **New this session:** the pod's environment had
  `HF_HUB_ENABLE_HF_TRANSFER=1` set with the `hf_transfer` package not
  installed, which makes `huggingface_hub`'s downloader raise instead of
  falling back. Fixed with `uv pip install hf_transfer` -- worth
  installing proactively in a future phase's runbook rather than
  discovering it mid-download.
- Direct SSH (the pod's own public IP/port) rejected this session's SSH
  key even though it was in the pod's own registered-keys list;
  RunPod's proxy (`ssh.runpod.io`) with the same key worked immediately.
  Not investigated further -- the proxy's known quirks (forces an
  interactive shell regardless of a trailing command; worked around with
  stdin heredoc scripts ending cleanly on EOF) were sufficient for
  everything this session needed.

## Cost

| Pod | GPU | Duration | Cost |
|---|---|---|---|
| `4u8rfiy5dz5vj6` | L40 (Community, out of stock) | 125s | $0.0274 |
| `wmv06rf1l2sm8s` | L40S (Community, host GPU bug) | 1024s | $0.2247 |
| `924l6eft4d8251` | L40 (Secure Cloud) | 7473s (2.08hr) | $1.7022 |
| **Total** | | | **$1.9543** |

Against the **$5 cap**: **39.1% used**, all three pods (including both
abandoned attempts) included. Full records:
`docs/findings/2026-09-17-phase-5a-community-attempt1-cost.md`,
`2026-09-17-phase-5a-community-attempt2-cost.md`,
`2026-09-17-phase-5a-quantization-cost.md`.
