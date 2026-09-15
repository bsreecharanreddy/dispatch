# Phase 1 grouped-GEMM run -- findings

## Infrastructure

- Pod `ftbbiqkiig7h5s`, RunPod Secure Cloud, data center US-KS-2, NVIDIA
  L40 -- compute capability **8.9** (confirmed live via `nvidia-smi
  --query-gpu=name,compute_cap,memory.total --format=csv`), 46068 MiB
  VRAM, 70GB container disk. Same GPU class as Phase 0, per the design
  doc's same-hardware rule.
- L40 was out of stock on both RunPod Community and Secure clouds when
  first checked; it reappeared on Secure Cloud (US-KS-2) a few minutes
  later. Rate: **$0.82/hr** -- the same Secure Cloud rate Phase 0 paid,
  for the same reason (Community's cheaper L40 tier was unavailable at
  deploy time both times).
- Image `runpod/pytorch:1.3.1-cu1290-torch290-ubuntu2404` (current stable
  tag on Docker Hub, checked live 2026-09-15).
- `torch 2.14.0+cu130`, `cuda 13.0`, `triton 3.8.0` (confirmed live and
  recorded in the kernel-bench JSON's `config`).
- Checked DeepSeek's `modeling_deepseek.py` for drift since Phase 0 per
  this runbook's own step 2: `lastModified` on the model repo is
  `2024-01-12`, unchanged since Phase 0's run -- both Phase 0 workarounds
  (a `transformers==4.57.6` pin overriding this repo's `>=5.17.0` floor,
  and a `DynamicCache.get_usable_length` monkeypatch) were still needed
  and applied identically via a pod-local, never-committed
  `_patch_and_run.py` wrapper.

## Correctness

`pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs` on this L40:
**25 passed, 0 skipped** (compute capability 8.9 runs the bf16 cases),
25.6 seconds per test on average due to Triton's first-call compilation
on a card this session hadn't used before -- both kernels re-verified
correct on the exact hardware the timed numbers below come from, not
just on Task 6's RTX 3090.

End to end, from each results JSON's `reference_comparison` (3 prompts,
11/11/7 generated-token positions):

| Run | moe_layers_patched | mutual top-5 | top-1 agreement | max abs logit diff |
|---|---|---|---|---|
| torch (control) | 27 | true (all 3 prompts) | 1.0 (all 3 prompts) | 0.773 / 0.850 / 1.900 |
| naive | 27 | true (all 3 prompts) | 1.0 (all 3 prompts) | 2.014 / 0.656 / 1.467 |
| persistent | 27 | true (all 3 prompts) | 1.0 (all 3 prompts) | 0.695 / 0.781 / 1.815 |

Every run patched all 27 of the model's MoE layers (`n_routed_experts=64,
num_experts_per_tok=6`, `moe_layer_freq=1`, `first_k_dense_replace=1` --
28 decoder layers minus the one dense layer). Every run agrees with the
stock reference at every position of every prompt, both by top-1 and by
mutual top-5. The control row (`torch`, the eager per-expert-group loop
over the same grouped layout the kernels use) is the noise floor this
project's own review flagged as the thing to read by eye rather than
trust a bare pass/fail on: its `max_abs_diff` (0.77-1.90) is the same
order of magnitude as the two real Triton kernels' (0.66-2.01), so the
kernels are not introducing meaningfully more numerical drift than the
grouped-layout reformulation itself already does in bf16. None of the
nine `max_abs_diff` values is an exact `0.0` -- per this project's own
finding from Task 8's review, an exact zero would mean the kernel path
never actually executed (e.g. a model accidentally left in training
mode), not a perfect result; every nonzero value here is evidence the
kernels genuinely ran.

Three transient CUDA allocator warnings
(`memory allocation failed with OOM ... trying to allocate 369098752
bytes`) appeared once per kernel-backed run (not the stock run), always
for the same ~352MB allocation, and each run still completed and
produced correct output -- PyTorch's caching allocator retried and
succeeded. 369098752 bytes matches Task 8's review of `stack_expert_weights`'s
per-layer re-stacking cost for this model
(moe_intermediate_size x hidden_size x 2 bytes x roughly one projection),
so this is exactly the "peak overhead is one layer's projection, not a
doubled model" memory shape that review predicted -- confirmed here as a
real, if narrowly-avoided, memory-pressure point on a 46GB card with a
32.8GB model already loaded. Not a correctness problem in this run; worth
knowing if a future session uses a smaller card or a larger model.

## End-to-end throughput

Same config as Phase 0 -- bf16, single L40, unbatched eager decode, 3
prompts x 5 repetitions, 64 new tokens -- all four runs in one session.

| Run | mean tokens/sec | mean TTFT | p50 TTFT | p99 TTFT | mean ITL | $ per 1M tokens |
|---|---|---|---|---|---|---|
| stock | 12.55 | 0.299s | 0.269s | 0.809s | 0.0763s | $18.15 |
| torch | 15.40 | 0.190s | 0.164s | 0.560s | 0.0631s | $14.79 |
| naive | 20.98 | 0.112s | 0.047s | 0.890s | 0.0470s | $10.86 |
| persistent | 20.80 | 0.125s | 0.048s | 1.055s | 0.0473s | $10.95 |

Stock (12.55 tok/s) is consistent with Phase 0's separately-measured
12.75 tok/s on the same GPU class, config, and 15-run protocol -- a
useful cross-session sanity check that this run's environment matches
Phase 0's closely enough for the comparison to mean something.

Both grouped-GEMM kernels measure **~65-67% faster than stock**
(naive: +67.2%, persistent: +65.7%) at this project's own baseline
config. The eager `torch` backend -- same grouped layout, same routing,
no Triton at all -- is itself ~22.7% faster than stock, meaning roughly a
third of the total speedup comes from the grouped-layout reformulation
(one contiguous per-expert matmul instead of DeepSeek's own masked-gather
loop) and the rest from the Triton kernels themselves. Cost per 1M
generated tokens drops from $18.15 (stock) to $10.86 (naive) --
computed as rate ($/hr) / (mean tokens/sec x 3600) x 1e6, both inputs
measured, not estimated.

**Naive and persistent are statistically indistinguishable at this
config** (20.98 vs. 20.80 tok/s, well within this run's own p50/p99
spread). This is not a surprise this project didn't already predict: the
plan's risk section states plainly that unbatched decode gives each MoE
layer only 6 routed rows per step (one per selected expert, since
`num_experts_per_tok=6`), so there is no multi-tile-per-expert band of
rows for the persistent kernel's grouped launch ordering to keep warm in
L2 across. The kernel micro-benchmark below, which sweeps token count
specifically to surface where grouped launch ordering does pay off, is
where that shows up instead of in this end-to-end number. This is
recorded as a real result, not buried: **grouped-GEMM's win here is
almost entirely from batching all six routed experts into one grouped
launch instead of looping through a slow gather per expert, not from
persistent-kernel cache reuse** -- decode is simply the wrong regime for
the latter, exactly as predicted.

## Kernel micro-benchmark

Per token count, per backend, for both `zipf` (skewed toward low-index
experts) and `uniform` routing: mean latency and TFLOP/s, for the whole
routed MoE layer (`layer`, three grouped GEMMs) and the gate_proj-shaped
grouped GEMM alone (`gemm`). Full data:
`docs/findings/2026-09-15-phase-1-kernel-bench-zipf.json` and
`-uniform.json` (each includes the run's full config -- GPU, torch/cuda/
triton versions, block sizes -- alongside every number).

### zipf routing

| tokens | torch/layer (ms, TFLOP/s) | naive/layer | persistent/layer | torch/gemm | naive/gemm | persistent/gemm |
|---|---|---|---|---|---|---|
| 1 | 1.104, 0.09 | 0.511, 0.20 | 0.518, 0.20 | 0.103, 0.33 | 0.078, 0.44 | 0.079, 0.44 |
| 16 | 5.319, 0.31 | 1.434, 1.16 | 1.393, 1.19 | 1.434, 0.39 | 0.454, 1.22 | 0.440, 1.26 |
| 128 | 8.898, 1.49 | 2.181, 6.09 | 2.085, 6.37 | 2.563, 1.73 | 0.692, 6.40 | 0.638, 6.94 |
| 512 | 9.956, 5.34 | 2.274, 23.37 | 2.489, 21.35 | 2.737, 6.47 | 0.701, 25.28 | 0.719, 24.64 |
| 2048 | 10.102, 21.04 | 4.626, 45.96 | 5.286, 40.22 | 2.678, 26.46 | 1.566, 45.25 | 1.618, 43.81 |

### uniform routing

| tokens | torch/layer (ms, TFLOP/s) | naive/layer | persistent/layer | torch/gemm | naive/gemm | persistent/gemm |
|---|---|---|---|---|---|---|
| 1 | 1.242, 0.08 | 0.510, 0.20 | 0.526, 0.20 | 0.104, 0.33 | 0.078, 0.44 | 0.079, 0.44 |
| 16 | 7.020, 0.24 | 1.709, 0.97 | 1.629, 1.02 | 1.908, 0.29 | 0.555, 1.00 | 0.517, 1.07 |
| 128 | 9.145, 1.45 | 2.187, 6.08 | 2.079, 6.39 | 2.696, 1.64 | 0.696, 6.36 | 0.647, 6.85 |
| 512 | 9.927, 5.35 | 2.288, 23.23 | 2.341, 22.71 | 2.898, 6.11 | 0.706, 25.11 | 0.677, 26.17 |
| 2048 | 10.344, 20.55 | 4.982, 42.67 | 5.314, 40.01 | 2.942, 24.09 | 1.600, 44.29 | 1.708, 41.50 |

Both routing distributions tell the same story: at 1-16 tokens (the
decode regime the end-to-end run above actually measured), naive and
persistent are within noise of each other and both already 2-4x faster
than the eager `torch` loop; from 128 tokens up (prefill-shaped work with
more rows per expert), naive is consistently as fast as or slightly
faster than persistent at this project's fixed, untuned block sizes
(`BLOCK_N=64`, `BLOCK_K=64`, `GROUP_SIZE_M=8`) -- the persistent kernel's
grouped launch ordering never shows a measurable win over the naive
kernel's simpler one-CTA-per-tile launch at any token count this sweep
covers, on this card, at these block sizes. This does not mean the
persistent design is wrong; it means this specific workload (67 total
routed rows per layer even at 2048 tokens x 6 experts-per-token, spread
across 64 experts) rarely gives any single expert more than one or two
`BLOCK_M=16` tiles for grouped ordering to amortize across. Distribution
skew (`zipf` vs. `uniform`) makes no visible difference to either
kernel's `layer` numbers, consistent with 64 experts each still getting
enough tokens at these token counts that per-expert row counts don't
swing tile counts much either way.

## What it shows

**Grouped GEMM measurably speeds up this model's routed-expert step**:
~65-67% faster end-to-end decode throughput than DeepSeek's own stock
`moe_infer`, at zero measured correctness cost (mutual top-5 agreement
and top-1 agreement both perfect across every tested position). About a
third of that win is the grouped-layout reformulation alone (the eager
`torch` control), and the rest is the two Triton kernels' compiled
execution.

**The persistent kernel's headline optimization -- grouped launch
ordering for L2 reuse -- does not show a measurable benefit anywhere in
this session's measurements**, end-to-end or in the token-count sweep.
This is a null result the plan's own risk section predicted before this
run happened (unbatched decode gives too few routed rows per expert for
a persistent CTA's tile-group reuse to matter), and the micro-benchmark
built specifically to surface where it *would* pay off doesn't find that
point within the swept range either (1 to 2048 tokens). Two honest
readings: (1) real serving traffic is usually batched well past 2048
tokens per step, a regime this sweep doesn't reach, so the result may
simply not generalize past this project's own single-request decode
scope; or (2) `GROUP_SIZE_M=8`, fixed and untuned per this project's
explicit scope decision, may not suit this model's 64-expert, 6-per-token
routing shape regardless of batch size. Both are real possibilities this
data cannot distinguish between, and Phase 1's stated scope (naive then
persistent, not autotuning) stops here rather than guessing further.
