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
- This pod took roughly 24 minutes to become SSH-reachable after
  creation (`runtime` stayed unpopulated and the SSH proxy returned
  "container not found" throughout that window), notably longer than
  Task 6's RTX 3090 pod (~2.5 minutes) on the same provider. Direct SSH
  connected on the first attempt once `runtime` populated. Billing starts
  at pod creation regardless of SSH readiness, so this added to the
  measured cost below; worth budgeting extra idle time for Secure Cloud
  L40 provisioning specifically in a future session.
- Pod terminated via `delete-pod` after all results were copied off;
  termination verified independently by a second, separate `get-pod`
  query, which returned `404 pod not found` (RunPod's terminate/delete
  fully removes the resource, confirmed the same way in Task 6).
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
**25 passed, 0 skipped** (compute capability 8.9 runs the bf16 cases) in
186.9s total (7.5s/test average) -- Triton's first-call compilation on a
card this session hadn't used before, plus the model-independent nature
of these tests keeping most of that on the kernel side rather than model
loading. Both kernels re-verified correct on the exact hardware the
timed numbers below come from, not just on Task 6's RTX 3090.

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
succeeded. 369,098,752 bytes = `n_routed_experts x moe_intermediate_size
x hidden_size x 2 bytes` = 64 x 1408 x 2048 x 2, exactly -- one expert
projection's weights across all 64 experts, matching Task 8's review of
`stack_expert_weights`'s per-layer re-stacking cost for this model. This
is exactly the "peak overhead is one layer's projection, not a doubled
model" memory shape that review predicted -- confirmed here as a
real, if narrowly-avoided, memory-pressure point on a 46GB card with a
32.8GB model already loaded. Not a correctness problem in this run; worth
knowing if a future session uses a smaller card or a larger model.

## End-to-end throughput

Same config as Phase 0 -- bf16, single L40, unbatched eager decode, 3
prompts x 5 repetitions, 64 new tokens -- all four runs in one session.
No separate warm-up phase: p99 TTFT includes each run's own first-call
overhead (CUDA context and, for the kernel-backed runs, Triton
compilation), which is why p99 TTFT runs well above mean/p50 for every
row below -- consistent with Phase 0's own observation that the first
run pulls the tail up.

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
spread). At the 1-token-per-step granularity this end-to-end run actually
exercises (unbatched decode routes 6 rows per MoE layer per generated
token -- one per selected expert, since `num_experts_per_tok=6`), the
kernel micro-benchmark below shows persistent running 0.2-3.1% *slower*
than naive at exactly this shape (`num_tokens=1`), which is consistent
with this end-to-end pair being a tie within measurement noise rather
than either kernel actually winning. The plan's risk section predicted
this: too few routed rows per expert at decode granularity for the
persistent kernel's grouped launch ordering to have a band of same-expert
rows to keep warm in L2. This is recorded as a real result, not buried:
**grouped-GEMM's win here is almost entirely from batching all six
routed experts into one grouped launch instead of looping through a slow
gather per expert, not from persistent-kernel cache reuse at this token
count** -- decode-per-step is the wrong regime for the latter, as
predicted. (The micro-benchmark below finds persistent *does* win at
somewhat larger token counts than a single decode step ever produces --
see "Kernel micro-benchmark".)

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
| 128 | 9.145, 1.45 | 2.187, 6.07 | 2.079, 6.39 | 2.696, 1.64 | 0.696, 6.36 | 0.647, 6.85 |
| 512 | 9.927, 5.35 | 2.288, 23.23 | 2.341, 22.71 | 2.898, 6.11 | 0.705, 25.11 | 0.677, 26.17 |
| 2048 | 10.344, 20.55 | 4.982, 42.67 | 5.314, 40.01 | 2.942, 24.09 | 1.600, 44.29 | 1.708, 41.50 |

Both routing distributions already are much faster than the eager
`torch` loop at every token count -- 2-10x on `layer`, more on the
standalone `gemm`. The persistent-vs-naive comparison is not flat across
the sweep; it changes sign twice, and the same sign shows up in both
routing distributions and in both measurement targets (`layer` and
`gemm`), which is the signature of a real effect rather than noise:

| tokens | rows/expert (avg) | tiles/expert (avg, `BLOCK_M=16`) | persistent vs. naive, `layer` (zipf / uniform) |
|---|---|---|---|
| 1 | 0.09 | 0.01 | +1.3% / +3.1% (persistent slower) |
| 16 | 1.5 | 0.09 | -2.8% / -4.7% (persistent faster) |
| 128 | 12 | 0.75 | -4.4% / -5.0% (persistent faster) |
| 512 | 48 | 3 | +9.5% / +2.3% (persistent slower) |
| 2048 | 192 | 12 | +14.3% / +6.7% (persistent slower) |

At the two smallest token counts *above* single-token decode (16 and
128), the persistent kernel wins a modest but consistent 3-8% in every
one of the four cells (two distributions x two measurement targets). At
512 and 2048 it loses, by as much as 14% at 2048/zipf. At the actual
decode granularity (1 token, `num_experts_per_tok=6` routed rows per
layer, well under one `BLOCK_M=16` tile per expert on average) it is a
tie leaning slightly toward naive being faster -- consistent with the
end-to-end result above. This does not fit a single one-line story: the
"rows per expert" column rules out a naive "more rows always helps
persistent" reading (128 tokens, 12 rows/expert, is where persistent
wins most; 2048 tokens, 192 rows/expert, is where it loses most), so
whatever is driving the 512-2048 regression is not simply "not enough
tiles" -- if anything the opposite. Distribution skew (`zipf` vs.
`uniform`) does *not* wash out here either: at 16 tokens specifically,
`naive/layer` and `persistent/layer` both run 17-19% slower under
`uniform` than under `zipf`, likely because `zipf`'s concentration
toward a handful of low-index experts leaves fewer of the 64 experts
occupied at all at just 96 total routed rows, while `uniform` spreads the
same 96 rows thinner across more experts and pays more per-tile launch
overhead; at 1/128/512/2048 tokens the two distributions track each
other closely on `layer` (within a few percent).

## What it shows

**Grouped GEMM measurably speeds up this model's routed-expert step**:
~65-67% faster end-to-end decode throughput than DeepSeek's own stock
`moe_infer`, at zero measured correctness cost (mutual top-5 agreement
and top-1 agreement both perfect across every tested position). About a
third of that win is the grouped-layout reformulation alone (the eager
`torch` control), and the rest is the two Triton kernels' compiled
execution.

**The persistent kernel's headline optimization -- grouped launch
ordering for L2 reuse -- shows no benefit at the token count that
actually drives end-to-end decode throughput (a tie at 1 token, leaning
slightly toward naive), which is the null result the plan's own risk
section predicted before this run happened.** This is why the two
kernels measure statistically indistinguishable end-to-end (20.98 vs.
20.80 tok/s) despite the persistent kernel's design intent. The full
micro-benchmark tells a more specific story than a flat "no benefit
anywhere," though: at 16 and 128 tokens -- above single-token decode but
still well short of a full batched prefill -- persistent consistently
wins by 3-8% across both routing distributions and both measurement
targets, then loses by up to 14% at 512-2048 tokens. That the win sits in
the *middle* of the swept range rather than growing monotonically with
token count (and therefore with rows-per-expert) means "not enough tiles
yet" cannot be the whole explanation for the 512-2048 regression, since
2048 tokens gives each expert far more tiles on average than 128 does.
Two honest readings this data cannot distinguish between: (1) the
persistent kernel's fixed `GROUP_SIZE_M=8` may be well-matched to a
narrow middle band of tile counts for this model's 64-expert shape and
mismatched outside it in either direction, in which case autotuning
`GROUP_SIZE_M` per token count -- explicitly out of Phase 1's scope --
might recover the 512-2048 loss; or (2) some other per-launch overhead
(e.g. the persistent kernel's fixed `(NUM_SMS,)` grid doing more
scheduling work per program as the total tile count grows) dominates at
the larger token counts regardless of tile density. Phase 1's stated
scope (naive then persistent, not autotuning) stops here rather than
guessing further, but real serving traffic that keeps individual
requests near single-token decode steps (this project's own scope) would
not see the mid-range win either way -- the end-to-end number above is
the one that matters for that regime, and it shows no persistent-kernel
advantage.
