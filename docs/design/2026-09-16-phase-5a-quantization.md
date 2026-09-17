# Phase 5a design: int8 weight-only quantized grouped-GEMM kernel

Per `docs/design/2026-09-14-dispatch-system-design.md` §7 (phase plan, row
5: "Quantization + speculative decoding layered on top"). That row bundles
two independent techniques; decided during brainstorming to split it into
two sequential sub-phases, each single-focus like every phase before it.
**This document covers 5a (quantization) only.** Speculative decoding is
5b, planned separately once 5a is complete.

## 1. Purpose and thesis

Every phase through Phase 4 has kept the routed-expert weights in bf16.
Weight-only int8 quantization is the standard first lever production
serving engines pull for memory: it roughly halves the routed-expert
weight footprint (the majority of DeepSeekMoE-16B's 32.8GB) with activations
left untouched, at some cost in output fidelity.

**Phase 5a's thesis: can the custom Triton grouped-GEMM kernel itself
consume int8 quantized expert weights -- not just call an existing
quantization library on the stock HF path -- and if so, at what memory,
throughput, and quality tradeoff, measured on the same real model and
hardware class this project has used throughout?**

This is a deliberate scope increase over "call bitsandbytes and report the
number": the kernel-engineering work (a new Triton kernel that dequantizes
inside the tile loop) is the differentiated contribution, consistent with
this project's overall thesis that the kernel is the custom part and
routing/communication/quantization *algorithms* are solved problems
(§3 of the system design doc; also its §11 out-of-scope note that
reimplementing bitsandbytes/AWQ's *algorithm* is not this project's job --
writing a kernel that executes one is a different claim).

## 2. Scope, decided during brainstorming

- **Weight-only int8 (W8A16).** Activations stay bf16. Only the routed
  expert weight tensors already flowing through Phase 1's kernel
  (`gate_proj`, `up_proj`, `down_proj`) are quantized. Shared experts,
  attention, the router/gate, and `lm_head` stay bf16 and untouched --
  exactly Phase 1's kernel scope, not expanded.
- **Per-output-channel symmetric round-to-nearest scales, self-computed.**
  `scale = max(abs(weight_row)) / 127` per expert per output channel,
  computed directly in `dispatch`'s own code rather than sourced from
  bitsandbytes/AWQ. Simple enough (a handful of lines) that it is not
  "reimplementing" a library's job -- it keeps the entire
  quantize -> kernel -> dequant-in-kernel path inside this project's own,
  fully inspectable code.
- **Built on the naive kernel, not persistent.** Phase 1's naive kernel won
  every benchmark that has run since (Phase 1's own token-count sweep,
  Phase 3's real-scale H200 measurement) -- no reason to add quantization
  complexity on top of the kernel variant that has never won.
- **Single GPU, reusing Phase 1's shape**, not Phase 3/4's multi-GPU
  rentals -- a kernel-correctness session, then one measured run.
  **Budget cap: $5**, set now, before any rental.
- **int4, activation quantization (W8A8), and bitsandbytes/AWQ-sourced
  scales are explicitly out of scope for 5a** -- see §7 Non-goals.

## 3. Architecture

The existing bf16 pluggable-kernel contract
(`GroupedMatmul = Callable[[Tensor, Tensor, TileSchedule], Tensor]` in
`moe_forward.py`, consumed by `torch_grouped_matmul` and the naive/persistent
Triton kernels) is **left untouched**. Quantized weights need a second
tensor (the per-channel scale) that doesn't fit that signature, and forcing
it in would mean every existing bf16 backend carries dead-weight branching
it doesn't need. Instead, Phase 5a adds a fully parallel set of types and
functions, the same pattern Phase 3 used for EP (`patch_moe_infer_ep`
added alongside `patch_moe_infer`, not folded into it):

- **`QuantizedTensor`** (new dataclass) -- `data: Tensor` (int8, shape
  `(E, N, K)`), `scale: Tensor` (fp32, shape `(E, N)`, one scale per
  expert per output channel).
- **`QuantizedStackedExpertWeights`** (new dataclass) -- `gate, up, down:
  QuantizedTensor`, mirroring `StackedExpertWeights`.
- **`QuantizedGroupedMatmul`** (new type alias) -- `Callable[[Tensor,
  QuantizedTensor, TileSchedule], Tensor]`.
- **`grouped_moe_routed_quantized`** (new function) -- the same body as
  `grouped_moe_routed` (group tokens, build tile schedule, three matmuls
  through SiLU, ungroup) but typed against `QuantizedStackedExpertWeights`
  and `QuantizedGroupedMatmul`. No behavior is shared by mutating the
  original; the logic is small enough (four lines) that duplicating it
  cleanly beats making the original generic over a union type.
- **`quantize_per_channel_int8`** (new function) -- `Tensor -> QuantizedTensor`,
  the round-to-nearest scale computation.
- **`quantize_stacked_weights`** (new function) -- `StackedExpertWeights ->
  QuantizedStackedExpertWeights`, applying the above to each projection.
  This is the hook point called once per layer at patch time (see below),
  never per forward pass.
- **`torch_grouped_matmul_dequant`** (new function) -- `(Tensor,
  QuantizedTensor, TileSchedule) -> Tensor`: dequantizes the given
  `QuantizedTensor` to fp32 and runs the same per-expert-slice matmul loop
  `torch_grouped_matmul` does. This is **not** a model-quality reference --
  it is the Triton int8 kernel's correctness oracle (see §5).
- **`grouped_matmul_int8`** (new Triton kernel, in `grouped_gemm.py` or a
  new sibling module if that file gets unwieldy -- decided in the
  implementation plan) -- built on the naive kernel's grid/launch
  structure; each tile loads an int8 weight sub-tile, widens and multiplies
  by its output-channel's fp32 scale, then proceeds through the same
  `tl.dot` accumulation the naive kernel uses.
- **`patch_moe_infer_quantized`** (new function, `integration.py`) --
  mirrors `patch_moe_infer` but calls `quantize_stacked_weights` on the
  freshly-stacked bf16 weights before building the per-layer closure, and
  wires `grouped_moe_routed_quantized` + `grouped_matmul_int8` instead of
  the bf16 path. Quantization happens once per layer, at patch time, not
  per forward pass.
- **CLI**: `scripts/run_baseline.py --moe-kernel` gains a new value
  (e.g. `quantized`) that routes to `patch_moe_infer_quantized` instead of
  `patch_moe_infer(model, resolve_backend(name))`. `resolve_backend`'s
  `BACKENDS` tuple and its bf16 contract are unchanged.

## 4. Data flow

1. **Patch time (once per layer, once per run):** `stack_expert_weights`
   builds the bf16 `StackedExpertWeights` exactly as today.
   `quantize_stacked_weights` converts it to `QuantizedStackedExpertWeights`
   -- three `QuantizedTensor`s, each with int8 data and fp32 per-channel
   scales. The original bf16 stacked weights are not retained after this
   (memory savings are real, not just reported).
2. **Forward pass (every call, unchanged token-routing logic):**
   `grouped_moe_routed_quantized` groups tokens by expert (identical
   `group_tokens_by_expert`/`build_tile_schedule` machinery, untouched),
   then calls `grouped_matmul_int8` three times (gate, up, down) exactly
   where `grouped_moe_routed` calls `matmul` today.
3. **Inside the kernel:** each tile dequantizes its int8 weight slice
   against that expert/channel's scale before the same tiled accumulation
   the naive kernel already does. Activations (`x`) stay bf16 throughout --
   only the weight side is ever int8.

## 5. Correctness -- two distinct layers, so the tight bar survives contact with lossy quantization

Quantization is lossy *by design*, so "correctness before speed" needs two
separate claims rather than one, or the project would either fake precision
it doesn't have or quietly drop the correctness discipline for this phase
alone:

- **Kernel-level (tight, same rigor as every prior phase).** The Triton
  `grouped_matmul_int8` kernel's output vs. `torch_grouped_matmul_dequant`
  fed the *same* `QuantizedTensor` -- both sides use identical already-quantized
  weights, so nothing here should diverge beyond float-precision noise.
  Held to `assert_matches_reference`'s existing tolerance discipline. A
  mutation (e.g. forcing every tile to use expert 0's scale) must turn this
  red, same bar Phase 1 set.
- **Model-level (stated, expected to show real divergence).** Mutual
  top-5 / top-1 logit agreement (`compare_top_k_agreement`, unchanged, no
  new machinery) between the int8-kernel-patched real model and Phase 1's
  bf16-naive-kernel-patched real model, same 3 prompts used throughout.
  This *is* expected to disagree somewhat -- quantization changing outputs
  is the entire point of measuring it. Whatever agreement percentage comes
  back gets reported as-is, not forced toward 100% or hidden if it's worse
  than hoped.

## 6. Testing

Extends CLAUDE.md's existing testing table:

| Layer | What's covered |
|---|---|
| Quantization math (CPU) | `quantize_per_channel_int8`: correct scale computation, round-trip error bounds on known tensors, an all-zero channel doesn't divide by zero, clamping at +-127. |
| Kernel contract (CPU) | `quantize_stacked_weights` produces correctly-shaped `QuantizedStackedExpertWeights`; tiling/grouping logic is reused unchanged from Phase 1, so no new tile-boundary tests are needed. |
| Kernel (`gpu`) | `grouped_matmul_int8` vs. `torch_grouped_matmul_dequant` at toy, decode-, and prefill-shaped dims -- the tight, kernel-level bar from §5. A mutation must turn this red. |
| End to end (`gpu`, paid) | Mutual top-5/top-1 agreement between the int8-patched real model and Phase 1's bf16-naive-patched real model, all 3 prompts -- the model-level bar from §5, reported honestly. **Gates the benchmark**: a run that fails to produce a patched count refuses to run, same discipline as every prior phase's `--moe-kernel`. |
| Benchmarks | Three-way `do_bench`/harness comparison (stock / bf16-naive / int8) on the same prompts and hardware; plus a new pure-function memory-footprint measurement (bytes of `StackedExpertWeights` vs. `QuantizedStackedExpertWeights`, CPU-testable, no GPU needed) since memory reduction is the actual point of quantizing. |

## 7. Non-goals

- **int4 or any bit-width below int8.** A real follow-on if 5a's int8 result
  is promising, not required for 5a to be "done."
- **Activation quantization (W8A8).** Weight-only only; activations stay
  bf16 throughout.
- **bitsandbytes/AWQ-sourced quantization scales.** Decided during
  brainstorming: self-computed round-to-nearest scales, to keep the whole
  path inside this project's own code (see §2). Not a comment on those
  libraries' quality -- their calibration-based schemes are the natural
  next comparison point, not part of 5a's claim.
- **Quantizing the persistent kernel.** Only the naive kernel gets an int8
  variant (§2) -- the persistent kernel has not won a single benchmark
  since Phase 1's own token-count sweep found its win region doesn't
  reach this project's real workload scale (Phase 3's finding).
  Revisiting it is not blocked by anything in this design if a future
  phase finds a reason to.
- **Combining with multi-GPU EP (Phase 3/4).** Single GPU only, matching
  the design doc's Phase 5 cost shape ("single/dual GPU"). A quantized +
  EP combination is a plausible future extension, not required here.
- **Speculative decoding.** Phase 5b, planned and executed separately once
  5a's PR merges.
- **Beating vLLM/SGLang's own quantized-serving numbers.** Phase 6 owns
  the vLLM/SGLang comparison; 5a's claim is limited to "this project's own
  kernel, bf16 vs. int8," an internal comparison like Phase 4's
  co-located-vs-disaggregated one.

## 8. Risk, cost, and rollout

- **Cost discipline is part of the result, not just a constraint.** $5 is
  a ceiling; the gap between cap and actual spend gets reported the way
  every prior phase has (Phase 1: $0.65 of a combined $8 across two
  sessions; Phase 3: $13.03 of $25; Phase 4: $10.94 of $40).
- **Maximize what's proven before any rental.** Every row in §6 above the
  `gpu` line is real, CPU-only, $0 to iterate on -- the quantization math,
  the dataclasses, the shape/round-trip tests. The paid session should
  start from that suite green and spend rented time only on what
  genuinely needs a GPU: the Triton kernel's own correctness, then the
  measured run.
- **Named risk: int8 dequant-in-kernel may need a numerically different
  tolerance than bf16-to-bf16 comparisons have used.** If
  `assert_matches_reference`'s existing `CORRECTNESS_RTOL` doesn't hold
  for the dequant path, that gets a stated, justified, and reported
  tolerance change -- not a silently loosened one.
- **Named risk: per-channel scale storage/broadcast inside the Triton
  kernel is new territory this project's kernels haven't needed before**
  (naive/persistent both operate on a single dtype with no side-channel
  per-tile metadata beyond the tile schedule already built). If this
  proves harder than expected inside the tile loop, that is itself a
  reportable finding, same as Phase 3's Fabric Manager discovery or
  Phase 1's persistent-kernel null result -- not a reason to force a
  workaround that muddies the kernel-level correctness gate.
- **GPU class**: RTX 3090 or L40 class (Phase 0/1's tier), marketplace/spot,
  checked live against real-time availability at rental time, same
  practice as every prior phase.
- **Session order**: CPU-testable suite green (quantization math, dataclass
  shapes) -> rent -> kernel correctness gate (§5, kernel-level) -> only if
  that passes, patch the real model and run the model-level agreement
  check (§5) -> only if a patched count is nonzero, the three-way
  benchmark and memory-footprint measurement -> tear down immediately.
