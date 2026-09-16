# Phase 3 design: multi-GPU expert-parallel serving

Per `docs/design/2026-09-14-dispatch-system-design.md` §5 and §7 (phase
plan), and `docs/adr/0002-deepep-over-hand-rolled-communication.md`'s
open question.

## 1. Purpose and thesis

Phase 3 puts DeepSeekMoE-16B's routed experts on two NVLink-connected
GPUs and uses DeepSeek's own DeepEP library for the cross-GPU
dispatch/combine, with Phase 1's Triton grouped-GEMM kernel doing the
local compute on each rank -- the split the system design doc already
calls for (DeepEP for the hard, solved communication problem; this
project's kernel for the differentiated compute work).

The headline result is not "DeepEP works." Phase 1 found a real,
workload-specific crossover: the persistent (cache-aware) kernel ties the
naive kernel at unbatched single-token decode, wins 3-8% at 16-128
tokens/expert, and loses by up to 14% at 512-2048 tokens/expert
(`docs/findings/2026-09-15-phase-1-grouped-gemm-run.md`). Cross-GPU EP
dispatch changes the per-expert token-count distribution each rank
actually sees -- it is not the same shape as local unbatched decode.

**Phase 3's thesis: does that crossover hold under DeepEP's real dispatch
pattern, or does multi-GPU expert-parallel routing shift where it falls?**
That is measured, not assumed, and it is what makes this phase more than
a library-integration exercise -- it reuses both of Phase 1's kernels and
Phase 2's skewed-load characterization instead of starting a new island.

## 2. Scope, decided during brainstorming

- **Topology: 2 GPUs, NVLink pair** (expert-parallel degree 2). Minimal
  degree that still exercises DeepEP's real cross-GPU path; DeepSeekMoE-16B's
  64 routed experts split 32/32. A larger node (4+ GPUs) was considered and
  rejected -- more cost and more to go wrong, for a model that already fits
  on one GPU, with no proportional increase in what it proves.
- **DeepEP vs. UCCL-EP: resolved by a live check before renting anything**
  (§5), not assumed from ADR-0002.
- **Benchmark scope: decode + prefill**, both. Prefill stresses the
  dispatch/combine path more heavily (many tokens per forward pass) and is
  arguably where EP's real cost or benefit shows up most; decode keeps
  continuity with Phase 0/1's methodology.
- **Small batch-size sweep, not a single point** -- a few batch sizes per
  mode, enough to see whether multi-GPU EP wins or loses depending on
  shape, consistent with how Phase 1 reported its own crossover rather than
  one flattering number.
- **Budget cap: $25**, set before rental (dedicated NVLink/SXM pricing is
  well above the $0.22-0.35/hr marketplace-spot tier Phases 0-2 used).

## 3. Architecture

Two GPU ranks, NVLink-connected, DeepSeekMoE-16B loaded on both.
Attention, the gate/router, embeddings, and the shared experts
(always-active, non-routed -- `src/dispatch/kernels/moe_forward.py`'s own
docstring already distinguishes "the routed-expert half" from the rest)
stay replicated and untouched on both ranks, exactly as in the existing
single-GPU path. Only the routed experts are sharded 32/32 across ranks by
expert id.

Process group: `torch.distributed` with the NCCL backend (`torchrun
--nproc_per_node=2`), which DeepEP's dispatch/combine calls run on top of.

## 4. Components

The key property of this design: `grouped_moe_routed` (in
`moe_forward.py`) and its pluggable `matmul` backend (`naive`/`persistent`,
resolved via `backends.py`) **do not change at all**. Today,
`integration.py`'s `patch_moe_infer` gathers a layer's tokens locally
before calling `grouped_moe_routed`. The EP version gathers the same way,
except tokens whose selected experts live on a remote rank get routed
there first via DeepEP's `dispatch()`, and each rank's
`grouped_moe_routed` call runs over whatever it now holds (local-origin +
received-from-remote), grouped only among that rank's local expert shard.

New files:

- `src/dispatch/kernels/expert_parallel.py` -- expert-id-to-rank
  assignment, and the DeepEP dispatch/combine wrapper around
  `grouped_moe_routed`.
- `patch_moe_infer_ep(...)` in `integration.py`, alongside the existing
  `patch_moe_infer` -- same swap-in pattern (replaces each MoE layer's
  `moe_infer`), EP-aware.

## 5. Data flow

Per MoE layer, per rank, for that rank's resident tokens:

1. DeepSeek's own gate (unchanged, replicated) computes top-k expert ids
   and softmax weights, same as today.
2. Shared experts run locally, no dispatch -- unchanged.
3. **Dispatch**: for the routed half, each token's selected experts are
   looked up against the expert-to-rank map; DeepEP's `dispatch()` sends
   the token's activation to every rank hosting at least one of them. With
   topk=6 over 64 experts split 32/32, a token landing entirely on one
   rank is rare (~3% by chance) -- most tokens dispatch to *both* ranks.
   This is the case DeepEP's API is built for, not a special case this
   project has to invent.
4. **Local compute**: each rank now holds a batch of (possibly
   remote-origin) token activations tagged by which local expert they
   need. This feeds directly into the existing `grouped_moe_routed(...)`,
   unchanged -- same grouping, same pluggable kernel, just fed by
   DeepEP-received rows instead of a purely local gather.
5. **Combine**: DeepEP's `combine()` sends each expert's output back to
   the token's origin rank; the origin rank forms the final weighted sum
   over all of a token's top-k contributions (local + remote) using
   `flat_expert_weights`, exactly as today's single-GPU caller does.

## 6. Testing

Extends CLAUDE.md's existing testing table with one new row, keeping the
same CPU/GPU split Phase 1 established:

| Layer | What's covered |
|---|---|
| EP bookkeeping (CPU, no GPU/DeepEP needed) | A test double simulates 2 "ranks" in a single process via plain tensor slicing -- no real DeepEP/NCCL call -- and asserts the sharded/dispatched/combined output matches the existing single-GPU `grouped_moe_routed` output exactly on the same input. Isolates "is the expert-to-rank sharding and weighted-combine arithmetic correct" from "does DeepEP's real transport work," the same separation Phase 1 drew between kernel-contract tests and real kernel tests. |
| EP correctness (`gpu`, paid) | Real 2-GPU rented run: mutual top-k logit agreement between the EP path (real DeepEP dispatch/combine) and the single-GPU reference, the same bar Phase 1 used. **Gates the benchmark** -- an EP run that fails correctness refuses to benchmark, same discipline as Phase 1's "a kernel run that patches no layers refuses to run." |
| Benchmark (`gpu`, paid) | Decode + prefill, small batch sweep, {naive, persistent} kernel x {single-GPU baseline, 2-GPU EP}. This grid answers the crossover-shift thesis (§1). Every number carries its full config, per CLAUDE.md. |

## 7. Risk, cost, and rollout

- **DeepEP vs. UCCL-EP live check first, no GPU cost.** Before renting
  anything: check both libraries' real repo activity, install path
  (prebuilt wheels vs. from-source NVSHMEM build -- DeepEP's known pain
  point), and actual NVLink/GPU support matrix. Resolves ADR-0002's open
  question honestly instead of deferring it again; whichever has the
  safer install path wins unless one is clearly unmaintained. Recorded as
  an ADR update once decided.
- **Hardware chosen live at rental time** against RunPod Secure Cloud's or
  Lambda's actual catalog and pricing -- not decided speculatively here,
  same practice as Phase 2's live GPU choice. Target: smallest
  NVLink-connected 2-GPU pair available.
- **Budget cap: $25**, timeboxed single session: spin up, run, capture
  evidence, tear down immediately. Cost measured and logged to
  `docs/findings/`, same as every phase.
- **Session order**: install/build the comms library -> confirm the
  CPU-testable EP-bookkeeping suite is already green before touching
  rented hardware -> the correctness gate -> only if correctness passes,
  the benchmark sweep. If the correctness gate fails and can't be fixed
  within budget, stop and document it as a real finding rather than force
  it.
- **Single-GPU baseline re-measured on the same GPU class** as one node of
  the rented pair -- not reused from Phase 0/1's L40/3090 numbers, since
  CLAUDE.md's benchmark rules require same-hardware comparisons.
- **Named risk**: DeepEP's NVSHMEM build is the single most likely thing
  to eat the session without producing a result. If the live pre-check
  finds this too fragile, the fallback (per ADR-0002) is DeepEP without
  its NVSHMEM fast path, or UCCL-EP -- decided live, not assumed here.

## 8. Non-goals

- Tensor parallelism, disaggregated prefill/decode (Phase 4), and
  quantization (Phase 5) are out of scope here.
- No claim beyond what's measured: if the crossover doesn't shift, or
  shifts only partially, that gets reported at exactly that strength, the
  same discipline Phase 1's null result and Phase 2's "moderate, not
  dramatic" divergence already set.
