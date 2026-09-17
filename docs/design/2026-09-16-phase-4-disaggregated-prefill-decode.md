# Phase 4 design: disaggregated prefill/decode

Per `docs/design/2026-09-14-dispatch-system-design.md` §7 (phase plan,
row 4: "Disaggregated prefill/decode across separate GPU pools").

## 1. Purpose and thesis

Every phase through Phase 3 stayed deliberately single-request and
unbatched (Phase 0's own methodology: "unbatched eager-mode token-by-token
decode"). At that scale there is no batching contention between prefill
and decode -- there is only ever one request in flight, so co-locating
both phases on the same GPU costs nothing extra. Disaggregated
prefill/decode only earns its keep once concurrent requests are actually
competing for the same GPU: prefill's large, batched, compute-bound
forward pass and decode's small, sequential, memory-bound steps interfere
with each other's ideal shape when several requests share one engine.

**Phase 4's thesis: does splitting prefill and decode onto separate GPU
pools measurably relieve that contention, for this project's own model
and kernels, holding total GPU count fixed?** That means Phase 4
introduces concurrent, continuous-batched serving for the first time in
this project -- new scope beyond anything Phase 0-3 built -- specifically
because the contention this phase measures does not exist without it.

The comparison is controlled, not a bigger-hardware-wins comparison: the
same 4 GPUs, once as one 4-rank EP pool doing both prefill and decode
for concurrent requests, once as two independent 2-rank EP pools (one
prefill, one decode). GPU count held constant; the only variable is
whether the two phases share GPUs.

## 2. Scope, decided during brainstorming

- **Batching: a real continuous-batching scheduler**, not static
  fixed-window batching. New requests are admitted into a live decode
  batch as slots free up, the pattern that actually produces prefill/decode
  contention on a shared engine -- a static batch-then-drain model would
  under-state the contention this phase exists to measure.
- **Combined with Phase 3's EP**, not isolated from it. Each pool (prefill,
  decode) is itself a 2-GPU DeepEP expert-parallel pool, reusing
  `patch_moe_infer_ep` and `expert_parallel.py` directly rather than
  falling back to single-GPU dense MoE. This is closer to what a real
  deployment would actually run, at the cost of stacking two previously
  separate subsystems (EP, disaggregation) together for the first time.
- **Topology: one 4-GPU NVLink/SXM node**, not two separate pods. GPUs 0-1
  = prefill EP pool, GPUs 2-3 = decode EP pool, all four ranks joined in
  one `torch.distributed` world so KV-cache handoff is a same-node NCCL
  `send`/`recv`, not a real network transfer. Logically disaggregated
  (separate process groups, separate schedulers, an explicit handoff
  step) without taking on cross-node networking as new scope.
- **Baseline: the same 4 GPUs as one co-located EP pool** (`n_ranks=4`)
  running both phases through a single continuous-batching loop. Isolates
  disaggregation's effect by holding GPU count constant between baseline
  and treatment.
- **Budget cap: $40**, set before rental, deliberately treated as a
  ceiling, not a target -- see §7.

## 3. Architecture

Four ranks, one `torch.distributed` NCCL world (`torchrun
--nproc_per_node=4`), split into two disjoint DeepEP EP sub-groups:

- **Prefill pool** -- ranks 0-1, `patch_moe_infer_ep(..., n_ranks=2)`,
  its own `Buffer`.
- **Decode pool** -- ranks 2-3, a second, independent
  `patch_moe_infer_ep(..., n_ranks=2)` and its own `Buffer`.

Each pool loads a full copy of DeepSeekMoE-16B (unavoidable -- they are
separate GPU groups with separate EP shards of the routed experts).
Attention, the gate, embeddings, and shared experts stay replicated on
every rank, exactly as in Phase 3 -- only the routed-expert compute is
EP-sharded, independently within each pool.

The co-located baseline reuses the same four ranks as a single EP pool
(`n_ranks=4`), no prefill/decode split, one scheduler loop doing both.

## 4. Components

- **`PrefillWorker`** (ranks 0-1) -- loop: drain up to `prefill_batch_size`
  waiting requests, batch and pad their prompts, one
  `model(input_ids=..., past_key_values=None, use_cache=True)` call across
  both ranks (EP dispatch/combine inside, unchanged from Phase 3), slice
  the batched `past_key_values` back into per-request caches, compute
  each request's first token, enqueue `(request_id, kv_cache, first_token)`
  onto the handoff queue.
- **`DecodeWorker`** (ranks 2-3) -- loop: admit anything waiting on the
  handoff queue into its live batch (up to `decode_batch_size`), run one
  batched decode step, drop any request that hit EOS or `max_new_tokens`,
  record its completion timing.
- **KV-cache handoff** -- EP shards only the MoE FFN, not attention, so
  both ranks in the prefill pool hold identical KV cache after prefill.
  Rank 0 -> rank 2 and rank 1 -> rank 3 each do a direct NCCL `send`/`recv`
  of that request's cache tensors; no broadcast step needed.
- **Load driver** -- issues a configurable number of concurrent requests
  with staggered arrivals (not a single simultaneous burst) against
  whichever mode (co-located or disaggregated) is under test, and records
  per-request TTFT, inter-token latency, and completion time.
- **Co-located baseline worker** -- one loop, four ranks, doing what
  `PrefillWorker` and `DecodeWorker` do separately: admit new requests'
  prefill inline, then step every in-flight request's decode by one
  token, every iteration, same 4 GPUs.

New file: `src/dispatch/serving/disaggregated.py` (workers, handoff
queue, scheduler). New file: `src/dispatch/serving/colocated.py` (the
baseline loop). Neither changes `grouped_moe_routed`, `expert_parallel.py`,
or Phase 3's EP integration -- both are new orchestration around the
existing, already-proven compute path.

## 5. Data flow

Per request, disaggregated mode:

1. Request arrives at the load driver, is pushed to the prefill pool's
   incoming queue. TTFT clock starts.
2. `PrefillWorker` admits it into the next prefill batch (padded,
   `attention_mask`-aware), runs one forward pass across ranks 0-1
   (identical EP dispatch/combine path to Phase 3), produces the first
   token and a per-request KV cache.
3. KV cache moves rank 0 -> rank 2 and rank 1 -> rank 3 via NCCL
   `send`/`recv`. First token and cache land on the decode pool.
4. `DecodeWorker` admits the request into its live batch on its next
   iteration. Every iteration thereafter, it steps one token for every
   active request (including this one) until EOS or `max_new_tokens`.
5. Completion timing recorded by the load driver.

Co-located mode: the same five steps happen on the same 4 ranks inside
one loop, with prefill and decode work interleaved on the same GPUs
instead of split across pools -- no handoff step, no separate queue.

## 6. A known simplification, disclosed up front

Continuous batching here uses plain dense, padded tensors, not a real
paged/block KV cache (PagedAttention). Every decode step re-pads all
active requests' caches to the batch's current max length -- real wasted
compute next to a production engine, and deliberately out of scope to
build (a paged cache is most of what makes vLLM's own engine hard).
It applies identically to the co-located baseline and the disaggregated
treatment, so it should not bias the *relative* comparison this phase
measures -- but absolute throughput/TTFT numbers here are not directly
comparable to vLLM's own, and the findings doc must say so explicitly,
same as Phase 3's H200-substitution and V1-API disclosures.

## 7. Testing

Extends CLAUDE.md's existing testing table:

| Layer | What's covered |
|---|---|
| Scheduler logic (CPU, no GPU) | Admission/eviction against a fake model call: batch-size limits, request lifecycle, queue draining. No real tensors needed. |
| KV-cache slice/reassemble (CPU, no GPU) | Splitting a batched `past_key_values` into per-request caches and back, against small CPU tensors and a tiny real HF model -- proves the bookkeeping before any GPU is involved. |
| Handoff mechanism (CPU, no GPU, no DeepEP) | `torch.distributed`'s `gloo` backend supports `send`/`recv` -- 4 CPU processes prove the right cache reaches the right rank before DeepEP or real GPUs enter the picture at all. |
| Batched correctness (`gpu`) | Batched prefill/decode output (padding, attention masking) matches Phase 0's single-request, unbatched reference for the same prompt. |
| Full pipeline correctness (`gpu`, paid) | Real 4-rank run, both EP pools live: mutual top-k logit agreement between the disaggregated path and the single-GPU reference, same bar Phase 1 and Phase 3 used. **Gates the benchmark** -- a run that fails correctness is not benchmarked. |
| Contention measurement (`gpu`, paid) | Co-located vs. disaggregated, same 4 GPUs, a couple of concurrency levels (e.g. 4 and 8 concurrent requests). Every number carries its full config. |

## 8. Risk, cost, and rollout

- **Cost discipline is itself part of this phase's result, not just a
  constraint on it.** $40 is a ceiling, not a target -- the goal is the
  smallest, most surgical paid session that still produces a trustworthy
  measurement, and the gap between the cap and the actual spend is worth
  reporting alongside the throughput numbers, the way Phase 3 reported
  $13.03 of its $25 cap.
- **Maximize what's proven before any rental.** Every layer in §7 above
  the `gpu` line is real, CPU-only, and can be fully debugged for $0 --
  including the NCCL handoff mechanism itself via `gloo`. The paid
  session should start from scheduler + handoff + batching logic already
  green, and spend rented time only on what genuinely requires DeepEP and
  real multi-GPU hardware: EP correctness, then the measurement.
- **Correctness gate before any throughput claim**, same discipline as
  every prior phase -- a disaggregated run that disagrees with the
  single-GPU reference does not get benchmarked, it gets debugged or the
  phase stops and reports why.
- **Session order**: confirm the CPU-testable suite (scheduler, cache
  slicing, `gloo` handoff) is green -> rent the 4-GPU node -> verify real
  NVLink across all 4 GPUs (`nvidia-smi topo -m`, same check Phase 3 used)
  -> correctness gate on both the co-located and disaggregated paths ->
  only if both pass, the concurrency-level measurement -> tear down
  immediately.
- **Named risk carried over from Phase 3**: DeepEP V1 (not V2) is the
  confirmed-working API on this project's rental tier (no GPU Fabric
  Manager available). Phase 4 should assume V1 from the start rather than
  re-discovering this -- two independent `Buffer` instances, one per EP
  pool, same as Phase 3's single pool.
- **Named risk, new to this phase**: running two independent 2-rank EP
  pools plus cross-pool NCCL `send`/`recv` in the same 4-rank world is
  more process-group plumbing than Phase 3 needed. If DeepEP's `Buffer`
  or process-group setup conflicts across sub-groups in a way that isn't
  resolvable within budget, that is itself a real, reportable finding --
  not a reason to force a workaround that muddies the correctness gate.
- **GPU class**: Hopper-class (H100/H200, SM90), same DeepEP V1
  requirement as Phase 3. Exact type/cloud chosen live at rental time
  against real-time availability, same practice as Phases 2-3.

## 9. Non-goals

- Cross-node networking, real distributed KV-cache transfer -- one
  NVLink node only (§2).
- Paged/block KV cache, priority scheduling, preemption, dynamic
  rebalancing between pools -- fixed 2+2 split, dense padded batching
  (§6).
- Beating vLLM/SGLang's real numbers -- Phase 6 owns that comparison.
  Phase 4's claim is limited to "disaggregation helps this project's own
  co-located baseline," an internal, controlled comparison.
- Quantization and speculative decoding -- Phase 5.
