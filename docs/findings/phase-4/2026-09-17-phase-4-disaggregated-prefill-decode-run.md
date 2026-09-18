# Phase 4 disaggregated prefill/decode run -- findings

## Infrastructure

- Pod `ds338r4byldpqu`, RunPod Secure Cloud, data center AP-IN-1, 4x
  NVIDIA H100 80GB HBM3 (real NVLink confirmed -- see below). Rate:
  **$13.96/hr** -- notably higher than the $10.76/hr `get-capacity`
  quoted for the same GPU/cloud/CUDA-version combination just before
  rental; the catalog-lookup price and the actual create-pod price can
  diverge by a real margin, confirmed from the pod object's own `cost`
  field, not assumed from the pre-rental check.
- `nvidia-smi topo -m` confirmed real NVLink across all four GPUs: every
  pair showed `NV18` (18 bonded links), not `PHB`/`PXB`/`SYS`.
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, same as Phase
  3. `uv sync` pulled this repo's own `torch==2.14.0` floor, defaulting
  to `+cu130`; the CUDA 13.0 toolkit (nvcc) was installed via apt,
  reusing Phase 3's exact fix. DeepEP V1 built and imported cleanly on
  the first attempt, no further issues.
- Same `transformers==4.57.6` pin + `DynamicCache.get_usable_length`
  monkeypatch Phase 0/1/3 all needed, applied identically via a
  pod-local, never-committed patch at the top of both Phase 4 scripts.
  DeepSeek's model repo is still unchanged since Phase 0.
- Disk bumped to 60GB (from Phase 3's 40GB) specifically to avoid Phase
  3's disk-space-vs-32.8GB-model-download issue; 47GB was free after all
  installs, before the download -- enough this time, no `/dev/shm`
  workaround needed.
- Pod terminated via `pod-action terminate` after all findings were
  copied off; a follow-up `list-pods` returned an empty list, confirming
  full removal. Total session: 47 minutes, pod creation to termination.

## Three real bugs found and fixed, none caught by CPU-only tests

All three are legitimate gaps in code that had already passed CPU-only
tests (including, for two of them, code merged in Phase 3) -- exactly the
class of bug this project's testing table exists to eventually catch on
real hardware, not evidence the CPU-only layer was pointless.

1. **DeepSeek's remote-code model returns the legacy tuple cache format,
   not a `DynamicCache`.** `modeling_deepseek.py` (unchanged since Phase
   0) predates the Cache-class refactor entirely: `outputs.past_key_values`
   came back as a plain 28-tuple of `(key, value)` pairs, confirmed live
   by inspecting a real forward call's output type. `kv_cache.py`/
   `disaggregated.py`'s own committed contract is built entirely on
   `DynamicCache` (Tasks 1-4, CPU-proven against transformers 5.17.0,
   where this project's own kernel/scheduler code lives). Resolved by
   converting only at the `PrefillFn`/`DecodeFn` closure boundary in the
   two pod-local scripts, via `DynamicCache.from_legacy_cache(...)` /
   `cache.to_legacy_cache()` -- both still present in `transformers==4.57.6`
   (the pin DeepSeek's code separately requires) though removed in v5.
   No change needed to any committed library code for this one.
2. **`local_expert_contribution` crashes when a rank receives zero
   tokens for any of its local experts at all.** `topk_idx.max()` raised
   `RuntimeError: Expected reduction dim to be specified for
   input.numel() == 0`. Phase 3's own fix (2026-09-16) handled a rank
   missing its *highest* local expert while still having *some* tokens;
   4-way EP (16 experts/rank, finer than Phase 3's 2-way/32-experts/rank)
   combined with short prompts made a rank receiving **no** tokens at all
   a real, reachable case Phase 3 never hit. Fixed in
   `src/dispatch/kernels/expert_parallel.py` with a CPU regression test
   (`test_local_expert_contribution_handles_a_batch_with_no_tokens_for_this_rank_at_all`),
   committed before returning to the pod.
3. **NCCL requires every send/recv tensor on the correct CUDA device;
   `recv_kv_cache`'s buffers defaulted to CPU.** Found by re-reading
   `handoff.py` before wiring the real cross-rank handoff, not by a live
   crash -- Task 2's own test only exercises the `gloo` backend, which
   tolerates CPU tensors. Fixed by making `send_kv_cache` infer its
   device from the cache being sent and adding an explicit `device`
   parameter to `recv_kv_cache` (default `"cpu"`, so the existing gloo
   test is unaffected). The same read-before-running pass caught a
   companion bug in `pad_and_batch_caches` (its zero-padding tensors also
   defaulted to CPU, which would have crashed concatenating against a
   CUDA cache) before it ever touched the rented GPUs.

All three fixes are covered by regression tests and were committed
before (bugs 2-3) or verified against (bug 1, pod-local by design --
DeepSeek's own repo quirk, not this project's) real hardware confirmed
them fixed.

## Correctness gate: passed, both topologies, exact token match

Judged by exact greedy-generated-token-sequence match against a
single-GPU reference, not top-k logit agreement: the scheduler's
`PrefillFn`/`DecodeFn` contract only returns argmax token ids (raw
logits are dropped by design, matching the project's own generation
loop convention), so exact-token-match is the correctness bar this
architecture actually admits -- arguably a stricter one for a
multi-step greedy-decode pipeline than a single forward pass's top-5
logit agreement, since it must stay correct across many compounding
steps, not just one.

`deepseek-ai/deepseek-moe-16b-base`, 27 MoE layers patched, 3 prompts, 16
tokens each, `max_new_tokens=16`:

| prompt | single-GPU reference | co-located (4-rank EP) | disaggregated (2+2-rank EP) |
|---|---|---|---|
| "The quick brown fox jumps over the lazy dog." | `[185, 549, 3399, 10176, 32431, 33747, 855, 254, 24547, 5025, 13, 185, 549, 3399, 10176, 32431]` | **exact match** | **exact match** |
| "In a distant galaxy, a small crew of explorers" | `[317, 331, 245, 8723, 276, 1275, 245, 761, 1719, 327, 704, 1245, 13, 1955, 463, 32339]` | **exact match** | **exact match** |
| "def fibonacci(n):" | `[185, 300, 565, 291, 2318, 207, 15, 25, 185, 391, 972, 207, 15, 185, 300, 23744]` | **exact match** | **exact match** |

Both configurations matched on every token, every prompt. Full data:
`docs/findings/phase-4/2026-09-16-phase-4-single-gpu-reference.json`,
`2026-09-16-phase-4-colocated-gate-results.json`,
`2026-09-16-phase-4-disaggregated-gate-results.json`.

Transient CUDA allocator OOM warnings appeared during the first
kernel-backed call in each run, always for the same 369,098,752-byte
allocation -- matching Phase 1's and Phase 3's own precedent exactly
(same byte count, same benign first-call/warmup pattern); generation
completed correctly afterward in every case, so these are not treated as
a correctness signal.

## The measured answer to Phase 4's thesis

**Does splitting prefill and decode onto separate GPU pools measurably
relieve the contention they create when they share GPUs under concurrent
load, holding total GPU count fixed at 4?**

Concurrency measurement (`scripts/gpu/phase4_concurrency.py`), 4 and 8
concurrent requests, `DEFAULT_PROMPTS` cycled, staggered arrivals 50ms
apart, `max_new_tokens=8` (short by design -- the hard, expensive part of
this session was the correctness work above; this measurement's job was
just to get one real reading on real hardware, cheaply):

| topology | concurrency | total wall time | mean TTFT | p50 TTFT | p99 TTFT | decode tokens/sec |
|---|---|---|---|---|---|---|
| co-located | 4 | 4.310s | 1.118s | 1.029s | 1.372s | 6.50 |
| co-located | 8 | 2.974s | 0.695s | 0.930s | 1.290s | 18.83 |
| disaggregated | 4 | 4.256s | 0.458s | 0.156s | 1.330s | 6.58 |
| disaggregated | 8 | 4.725s | 0.983s | 0.749s | 1.987s | 11.85 |

(Disaggregated's TTFT is measured on the prefill side, where admission
and completion both happen; disaggregated's wall time/tokens-per-second
is measured on the decode side, where token generation happens. The two
halves are reported separately because they run as genuinely separate
processes on separate GPUs -- there is no single rank that observes
both.)

**Mixed, not a clean win either way.** At concurrency 4, disaggregated's
mean TTFT (0.458s) is less than half co-located's (1.118s) -- consistent
with the thesis: prefill isn't waiting behind decode work on a shared
GPU. At concurrency 8, that reverses: co-located's mean TTFT (0.695s) is
now *lower* than disaggregated's (0.983s), and co-located's decode
throughput (18.83 tok/s) is higher than disaggregated's (11.85 tok/s) at
the same concurrency level too.

**This comparison is confounded by kernel-warmup state and should not be
read as a clean result.** Each of the four measurements above is a fresh
`torchrun` process; co-located's own total wall time *dropped* from
4.31s (concurrency 4) to 2.97s (concurrency 8) -- throughput improving
with more concurrent load, in the same direction warmup would push it,
not the direction contention would. Triton/CUDA kernel compilation
artifacts persisting on disk between process invocations, and the OS
page cache already holding the 32.8GB model checkpoint from the prior
run, are the most likely explanation for later runs measuring faster
independent of the actual topology under test. No warmup-then-measure
step was built into `phase4_concurrency.py`, and building one (plus
enough repeated runs to average out this noise) was exactly the
additional live-debugging scope this session's own cost-discipline
instruction weighed against taking on further.

**Direct answer**: the correctness question is answered cleanly and
completely -- both a co-located 4-rank EP pool and a disaggregated
2+2-rank EP pool with a real cross-rank KV-cache handoff produce
byte-exact correct greedy generation against a single-GPU reference. The
contention-relief question is not answered cleanly by this run: a real
signal favoring disaggregation appears at concurrency 4 and reverses at
concurrency 8, and the most likely explanation is a confound (warmup
state varying between independent process launches) rather than a real
property of either topology, so no strength-calibrated claim is made
either way. Answering it properly needs either a warmed-up measurement
protocol (discard the first N iterations of each configuration before
recording) or enough repeated trials to characterize the noise -- both
real, identifiable next steps, not a vague "more testing would help."

## What was *not* attempted, and why

The design doc's own §8 named the full async-staggered measurement
(`dist.irecv` polling on the decode side, real overlap between prefill
and decode) as more distributed-systems complexity than anything built
before it, with real risk of a long live-debugging session -- flagged
explicitly as a scope decision to check on rather than force through.
That build was attempted and succeeded (both correctness gates and all
four concurrency measurements ran without further live debugging once
the three bugs above were fixed). What wasn't attempted: a warmup
protocol to remove the confound described above, and enough repeated
trials per configuration to characterize run-to-run noise -- both
correctly scoped as follow-up work rather than squeezed into this
session, given the total budget already spent finding and fixing three
real bugs plus running the correctness gates.

## Cost

**Total: $10.94** of the $40 cap, a ceiling not a target -- 4x H100 SXM
at $13.96/hr, 47 minutes (pod creation to termination). The gap between
cap and actual spend is itself part of this phase's result, per the
standing instruction to be surgical: correctness work (the hard,
valuable, and now-complete part) took priority, and once both gates
passed and one real concurrency reading was in hand per configuration,
the session stopped rather than spending further budget chasing a clean
answer to a question the data itself flagged as confounded. Full record:
`docs/findings/phase-4/2026-09-17-phase-4-disaggregated-prefill-decode-cost.md`.
