# Phase 4 runbook: disaggregated prefill/decode

Budget cap: $40, a ceiling not a target. Hopper-class (H100/H200, SM90)
required, same as Phase 3.

**What actually happened (2026-09-17), read this before following the
steps below as written:** everything in this runbook ran successfully,
but real hardware and a real live `get-capacity` price quote both
corrected the plan before any measurement happened. Total cost:
**$10.94** of the $40 cap (docs/findings/
2026-09-17-phase-4-disaggregated-prefill-decode-cost.md). Full account,
including the correctness gate results and the measured contention
comparison: `docs/findings/2026-09-17-phase-4-disaggregated-prefill-decode-run.md`.

1. Create and wait for the 4-GPU pod, reusing Phase 3's generic
   `--gpu-count` support:

   ```bash
   uv run python scripts/gpu/provision.py create --name phase-4-disaggregated \
     --gpu-type <Hopper-class type available at rental time> --gpu-count 4 \
     --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 --cloud secure --disk-gb 60
   uv run python scripts/gpu/provision.py wait <pod-id>
   ```

   **What actually happened:** `get-capacity` quoted 4x H100 SXM at
   $10.76/hr (CUDA 13.0, Secure cloud) -- the pod actually billed at
   **$13.96/hr**, ~30% higher than the catalog check. Not a planning
   error to avoid, just a real gap between the catalog-lookup price and
   the actual create-pod price worth expecting; confirm the real rate
   from the pod object itself (its own `cost` field) immediately after
   creation, not just from the pre-rental catalog check. Disk bumped to
   60GB (from Phase 3's 40GB) specifically to avoid Phase 3's
   disk-space-vs-32.8GB-model-download issue; it was enough (47GB free
   after all installs, before the download).

2. **Verify real NVLink across all four GPUs**:

   ```bash
   nvidia-smi topo -m
   ```

   Expected: `NV#` between every pair of the four GPUs. **Confirmed**:
   every pair showed `NV18` (18 bonded links).

3. Install DeepEP V1 (legacy `Buffer`) directly -- do not attempt V2,
   Phase 3 already confirmed it cannot work on this rental tier (no GPU
   Fabric Manager). Install the matching CUDA toolkit first (this
   image's nvcc was missing on PATH; NVIDIA's apt repo was already
   configured):

   ```bash
   apt-get update -qq
   DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
     cuda-toolkit-13-0 libibverbs-dev rdma-core ninja-build
   export PATH=/usr/local/cuda-13.0/bin:$PATH

   uv sync   # pulls torch==2.14.0+cu130, matching the 13.0 toolkit

   git clone https://github.com/deepseek-ai/DeepEP.git
   cd DeepEP
   uv pip install --python ../.venv/bin/python "nvidia-nccl-cu13>=2.30.4"
   PATH=../.venv/bin:$PATH ../.venv/bin/python setup.py install
   cd ..
   .venv/bin/python -c "from deep_ep import Buffer; print('V1 Buffer OK')"
   ```

   **What actually happened:** this worked cleanly on the first attempt
   -- DeepEP V1 built and imported with no further issues, reusing
   exactly the CUDA-13-toolkit fix Phase 3 had to discover live.

4. Clone this repo, sync, and run the full CPU-testable suite one more
   time on the pod itself:

   ```bash
   git clone https://github.com/bsreecharanreddy/dispatch.git dispatch
   cd dispatch
   git checkout phase-4-disaggregated-prefill-decode
   uv sync
   uv run pytest tests/unit -v -m "not gpu"
   ```

   Expected and confirmed: all 100 pass (private repo -- cloned via an
   `x-access-token` HTTPS URL piped through stdin, never as a bare
   command-line argument).

5. **Build the two EP sub-groups and wire the real forward closures.**
   `torchrun --nproc_per_node=4` gives a 4-rank NCCL world; every rank
   calls `dist.new_group([0, 1])` then `dist.new_group([2, 3])`
   identically (a collective call every rank must join regardless of
   membership), then branches on `rank in (0, 1)` to pick its own
   sub-group. Each sub-group builds its own DeepEP `Buffer` scoped to
   that sub-group and calls `patch_moe_infer_ep(model, matmul, buffer,
   pool_rank, n_ranks=2)` unchanged, where `pool_rank =
   dist.get_rank(group=pool_group)` -- the **sub-group-local** rank (0 or
   1), not the global rank.

   **What actually happened, two real fixes needed before this worked:**
   - DeepSeek's remote-code modeling file (unchanged since Phase 0)
     predates the `transformers` Cache-class refactor entirely: it
     returns `past_key_values` as the **legacy tuple-of-(key,
     value)-per-layer** format, not a `DynamicCache` object, confirmed
     live by inspecting a real forward call's output type. Every
     `PrefillFn`/`DecodeFn` closure converts at this boundary only, via
     `DynamicCache.from_legacy_cache(...)` / `cache.to_legacy_cache()`
     (both still present in `transformers==4.57.6`, though removed in
     v5) -- `kv_cache.py`/`disaggregated.py`'s own committed
     `DynamicCache`-based contract, proven by Tasks 1-4's CPU tests,
     needed no changes at all.
   - `local_expert_contribution` (`expert_parallel.py`, committed in
     Phase 3) crashed with `RuntimeError: max(): Expected reduction dim
     to be specified for input.numel() == 0` -- a rank's `topk_idx` came
     back **completely empty** after DeepEP dispatch. Finer sharding
     (4-way EP, 16 experts/rank) than Phase 3's 2-way (32 experts/rank)
     plus short prompts made this a real, reachable case Phase 3 never
     hit. Fixed in `src/dispatch/kernels/expert_parallel.py`
     (`index_span` now sizes off `local_expert_ids` alone when
     `topk_idx` is empty), with a CPU regression test, found and fixed
     for $0 before spending more GPU time debugging it live.

   Rank-to-rank KV-cache handoff for the disaggregated path uses
   `handoff.py`'s `send_kv_cache`/`recv_kv_cache` over the full 4-rank
   `dist.group.WORLD`, not a sub-group (prefill and decode ranks must
   talk to each other). **A third real fix was needed here too, found
   by re-reading the committed code before running it, not by a live
   crash**: NCCL (unlike the `gloo` backend Task 2's own CPU test uses)
   requires every send/recv tensor to be on the correct CUDA device;
   `recv_kv_cache`'s buffers defaulted to CPU. Fixed by making
   `send_kv_cache` infer its device from the cache being sent and adding
   an explicit `device` parameter to `recv_kv_cache`. A companion device
   bug in `pad_and_batch_caches` (its zero-padding tensors also defaulted
   to CPU) was caught the same way, before it could crash a real batched
   decode step.

6. **Correctness gate -- must pass for both configurations before any
   measurement runs.** `scripts/gpu/phase4_correctness_gate.py --mode
   single` first (also warms the model cache), then `--mode colocated`
   and `--mode disaggregated` via `torchrun --nproc_per_node=4`. Every
   rank in a pool runs the identical scheduler logic (`PrefillWorker`/
   `DecodeWorker`/`ColocatedWorker`) against the identical prompt list --
   since that logic is pure and deterministic, every rank's `model(...)`
   calls stay naturally synchronized without a driver/follower split,
   the same pattern Phase 3's own correctness gate used. Correctness is
   judged by exact greedy-token-sequence match against the single-GPU
   reference (not top-k logit agreement -- the scheduler's `PrefillFn`/
   `DecodeFn` contract only returns argmax token ids, dropping raw
   logits by design), which is at least as strict a bar for a
   greedy-decode pipeline.

   **Confirmed: both passed, byte-exact token match**, all 3 prompts, 16
   tokens each. See the findings doc for the full token sequences.

7. **Only after both gates passed**, ran the concurrency measurement:
   co-located vs. disaggregated, same 4 GPUs, at 4 and 8 concurrent
   requests (`scripts/gpu/phase4_concurrency.py`, staggered arrivals 50ms
   apart, 8 max new tokens per request -- short by design, to keep this
   step cheap once the hard correctness work was already done). The
   disaggregated decode side uses `dist.irecv` + polling (`is_completed()`)
   to interleave receiving new prefill handoffs with stepping its own
   already-active decode requests, rather than blocking on `dist.recv`.

   Full measured numbers: findings doc. **Be surgical** held throughout
   -- total session was 47 minutes end to end.

8. Copied all console output and result JSON off the pod via `scp`
   before doing anything else.

9. Tore down immediately:

   ```
   pod-action terminate ds338r4byldpqu
   ```

   Confirmed via a follow-up `list-pods` returning an empty list.

10. Recorded the measured cost via `write_cost_record`, reused
    unmodified: `docs/findings/2026-09-17-phase-4-disaggregated-prefill-decode-cost.md`.
