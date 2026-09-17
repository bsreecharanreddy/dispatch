# Phase 4: Disaggregated Prefill/Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, on real hardware, whether splitting prefill and decode
onto separate GPU pools relieves the contention they create when they
share GPUs under concurrent load -- holding total GPU count fixed at 4,
comparing one co-located 4-rank EP pool against two independent 2-rank EP
pools (prefill, decode) connected by a KV-cache handoff.

**Architecture:** A continuous-batching scheduler (`PrefillWorker`,
`DecodeWorker` in `disaggregated.py`; `ColocatedWorker` in
`colocated.py`) built entirely against a pluggable `PrefillFn`/`DecodeFn`
callable pair -- the scheduler knows nothing about DeepEP, EP pools, or
even that this is a distributed setting. `kv_cache.py` supplies the
slice/pad/rebatch bookkeeping a dense (non-paged) continuous batch needs;
`handoff.py` supplies the cross-rank cache transfer over plain
`torch.distributed` send/recv, provably correct over CPU-only `gloo`
before it ever needs NCCL or real GPUs. Only the final, paid task wires
real closures around DeepSeekMoE-16B -- patched with Phase 3's existing,
unchanged `make_ep_moe_infer`/`patch_moe_infer_ep` -- into the
`PrefillFn`/`DecodeFn` shape.

**Tech Stack:** `transformers` v5's `DynamicCache` (its modern, surviving
public API only -- `to_legacy_cache`/`from_legacy_cache`/`key_cache`/
`value_cache` were removed in v5, confirmed live 2026-09-16 against
`transformers`' own v5 migration guide and `cache_utils.py` source).
`torch.distributed` (`gloo` for CPU tests, `NCCL` for the real run).
DeepEP V1 (legacy `Buffer`), Phase 3's `expert_parallel.py`, reused
unchanged. RunPod's REST API via `scripts/gpu/provision.py`
(`--gpu-count 4`, already generic from Phase 3's Task 1 -- no new
provisioning code needed).

**Spec:** `docs/design/2026-09-16-phase-4-disaggregated-prefill-decode.md`
(all sections); `docs/design/2026-09-15-phase-3-multi-gpu-expert-parallel-serving.md`
and `src/dispatch/kernels/expert_parallel.py` (Phase 3's EP machinery,
reused unchanged); `src/dispatch/benchmark/harness.py`,
`src/dispatch/benchmark/reference.py` (Phase 0/1's generation and
correctness-gate helpers, reused unchanged).

## Global Constraints

- **Budget cap: $40, a ceiling not a target.** Be surgical: maximize
  what's proven for $0 before any rental (design doc section 8), and the
  gap between the cap and the actual spend is itself a reportable part of
  Phase 4's result (Task 6), the same way Phase 3 reported $13.03 of its
  $25 cap.
- **Hardware: Hopper-class (H100/H200, SM90), one 4-GPU NVLink/SXM node.**
  Verify real NVLink across *all four* GPUs before spending budget on
  anything else -- `nvidia-smi topo -m` must show `NV#` between every
  pair, not just GPU 0-1.
- **DeepEP V1 (legacy `Buffer`), not V2.** Phase 3 confirmed live
  (2026-09-16) that V2's `ElasticBuffer` cannot work on this rental tier
  -- no GPU Fabric Manager, `nvidia-smi -q` reports `GPU Fabric GUID:
  N/A`. Do not re-attempt V2; start Task 5 from V1 directly.
- **`transformers` v5's `DynamicCache` API, confirmed live 2026-09-16**
  against `transformers`' own v5 migration guide and
  `src/transformers/cache_utils.py` (`main` branch): `to_legacy_cache`,
  `from_legacy_cache`, and the `key_cache`/`value_cache` list attributes
  were **removed** in v5. Use only the surviving public API this plan's
  code is built on: `cache.layers[i].keys` / `.values` (tensors shaped
  `(batch, num_heads, seq_len, head_dim)`), `DynamicCache(ddp_cache_data=
  [(key, value), ...])` to construct one from raw per-layer tensors,
  `cache.get_seq_length()`, `cache.batch_size`. Do not use
  `to_legacy_cache`/`from_legacy_cache`/`key_cache`/`value_cache` --
  they will raise `AttributeError` on this repo's pinned
  `transformers>=5.17.0`.
- **Dense, padded batching -- not a paged/block KV cache.** Every batched
  decode step re-pads every active request's cache to the batch's
  current max real length (design doc section 6). This applies
  identically to the co-located and disaggregated paths, so it should not
  bias their *relative* comparison, but it does mean absolute
  throughput/TTFT numbers here are not directly comparable to vLLM's own
  -- say so explicitly in Task 6's findings doc.
- **Left-padding convention.** Every batched forward call (prefill or
  decode) left-pads shorter sequences -- matches this repo's existing
  single-request convention implicitly (no prior padding existed before
  Phase 4) and HF's own default for batched causal-LM generation.
- **Reuse Phase 3's EP machinery unchanged.** `make_ep_moe_infer` and
  `patch_moe_infer_ep` (`src/dispatch/kernels/expert_parallel.py`) are
  not modified by this plan. Task 5 calls them twice -- once per EP
  sub-group (prefill ranks 0-1, decode ranks 2-3) -- exactly as Phase 3
  called them once.
- **Correctness gates every throughput/contention claim.** Both the
  co-located and disaggregated paths must independently pass mutual
  top-k logit agreement against the single-GPU reference (same bar Phase
  1 and Phase 3 used) before Task 5's concurrency measurement runs.
- **Never quote a benchmark number that wasn't measured** on this exact
  run (repo-wide rule).

## File Structure

```text
src/dispatch/serving/__init__.py         # Task 1
src/dispatch/serving/kv_cache.py         # Task 1 (slice_cache, pad_and_batch_caches)
tests/unit/test_kv_cache.py              # Task 1

src/dispatch/serving/handoff.py          # Task 2 (send_kv_cache, recv_kv_cache)
tests/unit/test_handoff.py               # Task 2

src/dispatch/serving/disaggregated.py    # Task 3 (Request, PrefillResult,
                                          # RequestResult, InFlightRequest,
                                          # PrefillFn, DecodeFn,
                                          # run_prefill_batch,
                                          # prefill_result_to_in_flight,
                                          # step_active_requests,
                                          # PrefillWorker, DecodeWorker)
tests/unit/test_disaggregated.py         # Task 3

src/dispatch/serving/colocated.py        # Task 4 (ColocatedWorker)
tests/unit/test_colocated.py             # Task 4

docs/runbooks/phase-4-disaggregated-prefill-decode.md              # Task 5
docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-cost.md  # Task 5 (write_cost_record)
docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-run.md   # Task 6
docs/STATUS.md                                                          # Task 6
```

---

### Task 1: KV-cache slice and pad-batch helpers, proven against a real tiny model on CPU

**Files:**

- Create: `src/dispatch/serving/__init__.py` (empty, matches
  `src/dispatch/{benchmark,kernels}/__init__.py`)
- Create: `src/dispatch/serving/kv_cache.py`
- Test: `tests/unit/test_kv_cache.py`

**Interfaces:**

- Produces: `slice_cache(cache: DynamicCache, index: int, *, keep_last:
  int | None = None) -> DynamicCache`; `pad_and_batch_caches(caches:
  list[DynamicCache], pad_to: int) -> tuple[DynamicCache, torch.Tensor]`.
  Tasks 3-4 build their entire batching loop on these two functions and
  nothing else from this file.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_kv_cache.py`:

```python
"""CPU-only: proves the dense-padding slice/rebatch bookkeeping a
continuous-batching scheduler needs, built entirely on transformers v5's
surviving public DynamicCache API (cache.layers[i].keys/.values, the
ddp_cache_data constructor path) -- to_legacy_cache/from_legacy_cache and
the old key_cache/value_cache attributes were removed in v5 (confirmed
live 2026-09-16 against transformers' own v5 migration guide).
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from dispatch.serving.kv_cache import pad_and_batch_caches, slice_cache

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4


def _make_cache(batch_size: int, seq_len: int, *, seed: int = 0) -> DynamicCache:
    generator = torch.Generator().manual_seed(seed)
    per_layer = [
        (
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM, generator=generator),
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM, generator=generator),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def test_slice_cache_extracts_one_request_without_mutating_the_original() -> None:
    cache = _make_cache(batch_size=2, seq_len=3)
    original_row_0 = cache.layers[0].keys[0].clone()

    sliced = slice_cache(cache, index=1)

    assert sliced.get_seq_length() == 3
    assert sliced.layers[0].keys.shape[0] == 1
    torch.testing.assert_close(sliced.layers[0].keys[0], cache.layers[0].keys[1])
    torch.testing.assert_close(cache.layers[0].keys[0], original_row_0)


def test_slice_cache_with_keep_last_drops_left_padding() -> None:
    cache = _make_cache(batch_size=1, seq_len=5)
    real_tail = cache.layers[0].keys[:, :, -2:, :].clone()

    sliced = slice_cache(cache, index=0, keep_last=2)

    assert sliced.get_seq_length() == 2
    torch.testing.assert_close(sliced.layers[0].keys, real_tail)


def test_pad_and_batch_caches_left_pads_to_a_common_length_and_masks_correctly() -> None:
    short = _make_cache(batch_size=1, seq_len=2, seed=1)
    long_ = _make_cache(batch_size=1, seq_len=5, seed=2)

    batched, attention_mask = pad_and_batch_caches([short, long_], pad_to=5)

    assert batched.get_seq_length() == 5
    assert batched.layers[0].keys.shape[0] == 2
    assert attention_mask.tolist() == [[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]
    torch.testing.assert_close(batched.layers[0].keys[0, :, -2:, :], short.layers[0].keys[0])
    torch.testing.assert_close(batched.layers[0].keys[1], long_.layers[0].keys[0])


def test_pad_and_batch_caches_rejects_a_cache_longer_than_pad_to() -> None:
    long_ = _make_cache(batch_size=1, seq_len=5)

    with pytest.raises(ValueError, match="exceeds"):
        pad_and_batch_caches([long_], pad_to=3)


def test_pad_and_batch_caches_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        pad_and_batch_caches([], pad_to=3)


@pytest.mark.slow
def test_slice_and_rebatch_round_trip_through_a_real_tiny_model() -> None:
    """Proves the helpers produce a cache the real model accepts back and
    continues generating from correctly -- against hf-internal-testing/
    tiny-random-gpt2, this repo's established tiny-model pattern
    (test_harness.py).
    """
    from dispatch.benchmark.harness import load_model

    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")
    input_ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = outputs.past_key_values
    assert isinstance(cache, DynamicCache)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sliced = slice_cache(cache, index=0)
    assert sliced.get_seq_length() == cache.get_seq_length()

    with torch.no_grad():
        continued = model(input_ids=next_token, past_key_values=sliced, use_cache=True)

    assert continued.logits.shape == (1, 1, continued.logits.shape[-1])
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_kv_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch.serving'`.

- [x] **Step 3: Write the implementation**

Create `src/dispatch/serving/__init__.py` (empty file).

Create `src/dispatch/serving/kv_cache.py`:

```python
"""KV-cache slicing and batch (re)assembly for Phase 4's continuous-
batching scheduler. Built entirely on transformers v5's surviving public
DynamicCache API -- to_legacy_cache/from_legacy_cache and the old
key_cache/value_cache attributes were removed in v5 (confirmed live
2026-09-16 against transformers' own v5 migration guide and
src/transformers/cache_utils.py). Dense, padded batching only -- not a
paged/block cache -- the known simplification this project's Phase 4
design doc discloses (section 6): real wasted compute next to a
production engine, applied identically to every configuration this
project measures, so it should not bias their relative comparison.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def slice_cache(cache: DynamicCache, index: int, *, keep_last: int | None = None) -> DynamicCache:
    """Extract one request's KV cache (by batch index) out of a batched
    cache, without mutating the original -- unlike Cache.batch_select_indices,
    which selects in place and offers no way to also drop padding. If
    keep_last is given, keeps only the last keep_last positions along the
    sequence dim, dropping this request's left-padding after a batched
    forward call where its real content is shorter than the batch's
    padded length.
    """
    per_layer = []
    for layer in cache.layers:
        key = layer.keys[index : index + 1]
        value = layer.values[index : index + 1]
        if keep_last is not None:
            key = key[:, :, -keep_last:, :]
            value = value[:, :, -keep_last:, :]
        per_layer.append((key.clone(), value.clone()))
    return DynamicCache(ddp_cache_data=per_layer)


def pad_and_batch_caches(
    caches: list[DynamicCache], pad_to: int
) -> tuple[DynamicCache, torch.Tensor]:
    """Left-pads each cache's key/value tensors to pad_to positions and
    concatenates them along the batch dim, returning the batched cache
    and a matching (batch, pad_to) attention_mask (0 = pad, 1 = real).
    """
    if not caches:
        raise ValueError("caches must not be empty")
    num_layers = len(caches[0].layers)
    per_layer_keys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    per_layer_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    attention_mask = torch.zeros(len(caches), pad_to, dtype=torch.long)

    for row, cache in enumerate(caches):
        seq_len = cache.get_seq_length()
        if seq_len > pad_to:
            raise ValueError(f"cache seq_len={seq_len} exceeds pad_to={pad_to}")
        attention_mask[row, pad_to - seq_len :] = 1
        pad_len = pad_to - seq_len
        for layer_idx, layer in enumerate(cache.layers):
            key, value = layer.keys, layer.values
            if pad_len > 0:
                pad_shape = (1, key.shape[1], pad_len, key.shape[3])
                key = torch.cat([torch.zeros(pad_shape, dtype=key.dtype), key], dim=2)
                value = torch.cat([torch.zeros(pad_shape, dtype=value.dtype), value], dim=2)
            per_layer_keys[layer_idx].append(key)
            per_layer_values[layer_idx].append(value)

    per_layer = [
        (torch.cat(per_layer_keys[i], dim=0), torch.cat(per_layer_values[i], dim=0))
        for i in range(num_layers)
    ]
    return DynamicCache(ddp_cache_data=per_layer), attention_mask
```

- [x] **Step 4: Run to verify the fast tests pass**

Run: `uv run pytest tests/unit/test_kv_cache.py -v -m "not slow"`
Expected: 5 pass (the `@pytest.mark.slow` real-model test is excluded).

- [x] **Step 5: Run the slow real-model test**

Run: `uv run pytest tests/unit/test_kv_cache.py -v -m slow`
Expected: PASS (downloads `hf-internal-testing/tiny-random-gpt2` on first
run, network required, no GPU).

- [x] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/dispatch/serving tests/unit/test_kv_cache.py && uv run ruff format --check src/dispatch/serving tests/unit/test_kv_cache.py && uv run mypy src tests`
Expected: clean.

- [x] **Step 7: Commit**

```bash
git add src/dispatch/serving/__init__.py src/dispatch/serving/kv_cache.py tests/unit/test_kv_cache.py
git commit -m "feat: add KV-cache slice/pad-batch helpers for continuous batching"
```

---

### Task 2: Cross-rank KV-cache handoff, proven over CPU-only gloo before any GPU is involved

**Files:**

- Create: `src/dispatch/serving/handoff.py`
- Test: `tests/unit/test_handoff.py`

**Interfaces:**

- Produces: `send_kv_cache(cache: DynamicCache, dst: int, group:
  dist.ProcessGroup) -> None`; `recv_kv_cache(src: int, group:
  dist.ProcessGroup, *, num_layers: int, num_heads: int, head_dim: int,
  dtype: torch.dtype) -> DynamicCache`. Task 5 calls these exact
  functions unchanged, with `backend="nccl"` instead of `"gloo"` and real
  ranks instead of a CPU 2-process test.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_handoff.py`:

```python
"""CPU-only, multi-process: proves send_kv_cache/recv_kv_cache round-trip
correctly using torch.distributed's gloo backend -- no GPU, no DeepEP, no
NCCL. Task 5 reuses these exact functions unchanged with the NCCL
backend on the real rented 4-GPU node; only init_process_group's backend
argument and the real ranks differ.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import DynamicCache

from dispatch.serving.handoff import recv_kv_cache, send_kv_cache

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4
SEQ_LEN = 3
PORT = 29513


def _make_cache(seed: int) -> DynamicCache:
    generator = torch.Generator().manual_seed(seed)
    per_layer = [
        (
            torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=generator),
            torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=generator),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def _worker(rank: int, world_size: int, result_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    if rank == 0:
        send_kv_cache(_make_cache(seed=42), dst=1, group=group)
    else:
        received = recv_kv_cache(
            src=0,
            group=group,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
        )
        expected = _make_cache(seed=42)
        ok = all(
            torch.equal(received.layers[i].keys, expected.layers[i].keys)
            and torch.equal(received.layers[i].values, expected.layers[i].values)
            for i in range(NUM_LAYERS)
        )
        torch.save({"ok": ok, "seq_len": received.get_seq_length()}, result_path)

    dist.destroy_process_group()


@pytest.mark.slow
def test_send_and_recv_kv_cache_round_trip_over_gloo(tmp_path: object) -> None:
    result_path = str(tmp_path / "result.pt")  # type: ignore[operator]

    mp.spawn(_worker, args=(2, result_path), nprocs=2, join=True)

    result = torch.load(result_path, weights_only=True)
    assert result["ok"] is True
    assert result["seq_len"] == SEQ_LEN
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_handoff.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch.serving.handoff'`.

- [x] **Step 3: Write the implementation**

Create `src/dispatch/serving/handoff.py`:

```python
"""Cross-rank KV-cache handoff between the prefill and decode EP pools,
built on plain torch.distributed point-to-point send/recv so the exact
same code runs over gloo (CPU, this file's own tests) and NCCL (real
GPUs, Task 5) -- the transport logic gets proven once, for $0, before it
is ever run against real hardware (design doc section 8's "maximize
what's proven before any rental").
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from transformers import DynamicCache


def send_kv_cache(cache: DynamicCache, dst: int, group: dist.ProcessGroup) -> None:
    seq_len = torch.tensor([cache.get_seq_length()], dtype=torch.int64)
    dist.send(seq_len, dst=dst, group=group)
    for layer in cache.layers:
        dist.send(layer.keys.contiguous(), dst=dst, group=group)
        dist.send(layer.values.contiguous(), dst=dst, group=group)


def recv_kv_cache(  # noqa: PLR0913 -- the receiver can't infer shape/dtype from the wire, they must be passed
    src: int,
    group: dist.ProcessGroup,
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> DynamicCache:
    seq_len_tensor = torch.zeros(1, dtype=torch.int64)
    dist.recv(seq_len_tensor, src=src, group=group)
    seq_len = int(seq_len_tensor.item())

    per_layer = []
    for _ in range(num_layers):
        key = torch.zeros(1, num_heads, seq_len, head_dim, dtype=dtype)
        value = torch.zeros(1, num_heads, seq_len, head_dim, dtype=dtype)
        dist.recv(key, src=src, group=group)
        dist.recv(value, src=src, group=group)
        per_layer.append((key, value))
    return DynamicCache(ddp_cache_data=per_layer)
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_handoff.py -v -m slow`
Expected: PASS.

- [x] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/dispatch/serving/handoff.py tests/unit/test_handoff.py && uv run ruff format --check src/dispatch/serving/handoff.py tests/unit/test_handoff.py && uv run mypy src tests`
Expected: clean. (Add `"src/dispatch/serving/handoff.py" = ["PLR0913"]`
to `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` only if the
inline `# noqa: PLR0913` above doesn't satisfy ruff.)

- [x] **Step 6: Commit**

```bash
git add src/dispatch/serving/handoff.py tests/unit/test_handoff.py
git commit -m "feat: add cross-rank KV-cache handoff, proven over gloo"
```

---

### Task 3: Continuous-batching prefill and decode workers

**Files:**

- Create: `src/dispatch/serving/disaggregated.py`
- Test: `tests/unit/test_disaggregated.py`

**Interfaces:**

- Consumes: `slice_cache`, `pad_and_batch_caches` (Task 1, unchanged).
- Produces: `Request`, `PrefillResult`, `RequestResult`,
  `InFlightRequest` (dataclasses); `PrefillFn = Callable[[torch.Tensor,
  torch.Tensor], tuple[torch.Tensor, DynamicCache]]`; `DecodeFn =
  Callable[[torch.Tensor, DynamicCache, torch.Tensor, torch.Tensor],
  tuple[torch.Tensor, DynamicCache]]`; `run_prefill_batch(prefill_fn,
  batch, clock_fn) -> list[PrefillResult]`;
  `prefill_result_to_in_flight(result) -> InFlightRequest`;
  `step_active_requests(active, decode_fn, clock_fn) ->
  tuple[list[InFlightRequest], list[RequestResult]]`; `PrefillWorker`;
  `DecodeWorker`. Task 4's `ColocatedWorker` imports
  `run_prefill_batch`, `prefill_result_to_in_flight`,
  `step_active_requests`, and every dataclass/type alias from this file
  unchanged. Task 5 wires real closures matching `PrefillFn`/`DecodeFn`
  exactly; nothing in this file changes at that point.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_disaggregated.py`:

```python
"""CPU-only: proves the continuous-batching scheduler's admission,
padding, and completion bookkeeping against a fake prefill/decode
forward pass -- no GPU, no DeepEP, no real model. Task 5 wires the same
PrefillFn/DecodeFn shapes to the real DeepSeekMoE-16B model and DeepEP EP
pools; nothing here changes at that point.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from dispatch.serving.disaggregated import DecodeWorker, PrefillResult, PrefillWorker, Request

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4


def _fake_cache(batch_size: int, seq_len: int) -> DynamicCache:
    per_layer = [
        (
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def _fake_prefill_fn(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask
    batch_size, seq_len = input_ids.shape
    first_tokens = torch.arange(batch_size) + 100  # distinctive, deterministic per row
    return first_tokens, _fake_cache(batch_size, seq_len)


def _fake_decode_fn(
    next_input_ids: torch.Tensor,
    cache: DynamicCache,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask, position_ids
    batch_size = next_input_ids.shape[0]
    new_seq_len = cache.get_seq_length() + 1
    next_tokens = next_input_ids.squeeze(1) + 1  # deterministic increment, easy to assert on
    return next_tokens, _fake_cache(batch_size, new_seq_len)


def test_prefill_worker_drains_up_to_batch_size_and_produces_per_request_caches() -> None:
    worker = PrefillWorker(_fake_prefill_fn, batch_size=2)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=4))
    worker.submit(Request("b", torch.tensor([[1, 2]]), max_new_tokens=4))
    worker.submit(Request("c", torch.tensor([[1]]), max_new_tokens=4))

    results = worker.step()

    assert [r.request_id for r in results] == ["a", "b"]
    assert results[0].cache.get_seq_length() == 3  # unpadded back to its own prompt length
    assert results[1].cache.get_seq_length() == 2
    assert results[0].first_token_id == 100
    assert results[1].first_token_id == 101

    remaining = worker.step()
    assert [r.request_id for r in remaining] == ["c"]


def test_prefill_worker_step_with_nothing_waiting_returns_empty() -> None:
    worker = PrefillWorker(_fake_prefill_fn, batch_size=2)

    assert worker.step() == []


def test_decode_worker_admits_and_completes_requests_after_max_new_tokens() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    cache = _fake_cache(1, 3)
    worker.admit(
        PrefillResult("a", cache, first_token_id=5, ttft=0.1, max_new_tokens=2, eos_token_id=None)
    )

    first_step = worker.step()
    assert first_step == []  # only 1 of 2 tokens generated so far

    second_step = worker.step()
    assert len(second_step) == 1
    assert second_step[0].request_id == "a"
    assert second_step[0].generated_token_ids == [5, 6]


def test_decode_worker_completes_a_request_early_on_eos() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    cache = _fake_cache(1, 3)
    worker.admit(
        PrefillResult("a", cache, first_token_id=5, ttft=0.1, max_new_tokens=10, eos_token_id=6)
    )

    results = worker.step()

    assert len(results) == 1
    assert results[0].generated_token_ids == [5, 6]


def test_decode_worker_batches_requests_with_different_cache_lengths() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=4)
    worker.admit(
        PrefillResult(
            "short",
            _fake_cache(1, 2),
            first_token_id=1,
            ttft=0.1,
            max_new_tokens=1,
            eos_token_id=None,
        )
    )
    worker.admit(
        PrefillResult(
            "long",
            _fake_cache(1, 5),
            first_token_id=1,
            ttft=0.1,
            max_new_tokens=1,
            eos_token_id=None,
        )
    )

    results = worker.step()

    assert {r.request_id for r in results} == {"short", "long"}


def test_decode_worker_respects_batch_size_admitting_only_free_slots() -> None:
    worker = DecodeWorker(_fake_decode_fn, batch_size=1)
    worker.admit(
        PrefillResult(
            "a", _fake_cache(1, 2), first_token_id=1, ttft=0.1, max_new_tokens=5, eos_token_id=None
        )
    )
    worker.admit(
        PrefillResult(
            "b", _fake_cache(1, 2), first_token_id=1, ttft=0.1, max_new_tokens=1, eos_token_id=None
        )
    )

    first_step = worker.step()
    assert first_step == []  # "a" admitted, needs 5 tokens; "b" still waiting

    for _ in range(4):
        worker.step()
    a_result = worker.step()
    assert [r.request_id for r in a_result] == ["a"]

    b_result = worker.step()  # "b" only now gets its first decode step
    assert b_result == []
    b_result = worker.step()
    assert [r.request_id for r in b_result] == ["b"]
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_disaggregated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch.serving.disaggregated'`.

- [x] **Step 3: Write the implementation**

Create `src/dispatch/serving/disaggregated.py`:

```python
"""Continuous-batching prefill and decode workers for Phase 4's
disaggregated serving path. Both workers are forward-pass-agnostic --
they take a plain PrefillFn/DecodeFn callable and know nothing about
DeepEP, EP pools, or even that this is a distributed setting. Task 5
wires the real DeepSeekMoE-16B model (patched with Phase 3's
make_ep_moe_infer/patch_moe_infer_ep, one EP pool per role) into exactly
these two callable shapes; nothing in this file changes when that
happens. run_prefill_batch, prefill_result_to_in_flight, and
step_active_requests are exported (not underscore-prefixed) because
colocated.py's ColocatedWorker reuses them unchanged -- a decode step is
a decode step regardless of whether prefill happened on the same GPU
this iteration or a different one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from dispatch.serving.kv_cache import pad_and_batch_caches, slice_cache

PrefillFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, DynamicCache]]
DecodeFn = Callable[
    [torch.Tensor, DynamicCache, torch.Tensor, torch.Tensor], tuple[torch.Tensor, DynamicCache]
]


@dataclass
class Request:
    request_id: str
    prompt_ids: torch.Tensor  # (1, prompt_len)
    max_new_tokens: int
    eos_token_id: int | None = None


@dataclass
class PrefillResult:
    request_id: str
    cache: DynamicCache
    first_token_id: int
    ttft: float
    max_new_tokens: int
    eos_token_id: int | None


@dataclass
class RequestResult:
    request_id: str
    generated_token_ids: list[int]
    ttft: float
    completion_time: float


@dataclass
class InFlightRequest:
    request_id: str
    cache: DynamicCache
    seq_len: int
    next_input_id: torch.Tensor
    generated_token_ids: list[int]
    max_new_tokens: int
    eos_token_id: int | None
    ttft: float


def run_prefill_batch(
    prefill_fn: PrefillFn,
    batch: list[tuple[Request, float]],
    clock_fn: Callable[[], float],
) -> list[PrefillResult]:
    """Batches a list of (request, admitted_at) pairs into one padded
    prefill_fn call and slices the result back into per-request, unpadded
    PrefillResults. Shared by PrefillWorker.step() and colocated.py's
    ColocatedWorker.
    """
    requests = [r for r, _ in batch]
    admitted_at = [t for _, t in batch]
    max_len = max(r.prompt_ids.shape[1] for r in requests)
    input_ids = torch.zeros(len(requests), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(requests), max_len, dtype=torch.long)
    for i, r in enumerate(requests):
        prompt_len = r.prompt_ids.shape[1]
        input_ids[i, max_len - prompt_len :] = r.prompt_ids[0]
        attention_mask[i, max_len - prompt_len :] = 1

    first_token_ids, cache = prefill_fn(input_ids, attention_mask)
    now = clock_fn()

    results = []
    for i, r in enumerate(requests):
        prompt_len = r.prompt_ids.shape[1]
        results.append(
            PrefillResult(
                request_id=r.request_id,
                cache=slice_cache(cache, i, keep_last=prompt_len),
                first_token_id=int(first_token_ids[i].item()),
                ttft=now - admitted_at[i],
                max_new_tokens=r.max_new_tokens,
                eos_token_id=r.eos_token_id,
            )
        )
    return results


def prefill_result_to_in_flight(result: PrefillResult) -> InFlightRequest:
    return InFlightRequest(
        request_id=result.request_id,
        cache=result.cache,
        seq_len=result.cache.get_seq_length(),
        next_input_id=torch.tensor([[result.first_token_id]], dtype=torch.long),
        generated_token_ids=[result.first_token_id],
        max_new_tokens=result.max_new_tokens,
        eos_token_id=result.eos_token_id,
        ttft=result.ttft,
    )


def step_active_requests(
    active: list[InFlightRequest], decode_fn: DecodeFn, clock_fn: Callable[[], float]
) -> tuple[list[InFlightRequest], list[RequestResult]]:
    """Pads every active request's cache to the batch's current max real
    length, runs one batched decode_fn call, and unpads each result back
    down to its own true length before storing it. Shared by
    DecodeWorker.step() and colocated.py's ColocatedWorker.step().
    """
    if not active:
        return [], []
    pad_to = max(r.seq_len for r in active)
    batched_cache, attention_mask = pad_and_batch_caches([r.cache for r in active], pad_to)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones(len(active), 1, dtype=torch.long)], dim=1
    )
    position_ids = attention_mask.sum(dim=1, keepdim=True)
    next_input_ids = torch.cat([r.next_input_id for r in active], dim=0)

    next_token_ids, updated_cache = decode_fn(
        next_input_ids, batched_cache, full_attention_mask, position_ids
    )
    now = clock_fn()

    still_active: list[InFlightRequest] = []
    completed: list[RequestResult] = []
    for i, r in enumerate(active):
        token_id = int(next_token_ids[i].item())
        r.generated_token_ids.append(token_id)
        r.seq_len += 1
        r.cache = slice_cache(updated_cache, i, keep_last=r.seq_len)
        r.next_input_id = torch.tensor([[token_id]], dtype=torch.long)
        done = len(r.generated_token_ids) >= r.max_new_tokens or (
            r.eos_token_id is not None and token_id == r.eos_token_id
        )
        if done:
            completed.append(
                RequestResult(
                    request_id=r.request_id,
                    generated_token_ids=r.generated_token_ids,
                    ttft=r.ttft,
                    completion_time=now,
                )
            )
        else:
            still_active.append(r)
    return still_active, completed


class PrefillWorker:
    """Drains up to batch_size waiting requests per step, runs one
    batched prefill forward pass, and returns each request's own
    (unpadded) KV cache and first generated token -- ready to hand off to
    a DecodeWorker (in-process admit() in CPU tests, handoff.py's
    send_kv_cache on the real rented node).
    """

    def __init__(
        self,
        prefill_fn: PrefillFn,
        batch_size: int,
        *,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._prefill_fn = prefill_fn
        self._batch_size = batch_size
        self._clock_fn = clock_fn
        self._waiting: list[tuple[Request, float]] = []

    def submit(self, request: Request) -> None:
        self._waiting.append((request, self._clock_fn()))

    def step(self) -> list[PrefillResult]:
        batch = self._waiting[: self._batch_size]
        self._waiting = self._waiting[self._batch_size :]
        if not batch:
            return []
        return run_prefill_batch(self._prefill_fn, batch, self._clock_fn)


class DecodeWorker:
    """Admits PrefillResults into a live, continuously-batched decode
    loop: every step, any waiting request gets admitted (up to
    batch_size), then every active request advances one token.
    """

    def __init__(
        self,
        decode_fn: DecodeFn,
        batch_size: int,
        *,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._decode_fn = decode_fn
        self._batch_size = batch_size
        self._clock_fn = clock_fn
        self._waiting: list[InFlightRequest] = []
        self._active: list[InFlightRequest] = []

    def admit(self, result: PrefillResult) -> None:
        self._waiting.append(prefill_result_to_in_flight(result))

    def step(self) -> list[RequestResult]:
        free_slots = self._batch_size - len(self._active)
        if free_slots > 0 and self._waiting:
            self._active.extend(self._waiting[:free_slots])
            self._waiting = self._waiting[free_slots:]
        self._active, completed = step_active_requests(
            self._active, self._decode_fn, self._clock_fn
        )
        return completed
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_disaggregated.py -v`
Expected: all pass.

- [x] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/dispatch/serving/disaggregated.py tests/unit/test_disaggregated.py && uv run ruff format --check src/dispatch/serving/disaggregated.py tests/unit/test_disaggregated.py && uv run mypy src tests`
Expected: clean.

- [x] **Step 6: Commit**

```bash
git add src/dispatch/serving/disaggregated.py tests/unit/test_disaggregated.py
git commit -m "feat: add continuous-batching prefill and decode workers"
```

---

### Task 4: Co-located baseline worker

**Files:**

- Create: `src/dispatch/serving/colocated.py`
- Test: `tests/unit/test_colocated.py`

**Interfaces:**

- Consumes: `Request`, `RequestResult`, `InFlightRequest`, `PrefillFn`,
  `DecodeFn`, `run_prefill_batch`, `prefill_result_to_in_flight`,
  `step_active_requests` (Task 3, unchanged).
- Produces: `ColocatedWorker`. Task 5 wires the same real `PrefillFn`/
  `DecodeFn` closures it built for Task 3's workers into this class too,
  just with both roles talking to the same 4-rank EP pool instead of two
  separate 2-rank pools.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_colocated.py`:

```python
"""CPU-only: proves ColocatedWorker interleaves inline prefill and
decode on the same loop -- new requests get their first token the same
step old ones advance, which is exactly what creates the contention
Phase 4 measures disaggregation against. Reuses test_disaggregated.py's
fake prefill/decode functions unchanged: the forward-pass contract is
identical, only which worker calls it differs.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from dispatch.serving.colocated import ColocatedWorker
from dispatch.serving.disaggregated import Request

NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4


def _fake_cache(batch_size: int, seq_len: int) -> DynamicCache:
    per_layer = [
        (
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
            torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM),
        )
        for _ in range(NUM_LAYERS)
    ]
    return DynamicCache(ddp_cache_data=per_layer)


def _fake_prefill_fn(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask
    batch_size, seq_len = input_ids.shape
    first_tokens = torch.arange(batch_size) + 100
    return first_tokens, _fake_cache(batch_size, seq_len)


def _fake_decode_fn(
    next_input_ids: torch.Tensor,
    cache: DynamicCache,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, DynamicCache]:
    del attention_mask, position_ids
    batch_size = next_input_ids.shape[0]
    new_seq_len = cache.get_seq_length() + 1
    next_tokens = next_input_ids.squeeze(1) + 1
    return next_tokens, _fake_cache(batch_size, new_seq_len)


def test_colocated_worker_admits_via_inline_prefill_and_completes_after_max_new_tokens() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=2))

    first_step = worker.step()  # prefill admits "a" and gives it token 1 of 2
    assert first_step == []

    second_step = worker.step()  # decode gives "a" its 2nd token -> complete
    assert len(second_step) == 1
    assert second_step[0].request_id == "a"
    assert len(second_step[0].generated_token_ids) == 2


def test_colocated_worker_interleaves_new_prefill_with_existing_decode_in_one_step() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("old", torch.tensor([[1, 2]]), max_new_tokens=3))
    worker.step()  # "old" now active, generated 1 of 3 tokens

    worker.submit(Request("new", torch.tensor([[9]]), max_new_tokens=1))
    step_result = worker.step()  # "new" gets prefilled AND decode-stepped this same call

    assert any(r.request_id == "new" for r in step_result)  # "new" already done at max_new_tokens=1


def test_colocated_worker_respects_batch_size_across_prefill_and_decode() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=1)
    worker.submit(Request("a", torch.tensor([[1]]), max_new_tokens=5))
    worker.submit(Request("b", torch.tensor([[2]]), max_new_tokens=1))

    worker.step()  # only "a" has a free slot
    for _ in range(3):
        worker.step()
    a_done = worker.step()
    assert [r.request_id for r in a_done] == ["a"]

    worker.step()  # "b" now gets its prefill turn
    b_done = worker.step()
    assert [r.request_id for r in b_done] == ["b"]
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_colocated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch.serving.colocated'`.

- [x] **Step 3: Write the implementation**

Create `src/dispatch/serving/colocated.py`:

```python
"""The co-located baseline: one worker, one EP pool, doing both prefill
and decode in a single loop on the same GPUs -- the contention baseline
Phase 4's disaggregated path (disaggregated.py) is measured against, on
the same 4 GPUs, so GPU count is never a confound (design doc section 1).
Reuses disaggregated.py's types and helpers unchanged: a decode step is a
decode step regardless of whether prefill happened on the same GPU this
iteration or a different one.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from dispatch.serving.disaggregated import (
    DecodeFn,
    InFlightRequest,
    PrefillFn,
    Request,
    RequestResult,
    prefill_result_to_in_flight,
    run_prefill_batch,
    step_active_requests,
)


class ColocatedWorker:
    """Every step: admit waiting requests via an inline prefill batch (up
    to free slots), then advance every active request -- new and old --
    by one decode token, all on the same GPU pool.
    """

    def __init__(
        self,
        prefill_fn: PrefillFn,
        decode_fn: DecodeFn,
        batch_size: int,
        *,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._prefill_fn = prefill_fn
        self._decode_fn = decode_fn
        self._batch_size = batch_size
        self._clock_fn = clock_fn
        self._waiting: list[tuple[Request, float]] = []
        self._active: list[InFlightRequest] = []

    def submit(self, request: Request) -> None:
        self._waiting.append((request, self._clock_fn()))

    def step(self) -> list[RequestResult]:
        free_slots = self._batch_size - len(self._active)
        if free_slots > 0 and self._waiting:
            batch = self._waiting[:free_slots]
            self._waiting = self._waiting[free_slots:]
            prefill_results = run_prefill_batch(self._prefill_fn, batch, self._clock_fn)
            self._active.extend(prefill_result_to_in_flight(r) for r in prefill_results)
        self._active, completed = step_active_requests(
            self._active, self._decode_fn, self._clock_fn
        )
        return completed
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_colocated.py -v`
Expected: all pass.

- [x] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/dispatch/serving/colocated.py tests/unit/test_colocated.py && uv run ruff format --check src/dispatch/serving/colocated.py tests/unit/test_colocated.py && uv run mypy src tests`
Expected: clean.

- [x] **Step 6: Run the full CPU-testable suite together**

Run: `uv run pytest tests/unit -v -m "not gpu"`
Expected: all pass, including Task 1-4's new tests. This is the
"everything not requiring GPU/DeepEP is already green" checkpoint the
Global Constraints require before Task 5's rental begins.

- [x] **Step 7: Commit**

```bash
git add src/dispatch/serving/colocated.py tests/unit/test_colocated.py
git commit -m "feat: add co-located prefill/decode baseline worker"
```

---

### Task 5: GPU rental runbook -- real EP pools, correctness gate, concurrency measurement

**Files:**

- Create: `docs/runbooks/phase-4-disaggregated-prefill-decode.md`
- Create (via `write_cost_record`, reused unmodified):
  `docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-cost.md`
- Creates live, on the pod (not pre-committed -- its exact shape depends
  on live-confirmed process-group/Buffer behavior, same practice Phase
  3's Task 4 used for `make_ep_moe_infer`): a load-driver script wiring
  real `PrefillFn`/`DecodeFn` closures around DeepSeekMoE-16B, patched
  twice with Phase 3's unchanged `make_ep_moe_infer`/`patch_moe_infer_ep`
  (once per EP sub-group).

**Budget cap: $40, a ceiling not a target -- be surgical, minimize actual
spend (Global Constraints). This is a live session with the user, not
something to run unattended -- get explicit go-ahead before renting.**

- [x] **Step 1: Write the runbook**

Create `docs/runbooks/phase-4-disaggregated-prefill-decode.md`:

````markdown
# Phase 4 runbook: disaggregated prefill/decode

Budget cap: $40, a ceiling not a target -- be surgical. Hopper-class
(H100/H200, SM90) required, same as Phase 3. Check RunPod's real catalog
at rental time for 4-GPU availability and pricing.

1. Create and wait for the 4-GPU pod, reusing Phase 3's generic
   `--gpu-count` support (Task 1 of Phase 3, no new provisioning code
   needed):

   ```bash
   uv run python scripts/gpu/provision.py create --name phase-4-disaggregated \
     --gpu-type <Hopper-class type available at rental time> --gpu-count 4 \
     --image <RunPod CUDA/PyTorch template, CUDA>=12.3> --cloud <community|secure> --disk-gb 40
   uv run python scripts/gpu/provision.py wait <pod-id>
   ```

2. **Verify real NVLink across all four GPUs** -- Phase 3's check
   extended to a 4-GPU pair matrix:

   ```bash
   nvidia-smi topo -m
   ```

   Expected: `NV#` between every pair of the four GPUs, not just 0-1. If
   any pair shows `PHB`/`PXB`/`SYS`, stop, terminate the pod, and
   re-provision -- do not spend budget against a partially-NVLink node.

3. Install DeepEP V1 (legacy `Buffer`) directly -- do not attempt V2,
   Phase 3 already confirmed it cannot work on this rental tier (no GPU
   Fabric Manager). Reuse Phase 3's exact install steps
   (`docs/runbooks/phase-3-multi-gpu-ep.md`, steps 3-4), adjusted to
   import `deep_ep.Buffer` rather than `ElasticBuffer`.

4. Clone this repo, sync, and run the full CPU-testable suite one more
   time on the pod itself, as a final sanity check before spending
   anything on the GPU-dependent work:

   ```bash
   git clone <this repo's URL> dispatch
   cd dispatch
   uv sync
   uv run pytest tests/unit -v -m "not gpu"
   ```

   Expected: all pass (same suite Task 4 step 6 already proved locally
   -- this just confirms the pod's environment doesn't disagree).

5. **Build the two EP sub-groups and wire the real forward closures.**
   `torchrun --nproc_per_node=4` gives a 4-rank NCCL world; split it with
   `dist.new_group([0, 1])` (prefill) and `dist.new_group([2, 3])`
   (decode). Each sub-group builds its own DeepEP `Buffer` scoped to
   that sub-group (same `Buffer(group, num_nvl_bytes, 0)` construction
   Phase 3 used, just called twice) and calls
   `patch_moe_infer_ep(model, matmul, buffer, rank, n_ranks=2)`
   unchanged -- `rank` here is the sub-group-local rank
   (`dist.get_rank(group=sub_group)`, which is 0 or 1), not the global
   rank. Live-confirm this distinction against DeepEP's actual behavior
   before trusting it -- if the printed/observed local ranks contradict
   this, fix the code to match what's actually observed, the same
   practice Phase 3's plan used for DeepEP's dispatch/combine shapes.

   Wrap the patched model's forward pass into `PrefillFn`/`DecodeFn`
   (Task 3's exact type aliases), doing the greedy-argmax step inside the
   closure -- matching `harness.py`'s existing
   `outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)` convention:

   ```python
   def make_prefill_fn(model, device):
       def prefill_fn(input_ids, attention_mask):
           outputs = model(
               input_ids=input_ids.to(device),
               attention_mask=attention_mask.to(device),
               use_cache=True,
           )
           first_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
           return first_tokens.cpu(), outputs.past_key_values

       return prefill_fn


   def make_decode_fn(model, device):
       def decode_fn(next_input_ids, cache, attention_mask, position_ids):
           outputs = model(
               input_ids=next_input_ids.to(device),
               past_key_values=cache,
               attention_mask=attention_mask.to(device),
               position_ids=position_ids.to(device),
               use_cache=True,
           )
           next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
           return next_tokens.cpu(), outputs.past_key_values

       return decode_fn
   ```

   For the disaggregated path, the prefill pool's rank-0-to-decode-rank-2
   and rank-1-to-decode-rank-3 handoff uses Task 2's `send_kv_cache`/
   `recv_kv_cache` unchanged, over the full 4-rank world group (not a
   sub-group, since prefill and decode ranks must talk to each other).

6. **Correctness gate -- must pass for both configurations before any
   measurement runs.** Reuse `dispatch.benchmark.reference`'s
   `capture_reference_logits`/`compare_top_k_agreement`/`save_reference`/
   `load_reference` unchanged (Phase 0/1's own functions). Get a
   single-GPU reference first (rank 0 only, `patch_moe_infer` with the
   naive kernel, `DEFAULT_PROMPTS` matching Phase 0/1/3's own set).
   Then, separately:
   - Co-located: one 4-rank EP pool (`patch_moe_infer_ep(..., n_ranks=4)`),
     `ColocatedWorker` driving the same prompts one at a time
     (concurrency 1, to isolate correctness from the concurrency
     measurement), compare logits to the reference.
   - Disaggregated: the two 2-rank sub-groups from step 5, `PrefillWorker`
     + `DecodeWorker` connected by the real handoff, same prompts at
     concurrency 1, compare logits to the reference.

   Both must show mutual top-5 and top-1 agreement, the same bar Phase 1
   and Phase 3 used. If either fails, stop -- do not measure contention
   on top of a wrong result. Record the exact agreement measured (perfect,
   or the first position/token where it diverges) in the findings doc
   either way.

7. **Only if step 6 passes for both configurations**, run the
   concurrency measurement: co-located vs. disaggregated, same 4 GPUs,
   at 4 and 8 concurrent requests (staggered arrivals, not a single
   simultaneous burst), `DEFAULT_PROMPTS` cycled to fill the concurrency
   level. Record per-request TTFT and inter-token latency, aggregate
   mean/p50/p99 TTFT and mean decode tokens/sec per configuration per
   concurrency level. Every number carries its full config (GPU model,
   concurrency level, batch sizes used, prompt set), per CLAUDE.md.

   **Be surgical.** Watch running cost throughout; if pace projects past
   the $40 cap before both concurrency levels finish, stop after the
   last fully-completed level and report exactly what was measured.
   The goal is the smallest paid session that produces a trustworthy
   result, not spending up to the cap.

8. Copy all console output and any saved JSON back to this repo's
   `docs/findings/` before doing anything else.

9. Tear down immediately:

   ```bash
   uv run python scripts/gpu/provision.py terminate <pod-id>
   ```

10. From this repo's root, record the measured cost:

    ```python
    from pathlib import Path
    from scripts.gpu.provision import write_cost_record

    write_cost_record(
        Path("docs/findings"),
        pod_id="<pod-id>",
        gpu_type_id="<GPU type actually used>",
        cost_per_hour=<combined rate for all 4 GPUs>,
        duration_s=<measured seconds>,
        note=(
            "Phase 4: DeepEP V1 install, 4-GPU NVLink verification, two EP "
            "sub-group setup, correctness gate (co-located + disaggregated), "
            "concurrency measurement at 4 and 8 concurrent requests"
        ),
        run_label="phase-4-disaggregated-prefill-decode",
    )
    ```
````

- [x] **Step 2: Get the user's explicit go-ahead, then execute the runbook**

Confirm the budget cap and GPU choice with the user before the first
`create` call -- this is a paid action, never taken unattended. Remind
the user of the standing instruction to be surgical: prefer stopping
early with a smaller, trustworthy result over spending toward the cap.

- [x] **Step 3: Commit the runbook, the pod-live load-driver script, and the cost record**

```bash
git add docs/runbooks/phase-4-disaggregated-prefill-decode.md docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-cost.md scripts/gpu/phase4_load_driver.py
git commit -m "docs: record Phase 4 GPU runbook and cost; add EP sub-group wiring"
```

If step 5 or 6 needed adjustment to match live-observed DeepEP/process-group
behavior, note the adjustment and why in the commit body -- not silently,
per this project's practice of recording what actually happened rather
than what was assumed.

---

### Task 6: Findings doc and STATUS.md

**Files:**

- Create: `docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-run.md`
- Modify: `docs/STATUS.md`

- [x] **Step 1: Write the findings doc**

Cover, in `docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-run.md`:
whether real 4-GPU NVLink was confirmed; whether DeepEP V1 installed
cleanly reusing Phase 3's steps or needed adjustment; the correctness-gate
result for both co-located and disaggregated configurations (pass, with
the measured agreement, or the honest reason it didn't); **the direct
answer to Phase 4's thesis** -- does disaggregation measurably relieve
prefill/decode contention at 4 and/or 8 concurrent requests, holding GPU
count fixed at 4 -- stated at exactly the strength the data supports; the
full measured TTFT/tokens-per-second table for both configurations at
both concurrency levels, if the gate passed; the known dense-padding
simplification's effect explicitly disclosed (design doc section 6); and
the actual GPU type, duration, and cost from Task 5's cost record,
**explicitly compared against the $40 cap** -- the gap between them is
itself part of this phase's reported result, per the standing instruction
to be surgical. If the correctness gate failed or the budget ran out
before both concurrency levels completed, say so plainly and report
exactly what was measured -- matching this project's practice of writing
down a null or partial result rather than a flattering guess.

- [x] **Step 2: Update STATUS.md**

Add a "## Phase 4 progress" section following the Phase 0/1/2/3 pattern:
plan link, hardware actually used, the correctness-gate result for both
configurations, the contention-relief thesis answer, and total GPU cost
against the $40 cap. Set "## Next step" to reflect what's actually next
(Phase 5 planning, or follow-up on Phase 4 if something didn't land
cleanly).

- [x] **Step 3: Commit**

```bash
git add docs/findings/2026-09-16-phase-4-disaggregated-prefill-decode-run.md docs/STATUS.md
git commit -m "docs: record Phase 4 disaggregated prefill/decode outcome"
```

---

## Self-Review Notes

- **Spec coverage:** design doc §3 (architecture: 4-rank world, two EP
  sub-groups) and §4 (components: PrefillWorker, DecodeWorker,
  KV-cache handoff, load driver, co-located worker) map to Tasks 1-5
  directly -- one task per component, in dependency order (cache
  bookkeeping and handoff proven before the scheduler that uses them,
  the scheduler proven before the baseline that reuses its internals).
  §5 (data flow) maps to Task 5's runbook steps 5-6. §6 (known dense-
  padding simplification) is implemented in Task 1's `pad_and_batch_caches`
  and explicitly disclosed again in Task 6's findings doc, not left
  implicit. §7 (testing table) maps one-to-one onto Tasks 1-4 (CPU rows)
  and Task 5 (the two `gpu`, paid rows). §8 (cost discipline, "be
  surgical") is a Global Constraint and repeated in Task 5's runbook and
  Task 6's findings doc, not stated once and forgotten. §1's thesis
  (does disaggregation relieve contention, GPU count held constant) is
  answered directly in Task 6, not left implicit.
- **New scope beyond the design doc, and why it's necessary rather than
  creep:** the design doc doesn't name `transformers` v5's `DynamicCache`
  API surface at all -- it was resolved during this plan's own writing
  (live-checked 2026-09-16 against `transformers`' migration guide and
  source, see Global Constraints) because the design doc's §4/§5 assume
  KV-cache slicing is possible without specifying how, and guessing the
  wrong API (e.g. the removed `to_legacy_cache`) would have surfaced as
  an `AttributeError` only once real GPU time was already being spent.
  Resolving it here, for $0, before Task 1 is dispatched, is the same
  "confirm live, don't guess" discipline Phase 3's plan applied to
  DeepEP's exact call shapes.
- **Ambiguity check:** "4 and 8 concurrent requests" (Task 5 step 7) is
  pinned, unlike Phase 3's plan which deliberately left its benchmark
  grid to be set live -- here the concurrency levels don't depend on
  anything that can only be observed on real hardware (unlike Phase 3's
  per-expert token counts, which came from DeepEP's real dispatch
  behavior), so pinning them now is reasonable rather than premature.
  `max_batch_size` for each worker is deliberately left for Task 5 to set
  against the rented node's real memory headroom, the same kind of
  live-set choice Phase 2 made for GPU type and Phase 0 made for model
  precision.
- **Type consistency:** `PrefillFn`/`DecodeFn` (Task 3) are used
  identically by `PrefillWorker`/`DecodeWorker` (Task 3),
  `ColocatedWorker` (Task 4), and Task 5's real closures --
  `(input_ids, attention_mask) -> (first_token_ids, cache)` and
  `(next_input_ids, cache, attention_mask, position_ids) -> (next_token_ids,
  cache)` respectively, never renamed or reshaped between tasks.
  `slice_cache`'s `keep_last` parameter (Task 1) is used identically by
  `run_prefill_batch` and `step_active_requests` (Task 3) to drop
  left-padding after every batched forward call, not just some of them.

## Deviation found during Task 3's execution

Tracing `DecodeWorker.step()`'s logic by hand before writing its
implementation surfaced a real bug this plan's original sketch didn't
have a check for: admission (waiting -> active) and decoding happen in
the *same* `step()` call, so a request whose `max_new_tokens` is 1 is
already complete from prefill's own first token alone -- calling
`step_active_requests` on it anyway would still take one decode step and
over-generate by one token. Fixed by adding `split_completed(active,
now) -> (still_active, completed)` to `disaggregated.py`, called in
`DecodeWorker.step()` (and, per the shared-helpers design, `colocated.py`'s
`ColocatedWorker.step()` in Task 4) immediately after admission and
before `step_active_requests`. Covered by a new regression test,
`test_decode_worker_completes_immediately_when_admission_alone_hits_max_new_tokens`.
This also forced `max_new_tokens` up from 1 to 2 or 3 in several of the
plan's other sketched test cases, where the degenerate value would have
made them stop exercising what they were meant to test (e.g.
`test_decode_worker_batches_requests_with_different_cache_lengths` would
never have reached `step_active_requests`'s padding logic at all with
`max_new_tokens=1` on both requests).
