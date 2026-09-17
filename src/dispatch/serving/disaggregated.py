"""Continuous-batching prefill and decode workers for Phase 4's
disaggregated serving path. Both workers are forward-pass-agnostic --
they take a plain PrefillFn/DecodeFn callable and know nothing about
DeepEP, EP pools, or even that this is a distributed setting. Task 5
wires the real DeepSeekMoE-16B model (patched with Phase 3's
make_ep_moe_infer/patch_moe_infer_ep, one EP pool per role) into exactly
these two callable shapes; nothing in this file changes when that
happens. run_prefill_batch, prefill_result_to_in_flight, split_completed,
and step_active_requests are exported (not underscore-prefixed) because
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


def split_completed(
    active: list[InFlightRequest], now: float
) -> tuple[list[InFlightRequest], list[RequestResult]]:
    """Separates requests that are already at or past max_new_tokens from
    those that still need at least one more decode step. Needed because
    admission and decoding share one step() call (continuous batching):
    a request whose max_new_tokens is 1 is already complete the moment
    it's admitted, from prefill's own first token alone -- without this
    check, step_active_requests would still run one decode_fn call on it
    and over-generate by one token.
    """
    still_active: list[InFlightRequest] = []
    completed: list[RequestResult] = []
    for r in active:
        if len(r.generated_token_ids) >= r.max_new_tokens:
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


def step_active_requests(
    active: list[InFlightRequest], decode_fn: DecodeFn, clock_fn: Callable[[], float]
) -> tuple[list[InFlightRequest], list[RequestResult]]:
    """Pads every active request's cache to the batch's current max real
    length, runs one batched decode_fn call, and unpads each result back
    down to its own true length before storing it. Shared by
    DecodeWorker.step() and colocated.py's ColocatedWorker.step(). Callers
    are expected to have already run split_completed on active -- this
    function always takes one decode step for everyone it's given.
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
        self._active, already_done = split_completed(self._active, self._clock_fn())
        self._active, completed = step_active_requests(
            self._active, self._decode_fn, self._clock_fn
        )
        return already_done + completed
