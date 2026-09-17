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
    split_completed,
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
        self._active, already_done = split_completed(self._active, self._clock_fn())
        self._active, completed = step_active_requests(
            self._active, self._decode_fn, self._clock_fn
        )
        return already_done + completed
