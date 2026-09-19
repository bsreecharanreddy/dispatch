"""Engine name -> contestants. vLLM and SGLang import lazily, so this module
is importable (and the driver's tests run) without either installed."""

from __future__ import annotations

from collections.abc import Sequence

from dispatch.benchmark.engines.base import MoEEngine
from dispatch.benchmark.engines.dispatch_kernels import DispatchEngine

ENGINE_NAMES = ("dispatch-naive", "dispatch-persistent", "vllm", "sglang")
DEFAULT_BLOCK_MS = (16, 32, 64, 128)


def build_engines(engine: str, block_ms: Sequence[int] = DEFAULT_BLOCK_MS) -> list[MoEEngine]:
    """dispatch-* returns one contestant per tile size in `block_ms` (its own
    tuning sweep); vllm and sglang return exactly one, tuned by their own
    tuners through their own config folders, not by this call."""
    if engine.startswith("dispatch-"):
        backend = engine.removeprefix("dispatch-")
        if f"dispatch-{backend}" not in ENGINE_NAMES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
        return [DispatchEngine(backend, block_m=block_m) for block_m in block_ms]
    if engine == "vllm":
        from dispatch.benchmark.engines.vllm_moe import VllmEngine  # noqa: PLC0415

        return [VllmEngine()]
    if engine == "sglang":
        from dispatch.benchmark.engines.sglang_moe import SglangEngine  # noqa: PLC0415

        return [SglangEngine()]
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
