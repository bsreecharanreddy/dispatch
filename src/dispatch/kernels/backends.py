"""Backend name -> GroupedMatmul. The Triton backends import lazily, so this
module -- and everything that only needs the eager backend -- stays
importable where triton isn't installed."""

from __future__ import annotations

from dispatch.kernels.moe_forward import GroupedMatmul, torch_grouped_matmul
from dispatch.kernels.quantization import QuantizedGroupedMatmul

BACKENDS = ("torch", "naive", "persistent")
QUANTIZED_BACKEND = "quantized"

# Which of BACKENDS has an int8 kernel: "torch" (the dequantized eager
# reference) and "naive" (Phase 5a's real int8 kernel, built on the naive
# launch order only). Named here, the one place backend capability is
# decided, instead of as an inline if/elif in each caller.
QUANTIZED_BACKENDS = ("torch", "naive")


def resolve_backend(name: str) -> GroupedMatmul:
    if name == "torch":
        return torch_grouped_matmul
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    from dispatch.kernels import grouped_gemm  # noqa: PLC0415 -- triton is Linux-only

    return (
        grouped_gemm.grouped_matmul if name == "naive" else grouped_gemm.grouped_matmul_persistent
    )


def resolve_quantized_backend() -> QuantizedGroupedMatmul:
    from dispatch.kernels import grouped_gemm_int8  # noqa: PLC0415 -- triton is Linux-only

    return grouped_gemm_int8.grouped_matmul_int8
