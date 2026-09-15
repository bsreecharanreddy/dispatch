"""Backend name -> GroupedMatmul. The Triton backends import lazily, so this
module -- and everything that only needs the eager backend -- stays
importable where triton isn't installed."""

from __future__ import annotations

from dispatch.kernels.moe_forward import GroupedMatmul, torch_grouped_matmul

BACKENDS = ("torch", "naive", "persistent")


def resolve_backend(name: str) -> GroupedMatmul:
    if name == "torch":
        return torch_grouped_matmul
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    from dispatch.kernels import grouped_gemm  # noqa: PLC0415 -- triton is Linux-only

    return (
        grouped_gemm.grouped_matmul if name == "naive" else grouped_gemm.grouped_matmul_persistent
    )
