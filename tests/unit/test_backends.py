"""CPU-only: the Triton names resolve against a stand-in module, so the
mapping itself is tested everywhere, not only on a GPU host."""

from __future__ import annotations

import sys
import types

import pytest

import dispatch.kernels
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.moe_forward import torch_grouped_matmul


def test_torch_backend_is_the_eager_contract() -> None:
    assert resolve_backend("torch") is torch_grouped_matmul


def test_triton_backend_names_map_to_their_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in = types.ModuleType("dispatch.kernels.grouped_gemm")
    stand_in.grouped_matmul = object()  # type: ignore[attr-defined]
    stand_in.grouped_matmul_persistent = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dispatch.kernels.grouped_gemm", stand_in)
    monkeypatch.setattr(dispatch.kernels, "grouped_gemm", stand_in, raising=False)

    assert resolve_backend("naive") is stand_in.grouped_matmul
    assert resolve_backend("persistent") is stand_in.grouped_matmul_persistent


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("cutlass")
