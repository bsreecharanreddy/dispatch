"""CPU-only: proves --responder stub needs nothing but this repo's normal
dependencies. --responder kernel is exercised only on a real GPU
(Task 9's gpu-marked test), never here.
"""

from __future__ import annotations

from scripts.run_model_server import build_responder, parse_args


def test_default_responder_is_stub() -> None:
    args = parse_args([])

    assert args.responder == "stub"
    assert args.port == 50051


def test_build_responder_stub_needs_no_gpu_imports() -> None:
    args = parse_args(["--responder", "stub"])

    responder = build_responder(args)

    events = list(responder.generate("hi", max_new_tokens=1))
    assert events[-1].is_final is True
