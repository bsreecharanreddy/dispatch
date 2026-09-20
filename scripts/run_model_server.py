"""CLI entrypoint for Phase 7's model server. `--responder stub` (the
default) needs nothing but this repo's normal dependencies and is what
tests, the cross-language router integration test, and docker-compose
local verification all run. `--responder kernel` additionally imports
scripts/gpu/phase7_kernel_responder.py (torch, the real model, Phase 6's
naive kernel) -- lazily, only when selected, so this module stays
importable on a machine with neither GPU nor torch installed.
"""

from __future__ import annotations

import argparse

from dispatch.serving.model_server import Responder, StubResponder, serve


def build_responder(args: argparse.Namespace) -> Responder:
    if args.responder == "stub":
        return StubResponder()
    # scripts/gpu/phase7_kernel_responder.py lands in Task 9 -- these two
    # ignores are real forward references to a module that doesn't exist
    # yet, not a permanent Any-typed boundary; Task 9 gives it a real
    # KernelResponder return type and removes both (mypy's
    # warn_unused_ignores will flag them as unused once it does).
    from scripts.gpu.phase7_kernel_responder import (  # type: ignore[import-not-found]  # noqa: PLC0415
        build_kernel_responder,
    )

    return build_kernel_responder(moe_kernel=args.moe_kernel)  # type: ignore[no-any-return]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responder", choices=["stub", "kernel"], default="stub")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--moe-kernel",
        default="naive",
        help="Only used with --responder kernel; passed to resolve_backend.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    responder = build_responder(args)
    server, bound_port = serve(responder, port=args.port)
    print(f"model server listening on port {bound_port}, responder={args.responder}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
