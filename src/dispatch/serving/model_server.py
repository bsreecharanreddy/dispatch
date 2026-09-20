"""gRPC model server for Phase 7's router demo: streams token-by-token
generation over the shared proto/dispatch.proto contract. Two Responders
share one Servicer so the wire contract is identical whether responses
come from StubResponder (canned, no model, used by tests/dev/docker-
compose) or the real KernelResponder (scripts/gpu/phase7_kernel_responder.py,
GPU-only, used only for the real demo session) -- this module itself
never imports torch, so it stays importable and testable anywhere.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass
from typing import Protocol

import grpc

from dispatch.proto import dispatch_pb2, dispatch_pb2_grpc


class ServerBusyError(RuntimeError):
    """Raised by a Responder that can only stream one request at a time
    when a second request arrives while the first is still in flight."""


@dataclass(frozen=True)
class TokenEvent:
    text: str
    is_final: bool
    t_emit: float


class Responder(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]: ...


class StubResponder:
    """Canned, deterministic tokens with a fixed per-token delay. No
    torch, no model weights -- used by unit tests, the cross-language
    router integration test (Task 8), and docker-compose local
    verification, all without a GPU.
    """

    def __init__(
        self,
        *,
        tokens: tuple[str, ...] = ("The", " quick", " brown", " fox", " jumps"),
        delay_seconds: float = 0.01,
    ) -> None:
        self._tokens = tokens
        self._delay_seconds = delay_seconds

    def generate(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        del prompt  # canned output does not depend on the prompt
        for text in self._tokens[:max_new_tokens]:
            time.sleep(self._delay_seconds)
            yield TokenEvent(text=text, is_final=False, t_emit=time.perf_counter())
        yield TokenEvent(text="", is_final=True, t_emit=time.perf_counter())


class ModelServerServicer(dispatch_pb2_grpc.ModelServerServicer):  # type: ignore[misc]
    # Generated base class is Any-typed (follow_imports="skip", pyproject.toml)
    # -- strict mode's disallow_subclassing_any is deliberately overridden
    # here, the one place this module actually needs the generated servicer.
    def __init__(self, responder: Responder) -> None:
        self._responder = responder

    def Generate(  # noqa: N802 -- overrides a grpc_tools-generated method name
        self, request: dispatch_pb2.GenerateRequest, context: grpc.ServicerContext
    ) -> Iterator[dispatch_pb2.GenerateResponse]:
        try:
            for event in self._responder.generate(request.prompt, request.max_new_tokens):
                yield dispatch_pb2.GenerateResponse(
                    request_id=request.request_id,
                    text=event.text,
                    is_final=event.is_final,
                    t_emit_unix=event.t_emit,
                )
        except ServerBusyError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))


def serve(responder: Responder, *, port: int = 0) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    dispatch_pb2_grpc.add_ModelServerServicer_to_server(ModelServerServicer(responder), server)
    bound_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, bound_port
