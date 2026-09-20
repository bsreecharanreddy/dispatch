"""No GPU, no torch -- StubResponder and the servicer wiring only."""

from __future__ import annotations

import grpc
import pytest

from dispatch.proto import dispatch_pb2, dispatch_pb2_grpc
from dispatch.serving.model_server import ServerBusyError, StubResponder, serve


def test_stub_responder_yields_requested_token_count_then_final() -> None:
    responder = StubResponder(tokens=("a", "b", "c", "d"), delay_seconds=0.0)

    events = list(responder.generate("ignored", max_new_tokens=2))

    assert [e.text for e in events] == ["a", "b", ""]
    assert [e.is_final for e in events] == [False, False, True]


def test_serve_streams_stub_responses_over_real_grpc() -> None:
    server, port = serve(StubResponder(tokens=("hi",), delay_seconds=0.0), port=0)
    try:
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = dispatch_pb2_grpc.ModelServerStub(channel)
        request = dispatch_pb2.GenerateRequest(request_id="r1", prompt="hello", max_new_tokens=1)

        responses = list(stub.Generate(request))

        assert [r.text for r in responses] == ["hi", ""]
        assert [r.is_final for r in responses] == [False, True]
        assert all(r.request_id == "r1" for r in responses)
    finally:
        server.stop(grace=None)


def test_generate_aborts_with_resource_exhausted_on_server_busy_error() -> None:
    class _BusyResponder:
        def generate(self, prompt: str, max_new_tokens: int):  # type: ignore[no-untyped-def]
            raise ServerBusyError("only one stream at a time")
            yield  # type: ignore[unreachable]  # pragma: no cover -- makes this a generator function

    server, port = serve(_BusyResponder(), port=0)
    try:
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = dispatch_pb2_grpc.ModelServerStub(channel)
        request = dispatch_pb2.GenerateRequest(request_id="r1", prompt="x", max_new_tokens=1)

        with pytest.raises(grpc.RpcError) as exc_info:
            list(stub.Generate(request))

        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        server.stop(grace=None)
