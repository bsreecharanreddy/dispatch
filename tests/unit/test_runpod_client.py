"""Tests for the RunPod REST client -- no real network calls, ever."""

from __future__ import annotations

from typing import Any

import pytest
from scripts.gpu.runpod_client import (
    PodHandle,
    RunPodAPIError,
    create_pod,
    get_pod,
    terminate_pod,
)


class FakeResponse:
    def __init__(
        self, status_code: int, payload: dict[str, Any] | None = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)
        self.content = b"x" if payload is not None else b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Any = None,
        timeout: Any = None,
    ) -> FakeResponse:
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return self.response


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")


def test_create_pod_sends_expected_body_and_parses_response() -> None:
    payload = {"id": "pod_123", "status": "PROVISIONING", "cost": 0.4}
    session = FakeSession(FakeResponse(200, payload))

    handle = create_pod(
        "dispatch-baseline",
        "NVIDIA A40",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        session=session,  # type: ignore[arg-type]
    )

    assert handle == PodHandle(id="pod_123", status="PROVISIONING", cost_per_hour=0.4)
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/pods")
    assert call["json"]["gpu"] == {"id": "NVIDIA A40", "count": 1}
    assert call["json"]["cloud"] == "COMMUNITY"
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_get_pod_parses_response() -> None:
    session = FakeSession(FakeResponse(200, {"id": "pod_123", "status": "RUNNING", "cost": 0.4}))

    handle = get_pod("pod_123", session=session)  # type: ignore[arg-type]

    assert handle.status == "RUNNING"


def test_terminate_pod_accepts_204_with_no_body() -> None:
    session = FakeSession(FakeResponse(204))

    terminate_pod("pod_123", session=session)  # type: ignore[arg-type]

    assert session.calls[0]["method"] == "DELETE"
    assert session.calls[0]["url"].endswith("/pods/pod_123")


def test_create_pod_raises_on_error_response() -> None:
    session = FakeSession(FakeResponse(422, text="bad gpu id"))

    with pytest.raises(RunPodAPIError, match="422"):
        create_pod("dispatch-baseline", "not-a-real-gpu", "some/image", session=session)  # type: ignore[arg-type]


def test_create_pod_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    with pytest.raises(RunPodAPIError, match="RUNPOD_API_KEY"):
        create_pod("x", "y", "z")
