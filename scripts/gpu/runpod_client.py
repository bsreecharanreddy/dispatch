"""Thin REST client for RunPod's pod-management API (v2).

No retry/backoff logic here on purpose -- that belongs in the orchestration
layer (scripts/gpu/provision.py), which knows what "retry" should mean for
each call (a pod create is not safe to blindly retry; a status poll is).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests

RUNPOD_API_BASE = "https://api.runpod.io/v2"


class RunPodAPIError(RuntimeError):
    """A RunPod API call returned an error, or the required API key is missing."""


@dataclass(frozen=True)
class PodHandle:
    id: str
    status: str
    cost_per_hour: float


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodAPIError("RUNPOD_API_KEY is not set")
    return key


def _request(
    session: requests.Session, method: str, path: str, *, json: dict[str, Any] | None = None
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    resp = session.request(
        method, f"{RUNPOD_API_BASE}{path}", json=json, headers=headers, timeout=30
    )
    if not resp.ok:
        raise RunPodAPIError(f"{method} {path} failed: {resp.status_code} {resp.text}")
    if resp.status_code == HTTPStatus.NO_CONTENT or not resp.content:
        return {}
    result: dict[str, Any] = resp.json()
    return result


def _pod_from_response(data: dict[str, Any]) -> PodHandle:
    return PodHandle(
        id=str(data["id"]), status=str(data["status"]), cost_per_hour=float(data.get("cost", 0.0))
    )


def create_pod(  # noqa: PLR0913 -- a pod-create request has this many independent knobs
    name: str,
    gpu_type_id: str,
    image: str,
    *,
    disk_gb: int = 60,
    gpu_count: int = 1,
    cloud: str = "COMMUNITY",
    ports: tuple[str, ...] = ("22/tcp",),
    start_ssh: bool = True,
    session: requests.Session | None = None,
) -> PodHandle:
    body = {
        "name": name,
        "image": image,
        "gpu": {"id": gpu_type_id, "count": gpu_count},
        "disk": disk_gb,
        "ports": list(ports),
        "cloud": cloud,
        "startSsh": start_ssh,
    }
    data = _request(session or requests.Session(), "POST", "/pods", json=body)
    return _pod_from_response(data)


def get_pod(pod_id: str, *, session: requests.Session | None = None) -> PodHandle:
    data = _request(session or requests.Session(), "GET", f"/pods/{pod_id}")
    return _pod_from_response(data)


def terminate_pod(pod_id: str, *, session: requests.Session | None = None) -> None:
    _request(session or requests.Session(), "DELETE", f"/pods/{pod_id}")
