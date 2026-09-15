"""Tests for pod-wait orchestration and cost-record writing.

wait_until_running takes get_pod_fn/sleep_fn/clock_fn so these tests never
sleep for real and never hit the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.gpu.provision import build_parser, main, wait_until_running, write_cost_record
from scripts.gpu.runpod_client import PodHandle, RunPodAPIError


def test_wait_until_running_returns_once_status_is_running() -> None:
    statuses = iter(["PROVISIONING", "STARTING", "RUNNING"])
    clock = iter([0.0, 1.0, 2.0])
    sleeps: list[float] = []

    def fake_get_pod(pod_id: str) -> PodHandle:
        return PodHandle(id=pod_id, status=next(statuses), cost_per_hour=0.4)

    pod = wait_until_running(
        "pod_1",
        get_pod_fn=fake_get_pod,
        timeout_s=100.0,
        poll_interval_s=5.0,
        sleep_fn=sleeps.append,
        clock_fn=lambda: next(clock),
    )

    assert pod.status == "RUNNING"
    assert sleeps == [5.0, 5.0]


def test_wait_until_running_raises_on_terminal_status() -> None:
    def fake_get_pod(pod_id: str) -> PodHandle:
        return PodHandle(id=pod_id, status="ERROR", cost_per_hour=0.0)

    with pytest.raises(RunPodAPIError, match="ERROR"):
        wait_until_running(
            "pod_1", get_pod_fn=fake_get_pod, sleep_fn=lambda _: None, clock_fn=lambda: 0.0
        )


def test_wait_until_running_raises_on_timeout() -> None:
    clock = iter([0.0, 50.0, 120.0])

    def fake_get_pod(pod_id: str) -> PodHandle:
        return PodHandle(id=pod_id, status="STARTING", cost_per_hour=0.0)

    with pytest.raises(TimeoutError, match="did not reach RUNNING"):
        wait_until_running(
            "pod_1",
            get_pod_fn=fake_get_pod,
            timeout_s=100.0,
            sleep_fn=lambda _: None,
            clock_fn=lambda: next(clock),
        )


def test_write_cost_record_computes_cost_and_writes_file(tmp_path: Path) -> None:
    path = write_cost_record(
        tmp_path,
        pod_id="pod_123",
        gpu_type_id="NVIDIA A40",
        cost_per_hour=0.40,
        duration_s=3600.0,
        note="phase 0 baseline run",
        now=datetime(2026, 9, 14, tzinfo=UTC),
    )

    assert path.name == "2026-09-14-phase-0-baseline-cost.md"
    content = path.read_text()
    assert "pod_123" in content
    assert "NVIDIA A40" in content
    assert "$0.4000/hr" in content
    assert "$0.4000" in content  # cost == rate at exactly 1hr duration
    assert "phase 0 baseline run" in content


def test_write_cost_record_names_the_file_for_its_run(tmp_path: Path) -> None:
    path = write_cost_record(
        tmp_path,
        pod_id="pod_9",
        gpu_type_id="NVIDIA L40",
        cost_per_hour=0.82,
        duration_s=1800.0,
        note="phase 1 kernel run",
        run_label="phase-1-grouped-gemm",
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )

    assert path.name == "2026-09-20-phase-1-grouped-gemm-cost.md"
    assert path.read_text().startswith("# phase-1-grouped-gemm -- GPU rental cost")


def test_build_parser_create_parses_expected_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "create",
            "--name",
            "dispatch-baseline",
            "--gpu-type",
            "NVIDIA A40",
            "--image",
            "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        ]
    )

    assert args.command == "create"
    assert args.name == "dispatch-baseline"
    assert args.gpu_type == "NVIDIA A40"
    assert args.cloud == "COMMUNITY"
    assert args.disk_gb == 60


def test_main_create_invokes_create_pod_and_prints_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_create_pod(
        name: str, gpu_type_id: str, image: str, *, cloud: str = "COMMUNITY", disk_gb: int = 60
    ) -> PodHandle:
        return PodHandle(id="pod_999", status="PROVISIONING", cost_per_hour=0.4)

    monkeypatch.setattr("scripts.gpu.provision.create_pod", fake_create_pod)

    main(["create", "--name", "x", "--gpu-type", "NVIDIA A40", "--image", "img"])

    assert "pod_999" in capsys.readouterr().out


def test_main_terminate_invokes_terminate_pod(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("scripts.gpu.provision.terminate_pod", calls.append)

    main(["terminate", "--pod-id", "pod_999"])

    assert calls == ["pod_999"]
    assert "pod_999" in capsys.readouterr().out
