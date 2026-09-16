"""Orchestration on top of runpod_client: wait for a pod to come up, and
record its billed cost once a run is done.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from scripts.gpu.runpod_client import PodHandle, RunPodAPIError, create_pod, get_pod, terminate_pod

_TERMINAL_STATUSES = frozenset({"ERROR", "TERMINATED"})


def wait_until_running(  # noqa: PLR0913 -- injectable clock/sleep/get_pod is what makes this testable
    pod_id: str,
    *,
    get_pod_fn: Callable[[str], PodHandle] = get_pod,
    timeout_s: float = 600.0,
    poll_interval_s: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> PodHandle:
    deadline = clock_fn() + timeout_s
    while True:
        pod = get_pod_fn(pod_id)
        if pod.status == "RUNNING":
            return pod
        if pod.status in _TERMINAL_STATUSES:
            raise RunPodAPIError(f"pod {pod_id} entered terminal status {pod.status} while waiting")
        if clock_fn() >= deadline:
            raise TimeoutError(
                f"pod {pod_id} did not reach RUNNING within {timeout_s}s "
                f"(last status: {pod.status})"
            )
        sleep_fn(poll_interval_s)


def write_cost_record(  # noqa: PLR0913 -- a cost record has this many independent fields
    findings_dir: Path,
    *,
    pod_id: str,
    gpu_type_id: str,
    cost_per_hour: float,
    duration_s: float,
    note: str,
    run_label: str = "phase-0-baseline",
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    cost_usd = cost_per_hour * (duration_s / 3600)
    findings_dir.mkdir(parents=True, exist_ok=True)
    path = findings_dir / f"{now:%Y-%m-%d}-{run_label}-cost.md"
    path.write_text(
        f"# {run_label} -- GPU rental cost\n\n"
        f"- Pod: `{pod_id}` ({gpu_type_id})\n"
        f"- Rate: ${cost_per_hour:.4f}/hr\n"
        f"- Duration: {duration_s:.0f}s ({duration_s / 3600:.3f}hr)\n"
        f"- Cost: ${cost_usd:.4f}\n"
        f"- Note: {note}\n"
    )
    return path


def _cmd_create(args: argparse.Namespace) -> None:
    pod = create_pod(
        args.name,
        args.gpu_type,
        args.image,
        cloud=args.cloud,
        disk_gb=args.disk_gb,
        gpu_count=args.gpu_count,
    )
    print(f"created pod {pod.id} status={pod.status} rate=${pod.cost_per_hour:.4f}/hr")


def _cmd_wait(args: argparse.Namespace) -> None:
    pod = wait_until_running(args.pod_id, timeout_s=args.timeout_s)
    print(f"pod {pod.id} is RUNNING, rate=${pod.cost_per_hour:.4f}/hr")


def _cmd_terminate(args: argparse.Namespace) -> None:
    terminate_pod(args.pod_id)
    print(f"terminated pod {args.pod_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provision", description="RunPod pod lifecycle for dispatch's Phase 0 baseline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="create a pod")
    create.add_argument("--name", required=True)
    create.add_argument("--gpu-type", required=True, dest="gpu_type")
    create.add_argument("--image", required=True)
    create.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    create.add_argument("--disk-gb", type=int, default=60, dest="disk_gb")
    create.add_argument("--gpu-count", type=int, default=1, dest="gpu_count")
    create.set_defaults(func=_cmd_create)

    wait = sub.add_parser("wait", help="wait for a pod to reach RUNNING")
    wait.add_argument("--pod-id", required=True, dest="pod_id")
    wait.add_argument("--timeout-s", type=float, default=600.0, dest="timeout_s")
    wait.set_defaults(func=_cmd_wait)

    terminate = sub.add_parser("terminate", help="terminate a pod")
    terminate.add_argument("--pod-id", required=True, dest="pod_id")
    terminate.set_defaults(func=_cmd_terminate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
