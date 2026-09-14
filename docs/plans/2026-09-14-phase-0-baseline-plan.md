# Phase 0: Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up Phase 0 of dispatch: a scripted, cost-tracked RunPod
GPU-rental workflow, and a plain-`transformers` baseline benchmark for
DeepSeekMoE-16B whose measured latency/throughput and captured reference
logits become the correctness oracle every later kernel phase (Phase 1+)
is checked against.

**Architecture:** Two independent pieces. (1) `scripts/gpu/` wraps
RunPod's REST v2 API for pod create/wait/terminate plus a cost-record
writer -- fully unit-tested against fake HTTP sessions, no real network
calls, no API key needed to run the test suite. (2) `src/dispatch/benchmark/`
separates pure latency/throughput math (`metrics.py`) from the HF
model-loading and token-by-token timing I/O (`harness.py`) and
reference-logit capture (`reference.py`) -- the pure piece is tested with
fabricated data, the I/O piece against a tiny public HF test model, marked
`slow` (network, no GPU needed). `scripts/run_baseline.py` is the CLI that
ties the benchmark pieces together for the one real run against the full
32.8GB DeepSeekMoE-16B model on a rented GPU -- that run (Task 8) is a
runbook, not code a TDD worker executes unattended, because it spends real
money.

**Tech Stack:** Python 3.12, `uv`, PyTorch >=2.14.0, `transformers`
>=5.17.0, `accelerate` >=1.15.0, `huggingface_hub` >=1.31.0, `safetensors`
>=0.8.0, `requests` >=2.34.2 (+ `types-requests` dev), RunPod REST API v2
(`https://api.runpod.io/v2`).

**Spec:** `docs/design/2026-09-14-dispatch-system-design.md` (S2 model
choice and its memory note, S7 Phase 0 row, S9 cost plan), plus
`docs/adr/0001` (why a kernel over another attention reimplementation --
context for why Phase 0's reference matters) and `docs/STATUS.md`'s
"Next step" entry, which this plan implements directly.

## Global Constraints

- Python 3.12+, `uv`, `ruff`, `mypy --strict`, `pytest`. `make check`
  (lint, typecheck, test) is green before every push.
- GPU-dependent tests are marked `gpu` and excluded from CI (no GPU
  runner). Network-dependent-but-CPU-only tests are marked `slow`
  (excluded from `make test-fast`, still run in `make test` and CI).
- Dependency floors are checked live against pypi.org's JSON API before
  being written down, never guessed or carried over -- done for this plan
  on 2026-09-14: `torch==2.14.0`, `transformers==5.17.0`,
  `accelerate==1.15.0`, `huggingface_hub==1.31.0`, `safetensors==0.8.0`,
  `requests==2.34.2`, `types-requests==2.33.0.20260906` were each the
  current release at that time; floors below use `>=` against those.
- `mypy --strict` may need targeted `# type: ignore[<code>]` comments
  where `torch`/`transformers`'s own type stubs are incomplete -- narrow,
  code-specific ignores only, added only after confirming the error is a
  stub gap and not a real bug. Never a blanket per-module ignore.
- **Correctness before speed** (CLAUDE.md's governing principle): the
  reference logits this plan captures (Task 6) are the numerical ground
  truth every later kernel version gets checked against within a stated
  tolerance. Getting this capture right matters more than anything else
  in this plan.
- **Never quote a benchmark number that wasn't measured**, on this exact
  hardware/config, by this repo. A benchmark claim states its config
  (batch size, sequence length, hardware, dtype) alongside the number.
- **Marketplace/spot instances only** for Phase 0 (`cloud: "COMMUNITY"`
  in the RunPod request body) -- no NVLink/SXM requirement at this phase.
- **Budget cap set before the first rental, not after.** Spin up, run,
  capture evidence, tear down immediately -- never leave a rented pod
  idle. Cost per run is measured (never estimated) and logged to
  `docs/findings/`.
- **One commit per completed task.** `docs/STATUS.md` updates in the
  same commit as the work. Conventional commit prefixes (`feat:`,
  `test:`, `docs:`). Commit messages are plain ASCII (`--`, never an
  em-dash).
- Renting a GPU costs real money and needs the user's RunPod account and
  API key. Task 8 (the only task that spends money) is a runbook to be
  followed with the user's explicit go-ahead -- not something this plan's
  executor runs unattended, regardless of which execution mode is chosen.
- CI's `uv sync` will pull PyPI's default (CUDA-capable) Linux `torch`
  wheel, which is large. Trimming that via a CPU-only index is a known
  possible future optimization, deliberately **not** included here --
  this plan doesn't guess at unverified `uv` index syntax, and CI time is
  a secondary concern next to Phase 0's actual deliverable. Revisit only
  if CI is observed to be slow or flaky because of it.

## File Structure

```
conftest.py                              # repo-root sys.path shim so tests/ can import scripts/
scripts/
  __init__.py
  gpu/
    __init__.py
    runpod_client.py                     # Task 1: thin REST wrapper, no orchestration logic
    provision.py                         # Task 2+3: wait_until_running, write_cost_record, CLI
  run_baseline.py                        # Task 7: the real-run CLI
src/dispatch/
  benchmark/
    __init__.py
    metrics.py                           # Task 4: pure TokenTimings/BenchmarkSummary math
    harness.py                           # Task 5: load_model, generate_with_timings (I/O)
    reference.py                         # Task 6: capture/save/load/compare reference logits
tests/unit/
  test_runpod_client.py
  test_provision.py
  test_metrics.py
  test_harness.py
  test_reference.py
  test_run_baseline.py
docs/
  runbooks/
    phase-0-baseline.md                  # Task 8: the manual, paid procedure
  findings/                              # created at Task 8 run time, not before
```

---

### Task 1: RunPod API client

**Files:**
- Create: `conftest.py`
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/gpu/__init__.py` (empty)
- Create: `scripts/gpu/runpod_client.py`
- Test: `tests/unit/test_runpod_client.py`
- Modify: `Makefile:23-24` (typecheck target)
- Modify: `.github/workflows/ci.yml` (mypy step)
- Modify: `pyproject.toml` (add `requests`, `types-requests`)
- Modify: `docs/STATUS.md` (Phase 0 progress checklist)

**Interfaces:**
- Produces: `PodHandle(id: str, status: str, cost_per_hour: float)` (frozen
  dataclass), `RunPodAPIError(RuntimeError)`, `create_pod(name: str,
  gpu_type_id: str, image: str, *, disk_gb: int = 60, cloud: str =
  "COMMUNITY", ports: tuple[str, ...] = ("22/tcp",), start_ssh: bool =
  True, session: requests.Session | None = None) -> PodHandle`,
  `get_pod(pod_id: str, *, session: requests.Session | None = None) ->
  PodHandle`, `terminate_pod(pod_id: str, *, session: requests.Session |
  None = None) -> None`. Task 2 imports all of these.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, replace the empty `dependencies = []` line with:

```toml
dependencies = [
    "requests>=2.34.2",
]
```

And add to the `dev` group in `[dependency-groups]`:

```toml
dev = [
    "mypy>=2.3.1",
    "pytest>=9.1.1",
    "pytest-cov>=7.1.0",
    "ruff>=0.16.7",
    "types-requests>=2.33.0.20260906",
]
```

Run: `uv sync --all-extras --dev`
Expected: resolves and installs cleanly.

- [ ] **Step 2: Make `scripts/` importable from tests, and type-checked**

Create `conftest.py` at the repo root:

```python
"""Puts the repo root on sys.path so tests can import scripts/ as a package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

Create empty `scripts/__init__.py` and `scripts/gpu/__init__.py`.

In `Makefile`, change:

```makefile
typecheck:
	uv run mypy src tests
```

to:

```makefile
typecheck:
	uv run mypy src tests scripts
```

In `.github/workflows/ci.yml`, change `- run: uv run mypy src tests` to
`- run: uv run mypy src tests scripts`.

- [ ] **Step 3: Write the failing tests**

Create `tests/unit/test_runpod_client.py`:

```python
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
    session = FakeSession(
        FakeResponse(200, {"id": "pod_123", "status": "PROVISIONING", "cost": 0.4})
    )

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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_runpod_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.gpu.runpod_client'`.

- [ ] **Step 5: Write the implementation**

Create `scripts/gpu/runpod_client.py`:

```python
"""Thin REST client for RunPod's pod-management API (v2).

No retry/backoff logic here on purpose -- that belongs in the orchestration
layer (scripts/gpu/provision.py), which knows what "retry" should mean for
each call (a pod create is not safe to blindly retry; a status poll is).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    if resp.status_code == 204 or not resp.content:
        return {}
    result: dict[str, Any] = resp.json()
    return result


def _pod_from_response(data: dict[str, Any]) -> PodHandle:
    return PodHandle(
        id=str(data["id"]), status=str(data["status"]), cost_per_hour=float(data.get("cost", 0.0))
    )


def create_pod(
    name: str,
    gpu_type_id: str,
    image: str,
    *,
    disk_gb: int = 60,
    cloud: str = "COMMUNITY",
    ports: tuple[str, ...] = ("22/tcp",),
    start_ssh: bool = True,
    session: requests.Session | None = None,
) -> PodHandle:
    body = {
        "name": name,
        "image": image,
        "gpu": {"id": gpu_type_id, "count": 1},
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_runpod_client.py -v`
Expected: 5 passed.

- [ ] **Step 7: Lint and typecheck**

Run: `make check-fast`
Expected: lint and typecheck both clean. Fix any `mypy --strict` findings
with the narrowest possible change before moving on (see Global
Constraints on `# type: ignore`).

- [ ] **Step 8: Update STATUS.md**

In `docs/STATUS.md`, add a new section after "## Next step":

```markdown
## Phase 0 progress

- [x] Task 1: RunPod API client (`scripts/gpu/runpod_client.py`)
- [ ] Task 2: Pod-wait orchestration + cost-record logger (`scripts/gpu/provision.py`)
- [ ] Task 3: Provisioning CLI (`scripts/gpu/provision.py` `main()`)
- [ ] Task 4: Pure benchmark metrics (`src/dispatch/benchmark/metrics.py`)
- [ ] Task 5: Generation harness (`src/dispatch/benchmark/harness.py`)
- [ ] Task 6: Reference-logit capture + tolerance compare (`src/dispatch/benchmark/reference.py`)
- [ ] Task 7: Baseline CLI (`scripts/run_baseline.py`)
- [ ] Task 8: Real rented-GPU run -- runbook executed, results + reference + cost recorded
```

- [ ] **Step 9: Commit**

```bash
git add conftest.py scripts/__init__.py scripts/gpu/__init__.py \
  scripts/gpu/runpod_client.py tests/unit/test_runpod_client.py \
  Makefile .github/workflows/ci.yml pyproject.toml uv.lock docs/STATUS.md
git commit -m "feat: add RunPod REST client for pod create/get/terminate"
```

---

### Task 2: Pod-wait orchestration and cost-record logger

**Files:**
- Create: `scripts/gpu/provision.py`
- Test: `tests/unit/test_provision.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `PodHandle`, `RunPodAPIError`, `get_pod` from
  `scripts.gpu.runpod_client` (Task 1).
- Produces: `wait_until_running(pod_id: str, *, get_pod_fn:
  Callable[[str], PodHandle] = get_pod, timeout_s: float = 600.0,
  poll_interval_s: float = 10.0, sleep_fn: Callable[[float], None] =
  time.sleep, clock_fn: Callable[[], float] = time.monotonic) ->
  PodHandle`, `write_cost_record(findings_dir: Path, *, pod_id: str,
  gpu_type_id: str, cost_per_hour: float, duration_s: float, note: str,
  now: datetime | None = None) -> Path`. Task 3's CLI and Task 8's runbook
  both call these directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_provision.py`:

```python
"""Tests for pod-wait orchestration and cost-record writing.

wait_until_running takes get_pod_fn/sleep_fn/clock_fn so these tests never
sleep for real and never hit the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.gpu.provision import wait_until_running, write_cost_record
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.gpu.provision'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/gpu/provision.py`:

```python
"""Orchestration on top of runpod_client: wait for a pod to come up, and
record its billed cost once a run is done.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from scripts.gpu.runpod_client import PodHandle, RunPodAPIError, get_pod

_TERMINAL_STATUSES = frozenset({"ERROR", "TERMINATED"})


def wait_until_running(
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
                f"pod {pod_id} did not reach RUNNING within {timeout_s}s (last status: {pod.status})"
            )
        sleep_fn(poll_interval_s)


def write_cost_record(
    findings_dir: Path,
    *,
    pod_id: str,
    gpu_type_id: str,
    cost_per_hour: float,
    duration_s: float,
    note: str,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    cost_usd = cost_per_hour * (duration_s / 3600)
    findings_dir.mkdir(parents=True, exist_ok=True)
    path = findings_dir / f"{now:%Y-%m-%d}-phase-0-baseline-cost.md"
    path.write_text(
        "# Phase 0 baseline -- GPU rental cost\n\n"
        f"- Pod: `{pod_id}` ({gpu_type_id})\n"
        f"- Rate: ${cost_per_hour:.4f}/hr\n"
        f"- Duration: {duration_s:.0f}s ({duration_s / 3600:.3f}hr)\n"
        f"- Cost: ${cost_usd:.4f}\n"
        f"- Note: {note}\n"
    )
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provision.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make check-fast`
Expected: clean.

- [ ] **Step 6: Update STATUS.md**

Flip Task 2's box in the Phase 0 progress checklist to `[x]`.

- [ ] **Step 7: Commit**

```bash
git add scripts/gpu/provision.py tests/unit/test_provision.py docs/STATUS.md
git commit -m "feat: add pod-wait orchestration and cost-record logger"
```

---

### Task 3: Provisioning CLI

**Files:**
- Modify: `scripts/gpu/provision.py` (add `build_parser`, `main`)
- Modify: `tests/unit/test_provision.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `create_pod`, `terminate_pod` from `scripts.gpu.runpod_client`
  (Task 1); `wait_until_running` from this file (Task 2).
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv:
  list[str] | None = None) -> None`. Task 8's runbook invokes this as
  `python -m scripts.gpu.provision <create|wait|terminate> ...`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_provision.py`:

```python
from scripts.gpu.provision import build_parser, main
from scripts.gpu.runpod_client import PodHandle


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_provision.py -v`
Expected: FAIL -- `build_parser`/`main` not defined.

- [ ] **Step 3: Write the implementation**

Append to `scripts/gpu/provision.py`:

```python
import argparse

from scripts.gpu.runpod_client import create_pod, terminate_pod


def _cmd_create(args: argparse.Namespace) -> None:
    pod = create_pod(args.name, args.gpu_type, args.image, cloud=args.cloud, disk_gb=args.disk_gb)
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
```

Note: `import pytest` must also be added to the top of
`tests/unit/test_provision.py` if not already present from Task 2 (it is
needed there too, for `pytest.raises`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provision.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make check-fast`
Expected: clean.

- [ ] **Step 6: Update STATUS.md**

Flip Task 3's box to `[x]`.

- [ ] **Step 7: Commit**

```bash
git add scripts/gpu/provision.py tests/unit/test_provision.py docs/STATUS.md
git commit -m "feat: add provisioning CLI (create/wait/terminate)"
```

---

### Task 4: Pure benchmark metrics

**Files:**
- Create: `src/dispatch/benchmark/__init__.py` (empty)
- Create: `src/dispatch/benchmark/metrics.py`
- Test: `tests/unit/test_metrics.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Produces: `TokenTimings(start_time: float, token_times: tuple[float,
  ...], prompt_token_count: int)` (frozen dataclass with properties
  `generated_token_count`, `time_to_first_token`, `inter_token_latencies`,
  `total_latency`, `tokens_per_second`); `BenchmarkSummary(run_count: int,
  mean_ttft: float, p50_ttft: float, p99_ttft: float,
  mean_inter_token_latency: float, mean_tokens_per_second: float)` (frozen
  dataclass); `summarize(runs: list[TokenTimings]) -> BenchmarkSummary`.
  Task 5 produces `TokenTimings` instances; Task 7 calls `summarize`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_metrics.py`:

```python
"""Pure math -- no model, no I/O, fabricated timestamps throughout."""

from __future__ import annotations

import pytest

from dispatch.benchmark.metrics import TokenTimings, summarize


def test_token_timings_computes_ttft_and_latency() -> None:
    timing = TokenTimings(start_time=10.0, token_times=(10.1, 10.15, 10.25), prompt_token_count=5)

    assert timing.time_to_first_token == pytest.approx(0.1)
    assert timing.generated_token_count == 3
    assert timing.total_latency == pytest.approx(0.25)
    assert timing.inter_token_latencies == pytest.approx((0.05, 0.10))
    assert timing.tokens_per_second == pytest.approx(3 / 0.25)


def test_token_timings_rejects_empty_token_times() -> None:
    with pytest.raises(ValueError, match="token_times"):
        TokenTimings(start_time=0.0, token_times=(), prompt_token_count=1)


def test_summarize_aggregates_across_runs() -> None:
    runs = [
        TokenTimings(start_time=0.0, token_times=(0.1, 0.2), prompt_token_count=1),
        TokenTimings(start_time=0.0, token_times=(0.2, 0.4), prompt_token_count=1),
    ]

    summary = summarize(runs)

    assert summary.run_count == 2
    assert summary.mean_ttft == pytest.approx((0.1 + 0.2) / 2)
    assert summary.p50_ttft == pytest.approx(0.15)
    assert summary.mean_inter_token_latency == pytest.approx((0.1 + 0.2) / 2)


def test_summarize_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="runs"):
        summarize([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'dispatch.benchmark'`.

- [ ] **Step 3: Write the implementation**

Create `src/dispatch/benchmark/metrics.py`:

```python
"""Pure latency/throughput metrics computed from per-token timestamps.

Kept free of torch/transformers imports on purpose: this is the one part
of the benchmark stack that needs no model, no GPU, and no network to test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenTimings:
    """Timestamps (seconds, from a monotonic clock) captured during one generation run."""

    start_time: float
    token_times: tuple[float, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        if not self.token_times:
            raise ValueError("token_times must have at least one entry")

    @property
    def generated_token_count(self) -> int:
        return len(self.token_times)

    @property
    def time_to_first_token(self) -> float:
        return self.token_times[0] - self.start_time

    @property
    def inter_token_latencies(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False))

    @property
    def total_latency(self) -> float:
        return self.token_times[-1] - self.start_time

    @property
    def tokens_per_second(self) -> float:
        return self.generated_token_count / self.total_latency


@dataclass(frozen=True)
class BenchmarkSummary:
    run_count: int
    mean_ttft: float
    p50_ttft: float
    p99_ttft: float
    mean_inter_token_latency: float
    mean_tokens_per_second: float


def summarize(runs: list[TokenTimings]) -> BenchmarkSummary:
    if not runs:
        raise ValueError("runs must not be empty")
    ttfts = sorted(run.time_to_first_token for run in runs)
    all_itls = [latency for run in runs for latency in run.inter_token_latencies]
    return BenchmarkSummary(
        run_count=len(runs),
        mean_ttft=statistics.mean(ttfts),
        p50_ttft=_percentile(ttfts, 0.50),
        p99_ttft=_percentile(ttfts, 0.99),
        mean_inter_token_latency=statistics.mean(all_itls) if all_itls else 0.0,
        mean_tokens_per_second=statistics.mean(run.tokens_per_second for run in runs),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and typecheck**

Run: `make check-fast`
Expected: clean.

- [ ] **Step 6: Update STATUS.md**

Flip Task 4's box to `[x]`.

- [ ] **Step 7: Commit**

```bash
git add src/dispatch/benchmark/__init__.py src/dispatch/benchmark/metrics.py \
  tests/unit/test_metrics.py docs/STATUS.md
git commit -m "feat: add pure latency/throughput metrics for the benchmark harness"
```

---

### Task 5: Generation harness (model loading + timed decoding)

**Files:**
- Create: `src/dispatch/benchmark/harness.py`
- Test: `tests/unit/test_harness.py`
- Modify: `pyproject.toml` (add `torch`, `transformers`, `accelerate`,
  `huggingface_hub`)
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `TokenTimings` from `dispatch.benchmark.metrics` (Task 4).
- Produces: `load_model(model_name: str, *, device: str = "cpu", dtype:
  torch.dtype = torch.float32, trust_remote_code: bool = False) ->
  tuple[PreTrainedModel, PreTrainedTokenizerBase]`,
  `generate_with_timings(model: PreTrainedModel, tokenizer:
  PreTrainedTokenizerBase, prompt: str, *, max_new_tokens: int = 32,
  device: str = "cpu", clock_fn: Callable[[], float] = time.perf_counter)
  -> TokenTimings`. Task 6 and Task 7 both call `load_model`; Task 7
  calls `generate_with_timings`.

- [ ] **Step 1: Add ML dependencies**

In `pyproject.toml`, extend `dependencies`:

```toml
dependencies = [
    "requests>=2.34.2",
    "torch>=2.14.0",
    "transformers>=5.17.0",
    "accelerate>=1.15.0",
    "huggingface_hub>=1.31.0",
]
```

Run: `uv sync --all-extras --dev`
Expected: resolves and installs cleanly (this is the first step in the
plan that downloads torch -- expect it to take a while and use several
GB of disk).

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_harness.py`:

```python
"""Two tiers: a fast fake-model test of the timing loop itself (no network,
no GPU), and a `slow` test against a real tiny public HF model (network,
still no GPU -- CPU inference on a few-KB model is fast).
"""

from __future__ import annotations

import pytest
import torch

from dispatch.benchmark.harness import generate_with_timings, load_model

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.past_key_values = None


class _FakeModel:
    def __init__(self, vocab_size: int = 10) -> None:
        self.vocab_size = vocab_size
        self.call_count = 0

    def __call__(
        self, *, input_ids: torch.Tensor, past_key_values: object, use_cache: bool
    ) -> _FakeOutputs:
        self.call_count += 1
        logits = torch.zeros((1, input_ids.shape[-1], self.vocab_size))
        logits[0, -1, self.call_count % self.vocab_size] = 10.0
        return _FakeOutputs(logits)


class _FakeBatchEncoding(dict):  # type: ignore[type-arg]
    def to(self, device: str) -> "_FakeBatchEncoding":
        return self


class _FakeTokenizer:
    eos_token_id = 999

    def __call__(self, prompt: str, return_tensors: str) -> _FakeBatchEncoding:
        return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))


def test_generate_with_timings_runs_max_new_tokens_steps_without_eos() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    fake_clock = iter([0.0, 0.1, 0.2, 0.3])

    timing = generate_with_timings(
        model,  # type: ignore[arg-type]
        tokenizer,  # type: ignore[arg-type]
        "prompt",
        max_new_tokens=3,
        clock_fn=lambda: next(fake_clock),
    )

    assert timing.generated_token_count == 3
    assert timing.prompt_token_count == 3
    assert timing.start_time == 0.0
    assert timing.token_times == (0.1, 0.2, 0.3)


@pytest.mark.slow
def test_generate_with_timings_against_a_real_tiny_model() -> None:
    model, tokenizer = load_model(TINY_MODEL)

    timing = generate_with_timings(model, tokenizer, "hello world", max_new_tokens=5)

    assert 1 <= timing.generated_token_count <= 5
    assert timing.time_to_first_token >= 0
    assert all(b >= a for a, b in zip(timing.token_times, timing.token_times[1:], strict=False))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_harness.py -v -m "not slow"`
Expected: FAIL -- `ModuleNotFoundError: No module named 'dispatch.benchmark.harness'`.

- [ ] **Step 4: Write the implementation**

Create `src/dispatch/benchmark/harness.py`:

```python
"""Loads a causal LM and runs greedy, token-by-token generation, timing
each decode step. Manual loop (not model.generate()) because per-token
timestamps are the whole point -- generate() only returns the final
sequence, not when each token was produced.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dispatch.benchmark.metrics import TokenTimings


def load_model(
    model_name: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_with_timings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    device: str = "cpu",
    clock_fn: Callable[[], float] = time.perf_counter,
) -> TokenTimings:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_token_count = int(input_ids.shape[-1])
    eos_token_id = tokenizer.eos_token_id

    start_time = clock_fn()
    token_times: list[float] = []
    past_key_values = None
    next_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            token_times.append(clock_fn())
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_input = next_token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return TokenTimings(
        start_time=start_time, token_times=tuple(token_times), prompt_token_count=prompt_token_count
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_harness.py -v -m "not slow"`
Expected: 1 passed (fake-model test). Then run the real one once, since it
needs network access CI will also have:

Run: `uv run pytest tests/unit/test_harness.py -v`
Expected: 2 passed (includes the `slow` real-model test; first run
downloads the tiny model to the local HF cache).

- [ ] **Step 6: Lint and typecheck**

Run: `make check-fast`
Expected: clean, or only stub-gap `type: ignore` comments per Global
Constraints (torch/transformers typing is the most likely source of
noise in this task specifically).

- [ ] **Step 7: Update STATUS.md**

Flip Task 5's box to `[x]`.

- [ ] **Step 8: Commit**

```bash
git add src/dispatch/benchmark/harness.py tests/unit/test_harness.py \
  pyproject.toml uv.lock docs/STATUS.md
git commit -m "feat: add model-loading and per-token-timed generation harness"
```

---

### Task 6: Reference-logit capture and tolerance comparison

**Files:**
- Create: `src/dispatch/benchmark/reference.py`
- Test: `tests/unit/test_reference.py`
- Modify: `pyproject.toml` (add `safetensors`)
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `load_model` from `dispatch.benchmark.harness` (Task 5, used
  only in the `slow` test).
- Produces: `capture_reference_logits(model: PreTrainedModel, tokenizer:
  PreTrainedTokenizerBase, prompts: list[str], *, device: str = "cpu") ->
  dict[str, torch.Tensor]`, `save_reference(tensors: dict[str,
  torch.Tensor], path: Path) -> None`, `load_reference(path: Path) ->
  dict[str, torch.Tensor]`, `compare_within_tolerance(actual: dict[str,
  torch.Tensor], reference: dict[str, torch.Tensor], *, rtol: float =
  1e-3, atol: float = 1e-5) -> dict[str, bool]`. Task 7 calls
  `capture_reference_logits` and `save_reference`; Phase 1's kernel work
  (not this plan) will call `load_reference` and `compare_within_tolerance`.

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, extend `dependencies` with `"safetensors>=0.8.0"`.

Run: `uv sync --all-extras --dev`
Expected: resolves cleanly.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_reference.py`:

```python
"""Reference-logit capture is Phase 1's correctness oracle -- get the
comparison semantics right here, before any kernel exists to check."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dispatch.benchmark.reference import (
    capture_reference_logits,
    compare_within_tolerance,
    load_reference,
    save_reference,
)


def test_compare_within_tolerance_true_for_identical_tensors() -> None:
    tensors = {"a": torch.tensor([1.0, 2.0, 3.0])}

    assert compare_within_tolerance(tensors, tensors) == {"a": True}


def test_compare_within_tolerance_false_beyond_tolerance() -> None:
    actual = {"a": torch.tensor([1.0, 2.0, 3.0])}
    reference = {"a": torch.tensor([1.0, 2.0, 30.0])}

    assert compare_within_tolerance(actual, reference, rtol=1e-3, atol=1e-5) == {"a": False}


def test_compare_within_tolerance_raises_on_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        compare_within_tolerance({"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])})


def test_save_and_load_reference_round_trips(tmp_path: Path) -> None:
    tensors = {"prompt_000_logits": torch.randn(4, 10)}
    path = tmp_path / "reference.safetensors"

    save_reference(tensors, path)
    loaded = load_reference(path)

    assert loaded.keys() == tensors.keys()
    assert torch.equal(loaded["prompt_000_logits"], tensors["prompt_000_logits"])


@pytest.mark.slow
def test_capture_reference_logits_is_deterministic_for_fixed_prompts() -> None:
    from dispatch.benchmark.harness import load_model

    model, tokenizer = load_model("hf-internal-testing/tiny-random-gpt2")

    first = capture_reference_logits(model, tokenizer, ["hello"])
    second = capture_reference_logits(model, tokenizer, ["hello"])

    assert compare_within_tolerance(first, second, rtol=0, atol=0) == {"prompt_000_logits": True}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_reference.py -v -m "not slow"`
Expected: FAIL -- `ModuleNotFoundError: No module named 'dispatch.benchmark.reference'`.

- [ ] **Step 4: Write the implementation**

Create `src/dispatch/benchmark/reference.py`:

```python
"""Captures reference logits for a fixed prompt set -- the numerical
ground truth Phase 1's kernel gets checked against within a stated
tolerance (CLAUDE.md's "correctness before speed").
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def capture_reference_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            logits = model(input_ids=input_ids).logits
            tensors[f"prompt_{i:03d}_logits"] = logits.squeeze(0).cpu().contiguous()
    return tensors


def save_reference(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def load_reference(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def compare_within_tolerance(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict[str, bool]:
    if actual.keys() != reference.keys():
        raise ValueError(
            f"key mismatch: actual has {sorted(actual.keys())}, reference has {sorted(reference.keys())}"
        )
    return {
        key: bool(torch.allclose(actual[key], reference[key], rtol=rtol, atol=atol))
        for key in reference
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_reference.py -v`
Expected: 5 passed.

- [ ] **Step 6: Lint and typecheck**

Run: `make check-fast`
Expected: clean (modulo stub-gap ignores, as in Task 5).

- [ ] **Step 7: Update STATUS.md**

Flip Task 6's box to `[x]`.

- [ ] **Step 8: Commit**

```bash
git add src/dispatch/benchmark/reference.py tests/unit/test_reference.py \
  pyproject.toml uv.lock docs/STATUS.md
git commit -m "feat: add reference-logit capture and tolerance comparison"
```

---

### Task 7: Baseline CLI

**Files:**
- Create: `scripts/run_baseline.py`
- Test: `tests/unit/test_run_baseline.py`
- Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `load_model`, `generate_with_timings` from
  `dispatch.benchmark.harness` (Task 5); `TokenTimings`, `summarize` from
  `dispatch.benchmark.metrics` (Task 4); `capture_reference_logits`,
  `save_reference` from `dispatch.benchmark.reference` (Task 6).
- Produces: `run_baseline(model_name: str, *, device: str, dtype:
  torch.dtype, trust_remote_code: bool, prompts: list[str], repetitions:
  int, max_new_tokens: int) -> tuple[list[TokenTimings], dict[str,
  torch.Tensor]]`, `main(argv: list[str] | None = None) -> None`. Task 8's
  runbook invokes this as `python -m scripts.run_baseline ...` against the
  real model on the rented GPU.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_run_baseline.py`:

```python
"""main()'s plumbing (args -> files) is tested fast with run_baseline and
summarize monkeypatched out; the one real end-to-end pass against a tiny
model is `slow`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import scripts.run_baseline as run_baseline_module
from dispatch.benchmark.metrics import BenchmarkSummary
from scripts.run_baseline import main


def test_main_writes_results_and_reference_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_summary = BenchmarkSummary(
        run_count=1,
        mean_ttft=0.1,
        p50_ttft=0.1,
        p99_ttft=0.1,
        mean_inter_token_latency=0.05,
        mean_tokens_per_second=20.0,
    )

    def fake_run_baseline(
        model_name: str, **kwargs: object
    ) -> tuple[list[object], dict[str, torch.Tensor]]:
        return [object()], {"prompt_000_logits": torch.zeros(1)}

    monkeypatch.setattr(run_baseline_module, "run_baseline", fake_run_baseline)
    monkeypatch.setattr(run_baseline_module, "summarize", lambda runs: fake_summary)

    main(
        [
            "--model-name",
            "tiny/test-model",
            "--output-dir",
            str(tmp_path),
            "--run-label",
            "test-run",
        ]
    )

    results = json.loads((tmp_path / "test-run-results.json").read_text())
    assert results["model"] == "tiny/test-model"
    assert results["mean_tokens_per_second"] == 20.0
    assert (tmp_path / "test-run-reference.safetensors").exists()


@pytest.mark.slow
def test_run_baseline_end_to_end_with_a_real_tiny_model() -> None:
    runs, reference = run_baseline_module.run_baseline(
        "hf-internal-testing/tiny-random-gpt2",
        device="cpu",
        dtype=torch.float32,
        trust_remote_code=False,
        prompts=["hello"],
        repetitions=1,
        max_new_tokens=3,
    )

    assert len(runs) == 1
    assert "prompt_000_logits" in reference
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_run_baseline.py -v -m "not slow"`
Expected: FAIL -- `ModuleNotFoundError: No module named 'scripts.run_baseline'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/run_baseline.py`:

```python
"""CLI: run dispatch's Phase 0 baseline -- a plain HF forward pass on one
rented GPU. Produces the honest "before" latency/throughput numbers and
the reference logits Phase 1's kernel gets checked against.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dispatch.benchmark.harness import generate_with_timings, load_model
from dispatch.benchmark.metrics import TokenTimings, summarize
from dispatch.benchmark.reference import capture_reference_logits, save_reference

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a small crew of explorers",
    "def fibonacci(n):",
]


def run_baseline(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    prompts: list[str],
    repetitions: int,
    max_new_tokens: int,
) -> tuple[list[TokenTimings], dict[str, torch.Tensor]]:
    model, tokenizer = load_model(
        model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )

    runs = [
        generate_with_timings(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        for prompt in prompts
        for _ in range(repetitions)
    ]
    reference = capture_reference_logits(model, tokenizer, prompts, device=device)
    return runs, reference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dispatch's Phase 0 baseline benchmark")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/findings"))
    parser.add_argument("--run-label", default=time.strftime("%Y-%m-%d-phase-0-baseline"))
    args = parser.parse_args(argv)

    dtype: torch.dtype = getattr(torch, args.dtype)
    runs, reference = run_baseline(
        args.model_name,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        prompts=DEFAULT_PROMPTS,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.run_label}-results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "device": args.device,
                "dtype": args.dtype,
                **asdict(summary),
            },
            indent=2,
        )
    )

    reference_path = args.output_dir / f"{args.run_label}-reference.safetensors"
    save_reference(reference, reference_path)

    print(f"wrote {results_path}")
    print(f"wrote {reference_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_run_baseline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Full suite, lint, typecheck**

Run: `make check`
Expected: all green -- this is the last code task before Task 8's real run.

- [ ] **Step 6: Update STATUS.md**

Flip Task 7's box to `[x]`. Also update the "Current position" paragraph:
replace "Still no code" with a short note that Phase 0's tooling is built
and tested, and only the real rented-GPU run (Task 8) remains.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_baseline.py tests/unit/test_run_baseline.py docs/STATUS.md
git commit -m "feat: add Phase 0 baseline CLI"
```

---

### Task 8: Real rented-GPU run (runbook, not automated)

**This task spends real money and needs the user's RunPod account and API
key. Do not run any pod-creating command in this task without the user's
explicit go-ahead and a stated budget cap, regardless of which execution
mode (subagent-driven or inline) is running this plan.**

**Files:**
- Create: `docs/runbooks/phase-0-baseline.md`
- Modify: `docs/STATUS.md`
- Modify: `CLAUDE.md` (Current status, if this is a natural stopping point)
- Creates at run time (not committed by this task's Step 1): `docs/findings/*`

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/phase-0-baseline.md`:

```markdown
# Runbook: Phase 0 baseline (real rented GPU)

Tasks 1-7 are developed and tested locally at zero cost. This is the one
step that spends real money. Confirm a budget cap with the user and get
explicit go-ahead before running any command that creates a pod.

## Prerequisites

- `RUNPOD_API_KEY` exported in the shell (RunPod console -> Settings ->
  API Keys).
- A budget cap stated out loud before the first `provision.py create`.
- `uv sync` run locally so `scripts/gpu/provision.py` and
  `scripts/run_baseline.py` are runnable.

## 1. Pick a GPU type and image

Query the live catalog rather than assuming a GPU type id or image tag:

    curl -s https://api.runpod.io/v2/catalog/gpus \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      | python3 -c "
import json, sys
gpus = json.load(sys.stdin)['gpus']
for g in gpus:
    if g['memory'] >= 40 and g['community']:
        print(g['id'], g['memory'], 'GB', g['price']['community'], '\$/hr community')
"

Pick the cheapest 40GB+ card on community cloud -- the design doc's memory
note is that 32.8GB of weights leaves limited headroom below 40GB.
Confirm a current `runpod/pytorch:*` image tag at
https://hub.docker.com/r/runpod/pytorch/tags; the tag in this runbook's
examples (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) was current
2026-09-14 and may have moved on.

## 2. Check whether the model needs trust_remote_code

    curl -s https://huggingface.co/api/models/deepseek-ai/deepseek-moe-16b-base \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('config', {}).get('model_type'))"

If the reported `model_type` is natively supported by the installed
`transformers` version, omit `--trust-remote-code` in step 5 (the
default). If not, add the flag.

## 3. Create the pod

    uv run python -m scripts.gpu.provision create \
      --name dispatch-phase-0-baseline \
      --gpu-type "<id from step 1>" \
      --image "<tag from step 1>" \
      --cloud COMMUNITY

Record the printed pod id.

## 4. Wait for it to come up

    uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

Connect via the SSH details shown in the RunPod console for that pod --
this project doesn't reimplement RunPod's own connection tooling.

## 5. Run the baseline on the pod

    git clone <this repo> && cd dispatch
    uv sync
    uv run python -m scripts.run_baseline \
      --model-name deepseek-ai/deepseek-moe-16b-base \
      --device cuda \
      --dtype bfloat16 \
      --repetitions 5 \
      --max-new-tokens 64 \
      --output-dir docs/findings \
      --run-label $(date +%Y-%m-%d)-phase-0-baseline

This writes `docs/findings/<label>-results.json` (the correctness-
reference latency/throughput numbers) and
`docs/findings/<label>-reference.safetensors` (the logits Phase 1's
kernel gets checked against).

## 6. Copy results back, log cost, tear down -- immediately

From your local machine:

    scp <pod>:dispatch/docs/findings/<label>-* docs/findings/

Then, before doing anything else:

    uv run python -c "
from pathlib import Path
from scripts.gpu.provision import write_cost_record
from scripts.gpu.runpod_client import get_pod

pod = get_pod('<pod_id>')
write_cost_record(
    Path('docs/findings'),
    pod_id=pod.id,
    gpu_type_id='<id from step 1>',
    cost_per_hour=pod.cost_per_hour,
    duration_s=<measured wall-clock seconds the pod was RUNNING>,
    note='Phase 0 baseline: deepseek-ai/deepseek-moe-16b-base, single GPU',
)
"
    uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>

Verify termination independently -- re-run `get_pod('<pod_id>').status`
and confirm it reports `TERMINATED` -- rather than trusting the
terminate command's exit code alone.

## 7. Record the finding

- Commit `docs/findings/<label>-results.json`,
  `<label>-reference.safetensors`, and `<label>-cost.md`.
- Update `docs/STATUS.md`: Phase 0 complete, quoting the measured numbers
  (never estimated), flip Task 8's checklist box.
- Update `CLAUDE.md`'s Current status if this is a natural stopping point
  a reader would want to know about (the proactive-refresh convention).
```

- [ ] **Step 2: Update STATUS.md and commit the runbook**

Flip Task 8's box in the Phase 0 progress checklist to reflect that the
runbook exists but has not yet been run (do not mark it fully `[x]` --
that happens only after Step 3 below actually completes). Use, e.g.:
`- [ ] Task 8: runbook written (\`docs/runbooks/phase-0-baseline.md\`), not yet run`.

```bash
git add docs/runbooks/phase-0-baseline.md docs/STATUS.md
git commit -m "docs: add Phase 0 baseline runbook"
```

- [ ] **Step 3: Execute the runbook -- gated on explicit user go-ahead**

Confirm with the user: budget cap, RunPod account ready,
`RUNPOD_API_KEY` set. Only then follow `docs/runbooks/phase-0-baseline.md`
end to end. This step's "test" is the runbook's own checklist: results
JSON and reference safetensors captured, cost logged, pod verified
`TERMINATED`.

- [ ] **Step 4: Final STATUS.md and CLAUDE.md update, commit**

Update `docs/STATUS.md`'s "Current position" to record Phase 0 complete
with the measured numbers (TTFT, tokens/sec, cost) and flip Task 8's box
to `[x]`. Update CLAUDE.md's Current status section per its own
proactive-refresh convention. Commit both together:

```bash
git add docs/STATUS.md CLAUDE.md docs/findings/
git commit -m "docs: Phase 0 baseline complete -- measured latency/throughput and cost"
```

---

## Self-Review

**Spec coverage:** S2's memory note (40GB headroom) -> runbook step 1's
GPU-selection filter. S7's Phase 0 row (plain HF transformers, one rented
GPU, measured latency/throughput as the correctness reference) -> Tasks
5-7. S9's cost plan (marketplace/spot, budget cap before first rental,
cost measured and logged to `docs/findings/`, never estimated) -> Task 2's
`write_cost_record` and the runbook's step 6. CLAUDE.md's Correctness-
before-speed governing principle -> Task 6's reference-logit capture,
called out explicitly in that task's docstring and this plan's Global
Constraints. STATUS.md's "Next step" (plan, then `scripts/gpu/`, before
any model-serving code) -> this plan's task ordering (GPU provisioning is
Tasks 1-3, before any model code in Tasks 4-7).

**Placeholder scan:** every step above either runs a real command or shows
complete code -- no "TBD", no "add appropriate error handling" without the
actual handling shown, no "similar to Task N" references.

**Type consistency:** `PodHandle`, `RunPodAPIError`, `create_pod`,
`get_pod`, `terminate_pod` (Task 1) are used with the same names and
signatures in Tasks 2, 3, and the Task 8 runbook. `TokenTimings` (Task 4)
is constructed identically in Task 4's own tests, Task 5's harness, and
Task 5/7's fake-model test. `BenchmarkSummary`/`summarize` (Task 4) match
between Task 4's tests and Task 7's `run_baseline`/`main`. `load_model`
and `generate_with_timings` (Task 5) are called with the same keyword
arguments in Task 6's `slow` test and Task 7's `run_baseline`.
`capture_reference_logits`/`save_reference` (Task 6) match their Task 7
call sites exactly.
