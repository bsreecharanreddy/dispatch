from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts.gpu.phase6_engine_reference import run_engine_reference


class _FakeServer:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class _StuckServer(_FakeServer):
    def wait(self, timeout: float) -> int:
        raise subprocess.TimeoutExpired("server", timeout)


def _good_raw() -> dict[str, Any]:
    return {
        "completed": 2,
        "failed": 0,
        "duration": 4.0,
        "output_throughput": 32.0,
        "output_lens": [64, 64],
        "ttfts": [0.1, 0.1],
        "itls": [[0.02], [0.02]],
        "latencies": [2.0, 2.0],
        "errors": ["", ""],
    }


def _run(
    tmp_path: Path, server: _FakeServer, raw_by_concurrency: dict[int, dict[str, Any]]
) -> Path:
    def fake_run(command: list[str], check: bool) -> None:
        concurrency = int(command[command.index("--max-concurrency") + 1])
        filename = command[command.index("--result-filename") + 1]
        (tmp_path / filename).write_text(json.dumps(raw_by_concurrency[concurrency]))

    return run_engine_reference(
        "vllm",
        model="m",
        port=8000,
        concurrencies=list(raw_by_concurrency),
        output_dir=tmp_path,
        run_label="ref",
        gpu_cost_per_hour=0.72,
        popen_fn=lambda command, **kwargs: server,
        run_fn=fake_run,
        get_fn=lambda url: 200,
        sleep_fn=lambda seconds: None,
        clock_fn=lambda: 0.0,
    )


def test_a_clean_run_writes_one_summary_per_concurrency_and_stops_the_server(
    tmp_path: Path,
) -> None:
    server = _FakeServer()

    path = _run(tmp_path, server, {1: _good_raw(), 4: _good_raw()})

    record = json.loads(path.read_text())
    assert [result["concurrency"] for result in record["results"]] == [1, 4]
    assert record["error"] is None
    assert record["config"]["output_len"] == 64
    assert record["config"]["gpu_cost_per_hour"] == 0.72
    assert server.terminated
    assert (tmp_path / "ref-trace.jsonl").exists()


def test_a_bad_run_still_writes_the_earlier_evidence_and_stops_the_server(
    tmp_path: Path,
) -> None:
    server = _FakeServer()
    bad = {**_good_raw(), "output_lens": [64, 3]}

    with pytest.raises(ValueError, match="ignore-eos was not honored"):
        _run(tmp_path, server, {1: _good_raw(), 4: bad})

    record = json.loads((tmp_path / "ref.json").read_text())
    assert [result["concurrency"] for result in record["results"]] == [1]
    assert "ignore-eos" in record["error"]
    assert server.terminated


def test_a_server_that_ignores_terminate_is_killed(tmp_path: Path) -> None:
    server = _StuckServer()

    _run(tmp_path, server, {1: _good_raw()})

    assert server.killed
