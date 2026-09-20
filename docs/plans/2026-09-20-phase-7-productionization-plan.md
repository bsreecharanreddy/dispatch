# Phase 7 productionization (Rust router + Docker + K8s) -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 13 rents a GPU and spends real money: never start it without the user's explicit go-ahead, and read "Cost and stop rules" first.**

**Goal:** Put dispatch's real kernel work behind a real, observable, deployable service -- a Rust router (gRPC to a Python model server, HTTP to clients, Prometheus metrics) demoed once through a local `kind` Kubernetes cluster against the real `deepseek-ai/deepseek-moe-16b-base` model on a rented GPU, with Grafana showing live TTFT/inter-token-latency/queue-depth from that real run.

**Architecture:** A Cargo binary+library crate (`router/`) owns HTTP ingress, single-slot admission, a tonic gRPC client, and a Prometheus registry. A Python gRPC server (`src/dispatch/serving/model_server.py`) implements the same shared `proto/dispatch.proto` contract behind a `Responder` protocol with two implementations: `StubResponder` (canned, no model, used everywhere except the final paid session) and `KernelResponder` (real model + Phase 6's naive kernel + Phase 4's `ColocatedWorker`, GPU-only). Everything is built and proven against the stub first -- unit tests, a cross-language integration test, Docker Compose, and a full `kind` rehearsal -- before the one real GPU session swaps the stub for the kernel behind an SSH tunnel.

**Tech Stack:** Rust (stable, via rustup), tonic 0.14.6, prost 0.14.4, axum 0.8.9, tokio 1.53.1, the `prometheus` crate 0.14.0 (all checked live 2026-09-20 against crates.io); Python 3.12+, `uv`, `ruff`, `mypy --strict`, `pytest`, grpcio/grpcio-tools 1.84.0 (checked live against pypi.org); Docker; `kind` v0.33.0 (checked live against GitHub releases); `kubectl` (already installed, client v1.30.5).

**Spec:** `docs/design/2026-09-20-phase-7-productionization.md`.

## Global Constraints

Every task's requirements implicitly include this section.

- Python side: 3.12+; `uv`, `ruff` (line length 100), `mypy --strict`, `pytest`. Rust side: stable toolchain via `rustup`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`. `make check` (Task 12 onward) runs both gates and is green before every commit and before any push.
- **Correctness before speed, still.** The real GPU session (Task 13) re-verifies `KernelResponder` against a same-session stock generation before it's ever wired behind the router -- the same bar every prior phase's kernel work met.
- **Never quote a number that wasn't measured.** The Grafana dashboard's TTFT/inter-token-latency/queue-depth values come only from the real Task 13 session; nothing in this plan fabricates or estimates them.
- `gpu`-marked pytest tests are excluded from CI, reason visible in the test, same as every earlier phase.
- One commit per task. `docs/STATUS.md` is updated in the same commit as the work it describes.
- **Commit messages are plain ASCII: `--`, never an em-dash. No attribution lines in commit messages or PR descriptions.**
- One branch for the phase: `phase-7-productionization`. One PR at the end. Do not push or open the PR without asking the user.
- Model: `deepseek-ai/deepseek-moe-16b-base`. Kernel: dispatch's **naive bf16** backend -- the exact config Phase 6 already used for its own concurrency-1 served row (21.04 tok/s), not re-decided here.
- Router-to-model-server connectivity: an SSH local-forward from the dev machine into the rented pod, reached from the local `kind` cluster via an `ExternalName` Service pointing at `host.docker.internal` (Task 1 verifies this concretely on this machine before anything else depends on it).
- Hardware: **one GPU pod, RunPod**, rented only for Task 13. Budget cap **$5**, a ceiling not a target, set before the first rental.
- Findings go to `docs/findings/phase-7/`.

## Pre-registered design decisions

Fixed here, carried over from the approved design doc and the API research done while writing this plan -- not re-decided mid-implementation.

- **Single-flight, not multi-request batching.** `KernelResponder` and the router's `AdmissionQueue` both enforce at most one in-flight generation at a time (a lock on the Python side, a capacity-1 semaphore on the Rust side). Phase 7's non-goals explicitly rule out concurrent multi-replica serving; this keeps the implementation honest about that scope rather than building unused concurrency machinery.
- **Router exposes HTTP to clients, gRPC to the model server only.** The design doc's data-flow line ("Client -> router, HTTP or gRPC") is resolved concretely to HTTP-only for the client-facing side, matching TGI's own actual router surface and avoiding a second, unused gRPC-facing listener.
- **`router/` is a combined library+binary Cargo crate** (`src/lib.rs` re-exporting `app`, `client`, `metrics`, `pb`, `queue`; `src/main.rs` as a thin entrypoint), not binary-only -- required so `router/tests/` integration tests (Task 8) can reach the same modules the unit tests use. This is a refinement of the design doc's "Cargo workspace" phrasing; a multi-crate workspace would be unused ceremony for one binary.
- **`protoc` is vendored via the `protoc-bin-vendored` crate** (Rust) and `grpcio-tools`' own bundled `protoc` (Python) -- neither side needs a system `protoc` install, confirmed against this machine's environment check (no system `protoc` found).
- **Real end-to-end correctness (Task 9's `gpu` test) compares `KernelResponder` against a same-session stock generation on the identical prompt**, not a previously-committed Phase 6 reference file -- this matches every earlier phase's actual correctness-gate pattern (Phase 1/3/4/5a/5b all compare against a same-session stock run, never a stale committed file) more closely than the design doc's shorthand wording.
- **The screen recording is a manual capture, the Grafana screenshot is automated.** Reliably automating a multi-window (terminal + browser) screen recording from this environment isn't something to overpromise; the Grafana dashboard screenshot alone (one page, one browser tab) is reliably automatable via browser tooling and stays automated as agreed. Task 13 gives the user the exact commands to run on-camera so the manual recording is clean.

## Cost and stop rules

- **State the live hourly price and get the user's explicit go-ahead before creating the pod (Task 13, Step 1).** Estimated cost is not a measurement.
- **Budget cap: $5.** This is a short demo session (bring up the model server, run the gate test, run the K8s demo, capture evidence), not a sweep -- expect well under an hour of rental.
- **Never leave the pod running across an unbounded wait.** Stop it first, restart on the answer, per CLAUDE.md's cost discipline and its own cited incident.
- **Pull every evidence file off the pod before `stop` or terminate.**
- Develop and test everything (router, proto, model server, Docker, K8s manifests) against `StubResponder` first -- rent only for Task 13.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `proto/dispatch.proto` | shared gRPC contract | 2 |
| `router/Cargo.toml`, `router/build.rs`, `router/src/lib.rs`, `router/src/pb.rs` | Rust crate scaffold, protoc-vendored codegen | 2 |
| `src/dispatch/proto/dispatch_pb2.py`, `dispatch_pb2_grpc.py`, `scripts/gen_proto.py` | Python side of the contract | 3 |
| `src/dispatch/serving/model_server.py`, `scripts/run_model_server.py` | Responder protocol, StubResponder, gRPC servicer, CLI | 3 |
| `router/src/queue.rs` | admission (single-flight semaphore + depth gauge) | 4 |
| `router/src/metrics.rs` | Prometheus registry | 5 |
| `router/src/client.rs`, `router/src/test_support.rs` | tonic client to the model server; test-only mock server helper | 6 |
| `router/src/app.rs`, `router/src/main.rs` | axum HTTP surface, wiring | 7 |
| `router/tests/cross_language_integration.rs` | router <-> real Python stub server, cross-language proof | 8 |
| `src/dispatch/serving/colocated.py` (modify), `scripts/gpu/phase7_kernel_responder.py` | real model server, GPU-only | 9 |
| `docker/router.Dockerfile`, `docker/model-server.Dockerfile`, `docker-compose.yml` | containers, local verification | 10 |
| `k8s/*.yaml`, `scripts/run_kind_demo.sh` | K8s manifests, scripted bring-up, rehearsed against the stub | 11 |
| `Makefile` (modify) | Rust-aware `make check` | 12 |
| `docs/runbooks/phase-7-productionization.md`, `docs/findings/phase-7/` | the paid session and its evidence | 13, 14 |

Tasks 1-12 are local, free, and TDD where code is involved. Task 13 is the paid pod session. Task 14 is findings, docs, and the PR.

---

### Task 1: Verify kind <-> host connectivity, install prerequisites

**Files:**
- Modify: `docs/STATUS.md` (new "Phase 7 progress" section, this task's finding)

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded, verified answer to "what hostname does a pod inside a local `kind` cluster on this machine use to reach a listener on the dev machine's own localhost" -- every later task (11, 13) that writes `k8s/model-server-external.yaml` depends on this exact value.

- [ ] **Step 1: Install `kind` and confirm Docker is running**

```bash
brew install kind
kind version   # expect v0.33.0 or newer
docker info >/dev/null && echo "docker OK"
```

- [ ] **Step 2: Create a throwaway cluster**

```bash
kind create cluster --name connectivity-check
kubectl config use-context kind-connectivity-check
kubectl get nodes
```

- [ ] **Step 3: Start a listener on the dev machine standing in for the future SSH-tunneled model server**

```bash
python3 -m http.server 50051 &
LISTENER_PID=$!
```

- [ ] **Step 4: From inside the cluster, try to reach it via `host.docker.internal`**

```bash
kubectl run connectivity-probe --image=busybox --rm -it --restart=Never -- \
  wget -qO- --timeout=3 http://host.docker.internal:50051/ || echo "FAILED: host.docker.internal"
```

If this succeeds (prints the directory listing HTML `python3 -m http.server` serves), `host.docker.internal` is the answer and Step 5 is skipped.

- [ ] **Step 5 (only if Step 4 failed): try the node's host-gateway alias**

```bash
kubectl get nodes -o jsonpath='{.items[0].status.addresses}'
# then probe the printed InternalIP-style address the same way as Step 4,
# and/or recreate the cluster with a kind config adding
# extraPortMappings / a node-level extraHosts entry for host-gateway,
# per kind's own networking docs -- record exactly which config worked.
```

- [ ] **Step 6: Record the finding and clean up**

Add a new "## Phase 7 progress" section to `docs/STATUS.md` stating: kind version installed, which hostname reached the dev-machine listener (`host.docker.internal` or the recorded alternative from Step 5), and that this value is what `k8s/model-server-external.yaml` (Task 11) hardcodes.

```bash
kill $LISTENER_PID
kind delete cluster --name connectivity-check
```

- [ ] **Step 7: Commit**

```bash
git checkout -b phase-7-productionization
git add docs/STATUS.md
git commit -m "docs: record Phase 7's kind-to-host connectivity finding"
```

---

### Task 2: Shared gRPC contract and Rust crate scaffold

**Files:**
- Create: `proto/dispatch.proto`
- Create: `router/Cargo.toml`, `router/build.rs`, `router/src/lib.rs`, `router/src/pb.rs`, `router/src/main.rs`
- Modify: `.gitignore` (add `/router/target/`)

**Interfaces:**
- Consumes: nothing.
- Produces: `dispatch_router::pb::dispatch_v1::{GenerateRequest, GenerateResponse}` (prost message types) and `dispatch_router::pb::dispatch_v1::model_server_client::ModelServerClient<T>` / `model_server_server::{ModelServer, ModelServerServer}` (tonic-generated client/server), used by every later router task.

- [ ] **Step 1: Install the Rust toolchain**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version && cargo --version
```

- [ ] **Step 2: Write the proto contract**

Create `proto/dispatch.proto`:

```proto
syntax = "proto3";

package dispatch.v1;

// The gRPC contract between the Rust router and the Python model server
// (docs/design/2026-09-20-phase-7-productionization.md SS3-4). Compiled
// by both sides -- tonic-build/prost-build (Rust, router/build.rs) and
// grpcio-tools (Python, scripts/gen_proto.py) -- each vendors its own
// protoc, so no system protoc install is required.
service ModelServer {
  // Streams generated text token-by-token. The final message in the
  // stream always has is_final=true and empty text; every message
  // before it has is_final=false and carries exactly one token's text.
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
}

message GenerateRequest {
  string request_id = 1;
  string prompt = 2;
  uint32 max_new_tokens = 3;
}

message GenerateResponse {
  string request_id = 1;
  string text = 2;
  bool is_final = 3;
  double t_emit_unix = 4;
}
```

- [ ] **Step 3: Scaffold the Cargo crate**

Create `router/Cargo.toml`:

```toml
[package]
name = "dispatch-router"
version = "0.1.0"
edition = "2021"

[dependencies]
tonic = "0.14.6"
prost = "0.14.4"
tokio = { version = "1.53.1", features = ["rt-multi-thread", "macros", "sync", "time"] }
axum = "0.8.9"
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1.0.151"
prometheus = "0.14.0"

[build-dependencies]
tonic-build = "0.14.6"
protoc-bin-vendored = "3.2.0"

[dev-dependencies]
tower = { version = "0.5.3", features = ["util"] }
http-body-util = "0.1.5"
tokio-stream = "0.1.19"
futures-core = "0.3.34"
```

Create `router/build.rs`:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc_path);
    tonic_build::compile_protos("../proto/dispatch.proto")?;
    Ok(())
}
```

Create `router/src/pb.rs`:

```rust
pub mod dispatch_v1 {
    tonic::include_proto!("dispatch.v1");
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::dispatch_v1::GenerateRequest;

    #[test]
    fn generate_request_round_trips_through_prost_encoding() {
        let original = GenerateRequest {
            request_id: "r1".to_string(),
            prompt: "hello".to_string(),
            max_new_tokens: 16,
        };

        let mut buf = Vec::new();
        original.encode(&mut buf).expect("encode");
        let decoded = GenerateRequest::decode(buf.as_slice()).expect("decode");

        assert_eq!(decoded, original);
    }
}
```

Create `router/src/lib.rs`:

```rust
pub mod pb;
```

Create `router/src/main.rs`:

```rust
fn main() {
    println!("dispatch-router (scaffold) -- HTTP/gRPC wiring lands in Task 7");
}
```

- [ ] **Step 4: Add the Rust build directory to `.gitignore`**

In `.gitignore`, add a new section:

```
# Rust
/router/target/
```

- [ ] **Step 5: Build and run the test**

```bash
cd router
cargo build
cargo test
```

Expected: the build succeeds (protoc is fetched via `protoc-bin-vendored`, no system `protoc` needed) and `generate_request_round_trips_through_prost_encoding` passes. If `tonic_build::compile_protos`'s exact method name has moved since this was written, `cargo doc -p tonic-build --open` (against the pinned 0.14.6) shows the current API -- adjust `build.rs` accordingly; this is the one place in this plan where a fast-moving crate's exact surface can't be guaranteed word-for-word, the same class of risk Phase 6's plan flagged for SGLang's API.

- [ ] **Step 6: Commit**

```bash
cd ..
git add proto/dispatch.proto router/Cargo.toml router/Cargo.lock router/build.rs router/src .gitignore
git commit -m "feat: add the shared gRPC contract and Rust router crate scaffold"
```

---

### Task 3: Python side of the contract -- model server skeleton

**Files:**
- Create: `scripts/gen_proto.py`
- Create: `src/dispatch/proto/__init__.py`, `src/dispatch/proto/dispatch_pb2.py`, `src/dispatch/proto/dispatch_pb2_grpc.py` (generated, then committed)
- Create: `src/dispatch/serving/model_server.py`
- Create: `scripts/run_model_server.py`
- Test: `tests/unit/test_model_server.py`, `tests/unit/test_run_model_server.py`
- Modify: `pyproject.toml` (add `grpcio` dependency, `grpcio-tools` dev dependency, mypy override, ruff extend-exclude)

**Interfaces:**
- Consumes: `proto/dispatch.proto` (Task 2).
- Produces: `dispatch.serving.model_server.{Responder, TokenEvent, StubResponder, ModelServerServicer, ServerBusyError, serve}`; `scripts.run_model_server.{build_responder, parse_args, main}`. Task 9 provides a second `Responder` implementation (`KernelResponder`) matching this same protocol; Task 8's Rust integration test spawns `scripts/run_model_server.py --responder stub` as a subprocess.

- [ ] **Step 1: Add Python dependencies**

In `pyproject.toml`'s `dependencies` list, add:

```toml
    "grpcio>=1.84.0",
```

In `[dependency-groups]` `dev`, add:

```toml
    "grpcio-tools>=1.84.0",
```

- [ ] **Step 2: Write the proto codegen script**

Create `scripts/gen_proto.py`:

```python
"""Regenerates the Python gRPC stubs from proto/dispatch.proto. grpc_tools
emits a bare `import dispatch_pb2` in the generated _grpc.py file, which
breaks once the file lives inside the dispatch.proto package (not on
sys.path directly) -- a known grpc_tools limitation, worked around here
by rewriting that one import to a relative one after generation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = REPO_ROOT / "src" / "dispatch" / "proto"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            str(PROTO_DIR / "dispatch.proto"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    grpc_file = OUT_DIR / "dispatch_pb2_grpc.py"
    text = grpc_file.read_text()
    patched = re.sub(
        r"^import dispatch_pb2 as dispatch__pb2$",
        "from . import dispatch_pb2 as dispatch__pb2",
        text,
        flags=re.MULTILINE,
    )
    if patched == text:
        raise RuntimeError(
            "expected bare 'import dispatch_pb2' line not found -- "
            "grpc_tools output format changed since this script was written"
        )
    grpc_file.write_text(patched)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run codegen and commit the generated files**

```bash
uv sync
uv run python scripts/gen_proto.py
```

Expected: `src/dispatch/proto/dispatch_pb2.py` and `dispatch_pb2_grpc.py` now exist, and `dispatch_pb2_grpc.py`'s import line reads `from . import dispatch_pb2 as dispatch__pb2`.

- [ ] **Step 3: Exempt the generated files from ruff and mypy strict**

In `pyproject.toml`'s `[tool.ruff]` section, add:

```toml
extend-exclude = ["src/dispatch/proto/dispatch_pb2.py", "src/dispatch/proto/dispatch_pb2_grpc.py"]
```

In `[tool.mypy]`, add a new override (following the existing `triton`/`deep_ep`/`vllm`/`sglang` pattern):

```toml
# Generated protobuf/grpc code -- untyped by construction, same boundary
# as triton/deep_ep/vllm/sglang above.
[[tool.mypy.overrides]]
module = ["dispatch.proto.*"]
ignore_errors = true
```

- [ ] **Step 4: Write the failing tests**

Create `tests/unit/test_model_server.py`:

```python
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
            yield  # pragma: no cover -- makes this a generator function

    server, port = serve(_BusyResponder(), port=0)  # type: ignore[arg-type]
    try:
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = dispatch_pb2_grpc.ModelServerStub(channel)
        request = dispatch_pb2.GenerateRequest(request_id="r1", prompt="x", max_new_tokens=1)

        with pytest.raises(grpc.RpcError) as exc_info:
            list(stub.Generate(request))

        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        server.stop(grace=None)
```

- [ ] **Step 5: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_model_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'dispatch.serving.model_server'`.

- [ ] **Step 6: Implement `model_server.py`**

Create `src/dispatch/serving/model_server.py`:

```python
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


class ModelServerServicer(dispatch_pb2_grpc.ModelServerServicer):
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
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_model_server.py -v
```

- [ ] **Step 8: Write the CLI, its failing test, then implement it**

Create `tests/unit/test_run_model_server.py`:

```python
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
```

Run it (`uv run pytest tests/unit/test_run_model_server.py -v`) and confirm it fails with `ModuleNotFoundError: No module named 'scripts.run_model_server'`.

Create `scripts/run_model_server.py`:

```python
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
    from scripts.gpu.phase7_kernel_responder import build_kernel_responder  # noqa: PLC0415

    return build_kernel_responder(moe_kernel=args.moe_kernel)


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
```

- [ ] **Step 9: Run the tests, lint, and typecheck**

```bash
uv run pytest tests/unit/test_run_model_server.py -v
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests scripts
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock scripts/gen_proto.py src/dispatch/proto \
  src/dispatch/serving/model_server.py scripts/run_model_server.py \
  tests/unit/test_model_server.py tests/unit/test_run_model_server.py
git commit -m "feat: add the Python model server skeleton (StubResponder, gRPC servicer, CLI)"
```

---

### Task 4: Router admission queue

**Files:**
- Create: `router/src/queue.rs`
- Modify: `router/src/lib.rs` (add `pub mod queue;`)

**Interfaces:**
- Consumes: nothing.
- Produces: `dispatch_router::queue::{AdmissionQueue, AdmissionTicket}` -- `AdmissionQueue::new(capacity: usize)`, `.depth() -> usize`, `async .admit() -> AdmissionTicket<'_>`. Task 7 consumes this directly.

- [ ] **Step 1: Write the failing tests**

Create `router/src/queue.rs`:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Semaphore, SemaphorePermit};

pub struct AdmissionQueue {
    semaphore: Semaphore,
    waiting: AtomicUsize,
}

pub struct AdmissionTicket<'a> {
    _permit: SemaphorePermit<'a>,
}

impl AdmissionQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            semaphore: Semaphore::new(capacity),
            waiting: AtomicUsize::new(0),
        }
    }

    pub fn depth(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    pub async fn admit(&self) -> AdmissionTicket<'_> {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("semaphore is never closed");
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        AdmissionTicket { _permit: permit }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};

    use super::*;

    #[tokio::test]
    async fn admit_serializes_single_capacity_access() {
        let queue = Arc::new(AdmissionQueue::new(1));
        let concurrent = Arc::new(StdAtomicUsize::new(0));
        let max_concurrent = Arc::new(StdAtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let queue = queue.clone();
            let concurrent = concurrent.clone();
            let max_concurrent = max_concurrent.clone();
            handles.push(tokio::spawn(async move {
                let _ticket = queue.admit().await;
                let now = concurrent.fetch_add(1, StdOrdering::SeqCst) + 1;
                max_concurrent.fetch_max(now, StdOrdering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                concurrent.fetch_sub(1, StdOrdering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.expect("task did not panic");
        }

        assert_eq!(max_concurrent.load(StdOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn depth_reflects_waiting_tasks() {
        let queue = Arc::new(AdmissionQueue::new(1));
        let held = queue.admit().await;

        let queue2 = queue.clone();
        let waiter = tokio::spawn(async move {
            let _ticket = queue2.admit().await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(queue.depth(), 1);

        drop(held);
        waiter.await.expect("task did not panic");
    }
}
```

- [ ] **Step 2: Add the module to `lib.rs`**

In `router/src/lib.rs`, add:

```rust
pub mod queue;
```

- [ ] **Step 3: Run the tests**

```bash
cd router && cargo test queue::
```

Expected: both tests pass (they were written alongside the implementation above rather than red-then-green, since the queue's behavior is simple enough to write once correctly -- but confirm by temporarily commenting out the `fetch_sub` in `admit` and re-running to see `depth_reflects_waiting_tasks` fail, then restore it).

- [ ] **Step 4: Lint**

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
cd ..
git add router/src/queue.rs router/src/lib.rs
git commit -m "feat: add the router's single-flight admission queue"
```

---

### Task 5: Router Prometheus metrics

**Files:**
- Create: `router/src/metrics.rs`
- Modify: `router/src/lib.rs` (add `pub mod metrics;`)

**Interfaces:**
- Consumes: nothing.
- Produces: `dispatch_router::metrics::RouterMetrics` with public fields `requests_total: IntCounter`, `ttft_seconds: Histogram`, `inter_token_latency_seconds: Histogram`, `queue_depth: IntGauge`, and `.encode() -> String`. Tasks 6 and 7 consume this.

- [ ] **Step 1: Write the metrics module with its test**

Create `router/src/metrics.rs`:

```rust
use prometheus::{Encoder, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry, TextEncoder};

pub struct RouterMetrics {
    registry: Registry,
    pub requests_total: IntCounter,
    pub ttft_seconds: Histogram,
    pub inter_token_latency_seconds: Histogram,
    pub queue_depth: IntGauge,
}

impl RouterMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounter::with_opts(Opts::new(
            "dispatch_router_requests_total",
            "Total requests admitted by the router",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(requests_total.clone()))
            .expect("first registration of this metric");

        let ttft_seconds = Histogram::with_opts(HistogramOpts::new(
            "dispatch_router_ttft_seconds",
            "Time to first token, seconds",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(ttft_seconds.clone()))
            .expect("first registration of this metric");

        let inter_token_latency_seconds = Histogram::with_opts(HistogramOpts::new(
            "dispatch_router_inter_token_latency_seconds",
            "Gap between consecutive tokens, seconds",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(inter_token_latency_seconds.clone()))
            .expect("first registration of this metric");

        let queue_depth = IntGauge::with_opts(Opts::new(
            "dispatch_router_queue_depth",
            "Requests currently waiting for the single-flight model server",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("first registration of this metric");

        Self {
            registry,
            requests_total,
            ttft_seconds,
            inter_token_latency_seconds,
            queue_depth,
        }
    }

    pub fn encode(&self) -> String {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&metric_families, &mut buffer)
            .expect("prometheus text encoding never fails for valid metric families");
        String::from_utf8(buffer).expect("prometheus TextEncoder always emits valid utf-8")
    }
}

impl Default for RouterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_reports_recorded_values() {
        let metrics = RouterMetrics::new();
        metrics.requests_total.inc();
        metrics.ttft_seconds.observe(0.25);
        metrics.queue_depth.set(3);

        let text = metrics.encode();

        assert!(text.contains("dispatch_router_requests_total 1"));
        assert!(text.contains("dispatch_router_queue_depth 3"));
        assert!(text.contains("dispatch_router_ttft_seconds_sum 0.25"));
    }
}
```

- [ ] **Step 2: Add the module to `lib.rs`**

In `router/src/lib.rs`, add:

```rust
pub mod metrics;
```

- [ ] **Step 3: Run the test, then lint**

```bash
cd router
cargo test metrics::
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd ..
git add router/src/metrics.rs router/src/lib.rs
git commit -m "feat: add the router's Prometheus metrics registry"
```

---

### Task 6: Router gRPC client and test-only mock server

**Files:**
- Create: `router/src/client.rs`, `router/src/test_support.rs`
- Modify: `router/src/lib.rs` (add `pub mod client;` and `#[cfg(test)] pub mod test_support;`)

**Interfaces:**
- Consumes: `dispatch_router::pb::dispatch_v1::*` (Task 2), `dispatch_router::metrics::RouterMetrics` (Task 5).
- Produces: `dispatch_router::client::{ModelServerClient, StreamedGeneration}` -- `ModelServerClient::connect(endpoint: String) -> Result<Self, tonic::transport::Error>`, `async .generate(request_id: &str, prompt: &str, max_new_tokens: u32, metrics: &RouterMetrics) -> Result<StreamedGeneration, tonic::Status>`. `dispatch_router::test_support::spawn_mock_model_server(chunks: Vec<&'static str>) -> String` (test-only). Task 7 and Task 8 consume `ModelServerClient`; Task 7's tests consume `test_support`.

- [ ] **Step 1: Write the test-only mock server helper**

Create `router/src/test_support.rs`:

```rust
use std::pin::Pin;

use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response, Status};

use crate::pb::dispatch_v1::model_server_server::{ModelServer, ModelServerServer};
use crate::pb::dispatch_v1::{GenerateRequest, GenerateResponse};

struct MockModelServer {
    chunks: Vec<&'static str>,
}

#[tonic::async_trait]
impl ModelServer for MockModelServer {
    type GenerateStream =
        Pin<Box<dyn futures_core::Stream<Item = Result<GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        let chunks = self.chunks.clone();
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            for text in chunks {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                let _ = tx
                    .send(Ok(GenerateResponse {
                        request_id: request_id.clone(),
                        text: text.to_string(),
                        is_final: false,
                        t_emit_unix: 0.0,
                    }))
                    .await;
            }
            let _ = tx
                .send(Ok(GenerateResponse {
                    request_id,
                    text: String::new(),
                    is_final: true,
                    t_emit_unix: 0.0,
                }))
                .await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Starts an in-process, real gRPC server implementing ModelServer that
/// streams `chunks` back as separate non-final tokens followed by one
/// final empty message, then returns its `http://host:port` endpoint.
/// Test-only (router/src/lib.rs gates this module behind `#[cfg(test)]`).
pub async fn spawn_mock_model_server(chunks: Vec<&'static str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(ModelServerServer::new(MockModelServer { chunks }))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("mock server exited unexpectedly");
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    format!("http://{addr}")
}
```

- [ ] **Step 2: Write the client with its test**

Create `router/src/client.rs`:

```rust
use std::time::{Duration, Instant};

use tonic::Request;
use tonic::transport::Channel;

use crate::metrics::RouterMetrics;
use crate::pb::dispatch_v1::model_server_client::ModelServerClient as GrpcClient;
use crate::pb::dispatch_v1::GenerateRequest;

pub struct ModelServerClient {
    inner: GrpcClient<Channel>,
}

#[derive(Debug, Clone)]
pub struct StreamedGeneration {
    pub full_text: String,
    pub ttft: Duration,
    pub inter_token_latencies: Vec<Duration>,
}

impl ModelServerClient {
    pub async fn connect(endpoint: String) -> Result<Self, tonic::transport::Error> {
        let inner = GrpcClient::connect(endpoint).await?;
        Ok(Self { inner })
    }

    pub async fn generate(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_new_tokens: u32,
        metrics: &RouterMetrics,
    ) -> Result<StreamedGeneration, tonic::Status> {
        let start = Instant::now();
        let mut stream = self
            .inner
            .generate(Request::new(GenerateRequest {
                request_id: request_id.to_string(),
                prompt: prompt.to_string(),
                max_new_tokens,
            }))
            .await?
            .into_inner();

        let mut full_text = String::new();
        let mut ttft: Option<Duration> = None;
        let mut inter_token_latencies = Vec::new();
        let mut last_token_at = start;

        while let Some(chunk) = stream.message().await? {
            if chunk.is_final {
                break;
            }
            let now = Instant::now();
            match ttft {
                None => {
                    let elapsed = now.duration_since(start);
                    metrics.ttft_seconds.observe(elapsed.as_secs_f64());
                    ttft = Some(elapsed);
                }
                Some(_) => {
                    let gap = now.duration_since(last_token_at);
                    metrics.inter_token_latency_seconds.observe(gap.as_secs_f64());
                    inter_token_latencies.push(gap);
                }
            }
            last_token_at = now;
            full_text.push_str(&chunk.text);
        }

        Ok(StreamedGeneration {
            full_text,
            ttft: ttft.unwrap_or_default(),
            inter_token_latencies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::spawn_mock_model_server;

    #[tokio::test]
    async fn generate_accumulates_text_and_records_latencies() {
        let endpoint = spawn_mock_model_server(vec!["Hello", " world", "!"]).await;
        let mut client = ModelServerClient::connect(endpoint).await.expect("connect");
        let metrics = RouterMetrics::new();

        let result = client
            .generate("req-1", "hi", 3, &metrics)
            .await
            .expect("generate succeeds");

        assert_eq!(result.full_text, "Hello world!");
        assert_eq!(result.inter_token_latencies.len(), 2);
        assert!(metrics.encode().contains("dispatch_router_ttft_seconds_count 1"));
    }
}
```

- [ ] **Step 3: Add the modules to `lib.rs`**

In `router/src/lib.rs`:

```rust
pub mod client;
pub mod metrics;
pub mod pb;
pub mod queue;

#[cfg(test)]
pub mod test_support;
```

- [ ] **Step 4: Run the test, then lint**

```bash
cd router
cargo test client::
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
cd ..
git add router/src/client.rs router/src/test_support.rs router/src/lib.rs router/Cargo.toml router/Cargo.lock
git commit -m "feat: add the router's gRPC client and a test-only mock model server"
```

---

### Task 7: Router HTTP surface and wiring

**Files:**
- Create: `router/src/app.rs`
- Modify: `router/src/main.rs` (full rewrite), `router/src/lib.rs` (add `pub mod app;`)

**Interfaces:**
- Consumes: `dispatch_router::queue::AdmissionQueue`, `dispatch_router::client::ModelServerClient`, `dispatch_router::metrics::RouterMetrics` (Tasks 4-6).
- Produces: `dispatch_router::app::{AppState, GenerateHttpRequest, GenerateHttpResponse, build_app}`. Task 8's integration test and the router binary (`main.rs`) both consume `build_app`.

- [ ] **Step 1: Write `app.rs` with its tests**

Create `router/src/app.rs`:

```rust
use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::client::ModelServerClient;
use crate::metrics::RouterMetrics;
use crate::queue::AdmissionQueue;

#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<AdmissionQueue>,
    pub client: Arc<Mutex<ModelServerClient>>,
    pub metrics: Arc<RouterMetrics>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateHttpRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateHttpResponse {
    pub text: String,
    pub ttft_ms: f64,
    pub inter_token_latencies_ms: Vec<f64>,
}

pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/generate", post(generate))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateHttpResponse>, axum::http::StatusCode> {
    let _ticket = state.queue.admit().await;
    state.metrics.queue_depth.set(state.queue.depth() as i64);
    state.metrics.requests_total.inc();

    let mut client = state.client.lock().await;
    let request_id = request_id();
    let result = client
        .generate(&request_id, &req.prompt, req.max_new_tokens, &state.metrics)
        .await
        .map_err(|_| axum::http::StatusCode::BAD_GATEWAY)?;

    Ok(Json(GenerateHttpResponse {
        text: result.full_text,
        ttft_ms: result.ttft.as_secs_f64() * 1000.0,
        inter_token_latencies_ms: result
            .inter_token_latencies
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect(),
    }))
}

async fn healthz() -> &'static str {
    "ok"
}

/// Startup blocks on a successful connection to the model server (see
/// main.rs) -- by the time this process is serving HTTP at all, it is
/// also ready, so readyz and healthz report the same thing here. Kept
/// as a separate endpoint because K8s conventionally wires liveness and
/// readiness probes to different paths.
async fn readyz() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.encode()
}

fn request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the epoch")
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use http_body_util::BodyExt;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use super::*;
    use crate::client::ModelServerClient;
    use crate::metrics::RouterMetrics;
    use crate::queue::AdmissionQueue;
    use crate::test_support::spawn_mock_model_server;

    async fn test_state(chunks: Vec<&'static str>) -> AppState {
        let endpoint = spawn_mock_model_server(chunks).await;
        let client = ModelServerClient::connect(endpoint).await.expect("connect");
        AppState {
            queue: Arc::new(AdmissionQueue::new(1)),
            client: Arc::new(Mutex::new(client)),
            metrics: Arc::new(RouterMetrics::new()),
        }
    }

    #[tokio::test]
    async fn generate_endpoint_returns_full_text_and_timing() {
        let app = build_app(test_state(vec!["Hello", " world"]).await);

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"prompt":"hi","max_new_tokens":2}"#))
            .expect("build request");

        let response = app.oneshot(request).await.expect("router did not panic");
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response.into_body().collect().await.expect("read body").to_bytes();
        let parsed: GenerateHttpResponse =
            serde_json::from_slice(&body).expect("valid json response");
        assert_eq!(parsed.text, "Hello world");
        assert!(parsed.ttft_ms >= 0.0);
    }

    #[tokio::test]
    async fn healthz_and_readyz_return_ok() {
        for path in ["/healthz", "/readyz"] {
            let app = build_app(test_state(vec!["ok"]).await);
            let request = axum::http::Request::builder()
                .uri(path)
                .body(axum::body::Body::empty())
                .expect("build request");
            let response = app.oneshot(request).await.expect("router did not panic");
            assert_eq!(response.status(), axum::http::StatusCode::OK, "path: {path}");
        }
    }

    #[tokio::test]
    async fn metrics_endpoint_exposes_prometheus_text_format() {
        let app = build_app(test_state(vec!["ok"]).await);

        let request = axum::http::Request::builder()
            .uri("/metrics")
            .body(axum::body::Body::empty())
            .expect("build request");
        let response = app.oneshot(request).await.expect("router did not panic");
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response.into_body().collect().await.expect("read body").to_bytes();
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("dispatch_router_queue_depth"));
    }
}
```

- [ ] **Step 2: Rewrite `main.rs`**

Replace `router/src/main.rs` entirely:

```rust
use std::env;
use std::sync::Arc;

use tokio::sync::Mutex;

use dispatch_router::app::{build_app, AppState};
use dispatch_router::client::ModelServerClient;
use dispatch_router::metrics::RouterMetrics;
use dispatch_router::queue::AdmissionQueue;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_server_endpoint = env::var("MODEL_SERVER_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let listen_addr = env::var("ROUTER_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let queue_capacity: usize = env::var("ROUTER_QUEUE_CAPACITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    println!("connecting to model server at {model_server_endpoint}");
    let client = ModelServerClient::connect(model_server_endpoint).await?;

    let state = AppState {
        queue: Arc::new(AdmissionQueue::new(queue_capacity)),
        client: Arc::new(Mutex::new(client)),
        metrics: Arc::new(RouterMetrics::new()),
    };

    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    println!("dispatch-router listening on {listen_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
```

- [ ] **Step 3: Add the module to `lib.rs`**

In `router/src/lib.rs`, add `pub mod app;` (alongside the existing `client`, `metrics`, `pb`, `queue`).

- [ ] **Step 4: Run the tests, then lint, then a manual smoke run**

```bash
cd router
cargo test app::
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build
```

- [ ] **Step 5: Commit**

```bash
cd ..
git add router/src/app.rs router/src/main.rs router/src/lib.rs
git commit -m "feat: add the router's HTTP surface (generate/healthz/readyz/metrics)"
```

---

### Task 8: Cross-language router integration test

**Files:**
- Create: `router/tests/cross_language_integration.rs`

**Interfaces:**
- Consumes: `dispatch_router::app::{build_app, AppState}`, `dispatch_router::client::ModelServerClient`, `dispatch_router::metrics::RouterMetrics`, `dispatch_router::queue::AdmissionQueue` (Task 7); `scripts/run_model_server.py --responder stub` (Task 3).
- Produces: proof that the Rust client and Python grpcio server actually agree on the wire format -- the one thing Task 6/7's Rust-only mock tests can't show.

- [ ] **Step 1: Write the integration test**

Create `router/tests/cross_language_integration.rs`:

```rust
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use dispatch_router::app::{build_app, AppState};
use dispatch_router::client::ModelServerClient;
use dispatch_router::metrics::RouterMetrics;
use dispatch_router::queue::AdmissionQueue;

const TEST_PORT: u16 = 50099;

struct StubServerGuard(Child);

impl Drop for StubServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("router/ has a parent directory")
        .to_path_buf()
}

fn spawn_stub_server() -> StubServerGuard {
    let child = Command::new("uv")
        .args([
            "run",
            "python",
            "scripts/run_model_server.py",
            "--responder",
            "stub",
            "--port",
            &TEST_PORT.to_string(),
        ])
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn scripts/run_model_server.py -- is uv on PATH?");
    StubServerGuard(child)
}

async fn wait_for_server_ready(endpoint: &str) {
    for _ in 0..50 {
        if ModelServerClient::connect(endpoint.to_string()).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("stub model server never became reachable at {endpoint}");
}

#[tokio::test]
async fn router_streams_real_python_stub_server_responses_over_http() {
    let _guard = spawn_stub_server();
    let endpoint = format!("http://127.0.0.1:{TEST_PORT}");
    wait_for_server_ready(&endpoint).await;

    let client = ModelServerClient::connect(endpoint)
        .await
        .expect("connect to the real Python stub server");
    let state = AppState {
        queue: Arc::new(AdmissionQueue::new(1)),
        client: Arc::new(Mutex::new(client)),
        metrics: Arc::new(RouterMetrics::new()),
    };
    let app = build_app(state);

    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(r#"{"prompt":"hi","max_new_tokens":3}"#))
        .expect("build request");

    use tower::ServiceExt;
    let response = app.oneshot(request).await.expect("router did not panic");
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    use http_body_util::BodyExt;
    let body = response.into_body().collect().await.expect("read body").to_bytes();
    let parsed: dispatch_router::app::GenerateHttpResponse =
        serde_json::from_slice(&body).expect("valid json response");

    // StubResponder's default tokens, from model_server.py.
    assert_eq!(parsed.text, "The quick brown");
}
```

- [ ] **Step 2: Run it**

```bash
cd router
cargo test --test cross_language_integration
```

Expected: the test spawns the real `scripts/run_model_server.py --responder stub` via `uv run`, connects a real tonic client to a real grpcio server, and gets back `"The quick brown"` (the first three of `StubResponder`'s default tokens) over the router's actual HTTP endpoint. If `uv` isn't resolvable from `cargo test`'s environment, the test's own panic message says so explicitly (`spawn_stub_server`'s `.expect(...)`) rather than failing silently.

- [ ] **Step 3: Lint**

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd ..
git add router/tests/cross_language_integration.rs
git commit -m "test: prove the router and the real Python stub server agree on the wire"
```

---

### Task 9: Real KernelResponder (GPU-only)

**Files:**
- Modify: `src/dispatch/serving/colocated.py` (add `snapshot_active_tokens`)
- Create: `scripts/gpu/phase7_kernel_responder.py`
- Test: `tests/unit/test_colocated.py` (add a case), `tests/unit/test_phase7_kernel_responder.py` (`gpu`-marked)

**Interfaces:**
- Consumes: `dispatch.serving.model_server.{Responder, TokenEvent, ServerBusyError}` (Task 3), `dispatch.serving.disaggregated.{Request, PrefillFn, DecodeFn}`, `dispatch.serving.colocated.ColocatedWorker` (Phase 4), `dispatch.benchmark.harness.load_model`, `dispatch.kernels.backends.resolve_backend`, `dispatch.kernels.integration.{fix_rope_inv_freq, patch_moe_infer}` (Phases 0-1).
- Produces: `scripts.gpu.phase7_kernel_responder.{KernelResponder, build_kernel_responder, MODEL_NAME}` -- `build_kernel_responder(*, moe_kernel: str = "naive") -> KernelResponder`, consumed by `scripts/run_model_server.py`'s lazy import (Task 3).

- [ ] **Step 1: Add `snapshot_active_tokens` to `ColocatedWorker`, with its test**

In `tests/unit/test_colocated.py`, add:

```python
def test_snapshot_active_tokens_reports_in_flight_generated_ids_so_far() -> None:
    worker = ColocatedWorker(_fake_prefill_fn, _fake_decode_fn, batch_size=4)
    worker.submit(Request("a", torch.tensor([[1, 2, 3]]), max_new_tokens=3))

    worker.step()  # prefilled and decoded once: 2 of 3 tokens so far, still active

    assert worker.snapshot_active_tokens() == {"a": [100, 101]}

    worker.step()  # completes; no longer active
    assert worker.snapshot_active_tokens() == {}
```

Run it (`uv run pytest tests/unit/test_colocated.py -v`) and confirm it fails with `AttributeError: 'ColocatedWorker' object has no attribute 'snapshot_active_tokens'`.

In `src/dispatch/serving/colocated.py`, add this method to `ColocatedWorker` (after `step`):

```python
    def snapshot_active_tokens(self) -> dict[str, list[int]]:
        """Every currently-active request's full generated-token-id list
        so far, keyed by request_id -- returns copies, never internal
        references. Additive to Phase 4's step()/submit(): step()
        already returns *completed* RequestResults; this exposes the
        still-in-flight ones too, which Phase 7's streaming model server
        needs to emit each token as soon as it's produced rather than
        only at completion.
        """
        return {r.request_id: list(r.generated_token_ids) for r in self._active}
```

Run the test again and confirm it passes.

- [ ] **Step 2: Write `phase7_kernel_responder.py`**

Create `scripts/gpu/phase7_kernel_responder.py`:

```python
"""GPU-only real inference for Phase 7's model server demo: wraps the
real deepseek-ai/deepseek-moe-16b-base model, Phase 6's naive bf16
kernel, and Phase 4's ColocatedWorker into model_server.py's Responder
protocol. Imported lazily by scripts/run_model_server.py only when
--responder kernel is passed -- never imported by CPU tests or CI.

Single-flight only: ColocatedWorker's continuous-batching machinery is
reused, but this responder enforces at most one in-flight stream at a
time (a lock, not a queue) -- Phase 7's design explicitly scopes out
concurrent multi-stream serving (design doc SS6, non-goals).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import torch
from transformers import DynamicCache

from dispatch.benchmark.harness import load_model
from dispatch.kernels.backends import resolve_backend
from dispatch.kernels.integration import fix_rope_inv_freq, patch_moe_infer
from dispatch.serving.colocated import ColocatedWorker
from dispatch.serving.disaggregated import DecodeFn, PrefillFn, Request
from dispatch.serving.model_server import ServerBusyError, TokenEvent

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
DEMO_REQUEST_ID = "demo"


# Same pre-v5 DynamicCache shim Phase 3/4's own GPU scripts use --
# DeepSeek's remote-code modeling file still calls the removed
# get_usable_length even against this repo's transformers>=5.17.0 floor.
def _get_usable_length(
    self: DynamicCache, new_seq_length: int | None = None, layer_idx: int = 0
) -> int:
    return self.get_seq_length(layer_idx)


DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]


def make_prefill_fn(model: torch.nn.Module, device: str) -> PrefillFn:
    def prefill_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=True,
            )
        first_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        cache = DynamicCache.from_legacy_cache(outputs.past_key_values)  # type: ignore[attr-defined]
        return first_tokens.cpu(), cache

    return prefill_fn


def make_decode_fn(model: torch.nn.Module, device: str) -> DecodeFn:
    def decode_fn(
        next_input_ids: torch.Tensor,
        cache: DynamicCache,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        with torch.no_grad():
            outputs = model(
                input_ids=next_input_ids.to(device),
                past_key_values=cache.to_legacy_cache(),  # type: ignore[attr-defined]
                attention_mask=attention_mask.to(device),
                position_ids=position_ids.to(device),
                use_cache=True,
            )
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        updated_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)  # type: ignore[attr-defined]
        return next_tokens.cpu(), updated_cache

    return decode_fn


class KernelResponder:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, *, device: str = "cuda") -> None:
        self.tokenizer = tokenizer
        self._device = device
        self._prefill_fn = make_prefill_fn(model, device)
        self._decode_fn = make_decode_fn(model, device)
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]:
        if not self._lock.acquire(blocking=False):
            raise ServerBusyError("model server is already streaming one request")
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            worker = ColocatedWorker(self._prefill_fn, self._decode_fn, batch_size=1)
            worker.submit(
                Request(
                    request_id=DEMO_REQUEST_ID,
                    prompt_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            )
            emitted = 0
            while True:
                completed = worker.step()
                active = worker.snapshot_active_tokens()
                if DEMO_REQUEST_ID in active:
                    new_ids = active[DEMO_REQUEST_ID][emitted:]
                    for token_id in new_ids:
                        yield TokenEvent(
                            text=self.tokenizer.decode([token_id]),
                            is_final=False,
                            t_emit=time.perf_counter(),
                        )
                    emitted = len(active[DEMO_REQUEST_ID])
                if completed:
                    tail = completed[0].generated_token_ids[emitted:]
                    for token_id in tail:
                        yield TokenEvent(
                            text=self.tokenizer.decode([token_id]),
                            is_final=False,
                            t_emit=time.perf_counter(),
                        )
                    yield TokenEvent(text="", is_final=True, t_emit=time.perf_counter())
                    return
        finally:
            self._lock.release()


def build_kernel_responder(*, moe_kernel: str = "naive") -> KernelResponder:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    fix_rope_inv_freq(model)
    patched = patch_moe_infer(model, resolve_backend(moe_kernel))
    if patched == 0:
        raise RuntimeError(f"patch_moe_infer patched zero layers for backend {moe_kernel!r}")
    return KernelResponder(model, tokenizer, device="cuda")
```

- [ ] **Step 3: Write the `gpu`-marked correctness test**

Create `tests/unit/test_phase7_kernel_responder.py`:

```python
"""Real end-to-end model-server correctness on a real GPU: KernelResponder's
streamed token sequence matches a same-session stock-model greedy
generation on the identical prompt, exactly -- Phase 4's own bar for its
correctness gate, applied here to the streaming responder. Excluded from
CI -- no GPU runner there.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("triton", reason="triton ships Linux wheels only")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from dispatch.benchmark.harness import load_model
from dispatch.kernels.integration import fix_rope_inv_freq
from scripts.gpu.phase7_kernel_responder import MODEL_NAME, build_kernel_responder

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 8


def _stock_greedy_text(prompt: str, max_new_tokens: int) -> str:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    fix_rope_inv_freq(model)
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated: list[int] = []
    past = None
    next_input = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            next_input = next_token
    return str(tokenizer.decode(generated))


def test_kernel_responder_matches_same_session_stock_greedy_text() -> None:
    expected_text = _stock_greedy_text(PROMPT, MAX_NEW_TOKENS)

    responder = build_kernel_responder(moe_kernel="naive")
    events = list(responder.generate(PROMPT, MAX_NEW_TOKENS))
    generated_text = "".join(e.text for e in events if not e.is_final)

    assert generated_text == expected_text
```

- [ ] **Step 4: Lint and typecheck the CPU-reachable parts**

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests scripts
uv run pytest -m "not gpu" -v
```

This test does not run yet (it needs a GPU); Task 13 runs it for real on the rented pod.

- [ ] **Step 5: Commit**

```bash
git add src/dispatch/serving/colocated.py tests/unit/test_colocated.py \
  scripts/gpu/phase7_kernel_responder.py tests/unit/test_phase7_kernel_responder.py
git commit -m "feat: add the real KernelResponder wrapping Phase 4's worker and Phase 6's kernel"
```

---

### Task 10: Dockerfiles and local Compose verification

**Files:**
- Create: `docker/router.Dockerfile`, `docker/model-server.Dockerfile`, `docker-compose.yml`

**Interfaces:**
- Consumes: `router/` (Task 7), `scripts/run_model_server.py` (Task 3).
- Produces: two runnable container images and a Compose file, verified against `StubResponder` end-to-end; Task 11's `kind` rehearsal reuses both Dockerfiles.

- [ ] **Step 1: Write the router Dockerfile**

Create `docker/router.Dockerfile`:

```dockerfile
FROM rust:1.98.1-slim-trixie AS build
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY proto ./proto
COPY router/Cargo.toml router/Cargo.lock ./router/
COPY router/build.rs ./router/build.rs
COPY router/src ./router/src
WORKDIR /build/router
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/router/target/release/dispatch-router /usr/local/bin/dispatch-router
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/dispatch-router"]
```

- [ ] **Step 2: Write the model server Dockerfile**

Create `docker/model-server.Dockerfile`:

```dockerfile
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY proto ./proto
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev
EXPOSE 50051
ENTRYPOINT ["uv", "run", "python", "scripts/run_model_server.py"]
CMD ["--responder", "stub", "--port", "50051"]
```

`CMD` defaults to the stub responder -- local Compose verification (this task) and the `kind` rehearsal (Task 11) never need a GPU. Task 13's real session overrides `CMD` (`docker run ... --responder kernel --moe-kernel naive`) on the rented pod only.

- [ ] **Step 3: Write `docker-compose.yml`**

Create `docker-compose.yml` at the repo root:

```yaml
services:
  model-server:
    build:
      context: .
      dockerfile: docker/model-server.Dockerfile
    command: ["--responder", "stub", "--port", "50051"]
    ports:
      - "50051:50051"

  router:
    build:
      context: .
      dockerfile: docker/router.Dockerfile
    environment:
      MODEL_SERVER_ENDPOINT: "http://model-server:50051"
      ROUTER_LISTEN_ADDR: "0.0.0.0:8080"
      ROUTER_QUEUE_CAPACITY: "1"
    ports:
      - "8080:8080"
    depends_on:
      - model-server
```

- [ ] **Step 4: Build and verify locally**

```bash
docker compose up -d --build
sleep 3
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"hi","max_new_tokens":3}'
```

Expected: a JSON body with `"text":"The quick brown"` (StubResponder's default tokens) and non-negative `ttft_ms`.

```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/metrics | grep dispatch_router
docker compose down
```

- [ ] **Step 5: Commit**

```bash
git add docker/router.Dockerfile docker/model-server.Dockerfile docker-compose.yml
git commit -m "feat: containerize the router and model server, verified via docker compose"
```

---

### Task 11: K8s manifests and scripted kind demo

**Files:**
- Create: `k8s/namespace.yaml`, `k8s/router-deployment.yaml`, `k8s/model-server-external.yaml`, `k8s/prometheus.yaml`, `k8s/grafana.yaml`, `scripts/run_kind_demo.sh`

**Interfaces:**
- Consumes: `docker/router.Dockerfile`, `docker/model-server.Dockerfile` (Task 10); Task 1's verified `host.docker.internal` (or its recorded alternative) finding.
- Produces: a scripted, repeatable `kind` bring-up that Task 13 reuses unmodified, pointing the same `ExternalName` Service at the real SSH tunnel instead of a local stub process.

- [ ] **Step 1: Write the namespace and router manifests**

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dispatch-demo
```

Create `k8s/router-deployment.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: router-config
  namespace: dispatch-demo
data:
  MODEL_SERVER_ENDPOINT: "http://model-server.dispatch-demo.svc.cluster.local:50051"
  ROUTER_LISTEN_ADDR: "0.0.0.0:8080"
  ROUTER_QUEUE_CAPACITY: "1"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: router
  namespace: dispatch-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dispatch-router
  template:
    metadata:
      labels:
        app: dispatch-router
    spec:
      containers:
        - name: router
          image: dispatch-router:demo
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: router-config
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 2
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8080
            initialDelaySeconds: 2
---
apiVersion: v1
kind: Service
metadata:
  name: router
  namespace: dispatch-demo
spec:
  selector:
    app: dispatch-router
  ports:
    - name: http
      port: 8080
      targetPort: 8080
```

- [ ] **Step 2: Write the model-server `ExternalName` Service**

Create `k8s/model-server-external.yaml` -- the hostname below is Task 1's verified finding (`host.docker.internal`, or the recorded alternative if Step 5 of Task 1 was needed):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
  namespace: dispatch-demo
spec:
  type: ExternalName
  externalName: host.docker.internal
  ports:
    - port: 50051
```

Note: `ExternalName` is pure DNS (a CNAME) -- it does not remap ports, so the router still connects on port 50051 directly to whatever `host.docker.internal` resolves to. That real destination is a local stub process in this task's rehearsal, and the SSH-tunneled real model server in Task 13; nothing else in this manifest changes between them.

- [ ] **Step 3: Write the Prometheus manifest**

Create `k8s/prometheus.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: dispatch-demo
data:
  prometheus.yml: |
    global:
      scrape_interval: 5s
    scrape_configs:
      - job_name: dispatch-router
        static_configs:
          - targets: ["router:8080"]
        metrics_path: /metrics
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: dispatch-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus:v3.13.3
          args: ["--config.file=/etc/prometheus/prometheus.yml"]
          ports:
            - containerPort: 9090
          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
      volumes:
        - name: config
          configMap:
            name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: dispatch-demo
spec:
  selector:
    app: prometheus
  ports:
    - port: 9090
      targetPort: 9090
```

- [ ] **Step 4: Write the Grafana manifest with an imported dashboard**

Create `k8s/grafana.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasource
  namespace: dispatch-demo
data:
  datasource.yaml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: dispatch-demo
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
      - name: dispatch
        type: file
        options:
          path: /var/lib/grafana/dashboards
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-dispatch
  namespace: dispatch-demo
data:
  dispatch-router.json: |
    {
      "title": "Dispatch Router - Phase 7 Demo",
      "uid": "dispatch-router-demo",
      "schemaVersion": 39,
      "refresh": "5s",
      "time": { "from": "now-5m", "to": "now" },
      "panels": [
        {
          "id": 1,
          "title": "Requests total",
          "type": "stat",
          "gridPos": { "h": 6, "w": 6, "x": 0, "y": 0 },
          "targets": [{ "expr": "dispatch_router_requests_total", "refId": "A" }]
        },
        {
          "id": 2,
          "title": "Queue depth",
          "type": "timeseries",
          "gridPos": { "h": 8, "w": 9, "x": 6, "y": 0 },
          "targets": [{ "expr": "dispatch_router_queue_depth", "refId": "A" }]
        },
        {
          "id": 3,
          "title": "Mean time to first token (s)",
          "type": "timeseries",
          "gridPos": { "h": 8, "w": 9, "x": 15, "y": 0 },
          "targets": [
            {
              "expr": "dispatch_router_ttft_seconds_sum / dispatch_router_ttft_seconds_count",
              "refId": "A"
            }
          ]
        }
      ]
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: dispatch-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
        - name: grafana
          image: grafana/grafana:13.2.2
          ports:
            - containerPort: 3000
          env:
            # Anonymous admin access is fine here: this cluster is local-
            # only, demoed once, and torn down (design doc SS6/SS7,
            # "no standing public deployment") -- it avoids a login step
            # in the automated screenshot capture (Task 13/14) for a
            # cluster that never leaves the dev machine.
            - name: GF_AUTH_ANONYMOUS_ENABLED
              value: "true"
            - name: GF_AUTH_ANONYMOUS_ORG_ROLE
              value: "Admin"
          volumeMounts:
            - name: datasource
              mountPath: /etc/grafana/provisioning/datasources
            - name: dashboard-provider
              mountPath: /etc/grafana/provisioning/dashboards
            - name: dashboard-dispatch
              mountPath: /var/lib/grafana/dashboards
      volumes:
        - name: datasource
          configMap:
            name: grafana-datasource
        - name: dashboard-provider
          configMap:
            name: grafana-dashboard-provider
        - name: dashboard-dispatch
          configMap:
            name: grafana-dashboard-dispatch
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: dispatch-demo
spec:
  selector:
    app: grafana
  ports:
    - port: 3000
      targetPort: 3000
```

- [ ] **Step 5: Write the demo script**

Create `scripts/run_kind_demo.sh`:

```bash
#!/usr/bin/env bash
# Brings up the K8s side of Phase 7's demo: a kind cluster with the
# router, Prometheus, and Grafana. Assumes something is already
# listening on the dev machine's port 50051 for the model-server
# ExternalName Service to reach -- the stub responder for this task's
# rehearsal (`uv run python scripts/run_model_server.py --responder stub
# --port 50051`, in another terminal), or the real SSH tunnel for
# Task 13's paid session. This script's own job is K8s bring-up only.
set -euo pipefail

CLUSTER_NAME="dispatch-demo"
NAMESPACE="dispatch-demo"

echo "[1/6] Creating kind cluster ${CLUSTER_NAME} (if not already up)"
if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}"
fi
kubectl config use-context "kind-${CLUSTER_NAME}"

echo "[2/6] Building and loading images into the kind cluster"
docker build -t dispatch-router:demo -f docker/router.Dockerfile .
docker build -t dispatch-model-server:demo -f docker/model-server.Dockerfile .
kind load docker-image dispatch-router:demo --name "${CLUSTER_NAME}"
kind load docker-image dispatch-model-server:demo --name "${CLUSTER_NAME}"

echo "[3/6] Applying manifests"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/router-deployment.yaml
kubectl apply -f k8s/model-server-external.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

echo "[4/6] Waiting for rollouts"
kubectl -n "${NAMESPACE}" rollout status deployment/router --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/prometheus --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/grafana --timeout=120s

echo "[5/6] Port-forwarding router (8080) and Grafana (3000) to localhost"
kubectl -n "${NAMESPACE}" port-forward svc/router 8080:8080 &
ROUTER_PF_PID=$!
kubectl -n "${NAMESPACE}" port-forward svc/grafana 3000:3000 &
GRAFANA_PF_PID=$!
trap 'kill ${ROUTER_PF_PID} ${GRAFANA_PF_PID} 2>/dev/null || true' EXIT
sleep 3

echo "[6/6] Ready."
echo "  Demo request: curl -s -X POST localhost:8080/generate -H 'content-type: application/json' -d '{\"prompt\":\"The quick brown fox\",\"max_new_tokens\":16}'"
echo "  Grafana: http://localhost:3000 (anonymous admin)"
echo "Press Ctrl-C to stop the port-forwards. Tear down the cluster separately with:"
echo "  kind delete cluster --name ${CLUSTER_NAME}"
wait
```

```bash
chmod +x scripts/run_kind_demo.sh
```

- [ ] **Step 6: Rehearse the full demo against the stub, no GPU**

In one terminal:

```bash
uv run python scripts/run_model_server.py --responder stub --port 50051
```

In another terminal:

```bash
./scripts/run_kind_demo.sh
```

In a third terminal, once the script prints "Ready.":

```bash
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"The quick brown fox","max_new_tokens":5}'
open http://localhost:3000  # confirm the Dispatch Router dashboard renders and updates
```

Expected: the curl call returns `"The quick brown fox jumps"` and Grafana's "Requests total" stat increments. Tear down:

```bash
kind delete cluster --name dispatch-demo
```

(Ctrl-C the stub model server and the demo script's port-forwards too.)

- [ ] **Step 7: Commit**

```bash
git add k8s/ scripts/run_kind_demo.sh
git commit -m "feat: add K8s manifests and a scripted kind demo, rehearsed against the stub"
```

---

### Task 12: `make check` goes Rust-aware

**Files:**
- Modify: `Makefile`

**Interfaces:**
- Consumes: `router/` (Tasks 2-8).
- Produces: `make router-lint`, `make router-test`, folded into `make check`/`make check-fast` so "green before push" stays one command covering both languages.

- [ ] **Step 1: Add Rust targets**

In `Makefile`, add (near the existing `lint`/`test` targets):

```makefile
router-lint:
	cd router && cargo fmt --check && cargo clippy --all-targets -- -D warnings

router-test:
	cd router && cargo test
```

- [ ] **Step 2: Fold them into `check` and `check-fast`**

Replace the existing `check` target:

```makefile
# The full gate. Runs before every push -- CI runs the same steps.
check:
	@echo "[1/5] lint"
	@$(MAKE) lint
	@echo "[2/5] typecheck"
	@$(MAKE) typecheck
	@echo "[3/5] test"
	@$(MAKE) test
	@echo "[4/5] router-lint"
	@$(MAKE) router-lint
	@echo "[5/5] router-test"
	@$(MAKE) router-test
```

Replace the existing `check-fast` target the same way, substituting `test-fast` for `test` at step 3.

- [ ] **Step 3: Run it**

```bash
make check
```

Expected: all five steps pass.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "build: make check runs the Rust router's fmt/clippy/test gate too"
```

---

### Task 13: Real GPU session -- PAID, needs explicit go-ahead

**Files:**
- Create (pod-local, never committed): none beyond what Task 9 already wrote.
- Produces: raw evidence under `docs/findings/phase-7/` (Grafana screenshot, metrics export, cost record) that Task 14 writes up.

**Interfaces:**
- Consumes: everything from Tasks 1-12, on a clean checkout of `phase-7-productionization`.
- Produces: `POD_ID`, `POD_SSH`, `RATE` (the live $/hr) for Task 14's cost writeup; the Grafana screenshot and the manually-captured screen recording, saved locally.

- [ ] **Step 1: Confirm the live price and get the go-ahead**

Query the live GPU price and availability (RunPod MCP `list-gpu-types` / `get-capacity`, or the console) for an L40-class card, RunPod Secure or Community Cloud per current stock. Tell the user: the GPU id, the hourly rate, the $5 cap, and that this session is expected to take well under an hour. **Do not create the pod until the user says yes.** Record the rate as `RATE`.

- [ ] **Step 2: Create the pod**

```bash
uv run python -m scripts.gpu.provision create --name dispatch-phase-7 \
  --gpu-type "<the GPU id from Step 1>" --image "<current runpod/pytorch tag>" \
  --cloud SECURE --disk-gb 60
uv run python -m scripts.gpu.provision wait --pod-id <pod_id>
```

Note `POD_ID` and `POD_SSH`. Start a wall-clock note (`date -u`).

- [ ] **Step 3: Ship the repo and build the environment**

Follow the established pattern from every prior phase's runbook (clone/ship the repo onto the pod, `uv sync --all-extras --dev`, download the model into a persistent mount, `HF_HOME` set accordingly). Confirm live which of the previously-documented `transformers` remote-code breaks (Phase 0/1/3/4/5a/5b) still reproduce at the pod's installed `transformers` version, and apply the same fixes if so.

- [ ] **Step 4: Re-verify correctness on this GPU**

```bash
.venv/bin/python -m pytest -m gpu tests/unit/test_phase7_kernel_responder.py -v
```

Expected: `test_kernel_responder_matches_same_session_stock_greedy_text` passes. If it doesn't, stop and debug before continuing -- correctness before the demo, same rule as every prior phase.

- [ ] **Step 5: Start the real model server**

```bash
.venv/bin/python scripts/run_model_server.py --responder kernel --moe-kernel naive --port 50051 &
```

- [ ] **Step 6: Open the SSH tunnel from the dev machine**

In a terminal on the dev machine (not the pod):

```bash
ssh -N -L 50051:localhost:50051 <POD_SSH>
```

- [ ] **Step 7: Bring up the K8s demo against the real tunnel**

```bash
./scripts/run_kind_demo.sh
```

The `ExternalName` Service (`k8s/model-server-external.yaml`) is unchanged from Task 11's rehearsal -- it still points at `host.docker.internal`, which now resolves through the SSH tunnel to the real GPU-backed model server instead of the local stub.

- [ ] **Step 8: Start the manual screen recording**

Before firing the demo request, start an OS-level screen recording covering both the terminal (for the `curl` command and its output) and the browser window (for Grafana). On macOS: Cmd+Shift+5, select "Record Selected Portion" or the full screen, start recording. Keep it short -- 15-30 seconds is enough to show the request, the response, and the dashboard updating.

- [ ] **Step 9: Fire the real demo request**

```bash
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"The quick brown fox jumps over the lazy dog.","max_new_tokens":16}'
```

Expected: real generated text (not the stub's canned tokens) and real `ttft_ms`/`inter_token_latencies_ms` values from actual GPU inference.

- [ ] **Step 10: Capture the Grafana screenshot via browser automation**

Wait a few seconds for Prometheus to scrape (5s interval), then use browser automation (Playwright or Chrome DevTools MCP) to navigate to `http://localhost:3000/d/dispatch-router-demo`, wait for the panels to render with non-empty data, and save a screenshot to `docs/findings/phase-7/2026-09-20-phase-7-grafana-dashboard.png`.

- [ ] **Step 11: Stop the screen recording and save it**

Stop the OS-level recording from Step 8. Save it as `docs/findings/phase-7/2026-09-20-phase-7-demo-recording.mov` (or convert to a short `.gif` if the file is large -- keep it well under a few MB for the README embed in Task 14).

- [ ] **Step 12: Export the raw metrics text and record cost**

```bash
curl -s localhost:8080/metrics > /tmp/phase7-metrics.txt
```

Copy `/tmp/phase7-metrics.txt` to `docs/findings/phase-7/2026-09-20-phase-7-metrics.txt` on the dev machine. Query RunPod's billing API for this pod's actual measured cost (Phase 6's established method, not rate x duration).

- [ ] **Step 13: Tear down**

```bash
kind delete cluster --name dispatch-demo
```

Kill the SSH tunnel (Step 6) and the model server (Step 5). Stop and terminate the pod:

```bash
uv run python -m scripts.gpu.provision stop --pod-id <pod_id>
uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>
```

Verify termination independently (pod lookup returns not-found or `TERMINATED`), per this repo's standing convention.

---

### Task 14: Findings, docs, and the PR

**Files:**
- Create: `docs/findings/phase-7/2026-09-20-phase-7-productionization-run.md`, `docs/runbooks/phase-7-productionization.md`
- Modify: `docs/STATUS.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: Task 13's evidence (screenshot, recording, metrics export, cost record).
- Produces: the phase's final write-up and the PR.

- [ ] **Step 1: Write the findings doc**

Create `docs/findings/phase-7/2026-09-20-phase-7-productionization-run.md` covering: what was built, the real demo request and its measured TTFT/inter-token-latency values, the Grafana screenshot (embedded), a link to the screen recording, the measured cost against the $5 cap, and any deviations from the plan discovered while executing Tasks 1-13 (the `host.docker.internal` finding from Task 1, any Rust API-surface adjustments from Task 2's Step 5 note, anything else) -- following this repo's established findings-doc style (see `docs/findings/phase-6/` for the pattern).

- [ ] **Step 2: Write the runbook**

Create `docs/runbooks/phase-7-productionization.md` from what actually worked in Task 13's session (not a re-statement of the plan -- the runbook records the real commands, the real pod id, the real timings), following the pattern of `docs/runbooks/phase-6-final-benchmark.md`.

- [ ] **Step 3: Update `docs/STATUS.md`**

Add a "Phase 7 progress" completion entry (extending the section Task 1 started): all tasks done, `make check` green throughout (both gates), the measured cost, and a note that this closes the system design's phase table (SS7) entirely -- Phase 0 through 7 now all complete.

- [ ] **Step 4: Refresh README and CLAUDE.md**

Update `README.md` with Phase 7's summary and embed the screen recording (or a GIF converted from it). Update `CLAUDE.md`'s "Current status" section with a Phase 7 entry matching the style of Phases 0-6's entries there, and note explicitly that this completes the system design's 8-phase plan (SS7).

- [ ] **Step 5: Final `make check` and commit**

```bash
make check
git add docs/findings/phase-7 docs/runbooks/phase-7-productionization.md docs/STATUS.md README.md CLAUDE.md
git commit -m "docs: record Phase 7 productionization run, close out the system design's phase table"
```

- [ ] **Step 6: Open the PR**

Ask the user before pushing or opening the PR, per this repo's standing convention. On approval:

```bash
git push -u origin phase-7-productionization
gh pr create --title "Phase 7: productionization (Rust router, Docker, K8s, observability)" --body "$(cat <<'EOF'
## Summary
- Rust router (tonic gRPC client, axum HTTP, Prometheus metrics) in front of a Python model server, per ADR-0003's TGI-shaped split
- Real deepseek-ai/deepseek-moe-16b-base inference (Phase 6's naive bf16 kernel, Phase 4's ColocatedWorker) demoed once through a local kind cluster, SSH-tunneled to a rented GPU
- Grafana dashboard showing real TTFT/inter-token-latency/queue-depth from the one real demo request
- Closes the system design's 8-phase plan (SS7) -- Phase 0 through 7 all complete

## Test plan
- [ ] `make check` green (Rust and Python gates)
- [ ] `gpu`-marked correctness test passed on the real rented GPU (Task 13, Step 4)
- [ ] Real demo request produced real generated text and real metrics (Task 13, Step 9)
- [ ] Grafana screenshot and screen recording captured as evidence
EOF
)"
```

