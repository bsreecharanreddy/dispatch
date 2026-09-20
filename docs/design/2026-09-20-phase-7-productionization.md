# Phase 7 design: productionization (Rust router + Docker + K8s)

Per `docs/design/2026-09-14-dispatch-system-design.md` §7 (phase plan, row
7) and §8 (Phase 7 detail), and `docs/adr/0003-rust-router-over-python-only-server.md`
(the TGI-shaped Rust-router/Python-model-server split, already accepted).
Phases 0-6 are complete and merged; the system design's phase table is now
fully executed except this one. This design fills in what §8 left at
architecture-sketch level: the concrete split of router responsibilities,
the demo topology, and the implementation plan's starting point.

## 1. Purpose and thesis

Every phase through Phase 6 answered a kernel- or engine-level performance
question. This phase answers a different one: **can dispatch's kernel work
be reached as a real, observable, deployable service**, the way a
production inference-engineering team would actually ship it -- not a new
performance claim, and not a rerun of Phase 6's race.

Added specifically because checking real inference-engineer job postings
(national and Atlanta-metro) against the original design found a gap this
project didn't otherwise cover: containerized, production-shaped
deployment with live observability, distinct from a one-shot benchmark
script. ADR-0003 already settled *how* to close that gap (TGI's own
documented three-tier split: Rust router, Python model server, gRPC
between them) and *why Rust* specifically (the router is the "last mile"
layer where Python's GIL and interpreter overhead matter most, per the
research ADR-0003 cites). This design is the *what exactly gets built*.

## 2. Scope, decided during brainstorming

- **The Python model server runs real inference on a real rented GPU**,
  not a stub. Reuses dispatch's naive bf16 kernel -- the exact config
  Phase 6 already used for its own concurrency-1 served row (21.04
  tok/s, `docs/findings/phase-6`), not re-decided from scratch here --
  serving the real `deepseek-ai/deepseek-moe-16b-base`. Consistent with
  CLAUDE.md's "never
  quote a benchmark number that wasn't measured" -- the Grafana dashboard
  this phase produces shows real TTFT/ITL from a real run, not a
  fixed-latency fake. Rejected: a lightweight/stub model, which would
  prove the plumbing but make the observability demo cosmetic.
- **K8s target is a local `kind` cluster**, not a rented cloud control
  plane. Free, reproducible, scriptable teardown -- proves the K8s
  manifests are real and deployable without the cost-discipline overhead
  a cloud cluster would add on top of the GPU rental this phase already
  needs. Matches design doc §8's "K8s demoed once and torn down, not run
  continuously."
- **Router-to-model-server connectivity is an SSH local-forward**, not a
  publicly exposed port on the rented pod. The in-cluster router reaches
  the tunnel endpoint through a Kubernetes `ExternalName` Service pointing
  at `localhost:<forwarded-port>` on the dev machine. This is the same
  SSH-based access pattern Phase 5b and Phase 6 already use to pull
  evidence off a rented pod -- no new public-exposure surface, no new
  auth mechanism to build. Rejected: a public gRPC port on the pod
  (would need real authn/allowlisting to be responsible with a real
  inference server); running the whole stack on the GPU pod itself
  (would need K8s tooling installed fresh on a rented box each session
  instead of persisting locally, and revisits the local-`kind` decision).
- **The router does not reimplement continuous batching.** Phase 4
  (`colocated.py`) already has a tested continuous-batching worker. The
  router owns the client-facing concerns TGI's own router actually owns
  -- HTTP/gRPC ingress, admission queueing, forwarding to the model
  server's streaming `Generate` call, and Prometheus metrics -- while the
  model server keeps the batching loop. Keeps "Rust is scoped to the
  router only" (ADR-0003) from quietly expanding into a second batching
  implementation.
- **Plain K8s manifests, not Helm.** No Helm is installed in this
  environment, and this project already prefers lightweight, direct
  tooling over heavier IaC for infrastructure it doesn't need to
  templatize (see CLAUDE.md's reasoning for RunPod scripts over
  Terraform, same logic applied here: one router, one model-server
  `ExternalName`, one Prometheus, one Grafana -- nothing that benefits
  from a chart's parameterization).
- **$5 GPU cap.** Smaller than every prior phase's cap because this phase
  doesn't run a new benchmark sweep -- it runs one real model server for
  long enough to demo a live request through the full stack and capture
  metrics, matching Phase 5a's actual cost class.
- **Screenshot capture is automated.** Once Grafana is port-forwarded
  locally, a browser-automation pass (Playwright/Chrome DevTools)
  navigates to the dashboard and saves the screenshot into
  `docs/findings/phase-7/` directly, rather than a manual step.
- **No standing public deployment -- considered and explicitly declined.**
  A real GPU-backed public endpoint would cost real money continuously
  (an L40/A40-class GPU left running prices out around $500-600/month)
  and directly contradicts this project's own cost-discipline rules
  (CLAUDE.md: "never leave a rented GPU idle," "budget cap set before the
  first rental") that every prior phase followed. In place of a live URL,
  the recruiter-facing artifact is a **short screen recording** of the
  live `kind` demo (terminal bring-up, then Grafana panels populating
  with real request traffic) captured alongside the screenshot and
  embedded in the README -- free to produce, no standing infrastructure,
  and no honesty-labeling burden (an always-on public endpoint that
  wasn't actually backed by the real model at click-time would need
  careful labeling to not misrepresent what's live, which this project's
  "never quote a number that wasn't measured" ethos leans hard against).

## 3. Components

```text
proto/dispatch.proto                shared gRPC contract: streaming Generate
                                     RPC (prompt in; per-token {text, t_first,
                                     t_emit} out). Compiled by both sides --
                                     tonic-build/prost-build (Rust) and
                                     grpcio-tools (Python) each vendor their
                                     own protoc, no new system dependency.

router/                             new Cargo workspace
  src/main.rs                        tonic gRPC + axum HTTP entrypoints
  src/queue.rs                       FIFO admission queue, backpressure
  src/client.rs                      tonic client -> model server's Generate
  src/metrics.rs                     Prometheus registry: TTFT, inter-token
                                     latency, queue depth, requests/sec
  src/health.rs                      /healthz, /readyz for K8s probes

src/dispatch/serving/model_server.py  new: gRPC server adapting Phase 4's
                                     colocated continuous-batching worker
                                     + Phase 6's chosen kernel to the shared
                                     proto's Generate RPC

docker/router.Dockerfile            multi-stage Rust build -> slim runtime
docker/model-server.Dockerfile      Python + CUDA base, installs dispatch
docker-compose.yml                  router + model server, local
                                     verification before any K8s step

k8s/
  namespace.yaml
  router-deployment.yaml            Deployment + Service + ConfigMap
  model-server-external.yaml        ExternalName Service -> SSH-tunnel
                                     endpoint (host.docker.internal or the
                                     kind node's host-gateway, checked live)
  prometheus.yaml                   minimal Deployment + ConfigMap (scrape
                                     config pointed at the router's /metrics)
  grafana.yaml                      Deployment + one imported dashboard JSON

scripts/run_kind_demo.sh            scripted bring-up: kind create, apply
                                     manifests, open the SSH tunnel, port-
                                     forward Grafana, fire one demo request,
                                     scripted teardown

docs/findings/phase-7/              demo evidence: metrics JSON/exported
                                     Grafana panel data, the screenshot,
                                     the screen recording, cost record
```

## 4. Data flow

Client -> router (`Generate`, HTTP or gRPC) -> admission queue -> router's
tonic client streams to the tunneled model server -> model server runs
continuous-batched decode on the real GPU using Phase 6's chosen kernel ->
tokens stream back through the router, each hop timestamped -> router
records TTFT/inter-token-latency/queue-depth into its Prometheus registry
-> Prometheus scrapes `/metrics` -> Grafana dashboard (imported once)
renders it live during the demo window -> one screenshot captured as
evidence.

## 5. Testing

Extends the repo's existing table (CLAUDE.md, "Testing policy") across the
new language boundary:

| Layer | What must be covered | Runs in CI? |
|---|---|---|
| Router unit (`cargo test`) | queue admission/ordering under backpressure; metrics recorded correctly per request; proto (de)serialization round-trips | yes |
| Router integration | router's tonic client against a fake in-process Python `Generate` server (grpcio test server, no real weights, no GPU) -- proves the gRPC wiring itself | yes |
| Model server unit (pytest) | the proto adapter around `colocated.py`'s already-tested loop; a malformed request is refused, not silently dropped | yes |
| Real end-to-end (`gpu`-marked, paid) | router -> SSH tunnel -> real model server -> real GPU -> one real streamed response, checked against a known-good token sequence from Phase 6's reference | no -- pod-only, excluded from CI like every other phase's `gpu` tests, reason visible in the test |
| K8s demo | scripted `kind` bring-up (`scripts/run_kind_demo.sh`), port-forward, one live request through the full stack, Grafana screenshot as evidence | manual/scripted, once, not part of the automated suite |

`make check` gains a Rust-aware counterpart (`cargo test` and `cargo
clippy` invoked alongside the existing lint/typecheck/test gate) so "green
before push" stays one rule covering both languages, not two separate
gates to remember.

## 6. Non-goals

- **A second benchmark race.** Phase 6 already answered the
  dispatch-vs-vLLM-vs-SGLang performance question. This phase's metrics
  are operational telemetry from one live demo run, not a new comparative
  claim.
- **Autoscaling, multi-replica routing, or a real load balancer.** One
  router replica, one model server, demoed once. Production-shaped
  plumbing, not a production-scale deployment.
- **A public-facing deployment.** Considered explicitly during
  brainstorming and declined -- see §2's "no standing public deployment"
  entry for the cost and honesty-labeling reasoning. The SSH-tunnel
  topology is deliberately local-demo-only; no cloud ingress, no TLS
  termination, no auth layer beyond what the tunnel itself provides. The
  recorded demo (§2, §7) is the recruiter-facing artifact instead.
- **Rewriting the model server's inference path in Rust.** ADR-0003
  already ruled this out; Rust stays scoped to the router.
- **A continuously running K8s deployment.** `kind` is created, demoed,
  and torn down in one scripted session, per design doc §8.

## 7. Risk, cost, and rollout

**Cost.** One rented GPU pod (RunPod, L40-class, checked live at rental
time), just long enough to bring up the model server and run the K8s demo
against it. **Budget cap: $5**, set before the first rental -- smaller
than every prior phase because this is a demo session, not a sweep.

**Cost discipline, from CLAUDE.md, applied to this phase:**

- Develop and test the router, the proto, and the K8s manifests entirely
  against the fake in-process Python server first -- rent only for the
  final real-GPU demo session.
- Open the SSH tunnel and run the full demo (including the Grafana
  screenshot capture) in one continuous session; never leave the pod
  idle waiting on a decision mid-session.
- Pull the demo evidence (metrics export, screenshot, cost record) before
  `stop`, not after -- same rule Phase 5b's incident established.

**Risks:**

- **Rust is new tooling for this environment** (no `cargo`/`rustc`
  installed). Budgeted as real setup time in the implementation plan, not
  assumed free -- per ADR-0003's own "Consequences" section.
- **`kind`'s network path to a host-machine SSH tunnel is environment-
  specific** (Docker Desktop's `host.docker.internal` vs. a Linux kind
  node's host-gateway alias). The plan's first task verifies this
  concretely on this machine before any other router/K8s work depends on
  it, rather than assuming a document's example works unmodified.
- **Streaming gRPC across the tunnel adds a real latency hop** the
  dashboard will show honestly (SSH forwarding overhead on top of the
  model's own TTFT). Reported as part of the topology, not hidden or
  subtracted out.
- **Phase 6's chosen kernel may need re-verifying on whatever GPU this
  phase actually rents** (a different L40 instance, possibly a different
  driver). The real end-to-end `gpu` test re-checks this before the demo
  runs, the same correctness-before-speed rule every prior phase followed.

**Rollout.** Branch `phase-7-productionization`, one commit per plan task,
`docs/STATUS.md` updated in the same commit as each piece of work, one PR
at the end. Findings, the Grafana screenshot, and the demo screen recording
in `docs/findings/phase-7/`. README, CLAUDE.md and the story-bank gist
refreshed at completion -- the README refresh specifically embeds the demo
recording, since it's this phase's main recruiter-facing artifact -- and
since this closes the system design's phase table (§7) entirely, that
completion note goes in CLAUDE.md's "Current status" too.
