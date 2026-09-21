# Contributing

This is a solo portfolio project, not one actively seeking outside
contributions — but it's built with real conventions, and this doc exists
so anyone reading the code (or considering a PR) can see how it's meant
to be worked in.

## Getting set up

```bash
uv sync --all-extras --dev   # Python
cd router && cargo build     # Rust router (needs a stable toolchain)
```

## Running the checks

```bash
make check       # the full gate: lint, mypy --strict, pytest, and the
                  # Rust router's own fmt/clippy/test -- same steps CI runs
make check-fast  # inner loop: skips slow-marked tests
make coverage    # pytest --cov, term + xml report
```

`make check` runs before every push in this repo's own workflow, and
matches CI exactly (`.github/workflows/ci.yml`).

## GPU-marked tests

Any test that needs a real CUDA device is marked `gpu` and excluded from
both `make check` and CI — this repo's CI has no GPU runner, and marking
is the explicit, visible boundary of what CI can verify (never a silent
skip). GPU-marked tests are run for real, on rented hardware, once per
phase; each phase's findings doc under
[`docs/findings/`](docs/findings/) records exactly which GPU, for how
long, and at what cost.

## Testing policy

No implementation code is committed without tests. The testing table this
project settled on (from Phase 1's actual kernel work):

| Layer | What must be covered |
|---|---|
| Kernel contract (CPU) | eager backend == `ReferenceMoE`; every row covered by exactly one tile; no tile crosses an expert boundary; experts that receive zero tokens |
| Kernels (`gpu`) | each kernel meets `assert_matches_reference` against an fp32 reference at toy, decode- and prefill-shaped dims, fp16 and bf16; a mutation must turn the suite red |
| End to end (`gpu`, paid) | mutual top-5 logit agreement with a same-session stock run; a kernel run that patches no layers refuses to run |
| Benchmarks | `triton.testing.do_bench`; a backend that disagrees with the eager one is refused, not timed; every JSON carries its full config |

**Correctness before speed** governs all of it: a kernel isn't "fast"
until it's been proven numerically correct against a reference
implementation within a stated tolerance. A kernel that's fast because
it's silently wrong is a bug, not a result.

**Never quote a benchmark number that wasn't measured** on the exact
hardware, at the exact config (batch size, sequence length, quantization
level, warm-up handled), by this repo. A benchmark claim states its
config alongside the number — "40% faster" isn't a result; "40% faster
at batch size 32, seq len 2048, on a single A100 80GB, vLLM 0.x.y as
baseline" is.

## Conventions

- **One branch per phase**, carrying the whole phase, pushed once as a
  single PR.
- **One commit per completed task**, not one bundled commit per phase.
- **`docs/STATUS.md` updates in the same commit as the work it
  describes.**
- Conventional commit prefixes (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `build:`).
- **Commit messages are plain ASCII** — `--`, never an em-dash. Docs use
  `—` freely; the git log does not.
- Every phase gets a `docs/design/` entry (for a new subsystem) or slots
  into the existing [system design](docs/design/2026-09-14-dispatch-system-design.md),
  then a `docs/plans/` implementation plan, written before any code.

## Language and tooling

Python 3.12+ (`uv`, `ruff`, `mypy --strict`, `pytest`), Rust (stable,
`cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`) for the Phase 7
router. GPU provisioning (`scripts/gpu/`) calls RunPod's REST API
directly rather than through Terraform — the actual workflow (rent, run
a benchmark, tear down) fits a script better than Terraform state
management, and RunPod's own Terraform provider is much less mature than
AWS/Azure's.

## Cost discipline

GPU rental is real money. If you're running the paid (`gpu`-marked) test
suite or a benchmark script yourself:

- Use marketplace/spot instances (RunPod Community Cloud, Vast.ai), not
  dedicated on-demand.
- Set a budget cap before the first rental, not after.
- Spin up, run the measured session, capture the evidence, tear down
  immediately — never leave a rented GPU idle between sessions or across
  an unbounded wait.
- Pull every evidence file off the pod before `stop`, not after; a
  stopped pod's disk is not guaranteed to come back.

## Full picture

[`CLAUDE.md`](CLAUDE.md) carries the complete set of engineering
patterns, phase-by-phase history, and the project's own written record of
what it got wrong and fixed along the way — written for an AI coding
assistant working in this repo, but equally readable as the fuller
version of this document.
