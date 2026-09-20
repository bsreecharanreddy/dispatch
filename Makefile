.PHONY: test test-fast lint fmt typecheck check check-fast coverage router-lint router-test

# GPU-dependent tests (anything that needs an actual CUDA device) are marked
# `gpu` and excluded here -- this repo's CI has no GPU runner. They run
# locally or on a rented instance; see docs/findings/ for how those runs are
# recorded.
test:
	uv run pytest -m "not gpu"

# Same selection today; kept as a separate target because it will diverge
# once slower CPU-only tests (numerical-tolerance checks against a reference
# implementation, say) show up and need their own opt-out for the inner loop.
test-fast:
	uv run pytest -m "not gpu and not slow"

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src tests scripts

router-lint:
	cd router && cargo fmt --check && cargo clippy --all-targets -- -D warnings

router-test:
	cd router && cargo test

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

check-fast:
	@echo "[1/5] lint"
	@$(MAKE) lint
	@echo "[2/5] typecheck"
	@$(MAKE) typecheck
	@echo "[3/5] test-fast"
	@$(MAKE) test-fast
	@echo "[4/5] router-lint"
	@$(MAKE) router-lint
	@echo "[5/5] router-test"
	@$(MAKE) router-test

coverage:
	uv run pytest -m "not gpu" --cov --cov-report=term --cov-report=xml
