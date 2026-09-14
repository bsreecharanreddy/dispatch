.PHONY: test test-fast lint fmt typecheck check check-fast coverage

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
	uv run mypy src tests

# The full gate. Runs before every push -- CI runs the same three steps.
check:
	@echo "[1/3] lint"
	@$(MAKE) lint
	@echo "[2/3] typecheck"
	@$(MAKE) typecheck
	@echo "[3/3] test"
	@$(MAKE) test

check-fast:
	@echo "[1/3] lint"
	@$(MAKE) lint
	@echo "[2/3] typecheck"
	@$(MAKE) typecheck
	@echo "[3/3] test-fast"
	@$(MAKE) test-fast

coverage:
	uv run pytest -m "not gpu" --cov --cov-report=term --cov-report=xml
