FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY proto ./proto
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev
EXPOSE 50051
# --no-dev here too: `uv run` re-syncs by default, and without it, every
# container start re-installs mypy/ruff/pytest into a production image
# (found live: a compose run spent its first several seconds downloading
# and installing dev tools before the server ever bound its port).
ENTRYPOINT ["uv", "run", "--no-dev", "python", "scripts/run_model_server.py"]
CMD ["--responder", "stub", "--port", "50051"]
