"""Turns the race driver's per-engine JSON records into comparable rows and a
markdown table. Pure, so the selection rule is unit-tested without a GPU.

Tuning rule, fixed before any run (docs/plans/2026-09-19-phase-6-final-
benchmark-plan.md): every contestant is tuned under *uniform* routing, the
distribution the engines' own tuners use, and then timed under both
distributions. vLLM and SGLang are tuned by their own tuners (one run per
tuning label); dispatch sweeps its tile size `block_m`, and its "tuned" row
is the variant fastest on uniform routing at that token count, applied to the
zipf row too. Its "default" row is `block_m` 16, the kernel's default.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dispatch.benchmark.engines.base import DEFAULT_BLOCK_M, TOKEN_COUNTS

TUNING_DISTRIBUTION = "uniform"


@dataclass(frozen=True)
class RaceRow:
    precision: str
    num_tokens: int
    distribution: str
    engine: str
    tuning: str  # "default" | "tuned"
    variant: str
    mean_ms: float
    p50_ms: float
    p99_ms: float
    tflops: float


def summarize_race(records: Iterable[dict[str, Any]]) -> list[RaceRow]:
    """Refused results are excluded (the driver already exits non-zero on
    them); only measured rows reach a table."""
    rows: list[RaceRow] = []
    for record in records:
        config = record["config"]
        ok = [result for result in record["results"] if result["status"] == "ok"]
        if config["engine"].startswith("dispatch-"):
            rows.extend(_dispatch_rows(config, ok))
        else:
            rows.extend(
                _row(config, result, tuning=config["tuning_label"], variant=result["variant"])
                for result in ok
            )
    return rows


def render_markdown(rows: list[RaceRow]) -> str:
    """One table per (precision, tuning): a row per (tokens, distribution), a
    column per engine, each cell 'mean / p99' in milliseconds."""
    blocks: list[str] = []
    for precision, tuning in sorted({(row.precision, row.tuning) for row in rows}):
        subset = [row for row in rows if (row.precision, row.tuning) == (precision, tuning)]
        engines = sorted({row.engine for row in subset})
        lookup: dict[tuple[int, str, str], RaceRow] = {}
        for row in subset:
            key = (row.num_tokens, row.distribution, row.engine)
            if key in lookup:
                raise ValueError(
                    f"two measurements for {row.engine} at {row.num_tokens} tokens, "
                    f"{row.distribution} routing ({precision}, {tuning}) -- merge input "
                    "has two conflicting race JSONs for the same cell; remove the stale one"
                )
            lookup[key] = row
        keys = sorted(
            {(row.num_tokens, row.distribution) for row in subset},
            key=lambda key: (_token_rank(key[0]), key[1]),
        )
        lines = [
            f"### {precision}, {tuning} (mean / p99 ms per routed-MoE layer call)",
            "",
            "| tokens | routing | " + " | ".join(engines) + " |",
            "|---|---|" + "---|" * len(engines),
        ]
        for tokens, distribution in keys:
            cells = [
                f"{found.mean_ms:.3f} / {found.p99_ms:.3f}"
                if (found := lookup.get((tokens, distribution, engine)))
                else "n/a"
                for engine in engines
            ]
            lines.append(f"| {tokens} | {distribution} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _dispatch_rows(config: dict[str, Any], ok: list[dict[str, Any]]) -> list[RaceRow]:
    rows: list[RaceRow] = []
    for num_tokens in sorted({result["num_tokens"] for result in ok}):
        at_tokens = [result for result in ok if result["num_tokens"] == num_tokens]
        default = [r for r in at_tokens if _block_m(r["variant"]) == DEFAULT_BLOCK_M]
        rows.extend(
            _row(config, result, tuning="default", variant=result["variant"]) for result in default
        )
        tuning_pool = [r for r in at_tokens if r["distribution"] == TUNING_DISTRIBUTION]
        if not tuning_pool:
            continue
        best_variant = min(tuning_pool, key=lambda r: r["mean_ms"])["variant"]
        rows.extend(
            _row(config, result, tuning="tuned", variant=best_variant)
            for result in at_tokens
            if result["variant"] == best_variant
        )
    return rows


def _row(config: dict[str, Any], result: dict[str, Any], *, tuning: str, variant: str) -> RaceRow:
    return RaceRow(
        precision=config["precision"],
        num_tokens=result["num_tokens"],
        distribution=result["distribution"],
        engine=config["engine"],
        tuning=tuning,
        variant=variant,
        mean_ms=result["mean_ms"],
        p50_ms=result["p50_ms"],
        p99_ms=result["p99_ms"],
        tflops=result["tflops"],
    )


def _block_m(variant: str) -> int:
    return int(variant.rsplit("-bm", 1)[1])


def _token_rank(num_tokens: int) -> tuple[int, int]:
    """Registered token counts sort first, in TOKEN_COUNTS order; any others
    sort after, by their own value -- never by a shared placeholder rank,
    which would make their relative order depend on set-iteration order."""
    if num_tokens in TOKEN_COUNTS:
        return (0, TOKEN_COUNTS.index(num_tokens))
    return (1, num_tokens)
