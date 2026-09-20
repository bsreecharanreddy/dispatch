from __future__ import annotations

from dispatch.benchmark.gate_prompts import GATE_PROMPTS


def test_gate_prompts_are_sixteen_distinct_nonempty_strings() -> None:
    assert len(GATE_PROMPTS) == 16
    assert len(set(GATE_PROMPTS)) == 16
    assert all(prompt.strip() for prompt in GATE_PROMPTS)


def test_gate_prompts_are_immutable_so_the_trace_cannot_drift_between_runs() -> None:
    assert isinstance(GATE_PROMPTS, tuple)


def test_gate_prompts_are_long_enough_to_reach_the_position_floor() -> None:
    # A cheap lower bound on the real-tokenizer count (measured 1036 positions
    # on 2026-09-19 with deepseek-moe-16b-base's tokenizer; the gate's runtime
    # MIN_GATE_POSITIONS check is the authoritative guard). Each prompt is
    # several dozen characters per expected token even at 3 chars/token.
    assert sum(len(prompt) for prompt in GATE_PROMPTS) / 6 > 500
