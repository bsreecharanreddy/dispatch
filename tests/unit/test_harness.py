"""Two tiers: a fast fake-model test of the timing loop itself (no network,
no GPU), and a `slow` test against a real tiny public HF model (network,
still no GPU -- CPU inference on a few-KB model is fast).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dispatch.benchmark.harness as harness_module
from dispatch.benchmark.harness import generate_with_timings, load_model

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


class _FakeLoadedModel:
    def to(self, device: str) -> _FakeLoadedModel:
        return self

    def eval(self) -> None:  # stub of nn.Module.eval() (train/eval mode), not the builtin
        pass

    def modules(self) -> Iterator[torch.nn.Module]:
        # A real nn.Module with no rotary-embedding submodules -- lets
        # fix_rope_inv_freq's real implementation run against this fake
        # (finding 0 matches) instead of needing its own mock.
        return iter(())


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
    def to(self, device: str) -> _FakeBatchEncoding:
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


def test_load_model_omits_attn_implementation_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_from_pretrained(model_name: str, **kwargs: object) -> _FakeLoadedModel:
        captured_kwargs.update(kwargs)
        return _FakeLoadedModel()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())

    load_model("some-model")

    assert "attn_implementation" not in captured_kwargs


def test_load_model_forwards_an_explicit_attn_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_from_pretrained(model_name: str, **kwargs: object) -> _FakeLoadedModel:
        captured_kwargs.update(kwargs)
        return _FakeLoadedModel()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())

    load_model("some-model", attn_implementation="sdpa")

    assert captured_kwargs["attn_implementation"] == "sdpa"


def test_load_model_applies_the_rope_fix_to_every_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the final-review finding that the rope fix was
    only wired into scripts/run_speculative_bench.py, leaving every other
    load_model caller (run_baseline.py included) still exposed to the
    uninitialized-inv_freq bug. Spies on fix_rope_inv_freq rather than
    re-testing its own logic (that's test_integration.py's job)."""
    fixed_models: list[object] = []
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _FakeLoadedModel())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(harness_module, "fix_rope_inv_freq", fixed_models.append)

    model, _ = load_model("some-model")

    assert fixed_models == [model]


@pytest.mark.slow
def test_generate_with_timings_against_a_real_tiny_model() -> None:
    model, tokenizer = load_model(TINY_MODEL)

    timing = generate_with_timings(model, tokenizer, "hello world", max_new_tokens=5)

    assert 1 <= timing.generated_token_count <= 5
    assert timing.time_to_first_token >= 0
    assert all(b >= a for a, b in zip(timing.token_times, timing.token_times[1:], strict=False))
