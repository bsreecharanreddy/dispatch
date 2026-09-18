"""Loads a causal LM and runs greedy, token-by-token generation, timing
each decode step. Manual loop (not model.generate()) because per-token
timestamps are the whole point -- generate() only returns the final
sequence, not when each token was produced.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dispatch.benchmark.metrics import TokenTimings
from dispatch.kernels.integration import fix_rope_inv_freq


def load_model(
    model_name: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    extra_kwargs = (
        {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=trust_remote_code, **extra_kwargs
    )
    # torch's Module.to() overloads don't resolve a str device; Module.eval() is untyped.
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    # Applied unconditionally, for every caller of load_model (not just
    # scripts/run_speculative_bench.py) -- transformers>=5.17.0 (this
    # project's own pinned floor) leaves DeepSeek's remote code's RoPE
    # inv_freq buffer uninitialized after from_pretrained (see
    # fix_rope_inv_freq's own docstring). A model that doesn't match its
    # duck-typed shape (anything but this bug's exact rotary-embedding
    # class) is left untouched -- harmless and idempotent either way.
    fix_rope_inv_freq(model)
    return model, tokenizer


def generate_with_timings(  # noqa: PLR0913 -- device/clock_fn are what make this testable
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    device: str = "cpu",
    clock_fn: Callable[[], float] = time.perf_counter,
) -> TokenTimings:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_token_count = int(input_ids.shape[-1])
    eos_token_id = tokenizer.eos_token_id

    start_time = clock_fn()
    token_times: list[float] = []
    past_key_values = None
    next_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            token_times.append(clock_fn())
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_input = next_token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return TokenTimings(
        start_time=start_time, token_times=tuple(token_times), prompt_token_count=prompt_token_count
    )
