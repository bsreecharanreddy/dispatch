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
from tokenizers import decoders, pre_tokenizers
from transformers import DynamicCache, PreTrainedTokenizerBase

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
            emitted_text = ""
            while True:
                completed = worker.step()
                active = worker.snapshot_active_tokens()
                token_ids = active.get(DEMO_REQUEST_ID)
                if token_ids is None and completed:
                    token_ids = completed[0].generated_token_ids
                if token_ids is not None:
                    # Decoding the whole sequence and diffing against the
                    # previous decode -- not each new token id in
                    # isolation -- because a byte-level BPE tokenizer's
                    # decode() only converts its internal space marker
                    # ('Ġ') into a real space when it has surrounding
                    # context. Found live on the real GPU session: per-
                    # token decode produced 'TheĠquickĠbrown'
                    # instead of 'The quick brown'.
                    full_text = self.tokenizer.decode(token_ids)
                    new_text = full_text[len(emitted_text) :]
                    if new_text:
                        yield TokenEvent(text=new_text, is_final=False, t_emit=time.perf_counter())
                    emitted_text = full_text
                if completed:
                    yield TokenEvent(text="", is_final=True, t_emit=time.perf_counter())
                    return
        finally:
            self._lock.release()


# deepseek-ai/deepseek-moe-16b-base ships a tokenizer.json whose vocab is
# majority GPT-2-style byte-level BPE (47,723 of 100,000 entries carry the
# 'Ġ' space marker) but is wired to a SentencePiece Metaspace
# pre-tokenizer/decoder pair (expects '▁', which appears in zero vocab
# entries). Found live on this GPU session: encode() glues words together
# with no space markers at all ("The quick brown fox" -> a token stream
# indistinguishable from "Thequickbrownfox"), and decode() then either
# drops spaces silently or leaks a literal 'Ġ'/'Ċ' glyph, depending on
# which vocab entries the greedy path happens to hit. This is a defect in
# the model's own shipped tokenizer file, not in transformers or dispatch
# -- Phases 0-6 never noticed because they only ever compared logits/token
# ids, never decoded text for a human to read. Rewiring both the
# pre-tokenizer and decoder to ByteLevel (matching what the vocab actually
# is) round-trips exactly: encode(decode(ids)) == ids, verified against
# several multi-word prompts.
def fix_tokenizer_byte_level(tokenizer: PreTrainedTokenizerBase) -> None:
    backend = tokenizer.backend_tokenizer
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()


def build_kernel_responder(*, moe_kernel: str = "naive") -> KernelResponder:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    fix_tokenizer_byte_level(tokenizer)
    fix_rope_inv_freq(model)
    patched = patch_moe_infer(model, resolve_backend(moe_kernel))
    if patched == 0:
        raise RuntimeError(f"patch_moe_infer patched zero layers for backend {moe_kernel!r}")
    return KernelResponder(model, tokenizer, device="cuda")
