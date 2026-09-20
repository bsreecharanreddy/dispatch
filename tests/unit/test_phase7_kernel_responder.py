"""Real end-to-end model-server correctness on a real GPU: KernelResponder's
streamed token sequence matches a same-session stock-model greedy
generation on the identical prompt, exactly -- Phase 4's own bar for its
correctness gate, applied here to the streaming responder. Excluded from
CI -- no GPU runner there.
"""

from __future__ import annotations

import pytest
import torch
from scripts.gpu.phase7_kernel_responder import MODEL_NAME, build_kernel_responder

from dispatch.benchmark.harness import load_model
from dispatch.kernels.integration import fix_rope_inv_freq

pytestmark = pytest.mark.gpu
pytest.importorskip("triton", reason="triton ships Linux wheels only")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 8


def _stock_greedy_text(prompt: str, max_new_tokens: int) -> str:
    model, tokenizer = load_model(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )
    fix_rope_inv_freq(model)
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated: list[int] = []
    past = None
    next_input = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            next_input = next_token
    return str(tokenizer.decode(generated))


def test_kernel_responder_matches_same_session_stock_greedy_text() -> None:
    expected_text = _stock_greedy_text(PROMPT, MAX_NEW_TOKENS)

    responder = build_kernel_responder(moe_kernel="naive")
    events = list(responder.generate(PROMPT, MAX_NEW_TOKENS))
    generated_text = "".join(e.text for e in events if not e.is_final)

    assert generated_text == expected_text
