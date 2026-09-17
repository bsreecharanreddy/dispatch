# Phase 5b design: speculative decoding (draft-model + prompt-lookup)

Per `docs/design/2026-09-14-dispatch-system-design.md` §7 (phase plan, row
5: "Quantization + speculative decoding layered on top"), split during
Phase 5a's own brainstorming into two sequential sub-phases. Phase 5a
(int8 weight-only quantization) is complete and merged (PR #5). **This
document covers 5b (speculative decoding) only.**

## 1. Purpose and thesis

Every phase through 5a has measured throughput of a single forward pass
per emitted token. Speculative decoding proposes several tokens cheaply,
then verifies all of them in one forward pass of the (expensive) target
model -- when verification accepts more than one token per target forward
pass, decode throughput rises without changing the target's own
per-token cost.

**Phase 5b's thesis: does layering speculative decoding on top of this
project's own quantized target kernel (Phase 5a) produce a real,
measured throughput win on DeepSeekMoE-16B, and does that hold for both
a real second-model drafter and a model-free n-gram drafter -- on the
same real hardware class this project has used throughout?**

This is explicitly an "existing tooling" phase, unlike Phases 1/5a's new
Triton kernels: the technique itself (Leviathan et al. 2023-style
draft-then-verify, and prompt-lookup decoding) is a solved algorithm.
This project's job is implementing it correctly against its own model,
own kernel, and own harness, and reporting an honestly measured number --
not writing a new kernel for it.

## 2. Scope, decided during brainstorming

- **Two drafters, compared, not one.** A real draft-model path
  (`deepseek-ai/deepseek-llm-7b-base`) and a model-free prompt-lookup
  (n-gram) path. Decided explicitly rather than picking one: they have
  different mechanisms, different costs, and different expected winners,
  and this project's pattern throughout has been "measure the comparison,
  don't assume the winner" (Phase 1's naive-vs-persistent, Phase 4's
  co-located-vs-disaggregated).
- **Target runs Phase 5a's int8-quantized kernel** (`patch_moe_infer_quantized`
  and `resolve_quantized_backend()`, unchanged from 5a), not bf16. Decided
  for a concrete reason, not just reuse: a bf16 target (~32GB) plus a
  bf16 7B draft (~14GB) does not comfortably fit on one 44-48GB GPU;
  freeing ~13.9GB via 5a's kernel (measured: 27.84GiB -> 13.95GiB expert
  weights) is what makes single-GPU feasible. This also ties 5a and 5b
  together the way the original design doc's Phase 5 row intended,
  rather than leaving them as two disconnected techniques.
- **Verification is exact-match against greedy, not sampling-based
  rejection.** This project's harness has always decoded greedy
  (`argmax`, `harness.py`'s `generate_with_timings`); speculative
  decoding's acceptance rule here is "accept a draft token iff it equals
  the target's own greedy argmax at that position." This makes
  speculative decoding's output a deterministic function that must be
  byte-identical to plain sequential greedy decoding -- see §5.
- **Single GPU.** Matches the design doc's Phase 5 cost shape
  ("single/dual GPU") and avoids the cost/complexity of cross-device
  coordination. **Budget cap: $10**, set now, before any rental --
  higher than 5a's $5 because this phase carries two new risks 5a didn't
  (two resident models; a new KV-cache-rollback code path).
- **Draft-model choice verified live, not assumed.** Checked directly
  against Hugging Face before committing to this design:
  `deepseek-moe-16b-base` uses a custom tokenizer (`LlamaTokenizerFast`,
  vocab_size 102400, DeepSeek's own `<|begin/end of sentence|>` special
  tokens). `deepseek-coder-1.3b-base` shares the same tokenizer *class*
  and special-token strings but a **different vocab** (32256) -- not
  usable for direct token-level verification. `deepseek-llm-7b-base` is
  the only standalone DeepSeek model confirmed to share the exact
  102400-vocab tokenizer (plain `LlamaForCausalLM`, no `trust_remote_code`
  needed). No standalone ~1B-class same-vocab model exists in the
  `deepseek-ai` namespace -- the smallest same-vocab dense sibling is 7B.
  This is carried into the design as a named risk (§8), not hidden: a 7B
  dense draft may cost close to, or more than, the target's ~2.8B
  active-per-token MoE compute.
- **Four prompts for this phase's own runs, not three.** The existing
  `DEFAULT_PROMPTS` (pangram, creative-writing opener, bare function
  signature) plus one new repetition-heavy prompt, so prompt-lookup gets
  a genuine chance to find n-gram matches. All four backends measured in
  this phase's session (quantized-target baseline, draft-model
  speculative, prompt-lookup speculative) run on the same four prompts;
  prior phases' three-prompt historical numbers are not recomputed.

## 3. Architecture

A new package, `src/dispatch/speculative/`, alongside `kernels/` and
`serving/`. Left untouched: `moe_forward.py`'s `GroupedMatmul` contract,
`integration.py`'s `patch_moe_infer`/`patch_moe_infer_quantized`, and
`harness.py`'s plain greedy loop (still used for the non-speculative
baseline and as the correctness oracle in §5).

- **`Drafter` (new `Protocol`, `drafters.py`)** -- the pluggable-callable
  pattern this project already uses for `GroupedMatmul`/`MoEInfer`:
  - `propose(token_ids: Tensor, num_tokens: int) -> Tensor` -- given the
    full token sequence so far, return **up to** `num_tokens` proposed
    token ids; a shorter (including empty) return is valid and the loop
    (§4) must handle it as a smaller round, not an error.
  - `on_accepted(accepted_len: int, rejected_len: int) -> None` -- hook
    called after each verification round so a drafter holding its own KV
    cache can roll it back; a no-op default for stateless drafters.
- **`DraftModelDrafter` (new class, `drafters.py`)** -- wraps a loaded
  `deepseek-llm-7b-base` and its own `past_key_values`. `propose` runs
  `num_tokens` steps of its own greedy decode (same pattern as
  `generate_with_timings`'s inner loop, reused rather than
  reimplemented). `on_accepted` calls `self.past_key_values.crop(rejected_len)`.
- **`PromptLookupDrafter` (new class, `drafters.py`)** -- no model, no
  cache. `propose` scans `token_ids` for the longest suffix match of the
  last `ngram_size` tokens against any earlier position in the sequence,
  and returns whatever tokens followed that earlier match, up to
  `num_tokens` -- fewer if the match occurred too close to the current
  end of the sequence for `num_tokens` of them to exist, and an empty
  proposal if no match is found at all (never padded). Both are real,
  expected outcomes on non-repetitive prompts, not errors. `on_accepted`
  is a no-op.
- **`speculative_generate()` (new function, `decode.py`)** -- the one
  shared propose/verify/accept/rollback loop, parameterized by a
  `Drafter`, mirroring the existing project pattern of one shared code
  path for multiple backends (`patch_moe_infer_quantized` sharing
  layer-iteration with `patch_moe_infer`). Writing this once means
  neither drafter duplicates verification or cache-rollback logic.
- **CLI**: new `scripts/run_speculative_bench.py`, reusing `load_model`,
  `resolve_quantized_backend`, `patch_moe_infer_quantized`, and
  `benchmark/metrics.py` from existing code. A new script rather than a
  flag on `run_baseline.py` because the decode loop itself is genuinely
  different (propose/verify/rollback vs. one-token-at-a-time), not a
  drop-in `GroupedMatmul` swap.

## 4. Data flow

1. **Setup (once per run):** load the target (`deepseek-moe-16b-base`,
   patched via `patch_moe_infer_quantized`) and, for the draft-model
   path, `deepseek-llm-7b-base`, both resident on one GPU.
2. **Each speculative round:**
   a. `drafter.propose(token_ids, k)` returns up to `k` candidate tokens.
   b. The target runs **one forward pass** over the candidate tokens
      appended to the sequence (using its existing KV cache from prior
      rounds), producing logits at every candidate position plus one
      bonus position past the last candidate.
   c. Compare the target's greedy argmax at each candidate position to
      the drafter's proposed token; find the longest matching prefix
      (`accepted_len`, `0 <= accepted_len <= k`).
   d. Emit the `accepted_len` matched tokens, plus **one more token**:
      the target's own argmax at the first mismatch position (or, if
      all `k` were accepted, the bonus position's argmax). This is the
      standard speculative-decoding guarantee that every round emits at
      least one token even on a full rejection.
   e. `target.past_key_values.crop(k - accepted_len)` discards the cache
      entries for the rejected candidate positions; `drafter.on_accepted(
      accepted_len, k - accepted_len)` does the same for any drafter
      holding its own cache.
   f. Repeat until `max_new_tokens` emitted or EOS.
3. **Non-speculative baseline (`--drafter none`):** the existing
   `generate_with_timings` loop, run on the target alone, over the new
   four-prompt set -- the reference this phase's speedup claims are
   measured against.

## 5. Correctness -- exact-match, not tolerance-based

Because both the target and (for the draft-model path) the drafter
decode strictly greedily, and the acceptance rule is defined as "accept
iff it equals the target's own greedy argmax," speculative decoding's
output is a deterministic function of the target model alone -- it
**must** produce byte-identical tokens to plain sequential greedy
decoding of the same target. This is a tighter, cheaper-to-check bar
than Phase 3/4's top-k logit agreement (those needed tolerance because
cross-GPU/EP execution order isn't bit-exact; nothing here introduces
that kind of divergence).

- **CPU contract tests, no GPU, no real models:** a toy model plus two
  deterministic stub drafters, `AlwaysCorrectDrafter` (proposes exactly
  what the toy model would produce) and `AlwaysWrongDrafter` (proposes
  tokens guaranteed to mismatch), to force both the full-accept and
  full-reject code paths without depending on a real model's actual
  behavior. Assert: `speculative_generate()`'s output token-for-token
  equals the existing plain greedy loop's output, for both stub drafters
  and for `PromptLookupDrafter` against a constructed repeating token
  sequence; after every round, the target's (and, for `DraftModelDrafter`,
  the drafter's) cache sequence length equals the number of tokens
  emitted so far -- catching a cache-rollback bug as a direct assertion
  failure rather than a silent divergence, the same category of bug
  Phase 5a's OOM was (a real invariant broken silently, not a crash).
- **GPU correctness gate (`gpu`-marked, paid):** both real drafters
  against the real quantized target on rented hardware, exact
  token-match vs. a same-session plain-greedy quantized-target run, at
  all four prompts. A run producing even one diverging token fails the
  gate and refuses to report a benchmark number -- "correctness before
  speed" applied literally here, not by analogy.

## 6. Testing

Extends CLAUDE.md's existing testing table:

| Layer | What's covered |
|---|---|
| Drafter contract (CPU) | `PromptLookupDrafter`: longest-suffix-match correctness on constructed sequences, empty proposal on no match, `ngram_size` edge cases (0, 1, longer than the sequence). `DraftModelDrafter`: cache-length bookkeeping with a stub model. |
| Speculative loop (CPU) | `speculative_generate()` byte-exact vs. plain greedy decode, for `AlwaysCorrectDrafter`/`AlwaysWrongDrafter`/`PromptLookupDrafter` against a toy model; cache-length invariant holds after every round; `k=0`/`k=1` edge cases. |
| End to end (`gpu`, paid) | Exact token-match, both drafters vs. plain-greedy quantized-target baseline, all 4 prompts -- the bar from §5. **Gates the benchmark**: a run with any mismatch refuses to report throughput, same discipline as every prior phase's correctness gate. |
| Benchmarks | Tokens/sec (wall-clock over *accepted* tokens -- the real user-facing rate) and acceptance rate (accepted/proposed) per prompt per backend, since prompt-lookup's payoff is expected to vary sharply across the 4 prompts. Every JSON carries drafter type, `k`, n-gram size, and target backend. A `k` sweep over {1, 2, 4, 8}, same shape as Phase 1's token-count sweep -- a speedup claim isn't quoted without knowing it holds across the relevant range. |

## 7. Non-goals

- **Sampling-based (non-greedy) speculative decoding.** This project's
  harness has always been greedy-only; adding stochastic rejection
  sampling now would be an unrelated scope expansion, not a Phase 5b
  requirement.
- **Combining with multi-GPU EP (Phase 3/4).** Single GPU only, matching
  the design doc's Phase 5 cost shape. A speculative + EP combination is
  a plausible future extension, not required here.
- **Medusa-style multi-head self-speculation, or any third drafter type**
  beyond the two decided in §2.
- **Beating vLLM/SGLang's own speculative-decoding numbers.** Phase 6
  owns that comparison; 5b's claim is limited to this project's own
  kernel/harness, quantized-target-alone vs. +speculative, an internal
  comparison like 5a's bf16-vs-int8 one.
- **A same-vocab draft model smaller than 7B.** Verified live (§2): none
  exists standalone in the `deepseek-ai` namespace at this generation.
  Not revisited unless a future phase finds one or accepts a
  cross-tokenizer ("universal assisted generation"-style) approach.

## 8. Risk, cost, and rollout

- **Cost discipline is part of the result, not just a constraint.** $10
  is a ceiling; the gap between cap and actual spend gets reported the
  way every prior phase has (5a: $1.95 of $5; Phase 3: $13.03 of $25;
  Phase 4: $10.94 of $40).
- **Named risk: the 7B draft may not be meaningfully cheaper than the
  target.** `deepseek-llm-7b-base` is dense 7B; the target activates
  ~2.8B params/token as an MoE. Draft-model speculative decoding's net
  speedup could be small or negative once the draft's own cost is
  counted -- reported honestly either way, the same posture as Phase 1's
  persistent-kernel null result. This is exactly why §2 scoped in
  prompt-lookup as a second, model-free comparison point rather than
  betting the whole phase on one drafter.
- **Named risk: prompt-lookup may show near-zero acceptance even with
  the added 4th prompt**, if that prompt doesn't repeat as cleanly as
  intended. Reported as-is; a workload-appropriateness finding, not a
  failure to hide.
- **Named risk: two real models resident on one GPU is new territory** --
  no prior phase held two separate model loads at once. The GPU-session
  runbook gets an explicit memory-budget checkpoint (measure resident
  memory after each model load, before running anything) rather than
  discovering pressure live the way Phase 5a's OOM was discovered.
- **GPU class**: L40 class (Phase 5a's tier), marketplace/spot, checked
  live against real-time availability at rental time, same practice as
  every prior phase.
- **Session order**: CPU-testable suite green (drafter contracts, the
  speculative loop against stub drafters and a toy model) -> rent ->
  load target, checkpoint memory -> load draft model, checkpoint memory
  again -> GPU correctness gate (§5) -> only if that passes, the
  four-prompt, two-drafter, `k`-swept benchmark -> tear down immediately.
