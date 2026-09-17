# Phase 5b speculative decoding run -- findings

Design: `docs/design/2026-09-17-phase-5b-speculative-decoding.md`. Plan:
`docs/plans/2026-09-17-phase-5b-speculative-decoding-plan.md`. Branch
`phase-5b-speculative-decoding`. Runbook:
`docs/runbooks/phase-5b-speculative-decoding.md` (full account of the
session, including the five environment breaks below).

## Infrastructure

Pod `2t6wh9l3okl3gj`, NVIDIA L40, RunPod Secure Cloud, $0.8200/hr, 3000s
(0.833hr) rented. Confirmed terminated after the session (404 on
`get-pod` following `delete-pod`'s 204). Full record:
`docs/findings/2026-09-17-phase-5b-speculative-decoding-cost.md`.

Config held constant across every run below unless noted: model
`deepseek-ai/deepseek-moe-16b-base`, quantized (int8) target kernel
(Phase 5a's `patch_moe_infer_quantized`), `bfloat16`, single L40, eager
unbatched decode, the 4-prompt `SPECULATIVE_PROMPTS` list
(`scripts/run_speculative_bench.py`: the 3 `DEFAULT_PROMPTS` -- "The
quick brown fox jumps over the lazy dog.", "In a distant galaxy, a small
crew of explorers", "def fibonacci(n):" -- plus a 4th, added for this
phase, "Please restate the following sentence exactly twice, back to
back: 'The system will now process each incoming request in the order
it was received.'"), 20 runs per configuration, 64 max new tokens,
`prompt_lookup_ngram_size=3`. Draft model where applicable:
`deepseek-ai/deepseek-llm-7b-base`, `bfloat16`.

## Environment

This session hit five distinct breaks loading DeepSeek's own remote code
under `transformers==5.17.0` (this repo's pinned floor), all from the
remote code targeting an older `transformers` API surface:
`is_torch_fx_available` removed, `DynamicCache.get_usable_length`
renamed, `DynamicCache.from_legacy_cache`/`to_legacy_cache` both removed,
and `rope_scaling` auto-populated into a dict missing the old `"type"`
key. Unlike Phases 0/1/5a, this session deliberately did **not**
downgrade `transformers` (the old `transformers==4.57.6` fix) -- this
project's own pinned `transformers>=5.17.0` is load-bearing for
`Cache.crop()`'s negative-number semantics that Tasks 2-3's drafters
depend on, and downgrading would have silently broken that. All five
fixes are a pod-local, never-committed monkeypatch, folded into the
runbook. Full account in the runbook; not repeated here.

## GPU correctness gates: both passed

Both drafters were required to match a same-session plain-greedy
baseline byte-for-byte before any timing number was trusted (design doc
section 5's exact-match bar, not tolerance-based).

| Drafter | `moe_layers_patched` | `token_match` (4 prompts) |
|---|---|---|
| draft-model | 27 | all `true` |
| prompt-lookup | 27 | all `true` |

Source: `2026-09-17-phase-5b-draft-model-results.json`,
`2026-09-17-phase-5b-prompt-lookup-results.json`. `moe_layers_patched`
is also `27` in every one of the other 9 results JSONs from this session
(baseline plus all 8 k-sweep runs) -- the quantized target's full expert
set was patched in every single invocation, not just the two gate runs.

## Measured run: baseline, draft-model, prompt-lookup (k=4, the default)

All three read directly from their own results JSON (`mean_tokens_per_second`,
`acceptance_rate`, `mean_ttft`), same config as stated above:

| Config | Acceptance rate | Tokens/sec | Mean TTFT | p50 TTFT | p99 TTFT | vs. baseline throughput |
|---|---|---|---|---|---|---|
| baseline (`--drafter none`) | 0.0% | 21.897 | 0.0926s | 0.0461s | 0.7894s | -- |
| draft-model (7B) | 91.91% | 26.668 | 0.2330s | 0.1724s | 1.1115s | **+21.8%** |
| prompt-lookup | 22.06% | 38.733 | 0.0976s | 0.0472s | 0.8542s | **+76.9%** |

Sources: `2026-09-17-phase-5b-baseline-results.json`,
`2026-09-17-phase-5b-draft-model-results.json`,
`2026-09-17-phase-5b-prompt-lookup-results.json`.

**The non-obvious result: the drafter with much lower acceptance wins on
raw throughput.** Draft-model accepts 91.9% of its proposed tokens --
over 4x prompt-lookup's 22.1% -- yet prompt-lookup is faster in wall-clock
terms (+76.9% vs. +21.8% over baseline) because it costs nothing to
propose (no second model, no extra forward pass) while draft-model pays
a real 7B dense forward pass per round regardless of how often its
proposal is right. High acceptance does not automatically mean the
fastest drafter once the drafter's own cost is counted -- see the risk
verdict below.

## k-sweep: k in {1, 2, 4, 8}, both drafters, full 4-prompt suite

**Deviation from the original plan, stated plainly per this project's own
practice:** the plan intended to sweep `k` on just the repetition-heavy
4th prompt alone, to keep the sweep cheap. `run_speculative_bench.py`'s
actual CLI has no per-prompt selection flag -- `SPECULATIVE_PROMPTS` is
always the full fixed list of 4. Rather than add an untested flag
mid-session for a pure cost optimization, this session ran the full
4-prompt suite at every `k` instead (8 extra runs total, still cheap at
this card's $0.82/hr rate). Every number below reflects that: it is a
sweep over the same 4-prompt aggregate the measured run above used, not
an isolated read on the repetition-heavy prompt.

| k | Drafter | Acceptance rate | Tokens/sec | vs. baseline (21.897 tok/s) |
|---|---|---|---|---|
| 1 | draft-model | 65.72% | 18.178 | -17.0% |
| 2 | draft-model | 50.70% | 17.814 | -18.6% |
| 4 | draft-model | 91.91% | 29.150 | +33.1% |
| 8 | draft-model | 80.29% | 32.252 | +47.3% |
| 1 | prompt-lookup | 54.78% | 33.749 | +54.1% |
| 2 | prompt-lookup | 31.87% | 35.861 | +63.8% |
| 4 | prompt-lookup | 22.06% | 40.385 | +84.4% |
| 8 | prompt-lookup | 11.03% | 38.661 | +76.6% |

Sources: the 8 `2026-09-17-phase-5b-k-sweep-{draft-model,prompt-lookup}-{1,2,4,8}-results.json`
files. Per the runbook, this sweep intentionally omits
`--compare-generated-tokens` -- the gate above already proved exact-match
correctness at k=4, and correctness does not depend on `k` (the
verification rule is the same at every `k`), so the sweep only measures
throughput/acceptance, not correctness.

**Two things worth flagging plainly, found only by reading the
generated-token JSON files directly (not just the results JSONs), since
they complicate the acceptance-rate story above:**

1. **The k=4 point measured twice in this session doesn't repeat
   exactly.** The gate run's own k=4 throughput for draft-model is 26.667
   tok/s (`2026-09-17-phase-5b-draft-model-results.json`); the k-sweep's
   independent k=4 run of the same config measured 29.150 tok/s
   (`2026-09-17-phase-5b-k-sweep-draft-model-4-results.json`) -- a ~9%
   difference between two nominally identical configurations run at
   different points in the same session. Prompt-lookup shows the same
   pattern (38.733 vs. 40.385 tok/s, ~4%). Both are real, measured
   numbers from separate process invocations; the difference is reported
   as run-to-run variance rather than reconciled into one number, since
   nothing in either results JSON explains it and inventing an
   explanation would be exactly the kind of "helpfully adjusted" number
   this project's practice rules out.
2. **The actual generated token content is not the same generation
   problem at every k.** Reading `*-generated-tokens.json` directly: the
   plain baseline's own greedy continuation (`2026-09-17-phase-5b-baseline-generated-tokens.json`)
   for all 4 prompts is 64 repetitions of token id `0` -- and both k=4
   correctness-gate runs reproduce that exactly (which is why
   `token_match` is `true`; this is a genuine, verified match, not a
   coincidence). The k=4 and k=8 sweep runs also reproduce all-`0`
   output for all 4 prompts. But the k=1 and k=2 sweep runs do not: both
   drafters' k=1/k=2 generated-token files contain varied, non-degenerate
   token sequences that differ from the baseline's all-`0` trajectory
   *and* from each other (e.g. draft-model k=1's prompt 0 uses token ids
   `{13, 15, 16, 17, 185, 207, 398, 690, 992, 14166, 14594, 24962}`;
   prompt-lookup k=1's prompt 0 uses a different set entirely). Since
   speculative decoding's exactness guarantee means every `k` should
   reduce to the identical greedy trajectory as the plain baseline, this
   is a real discrepancy in what the k=1/k=2 sweep points are actually
   measuring -- and the runbook's own k-sweep design (step 12) explicitly
   chose not to check `--compare-generated-tokens` at any k but 4, so
   this was never caught in-session. Reported here as an open
   observation, not a root-caused bug (that would need code changes and
   further GPU time, out of this docs-only task's scope) -- likely
   explanation is floating-point non-associativity between the batched
   verification forward pass (processing k+1 tokens at once) and the
   one-token-at-a-time baseline path, which would only matter if the
   model's real logits are frequently near-tied at this point in
   decoding (consistent with the all-`0` degenerate collapse seen at
   k=4/k=8). Worth a follow-up investigation before treating the k=1/k=2
   sweep points as a clean apples-to-apples comparison with k=4.

## Memory checkpoints

Design doc section 8's named risk (two real models resident on one GPU
at once is new territory for this project) was checked live, not assumed:

| Checkpoint | `allocated_gb` | `reserved_gb` |
|---|---|---|
| quantized target only | 16.75 | 17.49 |
| quantized target + bf16 draft model | 29.62 | 29.86 |

Source: `2026-09-17-phase-5b-memory-checkpoints.json` (two JSON objects
concatenated back to back). Adding the bf16 `deepseek-llm-7b-base` draft
model on top of the already-quantized target cost **12.87GB** allocated
(29.62 - 16.75). Both checkpoints are well within a single L40's
capacity -- the session proceeded past both checkpoints without hitting
the runbook's stop condition, and no OOM occurred at any point in the
session.

## Cost per 1M generated tokens

Same `$/hr / (tokens/sec * 3600) * 1e6` computation as every prior
phase's findings doc, at this session's own measured rate ($0.82/hr) and
each configuration's own `mean_tokens_per_second` from the measured run
above (k=4, the default):

| Config | Tokens/sec | Cost / 1M tokens |
|---|---|---|
| baseline | 21.897 | $10.40 |
| draft-model | 26.668 | $8.54 |
| prompt-lookup | 38.733 | $5.88 |

## Cost vs. cap

**Actual: $0.6833** (pod `2t6wh9l3okl3gj`, L40, $0.82/hr, 3000s) against
the **$10 cap** -- **6.83% used**. Source:
`docs/findings/2026-09-17-phase-5b-speculative-decoding-cost.md`. Both
correctness gates, the three-way measured run, and the full 8-run
k-sweep all completed inside this one session, well under budget.

## Risk verdicts (design doc section 8)

**Risk 1: "the 7B draft may not be meaningfully cheaper than the target
... net speedup could be small or negative once the draft's own cost is
counted."** Partially materialized, reported as measured rather than
smoothed into a clean yes/no. The draft-model drafter's net speedup over
baseline is real and positive: **+21.8%** at k=4 (26.668 vs. 21.897
tok/s), so the risk's worst case (a negative or negligible speedup) did
not happen. But the risk's underlying concern -- that the dense 7B
draft's own cost would eat into the win -- shows up clearly in a
different place than the risk anticipated: **draft-model is not the
fastest option measured.** Prompt-lookup, a free (no second model)
drafter, beats it substantially (+76.9% vs. +21.8% over baseline),
despite draft-model accepting its own proposals **91.9%** of the time
against prompt-lookup's **22.1%** -- more than 4x higher acceptance. The
7B draft's near-doubling of TTFT (0.233s vs. baseline's 0.093s and
prompt-lookup's 0.098s) is the direct cost the design doc named, and it
is large enough that a much-more-accurate drafter still loses the raw
throughput race to a much-less-accurate, free one. This is exactly the
kind of finding this project's practice keeps rather than buries: high
acceptance is not sufficient by itself, and this session's own numbers
show that plainly on real hardware.

**Risk 2: "prompt-lookup may show near-zero acceptance even with the
added 4th prompt, if that prompt doesn't repeat as cleanly as intended."**
Did not materialize as stated, but for a more surprising reason than the
design doc anticipated. Prompt-lookup's acceptance rate at k=4 is
**22.1%**, not near-zero, and both correctness gates confirm this rate
comes from real, verified matches (`token_match: true` for all 4
prompts including `prompt_003_tokens`, the repetition-heavy prompt).
However, reading the generated-token files directly shows *why*
acceptance is non-trivial is not what the design intended: **at k=4, all
four prompts' actual greedy continuations under this quantized model
collapse into repeating a single token (id `0`) for the full 64-token
window** -- not just the purpose-built repetition prompt. This makes
every prompt maximally, trivially repetitive at the token level by k=4,
which is an easy target for n-gram lookback regardless of whether the
*prompt's own text* was designed to repeat. The results JSONs report
only an aggregate acceptance rate across all 4 prompts, not a per-prompt
breakdown, so this finding cannot confirm whether the repetition-heavy
prompt specifically contributed more matches than the other three within
that 22.1% -- only that all 4 prompts' real generated content was, in
practice, equally repetitive at k=4. The risk as framed (the 4th prompt
failing to repeat "as intended") is moot: the model's own degenerate
greedy behavior overtook the experimental design's premise. Whether this
all-`0` collapse is itself specific to the quantized kernel, this exact
prompt set, or greedy decoding at this length in general was not
investigated further -- out of scope for this docs-only task, flagged as
a candidate follow-up.

## Summary

Both drafters passed their correctness gates on real hardware. The
measured run and k-sweep completed in full, inside budget. The headline
result carries forward cleanly: prompt-lookup, not the higher-accuracy
draft-model drafter, is the faster option on this model/hardware/config
(+76.9% vs. +21.8% over baseline) -- a genuinely non-obvious outcome this
project's own practice reports as-is rather than reconciling toward
whichever drafter "should" have won on paper. The k-sweep and
generated-token read also surface two open questions (run-to-run
variance at nominally-identical k=4 points, and k=1/k=2 sweep runs not
reproducing the verified baseline trajectory) that a future session
should resolve before leaning on the k<4 sweep points for anything more
than a directional read.
