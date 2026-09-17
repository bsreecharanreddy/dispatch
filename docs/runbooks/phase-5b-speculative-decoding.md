# Runbook: Phase 5b speculative decoding (rented GPU)

One combined session -- correctness gate, then the measured run and a
targeted k-sweep -- per
docs/design/2026-09-17-phase-5b-speculative-decoding.md section 8.
Budget cap: $10, a ceiling not a target.

1. Pick the cheapest L40-class card (Phase 5a's tier) with enough memory
   for the quantized target plus the bf16 draft model resident together
   (design doc section 2: quantizing the target frees ~13.9GB, per Phase
   5a's own measurement, specifically to make room for this), checked
   live against real-time availability.
2. Create and wait, with a 60GB+ volume (Phase 0's real trap: the target
   model alone is 32.8GB and this session also downloads a 7B draft
   model):

       uv run python -m scripts.gpu.provision create --name dispatch-phase-5b \
         --gpu-type "<id from step 1>" --image "<current runpod/pytorch tag>" \
         --cloud SECURE --disk-gb 70
       uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

3. Confirm the card:

       nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

4. Transfer and set up. **2026-09-17 session's actual finding: do NOT
   downgrade `transformers` globally.** This project's own pinned
   `transformers>=5.17.0` is load-bearing for `Cache.crop()`'s
   negative-number "remove N tokens" semantics (Tasks 2-3's own fix
   rounds depend on it); DeepSeek's remote code was written against a
   much older `transformers` API surface, and downgrading to fix it (the
   old `transformers==4.57.6` approach from Phase 0/1/5a) would silently
   break this phase's own cache-rollback correctness instead. Keep the
   pinned version and monkeypatch only the specific broken symbols, via
   the pod-local, never-committed wrapper below.

   If the ssh proxy used to reach the pod requires a PTY for command
   execution and rejects SFTP/`scp` (both true for RunPod's
   `ssh.runpod.io` proxy this session), transfer the repo as base64 text
   piped through an interactive `ssh -tt` session instead -- wrap the
   base64 output at a short line length (e.g. 76 chars/line); an
   unwrapped multi-megabyte single line risks silent truncation by the
   PTY's line-discipline buffer (~4096 bytes/line on Linux):

       cd /local/repo && git archive HEAD -o repo.tar
       base64 -b 76 -i repo.tar -o repo.tar.b64   # macOS; use -w 76 on GNU base64
       { echo "mkdir -p /root/dispatch"; echo "cat > /root/dispatch.tar.b64 <<'B64EOF'"; \
         cat repo.tar.b64; echo "B64EOF"; \
         echo "base64 -d < /root/dispatch.tar.b64 > /root/dispatch.tar"; \
         echo "tar -xf /root/dispatch.tar -C /root/dispatch"; echo "exit"; } \
         > payload.txt
       ssh -tt <pod-proxy-user>@ssh.runpod.io < payload.txt

   Then set up the environment:

       ssh -tt <pod-proxy-user>@ssh.runpod.io
       cd /root/dispatch
       mkdir -p /workspace/hf_cache   # confirm /workspace exists first --
                                       # it may just be a plain directory on
                                       # the container's own disk, not a
                                       # separate persistent mount, if the
                                       # pod was created without one
       export HF_HOME=/workspace/hf_cache
       export HF_HUB_ENABLE_HF_TRANSFER=1
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv pip install hf_transfer   # uv sync alone won't pull this in

   **Confirm live whether DeepSeek's `modeling_deepseek.py` still needs
   patching** (check
   https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/commits/main
   since this session) -- this session hit FIVE distinct breaks loading
   `deepseek-ai/deepseek-moe-16b-base` under `transformers==5.17.0`
   (`trust_remote_code=True`), all from the remote code targeting an
   older `transformers` API surface than what's pinned:

   1. `ImportError: cannot import name 'is_torch_fx_available'` --
      removed from `transformers.utils.import_utils`. Only guards an
      optional `torch.fx.wrap` decoration on one helper (not used in
      eager inference); safe to stub to `False`.
   2. `DynamicCache` has no `get_usable_length` (Phase 0/1/5a's own
      already-known trap, still present).
   3. `DynamicCache` has no `from_legacy_cache` -- only ever called with
      `None` in this project's own call pattern (guarded by
      `not isinstance(past_key_values, Cache)` in the remote code), so a
      minimal stub covers it.
   4. `DynamicCache` has no `to_legacy_cache` either -- the same
      `use_legacy_cache` branch converts the OUTPUT cache back to a
      legacy tuple, which would silently strip this project's own
      required `.crop()` method from `outputs.past_key_values`. Stub as
      an **identity function** (`lambda self: self`), not a real legacy
      conversion -- this project always wants a real `Cache` object back.
   5. `KeyError: 'type'` in `_init_rope` at
      `self.config.rope_scaling["type"]`, even though the model's own
      `config.json` says `"rope_scaling": null` -- `transformers==5.17.0`
      auto-populates `rope_scaling` into a dict without the old `"type"`
      key by the time the remote code reads it. This one isn't a missing
      *symbol* (nothing to monkeypatch on import), so it needs a direct,
      pod-local edit to the cached remote-code file itself -- find the
      real imported copy first (`transformers`' dynamic-module cache
      copies the hub snapshot into a hyphen-escaped path under
      `$HF_HOME/modules/transformers_modules/...`; a glob on the
      original repo name like `*deepseek-moe-16b*` matches the WRONG
      copy, the raw `hub/snapshots/...` one that isn't actually
      imported):

          FILE=$(python3 -c "
          import transformers.utils.import_utils as _iu
          _iu.is_torch_fx_available = lambda: False
          from transformers import AutoConfig
          # trigger the dynamic module download/cache without instantiating a model
          AutoConfig.from_pretrained('deepseek-ai/deepseek-moe-16b-base', trust_remote_code=True)
          " 2>/dev/null; find "\$HF_HOME/modules/transformers_modules" -iname "modeling_deepseek.py")
          sed -i 's/if self.config.rope_scaling is None:/if not self.config.rope_scaling or "type" not in self.config.rope_scaling:/' "\$FILE"

   Combine fixes 1-4 into one pod-local, never-committed wrapper (this
   session's actual working version, reused unmodified across every
   subsequent CLI invocation this session):

       cat > _patch_and_run.py <<'EOF'
       import sys

       import transformers.utils.import_utils as _iu

       _iu.is_torch_fx_available = lambda: False

       from transformers.cache_utils import DynamicCache


       def _get_usable_length(self, new_seq_length=None, layer_idx=0):
           return self.get_seq_length(layer_idx)


       DynamicCache.get_usable_length = _get_usable_length


       @classmethod
       def _from_legacy_cache(cls, past_key_values=None):
           if past_key_values is None:
               return cls()
           if isinstance(past_key_values, cls):
               return past_key_values
           raise NotImplementedError(
               "pod-local shim: only None/DynamicCache inputs are supported here"
           )


       DynamicCache.from_legacy_cache = _from_legacy_cache

       # Identity, not a real legacy conversion -- see finding 4 above.
       DynamicCache.to_legacy_cache = lambda self: self

       from scripts.run_speculative_bench import main

       main(sys.argv[1:])
       EOF

   From here on invoke `.venv/bin/python` directly (`uv run` re-syncs to
   `uv.lock` on every call and silently undoes any environment override
   -- there isn't one left after this session's fix, since `transformers`
   stays pinned, but this still matters if a future session needs its own
   temporary override). If none of the five breaks reproduce live,
   invoke `scripts/run_speculative_bench.py` directly instead of
   `_patch_and_run.py` in every command below.

5. Full CPU-testable suite, one more time, on the pod itself:

       .venv/bin/python -m pytest -m "not gpu" -v

   Expected: all pass (confirms the pod's environment doesn't disagree
   with what already passed locally, including the `slow` real-tiny-model
   tests from Tasks 3 and 5).

6. **Memory checkpoint after loading the target alone** (design doc
   section 8's named risk: two real models resident on one GPU is new
   territory for this project):

       DATE=$(date +%Y-%m-%d)
       .venv/bin/python <<'EOF'
       import json
       import torch

       from dispatch.benchmark.harness import load_model
       from dispatch.kernels.backends import resolve_quantized_backend
       from dispatch.kernels.integration import patch_moe_infer_quantized

       model, _ = load_model(
           "deepseek-ai/deepseek-moe-16b-base",
           device="cuda", dtype=torch.bfloat16, trust_remote_code=True,
       )
       patched = patch_moe_infer_quantized(model, resolve_quantized_backend())
       checkpoint = {
           "after": "quantized target only",
           "moe_layers_patched": patched,
           "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
           "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
       }
       print(json.dumps(checkpoint, indent=2))
       with open("target_checkpoint.json", "w") as f:
           json.dump(checkpoint, f)
       EOF

   If `allocated_gb` already leaves no plausible room for a 7B bf16 draft
   (~14GB) under the card's total memory (from step 3), stop and report
   this as a named finding -- do not proceed to load the draft model on a
   card that clearly can't hold both; downsize the target's dtype further
   or move to a card with more memory instead of discovering an OOM live.

7. **Memory checkpoint after also loading the draft model:**

       .venv/bin/python <<'EOF'
       import json
       import torch

       from dispatch.benchmark.harness import load_model

       # Continues from step 6's session if run in the same process (a
       # notebook/REPL); otherwise re-run step 6's loading first, then:
       draft_model, _ = load_model(
           "deepseek-ai/deepseek-llm-7b-base", device="cuda", dtype=torch.bfloat16,
       )
       checkpoint = {
           "after": "quantized target + bf16 draft model",
           "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
           "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
       }
       print(json.dumps(checkpoint, indent=2))
       with open("both_checkpoint.json", "w") as f:
           json.dump(checkpoint, f)
       EOF

       cat target_checkpoint.json both_checkpoint.json > \
         docs/findings/$DATE-phase-5b-memory-checkpoints.json

   Anything that OOMs here: stop, report exactly what was measured before
   the failure (same discipline as Phase 5a's own mid-session OOM
   finding), and decide live whether to move to a larger card within the
   $10 cap before continuing.

8. **The baseline run** (`--drafter none`), this session's own reference:

       .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
         --drafter none \
         --run-label $DATE-phase-5b-baseline

   Must show `"moe_layers_patched": 27`.

9. **Correctness gate -- draft-model drafter, exact match against the
   baseline. Must pass before any timing is trusted.**

       .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
         --drafter draft-model \
         --compare-generated-tokens docs/findings/$DATE-phase-5b-baseline-generated-tokens.json \
         --run-label $DATE-phase-5b-draft-model

   Must show `"moe_layers_patched": 27` and every prompt's `token_match`
   `true`. Any `false` is a real bug -- stop and debug before proceeding
   to step 11's throughput numbers; the exact-match bar (design doc
   section 5) is not a suggestion.

10. **Correctness gate -- prompt-lookup drafter, same check:**

        .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
          --drafter prompt-lookup \
          --compare-generated-tokens docs/findings/$DATE-phase-5b-baseline-generated-tokens.json \
          --run-label $DATE-phase-5b-prompt-lookup

    Must show `"moe_layers_patched": 27` and every prompt's `token_match`
    `true`.

11. **Throughput and acceptance-rate comparison.** Read
    `mean_tokens_per_second` and `acceptance_rate` out of the three
    results JSONs from steps 8-10 -- no separate benchmark tool needed,
    `run_speculative_bench`'s own harness already measured both
    identically for all three configurations, at the default
    `--num-speculative-tokens 4`.

12. **k-sweep**, draft-model and prompt-lookup only. The original intent
    was to target just the repetition-heavy prompt (the 4th of
    `SPECULATIVE_PROMPTS`) to keep this cheap, matching Phase 1's own
    token-count sweep's shape -- but `run_speculative_bench.py`'s actual
    CLI (Task 5) has no per-prompt selection flag; `SPECULATIVE_PROMPTS`
    is always the full fixed list. Rather than add a new flag mid-session
    (risking an untested bug under time/cost pressure for a purely
    cost-saving optimization), this session ran the full 4-prompt suite
    at each k instead. Still cheap in absolute terms ($0.82/hr card, 8
    extra runs) -- confirm this reasoning still holds before repeating it,
    or add the flag properly (with its own test) in a calmer moment if a
    future phase wants the original narrower scope:

        for K in 1 2 4 8; do
          .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
            --drafter draft-model --num-speculative-tokens $K \
            --run-label $DATE-phase-5b-k-sweep-draft-model-$K
          .venv/bin/python -m scripts.run_speculative_bench --trust-remote-code \
            --drafter prompt-lookup --num-speculative-tokens $K \
            --run-label $DATE-phase-5b-k-sweep-prompt-lookup-$K
        done

    (This sweep intentionally omits `--compare-generated-tokens`: steps
    9-10 already proved exact-match correctness at k=4 for both drafters
    across all 4 prompts; correctness does not depend on k, since the
    verification rule is the same at every k, so the sweep only needs to
    measure throughput/acceptance-rate, not re-verify correctness 8 more
    times.)

13. **Cost record and teardown:**

        .venv/bin/python -c "
        from pathlib import Path
        from scripts.gpu.provision import write_cost_record
        write_cost_record(
            Path('docs/findings'),
            pod_id='<pod_id>',
            gpu_type_id='<id from step 1>',
            cost_per_hour=<rate>,
            duration_s=<elapsed>,
            note='Phase 5b speculative decoding: correctness gates + measured run + k-sweep',
            run_label='phase-5b-speculative-decoding',
        )
        "

    Then tear the pod down immediately (`scripts/gpu/provision.py` exposes
    `create`/`wait`/`terminate` only -- no `delete`/`get` subcommand, so
    status confirmation goes through the RunPod API/dashboard directly):

        uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>
        # confirm terminated via the RunPod dashboard or API (get-pod <pod_id>)
