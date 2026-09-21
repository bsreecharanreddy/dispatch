# Runbook: Phase 7 productionization (rented GPU + local kind demo)

One combined session -- real model server, SSH tunnel, `kind` demo,
Grafana screenshot, screen recording -- per
`docs/plans/2026-09-20-phase-7-productionization-plan.md` Task 13. Budget
cap: $5, a ceiling not a target. Actual cost: **$6.8563** (RunPod's
billing API), exceeded and disclosed mid-session -- see the findings doc
for why. This runbook records what actually worked on 2026-09-21, not the
plan's first-guess commands.

1. **L40 sold out during pod creation; A40 was the live substitute.**
   Quoted at $0.82/hr at session start via `list-gpu-types`/`get-capacity`;
   by the time `create-pod` ran, stock was gone. A40 ($0.49/hr, 46068MiB)
   was in stock and is what actually ran this session -- check live stock
   immediately before creating, not just at the start of the
   conversation.

2. **Direct SSH, same pattern as Phase 6's runbook**:

       ssh -o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
         -i ~/.ssh/dispatch_runpod_ed25519 -p 22094 root@63.141.33.106 '<cmd>'

   The `ServerAliveInterval`/`ServerAliveCountMax` pair matters here more
   than in prior phases -- this session's SSH connection dropped
   mid-command at least twice (`Timeout, server ... not responding`,
   `Connection reset`) with no other symptom. Retrying the exact same
   command after a fresh connectivity check (`ssh ... echo PING_OK`)
   always worked; there was no underlying pod problem.

3. **Start the model server as a module, not a script**:

       tmux new-session -d -s modelserver -x 200 -y 50 \
         'export HF_HOME=/workspace/hf_cache; .venv/bin/python -m scripts.run_model_server --responder kernel --port 50051 > /workspace/modelserver.log 2>&1'

   `python scripts/run_model_server.py` puts `scripts/` itself on
   `sys.path[0]`, not the repo root, so the lazy
   `from scripts.gpu.phase7_kernel_responder import ...` import inside
   `build_responder()` raises `ModuleNotFoundError: No module named
   'scripts'`. `-m scripts.run_model_server` (run from
   `/workspace/dispatch`) puts the repo root on `sys.path[0]` instead and
   resolves cleanly. Same fix needed for the correctness-test invocation
   if using a wrapper script rather than pytest directly (pytest itself
   is unaffected -- it inserts the repo root on its own).

4. **Same `transformers` remote-code shims as every prior phase's
   runbook**, applied via a pod-local wrapper (never committed) that
   monkeypatches before importing `scripts.run_model_server.main`:

       import transformers.utils.import_utils as _iu
       _iu.is_torch_fx_available = lambda: False

       from transformers.cache_utils import DynamicCache
       DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length(0)

       @classmethod
       def _from_legacy_cache(cls, past_key_values=None):
           if past_key_values is None:
               return cls()
           if isinstance(past_key_values, cls):
               return past_key_values
           raise NotImplementedError("pod-local shim: only None/DynamicCache supported")
       DynamicCache.from_legacy_cache = _from_legacy_cache
       DynamicCache.to_legacy_cache = lambda self: self

   All four still reproduced at this pod's installed `transformers`
   version, exactly as documented in Phase 5b's runbook.

5. **If the gpu-marked correctness test fails identically after a fix
   ships, suspect stale bytecode before suspecting the fix.**
   `/workspace` on this pod is `mfs#ca-mtl-1.runpod.net`, a FUSE network
   filesystem; a checksum-verified-identical file still reproduced the
   exact pre-fix failure on a second run, and took *longer* (1422s vs.
   1247s) -- consistent with Python recompiling and re-caching bytecode
   under filesystem-level contention rather than trusting a possibly
   stale `.pyc`. Fix:

       find /workspace/dispatch -iname '__pycache__' -exec rm -rf {} +
       .venv/bin/python -B _patch_and_test.py -m gpu tests/unit/test_phase7_kernel_responder.py -v

   `-B` disables bytecode caching outright -- slower, but removes the
   variable entirely rather than hoping the manual clear was enough.

6. **If real generated text comes back with no spaces between words
   (or literal `Ġ`/`Ċ` characters), check the tokenizer's own wiring
   before touching the serving code.** This is not the same bug as
   per-token-vs-whole-sequence decode (which *was* a real, separate,
   already-fixed issue in `KernelResponder.generate()` before this
   session). Diagnose in order:

       python -c "
       from transformers import AutoTokenizer
       tok = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base', trust_remote_code=True)
       print(tok.tokenize('hello world'))                    # word boundaries lost already?
       print(tok.backend_tokenizer.pre_tokenizer)             # Metaspace (expects SentencePiece '▁')?
       import json, glob
       vocab = json.load(open(glob.glob('/workspace/hf_cache/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/*/tokenizer.json')[0]))['model']['vocab']
       print(sum(1 for k in vocab if '▁' in k), sum(1 for k in vocab if 'Ġ' in k))  # which marker does the vocab actually use?
       "

   If the vocab is majority `Ġ`-marked (byte-level BPE) but the
   pre-tokenizer/decoder expect `▁` (SentencePiece), rewire both:

       from tokenizers import pre_tokenizers, decoders
       tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
       tok.backend_tokenizer.decoder = decoders.ByteLevel()

   Verify with a round trip (`tok(tok.decode(ids))["input_ids"] == ids`)
   before trusting it against the real model. This fix is committed
   (`scripts/gpu/phase7_kernel_responder.py`'s `fix_tokenizer_byte_level`,
   called once in `build_kernel_responder`) since it corrects the
   tokenizer's own shipped configuration, not a version-specific
   environment mismatch -- unlike the `transformers` shims in step 4,
   there's no reason to re-derive it on the next GPU session for this
   model.

7. **SSH tunnel from the dev machine, `kind` demo unmodified from the
   stub rehearsal**:

       ssh -f -N -i ~/.ssh/dispatch_runpod_ed25519 -p 22094 -L 50051:localhost:50051 root@<pod-ip>
       ./scripts/run_kind_demo.sh

   `k8s/model-server-external.yaml`'s `ExternalName` Service needed zero
   changes -- `host.docker.internal` resolves through the tunnel exactly
   as it resolved to the local stub in Task 11's rehearsal.

8. **Fire a few warm-up requests before capturing evidence.** The Grafana
   dashboard's panels are far more legible with 5-10 data points than
   with 1; fired several varied prompts before the screenshot and the
   on-camera recording, all counted honestly in the final
   `dispatch_router_requests_total`.

9. **PII lives in unexpected corners of a screen recording, not just the
   terminal prompt.** A `Cmd+Tab` window-switch mid-recording produced a
   warped/rotated transition frame (a static redaction box can't track
   it) and, moments later, an app-switcher tooltip that briefly showed a
   VS Code workspace name containing the same username the terminal
   prompt already leaked. Frame-by-frame review (`ffmpeg -vf fps=N` at
   decreasing intervals to bound each transient element's exact start/end
   time) found both; a short solid-black cut over the transition window
   was simpler and more reliable than trying to track warped geometry.

10. **Pull evidence before tearing down anything.** Order that avoided
    re-provisioning: copy pod-side logs off first (`ssh ... cat
    /workspace/*.log > local-file`), export the router's `/metrics` via a
    fresh `kubectl port-forward` (needed *before* `kind delete cluster`,
    not after), *then* delete the `kind` cluster, *then* kill the SSH
    tunnel and the model-server `tmux` session, *then* terminate the pod
    via the RunPod MCP `delete-pod`, and verify with `get-pod` returning
    `404`.

11. **Real cost, from RunPod's billing API, not rate × duration**: query
    `list-pod-billing` with the pod id after termination, not before --
    the last partial hour only settles once the pod actually stops.
