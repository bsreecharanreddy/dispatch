# Phase 5a runbook: int8 quantized kernel

Budget cap: $5, a ceiling not a target.

**What actually happened (2026-09-17), read this before following the
steps below as written:** `RUNPOD_API_KEY` wasn't set in the shell this
session, so the pod lifecycle ran through the RunPod MCP plugin's tools
(`create-pod`, `get-pod`, `delete-pod`, `list-pod-billing`) instead of
`scripts/gpu/provision.py`'s CLI -- same underlying RunPod API, different
caller. Two Community Cloud pods hit a real host-level GPU bug before a
Secure Cloud L40 worked. Total cost: **$1.9543** of the $5 cap. Full
account, including the correctness gate, the three-way measured run, and
a real bug found and fixed mid-session:
`docs/findings/2026-09-17-phase-5a-quantization-run.md`.

1. **Create the pod.** Quoted L40 on Community Cloud at $0.69/hr,
   confirmed with the user before creating anything.

   **What actually happened:** the first L40 (Community) create attempt
   failed -- "no longer any instances available" -- and a retry failed
   the same way. Fell back to L40S (Community), which succeeded at
   $0.79/hr, but that pod's `cuInit()` (the raw CUDA driver API) returned
   `CUDA_ERROR_UNKNOWN` (999) even against the base image's own
   preinstalled torch -- confirmed via a standalone ctypes-level test
   that doesn't touch this project's own torch/venv choice at all. A pod
   restart changed which `/dev/nvidiaN` node appeared but did not fix
   it. Reproduced identically on a second, separately-created Community
   Cloud pod. **Abandoned Community Cloud entirely** and created an L40
   on **Secure Cloud** instead, at $0.82/hr -- the same ctypes test
   returned `0` (success) immediately. Everything below ran on that pod
   (`924l6eft4d8251`).

2. **Transfer the repo.** No `RUNPOD_API_KEY` locally means no `scp`
   integration either; this pod's SSH was proxy-only
   (`ssh.runpod.io`), which forces an interactive shell regardless of a
   trailing command and rejects non-PTY connections outright. Worked
   around by piping whole scripts via stdin (`ssh -tt ... < script.sh`,
   ending on EOF) for every remote command, and by transferring the repo
   as a `git archive HEAD | base64` blob written into a heredoc script
   (`base64 -d <<'B64EOF' ... | tar -xzf -`) rather than ever putting the
   ~465KB blob directly in a shell command.

3. **Reinstall `transformers==4.57.6` and apply the known monkeypatch.**
   The pod's `transformers==5.17.0` (this repo's pinned floor) breaks
   DeepSeek's own remote-code model file the same two ways Phases
   0/1/3/4 documented (`is_torch_fx_available` removed;
   `DynamicCache.get_usable_length` renamed to `get_seq_length`) --
   fourth phase running into the same, still-unresolved-upstream break.

   ```bash
   uv pip install --reinstall "transformers==4.57.6"
   ```

   (`uv run` re-syncs to the lockfile on every invocation and will
   silently revert this -- call `.venv/bin/python` directly instead, same
   as every prior phase learned.) Applied via a pod-local, never-committed
   `_patch_and_run.py` that monkeypatches `DynamicCache.get_usable_length`
   then calls `scripts.run_baseline.main()` -- this is a DeepSeek-repo
   problem, not a `dispatch` one.

   **New this session:** the pod also had `HF_HUB_ENABLE_HF_TRANSFER=1`
   set with the `hf_transfer` package not installed, which makes
   `huggingface_hub`'s downloader raise instead of silently falling back.
   Fixed with `uv pip install hf_transfer`.

4. **Kernel correctness gate:**

   ```bash
   .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_int8_kernel.py -v
   .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v
   ```

   **Result:** 15/15 and 25/25, both passed -- the int8 kernel's first
   real-hardware run.

5. **Three-way measured run**, each via the same `_patch_and_run.py`
   wrapper:

   ```bash
   .venv/bin/python _patch_and_run.py --trust-remote-code \
     --run-label $DATE-phase-5a-stock
   .venv/bin/python _patch_and_run.py --trust-remote-code \
     --run-label $DATE-phase-5a-naive --moe-kernel naive \
     --compare-reference docs/findings/$DATE-phase-5a-stock-reference.safetensors
   .venv/bin/python _patch_and_run.py --trust-remote-code \
     --run-label $DATE-phase-5a-quantized --moe-kernel quantized \
     --compare-reference docs/findings/$DATE-phase-5a-naive-reference.safetensors
   ```

   **What actually happened:** the quantized run OOMed on the first
   attempt, before any generation -- a real bug in `patch_moe_infer_quantized`
   (it quantized each layer's weights into int8 but never freed the bf16
   originals, so the model held both at once). Fixed
   (`_free_expert_weights` in `src/dispatch/kernels/integration.py`,
   commit `c75da79`), the fix pushed to the pod, and the quantized run
   re-executed successfully. Full root-cause and fix details in the
   findings doc.

6. **Memory footprint** -- a standalone script loading the model fresh
   and measuring `stack_expert_weights`/`quantize_stacked_weights`'
   byte counts directly, not sampled from `nvidia-smi`:

   ```bash
   .venv/bin/python <<'PYEOF'
   from transformers.cache_utils import DynamicCache
   DynamicCache.get_usable_length = lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)

   import json
   import torch
   from dispatch.benchmark.harness import load_model
   from dispatch.kernels.moe_forward import stack_expert_weights
   from dispatch.kernels.quantization import (
       quantize_stacked_weights,
       quantized_stacked_weights_nbytes,
       stacked_weights_nbytes,
   )

   model, _ = load_model(
       "deepseek-ai/deepseek-moe-16b-base", device="cuda",
       dtype=torch.bfloat16, trust_remote_code=True,
   )
   total_bf16 = total_int8 = layers = 0
   for module in model.modules():
       if not hasattr(module, "moe_infer"):
           continue
       weights = stack_expert_weights(module.experts)
       total_bf16 += stacked_weights_nbytes(weights)
       total_int8 += quantized_stacked_weights_nbytes(quantize_stacked_weights(weights))
       layers += 1
   print(json.dumps({
       "model": "deepseek-ai/deepseek-moe-16b-base", "dtype": "bfloat16", "device": "cuda",
       "layers": layers, "bf16_bytes": total_bf16, "int8_bytes": total_int8,
       "reduction_pct": round(100 * (1 - total_int8 / total_bf16), 2),
   }, indent=2))
   PYEOF
   ```

   **Result:** 27 layers, 27.84 GiB bf16 -> 13.95 GiB int8, **49.89%**
   reduction.

7. **Copy evidence back, record cost, tear down.** Result JSONs were
   small enough to `cat` back over the same proxy SSH session rather
   than needing a file-transfer tool; the `.safetensors` reference files
   were not copied back (`.gitignore` excludes them repo-wide, same as
   every prior phase -- they're regenerated by `capture_reference_logits`
   rather than relied on as checked-in blobs). Cost recorded locally via
   `write_cost_record` (a pure function needing no API key) once per pod
   this session, including both abandoned Community Cloud attempts:

   ```python
   from pathlib import Path
   from scripts.gpu.provision import write_cost_record
   write_cost_record(
       Path("docs/findings"), pod_id="<id>", gpu_type_id="<type>",
       cost_per_hour=<rate>, duration_s=<elapsed>,
       note="...", run_label="<label>",
   )
   ```

   Pod torn down via the RunPod MCP plugin's `delete-pod`, confirmed
   `404 pod not found` on a follow-up `get-pod` (this account's
   equivalent of `provision.py`'s own `TERMINATED` check).
