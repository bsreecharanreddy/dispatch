# Phase 2 runbook: vLLM skewed-load benchmark contribution

Budget cap: $3. Spot/marketplace only. Compute capability 8.0+ required
(Triton floor). No cross-phase throughput claim is being made here, so
the cheapest available Ampere+ spot card is the right choice -- check
`list-gpu-types`/RunPod marketplace pricing at rental time rather than
assuming Phase 0/1's L40 rate.

1. Create and wait for the pod (60GB disk: deepseek-moe-16b-base is
   ~32.8GB in bf16, per Phase 0/1's own findings, plus vLLM's own
   footprint):

   ```bash
   uv run python scripts/gpu/provision.py create --name phase-2-vllm-bench \
     --gpu-type <cheapest 8.0+ spot card available at rental time> \
     --image <RunPod CUDA/PyTorch template> --cloud community --disk-gb 60
   uv run python scripts/gpu/provision.py wait <pod-id>
   ```

2. SSH in. Clone the fork and check out `feat/moe-benchmark-skewed-load`
   (Task 1-4's commits; rebase onto `fix/deepseek-v1-get-model-params` if
   PR 1 hasn't merged yet -- see Task 6 Step 1):

   ```bash
   git clone git@github.com:bsreecharanreddy/vllm.git
   cd vllm
   git checkout feat/moe-benchmark-skewed-load
   ```

3. Install, using vLLM's own precompiled-wheel editable-install path (not
   a from-source build):

   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
   ```

   If this fails (precompiled wheel unavailable for the pod's CUDA/torch
   combination), fall back to `uv pip install vllm` (latest released
   wheel) and copy just the locally-edited
   `benchmarks/kernels/benchmark_moe.py` over it -- the two changes touch
   only that script, not the installed package, so a version-mismatched
   released wheel still works as long as its
   `vllm.model_executor.layers.fused_moe` exports the names this script's
   top-of-file imports list. If neither install path works against this
   script's current imports, stop, record why in the findings doc, and do
   not claim a measurement was taken.

4. **Verify PR 1's fix actually unblocks the model** before spending any
   more GPU time -- no `--tune`, one batch size, fast:

   ```bash
   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base \
     --trust-remote-code --batch-size 1 --seed 0
   ```

   Expected: it runs and prints a kernel time, instead of the pre-fix
   `AttributeError`. If it still errors, stop and diagnose before step 6
   -- do not proceed to a paid `--tune` sweep against a broken model
   resolution path.

5. Confirm the model is genuinely unquantized (already known from Phase
   0/1: bf16, no `quantization_config` in `config.json`), so
   `benchmark_config`'s unquantized branch (`args.dtype auto`, the
   default) is what actually runs -- the same branch that would execute
   at real vLLM serve time for this model on this hardware, since none of
   the quantized backends (fp8/int8/int4/deep_gemm) apply without a
   `quantization_config`. This is the design doc's "verify the backend
   that's actually live" check, satisfied by the model's own config
   rather than needing to trace vLLM's kernel-selection code further.

6. Run the actual comparison, uniform first, then zipf:

   ```bash
   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base --trust-remote-code \
     --tune --batch-size 1 2 4 8 16 \
     --expert-load-distribution uniform --save-dir ./tuned-uniform/ --seed 0

   python benchmarks/kernels/benchmark_moe.py \
     --model deepseek-ai/deepseek-moe-16b-base --trust-remote-code \
     --tune --batch-size 1 2 4 8 16 \
     --expert-load-distribution zipf --save-dir ./tuned-zipf/ --seed 0
   ```

   Watch elapsed time and the pod's running cost after batch size 1
   completes on each distribution; if the pace projects past the $3 cap
   before both distributions finish all five sizes, stop after the
   current distribution's last completed batch size (per the plan's
   Global Constraints) rather than letting it run to a budget overrun.

7. Compare and capture the evidence:

   ```bash
   diff -u ./tuned-uniform/*.json ./tuned-zipf/*.json
   ```

   Record, per batch size, whether the winning `BLOCK_SIZE_M/N/K`,
   `GROUP_SIZE_M`, `num_warps`, or `num_stages` differs between the two
   distributions. Copy both JSON files and the full console output back
   to this repo's `docs/findings/` before doing anything else.

8. Tear down immediately, right after step 7's evidence is copied out:

   ```bash
   uv run python scripts/gpu/provision.py terminate <pod-id>
   ```

9. From dispatch's own repo root, after the pod's evidence is copied out,
   record the measured cost (reuses `write_cost_record` from Phase 1's
   `scripts/gpu/provision.py` as-is -- no new cost-recording code):

   ```python
   from pathlib import Path
   from scripts.gpu.provision import write_cost_record

   write_cost_record(
       Path("docs/findings"),
       pod_id="<pod-id>",
       gpu_type_id="<gpu-type>",
       cost_per_hour=<rate>,
       duration_s=<measured seconds>,
       note=(
           "Phase 2: verify get_model_params fix + --tune under uniform "
           "and zipf expert-load distributions, batch sizes 1/2/4/8/16"
       ),
       run_label="phase-2-vllm-benchmark",
   )
   ```
