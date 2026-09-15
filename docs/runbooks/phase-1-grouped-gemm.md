# Runbook: Phase 1 grouped-GEMM (rented GPU)

Two paid sessions. Both need the user's explicit go-ahead and a budget cap
stated before a pod exists. Tasks 1-5, 7 and 8 cost nothing.

## Session A -- kernel correctness (Task 6)

Synthetic weights only; no model download; any GPU Triton supports.

1. Pick the cheapest Community Cloud card with compute capability 8.0+ --
   Triton's README supports "NVIDIA GPUs (Compute Capability 8.0+)", so an
   Ampere-or-newer part (RTX A4000/A5000/A6000, RTX 3090/4090, A40, L4,
   L40, A100). A T4 is 7.5 and a P100 is 6.0: neither qualifies. Query the
   live catalog the same way `docs/runbooks/phase-0-baseline.md` step 1
   does, or use the RunPod MCP plugin, which is what Phase 0 actually used.
2. Create and wait:

       uv run python -m scripts.gpu.provision create --name dispatch-phase-1-kernels \
         --gpu-type "<id from step 1>" --image "<current runpod/pytorch tag>" --cloud COMMUNITY
       uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

3. Confirm the card before spending time on it:

       nvidia-smi --query-gpu=name,compute_cap --format=csv

   Below 8.0: terminate now and pick another.
4. Transfer the committed branch without pushing it (this repo pushes a
   phase once, at the end -- Phase 0 used the same trick):

       git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"

5. On the pod:

       cd dispatch
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv run python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
       uv run pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs

6. Iterate: edit locally (the Mac stays the source of truth), re-sync, rerun:

       command -v rsync || apt-get install -y rsync   # on the pod, once
       rsync -az --exclude .venv --exclude .git --exclude '*.safetensors' ./ <pod>:dispatch/

7. Mutation check, once green -- prove the suite can fail. In
   `_matmul_tile`, temporarily make every tile read expert 0's weights:

       expert_id = tl.load(tile_expert_ptr + m_tile) * 0

   Re-sync, rerun: every multi-expert case must fail. Revert, re-sync,
   rerun: green again.
8. Record the cost, terminate, and verify termination independently:

       uv run python -c "
       from pathlib import Path
       from scripts.gpu.provision import write_cost_record
       from scripts.gpu.runpod_client import get_pod
       pod = get_pod('<pod_id>')
       write_cost_record(
           Path('docs/findings'),
           pod_id=pod.id,
           gpu_type_id='<id from step 1>',
           cost_per_hour=pod.cost_per_hour,
           duration_s=<measured wall-clock seconds RUNNING>,
           note='Phase 1 kernel correctness, synthetic weights',
           run_label='phase-1-kernel-correctness',
       )
       "
       uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>

   Then re-query the pod independently rather than trusting the terminate
   command's exit code alone. Confirmed live 2026-09-15: RunPod's
   terminate/delete removes the pod entirely rather than moving it to a
   `TERMINATED` status field -- `get_pod('<pod_id>')` (a `GET
   /pods/{id}`) raises `RunPodAPIError` from a `404`, which is a stronger
   confirmation than a status string would be. Expect the 404, not a
   `.status == "TERMINATED"` read.

## Session B -- the measured run (Task 9)

Same GPU class as Phase 0 -- NVIDIA L40 -- so the comparison holds. If no
L40 is available, stop and ask: switching class silently breaks the
same-hardware rule that makes these numbers mean anything.

1. Locally: `make check` green, `git status` clean, Tasks 1-8 committed,
   budget cap stated out loud.
2. Check whether DeepSeek has changed `modeling_deepseek.py` since Phase 0
   (https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/commits/main).
   If it has, the two workarounds below may be unnecessary or wrong.
3. Create an L40 pod with a 60GB+ volume, wait, and confirm the card:

       nvidia-smi --query-gpu=name,compute_cap --format=csv

4. Transfer: `git archive HEAD | ssh <pod> "mkdir -p dispatch && tar -x -C dispatch"`
5. Environment, on the pod -- all three of Phase 0's traps at once:

       cd dispatch
       export HF_HOME=/workspace/hf_cache        # NOT the 30GB container disk
       command -v uv || pip install uv
       uv sync --all-extras --dev
       uv pip install transformers==4.57.6       # Phase 0 bug 1

   From here on invoke `.venv/bin/python` directly. `uv run` re-syncs to
   `uv.lock` on every call and silently undoes that override.

   For Phase 0 bug 2, create a pod-local wrapper -- never committed:

       cat > _patch_and_run.py <<'EOF'
       import sys
       from transformers.cache_utils import DynamicCache


       def _get_usable_length(self, new_seq_length=None, layer_idx=0):
           return self.get_seq_length(layer_idx)


       DynamicCache.get_usable_length = _get_usable_length

       from scripts.run_baseline import main

       main(sys.argv[1:])
       EOF

6. Kernel correctness on the measurement hardware, before any timing:

       .venv/bin/python -m pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v -rs

   Expected: 25 passed, 0 skipped (an L40 is compute capability 8.9, so the
   bf16 cases run). Anything red: stop. Nothing below is meaningful.
7. The stock run -- this session's own "before" number and the reference
   every kernel run is compared against:

       DATE=$(date +%Y-%m-%d)
       .venv/bin/python _patch_and_run.py --trust-remote-code \
         --run-label $DATE-phase-1-stock

8. The three backend runs. The CLI exits non-zero on disagreement, after
   writing its evidence:

       for kernel in torch naive persistent; do
         .venv/bin/python _patch_and_run.py --trust-remote-code \
           --moe-kernel $kernel \
           --compare-reference docs/findings/$DATE-phase-1-stock-reference.safetensors \
           --run-label $DATE-phase-1-$kernel || break
       done

   Every results JSON must show `"moe_layers_patched": 27`. Per Task 8's
   review: an exact `"max_abs_diff": 0.0` is suspicious, not a perfect
   score -- it would mean the kernel path never executed (e.g. the model
   was left in training mode) rather than that it matched exactly. Read
   `top1_agreement` and `max_abs_diff` by eye for each run; the CLI's own
   exit code only checks `mutual_top_k`.
9. The micro-benchmark, both routing shapes:

       .venv/bin/python -m scripts.run_kernel_bench --distribution zipf \
         --run-label $DATE-phase-1-kernel-bench-zipf
       .venv/bin/python -m scripts.run_kernel_bench --distribution uniform \
         --run-label $DATE-phase-1-kernel-bench-uniform

10. Copy the JSON back (the .safetensors stay on the pod -- `.gitignore`
    excludes them repo-wide by design):

        scp '<pod>:dispatch/docs/findings/'"$DATE"'-phase-1-*.json' docs/findings/

11. Cost, teardown, and an independent verification that the pod reports
    gone -- same as session A step 8, with `run_label='phase-1-grouped-gemm'`.
    Per session A's finding: expect a 404 on re-query, not a `TERMINATED`
    status string.
