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

   Then re-read `get_pod('<pod_id>').status` and confirm `TERMINATED`
   rather than trusting the terminate command's exit code.
