# Runbook: Phase 0 baseline (real rented GPU)

Tasks 1-7 are developed and tested locally at zero cost. This is the one
step that spends real money. Confirm a budget cap with the user and get
explicit go-ahead before running any command that creates a pod.

## Prerequisites

- `RUNPOD_API_KEY` exported in the shell (RunPod console -> Settings ->
  API Keys).
- A budget cap stated out loud before the first `provision.py create`.
- `uv sync` run locally so `scripts/gpu/provision.py` and
  `scripts/run_baseline.py` are runnable.

## 1. Pick a GPU type and image

Query the live catalog rather than assuming a GPU type id or image tag:

    curl -s https://api.runpod.io/v2/catalog/gpus \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      | python3 -c "
import json, sys
gpus = json.load(sys.stdin)['gpus']
for g in gpus:
    if g['memory'] >= 40 and g['community']:
        print(g['id'], g['memory'], 'GB', g['price']['community'], '\$/hr community')
"

Pick the cheapest 40GB+ card on community cloud -- the design doc's memory
note is that 32.8GB of weights leaves limited headroom below 40GB.
Confirm a current `runpod/pytorch:*` image tag at
https://hub.docker.com/r/runpod/pytorch/tags; the tag in this runbook's
examples (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) was current
2026-09-14 and may have moved on.

## 2. Check whether the model needs trust_remote_code

    curl -s https://huggingface.co/api/models/deepseek-ai/deepseek-moe-16b-base \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('config', {}).get('model_type'))"

If the reported `model_type` is natively supported by the installed
`transformers` version, omit `--trust-remote-code` in step 5 (the
default). If not, add the flag.

## 3. Create the pod

    uv run python -m scripts.gpu.provision create \
      --name dispatch-phase-0-baseline \
      --gpu-type "<id from step 1>" \
      --image "<tag from step 1>" \
      --cloud COMMUNITY

Record the printed pod id.

## 4. Wait for it to come up

    uv run python -m scripts.gpu.provision wait --pod-id <pod_id>

Connect via the SSH details shown in the RunPod console for that pod --
this project doesn't reimplement RunPod's own connection tooling.

## 5. Run the baseline on the pod

    git clone <this repo> && cd dispatch
    uv sync
    uv run python -m scripts.run_baseline \
      --model-name deepseek-ai/deepseek-moe-16b-base \
      --device cuda \
      --dtype bfloat16 \
      --repetitions 5 \
      --max-new-tokens 64 \
      --output-dir docs/findings \
      --run-label $(date +%Y-%m-%d)-phase-0-baseline

This writes `docs/findings/<label>-results.json` (the correctness-
reference latency/throughput numbers) and
`docs/findings/<label>-reference.safetensors` (the logits Phase 1's
kernel gets checked against).

## 6. Copy results back, log cost, tear down -- immediately

From your local machine:

    scp <pod>:dispatch/docs/findings/<label>-* docs/findings/

Then, before doing anything else:

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
    duration_s=<measured wall-clock seconds the pod was RUNNING>,
    note='Phase 0 baseline: deepseek-ai/deepseek-moe-16b-base, single GPU',
)
"
    uv run python -m scripts.gpu.provision terminate --pod-id <pod_id>

Verify termination independently -- re-run `get_pod('<pod_id>').status`
and confirm it reports `TERMINATED` -- rather than trusting the
terminate command's exit code alone.

## 7. Record the finding

- Commit `docs/findings/<label>-results.json`,
  `<label>-reference.safetensors`, and `<label>-cost.md`.
- Update `docs/STATUS.md`: Phase 0 complete, quoting the measured numbers
  (never estimated), flip Task 8's checklist box.
- Update `CLAUDE.md`'s Current status if this is a natural stopping point
  a reader would want to know about (the proactive-refresh convention).
