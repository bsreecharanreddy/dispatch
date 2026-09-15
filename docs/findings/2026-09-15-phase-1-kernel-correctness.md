# Phase 1 kernel correctness session -- findings

## Infrastructure

- Pod `sta1ejhhrg5bc4`, RunPod Community Cloud, NVIDIA GeForce RTX 3090
  (compute capability **8.6** -- confirmed live via `nvidia-smi
  --query-gpu=name,compute_cap,memory.total --format=csv`, meets Triton's
  8.0+ floor), 24GB VRAM, 30GB container disk.
- RTX A4000 ($0.17/hr, the originally quoted card) sold out between the
  price quote and pod creation -- Community Cloud "LOW" availability is
  volatile. Substituted RTX 3090 ($0.22/hr) with the user's go-ahead,
  same compute-capability-8.0+ requirement, same $3 cap.
- Image: `runpod/pytorch:1.3.1-cu1290-torch290-ubuntu2404` (current
  stable tag on Docker Hub, checked live 2026-09-15).
- Environment on the pod, confirmed live: `torch 2.14.0+cu130`, `cuda
  13.0`, `triton 3.8.0`.
- No model download -- synthetic weights only, per the plan's Task 6
  scope.
- No usable local SSH private key existed for the key already registered
  with RunPod from Phase 0 (`dispatch-phase-0-baseline-20260914`). A new
  ed25519 keypair was generated locally
  (`~/.ssh/dispatch_runpod_ed25519`, comment
  `dispatch-phase-1-kernels-20260915`) and registered on the account
  ADDITIVELY (the Phase 0 key was resent in the same call, not dropped --
  `update-ssh-keys` is a full replacement, not a merge). Both keys are
  now authorized on pods created with `startSsh`; Task 9's operator can
  use either.

## Correctness

`pytest -m gpu tests/unit/test_grouped_gemm_kernel.py -v`: **25 passed,
0 skipped** -- both `grouped_matmul` (naive) and `grouped_matmul_persistent`
(persistent, cache-aware) match `torch_grouped_matmul`'s fp32 reference at
every tested configuration: toy dims (24x16, an empty expert, a multi-tile
expert), decode-shaped (six single-row experts, gate/up_proj dims
1408x2048), and prefill-shaped (partial tiles, down_proj dims 2048x1408),
each in both fp16 and bf16 (bf16 gated on compute capability 8.0+, which
this card meets). Also covers the full DeepSeekMoE-16B-shaped routed layer
(64 experts, 6-of-64 routing) at 1, 37, and 256 tokens, and the
`block_m < MIN_BLOCK_M` rejection.

**No kernel bugs found.** Both kernels were correct on the very first real
execution -- an unusually clean outcome; this project's own Phase 0 run
found three environment bugs, and the plan's risk section explicitly
expected at least one kernel bug here.

**One real, non-kernel bug found and fixed**: the first run produced a
`UserWarning: Converting a tensor with requires_grad=True to a scalar may
lead to unexpected behavior` from `assert_matches_reference`
(`src/dispatch/kernels/moe_forward.py`, from Task 3). `ReferenceMoE`'s
parameters carry `requires_grad=True`, so `expected.abs().max()` inside
`assert_matches_reference` triggered the warning the first time this
function was ever called with a real gradient-carrying tensor on a real
device (no CPU test in Tasks 1-5 happened to exercise that path). Fixed by
detaching both tensors before conversion:

```python
atol = CORRECTNESS_RTOL * float(expected.detach().abs().max())
torch.testing.assert_close(
    actual.detach().float(), expected.detach().float(), rtol=CORRECTNESS_RTOL, atol=atol
)
```

Re-synced and re-ran: 25 passed, 0 warnings. Regression-guarded with a new
CPU test, `test_assert_matches_reference_accepts_gradient_carrying_tensors`
(`tests/unit/test_moe_forward.py`), which fails under
`warnings.simplefilter("error")` on the pre-fix code and passes on the
fix -- confirmed by temporarily reverting the fix locally, observing the
same `UserWarning` this session hit on real hardware, then restoring it.

## Mutation check

Per the plan's discipline ("a test suite that cannot fail is not a
gate"), `_matmul_tile`'s expert-selection load was mutated to
`expert_id = tl.load(tile_expert_ptr + m_tile) * 0` (forcing every tile to
read expert 0's weights), synced, and re-run:

- **24 of 25 failed** -- every case that exercises more than one expert
  went red, with large, non-tolerance-adjacent mismatches (example:
  `Mismatched elements: 507412 / 524288 (96.8%)`, greatest absolute
  difference 0.141 against an allowed 0.00088).
- The one pass (`test_rejects_a_schedule_below_tl_dots_minimum_block_m`)
  is a pure host-side validation test that never reaches the mutated
  kernel body, so it is expected to be unaffected.

Reverted (`git diff -- src/dispatch/kernels/grouped_gemm.py` against the
pre-mutation commit confirmed empty -- the committed kernel file carries
no trace of the mutation), re-synced, re-ran: **25 passed** again. The
suite can fail, and the fix restores it.

## Cost

$0.0606 for 991 seconds (16.5 minutes) of RTX 3090 rental at $0.22/hr --
well under the $3 cap. Full record:
`docs/findings/2026-09-15-phase-1-kernel-correctness-cost.md`. Duration
provenance: pod `createdAt` was `2026-09-15T15:43:56.898Z` (from the
create-pod response's `startedAt` field); wall-clock at the point of
`delete-pod` was `2026-09-15T16:00:27Z` (local `date -u`, immediately
before the terminate call) -- 991.25s between them, rounded to 991s.

Pod `sta1ejhhrg5bc4` terminated via `delete-pod`; termination verified
independently by re-querying the pod afterward, which returned `404 pod
not found` (RunPod's terminate fully removes the resource, unlike
stop/exit).
