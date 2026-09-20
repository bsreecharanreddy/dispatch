# phase-6-final-benchmark -- GPU rental cost

- Pod: `97jlyai5sowyeq` (NVIDIA L40, Secure Cloud, US-KS-2)
- Session: 2026-09-19T17:54:26Z (created) -- 2026-09-20T00:10Z (terminated), ~6.26hr
- Nominal rate: $0.8200/hr
- **Cost (measured from RunPod's billing API, `list-pod-billing`, hourly buckets
  17:00-00:00 UTC 2026-09-19/20, not computed from rate x duration): $4.5448**
  ($4.3745 GPU, $0.1704 disk). Of the $10 cap.
- Note: Phase 6 single session: at-scale correctness gate (Stage 1, all
  configs passed), kernel race at default configs (Stage 2 -- tuner runs
  skipped after a live cost check found each token-count/dtype/engine tuning
  run takes ~18 minutes and only saves after every requested count finishes;
  see `docs/STATUS.md`'s Phase 6 section), and the vLLM/SGLang engine
  reference (Stage 3, both engines served cleanly across concurrency
  1/4/16/64). One real API-drift fix shipped mid-session: vLLM 0.29.0's
  `--save-detailed` output carries no per-request end-to-end latency field,
  fixed to derive it from ttft + inter-token gaps (commit `3acf14c`).
