# phase-2-vllm-benchmark -- GPU rental cost

- Pod: `ltdb2e2v0ufsrv` (NVIDIA GeForce RTX 3090)
- Rate: $0.2200/hr
- Duration: 12609s (3.503hr)
- Cost: $0.7706
- Note: Phase 2: verified PR 1 get_model_params fix unblocks deepseek-ai/deepseek-moe-16b-base, then ran --tune under uniform and zipf expert-load distributions, batch sizes 1/2/4/8/16 (--tp-size 1). Community cloud RTX 3090. Uniform sweep 8838.73s, zipf sweep 1985.18s (Triton kernel-cache warm from the first run), plus setup/verification time.
