# phase-3-multi-gpu-ep -- GPU rental cost

- Pod: `3i210zz3qvgvo1` (NVIDIA H200 (2x, SXM))
- Rate: $9.1800/hr
- Duration: 5109s (1.419hr)
- Cost: $13.0280
- Note: Phase 3: DeepEP install (V1 legacy API after V2 GIN backend proved unavailable -- no GPU Fabric Manager on this rental), NVLink verification, real EP MoE layer built and validated on toy + real 64-expert model, correctness gate passed (perfect top-1, mutual top-5 across 3 prompts), single-GPU H200 kernel bench, real per-expert token-count measurement, real-scale kernel bench
