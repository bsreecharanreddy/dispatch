# phase-5a-quantization -- GPU rental cost

- Pod: `924l6eft4d8251` (NVIDIA L40 (Secure Cloud))
- Rate: $0.8200/hr
- Duration: 7473s (2.076hr)
- Cost: $1.7022
- Note: Phase 5a: int8 quantized Triton kernel correctness gate (15/15 GPU tests) plus bf16 kernel re-verification (25/25), then the three-way measured run (stock/naive/quantized) and memory-footprint measurement on the real deepseek-ai/deepseek-moe-16b-base model. Includes a mid-session fix: patch_moe_infer_quantized was not freeing the original bf16 expert weights (stack_expert_weights re-points each Linear at a *view* into a shared bf16 tensor), so the quantized run OOMed on first attempt; fixed and rerun successfully.
