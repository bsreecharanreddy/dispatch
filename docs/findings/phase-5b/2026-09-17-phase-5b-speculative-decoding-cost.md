# phase-5b-speculative-decoding -- GPU rental cost

- Pod: `2t6wh9l3okl3gj` (NVIDIA L40)
- Rate: $0.8200/hr
- Duration: 3000s (0.833hr)
- Cost: $0.6833
- Note: Phase 5b speculative decoding: correctness gates (draft-model + prompt-lookup, both byte-exact vs baseline) + measured run + k-sweep (k=1,2,4,8 x 2 drafters, full 4-prompt suite each -- CLI has no per-prompt filter flag, so ran full suite rather than write new code mid-session). Pod confirmed terminated (404 on get-pod after delete-pod returned 204).
