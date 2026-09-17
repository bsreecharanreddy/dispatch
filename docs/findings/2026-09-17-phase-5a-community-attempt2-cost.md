# phase-5a-community-attempt2 -- GPU rental cost

- Pod: `wmv06rf1l2sm8s` (NVIDIA L40S (Community Cloud))
- Rate: $0.7900/hr
- Duration: 1024s (0.284hr)
- Cost: $0.2247
- Note: Phase 5a attempt 2: RunPod Community Cloud L40S. Host-level GPU passthrough bug -- cuInit() returned CUDA_ERROR_UNKNOWN (999) even against the base images own stock torch, confirmed via a raw ctypes test, reproduced on a second Community Cloud pod too. Abandoned Community Cloud entirely after this. Duration/cost derived from RunPod billing APIs exact recorded total ($0.2247) at this pods quoted $0.79/hr rate.
