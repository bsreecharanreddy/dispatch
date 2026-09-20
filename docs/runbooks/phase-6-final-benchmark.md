# Runbook: Phase 6 final benchmark vs. vLLM and SGLang (rented GPU)

One combined session -- correctness gate, kernel race, engine reference --
per `docs/plans/2026-09-19-phase-6-final-benchmark-plan.md` Tasks 10-14.
Budget cap: $10, a ceiling not a target. Actual cost: $4.5448 (RunPod's
billing API). This runbook records what actually worked on
2026-09-19/20, not the plan's first-guess commands.

1. **Wait for real L40 stock before renting anything else.** RunPod's L40
   availability flapped between `NONE` and `Low` over about 3 hours on
   2026-09-19 (checked live via the `runpod` MCP's `get-capacity`, not
   assumed). The L40S ($1.09/hr) and A40 ($0.49/hr) were both in stock the
   whole time as fallbacks; the user chose to wait rather than switch GPU
   class, to keep this phase's tables on the same hardware as Phase 1 and
   5a.

2. **Direct SSH works on this pod, not just the PTY-only `ssh.runpod.io`
   proxy** Phase 5b's runbook needed. `create-pod`'s response includes both
   `ssh.direct` and `ssh.proxy`; check both live before assuming the PTY
   dance is necessary:

       ssh -o ConnectTimeout=20 -o BatchMode=yes -i ~/.ssh/dispatch_runpod_ed25519 \
         root@<direct-ip> -p <direct-port> 'echo OK'

   Direct SSH takes multi-line heredocs and non-interactive commands
   cleanly, and **`scp`/`rsync` work over it** -- evidence retrieval at the
   end of the session was one `rsync`, not a base64-through-PTY transfer.
   Use the proxy only if the direct route is unreachable (e.g. the pod has
   no public IP).

3. **`pkill -f <pattern>` can kill its own invoking shell.** `pkill -f
   benchmark_moe.py` run over SSH matches the *remote shell's own command
   line* (which literally contains the string `benchmark_moe.py` as the
   text of the command being run), killing the SSH session itself with no
   output and exit 255 -- looks exactly like a dropped connection. Fix:
   `pkill -f "[b]enchmark_moe.py"` (the bracket trick prevents grep-family
   tools from matching their own invocation).

4. **Create the pod with a disk that fits everything, under `/workspace`**:

       # state the live price and get sign-off before this call
       # (RunPod MCP tool used directly this session, not scripts/gpu/provision.py --
       #  either works; the MCP tool was already open in this session)
       create-pod: name=dispatch-phase-6, gpu.id="NVIDIA L40",
         gpu.allowedCudaVersions=["13.0","13.2"], cloud=SECURE,
         disk=80 (container), mounts.persistent={size:150, path:/workspace}

   150GB persistent + 80GB container was enough for the 32.8GB model, two
   engine venvs, and all evidence.

5. **Ship the repo with `git archive` + base64, same as runbook 5b**, but
   verify with a `sha256sum` on both ends rather than trusting the
   transfer -- catches silent truncation immediately instead of surfacing
   as a confusing import error later.

6. **`uv sync --all-extras --dev` in `/workspace/dispatch`, then confirm the
   RoPE fix is still needed** the same way runbook 5b does. This session hit
   the same `KeyError: 'type'` in `_init_rope` (transformers auto-populates
   `rope_scaling` without a `"type"` key) and applied the same sed fix to
   the cached remote-code file. `fix_rope_inv_freq` (in-repo, applied by
   `load_model` for every caller since Phase 5b) was not needed here --
   different bug, same rope_scaling code path, both still present in the
   model repo's `modeling_deepseek.py` as of this session.

7. **vLLM 0.29.0 and SGLang 0.5.20 do NOT co-install in one venv**
   (`compressed-tensors==0.17.0` vs. `==0.18.0`), so use two venvs. Put them
   **outside `/workspace`** (e.g. `/root/vllm-venv`, `/root/sglang-venv`):
   `/workspace` is a network filesystem on this pod, and `import sglang`
   alone took minutes from there versus seconds from local disk.

       uv venv /root/vllm-venv --python 3.12
       uv pip install --python /root/vllm-venv/bin/python \
         torch==2.13.0 vllm==0.29.0 safetensors requests pytest pandas ray
       uv pip install --python /root/vllm-venv/bin/python --no-deps -e /workspace/dispatch

       uv venv /root/sglang-venv --python 3.12
       uv pip install --prerelease=allow --python /root/sglang-venv/bin/python \
         torch==2.13.0 sglang==0.5.20 cuda-tile==1.6.0rc5 safetensors requests pytest pandas ray
       uv pip install --python /root/sglang-venv/bin/python --no-deps -e /workspace/dispatch

   `--prerelease=allow` is required for SGLang alone (`cuda-tile==1.6.0rc5`,
   a `flash-attn-4` beta pin); vLLM's install does not need it. `ninja`
   (`uv pip install ninja`) is also required in the SGLang venv or its
   kernel gate fails with `FileNotFoundError: 'ninja'`. `pandas` and `ray`
   are needed for `vllm.benchmarks.serve` and `benchmark_moe.py --tune`
   respectively -- neither is a `vllm`/`sglang` dependency by default.

8. **SGLang's `fused_experts` needs a published `ServerArgs`, not just a
   distributed group.** First real run failed with `config namespace 'exec'
   not published` -- `fused_experts` reads `get_exec()`, which SGLang fills
   only from a `set_global_server_args_for_scheduler(ServerArgs(...))`
   call. Fixed in `sglang_moe.init_distributed()` (commit `c31d1f4`) to
   publish one, once, alongside the existing `initialize_model_parallel`
   call.

9. **Both tuners need a `DeepseekForCausalLM` -> `DeepseekV2ForCausalLM`
   architecture shim**, the same one this session built for SGLang's own
   tuner (`benchmark_moe.py`'s `get_model_params` and SGLang's tuner's
   `get_model_config` both key off `config.architectures[0]` and neither
   recognizes the V1 architecture `deepseek-moe-16b-base` actually reports):

       from huggingface_hub import snapshot_download
       import json, pathlib
       p = pathlib.Path(snapshot_download('deepseek-ai/deepseek-moe-16b-base',
           allow_patterns=['*.json','*.py'], local_dir='/workspace/deepseek-config-shim')) / 'config.json'
       c = json.loads(p.read_text()); c['architectures'] = ['DeepseekV2ForCausalLM']
       p.write_text(json.dumps(c, indent=2))

   Then pass `--model /workspace/deepseek-config-shim` in place of the real
   model id to either tuner (weights are never read by the tuner, only the
   config's routed-expert dims).

10. **Check the tuner's actual save behavior before budgeting time for it.**
    `benchmark_moe.py --tune` over the plan's 7 batch sizes took ~18 minutes
    *per batch size*, running single-worker and sequential (one Ray task at
    a time, not 7 in parallel), and its `save_configs` call happens exactly
    once, after every requested batch size finishes -- reading the source
    (`grep -n "save_configs\|_distribute" benchmarks/kernels/benchmark_moe.py`)
    before launching would have caught this without spending 5 minutes of
    real GPU time to confirm it live. **Budget real per-token-count tuning
    time before committing to the plan's 60-minute-per-engine-per-precision
    ceiling**, or reduce the requested batch-size list up front if the
    total cap can't absorb the full matrix.

11. **`vllm bench serve --save-detailed`'s real schema (0.29.0) has no
    per-request end-to-end `"latencies"` key** -- only per-request `ttfts`
    and `itls` (a list of per-token gaps, length `output_len - 1`). Derive
    end-to-end latency as `ttft + sum(itls)` (fixed in `serving_bench.py`,
    commit `3acf14c`). Confirm a new engine version's schema with
    `python -c "import json; print(sorted(json.load(open(f)).keys()))"`
    on one raw result before trusting field names from documentation.

12. **When launching a script that itself launches a subprocess by bare
    name** (`phase6_engine_reference.py` calls `subprocess.Popen(["python",
    ...])` for SGLang's server and `["vllm", ...]` for the bench client),
    **put the venv whose binary you need resolved FIRST on `PATH`**:

        export PATH=/root/sglang-venv/bin:/root/vllm-venv/bin:$PATH  # sglang server + vllm bench client
        /root/sglang-venv/bin/python -m scripts.gpu.phase6_engine_reference --engine sglang ...

    Reversing the order silently launches `sglang.launch_server` under the
    wrong interpreter (`ModuleNotFoundError: No module named 'sglang'`) even
    though the outer `python -m scripts...` command itself ran fine.

13. **Pull evidence with `rsync` over direct SSH, excluding `*.safetensors`**
    (the gate's reference logits, ~424MB each -- regenerable, gitignored,
    not committed):

        rsync -avz -e "ssh -i ~/.ssh/dispatch_runpod_ed25519 -p <port>" \
          --exclude='*.safetensors' root@<ip>:/workspace/evidence/phase-6/ \
          docs/findings/phase-6/

    Diff the local and remote file *lists* (not just a byte count) before
    terminating -- `diff <(ls local | sort) <(ssh ... find ... | sort)`.

14. **Terminate, then confirm with a `get-pod` call that expects a 404.**
    A 404 on the terminated pod's id is the positive confirmation, not an
    error to retry past.

15. **Read the RunPod billing API for the actual cost, don't compute
    rate x duration.** `list-pod-billing` with `podId` and hourly buckets
    gives GPU and disk cost separately and matches what actually gets
    billed; `rate * wall-clock-duration` overstated this session's cost by
    about $0.60 against the billing API's own total.
