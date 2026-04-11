# Hyak klone Probe Findings (2026-04-10)

Full probe log lives in the user's work directory at `/gscratch/stf/naladala/agentbreed/real_study_v1/logs/hyak_probe_20260410_195459.log`. This file summarizes the findings that are load-bearing for the harness and docs. Any subsequent edits to the compute plan or Slurm scripts that invoke Hyak-specific parameters should cite this file.

## Cluster identity
- Host: `klone-login03`, the UW Hyak klone cluster.
- User: `naladala`.
- Groups: `all stf test klone`.
- Accounts: `stf` (for `stf`, `stf-cpu-g2` partitions) and `stf-ckpt` (for `ckpt`, `ckpt-gpu`, `ckpt-all`, `ckpt-g2`).
- Home filesystem: GPFS on `/mmfs1`, 2.7 PB total, 345 TB free at probe time.
- Work directory: `/gscratch/stf/naladala/agentbreed/real_study_v1` (writable, persistent).

## GPU reality check
**Hyak klone does NOT have H100.** The actual GPU inventory that matters for this study:

| Partition | GPU type | Cards/node | Notes |
|---|---|---|---|
| `ckpt-g2` | **H200** | 8 | Newest-generation card. Clean H200-only partition (plus L40 and L40s on same partition). Our primary target. |
| `ckpt-all` | **H200** | 8 | Also has H200, plus a mix of older cards (A100, A40, L40, L40s, RTX6k, RTX2080Ti, P100). |
| `ckpt` | A100 80GB, A40 48GB | 8 | Fallback for 32B models or Llama-70B AWQ. Many A40 nodes available. |
| `ckpt` | 2080Ti, RTX6k | 4–8 | Too small for our study — ignore. |
| `gpu-2080ti` | 2080Ti | 8 | Same — ignore. |
| `stf` partition | (CPU / no GPU nodes visible) | — | Not for GPU work. |

**All GPU partitions are preemptable (`ckpt`-family).** Jobs can be killed mid-run. Our atomic per-cell JSON writes + `--requeue` Slurm flag handle this cleanly.

## Memory fit implications

With H200's 141 GB VRAM, the model deployment plan simplifies:

| Model | Precision | Cards needed | KV headroom |
|---|---|---|---|
| Qwen3-32B | fp16 | 1× H200 | comfortable (~77 GB for KV at batch 32) |
| Qwen3-32B | AWQ 4-bit | 1× H200 | excessive (fits on L40 48GB if H200 preempted) |
| Llama-3.3-70B | fp16 | **2× H200** (tight at 1×, safer at 2× with TP=2) | comfortable at TP=2 |
| Llama-3.3-70B | AWQ 4-bit | 1× H200 | comfortable (~105 GB for KV at batch 32) |
| Qwen3-14B | fp16 | 1× H200 or 1× L40 48GB | trivial |

**Primary plan:** Qwen3-32B fp16 on 1× H200. Llama-3.3-70B AWQ 4-bit on 1× H200 (cheaper and more KV headroom than 2× fp16 tensor parallel, at the cost of ~2% accuracy on published benchmarks — acceptable for our purposes). The fp16 TP=2 configuration stays as a fallback if AWQ weights are unavailable or buggy.

## Throughput expectations

H200 vs H100 memory bandwidth: 4.8 TB/s vs 3.35 TB/s (+43%). For inference-bottlenecked workloads (which ours is), effective throughput scales close to bandwidth. Revised per-call wall-clock estimates:

| Model | vLLM config | Per-call @ steady-state batch | Per-call wall-clock with ckpt preemption overhead (~25%) |
|---|---|---|---|
| Qwen3-32B fp16 | 1× H200 | ~0.7 s | ~0.9 s |
| Llama-3.3-70B AWQ | 1× H200 | ~1.1 s | ~1.4 s |
| Llama-3.3-70B fp16 TP=2 | 2× H200 | ~1.2 s (but 2 GPU-hours consumed per hour) | ~1.5 s effective |

The ~25% preemption overhead is conservative. In practice, `ckpt-g2` is relatively uncontested for H200 at off-peak hours, but we plan for the worst case.

## Software stack
- CUDA 13.0, driver 580.126.20.
- Python 3.11.15 already installed in user's miniconda `parchment` env.
- `uv` installed fresh in `$HOME/.local/bin` (Python package/env manager).
- No `vllm`, `torch`, or other ML libraries pre-installed in the `agentbreed` venv — we install them from scratch.
- No `module load` is strictly required; the venv handles it.
- `git` 2.43.7 available.

## Srun verification
```
srun 1-min GPU probe on ckpt (will request h100 or any GPU)
srun: No account specified, defaulting to: stf
srun: job 34555908 queued and waiting for resources
srun: job 34555908 has been allocated resources
srun hostname: z3001
Fri Apr 10 19:55:20 2026
NVIDIA GeForce RTX 2080 Ti, driver 580.126.20, CUDA 13.0
(srun exit code: 0)
```

- The default account (`stf`) was accepted for a tiny GPU job on `ckpt`, so Slurm is not strictly requiring `-A stf-ckpt`. That said, we explicitly use `-A stf-ckpt` in all GPU job scripts for clarity and to avoid any "defaulting" surprises.
- The probe landed on a 2080Ti node because we did not specify `--gres=gpu:h200:1`. All our real jobs must specify the GPU type to land on H200.

## Implications for the committed docs

- `real_study/docs/03_model_suite.md` and `07_compute_plan.md` were written assuming a generic "H100/H200" target. Where H100-specific numbers appear, they should be read as "approximate, see PROBE_FINDINGS.md for Hyak-klone-specific values." A precise rewrite is deferred to after Stage A completes and we have real throughput measurements on an actual Hyak H200 run.
- The Slurm scripts under `real_study/harness/slurm/` are updated to target `ckpt-g2`, `stf-ckpt` account, and `--gres=gpu:h200:1` as the primary configuration. An A100 fallback path is provided in `submit_smoke_fallback.sh`.
- Throughput assumptions in `07_compute_plan.md` will be validated against the real Stage B pilot data and corrected if off by more than 25%.

## Committed-to actions (this session)

1. Rewrite `real_study/harness/slurm/submit_smoke.sh` with Hyak-correct parameters.
2. Add `submit_smoke_tiny.sh` for a fast (5-minute) stack-verification job using `Qwen/Qwen2.5-0.5B-Instruct`.
3. Add `submit_smoke_qwen32b.sh` for the real Stage A smoke job.
4. Add `real_study/harness/README.md` with Hyak-specific quickstart.
5. Commit and push to GitHub so the clone on Hyak gets the corrected version.
