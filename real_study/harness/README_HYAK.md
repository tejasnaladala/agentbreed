# Real-study harness — Hyak klone quickstart

Concrete, copy-pasteable setup for running the agentbreed real-LLM study on **UW Hyak klone** with STF allocation. Based on the 2026-04-10 probe (see `../docs/PROBE_FINDINGS.md` for the discovered cluster state).

## One-time setup (do this once per clean clone)

### 1. Clone into gscratch (never into home)

```bash
cd /gscratch/stf/$USER
mkdir -p agentbreed
cd agentbreed
git clone https://github.com/tejasnaladala/agentbreed.git repo
cd repo
```

All work happens inside `repo/`. The `real_study_v1/` directory you created earlier (where the `probe_hyak.sh` and `.venv` live) is sibling to `repo/`, not inside it — that's intentional. The venv is shared across clones.

### 2. Point the venv at the clone

```bash
# Activate the shared venv (already set up during Stage A env bootstrap)
source /gscratch/stf/$USER/agentbreed/real_study_v1/.venv/bin/activate

# Install the repo in editable mode so breed/ imports work
cd /gscratch/stf/$USER/agentbreed/repo
uv pip install -e .

# Verify
python -c "import breed; print('breed:', breed.__version__)"
python -c "from real_study.harness.genome import default_genome_for; print(default_genome_for('forecastbench').benchmark)"
```

Expected: `breed: 0.1.0` and `forecastbench`.

### 3. Run the harness unit tests

```bash
cd /gscratch/stf/$USER/agentbreed/repo
python -m pytest real_study/harness/tests/ -q
```

Expected: **35 passed in ~0.1s**. If any test fails, stop — the schema lock is broken.

## Stage A — smoke tests (in order)

Run these in order. Do not skip. Each one has a gate.

### A1 — Tiny smoke (~5 min wall clock, costs < 0.01 GPU-day)

Purpose: verify the full stack (vllm, httpx, genome validation, search loop, one real LLM call round-trip) on a cheap GPU with a tiny 0.5B model.

```bash
cd /gscratch/stf/$USER/agentbreed/repo
sbatch real_study/harness/slurm/submit_smoke_tiny.sh
```

Watch:
```bash
# In another terminal (or after sbatch prints the job id)
squeue -u $USER
# Once running:
tail -f real_study/logs/smoke_tiny_<JOB_ID>/harness.log
```

**Gate A1:** `STAGE A TINY SMOKE: PASS` at the bottom of `harness.log`, and a `smoke_tiny_result.json` with `"status": "pass"`.

Typical failure modes: (a) vllm doesn't start within 5 min (`--download-dir` points to an unwritable path, network hiccup downloading the tiny model), (b) `CUDA_HOME` errors (shouldn't happen with pre-built wheels), (c) benchmark import errors.

### A2 — Real smoke on Qwen3-32B (~15-30 min wall clock, ~0.3 GPU-day)

Purpose: verify Qwen3-32B serves cleanly on H200 and vllm throughput matches the planned ~700 tok/s.

```bash
# Pre-download Qwen3-32B into the shared HF cache first (one time, ~60 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', cache_dir='/gscratch/stf/naladala/cache/huggingface')
"

# Then submit the real smoke job
sbatch real_study/harness/slurm/submit_smoke_qwen32b.sh
```

**Gate A2:** pass line + a steady-state throughput of at least 500 tok/s logged in the vllm output. If throughput is < 400 tok/s, we have an issue with the H200 allocation or max-num-seqs and need to investigate before Stage B.

### A3 — Real smoke on Llama-3.3-70B (AWQ, ~20-40 min wall clock, ~0.5 GPU-day)

Purpose: verify the 70B path works. We use AWQ 4-bit to fit comfortably on 1× H200 with plenty of KV cache headroom.

```bash
# Pre-download Llama-3.3-70B-AWQ (one time, ~35 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', cache_dir='/gscratch/stf/naladala/cache/huggingface')
"
# Note: if Llama-3.3-70B-Instruct-AWQ is available under a similar repo, use that instead.
# The preregistration says 'Llama-3.3-70B-Instruct'; AWQ is our fallback quantization.

sbatch real_study/harness/slurm/submit_smoke_llama70b.sh
```

**Gate A3:** pass line + steady-state throughput of at least 300 tok/s.

## Stage B — pilot (after all Stage A gates pass)

See `../docs/09_execution_stages.md` § Stage B. 40 cells (1 model × 1 benchmark × 8 methods × 5 seeds) takes ~1.5 GPU-days.

## Environment variables you ALWAYS need

These are appended to `.venv/bin/activate` but are critical enough to list here:

```bash
export UV_CACHE_DIR=/gscratch/stf/$USER/cache/uv
export PIP_CACHE_DIR=/gscratch/stf/$USER/cache/pip
export HF_HOME=/gscratch/stf/$USER/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/gscratch/stf/$USER/cache/huggingface
export TORCH_HOME=/gscratch/stf/$USER/cache/torch
export TMPDIR=/gscratch/stf/$USER/cache/tmp
```

**Never let any of these default to `~/.cache/`** — home quota on Hyak is ~10 GB and a single torch install exhausts it.

## Slurm quick reference for our study

| What | Flag |
|---|---|
| Account | `-A stf-ckpt` |
| Partition (primary, H200/L40/L40s) | `-p ckpt-g2` |
| Partition (fallback, mixed GPUs) | `-p ckpt-all` |
| Partition (A100 / A40 fallback) | `-p ckpt` |
| Specific GPU type | `--gres=gpu:h200:1` or `--gpus=h200:1` |
| Memory | `--mem=128G` (for Qwen-32B), `--mem=256G` (for Llama-70B) |
| Time | `--time=HH:MM:SS`, preemption can kill earlier |
| Requeue on preemption | `--requeue` |
| Preserve environment | default — inherits from submission shell |

## Preemption etiquette

All GPU jobs run on `ckpt-*` partitions, meaning they can be preempted. The harness is designed to handle this:

- Every cell writes its output JSON atomically (`.tmp` then rename).
- Slurm `--requeue` flag means preempted jobs auto-requeue.
- The runner skips cells whose output JSON already exists.

If you see a job get preempted (`State=PREEMPTED` in `sacct`), it will be automatically requeued. The only time you need to intervene is if the same cell fails repeatedly — then check the harness log.

## Disk hygiene

- `/gscratch/stf/$USER/cache/` — pip/uv/torch/HF caches. Can grow to 50+ GB. Prune with:
  ```bash
  uv cache prune  # removes unused wheels
  rm -rf /gscratch/stf/$USER/cache/tmp/*
  ```
- `/gscratch/stf/$USER/cache/huggingface/` — model weights. Do NOT prune unless you're re-downloading.
- `/gscratch/stf/$USER/agentbreed/repo/real_study/logs/` — Slurm output. Can be moved to `../archive_logs/` after each stage completes.
- `/gscratch/stf/$USER/agentbreed/repo/real_study/results/` — per-cell JSONs. **Never delete.** These are the paper's evidence.

## Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `Disk quota exceeded` during install | Cache defaulted to `~/.cache/` | `export UV_CACHE_DIR=/gscratch/...` and re-run |
| `CUDA_HOME is not set` during vllm install | Source build on login node | Pin to `vllm==0.7.3` (pre-built wheels) |
| vllm server fails to start | Model weights not downloaded | Pre-download via `huggingface_hub.snapshot_download` |
| `ImportError: no module named real_study` | Repo not pip-installed | `uv pip install -e /gscratch/stf/$USER/agentbreed/repo` |
| Job preempted and disappeared | Normal on ckpt | Check `sacct -j <job_id>` — if `State=PREEMPTED`, it will requeue |
| Parse failures > 10% on a benchmark | Scorer bug OR temperature too high | Inspect `real_study/results/{run_id}.json`, check `details` of BenchmarkScore |
| vLLM OOM | Too many concurrent seqs or too-long context | Reduce `--max-num-seqs` or `--max-model-len` |
