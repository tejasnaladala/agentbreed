#!/bin/bash
# Stage A tiny smoke job for Hyak klone.
# Goal: verify the end-to-end stack (vllm serves, benchmark loads, LLM call + score
#       round-trip works) in ~5 minutes on a cheap GPU before burning real H200 time.
#
# Model: Qwen/Qwen2.5-0.5B-Instruct (~1 GB, loads in seconds)
# GPU:   1x L40 48GB (or A40 / any available GPU)
# Partition: ckpt (broadest availability)
# Account:   stf-ckpt
#
# USAGE (from agentbreed repo root on Hyak):
#   sbatch real_study/harness/slurm/submit_smoke_tiny.sh
#
# What this job does:
#   1. Activates the uv venv at /gscratch/stf/naladala/agentbreed/real_study_v1/.venv
#   2. Starts vLLM serving Qwen2.5-0.5B on port 8000 as a background process
#   3. Waits up to 120 s for /v1/models to respond
#   4. Makes a single test LLM call via httpx
#   5. Validates the genome schema round-trip (validate, hash, serialize)
#   6. Writes logs to real_study/logs/smoke_tiny_${SLURM_JOB_ID}/
#   7. Tears down vllm cleanly

#SBATCH --job-name=ab_tiny
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=real_study/logs/smoke_tiny_%j.out
#SBATCH --error=real_study/logs/smoke_tiny_%j.err

set -euo pipefail

# Slurm copies batch scripts to a spool dir before running them, so
# ${BASH_SOURCE[0]} does NOT resolve to the original script location.
# Use SLURM_SUBMIT_DIR (set by Slurm to the sbatch invocation directory)
# as the repo root. If running outside Slurm (direct bash invocation),
# fall back to the script's own directory.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"
echo "[$(date -Is)] REPO_ROOT=${REPO_ROOT}"
echo "[$(date -Is)] pwd=$(pwd)"
# Sanity: the repo root must contain real_study/
if [ ! -d "real_study" ]; then
    echo "FATAL: real_study/ not found in REPO_ROOT=${REPO_ROOT}" >&2
    echo "contents of REPO_ROOT:" >&2
    ls -la "${REPO_ROOT}" >&2
    exit 10
fi

# Log directory for this job
LOG_DIR="real_study/logs/smoke_tiny_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"
HARNESS_LOG="${LOG_DIR}/harness.log"
VLLM_LOG="${LOG_DIR}/vllm.log"

echo "[$(date -Is)] === Stage A tiny smoke job ===" | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] node: $(hostname)" | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] cwd:  $(pwd)" | tee -a "${HARNESS_LOG}"
nvidia-smi 2>&1 | head -15 | tee -a "${HARNESS_LOG}" || true

# Activate the venv
VENV=/gscratch/stf/naladala/agentbreed/real_study_v1/.venv
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "[$(date -Is)] FATAL: venv not found at ${VENV}" | tee -a "${HARNESS_LOG}"
    exit 2
fi
source "${VENV}/bin/activate"
echo "[$(date -Is)] venv: $(which python)" | tee -a "${HARNESS_LOG}"
python --version 2>&1 | tee -a "${HARNESS_LOG}"

# Force unbuffered Python output so vllm.log captures everything in real time,
# even if vllm crashes before it would normally flush.
export PYTHONUNBUFFERED=1

# Force vllm + huggingface_hub to use ONLY the local cache.
# Requires that the model is pre-downloaded on the login node into HF_HOME.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Diagnostics
echo "[$(date -Is)] vllm binary: $(which vllm)" | tee -a "${HARNESS_LOG}"
vllm --version 2>&1 | head -3 | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] HF_HOME=${HF_HOME:-unset}" | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] HF cache contents:" | tee -a "${HARNESS_LOG}"
find "${HF_HOME:-/gscratch/stf/naladala/cache/huggingface}" -maxdepth 4 -type d 2>&1 | head -20 | tee -a "${HARNESS_LOG}"

# Start vLLM serving a tiny model
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000
echo "[$(date -Is)] starting vllm server for ${MODEL} on port ${PORT}" | tee -a "${HARNESS_LOG}"

# Use stdbuf to force line-buffered output from vllm's stdio too, so the log
# captures startup messages immediately.
stdbuf -oL -eL vllm serve "${MODEL}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --tensor-parallel-size 1 \
    --max-num-seqs 8 \
    --max-model-len 4096 \
    --dtype float16 \
    --download-dir "${HF_HOME:-/gscratch/stf/naladala/cache/huggingface}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[$(date -Is)] vllm started as PID ${VLLM_PID}" | tee -a "${HARNESS_LOG}"

# Teardown hook: kill vllm no matter how we exit
trap 'echo "[$(date -Is)] trap: killing vllm pid ${VLLM_PID}" | tee -a "${HARNESS_LOG}"; kill -TERM "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

# Wait for vllm to be responsive (10 min cap, 5-second poll)
echo "[$(date -Is)] waiting for vllm /v1/models endpoint" | tee -a "${HARNESS_LOG}"
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        READY=1
        echo "[$(date -Is)] vllm ready after ${i} probe(s) (~$((i * 5)) s)" | tee -a "${HARNESS_LOG}"
        break
    fi
    # Every 12 polls (60 s), check if vllm is still alive and report
    if [ $((i % 12)) -eq 0 ]; then
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[$(date -Is)] still waiting (~$((i * 5)) s elapsed, vllm PID ${VLLM_PID} alive)" | tee -a "${HARNESS_LOG}"
            echo "[$(date -Is)] vllm.log size: $(stat -c%s "${VLLM_LOG}" 2>/dev/null || echo 0) bytes" | tee -a "${HARNESS_LOG}"
            echo "[$(date -Is)] last 5 vllm.log lines:" | tee -a "${HARNESS_LOG}"
            tail -5 "${VLLM_LOG}" 2>&1 | sed 's/^/    /' | tee -a "${HARNESS_LOG}"
        else
            echo "[$(date -Is)] FATAL: vllm PID ${VLLM_PID} is DEAD after ~$((i * 5)) s" | tee -a "${HARNESS_LOG}"
            echo "--- full vllm.log ---" | tee -a "${HARNESS_LOG}"
            cat "${VLLM_LOG}" 2>&1 | tee -a "${HARNESS_LOG}" || echo "vllm.log is empty" | tee -a "${HARNESS_LOG}"
            exit 4
        fi
    fi
    sleep 5
done

if [ "${READY}" -ne 1 ]; then
    echo "[$(date -Is)] FATAL: vllm did not become ready within 10 minutes" | tee -a "${HARNESS_LOG}"
    echo "--- full vllm.log ---" | tee -a "${HARNESS_LOG}"
    cat "${VLLM_LOG}" 2>&1 | tee -a "${HARNESS_LOG}" || echo "vllm.log is empty" | tee -a "${HARNESS_LOG}"
    echo "--- last 50 lines of any python output anywhere ---" | tee -a "${HARNESS_LOG}"
    find "${LOG_DIR}" -type f -name "*.log" -exec ls -la {} \; | tee -a "${HARNESS_LOG}"
    exit 3
fi

# Run the Python-side smoke test
echo "[$(date -Is)] running Python smoke test" | tee -a "${HARNESS_LOG}"
python -u real_study/harness/scripts/smoke_test_tiny.py \
    --model "${MODEL}" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --benchmark forecastbench \
    --out "${LOG_DIR}/smoke_tiny_result.json" \
    2>&1 | tee -a "${HARNESS_LOG}"
PYTHON_EXIT=$?

echo "[$(date -Is)] python smoke exit=${PYTHON_EXIT}" | tee -a "${HARNESS_LOG}"

if [ "${PYTHON_EXIT}" -eq 0 ]; then
    echo "[$(date -Is)] === STAGE A TINY SMOKE: PASS ===" | tee -a "${HARNESS_LOG}"
    exit 0
else
    echo "[$(date -Is)] === STAGE A TINY SMOKE: FAIL ===" | tee -a "${HARNESS_LOG}"
    exit 1
fi
