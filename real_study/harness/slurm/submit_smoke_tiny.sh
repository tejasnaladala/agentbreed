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

# Resolve the repo root from the script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

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

# Start vLLM serving a tiny model
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000
echo "[$(date -Is)] starting vllm server for ${MODEL} on port ${PORT}" | tee -a "${HARNESS_LOG}"

vllm serve "${MODEL}" \
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

# Wait for vllm to be responsive
echo "[$(date -Is)] waiting for vllm /v1/models endpoint" | tee -a "${HARNESS_LOG}"
READY=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        READY=1
        echo "[$(date -Is)] vllm ready after ${i} probe(s)" | tee -a "${HARNESS_LOG}"
        break
    fi
    sleep 5
done

if [ "${READY}" -ne 1 ]; then
    echo "[$(date -Is)] FATAL: vllm did not become ready within 5 minutes" | tee -a "${HARNESS_LOG}"
    echo "--- last 30 lines of vllm.log ---" | tee -a "${HARNESS_LOG}"
    tail -30 "${VLLM_LOG}" 2>&1 | tee -a "${HARNESS_LOG}" || true
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
