#!/bin/bash
# Stage A2 real-smoke job: Qwen3-32B on H200 via Hyak klone ckpt-g2 partition.
# Purpose: verify Qwen3-32B serves correctly, measure actual throughput, make
# a handful of real LLM calls end-to-end.
#
# Time budget: ~30 min wall clock, ~0.3 GPU-day.
#
# Pre-requisite: model weights already cached in $HF_HOME. Run this once:
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('Qwen/Qwen3-32B', \
#                 cache_dir='/gscratch/stf/naladala/cache/huggingface')"
#
# USAGE:
#   cd /gscratch/stf/naladala/agentbreed/repo
#   sbatch real_study/harness/slurm/submit_smoke_qwen32b.sh

#SBATCH --job-name=ab_sq32
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=h200:1
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=real_study/logs/smoke_qwen32b_%j.out
#SBATCH --error=real_study/logs/smoke_qwen32b_%j.err

set -euo pipefail

# Slurm copies batch scripts to a spool dir before running them.
# Use SLURM_SUBMIT_DIR to locate the repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"
echo "[$(date -Is)] REPO_ROOT=${REPO_ROOT}"
if [ ! -d "real_study" ]; then
    echo "FATAL: real_study/ not found in REPO_ROOT=${REPO_ROOT}" >&2
    ls -la "${REPO_ROOT}" >&2
    exit 10
fi

LOG_DIR="real_study/logs/smoke_qwen32b_${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_DIR}"
HARNESS_LOG="${LOG_DIR}/harness.log"
VLLM_LOG="${LOG_DIR}/vllm.log"

echo "[$(date -Is)] === Stage A2 smoke: Qwen3-32B ===" | tee -a "${HARNESS_LOG}"
echo "[$(date -Is)] node: $(hostname)" | tee -a "${HARNESS_LOG}"
nvidia-smi 2>&1 | head -15 | tee -a "${HARNESS_LOG}" || true

# Verify H200
if ! nvidia-smi 2>&1 | grep -qi "H200"; then
    echo "[$(date -Is)] WARNING: expected H200, nvidia-smi shows different GPU" | tee -a "${HARNESS_LOG}"
    nvidia-smi 2>&1 | grep -E "NVIDIA|Tesla" | tee -a "${HARNESS_LOG}"
fi

# Activate the venv
VENV=/gscratch/stf/naladala/agentbreed/real_study_v1/.venv
source "${VENV}/bin/activate"

# Start vLLM
MODEL="Qwen/Qwen3-32B"
PORT=8000
echo "[$(date -Is)] starting vllm server for ${MODEL} on port ${PORT}" | tee -a "${HARNESS_LOG}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --tensor-parallel-size 1 \
    --max-num-seqs 32 \
    --max-model-len 8192 \
    --dtype float16 \
    --download-dir "${HF_HOME:-/gscratch/stf/naladala/cache/huggingface}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[$(date -Is)] vllm started as PID ${VLLM_PID}" | tee -a "${HARNESS_LOG}"

trap 'echo "[$(date -Is)] trap: killing vllm pid ${VLLM_PID}" | tee -a "${HARNESS_LOG}"; kill -TERM "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

# Wait for vLLM to be responsive (up to 10 min for 32B model load)
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        READY=1
        echo "[$(date -Is)] vllm ready after ${i} probe(s) (~$((i*5)) s)" | tee -a "${HARNESS_LOG}"
        break
    fi
    sleep 5
done

if [ "${READY}" -ne 1 ]; then
    echo "[$(date -Is)] FATAL: vllm did not become ready within 10 minutes" | tee -a "${HARNESS_LOG}"
    tail -50 "${VLLM_LOG}" 2>&1 | tee -a "${HARNESS_LOG}" || true
    exit 3
fi

# Throughput probe: 16 concurrent dummy calls, measure tok/s
echo "[$(date -Is)] running throughput probe" | tee -a "${HARNESS_LOG}"
python -u real_study/harness/scripts/throughput_probe.py \
    --model "${MODEL}" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --concurrent 16 \
    --n-calls 64 \
    --max-tokens 200 \
    --out "${LOG_DIR}/throughput.json" \
    2>&1 | tee -a "${HARNESS_LOG}"

# Tiny end-to-end smoke with real Qwen3-32B
echo "[$(date -Is)] running end-to-end smoke" | tee -a "${HARNESS_LOG}"
python -u real_study/harness/scripts/smoke_test_tiny.py \
    --model "${MODEL}" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --benchmark forecastbench \
    --out "${LOG_DIR}/smoke_result.json" \
    2>&1 | tee -a "${HARNESS_LOG}"
PYTHON_EXIT=$?

if [ "${PYTHON_EXIT}" -eq 0 ]; then
    echo "[$(date -Is)] === STAGE A2 SMOKE: PASS ===" | tee -a "${HARNESS_LOG}"
    exit 0
else
    echo "[$(date -Is)] === STAGE A2 SMOKE: FAIL ===" | tee -a "${HARNESS_LOG}"
    exit 1
fi
