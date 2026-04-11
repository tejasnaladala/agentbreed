#!/bin/bash
# Smoke-test Slurm job: 1 model × 1 benchmark × full_evolution × seed=1.
# Runs on Hyak klone after Stage A environment setup.
#
# USAGE (from login node):
#   sbatch real_study/harness/slurm/submit_smoke.sh
#
# The job:
#   1. Activates the Python env
#   2. Starts a vllm server for Qwen3-32B as a background process
#   3. Waits for vllm to become responsive
#   4. Runs real_study/harness/scripts/smoke_test.py
#   5. Captures logs into real_study/logs/smoke_${SLURM_JOB_ID}/
#   6. Tears down vllm cleanly
#
# Hyak-specific notes:
#   * Adjust --account, --partition, and --gpus based on the probe output.
#   * Adjust --time as needed; smoke test should finish in under 60 min.

#SBATCH --job-name=ab_smoke
#SBATCH --account=CHANGE_ME_GROUP
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=real_study/logs/smoke_%j.out
#SBATCH --error=real_study/logs/smoke_%j.err

set -euo pipefail

cd "$(dirname "$0")/../../.."

# Environment
module load python/3.11  2>/dev/null || true
source .venv/bin/activate

mkdir -p real_study/logs "real_study/logs/smoke_${SLURM_JOB_ID}"
LOG_DIR="real_study/logs/smoke_${SLURM_JOB_ID}"

# Start vllm
MODEL="Qwen/Qwen3-32B"
VLLM_PORT=8000
VLLM_LOG="${LOG_DIR}/vllm.log"

echo "[$(date)] starting vllm server for ${MODEL}" | tee -a "${LOG_DIR}/harness.log"
vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --dtype float16 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# Wait until vllm is responsive
echo "[$(date)] waiting for vllm to respond" | tee -a "${LOG_DIR}/harness.log"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
        echo "[$(date)] vllm is up (attempt $i)" | tee -a "${LOG_DIR}/harness.log"
        break
    fi
    sleep 5
done

# Run the smoke test
echo "[$(date)] running smoke test" | tee -a "${LOG_DIR}/harness.log"
python -u real_study/harness/scripts/smoke_test.py \
    --benchmark forecastbench \
    --model "${MODEL}" \
    --method full_evolution \
    --seed 1 \
    --population-size 4 \
    --generations 2 \
    --results-dir "${LOG_DIR}/results" \
    2>&1 | tee -a "${LOG_DIR}/harness.log"
SMOKE_EXIT=$?

# Teardown
echo "[$(date)] stopping vllm" | tee -a "${LOG_DIR}/harness.log"
kill -TERM "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true

echo "[$(date)] smoke test exit=${SMOKE_EXIT}" | tee -a "${LOG_DIR}/harness.log"
exit "${SMOKE_EXIT}"
