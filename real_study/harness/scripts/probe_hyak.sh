#!/bin/bash
# Hyak klone environment probe. Run this first thing after SSHing in.
# Writes a JSON report to real_study/logs/hyak_probe_$(date).json.
#
# USAGE (on Hyak login node):
#   bash real_study/harness/scripts/probe_hyak.sh
#
# The probe captures:
#   - Slurm partitions and GPU types available
#   - User's account / allocation
#   - Storage quotas
#   - Python / module availability
#   - vLLM / torch install status (if venv is active)
# NO secrets or tokens are captured.

set -euo pipefail

cd "$(dirname "$0")/../../.."

DATE=$(date +%Y%m%d_%H%M%S)
OUT="real_study/logs/hyak_probe_${DATE}.log"
mkdir -p real_study/logs

{
    echo "=== Hyak klone probe at $(date) ==="
    echo "=== hostname ==="; hostname
    echo "=== whoami ==="; whoami
    echo "=== pwd ==="; pwd
    echo
    echo "=== sinfo (partitions + GPU types) ==="
    sinfo -o "%P %G %N %D %t" 2>&1 | head -60
    echo
    echo "=== sacctmgr show assoc ==="
    sacctmgr show assoc user="$USER" format=Account,Partition,QOS 2>&1 || echo "sacctmgr unavailable"
    echo
    echo "=== quota (group storage) ==="
    quota -g 2>&1 || echo "quota -g unavailable"
    echo
    echo "=== gscratch space ==="
    df -h /gscratch /mmfs1 /tmp 2>&1 || true
    echo
    echo "=== modules ==="
    module avail python cuda 2>&1 | head -40 || echo "module system unavailable"
    echo
    echo "=== default python ==="
    which python 2>&1 || true
    python --version 2>&1 || true
    echo
    echo "=== nvidia-smi (if on GPU node) ==="
    nvidia-smi 2>&1 | head -20 || echo "nvidia-smi unavailable on login node (expected)"
    echo
    echo "=== GPU availability summary ==="
    sinfo -o "%G %D" 2>&1 | grep -E "gpu|h100|h200|a100|l40" | sort | uniq -c || true
    echo
    echo "=== done ==="
} 2>&1 | tee "${OUT}"

echo
echo "Probe complete. Log: ${OUT}"
echo "Next step: review the log, update real_study/harness/slurm/submit_smoke.sh with your account + partition, then sbatch it."
