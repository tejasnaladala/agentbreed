#!/bin/bash
# DEPRECATED — this was the original Slurm template written before we had the
# Hyak klone probe. It targeted a generic "H100" config that does not exist
# on Hyak. Do not use.
#
# Use these Hyak-specific scripts instead:
#
#   submit_smoke_tiny.sh       Stage A1  — tiny model, cheap GPU, ~5 min
#   submit_smoke_qwen32b.sh    Stage A2  — Qwen3-32B on H200, ~30 min
#   submit_smoke_llama70b.sh   Stage A3  — Llama-3.3-70B AWQ on H200, ~40 min
#
# See real_study/harness/README_HYAK.md and real_study/docs/PROBE_FINDINGS.md
# for the actual cluster setup and why this file was replaced.

echo "ERROR: submit_smoke.sh is deprecated. Use submit_smoke_tiny.sh or submit_smoke_qwen32b.sh." >&2
exit 64
