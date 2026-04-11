# Phase 11 — Execution Stages

A staged rollout protects against wasted compute if something is wrong in the pipeline. Each stage has a gate; the next stage does not start until the previous stage passes its gate.

## Stage A — Environment probe and smoke (1 wall-clock day, < 0.5 GPU-day)

### A.1 Hyak probe
- SSH into `klone.hyak.uw.edu`.
- `bash real_study/harness/scripts/probe_hyak.sh`
- Review `real_study/logs/hyak_probe_*.log`.
- Update `real_study/harness/slurm/submit_smoke.sh` with the correct `--account`, `--partition`, and `--gpus` flags based on the probe.

### A.2 Environment setup
- Create a venv at `/gscratch/${group}/agentbreed_venv` (fast local disk).
- `pip install -e ".[dev,charts]"` inside the venv.
- `pip install vllm httpx optuna statsmodels scikit-optimize scipy numpy`.
- Pre-download model weights to `/gscratch/${group}/hf_cache`:
  ```bash
  HF_HOME=/gscratch/${group}/hf_cache huggingface-cli download Qwen/Qwen3-32B
  HF_HOME=/gscratch/${group}/hf_cache huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
  ```

### A.3 Benchmark snapshots
- Download ForecastBench snapshot → `real_study/results/snapshots/forecastbench_frozen.json`.
- Download LiveCodeBench v6 → `real_study/results/snapshots/livecodebench_v6_frozen.json`.
- Download GPQA Diamond → `real_study/results/snapshots/gpqa_diamond_frozen.json`.
- Compute SHA-256 for each and commit to the preregistration's amendment log (as a dated informational entry, not an amendment to hypotheses).

### A.4 Smoke test
- `sbatch real_study/harness/slurm/submit_smoke.sh`
- This runs full_evolution with population=4, generations=2 on ForecastBench × Qwen3-32B.
- Expected wall-clock: ~20 minutes on 1× H100.

### Stage A gate

All of these must be true to proceed to Stage B:
1. Slurm job exited 0.
2. `real_study/logs/smoke_${JOB_ID}/smoke_summary.json` contains a plausible champion fitness (between 0.3 and 0.95 on ForecastBench).
3. Per-call tokens per second match assumption within 25% (~1000 tok/s for Qwen3-32B).
4. Contamination probe dropped < 10% of probed items.
5. Two reruns of the same cell give identical fitness (deterministic seeding works).
6. vLLM did not crash during the smoke run.

If any gate fails: debug, re-probe, do not proceed to Stage B.

## Stage B — Pilot (2 wall-clock days, ~1.5 GPU-day)

### B.1 Single-model single-benchmark pilot
- Run all 8 methods × ForecastBench × Qwen3-32B × 5 seeds = 40 cells.
- Uses full `population_size = 20, generations = 15`.
- Submit via job array: `real_study/harness/slurm/submit_pilot.sh` (script to be written at Stage A time — template below).

### B.2 Analysis dry-run
- Run `python real_study/harness/stats/analysis.py --stage pilot` on the 40-cell output.
- Verify the LMM converges on a 40-observation dataset.
- Verify the TOST bounds are computed.
- Verify figures render without errors.

### Stage B gate
1. All 40 pilot cells completed with no errors.
2. LMM converged.
3. Pilot per-seed variance ≤ 0.08 (anything higher suggests a parser bug or flaky vLLM).
4. Champion test scores are in the expected range (ForecastBench usually produces 0.7–0.85 with Qwen3-32B).
5. `full_evolution` vs `prompt_only_evolution` has a visible gap (even if non-significant at n=5).

If any gate fails: debug, do not proceed.

## Stage C — Main confirmatory sweep (5–14 wall-clock days, ~7 GPU-days compute)

### C.1 Launch the full main matrix
- 8 methods × 3 benchmarks × 2 models × 15 seeds = **720 cells**.
- Submit as 6 Slurm job arrays (one per (model, benchmark)), each with 120 tasks.
- **Locked:** no new methods, no benchmark changes, no hyperparameter tuning.
- `real_study/harness/slurm/submit_main.sh` (template below).

### C.2 Checkpoint monitoring
- Every 2 hours, `python real_study/harness/scripts/progress_report.py` — summarizes which cells are done, which are running, which have failed.
- Failed cells are re-queued automatically as long as they are reproducible (same config re-runs should be bit-identical).

### Stage C gate
1. All 720 cells have a valid JSON in `results/main_matrix/`.
2. No parse-failure rate exceeds 10% on any benchmark for any (model, method, seed) cell.
3. Per-run vLLM wall-time matches Stage B estimates within 30%.

## Stage D — E_decisive + H3 Sobol + transfer (2–7 wall-clock days, ~9 GPU-days)

Run in parallel with Stage C if GPU allocation allows.

### D.1 E_decisive
- 2 methods × 2 benchmarks × 2 models × 6 K × 15 seeds = 720 cells.
- Uses the same runner with the `--sweep-k K` flag to freeze (9−K) genes.

### D.2 H3 Sobol
- Pick 2 `(b, l)` cells from the main matrix where Stage C shows the cleanest signal.
- Run Saltelli-Sobol with `n_base = 2048` → ~45k evaluations per cell.

### D.3 Transfer matrix
- Evaluate each main-matrix champion on every other `(b, l)` cell's test set.
- ~10k calls total across all combinations.

### Stage D gate
1. E_decisive completes all 720 cells.
2. H3 Sobol produces a valid decomposition (`sum ≤ 1.05`).
3. Transfer evaluations complete.

## Stage E — Exploratory analyses (1–3 wall-clock days, ~2 GPU-days)

- Budget-scaling curves
- Category ablations
- Seed-count sensitivity
- Wall-clock-normalized ranking
- Failure-mode qualitative review

**Labeled as exploratory in every output path.**

## Stage F — Analysis and paper (1 wall-clock day, 0 GPU-days)

- `python real_study/harness/stats/analysis.py --stage main`
- Generates all preregistered tables and figures.
- **Frozen at this point.** Any further tweaks require a dated amendment.
- Paper draft written against the locked analysis output.

## Running ledger

Every stage writes a JSON entry to `real_study/logs/ledger.jsonl`:

```json
{
    "timestamp": "2026-04-XX T XX:XX:XX Z",
    "stage": "C",
    "sub_stage": "main_matrix",
    "event": "stage_complete",
    "cells_total": 720,
    "cells_ok": 720,
    "cells_failed": 0,
    "gpu_hours_consumed": 168.4,
    "slurm_job_ids": ["12345", "12346", ...]
}
```

The ledger is the reproducibility source of truth.

## Rerun policy

Any cell that fails can be rerun without contamination, because:
- run_id is a hash of (method, benchmark, model, seed, code_sha, prereg_sha, snapshot_sha).
- Rerunning the same config produces the same run_id, overwriting the previous (tmp + rename) atomic JSON.
- Slurm arrays skip cells whose output JSON already exists in `results/main_matrix/`.

## Stop conditions

The pipeline stops (and is reported to the user) if any of:
- A gate fails and cannot be fixed within 24 hours.
- The GPU allocation is revoked.
- A parse-failure rate above 20% on any benchmark indicates a real scoring bug.
- Stage C runs 1.5× over budget.

## Timeline summary

| Stage | Wall-clock | GPU compute | Gate criteria |
|---|---|---|---|
| A — Probe + smoke | 1 day | 0.5 GPU-day | vLLM up, contamination < 10%, smoke runs clean |
| B — Pilot | 2 days | 1.5 GPU-days | 40 cells OK, LMM converges |
| C — Main matrix | 5–14 days | 7 GPU-days | 720 cells OK |
| D — Decisive + Sobol + transfer | 2–7 days | 9 GPU-days | All cells OK, Sobol valid |
| E — Exploratory | 1–3 days | 2 GPU-days | best-effort |
| F — Analysis + paper | 1 day | 0 | Frozen |
| **Total** | **12–28 days wall-clock** | **~20 GPU-days** | |

Fits comfortably within the NeurIPS 2026 window assuming the Stage A start is within 6 weeks of the deadline.
