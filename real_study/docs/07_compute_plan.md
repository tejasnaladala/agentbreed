# Phase 8 — Compute Plan for UW Hyak klone

## Environment assumption

The target cluster is **UW Hyak klone**, accessed via SSH to `klone.hyak.uw.edu`. Exact GPU allocation, partition, and quota are **unknown until the SSH probe is run**. This document assumes a reasonable configuration and flags every assumption for verification.

### Assumptions (to verify at probe time)

1. Access to a GPU partition (either a research-group allocation or the opportunistic `ckpt` queue).
2. At least some nodes with H100 80 GB GPUs. Likely 2× to 8× H100 per node.
3. At least 50 GB of per-user storage on `gscratch`.
4. Slurm version ≥ 22.x with job arrays and `--gres=gpu:h100:N` syntax.
5. Network egress for downloading model weights and benchmark data (or a shared HuggingFace cache accessible from the compute nodes).
6. Python 3.11+ available via module system, or via `miniconda` in gscratch.

### Probe script (Phase 10 will implement it)

`real_study/harness/scripts/probe_hyak.sh` — runs at first SSH login and captures:
- `sinfo -o "%P %G %N %D %t"` → partitions, GPU types, node list
- `sacctmgr show assoc user=$USER` → quota / account
- `quota -g` → group storage
- `nvidia-smi` (on a tiny GPU job) → driver, CUDA, GPU model
- `module avail python` + `which python`
- `df -h /gscratch /tmp /mmfs1` → available scratch space

Output goes to `real_study/logs/hyak_probe_{date}.log` and is committed (redacted of anything user-specific) as part of the reproducibility record.

## Budget bookkeeping

Budget cap: **30 GPU-days** over the study. This is the envelope the user committed in Phase 0. The plan below fits within that envelope with ~20% headroom for debugging.

### Cost per (model, benchmark, method, seed) cell

From Phase 3 + Phase 4, estimated LLM wall-clock per call at steady-state concurrent throughput:

| Model | ForecastBench (~200 out tok) | LiveCodeBench (~800 out tok) | GPQA Diamond (~400 out tok) | Mean |
|---|---|---|---|---|
| Qwen3-32B (1× H100, fp16) | 1.5 s | 5.5 s | 2.8 s | 3.3 s |
| Llama-3.3-70B (2× H100, fp16) | 2.5 s | 10.0 s | 4.5 s | 5.7 s |

With vLLM continuous batching at ~32 concurrent requests, effective throughput scales ~3–4×. Realistic per-call amortized cost:

| Model | Amortized per-call |
|---|---|
| Qwen3-32B | ~1.0 s |
| Llama-3.3-70B | ~1.7 s |

Per cell (300 search calls + 80 or 40 test calls ≈ 380 calls):

| Model | Per-cell wall clock |
|---|---|
| Qwen3-32B | 380 × 1.0 s = ~6.3 min |
| Llama-3.3-70B | 380 × 1.7 s = ~10.8 min |

### Main matrix cost

**Main confirmatory matrix:** `|M| × |B| × |L| × n_seeds = 8 × 3 × 2 × 15 = 720` cells.

- Qwen3-32B: 8 × 3 × 15 = 360 cells × 6.3 min = 2,268 min = **37.8 GPU-hours (1.6 GPU-days)**
- Llama-3.3-70B: 8 × 3 × 15 = 360 cells × 10.8 min = 3,888 min = **64.8 GPU-hours (2.7 GPU-days)**, running on 2× GPUs so **5.4 GPU-days of GPU-hours**

**Main matrix subtotal: 7.0 GPU-days.**

### E_decisive (dimensionality sweep) cost

2 methods (`full_evolution`, `random_search`) × 2 benchmarks (ForecastBench, LiveCodeBench) × 2 models × 6 K values × 15 seeds = 720 cells.

Same per-cell cost as above:
- Qwen3-32B: 360 × 6.3 min = 1.6 GPU-days
- Llama-3.3-70B: 360 × 10.8 min = 5.4 GPU-days (of GPU-hours including 2× TP)

**E_decisive subtotal: 7.0 GPU-days.**

### H3 Sobol cost

2 of 6 `(b, l)` cells × ~45,000 evaluations per cell × avg 1.3 s per call:
- 90,000 × 1.3 s = 117,000 s ≈ 32.5 GPU-hours = **1.4 GPU-days**

**H3 subtotal: 1.4 GPU-days.**

### Transfer matrix cost (S1 + S2)

- Champion evaluation only (no search). 9 (source × target) combinations × 3 benchmarks × 5 seeds each × 80 test items.
- `9 × 3 × 5 × 80 = 10,800` calls across both models. At ~1.3 s avg = 14,040 s = **0.16 GPU-days**

**Transfer subtotal: 0.2 GPU-days.**

### Budget-scaling curves (secondary)

3 methods (`full_evolution`, `random_search`, `best_of_20_init`) × 3 budgets (100/300/1000) × 2 benchmarks × 2 models × 8 seeds = 288 cells. At average 1.3 s per call:
- budget=100: 288/3 cells × 100 calls × 1.3 s = 12,480 s per model per benchmark
- budget=300: (already in main)
- budget=1000: 288/3 cells × 1000 calls × 1.3 s = 124,800 s
- Total: ~13 GPU-hours = **0.5 GPU-days**

**Budget-scaling subtotal: 0.5 GPU-days.**

### Category ablations (exploratory)

4 categories × 3 methods × 2 benchmarks × 2 models × 8 seeds = 384 cells. Per cell ~8 min avg.
- ~51 GPU-hours = **2.1 GPU-days**

**Exploratory ablations subtotal: 2.1 GPU-days.**

### Smoke-test + Stage B pilot cost

Small-scale probes: ~0.5 GPU-days for smoke, ~1.5 GPU-days for Stage B pilot.

**Smoke + pilot subtotal: 2.0 GPU-days.**

## Budget summary

| Stage | GPU-days |
|---|---|
| Smoke + pilot | 2.0 |
| Main matrix (confirmatory) | 7.0 |
| E_decisive | 7.0 |
| H3 Sobol | 1.4 |
| Transfer | 0.2 |
| Budget-scaling | 0.5 |
| Exploratory category ablations | 2.1 |
| **Subtotal** | **20.2** |
| Debugging / rerun headroom (25%) | 5.0 |
| **Total** | **25.2 GPU-days** |

Well inside the 30-GPU-day budget. Leaves ~5 GPU-days for unforeseen work (e.g. optional Qwen3-14B scaling run, MMLU-Pro bonus benchmark, a failed-node reruns).

## Execution schedule

Each "stage" is a self-contained block that can be run in any order, but ordering is chosen to surface failures early.

### Stage A — Smoke (1 day wall clock, 0.5 GPU-day)

- Probe Hyak klone.
- Pull Qwen3-32B and Llama-3.3-70B weights into `gscratch` cache.
- Download ForecastBench snapshot, LiveCodeBench v6, GPQA Diamond.
- Run contamination probes (Phase 3).
- Single (model, benchmark, method=full_evolution, seed=1) smoke cell per model per benchmark → 6 cells → ~1 hour.
- Verify per-run JSON logs, parse paths, reproducibility across re-runs of the same seed.
- Verify vLLM throughput matches assumption within 20%.
- **Gate:** if throughput is < 80% of assumption, revise the plan before proceeding.

### Stage B — Pilot (2 days wall clock, 1.5 GPU-day)

- Run 5 seeds × all methods × 1 benchmark (ForecastBench) × 1 model (Qwen3-32B). 8 methods × 5 seeds = 40 cells × 6.3 min ≈ 4.2 GPU-hours.
- Verify the analysis pipeline on the pilot data: LMM converges, TOST runs, bootstraps run, figure generation works.
- Sanity-check that paired differences and variance estimates are in the expected range (pilot SD ≈ 0.05).
- **Gate:** if the pipeline produces any error, fix it before Stage C. No exceptions.

### Stage C — Main confirmatory sweep (7–14 days wall clock depending on parallelism)

- Launch the full 720-cell main matrix.
- Run in Slurm job arrays, one array per (model, benchmark) — this gives 6 arrays, each with `8 methods × 15 seeds = 120 jobs`.
- Each job is a single cell. Output JSON goes to `results/main_matrix/{run_id}.json`.
- Concurrent array tasks per job: 8 (matches vLLM's preferred concurrency for our model tier).
- Wall-clock: main matrix alone is ~7 GPU-days. On 2 H100 nodes running in parallel, that's ~3.5 days of wall clock. On 4 H100 nodes, ~1.8 days.
- **No exploratory analyses during this stage.** Results are frozen at the end.

### Stage D — E_decisive + H3 + transfer (4–8 days wall clock)

- Run in parallel with Stage C if GPU availability allows. Otherwise serial.
- E_decisive is a job array of `2 × 2 × 6 × 15 × 2 = 720` cells.
- H3 Sobol is a separate job (2 × 45k evaluations).
- Transfer is a small run (10k calls total).

### Stage E — Exploratory (2–3 days wall clock)

- Budget-scaling, category ablations, sensitivity analyses.
- Labeled "exploratory" in all output paths.

### Stage F — Analysis + figures (1 day wall clock, 0 GPU-days)

- Run `real_study/harness/stats/analysis.py` on all committed results.
- Generate tables and figures.
- **Frozen analysis.** No tweaking beyond this point.

## Wall-clock estimate

| Scenario | Wall clock |
|---|---|
| Serial (1 job at a time) | ~30 days |
| 2 concurrent jobs (1 per model) | ~15 days |
| 4 concurrent jobs | ~8 days |
| 8 concurrent jobs (aggressive) | ~4 days |

With 4× H100 available on Hyak at reasonable utilization, the full main matrix completes in about **5–8 wall-clock days**. Comfortable for a NeurIPS July 2026 abstract deadline (which is typically 6–8 weeks before main deadline).

## Cost-per-dollar sanity

Hyak klone is a UW shared resource; the "cost" is GPU-hour allocation, not dollars. The dollar-equivalent of 25 GPU-days on 1× H100 at commercial rental rates (~$2/hr) would be ~$1,200. On 2× H100 for Llama-70B work, ~$2,400. Scale up by 1.3× for debugging overhead: ~$3,100 total equivalent commercial cost. This is consistent with typical NeurIPS-submission compute budgets for evaluation papers.

## Failure modes and mitigations

| Failure | Probability | Mitigation |
|---|---|---|
| Hyak partition is unavailable or quota exhausted | Medium | Fall back to `ckpt` queue with preemption-tolerant job scripts (checkpoint every cell). |
| Single node goes down mid-array | High (HPC life) | Per-cell atomic JSON writes mean we lose at most one cell. Slurm array resume-skip mode. |
| vLLM crashes under high concurrency | Medium | Cap `max-num-seqs` at 32. Watchdog script restarts the server on crash. |
| Model weights not in cache, slow download | Medium | Pre-download during Stage A; bake into cache before Stage C. |
| vLLM version incompatibility with Qwen3 kernel | Low | Pin to a vetted vLLM version in `requirements.txt`. |
| Out-of-memory on long-context coding problems | Medium | Cap `max_tokens` per gene; enforce at request construction. |
| Benchmark server down (ForecastBench) | Low | Use frozen snapshot, don't query ForecastBench at runtime. |
| Seed non-reproducibility drift | Medium | Documented tolerance (±0.008). Flag if drift exceeds. |
| We need more seeds for a tight CI | Medium | 25% headroom built in; if we need > 25% more, we drop scaling experiments first. |
| Compute allocation rotation / group preempts | Medium | Checkpointed jobs; per-cell atomic writes. |

## Reproducibility artifacts committed per cell

Each `results/{run_id}.json` contains:

```json
{
  "run_id": "sha256(...)",
  "benchmark": "forecastbench",
  "model": "Qwen3-32B",
  "model_quantization": "fp16",
  "method": "full_evolution",
  "seed": 1,
  "population_size": 20,
  "generations": 15,
  "budget_llm_calls": 300,
  "actual_llm_calls": 300,
  "total_input_tokens": 1234567,
  "total_output_tokens": 89012,
  "total_vllm_wall_time_s": 378.4,
  "gpu_type": "H100",
  "gpu_count": 1,
  "tp_size": 1,
  "started_at": "2026-04-XX T XX:XX:XX Z",
  "completed_at": "2026-04-XX T XX:XX:XX Z",
  "champion_genome": {...},
  "champion_test_score": 0.742,
  "per_generation_best": [0.612, 0.658, ...],
  "code_version": "git sha",
  "preregistration_sha": "sha of preregistration_real_v1.md at run time"
}
```

## Data transfer plan

- **In:** model weights (Qwen3-32B ≈ 64 GB; Llama-3.3-70B ≈ 140 GB; optional Qwen3-14B ≈ 28 GB) pulled from HuggingFace at Stage A to `/gscratch/.../hf_cache`.
- **Out:** `results/` directory (< 1 GB of JSONs) rsynced back to local after each stage. No model weights exfiltrated.
- **Size of final reproducibility bundle:** approximately 2 GB uncompressed (including intermediate per-call logs). Fits in a single tar.gz for artifact release.
