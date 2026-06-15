# Reproducibility Instructions

This document explains how to reproduce every number, figure, and table from a fresh clone.

> **Two analyses in this package.** Sections 3–8 reproduce the single-domain workshop pilot. Section 12 reproduces the main-track analysis (3 domains × 10 methods × 20 seeds, preregistered H1/H2/H3). If you are here for the main-track numbers, skip to §12.
>
> The paper drafts themselves are not in this repo — they target a blind submission and are kept out of the public tree. Everything below reproduces the numbers, figures, and tables the drafts reference, directly from the committed `breed` library and the scripts in `paper/03_experiments/`.

## 1. Environment

```bash
git clone <repo-url> agentbreed
cd agentbreed
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,charts]"
```

Requirements: Python 3.11+, ~500 MB disk. No GPU. No API keys.

## 2. Verify the installation

```bash
python -m pytest tests/ -q
```

Expected: **431 tests passing in ~16 seconds**. If any test fails, the pipeline below will not produce the documented numbers.

## 3. Run the main experiment

```bash
python -u paper/03_experiments/scripts/run_main_experiment.py \
    --population 20 \
    --generations 15 \
    --seeds 1 2 3 4 5 6 7 8 \
    --output-dir paper/04_results/logs/main_experiment
```

Expected runtime: **~12 seconds** on a single workstation.

This produces:
- `paper/04_results/logs/main_experiment/runs/{method}__seed{N}.json` — one file per (method, seed) pair, 56 files total.
- `paper/04_results/logs/main_experiment/results.jsonl` — aggregated results in JSON Lines format.

## 4. Run the analysis

```bash
python -u paper/03_experiments/scripts/analyze_results.py
```

Expected runtime: **~5 seconds**.

This produces:
- `paper/04_results/figures/fig1_fitness_curves.pdf` — Figure 1
- `paper/04_results/figures/fig2_ablation_bars.pdf` — Figure 2
- `paper/04_results/figures/fig3_cost_performance.pdf` — Figure 3
- `paper/04_results/tables/table1_summary.md` — Table 1
- `paper/04_results/tables/table2_paired_comparisons.md` — Table 2
- `paper/04_results/analysis_summary.json` — full statistical analysis

## 5. Verify the paper's headline numbers

Open `paper/04_results/tables/table2_paired_comparisons.md` and confirm:

- Full evolution vs. Prompt-only evolution: diff = +0.0538, *d<sub>z</sub>* = 1.451, raw *p* = 0.0045, Holm *p* = 0.0273 (**significant**)
- Full evolution vs. Mutation only: diff = +0.0235, *d<sub>z</sub>* = 0.418, raw *p* = 0.275
- Full evolution vs. Crossover only: diff = −0.0056, *d<sub>z</sub>* = −0.094, raw *p* = 0.799
- Full evolution vs. Random search: diff = +0.0208, *d<sub>z</sub>* = 0.443, raw *p* = 0.250
- Full evolution vs. Best random init: diff = −0.0047, *d<sub>z</sub>* = −0.100, raw *p* = 0.786
- Full evolution vs. Static ensemble: diff = +0.0207, *d<sub>z</sub>* = 0.544, raw *p* = 0.168

These should match to 4 decimal places given the fixed seeds. The pipeline
is fully deterministic as of commit v0.1.0+det-fix. If your numbers differ,
check that you are on the latest commit (the earlier prototype used
`uuid.uuid4()` in the synthetic agent which broke reproducibility).

## 6. Run the synthetic agent sanity tests

```bash
python paper/03_experiments/scripts/test_synthetic_agent.py
```

Expected output:
```
PASS: minimal genome has low skill
PASS: optimal genome has high skill
PASS: cross-component interaction bonus fires
PASS: high-skill beats low-skill on Brier
PASS: agent is deterministic
PASS: random genomes span meaningful skill range
ALL SANITY TESTS PASSED
```

## 7. Expected artifacts

After running steps 3-4, your `paper/04_results/` directory should contain:

```
paper/04_results/
├── analysis_summary.json
├── figures/
│   ├── fig1_fitness_curves.pdf
│   ├── fig1_fitness_curves.png
│   ├── fig2_ablation_bars.pdf
│   ├── fig2_ablation_bars.png
│   └── fig3_cost_performance.pdf
├── logs/
│   └── main_experiment/
│       ├── results.jsonl       (57 lines: 1 header + 56 runs)
│       └── runs/               (56 JSON files)
└── tables/
    ├── table1_summary.md
    └── table2_paired_comparisons.md
```

## 8. Determinism notes

All runs are deterministic given the seeds:
- `run_main_experiment.py` uses `seeds=[1..8]` by default.
- Each seed seeds a `random.Random` chain that propagates deterministically through population spawning, selection, crossover, mutation, and immigration.
- The synthetic agent is deterministic via SHA-256 hashing of `(genome_id, task, purpose)`.
- Bootstrap confidence intervals use a fixed bootstrap seed of 42 in the analysis script.

If you run with different seeds, the per-seed numbers will change but the overall qualitative findings (full > prompt-only by a large margin; full ≈ mutation-only ≈ random search) should persist at *n* ≥ 6.

## 9. Running on a real LLM

The synthetic agent can be replaced with a real LLM adapter. Example replacing the agent in `run_main_experiment.py`:

```python
from breed.adapters.anthropic_adapter import AnthropicMessagesAdapter
# OR
from breed.adapters.openai_adapter import OpenAIAdapter

# Replace make_synthetic_agent(...) with:
adapter = AnthropicMessagesAdapter(model="claude-sonnet-4-6")
# or
adapter = OpenAIAdapter(model="gpt-4o")

# Then wire it into ExperimentRunner via agent_fn
```

Expected cost for a single full run with 8 seeds, 7 methods, 20 pop × 15 gen, 50 test questions: approximately 30,000-60,000 LLM calls. At ~$0.01/call this is ~$300-600 per full experimental run.

## 10. Known issues

- **Windows legacy terminals:** The CLI banner uses ASCII box-drawing characters that may render incorrectly in cp1252 terminals. All data files are plain JSON/PDF and unaffected.
- **Git Bash stdout buffering on Windows:** If you run the experiment script through `python ... | tail` in Git Bash, stdout may not flush until the subprocess exits. Use `-u` (unbuffered) or redirect to a file to see real-time progress.

## 11. Contact

Issues and questions: [REPO URL]/issues

---

## 12. Main-track analysis reproduction

The main-track analysis has preregistered H1/H2/H3 (locked 2026-04-10 in `paper/01_corpus/preregistration.md`). It subsumes the workshop pipeline; the workshop numbers in §3–8 are preserved only for historical comparison.

### 12.1 Run every main-track experiment

```bash
python -u paper/03_experiments/scripts/run_main_track_experiments.py --which all
```

Expected runtime: **~400 seconds** on a single CPU. Produces per-run JSON files under:

```
paper/04_results/E1/           # 600 runs: 3 domains × 10 methods × 20 seeds
paper/04_results/E_decisive/   # 480 runs: 2 domains × 2 methods × 6 K-values × 20 seeds
paper/04_results/E2/           # 100 runs: gene dropout ablation
paper/04_results/E3/           # Sobol epistasis (n_base=512 by default)
paper/04_results/E_transfer/   # 3×3 champion transfer matrix
```

If the runner stalls on Windows (documented Git Bash subprocess issue after ~200 runs), chunk it:

```bash
python -u paper/03_experiments/scripts/run_chunk.py --max-runs 200
```

…and re-invoke until "no remaining runs."

### 12.2 Run the main-track analysis

```bash
python -u paper/03_experiments/scripts/analyze_main_track.py
```

Expected runtime: **~5 seconds**. Produces:

- `paper/04_results/tables/main_track/table_E1_summary.md`
- `paper/04_results/tables/main_track/table_E1_comparisons.md` ← the centerpiece
- `paper/04_results/tables/main_track/table_E_decisive.md`
- `paper/04_results/tables/main_track/table_transfer.md`
- `paper/04_results/figures/main_track/fig_ablation_{coding,forecasting,knowledge}.{pdf,png}`
- `paper/04_results/figures/main_track/fig_decisive_{coding,forecasting}.{pdf,png}`
- `paper/04_results/figures/main_track/fig_e2_dropout.{pdf,png}`
- `paper/04_results/figures/main_track/fig_transfer_matrix.{pdf,png}`
- `paper/04_results/analysis_main_track.json`

### 12.3 Verify the headline numbers

Open `paper/04_results/tables/main_track/table_E1_comparisons.md` and confirm these cells exist:

| Domain | Baseline | Diff | Holm *p* | *d<sub>z</sub>* |
|---|---|---|---|---|
| coding | Prompt-only evolution | +0.4327 | 0.0000 | 6.064 |
| coding | Static ensemble | +0.2642 | 0.0000 | 3.573 |
| coding | Best random init | +0.1115 | 0.0008 | 1.087 |
| coding | Mutation only | +0.0368 | 0.5282 (ns) | 0.352 |
| coding | Random search | −0.0102 | 0.7437 (ns) | −0.111 |
| coding | bayesian_opt | −0.0350 | 0.4655 (ns) | −0.395 |
| knowledge | Prompt-only evolution | +0.3187 | 0.0000 | 3.098 |
| knowledge | Static ensemble | +0.1378 | 0.0001 | 1.349 |
| forecasting | Static ensemble | +0.0245 | 0.0137 | 0.827 |

Also verify the pooled cross-domain aggregate (in `analysis_main_track.json` under `aggregate.full_vs_prompt_only`):

- Mean diff `+0.2549`, 95% CI `[+0.206, +0.303]`, *t* `11.5`, *p* `≈ 1e-14`, *d<sub>z</sub>* `1.322`.

These are the H1 confirmation numbers reported in the main-track abstract.

### 12.4 Preregistration check

```bash
git log --follow paper/01_corpus/preregistration.md
```

The first commit of `preregistration.md` must predate any file under `paper/04_results/E1/`, `E_decisive/`, `E2/`, `E3/`, or `E_transfer/`. This is the audit trail that makes H1/H2/H3 confirmatory rather than exploratory.
