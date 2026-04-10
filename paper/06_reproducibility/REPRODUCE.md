# Reproducibility Instructions

This document explains how to reproduce every number, figure, and table in the paper from a fresh clone.

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

Expected: **422+ tests passing in ~10 seconds**. If any test fails, the pipeline below will not produce the paper's numbers.

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
