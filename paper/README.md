# Research Package: Lineage-Aware Configuration Search for LLM Forecasting Agents

This directory is the **complete research package** for the paper
"Lineage-Aware Configuration Search for LLM Forecasting Agents".

## Contents

```
paper/
├── README.md                        # This file
├── 01_corpus/                       # Literature review
│   ├── sources/                     # Per-topic literature notes
│   │   └── 03_agent_configuration_search.md
│   └── novelty_memo.md              # Honest novelty assessment + scope decision
├── 02_data/                         # Dataset registry (placeholder)
├── 03_experiments/
│   └── scripts/
│       ├── synthetic_agent.py       # Deterministic forecasting agent with
│       │                            #   cross-component interactions
│       ├── test_synthetic_agent.py  # Sanity tests for the synthetic agent
│       ├── run_main_experiment.py   # Main multi-seed multi-method runner
│       ├── analyze_results.py       # Statistical tests + figure generation
│       └── finish_missing_runs.py   # Recovery helper
├── 04_results/                      # Experimental artifacts
│   ├── analysis_summary.json        # Full stats + comparisons
│   ├── figures/                     # PDF + PNG publication figures
│   ├── tables/                      # Markdown tables
│   └── logs/main_experiment/        # Raw per-run JSON + aggregated JSONL
├── 05_draft/
│   └── paper.md                     # Full paper draft
└── 06_reproducibility/
    └── REPRODUCE.md                 # Step-by-step reproduction guide
```

## Quick start

```bash
# Install the library
pip install -e ".[dev,charts]"

# Run all tests (~10 seconds)
python -m pytest tests/ -q

# Run the main experiment (~12 seconds)
python -u paper/03_experiments/scripts/run_main_experiment.py \
    --population 20 --generations 15 --seeds 1 2 3 4 5 6 7 8

# Run the analysis (~5 seconds)
python -u paper/03_experiments/scripts/analyze_results.py

# View the paper
cat paper/05_draft/paper.md
```

## What's in this package

1. **Literature corpus** (`01_corpus/`): a structured review of adjacent work and a *novelty memo* that explicitly decides which claims are defensible and which are not.

2. **Experimental infrastructure** (`03_experiments/`): the scripts that turn the `breed` library into a publication-grade experimental pipeline, including the synthetic agent used for validation.

3. **Results artifacts** (`04_results/`): the raw per-run JSON files, the aggregated JSONL, the publication figures (PDF + PNG), the summary tables, and the full statistical analysis JSON. These are the *canonical* versions that `paper.md` references.

4. **Paper draft** (`05_draft/`): the full paper draft. Honest about scope, limitations, and what the experiments support versus what they do not.

5. **Reproducibility guide** (`06_reproducibility/`): every step needed to go from a clean clone to the paper's numbers, figures, and tables.

## Decision: is this worth pursuing as a paper?

**Yes, as a workshop-track contribution with honest framing.** See `01_corpus/novelty_memo.md` for the full decision rationale. Summary:

- The infrastructure is already built (422+ tests passing).
- The scientific question is genuinely open and under-studied.
- The honest answer from preliminary experiments is "multi-component search beats single-component search; crossover versus mutation is a wash at our scale."
- The released `breed` library is a useful community artifact regardless of the paper outcome.
- **Next step:** replicate these results with real LLM-instantiated agents on ForecastBench or Metaculus data at n ≥ 20 seeds before targeting a main-track venue.

## What this package does NOT contain

- Real LLM API experiments (synthetic agent only).
- Results on datasets larger than 50 questions.
- Comparisons against Halawi et al. 2024 or other SOTA forecasting systems.
- Experiments on non-forecasting domains.

These are identified as required follow-up work in the paper's Limitations section.
