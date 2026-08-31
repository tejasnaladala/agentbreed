# Phase 3 — Benchmark Suite

## Design principles

1. **Contamination-controlled.** Every benchmark either has a natural contamination-free property (future events, dated problems) or a documented filter to remove pre-training exposure.
2. **Real LLM evaluation.** Every benchmark is designed for LLM-in-the-loop scoring with well-defined metrics.
3. **Different task families.** Forecasting, coding, scientific reasoning — no shared heuristic vocabulary (unlike the synthetic pilot).
4. **Feasible at our compute budget.** Each benchmark has ≥ 80 evaluable items to give 40 train / 40 test or better, but not so many that scoring dominates the compute budget.
5. **Respected, reproducible, and legally usable.** No custom-scraped datasets, no gated academic data without access.

## Candidates considered

### Forecasting
- **ForecastBench** ([Karger et al., 2024](https://arxiv.org/abs/2409.19839))
- **Metaculus historical resolved questions (2022–2025 cutoff)** — contamination risk medium, must time-filter
- **Autocast** ([Zou et al., NeurIPS 2022](https://arxiv.org/abs/2206.15474)) — older, may be in pretraining data
- **Judgemark** — too small, not widely cited

### Coding
- **LiveCodeBench v6** ([Jain et al., 2024](https://livecodebench.github.io/)) — 400+ problems, dated from May 2023 to April 2025
- **HumanEval** (Chen et al., 2021) — **saturated and contaminated**, known to be in most pretraining sets since 2021
- **MBPP sanitized** (Austin et al., 2021) — same contamination issue
- **BigCodeBench** — complex eval, high compute per item
- **SWE-bench Verified** — too hard for our budget, too much per-item setup
- **APPS** — older, contamination risk

### Scientific reasoning / knowledge
- **GPQA Diamond** ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)) — 198 graduate-level questions, SOTA at ~94% as of Feb 2026
- **MMLU-Pro** ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)) — 12k items, 5-option, harder than MMLU
- **Humanity's Last Exam** — newest, hardest, small (2.5k items), probably too hard for our model tier
- **BigBench Hard** — older, partial contamination
- **MuSR** — multi-hop reasoning, moderately sized

## Final choice

### 1. ForecastBench (forecasting)

**Data source:** https://www.forecastbench.org/ and https://github.com/forecastingresearch/forecastbench (public GitHub repo, Apache 2.0). Leaderboard updated nightly; snapshots downloadable in JSON.

**Why this one:** it is the *only* respected forecasting benchmark that is fundamentally contamination-immune by design: all questions are about future events with no known answer at submission time. The pilot's original preregistration listed it; we use the real thing. Funded by Open Philanthropy through mid-2027, so the benchmark is alive.

**Our subset:**
- We use a **frozen snapshot** downloaded on the day the first main run begins. Snapshot hash and download timestamp committed to `results/snapshots/forecastbench_frozen.json`.
- From the snapshot, we take only **resolved** binary questions (where the ground truth is known at snapshot time).
- Strict time-based split: questions resolved before 2025-06-01 → train (search set); questions resolved 2025-06-01 to snapshot date → test. Target: 80 train / 80 test.
- If the snapshot has < 160 resolved binary items, we extend the test window backward in 1-month steps until the quotas are met, and the extension is documented.
- Non-binary (numeric / categorical / ordinal) questions are excluded from v1 for scoring simplicity.

**Scoring function:**
- Primary metric: **1 − Brier score** for binary forecasts. Brier = (p − y)². Higher is better.
- Agent outputs a probability `p ∈ [0, 1]` (parsed from structured response).
- Ground truth `y ∈ {0, 1}`.
- Per-benchmark score = mean(1 − Brier) over the 80 test items.

**Contamination / leakage concerns:** minimal, by construction. The only risk is that a pre-training cut-off later than our train/test split date could leak partial information — we address this by choosing splits that postdate the training cutoffs of Llama-3.3-70B (Dec 2024) and Qwen3-32B (early 2025). Specifically: test questions must resolve **after** `max(model_training_cutoff, 2025-06-01)`.

**Evaluation protocol:**
- Each agent run (genome, question) → single LLM call with structured prompt that requests a probability in a parseable format. Retry up to 2 times on parse failure; if 3rd attempt fails, score as the worst-case 0.25 Brier (i.e. the fitness is penalized).
- No external tool calls for v1. Some genes control reasoning scaffold (decomposition, calibration), but no web search.

**Cost per run:**
- Search: 20 population × 15 generations × 80 search-set items = 24,000 calls per (model, method, seed) cell.
- Test: 80 calls per cell.
- Total: 24,080 calls × (seconds/call varies by model) — see Phase 8.

### 2. LiveCodeBench v6 (coding)

**Data source:** https://livecodebench.github.io/ and https://huggingface.co/datasets/livecodebench/code_generation_lite (MIT-licensed). Problems are released in dated batches from LeetCode, AtCoder, CodeForces.

**Why this one:** the only contamination-controlled coding benchmark with a reliable dated-release filter. HumanEval is saturated. MBPP is contaminated. SWE-bench is too heavy. LCB v6 is the maintained, filterable, current standard.

**Our subset:**
- **Release-window filter:** problems released between **2024-12-01** and **2025-04-30** (the v6 cutoff). These postdate Llama-3.3-70B's Dec 2024 training cutoff and are very unlikely to have been in Qwen3-32B's training data (exact cutoff TBD by model card).
- This window yields approximately 100–130 problems in v6. Target: 80 train / 40 test, with time-based split (earlier → train, later → test).

**Scoring function:**
- Primary metric: **pass@1** on a held-out test-case set. Each problem has reference test cases (typically 3–6).
- Agent output: a single Python function or full program, extracted from the LLM response via fenced-block parsing.
- Execution: not implemented. A subprocess and timeout are not a sandbox; this benchmark must remain disabled until submissions run in a disposable OS-level sandbox with no host secrets or network and strict process, filesystem, CPU, memory, PID, and wall-time limits.
- Per-benchmark score = mean(pass@1) over the 40 test items.

**Contamination concerns:** the release-date filter is the defense. We additionally exclude any problem whose text appears verbatim in the pretraining-data-probes we run at smoke-test time (a small set of `"<problem text>"` prompts to each model with temperature 0; if the model recites the problem, we drop it).

**Evaluation protocol:**
- Single LLM call per (genome, problem). The agent prompt includes the problem statement and output format instructions (specified by genes like `answer_format`).
- No multi-turn for v1. An ablation with a fixed 2-turn reflection scaffold is deferred to exploratory.
- Sandboxing requirement: use a disposable OS-level sandbox with no host secrets or network, a read-only root, a dedicated temporary working directory, and strict process, filesystem, CPU, memory, PID, syscall, and wall-time limits. `RestrictedPython`, regex or AST filtering, and `subprocess.run(..., timeout=5, ...)` are not security boundaries.
- Parse failures scored as 0 (complete failure).

**Cost per run:** same as ForecastBench, 24,000 search calls + 40 test calls. Coding responses are longer (≈ 800 output tokens vs ≈ 200 for forecasting), so per-call wall clock is ~1.5× longer.

### 3. GPQA Diamond (scientific reasoning)

**Data source:** https://github.com/idavidrein/gpqa (MIT-licensed). 198 graduate-level physics / biology / chemistry MCQ questions.

**Why this one:** though GPQA is approaching saturation on frontier closed models (Gemini 3.1 Pro ≈ 94%, GPT-5.4 ≈ 92%), **open-weight 70B models are around 50–65%**, leaving substantial headroom for agent-configuration gains to be measurable. It is the cleanest graduate-level reasoning benchmark with a stable evaluation protocol and no contamination — the questions were explicitly written to be Google-proof.

**Why not MMLU-Pro as primary:** MMLU-Pro is larger (12k items) and cheaper per item, but the questions are shallower and the per-item signal is noisier. We include MMLU-Pro as an **exploratory add-on** if compute permits (see Phase 8 budget), but GPQA Diamond is the committed primary reasoning benchmark.

**Our subset:**
- All 198 Diamond items.
- Split: random 80/80 split with seed 42 (frozen). Remaining 38 items are a held-out "exploratory validation set" used only at paper-writing time to sanity-check one or two claims, not for confirmatory inference.

**Scoring function:**
- Primary metric: **accuracy (exact-match on the selected option letter)** against ground truth.
- Four-choice MCQ. The agent outputs an option letter `A`/`B`/`C`/`D` after any reasoning. Parser extracts the final letter via a robust "after `Answer:`" pattern; on parse failure, retry twice; on 3rd failure, score as random (0.25 expected).
- Per-benchmark score = mean(accuracy) over the 80 test items.

**Contamination concerns:** GPQA is designed to be Google-proof, which implies robustness against pretraining memorization. Additionally, the answer option letters are shuffled per-run (seeded per genome) so memorized letter patterns do not help.

**Evaluation protocol:** single LLM call per (genome, question). Reasoning genes (e.g. `decomposition_style`) may expand this into multi-step reasoning within a single call, but no multi-turn conversation.

**Cost per run:** 24,000 search calls + 80 test calls. Reasoning responses are longer (~300–500 tokens) due to chain-of-thought, placing per-call wall clock between forecasting and coding.

## Contamination protocol (all benchmarks)

Before locking any benchmark snapshot, we run the following contamination probe at smoke-test time:

1. For each benchmark, draw 10 random problems.
2. For each model, submit the problem text verbatim with temperature 0 and a completion-style prompt ("Continue: [first 200 chars of problem]").
3. If the model's continuation matches the problem text at > 80% BLEU or the model cites the correct answer without prompting, the problem is flagged and removed from the pool.
4. Log the drop list. If > 10% of probed problems are flagged, the entire benchmark is re-evaluated for usability.

## Evaluation protocol (cross-benchmark)

- **Deterministic agent invocation:** `temperature = 0` for all non-temperature-evolved runs (we set temperature via the genome, but if the gene says a specific value, we use that value). `seed` passed to the vLLM backend where supported.
- **Single-call-per-item:** every (genome, item) is exactly one API call. No prompt ensembling, no self-consistency voting, unless a gene explicitly encodes it.
- **Token budget per call:** max output tokens 1024 for coding, 512 for reasoning, 256 for forecasting. Exceeding is truncation — parser treats truncated output as parse failure.
- **Error handling:** network / backend errors are retried once; persistent failures are logged and the item is scored as worst-case for fitness (0 for coding/reasoning, 0.25 Brier for forecasting).
- **Per-call logging:** every call logged with `{run_id, genome_hash, item_id, input_tokens, output_tokens, wall_time_ms, score}` as a line in `results/{run_id}/calls.jsonl`.
- **Cost accounting:** total LLM calls, total input tokens, total output tokens logged per run.

## Final benchmark summary

| Benchmark | Items (train/test) | Metric | Contamination defense | Agent call/item | Avg out-tok | Per-call wall-time estimate (Llama-3.3-70B @ 1× H100 AWQ) |
|---|---|---|---|---|---|---|
| ForecastBench (binary) | 80 / 80 | 1 − Brier | Future events (resolution date > training cutoff) | 1 | 200 | ~4 s |
| LiveCodeBench v6 | 80 / 40 | pass@1 | Release-date filter (post-training-cutoff problems only) + text probe | 1 | 800 | ~12 s |
| GPQA Diamond | 80 / 80 | accuracy | Google-proof by design + shuffled options | 1 | 400 | ~6 s |

**Total evaluations per (model, method, seed) cell = (80 × 20 × 15) + (40 or 80) = ~24,100.**

**Wall-clock cost for one Llama-3.3-70B cell, summed across all three benchmarks: ~24,000 × (4 + 12 + 6) / 3 seconds ≈ 176,000 seconds ≈ 49 hours ≈ 2.0 GPU-days single-GPU.**

This single-cell cost, multiplied across methods × seeds × models, is the primary driver of the compute plan in Phase 8.

## What we are NOT using

- HumanEval (contamination).
- MBPP (contamination, partial).
- GSM8K (saturated for 70B).
- MATH (too high variance without >> 100 seeds).
- BBH (older, partial contamination).
- Metaculus (would need a separate scraping pipeline; ForecastBench already aggregates Metaculus as one of its 9 sources).
- SWE-bench (too expensive per problem, too much per-item setup noise).
- MMLU-Pro (kept as exploratory add-on only).
- Any closed-book benchmark that does not ship a machine-readable test set.
