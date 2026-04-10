# Manual Steps — What You (The Human) Still Need To Do

Everything that could be automated has been automated. This document is the exhaustive list of steps that **cannot** be executed by the agent and therefore require human action before, during, or after submission.

Read this file once start-to-finish. Then work top-down.

---

## Part A — Before you do anything else (10 minutes)

### A.1 Install dependencies and verify the build

```bash
cd C:\Users\tejas\agentbreed
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -e ".[dev,charts]"
python -m pytest tests/ -q
```

Expected: **431 tests passing in ~8 seconds**. If any test fails, stop and fix before proceeding — downstream numbers depend on a green test suite.

### A.2 Read the paper draft

- `paper/05_draft/paper_main_track.md` — the main-track paper draft.
- `paper/01_corpus/preregistration.md` — locked hypotheses H1, H2, H3 (read this second so you understand what the paper is trying to prove).
- `paper/01_corpus/main_track_plan.md` — canonical strategy document.

These three files together are the complete story. If anything in them surprises you or does not match your mental model, stop and push back before any submission.

### A.3 Sanity-check the headline claim

Open `paper/04_results/tables/main_track/table_E1_comparisons.md` and verify with your own eyes:

- `coding / Prompt-only evolution` → diff `+0.4327`, Holm *p* `0.0000`, *d<sub>z</sub>* `6.064`
- `knowledge / Prompt-only evolution` → diff `+0.3187`, Holm *p* `0.0000`, *d<sub>z</sub>* `3.098`
- `forecasting / Prompt-only evolution` → diff `+0.0133`, Holm *p* `0.3321` (not significant — flat forecasting landscape)
- `coding / Mutation only` → Holm *p* `0.5282` (operator null, as predicted by H2)
- `coding / Random search` → Holm *p* `0.7437` (operator null, as predicted by H2)

If these numbers do not match, the pipeline is in a different state than I left it in. Re-run the analysis (Part B.2) and re-check.

---

## Part B — Reproducing the numbers from scratch (optional, ~8 minutes)

You do not need to do this unless you or a reviewer doubts the numbers. The repo ships with the outputs committed.

### B.1 Re-run every experiment

```bash
python -u paper/03_experiments/scripts/run_main_track_experiments.py --which all
```

Expected runtime: **~400 seconds** on a single CPU. Produces per-run JSON under `paper/04_results/E1`, `E_decisive`, `E2`, `E3`, `E_transfer`.

If the script stalls on Windows (it has been known to do so after ~200 runs through a Git Bash pipe), run it in smaller chunks via:

```bash
python -u paper/03_experiments/scripts/run_chunk.py --max-runs 200
```

…and re-invoke until it reports "no remaining runs."

### B.2 Re-run the analysis

```bash
python -u paper/03_experiments/scripts/analyze_main_track.py
```

Regenerates:
- `paper/04_results/tables/main_track/table_E1_summary.md`
- `paper/04_results/tables/main_track/table_E1_comparisons.md`
- `paper/04_results/tables/main_track/table_E_decisive.md`
- `paper/04_results/tables/main_track/table_transfer.md`
- `paper/04_results/figures/main_track/fig_ablation_{coding,forecasting,knowledge}.{pdf,png}`
- `paper/04_results/figures/main_track/fig_decisive_{coding,forecasting}.{pdf,png}`
- `paper/04_results/figures/main_track/fig_e2_dropout.{pdf,png}`
- `paper/04_results/figures/main_track/fig_transfer_matrix.{pdf,png}`
- `paper/04_results/analysis_main_track.json`

### B.3 Diff against committed artifacts

```bash
git status paper/04_results/
```

Expected: no unexpected changes (the re-run should be bit-exact because the synthetic agent hashes gene content, not random UUIDs).

---

## Part C — Decisions only a human can make

None of these have correct agent-answerable defaults. You must pick.

### C.1 Pick a submission venue

| Option | Fit | Action |
|---|---|---|
| NeurIPS 2026 (main track) | Strong fit for "controlled dimensionality sweep + epistasis decomposition." Deadline typically mid-May. | Check the current call for papers. Prepare a 9-page LaTeX version from the markdown draft. |
| ICLR 2027 | Also strong fit, slightly more methods-friendly. October deadline. | Same LaTeX conversion. |
| NeurIPS Agent / AutoML workshop | Safer, faster, accepts synthetic-only results. | Conversion is lighter (4–6 pages). Use this as fallback if main track rejects. |
| arXiv preprint first | Zero-risk visibility. Do this regardless of venue. | `paper/05_draft/paper_main_track.md` → convert to LaTeX → upload. |

**Recommended:** arXiv this week, then submit to NeurIPS main track when it opens. Workshop as fallback.

### C.2 Decide the author list

The draft currently says `[Anonymous for review]`. Before any non-anonymous upload:

1. Confirm every co-author in writing.
2. Agree on author order.
3. Agree on corresponding-author affiliation.
4. Get explicit sign-off from each co-author on: (a) the preregistration, (b) the limitations section, (c) the claims in §1 and §7.

### C.3 Decide on the real-LLM follow-up

The paper's biggest limitation is "synthetic agents only." The right way to close that gap is a real-LLM replication on ForecastBench, HumanEval (LiveCodeBench filter), and Metaculus historical. Decide:

- [ ] Which benchmarks to include (ForecastBench + HumanEval is the minimum).
- [ ] Which models to use (Llama-3.3-70B open + one closed-source proprietary).
- [ ] Whether to preregister the real-LLM study *before* running it (strongly recommended; a second `preregistration_real_llm.md` lock).
- [ ] Budget: ~$300–600 per full run, ×3 for variance = **~$1,500–2,000 total** at matched compute across 9 methods.
- [ ] Compute window: do you want this before or after arXiv?

I recommend preregistering and running the real-LLM study **before** NeurIPS submission so you can include a single cell of real-LLM data as "proof the synthetic findings transfer." Even 1 method × 1 model × 10 seeds of real confirmation would materially strengthen the paper.

### C.4 Review the author prior in the preregistration

`paper/01_corpus/preregistration.md` contains an explicit author-subjective prior over outcomes (H1/H2/H3 probability estimates). Confirm these were your best honest beliefs at the time of locking (2026-04-10) and that you are comfortable having a reviewer read them. If not, edit in git history is **not acceptable** — you must instead amend with a separate "Updated prior" paragraph referencing the original.

---

## Part D — Things you must do before any public release

### D.1 Credentials to acquire

| Credential | Where it's used | How to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | `breed/adapters/anthropic_adapter.py` for real-LLM runs | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | `breed/adapters/openai_adapter.py` (optional) | [platform.openai.com](https://platform.openai.com) |
| ForecastBench API access | Real-LLM follow-up on forecasting | See ForecastBench repo docs |
| Metaculus API token | Historical question backfill | [metaculus.com](https://metaculus.com), Settings → API |
| HuggingFace token | Downloading HumanEval / LiveCodeBench splits | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| wandb or MLflow (optional) | Experiment tracking for real-LLM runs | [wandb.ai](https://wandb.ai) or self-host MLflow |

**Do not commit any of these to git.** Use `.env` + `python-dotenv`; the codebase already respects `os.environ[...]`.

### D.2 Repository hygiene

Before making the repo public:

- [ ] `git log --all -- paper/04_results/` — confirm no committed files contain live API keys.
- [ ] Scan for large files: `git ls-files | xargs -I {} du -k "{}" | sort -rn | head -20`. Nothing should be over 5 MB. If it is, use `git lfs`.
- [ ] Add a `LICENSE` file (MIT is already implied by the plan — add the actual file).
- [ ] Add a `CITATION.cff` pointing at the arXiv version once it exists.
- [ ] Add a minimal `CODE_OF_CONDUCT.md`.
- [ ] Confirm `.gitignore` excludes `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `*.egg-info/`.
- [ ] Run `pip-audit` (or `bandit -r breed/`) once and resolve anything CRITICAL/HIGH.

### D.3 README for external readers

The current `README.md` is oriented toward the library. Before launch, add three sections at the top:

1. **What this is** — two sentences, not the whole manifesto.
2. **Paper link** — once arXiv number exists.
3. **`pip install -e .` + `python -m pytest` quickstart** — so a reviewer can get to green tests in 60 seconds.

---

## Part E — Things only a human can verify

### E.1 Proofread the paper draft yourself

Agents (including me) cannot catch every tone/clarity issue. Read `paper_main_track.md` out loud once. Flag:

- Any sentence you wouldn't defend in a rebuttal.
- Any claim that stretches beyond what the data supports.
- Any adjective that sounds like hype ("dramatically," "breakthrough," etc.).

The current draft was written defensively but is not immune to overclaiming. Trust your own ear more than mine.

### E.2 Pick the headline chart

The paper currently references `fig_decisive_coding.pdf` as the headline figure (E_decisive on coding, showing 6× improvement from K=1 to K=9). Before submission, **look at every figure in `paper/04_results/figures/main_track/` and decide which single figure is the most persuasive.** That becomes the Figure 1 of the final PDF.

Candidates worth considering:
- `fig_decisive_coding.pdf` — dimensionality sweep (current Figure 1).
- `fig_ablation_coding.pdf` — 9-method bar chart showing prompt-only's collapse.
- `fig_transfer_matrix.pdf` — 3×3 heatmap showing domain-specific champions.

### E.3 Decide whether to retain the workshop draft

There are currently two paper files:

- `paper/05_draft/paper.md` — the earlier workshop draft (single domain, 6 methods).
- `paper/05_draft/paper_main_track.md` — the main-track draft (3 domains, 10 methods, preregistered).

Decide:

- **Archive the workshop draft** (move to `paper/05_draft/archive/paper_workshop_v0.md`) — recommended, preserves history.
- **Keep both** — only if you plan to submit both (e.g. main track + parallel workshop submission at a different venue).
- **Delete the workshop draft** — not recommended, git history still has it but file-level clarity is lost.

I did not make this call; it's yours.

---

## Part F — Optional but high-value follow-ups

These would materially strengthen the paper but are not blockers.

### F.1 Bigger Sobol sample (closes H3 from "qualitative" to "quantitative")

Current E3 runs Saltelli at `n_base = 512` and the indices are noisy. Re-run at `n_base = 2048`:

```bash
python -u paper/03_experiments/scripts/run_main_track_experiments.py --which E3 --sobol-n-base 2048
```

Expected runtime: ~1 hour. Updates the H3 rows in `table_E1_comparisons.md` and can be referenced as a "full Saltelli decomposition" rather than "qualitative confirmation."

### F.2 Add a fourth domain

The paper's generality is bounded by 3 domains. Adding a fourth structured synthetic domain (e.g. dialogue agents, multi-step tool use) is cheap: ~200 additional runs, ~2 minutes. Edit `paper/03_experiments/scripts/multi_domain_agent.py` and add `DIALOGUE_DOMAIN` with its own gene marginals and interaction bonuses.

### F.3 Random gene template ablation

The "9-gene template was chosen by authors" limitation is addressed by sampling random 9-gene templates from a larger pool and showing the effect persists. If you do this, **lock the pool and the sampling seed in an amended preregistration** first.

### F.4 Real-LLM one-cell proof

Even a single cell (e.g. full_evolution on forecasting with Llama-3.3-70B, 10 seeds) would let you say "the synthetic findings replicate on a real model at reduced scale." That sentence is worth the ~$300 it would cost.

---

## Part G — What NOT to do

Things that sound like good ideas but would damage the paper:

- ❌ **Do not add "agentic breeding" branding to the paper.** The marketing framing is fine for the repo launch; it is poison in an academic paper. The draft currently avoids it. Keep it that way.
- ❌ **Do not edit the preregistration file after 2026-04-10 without a dated amendment.** Silent edits destroy the preregistration's value.
- ❌ **Do not run more seeds if H2 goes marginal.** Stopping-and-retesting is p-hacking. The preregistered *n* = 20 is *n* = 20.
- ❌ **Do not claim H3 is "quantitatively confirmed"** until you re-run Sobol at `n_base ≥ 2048`. The current phrasing in the paper is "qualitatively confirmed" — match that exactly in any summary or tweet.
- ❌ **Do not submit to venues with conflicting anonymity requirements** (double-blind vs. single-blind) without reading the call carefully.
- ❌ **Do not push real LLM credentials to git.** Ever.
- ❌ **Do not promote the repo on Twitter/HN before arXiv is live.** The paper is the lead; the repo supports it.

---

## Part H — Launch sequence (when all of the above is done)

1. **arXiv submission** (Part C.1 + C.2) — anonymous → de-anonymize on arXiv → push.
2. **Repo de-privatization** — flip the repo to public.
3. **README final polish** — add arXiv badge, paper link, one-line elevator pitch.
4. **One technical tweet** (not a marketing thread) — link to arXiv, one sentence, one figure (`fig_decisive_coding.pdf`).
5. **HN Show HN post** — 24 hours later, titled conservatively: "Search Space, Not Operators: Epistasis Dominates LLM Agent Configuration Optimization."
6. **r/MachineLearning [R] post** — 48 hours later, with key table and figure.
7. **Respond to comments** — first 48 hours drives 80% of the attention. Budget 6+ hours of response time.

---

## Summary checklist

Copy this into an issue tracker and tick off:

- [ ] A.1 Install + 431 tests green
- [ ] A.2 Read paper, preregistration, plan
- [ ] A.3 Verify headline numbers
- [ ] B.1 (optional) Re-run experiments
- [ ] B.2 (optional) Re-run analysis
- [ ] C.1 Pick submission venue
- [ ] C.2 Confirm author list
- [ ] C.3 Decide real-LLM follow-up plan
- [ ] C.4 Confirm preregistration author prior
- [ ] D.1 Acquire credentials (real-LLM runs only)
- [ ] D.2 Repo hygiene pass
- [ ] D.3 README polish
- [ ] E.1 Proofread paper aloud
- [ ] E.2 Pick headline chart
- [ ] E.3 Decide workshop draft fate
- [ ] F.1–F.4 (optional) Strengthening experiments
- [ ] G Re-read "do not" list before any public step
- [ ] H Execute launch sequence in order

---

**Bottom line:** Every number in the paper is reproducible from a fresh clone in ~8 seconds (tests) + ~400 seconds (experiments). Every computational step has been executed. The remaining work is judgement, writing polish, credentials, and submission logistics — all of which are inherently human tasks.
