# Phase 14 — Risk Memo

Blunt list of what could still kill this paper.

## Scientific / empirical risks

### R1 — H1 does not replicate on real LLMs (Medium-High, impact Critical)

The synthetic pilot showed d_z = 1.32, but real LLMs may have a much smaller or zero dimensionality effect.

**Mitigation:**
- The paper's framing is set up to accept either outcome (see docs/01_research_brief.md). If H1 fails, we pivot to a "negative result / re-evaluation" framing and emphasize the surprising-to-community aspect.
- Preregistered power analysis says we can detect a 5-percentage-point effect at 95% power; if the true effect is > 5pp, we will see it.

### R2 — H2 equivalence tests are inconclusive (Medium, impact High)

With `Δ_eq = 0.03` and n=15 seeds, the TOST power to establish equivalence depends on the underlying SD. If real-LLM SDs are larger than pilot estimates (entirely plausible), several of the 21 pairs may be inconclusive — neither "equivalent" nor "different."

**Mitigation:**
- Report the pair-by-pair equivalence bounds even if the overall H2 does not cleanly confirm.
- If needed, tighten the paper's framing from "operators are equivalent" to "operator effects are bounded in magnitude by [X], and only [K] of 21 pairs achieve strict statistical equivalence at our preregistered margin."
- Budget 5 additional seeds per cell as an exploratory extension if the main matrix's H2 is borderline.

### R3 — H3 Sobol at n_base=2048 is still numerically unreliable (Low-Medium, impact Medium)

Even at 2048, the second-order Saltelli estimator can be noisy for mixed-type (ENUM + FLOAT + SET) inputs. We might get `Σ S > 1.05`, flagging the decomposition as invalid on some cells.

**Mitigation:**
- We have a validity check. If H3 fails validity, we mark it inconclusive and publish that honestly.
- Fallback: report only first-order and total-order indices (which are more robust), and estimate `Σ S_{ij}` as `Σ (S_T - S_first)` which is an upper bound on pairwise effects.

### R4 — Benchmark contamination leak (Low, impact Critical)

If a Qwen3-32B pretraining cutoff turns out to include ForecastBench resolutions or LCB v6 problems, the paper's "contamination-controlled" claim fails.

**Mitigation:**
- Contamination probe at Stage A.
- Training cutoff documented in the paper and in the prereg.
- LCB v6's release-date filter is our primary defense for coding; it is known to be reliable.

### R5 — Forecasting benchmark is saturated on Qwen3-32B/Llama-3.3-70B (Low, impact High)

If both models score > 0.9 on ForecastBench from the default prompt, there is no headroom for agent configuration to matter. H1 would fail by default.

**Mitigation:**
- Stage B pilot catches this. If ForecastBench is saturated on our model tier, we switch the third benchmark to something harder (MMLU-Pro Chemistry/Physics subset), with the switch documented as an amendment.
- Probability: low for Qwen3-32B (pilot forecasting scores were ~0.77–0.83). Llama-3.3-70B is a slightly weaker LLM on forecasting than Qwen3.

### R6 — GPQA Diamond is TOO hard (Medium, impact High)

Open-weight 70B models score around 50–65% on GPQA Diamond as of early 2026. If our models score below 30%, the variance between methods will be dominated by noise and H1/H2 tests will be underpowered.

**Mitigation:**
- Stage B pilot catches this. If GPQA mean is below 30%, we swap to MMLU-Pro reasoning subset.
- Alternative: use only the GPQA biology subset (highest-performing for open-weight models).

### R7 — Variance inflation from decoding stochasticity (Medium, impact Medium)

When `temperature > 0` is in the genome, the same genome produces different scores across reruns. This inflates per-cell variance and reduces power.

**Mitigation:**
- Pin the vLLM seed per cell.
- Report variance-decomposed results separating "between-genome" variance from "within-genome decoding" variance.
- If decoding variance dominates, constrain the temperature gene range to `[0.0, 0.3]` via an amendment.

## Compute / infrastructure risks

### R8 — Hyak klone allocation is revoked or quota exhausted (Medium, impact Critical)

UW shared clusters have tight GPU quotas. If the group account runs out of GPU-hours mid-Stage-C, the study halts.

**Mitigation:**
- Stage A probe will reveal the quota.
- Run on `ckpt` (preemptable) queue as fallback. Our atomic per-cell JSONs survive preemption.
- Emergency plan: rent comparable GPUs on Lambda/RunPod (~$2-4/GPU-hour). ~$200-400 to finish Stage C on commercial hardware.

### R9 — vLLM version incompatibility (Medium, impact Medium)

A new vLLM release or a kernel regression mid-study would break one or both models.

**Mitigation:**
- Pin vLLM version in `requirements.txt` once Stage A is clean.
- Rollback procedure documented: `pip install vllm==<frozen_version>`.
- Model weights downloaded at Stage A, cached in gscratch.

### R10 — Throughput is lower than estimated (Low-Medium, impact Medium)

The 1000 tok/s / 1700 tok/s estimates are from vLLM benchmark blogs. Real-world throughput on our benchmarks' prompt-length distribution may be 30–50% lower.

**Mitigation:**
- Stage A measures real throughput. If it's > 25% lower, we reduce seeds from 15 to 12 (still > preregistered minimum) or drop the Qwen3-14B scaling reference.
- 25% compute headroom already built into the budget.

## Process / reviewer risks

### R11 — Reviewer says "synthetic pilot is contaminated" (Low, impact Medium)

A reviewer who reads both the paper and the `synthetic_pilot/` directory might argue that the real paper's framing is tainted by the pilot's flawed preregistration.

**Mitigation:**
- Paper §2 Related Work cites the synthetic pilot as "prior exploratory work by the same author" and explicitly says the pilot's numerical claims are not relied upon.
- The new preregistration (`preregistration_real_v1.md`) is locked BEFORE any real-LLM run, not after. This is what "preregistration" means; reviewers can verify the git history.
- `synthetic_pilot/README.md` makes the boundary explicit.

### R12 — Reviewer says "equivalence testing is unfamiliar" (Low, impact Low)

Some NeurIPS reviewers don't know TOST. This is less a risk than an opportunity — we explain it clearly in §4.5.

**Mitigation:** explain TOST in one paragraph in methods, with a one-figure visualization of equivalence bounds. Standard in clinical statistics.

### R13 — Reviewer says "8 methods is too few / too many" (Low, impact Low)

If too few: we can point to 21 pairwise tests and the breadth of the ladder.
If too many: we can point to the preregistration and the fact that each method has a distinct theoretical motivation.

**Mitigation:** none needed.

### R14 — Reviewer says "why not include SMAC / DEHB / BOHB?" (Medium, impact Medium)

Standard AutoML reviewers will ask why TPE + SH but not the full AutoML ladder.

**Mitigation:** the preregistration explains the choice. If compute permits at Stage E, we can include SMAC as an exploratory add-on, clearly labeled post-hoc.

### R15 — Reviewer says "your 9-gene space is ad hoc" (High, impact Medium)

The gene selection was author judgment informed by the literature. A reviewer might say "you chose genes that make H1 look good."

**Mitigation:**
- The gene list is frozen in the preregistration BEFORE any run.
- Cite each gene choice to literature precedent (system prompt → DSPy/PromptBreeder, decomposition → Wei et al CoT, self-critique → Reflexion, etc.).
- Exploratory ablation: freeze one gene at a time and show the effect persists (or not) across gene removal.

## Writing / presentation risks

### R16 — Paper reads as evaluation-only, gets routed to workshop (Low, impact Medium)

NeurIPS E&D is a valid main venue, but some program chairs route papers there that the authors wanted in main track.

**Mitigation:**
- Submit to main track if H2's equivalence is clean; submit to E&D otherwise.
- Either way, the paper structure and story survive the routing.

### R17 — The story is not surprising enough to make main track (Medium, impact Low)

If the real data replicates the pilot (H1 confirmed, H2 confirmed), reviewers at main track might say "interesting but not novel enough; please submit to E&D."

**Mitigation:**
- Preemptively target E&D as primary. No downside.
- If results are unusually clean, escalate in revision.

## Risks I am NOT worried about

- Code bugs: 466 tests passing, atomic per-cell logging, reproducibility by construction.
- Data availability: three benchmarks are all open and stable.
- Author credit disputes: solo author.
- Legal IP: MIT-licensed code, Qwen/Llama research licenses cover academic publication.
- Time zone / operational overhead: not a scientific risk.

## Blunt summary

The single biggest risk is **R1 — the synthetic pilot result may not replicate**. Everything else is manageable with documented fallbacks. The second biggest risk is **R6 — GPQA may be too hard for our model tier**, for which we have a swap-to-MMLU-Pro fallback.

If both R1 and R6 materialize, the paper becomes "we ran a careful preregistered evaluation and found a mix of positive and negative results across three real benchmarks" — which is still a valid E&D submission, just not a main-track candidate.

If neither materializes, the paper is a clean E&D submission with main-track potential.
