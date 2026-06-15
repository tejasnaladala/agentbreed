# What Must Be Retested

Every claim in the pilot main-track draft (kept in git history only, out of the public tree because it targets a blind submission) that does not survive into the real study must be either dropped or retested on real data. This is the disposition list.

## Claims that must be retested on real data

| # | Pilot claim | Where it appears | Status for real study |
|---|---|---|---|
| 1 | "Multi-component evolutionary search outperforms single-component (prompt-only) search with a large paired effect (d_z = 1.32, p = 1e-14)" | Abstract, §1, §5.4 | **Retest as H1** on 3 benchmarks × 2 models, no pooling across non-independent domains. Use mixed-effects LMM. |
| 2 | "Pairwise comparisons between full evolution and every other multi-component baseline fail to reach Holm-corrected significance in any domain" | Abstract, §1, §5.3 | **Retest as H2** using equivalence testing (TOST or paired bootstrap with preregistered equivalence margin), not the misleading "fails to reject H0 = confirms null" logic. |
| 3 | "A controlled dimensionality sweep across K ∈ {1, 2, 3, 5, 7, 9} gene axes confirms the effect is monotonic in search space size" | Abstract, §4 | **Retest on real benchmarks.** The sweep design is valid; the data is not. |
| 4 | "Functional ANOVA on the fitness landscape detects meaningful pairwise epistasis (H3 confirmed)" | Abstract, §5.6 | **Retest with Sobol n_base ≥ 2048** on a landscape estimated from real LLM calls, not a synthetic hash. Must re-verify that `sum(first_order) + sum(pairwise) ≤ 1.0`. |
| 5 | "Champion configurations do not transfer across domains (off-diagonal ≈ 40% of diagonal)" | Abstract, §5.5 | **Retest on real benchmarks × real models.** Compute the preregistered retention-of-improvement metric, not raw fitness. |
| 6 | "For LLM agent optimization, invest in defining a richer configuration space, not in building a more sophisticated search operator" | Abstract prescription | **Cannot be asserted from synthetic evidence.** Retest in the real study or soften to "our synthetic pilot suggests, and the real study must verify." |
| 7 | "To our knowledge this is the first rigorous demonstration of the same phenomenon for LLM agent configuration search" | §1 | **Factually false of the pilot** (zero LLM calls). The real study earns this claim only if it runs on real LLMs. |
| 8 | "full_evolution's test fitness is 0.846 on coding, 0.770 on forecasting, and 0.430 on knowledge" | §5.2 | Specific pilot numbers. Real study replaces with real-benchmark numbers. |
| 9 | "Runtime: 223 seconds on a single CPU ... 600 runs" | §5.1 | Real study runtime will be orders of magnitude higher (days, not seconds). Replace. |
| 10 | "our experiments rule out medium effects (d_z > 0.5) with high confidence" | §7 | Retest with appropriate power analysis for the real-study seed count. |

## Claims that must be DROPPED from the real paper

- "Preregistered" as applied to any synthetic run.
- "3 of 3 synthetic domains confirm H1" — decision rule was undefined for N=3.
- "H2 confirmed" per pilot threshold interpretation — the preregistered |diff| ≤ 0.03 rule was violated and silently relaxed.
- "H3 confirmed qualitatively" — invalid decomposition.
- "Mirrors a classical result from Neural Architecture Search" — may be true but not supportable from synthetic evidence; attach to real-study results only.
- The transfer matrix asymmetry argument about "wide" vs "narrow" basins — pilot-artifact, not real signal.

## Claims the real study must newly address

These are questions the pilot never addressed but a NeurIPS reviewer will ask:

- **Inference-cost normalization.** Under matched LLM-call budget, do the methods still tie after accounting for auxiliary compute (TPE fits, SH partial-fidelity calls)?
- **Wall-clock normalization.** Some methods have longer wall-clock per evaluation due to sequential dependencies (successive halving rounds, coordinate descent trajectories). Are results stable under wall-clock-normalized comparisons?
- **Cross-model transfer.** Does a champion from Qwen3-32B evaluated on Llama-3.3-70B retain fitness improvement? The preregistered S1 ("retain ≥ 80% of fitness improvement over random search").
- **Cross-benchmark transfer.** The preregistered S2 ("retain < 50% of fitness improvement").
- **Budget-scaling curves.** Does the operator null hold at `population_size × generations = 100, 300, 1000`?
- **Sensitivity to decoding stochasticity.** Temperature is a genome gene in the pilot. In the real study, the benchmark evaluation itself has stochasticity. How does sampling variance interact with genome-induced variance?
- **Effect size floors at realistic N.** With 15 seeds per cell (the real-study feasibility limit), what is the minimum detectable d_z? The real-paper power analysis must be honest about this.

## Summary

**Nothing numerical from the pilot propagates.** The pilot's *question*, its *design template* for the dimensionality sweep and method matrix, and its *code infrastructure* propagate. Its *evidence* does not.
