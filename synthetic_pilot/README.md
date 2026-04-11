# Synthetic Pilot

**Status:** labeled as an exploratory synthetic pilot. **Not** the preregistered main-track execution.

## What this directory is

This directory is a pointer to the pilot-phase artifacts in `../paper/`. The `paper/` subtree was originally framed as a main-track NeurIPS paper; an adversarial review on 2026-04-10 found that the framing was incompatible with the locked preregistration (`paper/01_corpus/preregistration.md`). Specifically:

- The preregistration locks **real benchmarks** (ForecastBench, HumanEval, MBPP, Metaculus historical) and **real open-weight models** (Llama-3.3-70B, Qwen-2.5-72B).
- The pilot ran only on **synthetic analogs** driven by a deterministic content-hash agent (`multi_domain_agent.py`).
- The decision rules for H1 ("3 of 4 benchmarks") and H2 ("|diff| ≤ 0.03") were silently reinterpreted when applied to the 3 synthetic domains.
- The Sobol decomposition for H3 was run at `n_base = 512`, producing invalid (> 1.0) variance shares, and re-labelled as "qualitatively confirmed."

The honest framing is: the work in `../paper/` is a **controlled synthetic pilot** that produced three findings worth propagating to a real study, and nothing it claims about LLM agent configuration search can be concluded without a real-LLM replication.

## Honest relabeling

The paper draft at `../paper/05_draft/paper_main_track.md` is retained as-is in git history for accountability. Any use of it going forward must treat it as a pilot artifact, not a preregistered main-track execution. The word `preregistered` must not be applied to its 600 synthetic runs.

The real study lives in `../real_study/`. Its preregistration is `../real_study/preregistration_real_v1.md` and is locked with a dated commit **before any real-LLM experiment runs**.

See also:
- `WHAT_SURVIVES.md` — which pilot components are reusable in the real study
- `WHAT_MUST_BE_RETESTED.md` — which claims cannot propagate without real evidence
- `../real_study/README.md` — the new study
