"""Synthetic forecasting agent for paper validation experiments.

This module defines a deterministic, seedable agent simulator used to
validate the agentbreed pipeline before any API-cost experiments.

The agent's quality is a structured function of its genome, with
genuine cross-component interactions: certain calibration rules work
best with certain reasoning strategies, certain tools amplify certain
prompts, and so on. This means the fitness landscape is NOT trivially
additive across genes -- which is the precondition under which crossover
across heterogeneous components could plausibly help.

Crucially, the agent is also stochastic in a controlled way: predictions
are deterministic given (genome_id, question_id) but vary across genomes
even when their gene values are similar. This mimics the real-world
property that two LLM-instantiated agents with similar configs still
produce different outputs.

Why a synthetic agent?
----------------------
The paper makes one specific empirical claim: "cross-component crossover
on heterogeneous agent configurations outperforms mutation-only search
when the fitness landscape has cross-component interactions." Testing
that claim requires a fitness landscape with KNOWN structure, which a
synthetic agent provides. A real LLM agent would also have cross-component
interactions, but we couldn't isolate them from confounds.

Limitations
-----------
Results from this synthetic agent demonstrate that the breeding
*algorithm* works on cross-component fitness landscapes. They do NOT
demonstrate that real LLM agents on real forecasting tasks have such
landscapes. The full paper would need to validate against real LLMs.
This synthetic study is a *necessary* but not *sufficient* validation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Trait modeling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneEffects:
    """How each gene value affects baseline forecasting quality.

    Quality is modeled in [0, 1] where 0 is uninformative (always 0.5)
    and 1 is perfect (matches the true outcome). Genes contribute
    additively to a "skill" parameter, plus interaction terms add or
    subtract bonuses based on specific gene combinations.
    """

    # Per-gene marginal contributions (how much a particular value adds to skill)
    prompt_template_marginals: dict[str, float]
    decision_heuristic_marginals: dict[str, float]
    calibration_rule_marginals: dict[str, float]
    decomposition_style_marginals: dict[str, float]
    memory_structure_marginals: dict[str, float]
    answer_format_marginals: dict[str, float]


# Default marginal effects -- chosen so no single value dominates and
# the strongest results require finding good combinations.
DEFAULT_EFFECTS = GeneEffects(
    prompt_template_marginals={
        "superforecaster_decompose": 0.10,
        "calibrated_bayesian": 0.08,
        "pro_con_analytical": 0.05,
        "outside_view_first": 0.09,
        "market_trader": 0.04,
    },
    decision_heuristic_marginals={
        "expected_value": 0.07,
        "minimax_regret": 0.04,
        "recognition_primed": 0.05,
        "fermi_decomposition": 0.08,
        "competing_hypotheses": 0.06,
    },
    calibration_rule_marginals={
        "shrink_to_base_rate": 0.09,
        "tetlock_curve_check": 0.07,
        "extremeness_aversion": 0.08,
        "three_interval_check": 0.05,
    },
    decomposition_style_marginals={
        "reference_class": 0.08,
        "causal_chain": 0.05,
        "scenario_tree": 0.06,
        "fermi": 0.07,
        "analogy": 0.03,
    },
    memory_structure_marginals={
        "sliding_window": 0.03,
        "episodic": 0.05,
        "semantic": 0.04,
        "graph": 0.02,
        "none": 0.01,
    },
    answer_format_marginals={
        "probability_only": 0.02,
        "probability_with_reasoning": 0.04,
        "structured_json": 0.03,
    },
)

# Cross-component interaction bonuses: certain combinations work
# especially well together. These are the "co-adaptations" that
# crossover-based search should be able to exploit and that
# mutation-only search may struggle to find.
INTERACTION_BONUSES: dict[tuple[str, str, str, str], float] = {
    # (gene_a, value_a, gene_b, value_b) -> bonus to skill
    # Outside-view + reference-class (the Tetlock combo)
    ("prompt_template", "outside_view_first", "decomposition_style", "reference_class"): 0.08,
    # Bayesian + tetlock check (calibrated reasoning + good calibration)
    ("prompt_template", "calibrated_bayesian", "calibration_rule", "tetlock_curve_check"): 0.07,
    # Superforecaster decomposition + expected value (decompose then weigh)
    ("prompt_template", "superforecaster_decompose", "decision_heuristic", "expected_value"): 0.06,
    # Fermi decomposition style + Fermi heuristic
    ("decomposition_style", "fermi", "decision_heuristic", "fermi_decomposition"): 0.07,
    # Competing hypotheses + episodic memory (recall past evidence)
    ("decision_heuristic", "competing_hypotheses", "memory_structure", "episodic"): 0.05,
    # Extremeness aversion + market trader prompt (counters market overconfidence)
    ("calibration_rule", "extremeness_aversion", "prompt_template", "market_trader"): 0.06,
    # Structured JSON output + reference class (forces explicit prior)
    ("answer_format", "structured_json", "decomposition_style", "reference_class"): 0.04,
}


# Anti-interactions: certain combinations actively hurt
INTERACTION_PENALTIES: dict[tuple[str, str, str, str], float] = {
    # Outside view shouldn't be combined with analogy (redundant)
    ("prompt_template", "outside_view_first", "decomposition_style", "analogy"): 0.04,
    # Pro/con analytical with no memory is wasteful
    ("prompt_template", "pro_con_analytical", "memory_structure", "none"): 0.03,
}


# ---------------------------------------------------------------------------
# Agent simulator
# ---------------------------------------------------------------------------

def _stable_uniform(*parts: str) -> float:
    """Deterministic float in [0, 1] from string parts via SHA-256."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:16], 16) / float(0xFFFF_FFFF_FFFF_FFFF)


def _genome_content_hash(genome_dict: dict[str, Any]) -> str:
    """Compute a stable hash of a genome's *content* (gene values), ignoring IDs.

    Two genomes with identical gene values produce the same hash. This is
    critical for determinism in the synthetic agent: if we hashed
    ``genome_id`` (a random UUID) instead, two structurally-identical
    genomes would produce different predictions, making experiments
    non-reproducible.
    """
    genes = genome_dict.get("genes", {}) or {}
    parts: list[str] = []
    for name in sorted(genes.keys()):
        gene = genes[name]
        if isinstance(gene, dict):
            value = gene.get("value")
        else:
            value = gene
        parts.append(f"{name}={value!r}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _genome_skill(genome_dict: dict[str, Any], effects: GeneEffects = DEFAULT_EFFECTS) -> float:
    """Compute the "skill" of a genome in [0, 1].

    Skill is the sum of:
    - Per-gene marginal contributions
    - Cross-component interaction bonuses
    - Cross-component interaction penalties (subtracted)
    - A small temperature-based smoothing
    Bounded to [0, 1].
    """
    genes = genome_dict.get("genes", {})
    skill = 0.0

    def gene_value(name: str) -> str | None:
        g = genes.get(name)
        if g is None:
            return None
        return str(g.get("value")) if isinstance(g, dict) else None

    # Gene-by-gene marginals
    marginal_maps = {
        "prompt_template": effects.prompt_template_marginals,
        "decision_heuristic": effects.decision_heuristic_marginals,
        "calibration_rule": effects.calibration_rule_marginals,
        "decomposition_style": effects.decomposition_style_marginals,
        "memory_structure": effects.memory_structure_marginals,
        "answer_format": effects.answer_format_marginals,
    }
    for gene_name, marginal_map in marginal_maps.items():
        v = gene_value(gene_name)
        if v in marginal_map:
            skill += marginal_map[v]

    # Cross-component interactions
    for (g1, v1, g2, v2), bonus in INTERACTION_BONUSES.items():
        if gene_value(g1) == v1 and gene_value(g2) == v2:
            skill += bonus
    for (g1, v1, g2, v2), penalty in INTERACTION_PENALTIES.items():
        if gene_value(g1) == v1 and gene_value(g2) == v2:
            skill -= penalty

    # Temperature: too high = noisy, too low = brittle
    temp_gene = genes.get("temperature")
    if isinstance(temp_gene, dict):
        try:
            temp = float(temp_gene.get("value", 0.7))
            # Sweet spot around 0.5: bell curve
            skill += 0.04 * math.exp(-((temp - 0.5) ** 2) / 0.2)
        except (TypeError, ValueError):
            pass

    # Risk threshold: lower = better for forecasting (less Dunning-Kruger)
    risk_gene = genes.get("risk_threshold")
    if isinstance(risk_gene, dict):
        try:
            risk = float(risk_gene.get("value", 0.25))
            skill += 0.04 * (1.0 - risk * 2.0)  # peaks at risk=0
        except (TypeError, ValueError):
            pass

    # Tool count: 2-3 is the sweet spot, more wastes attention
    tool_gene = genes.get("tool_policy")
    if isinstance(tool_gene, dict):
        v = tool_gene.get("value", [])
        if isinstance(v, list):
            n = len(v)
            tool_bonus = {0: -0.05, 1: 0.0, 2: 0.04, 3: 0.05, 4: 0.03, 5: 0.0, 6: -0.02}
            skill += tool_bonus.get(n, -0.02)

    return max(0.0, min(1.0, skill))


def make_synthetic_agent(
    truth_lookup: dict[str, float],
    noise_level: float = 0.08,
    effects: GeneEffects = DEFAULT_EFFECTS,
):
    """Build a synthetic forecasting agent function.

    The returned async callable takes (genome_dict, task_str) and returns
    a string probability. The probability is:
    - Correlated with the true outcome (correlation strength = skill).
    - Deterministic given (genome_id, task).
    - Bounded in [0.02, 0.98] to avoid degenerate Brier scores.

    The task string is matched against ``truth_lookup`` to find the true
    outcome. The agent has imperfect knowledge: with probability
    proportional to ``skill``, it points in the right direction; otherwise
    it makes a random-direction prediction. Higher-skill genomes both
    point in the right direction more often AND express higher confidence
    when correct.

    Parameters
    ----------
    truth_lookup:
        Mapping from task prompt -> true binary outcome (0.0 or 1.0).
        The agent uses this to simulate "knowing" the answer with
        skill-weighted probability.
    noise_level:
        Per-prediction Gaussian noise level applied to the final probability.
    effects:
        GeneEffects determining the fitness landscape.
    """

    async def agent(genome_dict: dict[str, Any], task: str) -> str:
        skill = _genome_skill(genome_dict, effects)  # in [0, 1]
        # Use content hash, not genome_id, so identical gene sets produce
        # identical predictions regardless of UUID -- ensures reproducibility.
        content_hash = _genome_content_hash(genome_dict)
        truth = truth_lookup.get(task)

        # Deterministic per-(content, task) randomness
        u_correct = _stable_uniform(content_hash, task, "correctness")
        u_noise = (_stable_uniform(content_hash, task, "noise") - 0.5) * 2.0 * noise_level

        # Skill probability of "knowing" the right answer
        # We use a softer scaling: even a 0-skill genome has 50% chance of
        # being correct (random); a perfect-skill genome is correct ~95%.
        p_correct = 0.5 + 0.45 * skill

        if truth is not None:
            wants_yes = (truth >= 0.5) if u_correct < p_correct else (truth < 0.5)
        else:
            # No truth available -- pure random
            wants_yes = u_correct >= 0.5

        # Confidence is also skill-dependent: high-skill genomes are
        # more confident when correct, lower when uncertain.
        confidence = 0.55 + 0.40 * skill  # in [0.55, 0.95]

        if wants_yes:
            probability = confidence
        else:
            probability = 1.0 - confidence

        probability += u_noise
        probability = max(0.02, min(0.98, probability))
        return f"{probability:.4f}"

    return agent
