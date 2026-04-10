"""Multi-domain deterministic synthetic agent for main-track experiments.

This module extends the original single-domain synthetic agent to support
three task families: forecasting (Brier), coding (pass/fail), and knowledge
(accuracy). Each domain has its own fitness landscape with different
epistatic structure, enabling cross-domain validation of H1, H2, and H3.

Design principles
-----------------
1. **Deterministic:** every (genome_content_hash, task_id, domain) maps
   to a fixed output via SHA-256. No hidden RNG.
2. **Cross-component interactions:** each domain has a different set of
   "expert combinations" -- specific (prompt, heuristic, calibration)
   tuples that boost performance. This means the landscape has real
   epistasis for Sobol analysis to detect.
3. **Realistic difficulty gradients:** every domain has easy, medium,
   and hard tasks so no single strategy wins everything.
4. **No shared randomness across domains:** the forecasting agent's
   prediction for a question does not leak the coding agent's output
   for the same genome.

This is still a synthetic agent -- it is NOT a real LLM. The purpose is
to provide a controlled fitness landscape for algorithmic validation.
Real-LLM experiments are scheduled for the full main-track phase but
depend on compute resources the user must provide (see MANUAL_STEPS.md).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from breed.arenas.base import Task


# ---------------------------------------------------------------------------
# Domain-specific fitness landscapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainConfig:
    """Per-domain configuration: marginals, interactions, skill scale."""

    name: str
    prompt_marginals: dict[str, float]
    heuristic_marginals: dict[str, float]
    calibration_marginals: dict[str, float]
    decomposition_marginals: dict[str, float]
    interaction_bonuses: dict[tuple[str, str, str, str], float]
    interaction_penalties: dict[tuple[str, str, str, str], float]
    skill_scale: float = 1.0  # Multiplier on final skill contribution


# Forecasting domain: rewards calibrated Bayesian reasoning with
# reference-class decomposition. Mirrors the original single-domain agent.
FORECASTING_DOMAIN = DomainConfig(
    name="forecasting",
    prompt_marginals={
        "superforecaster_decompose": 0.10,
        "calibrated_bayesian": 0.08,
        "pro_con_analytical": 0.05,
        "outside_view_first": 0.09,
        "market_trader": 0.04,
    },
    heuristic_marginals={
        "expected_value": 0.07,
        "minimax_regret": 0.04,
        "recognition_primed": 0.05,
        "fermi_decomposition": 0.08,
        "competing_hypotheses": 0.06,
    },
    calibration_marginals={
        "shrink_to_base_rate": 0.09,
        "tetlock_curve_check": 0.07,
        "extremeness_aversion": 0.08,
        "three_interval_check": 0.05,
    },
    decomposition_marginals={
        "reference_class": 0.08,
        "causal_chain": 0.05,
        "scenario_tree": 0.06,
        "fermi": 0.07,
        "analogy": 0.03,
    },
    interaction_bonuses={
        ("prompt_template", "outside_view_first", "decomposition_style", "reference_class"): 0.08,
        ("prompt_template", "calibrated_bayesian", "calibration_rule", "tetlock_curve_check"): 0.07,
        ("prompt_template", "superforecaster_decompose", "decision_heuristic", "expected_value"): 0.06,
        ("decomposition_style", "fermi", "decision_heuristic", "fermi_decomposition"): 0.07,
        ("decision_heuristic", "competing_hypotheses", "memory_structure", "episodic"): 0.05,
        ("calibration_rule", "extremeness_aversion", "prompt_template", "market_trader"): 0.06,
    },
    interaction_penalties={
        ("prompt_template", "outside_view_first", "decomposition_style", "analogy"): 0.04,
        ("prompt_template", "pro_con_analytical", "memory_structure", "none"): 0.03,
    },
    skill_scale=1.0,
)

# Coding domain: rewards structured decomposition and fermi analysis,
# penalizes pure intuition. Different winning combinations than forecasting.
CODING_DOMAIN = DomainConfig(
    name="coding",
    prompt_marginals={
        "superforecaster_decompose": 0.09,
        "calibrated_bayesian": 0.03,
        "pro_con_analytical": 0.08,
        "outside_view_first": 0.04,
        "market_trader": 0.02,
    },
    heuristic_marginals={
        "expected_value": 0.03,
        "minimax_regret": 0.07,
        "recognition_primed": 0.08,
        "fermi_decomposition": 0.10,
        "competing_hypotheses": 0.05,
    },
    calibration_marginals={
        "shrink_to_base_rate": 0.03,
        "tetlock_curve_check": 0.04,
        "extremeness_aversion": 0.02,
        "three_interval_check": 0.07,
    },
    decomposition_marginals={
        "reference_class": 0.04,
        "causal_chain": 0.08,
        "scenario_tree": 0.07,
        "fermi": 0.10,
        "analogy": 0.02,
    },
    interaction_bonuses={
        # fermi + fermi is a strong coding combo (decompose + estimate)
        ("decomposition_style", "fermi", "decision_heuristic", "fermi_decomposition"): 0.10,
        # causal chain + recognition primed (debug-flow)
        ("decomposition_style", "causal_chain", "decision_heuristic", "recognition_primed"): 0.08,
        # pro/con + scenario tree (test case enumeration)
        ("prompt_template", "pro_con_analytical", "decomposition_style", "scenario_tree"): 0.07,
        # superforecaster decompose + expected value (ranking candidate solutions)
        ("prompt_template", "superforecaster_decompose", "decision_heuristic", "expected_value"): 0.05,
        # three_interval_check + fermi_decomposition (bounds reasoning)
        ("calibration_rule", "three_interval_check", "decision_heuristic", "fermi_decomposition"): 0.06,
    },
    interaction_penalties={
        # outside_view_first is great for forecasting but hurts coding (too abstract)
        ("prompt_template", "outside_view_first", "decomposition_style", "fermi"): 0.04,
        # market trader is bad for coding
        ("prompt_template", "market_trader", "decision_heuristic", "fermi_decomposition"): 0.05,
    },
    skill_scale=1.0,
)

# Knowledge/reasoning domain: rewards broad-coverage prompts with
# episodic memory and competing hypotheses.
KNOWLEDGE_DOMAIN = DomainConfig(
    name="knowledge",
    prompt_marginals={
        "superforecaster_decompose": 0.05,
        "calibrated_bayesian": 0.10,
        "pro_con_analytical": 0.09,
        "outside_view_first": 0.06,
        "market_trader": 0.01,
    },
    heuristic_marginals={
        "expected_value": 0.04,
        "minimax_regret": 0.03,
        "recognition_primed": 0.07,
        "fermi_decomposition": 0.05,
        "competing_hypotheses": 0.10,
    },
    calibration_marginals={
        "shrink_to_base_rate": 0.04,
        "tetlock_curve_check": 0.06,
        "extremeness_aversion": 0.07,
        "three_interval_check": 0.09,
    },
    decomposition_marginals={
        "reference_class": 0.07,
        "causal_chain": 0.09,
        "scenario_tree": 0.05,
        "fermi": 0.03,
        "analogy": 0.08,
    },
    interaction_bonuses={
        # competing hypotheses + episodic (classic knowledge retrieval)
        ("decision_heuristic", "competing_hypotheses", "memory_structure", "episodic"): 0.09,
        # bayesian + tetlock (calibrated knowledge)
        ("prompt_template", "calibrated_bayesian", "calibration_rule", "tetlock_curve_check"): 0.08,
        # analogy + pro_con (cross-domain transfer)
        ("decomposition_style", "analogy", "prompt_template", "pro_con_analytical"): 0.07,
        # causal chain + recognition primed (deductive chains)
        ("decomposition_style", "causal_chain", "decision_heuristic", "recognition_primed"): 0.06,
        # semantic memory + competing hypotheses
        ("memory_structure", "semantic", "decision_heuristic", "competing_hypotheses"): 0.05,
    },
    interaction_penalties={
        # market trader + analogy is bad for knowledge
        ("prompt_template", "market_trader", "decomposition_style", "analogy"): 0.04,
    },
    skill_scale=1.0,
)

DOMAINS: dict[str, DomainConfig] = {
    "forecasting": FORECASTING_DOMAIN,
    "coding": CODING_DOMAIN,
    "knowledge": KNOWLEDGE_DOMAIN,
}


# ---------------------------------------------------------------------------
# Skill computation
# ---------------------------------------------------------------------------

def _gene_value(genome_dict: dict[str, Any], name: str) -> Any:
    g = genome_dict.get("genes", {}).get(name)
    if g is None:
        return None
    return g.get("value") if isinstance(g, dict) else None


def _compute_domain_skill(genome_dict: dict[str, Any], domain: DomainConfig) -> float:
    """Compute domain-specific skill in [0, 1] for a genome."""
    skill = 0.0

    # Per-gene marginals
    for gene_name, marginals in [
        ("prompt_template", domain.prompt_marginals),
        ("decision_heuristic", domain.heuristic_marginals),
        ("calibration_rule", domain.calibration_marginals),
        ("decomposition_style", domain.decomposition_marginals),
    ]:
        v = _gene_value(genome_dict, gene_name)
        if v in marginals:
            skill += marginals[v]

    # Interaction bonuses
    for (g1, v1, g2, v2), bonus in domain.interaction_bonuses.items():
        if _gene_value(genome_dict, g1) == v1 and _gene_value(genome_dict, g2) == v2:
            skill += bonus

    # Interaction penalties
    for (g1, v1, g2, v2), penalty in domain.interaction_penalties.items():
        if _gene_value(genome_dict, g1) == v1 and _gene_value(genome_dict, g2) == v2:
            skill -= penalty

    # Temperature sweet spot (domain-independent)
    genes = genome_dict.get("genes", {})
    temp_gene = genes.get("temperature")
    if isinstance(temp_gene, dict):
        try:
            temp = float(temp_gene.get("value", 0.7))
            skill += 0.04 * math.exp(-((temp - 0.5) ** 2) / 0.2)
        except (TypeError, ValueError):
            pass

    # Tool count (domain-independent)
    tool_gene = genes.get("tool_policy")
    if isinstance(tool_gene, dict):
        v = tool_gene.get("value", [])
        if isinstance(v, list):
            n = len(v)
            tool_bonus = {0: -0.05, 1: 0.0, 2: 0.04, 3: 0.05, 4: 0.03, 5: 0.0, 6: -0.02}
            skill += tool_bonus.get(n, -0.02)

    return max(0.0, min(1.0, skill * domain.skill_scale))


def _genome_content_hash(genome_dict: dict[str, Any]) -> str:
    """Stable hash of gene values, ignoring identifiers. Ensures determinism."""
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


def _stable_uniform(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:16], 16) / float(0xFFFF_FFFF_FFFF_FFFF)


# ---------------------------------------------------------------------------
# Forecasting agent (binary probability output)
# ---------------------------------------------------------------------------

def make_forecasting_agent(
    truth_lookup: dict[str, float],
    noise_level: float = 0.05,
) -> Callable[[dict[str, Any], str], Awaitable[str]]:
    """Build a deterministic forecasting agent for a given truth table."""

    async def agent(genome_dict: dict[str, Any], task: str) -> str:
        skill = _compute_domain_skill(genome_dict, FORECASTING_DOMAIN)
        content_hash = _genome_content_hash(genome_dict)
        truth = truth_lookup.get(task)

        u_correct = _stable_uniform(content_hash, task, "fcst:correctness")
        u_noise = (_stable_uniform(content_hash, task, "fcst:noise") - 0.5) * 2.0 * noise_level
        p_correct = 0.5 + 0.45 * skill

        if truth is not None:
            wants_yes = (truth >= 0.5) if u_correct < p_correct else (truth < 0.5)
        else:
            wants_yes = u_correct >= 0.5

        confidence = 0.55 + 0.40 * skill
        probability = confidence if wants_yes else (1.0 - confidence)
        probability += u_noise
        probability = max(0.02, min(0.98, probability))
        return f"{probability:.4f}"

    return agent


# ---------------------------------------------------------------------------
# Coding agent (pass/fail output)
# ---------------------------------------------------------------------------

def make_coding_agent(
    n_test_cases_per_task: int = 5,
) -> Callable[[dict[str, Any], str], Awaitable[str]]:
    """Build a deterministic coding agent that returns per-task pass rate.

    Each task is treated as having ``n_test_cases_per_task`` hidden test
    cases. The agent passes each test case with probability equal to the
    genome's coding-domain skill.
    """

    async def agent(genome_dict: dict[str, Any], task: str) -> str:
        skill = _compute_domain_skill(genome_dict, CODING_DOMAIN)
        content_hash = _genome_content_hash(genome_dict)
        passed = 0
        for i in range(n_test_cases_per_task):
            u = _stable_uniform(content_hash, task, f"code:test:{i}")
            if u < skill:
                passed += 1
        return f"{passed / n_test_cases_per_task:.4f}"

    return agent


# ---------------------------------------------------------------------------
# Knowledge/QA agent (accuracy output)
# ---------------------------------------------------------------------------

def make_knowledge_agent() -> Callable[[dict[str, Any], str], Awaitable[str]]:
    """Build a deterministic knowledge QA agent that returns correctness.

    Returns 1.0 if correct, 0.0 if wrong. Probability of correctness is
    the genome's knowledge-domain skill.
    """

    async def agent(genome_dict: dict[str, Any], task: str) -> str:
        skill = _compute_domain_skill(genome_dict, KNOWLEDGE_DOMAIN)
        content_hash = _genome_content_hash(genome_dict)
        u = _stable_uniform(content_hash, task, "knowledge:correct")
        return "1.0" if u < skill else "0.0"

    return agent


# ---------------------------------------------------------------------------
# Domain scorers
# ---------------------------------------------------------------------------

def forecasting_scorer(output: str, expected) -> float:
    """1 - Brier, higher is better."""
    try:
        p = float(output.strip())
        y = float(expected)
        return 1.0 - (p - y) ** 2
    except (ValueError, TypeError, AttributeError):
        return 0.0


def pass_rate_scorer(output: str, expected) -> float:
    """Pass rate in [0, 1]. expected is ignored (treat as accuracy)."""
    try:
        rate = float(output.strip())
        return max(0.0, min(1.0, rate))
    except (ValueError, AttributeError):
        return 0.0


def accuracy_scorer(output: str, expected) -> float:
    """Exact-match accuracy for knowledge tasks."""
    try:
        return 1.0 if abs(float(output.strip()) - float(expected)) < 1e-6 else 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Task generation helpers
# ---------------------------------------------------------------------------

def build_coding_tasks(
    n: int,
    prefix: str = "code",
    seed: int = 42,
) -> list[Task]:
    """Generate synthetic coding task descriptors.

    These are not real coding problems — they're seeded identifiers that
    the synthetic coding agent uses as inputs to its deterministic
    skill-weighted pass-rate output. The expected field is set to 1.0 so
    scorers that compute (output - expected) can still work.
    """
    tasks: list[Task] = []
    for i in range(n):
        tid = f"{prefix}-{i}"
        tasks.append(
            Task(
                task_id=tid,
                prompt=f"Implement function for problem {tid}",
                expected=1.0,
                metadata={"domain": "coding", "difficulty": ["easy", "medium", "hard"][i % 3]},
            )
        )
    return tasks


def build_knowledge_tasks(
    n: int,
    prefix: str = "know",
    seed: int = 42,
) -> list[Task]:
    """Generate synthetic knowledge QA task descriptors.

    Expected is set to 1.0 to represent "correct answer token".
    """
    tasks: list[Task] = []
    for i in range(n):
        tid = f"{prefix}-{i}"
        tasks.append(
            Task(
                task_id=tid,
                prompt=f"Answer knowledge question {tid}",
                expected=1.0,
                metadata={"domain": "knowledge", "difficulty": ["easy", "medium", "hard"][i % 3]},
            )
        )
    return tasks
