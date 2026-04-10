"""Sanity tests for the synthetic forecasting agent.

These tests verify that:
1. Higher-skill genomes produce better Brier scores.
2. Cross-component interactions matter (specific gene combinations
   beat the sum of individual gene contributions).
3. The agent is deterministic given seeds.
4. The fitness landscape is not trivially additive.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from breed.datasets import load_builtin_questions
from breed.genome import Gene, GeneType, Genome
from breed.stats import brier_score
from synthetic_agent import (
    DEFAULT_EFFECTS,
    INTERACTION_BONUSES,
    _genome_skill,
    make_synthetic_agent,
)


def build_test_genome(
    prompt: str = "calibrated_bayesian",
    heuristic: str = "expected_value",
    calibration: str = "tetlock_curve_check",
    decomposition: str = "reference_class",
    memory: str = "episodic",
    answer_format: str = "probability_with_reasoning",
    temperature: float = 0.5,
    risk: float = 0.1,
    tools: list[str] | None = None,
    genome_id: str = "test",
) -> Genome:
    """Construct a Genome from explicit gene values for testing."""
    if tools is None:
        tools = ["web_search", "calculator"]
    available_tools = [
        "web_search", "web_fetch", "calculator", "code_exec", "read", "write",
    ]
    return Genome(
        genome_id=genome_id,
        generation=0,
        parents=[],
        genes={
            "prompt_template": Gene(
                name="prompt_template", type=GeneType.ENUM,
                value=prompt,
                options=[
                    "superforecaster_decompose", "calibrated_bayesian",
                    "pro_con_analytical", "outside_view_first", "market_trader",
                ],
            ),
            "decision_heuristic": Gene(
                name="decision_heuristic", type=GeneType.ENUM,
                value=heuristic,
                options=[
                    "expected_value", "minimax_regret", "recognition_primed",
                    "fermi_decomposition", "competing_hypotheses",
                ],
            ),
            "calibration_rule": Gene(
                name="calibration_rule", type=GeneType.ENUM,
                value=calibration,
                options=[
                    "shrink_to_base_rate", "tetlock_curve_check",
                    "extremeness_aversion", "three_interval_check",
                ],
            ),
            "decomposition_style": Gene(
                name="decomposition_style", type=GeneType.ENUM,
                value=decomposition,
                options=[
                    "reference_class", "causal_chain", "scenario_tree",
                    "fermi", "analogy",
                ],
            ),
            "memory_structure": Gene(
                name="memory_structure", type=GeneType.ENUM,
                value=memory,
                options=["sliding_window", "episodic", "semantic", "graph", "none"],
            ),
            "answer_format": Gene(
                name="answer_format", type=GeneType.ENUM,
                value=answer_format,
                options=[
                    "probability_only", "probability_with_reasoning",
                    "structured_json",
                ],
            ),
            "temperature": Gene(
                name="temperature", type=GeneType.FLOAT,
                value=temperature, range=(0.0, 1.5),
            ),
            "risk_threshold": Gene(
                name="risk_threshold", type=GeneType.FLOAT,
                value=risk, range=(0.01, 0.5),
            ),
            "tool_policy": Gene(
                name="tool_policy", type=GeneType.SET,
                value=tools, available=available_tools,
            ),
        },
    )


async def evaluate_genome(genome: Genome, agent, questions) -> tuple[float, float]:
    """Run a genome on the question set, return (mean_brier, mean_skill)."""
    truth_lookup = {q.prompt: q.outcome for q in questions}
    preds = []
    outs = []
    for q in questions:
        result = await agent(genome.to_dict(), q.prompt)
        preds.append(float(result))
        outs.append(q.outcome)
    return brier_score(preds, outs), 1.0 - brier_score(preds, outs)


def test_skill_function_zero_for_minimal_genome():
    """A genome with bad gene values should have low skill."""
    g = build_test_genome(
        prompt="market_trader",
        heuristic="minimax_regret",
        calibration="three_interval_check",
        decomposition="analogy",
        memory="none",
        answer_format="probability_only",
        temperature=1.4,
        risk=0.49,
        tools=[],
    )
    skill = _genome_skill(g.to_dict())
    assert skill < 0.30, f"Bad genome should have low skill, got {skill:.3f}"


def test_skill_function_high_for_optimal_genome():
    """A genome with the best gene values + interactions should have high skill."""
    g = build_test_genome(
        prompt="outside_view_first",
        heuristic="expected_value",
        calibration="tetlock_curve_check",
        decomposition="reference_class",  # Triggers interaction bonus with outside_view_first
        memory="episodic",
        answer_format="structured_json",  # Triggers bonus with reference_class
        temperature=0.5,
        risk=0.05,
        tools=["web_search", "calculator"],
    )
    skill = _genome_skill(g.to_dict())
    assert skill > 0.45, f"Optimal genome should have high skill, got {skill:.3f}"


def test_interaction_bonus_active():
    """The (outside_view_first, reference_class) bonus should fire."""
    g_with = build_test_genome(
        prompt="outside_view_first",
        decomposition="reference_class",
    )
    g_without = build_test_genome(
        prompt="outside_view_first",
        decomposition="causal_chain",  # No bonus
    )
    skill_with = _genome_skill(g_with.to_dict())
    skill_without = _genome_skill(g_without.to_dict())
    # The interaction bonus is 0.08; should appear in the difference
    assert skill_with > skill_without
    diff = skill_with - skill_without
    # 0.08 (interaction) + (0.08 - 0.05) marginal diff = 0.11 expected
    assert diff > 0.05, f"Interaction should add ~0.08 to skill, got diff {diff:.3f}"


async def test_high_skill_beats_low_skill_on_brier():
    """A high-skill genome should achieve a better Brier score than a low-skill one."""
    questions = load_builtin_questions()
    truth_lookup = {q.prompt: q.outcome for q in questions}
    agent = make_synthetic_agent(truth_lookup, noise_level=0.05)

    high = build_test_genome(
        prompt="outside_view_first", heuristic="expected_value",
        calibration="tetlock_curve_check", decomposition="reference_class",
        memory="episodic", answer_format="structured_json",
        temperature=0.5, risk=0.05, tools=["web_search", "calculator"],
        genome_id="high",
    )
    low = build_test_genome(
        prompt="market_trader", heuristic="minimax_regret",
        calibration="three_interval_check", decomposition="analogy",
        memory="none", answer_format="probability_only",
        temperature=1.4, risk=0.49, tools=[],
        genome_id="low",
    )

    high_brier, _ = await evaluate_genome(high, agent, questions)
    low_brier, _ = await evaluate_genome(low, agent, questions)

    print(f"High-skill Brier: {high_brier:.4f}")
    print(f"Low-skill  Brier: {low_brier:.4f}")
    # Threshold tuned against the deterministic content-hash agent on the
    # 50-question built-in dataset. With Brier in [0, 1] and binary outcomes,
    # a 0.025 gap corresponds to ~2-3 questions flipped between correct and
    # wrong, which is enough to confirm skill matters at this dataset size.
    gap = low_brier - high_brier
    assert gap > 0.025, (
        f"High-skill should have meaningfully lower Brier than low-skill; "
        f"got high={high_brier:.4f}, low={low_brier:.4f}, gap={gap:.4f}"
    )


async def test_determinism():
    """Same genome+task = same prediction across calls."""
    questions = load_builtin_questions()
    truth_lookup = {q.prompt: q.outcome for q in questions}
    agent = make_synthetic_agent(truth_lookup)
    g = build_test_genome(genome_id="det-test")

    p1 = await agent(g.to_dict(), questions[0].prompt)
    p2 = await agent(g.to_dict(), questions[0].prompt)
    assert p1 == p2


async def test_random_genomes_skill_distribution():
    """Random genomes should span a meaningful skill range."""
    rng = random.Random(42)
    skills = []
    prompts = list(DEFAULT_EFFECTS.prompt_template_marginals.keys())
    heuristics = list(DEFAULT_EFFECTS.decision_heuristic_marginals.keys())
    calibrations = list(DEFAULT_EFFECTS.calibration_rule_marginals.keys())
    decompositions = list(DEFAULT_EFFECTS.decomposition_style_marginals.keys())
    memories = list(DEFAULT_EFFECTS.memory_structure_marginals.keys())
    formats = list(DEFAULT_EFFECTS.answer_format_marginals.keys())
    available_tools = ["web_search", "calculator", "code_exec", "read"]

    for i in range(50):
        g = build_test_genome(
            prompt=rng.choice(prompts),
            heuristic=rng.choice(heuristics),
            calibration=rng.choice(calibrations),
            decomposition=rng.choice(decompositions),
            memory=rng.choice(memories),
            answer_format=rng.choice(formats),
            temperature=rng.uniform(0.0, 1.5),
            risk=rng.uniform(0.01, 0.5),
            tools=rng.sample(available_tools, k=rng.randint(0, 4)),
            genome_id=f"r{i}",
        )
        skills.append(_genome_skill(g.to_dict()))

    skills = np.array(skills)
    print(f"Random skill distribution: min={skills.min():.3f} max={skills.max():.3f} "
          f"mean={skills.mean():.3f} std={skills.std():.3f}")
    assert skills.std() > 0.05, "Skill should vary across random genomes"
    assert skills.max() - skills.min() > 0.20, "Skill range should be meaningful"


async def main():
    print("=" * 60)
    print("Synthetic Agent Sanity Tests")
    print("=" * 60)

    test_skill_function_zero_for_minimal_genome()
    print("PASS: minimal genome has low skill")

    test_skill_function_high_for_optimal_genome()
    print("PASS: optimal genome has high skill")

    test_interaction_bonus_active()
    print("PASS: cross-component interaction bonus fires")

    await test_high_skill_beats_low_skill_on_brier()
    print("PASS: high-skill beats low-skill on Brier")

    await test_determinism()
    print("PASS: agent is deterministic")

    await test_random_genomes_skill_distribution()
    print("PASS: random genomes span meaningful skill range")

    print("=" * 60)
    print("ALL SANITY TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
