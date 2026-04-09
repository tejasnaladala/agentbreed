"""Forecasting arena: probability estimation on binary-outcome questions.

Agents receive questions with known binary outcomes and must output a
probability estimate.  Fitness is scored via Brier score and calibration
error.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import Arena, EvalResult, Task
from breed.fitness import BrierScoreEvaluator, calibration_error
from breed.genome import Genome

# ---------------------------------------------------------------------------
# Built-in sample questions (for offline testing / demos)
# ---------------------------------------------------------------------------

_BUILTIN_QUESTIONS: list[dict[str, Any]] = [
    # Technology
    {
        "prompt": "Will global average temperature in 2025 exceed 1.5C above pre-industrial levels?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will SpaceX Starship complete an orbital flight by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will Apple release a mixed-reality headset by end of 2023?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will a fully autonomous robotaxi service operate in a major US city by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will OpenAI release GPT-5 by end of 2024?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will quantum computers break RSA-2048 encryption by 2026?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will a humanoid robot be commercially available for under $20,000 by end of 2025?",
        "outcome": 0.0,
    },
    # Science
    {
        "prompt": "Will the first pig-to-human heart transplant recipient survive more than 2 years?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will nuclear fusion achieve net energy gain in a lab by end of 2023?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will CRISPR gene therapy receive FDA approval for sickle cell disease by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will an asteroid mining mission launch by 2026?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will the James Webb Space Telescope discover signs of life on an exoplanet by 2025?",
        "outcome": 0.0,
    },
    # Economics
    {
        "prompt": "Will the US enter a recession (two consecutive quarters of negative GDP growth) in 2024?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will Bitcoin exceed $100,000 USD at any point in 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will global inflation average below 4% in 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will the US federal funds rate drop below 4% by end of 2024?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will the EU impose a digital services tax exceeding 5% on Big Tech by 2025?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will China's GDP growth exceed 5% in 2024?",
        "outcome": 1.0,
    },
    # Politics / Geopolitics
    {
        "prompt": "Will there be a ceasefire in the Russia-Ukraine conflict lasting >30 days by end of 2024?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will the 2024 US presidential election have a voter turnout exceeding 60%?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will Sweden join NATO by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will the UN adopt a binding global AI governance treaty by 2025?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will India surpass China in population by end of 2023?",
        "outcome": 1.0,
    },
    # AI / Machine Learning
    {
        "prompt": "Will an AI system score above 90th percentile on the bar exam by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will the EU AI Act be formally adopted by mid-2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will an AI-generated movie win a major film festival award by 2025?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will deepfake detection achieve above 99% accuracy on standard benchmarks by 2025?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will any country ban generative AI for public use by end of 2024?",
        "outcome": 0.0,
    },
    # Miscellaneous
    {
        "prompt": "Will the 2024 Summer Olympics in Paris proceed without major security incidents?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will a self-driving truck complete a cross-country US delivery by end of 2024?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will global CO2 emissions decrease year-over-year in 2024 compared to 2023?",
        "outcome": 0.0,
    },
    {
        "prompt": "Will the world population exceed 8.1 billion by end of 2024?",
        "outcome": 1.0,
    },
    {
        "prompt": "Will lab-grown meat be available in US grocery stores by end of 2025?",
        "outcome": 0.0,
    },
]


# ---------------------------------------------------------------------------
# Probability parser
# ---------------------------------------------------------------------------


def parse_probability(text: str) -> float:
    """Extract a probability value from agent output.

    Handles:
        - Plain float: "0.75"
        - Percentage: "75%"
        - Embedded text: "I estimate 75% probability"
        - JSON object: {"probability": 0.75}
        - Fallback: 0.5 if nothing parseable is found.

    Returns:
        A float clamped to [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.5

    cleaned = text.strip()

    # Attempt JSON parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            for key in ("probability", "prob", "estimate", "prediction", "p"):
                if key in parsed:
                    val = float(parsed[key])
                    return _clamp_probability(val)
        if isinstance(parsed, (int, float)):
            val = float(parsed)
            return _clamp_probability(val)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Percentage pattern: "75%" or "75 %"
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", cleaned)
    if pct_match:
        val = float(pct_match.group(1)) / 100.0
        return _clamp_probability(val)

    # Plain float anywhere in text: "0.75" or "probability is 0.82"
    float_match = re.search(r"(?<!\d)([01](?:\.\d+)?)\b", cleaned)
    if float_match:
        val = float(float_match.group(1))
        return _clamp_probability(val)

    # Broader float search for values like "0.75" embedded in longer strings
    broad_match = re.search(r"(\d+\.\d+)", cleaned)
    if broad_match:
        val = float(broad_match.group(1))
        if 0.0 <= val <= 1.0:
            return val
        if 0.0 < val <= 100.0:
            return _clamp_probability(val / 100.0)

    return 0.5


def _clamp_probability(val: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, val))


# ---------------------------------------------------------------------------
# Forecasting Arena
# ---------------------------------------------------------------------------


class ForecastingArena(Arena):
    """Arena for probability estimation on binary-outcome questions.

    Uses BrierScoreEvaluator for fitness and reports calibration error
    in the breakdown.

    Attributes:
        questions: The pool of questions to draw from.
    """

    def __init__(self, questions: list[dict[str, Any]] | None = None) -> None:
        """Initialise with an optional custom question pool.

        Args:
            questions: List of dicts with 'prompt' and 'outcome' keys.
                       Defaults to the built-in sample set.
        """
        self.questions: list[dict[str, Any]] = (
            list(questions) if questions is not None else list(_BUILTIN_QUESTIONS)
        )
        self._brier = BrierScoreEvaluator()

    async def generate_tasks(self, count: int, seed: int | None = None) -> list[Task]:
        """Sample *count* forecasting tasks from the question pool.

        Args:
            count: Number of tasks to generate.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A list of Task objects, each with expected=outcome float.
        """
        rng = random.Random(seed)
        pool = list(self.questions)

        if count > len(pool):
            # Allow oversampling with replacement
            selected = rng.choices(pool, k=count)
        else:
            selected = rng.sample(pool, count)

        tasks: list[Task] = []
        for idx, q in enumerate(selected):
            tasks.append(
                Task(
                    task_id=f"forecast-{idx:04d}",
                    prompt=q["prompt"],
                    expected=float(q["outcome"]),
                    metadata={"domain": "forecasting"},
                )
            )
        return tasks

    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        """Run the agent on each task, parse probabilities, and score.

        Args:
            genome: The genome defining the agent.
            adapter: Translates the genome into a runnable agent.
            tasks: The forecasting tasks to evaluate on.

        Returns:
            EvalResult with Brier-based fitness and calibration breakdown.
        """
        predictions: list[float] = []
        outcomes: list[float] = []
        agent_results: list[AgentResult] = []
        total_tokens = 0

        for task in tasks:
            result = await adapter.run(genome, task.prompt)
            agent_results.append(result)
            total_tokens += result.cost_tokens

            if result.success:
                prob = parse_probability(result.output)
            else:
                prob = 0.5  # default on failure

            predictions.append(prob)
            outcomes.append(float(task.expected))

        brier_fitness = self._brier.evaluate(predictions, outcomes)
        cal_error = calibration_error(predictions, outcomes)

        return EvalResult(
            genome_id=genome.genome_id,
            fitness_score=brier_fitness,
            fitness_breakdown={
                "brier_fitness": brier_fitness,
                "calibration_error": cal_error,
            },
            task_results=agent_results,
            cost_tokens=total_tokens,
        )
