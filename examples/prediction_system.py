"""Example: Evolve a prediction system.

Works with any prediction / forecasting pipeline.
breed optimizes the reasoning strategy, calibration, and parameters.
"""

import asyncio
import hashlib
from breed import evolve

# Your gene template -- these are the knobs breed will evolve
PREDICTION_GENES = {
    "reasoning_strategy": {
        "type": "text",
        "templates": [
            "Use base rates from historical data, then adjust with current evidence.",
            "Decompose into sub-questions, estimate each independently, combine.",
            "Find 3 reference cases, average their outcomes, adjust for differences.",
            "Consider bull case, bear case, and base case. Weight by probability.",
            "Start from market consensus, then adjust for information others might miss.",
        ],
        "mutation_rate": 0.2,
    },
    "calibration": {
        "type": "text",
        "templates": [
            "If confidence > 90%, shrink 10% toward 50%. Overconfidence kills.",
            "Apply no correction. Trust the analysis.",
            "Always widen confidence intervals by 20%. Uncertainty is underestimated.",
            "Use the outside view: your prediction should match base rates within 10%.",
        ],
        "mutation_rate": 0.15,
    },
    "evidence_weights": {
        "type": "numeric_vector",
        "templates": [
            [0.4, 0.35, 0.25],  # recency, quality, consensus
            [0.6, 0.2, 0.2],    # recency-heavy
            [0.2, 0.5, 0.3],    # quality-heavy
            [0.33, 0.33, 0.34], # balanced
        ],
        "mutation_rate": 0.1,
    },
    "risk_appetite": {
        "type": "float",
        "range": (0.01, 0.5),
        "mutation_rate": 0.1,
    },
    "model_choice": {
        "type": "enum",
        "options": ["conservative", "moderate", "aggressive", "contrarian"],
        "mutation_rate": 0.1,
    },
}


async def prediction_agent(config: dict, task: str) -> str:
    """Your prediction agent. Replace this with your actual prediction logic."""
    genes = config["genes"]
    strategy = genes["reasoning_strategy"]["value"]
    model = genes["model_choice"]["value"]
    risk = genes["risk_appetite"]["value"]

    # Simulate: in reality, you'd call your LLM / model here
    # The genome config determines HOW it reasons
    h = hashlib.md5((config["genome_id"] + task + model).encode()).hexdigest()
    base = int(h[:8], 16) / 0xFFFFFFFF

    # Different strategies produce different quality predictions
    if "base rates" in strategy:
        base += 0.05
    if "contrarian" in model:
        base += 0.03
    if risk < 0.2:
        base += 0.02  # conservative = slightly better calibrated

    return f"{min(base, 0.99):.4f}"


def brier_score(output: str, expected) -> float:
    """Score predictions using Brier score (lower = better, we invert)."""
    try:
        predicted = float(output.strip())
        actual = float(expected) if expected is not None else 0.5
        return 1.0 - (predicted - actual) ** 2
    except (ValueError, TypeError):
        return 0.0


async def main():
    # Your prediction tasks with known outcomes
    from breed import Task
    tasks = [
        Task(task_id="q1", prompt="Will AI revenue exceed $500B in 2026?", expected=0.8),
        Task(task_id="q2", prompt="Will BTC exceed $200k by end of 2026?", expected=0.3),
        Task(task_id="q3", prompt="Will US unemployment rise above 5% in 2026?", expected=0.2),
        Task(task_id="q4", prompt="Will GPT-5 be released before July 2026?", expected=0.7),
        Task(task_id="q5", prompt="Will EU pass new AI regulation in 2026?", expected=0.9),
    ]

    champion = await evolve(
        agent=prediction_agent,
        tasks=tasks,
        scorer=brier_score,
        genes=PREDICTION_GENES,
        generations=15,
        population_size=20,
        seed=42,
        verbose=True,
    )

    print("\n=== Winning Prediction Strategy ===")
    for name, gene in champion.genes.items():
        print(f"  {name}: {gene.value}")


if __name__ == "__main__":
    asyncio.run(main())
