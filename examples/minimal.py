"""Minimal example -- evolve an agent in 10 lines."""

import asyncio
from breed import evolve

async def my_agent(config: dict, task: str) -> str:
    """Your agent. breed passes the genome config + task, you return a result."""
    prompt = config["genes"]["approach"]["value"]
    # In a real project: call your LLM / swarm / pipeline here
    return f"Answer using {prompt}: {task}"

def score(output: str, expected) -> float:
    """Your scorer. 0.0 = bad, 1.0 = perfect."""
    return 1.0 if len(output) > 20 else 0.0

async def main():
    champion = await evolve(
        agent=my_agent,
        tasks=["What is AI?", "Explain quantum computing", "Summarize climate change"],
        scorer=score,
        genes={
            "approach": {
                "type": "text",
                "templates": [
                    "Be concise and factual",
                    "Use analogies and examples",
                    "Think step by step",
                    "Consider multiple perspectives",
                ],
                "mutation_rate": 0.3,
            },
            "temperature": {
                "type": "float",
                "range": (0.0, 1.5),
                "mutation_rate": 0.2,
            },
        },
        generations=5,
        population_size=10,
        verbose=True,
    )
    print(f"\nChampion: {champion.genes['approach'].value}")

if __name__ == "__main__":
    asyncio.run(main())
