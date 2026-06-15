"""Example: Evolving a multi-agent swarm with breed.

This shows how to define a CUSTOM genome for your own swarm,
with genes that match YOUR agents' configuration knobs.
"""

from __future__ import annotations

import asyncio
import hashlib

from breed.adapters import CallableAdapter
from breed.arenas.base import Task
from breed.arenas.custom import CustomArena
from breed.engine import BreedingConfig, BreedingEngine

# ---------------------------------------------------------------------------
# Step 1: Define YOUR swarm's genome -- these are YOUR configuration knobs
# ---------------------------------------------------------------------------

MY_SWARM_GENES: dict = {
    "researcher_prompt": {
        "type": "text",
        "templates": [
            "You are a thorough researcher. Find primary sources and cite them.",
            "You are a fast researcher. Prioritize speed over depth.",
            "You are a contrarian researcher. Look for evidence against the consensus.",
            "You are a systematic researcher. Use structured frameworks for analysis.",
        ],
        "mutation_rate": 0.2,
    },
    "writer_prompt": {
        "type": "text",
        "templates": [
            "You are a concise technical writer. Use bullet points and clear structure.",
            "You are a narrative writer. Tell a compelling story with the research.",
            "You are an academic writer. Use formal language and citations.",
        ],
        "mutation_rate": 0.2,
    },
    "delegation_strategy": {
        "type": "enum",
        "options": ["sequential", "parallel", "hierarchical", "consensus"],
        "mutation_rate": 0.15,
    },
    "tools_enabled": {
        "type": "set",
        "available": [
            "web_search",
            "code_exec",
            "file_read",
            "api_call",
            "database_query",
        ],
        "mutation_rate": 0.1,
    },
    "max_iterations": {
        "type": "float",
        "range": (1.0, 10.0),
        "mutation_rate": 0.1,
    },
    "creativity": {
        "type": "float",
        "range": (0.0, 1.5),
        "mutation_rate": 0.1,
    },
    "quality_weights": {
        "type": "numeric_vector",
        "templates": [
            [0.4, 0.3, 0.3],  # balanced
            [0.7, 0.2, 0.1],  # accuracy-focused
            [0.2, 0.5, 0.3],  # speed-focused
        ],
        "mutation_rate": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Step 2: Define how to run your swarm with a genome
# ---------------------------------------------------------------------------


async def run_my_swarm(genome_dict: dict, task: str) -> str:
    """Your swarm execution logic. breed calls this for every genome on every task."""
    genes = genome_dict["genes"]

    # In a real integration, you'd configure your actual swarm here:
    # researcher = Agent(prompt=genes["researcher_prompt"]["value"], ...)
    # writer = Agent(prompt=genes["writer_prompt"]["value"], ...)
    # crew = Crew(agents=[researcher, writer],
    #             process=genes["delegation_strategy"]["value"])
    # result = crew.kickoff(task)

    # For this demo, simulate based on genome traits
    h = hashlib.md5((genome_dict["genome_id"] + task).encode()).hexdigest()
    base_score = int(h[:8], 16) / 0xFFFFFFFF

    # Higher creativity + more tools = better results in this sim
    creativity = genes["creativity"]["value"]
    tool_count = len(genes["tools_enabled"]["value"])
    bonus = (creativity * 0.1) + (tool_count * 0.05)

    return f"Research output (quality: {base_score + bonus:.2f})"


# ---------------------------------------------------------------------------
# Step 3: Define your evaluation tasks and scorer
# ---------------------------------------------------------------------------

TASKS = [
    Task(task_id="1", prompt="Research AI agent frameworks", expected="comprehensive"),
    Task(
        task_id="2",
        prompt="Research quantum computing applications",
        expected="comprehensive",
    ),
    Task(
        task_id="3",
        prompt="Research climate tech investment trends",
        expected="comprehensive",
    ),
]


def score_output(output: str, expected: str) -> float:
    """Your quality scorer."""
    try:
        quality = float(output.split("quality: ")[1].rstrip(")"))
        return min(max(quality, 0.0), 1.0)
    except (IndexError, ValueError):
        return 0.3


# ---------------------------------------------------------------------------
# Step 4: Run the breeding loop
# ---------------------------------------------------------------------------


async def main() -> None:
    # Tip: increase population_size and generations for real workloads.
    # These small values are for quick demonstration.
    config = BreedingConfig(
        population_size=8,
        elite_count=3,
        generations=5,
        mutation_rate=0.2,
        num_tasks=3,
        immigration_rate=0.0,
        seed=42,
    )

    arena = CustomArena(tasks=TASKS, scorer=score_output)
    adapter = CallableAdapter(run_my_swarm)

    engine = BreedingEngine(
        config=config,
        arena=arena,
        adapter=adapter,
        gene_template=MY_SWARM_GENES,  # <-- YOUR custom genes
    )

    champion = await engine.run()

    print("\n=== Champion Swarm Configuration ===")
    print(champion.to_yaml())


if __name__ == "__main__":
    asyncio.run(main())
