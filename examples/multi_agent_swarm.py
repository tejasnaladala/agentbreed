"""Example: Evolve a multi-agent swarm.

This shows how to evolve the configuration of an entire swarm --
researcher, writer, reviewer -- where breed optimizes each agent's
prompt, the tools, the delegation strategy, and the team dynamics.

Works with any swarm framework: CrewAI, LangGraph, AutoGen, MiroFish, custom.
"""

import asyncio
import hashlib
from breed import evolve

# Define what's evolvable about YOUR swarm
SWARM_GENES = {
    # Agent-level genes
    "researcher_prompt": {
        "type": "text",
        "templates": [
            "You are a thorough researcher. Find primary sources and verify claims.",
            "You are a fast researcher. Prioritize breadth over depth. Cover many angles.",
            "You are a contrarian researcher. Actively seek disconfirming evidence.",
            "You are a systematic researcher. Use MECE frameworks to structure analysis.",
        ],
        "mutation_rate": 0.2,
    },
    "writer_prompt": {
        "type": "text",
        "templates": [
            "Write concise, structured reports with bullet points and clear headers.",
            "Write narrative reports that tell a compelling story with the data.",
            "Write executive summaries -- one page max, key findings only.",
            "Write detailed technical reports with methodology and limitations sections.",
        ],
        "mutation_rate": 0.2,
    },
    "reviewer_prompt": {
        "type": "text",
        "templates": [
            "Check for factual accuracy and logical consistency. Flag unsupported claims.",
            "Review for clarity and actionability. Can a decision-maker act on this?",
            "Look for missing perspectives and blind spots. What was overlooked?",
            "Grade on a rubric: accuracy (40%), clarity (30%), completeness (30%).",
        ],
        "mutation_rate": 0.2,
    },

    # Swarm-level genes
    "delegation_strategy": {
        "type": "enum",
        "options": ["sequential", "parallel", "hierarchical", "debate"],
        "mutation_rate": 0.15,
    },
    "tools_enabled": {
        "type": "set",
        "available": ["web_search", "code_exec", "file_read", "api_call", "database"],
        "mutation_rate": 0.1,
    },
    "max_rounds": {
        "type": "float",
        "range": (1.0, 5.0),
        "mutation_rate": 0.1,
    },
    "temperature": {
        "type": "float",
        "range": (0.0, 1.2),
        "mutation_rate": 0.1,
    },
}


async def run_swarm(config: dict, task: str) -> str:
    """Run your multi-agent swarm with the evolved configuration.

    REPLACE THIS with your actual swarm framework code.
    breed doesn't care what framework you use -- it just needs:
    - input: a genome config dict + a task string
    - output: a result string
    """
    genes = config["genes"]

    # In a real project, you'd do something like:
    #
    # from crewai import Agent, Crew, Task
    # researcher = Agent(role="Researcher", goal=genes["researcher_prompt"]["value"])
    # writer = Agent(role="Writer", goal=genes["writer_prompt"]["value"])
    # reviewer = Agent(role="Reviewer", goal=genes["reviewer_prompt"]["value"])
    # crew = Crew(agents=[researcher, writer, reviewer], process=genes["delegation_strategy"]["value"])
    # return str(crew.kickoff(inputs={"topic": task}))
    #
    # Or with MiroFish:
    # swarm = MiroFish(agents=[...], orchestration=genes["delegation_strategy"]["value"])
    # return swarm.run(task)

    # Simulation for demo purposes
    h = hashlib.md5((config["genome_id"] + task).encode()).hexdigest()
    base_quality = int(h[:8], 16) / 0xFFFFFFFF

    # Genome traits affect output quality in this sim
    strategy = genes["delegation_strategy"]["value"]
    tool_count = len(genes["tools_enabled"]["value"])
    temp = genes["temperature"]["value"]

    quality = base_quality
    if strategy == "debate":
        quality += 0.08  # debate produces better reasoning
    if strategy == "hierarchical":
        quality += 0.05
    quality += tool_count * 0.02  # more tools = more capability
    quality -= abs(temp - 0.7) * 0.1  # temperature sweet spot around 0.7

    output_lines = [
        f"Research by: {genes['researcher_prompt']['value'][:50]}...",
        f"Written by: {genes['writer_prompt']['value'][:50]}...",
        f"Reviewed by: {genes['reviewer_prompt']['value'][:50]}...",
        f"Strategy: {strategy}, Tools: {tool_count}, Quality: {quality:.3f}",
    ]
    return "\n".join(output_lines)


def score_output(output: str, expected) -> float:
    """Score the swarm's output quality."""
    try:
        # Extract quality from our simulation
        for line in output.split("\n"):
            if "Quality:" in line:
                q = float(line.split("Quality:")[1].strip())
                return min(max(q, 0.0), 1.0)
    except (ValueError, IndexError):
        pass
    return 0.3


async def main():
    tasks = [
        "Research the current state of AI agent frameworks and recommend the best one",
        "Analyze the competitive landscape for developer tools in 2026",
        "Write a technical brief on evolutionary optimization for non-technical stakeholders",
        "Compare and contrast 5 approaches to multi-agent orchestration",
    ]

    print("Evolving swarm configuration...\n")

    champion = await evolve(
        agent=run_swarm,
        tasks=tasks,
        scorer=score_output,
        genes=SWARM_GENES,
        generations=15,
        population_size=16,
        seed=42,
        verbose=True,
    )

    print("\n=== Champion Swarm Configuration ===")
    for name, gene in sorted(champion.genes.items()):
        val = gene.value
        if isinstance(val, str) and len(val) > 60:
            val = val[:60] + "..."
        print(f"  {name}: {val}")


if __name__ == "__main__":
    asyncio.run(main())
