<div align="center">

# breed

**Evolve AI agents through Darwinian selection.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen.svg)]()

*Don't design agents. Breed them.*

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [The Breeding Pit](#the-breeding-pit) · [Adapters](#adapters) · [Arenas](#arenas) · [Research](#research)

</div>

---

We stopped tuning prompts. We started breeding minds.

```
  Gen 0   avg fitness 0.3214   champion: 6a1f9c2e...
  Gen 5   avg fitness 0.5871   champion: d04b8e31...
  Gen 10  avg fitness 0.7403   champion: a99e7f10...
  Gen 24  avg fitness 0.8216   champion: a99e7f10...   <-- 2.5x improvement, zero manual tuning
```

## What is breed?

breed treats AI agents as organisms with genomes -- 10 evolvable genes that control everything from prompt strategy to tool selection to calibration rules. Populations of agents compete in arenas, the fittest survive and reproduce, and over generations the population converges on strategies no human would have designed. It works with Claude, GPT-4o, or any async callable you can write.

## Why breed?

| Approach | What you do | What you get |
|---|---|---|
| Manual prompt tuning | Tweak system prompts by hand, test, repeat | Local optima you could imagine |
| DSPy / EvoPrompt | Optimize prompt tokens via gradient signal | Better prompts, same architecture |
| **breed** | Evolve the entire agent phenotype -- prompt, tools, memory, calibration, decomposition, temperature | **Strategies you couldn't have designed** |

breed doesn't optimize a string. It evolves a *mind*: which tools the agent reaches for, how it decomposes problems, how much risk it tolerates, how it weights evidence, and how it calibrates confidence. These traits recombine through crossover and mutate across generations. The result is agents that are genuinely alien in their approach -- and measurably better.

## Quick Start

```bash
pip install agentbreed
breed init --arena forecasting
breed run --generations 10
```

**No API key needed.** Demo mode uses a `CallableAdapter` with deterministic random probabilities, so you see the full breeding loop -- spawn, evaluate, select, breed, mutate, immigrate -- without spending a cent.

You'll see output like this:

```
  breed v0.1.0  --  Evolving agents

  Generation 7/10  ████████████████████████░░░░░░  70%  elapsed 4.2s

  Generation Stats
  ┌─────┬────────┬────────┬────────┬─────┬──────────────────┐
  │ Gen │   Best │    Avg │  Worst │ Pop │ Champion         │
  ├─────┼────────┼────────┼────────┼─────┼──────────────────┤
  │   0 │ 0.7218 │ 0.3214 │ 0.0512 │  20 │ 6a1f9c2e1b34...  │
  │   1 │ 0.7301 │ 0.3892 │ 0.0891 │  20 │ 6a1f9c2e1b34...  │
  │   2 │ 0.7544 │ 0.4201 │ 0.1024 │  20 │ d04b8e31aa92...  │
  │   ...                                                     │
  │   7 │ 0.8102 │ 0.6217 │ 0.2901 │  20 │ a99e7f10cc43...  │
  └─────┴────────┴────────┴────────┴─────┴──────────────────┘

  * New champion! a99e7f10... (fitness 0.8102)
  x Died: 3b2c1d44... (fitness 0.0891)
  > 2 immigrants injected
```

Then inspect the winner:

```bash
breed champion
```

```
  Champion Genome: a99e7f10cc43...

  Gene                   Type     Value
  ─────────────────────────────────────────────────────────────
  prompt_template        text     "Adopt the outside view first: find the historical
                                   base rate for this category of event, then adjust
                                   inward based on inside-view evidence."
  memory_structure       enum     episodic
  tool_policy            set      [calculator, web_search]
  decision_heuristic     text     "Decompose into Fermi sub-estimates, sanity-check
                                   each against known anchors..."
  calibration_rule       text     "Apply the extremeness aversion correction: if your
                                   estimate is above 90% or below 10%..."
  evidence_weighting     vector   [0.5, 0.2, 0.3]
  risk_threshold         float    0.127
  decomposition_style    enum     reference_class
  temperature            float    0.42
  answer_format          enum     probability_with_reasoning
```

The agent evolved to prefer outside-view reasoning, low temperature, and a reference-class decomposition. Nobody told it to do that.

## 30-Second Python API

```python
import asyncio
from breed import __version__
from breed.engine import BreedingEngine, BreedingConfig
from breed.arenas.forecasting import ForecastingArena
from breed.adapters import CallableAdapter

async def my_agent(genome_dict: dict, task: str) -> str:
    """Your agent logic here. Return a string."""
    return "0.75"

async def main():
    engine = BreedingEngine(
        config=BreedingConfig(
            population_size=20,
            generations=10,
            elite_count=4,
            mutation_rate=0.15,
        ),
        arena=ForecastingArena(),
        adapter=CallableAdapter(my_agent),
    )
    champion = await engine.run()
    print(f"Champion: {champion.genome_id}")
    print(f"Fitness:  {champion.fitness_history}")

asyncio.run(main())
```

## How It Works

```
                         ┌─────────────────────┐
                         │   Spawn Population   │
                         │   (random genomes)   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │         Evaluate               │
                    │  (run agents against arena)    │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          Score                  │
                    │  (Brier score, pass rate, ...)  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          Select                 │
                    │  (tournament / truncation)      │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼──────────┐ ┌───────▼────────┐ ┌──────────▼─────────┐
    │      Breed          │ │    Mutate       │ │    Immigrate       │
    │  (crossover ops)    │ │  (6 operators)  │ │  (diversity boost) │
    └─────────┬──────────┘ └───────┬────────┘ └──────────┬─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                            ┌───────▼───────┐
                            │    Repeat      │
                            └───────────────┘
```

**Crossover operators** (3): uniform, component_swap, weighted -- different strategies for combining parent genomes into offspring.

**Mutation operators** (6): parameter_jitter, tool_swap, strategy_mutation, prompt_perturb_simple, vector_jitter, calibration_adjustment -- each targeting a specific gene type.

**Selection**: Tournament (default, k=3) or truncation. Elites survive automatically.

**Immigration**: Random genomes injected each generation to maintain diversity and prevent premature convergence.

## The Genome

Every agent carries a genome of 10 genes:

| Gene | Type | What it controls |
|---|---|---|
| `prompt_template` | text | Core system prompt strategy |
| `memory_structure` | enum | How the agent organizes context (sliding_window, episodic, semantic, graph, none) |
| `tool_policy` | set | Which tools the agent is allowed to use |
| `decision_heuristic` | text | Reasoning strategy (expected-value, minimax regret, Fermi decomposition, ...) |
| `calibration_rule` | text | How the agent corrects for overconfidence |
| `evidence_weighting` | vector | Relative weights for different evidence categories |
| `risk_threshold` | float | Willingness to make extreme predictions (0.01 - 0.50) |
| `decomposition_style` | enum | Problem breakdown method (reference_class, causal_chain, scenario_tree, fermi, analogy) |
| `temperature` | float | LLM sampling temperature (0.0 - 1.5) |
| `answer_format` | enum | Output structure (probability_only, probability_with_reasoning, structured_json) |

Genomes are serializable to YAML and JSON. They carry full lineage metadata: genome ID, generation number, parent IDs, and fitness history.

## The Breeding Pit

The Breeding Pit is a real-time web UI for watching evolution happen.

```bash
breed serve --port 8420
```

Built with FastAPI and canvas-based physics, it renders the population as a living system:

- **Agent blobs** float in the arena, sized by fitness score and colored on a red-to-green gradient. Low-fitness agents are small and red. High-fitness agents are large and green.
- **Breeding**: when parents are selected for crossover, their blobs drift together, merge with a heart animation, and burst into particles as the offspring appears.
- **Death**: culled agents shrink, fade, and receive a tombstone epitaph with their final fitness score.
- **Champion coronation**: when a new champion emerges, it gets a golden crown and confetti animation.
- **Live stats**: generation counter, fitness curves, and event log update via WebSocket in real time.

The web UI is the memetic hook. Screen-record a run and post it.

## Adapters

breed wraps any agent framework. The adapter layer translates genomes into runnable agents.

```python
# Claude (requires: pip install agentbreed[anthropic])
from breed.adapters import AnthropicMessagesAdapter
adapter = AnthropicMessagesAdapter(model="claude-sonnet-4-6")

# OpenAI (requires: pip install agentbreed[openai])
from breed.adapters import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-4o")

# Any async function -- zero dependencies
from breed.adapters import CallableAdapter

async def my_agent(genome_dict: dict, task: str) -> str:
    # genome_dict contains all 10 genes -- use them however you want
    prompt = genome_dict["genes"]["prompt_template"]["value"]
    temp = genome_dict["genes"]["temperature"]["value"]
    # ... call your agent framework, local model, whatever
    return "0.85"

adapter = CallableAdapter(my_agent)
```

The `CallableAdapter` is the escape hatch. If you can write an async function that takes `(genome_dict, task_str) -> result_str`, you can breed it.

## Arenas

Arenas define what agents compete on and how they're scored.

### ForecastingArena

32 built-in binary prediction questions with known outcomes. Agents output probability estimates, scored by Brier score and calibration error. Good for evolving calibrated reasoners.

```python
from breed.arenas import ForecastingArena
arena = ForecastingArena()
```

### CodingArena

15 Python coding challenges with test cases. Agents write functions, code is extracted and executed in a sandbox, fitness = pass rate. Good for evolving code generation strategies.

```python
from breed.arenas import CodingArena
arena = CodingArena()
```

### CustomArena

Bring your own tasks and scoring function:

```python
from breed.arenas import CustomArena
from breed.arenas.base import Task

tasks = [
    Task(task_id="1", prompt="Summarize this article...", expected="key_points"),
    Task(task_id="2", prompt="Translate to French...", expected="Bonjour"),
]

def my_scorer(agent_output: str, expected) -> float:
    """Return a score in [0, 1]."""
    return 1.0 if expected.lower() in agent_output.lower() else 0.0

arena = CustomArena(tasks=tasks, scorer=my_scorer)
```

## Dual Mode: Headless + Interactive

breed adapts to your workflow through an observer pattern with 10 event types:

```bash
# Rich terminal display (default) -- progress bars, live stats, event icons
breed run

# Headless JSON output -- for CI/CD pipelines and scripting
breed run --headless

# Live web UI -- the Breeding Pit in your browser
breed run --serve
```

Programmatically, attach any observer to the event bus:

```python
from breed.events import EventBus, CallbackObserver

bus = EventBus()
bus.add_observer(CallbackObserver(lambda event: print(event.to_dict())))

engine = BreedingEngine(config=config, arena=arena, adapter=adapter, event_bus=bus)
```

Event types: `AgentBorn`, `AgentDied`, `AgentEvaluated`, `ChampionChanged`, `GenerationComplete`, `BreedingStarted`, `BreedingComplete`, `ImmigrantsInjected`, `CrossoverEvent`, `MutationEvent`.

## CLI Reference

| Command | Description |
|---|---|
| `breed init` | Initialize a new breeding project with config and directory structure |
| `breed run` | Execute the evolutionary loop |
| `breed results` | Display summary statistics from a completed run |
| `breed champion` | Print the winning genome in full detail |
| `breed lineage` | Trace ancestry of any genome (tree or table format) |
| `breed diff --gen-a 0 --gen-b 10` | Compare champions across generations |
| `breed export` | Export a genome as YAML or JSON |
| `breed serve` | Launch the Breeding Pit web UI on port 8420 |

All commands accept `--config path/to/breed.yaml` to point at a custom config.

## Research Use

breed is built for reproducible experiments.

**Seed-controlled reproducibility**: Set `seed` in `BreedingConfig` and every random decision -- population spawn, crossover, mutation, immigration -- is deterministic. Same seed, same evolution.

**JSONL lineage tracking**: Every genome, every generation, every fitness score is logged to an append-only JSONL file. Load it into pandas, R, or whatever you use for analysis.

**Ablations via config**: Want to test whether immigration helps? Set `immigration_rate: 0.0`. Want to compare tournament vs. truncation selection? Change one line. The config surface is small and explicit.

**Paper-ready charts** (requires `pip install agentbreed[charts]`):

```python
from breed.charts import fitness_curve, population_distribution, gene_persistence, lineage_tree_static

fitness_curve("results/lineage.jsonl", output="fitness.png")
population_distribution("results/lineage.jsonl", output="pop_dist.png")
gene_persistence("results/lineage.jsonl", gene="temperature", output="temp_persistence.png")
lineage_tree_static("results/lineage.jsonl", output="lineage_tree.png")
```

## Installation

```bash
# Core (no API dependencies)
pip install agentbreed

# With specific providers
pip install agentbreed[anthropic]     # Claude support
pip install agentbreed[openai]        # OpenAI support
pip install agentbreed[web]           # Breeding Pit web UI
pip install agentbreed[charts]        # Matplotlib charts

# Everything
pip install agentbreed[all]
```

Requires Python 3.11+.

## Contributing

breed is MIT licensed. PRs welcome.

Areas where contributions would be particularly valuable:

- **New arenas**: reasoning, debate, agentic tool use, multi-agent negotiation
- **New mutation operators**: LLM-guided prompt mutation, crossover with semantic similarity
- **New adapters**: LangChain, CrewAI, LlamaIndex, local models via vLLM/Ollama
- **Visualization**: richer web UI animations, D3-based lineage graphs
- **Distributed breeding**: run populations across multiple machines

```bash
git clone https://github.com/YOUR_USERNAME/agentbreed.git
cd agentbreed
pip install -e ".[dev,all]"
pytest
```

274 tests. They should all pass.

## Star History

<!-- Replace with actual star history chart once repo is public -->
<!-- [![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/agentbreed&type=Date)](https://star-history.com/#YOUR_USERNAME/agentbreed&Date) -->

---

<div align="center">

*breed: because the best agent architecture is the one that evolved to win.*

</div>
