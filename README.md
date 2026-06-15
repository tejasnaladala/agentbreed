# breed

Evolve any AI agent through Darwinian selection. `pip install` and add 3 lines to your project.

```
pip install agentbreed
```

## Quickstart

```python
from breed import evolve

async def my_agent(config, task):
    prompt = config["genes"]["strategy"]["value"]
    # your agent/swarm/pipeline logic here
    return result

champion = await evolve(
    agent=my_agent,
    tasks=["task 1", "task 2", "task 3"],
    scorer=lambda output, expected: score(output),
    genes={
        "strategy": {
            "type": "text",
            "templates": ["Be precise", "Think step by step", "Use examples"],
            "mutation_rate": 0.2,
        },
        "temperature": {
            "type": "float",
            "range": (0.0, 1.5),
            "mutation_rate": 0.1,
        },
    },
    generations=10,
    verbose=True,
)

# champion.genes has the winning config
```

That's it. breed spawns a population of agents with different configs, evaluates them on your tasks, keeps the fittest, breeds them, mutates the offspring, and repeats for N generations. You get back the champion genome.

## What it does

```
Spawn 20 random agents
    |
    v
Evaluate all on your tasks  -->  Score with your scorer
    |
    v
Kill bottom 60%
    |
    v
Breed survivors (crossover genes between top agents)
    |
    v
Mutate offspring (tweak prompts, tools, parameters)
    |
    v
Inject random immigrants (maintain diversity)
    |
    v
Repeat for N generations
    |
    v
Return champion genome
```

## Define your genes

A gene is any configuration knob you want breed to optimize. 5 types:

| Type | What it is | Example |
|------|-----------|---------|
| `text` | A prompt/instruction string, chosen from templates | System prompts, reasoning strategies |
| `enum` | One value from a fixed set | `"gpt-4o"` or `"claude-sonnet-4-6"`, delegation strategy |
| `float` | A number in a range | Temperature, risk threshold, confidence cutoff |
| `set` | A subset of available items | Which tools to enable, which APIs to call |
| `numeric_vector` | A list of floats (auto-normalized to sum=1) | Evidence weights, priority scores |

```python
genes = {
    "researcher_prompt": {
        "type": "text",
        "templates": [
            "Find primary sources and verify claims.",
            "Prioritize speed. Cover many angles.",
            "Actively seek disconfirming evidence.",
        ],
        "mutation_rate": 0.2,
    },
    "delegation": {
        "type": "enum",
        "options": ["sequential", "parallel", "hierarchical", "debate"],
        "mutation_rate": 0.15,
    },
    "tools": {
        "type": "set",
        "available": ["web_search", "code_exec", "database", "api_call"],
        "mutation_rate": 0.1,
    },
    "temperature": {
        "type": "float",
        "range": (0.0, 1.5),
        "mutation_rate": 0.1,
    },
    "priority_weights": {
        "type": "numeric_vector",
        "templates": [[0.5, 0.3, 0.2], [0.33, 0.33, 0.34]],
        "mutation_rate": 0.1,
    },
}
```

## Integrate with any framework

breed doesn't care what agent framework you use. Your agent function gets a config dict and a task string, returns a result string.

**CrewAI:**
```python
async def my_crew(config, task):
    genes = config["genes"]
    researcher = Agent(role="Researcher", goal=genes["researcher_prompt"]["value"])
    writer = Agent(role="Writer", goal=genes["writer_prompt"]["value"])
    crew = Crew(agents=[researcher, writer], process=genes["delegation"]["value"])
    return str(crew.kickoff(inputs={"topic": task}))
```

**LangGraph:**
```python
async def my_graph(config, task):
    genes = config["genes"]
    graph = build_graph(strategy=genes["routing"]["value"], tools=genes["tools"]["value"])
    return graph.invoke({"input": task})["output"]
```

**OpenAI / Claude / any LLM:**
```python
async def my_llm(config, task):
    genes = config["genes"]
    response = await client.chat.completions.create(
        model=genes["model"]["value"],
        temperature=genes["temperature"]["value"],
        messages=[{"role": "system", "content": genes["prompt"]["value"]},
                  {"role": "user", "content": task}],
    )
    return response.choices[0].message.content
```

**Any custom system:**
```python
async def my_pipeline(config, task):
    # config["genes"] has your evolved params -- use them however you want
    return run_my_thing(config["genes"], task)
```

## What breed evolves

- 3 crossover operators (uniform, component swap, fitness-weighted)
- 6 mutation operators (text swap, enum swap, float jitter, set add/remove, vector jitter, generic)
- Tournament and truncation selection
- Diversity preservation (novelty scoring, random immigrant injection)
- Immutable JSONL lineage tracking

## The research behind it

breed started as the artifact for a preregistered study asking a narrow question: when you optimize an LLM agent's configuration, what actually moves the needle — the search *operator* (crossover vs mutation vs Bayesian optimization vs successive halving), or the richness of the configuration *space* itself?

The pilot ran 700 preregistered runs (the headline set is 600: 3 domains × 10 methods × 20 seeds) against a deterministic synthetic agent built with explicit cross-component interactions, under matched compute budgets. Two findings held up:

- Multi-component search beats prompt-only search by a wide margin (pooled paired *d_z* = 1.32, 95% CI on the mean difference [+0.206, +0.303], *p* ≈ 1e-14 across 60 seed-domain pairs).
- Within multi-component methods, no operator pulled ahead of any other at Holm-corrected significance. Full evolution, mutation-only, crossover-only, random search, and Bayesian optimization were a wash.

The takeaway: spend your effort defining a richer configuration space, not on a fancier search operator. A Sobol/Saltelli variance decomposition (`breed/analysis/epistasis.py`) confirms that pairwise gene interactions, not single genes acting alone, account for a real share of the fitness variance.

Two things to be precise about:

- **The pilot agent is synthetic** (a content-hash function with designed interaction structure), not a real LLM. The repo labels it that way everywhere. It tests the *search question*, not real-model performance.
- **The real-LLM replication is designed, preregistered, and not yet run.** It lives in [`real_study/`](real_study/) as a locked preregistration plus a harness skeleton (benchmark fetch stubbed, one of the search methods wired up). Read its `README.md` before assuming any real-model result exists — none do yet.

The `breed` library itself is finished and tested: 431 unit tests in `tests/` (plus 35 in `real_study/harness/`), all passing. Reproduction steps for every number above are in [`paper/06_reproducibility/REPRODUCE.md`](paper/06_reproducibility/REPRODUCE.md).

## Python API

```python
from breed import evolve, evolve_sync, BreedingEngine, BreedingConfig
from breed import Population, Genome, Gene, GeneType
from breed import CallableAdapter, CustomArena, Task
from breed import EventBus, CallbackObserver, JSONObserver

# Simple: one function call
champion = await evolve(agent=fn, tasks=[...], scorer=score_fn, genes={...})

# Sync version (no async needed)
champion = evolve_sync(agent=fn, tasks=[...], scorer=score_fn, genes={...})

# Full control: build the engine yourself
engine = BreedingEngine(
    config=BreedingConfig(population_size=20, generations=10),
    arena=CustomArena(tasks=[...], scorer=score_fn),
    adapter=CallableAdapter(my_fn),
    gene_template=my_genes,
)
champion = await engine.run()
```

## Built-in adapters

```python
from breed.adapters import CallableAdapter          # any async function
from breed.adapters import AnthropicMessagesAdapter  # pip install agentbreed[anthropic]
from breed.adapters import OpenAIAdapter             # pip install agentbreed[openai]
```

## Built-in arenas

```python
from breed import ForecastingArena  # 33 built-in prediction questions, Brier score
from breed import CodingArena       # 15 coding problems with test cases
from breed import CustomArena       # your tasks + your scorer
```

## Events

```python
from breed import evolve, CallbackObserver

# Track progress
champion = await evolve(
    ...,
    on_generation=lambda gen, best, avg: print(f"Gen {gen}: {best:.4f}"),
)

# Or use the full event system
from breed.events import AgentBorn, AgentDied, ChampionChanged
```

## CLI (optional)

```
pip install agentbreed[cli]
breed init --arena forecasting
breed run --generations 10
breed results
breed champion
breed lineage
breed diff --gen-a 0 --gen-b 9
breed export --format json
```

## Install

```bash
pip install agentbreed              # core library (pydantic + pyyaml only)
pip install agentbreed[anthropic]   # + Claude adapter
pip install agentbreed[openai]      # + OpenAI adapter
pip install agentbreed[cli]         # + terminal commands
pip install agentbreed[charts]      # + matplotlib charts
pip install agentbreed[all]         # everything
```

## Examples

- [`examples/minimal.py`](examples/minimal.py) -- 10-line quickstart
- [`examples/prediction_system.py`](examples/prediction_system.py) -- evolve a forecasting pipeline
- [`examples/multi_agent_swarm.py`](examples/multi_agent_swarm.py) -- evolve a multi-agent crew
- [`examples/swarm_integration.py`](examples/swarm_integration.py) -- custom 7-gene swarm genome

## License

MIT
