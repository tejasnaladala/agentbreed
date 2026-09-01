# AgentBreed

**700-evaluation deterministic synthetic pilot complete; real-LLM replication
preregistered and pending.**

AgentBreed is a Python library and research package for searching typed agent
configurations: prompts, tool sets, model choices, numeric parameters, and
multi-component policies.

## Research status

The pilot comprises 600 E1 run records (3 synthetic domains x 10 methods x 20
seeds) and 100 E2 gene-dropout records. It uses a content-hash agent with
designed cross-gene interactions and contains no real-LLM outputs.

[`real_study/`](real_study/) contains the locked preregistration and a harness
skeleton. Its status states that no real-LLM runs have been executed, so the
pilot statistics apply only to the synthetic setup.

## Install from source

```bash
git clone https://github.com/tejasnaladala/agentbreed.git
cd agentbreed
python -m pip install -e .
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

`evolve` creates a population of configurations, evaluates each one with the
supplied tasks and scorer, applies selection, crossover, and mutation, and
returns the highest-scoring genome.

## What it does

```
Initialize 20 random genomes
    |
    v
Evaluate all on your tasks  -->  Score with your scorer
    |
    v
Retain top 40%
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

Genes are typed configuration variables. Five types are implemented:

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

## Framework integration

The integration boundary is an async callable that receives a configuration
dictionary and a task string, then returns a result string.

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

## Research package

The synthetic pilot studies search operators and configuration-space richness
under matched compute budgets. Its 700 run-level evaluations use a
deterministic content-hash agent with explicit cross-component interactions;
they were exploratory and were not the preregistered real-LLM execution.

Within that synthetic setup:

- Multi-component search exceeded prompt-only search (pooled paired *d_z* =
  1.32, 95% CI for the mean difference [+0.206, +0.303], *p* approximately
  1e-14 across 60 seed-domain pairs).
- Pairwise comparisons among full evolution, mutation-only, crossover-only,
  random search, and Bayesian optimization did not reach Holm-corrected
  significance. This result does not establish equivalence between operators.

The pilot's Sobol/Saltelli H3 output is retained for audit but is not treated as
a valid result: `n_base = 512` produced variance shares above 1.0. The locked
real-study protocol requires `n_base >= 2048`.

[`synthetic_pilot/`](synthetic_pilot/) documents the scope correction and the
claims that require replication. [`real_study/`](real_study/) contains the
pending real-LLM study. Pilot reproduction commands and artifact paths are in
[`paper/06_reproducibility/REPRODUCE.md`](paper/06_reproducibility/REPRODUCE.md).
Library tests live in [`tests/`](tests/), with separate harness tests under
[`real_study/harness/tests/`](real_study/harness/tests/).

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
from breed.adapters import CallableAdapter           # any async function
from breed.adapters import AnthropicMessagesAdapter  # optional "anthropic" extra
from breed.adapters import OpenAIAdapter              # optional "openai" extra
```

## Built-in arenas

```python
from breed import ForecastingArena  # 33 built-in prediction questions, Brier score
from breed import CodingArena       # task generation only; evaluation is disabled
from breed import CustomArena       # your tasks + your scorer
```

`CodingArena.evaluate()` is intentionally disabled because it previously ran
model-generated Python inside the coordinator process. Agent-generated code is
not executed. The arena will remain disabled until evaluation runs in a
disposable OS-level sandbox with no host secrets or network and strict process,
filesystem, CPU, memory, and wall-time limits. A Python namespace, regex or AST
filter, or subprocess timeout alone is not a sandbox.

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
python -m pip install -e ".[cli]"
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
python -m pip install -e .                 # core library
python -m pip install -e ".[anthropic]"   # + Claude adapter
python -m pip install -e ".[openai]"      # + OpenAI adapter
python -m pip install -e ".[cli]"         # + terminal commands
python -m pip install -e ".[charts]"      # + matplotlib charts
python -m pip install -e ".[all]"         # all optional dependencies
```

## Examples

- [`examples/minimal.py`](examples/minimal.py) -- 10-line quickstart
- [`examples/prediction_system.py`](examples/prediction_system.py) -- evolve a forecasting pipeline
- [`examples/multi_agent_swarm.py`](examples/multi_agent_swarm.py) -- evolve a multi-agent crew
- [`examples/swarm_integration.py`](examples/swarm_integration.py) -- custom 7-gene swarm genome

## License

MIT
