# agentbreed Implementation Plan

**Goal:** Build `breed`, an open-source Python framework for evolving **any** AI agent through Darwinian selection. Provider-agnostic: works with Claude, OpenAI, local models, CrewAI, LangGraph, AutoGen, or any custom agent. Breeds, mutates, and optimizes whatever agents you're already running.

**Architecture:** Provider-agnostic evolutionary optimization. The genome schema is a universal agent configuration (prompt, tools, memory, heuristics, etc.) that adapts to any agent framework via pluggable **adapters**. Adapters translate genomes into framework-specific configs (Claude Managed Agents, OpenAI Agents SDK, CrewAI crews, LangGraph graphs, raw API calls, or custom callables). Arenas define task domains and fitness functions. The breeding engine orchestrates the evolutionary loop. Lineage is tracked in an immutable log and visualized via CLI and web UI.

**Key design principle:** breed doesn't replace your agent framework -- it wraps it. You bring your agents, breed evolves them. Over time, the agents in question get optimized through selection pressure.

**Adapter model:**
- `AnthropicMessagesAdapter` -- genome -> system prompt + tools -> `messages.create()`
- `ManagedAgentsAdapter` -- genome -> `POST /v1/agents` -> sessions -> SSE streaming
- `AgentSDKAdapter` -- genome -> `ClaudeAgentOptions` -> `query()`
- `OpenAIAdapter` -- genome -> `client.chat.completions.create()` with tools
- `CrewAIAdapter` -- genome -> CrewAI Agent/Task/Crew config
- `LangGraphAdapter` -- genome -> graph node/edge config
- `CallableAdapter` -- genome -> any `async (genome, task) -> result` function (escape hatch)

Users can write custom adapters in ~20 lines to evolve any agent system.

**Tech Stack:** Python 3.11+, Pydantic v2, Click, anthropic SDK (optional), openai SDK (optional), FastAPI, D3.js, matplotlib, pytest, YAML/JSON

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `breed/__init__.py`
- Create: `breed/py.typed`
- Create: `configs/forecasting.yaml`
- Create: `configs/coding.yaml`
- Create: `LICENSE`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentbreed"
version = "0.1.0"
description = "Evolve AI agents through Darwinian selection. Track lineage. Benchmark generations."
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{name = "Tejas"}]
keywords = ["ai", "agents", "evolution", "breeding", "darwinian", "llm", "anthropic"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "anthropic>=0.52.0",
    "pydantic>=2.0",
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.optional-dependencies]
web = ["fastapi>=0.110", "uvicorn>=0.29", "jinja2>=3.1"]
charts = ["matplotlib>=3.8"]
agent-sdk = ["claude-agent-sdk>=0.1"]
all = ["agentbreed[web,charts,agent-sdk]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4"]

[project.scripts]
breed = "breed.cli:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create breed/__init__.py**

```python
"""breed -- Evolve AI agents through Darwinian selection."""

__version__ = "0.1.0"
```

**Step 3: Create breed/py.typed (empty marker file)**

```
```

**Step 4: Create configs/forecasting.yaml**

```yaml
arena:
  type: forecasting
  source: metaculus  # or forecastbench
  num_questions: 50
  categories: [technology, science, politics, economics]

population:
  size: 20
  elite_count: 4

breeding:
  generations: 25
  crossover_rate: 0.7
  mutation_rate: 0.15
  immigration_rate: 0.1
  selection: tournament
  tournament_size: 3

fitness:
  weights:
    accuracy: 0.50
    calibration: 0.25
    cost_efficiency: 0.15
    diversity: 0.10

backend:
  type: messages  # or managed, agent_sdk
  model: claude-sonnet-4-6

logging:
  lineage_file: lineage.jsonl
  results_dir: results/
```

**Step 5: Create configs/coding.yaml**

```yaml
arena:
  type: coding
  problems_dir: problems/
  num_problems: 20
  difficulty: [easy, medium]

population:
  size: 15
  elite_count: 3

breeding:
  generations: 15
  crossover_rate: 0.7
  mutation_rate: 0.2
  immigration_rate: 0.1
  selection: tournament
  tournament_size: 3

fitness:
  weights:
    accuracy: 0.60
    cost_efficiency: 0.20
    diversity: 0.20

backend:
  type: messages
  model: claude-sonnet-4-6

logging:
  lineage_file: lineage.jsonl
  results_dir: results/
```

**Step 6: Create MIT LICENSE**

Standard MIT license text with current year and author.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml, configs, and license"
```

---

## Task 2: Genome Schema + Validation

**Files:**
- Create: `breed/genome.py`
- Test: `tests/test_genome.py`

**Step 1: Write failing tests**

```python
# tests/test_genome.py
import pytest
from breed.genome import Genome, Gene, GeneType, create_random_genome


class TestGene:
    def test_text_gene_creation(self):
        gene = Gene(
            name="prompt_template",
            type=GeneType.TEXT,
            value="You are a forecasting agent.",
            mutation_rate=0.15,
        )
        assert gene.name == "prompt_template"
        assert gene.value == "You are a forecasting agent."

    def test_enum_gene_creation(self):
        gene = Gene(
            name="decomposition_style",
            type=GeneType.ENUM,
            value="reference_class",
            options=["reference_class", "causal_chain", "scenario_tree", "fermi"],
            mutation_rate=0.10,
        )
        assert gene.value == "reference_class"

    def test_enum_gene_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            Gene(
                name="decomposition_style",
                type=GeneType.ENUM,
                value="invalid_style",
                options=["reference_class", "causal_chain"],
                mutation_rate=0.10,
            )

    def test_float_gene_clamps_to_range(self):
        gene = Gene(
            name="temperature",
            type=GeneType.FLOAT,
            value=0.7,
            range=(0.0, 1.5),
            mutation_rate=0.08,
        )
        assert gene.value == 0.7

    def test_float_gene_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            Gene(
                name="temperature",
                type=GeneType.FLOAT,
                value=2.0,
                range=(0.0, 1.5),
                mutation_rate=0.08,
            )

    def test_set_gene_creation(self):
        gene = Gene(
            name="tool_policy",
            type=GeneType.SET,
            value=["web_search", "calculator"],
            available=["web_search", "calculator", "code_exec", "api_call"],
            mutation_rate=0.10,
        )
        assert set(gene.value) == {"web_search", "calculator"}

    def test_numeric_vector_gene(self):
        gene = Gene(
            name="evidence_weighting",
            type=GeneType.NUMERIC_VECTOR,
            value={"recency": 0.3, "source_quality": 0.4, "consensus": 0.3},
            mutation_rate=0.10,
        )
        assert abs(sum(gene.value.values()) - 1.0) < 1e-6


class TestGenome:
    def test_genome_creation(self):
        genome = create_random_genome(seed=42)
        assert genome.genome_id is not None
        assert genome.generation == 0
        assert len(genome.genes) >= 9
        assert genome.parents == []

    def test_genome_serialization_roundtrip(self):
        genome = create_random_genome(seed=42)
        data = genome.to_dict()
        restored = Genome.from_dict(data)
        assert restored.genome_id == genome.genome_id
        assert restored.genes["prompt_template"].value == genome.genes["prompt_template"].value

    def test_genome_yaml_roundtrip(self):
        genome = create_random_genome(seed=42)
        yaml_str = genome.to_yaml()
        restored = Genome.from_yaml(yaml_str)
        assert restored.genome_id == genome.genome_id

    def test_genome_distance(self):
        g1 = create_random_genome(seed=1)
        g2 = create_random_genome(seed=2)
        dist = g1.distance(g2)
        assert dist > 0.0
        assert g1.distance(g1) == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
cd /c/Users/tejas/agentbreed && python -m pytest tests/test_genome.py -v
```
Expected: FAIL (module not found)

**Step 3: Implement breed/genome.py**

```python
"""Agent genome schema with 10 genes, validation, and serialization."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator


class GeneType(str, Enum):
    TEXT = "text"
    ENUM = "enum"
    FLOAT = "float"
    SET = "set"
    NUMERIC_VECTOR = "numeric_vector"


class Gene(BaseModel):
    name: str
    type: GeneType
    value: Any
    mutation_rate: float = 0.10
    # Type-specific constraints
    options: list[str] | None = None        # for ENUM
    range: tuple[float, float] | None = None  # for FLOAT
    available: list[str] | None = None       # for SET

    @model_validator(mode="after")
    def validate_value(self) -> "Gene":
        if self.type == GeneType.ENUM:
            if self.options and self.value not in self.options:
                raise ValueError(
                    f"Gene '{self.name}': value '{self.value}' not in options {self.options}"
                )
        elif self.type == GeneType.FLOAT:
            if self.range:
                lo, hi = self.range
                if not (lo <= self.value <= hi):
                    raise ValueError(
                        f"Gene '{self.name}': value {self.value} not in range [{lo}, {hi}]"
                    )
        elif self.type == GeneType.SET:
            if self.available:
                invalid = set(self.value) - set(self.available)
                if invalid:
                    raise ValueError(
                        f"Gene '{self.name}': values {invalid} not in available {self.available}"
                    )
        elif self.type == GeneType.NUMERIC_VECTOR:
            if not isinstance(self.value, dict):
                raise ValueError(f"Gene '{self.name}': numeric_vector value must be a dict")
        return self


class Genome(BaseModel):
    genome_id: str
    generation: int = 0
    parents: list[str] = []
    created_at: str = ""
    genes: dict[str, Gene] = {}
    fitness_history: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def set_defaults(self) -> "Genome":
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        genes = {}
        for name, gene_data in data.get("genes", {}).items():
            if isinstance(gene_data, dict) and "name" in gene_data:
                genes[name] = Gene(**gene_data)
            elif isinstance(gene_data, dict):
                genes[name] = Gene(name=name, **gene_data)
        data_copy = {**data, "genes": genes}
        return cls(**data_copy)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Genome":
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def distance(self, other: "Genome") -> float:
        """Compute normalized genome distance (0.0 = identical, 1.0 = maximally different)."""
        if self.genome_id == other.genome_id:
            return 0.0
        diffs = 0
        total = 0
        for name, gene in self.genes.items():
            if name not in other.genes:
                diffs += 1
                total += 1
                continue
            other_gene = other.genes[name]
            total += 1
            if gene.type == GeneType.TEXT:
                # Simple word-level Jaccard distance
                words_a = set(str(gene.value).lower().split())
                words_b = set(str(other_gene.value).lower().split())
                if words_a or words_b:
                    jaccard = 1.0 - len(words_a & words_b) / max(len(words_a | words_b), 1)
                    diffs += jaccard
                # else both empty, diff = 0
            elif gene.type == GeneType.ENUM:
                if gene.value != other_gene.value:
                    diffs += 1
            elif gene.type == GeneType.FLOAT:
                if gene.range:
                    lo, hi = gene.range
                    span = hi - lo if hi > lo else 1.0
                    diffs += abs(gene.value - other_gene.value) / span
                else:
                    diffs += 0 if gene.value == other_gene.value else 1
            elif gene.type == GeneType.SET:
                set_a = set(gene.value)
                set_b = set(other_gene.value)
                if set_a or set_b:
                    diffs += 1.0 - len(set_a & set_b) / max(len(set_a | set_b), 1)
            elif gene.type == GeneType.NUMERIC_VECTOR:
                keys = set(gene.value.keys()) | set(other_gene.value.keys())
                if keys:
                    diff_sum = sum(
                        abs(gene.value.get(k, 0) - other_gene.value.get(k, 0))
                        for k in keys
                    )
                    diffs += min(diff_sum / 2.0, 1.0)  # normalize
        return diffs / max(total, 1)


# Default gene templates for agent genomes
DEFAULT_GENE_TEMPLATES: dict[str, dict[str, Any]] = {
    "prompt_template": {
        "type": "text",
        "mutation_rate": 0.15,
        "templates": [
            "You are a calibrated forecasting agent. Consider base rates first, then update with specific evidence. Express uncertainty honestly.",
            "You are an analytical prediction agent. Decompose complex questions into measurable components. Reason step by step before giving a probability.",
            "You are a superforecaster-style agent. Use reference class forecasting, consider multiple scenarios, and avoid anchoring on initial estimates.",
            "You are a Bayesian reasoning agent. Start with prior probabilities from historical data, then systematically update based on new evidence.",
            "You are a contrarian forecasting agent. Actively seek evidence that contradicts the consensus view. Weight surprising information highly.",
        ],
    },
    "memory_structure": {
        "type": "enum",
        "options": ["sliding_window", "episodic", "semantic", "graph", "none"],
        "mutation_rate": 0.10,
    },
    "tool_policy": {
        "type": "set",
        "available": ["web_search", "web_fetch", "calculator", "code_exec", "read", "write"],
        "mutation_rate": 0.10,
    },
    "decision_heuristic": {
        "type": "text",
        "mutation_rate": 0.12,
        "templates": [
            "Consider base rates first, then update with evidence. Weight recent evidence more heavily. Flag when confidence exceeds 0.9.",
            "Use reference class forecasting: find 3-5 historical analogies, compute their average outcome, then adjust for unique factors.",
            "Decompose the question into independent sub-questions. Estimate each sub-probability. Combine using appropriate rules (AND/OR).",
            "Start with the outside view (base rate), then apply the inside view (specific details). Explicitly state your adjustment rationale.",
            "Consider three scenarios: optimistic, pessimistic, and most likely. Weight by plausibility. Your final answer is the weighted average.",
        ],
    },
    "calibration_rule": {
        "type": "text",
        "mutation_rate": 0.08,
        "templates": [
            "When confidence > 0.9, apply 10% shrinkage toward base rate. When confidence < 0.1, apply 10% push toward 0.15.",
            "Never give probabilities below 0.02 or above 0.98. Apply extremeness aversion: shrink all estimates 5% toward 0.5.",
            "Use the calibration heuristic: for events you estimate at X%, historically you are right X% of the time only if X < 0.7. Above 0.7, reduce by 10%.",
            "Apply no calibration correction. Trust your reasoning process. Only flag extreme probabilities (>0.95 or <0.05) for review.",
        ],
    },
    "evidence_weighting": {
        "type": "numeric_vector",
        "mutation_rate": 0.10,
        "templates": [
            {"recency": 0.3, "source_quality": 0.4, "consensus": 0.3},
            {"recency": 0.5, "source_quality": 0.3, "consensus": 0.2},
            {"recency": 0.2, "source_quality": 0.5, "consensus": 0.3},
            {"recency": 0.4, "source_quality": 0.2, "consensus": 0.4},
        ],
    },
    "risk_threshold": {
        "type": "float",
        "range": [0.01, 0.50],
        "mutation_rate": 0.10,
    },
    "decomposition_style": {
        "type": "enum",
        "options": ["reference_class", "causal_chain", "scenario_tree", "fermi", "analogy"],
        "mutation_rate": 0.10,
    },
    "temperature": {
        "type": "float",
        "range": [0.0, 1.5],
        "mutation_rate": 0.08,
    },
    "answer_format": {
        "type": "enum",
        "options": ["probability_only", "probability_with_reasoning", "structured_json"],
        "mutation_rate": 0.05,
    },
}


def create_random_genome(seed: int | None = None) -> Genome:
    """Create a random genome from default gene templates."""
    rng = random.Random(seed)
    genes: dict[str, Gene] = {}

    for name, template in DEFAULT_GENE_TEMPLATES.items():
        gene_type = GeneType(template["type"])
        mutation_rate = template.get("mutation_rate", 0.10)

        if gene_type == GeneType.TEXT:
            value = rng.choice(template["templates"])
            genes[name] = Gene(name=name, type=gene_type, value=value, mutation_rate=mutation_rate)

        elif gene_type == GeneType.ENUM:
            value = rng.choice(template["options"])
            genes[name] = Gene(
                name=name, type=gene_type, value=value,
                options=template["options"], mutation_rate=mutation_rate,
            )

        elif gene_type == GeneType.FLOAT:
            lo, hi = template["range"]
            value = round(rng.uniform(lo, hi), 3)
            genes[name] = Gene(
                name=name, type=gene_type, value=value,
                range=tuple(template["range"]), mutation_rate=mutation_rate,
            )

        elif gene_type == GeneType.SET:
            available = template["available"]
            count = rng.randint(1, len(available))
            value = sorted(rng.sample(available, count))
            genes[name] = Gene(
                name=name, type=gene_type, value=value,
                available=available, mutation_rate=mutation_rate,
            )

        elif gene_type == GeneType.NUMERIC_VECTOR:
            template_value = rng.choice(template["templates"])
            genes[name] = Gene(
                name=name, type=gene_type, value=dict(template_value),
                mutation_rate=mutation_rate,
            )

    return Genome(
        genome_id=str(uuid.uuid4()),
        generation=0,
        parents=[],
        genes=genes,
    )
```

**Step 4: Run tests**

```bash
cd /c/Users/tejas/agentbreed && python -m pytest tests/test_genome.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add breed/genome.py tests/test_genome.py
git commit -m "feat: genome schema with 10 genes, validation, serialization, and distance metric"
```

---

## Task 3: Crossover + Mutation Operators

**Files:**
- Create: `breed/operators/crossover.py`
- Create: `breed/operators/mutation.py`
- Create: `breed/operators/__init__.py`
- Test: `tests/test_operators.py`

**Step 1: Write failing tests**

```python
# tests/test_operators.py
import pytest
from breed.genome import Genome, create_random_genome
from breed.operators.crossover import uniform_crossover, component_swap, weighted_crossover
from breed.operators.mutation import (
    parameter_jitter,
    tool_swap,
    strategy_mutation,
    prompt_perturb_simple,
)


class TestCrossover:
    def test_uniform_crossover_produces_child(self):
        p1 = create_random_genome(seed=1)
        p2 = create_random_genome(seed=2)
        child = uniform_crossover(p1, p2, seed=42)
        assert child.genome_id != p1.genome_id
        assert child.genome_id != p2.genome_id
        assert child.generation == max(p1.generation, p2.generation) + 1
        assert set(child.parents) == {p1.genome_id, p2.genome_id}
        assert len(child.genes) == len(p1.genes)

    def test_uniform_crossover_genes_from_parents(self):
        p1 = create_random_genome(seed=1)
        p2 = create_random_genome(seed=2)
        child = uniform_crossover(p1, p2, seed=42)
        for name, gene in child.genes.items():
            assert (
                gene.value == p1.genes[name].value or
                gene.value == p2.genes[name].value
            )

    def test_component_swap_crossover(self):
        p1 = create_random_genome(seed=1)
        p2 = create_random_genome(seed=2)
        child = component_swap(p1, p2, seed=42)
        assert child.generation == max(p1.generation, p2.generation) + 1
        assert len(child.genes) == len(p1.genes)

    def test_weighted_crossover_biases_fitter(self):
        p1 = create_random_genome(seed=1)
        p2 = create_random_genome(seed=2)
        # Run many times, fitter parent's genes should appear more often
        p1_count = 0
        total = 0
        for i in range(100):
            child = weighted_crossover(p1, p2, fitness_a=0.9, fitness_b=0.1, seed=i)
            for name in child.genes:
                total += 1
                if child.genes[name].value == p1.genes[name].value:
                    p1_count += 1
        # p1 is much fitter, should get >60% of genes
        assert p1_count / total > 0.6


class TestMutation:
    def test_parameter_jitter_changes_float(self):
        genome = create_random_genome(seed=42)
        original_temp = genome.genes["temperature"].value
        mutated = parameter_jitter(genome, gene_name="temperature", strength=0.5, seed=1)
        assert mutated.genes["temperature"].value != original_temp
        # Should still be within range
        lo, hi = mutated.genes["temperature"].range
        assert lo <= mutated.genes["temperature"].value <= hi

    def test_tool_swap_changes_tools(self):
        genome = create_random_genome(seed=42)
        original_tools = set(genome.genes["tool_policy"].value)
        mutated = tool_swap(genome, seed=1)
        new_tools = set(mutated.genes["tool_policy"].value)
        # At least one tool should differ
        assert original_tools != new_tools

    def test_strategy_mutation_changes_enum(self):
        genome = create_random_genome(seed=42)
        original = genome.genes["decomposition_style"].value
        # Try multiple seeds until we get a different value
        changed = False
        for s in range(20):
            mutated = strategy_mutation(genome, gene_name="decomposition_style", seed=s)
            if mutated.genes["decomposition_style"].value != original:
                changed = True
                break
        assert changed

    def test_mutation_preserves_other_genes(self):
        genome = create_random_genome(seed=42)
        mutated = parameter_jitter(genome, gene_name="temperature", strength=0.5, seed=1)
        # All non-temperature genes should be identical
        for name, gene in genome.genes.items():
            if name != "temperature":
                assert mutated.genes[name].value == gene.value

    def test_prompt_perturb_simple_changes_prompt(self):
        genome = create_random_genome(seed=42)
        original = genome.genes["prompt_template"].value
        mutated = prompt_perturb_simple(genome, seed=1)
        # Simple perturbation should produce a different prompt
        assert mutated.genes["prompt_template"].value != original
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_operators.py -v
```

**Step 3: Implement breed/operators/__init__.py**

```python
"""Genetic operators for agent breeding."""
```

**Step 4: Implement breed/operators/crossover.py**

```python
"""Crossover operators for combining parent genomes."""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from datetime import datetime, timezone

from breed.genome import Gene, Genome, GeneType


def _make_child(parent_a: Genome, parent_b: Genome, genes: dict[str, Gene]) -> Genome:
    return Genome(
        genome_id=str(uuid.uuid4()),
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parents=[parent_a.genome_id, parent_b.genome_id],
        genes=genes,
    )


def uniform_crossover(parent_a: Genome, parent_b: Genome, seed: int | None = None) -> Genome:
    """For each gene, randomly pick from parent A or parent B (50/50)."""
    rng = random.Random(seed)
    genes: dict[str, Gene] = {}
    for name in parent_a.genes:
        source = parent_a if rng.random() < 0.5 else parent_b
        genes[name] = deepcopy(source.genes[name])
    return _make_child(parent_a, parent_b, genes)


def component_swap(parent_a: Genome, parent_b: Genome, seed: int | None = None) -> Genome:
    """Swap gene components: text genes from one parent, numeric from the other."""
    rng = random.Random(seed)
    text_source = parent_a if rng.random() < 0.5 else parent_b
    other_source = parent_b if text_source is parent_a else parent_a
    genes: dict[str, Gene] = {}
    for name, gene in parent_a.genes.items():
        if gene.type == GeneType.TEXT:
            genes[name] = deepcopy(text_source.genes[name])
        else:
            genes[name] = deepcopy(other_source.genes[name])
    return _make_child(parent_a, parent_b, genes)


def weighted_crossover(
    parent_a: Genome, parent_b: Genome,
    fitness_a: float, fitness_b: float,
    seed: int | None = None,
) -> Genome:
    """Bias gene selection toward fitter parent."""
    rng = random.Random(seed)
    total_fitness = fitness_a + fitness_b
    prob_a = fitness_a / total_fitness if total_fitness > 0 else 0.5
    genes: dict[str, Gene] = {}
    for name in parent_a.genes:
        source = parent_a if rng.random() < prob_a else parent_b
        gene = deepcopy(source.genes[name])
        # For numeric genes, optionally blend
        if gene.type == GeneType.FLOAT:
            val_a = parent_a.genes[name].value
            val_b = parent_b.genes[name].value
            gene.value = round(val_a * prob_a + val_b * (1 - prob_a), 4)
            if gene.range:
                lo, hi = gene.range
                gene.value = max(lo, min(hi, gene.value))
        genes[name] = gene
    return _make_child(parent_a, parent_b, genes)
```

**Step 5: Implement breed/operators/mutation.py**

```python
"""Mutation operators for modifying agent genomes."""

from __future__ import annotations

import random
from copy import deepcopy

from breed.genome import Gene, GeneType, Genome, DEFAULT_GENE_TEMPLATES


def _clone_genome(genome: Genome) -> Genome:
    """Create a deep copy of a genome with the same ID (caller replaces genes)."""
    return genome.model_copy(deep=True)


def parameter_jitter(
    genome: Genome, gene_name: str = "temperature",
    strength: float = 0.3, seed: int | None = None,
) -> Genome:
    """Add Gaussian noise to a FLOAT gene, clamped to valid range."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes[gene_name]
    if gene.type != GeneType.FLOAT:
        return mutated
    noise = rng.gauss(0, strength)
    new_val = gene.value + noise
    if gene.range:
        lo, hi = gene.range
        new_val = max(lo, min(hi, new_val))
    gene.value = round(new_val, 4)
    return mutated


def tool_swap(genome: Genome, seed: int | None = None) -> Genome:
    """Add, remove, or swap one tool in the tool_policy gene."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes.get("tool_policy")
    if gene is None or gene.type != GeneType.SET or not gene.available:
        return mutated
    current = list(gene.value)
    available = gene.available
    not_selected = [t for t in available if t not in current]
    action = rng.choice(["add", "remove", "swap"])
    if action == "add" and not_selected:
        current.append(rng.choice(not_selected))
    elif action == "remove" and len(current) > 1:
        current.remove(rng.choice(current))
    elif action == "swap" and current and not_selected:
        current.remove(rng.choice(current))
        current.append(rng.choice(not_selected))
    gene.value = sorted(current)
    return mutated


def strategy_mutation(
    genome: Genome, gene_name: str = "decomposition_style", seed: int | None = None,
) -> Genome:
    """Swap an ENUM gene to a random alternative."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes.get(gene_name)
    if gene is None or gene.type != GeneType.ENUM or not gene.options:
        return mutated
    alternatives = [o for o in gene.options if o != gene.value]
    if alternatives:
        gene.value = rng.choice(alternatives)
    return mutated


def prompt_perturb_simple(genome: Genome, seed: int | None = None) -> Genome:
    """Replace a prompt gene with a different template (non-LLM mutation)."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes.get("prompt_template")
    if gene is None or gene.type != GeneType.TEXT:
        return mutated
    templates = DEFAULT_GENE_TEMPLATES.get("prompt_template", {}).get("templates", [])
    alternatives = [t for t in templates if t != gene.value]
    if alternatives:
        gene.value = rng.choice(alternatives)
    return mutated


def vector_jitter(
    genome: Genome, gene_name: str = "evidence_weighting",
    strength: float = 0.1, seed: int | None = None,
) -> Genome:
    """Add noise to a NUMERIC_VECTOR gene, re-normalize to sum to 1."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes.get(gene_name)
    if gene is None or gene.type != GeneType.NUMERIC_VECTOR:
        return mutated
    new_values = {}
    for key, val in gene.value.items():
        new_val = max(0.01, val + rng.gauss(0, strength))
        new_values[key] = new_val
    total = sum(new_values.values())
    gene.value = {k: round(v / total, 4) for k, v in new_values.items()}
    return mutated


def calibration_adjustment(genome: Genome, seed: int | None = None) -> Genome:
    """Replace calibration_rule with a different template."""
    rng = random.Random(seed)
    mutated = _clone_genome(genome)
    gene = mutated.genes.get("calibration_rule")
    if gene is None or gene.type != GeneType.TEXT:
        return mutated
    templates = DEFAULT_GENE_TEMPLATES.get("calibration_rule", {}).get("templates", [])
    alternatives = [t for t in templates if t != gene.value]
    if alternatives:
        gene.value = rng.choice(alternatives)
    return mutated
```

**Step 6: Run tests**

```bash
python -m pytest tests/test_operators.py -v
```

**Step 7: Commit**

```bash
git add breed/operators/ tests/test_operators.py
git commit -m "feat: crossover (uniform, component_swap, weighted) and mutation (jitter, tool_swap, strategy, prompt_perturb, vector_jitter, calibration) operators"
```

---

## Task 4: Selection + Diversity Preservation

**Files:**
- Create: `breed/operators/selection.py`
- Create: `breed/diversity.py`
- Test: `tests/test_selection.py`

**Step 1: Write failing tests**

```python
# tests/test_selection.py
import pytest
from breed.genome import create_random_genome
from breed.operators.selection import tournament_select, truncation_select
from breed.diversity import novelty_score, inject_immigrants


class TestSelection:
    def test_tournament_select_returns_correct_count(self):
        pop = [create_random_genome(seed=i) for i in range(10)]
        scores = {g.genome_id: float(i) / 10 for i, g in enumerate(pop)}
        selected = tournament_select(pop, scores, k=5, tournament_size=3, seed=42)
        assert len(selected) == 5

    def test_tournament_select_favors_fitter(self):
        pop = [create_random_genome(seed=i) for i in range(20)]
        # Give last agents highest fitness
        scores = {g.genome_id: float(i) for i, g in enumerate(pop)}
        selected = tournament_select(pop, scores, k=5, tournament_size=3, seed=42)
        selected_indices = [pop.index(s) for s in selected]
        # Selected agents should have above-median indices on average
        assert sum(selected_indices) / len(selected_indices) > 10

    def test_truncation_select(self):
        pop = [create_random_genome(seed=i) for i in range(10)]
        scores = {g.genome_id: float(i) for i, g in enumerate(pop)}
        selected = truncation_select(pop, scores, k=3)
        # Should be the top 3 (indices 7, 8, 9)
        for s in selected:
            assert scores[s.genome_id] >= 7.0


class TestDiversity:
    def test_novelty_score_self_is_zero(self):
        pop = [create_random_genome(seed=i) for i in range(5)]
        # Novelty relative to population center
        scores = novelty_score(pop)
        # All scores should be non-negative
        assert all(s >= 0 for s in scores.values())

    def test_inject_immigrants_adds_agents(self):
        pop = [create_random_genome(seed=i) for i in range(10)]
        expanded = inject_immigrants(pop, count=3, seed=99)
        assert len(expanded) == 13
        # New agents should have unique IDs
        ids = [g.genome_id for g in expanded]
        assert len(set(ids)) == 13
```

**Step 2: Run tests**
```bash
python -m pytest tests/test_selection.py -v
```

**Step 3: Implement breed/operators/selection.py**

```python
"""Selection operators for choosing parents from population."""

from __future__ import annotations

import random

from breed.genome import Genome


def tournament_select(
    population: list[Genome],
    fitness_scores: dict[str, float],
    k: int,
    tournament_size: int = 3,
    seed: int | None = None,
) -> list[Genome]:
    """Select k individuals via tournament selection."""
    rng = random.Random(seed)
    selected: list[Genome] = []
    for _ in range(k):
        tournament = rng.sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda g: fitness_scores.get(g.genome_id, 0.0))
        selected.append(winner)
    return selected


def truncation_select(
    population: list[Genome],
    fitness_scores: dict[str, float],
    k: int,
) -> list[Genome]:
    """Select the top-k individuals by fitness."""
    sorted_pop = sorted(
        population,
        key=lambda g: fitness_scores.get(g.genome_id, 0.0),
        reverse=True,
    )
    return sorted_pop[:k]
```

**Step 4: Implement breed/diversity.py**

```python
"""Diversity preservation mechanisms."""

from __future__ import annotations

from breed.genome import Genome, create_random_genome


def novelty_score(population: list[Genome]) -> dict[str, float]:
    """Compute novelty score for each genome (average distance to all others)."""
    scores: dict[str, float] = {}
    n = len(population)
    if n <= 1:
        return {g.genome_id: 0.0 for g in population}
    for genome in population:
        total_dist = sum(genome.distance(other) for other in population if other is not genome)
        scores[genome.genome_id] = total_dist / (n - 1)
    return scores


def inject_immigrants(
    population: list[Genome], count: int, seed: int | None = None,
) -> list[Genome]:
    """Add random immigrant genomes to maintain diversity."""
    import random as _random
    rng = _random.Random(seed)
    immigrants = [create_random_genome(seed=rng.randint(0, 999999)) for _ in range(count)]
    return population + immigrants
```

**Step 5: Run tests**
```bash
python -m pytest tests/test_selection.py -v
```

**Step 6: Commit**
```bash
git add breed/operators/selection.py breed/diversity.py tests/test_selection.py
git commit -m "feat: tournament/truncation selection and diversity preservation (novelty score, immigration)"
```

---

## Task 5: Population Management

**Files:**
- Create: `breed/population.py`
- Test: `tests/test_population.py`

**Step 1: Write failing tests**

```python
# tests/test_population.py
import pytest
from breed.population import Population


class TestPopulation:
    def test_spawn_creates_correct_size(self):
        pop = Population.spawn(size=10, seed=42)
        assert len(pop.agents) == 10

    def test_spawn_all_unique(self):
        pop = Population.spawn(size=10, seed=42)
        ids = [g.genome_id for g in pop.agents]
        assert len(set(ids)) == 10

    def test_select_reduces_size(self):
        pop = Population.spawn(size=10, seed=42)
        scores = {g.genome_id: float(i) for i, g in enumerate(pop.agents)}
        selected = pop.select(scores, top_k=5)
        assert len(selected.agents) == 5

    def test_breed_produces_offspring(self):
        pop = Population.spawn(size=10, seed=42)
        scores = {g.genome_id: float(i) for i, g in enumerate(pop.agents)}
        selected = pop.select(scores, top_k=5)
        bred = selected.breed(target_size=10, seed=42)
        assert len(bred.agents) == 10
        # Bred agents should have parents
        new_agents = [g for g in bred.agents if g.parents]
        assert len(new_agents) > 0

    def test_mutate_changes_some_agents(self):
        pop = Population.spawn(size=10, seed=42)
        mutated = pop.mutate(rate=1.0, seed=42)  # 100% mutation rate
        # At least some agents should differ
        changed = sum(
            1 for orig, mut in zip(pop.agents, mutated.agents)
            if orig.genes["temperature"].value != mut.genes["temperature"].value
        )
        assert changed > 0

    def test_generation_increments(self):
        pop = Population.spawn(size=10, seed=42)
        assert pop.generation == 0
        scores = {g.genome_id: float(i) for i, g in enumerate(pop.agents)}
        selected = pop.select(scores, top_k=5)
        bred = selected.breed(target_size=10, seed=42)
        assert bred.generation == 1
```

**Step 2: Run tests, verify fail**

**Step 3: Implement breed/population.py**

```python
"""Population management: spawn, select, breed, mutate."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field

from breed.genome import Genome, create_random_genome
from breed.operators.crossover import uniform_crossover, component_swap, weighted_crossover
from breed.operators.mutation import (
    parameter_jitter, tool_swap, strategy_mutation,
    prompt_perturb_simple, vector_jitter, calibration_adjustment,
)
from breed.operators.selection import tournament_select, truncation_select
from breed.diversity import inject_immigrants


@dataclass
class Population:
    agents: list[Genome]
    generation: int = 0

    @classmethod
    def spawn(cls, size: int, seed: int | None = None) -> "Population":
        rng = random.Random(seed)
        agents = [create_random_genome(seed=rng.randint(0, 999999)) for _ in range(size)]
        return cls(agents=agents, generation=0)

    def select(
        self, fitness_scores: dict[str, float], top_k: int,
        method: str = "tournament", tournament_size: int = 3,
        seed: int | None = None,
    ) -> "Population":
        if method == "tournament":
            selected = tournament_select(
                self.agents, fitness_scores, k=top_k,
                tournament_size=tournament_size, seed=seed,
            )
        else:
            selected = truncation_select(self.agents, fitness_scores, k=top_k)
        return Population(agents=selected, generation=self.generation)

    def breed(
        self, target_size: int, seed: int | None = None,
        fitness_scores: dict[str, float] | None = None,
    ) -> "Population":
        rng = random.Random(seed)
        offspring: list[Genome] = list(deepcopy(self.agents))  # elites survive

        crossover_ops = [uniform_crossover, component_swap]

        while len(offspring) < target_size:
            p1, p2 = rng.sample(self.agents, 2)
            op = rng.choice(crossover_ops)
            if op == weighted_crossover and fitness_scores:
                child = weighted_crossover(
                    p1, p2,
                    fitness_a=fitness_scores.get(p1.genome_id, 0.5),
                    fitness_b=fitness_scores.get(p2.genome_id, 0.5),
                    seed=rng.randint(0, 999999),
                )
            else:
                child = op(p1, p2, seed=rng.randint(0, 999999))
            offspring.append(child)

        return Population(agents=offspring[:target_size], generation=self.generation + 1)

    def mutate(self, rate: float = 0.15, seed: int | None = None) -> "Population":
        rng = random.Random(seed)
        mutation_ops = [
            lambda g, s: parameter_jitter(g, "temperature", 0.3, s),
            lambda g, s: parameter_jitter(g, "risk_threshold", 0.1, s),
            lambda g, s: tool_swap(g, s),
            lambda g, s: strategy_mutation(g, "decomposition_style", s),
            lambda g, s: strategy_mutation(g, "answer_format", s),
            lambda g, s: prompt_perturb_simple(g, s),
            lambda g, s: vector_jitter(g, "evidence_weighting", 0.1, s),
            lambda g, s: calibration_adjustment(g, s),
        ]

        mutated_agents: list[Genome] = []
        for genome in self.agents:
            if rng.random() < rate:
                op = rng.choice(mutation_ops)
                mutated_agents.append(op(genome, rng.randint(0, 999999)))
            else:
                mutated_agents.append(deepcopy(genome))

        return Population(agents=mutated_agents, generation=self.generation)

    def add_immigrants(self, count: int, seed: int | None = None) -> "Population":
        expanded = inject_immigrants(self.agents, count, seed)
        return Population(agents=expanded, generation=self.generation)

    @property
    def size(self) -> int:
        return len(self.agents)
```

**Step 4: Run tests**

**Step 5: Commit**
```bash
git add breed/population.py tests/test_population.py
git commit -m "feat: population management with spawn, select, breed, mutate, immigrate"
```

---

## Task 6: Lineage Tracker

**Files:**
- Create: `breed/lineage.py`
- Test: `tests/test_lineage.py`

(Implementation: immutable JSONL append log, parent tracking, lineage tree reconstruction, champion path extraction)

---

## Task 7: Fitness Evaluator + Composite Scoring

**Files:**
- Create: `breed/fitness.py`
- Test: `tests/test_fitness.py`

(Implementation: abstract FitnessEvaluator, BrierScoreEvaluator, PassRateEvaluator, CompositeFitness with configurable weights)

---

## Task 8: Messages API Backend

**Files:**
- Create: `breed/backends/base.py`
- Create: `breed/backends/messages.py`
- Test: `tests/test_backends.py`

(Implementation: AbstractBackend, MessagesBackend using `anthropic.Anthropic().messages.create()`, genome-to-system-prompt conversion, tool config generation)

---

## Task 9: Managed Agents Backend

**Files:**
- Create: `breed/backends/managed.py`

(Implementation: ManagedAgentsBackend using `client.beta.agents.create()`, `client.beta.sessions.create()`, `client.beta.sessions.events.stream()`, genome-to-agent config mapping, SSE event processing)

---

## Task 10: Agent SDK Backend

**Files:**
- Create: `breed/backends/agent_sdk.py`

(Implementation: AgentSDKBackend using `claude_agent_sdk.query()`, genome-to-ClaudeAgentOptions mapping)

---

## Task 11: Forecasting Arena

**Files:**
- Create: `breed/arenas/base.py`
- Create: `breed/arenas/forecasting.py`

(Implementation: abstract Arena, ForecastingArena with Metaculus API integration, question sampling, Brier score computation, masked-price evaluation)

---

## Task 12: Coding Arena

**Files:**
- Create: `breed/arenas/coding.py`

(Historical plan: CodingArena with HumanEval-style problems and pass-rate scoring. The shipped in-process executor was not a sandbox and is now disabled. Do not restore execution until a disposable OS-level sandbox provides process, filesystem, environment, network, and resource isolation.)

---

## Task 13: Breeding Engine (Main Loop)

**Files:**
- Create: `breed/engine.py`
- Test: `tests/test_engine.py`

(Implementation: BreedingEngine orchestrating the full loop: spawn -> evaluate -> score -> select -> breed -> mutate -> immigrate -> log -> repeat. Rich terminal output with real-time progress bars.)

---

## Task 14: CLI (Click)

**Files:**
- Create: `breed/cli.py`

(Implementation: `breed init`, `breed run`, `breed results`, `breed champion`, `breed lineage`, `breed diff`, `breed export`, `breed benchmark`, `breed serve`)

---

## Task 15: Chart Generation

**Files:**
- Create: `breed/charts.py`

(Implementation: fitness_curve, calibration_plot, population_distribution, gene_importance_heatmap, lineage_tree_static)

---

## Task 16: Web UI (FastAPI + D3.js)

**Files:**
- Create: `breed/web/app.py`
- Create: `breed/web/static/lineage.js`
- Create: `breed/web/static/index.html`
- Create: `breed/web/static/style.css`
- Create: `breed/web/templates/index.html`

(Implementation: FastAPI serving lineage explorer with interactive D3.js phylogenetic tree, genome detail panel, fitness heatmap, real-time breeding dashboard)

---

## Task 17: Experiment Runner + Ablation Support

**Files:**
- Create: `breed/experiment.py`

(Implementation: ExperimentRunner loading YAML configs, running ablations (mutation-only, crossover-only, random-search, prompt-only), collecting statistics, generating comparison charts)

---

## Task 18: README + Docs + Hero Visuals

**Files:**
- Create: `README.md`
- Create: `docs/quickstart.md`
- Create: `docs/genome-schema.md`
- Create: `docs/managed-agents-integration.md`
- Create: `docs/breedBench-protocol.md`

(Implementation: star-maximizing README with 30-second quickstart, results table, architecture diagram, hero visual placeholder, badges)

---

## Task 19: CI/CD + Packaging

**Files:**
- Create: `.github/workflows/ci.yml`

(Implementation: pytest, ruff lint, type check, build wheel)

---

## Task 20: Integration Test + End-to-End Verification

Run full breeding loop with Messages API backend, 5 generations, 10 agents, forecasting arena with synthetic questions. Verify lineage file, charts, and CLI commands all work.

```bash
breed init --arena forecasting --backend messages
breed run --generations 5 --population 10
breed results
breed champion
breed lineage --format tree
breed diff --gen 0 --gen 5
breed serve
```
