"""Tests for custom genome template support.

Covers create_genome_from_template, generic mutation operators,
Population.spawn with custom templates, and full breeding loops.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import Arena, EvalResult, Task
from breed.engine import BreedingConfig, BreedingEngine
from breed.genome import (
    Gene,
    GeneType,
    Genome,
    create_genome_from_template,
)
from breed.operators.crossover import uniform_crossover
from breed.operators.mutation import (
    mutate_gene,
    set_swap,
    text_swap,
)
from breed.population import Population


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_TEMPLATE: dict = {
    "my_text": {
        "type": GeneType.TEXT,
        "templates": ["Hello world", "Goodbye world", "Neutral statement"],
        "mutation_rate": 0.2,
    },
    "my_float": {
        "type": GeneType.FLOAT,
        "range": (0.0, 10.0),
        "mutation_rate": 0.1,
    },
    "my_enum": {
        "type": GeneType.ENUM,
        "options": ["alpha", "beta", "gamma"],
        "mutation_rate": 0.15,
    },
}

FULL_TEMPLATE: dict = {
    "agent_prompt": {
        "type": GeneType.TEXT,
        "templates": [
            "You are a helpful assistant.",
            "You are a research analyst.",
            "You are a creative writer.",
        ],
        "mutation_rate": 0.2,
    },
    "strategy": {
        "type": GeneType.ENUM,
        "options": ["sequential", "parallel", "hierarchical"],
        "mutation_rate": 0.15,
    },
    "tools_enabled": {
        "type": GeneType.SET,
        "available": ["web_search", "code_exec", "file_read", "api_call"],
        "mutation_rate": 0.1,
    },
    "temperature": {
        "type": GeneType.FLOAT,
        "range": (0.0, 1.5),
        "mutation_rate": 0.1,
    },
    "quality_weights": {
        "type": GeneType.NUMERIC_VECTOR,
        "templates": [
            [0.4, 0.3, 0.3],
            [0.7, 0.2, 0.1],
        ],
        "mutation_rate": 0.1,
    },
}


# ---------------------------------------------------------------------------
# create_genome_from_template
# ---------------------------------------------------------------------------


class TestCreateGenomeFromTemplate:
    """Tests for create_genome_from_template."""

    def test_basic_three_gene_template(self) -> None:
        genome = create_genome_from_template(SIMPLE_TEMPLATE, seed=42)

        assert isinstance(genome, Genome)
        assert len(genome.genes) == 3
        assert set(genome.genes.keys()) == {"my_text", "my_float", "my_enum"}

        # TEXT gene
        text_gene = genome.genes["my_text"]
        assert text_gene.type == GeneType.TEXT
        assert text_gene.value in SIMPLE_TEMPLATE["my_text"]["templates"]
        assert text_gene.mutation_rate == 0.2

        # FLOAT gene
        float_gene = genome.genes["my_float"]
        assert float_gene.type == GeneType.FLOAT
        assert 0.0 <= float_gene.value <= 10.0
        assert float_gene.range == (0.0, 10.0)

        # ENUM gene
        enum_gene = genome.genes["my_enum"]
        assert enum_gene.type == GeneType.ENUM
        assert enum_gene.value in ["alpha", "beta", "gamma"]
        assert enum_gene.options == ["alpha", "beta", "gamma"]

    def test_string_type_names(self) -> None:
        """type can be a plain string like 'text', 'float', 'enum'."""
        template = {
            "my_text": {
                "type": "text",
                "templates": ["Hello", "World"],
                "mutation_rate": 0.1,
            },
            "my_float": {
                "type": "float",
                "range": (0.0, 1.0),
                "mutation_rate": 0.1,
            },
            "my_enum": {
                "type": "enum",
                "options": ["a", "b", "c"],
                "mutation_rate": 0.1,
            },
        }
        genome = create_genome_from_template(template, seed=99)

        assert genome.genes["my_text"].type == GeneType.TEXT
        assert genome.genes["my_float"].type == GeneType.FLOAT
        assert genome.genes["my_enum"].type == GeneType.ENUM

    def test_all_five_gene_types(self) -> None:
        """Template with all five gene types works correctly."""
        genome = create_genome_from_template(FULL_TEMPLATE, seed=7)

        assert len(genome.genes) == 5
        assert genome.genes["agent_prompt"].type == GeneType.TEXT
        assert genome.genes["strategy"].type == GeneType.ENUM
        assert genome.genes["tools_enabled"].type == GeneType.SET
        assert genome.genes["temperature"].type == GeneType.FLOAT
        assert genome.genes["quality_weights"].type == GeneType.NUMERIC_VECTOR

        # SET gene should have items from the available list
        tools = genome.genes["tools_enabled"]
        assert all(t in tools.available for t in tools.value)

    def test_reproducibility_with_seed(self) -> None:
        g1 = create_genome_from_template(FULL_TEMPLATE, seed=123)
        g2 = create_genome_from_template(FULL_TEMPLATE, seed=123)

        for name in FULL_TEMPLATE:
            assert g1.genes[name].value == g2.genes[name].value

    def test_invalid_string_type_raises(self) -> None:
        bad_template = {
            "bad": {
                "type": "nonexistent",
                "mutation_rate": 0.1,
            }
        }
        with pytest.raises(ValueError, match="Unknown gene type"):
            create_genome_from_template(bad_template)

    def test_serialization_roundtrip(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)
        yaml_str = genome.to_yaml()
        restored = Genome.from_yaml(yaml_str)

        assert set(restored.genes.keys()) == set(genome.genes.keys())
        for name in genome.genes:
            assert restored.genes[name].value == genome.genes[name].value


# ---------------------------------------------------------------------------
# Generic mutation operators
# ---------------------------------------------------------------------------


class TestMutateGene:
    """Tests for mutate_gene dispatching to the right operator."""

    def test_text_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)
        original_val = genome.genes["agent_prompt"].value

        mutated = mutate_gene(
            genome,
            "agent_prompt",
            templates=FULL_TEMPLATE["agent_prompt"]["templates"],
            seed=1,
        )

        assert mutated.genome_id != genome.genome_id
        assert mutated.genes["agent_prompt"].type == GeneType.TEXT
        # Should be different (with high probability given seed)
        # But at minimum the genome is cloned
        assert mutated is not genome

    def test_float_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)
        original_val = genome.genes["temperature"].value

        mutated = mutate_gene(genome, "temperature", seed=1)

        assert mutated.genes["temperature"].type == GeneType.FLOAT
        # Value should be within range
        lo, hi = mutated.genes["temperature"].range
        assert lo <= mutated.genes["temperature"].value <= hi

    def test_enum_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)

        mutated = mutate_gene(genome, "strategy", seed=1)

        assert mutated.genes["strategy"].type == GeneType.ENUM
        assert mutated.genes["strategy"].value in ["sequential", "parallel", "hierarchical"]

    def test_set_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)

        mutated = mutate_gene(genome, "tools_enabled", seed=1)

        assert mutated.genes["tools_enabled"].type == GeneType.SET
        # All items should be from the available list
        available = set(FULL_TEMPLATE["tools_enabled"]["available"])
        assert all(t in available for t in mutated.genes["tools_enabled"].value)

    def test_numeric_vector_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)

        mutated = mutate_gene(genome, "quality_weights", seed=1)

        assert mutated.genes["quality_weights"].type == GeneType.NUMERIC_VECTOR
        vec = mutated.genes["quality_weights"].value
        assert isinstance(vec, list)
        assert all(isinstance(v, (int, float)) for v in vec)

    def test_nonexistent_gene_raises(self) -> None:
        genome = create_genome_from_template(SIMPLE_TEMPLATE, seed=42)
        with pytest.raises(ValueError, match="not found"):
            mutate_gene(genome, "does_not_exist")

    def test_immutability(self) -> None:
        """Original genome is not modified."""
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)
        original_val = genome.genes["temperature"].value

        _ = mutate_gene(genome, "temperature", seed=1)

        assert genome.genes["temperature"].value == original_val


# ---------------------------------------------------------------------------
# set_swap generic
# ---------------------------------------------------------------------------


class TestSetSwap:
    """Tests for set_swap on arbitrary SET genes."""

    def test_works_on_custom_set_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)
        original_tools = set(genome.genes["tools_enabled"].value)

        mutated = set_swap(genome, "tools_enabled", seed=1)

        new_tools = set(mutated.genes["tools_enabled"].value)
        # Should differ by at most one item (add, remove, or swap)
        symmetric_diff = original_tools.symmetric_difference(new_tools)
        assert len(symmetric_diff) <= 2  # swap changes 2, add/remove changes 1

    def test_raises_for_non_set_gene(self) -> None:
        genome = create_genome_from_template(SIMPLE_TEMPLATE, seed=42)
        with pytest.raises(ValueError, match="expected SET"):
            set_swap(genome, "my_text")

    def test_raises_for_missing_gene(self) -> None:
        genome = create_genome_from_template(SIMPLE_TEMPLATE, seed=42)
        with pytest.raises(ValueError, match="not found"):
            set_swap(genome, "nonexistent")


# ---------------------------------------------------------------------------
# text_swap generic
# ---------------------------------------------------------------------------


class TestTextSwap:
    """Tests for text_swap on arbitrary TEXT genes."""

    def test_works_on_custom_text_gene(self) -> None:
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)

        mutated = text_swap(
            genome,
            "agent_prompt",
            templates=FULL_TEMPLATE["agent_prompt"]["templates"],
            seed=1,
        )

        assert mutated.genes["agent_prompt"].type == GeneType.TEXT
        assert mutated.genome_id != genome.genome_id

    def test_shuffles_when_no_templates(self) -> None:
        """When no templates are provided and gene is not in DEFAULT, shuffles."""
        genome = create_genome_from_template(FULL_TEMPLATE, seed=42)

        # agent_prompt is not in DEFAULT_GENE_TEMPLATES, so it should shuffle
        mutated = text_swap(genome, "agent_prompt", seed=1)

        assert mutated.genes["agent_prompt"].type == GeneType.TEXT
        assert mutated.genome_id != genome.genome_id

    def test_raises_for_non_text_gene(self) -> None:
        genome = create_genome_from_template(SIMPLE_TEMPLATE, seed=42)
        with pytest.raises(ValueError, match="expected TEXT"):
            text_swap(genome, "my_float")


# ---------------------------------------------------------------------------
# Population.spawn with custom template
# ---------------------------------------------------------------------------


class TestPopulationSpawnCustom:
    """Tests for Population.spawn with gene_template."""

    def test_spawn_with_custom_template(self) -> None:
        pop = Population.spawn(5, seed=42, gene_template=FULL_TEMPLATE)

        assert pop.size == 5
        assert pop.generation == 0

        for genome in pop.members:
            assert set(genome.genes.keys()) == set(FULL_TEMPLATE.keys())
            assert genome.genes["agent_prompt"].type == GeneType.TEXT
            assert genome.genes["strategy"].type == GeneType.ENUM
            assert genome.genes["tools_enabled"].type == GeneType.SET
            assert genome.genes["temperature"].type == GeneType.FLOAT
            assert genome.genes["quality_weights"].type == GeneType.NUMERIC_VECTOR

    def test_spawn_without_template_uses_defaults(self) -> None:
        """Backward compatibility: spawn without template uses DEFAULT_GENE_TEMPLATES."""
        pop = Population.spawn(3, seed=42)

        assert pop.size == 3
        for genome in pop.members:
            assert "prompt_template" in genome.genes
            assert "tool_policy" in genome.genes

    def test_gene_template_stored_on_population(self) -> None:
        pop = Population.spawn(3, seed=42, gene_template=FULL_TEMPLATE)
        assert pop.gene_template is FULL_TEMPLATE


# ---------------------------------------------------------------------------
# Full breeding loop with custom genome
# ---------------------------------------------------------------------------


class TestFullBreedingLoop:
    """End-to-end test: spawn -> select -> breed -> mutate with custom genomes."""

    def test_full_loop(self) -> None:
        pop = Population.spawn(10, seed=42, gene_template=FULL_TEMPLATE)
        assert pop.size == 10

        # Simulate fitness scores
        import random as stdlib_random

        rng = stdlib_random.Random(99)
        scores = [rng.random() for _ in range(pop.size)]

        # Select top 4
        pop.select(fitness_scores=scores, top_k=4, method="truncation")
        assert pop.size == 4

        # Breed back to 10
        pop.breed(target_size=10, seed=42)
        assert pop.size == 10

        # Mutate
        pop.mutate(rate=0.5, seed=42)
        assert pop.size == 10

        # All genomes should still have the custom gene set
        for genome in pop.members:
            assert set(genome.genes.keys()) == set(FULL_TEMPLATE.keys())

    def test_immigration_uses_custom_template(self) -> None:
        pop = Population.spawn(5, seed=42, gene_template=FULL_TEMPLATE)
        pop.add_immigrants(3, seed=99)

        assert pop.size == 8
        # Immigrants should also have the custom gene set
        for genome in pop.members:
            assert set(genome.genes.keys()) == set(FULL_TEMPLATE.keys())


# ---------------------------------------------------------------------------
# Crossover with custom genomes
# ---------------------------------------------------------------------------


class TestCrossoverCustomGenomes:
    """Crossover operators are generic -- verify they work with custom genomes."""

    def test_uniform_crossover(self) -> None:
        g1 = create_genome_from_template(FULL_TEMPLATE, seed=1)
        g2 = create_genome_from_template(FULL_TEMPLATE, seed=2)

        child = uniform_crossover(g1, g2, seed=42)

        assert set(child.genes.keys()) == set(FULL_TEMPLATE.keys())
        assert child.genome_id != g1.genome_id
        assert child.genome_id != g2.genome_id
        assert len(child.parents) == 2

    def test_child_genes_come_from_parents(self) -> None:
        g1 = create_genome_from_template(FULL_TEMPLATE, seed=1)
        g2 = create_genome_from_template(FULL_TEMPLATE, seed=2)

        child = uniform_crossover(g1, g2, seed=42)

        for name in FULL_TEMPLATE:
            gene = child.genes[name]
            # For scalar types, value must come from one of the parents
            if gene.type in (GeneType.TEXT, GeneType.ENUM, GeneType.FLOAT):
                assert gene.value in (
                    g1.genes[name].value,
                    g2.genes[name].value,
                )


# ---------------------------------------------------------------------------
# Distance with custom genomes
# ---------------------------------------------------------------------------


class TestDistanceCustomGenomes:
    """Genome.distance works with custom gene sets."""

    def test_distance_same_genome(self) -> None:
        g = create_genome_from_template(FULL_TEMPLATE, seed=42)
        assert g.distance(g) == 0.0

    def test_distance_different_genomes(self) -> None:
        g1 = create_genome_from_template(FULL_TEMPLATE, seed=1)
        g2 = create_genome_from_template(FULL_TEMPLATE, seed=2)

        d = g1.distance(g2)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# BreedingEngine with custom gene_template
# ---------------------------------------------------------------------------

_MINI_TEMPLATE: dict = {
    "prompt": {
        "type": "text",
        "templates": ["Be concise.", "Be thorough.", "Be creative."],
        "mutation_rate": 0.2,
    },
    "mode": {
        "type": "enum",
        "options": ["fast", "slow", "balanced"],
        "mutation_rate": 0.15,
    },
    "temp": {
        "type": "float",
        "range": (0.0, 1.0),
        "mutation_rate": 0.1,
    },
}


class _CustomMockArena(Arena):
    """Arena that scores genomes based on their float gene value."""

    async def generate_tasks(
        self, count: int, seed: int | None = None
    ) -> list[Task]:
        return [
            Task(task_id=f"t-{i}", prompt=f"Q{i}", expected=1.0)
            for i in range(count)
        ]

    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        result = await adapter.run(genome, tasks[0].prompt)
        temp_gene = genome.genes.get("temp")
        fitness = 1.0 - float(temp_gene.value) if temp_gene else 0.5
        return EvalResult(
            genome_id=genome.genome_id,
            fitness_score=max(0.0, min(1.0, fitness)),
            task_results=[result],
        )


class _MockAdapter(Adapter):
    async def run(self, genome: Genome, task: str) -> AgentResult:
        return AgentResult(output="ok", duration_ms=1)


class TestBreedingEngineCustomTemplate:
    """BreedingEngine works end-to-end with a custom gene_template."""

    @pytest.fixture
    def small_config(self, tmp_path: Any) -> BreedingConfig:
        return BreedingConfig(
            population_size=6,
            elite_count=2,
            generations=3,
            mutation_rate=0.2,
            immigration_rate=0.0,
            seed=42,
            num_tasks=2,
            results_dir=str(tmp_path / "results"),
        )

    @pytest.mark.asyncio
    async def test_engine_returns_champion_with_custom_genes(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_CustomMockArena(),
            adapter=_MockAdapter(),
            gene_template=_MINI_TEMPLATE,
        )

        champion = await engine.run()

        assert isinstance(champion, Genome)
        assert set(champion.genes.keys()) == {"prompt", "mode", "temp"}
        assert engine.champion_fitness > 0.0

    @pytest.mark.asyncio
    async def test_engine_population_uses_custom_template(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_CustomMockArena(),
            adapter=_MockAdapter(),
            gene_template=_MINI_TEMPLATE,
        )

        await engine.run()

        pop = engine.population
        assert pop is not None
        for genome in pop.members:
            assert set(genome.genes.keys()) == {"prompt", "mode", "temp"}
