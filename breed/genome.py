"""Agent genome schema and gene templates for evolutionary breeding.

Defines the genetic representation of an AI agent: a collection of typed genes
that control prompt strategy, tool usage, calibration, and other behavioral axes.
Provider-agnostic -- works for any agent framework.
"""

from __future__ import annotations

import copy
import math
import random
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Gene type enum
# ---------------------------------------------------------------------------

class GeneType(str, Enum):
    """Supported gene value types."""

    TEXT = "text"
    ENUM = "enum"
    FLOAT = "float"
    SET = "set"
    NUMERIC_VECTOR = "numeric_vector"


# ---------------------------------------------------------------------------
# Gene model
# ---------------------------------------------------------------------------

class Gene(BaseModel):
    """A single gene within an agent genome.

    Attributes:
        name: Human-readable gene identifier.
        type: The data type / interpretation of the gene value.
        value: Current gene value (type depends on *type*).
        mutation_rate: Probability of mutation per generation (0.0-1.0).
        options: Valid choices when *type* is ENUM.
        range: (min, max) tuple when *type* is FLOAT.
        available: Universe of legal items when *type* is SET.
    """

    name: str
    type: GeneType
    value: Any
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    options: list[str] | None = None
    range: tuple[float, float] | None = None
    available: list[str] | None = None

    @model_validator(mode="after")
    def _validate_value_against_constraints(self) -> "Gene":
        if self.type == GeneType.ENUM:
            if self.options is None:
                raise ValueError(f"Gene '{self.name}': ENUM gene requires 'options'")
            if self.value not in self.options:
                raise ValueError(
                    f"Gene '{self.name}': value '{self.value}' not in options {self.options}"
                )

        if self.type == GeneType.FLOAT:
            if self.range is None:
                raise ValueError(f"Gene '{self.name}': FLOAT gene requires 'range'")
            lo, hi = self.range
            if not (lo <= self.value <= hi):
                raise ValueError(
                    f"Gene '{self.name}': value {self.value} outside range [{lo}, {hi}]"
                )

        if self.type == GeneType.SET:
            if self.available is None:
                raise ValueError(f"Gene '{self.name}': SET gene requires 'available'")
            invalid = set(self.value) - set(self.available)
            if invalid:
                raise ValueError(
                    f"Gene '{self.name}': values {invalid} not in available {self.available}"
                )

        if self.type == GeneType.NUMERIC_VECTOR:
            if not isinstance(self.value, list) or not all(
                isinstance(v, (int, float)) for v in self.value
            ):
                raise ValueError(
                    f"Gene '{self.name}': NUMERIC_VECTOR value must be a list of numbers"
                )

        return self


# ---------------------------------------------------------------------------
# Genome model
# ---------------------------------------------------------------------------

class Genome(BaseModel):
    """Complete genetic representation of an AI agent.

    Attributes:
        genome_id: Unique identifier (UUID4 hex string).
        generation: Generation number (0 for seed genomes).
        parents: IDs of parent genomes (empty for seed).
        created_at: ISO-8601 creation timestamp.
        genes: Mapping of gene name -> Gene.
        fitness_history: Ordered list of fitness scores across evaluations.
        metadata: Arbitrary extra data attached to the genome.
    """

    genome_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    generation: int = 0
    parents: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    genes: dict[str, Gene] = Field(default_factory=dict)
    fitness_history: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the genome to a plain dictionary.

        Uses JSON-compatible mode so enum values become plain strings and
        tuples become lists, ensuring safe YAML / JSON roundtrips.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        """Deserialize a genome from a plain dictionary."""
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize the genome to a YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Genome":
        """Deserialize a genome from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    # -- distance ------------------------------------------------------------

    def distance(self, other: "Genome") -> float:
        """Compute a normalised 0.0-1.0 distance to *other*.

        Strategy per gene type:
        * TEXT: Jaccard distance on word-level bigrams.
        * ENUM: 0.0 if equal, 1.0 otherwise.
        * FLOAT: |a - b| / (hi - lo), clamped to [0, 1].
        * SET: 1 - Jaccard similarity of the two sets.
        * NUMERIC_VECTOR: normalised Euclidean distance.
        """
        shared_genes = set(self.genes.keys()) & set(other.genes.keys())
        if not shared_genes:
            return 1.0

        total = 0.0
        for name in shared_genes:
            gene_a = self.genes[name]
            gene_b = other.genes[name]
            total += _gene_distance(gene_a, gene_b)

        return total / len(shared_genes)


# ---------------------------------------------------------------------------
# Gene distance helpers
# ---------------------------------------------------------------------------

def _word_bigrams(text: str) -> set[str]:
    words = text.lower().split()
    if len(words) < 2:
        return set(words)
    return {f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)}


def _jaccard(set_a: set[Any], set_b: set[Any]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    inter = set_a & set_b
    return 1.0 - len(inter) / len(union)


def _gene_distance(a: Gene, b: Gene) -> float:
    if a.type != b.type:
        return 1.0

    if a.type == GeneType.TEXT:
        return _jaccard(_word_bigrams(str(a.value)), _word_bigrams(str(b.value)))

    if a.type == GeneType.ENUM:
        return 0.0 if a.value == b.value else 1.0

    if a.type == GeneType.FLOAT:
        lo, hi = a.range or (0.0, 1.0)
        span = hi - lo
        if span == 0:
            return 0.0
        return min(abs(a.value - b.value) / span, 1.0)

    if a.type == GeneType.SET:
        return _jaccard(set(a.value), set(b.value))

    if a.type == GeneType.NUMERIC_VECTOR:
        vec_a = a.value
        vec_b = b.value
        if len(vec_a) != len(vec_b):
            return 1.0
        sq_sum = sum((x - y) ** 2 for x, y in zip(vec_a, vec_b))
        # max possible distance when each element is in [0,1] is sqrt(n)
        max_dist = math.sqrt(len(vec_a))
        if max_dist == 0:
            return 0.0
        return min(math.sqrt(sq_sum) / max_dist, 1.0)

    return 1.0


# ---------------------------------------------------------------------------
# Default gene templates (10 genes)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES: list[str] = [
    (
        "You are a superforecaster. Break the question into independent sub-questions, "
        "estimate each, then combine using the chain rule of probability."
    ),
    (
        "Act as an expert calibrated predictor. Use base rates from reference classes, "
        "then adjust for specific evidence using Bayesian updating."
    ),
    (
        "You are an analytical forecaster. Identify the three strongest arguments for and "
        "against, weigh them explicitly, and output a probability."
    ),
    (
        "Adopt the outside view first: find the historical base rate for this category of "
        "event, then adjust inward based on inside-view evidence."
    ),
    (
        "Think step by step like a prediction market trader. Consider liquidity of "
        "information, market consensus, and contrarian signals before estimating."
    ),
]

_DECISION_HEURISTICS: list[str] = [
    (
        "Use expected-value reasoning: enumerate possible outcomes, assign probabilities "
        "and payoffs, then choose the option that maximises expected value."
    ),
    (
        "Apply minimax regret: consider what you would regret most if wrong, and choose "
        "the action that minimises worst-case regret."
    ),
    (
        "Use recognition-primed decision making: match the current situation to the most "
        "similar past case and adapt that solution."
    ),
    (
        "Decompose into Fermi sub-estimates, sanity-check each against known anchors, "
        "then multiply/add to form a final answer."
    ),
    (
        "Generate three competing hypotheses, seek discriminating evidence for each, "
        "then weight by evidence strength."
    ),
]

_CALIBRATION_RULES: list[str] = [
    (
        "After forming an initial estimate, apply the calibration nudge: move 10% toward "
        "the base rate to counteract overconfidence."
    ),
    (
        "Check your estimate against Tetlock's calibration curve: 70% predictions should "
        "resolve true about 70% of the time. Adjust if your track record diverges."
    ),
    (
        "Apply the extremeness aversion correction: if your estimate is above 90% or "
        "below 10%, justify each percentage point beyond 80%/20%."
    ),
    (
        "Use the three-interval check: express uncertainty as a 50% confidence interval "
        "first, then widen to 90%. If the 90% interval is narrower than 3x the 50%, "
        "you are overconfident."
    ),
]

_EVIDENCE_WEIGHT_TEMPLATES: list[list[float]] = [
    [0.4, 0.35, 0.25],
    [0.2, 0.5, 0.3],
    [0.5, 0.2, 0.3],
    [0.33, 0.34, 0.33],
]

DEFAULT_GENE_TEMPLATES: dict[str, dict[str, Any]] = {
    "prompt_template": {
        "type": GeneType.TEXT,
        "templates": _PROMPT_TEMPLATES,
        "mutation_rate": 0.15,
    },
    "memory_structure": {
        "type": GeneType.ENUM,
        "options": ["sliding_window", "episodic", "semantic", "graph", "none"],
        "mutation_rate": 0.1,
    },
    "tool_policy": {
        "type": GeneType.SET,
        "available": ["web_search", "web_fetch", "calculator", "code_exec", "read", "write"],
        "mutation_rate": 0.1,
    },
    "decision_heuristic": {
        "type": GeneType.TEXT,
        "templates": _DECISION_HEURISTICS,
        "mutation_rate": 0.15,
    },
    "calibration_rule": {
        "type": GeneType.TEXT,
        "templates": _CALIBRATION_RULES,
        "mutation_rate": 0.12,
    },
    "evidence_weighting": {
        "type": GeneType.NUMERIC_VECTOR,
        "templates": _EVIDENCE_WEIGHT_TEMPLATES,
        "mutation_rate": 0.1,
    },
    "risk_threshold": {
        "type": GeneType.FLOAT,
        "range": (0.01, 0.50),
        "mutation_rate": 0.1,
    },
    "decomposition_style": {
        "type": GeneType.ENUM,
        "options": ["reference_class", "causal_chain", "scenario_tree", "fermi", "analogy"],
        "mutation_rate": 0.1,
    },
    "temperature": {
        "type": GeneType.FLOAT,
        "range": (0.0, 1.5),
        "mutation_rate": 0.1,
    },
    "answer_format": {
        "type": GeneType.ENUM,
        "options": ["probability_only", "probability_with_reasoning", "structured_json"],
        "mutation_rate": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Random genome factory
# ---------------------------------------------------------------------------

def create_random_genome(seed: int | None = None) -> Genome:
    """Create a genome with randomised gene values drawn from the default templates.

    Args:
        seed: Optional RNG seed for reproducibility.

    Returns:
        A fully-populated :class:`Genome` with generation 0.
    """
    rng = random.Random(seed)
    genes: dict[str, Gene] = {}

    for gene_name, template in DEFAULT_GENE_TEMPLATES.items():
        gene_type: GeneType = template["type"]
        mutation_rate: float = template["mutation_rate"]

        if gene_type == GeneType.TEXT:
            templates: list[str] = template["templates"]
            value = rng.choice(templates)
            genes[gene_name] = Gene(
                name=gene_name,
                type=gene_type,
                value=value,
                mutation_rate=mutation_rate,
            )

        elif gene_type == GeneType.ENUM:
            options: list[str] = template["options"]
            value = rng.choice(options)
            genes[gene_name] = Gene(
                name=gene_name,
                type=gene_type,
                value=value,
                mutation_rate=mutation_rate,
                options=options,
            )

        elif gene_type == GeneType.FLOAT:
            lo, hi = template["range"]
            value = rng.uniform(lo, hi)
            genes[gene_name] = Gene(
                name=gene_name,
                type=gene_type,
                value=round(value, 6),
                mutation_rate=mutation_rate,
                range=(lo, hi),
            )

        elif gene_type == GeneType.SET:
            available: list[str] = template["available"]
            k = rng.randint(1, len(available))
            value = sorted(rng.sample(available, k))
            genes[gene_name] = Gene(
                name=gene_name,
                type=gene_type,
                value=value,
                mutation_rate=mutation_rate,
                available=available,
            )

        elif gene_type == GeneType.NUMERIC_VECTOR:
            vec_templates: list[list[float]] = template["templates"]
            value = list(rng.choice(vec_templates))
            genes[gene_name] = Gene(
                name=gene_name,
                type=gene_type,
                value=value,
                mutation_rate=mutation_rate,
            )

    return Genome(genes=genes)
