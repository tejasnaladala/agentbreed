"""Analysis modules for fitness landscape decomposition."""

from breed.analysis.schema_preservation import (
    RuggednessReport,
    SchemaPreservationReport,
    compute_ruggedness,
    effective_dimensionality_via_pca,
    schema_preservation_rate,
    typed_hamming_distance,
)
from breed.analysis.landscape import (
    LandscapeSummary,
    genomes_to_feature_matrix,
    summarize_landscape,
)

__all__ = [
    "RuggednessReport",
    "SchemaPreservationReport",
    "LandscapeSummary",
    "compute_ruggedness",
    "effective_dimensionality_via_pca",
    "schema_preservation_rate",
    "typed_hamming_distance",
    "genomes_to_feature_matrix",
    "summarize_landscape",
]

# Epistasis module is imported lazily to allow partial-install use cases
# during development.
try:
    from breed.analysis.epistasis import (  # noqa: F401
        EpistasisReport,
        SobolIndices,
        compute_pairwise_interactions,
        compute_sobol_indices,
        functional_anova_decomposition,
        sobol_sample,
    )

    __all__ += [
        "EpistasisReport",
        "SobolIndices",
        "compute_pairwise_interactions",
        "compute_sobol_indices",
        "functional_anova_decomposition",
        "sobol_sample",
    ]
except ImportError:
    pass
