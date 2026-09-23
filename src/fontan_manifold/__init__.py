from .core import (
    FontanManifold,
    FontanOutcomeAnalyzer,
    ProjectionResult,
    default_sex_binary,
    DEFAULT_DOMAIN_DEFINITION,
    DEFAULT_CONTINUOUS_MI_VARS,
    DEFAULT_OUTCOMES,
)

__all__ = [
    "FontanManifold",
    "FontanOutcomeAnalyzer",
    "ProjectionResult",
    "default_sex_binary",
    "DEFAULT_DOMAIN_DEFINITION",
    "DEFAULT_CONTINUOUS_MI_VARS",
    "DEFAULT_OUTCOMES",
]

from .plotting import (
    plot_manifold_3d,
    plot_component_characterization,
    plot_pt_scatter,
    plot_outcome_forest,
)

from .figure1 import make_figure1
