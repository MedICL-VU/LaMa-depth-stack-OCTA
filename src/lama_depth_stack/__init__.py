"""Canonical LaMa-depth-stack OCTA reconstruction package."""

from .core.normalization import ObservedPositiveScale, fit_observed_positive_scale
from .core.reconstruction import (
    CanonicalConfig,
    CanonicalDepthStackReconstructor,
    ReconstructionResult,
)

__all__ = [
    "CanonicalConfig",
    "CanonicalDepthStackReconstructor",
    "ObservedPositiveScale",
    "ReconstructionResult",
    "fit_observed_positive_scale",
]

__version__ = "0.1.0"
