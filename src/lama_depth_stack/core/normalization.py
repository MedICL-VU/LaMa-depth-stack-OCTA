"""Leakage-safe intensity normalization shared by reconstruction methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservedPositiveScale:
    """Per-volume scale estimated from positive observed corrupted voxels."""

    value: float
    percentile: float
    eligible_voxels: int
    observed_voxels: int

    def apply(self, volume: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(volume, dtype=np.float32) / self.value, 0.0, 1.0)

    def invert(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=np.float32) * self.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_volume_and_mask(
    volume: np.ndarray, missing_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate canonical `(B,Z,W)` arrays and return float/bool views."""

    values = np.asarray(volume)
    missing = np.asarray(missing_mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError(
            f"Expected volume in `(B,Z,W)` order, got shape {values.shape}."
        )
    if values.shape != missing.shape:
        raise ValueError(
            f"Volume/mask shape mismatch: {values.shape} != {missing.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Corrupted volume contains non-finite values.")
    return values.astype(np.float32, copy=False), missing


def fit_observed_positive_scale(
    corrupted: np.ndarray,
    missing_mask: np.ndarray,
    *,
    percentile: float = 99.9,
) -> ObservedPositiveScale:
    """Fit `P99.9({C_i: M_i=0, C_i>0})` without requiring clean GT."""

    values, missing = validate_volume_and_mask(corrupted, missing_mask)
    eligible = (~missing) & (values > 0)
    observed_positive = values[eligible]
    if observed_positive.size == 0:
        raise ValueError(
            "No finite positive observed voxels are available for normalization."
        )
    scale = float(np.percentile(observed_positive, percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid observed-positive percentile scale: {scale!r}.")
    return ObservedPositiveScale(
        value=scale,
        percentile=float(percentile),
        eligible_voxels=int(observed_positive.size),
        observed_voxels=int((~missing).sum()),
    )
