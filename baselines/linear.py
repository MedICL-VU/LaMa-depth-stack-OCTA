"""Mask-aware linear interpolation baseline along the B-scan axis."""

from __future__ import annotations

import time

import numpy as np

from ..core.normalization import fit_observed_positive_scale, validate_volume_and_mask
from ..core.reconstruction import (
    CanonicalConfig,
    ReconstructionResult,
    compose_missing_only,
)


def reconstruct_linear(
    corrupted: np.ndarray,
    missing_mask: np.ndarray,
    config: CanonicalConfig | None = None,
) -> ReconstructionResult:
    """Interpolate every `(Z,W)` line independently along `B`.

    Interior gaps use linear interpolation. Leading and trailing gaps use the
    nearest observed endpoint, matching `numpy.interp`. A line with one
    observation is constant; a line with none remains unchanged.
    """

    config = config or CanonicalConfig()
    values, missing = validate_volume_and_mask(corrupted, missing_mask)
    scale = fit_observed_positive_scale(
        values, missing, percentile=config.positive_percentile
    )
    model_input = scale.apply(values)
    raw_prediction_01 = model_input.copy()
    coordinates = np.arange(values.shape[0], dtype=np.float32)
    started = time.perf_counter()
    interpolated_lines = 0
    empty_lines = 0

    for z in range(values.shape[1]):
        for w in range(values.shape[2]):
            line_missing = missing[:, z, w]
            if not np.any(line_missing):
                continue
            observed_indices = coordinates[~line_missing]
            if observed_indices.size == 0:
                empty_lines += 1
                continue
            observed_values = model_input[~line_missing, z, w]
            raw_prediction_01[line_missing, z, w] = np.interp(
                coordinates[line_missing], observed_indices, observed_values
            )
            interpolated_lines += 1

    raw_native = scale.invert(raw_prediction_01)
    final = compose_missing_only(values, raw_native, missing)
    metadata = {
        "method": "linear_interpolation",
        "axis_order": "BZW",
        "interpolation_axis": "B",
        "normalization": scale.to_dict(),
        "interpolated_lines": interpolated_lines,
        "lines_without_observed_samples": empty_lines,
        "observed_max_abs_error": float(
            np.max(np.abs(final[~missing] - values[~missing]))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "gt_required": False,
        "config": config.to_dict(),
        "backend": {"backend": "numpy.interp"},
    }
    return ReconstructionResult(raw_native, final, model_input, scale, metadata)
