"""Canonical fixed-depth LaMa stacking and missing-only composition."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .backend import BIG_LAMA_SHA256, InpaintBackend, inpaint_plane
from .normalization import (
    ObservedPositiveScale,
    fit_observed_positive_scale,
    validate_volume_and_mask,
)


@dataclass(frozen=True)
class CanonicalConfig:
    """Configuration values that define canonical LaMa-depth-stack."""

    positive_percentile: float = 99.9
    checkpoint_sha256: str = BIG_LAMA_SHA256
    verify_checkpoint: bool = True
    backend_input_preview_count: int = 2
    depth_panel_count: int = 5
    bscan_panel_count: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.positive_percentile <= 100.0:
            raise ValueError("positive_percentile must be in (0, 100].")
        preview_counts = (
            self.backend_input_preview_count,
            self.depth_panel_count,
            self.bscan_panel_count,
        )
        if any(count < 0 for count in preview_counts):
            raise ValueError("Artifact preview counts must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionResult:
    """Native-intensity outputs from canonical LaMa-depth-stack inference."""

    raw_prediction: np.ndarray
    final_reconstruction: np.ndarray
    model_input: np.ndarray
    scale: ObservedPositiveScale
    metadata: dict[str, Any]


def compose_missing_only(
    corrupted: np.ndarray,
    raw_prediction: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    """Paste predictions only inside the missing support."""

    corrupted_arr = np.asarray(corrupted)
    raw = np.asarray(raw_prediction, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    if corrupted_arr.shape != raw.shape or raw.shape != missing.shape:
        raise ValueError(
            "Corrupted, raw prediction, and mask must have identical shapes."
        )
    return np.where(missing, raw, corrupted_arr).astype(np.float32, copy=False)


class CanonicalDepthStackReconstructor:
    """Apply a frozen 2D backend independently to `C[:, z, :]` planes."""

    def __init__(
        self, backend: InpaintBackend, config: CanonicalConfig | None = None
    ) -> None:
        self.backend = backend
        self.config = config or CanonicalConfig()

    def reconstruct(
        self, corrupted: np.ndarray, missing_mask: np.ndarray
    ) -> ReconstructionResult:
        corrupted_values, missing = validate_volume_and_mask(corrupted, missing_mask)
        if not np.any(missing):
            raise ValueError("Missing mask contains no corrupted voxels.")

        scale = fit_observed_positive_scale(
            corrupted_values,
            missing,
            percentile=self.config.positive_percentile,
        )
        model_input = scale.apply(corrupted_values)
        raw_prediction = corrupted_values.copy()
        processed_planes = 0
        max_bottom_crop = 0
        max_right_crop = 0
        started = time.perf_counter()

        # Axis contract: every target plane is `(B,W)` at one axial depth `z`.
        for z in range(corrupted_values.shape[1]):
            plane_mask = missing[:, z, :]
            if not np.any(plane_mask):
                continue
            prediction_01, crop = inpaint_plane(
                self.backend,
                model_input[:, z, :],
                plane_mask,
            )
            raw_prediction[:, z, :] = scale.invert(prediction_01)
            processed_planes += 1
            max_bottom_crop = max(max_bottom_crop, crop["bottom_padding_cropped"])
            max_right_crop = max(max_right_crop, crop["right_padding_cropped"])

        final = compose_missing_only(corrupted_values, raw_prediction, missing)
        observed_error = float(
            np.max(np.abs(final[~missing] - corrupted_values[~missing]))
        )
        if observed_error != 0.0:
            raise RuntimeError(
                f"Observed-voxel invariance failed: max error={observed_error}."
            )

        metadata = {
            "method": "lama_depth_stack",
            "axis_order": "BZW",
            "processed_plane": "volume[:, z, :] with shape (B,W)",
            "volume_shape_bzw": [int(value) for value in corrupted_values.shape],
            "mask_true_means": "missing",
            "normalization": scale.to_dict(),
            "quantization": "round(clip(x,0,1)*255) to uint8; grayscale replicated to RGB",
            "output_channel_reduction": "floating_point_rgb_mean",
            "processed_depth_planes": int(processed_planes),
            "total_depth_planes": int(corrupted_values.shape[1]),
            "max_bottom_padding_cropped": int(max_bottom_crop),
            "max_right_padding_cropped": int(max_right_crop),
            "observed_max_abs_error": observed_error,
            "gt_required": False,
            "runtime_seconds": float(time.perf_counter() - started),
            "backend": self.backend.metadata,
            "config": self.config.to_dict(),
        }
        return ReconstructionResult(
            raw_prediction=raw_prediction,
            final_reconstruction=final,
            model_input=model_input,
            scale=scale,
            metadata=metadata,
        )
