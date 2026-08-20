"""Compact, model-facing and qualitative artifacts for one reconstructed case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io import save_volume
from .reconstruction import CanonicalConfig, ReconstructionResult


def _display_scale(images: list[np.ndarray]) -> float:
    positive = np.concatenate(
        [np.asarray(image)[np.asarray(image) > 0] for image in images]
    )
    return float(np.percentile(positive, 99.5)) if positive.size else 1.0


def _save_gray(path: Path, image: np.ndarray, *, scale: float) -> None:
    normalized = np.clip(
        np.asarray(image, dtype=np.float32) / max(scale, 1e-8), 0.0, 1.0
    )
    Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8)).save(path)


def write_case_artifacts(
    *,
    output_dir: Path,
    case_id: str,
    corrupted: np.ndarray,
    missing_mask: np.ndarray,
    result: ReconstructionResult,
    source_dtype: np.dtype,
    config: CanonicalConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write native-volume outputs, MIPs, representative planes, and provenance."""

    volumes = output_dir / "volumes"
    images = output_dir / "images"
    planes = images / "fixed_depth_planes"
    bscans = images / "bscans"
    backend_inputs = images / "backend_inputs"
    for directory in (volumes, images, planes, bscans, backend_inputs):
        directory.mkdir(parents=True, exist_ok=True)

    save_volume(
        volumes / "raw_prediction.tif", result.raw_prediction, dtype=source_dtype
    )
    save_volume(
        volumes / "final_reconstruction.tif",
        result.final_reconstruction,
        dtype=source_dtype,
    )
    save_volume(
        volumes / "model_input_native_scale.tif",
        result.scale.invert(result.model_input),
        dtype=source_dtype,
    )
    save_volume(
        volumes / "missing_mask.tif",
        np.asarray(missing_mask, dtype=np.uint8),
        dtype=np.uint8,
    )

    corrupted_mip = np.max(corrupted, axis=1)
    raw_mip = np.max(result.raw_prediction, axis=1)
    final_mip = np.max(result.final_reconstruction, axis=1)
    mip_scale = _display_scale([corrupted_mip, final_mip])
    _save_gray(images / "corrupted_full_axial_mip.png", corrupted_mip, scale=mip_scale)
    _save_gray(images / "raw_prediction_full_axial_mip.png", raw_mip, scale=mip_scale)
    _save_gray(images / "final_full_axial_mip.png", final_mip, scale=mip_scale)

    z_indices = np.linspace(
        0, corrupted.shape[1] - 1, config.depth_panel_count, dtype=int
    )
    for z in np.unique(z_indices):
        scale = _display_scale(
            [corrupted[:, z, :], result.final_reconstruction[:, z, :]]
        )
        _save_gray(planes / f"z{z:04d}_corrupted.png", corrupted[:, z, :], scale=scale)
        _save_gray(
            planes / f"z{z:04d}_final.png",
            result.final_reconstruction[:, z, :],
            scale=scale,
        )

    processed_z = np.flatnonzero(np.any(missing_mask, axis=(0, 2)))
    for z in processed_z[: config.backend_input_preview_count]:
        _save_gray(
            backend_inputs / f"z{z:04d}_normalized_input.png",
            result.model_input[:, z, :],
            scale=1.0,
        )
        Image.fromarray(np.where(missing_mask[:, z, :], 255, 0).astype(np.uint8)).save(
            backend_inputs / f"z{z:04d}_white_missing_mask.png"
        )

    affected_b = np.flatnonzero(np.any(missing_mask, axis=(1, 2)))
    if affected_b.size:
        selected = affected_b[
            np.linspace(0, affected_b.size - 1, config.bscan_panel_count, dtype=int)
        ]
        for b in np.unique(selected):
            scale = _display_scale([corrupted[b], result.final_reconstruction[b]])
            _save_gray(bscans / f"b{b:04d}_corrupted.png", corrupted[b], scale=scale)
            _save_gray(
                bscans / f"b{b:04d}_final.png",
                result.final_reconstruction[b],
                scale=scale,
            )

    metadata = {"case_id": case_id, **result.metadata, **(extra_metadata or {})}
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
