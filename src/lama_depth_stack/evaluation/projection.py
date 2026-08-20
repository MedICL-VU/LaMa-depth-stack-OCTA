"""Explicit full-depth and OCTA-500 ILM--OPL projection geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LayerSurfaces:
    """OCTA-500 surfaces in `(surface,B,W)` order and zero-based axial units."""

    values: np.ndarray
    source_path: Path

    @property
    def ilm(self) -> np.ndarray:
        return self.values[0]

    @property
    def opl(self) -> np.ndarray:
        return self.values[2]

    def shifted(self, axial_offset: int) -> LayerSurfaces:
        return LayerSurfaces(self.values - int(axial_offset), self.source_path)


@dataclass(frozen=True)
class EvaluationSupports:
    """Mutually exclusive voxel and projected supports for regional metrics."""

    full_bscan_voxels: np.ndarray
    lateral_voxels: np.ndarray
    full_bscan_projection: np.ndarray
    lateral_projection: np.ndarray


def load_layer_surfaces(path: Path) -> LayerSurfaces:
    """Load `(>=3,B,W)` surfaces from NPY, NPZ, or OCTA-500 MATLAB files."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            keys = list(archive.keys())
            if not keys:
                raise ValueError(f"Layer archive contains no arrays: {path}")
            key = "Layer" if "Layer" in archive else keys[0]
            values = np.asarray(archive[key])
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MATLAB layer files require the `evaluation` optional dependencies."
            ) from exc
        payload = loadmat(path)
        if "Layer" not in payload:
            raise KeyError(f"{path} does not contain the OCTA-500 `Layer` variable.")
        values = payload["Layer"]
    else:
        raise ValueError(f"Unsupported layer format: {path.suffix}")
    raw_surfaces = np.asarray(values)
    if raw_surfaces.ndim != 3 or raw_surfaces.shape[0] < 3:
        raise ValueError(f"Expected layer shape `(>=3,B,W)`, got {raw_surfaces.shape}.")
    if not np.isfinite(raw_surfaces).all():
        raise ValueError(f"Layer surfaces contain non-finite values: {path}")
    surfaces = np.rint(raw_surfaces).astype(np.int32)
    return LayerSurfaces(surfaces, path)


def project_full_axial(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume)
    if values.ndim != 3:
        raise ValueError(f"Expected `(B,Z,W)` volume, got {values.shape}.")
    return np.max(values, axis=1)


def project_layer_slab(
    volume: np.ndarray,
    lower_bw: np.ndarray,
    upper_bw: np.ndarray,
) -> np.ndarray:
    """Maximum-project each ray over inclusive, clipped anatomical bounds."""

    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected `(B,Z,W)` volume, got {values.shape}.")
    b_size, z_size, w_size = values.shape
    lower = np.asarray(lower_bw)
    upper = np.asarray(upper_bw)
    if lower.shape != (b_size, w_size) or upper.shape != (b_size, w_size):
        raise ValueError(
            f"Layer bounds must have shape {(b_size, w_size)}, got {lower.shape}/{upper.shape}."
        )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Layer surfaces contain non-finite bounds.")
    lo = np.clip(np.minimum(lower, upper).astype(np.int32), 0, z_size - 1)
    hi = np.clip(np.maximum(lower, upper).astype(np.int32), 0, z_size - 1)
    output = np.empty((b_size, w_size), dtype=np.float32)
    depth = np.arange(z_size)[:, None]
    for b in range(b_size):
        valid = (depth >= lo[b][None, :]) & (depth <= hi[b][None, :])
        if not np.all(np.any(valid, axis=0)):
            raise RuntimeError(
                "At least one ILM--OPL ray has zero valid axial samples."
            )
        output[b] = np.max(np.where(valid, values[b], -np.inf), axis=0)
    return output


def project_volume(
    volume: np.ndarray,
    *,
    projection_kind: str,
    layers: LayerSurfaces | None = None,
) -> np.ndarray:
    if projection_kind == "full_axial":
        return project_full_axial(volume)
    if projection_kind == "ilm_opl":
        if layers is None:
            raise ValueError("ILM--OPL projection requires layer surfaces.")
        return project_layer_slab(volume, layers.ilm, layers.opl)
    raise ValueError(f"Unknown projection kind `{projection_kind}`.")


def build_disjoint_supports(
    *,
    full_bscan_mask: np.ndarray,
    lateral_mask: np.ndarray,
    projection_kind: str,
    layers: LayerSurfaces | None,
) -> EvaluationSupports:
    """Assign lateral overlap to lateral support, then remove it from full support."""

    full = np.asarray(full_bscan_mask, dtype=bool)
    lateral = np.asarray(lateral_mask, dtype=bool)
    if full.ndim != 3 or full.shape != lateral.shape:
        raise ValueError(
            f"Component masks must share `(B,Z,W)` shape: {full.shape}/{lateral.shape}."
        )
    disjoint_full = full & ~lateral
    full_projection_raw = (
        project_volume(
            full.astype(np.uint8), projection_kind=projection_kind, layers=layers
        )
        > 0
    )
    lateral_projection = (
        project_volume(
            lateral.astype(np.uint8), projection_kind=projection_kind, layers=layers
        )
        > 0
    )
    full_projection = full_projection_raw & ~lateral_projection
    if np.any(disjoint_full & lateral) or np.any(full_projection & lateral_projection):
        raise RuntimeError("Evaluation supports are not disjoint.")
    return EvaluationSupports(
        disjoint_full, lateral, full_projection, lateral_projection
    )
