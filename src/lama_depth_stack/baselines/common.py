"""Shared, leakage-safe infrastructure for learned OCTA baselines.

The public baselines consume volumes in ``(B,Z,W)`` order. ``True`` mask
values are missing. Model-facing intensities use a per-volume scale fitted only
to positive observed values in the corrupted input; clean targets never
participate in normalization.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.artifacts import write_case_artifacts
from ..core.io import load_volume
from ..core.manifest import CaseRecord
from ..core.normalization import ObservedPositiveScale, fit_observed_positive_scale
from ..core.reconstruction import (
    CanonicalConfig,
    ReconstructionResult,
    compose_missing_only,
)


@dataclass(frozen=True)
class PreparedCase:
    """One normalized case used by a learned baseline.

    ``clean_01`` is populated only by supervised training code. Prediction
    paths call :func:`prepare_case` with ``require_clean=False`` and therefore
    cannot expose a clean target to a model input constructor.
    """

    record: CaseRecord
    corrupted_native: np.ndarray
    missing_mask: np.ndarray
    corrupted_01: np.ndarray
    scale: ObservedPositiveScale
    clean_01: np.ndarray | None = None


def require_torch() -> Any:
    """Import PyTorch lazily so reconstruction-only installs remain lightweight."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Learned baselines require PyTorch. Install `lama-depth-stack-octa[baselines]`."
        ) from exc
    return torch


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing deterministic CUDA kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_case(record: CaseRecord, *, require_clean: bool) -> PreparedCase:
    """Load and normalize a case without using GT to determine intensity scale."""

    required = ["corrupted_path", "mask_path"]
    if require_clean:
        required.append("gt_path")
    record.require(*required)

    corrupted = load_volume(record.corrupted_path)  # type: ignore[arg-type]
    missing = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
    if corrupted.shape != missing.shape:
        raise ValueError(f"Case `{record.case_id}` volume/mask shapes do not match.")
    scale = fit_observed_positive_scale(corrupted, missing, percentile=99.9)
    clean_01: np.ndarray | None = None
    if require_clean:
        clean = load_volume(record.gt_path)  # type: ignore[arg-type]
        if clean.shape != corrupted.shape:
            raise ValueError(
                f"Case `{record.case_id}` clean/corrupted shapes do not match."
            )
        clean_01 = scale.apply(clean)
    return PreparedCase(
        record=record,
        corrupted_native=corrupted,
        missing_mask=missing,
        corrupted_01=scale.apply(corrupted),
        scale=scale,
        clean_01=clean_01,
    )


def edge_padded_stack(volume: np.ndarray, center: int, size: int) -> np.ndarray:
    """Return an edge-padded odd B-scan window as ``(S,Z,W)``."""

    if size < 1 or size % 2 != 1:
        raise ValueError("B-scan window size must be a positive odd integer.")
    pad = size // 2
    padded = np.pad(np.asarray(volume), ((pad, pad), (0, 0), (0, 0)), mode="edge")
    return padded[int(center) : int(center) + size]


def affected_bscans(missing_mask: np.ndarray) -> np.ndarray:
    """Indices of B-scans containing at least one missing voxel."""

    return np.flatnonzero(np.any(np.asarray(missing_mask, dtype=bool), axis=(1, 2)))


def pad_bottom_right(
    array: np.ndarray, multiple: int = 16
) -> tuple[np.ndarray, tuple[int, int]]:
    """Edge-pad the last two dimensions to a model-compatible multiple."""

    z, w = array.shape[-2:]
    pad_z = (multiple - z % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_z == 0 and pad_w == 0:
        return array, (0, 0)
    padding = [(0, 0)] * array.ndim
    padding[-2] = (0, pad_z)
    padding[-1] = (0, pad_w)
    return np.pad(array, padding, mode="edge"), (pad_z, pad_w)


def crop_bottom_right(array: np.ndarray, crop: tuple[int, int]) -> np.ndarray:
    """Undo :func:`pad_bottom_right` on the final two dimensions."""

    pad_z, pad_w = crop
    z_slice = slice(None, -pad_z) if pad_z else slice(None)
    w_slice = slice(None, -pad_w) if pad_w else slice(None)
    return array[..., z_slice, w_slice]


def save_checkpoint(
    path: Path,
    *,
    method: str,
    model: Any,
    optimizer: Any,
    epoch: int,
    config: dict[str, Any],
    history: list[dict[str, float | int]],
    best_validation_loss: float,
) -> None:
    """Write a structured, reloadable training checkpoint."""

    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "method": method,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "history": history,
            "best_validation_loss": float(best_validation_loss),
        },
        path,
    )


def load_checkpoint(path: Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    """Load and minimally validate a public baseline checkpoint."""

    torch = require_torch()
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint `{path}` does not contain a mapping.")
    required = {"format_version", "method", "model_state", "config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint `{path}` is missing keys: {sorted(missing)}.")
    return payload


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Persist epoch-level history as CSV without requiring pandas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_prediction_artifacts(
    *,
    case: PreparedCase,
    raw_prediction_01: np.ndarray,
    output_root: Path,
    method: str,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> Path:
    """Invert model scaling, enforce paste-back, and save standard artifacts."""

    raw_native = case.scale.invert(np.clip(raw_prediction_01, 0.0, 1.0))
    final_native = compose_missing_only(
        case.corrupted_native,
        raw_native,
        case.missing_mask,
    )
    observed_error = float(
        np.max(
            np.abs(
                final_native[~case.missing_mask]
                - case.corrupted_native[~case.missing_mask]
            )
        )
    )
    if observed_error != 0.0:
        raise RuntimeError(
            f"Observed-voxel invariance failed: max error={observed_error}."
        )
    result = ReconstructionResult(
        raw_prediction=raw_native,
        final_reconstruction=final_native,
        model_input=case.corrupted_01,
        scale=case.scale,
        metadata={
            "method": method,
            "axis_order": "BZW",
            "mask_true_means": "missing",
            "normalization": case.scale.to_dict(),
            "final_reconstruction": "where(mask, prediction, corrupted)",
            "observed_max_abs_error": observed_error,
            "gt_required_for_inference": False,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "config": config,
        },
    )
    output_dir = output_root / case.record.case_id
    write_case_artifacts(
        output_dir=output_dir,
        case_id=case.record.case_id,
        corrupted=case.corrupted_native,
        missing_mask=case.missing_mask,
        result=result,
        source_dtype=case.corrupted_native.dtype,
        config=CanonicalConfig(),
        extra_metadata={
            "cohort": case.record.cohort,
            "parent_id": case.record.parent_id,
        },
    )
    return output_dir / "volumes" / "final_reconstruction.tif"


def config_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass config to a JSON-ready dictionary."""

    return asdict(config)
