"""One shared evaluation protocol for every reconstruction method."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .projection import EvaluationSupports, LayerSurfaces, project_volume


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float | int | str]
    gt_projection: np.ndarray
    prediction_projection: np.ndarray


_LPIPS_MODEL: Any | None = None


def normalize_by_gt_max(
    gt: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply evaluation-only GT-maximum normalization."""

    gt_values = np.asarray(gt, dtype=np.float32)
    prediction_values = np.asarray(prediction, dtype=np.float32)
    if gt_values.shape != prediction_values.shape:
        raise ValueError(
            f"GT/prediction shape mismatch: {gt_values.shape}/{prediction_values.shape}."
        )
    scale = float(np.max(gt_values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid GT maximum: {scale!r}.")
    return gt_values / scale, prediction_values / scale, scale


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    grad_0 = np.diff(image, axis=0, append=image[-1:, :])
    grad_1 = np.diff(image, axis=1, append=image[:, -1:])
    return np.sqrt(grad_0**2 + grad_1**2)


def pearson_ncc(reference: np.ndarray, prediction: np.ndarray) -> float:
    first = np.asarray(reference, dtype=np.float64).ravel()
    second = np.asarray(prediction, dtype=np.float64).ravel()
    if first.size == 0:
        return float("nan")
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.sqrt(np.sum(first**2) * np.sum(second**2)))
    if denominator < 1e-12:
        return 1.0 if np.allclose(first, second, atol=1e-8) else 0.0
    return float(np.sum(first * second) / denominator)


def _lpips_model() -> Any:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LPIPS evaluation requires `pip install -e '.[evaluation]'`."
            ) from exc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _LPIPS_MODEL = lpips.LPIPS(net="alex").eval()
        if torch.cuda.is_available():
            _LPIPS_MODEL = _LPIPS_MODEL.cuda()
    return _LPIPS_MODEL


def _lpips_value(reference: np.ndarray, prediction: np.ndarray) -> float:
    import torch

    def tensor(image: np.ndarray) -> Any:
        array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        value = torch.from_numpy(np.ascontiguousarray(array))[None, None]
        return value.repeat(1, 3, 1, 1) * 2.0 - 1.0

    model = _lpips_model()
    device = next(model.parameters()).device
    with torch.no_grad():
        return float(
            model(tensor(reference).to(device), tensor(prediction).to(device)).item()
        )


def _masked_region_metric(
    reference: np.ndarray,
    prediction: np.ndarray,
    support: np.ndarray,
) -> tuple[float, float]:
    selected = np.asarray(support, dtype=bool)
    if not np.any(selected):
        return float("nan"), float("nan")
    return (
        float(np.mean(np.abs(prediction[selected] - reference[selected]))),
        pearson_ncc(reference[selected], prediction[selected]),
    )


def evaluate_reconstruction(
    *,
    gt: np.ndarray,
    prediction: np.ndarray,
    supports: EvaluationSupports,
    projection_kind: str,
    layers: LayerSurfaces | None,
    compute_lpips: bool = True,
) -> EvaluationResult:
    """Evaluate one case under the shared GT-max and disjoint-support protocol."""

    gt_01, prediction_01, gt_max = normalize_by_gt_max(gt, prediction)
    if supports.full_bscan_voxels.shape != gt_01.shape:
        raise ValueError(
            "Evaluation support shape does not match reconstruction volume."
        )

    grad_values: list[float] = []
    lpips_values: list[float] = []
    bscan_ncc_values: list[float] = []
    selected_rows = np.flatnonzero(np.any(supports.full_bscan_voxels, axis=(1, 2)))
    for b in selected_rows:
        support = supports.full_bscan_voxels[b]
        region_prediction = np.where(support, prediction_01[b], gt_01[b])
        grad_values.append(
            float(
                np.mean(
                    np.abs(
                        gradient_magnitude(region_prediction)
                        - gradient_magnitude(gt_01[b])
                    )
                )
            )
        )
        bscan_ncc_values.append(
            pearson_ncc(gt_01[b][support], prediction_01[b][support])
        )
        if compute_lpips:
            lpips_values.append(_lpips_value(gt_01[b], region_prediction))

    gt_projection = project_volume(
        gt_01, projection_kind=projection_kind, layers=layers
    )
    prediction_projection = project_volume(
        prediction_01, projection_kind=projection_kind, layers=layers
    )
    full_l1, full_ncc = _masked_region_metric(
        gt_projection,
        prediction_projection,
        supports.full_bscan_projection,
    )
    lateral_l1, lateral_ncc = _masked_region_metric(
        gt_projection,
        prediction_projection,
        supports.lateral_projection,
    )
    metrics: dict[str, float | int | str] = {
        "evaluation_normalization": "gt_max",
        "gt_max": gt_max,
        "evaluated_full_bscan_rows": int(selected_rows.size),
        "bscan_grad_l1": float(np.mean(grad_values)) if grad_values else float("nan"),
        "bscan_lpips": float(np.mean(lpips_values)) if lpips_values else float("nan"),
        "bscan_ncc": float(np.mean(bscan_ncc_values))
        if bscan_ncc_values
        else float("nan"),
        "full_bscan_mip_pixels": int(supports.full_bscan_projection.sum()),
        "full_bscan_mip_l1": full_l1,
        "full_bscan_mip_ncc": full_ncc,
        "lateral_mip_pixels": int(supports.lateral_projection.sum()),
        "lateral_mip_l1": lateral_l1,
        "lateral_mip_ncc": lateral_ncc,
    }
    return EvaluationResult(metrics, gt_projection, prediction_projection)
