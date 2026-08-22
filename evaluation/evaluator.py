"""Manifest-driven, method-agnostic evaluation and hierarchy-aware aggregation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..core.io import load_volume
from ..core.manifest import CaseRecord
from ..core.reconstruction import compose_missing_only
from .metrics import evaluate_reconstruction
from .projection import (
    LayerSurfaces,
    build_disjoint_supports,
    load_layer_surfaces,
    project_volume,
)

METRIC_COLUMNS = (
    "bscan_grad_l1",
    "bscan_lpips",
    "bscan_ncc",
    "full_bscan_mip_l1",
    "full_bscan_mip_ncc",
    "lateral_mip_l1",
    "lateral_mip_ncc",
)


def _to_u8(image: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(np.asarray(image, dtype=np.float32) / max(scale, 1e-8), 0, 1)
    return np.rint(normalized * 255).astype(np.uint8)


def write_projection_panel(
    *,
    path: Path,
    gt_projection: np.ndarray,
    corrupted_projection: np.ndarray,
    prediction_projection: np.ndarray,
    full_support: np.ndarray,
    lateral_support: np.ndarray,
) -> None:
    """Write consistently scaled GT, corrupted, reconstruction, and support MIPs."""

    positive = np.asarray(gt_projection)[np.asarray(gt_projection) > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    labels = ["GT", "Corrupted", "Reconstruction", "Full support", "Lateral support"]
    images = [
        _to_u8(gt_projection, scale),
        _to_u8(corrupted_projection, scale),
        _to_u8(prediction_projection, scale),
        np.where(full_support, 255, 0).astype(np.uint8),
        np.where(lateral_support, 255, 0).astype(np.uint8),
    ]
    width = max(image.shape[1] for image in images)
    height = max(image.shape[0] for image in images)
    header = 24
    canvas = Image.new("L", (width * len(images), height + header), color=0)
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(Image.fromarray(image), (index * width, header))
        draw.text((index * width + 4, 5), label, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def _layers_for_record(record: CaseRecord) -> LayerSurfaces | None:
    if record.projection_kind == "full_axial":
        return None
    if record.projection_kind != "ilm_opl":
        raise ValueError(
            f"Case `{record.case_id}` requires projection_kind `full_axial` or `ilm_opl`."
        )
    record.require("layer_path")
    layers = load_layer_surfaces(record.layer_path)  # type: ignore[arg-type]
    return layers.shifted(record.axial_offset or 0)


def evaluate_cases(
    *,
    records: list[CaseRecord],
    predictions: dict[str, Path],
    output_dir: Path,
    compute_lpips: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all cases and return case, independent-unit, and cohort rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    for record in records:
        record.require(
            "gt_path",
            "corrupted_path",
            "mask_path",
            "full_bscan_mask_path",
            "lateral_mask_path",
            "projection_kind",
        )
        if record.case_id not in predictions:
            raise KeyError(f"No prediction was provided for `{record.case_id}`.")
        gt = load_volume(record.gt_path)  # type: ignore[arg-type]
        corrupted = load_volume(record.corrupted_path)  # type: ignore[arg-type]
        combined_mask = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
        full_mask = load_volume(record.full_bscan_mask_path) != 0  # type: ignore[arg-type]
        lateral_mask = load_volume(record.lateral_mask_path) != 0  # type: ignore[arg-type]
        prediction_raw = load_volume(predictions[record.case_id])
        if not (
            gt.shape == corrupted.shape == combined_mask.shape == prediction_raw.shape
        ):
            raise ValueError(f"Shape mismatch while evaluating `{record.case_id}`.")
        if not np.array_equal(combined_mask, full_mask | lateral_mask):
            raise ValueError(
                f"Combined mask does not equal component-mask union for `{record.case_id}`."
            )

        # Common composition prevents method-specific observed values from
        # affecting metrics and makes the evaluator safe for raw predictions.
        prediction = compose_missing_only(corrupted, prediction_raw, combined_mask)
        observed_before = float(
            np.max(
                np.abs(
                    prediction_raw.astype(np.float32)[~combined_mask]
                    - corrupted.astype(np.float32)[~combined_mask]
                )
            )
        )
        layers = _layers_for_record(record)
        supports = build_disjoint_supports(
            full_bscan_mask=full_mask,
            lateral_mask=lateral_mask,
            projection_kind=record.projection_kind,  # type: ignore[arg-type]
            layers=layers,
        )
        evaluated = evaluate_reconstruction(
            gt=gt,
            prediction=prediction,
            supports=supports,
            projection_kind=record.projection_kind,  # type: ignore[arg-type]
            layers=layers,
            compute_lpips=compute_lpips,
        )
        gt_max = float(evaluated.metrics["gt_max"])
        corrupted_projection = project_volume(
            corrupted.astype(np.float32) / gt_max,
            projection_kind=record.projection_kind,  # type: ignore[arg-type]
            layers=layers,
        )
        write_projection_panel(
            path=output_dir / "visuals" / f"{record.case_id}_projection_panel.png",
            gt_projection=evaluated.gt_projection,
            corrupted_projection=corrupted_projection,
            prediction_projection=evaluated.prediction_projection,
            full_support=supports.full_bscan_projection,
            lateral_support=supports.lateral_projection,
        )
        unit_id = (
            record.parent_id
            if record.cohort == "inhouse" and record.parent_id
            else record.case_id
        )
        row: dict[str, Any] = {
            "case_id": record.case_id,
            "cohort": record.cohort or "unspecified",
            "parent_id": record.parent_id or "",
            "independent_unit_id": unit_id,
            "prediction_path": str(predictions[record.case_id]),
            "raw_prediction_observed_max_abs_error": observed_before,
            "evaluated_prediction_observed_max_abs_error": 0.0,
            **evaluated.metrics,
        }
        case_rows.append(row)

    unit_rows = _aggregate_by_keys(case_rows, ("cohort", "independent_unit_id"))
    cohort_rows = _summarize_cohorts(unit_rows)
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    _write_csv(output_dir / "per_independent_unit_metrics.csv", unit_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", cohort_rows)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "evaluation_normalization": "GT maximum per case",
                "bscan_support": "full-B-scan component minus lateral overlap",
                "projection_support": "project first; lateral assigned first; full excludes lateral",
                "inhouse_aggregation": "mean subvolumes within parent acquisition, then summarize parents",
                "octa500_aggregation": "independent volumes",
                "lpips_enabled": compute_lpips,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return case_rows, unit_rows, cohort_rows


def _aggregate_by_keys(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, members in sorted(grouped.items()):
        result: dict[str, Any] = dict(zip(keys, key_values))
        result["case_count"] = len(members)
        for metric in METRIC_COLUMNS:
            values = np.asarray(
                [float(member[metric]) for member in members], dtype=np.float64
            )
            result[metric] = _finite_mean(values)
        output.append(result)
    return output


def _summarize_cohorts(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        grouped[str(row["cohort"])].append(row)
    output: list[dict[str, Any]] = []
    for cohort, members in sorted(grouped.items()):
        result: dict[str, Any] = {
            "cohort": cohort,
            "independent_unit_count": len(members),
        }
        for metric in METRIC_COLUMNS:
            values = np.asarray(
                [float(member[metric]) for member in members], dtype=np.float64
            )
            result[f"{metric}_mean"] = _finite_mean(values)
            result[f"{metric}_std"] = _finite_std(values)
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
