"""Adjacent-en-face consistency and 5x5 per-B-scan median diagnostic.

An en-face plane is ``volume[:, z, :]`` with shape ``(B,W)``. The primary
consistency statistic computes Pearson NCC independently for each adjacent
pair of these planes and then averages pair values within a volume. This is
deliberately different from one NCC over two concatenated 3D arrays.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import load_volume, save_volume
from ..core.manifest import CaseRecord, read_manifest, write_manifest
from ..evaluation.metrics import pearson_ncc


def median_filter_bscan_planes(volume: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median-filter each conventional ``(Z,W)`` B-scan independently.

    The corresponding 3D kernel is ``(1,k,k)`` in repository ``(B,Z,W)``
    order, so no samples cross B-scan boundaries.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("B-scan median kernel size must be a positive odd integer.")
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The median diagnostic requires `pip install -e '.[diagnostics]'`."
        ) from exc
    return ndimage.median_filter(
        np.asarray(volume), size=(1, kernel_size, kernel_size), mode="nearest"
    )


def _masked_median_filter_bscan_planes(
    volume: np.ndarray, missing_mask: np.ndarray, kernel_size: int
) -> np.ndarray:
    """Filter observed B-scan pixels without using missing placeholders.

    GT is filtered conventionally. Corrupted input uses this mask-aware form
    so zero-filled missing samples cannot contaminate neighboring observations.
    Missing output values remain exactly as supplied because the missing mask
    is still consumed by the downstream restoration method.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("B-scan median kernel size must be a positive odd integer.")
    source = np.asarray(volume)
    missing = np.asarray(missing_mask, dtype=bool)
    if source.shape != missing.shape:
        raise ValueError(
            "Corrupted volume and missing mask must have identical shapes."
        )
    radius = kernel_size // 2
    filtered = np.empty(source.shape, dtype=np.float64)
    _, z_size, w_size = source.shape
    # Process one B-scan at a time. A full-volume neighborhood stack would be
    # unnecessarily large; this filter never crosses B-scan borders.
    for b_index in range(source.shape[0]):
        plane = source[b_index]
        plane_observed = ~missing[b_index]
        padded_plane = np.pad(plane, ((radius, radius), (radius, radius)), mode="edge")
        padded_observed = np.pad(
            plane_observed, ((radius, radius), (radius, radius)), mode="edge"
        )
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for offset_z in range(kernel_size):
            for offset_w in range(kernel_size):
                values.append(
                    padded_plane[
                        offset_z : offset_z + z_size, offset_w : offset_w + w_size
                    ]
                )
                valid.append(
                    padded_observed[
                        offset_z : offset_z + z_size, offset_w : offset_w + w_size
                    ]
                )
        value_stack = np.stack(values, axis=0).astype(np.float64)
        valid_stack = np.stack(valid, axis=0)
        counts = valid_stack.sum(axis=0)
        ordered = np.sort(np.where(valid_stack, value_stack, np.inf), axis=0)
        lower_index = np.maximum(counts - 1, 0) // 2
        upper_index = counts // 2
        lower = np.take_along_axis(ordered, lower_index[None], axis=0)[0]
        upper = np.take_along_axis(ordered, upper_index[None], axis=0)[0]
        filtered[b_index] = np.where(counts > 0, 0.5 * (lower + upper), plane)
    filtered[missing] = source[missing]
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        return np.clip(np.rint(filtered), limits.min, limits.max).astype(source.dtype)
    return filtered.astype(source.dtype, copy=False)


def prepare_bscan_median_diagnostic(
    *,
    manifest_path: Path,
    output_root: Path,
    kernel_size: int = 5,
    cohort: str | None = "inhouse",
    overwrite: bool = False,
) -> list[CaseRecord]:
    """Create filtered GT/corrupted cases for the median-filter diagnostic."""

    records = read_manifest(manifest_path)
    if cohort is not None:
        records = [record for record in records if record.cohort == cohort]
    if not records:
        raise ValueError(f"Manifest contains no cases for cohort={cohort!r}.")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --overwrite intentionally."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[CaseRecord] = []

    for record in records:
        record.require("gt_path", "corrupted_path", "mask_path")
        gt = load_volume(record.gt_path)  # type: ignore[arg-type]
        corrupted = load_volume(record.corrupted_path)  # type: ignore[arg-type]
        missing = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
        if not (gt.shape == corrupted.shape == missing.shape):
            raise ValueError(
                f"Shape mismatch for median diagnostic case {record.case_id}."
            )
        filtered_gt = median_filter_bscan_planes(gt, kernel_size)
        filtered_corrupted = _masked_median_filter_bscan_planes(
            corrupted, missing, kernel_size
        )
        case_dir = output_root / "cases" / record.case_id
        gt_path = case_dir / "gt_median_bscan.tif"
        corrupted_path = case_dir / "corrupted_median_bscan.tif"
        save_volume(gt_path, filtered_gt, dtype=gt.dtype)
        save_volume(corrupted_path, filtered_corrupted, dtype=corrupted.dtype)
        metadata_path = case_dir / "median_diagnostic_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "case_id": record.case_id,
                    "axis_order": "(B,Z,W)",
                    "filter": "independent 2D median per conventional B-scan",
                    "kernel_bzw": [1, kernel_size, kernel_size],
                    "boundary_mode": "nearest",
                    "gt_filtering": "ordinary median filter",
                    "corrupted_filtering": "observed-only median; missing placeholders excluded",
                    "mask_modified": False,
                    "source_gt_path": str(record.gt_path),
                    "source_corrupted_path": str(record.corrupted_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        generated.append(
            CaseRecord(
                case_id=record.case_id,
                cohort=record.cohort,
                parent_id=record.parent_id,
                gt_path=gt_path,
                corrupted_path=corrupted_path,
                mask_path=record.mask_path,
                full_bscan_mask_path=record.full_bscan_mask_path,
                lateral_mask_path=record.lateral_mask_path,
                corruption_metadata_path=metadata_path,
                projection_kind=record.projection_kind,
                layer_path=record.layer_path,
                axial_offset=record.axial_offset,
            )
        )

    write_manifest(output_root / "manifest.csv", generated)
    (output_root / "diagnostic_config.json").write_text(
        json.dumps(
            {
                "protocol": "bscan_median_diagnostic_v1",
                "source_manifest": str(manifest_path.resolve()),
                "selected_cohort": cohort,
                "kernel_bzw": [1, kernel_size, kernel_size],
                "boundary_mode": "nearest",
                "case_count": len(generated),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return generated


def adjacent_enface_pair_ncc(volume: np.ndarray) -> list[float]:
    """Return NCC for every adjacent pair ``V[:,z,:]`` and ``V[:,z+1,:]``."""

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected (B,Z,W), found {array.shape}.")
    if array.shape[1] < 2:
        raise ValueError("Adjacent-en-face NCC requires at least two Z planes.")
    return [
        pearson_ncc(array[:, z, :], array[:, z + 1, :])
        for z in range(array.shape[1] - 1)
    ]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Cannot write empty diagnostic table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_std(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0 if finite.size == 1 else float("nan")
    return float(finite.std(ddof=1))


def analyze_adjacent_enface_consistency(
    *,
    manifest_path: Path,
    output_dir: Path,
    volume_field: str = "gt_path",
    median_kernel_size: int = 5,
) -> None:
    """Analyze original and median-filtered adjacent en-face consistency."""

    if volume_field not in {"gt_path", "corrupted_path"}:
        raise ValueError("volume_field must be `gt_path` or `corrupted_path`.")
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError("Consistency manifest is empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for record in records:
        record.require(volume_field)
        path = getattr(record, volume_field)
        original_raw = load_volume(path)  # type: ignore[arg-type]
        scale = float(np.max(original_raw))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid volume maximum for {record.case_id}: {scale}.")
        original = original_raw.astype(np.float32) / scale
        filtered = median_filter_bscan_planes(original, median_kernel_size).astype(
            np.float32
        )
        condition_values: dict[str, list[float]] = {
            "original": adjacent_enface_pair_ncc(original),
            f"bscan_median_{median_kernel_size}x{median_kernel_size}": (
                adjacent_enface_pair_ncc(filtered)
            ),
        }
        for condition, values in condition_values.items():
            for z_index, ncc in enumerate(values):
                pair_rows.append(
                    {
                        "case_id": record.case_id,
                        "cohort": record.cohort or "unspecified",
                        "parent_id": record.parent_id or "",
                        "condition": condition,
                        "z_left": z_index,
                        "z_right": z_index + 1,
                        "adjacent_enface_ncc": ncc,
                    }
                )
            case_rows.append(
                {
                    "case_id": record.case_id,
                    "cohort": record.cohort or "unspecified",
                    "parent_id": record.parent_id or "",
                    "condition": condition,
                    "mean_adjacent_enface_ncc": _finite_mean(values),
                    "std_adjacent_enface_ncc": _finite_std(values),
                    "n_adjacent_plane_pairs": len(values),
                    "volume_field": volume_field,
                    "volume_path": str(path),
                }
            )

    unit_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        unit_id = (
            str(row["parent_id"])
            if row["cohort"] == "inhouse" and row["parent_id"]
            else str(row["case_id"])
        )
        unit_groups[(str(row["cohort"]), unit_id, str(row["condition"]))].append(row)
    unit_rows: list[dict[str, Any]] = []
    for (cohort, unit_id, condition), members in sorted(unit_groups.items()):
        unit_rows.append(
            {
                "cohort": cohort,
                "independent_unit_id": unit_id,
                "condition": condition,
                "mean_adjacent_enface_ncc": _finite_mean(
                    [float(row["mean_adjacent_enface_ncc"]) for row in members]
                ),
                "n_source_cases": len(members),
            }
        )

    cohort_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        cohort_groups[(str(row["cohort"]), str(row["condition"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (cohort, condition), members in sorted(cohort_groups.items()):
        values = [float(row["mean_adjacent_enface_ncc"]) for row in members]
        summary_rows.append(
            {
                "cohort": cohort,
                "condition": condition,
                "mean_adjacent_enface_ncc": _finite_mean(values),
                "std_adjacent_enface_ncc": _finite_std(values),
                "n_independent_units": len(values),
            }
        )

    _write_csv(output_dir / "adjacent_enface_pair_ncc.csv", pair_rows)
    _write_csv(output_dir / "per_case_consistency.csv", case_rows)
    _write_csv(output_dir / "per_independent_unit_consistency.csv", unit_rows)
    _write_csv(output_dir / "aggregate_consistency.csv", summary_rows)
    _plot_consistency(unit_rows, summary_rows, output_dir)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "axis_order": "(B,Z,W)",
                "enface_plane": "V[:,z,:] with shape (B,W)",
                "primary_metric": "mean of independently computed Pearson NCC values for adjacent en-face plane pairs",
                "preprocessing": "common positive volume scaling only; NCC is scale invariant",
                "alignment_or_crop": "none",
                "masking": "none; full prepared representation",
                "volume_field": volume_field,
                "median_diagnostic_kernel_bzw": [
                    1,
                    median_kernel_size,
                    median_kernel_size,
                ],
                "median_boundary_mode": "nearest",
                "aggregation": "pair mean within case; in-house subvolumes mean within parent; independent-unit cohort mean/std",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_consistency(
    unit_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Diagnostic plotting requires `pip install -e '.[diagnostics]'`."
        ) from exc

    conditions = ["original"] + sorted(
        {
            str(row["condition"])
            for row in summary_rows
            if row["condition"] != "original"
        }
    )
    cohorts = sorted({str(row["cohort"]) for row in summary_rows})
    figure, axes = plt.subplots(
        1,
        len(cohorts),
        figsize=(max(3.45, 2.25 * len(cohorts)), 2.55),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, cohort in zip(axes[0], cohorts):
        cohort_units = [row for row in unit_rows if row["cohort"] == cohort]
        by_unit: dict[str, dict[str, float]] = defaultdict(dict)
        for row in cohort_units:
            by_unit[str(row["independent_unit_id"])][str(row["condition"])] = float(
                row["mean_adjacent_enface_ncc"]
            )
        for values in by_unit.values():
            if all(condition in values for condition in conditions):
                axis.plot(
                    np.arange(len(conditions)),
                    [values[condition] for condition in conditions],
                    color="#A8A8A8",
                    linewidth=0.7,
                    alpha=0.65,
                    zorder=1,
                )
        summary = {
            str(row["condition"]): row
            for row in summary_rows
            if row["cohort"] == cohort
        }
        means = [
            float(summary[condition]["mean_adjacent_enface_ncc"])
            for condition in conditions
        ]
        stds = [
            float(summary[condition]["std_adjacent_enface_ncc"])
            for condition in conditions
        ]
        axis.errorbar(
            np.arange(len(conditions)),
            means,
            yerr=stds,
            color="#0072B2",
            marker="o",
            linewidth=1.5,
            capsize=2,
            zorder=2,
        )
        axis.set_title(cohort.replace("_", " "), fontsize=8.5)
        condition_labels = [
            "Original"
            if condition == "original"
            else condition.replace("bscan_", "").replace("_", " ")
            for condition in conditions
        ]
        axis.set_xticks(np.arange(len(conditions)), condition_labels, fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean adjacent en-face NCC", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"adjacent_enface_consistency.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(figure)
