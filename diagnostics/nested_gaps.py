"""Manifest-driven nested consecutive-missing-B-scan diagnostic.

The diagnostic creates one centered full-B-scan gap per condition. For a
fixed source case and center, larger conditions strictly contain the smaller
ones. Arrays follow the repository convention ``(B,Z,W)`` and masks use
``True`` for missing samples.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import load_volume, save_volume
from ..core.manifest import (
    CaseRecord,
    read_manifest,
    resolve_predictions,
    write_manifest,
)
from ..core.reconstruction import compose_missing_only
from ..evaluation.metrics import pearson_ncc

DEFAULT_GAP_LENGTHS = (1, 2, 4, 6, 9, 12)
DEFAULT_CENTER_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class NestedCondition:
    """Metadata for one source-case, center, and gap-length condition."""

    condition_id: str
    source_case_id: str
    cohort: str
    parent_id: str
    center_id: str
    requested_center_fraction: float
    anchor_b: int
    gap_length: int
    start_b: int
    end_b_exclusive: int
    shape_b: int
    shape_z: int
    shape_w: int


def nested_bscan_indices(anchor_b: int, gap_length: int) -> np.ndarray:
    """Return a deterministic interval nested around ``anchor_b``.

    For even lengths, the interval gains its first extra row on the positive
    side. This preserves the same anatomical anchor at every gap length.
    """

    if gap_length < 1:
        raise ValueError("Gap length must be positive.")
    start = int(anchor_b) - (int(gap_length) - 1) // 2
    return np.arange(start, start + int(gap_length), dtype=np.int64)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Cannot write empty diagnostic table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _read_conditions(path: Path) -> list[NestedCondition]:
    rows: list[NestedCondition] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            field.name for field in NestedCondition.__dataclass_fields__.values()
        }
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Nested condition columns must be exactly {sorted(expected)}, "
                f"found {sorted(reader.fieldnames or [])}."
            )
        for row in reader:
            rows.append(
                NestedCondition(
                    condition_id=row["condition_id"],
                    source_case_id=row["source_case_id"],
                    cohort=row["cohort"],
                    parent_id=row["parent_id"],
                    center_id=row["center_id"],
                    requested_center_fraction=float(row["requested_center_fraction"]),
                    anchor_b=int(row["anchor_b"]),
                    gap_length=int(row["gap_length"]),
                    start_b=int(row["start_b"]),
                    end_b_exclusive=int(row["end_b_exclusive"]),
                    shape_b=int(row["shape_b"]),
                    shape_z=int(row["shape_z"]),
                    shape_w=int(row["shape_w"]),
                )
            )
    return rows


def _center_definitions(
    records: Sequence[CaseRecord],
    *,
    center_fractions: Sequence[float],
    centers_csv: Path | None,
) -> dict[str, list[tuple[str, float, int | None]]]:
    """Resolve generic fractions or explicit case-specific B-scan anchors."""

    if centers_csv is None:
        if not center_fractions:
            raise ValueError("At least one center fraction is required.")
        if any(not 0.0 < value < 1.0 for value in center_fractions):
            raise ValueError("Center fractions must be strictly between zero and one.")
        return {
            record.case_id: [
                (f"c{index}", float(fraction), None)
                for index, fraction in enumerate(center_fractions, start=1)
            ]
            for record in records
        }

    resolved: dict[str, list[tuple[str, float, int | None]]] = defaultdict(list)
    with centers_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "center_id", "anchor_b"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Centers CSV requires case_id,center_id,anchor_b columns.")
        for row in reader:
            resolved[row["case_id"].strip()].append(
                (row["center_id"].strip(), float("nan"), int(row["anchor_b"]))
            )
    expected_ids = {record.case_id for record in records}
    if set(resolved) != expected_ids:
        raise ValueError(
            "Centers CSV case IDs do not match the selected manifest cases: "
            f"missing={sorted(expected_ids - set(resolved))}, "
            f"unexpected={sorted(set(resolved) - expected_ids)}."
        )
    for case_id, centers in resolved.items():
        center_ids = [center[0] for center in centers]
        if len(center_ids) != len(set(center_ids)):
            raise ValueError(f"Duplicate center_id for {case_id}.")
    return dict(resolved)


def prepare_nested_gaps(
    *,
    manifest_path: Path,
    output_root: Path,
    lengths: Sequence[int] = DEFAULT_GAP_LENGTHS,
    center_fractions: Sequence[float] = DEFAULT_CENTER_FRACTIONS,
    centers_csv: Path | None = None,
    cohort: str = "octa500_3mm",
    overwrite: bool = False,
) -> tuple[list[CaseRecord], list[NestedCondition]]:
    """Create a reusable nested-gap dataset without copying source GT volumes."""

    lengths = tuple(sorted({int(length) for length in lengths}))
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Nested gap lengths must be positive integers.")
    records = [
        record for record in read_manifest(manifest_path) if record.cohort == cohort
    ]
    if not records:
        raise ValueError(f"Manifest contains no cases in cohort `{cohort}`.")
    for record in records:
        record.require("gt_path", "projection_kind")

    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --overwrite intentionally."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    centers = _center_definitions(
        records,
        center_fractions=center_fractions,
        centers_csv=centers_csv,
    )
    largest = max(lengths)
    generated_records: list[CaseRecord] = []
    conditions: list[NestedCondition] = []
    missing_rows: list[dict[str, Any]] = []

    for record in records:
        gt = load_volume(record.gt_path)  # type: ignore[arg-type]
        b_size, z_size, w_size = (int(value) for value in gt.shape)
        for center_id, fraction, explicit_anchor in centers[record.case_id]:
            anchor = (
                int(explicit_anchor)
                if explicit_anchor is not None
                else round(float(fraction) * (b_size - 1))
            )
            # Keep one observed row on both sides of the largest gap.
            minimum_anchor = (largest - 1) // 2 + 1
            maximum_anchor = b_size - 1 - (largest // 2 + 1)
            if explicit_anchor is None:
                anchor = min(max(anchor, minimum_anchor), maximum_anchor)
            elif not minimum_anchor <= anchor <= maximum_anchor:
                raise ValueError(
                    f"Explicit anchor {anchor} for {record.case_id}/{center_id} cannot "
                    f"support L={largest} with observed boundary rows."
                )

            previous: set[int] = set()
            for length in lengths:
                indices = nested_bscan_indices(anchor, length)
                current = {int(value) for value in indices}
                if previous and not previous.issubset(current):
                    raise AssertionError(
                        "Nested gap construction lost subset containment."
                    )
                previous = current

                condition_id = (
                    f"{record.case_id}__nested_{center_id}_b{anchor:04d}_L{length:02d}"
                )
                case_dir = output_root / "conditions" / condition_id
                case_dir.mkdir(parents=True, exist_ok=True)
                full_mask = np.zeros(gt.shape, dtype=bool)
                full_mask[indices] = True
                corrupted = np.array(gt, copy=True)
                corrupted[full_mask] = 0
                lateral_mask = np.zeros(gt.shape, dtype=np.uint8)
                corrupted_path = case_dir / "corrupted.tif"
                mask_path = case_dir / "missing_mask.tif"
                lateral_path = case_dir / "lateral_mask.tif"
                save_volume(corrupted_path, corrupted, dtype=gt.dtype)
                save_volume(mask_path, full_mask, dtype=np.uint8)
                save_volume(lateral_path, lateral_mask, dtype=np.uint8)

                requested_fraction = float(fraction)
                condition = NestedCondition(
                    condition_id=condition_id,
                    source_case_id=record.case_id,
                    cohort=record.cohort or "unspecified",
                    parent_id=record.parent_id or "",
                    center_id=center_id,
                    requested_center_fraction=requested_fraction,
                    anchor_b=anchor,
                    gap_length=length,
                    start_b=int(indices[0]),
                    end_b_exclusive=int(indices[-1] + 1),
                    shape_b=b_size,
                    shape_z=z_size,
                    shape_w=w_size,
                )
                conditions.append(condition)
                metadata_path = case_dir / "corruption_metadata.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "condition_id": condition_id,
                            "source_case_id": record.case_id,
                            "axis_order": "(B,Z,W)",
                            "mask_polarity": "1/True means missing",
                            "corruption": "one centered consecutive full-B-scan gap",
                            "center_id": center_id,
                            "requested_center_fraction": requested_fraction,
                            "anchor_b": anchor,
                            "gap_length": length,
                            "missing_bscan_indices": indices.tolist(),
                            "lateral_corruption_included": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                generated_records.append(
                    CaseRecord(
                        case_id=condition_id,
                        cohort=record.cohort,
                        parent_id=record.parent_id,
                        gt_path=record.gt_path,
                        corrupted_path=corrupted_path,
                        mask_path=mask_path,
                        full_bscan_mask_path=mask_path,
                        lateral_mask_path=lateral_path,
                        corruption_metadata_path=metadata_path,
                        projection_kind=record.projection_kind,
                        layer_path=record.layer_path,
                        axial_offset=record.axial_offset,
                    )
                )
                for position, b_index in enumerate(indices):
                    missing_rows.append(
                        {
                            "condition_id": condition_id,
                            "source_case_id": record.case_id,
                            "center_id": center_id,
                            "gap_length": length,
                            "b_index": int(b_index),
                            "position_in_gap": position,
                            "distance_to_nearest_observed_bscan": min(
                                position + 1, length - position
                            ),
                        }
                    )

    write_manifest(output_root / "manifest.csv", generated_records)
    _write_csv(
        output_root / "nested_conditions.csv", [vars(item) for item in conditions]
    )
    _write_csv(output_root / "missing_bscans.csv", missing_rows)
    (output_root / "diagnostic_config.json").write_text(
        json.dumps(
            {
                "protocol": "nested_gap_octa500_3mm_v1",
                "source_manifest": str(manifest_path.resolve()),
                "selected_cohort": cohort,
                "gap_lengths": list(lengths),
                "center_source": str(centers_csv.resolve())
                if centers_csv
                else "fractions",
                "center_fractions": list(center_fractions)
                if centers_csv is None
                else None,
                "condition_count": len(conditions),
                "gt_storage": "referenced from source manifest; not duplicated",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return generated_records, conditions


def _parse_prediction_specs(
    specifications: Sequence[str], records: Sequence[CaseRecord]
) -> dict[str, dict[str, Path]]:
    methods: dict[str, dict[str, Path]] = {}
    case_ids = [record.case_id for record in records]
    for value in specifications:
        if "=" not in value:
            raise ValueError("Prediction specifications use METHOD=PATH syntax.")
        method, raw_path = value.split("=", 1)
        method = method.strip()
        if not method or method in methods:
            raise ValueError(f"Invalid or duplicate method name in `{value}`.")
        methods[method] = resolve_predictions(Path(raw_path).expanduser(), case_ids)
    if not methods:
        raise ValueError(
            "At least one METHOD=PATH prediction specification is required."
        )
    return methods


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


def _bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def analyze_nested_predictions(
    *,
    manifest_path: Path,
    conditions_path: Path,
    prediction_specs: Sequence[str],
    output_dir: Path,
    uncertainty: str = "ci95",
    bootstrap_samples: int = 10_000,
    seed: int = 2027,
) -> None:
    """Compute NCC trajectories from already reconstructed conditions."""

    records = read_manifest(manifest_path)
    records_by_id = {record.case_id: record for record in records}
    conditions = _read_conditions(conditions_path)
    if {condition.condition_id for condition in conditions} != set(records_by_id):
        raise ValueError(
            "Condition table and nested manifest contain different condition IDs."
        )
    methods = _parse_prediction_specs(prediction_specs, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    bscan_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for condition in conditions:
        record = records_by_id[condition.condition_id]
        record.require("gt_path", "corrupted_path", "mask_path", "full_bscan_mask_path")
        gt_raw = load_volume(record.gt_path)  # type: ignore[arg-type]
        corrupted_raw = load_volume(record.corrupted_path)  # type: ignore[arg-type]
        missing = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
        full_mask = load_volume(record.full_bscan_mask_path) != 0  # type: ignore[arg-type]
        if not (
            gt_raw.shape == corrupted_raw.shape == missing.shape == full_mask.shape
        ):
            raise ValueError(
                f"Shape mismatch for nested condition {condition.condition_id}."
            )
        selected_bscans = np.flatnonzero(np.all(full_mask, axis=(1, 2)))
        expected = np.arange(condition.start_b, condition.end_b_exclusive)
        if not np.array_equal(selected_bscans, expected):
            raise ValueError(
                f"Full-B-scan support disagrees with condition metadata for "
                f"{condition.condition_id}."
            )
        gt_max = float(np.max(gt_raw))
        if not np.isfinite(gt_max) or gt_max <= 0:
            raise ValueError(
                f"Invalid GT maximum for {condition.condition_id}: {gt_max}."
            )
        gt = gt_raw.astype(np.float32) / gt_max
        corrupted = corrupted_raw.astype(np.float32) / gt_max

        for method, paths in methods.items():
            raw_prediction = load_volume(paths[condition.condition_id])
            if raw_prediction.shape != gt_raw.shape:
                raise ValueError(
                    f"Prediction shape mismatch for {method}/{condition.condition_id}."
                )
            raw_prediction_01 = raw_prediction.astype(np.float32) / gt_max
            prediction = compose_missing_only(corrupted, raw_prediction_01, missing)
            observed_error = float(
                np.max(np.abs(prediction[~missing] - corrupted[~missing]))
            )
            values: list[float] = []
            for b_index in selected_bscans:
                ncc = pearson_ncc(gt[b_index], prediction[b_index])
                values.append(ncc)
                bscan_rows.append(
                    {
                        "method": method,
                        "condition_id": condition.condition_id,
                        "source_case_id": condition.source_case_id,
                        "cohort": condition.cohort,
                        "parent_id": condition.parent_id,
                        "center_id": condition.center_id,
                        "gap_length": condition.gap_length,
                        "b_index": int(b_index),
                        "distance_to_nearest_observed_bscan": min(
                            int(b_index - condition.start_b + 1),
                            int(condition.end_b_exclusive - b_index),
                        ),
                        "bscan_ncc": ncc,
                    }
                )
            condition_rows.append(
                {
                    "method": method,
                    "condition_id": condition.condition_id,
                    "source_case_id": condition.source_case_id,
                    "cohort": condition.cohort,
                    "parent_id": condition.parent_id,
                    "center_id": condition.center_id,
                    "gap_length": condition.gap_length,
                    "mean_reconstructed_bscan_ncc": _finite_mean(values),
                    "n_missing_bscans": len(values),
                    "gt_max": gt_max,
                    "observed_pasteback_max_abs_error": observed_error,
                    "prediction_path": str(paths[condition.condition_id]),
                }
            )

    # First average anatomical centers within a source volume. These source
    # volumes, not centers or individual missing slices, are the sample units.
    source_groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in condition_rows:
        key = (
            str(row["method"]),
            str(row["source_case_id"]),
            str(row["cohort"]),
            str(row["parent_id"]),
            int(row["gap_length"]),
        )
        source_groups[key].append(row)
    source_rows: list[dict[str, Any]] = []
    for key, members in sorted(source_groups.items()):
        method, source_case_id, cohort, parent_id, gap_length = key
        source_rows.append(
            {
                "method": method,
                "source_case_id": source_case_id,
                "cohort": cohort,
                "parent_id": parent_id,
                "gap_length": gap_length,
                "mean_reconstructed_bscan_ncc": _finite_mean(
                    [float(item["mean_reconstructed_bscan_ncc"]) for item in members]
                ),
                "n_centers": len({str(item["center_id"]) for item in members}),
            }
        )

    # Retain the shared hierarchy rule if this utility is deliberately reused
    # outside the default OCTA-500 3 mm diagnostic.
    unit_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in source_rows:
        independent_id = (
            str(row["parent_id"])
            if row["cohort"] == "inhouse" and row["parent_id"]
            else str(row["source_case_id"])
        )
        unit_groups[
            (
                str(row["method"]),
                str(row["cohort"]),
                independent_id,
                int(row["gap_length"]),
            )
        ].append(row)
    unit_rows: list[dict[str, Any]] = []
    for (method, cohort, unit_id, gap_length), members in sorted(unit_groups.items()):
        unit_rows.append(
            {
                "method": method,
                "independent_unit_id": unit_id,
                "cohort": cohort,
                "gap_length": gap_length,
                "mean_reconstructed_bscan_ncc": _finite_mean(
                    [float(item["mean_reconstructed_bscan_ncc"]) for item in members]
                ),
                "n_source_volumes": len(members),
            }
        )

    summary_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        summary_groups[
            (str(row["method"]), str(row["cohort"]), int(row["gap_length"]))
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (method, cohort, gap_length), members in sorted(summary_groups.items()):
        values = [float(item["mean_reconstructed_bscan_ncc"]) for item in members]
        ci_low, ci_high = _bootstrap_ci(
            values, samples=bootstrap_samples, seed=seed + gap_length
        )
        summary_rows.append(
            {
                "method": method,
                "cohort": cohort,
                "gap_length": gap_length,
                "mean_reconstructed_bscan_ncc": _finite_mean(values),
                "std_reconstructed_bscan_ncc": _finite_std(values),
                "sem_reconstructed_bscan_ncc": (
                    _finite_std(values) / math.sqrt(len(values))
                    if values
                    else float("nan")
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_independent_units": len(values),
            }
        )

    _write_csv(output_dir / "per_bscan_ncc.csv", bscan_rows)
    _write_csv(output_dir / "per_condition_ncc.csv", condition_rows)
    _write_csv(output_dir / "per_source_volume_ncc.csv", source_rows)
    _write_csv(output_dir / "per_independent_unit_ncc.csv", unit_rows)
    _write_csv(output_dir / "summary_by_gap_length.csv", summary_rows)
    _plot_nested_summary(summary_rows, output_dir, uncertainty=uncertainty)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "metric": "Pearson NCC per fully missing reconstructed (Z,W) B-scan",
                "evaluation_order": "B-scan NCC -> condition mean -> center mean within source volume -> cohort summary",
                "prediction_composition": "original observed voxels plus prediction inside mask",
                "uncertainty": uncertainty,
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
                "prediction_specs": list(prediction_specs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_nested_summary(
    rows: Sequence[dict[str, Any]], output_dir: Path, *, uncertainty: str
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Diagnostic plotting requires `pip install -e '.[diagnostics]'`."
        ) from exc

    methods = sorted({str(row["method"]) for row in rows})
    cohorts = sorted({str(row["cohort"]) for row in rows})
    colors = plt.get_cmap("tab10")
    for cohort in cohorts:
        figure, axis = plt.subplots(figsize=(3.45, 2.6), constrained_layout=True)
        cohort_lengths = sorted(
            {int(row["gap_length"]) for row in rows if row["cohort"] == cohort}
        )
        for method_index, method in enumerate(methods):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["cohort"] == cohort and row["method"] == method
                ),
                key=lambda row: int(row["gap_length"]),
            )
            if not selected:
                continue
            x = np.asarray([row["gap_length"] for row in selected], dtype=float)
            y = np.asarray(
                [row["mean_reconstructed_bscan_ncc"] for row in selected], dtype=float
            )
            axis.plot(
                x,
                y,
                marker="o",
                markersize=4.0,
                linewidth=1.5,
                label=method,
                color=colors(method_index),
            )
            if uncertainty == "ci95":
                lower = np.asarray([row["ci95_low"] for row in selected], dtype=float)
                upper = np.asarray([row["ci95_high"] for row in selected], dtype=float)
                errors = np.vstack([y - lower, upper - y])
            elif uncertainty in {"std", "sem"}:
                key = f"{uncertainty}_reconstructed_bscan_ncc"
                delta = np.asarray([row[key] for row in selected], dtype=float)
                errors = np.vstack([delta, delta])
            else:
                errors = None
            if errors is not None:
                axis.errorbar(
                    x, y, yerr=errors, fmt="none", capsize=2, color=colors(method_index)
                )
        axis.set_xlabel("Consecutive missing B-scans, $L$", fontsize=9)
        axis.set_ylabel("Mean reconstructed B-scan NCC", fontsize=9)
        axis.set_xticks(cohort_lengths)
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(
                output_dir / f"nested_bscan_ncc_{cohort}.{suffix}",
                dpi=400,
                bbox_inches="tight",
            )
        plt.close(figure)
