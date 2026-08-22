"""Generic, path-relative manifests for OCTA data and method predictions."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    """One volume and its scientific metadata; all paths are optional by task.

    Commands validate only the fields they require. This permits one manifest
    schema to drive clean-data corruption, reconstruction, and evaluation.
    """

    case_id: str
    cohort: str | None = None
    parent_id: str | None = None
    gt_path: Path | None = None
    corrupted_path: Path | None = None
    mask_path: Path | None = None
    full_bscan_mask_path: Path | None = None
    lateral_mask_path: Path | None = None
    corruption_metadata_path: Path | None = None
    projection_kind: str | None = None
    layer_path: Path | None = None
    axial_offset: int | None = None

    def require(self, *field_names: str) -> None:
        missing = [name for name in field_names if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"Case `{self.case_id}` is missing required fields: {missing}."
            )


PATH_FIELDS = {
    "gt_path",
    "corrupted_path",
    "mask_path",
    "full_bscan_mask_path",
    "lateral_mask_path",
    "corruption_metadata_path",
    "layer_path",
}


def _optional(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_manifest(path: Path) -> list[CaseRecord]:
    """Read a CSV manifest; relative paths resolve from its parent directory."""

    base = path.resolve().parent
    records: list[CaseRecord] = []
    valid_fields = {field.name for field in fields(CaseRecord)}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "case_id" not in (reader.fieldnames or []):
            raise ValueError("Manifest must contain a `case_id` column.")
        unknown = set(reader.fieldnames or []) - valid_fields
        if unknown:
            raise ValueError(f"Manifest contains unknown columns: {sorted(unknown)}.")
        for row_index, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"Manifest row {row_index} has an empty case_id.")
            kwargs: dict[str, object] = {"case_id": case_id}
            for name in valid_fields - {"case_id", "axial_offset"}:
                value = _optional(row.get(name))
                kwargs[name] = _resolve(base, value) if name in PATH_FIELDS else value
            offset = _optional(row.get("axial_offset"))
            kwargs["axial_offset"] = int(offset) if offset is not None else None
            records.append(CaseRecord(**kwargs))
    _validate_unique_ids((record.case_id for record in records), label="case")
    return records


def write_manifest(path: Path, records: Iterable[CaseRecord]) -> None:
    """Write an absolute-path manifest suitable for downstream commands."""

    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(CaseRecord)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow(
                {key: "" if value is None else str(value) for key, value in row.items()}
            )


def read_prediction_manifest(path: Path) -> dict[str, Path]:
    """Read `case_id,prediction_path` mappings for arbitrary methods."""

    base = path.resolve().parent
    predictions: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "prediction_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Prediction manifest requires `case_id,prediction_path` columns."
            )
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            raw_path = _optional(row.get("prediction_path"))
            if not case_id or raw_path is None:
                raise ValueError("Prediction manifest contains an incomplete row.")
            if case_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {case_id}")
            prediction_path = _resolve(base, raw_path)
            if prediction_path is None:  # Guard for future changes to `_resolve`.
                raise ValueError(f"Prediction path is empty for `{case_id}`.")
            predictions[case_id] = prediction_path
    return predictions


def resolve_predictions(
    specification: Path, case_ids: Iterable[str]
) -> dict[str, Path]:
    """Resolve predictions from a CSV mapping or standard output directory."""

    if specification.is_file():
        return read_prediction_manifest(specification)
    if not specification.is_dir():
        raise FileNotFoundError(f"Prediction path does not exist: {specification}")

    resolved: dict[str, Path] = {}
    for case_id in case_ids:
        candidates = [
            specification / case_id / "volumes" / "final_reconstruction.tif",
            specification / case_id / "final_reconstruction.tif",
            specification / f"{case_id}.tif",
            specification / f"{case_id}.npy",
        ]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one prediction for `{case_id}` below {specification}; found {matches}."
            )
        resolved[case_id] = matches[0]
    return resolved


def _validate_unique_ids(values: Iterable[str], *, label: str) -> None:
    values_list = list(values)
    if len(set(values_list)) != len(values_list):
        raise ValueError(f"Duplicate {label} IDs are not allowed.")
