"""Unified commands for corruption, reconstruction, baselines, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines.linear import reconstruct_linear
from .core.artifacts import write_case_artifacts
from .core.backend import SimpleLamaBackend
from .core.io import load_volume, save_volume
from .core.manifest import (
    CaseRecord,
    read_manifest,
    resolve_predictions,
    write_manifest,
)
from .core.reconstruction import (
    CanonicalConfig,
    CanonicalDepthStackReconstructor,
    ReconstructionResult,
)
from .corruption import (
    CorruptionConfig,
    apply_synthetic_corruption,
    deterministic_case_seed,
)
from .evaluation.evaluator import evaluate_cases


def _case_output(root: Path, case_id: str) -> Path:
    return root / case_id


def _write_result(
    *,
    record: CaseRecord,
    output_root: Path,
    corrupted: np.ndarray,
    mask: np.ndarray,
    result: ReconstructionResult,
    config: CanonicalConfig,
) -> None:
    write_case_artifacts(
        output_dir=_case_output(output_root, record.case_id),
        case_id=record.case_id,
        corrupted=corrupted,
        missing_mask=mask,
        result=result,
        source_dtype=corrupted.dtype,
        config=config,
        extra_metadata={
            "input_paths": {
                "corrupted": str(record.corrupted_path),
                "mask": str(record.mask_path),
            },
            "cohort": record.cohort,
            "parent_id": record.parent_id,
        },
    )


def command_corrupt(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    config = CorruptionConfig()
    generated: list[CaseRecord] = []
    for record in records:
        record.require("gt_path", "projection_kind")
        clean = load_volume(record.gt_path)  # type: ignore[arg-type]
        case_seed = deterministic_case_seed(args.seed, record.case_id)
        result = apply_synthetic_corruption(clean, seed=case_seed, config=config)
        case_dir = _case_output(args.output_root, record.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = case_dir / "corrupted.tif"
        mask_path = case_dir / "missing_mask.tif"
        full_path = case_dir / "full_bscan_mask.tif"
        lateral_path = case_dir / "lateral_mask.tif"
        metadata_path = case_dir / "corruption_metadata.json"
        save_volume(corrupted_path, result.corrupted, dtype=clean.dtype)
        save_volume(mask_path, result.combined_mask, dtype=np.uint8)
        save_volume(full_path, result.full_bscan_mask, dtype=np.uint8)
        save_volume(lateral_path, result.lateral_mask, dtype=np.uint8)
        metadata_path.write_text(
            json.dumps({"case_id": record.case_id, **result.metadata}, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(
            CaseRecord(
                case_id=record.case_id,
                cohort=record.cohort,
                parent_id=record.parent_id,
                gt_path=record.gt_path,
                corrupted_path=corrupted_path,
                mask_path=mask_path,
                full_bscan_mask_path=full_path,
                lateral_mask_path=lateral_path,
                corruption_metadata_path=metadata_path,
                projection_kind=record.projection_kind,
                layer_path=record.layer_path,
                axial_offset=record.axial_offset,
            )
        )
    write_manifest(args.output_root / "manifest.csv", generated)


def command_reconstruct(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    config = CanonicalConfig()
    backend = SimpleLamaBackend(
        device=args.device,
        checkpoint_path=args.checkpoint,
        expected_sha256=config.checkpoint_sha256,
        verify_checkpoint=config.verify_checkpoint,
    )
    reconstructor = CanonicalDepthStackReconstructor(backend, config)
    for record in records:
        record.require("corrupted_path", "mask_path")
        corrupted = load_volume(record.corrupted_path)  # type: ignore[arg-type]
        mask = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
        result = reconstructor.reconstruct(corrupted, mask)
        _write_result(
            record=record,
            output_root=args.output_root,
            corrupted=corrupted,
            mask=mask,
            result=result,
            config=config,
        )


def command_linear(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    config = CanonicalConfig()
    for record in records:
        record.require("corrupted_path", "mask_path")
        corrupted = load_volume(record.corrupted_path)  # type: ignore[arg-type]
        mask = load_volume(record.mask_path) != 0  # type: ignore[arg-type]
        result = reconstruct_linear(corrupted, mask, config)
        _write_result(
            record=record,
            output_root=args.output_root,
            corrupted=corrupted,
            mask=mask,
            result=result,
            config=config,
        )


def command_evaluate(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    predictions = resolve_predictions(
        args.predictions, (record.case_id for record in records)
    )
    evaluate_cases(
        records=records,
        predictions=predictions,
        output_dir=args.output_dir,
        compute_lpips=not args.skip_lpips,
    )


def command_train_vamos(args: argparse.Namespace) -> None:
    from .baselines.vamos_octa import train_from_manifests

    checkpoint = train_from_manifests(
        args.train_manifest,
        args.validation_manifest,
        args.output_dir,
        args.device,
        args.config,
        args.epochs,
        args.max_batches,
    )
    print(f"Best VAMOS-OCTA checkpoint: {checkpoint}")


def command_predict_vamos(args: argparse.Namespace) -> None:
    from .baselines.vamos_octa import predict

    predict(
        records=read_manifest(args.manifest),
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
    )


def command_train_soad(args: argparse.Namespace) -> None:
    from .baselines.soad import train_from_manifests

    checkpoint = train_from_manifests(
        args.train_manifest,
        args.validation_manifest,
        args.output_dir,
        args.device,
        args.config,
        args.epochs,
        args.max_patches,
        args.max_batches,
    )
    print(f"Best SOAD blind-slice checkpoint: {checkpoint}")


def command_predict_soad(args: argparse.Namespace) -> None:
    from .baselines.soad import predict

    predict(
        records=read_manifest(args.manifest),
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
    )


def command_prepare_nested_gaps(args: argparse.Namespace) -> None:
    from .diagnostics.nested_gaps import prepare_nested_gaps

    _records, conditions = prepare_nested_gaps(
        manifest_path=args.manifest,
        output_root=args.output_root,
        lengths=args.lengths,
        center_fractions=args.center_fractions,
        centers_csv=args.centers,
        cohort=args.cohort,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {len(conditions)} nested-gap conditions from "
        f"{len({condition.source_case_id for condition in conditions})} source volumes; "
        f"manifest: {args.output_root / 'manifest.csv'}"
    )


def command_analyze_nested_gaps(args: argparse.Namespace) -> None:
    from .diagnostics.nested_gaps import analyze_nested_predictions

    analyze_nested_predictions(
        manifest_path=args.manifest,
        conditions_path=args.conditions,
        prediction_specs=args.predictions,
        output_dir=args.output_dir,
        uncertainty=args.uncertainty,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Nested-gap tables and plots: {args.output_dir}")


def command_prepare_median_diagnostic(args: argparse.Namespace) -> None:
    from .diagnostics.consistency import prepare_bscan_median_diagnostic

    records = prepare_bscan_median_diagnostic(
        manifest_path=args.manifest,
        output_root=args.output_root,
        kernel_size=args.kernel_size,
        cohort=None if args.cohort == "all" else args.cohort,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {len(records)} B-scan-median cases; "
        f"manifest: {args.output_root / 'manifest.csv'}"
    )


def command_analyze_consistency(args: argparse.Namespace) -> None:
    from .diagnostics.consistency import analyze_adjacent_enface_consistency

    analyze_adjacent_enface_consistency(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        volume_field=args.volume_field,
        median_kernel_size=args.kernel_size,
    )
    print(f"Adjacent-en-face consistency tables and plots: {args.output_dir}")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, help="Optional strict JSON configuration."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--epochs", type=int, help="Optional override of the epoch cap."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional batch cap per epoch, useful for lightweight validation.",
    )


def _add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    corrupt = subparsers.add_parser(
        "corrupt", help="Generate reproducible synthetic OCTA corruptions."
    )
    corrupt.add_argument(
        "--manifest", type=Path, required=True, help="Clean-data case manifest."
    )
    corrupt.add_argument("--output-root", type=Path, required=True)
    corrupt.add_argument("--seed", type=int, default=42)
    corrupt.set_defaults(handler=command_corrupt)

    reconstruct = subparsers.add_parser(
        "reconstruct", help="Run canonical frozen LaMa-depth-stack."
    )
    reconstruct.add_argument("--manifest", type=Path, required=True)
    reconstruct.add_argument("--output-root", type=Path, required=True)
    reconstruct.add_argument("--device", default="cpu")
    reconstruct.add_argument("--checkpoint", type=Path)
    reconstruct.set_defaults(handler=command_reconstruct)

    linear = subparsers.add_parser(
        "linear", help="Run mask-aware interpolation along B."
    )
    linear.add_argument("--manifest", type=Path, required=True)
    linear.add_argument("--output-root", type=Path, required=True)
    linear.set_defaults(handler=command_linear)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate any method with the shared protocol."
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction root or CSV with case_id,prediction_path.",
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip optional LPIPS when only lightweight metrics are needed.",
    )
    evaluate.set_defaults(handler=command_evaluate)

    train_vamos = subparsers.add_parser(
        "train-vamos", help="Train the supervised VAMOS-OCTA baseline."
    )
    _add_training_arguments(train_vamos)
    train_vamos.set_defaults(handler=command_train_vamos)

    predict_vamos = subparsers.add_parser(
        "predict-vamos", help="Reconstruct a manifest with a VAMOS-OCTA checkpoint."
    )
    _add_prediction_arguments(predict_vamos)
    predict_vamos.set_defaults(handler=command_predict_vamos)

    train_soad = subparsers.add_parser(
        "train-soad", help="Train SOAD blind-slice self-supervision."
    )
    _add_training_arguments(train_soad)
    train_soad.add_argument(
        "--max-patches",
        type=int,
        help="Optional patch cap for lightweight training validation.",
    )
    train_soad.set_defaults(handler=command_train_soad)

    predict_soad = subparsers.add_parser(
        "predict-soad",
        help="Reconstruct a manifest with a SOAD blind-slice checkpoint.",
    )
    _add_prediction_arguments(predict_soad)
    predict_soad.set_defaults(handler=command_predict_soad)

    nested_prepare = subparsers.add_parser(
        "diagnostic-nested-prepare",
        help="Prepare fixed-center nested full-B-scan gaps without running inference.",
    )
    nested_prepare.add_argument("--manifest", type=Path, required=True)
    nested_prepare.add_argument("--output-root", type=Path, required=True)
    nested_prepare.add_argument("--cohort", default="octa500_3mm")
    nested_prepare.add_argument(
        "--lengths", nargs="+", type=int, default=[1, 2, 4, 6, 9, 12]
    )
    nested_prepare.add_argument(
        "--center-fractions", nargs="+", type=float, default=[0.25, 0.50, 0.75]
    )
    nested_prepare.add_argument(
        "--centers",
        type=Path,
        help="Optional CSV with case_id,center_id,anchor_b for explicit anatomical locations.",
    )
    nested_prepare.add_argument("--overwrite", action="store_true")
    nested_prepare.set_defaults(handler=command_prepare_nested_gaps)

    nested_analyze = subparsers.add_parser(
        "diagnostic-nested-analyze",
        help="Compute and plot reconstructed B-scan NCC trajectories from saved predictions.",
    )
    nested_analyze.add_argument("--manifest", type=Path, required=True)
    nested_analyze.add_argument("--conditions", type=Path, required=True)
    nested_analyze.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Repeat for each method; PATH is a prediction CSV or standard output root.",
    )
    nested_analyze.add_argument("--output-dir", type=Path, required=True)
    nested_analyze.add_argument(
        "--uncertainty", choices=["ci95", "sd", "sem", "none"], default="ci95"
    )
    nested_analyze.add_argument("--bootstrap-samples", type=int, default=10_000)
    nested_analyze.add_argument("--seed", type=int, default=2027)
    nested_analyze.set_defaults(handler=command_analyze_nested_gaps)

    median_prepare = subparsers.add_parser(
        "diagnostic-median-prepare",
        help="Prepare an independent 5x5 per-B-scan median representation.",
    )
    median_prepare.add_argument("--manifest", type=Path, required=True)
    median_prepare.add_argument("--output-root", type=Path, required=True)
    median_prepare.add_argument("--kernel-size", type=int, default=5)
    median_prepare.add_argument(
        "--cohort", default="inhouse", help="Use `all` to disable cohort selection."
    )
    median_prepare.add_argument("--overwrite", action="store_true")
    median_prepare.set_defaults(handler=command_prepare_median_diagnostic)

    consistency = subparsers.add_parser(
        "diagnostic-consistency",
        help="Analyze adjacent en-face NCC before and after per-B-scan median filtering.",
    )
    consistency.add_argument("--manifest", type=Path, required=True)
    consistency.add_argument("--output-dir", type=Path, required=True)
    consistency.add_argument(
        "--volume-field", choices=["gt_path", "corrupted_path"], default="gt_path"
    )
    consistency.add_argument("--kernel-size", type=int, default=5)
    consistency.set_defaults(handler=command_analyze_consistency)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
