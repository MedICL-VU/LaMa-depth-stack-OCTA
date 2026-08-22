"""Minimal VAMOS-OCTA baseline.

VAMOS-OCTA predicts one ``(Z,W)`` B-scan from a nine-B-scan corrupted window.
Unlike SOAD, the corrupted center B-scan remains in the input, so observed
same-slice pixels are available for lateral restoration. The clean target is
used only by supervised training; prediction requires only corrupted data and
the missing mask.

The architecture and objective are adapted from the official VAMOS-OCTA
implementation, distributed under the MIT License retained below.
"""

# VAMOS-OCTA upstream MIT license
# Copyright (c) 2025 Nick DiSanto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Sampler

from ..core.manifest import CaseRecord, read_manifest
from .common import (
    PreparedCase,
    affected_bscans,
    config_dict,
    crop_bottom_right,
    edge_padded_stack,
    load_checkpoint,
    pad_bottom_right,
    prepare_case,
    save_checkpoint,
    set_deterministic_seed,
    write_history,
    write_prediction_artifacts,
    write_run_config,
)


@dataclass(frozen=True)
class VAMOSConfig:
    """Default VAMOS-OCTA configuration."""

    input_bscans: int = 9
    features: tuple[int, ...] = (64, 128, 256, 512)
    dropout: float = 0.0
    batch_size: int = 8
    maximum_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 5e-5
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    weighted_mse_weight: float = 1.0
    axial_mip_weight: float = 3.0
    lateral_mip_multiplier: float = 3.0
    axial_aip_weight: float = 3.0
    lateral_aip_weight: float = 3.0
    vessel_alpha: float = 100.0
    vessel_gamma: float = 1.0 / 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_bscans != 9:
            raise ValueError("The VAMOS-OCTA baseline uses exactly 9 B-scans.")
        if len(self.features) != 4 or any(value <= 0 for value in self.features):
            raise ValueError(
                "VAMOS features must contain four positive channel counts."
            )
        if self.batch_size <= 0 or self.maximum_epochs <= 0:
            raise ValueError("VAMOS batch size and maximum epochs must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("VAMOS dropout must be in [0,1).")
        if self.early_stopping_patience < 0 or self.plateau_patience < 0:
            raise ValueError("VAMOS patience values must be nonnegative.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("VAMOS early-stopping minimum delta must be nonnegative.")
        if not 0.0 < self.plateau_factor < 1.0:
            raise ValueError("VAMOS plateau_factor must be in (0,1).")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "VAMOS learning rate must be positive and weight decay nonnegative."
            )
        loss_weights = (
            self.weighted_mse_weight,
            self.axial_mip_weight,
            self.lateral_mip_multiplier,
            self.axial_aip_weight,
            self.lateral_aip_weight,
            self.vessel_alpha,
            self.vessel_gamma,
        )
        if any(value < 0 for value in loss_weights):
            raise ValueError(
                "VAMOS loss weights and vessel parameters must be nonnegative."
            )
        effective_terms = (
            self.weighted_mse_weight,
            self.axial_mip_weight,
            self.axial_aip_weight,
            self.lateral_aip_weight,
        )
        if not any(value > 0 for value in effective_terms):
            raise ValueError("VAMOS requires at least one positive loss weight.")


def load_config(path: Path | None) -> VAMOSConfig:
    """Load and validate a strict VAMOS-OCTA configuration JSON file."""

    if path is None:
        return VAMOSConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_metadata = {
        "method": "vamos_octa",
        "normalization": "observed_positive_p99p9",
        "optimizer": "adamw",
        "corrupted_center_observed_context": True,
        "preserve_observed": True,
    }
    for key, expected in required_metadata.items():
        if key in payload and payload[key] != expected:
            raise ValueError(f"VAMOS-OCTA requires `{key}={expected!r}`.")
    aliases = {
        "unet_features": "features",
        "optimizer": None,
        "method": None,
        "normalization": None,
        "corrupted_center_observed_context": None,
        "preserve_observed": None,
    }
    valid = {field.name for field in fields(VAMOSConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        mapped = aliases.get(key, key)
        if mapped is None:
            continue
        if mapped not in valid:
            raise ValueError(f"Unknown VAMOS-OCTA configuration field: {key}")
        kwargs[mapped] = tuple(value) if mapped == "features" else value
    return VAMOSConfig(**kwargs)


class DoubleConv(nn.Module):
    """Two ``Conv2d -> ReLU -> BatchNorm2d`` blocks from VAMOS-OCTA."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        delta_z = skip.size(2) - inputs.size(2)
        delta_w = skip.size(3) - inputs.size(3)
        inputs = F.pad(
            inputs,
            [
                delta_w // 2,
                delta_w - delta_w // 2,
                delta_z // 2,
                delta_z - delta_z // 2,
            ],
        )
        return self.conv(torch.cat([skip, inputs], dim=1))


class VAMOSUNet(nn.Module):
    """The 2.5D U-Net used by the VAMOS-OCTA baseline."""

    def __init__(self, config: VAMOSConfig) -> None:
        super().__init__()
        f0, f1, f2, f3 = config.features
        self.input = DoubleConv(config.input_bscans, f0)
        self.down1 = Down(f0, f1)
        self.down2 = Down(f1, f2)
        self.down3 = Down(f2, f3)
        self.dropout = (
            nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.up1 = Up(f3, f2)
        self.up2 = Up(f2, f1)
        self.up3 = Up(f1, f0)
        self.output = nn.Conv2d(f0, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.input(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        decoded = self.dropout(self.up1(x4, x3))
        decoded = self.dropout(self.up2(decoded, x2))
        decoded = self.up3(decoded, x1)
        return self.output(decoded)


class VAMOSDataset(Dataset[dict[str, torch.Tensor]]):
    """Static-corruption supervised samples for affected center B-scans."""

    def __init__(self, cases: Sequence[PreparedCase], config: VAMOSConfig) -> None:
        self.cases = list(cases)
        self.config = config
        self.samples: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            if case.clean_01 is None:
                raise ValueError("VAMOS training samples require a clean target.")
            self.samples.extend(
                (case_index, int(center))
                for center in affected_bscans(case.missing_mask)
            )
        if not self.samples:
            raise ValueError("No affected B-scans are available for VAMOS training.")

    def __len__(self) -> int:
        return len(self.samples)

    def shape_for_index(self, index: int) -> tuple[int, int]:
        case_index, _ = self.samples[index]
        return tuple(
            int(value) for value in self.cases[case_index].corrupted_01.shape[1:]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_index, center = self.samples[index]
        case = self.cases[case_index]
        stack = edge_padded_stack(case.corrupted_01, center, self.config.input_bscans)
        return {
            "input": torch.from_numpy(stack.astype(np.float32)),
            "target": torch.from_numpy(case.clean_01[center][None].astype(np.float32)),  # type: ignore[index]
            "corrupted_center": torch.from_numpy(
                case.corrupted_01[center][None].astype(np.float32)
            ),
            "missing": torch.from_numpy(
                case.missing_mask[center][None].astype(np.float32)
            ),
        }


class ShapeBatchSampler(Sampler[list[int]]):
    """Batch native-size samples without resizing mixed-resolution OCTA volumes."""

    def __init__(
        self, dataset: VAMOSDataset, batch_size: int, *, shuffle: bool, seed: int
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[tuple[int, int], list[int]] = {}
        for index in range(len(dataset)):
            groups.setdefault(dataset.shape_for_index(index), []).append(index)
        self.groups = groups

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.groups.values():
            selected = list(indices)
            if self.shuffle:
                generator.shuffle(selected)
            batches.extend(
                selected[start : start + self.batch_size]
                for start in range(0, len(selected), self.batch_size)
            )
        if self.shuffle:
            generator.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups.values()
        )


def vamos_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    corrupted_center: torch.Tensor,
    missing: torch.Tensor,
    config: VAMOSConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask-aware VAMOS composite loss used by the default baseline.

    Observed center-slice pixels are pasted back before all loss terms, so
    optimization is restricted to model-supplied missing content.
    """

    composed = prediction * missing + corrupted_center * (1.0 - missing)
    with torch.no_grad():
        epsilon = 1e-5
        weights = (
            config.vessel_alpha
            * composed.detach().clamp_min(epsilon).pow(config.vessel_gamma)
            + target.detach().clamp_min(epsilon).pow(config.vessel_gamma)
            + 0.5
        )
    weighted_mse = torch.mean(weights * (composed - target).square())
    axial_mip = F.l1_loss(composed.max(dim=2).values, target.max(dim=2).values)
    lateral_mip = F.l1_loss(composed.max(dim=3).values, target.max(dim=3).values)
    axial_aip = F.l1_loss(composed.mean(dim=2), target.mean(dim=2))
    lateral_aip = F.l1_loss(composed.mean(dim=3), target.mean(dim=3))
    total = (
        config.weighted_mse_weight * weighted_mse
        + config.axial_mip_weight * axial_mip
        + config.lateral_mip_multiplier * config.axial_mip_weight * lateral_mip
        + config.axial_aip_weight * axial_aip
        + config.lateral_aip_weight * lateral_aip
    )
    terms = {
        "loss": float(total.detach()),
        "weighted_mse": float(weighted_mse.detach()),
        "axial_mip_l1": float(axial_mip.detach()),
        "lateral_mip_l1": float(lateral_mip.detach()),
        "axial_aip_l1": float(axial_aip.detach()),
        "lateral_aip_l1": float(lateral_aip.detach()),
    }
    return total, terms


def _epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: VAMOSConfig,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    examples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            center = batch["corrupted_center"].to(device)
            missing = batch["missing"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss, terms = vamos_loss(prediction, target, center, missing, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("VAMOS-OCTA produced a non-finite loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(inputs.shape[0])
            examples += count
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value * count
    if examples == 0:
        raise RuntimeError("VAMOS-OCTA epoch processed no samples.")
    return {key: value / examples for key, value in totals.items()}


def train(
    *,
    train_records: Sequence[CaseRecord],
    validation_records: Sequence[CaseRecord],
    output_dir: Path,
    device: str,
    config: VAMOSConfig,
    epochs: int | None = None,
    max_batches: int | None = None,
) -> Path:
    """Train VAMOS-OCTA and return the best checkpoint path."""

    set_deterministic_seed(config.seed)
    train_cases = [prepare_case(record, require_clean=True) for record in train_records]
    validation_cases = [
        prepare_case(record, require_clean=True) for record in validation_records
    ]
    train_dataset = VAMOSDataset(train_cases, config)
    validation_dataset = VAMOSDataset(validation_cases, config)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ShapeBatchSampler(
            train_dataset, config.batch_size, shuffle=True, seed=config.seed
        ),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=ShapeBatchSampler(
            validation_dataset, config.batch_size, shuffle=False, seed=config.seed
        ),
    )
    torch_device = torch.device(device)
    model = VAMOSUNet(config).to(torch_device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.plateau_patience,
        factor=config.plateau_factor,
    )
    run_config = config_dict(config)
    write_run_config(
        output_dir / "run_config.json",
        {
            "method": "vamos_octa",
            "config": run_config,
            "train_case_ids": [record.case_id for record in train_records],
            "validation_case_ids": [record.case_id for record in validation_records],
            "normalization": "observed_positive_p99p9",
            "inference_requires_gt": False,
            "center_bscan_policy": "corrupted center retained; missing voxels remain absent",
        },
    )
    history: list[dict[str, float | int]] = []
    best = float("inf")
    stale_epochs = 0
    best_path = output_dir / "checkpoints" / "best.pt"
    maximum_epochs = int(config.maximum_epochs if epochs is None else epochs)
    if maximum_epochs <= 0:
        raise ValueError("VAMOS training requires at least one epoch.")
    for epoch in range(1, maximum_epochs + 1):
        train_terms = _epoch(
            model, train_loader, config, torch_device, optimizer, max_batches
        )
        validation_terms = _epoch(
            model, validation_loader, config, torch_device, None, max_batches
        )
        validation_loss = validation_terms["loss"]
        scheduler.step(validation_loss)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_terms.items()},
            **{f"validation_{key}": value for key, value in validation_terms.items()},
        }
        history.append(row)
        write_history(output_dir / "logs" / "training_history.csv", history)
        improved = validation_loss < best - config.early_stopping_min_delta
        if improved:
            best = validation_loss
            stale_epochs = 0
            save_checkpoint(
                best_path,
                method="vamos_octa",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=run_config,
                history=history,
                best_validation_loss=best,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            method="vamos_octa",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=run_config,
            history=history,
            best_validation_loss=best,
        )
        print(
            f"[VAMOS-OCTA] epoch={epoch:03d} train={train_terms['loss']:.6f} "
            f"validation={validation_loss:.6f}"
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    return best_path


@torch.no_grad()
def predict_case(
    model: nn.Module, case: PreparedCase, config: VAMOSConfig, device: str
) -> np.ndarray:
    """Predict all affected B-scans without accessing a clean volume."""

    model.eval()
    torch_device = torch.device(device)
    raw = case.corrupted_01.copy()
    for center in affected_bscans(case.missing_mask):
        stack = edge_padded_stack(case.corrupted_01, int(center), config.input_bscans)
        padded, crop = pad_bottom_right(stack, multiple=8)
        inputs = torch.from_numpy(padded[None].astype(np.float32)).to(torch_device)
        prediction = model(inputs).squeeze(0).squeeze(0).cpu().numpy()
        raw[int(center)] = crop_bottom_right(prediction, crop)
    return raw


def predict(
    *,
    records: Sequence[CaseRecord],
    checkpoint_path: Path,
    output_root: Path,
    device: str,
) -> None:
    """Reload a VAMOS checkpoint and reconstruct every manifest case."""

    payload = load_checkpoint(checkpoint_path, map_location=device)
    if payload["method"] != "vamos_octa":
        raise ValueError(
            f"Expected VAMOS-OCTA checkpoint, found `{payload['method']}`."
        )
    config = VAMOSConfig(**payload["config"])
    model = VAMOSUNet(config).to(torch.device(device))
    model.load_state_dict(payload["model_state"])
    for record in records:
        case = prepare_case(record, require_clean=False)
        raw = predict_case(model, case, config, device)
        write_prediction_artifacts(
            case=case,
            raw_prediction_01=raw,
            output_root=output_root,
            method="vamos_octa",
            checkpoint_path=checkpoint_path,
            config=config_dict(config),
        )


def train_from_manifests(
    train_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    device: str,
    config_path: Path | None,
    epochs: int | None,
    max_batches: int | None,
) -> Path:
    return train(
        train_records=read_manifest(train_manifest),
        validation_records=read_manifest(validation_manifest),
        output_dir=output_dir,
        device=device,
        config=load_config(config_path),
        epochs=epochs,
        max_batches=max_batches,
    )
