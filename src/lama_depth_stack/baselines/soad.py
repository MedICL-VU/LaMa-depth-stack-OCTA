"""SOAD blind-slice baseline for controlled OCTA restoration.

The architecture is adapted from the official SOAD implementation by Zhenghong
Li et al. (MICCAI 2024), distributed under the MIT License retained below.

Only the blind-slice path is exposed: the center B-scan is zero
for every model input. Training is self-supervised from observed corrupted
center-slice pixels. Clean targets are not loaded for training or prediction.
"""

# SOAD upstream MIT license
# Copyright (c) 2024 ZhenghLi
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
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

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
class SOADConfig:
    """Default SOAD blind-slice configuration."""

    input_bscans: int = 7
    patch_size: int = 128
    patch_stride: int = 64
    features: tuple[int, ...] = (16, 32, 64, 128, 256)
    residual_blocks: tuple[int, ...] = (1, 2, 3, 3)
    batch_size: int = 128
    maximum_epochs: int = 30
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    learning_rate: float = 1e-3
    learning_rate_milestones: tuple[int, ...] = (4, 8, 12)
    learning_rate_gamma: float = 0.1
    alpha: float = 100.0
    beta: float = 1.0
    gamma: float = 3.0
    epsilon: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_bscans != 7:
            raise ValueError("The SOAD blind-slice baseline uses exactly 7 B-scans.")
        if self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError("SOAD patch size and stride must be positive.")
        if len(self.features) != 5 or len(self.residual_blocks) != 4:
            raise ValueError(
                "SOAD requires five feature widths and four residual-block counts."
            )
        if any(value <= 0 for value in (*self.features, *self.residual_blocks)):
            raise ValueError(
                "SOAD feature widths and residual-block counts must be positive."
            )
        if self.batch_size <= 0 or self.maximum_epochs <= 0:
            raise ValueError("SOAD batch size and maximum epochs must be positive.")
        if self.early_stopping_patience < 0:
            raise ValueError("SOAD early-stopping patience must be nonnegative.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("SOAD early-stopping minimum delta must be nonnegative.")
        if self.learning_rate <= 0 or self.learning_rate_gamma <= 0:
            raise ValueError("SOAD learning-rate values must be positive.")
        if any(epoch <= 0 for epoch in self.learning_rate_milestones):
            raise ValueError("SOAD learning-rate milestones must be positive epochs.")
        if self.alpha < 0 or self.beta < 0 or self.epsilon < 0 or self.gamma <= 0:
            raise ValueError(
                "SOAD loss parameters require alpha/beta/epsilon >= 0 and gamma > 0."
            )


def load_config(path: Path | None) -> SOADConfig:
    """Load and validate a strict SOAD configuration JSON file."""

    if path is None:
        return SOADConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_metadata = {
        "method": "soad_blind_slice",
        "normalization": "observed_positive_p99p9",
        "center_bscan_input": "fully_zeroed",
        "optimizer": "adam",
        "preserve_observed": True,
    }
    for key, expected in required_metadata.items():
        if key in payload and payload[key] != expected:
            raise ValueError(f"SOAD blind-slice requires `{key}={expected!r}`.")
    aliases = {
        "method": None,
        "normalization": None,
        "center_bscan_input": None,
        "optimizer": None,
        "preserve_observed": None,
    }
    valid = {field.name for field in fields(SOADConfig)}
    kwargs: dict[str, Any] = {}
    tuple_fields = {"features", "residual_blocks", "learning_rate_milestones"}
    for key, value in payload.items():
        mapped = aliases.get(key, key)
        if mapped is None:
            continue
        if mapped not in valid:
            raise ValueError(f"Unknown SOAD configuration field: {key}")
        kwargs[mapped] = tuple(value) if mapped in tuple_fields else value
    config = SOADConfig(**kwargs)
    if payload.get("center_bscan_input", "fully_zeroed") != "fully_zeroed":
        raise ValueError("Only SOAD blind-slice inference is supported.")
    return config


class ConvNormAct3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] = 1,
        stride: int | tuple[int, int, int] = 1,
        normalize: bool = True,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride
        )
        self.norm = nn.BatchNorm3d(out_channels) if normalize else nn.Identity()
        self.activation = nn.ReLU() if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
        normalize: bool = True,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.activation = nn.ReLU() if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class Residual3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = ConvNormAct3d(channels, channels, activate=False)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.residual(inputs))


class Residual2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = ConvNormAct2d(channels, channels, activate=False)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.residual(inputs))


class Down3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, down_time: bool, blocks: int
    ) -> None:
        super().__init__()
        if down_time:
            self.down = ConvNormAct3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=(0, 1, 1),
                stride=2,
            )
        else:
            self.down = ConvNormAct3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=(1, 2, 2),
            )
        # The official implementation repeats one block instance, sharing its
        # parameters across the configured residual depth. Preserve that
        # unusual but checkpoint-relevant behavior exactly.
        residual = Residual3d(out_channels)
        self.residual = nn.Sequential(*(residual for _ in range(blocks)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.residual(self.down(inputs))


class Up2d(nn.Module):
    def __init__(
        self, up_channels: int, skip_channels: int, out_channels: int, *, blocks: int
    ) -> None:
        super().__init__()
        inner_channels = up_channels // 2
        self.up = ConvNormAct2d(up_channels, inner_channels)
        self.transition = ConvNormAct2d(
            inner_channels + skip_channels, out_channels, kernel_size=1, padding=0
        )
        residual = Residual2d(out_channels)
        self.residual = nn.Sequential(*(residual for _ in range(blocks)))

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="nearest")
        return self.residual(self.transition(torch.cat([self.up(inputs), skip], dim=1)))


class NonLocal2d(nn.Module):
    """Embedded-Gaussian non-local block used at the SOAD decoder bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        inner = channels // 2
        self.inner = inner
        self.g = nn.Sequential(nn.Conv2d(channels, inner, 1), nn.MaxPool2d(2))
        self.theta = nn.Conv2d(channels, inner, 1)
        self.phi = nn.Sequential(nn.Conv2d(channels, inner, 1), nn.MaxPool2d(2))
        self.output = nn.Sequential(
            nn.Conv2d(inner, channels, 1), nn.BatchNorm2d(channels)
        )
        nn.init.constant_(self.output[1].weight, 0)
        nn.init.constant_(self.output[1].bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        g = self.g(inputs).view(batch, self.inner, -1).permute(0, 2, 1)
        theta = self.theta(inputs).view(batch, self.inner, -1).permute(0, 2, 1)
        phi = self.phi(inputs).view(batch, self.inner, -1)
        affinity = F.softmax(torch.matmul(theta, phi), dim=-1)
        response = torch.matmul(affinity, g).permute(0, 2, 1).contiguous()
        response = response.view(batch, self.inner, *inputs.shape[-2:])
        return self.output(response) + inputs


class SOADNet(nn.Module):
    """Official SOAD VNetProj-nonlocal architecture specialized to seven frames."""

    def __init__(self, config: SOADConfig) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = config.features
        n0, n1, n2, n3 = config.residual_blocks
        frames = config.input_bscans
        self.input = ConvNormAct3d(1, f0, kernel_size=5, padding=2)
        self.down1 = Down3d(f0, f1, down_time=True, blocks=n0)
        self.down2 = Down3d(f1, f2, down_time=False, blocks=n1)
        self.down3 = Down3d(f2, f3, down_time=True, blocks=n2)
        self.down4 = Down3d(f3, f4, down_time=False, blocks=n3)
        self.project0 = nn.Sequential(
            nn.Conv3d(f0, f0, kernel_size=(frames, 5, 5), padding=(0, 2, 2)), nn.ReLU()
        )
        self.project1 = nn.Sequential(
            nn.Conv3d(f1, f1, kernel_size=(frames // 2, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
        )
        self.project2 = nn.Sequential(
            nn.Conv3d(f2, f2, kernel_size=(frames // 2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        self.project3 = nn.Sequential(
            nn.Conv3d(f3, f3, kernel_size=(frames // 4, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        self.project4 = nn.Sequential(
            nn.Conv3d(f4, f4, kernel_size=(frames // 4, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        self.up4 = Up2d(f4, f3, f4, blocks=n3)
        self.nonlocal_block = NonLocal2d(f4)
        self.up3 = Up2d(f4, f2, f3, blocks=n2)
        self.up2 = Up2d(f3, f1, f2, blocks=n1)
        self.up1 = Up2d(f2, f0, f1, blocks=n0)
        self.output = nn.Sequential(
            ConvNormAct2d(f1, f1),
            ConvNormAct2d(
                f1, 1, kernel_size=1, padding=0, normalize=False, activate=False
            ),
        )
        self._initialize_official_3d_layers()

    def _initialize_official_3d_layers(self) -> None:
        # The upstream initializer deliberately initializes only 3D layers.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level0 = self.input(inputs)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level4 = self.down4(level3)
        projection0 = self.project0(level0).squeeze(2)
        projection1 = self.project1(level1).squeeze(2)
        projection2 = self.project2(level2).squeeze(2)
        projection3 = self.project3(level3).squeeze(2)
        projection4 = self.project4(level4).squeeze(2)
        decoded = self.nonlocal_block(self.up4(projection4, projection3))
        decoded = self.up3(decoded, projection2)
        decoded = self.up2(decoded, projection1)
        decoded = self.up1(decoded, projection0)
        return self.output(decoded)


def _pad_spatial_minimum(
    array: np.ndarray, minimum: int, *, mode: str, constant: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``(B,Z,W)`` to the configured patch size and return native-pixel support."""

    b, z, w = array.shape
    pad_z = max(0, minimum - z)
    pad_w = max(0, minimum - w)
    padding = ((0, 0), (0, pad_z), (0, pad_w))
    if mode == "constant":
        padded = np.pad(array, padding, mode=mode, constant_values=constant)
    else:
        padded = np.pad(array, padding, mode=mode)
    valid = np.pad(
        np.ones((b, z, w), dtype=bool), padding, mode="constant", constant_values=False
    )
    return padded, valid


class SOADPatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Self-supervised 128x128 blind-center patches from corrupted volumes only."""

    def __init__(
        self,
        cases: Sequence[PreparedCase],
        config: SOADConfig,
        *,
        max_patches: int | None = None,
    ) -> None:
        self.cases = list(cases)
        self.config = config
        self.samples: list[tuple[int, int, int, int]] = []
        for case_index, case in enumerate(self.cases):
            if case.clean_01 is not None:
                raise ValueError(
                    "SOAD self-supervised training must not load clean targets."
                )
            _, z, w = case.corrupted_01.shape
            z_effective = max(z, config.patch_size)
            w_effective = max(w, config.patch_size)
            padded_mask, native = _pad_spatial_minimum(
                case.missing_mask, config.patch_size, mode="constant", constant=1.0
            )
            for center in range(case.corrupted_01.shape[0]):
                for z0 in range(
                    0, z_effective - config.patch_size + 1, config.patch_stride
                ):
                    for w0 in range(
                        0, w_effective - config.patch_size + 1, config.patch_stride
                    ):
                        observed = (
                            ~padded_mask[
                                center,
                                z0 : z0 + config.patch_size,
                                w0 : w0 + config.patch_size,
                            ].astype(bool)
                            & native[
                                center,
                                z0 : z0 + config.patch_size,
                                w0 : w0 + config.patch_size,
                            ]
                        )
                        if np.any(observed):
                            self.samples.append((case_index, center, z0, w0))
        if max_patches is not None:
            self.samples = self.samples[: int(max_patches)]
        if not self.samples:
            raise ValueError("No observed SOAD training patches are available.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_index, center, z0, w0 = self.samples[index]
        case = self.cases[case_index]
        corrupted, native = _pad_spatial_minimum(
            case.corrupted_01, self.config.patch_size, mode="edge"
        )
        missing, _ = _pad_spatial_minimum(
            case.missing_mask, self.config.patch_size, mode="constant", constant=1.0
        )
        stack = edge_padded_stack(corrupted, center, self.config.input_bscans)[
            :, z0 : z0 + self.config.patch_size, w0 : w0 + self.config.patch_size
        ].copy()
        target = corrupted[
            center, z0 : z0 + self.config.patch_size, w0 : w0 + self.config.patch_size
        ].copy()
        observed = (
            ~missing[
                center,
                z0 : z0 + self.config.patch_size,
                w0 : w0 + self.config.patch_size,
            ].astype(bool)
            & native[
                center,
                z0 : z0 + self.config.patch_size,
                w0 : w0 + self.config.patch_size,
            ]
        )
        stack[self.config.input_bscans // 2] = 0.0
        return {
            "input": torch.from_numpy(stack[None].astype(np.float32)),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "observed": torch.from_numpy(observed[None].astype(np.float32)),
        }


def soad_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    config: SOADConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Official weighted-recovery form restricted to valid observed targets."""

    count = observed.sum()
    if count <= 0:
        raise ValueError("SOAD loss received a patch with no observed target pixels.")
    exponent = 1.0 / config.gamma
    weights = (
        config.alpha * prediction.detach().clamp(0.0, 1.0).pow(exponent)
        + config.beta * target.detach().clamp(0.0, 1.0).pow(exponent)
        + config.epsilon
    )
    squared = (prediction - target).square()
    loss = torch.sum(weights * squared * observed) / count
    mse = torch.sum(squared * observed) / count
    return loss, {"loss": float(loss.detach()), "observed_mse": float(mse.detach())}


def _epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: SOADConfig,
    device: torch.device,
    optimizer: Adam | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "observed_mse": 0.0}
    examples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            observed = batch["observed"].to(device)
            if not torch.all(inputs[:, :, config.input_bscans // 2] == 0):
                raise RuntimeError("SOAD center-frame blindness invariant failed.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss, terms = soad_loss(prediction, target, observed, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("SOAD produced a non-finite loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(inputs.shape[0])
            examples += count
            for key, value in terms.items():
                totals[key] += value * count
    if examples == 0:
        raise RuntimeError("SOAD epoch processed no samples.")
    return {key: value / examples for key, value in totals.items()}


def train(
    *,
    train_records: Sequence[CaseRecord],
    validation_records: Sequence[CaseRecord],
    output_dir: Path,
    device: str,
    config: SOADConfig,
    epochs: int | None = None,
    max_patches: int | None = None,
    max_batches: int | None = None,
) -> Path:
    """Train SOAD blind-slice and return the best checkpoint path."""

    set_deterministic_seed(config.seed)
    train_cases = [
        prepare_case(record, require_clean=False) for record in train_records
    ]
    validation_cases = [
        prepare_case(record, require_clean=False) for record in validation_records
    ]
    train_dataset = SOADPatchDataset(train_cases, config, max_patches=max_patches)
    validation_dataset = SOADPatchDataset(
        validation_cases, config, max_patches=max_patches
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    torch_device = torch.device(device)
    model = SOADNet(config).to(torch_device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(config.learning_rate_milestones),
        gamma=config.learning_rate_gamma,
    )
    run_config = config_dict(config)
    write_run_config(
        output_dir / "run_config.json",
        {
            "method": "soad_blind_slice",
            "config": run_config,
            "train_case_ids": [record.case_id for record in train_records],
            "validation_case_ids": [record.case_id for record in validation_records],
            "training_supervision": "observed corrupted center pixels only",
            "clean_target_loaded": False,
            "center_bscan_policy": "fully zeroed before every forward pass",
        },
    )
    history: list[dict[str, float | int]] = []
    best = float("inf")
    stale_epochs = 0
    best_path = output_dir / "checkpoints" / "best.pt"
    maximum_epochs = int(config.maximum_epochs if epochs is None else epochs)
    if maximum_epochs <= 0:
        raise ValueError("SOAD training requires at least one epoch.")
    for epoch in range(1, maximum_epochs + 1):
        train_terms = _epoch(
            model, train_loader, config, torch_device, optimizer, max_batches
        )
        validation_terms = _epoch(
            model, validation_loader, config, torch_device, None, max_batches
        )
        scheduler.step()
        validation_loss = validation_terms["loss"]
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
                method="soad_blind_slice",
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
            method="soad_blind_slice",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=run_config,
            history=history,
            best_validation_loss=best,
        )
        print(
            f"[SOAD-blind-slice] epoch={epoch:03d} train={train_terms['loss']:.6f} "
            f"validation={validation_loss:.6f}"
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    return best_path


@torch.no_grad()
def predict_case(
    model: nn.Module, case: PreparedCase, config: SOADConfig, device: str
) -> np.ndarray:
    """Predict affected centers from seven-frame windows with a zero center."""

    model.eval()
    torch_device = torch.device(device)
    raw = case.corrupted_01.copy()
    for center in affected_bscans(case.missing_mask):
        stack = edge_padded_stack(
            case.corrupted_01, int(center), config.input_bscans
        ).copy()
        stack[config.input_bscans // 2] = 0.0
        padded, crop = pad_bottom_right(stack, multiple=16)
        inputs = torch.from_numpy(padded[None, None].astype(np.float32)).to(
            torch_device
        )
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
    payload = load_checkpoint(checkpoint_path, map_location=device)
    if payload["method"] != "soad_blind_slice":
        raise ValueError(f"Expected SOAD checkpoint, found `{payload['method']}`.")
    config = SOADConfig(**payload["config"])
    model = SOADNet(config).to(torch.device(device))
    model.load_state_dict(payload["model_state"])
    for record in records:
        case = prepare_case(record, require_clean=False)
        raw = predict_case(model, case, config, device)
        write_prediction_artifacts(
            case=case,
            raw_prediction_01=raw,
            output_root=output_root,
            method="soad_blind_slice",
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
    max_patches: int | None,
    max_batches: int | None,
) -> Path:
    return train(
        train_records=read_manifest(train_manifest),
        validation_records=read_manifest(validation_manifest),
        output_dir=output_dir,
        device=device,
        config=load_config(config_path),
        epochs=epochs,
        max_patches=max_patches,
        max_batches=max_batches,
    )
