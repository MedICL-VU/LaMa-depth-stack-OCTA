"""Reproducible synthetic motion corruptions for OCTA restoration."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorruptionConfig:
    """Synthetic corruption protocol with explicit lateral-event placement."""

    full_bscan_fraction: float = 0.25
    geometric_probability: float = 0.4
    maximum_group_length: int = 6
    lateral_width_fraction_each_boundary: float = 0.15
    lateral_span_bscans_min: int = 40
    lateral_span_bscans_max: int = 60
    slope_max_pixels: float = 10.0
    noise_standard_deviation_pixels: float = 4.0
    noise_smoothing_window: int = 9
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.full_bscan_fraction < 1:
            raise ValueError("full_bscan_fraction must be in (0,1).")
        if not 0 < self.geometric_probability <= 1:
            raise ValueError("geometric_probability must be in (0,1].")
        if self.maximum_group_length < 1:
            raise ValueError("maximum_group_length must be positive.")
        if not 0 < self.lateral_width_fraction_each_boundary < 0.5:
            raise ValueError("Each lateral width fraction must be in (0,0.5).")
        if self.lateral_span_bscans_min < 1:
            raise ValueError("Minimum lateral span must be positive.")
        if self.lateral_span_bscans_max < self.lateral_span_bscans_min:
            raise ValueError(
                "Maximum lateral span must not be smaller than the minimum."
            )
        if self.slope_max_pixels < 0 or self.noise_standard_deviation_pixels < 0:
            raise ValueError("Lateral slope and noise scales must be nonnegative.")
        if self.noise_smoothing_window < 1:
            raise ValueError("noise_smoothing_window must be positive.")
        if not np.isfinite(self.fill_value):
            raise ValueError("fill_value must be finite.")


@dataclass(frozen=True)
class CorruptionResult:
    corrupted: np.ndarray
    combined_mask: np.ndarray
    full_bscan_mask: np.ndarray
    lateral_mask: np.ndarray
    metadata: dict[str, Any]


def deterministic_case_seed(global_seed: int, case_id: str) -> int:
    """Derive an order-independent uint64 seed from one run seed and case ID."""

    payload = f"{int(global_seed)}:{case_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def smooth_noise(noise: np.ndarray, window: int) -> np.ndarray:
    """Apply an edge-padded moving average to lateral-width noise."""

    if window <= 1:
        return np.asarray(noise, dtype=np.float32)
    effective = int(window) if int(window) % 2 == 1 else int(window) + 1
    pad = effective // 2
    padded = np.pad(np.asarray(noise, dtype=np.float32), (pad, pad), mode="edge")
    kernel = np.full(effective, 1.0 / effective, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid")


def _would_exceed_run(existing: set[int], candidate: range, maximum: int) -> bool:
    run = 1
    indices = sorted(existing | set(candidate))
    for previous, current in pairwise(indices):
        run = run + 1 if current == previous + 1 else 1
        if run > maximum:
            return True
    return False


def sample_full_bscans(
    b_size: int,
    rng: np.random.Generator,
    config: CorruptionConfig,
) -> list[int]:
    target = round(config.full_bscan_fraction * b_size)
    selected: set[int] = set()
    for _ in range(100_000):
        if len(selected) == target:
            break
        length = min(
            int(rng.geometric(config.geometric_probability)),
            config.maximum_group_length,
            target - len(selected),
        )
        start = int(rng.integers(0, b_size - length + 1))
        candidate = range(start, start + length)
        if any(index in selected for index in candidate):
            continue
        if _would_exceed_run(selected, candidate, config.maximum_group_length):
            continue
        selected.update(candidate)
    if len(selected) != target:
        raise RuntimeError(
            f"Could place only {len(selected)}/{target} fully missing B-scans."
        )
    return sorted(selected)


def _width_profile(
    *,
    length: int,
    lateral_size: int,
    rng: np.random.Generator,
    config: CorruptionConfig,
) -> tuple[np.ndarray, float]:
    nominal = float(config.lateral_width_fraction_each_boundary * lateral_size)
    slope_endpoint = float(
        rng.uniform(-config.slope_max_pixels, config.slope_max_pixels)
    )
    slope = np.linspace(0.0, slope_endpoint, length, dtype=np.float32)
    noise = rng.normal(
        0.0,
        config.noise_standard_deviation_pixels,
        size=length,
    ).astype(np.float32)
    widths = np.rint(
        nominal + slope + smooth_noise(noise, config.noise_smoothing_window)
    )
    return np.clip(widths, 1, lateral_size).astype(np.int32), slope_endpoint


def apply_synthetic_corruption(
    volume: np.ndarray,
    *,
    seed: int,
    config: CorruptionConfig | None = None,
) -> CorruptionResult:
    """Apply full-B-scan and independent left/right lateral corruption.

    Args:
        volume: Clean OCTA volume in `(B,Z,W)` order.
        seed: Per-case seed, normally from :func:`deterministic_case_seed`.
    """

    config = config or CorruptionConfig()
    clean = np.asarray(volume)
    if clean.ndim != 3:
        raise ValueError(f"Expected `(B,Z,W)` volume, got {clean.shape}.")
    b_size, _z_size, w_size = clean.shape
    rng = np.random.default_rng(int(seed))

    full_mask = np.zeros(clean.shape, dtype=bool)
    full_indices = sample_full_bscans(b_size, rng, config)
    full_mask[full_indices, :, :] = True

    lateral_mask = np.zeros(clean.shape, dtype=bool)
    lateral_events: list[dict[str, Any]] = []
    occupied_intervals: list[tuple[int, int]] = []
    for side in ("left", "right"):
        if b_size < 2:
            raise ValueError(
                "At least two B-scans are required for independent lateral events."
            )
        length = min(
            int(
                rng.integers(
                    config.lateral_span_bscans_min, config.lateral_span_bscans_max + 1
                )
            ),
            b_size // 2,
        )
        valid_starts: list[int] = []
        while length >= 1 and not valid_starts:
            valid_starts = [
                start
                for start in range(b_size - length + 1)
                if all(
                    start + length <= lo or start >= hi for lo, hi in occupied_intervals
                )
            ]
            if not valid_starts:
                # This affects only volumes too short for two 40--60 B-scan
                # events. Typical OCTA volumes retain the sampled span.
                length -= 1
        if length < 1:
            raise RuntimeError(
                "Cannot place non-overlapping left/right lateral intervals."
            )
        start = int(rng.choice(valid_starts))
        stop = start + length
        occupied_intervals.append((start, stop))
        widths, slope_endpoint = _width_profile(
            length=length,
            lateral_size=w_size,
            rng=rng,
            config=config,
        )
        for offset, b_index in enumerate(range(start, stop)):
            width = int(widths[offset])
            if side == "left":
                lateral_mask[b_index, :, :width] = True
            else:
                lateral_mask[b_index, :, w_size - width :] = True
        lateral_events.append(
            {
                "side": side,
                "b_start": start,
                "b_stop_exclusive": stop,
                "nominal_width_fraction": config.lateral_width_fraction_each_boundary,
                "slope_endpoint_pixels": slope_endpoint,
                "widths_pixels": widths.tolist(),
            }
        )

    combined = full_mask | lateral_mask
    corrupted = clean.copy()
    corrupted[combined] = np.asarray(config.fill_value, dtype=clean.dtype)
    metadata = {
        "protocol": "octa_motion_corruption_v1",
        "seed": int(seed),
        "shape_bzw": [int(value) for value in clean.shape],
        "mask_convention": "true/1 means missing",
        "config": asdict(config),
        "full_bscan_indices": full_indices,
        "lateral_events": lateral_events,
        "component_masks_may_overlap": True,
        "combined_missing_fraction": float(combined.mean()),
    }
    return CorruptionResult(corrupted, combined, full_mask, lateral_mask, metadata)
