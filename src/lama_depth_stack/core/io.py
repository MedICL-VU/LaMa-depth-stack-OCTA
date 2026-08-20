"""Volume I/O with explicit preservation of `(B,Z,W)` array order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def load_volume(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        volume = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported volume format `{path.suffix}` for {path}.")
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected `(B,Z,W)` volume at {path}, got {volume.shape}.")
    return volume


def save_volume(
    path: Path, volume: np.ndarray, *, dtype: np.dtype | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.asarray(volume)
    if dtype is not None:
        target = np.dtype(dtype)
        if np.issubdtype(target, np.integer):
            limits = np.iinfo(target)
            output = np.clip(np.rint(output), limits.min, limits.max).astype(target)
        else:
            output = output.astype(target)
    if path.suffix.lower() == ".npy":
        np.save(path, output)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, output)
    else:
        raise ValueError(f"Unsupported output format `{path.suffix}` for {path}.")
