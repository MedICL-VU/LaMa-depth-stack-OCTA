"""Optional frozen Big-LaMa backend and backend-independent plane adapter."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

BIG_LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


@runtime_checkable
class InpaintBackend(Protocol):
    """Minimal interface needed by canonical fixed-depth reconstruction."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def inpaint(self, image_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Return an RGB uint8 image, optionally padded on bottom/right."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SimpleLamaBackend:
    """Persistent frozen Big-LaMa adapter using `simple-lama-inpainting`.

    PyTorch and the optional dependency are imported lazily so manifest,
    preprocessing, and evaluation utilities remain lightweight.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        checkpoint_path: Path | None = None,
        expected_sha256: str = BIG_LAMA_SHA256,
        verify_checkpoint: bool = True,
    ) -> None:
        checkpoint = checkpoint_path
        if checkpoint is None:
            checkpoint = Path(
                os.environ.get(
                    "LAMA_MODEL",
                    Path.home()
                    / ".cache"
                    / "torch"
                    / "hub"
                    / "checkpoints"
                    / "big-lama.pt",
                )
            )
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Big-LaMa checkpoint not found. Pass `--checkpoint` or set `LAMA_MODEL`. "
                f"Resolved path: {checkpoint}"
            )
        digest = sha256_file(checkpoint)
        if verify_checkpoint and digest != expected_sha256:
            raise RuntimeError(
                f"Big-LaMa checkpoint hash mismatch: expected {expected_sha256}, got {digest}. "
                f"Path: {checkpoint}"
            )

        try:
            import torch
            from simple_lama_inpainting import SimpleLama  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install the optional LaMa dependencies with `pip install -e '.[lama]'`."
            ) from exc

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch reports no available CUDA device."
            )

        # simple-lama-inpainting selects its TorchScript file through this
        # environment variable. Set it only during construction so the file
        # whose provenance we record is exactly the file loaded by the model.
        previous_model_path = os.environ.get("LAMA_MODEL")
        os.environ["LAMA_MODEL"] = str(checkpoint)
        try:
            self._model = SimpleLama(device=requested_device)
        finally:
            if previous_model_path is None:
                os.environ.pop("LAMA_MODEL", None)
            else:
                os.environ["LAMA_MODEL"] = previous_model_path
        self._metadata = {
            "backend": "simple_lama_inpainting",
            "device": str(requested_device),
            "model_eval_mode": bool(not getattr(self._model.model, "training", True)),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_size_bytes": checkpoint.stat().st_size
            if checkpoint.is_file()
            else None,
            "mask_convention": "255/white means missing; 0/black means observed",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def inpaint(self, image_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        result = self._model(Image.fromarray(image_rgb_u8), Image.fromarray(mask_u8))
        return np.asarray(result.convert("RGB"), dtype=np.uint8)


def prepare_plane_inputs(
    plane_01: np.ndarray, missing_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one `(B,W)` plane and create LaMa's white-missing mask."""

    plane = np.asarray(plane_01, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    if plane.ndim != 2 or plane.shape != missing.shape:
        raise ValueError(
            f"Plane/mask must share `(B,W)` shape, got {plane.shape}/{missing.shape}."
        )
    if not np.isfinite(plane).all():
        raise ValueError("Model-facing plane contains non-finite values.")
    gray_u8 = np.rint(np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_rgb_u8 = np.repeat(gray_u8[..., None], 3, axis=-1)
    mask_u8 = np.where(missing, 255, 0).astype(np.uint8)
    return image_rgb_u8, mask_u8


def inpaint_plane(
    backend: InpaintBackend,
    plane_01: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run one plane, crop backend padding, and average RGB in floating point."""

    image_rgb_u8, mask_u8 = prepare_plane_inputs(plane_01, missing_mask)
    output = np.asarray(backend.inpaint(image_rgb_u8, mask_u8))
    if output.ndim != 3 or output.shape[-1] != 3 or output.dtype != np.uint8:
        raise RuntimeError(
            f"Backend must return RGB uint8, got {output.shape}/{output.dtype}."
        )

    target_b, target_w = plane_01.shape
    returned_b, returned_w = output.shape[:2]
    if returned_b < target_b or returned_w < target_w:
        raise RuntimeError(
            f"Backend output {(returned_b, returned_w)} is smaller than input {(target_b, target_w)}."
        )
    crop = {
        "bottom_padding_cropped": int(returned_b - target_b),
        "right_padding_cropped": int(returned_w - target_w),
    }
    output = output[:target_b, :target_w, :]
    return output.astype(np.float32).mean(axis=2) / 255.0, crop
