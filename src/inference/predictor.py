"""Standalone inference for single images or folders.

Loads a trained OSDFD checkpoint and predicts forgery probability / label for
one image or every image in a directory. The Forgery Style Mixture module is
always disabled at inference time (paper Sec. III-C).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.face_detection import FaceCropper
from ..data.transforms import build_transform
from ..lightning.module import OSDFDLightningModule

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


class _ImageFolderDataset(Dataset):
    """Loads + transforms a fixed list of image paths, returning their index.

    The index is threaded through so :meth:`OSDFDPredictor.predict_folder` can
    reassemble per-image :class:`Prediction` objects after batched inference.
    """

    def __init__(self, paths: list[str], transform, cropper: FaceCropper | None = None) -> None:
        self.paths = paths
        self.transform = transform
        self.cropper = cropper

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = Image.open(self.paths[i]).convert("RGB")
        if self.cropper is not None:
            image = self.cropper(image)
        return self.transform(image), i


@dataclass
class Prediction:
    """One image prediction."""

    path: str
    label: str          # "fake" or "real"
    probability: float  # P(fake)
    confidence: float   # max(P(fake), 1 - P(fake))
    logit: float


class OSDFDPredictor:
    """Thin inference wrapper around a trained LightningModule.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` checkpoint.
        device: Torch device string (default: cuda if available).
        image_size: Model input resolution. ``None`` (default) resolves it
            from the checkpoint's own ``backbone.image_size`` (falls back to
            224 with a warning if that hparam is missing).
        use_face_crop: Enable face detection before preprocessing.
        face_backend: Face-detection backend.
        face_margin: Crop enlargement factor.
        threshold: Decision threshold for the ``"fake"``/``"real"`` label.
            Defaults to the checkpoint's EER-calibrated threshold (see
            ``OSDFDLightningModule.calibrated_threshold``), falling back to
            0.5 for older checkpoints that predate calibration.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        image_size: int | None = None,
        use_face_crop: bool = False,
        face_backend: str = "opencv",
        face_margin: float = 1.3,
        threshold: float | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.module = OSDFDLightningModule.load_for_inference(ckpt_path, map_location=self.device)
        self.module.eval()
        self.module.to(self.device)
        if image_size is None:
            try:
                image_size = int(self.module.cfg.backbone.image_size)
            except Exception:
                print("[OSDFDPredictor] backbone.image_size missing from checkpoint; defaulting to 224")
                image_size = 224
        self.transform = build_transform(image_size, train=False)
        self.cropper = FaceCropper(face_backend, face_margin) if use_face_crop else None
        self.threshold = (
            threshold if threshold is not None else getattr(self.module, "calibrated_threshold", 0.5)
        )

    @torch.no_grad()
    def predict_image(self, path: str) -> Prediction:
        image = Image.open(path).convert("RGB")
        if self.cropper is not None:
            image = self.cropper(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        out = self.module.model(x, apply_fsm=False)
        logit = float(out.logits.item())
        prob = float(torch.sigmoid(out.logits).item())
        return Prediction(
            path=path,
            label="fake" if prob >= self.threshold else "real",
            probability=prob,
            confidence=max(prob, 1.0 - prob),
            logit=logit,
        )

    @torch.no_grad()
    def predict_folder(
        self,
        folder: str,
        batch_size: int = 64,
        num_workers: int = 8,
    ) -> list[Prediction]:
        """Batched inference over every image found under ``folder``.

        Args:
            folder: Directory to scan recursively for images.
            batch_size: Inference batch size.
            num_workers: Dataloader worker processes.

        Returns:
            One :class:`Prediction` per image, in the same sorted-path order
            as the single-image ``predict_image`` would produce.
        """
        files: list[str] = []
        for ext in _IMG_EXTS:
            files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        files = sorted(files)

        dataset = _ImageFolderDataset(files, self.transform, self.cropper)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        results: list[Prediction | None] = [None] * len(files)
        for pixel_values, indices in loader:
            pixel_values = pixel_values.to(self.device)
            out = self.module.model(pixel_values, apply_fsm=False)
            probs = torch.sigmoid(out.logits)
            for logit, prob, idx in zip(out.logits.tolist(), probs.tolist(), indices.tolist()):
                results[idx] = Prediction(
                    path=files[idx],
                    label="fake" if prob >= self.threshold else "real",
                    probability=prob,
                    confidence=max(prob, 1.0 - prob),
                    logit=logit,
                )
        return results
