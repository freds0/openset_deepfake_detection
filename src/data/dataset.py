"""FaceForensics++-style frame dataset.

Supports two ways of specifying data:

1. **Folder layout** (recommended for FF++). Frames are pre-extracted / cropped
   and organised per split and per manipulation type::

       <root>/<split>/real/**/*.png              # bona-fide faces  (label 0)
       <root>/<split>/Deepfakes/**/*.png         # forgery domain 1 (label 1)
       <root>/<split>/Face2Face/**/*.png         # forgery domain 2 (label 1)
       <root>/<split>/FaceSwap/**/*.png          # forgery domain 3 (label 1)
       <root>/<split>/NeuralTextures/**/*.png    # forgery domain 4 (label 1)

   The user pre-splits videos into ``train``/``val``/``test`` following the
   official FF++ protocol (720 / 140 / 140 videos).

2. **Manifest CSV** with columns ``path,label,domain[,split]`` for arbitrary
   datasets (e.g. cross-dataset evaluation on CDF, DFDC, WDF, ...).

Each item is a dict with keys ``pixel_values``, ``label`` (binary),
``domain`` (forgery source-domain id; 0 for real) and ``path``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Forgery source-domain ids used by the Forgery Style Mixture module.
# 0 is reserved for real faces; each manipulation type is a distinct domain.
FFPP_DOMAINS: dict[str, int] = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
}

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


@dataclass
class Record:
    """A single frame sample."""

    path: str
    label: int   # 1 = fake, 0 = real
    domain: int  # forgery source-domain id (0 for real)


def _scan_dir(directory: str) -> list[str]:
    files: list[str] = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)


def records_from_faceforensics(
    root: str,
    split: str,
    domain_map: dict[str, int] | None = None,
    classes: list[str] | None = None,
) -> list[Record]:
    """Build records by scanning an FF++-style split folder.

    Args:
        root: Dataset root containing per-split subfolders.
        split: ``"train"``, ``"val"`` or ``"test"``.
        domain_map: Mapping ``{subfolder_name: domain_id}``; ``"real"`` (or any
            entry mapped to 0) is treated as bona-fide. Defaults to
            :data:`FFPP_DOMAINS`.
        classes: Optional subset of subfolder names to include (e.g. the source
            manipulations of a leave-one-manipulation-out split). ``None`` (the
            default) scans every entry of ``domain_map``. Names not present in
            ``domain_map`` are ignored.

    Returns:
        A list of :class:`Record`.
    """
    domain_map = domain_map or FFPP_DOMAINS
    if classes is not None:
        classes = set(classes)
        domain_map = {k: v for k, v in domain_map.items() if k in classes}
        if not domain_map:
            raise ValueError(f"`classes`={sorted(classes)} matched no entry of the domain map")
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    records: list[Record] = []
    for name, domain in domain_map.items():
        sub = os.path.join(split_dir, name)
        if not os.path.isdir(sub):
            continue
        label = 0 if domain == 0 else 1
        for path in _scan_dir(sub):
            records.append(Record(path=path, label=label, domain=domain))
    if not records:
        raise RuntimeError(f"No images found under {split_dir}")
    return records


def records_from_manifest(csv_path: str, split: str | None = None) -> list[Record]:
    """Build records from a manifest CSV (``path,label,domain[,split]``)."""
    df = pd.read_csv(csv_path)
    required = {"path", "label", "domain"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {required}, got {list(df.columns)}")
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    return [
        Record(path=r.path, label=int(r.label), domain=int(r.domain))
        for r in df.itertuples(index=False)
    ]


def oversample_real(records: list[Record], factor: int) -> list[Record]:
    """Replicate real records ``factor`` times to balance positives/negatives.

    Paper (Sec. IV-B): "To balance positive and negative samples during
    training, real faces are augmented fourfold."
    """
    if factor <= 1:
        return records
    reals = [r for r in records if r.label == 0]
    return records + reals * (factor - 1)


class ForgeryFrameDataset(Dataset):
    """Dataset of forgery/real face frames.

    Args:
        records: List of :class:`Record`.
        transform: A callable mapping a PIL image to a tensor.
        face_cropper: Optional :class:`~src.data.face_detection.FaceCropper`.
        jpeg_draft_size: If set, JPEG images are decoded via ``Image.draft``
            at the largest DCT scale that still covers this size (2-8x faster
            decode for large "in the wild" JPEGs, e.g. NTIRE) before the
            final resize in ``transform``. ``None`` (default) decodes at full
            resolution -- required for lossless formats like PNG.
    """

    def __init__(
        self,
        records: list[Record],
        transform,
        face_cropper=None,
        jpeg_draft_size: int | None = None,
    ) -> None:
        self.records = records
        self.transform = transform
        self.face_cropper = face_cropper
        self.jpeg_draft_size = jpeg_draft_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec.path)
        if self.jpeg_draft_size and image.format == "JPEG":
            # Decodes at the largest DCT-scaled factor that still covers the
            # target size; never upscales, so quality loss is sub-visual once
            # downstream-resized to a smaller model input.
            image.draft("RGB", (self.jpeg_draft_size, self.jpeg_draft_size))
        image = image.convert("RGB")
        if self.face_cropper is not None:
            image = self.face_cropper(image)
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "domain": torch.tensor(rec.domain, dtype=torch.long),
            "path": rec.path,
        }
