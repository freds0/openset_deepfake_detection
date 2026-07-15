"""LightningDataModule for FaceForensics++-style forgery detection.

Wires the :class:`ForgeryFrameDataset` into PyTorch Lightning, handling the
train / val / test splits, SigLIP 2 preprocessing, optional face cropping and
the paper's 4x real-face oversampling for class balance.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import (
    ForgeryFrameDataset,
    Record,
    oversample_real,
    records_from_faceforensics,
    records_from_manifest,
)
from .face_detection import FaceCropper
from .transforms import build_transform


class ForgeryDataModule(L.LightningDataModule):
    """Data module for cross-manipulation / cross-dataset forgery detection.

    Args:
        source: ``"faceforensics"`` (folder scan) or ``"manifest"`` (CSV).
        root: Dataset root (folder mode).
        manifest: CSV path (manifest mode).
        image_size: Model input resolution.
        batch_size: Batch size.
        num_workers: Dataloader workers.
        real_oversample: Replication factor for real training frames (paper: 4).
            Ignored (must be ``1``) when ``balance_sampler`` is True.
        balance_sampler: Balance real/fake per batch with a
            ``WeightedRandomSampler`` over the (non-duplicated) train records,
            instead of physically duplicating real records via
            ``real_oversample``. Mutually exclusive with ``real_oversample>1``.
        use_face_crop: Enable face detection / cropping.
        face_backend: Face-detection backend (see :class:`FaceCropper`).
        face_margin: Crop enlargement factor (paper: 1.3).
        augmentation: Advanced augmentation config (see
            :data:`src.data.transforms.DEFAULT_AUG`). Disabled by default to
            match the paper; only applied to the training split when
            ``augmentation['enabled']`` is True.
        resize_mode: ``"squash"`` (default, paper-faithful) or ``"crop"`` --
            see :func:`src.data.transforms.build_transform`.
        jpeg_draft_size: Fast approximate JPEG decode target size (see
            :class:`~src.data.dataset.ForgeryFrameDataset`); ``None`` (default)
            decodes at full resolution.
        train_classes: Optional subset of manipulation subfolders (folder mode
            only) used as the *source* domains -- restricts both the train and
            the val splits. Used for the leave-one-manipulation-out (open-set /
            cross-manipulation) protocol of the OSDFD paper (Tables I-II), where
            val must contain only the seen manipulations. ``None`` = all.
        test_classes: Optional subset of manipulation subfolders for the *test*
            split -- the held-out target manipulation(s) plus ``real``. ``None``
            = all. Both flags are ignored in manifest mode (no subfolders).
        domain_map: Optional ``{subfolder: domain_id}`` override (folder mode).
        pin_memory: Pin dataloader memory.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches pre-loaded per worker (``None`` = torch
            default of 2). Raise (e.g. 4) to smooth I/O latency spikes when
            the storage medium has variable read latency.
    """

    def __init__(
        self,
        source: str = "faceforensics",
        root: str | None = None,
        manifest: str | None = None,
        image_size: int = 224,
        batch_size: int = 48,
        num_workers: int = 8,
        real_oversample: int = 4,
        balance_sampler: bool = False,
        use_face_crop: bool = False,
        face_backend: str = "opencv",
        face_margin: float = 1.3,
        augmentation: dict | None = None,
        resize_mode: str = "squash",
        jpeg_draft_size: int | None = None,
        train_classes: list[str] | None = None,
        test_classes: list[str] | None = None,
        domain_map: dict[str, int] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
    ) -> None:
        super().__init__()
        if balance_sampler and real_oversample > 1:
            raise ValueError(
                "balance_sampler=True already balances classes via sampling; "
                "set real_oversample=1 (duplication would double-balance on top of it)."
            )
        self.save_hyperparameters()
        self._train: ForgeryFrameDataset | None = None
        self._val: ForgeryFrameDataset | None = None
        self._test: ForgeryFrameDataset | None = None
        self._train_sampler: WeightedRandomSampler | None = None

    def _load_records(self, split: str, classes: list[str] | None = None) -> list[Record]:
        h = self.hparams
        if h.source == "faceforensics":
            if h.root is None:
                raise ValueError("`root` is required for source='faceforensics'")
            return records_from_faceforensics(
                h.root, split, domain_map=h.domain_map, classes=classes
            )
        if h.source == "manifest":
            if h.manifest is None:
                raise ValueError("`manifest` is required for source='manifest'")
            return records_from_manifest(h.manifest, split=split)
        raise ValueError(f"Unknown data source: {h.source}")

    def _cropper(self) -> FaceCropper | None:
        h = self.hparams
        if not h.use_face_crop:
            return None
        return FaceCropper(backend=h.face_backend, margin=h.face_margin)

    def setup(self, stage: str | None = None) -> None:
        h = self.hparams
        cropper = self._cropper()

        # Source manipulations (train_classes) gate both train and val so that
        # validation stays open-set (no target manipulation leakage); the
        # held-out target manipulation lives in test (test_classes).
        if stage in (None, "fit"):
            if h.balance_sampler:
                train_records = self._load_records("train", classes=h.train_classes)
                labels = np.array([r.label for r in train_records])
                class_counts = np.bincount(labels)
                weights = 1.0 / class_counts[labels]
                self._train_sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(weights, dtype=torch.double),
                    num_samples=len(train_records),
                    replacement=True,
                )
            else:
                train_records = oversample_real(
                    self._load_records("train", classes=h.train_classes), h.real_oversample
                )
                self._train_sampler = None
            self._train = ForgeryFrameDataset(
                train_records,
                transform=build_transform(
                    h.image_size, train=True, augmentation=h.augmentation, resize_mode=h.resize_mode
                ),
                face_cropper=cropper,
                jpeg_draft_size=h.jpeg_draft_size,
            )
            self._val = ForgeryFrameDataset(
                self._load_records("val", classes=h.train_classes),
                transform=build_transform(h.image_size, train=False, resize_mode=h.resize_mode),
                face_cropper=cropper,
                jpeg_draft_size=h.jpeg_draft_size,
            )
        if stage in ("test", "predict") or stage is None:
            self._test = ForgeryFrameDataset(
                self._load_records("test", classes=h.test_classes),
                transform=build_transform(h.image_size, train=False, resize_mode=h.resize_mode),
                face_cropper=cropper,
                jpeg_draft_size=h.jpeg_draft_size,
            )
        if stage == "validate" and self._val is None:
            self._val = ForgeryFrameDataset(
                self._load_records("val", classes=h.train_classes),
                transform=build_transform(h.image_size, train=False, resize_mode=h.resize_mode),
                face_cropper=cropper,
                jpeg_draft_size=h.jpeg_draft_size,
            )

    def _loader(
        self,
        dataset: ForgeryFrameDataset,
        shuffle: bool,
        sampler: WeightedRandomSampler | None = None,
    ) -> DataLoader:
        h = self.hparams
        return DataLoader(
            dataset,
            batch_size=h.batch_size,
            shuffle=shuffle if sampler is None else False,  # mutually exclusive with sampler
            sampler=sampler,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            drop_last=shuffle or sampler is not None,
            persistent_workers=h.persistent_workers and h.num_workers > 0,
            prefetch_factor=h.prefetch_factor if h.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=self._train_sampler is None, sampler=self._train_sampler)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        # Predict over the test split (used by test.py to write per-image CSVs).
        return self._loader(self._test, shuffle=False)
