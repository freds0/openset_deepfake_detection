"""SigLIP 2 image preprocessing and (optional) data augmentation.

SigLIP / SigLIP 2 normalise inputs to the ``[-1, 1]`` range (mean = std = 0.5)
and resize with bicubic interpolation (SigLIP 2 paper, Sec. 2.1; matches the
``Siglip2ImageProcessor`` defaults).

The OSDFD paper uses **no** data augmentation during training ("OSDFD does not
employ data augmentations during training"), so augmentation is **disabled by
default**. An advanced, forgery-aware augmentation pipeline is available and can
be switched on via config (``data.augmentation.enabled=true``). It favours
transforms that are relevant to face-forgery detection -- JPEG recompression and
down/up-scaling (compression artefacts), Gaussian blur, colour jitter, random
resized crop and random erasing (cutout) -- so the model can be made robust to
the perturbations studied in the paper's robustness section without changing the
default paper-faithful behaviour.
"""

from __future__ import annotations

import io
import random

import torchvision.transforms as T
from PIL import Image

# SigLIP normalisation constants (range [-1, 1]).
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# Default augmentation settings (all effectively off unless `enabled=true`).
DEFAULT_AUG: dict = {
    "enabled": False,
    "hflip": 0.5,               # prob. of horizontal flip
    "color_jitter": 0.0,        # brightness/contrast/saturation strength (0 = off)
    "hue": 0.0,                 # hue jitter strength
    "random_resized_crop": False,
    "rrc_scale": (0.8, 1.0),    # area scale range for RandomResizedCrop
    "gaussian_blur": 0.0,       # prob. of Gaussian blur
    "blur_sigma": (0.1, 2.0),   # sigma range for the blur
    "jpeg": 0.0,                # prob. of random JPEG recompression
    "jpeg_quality": (40, 90),   # JPEG quality range
    "downscale": 0.0,           # prob. of random down+up scaling
    "downscale_range": (0.25, 0.75),  # relative scale for the downscale step
    "random_erasing": 0.0,      # prob. of random erasing (cutout, on tensor)
}


class RandomJPEG:
    """Randomly recompress a PIL image as JPEG (compression artefacts)."""

    def __init__(self, p: float, quality: tuple[int, int]) -> None:
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = random.randint(self.quality[0], self.quality[1])
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscale:
    """Randomly down-sample then up-sample a PIL image (blockiness/blur)."""

    def __init__(self, p: float, scale: tuple[float, float]) -> None:
        self.p = p
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(self.scale[0], self.scale[1])
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def _merge_aug(augmentation: dict | None) -> dict:
    cfg = dict(DEFAULT_AUG)
    if augmentation:
        cfg.update({k: v for k, v in augmentation.items() if v is not None})
    return cfg


def build_transform(
    image_size: int = 224,
    train: bool = False,
    augmentation: dict | None = None,
    resize_mode: str = "squash",
) -> T.Compose:
    """Build a SigLIP 2 preprocessing pipeline, optionally with augmentation.

    Args:
        image_size: Target square resolution (224 for the default checkpoint).
        train: Whether this is the training pipeline.
        augmentation: Augmentation config dict (see :data:`DEFAULT_AUG`). Only
            applied when ``train`` is True and ``augmentation['enabled']`` is
            True; otherwise a plain resize + normalise pipeline is returned
            (paper-faithful default).
        resize_mode: ``"squash"`` (default) resizes directly to
            ``(image_size, image_size)``, matching the SigLIP preprocessor and
            ignoring aspect ratio. ``"crop"`` instead uses a
            ``RandomResizedCrop`` at train time and a shorter-side resize +
            center crop at val/test time -- preserves aspect ratio, at the
            cost of not seeing the full frame. Ignored when the augmentation
            pipeline's own ``random_resized_crop`` is active (train only) to
            avoid stacking two crops.

    Returns:
        A ``torchvision`` transform mapping a PIL image to a normalised tensor.
    """
    if resize_mode not in ("squash", "crop"):
        raise ValueError(f"Unknown resize_mode: {resize_mode}")
    aug = _merge_aug(augmentation)
    use_aug = train and aug["enabled"]

    ops: list = []

    # Spatial resize. The augmentation pipeline's own RandomResizedCrop (train
    # only, when enabled) wins over `resize_mode` so the two never stack.
    if use_aug and aug["random_resized_crop"]:
        ops.append(
            T.RandomResizedCrop(
                image_size,
                scale=tuple(aug["rrc_scale"]),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    elif resize_mode == "crop" and train:
        ops.append(
            T.RandomResizedCrop(
                image_size,
                scale=tuple(aug["rrc_scale"]),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    elif resize_mode == "crop":
        ops.append(T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC))
        ops.append(T.CenterCrop(image_size))
    else:
        ops.append(
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        )

    if use_aug:
        if aug["hflip"] > 0:
            ops.append(T.RandomHorizontalFlip(p=aug["hflip"]))
        if aug["color_jitter"] > 0 or aug["hue"] > 0:
            cj = aug["color_jitter"]
            ops.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=aug["hue"]))
        if aug["gaussian_blur"] > 0:
            ops.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=5, sigma=tuple(aug["blur_sigma"]))],
                    p=aug["gaussian_blur"],
                )
            )
        if aug["jpeg"] > 0:
            ops.append(RandomJPEG(aug["jpeg"], tuple(aug["jpeg_quality"])))
        if aug["downscale"] > 0:
            ops.append(RandomDownscale(aug["downscale"], tuple(aug["downscale_range"])))

    ops += [
        T.ToTensor(),
        T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ]

    # Random erasing operates on the normalised tensor (cutout).
    if use_aug and aug["random_erasing"] > 0:
        ops.append(T.RandomErasing(p=aug["random_erasing"]))

    return T.Compose(ops)
