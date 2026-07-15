"""Assign k-means pseudo-domains to NTIRE fakes so FSM can pair informed
domains instead of falling back to random pairing (Fase 2, item 2.1 of
PLAN_v0.1.md).

NTIRE's ``labels.csv`` has no per-generator column, so
``scripts/preprocess_ntire.py`` sets ``domain = label`` (a single fake
domain) and the Forgery Style Mixture module (``src/models/fsm.py``) falls
back to random pairing among fakes. This script instead clusters fake images
by their frozen-backbone embedding (a proxy for "generator style") and writes
those cluster ids as the ``domain`` column, so FSM can go back to pairing
across *distinct* pseudo-domains, as it does natively on FF++.

Usage:
    python scripts/assign_pseudo_domains.py \
        --manifest data/ntire_manifest.csv --out data/ntire_manifest_k8.csv --k 8

Train with the result via:
    ./scripts/train.sh data=ntire data.manifest=data/ntire_manifest_k8.csv \
        fsm.single_domain_fallback=off
(``off`` makes any single-domain regression fail visibly via
``train/fsm_fired`` dropping to 0, instead of silently degrading to the
random-pairing fallback.)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_transform
from src.models.backbone import Siglip2Backbone


class _ManifestImageDataset(Dataset):
    """Loads + transforms images from a list of manifest paths."""

    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign k-means pseudo-domains to NTIRE fakes")
    p.add_argument("--manifest", default="data/ntire_manifest.csv")
    p.add_argument("--out", default="data/ntire_manifest_k8.csv")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--backbone", default="google/siglip2-base-patch16-224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(
    paths: list[str],
    backbone: Siglip2Backbone,
    device: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """MAP-pooled, frozen-backbone embedding for every image in ``paths``."""
    transform = build_transform(image_size, train=False)
    loader = DataLoader(
        _ManifestImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    embeddings = []
    for batch in loader:
        batch = batch.to(device, dtype=torch.bfloat16)
        pooled = backbone.pool(backbone(batch))
        embeddings.append(pooled.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)

    backbone = Siglip2Backbone(model_name=args.backbone, freeze=True)
    backbone = backbone.to(args.device, dtype=torch.bfloat16)
    backbone.eval()

    # Domain is used only during training, but computed for every split so
    # the manifest stays consistent if val/test are ever re-used for FSM.
    fake_mask = (manifest["label"] == 1).to_numpy()
    fake_paths = manifest.loc[fake_mask, "path"].tolist()

    embeddings = extract_embeddings(
        fake_paths, backbone, args.device, args.image_size, args.batch_size, args.num_workers
    )
    # L2-normalise so k-means clusters on embedding *direction* (style),
    # not magnitude.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    manifest["domain"] = 0
    manifest.loc[fake_mask, "domain"] = cluster_ids + 1  # 1..k; 0 stays real-only

    manifest.to_csv(args.out, index=False)

    print(f"=== Done: {len(manifest)} rows -> {args.out} ===")
    for cluster_id, size in pd.Series(cluster_ids).value_counts().sort_index().items():
        print(f"domain {cluster_id + 1}: {size}")


if __name__ == "__main__":
    main()
