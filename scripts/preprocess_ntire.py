"""NTIRE RobustAIGenDetection preprocessing: build a manifest CSV for
:class:`src.data.datamodule.ForgeryDataModule` (``source="manifest"``).

Per the dataset card (deepfakesMSU/NTIRE-RobustAIGenDetection-train), each
shard directory contains::

    shard_N/images/<name>.jpg
    shard_N/labels.csv       # columns: (index), image_name, label (0=real, 1=fake)

Only the train set ships with real labels at this stage of the challenge (the
separate ``-val`` dataset's ``val_labels.csv`` / ``val_hard_labels.csv`` are
NOT ground truth per its README -- it is meant for unlabeled challenge
submissions, see ``predict.py``). So this script carves train/val/test out of
the labeled train pool itself: a stratified random split (by label) with a
fixed seed for reproducibility.

Output manifest columns: ``path,label,domain,split``
  - domain 0 = real, domain 1 = fake (NTIRE has no per-generator domain info,
    unlike FF++'s 4 manipulation domains).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a manifest CSV for the NTIRE dataset")
    p.add_argument("--data-root", default="data/NTIRE-RobustAIGenDetection-train")
    p.add_argument("--out", default="data/ntire_manifest.csv")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_root, "shard_*"))
        if os.path.isfile(os.path.join(d, "labels.csv"))
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_*/labels.csv found under {args.data_root}")

    rows = []
    for shard_dir in tqdm(shard_dirs, desc="shards", unit="shard"):
        df = pd.read_csv(os.path.join(shard_dir, "labels.csv"))
        for name, label in zip(df["image_name"], df["label"]):
            rows.append((os.path.join(shard_dir, "images", name), int(label)))
    manifest = pd.DataFrame(rows, columns=["path", "label"])
    manifest["domain"] = manifest["label"]  # 0 = real, 1 = fake (no per-generator info)

    # Stratified random split by label so val/test keep the same class balance.
    rng = np.random.default_rng(args.seed)
    split = pd.Series("train", index=manifest.index)
    for label in manifest["label"].unique():
        idx = manifest.index[manifest["label"] == label].to_numpy()
        rng.shuffle(idx)
        n_val = int(len(idx) * args.val_frac)
        n_test = int(len(idx) * args.test_frac)
        split.loc[idx[:n_val]] = "val"
        split.loc[idx[n_val:n_val + n_test]] = "test"
    manifest["split"] = split

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifest.to_csv(args.out, index=False)

    print(f"\n=== Done: {len(manifest)} rows -> {args.out} ===")
    print(manifest.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
