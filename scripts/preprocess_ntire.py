"""NTIRE RobustAIGenDetection preprocessing: build a manifest CSV for
:class:`src.data.datamodule.ForgeryDataModule` (``source="manifest"``).

Per the dataset card (deepfakesMSU/NTIRE-RobustAIGenDetection-train/-val), each
shard directory contains::

    shard_N/images/<name>.jpg
    shard_N/labels.csv       # columns: (index), image_name, label (0=real, 1=fake)

The challenge ships train and val as separate HuggingFace datasets (download
with ``scripts/download/download_ntire_dataset.py`` and
``download_ntire_val_dataset.py``); the official test split is not released
yet. All of ``--train-dir`` becomes the ``train`` split. ``--val-dir`` is
split by label into ``val``/``test`` (``--test-frac-of-val``, default 50/50)
until a real test set ships.

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
    p.add_argument("--train-dir", default="data/NTIRE-RobustAIGenDetection-train")
    p.add_argument("--val-dir", default="data/NTIRE-RobustAIGenDetection-val")
    p.add_argument("--out", default="data/ntire_manifest.csv")
    p.add_argument("--test-frac-of-val", type=float, default=0.5,
                   help="fraction of --val-dir routed to `test` (no official test split yet)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_shards(data_dir: str) -> pd.DataFrame:
    shard_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "shard_*"))
        if os.path.isfile(os.path.join(d, "labels.csv"))
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_*/labels.csv found under {data_dir}")

    rows = []
    for shard_dir in tqdm(shard_dirs, desc=f"shards ({os.path.basename(data_dir)})", unit="shard"):
        df = pd.read_csv(os.path.join(shard_dir, "labels.csv"))
        for name, label in zip(df["image_name"], df["label"]):
            rows.append((os.path.join(shard_dir, "images", name), int(label)))
    return pd.DataFrame(rows, columns=["path", "label"])


def main() -> None:
    args = parse_args()

    train_df = load_shards(args.train_dir)
    train_df["split"] = "train"

    val_df = load_shards(args.val_dir)
    rng = np.random.default_rng(args.seed)
    val_df["split"] = "val"
    for label in val_df["label"].unique():
        idx = val_df.index[val_df["label"] == label].to_numpy()
        rng.shuffle(idx)
        n_test = int(len(idx) * args.test_frac_of_val)
        val_df.loc[idx[:n_test], "split"] = "test"

    manifest = pd.concat([train_df, val_df], ignore_index=True)
    manifest["domain"] = manifest["label"]  # 0 = real, 1 = fake (no per-generator info)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifest.to_csv(args.out, index=False)

    print(f"\n=== Done: {len(manifest)} rows -> {args.out} ===")
    print(manifest.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
