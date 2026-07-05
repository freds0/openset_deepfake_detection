"""Build a manifest CSV for the NTIRE 2026 Robust AI-generated Image Detection
In The Wild challenge dataset, so it can be trained through the existing
:class:`src.data.datamodule.ForgeryDataModule` (``source: manifest``).

Input layout (as downloaded by ``scripts/download/download_ntire_dataset.py``)::

    <root>/shard_0/images/<image_name>.jpg
    <root>/shard_0/labels.csv          # columns: ,image_name,label
    <root>/shard_1/...
    ...

``label`` is binary (0 = real, 1 = fake); there is no forgery-source/generator
column, so ``domain`` is set equal to ``label`` (single fake domain). With only
one fake domain, the Forgery Style Mixture module has nothing to mix between
same-domain fakes (see ``src/models/fsm.py::_domain_shuffle``) -- disable it
for this dataset (``fsm=disabled``) rather than leaving it silently inert.

Split is assigned randomly *within each shard*, stratified by label so
train/val/test keep the same real/fake ratio.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def build_manifest(root: str, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    shard_dirs = sorted(glob.glob(os.path.join(root, "shard_*")))
    if not shard_dirs:
        raise RuntimeError(f"No shard_* directories found under {root}")

    rng = np.random.default_rng(seed)
    rows = []
    for shard_dir in shard_dirs:
        shard_name = os.path.basename(shard_dir)
        labels_csv = os.path.join(shard_dir, "labels.csv")
        df = pd.read_csv(labels_csv, index_col=0)

        splits = np.empty(len(df), dtype=object)
        for label in df["label"].unique():
            idx = df.index[df["label"] == label].to_numpy()
            rng.shuffle(idx)
            n_val = int(round(len(idx) * val_frac))
            n_test = int(round(len(idx) * test_frac))
            pos = df.index.get_indexer(idx)
            splits[pos[:n_val]] = "val"
            splits[pos[n_val : n_val + n_test]] = "test"
            splits[pos[n_val + n_test :]] = "train"

        for (_, r), split in zip(df.iterrows(), splits):
            path = os.path.join(root, shard_name, "images", r["image_name"])
            label = int(r["label"])
            rows.append({"path": path, "label": label, "domain": label, "split": split})

    return pd.DataFrame(rows, columns=["path", "label", "domain", "split"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/NTIRE-RobustAIGenDetection-train")
    parser.add_argument("--output", default="data/ntire_manifest.csv")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_manifest(args.root, args.val_frac, args.test_frac, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    manifest.to_csv(args.output, index=False)

    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
