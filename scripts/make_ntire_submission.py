"""Generate a NTIRE 2026 challenge submission CSV from a trained checkpoint.

Runs batched inference over a folder of challenge images (e.g. the official,
unlabeled ``val_images/`` or ``val_images_hard/``) and writes a
``image_name,pred`` CSV in the format expected by the challenge submission
(``pred`` is P(fake) in [0, 1]; see the dataset card's ``predict_on_val``
snippet).

Usage:
    python scripts/make_ntire_submission.py --ckpt outputs/.../last.ckpt \
        --images-dir data/NTIRE-RobustAIGenDetection-val/val_images \
        --out submission.csv

    python scripts/make_ntire_submission.py --ckpt outputs/.../last.ckpt \
        --images-dir data/NTIRE-RobustAIGenDetection-val/val_images_hard \
        --out submission_hard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.inference.predictor import OSDFDPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a NTIRE challenge submission CSV")
    p.add_argument("--ckpt", required=True, help="Path to a Lightning .ckpt")
    p.add_argument(
        "--images-dir",
        default="data/NTIRE-RobustAIGenDetection-val/val_images",
        help="Folder of challenge images to predict on",
    )
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda | cpu (auto by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = OSDFDPredictor(ckpt_path=args.ckpt, device=args.device)
    results = predictor.predict_folder(
        args.images_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    df = pd.DataFrame(
        {
            "image_name": [Path(r.path).stem for r in results],
            "pred": [r.probability for r in results],
        }
    )
    df.to_csv(args.out)
    print(f"{len(df)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
