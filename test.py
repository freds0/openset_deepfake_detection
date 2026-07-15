"""OSDFD evaluation entry point.

Runs in-domain, cross-manipulation or cross-dataset evaluation depending on the
data config you point it at, reporting ACC / AUC / F1 / AP / EER / FPR / FNR and
writing per-image predictions to CSV.

Usage:
    # In-domain / cross-manipulation (FF++ test split of the trained root):
    python test.py ckpt_path=outputs/.../checkpoints/last.ckpt

    # Cross-dataset (e.g. Celeb-DF via a manifest CSV):
    python test.py ckpt_path=... data.source=manifest data.manifest=cdf.csv
"""

from __future__ import annotations

import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datamodule import ForgeryDataModule
from src.lightning.module import OSDFDLightningModule
from src.training.metrics import compute_metrics, log_figures
from src.utils.lightning_setup import build_loggers
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")  # TF32 in fp32 matmuls (Ampere+/B200)
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    seed_everything(cfg.seed)
    if cfg.ckpt_path is None:
        raise ValueError("Set ckpt_path=/path/to/model.ckpt to evaluate.")

    datamodule = ForgeryDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    model = OSDFDLightningModule.load_for_inference(cfg.ckpt_path)

    import lightning as L

    loggers = build_loggers(cfg)
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        precision=cfg.trainer.precision,
        logger=loggers,
    )

    # Single pass over the test split: per-image predictions -> CSV, metrics
    # and figures are derived from the same predictions (no second pass).
    predictions = trainer.predict(model, datamodule=datamodule)
    paths, probs, preds, labels = [], [], [], []
    for batch in predictions:
        if batch["path"] is not None:
            paths.extend(list(batch["path"]))
        probs.append(batch["prob"].numpy())
        preds.append(batch["pred"].numpy())
        if "label" in batch:
            labels.append(batch["label"].numpy())

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    df = {"prob": probs, "pred": preds}
    if paths:
        df = {"path": paths, **df}
    if labels:
        y_true = np.concatenate(labels)
        df["label"] = y_true
        metrics = compute_metrics(y_true, probs)
        print("\n=== Evaluation metrics ===")
        for k, v in metrics.items():
            print(f"{k:>5}: {v:.4f}")
        for logger in loggers:
            logger.log_metrics({f"test/{k}": v for k, v in metrics.items()}, step=0)
            log_figures(logger, y_true, probs, step=0, prefix="test")

    out_csv = os.path.join(cfg.output_dir, "predictions.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    pd.DataFrame(df).to_csv(out_csv, index=False)
    print(f"\nPredictions written to {out_csv}")


if __name__ == "__main__":
    main()
