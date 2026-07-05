"""OSDFD training entry point (PyTorch Lightning + Hydra).

Usage:
    python train.py                                  # defaults
    python train.py data.root=/data/ffpp_frames
    python train.py optimizer.lr=1e-4 fsm.enabled=false trainer.precision=32-true

The Forgery Style Mixture module is active only during training; validation uses
AUC as the primary metric (paper Sec. IV-A). Only LoRA, the CDC adapter and the
classification head are optimised.
"""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datamodule import ForgeryDataModule
from src.lightning.module import OSDFDLightningModule
from src.utils.lightning_setup import build_callbacks, build_loggers, build_trainer
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Bake the concrete Hydra run directory into the config so checkpoints do
    # not store an unresolvable `${hydra:...}` interpolation.
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    datamodule = ForgeryDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    model = OSDFDLightningModule(cfg)

    loggers = build_loggers(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, loggers, callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
