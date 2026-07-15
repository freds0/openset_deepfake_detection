"""Exponential Moving Average (EMA) of the trainable parameters.

Fase 2 ablation (item 2.5 of PLAN_v0.1.md): maintains a shadow copy of every
*trainable* parameter (the frozen SigLIP 2 backbone is excluded), updated
after each optimizer step, and swapped in for validation / test / predict.
EMA weights often generalise slightly better than the raw, noisier
end-of-training weights; the cost here is negligible since only the ~2.3M
LoRA + CDC + head parameters are tracked, not the full backbone.
"""

from __future__ import annotations

import lightning as L
import torch


class EMACallback(L.Callback):
    """Maintains an EMA shadow of trainable parameters; swaps them in for eval.

    Args:
        decay: EMA decay rate (``shadow = decay*shadow + (1-decay)*param``).
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    @staticmethod
    def _trainable_named_parameters(pl_module: L.LightningModule):
        return ((n, p) for n, p in pl_module.named_parameters() if p.requires_grad)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._shadow:  # skip re-init when resuming from a checkpoint
            for name, param in self._trainable_named_parameters(pl_module):
                self._shadow[name] = param.detach().clone()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs) -> None:
        for name, param in self._trainable_named_parameters(pl_module):
            self._shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def _swap_in(self, pl_module: L.LightningModule) -> None:
        self._backup = {}
        for name, param in self._trainable_named_parameters(pl_module):
            self._backup[name] = param.detach().clone().cpu()
            param.data.copy_(self._shadow[name].to(device=param.device, dtype=param.dtype))

    def _swap_out(self, pl_module: L.LightningModule) -> None:
        for name, param in self._trainable_named_parameters(pl_module):
            param.data.copy_(self._backup[name].to(device=param.device, dtype=param.dtype))
        self._backup = {}

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self._shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self._shadow = state_dict["shadow"]
