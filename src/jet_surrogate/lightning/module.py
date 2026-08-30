"""LightningModule wrapping ``ParticleTransformer`` for binary jet classification.

The preprocessing spec (normalization statistics and PDG vocabulary) is
taken from the DataModule at ``setup()`` and written into the module's
hyperparameters, so ``load_from_checkpoint`` rebuilds a self-contained
model (net + preprocessor) with no access to training data, exactly as the
old ``tagger.pt``/``surrogate.pt`` files did.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..models import ParticleTransformer
from ..training import Preprocessor


class JetClassifier(LightningModule):
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4, d_ff: int | None = None,
                 dropout: float = 0.1, cat_dim: int = 16, lr: float = 3e-4, weight_decay: float = 1e-2,
                 warmup_steps: int = 500, preprocessor: dict | None = None, name: str = "jet-surrogate"):
        super().__init__()
        self.save_hyperparameters()
        self.net: ParticleTransformer | None = None
        self.pre: Preprocessor | None = None
        self.loss = nn.BCEWithLogitsLoss()
        if preprocessor is not None:
            self._build(Preprocessor.from_dict(preprocessor))

    # ------------------------------------------------------------------ construction
    def _build(self, pre: Preprocessor) -> None:
        h = self.hparams
        self.pre = pre
        self.net = ParticleTransformer(len(pre.floats), pre.cat_sizes, cat_dim=h.cat_dim, d_model=h.d_model,
                                       n_heads=h.n_heads, n_layers=h.n_layers,
                                       d_ff=h.d_ff or 2 * h.d_model, dropout=h.dropout)
        self.hparams["preprocessor"] = pre.to_dict()

    def setup(self, stage: str | None = None) -> None:
        if self.net is None:
            dm = self.trainer.datamodule
            if dm.preprocessor is None:
                dm.setup(stage)
            self._build(dm.preprocessor)
            print(f"parameters: {sum(p.numel() for p in self.net.parameters()) / 1e6:.2f} M; "
                  f"vocab sizes {self.pre.cat_sizes}", flush=True)

    # ------------------------------------------------------------------ steps
    def forward(self, x, cats, mask):
        return self.net(x, cats, mask)

    def _step(self, batch, prefix: str):
        x, c, m, y = batch
        logit = self(x, c, m)
        loss = self.loss(logit, y)
        if not torch.isfinite(loss):
            raise ValueError(f"{prefix} loss is not finite")
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=len(y), sync_dist=True)
        if prefix != "train":
            acc = ((logit > 0) == (y > 0.5)).float().mean()
            self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, c, m, _ = batch
        return self(x, c, m).float()

    # ------------------------------------------------------------------ optimization
    def configure_optimizers(self):
        h = self.hparams
        opt = torch.optim.AdamW(self.parameters(), lr=h.lr, weight_decay=h.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warm = min(h.warmup_steps, total // 10)

        def sched_fn(step):                       # linear warm-up, then cosine to zero
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
