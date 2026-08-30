"""Callbacks in the ej-vae style: one checkpoint per epoch named by its
validation loss, the run config and normalization saved into the run dir,
and best-epoch selection by parsing the checkpoint filenames."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=([\d.]+)\.ckpt$")


class Checkpoint(ModelCheckpoint):
    """Keep every epoch: ``ckpts/epoch=012-val_loss=0.17701.ckpt``."""

    def __init__(self, monitor_loss: str = "val/loss"):
        super().__init__(save_top_k=-1, auto_insert_metric_name=False,
                         filename="epoch={epoch:03d}-val_loss={" + monitor_loss + ":.5f}")

    def setup(self, trainer, pl_module, stage=None):
        # trainer.log_dir follows the logger (CometLogger: ./.cometml-runs); the run dir is default_root_dir
        self.dirpath = str(Path(trainer.default_root_dir) / "ckpts")
        super().setup(trainer, pl_module, stage)


class SaveRunConfig(Callback):
    """Write the normalization spec (``norm.yaml``) into the run directory so
    every run is self-contained; LightningCLI writes ``config.yaml`` itself."""

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero and pl_module.pre is not None:
            out = Path(trainer.default_root_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "norm.yaml").write_text(yaml.safe_dump(pl_module.pre.to_dict()))


def get_best_epoch(run_dir: str | Path) -> Path:
    """Checkpoint with the lowest ``val_loss`` in its filename."""
    best, best_loss = None, float("inf")
    for p in Path(run_dir).glob("ckpts/*.ckpt"):
        m = CKPT_RE.search(p.name)
        if m and float(m.group(2)) < best_loss:
            best, best_loss = p, float(m.group(2))
    if best is None:
        raise FileNotFoundError(f"no epoch=*-val_loss=*.ckpt under {run_dir}/ckpts")
    return best


class EpochHistory(Callback):
    """Per-epoch ``history.json`` in the run directory, in the format the old
    plain-torch trainer wrote (epoch, train_loss, val_loss, val_acc, time)."""

    def __init__(self):
        self.history: list[dict] = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        # validation runs before on_train_epoch_end, so stash the val metrics and
        # complete the record (train loss) when the training epoch closes
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._pending = {"epoch": trainer.current_epoch, "val_loss": float(m["val/loss"]),
                         "val_acc": float(m.get("val/acc", float("nan")))}

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        rec = getattr(self, "_pending", None)
        if rec is None or rec["epoch"] != trainer.current_epoch:
            return
        m = trainer.callback_metrics
        rec["train_loss"] = float(m["train/loss"]) if "train/loss" in m else float("nan")
        rec["time"] = time.time() - self._t0 if self._t0 else None
        self.history.append(rec); self._pending = None
        if trainer.is_global_zero:
            print(f"epoch {rec['epoch']:3d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}"
                  f"  acc {rec['val_acc']:.4f}  {rec['time'] or 0:.0f}s", flush=True)
            out = Path(trainer.default_root_dir)
            (out / "history.json").write_text(json.dumps(
                {"history": self.history, "best_val_loss": min(h["val_loss"] for h in self.history)}, indent=1))
