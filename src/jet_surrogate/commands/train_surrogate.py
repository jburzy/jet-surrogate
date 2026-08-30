"""Train the truth-level surrogate (GPU, Lightning): generator particles in a
truth large-R jet -> probability that the matched reco jet passes the tagger.

    sbatch slurm/gpu.sbatch train-surrogate [--config src/jet_surrogate/configs/surrogate.yaml]
                                            [--name surrogate] [--trainer.max_epochs=30]

Trained on the train + val seeds of the nominal signal (labels from
apply-tagger). Signal only by default: the surrogate is never applied to
background, and background jets would only bias the calibration on
QCD-like signal jets (``--data.qcd_jets_per_file=N`` mixes QCD back in). Run directory as for train-tagger. ``--out`` receives
``surrogate.pt``, ``surrogate.onnx``, ``history.json``, ``summary.json``,
``best.ckpt`` and ``run_dir.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import features as F
from .train_tagger import _overrides, finish, history_from_logger

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "surrogate.yaml"


def add_arguments(ap) -> None:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default="models/surrogate")
    ap.add_argument("--epochs", type=int, default=None, help="shorthand for --trainer.max_epochs")
    ap.add_argument("--max-files", type=int, default=None, help="shorthand for --data.max_files")
    ap.set_defaults(overrides=[])   # every unrecognized --section.key=value is forwarded to LightningCLI


def run(args) -> None:
    from sklearn.metrics import roc_auc_score  # lazy: keeps the inference env free of sklearn
    from ..lightning.cli import run_fit
    from ..training import export_onnx, pick_device, predict, save_checkpoint

    out = Path(args.out)
    cli = run_fit(Path(args.config), _overrides(args))
    model, best, hist = finish(cli, out)
    net, pre = model.net.eval(), model.pre
    device = pick_device()
    net.to(device)
    dm = cli.datamodule
    (out / "history.json").write_text(json.dumps(hist or history_from_logger(cli), indent=1))

    ds_va = dm.datasets["val"]
    vlogit = predict(net, ds_va, device)
    prob = 1 / (1 + np.exp(-vlogit))
    yv, sv = ds_va.labels, dm.val_is_signal
    summary = {"val_auc_all": float(roc_auc_score(yv, vlogit)) if 0 < yv.sum() < len(yv) else None,
               "val_auc_signal_jets": float(roc_auc_score(yv[sv], vlogit[sv])) if 0 < yv[sv].sum() < sv.sum() else None,
               "val_mean_label_signal": float(yv[sv].mean()), "val_mean_prob_signal": float(prob[sv].mean()),
               "val_mean_label_qcd": float(yv[~sv].mean()) if (~sv).any() else None,
               "val_mean_prob_qcd": float(prob[~sv].mean()) if (~sv).any() else None,
               "n_train": len(dm.datasets["train"]), "n_val": int(len(yv)), "checkpoint": str(best)}
    print(json.dumps(summary, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    save_checkpoint(out / "surrogate.pt", net, pre, {"summary": summary, "config": str(args.config),
                                                     "checkpoint": str(best)})
    export_onnx(net, pre, out / "surrogate.onnx", F.MAX_PART)
    print(f"wrote {out}")
