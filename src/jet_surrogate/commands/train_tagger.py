"""Train the detector-level transformer tagger on Delphes tracks (GPU, Lightning).

    sbatch slurm/gpu.sbatch train-tagger [--config src/jet_surrogate/configs/tagger.yaml]
                                         [--name tagger] [--trainer.max_epochs=30] [--data.max_files=3]

Any ``--section.key=value`` argument overrides the YAML config (jsonargparse).
Training writes a self-contained run directory
``logs/<name>_<timestamp>/{config.yaml, norm.yaml, ckpts/epoch=NNN-val_loss=X.ckpt}``
with Comet logging (offline without COMET_API_KEY). After the fit the best
checkpoint is loaded, the working point (logit threshold at the requested QCD
jet rejection) is fixed on the test seeds, and ``--out`` receives
``tagger.pt`` (inference checkpoint), ``tagger.onnx``, ``working_point.yaml``,
``roc.json``, ``history.json`` and ``run_dir.txt``; figures come from
``jet-surrogate visualize``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from .. import features as F
from ..metrics import efficiency, roc, threshold_at_rejection

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tagger.yaml"


def add_arguments(ap) -> None:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default="models/tagger")
    ap.add_argument("--rejection", type=float, default=1000.0)
    ap.add_argument("--epochs", type=int, default=None, help="shorthand for --trainer.max_epochs")
    ap.add_argument("--max-files", type=int, default=None, help="shorthand for --data.max_files")
    ap.set_defaults(overrides=[])   # every unrecognized --section.key=value is forwarded to LightningCLI


def _overrides(args) -> list[str]:
    ov = list(args.overrides)
    if args.epochs is not None:
        ov.append(f"--trainer.max_epochs={args.epochs}")
    if args.max_files is not None:
        ov.append(f"--data.max_files={args.max_files}")
    return ov


def finish(cli, out: Path) -> tuple:
    """Common post-fit bookkeeping: history, best checkpoint, inference model."""
    from ..lightning.callbacks import get_best_epoch
    from ..lightning.module import JetClassifier

    out.mkdir(parents=True, exist_ok=True)
    best = get_best_epoch(cli.run_dir)
    model = JetClassifier.load_from_checkpoint(best, map_location="cpu")
    hist = json.loads((cli.run_dir / "history.json").read_text()) if (cli.run_dir / "history.json").exists() else {}
    (out / "run_dir.txt").write_text(f"{cli.run_dir}\n{best}\n")
    shutil.copy(best, out / "best.ckpt")
    print(f"best checkpoint {best}", flush=True)
    return model, best, hist


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

    # ---- working point and ROC curves on the test seeds (data only, no plots)
    t = dm.meta["test"]
    logit = predict(net, dm.datasets["test"], device)
    sig_mask = t.label == 1
    bkg = logit[~sig_mask]
    thr = threshold_at_rejection(bkg, args.rejection)
    auc = float(roc_auc_score(t.label, logit)) if 0 < sig_mask.sum() < len(t) else None
    wp = {"threshold_logit": thr, "target_rejection": args.rejection,
          "eff_qcd": efficiency(bkg > thr)[0], "n_qcd_test": int(len(bkg)),
          "n_qcd_pass": int((bkg > thr).sum()), "auc": auc, "eff_signal": {},
          "checkpoint": str(best)}
    curves = {}
    for ct in sorted(set(t.ctau[sig_mask].tolist())):
        s = logit[sig_mask & (t.ctau == ct)]
        e, err = efficiency(s > thr)
        wp["eff_signal"][f"{ct:g}"] = {"eff": e, "err": err, "n": int(len(s))}
        es, eb, _ = roc(s, bkg)
        curves[f"{ct:g}"] = {"eff_sig": es.tolist(), "eff_bkg": eb.tolist()}
        print(f"signal ctau={ct:g} mm: jet eff at 1/{args.rejection:g} = {e:.4f} +- {err:.4f}")
    print(f"AUC {auc}  threshold {thr:.3f}  qcd eff {wp['eff_qcd']:.2e} ({wp['n_qcd_pass']} / {wp['n_qcd_test']})")
    (out / "working_point.yaml").write_text(yaml.safe_dump(wp, sort_keys=False))
    (out / "roc.json").write_text(json.dumps({"rejection": args.rejection, "curves": curves}))
    save_checkpoint(out / "tagger.pt", net, pre, {"working_point": wp, "config": str(args.config),
                                                  "checkpoint": str(best)})
    export_onnx(net, pre, out / "tagger.onnx", F.MAX_TRK)
    print(f"wrote {out}")


def history_from_logger(cli) -> dict:
    """Epoch history from the EpochHistory callback if present."""
    for cb in cli.trainer.callbacks:
        if hasattr(cb, "history"):
            return {"history": cb.history, "best_val_loss": min((h["val_loss"] for h in cb.history), default=None)}
    return {}
