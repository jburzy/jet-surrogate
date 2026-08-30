"""``LightningCLI`` subclass and the ``run_fit`` helper used by the
``train-tagger`` / ``train-surrogate`` subcommands.

Conventions (ej-vae): ``--name`` becomes the Comet experiment name and the
run directory ``logs/<name>_<YYYYmmdd-THHMMSS>``; without ``COMET_API_KEY``
the logger goes offline into the run directory. Any ``--section.key=value``
argument is a jsonargparse override of the YAML config.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("COMET_LOG_ENV_CONDA", "false")
try:                                   # comet_ml must be imported before torch
    import comet_ml  # noqa: F401
except Exception:                      # pragma: no cover
    pass

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from .data import JetDataModule  # noqa: E402
from .module import JetClassifier  # noqa: E402


class JetCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("-n", "--name", default="jet-surrogate", help="run and Comet experiment name")
        parser.add_argument("--log_suffix", default=None, help="fixed run-dir suffix instead of a timestamp")
        parser.link_arguments("name", "model.name")

    def before_instantiate_classes(self):
        cfg = self.config
        suffix = cfg.log_suffix or datetime.now().strftime("%Y%m%d-T%H%M%S")
        root = Path(cfg.trainer.default_root_dir or "logs")
        run_dir = root / f"{cfg.name}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.trainer.default_root_dir = str(run_dir)
        logger = cfg.trainer.logger
        if logger is not None and not isinstance(logger, bool):
            init = logger.init_args
            if not os.environ.get("COMET_API_KEY"):
                init.online = False
                os.environ.setdefault("COMET_OFFLINE_DIRECTORY", str(run_dir))
        self.run_dir = run_dir

    def after_instantiate_classes(self):
        # the experiment name is a comet_ml.start() kwarg, not a CometLogger init arg
        logger = self.trainer.logger
        if logger is not None and hasattr(logger, "experiment"):
            try:
                logger.experiment.set_name(self.config.name)
            except Exception as e:                       # pragma: no cover
                print(f"[comet] could not set the experiment name: {e}")


def run_fit(config: Path, overrides: list[str]) -> JetCLI:
    """Instantiate from ``config`` (+ overrides), fit, and return the CLI object
    (``cli.trainer``, ``cli.model``, ``cli.datamodule``, ``cli.run_dir``)."""
    import sys
    args = ["--config", str(config), *overrides]
    sys.argv = sys.argv[:1]            # LightningCLI warns if sys.argv also carries arguments
    cli = JetCLI(JetClassifier, JetDataModule, args=args, run=False, save_config_kwargs={"overwrite": True})
    cli.trainer.fit(cli.model, cli.datamodule)
    return cli
