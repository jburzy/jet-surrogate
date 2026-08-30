"""LightningDataModule for the tagger (reco tracks) and the surrogate (truth particles).

Data flow, unchanged from the validated plain-torch chain:
    skim HDF5 -> Preprocessor (model_space transform, z-score, PDG vocab)
              -> in-memory padded tensors -> whole-batch slicing

The normalization statistics are fitted on a subset of the training files
in ``setup()`` and exposed as ``self.preprocessor``; the LightningModule
picks them up in its own ``setup()`` and stores them in its hyperparameters,
so a checkpoint is self-contained. Batches are produced by slicing the
tensors directly (``batch_size=None`` + ``BatchSampler``), which is what
lets ``num_workers=0`` be fast and sidesteps the forked-worker deadlock
seen on the GPU node.
"""

from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from .. import features as F
from ..data import JetTable, load_table, load_transformed, skim_files
from ..generate import NOMINAL_MPID
from ..training import PaddedDataset, Preprocessor


class PaddedTensors(PaddedDataset, Dataset):
    """``PaddedDataset`` (in-memory padded tensors, ``batches()`` iterator used
    by ``predict``) that is also a torch ``Dataset`` indexed by whole batches
    (lists of indices), for the Lightning DataLoaders."""

    def subset(self, indices) -> "PaddedTensors":
        ds = PaddedTensors.__new__(PaddedTensors)
        ds.x, ds.cats, ds.mask, ds.y = self.x, self.cats, self.mask, self.y
        ds.indices = torch.as_tensor(np.asarray(indices), dtype=torch.int64)
        return ds

    def __getitem__(self, idx):
        b = torch.as_tensor(idx, dtype=torch.int64)
        if self.indices is not None:
            b = self.indices[b]
        return self.x[b], self.cats[b].long(), self.mask[b], self.y[b]


def batch_loader(ds: PaddedTensors, batch_size: int, shuffle: bool, drop_last: bool = False,
                 num_workers: int = 0, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, generator=gen) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, batch_size=None, sampler=BatchSampler(sampler, batch_size, drop_last),
                      num_workers=num_workers, pin_memory=True)


class JetDataModule(LightningDataModule):
    """``task="tagger"``: reco tracks, train/val/test by generator seed,
    signal = nominal mass, background = QCD.
    ``task="surrogate"``: truth particles of the dedicated surrogate seeds
    (disjoint from every tagger seed) with labels from apply-tagger,
    jet-level ``val_frac`` hold-out. Signal only by default
    (``qcd_jets_per_file=0``): the surrogate is never applied to background,
    and background jets at label ~0 only distort the calibration on
    QCD-like signal jets. Set ``qcd_jets_per_file > 0`` to mix in that many
    QCD truth jets per file for a controlled comparison.
    """

    def __init__(self, task: str = "tagger", data_dir: str = "data/skim", batch_size: int = 512,
                 num_workers: int = 0, fit_files: int = 6, max_jets_per_file: int | None = None,
                 qcd_jets_per_file: int = 0, val_frac: float = 0.1, max_files: int | None = None,
                 ctaus: list[float] | None = None, seed: int = 0):
        super().__init__()
        assert task in ("tagger", "surrogate")
        self.save_hyperparameters()
        self.task = task
        self.side = "reco" if task == "tagger" else "truth"
        self.floats = F.TRACK_FLOATS if task == "tagger" else F.PART_FLOATS
        self.cats = F.TRACK_CATS if task == "tagger" else F.PART_CATS
        self.preprocessor: Preprocessor | None = None
        self.datasets: dict[str, PaddedTensors] = {}
        self.meta: dict[str, JetTable] = {}
        self.is_signal: np.ndarray | None = None      # surrogate: per-jet sample flag

    # ------------------------------------------------------------------ files
    def _files(self, split: str):
        h = self.hparams
        if self.task == "tagger":
            fs = skim_files(h.data_dir, samples=("qcd", "signal"), splits=(split,), mpid=NOMINAL_MPID,
                            ctaus=h.ctaus)
        else:
            fs = skim_files(h.data_dir, samples=("qcd", "signal"), splits=("surrogate", "train", "val"),
                            mpid=NOMINAL_MPID, ctaus=h.ctaus, require_scores=True)
            # signal: only the dedicated surrogate seeds (disjoint from the tagger's); QCD, if
            # mixed in at all, may come from the tagger's train/val seeds (it never sees labels)
            fs = [f for f in fs if f.sample == "qcd" or f.split == "surrogate"]
        if h.max_files:
            fs = ([f for f in fs if f.sample == "qcd"][: h.max_files]
                  + [f for f in fs if f.sample == "signal"][: h.max_files])
        return fs

    def _fit_preprocessor(self, files) -> Preprocessor:
        h = self.hparams
        fit = ([f for f in files if f.sample == "qcd"][: h.fit_files]
               + [f for f in files if f.sample == "signal"][: h.fit_files])
        table, _ = load_table(fit, self.side, max_jets_per_file=20000 if self.task == "tagger" else 10000)
        return Preprocessor(self.floats, self.cats).fit(table.objs)

    # ------------------------------------------------------------------ setup
    def setup(self, stage: str | None = None) -> None:
        if self.datasets:
            return
        h = self.hparams
        if self.task == "tagger":
            train_files = self._files("train")
            self.preprocessor = self._fit_preprocessor(train_files)
            for split, files in (("train", train_files), ("val", self._files("val")), ("test", self._files("test"))):
                x, c, m, meta, _ = load_transformed(files, "reco", self.preprocessor,
                                                    max_jets_per_file=h.max_jets_per_file)
                self.datasets[split] = PaddedTensors(x, c, m, meta.label.astype(np.float32))
                self.meta[split] = meta
                print(f"{split}: {len(meta)} jets from {len(meta.files)} files "
                      f"(signal {int(meta.label.sum())}, qcd {int((meta.label == 0).sum())})", flush=True)
        else:
            files = self._files("train")
            sig = [f for f in files if f.sample == "signal"]
            qcd = [f for f in files if f.sample == "qcd"]
            if not sig:
                raise SystemExit("surrogate: no scored signal skims in the surrogate seeds "
                                 "(seeds >= 35 of the nominal mass); run apply-tagger first")
            self.preprocessor = self._fit_preprocessor(files if h.qcd_jets_per_file > 0 else sig)
            xs, cs, ms, meta_s, sc_s = load_transformed(sig, "truth", self.preprocessor, with_scores=("truth_label",))
            if h.qcd_jets_per_file > 0:
                xq, cq, mq, meta_q, sc_q = load_transformed(qcd, "truth", self.preprocessor,
                                                            max_jets_per_file=h.qcd_jets_per_file,
                                                            with_scores=("truth_label",))
                x = np.concatenate([xs, xq]); del xs, xq
                c = np.concatenate([cs, cq]); del cs, cq
                m = np.concatenate([ms, mq]); del ms, mq
                y = np.concatenate([sc_s["truth_label"], sc_q["truth_label"]]).astype(np.float32)
                self.is_signal = np.concatenate([np.ones(len(meta_s), bool), np.zeros(len(meta_q), bool)])
                print(f"signal truth jets {len(meta_s)} (label rate {sc_s['truth_label'].mean():.3f}), "
                      f"qcd truth jets {len(meta_q)} (label rate {sc_q['truth_label'].mean():.2e})", flush=True)
            else:
                x, c, m = xs, cs, ms
                y = sc_s["truth_label"].astype(np.float32)
                self.is_signal = np.ones(len(meta_s), bool)
                print(f"signal truth jets {len(meta_s)} (label rate {sc_s['truth_label'].mean():.3f}), "
                      f"no QCD (signal-only surrogate)", flush=True)
            rng = np.random.default_rng(h.seed)
            val = rng.random(len(y)) < h.val_frac
            full = PaddedTensors(x, c, m, y)
            del x, c, m
            self.datasets["train"] = full.subset(np.flatnonzero(~val))
            self.datasets["val"] = full.subset(np.flatnonzero(val))
            self.val_is_signal = self.is_signal[val]

    # ------------------------------------------------------------------ loaders
    def train_dataloader(self):
        h = self.hparams
        return batch_loader(self.datasets["train"], h.batch_size, True, drop_last=True,
                            num_workers=h.num_workers, seed=h.seed)

    def val_dataloader(self):
        h = self.hparams
        return batch_loader(self.datasets["val"], 2 * h.batch_size, False, num_workers=h.num_workers)

    def test_dataloader(self):
        h = self.hparams
        return batch_loader(self.datasets["test"], 2 * h.batch_size, False, num_workers=h.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
