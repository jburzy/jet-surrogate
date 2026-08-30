"""Preprocessing, in-memory padded datasets, inference checkpoints, ONNX export.

Training itself lives in ``jet_surrogate.lightning`` (PyTorch Lightning).
Both networks consume padded (jet, slot) arrays:
    floats -> model_space() transform -> z-score with training-set stats
    cats   -> vocabulary index (0 = pad/unknown)
An inference checkpoint (``*.pt``) stores everything needed to score new
jets (architecture, normalization, vocabulary); a Lightning ``*.ckpt`` holds
the same information in its hyperparameters and is accepted everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import model_space
from .models import ParticleTransformer


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ preprocessing
@dataclass
class Preprocessor:
    floats: list[str]
    cats: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    vocab: list[dict[int, int]] = field(default_factory=list)   # per cat: raw id -> index

    def fit(self, padded: np.ndarray, max_vocab: int = 200) -> "Preprocessor":
        valid = padded["valid"]
        x = np.stack([model_space(f, padded[f]) for f in self.floats], axis=-1)[valid]
        self.mean = x.mean(0).astype(np.float32)
        self.std = (x.std(0) + 1e-6).astype(np.float32)
        self.vocab = []
        for c in self.cats:
            ids, counts = np.unique(padded[c][valid], return_counts=True)
            top = ids[np.argsort(-counts)][:max_vocab]
            # 0 = padding, 1 = unknown, 2.. = known ids
            self.vocab.append({int(i): k + 2 for k, i in enumerate(sorted(top.tolist()))})
        return self

    @property
    def cat_sizes(self) -> list[int]:
        return [len(v) + 2 for v in self.vocab]

    def transform(self, padded: np.ndarray):
        valid = padded["valid"]
        x = np.stack([model_space(f, padded[f]) for f in self.floats], axis=-1)
        x = (x - self.mean) / self.std
        x = np.where(valid[..., None], x, 0.0).astype(np.float32)
        cats = np.zeros(padded.shape + (len(self.cats),), dtype=np.int64)
        for k, c in enumerate(self.cats):
            raw = padded[c]
            lut_keys = np.array(list(self.vocab[k].keys()), dtype=np.int64)
            lut_vals = np.array(list(self.vocab[k].values()), dtype=np.int64)
            pos = np.searchsorted(lut_keys, raw)
            pos = np.clip(pos, 0, len(lut_keys) - 1)
            known = lut_keys[pos] == raw
            idx = np.where(known, lut_vals[pos], 1)
            cats[..., k] = np.where(valid, idx, 0)
        return x, cats, valid

    def to_dict(self) -> dict:
        return {"floats": self.floats, "cats": self.cats, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "vocab": [{str(k): v for k, v in d.items()} for d in self.vocab]}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(d["floats"], d["cats"], np.array(d["mean"], np.float32), np.array(d["std"], np.float32),
                   [{int(k): v for k, v in dd.items()} for dd in d["vocab"]])


# ------------------------------------------------------------------ data
class PaddedDataset:
    """In-memory padded tensors with a batch iterator that slices them directly.

    No torch DataLoader: forked worker processes deadlocked (futex) after the
    first epoch on the GPU node, and per-sample __getitem__ is pointless when
    everything already sits in RAM. Categorical indices are stored as int32
    and widened to int64 per batch. ``indices`` makes a zero-copy view of a
    subset (train/val splits share one set of arrays).
    """

    def __init__(self, x, cats, mask, y, indices=None):
        self.x = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32))
        self.cats = torch.as_tensor(np.ascontiguousarray(cats, dtype=np.int32))
        self.mask = torch.as_tensor(np.ascontiguousarray(mask, dtype=bool))
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32))
        self.indices = None if indices is None else torch.as_tensor(np.asarray(indices), dtype=torch.int64)

    def subset(self, indices) -> "PaddedDataset":
        ds = PaddedDataset.__new__(PaddedDataset)
        ds.x, ds.cats, ds.mask, ds.y = self.x, self.cats, self.mask, self.y
        ds.indices = torch.as_tensor(np.asarray(indices), dtype=torch.int64)
        return ds

    def __len__(self):
        return len(self.y) if self.indices is None else len(self.indices)

    @property
    def labels(self) -> np.ndarray:
        return (self.y if self.indices is None else self.y[self.indices]).numpy()

    def batches(self, batch_size: int, shuffle: bool, drop_last: bool = False, seed: int = 0):
        n = len(self)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
        if self.indices is not None:
            order = self.indices[order]
        for lo in range(0, n, batch_size):
            b = order[lo:lo + batch_size]
            if drop_last and len(b) < batch_size:
                break
            yield self.x[b], self.cats[b].long(), self.mask[b], self.y[b]

    def n_batches(self, batch_size: int, drop_last: bool = False) -> int:
        return len(self) // batch_size if drop_last else -(-len(self) // batch_size)


# ------------------------------------------------------------------ train
@torch.no_grad()
def predict(model: nn.Module, ds: PaddedDataset, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for x, c, m, _ in ds.batches(batch_size, shuffle=False):
        out.append(model(x.to(device), c.to(device), m.to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path: Path, model: ParticleTransformer, pre: Preprocessor, extra: dict | None = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "preprocessor": pre.to_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: Path, device=None) -> tuple[ParticleTransformer, Preprocessor, dict]:
    """Load either an inference checkpoint (``.pt``) or a Lightning ``.ckpt``.
    Returns (net, preprocessor, extra); for a ``.ckpt`` ``extra`` holds the
    hyperparameters and, if a ``working_point.yaml`` sits next to the run's
    ``models`` output, callers must pass it explicitly."""
    if str(path).endswith(".ckpt"):
        from .lightning.module import JetClassifier
        m = JetClassifier.load_from_checkpoint(path, map_location=device or "cpu")
        return m.net.eval(), m.pre, {"hparams": dict(m.hparams)}
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = ParticleTransformer(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Preprocessor.from_dict(ck["preprocessor"]), ck["extra"]


def score_padded(model, pre: Preprocessor, padded: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """Logits for a padded array (any size), batched."""
    x, c, m = pre.transform(padded)
    ds = PaddedDataset(x, c, m, np.zeros(len(x), np.float32))
    return predict(model.to(device), ds, device, batch_size)


def export_onnx(model: ParticleTransformer, pre: Preprocessor, path: Path, n_slots: int):
    """ONNX graph with inputs (x[B,N,F], cats[B,N,C], mask[B,N]) -> logit[B]."""
    model = model.cpu().eval()
    x = torch.zeros(2, n_slots, len(pre.floats))
    c = torch.zeros(2, n_slots, len(pre.cats), dtype=torch.long)
    m = torch.zeros(2, n_slots, dtype=torch.bool); m[:, 0] = True
    torch.onnx.export(model, (x, c, m), str(path), input_names=["x", "cats", "mask"],
                      output_names=["logit"], dynamo=False, opset_version=17,
                      dynamic_axes={"x": {0: "batch"}, "cats": {0: "batch"}, "mask": {0: "batch"},
                                    "logit": {0: "batch"}})
    Path(path).with_suffix(".preprocessor.json").write_text(json.dumps(pre.to_dict()))
