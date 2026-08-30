"""Consistency check for a trained model directory: the Lightning .ckpt, the
inference .pt and the ONNX graph must give the same logits on one skim file.

    python slurm/tools/check_checkpoints.py models/tagger data/skim
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import torch

from jet_surrogate.data import skim_files
from jet_surrogate.training import load_checkpoint, score_padded

out, skim_dir = Path(sys.argv[1]), sys.argv[2]
side = "reco_tracks" if (out / "tagger.pt").exists() else "truth_parts"
pt = out / ("tagger.pt" if side == "reco_tracks" else "surrogate.pt")
onnx = pt.with_suffix(".onnx")
f = skim_files(skim_dir, samples=("signal",))[0]
with h5py.File(f.path) as h:
    padded = h[side][:2000]

net_pt, pre_pt, _ = load_checkpoint(pt)
net_ck, pre_ck, _ = load_checkpoint(out / "best.ckpt")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l_pt = score_padded(net_pt, pre_pt, padded, dev)
l_ck = score_padded(net_ck, pre_ck, padded, dev)
l_pt_cpu = score_padded(net_pt, pre_pt, padded, torch.device("cpu"))
x, c, m = pre_pt.transform(padded)
sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
l_onnx = np.concatenate([sess.run(["logit"], {"x": x[i:i + 500], "cats": c[i:i + 500], "mask": m[i:i + 500]})[0]
                         for i in range(0, len(x), 500)])
print(f"file {f.path.name}: {len(padded)} jets")
print(f"max |ckpt - pt| (gpu)      = {np.abs(l_ck - l_pt).max():.3e}")
print(f"max |pt gpu - pt cpu|      = {np.abs(l_pt - l_pt_cpu).max():.3e}")
print(f"max |onnx - pt cpu|        = {np.abs(l_onnx - l_pt_cpu).max():.3e}")
assert np.abs(l_ck - l_pt).max() == 0.0, "ckpt and pt differ"
assert np.abs(l_onnx - l_pt_cpu).max() < 1e-3, "onnx differs"
print("CHECKPOINTS-CONSISTENT")
