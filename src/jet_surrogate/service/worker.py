"""Worker loop: claim a queued job, run the surrogate on its HepMC file,
store the result. One model load per process."""

from __future__ import annotations

import gzip
import json
import shutil
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

from .jobs import JobStore, settings


def _prepare_input(job_dir: Path) -> Path:
    src = job_dir / "input.dat"
    dst = job_dir / "events.hepmc"
    if dst.exists():
        return dst
    with open(src, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        src.rename(dst)
    return dst


def run_job(store: JobStore, job, model, pre, device, max_events_cap: int) -> None:
    from ..commands.predict import predict_sample
    from ..hepmc_io import read_hepmc
    from ..skim import skim_truth

    job_dir = store.job_dir(job.id)
    log = open(job_dir / "predict.log", "a")
    try:
        hepmc = _prepare_input(job_dir)
        max_events = min(job.max_events or max_events_cap, max_events_cap)
        jets, parts, n_ev = [], [], 0
        for batch in read_hepmc(hepmc, max_events=max_events, chunk=1000):
            tj, tp = skim_truth(batch.part)
            tj["event"] += n_ev
            jets.append(tj); parts.append(tp); n_ev += len(batch)
            store.update(job.id, progress=f"{n_ev} events read")
            print(f"{n_ev} events, {sum(len(j) for j in jets)} truth jets", file=log, flush=True)
        if n_ev == 0:
            raise ValueError("no events could be read from the input (is it HepMC2/3 ASCII?)")
        jets = np.concatenate(jets); parts = np.concatenate(parts)
        summary, per_event, prob = predict_sample(jets, parts, n_ev, model, pre, device)
        summary["model"] = str(settings()["model"])
        out = job_dir / "results"; out.mkdir(exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        with h5py.File(out / "prediction.h5", "w") as h:
            h.create_dataset("event_probability", data=per_event.astype(np.float32))
            h.create_dataset("jet_probability", data=prob.astype(np.float32))
            h.create_dataset("truth_jets", data=jets)
        hepmc.unlink(missing_ok=True)                    # uploads are not kept once processed
        store.update(job.id, status="done", finished=time.time(), result=summary, progress="done")
    except Exception as e:                               # noqa: BLE001 - report every failure to the user
        traceback.print_exc(file=log)
        store.update(job.id, status="failed", finished=time.time(), error=f"{type(e).__name__}: {e}")
    finally:
        log.close()


def main_loop(poll_seconds: float = 3.0, once: bool = False, cleanup_only: bool = False) -> None:
    from ..training import load_checkpoint, pick_device

    cfg = settings()
    store = JobStore(cfg["root"])
    if cleanup_only:
        print(f"removed {store.cleanup(cfg['ttl_hours'])} expired jobs"); return
    device = pick_device("cpu")
    model, pre, _ = load_checkpoint(cfg["model"], device)
    model.to(device)
    print(f"worker ready: model {cfg['model']}, jobs in {cfg['root']}", flush=True)
    last_cleanup = 0.0
    while True:
        job = store.claim_next()
        if job is not None:
            print(f"job {job.id}: {job.label or job.source}", flush=True)
            run_job(store, job, model, pre, device, cfg["max_events"])
            j = store.get(job.id)
            print(f"job {job.id}: {j.status} {j.error or ''}", flush=True)
        elif once:
            return
        else:
            time.sleep(poll_seconds)
        if time.time() - last_cleanup > 3600:
            store.cleanup(cfg["ttl_hours"]); last_cleanup = time.time()
