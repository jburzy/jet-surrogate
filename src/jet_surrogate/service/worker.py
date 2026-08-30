"""Worker loop: claim a queued job, run the analysis' predictor on its HepMC
file, store the result. Predictors (one model each) are loaded on first use
and cached for the life of the process."""

from __future__ import annotations

import gzip
import json
import shutil
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

from . import registry
from .jobs import JobStore, settings


def _prepare_input(job_dir: Path) -> Path:
    src, dst = job_dir / "input.dat", job_dir / "events.hepmc"
    if dst.exists():
        return dst
    with open(src, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        src.unlink()
    else:
        src.rename(dst)
    return dst


class Predictors:
    def __init__(self, analyses: dict[str, registry.Analysis], device="cpu"):
        self.analyses, self.device, self._cache = analyses, device, {}

    def get(self, analysis_id: str):
        if analysis_id not in self._cache:
            a = self.analyses.get(analysis_id)
            if a is None:
                raise KeyError(f"unknown analysis '{analysis_id}'")
            if a.record.get("assets"):
                a.fetch_assets()
            self._cache[analysis_id] = a.predictor(self.device)
        return self._cache[analysis_id]


def run_job(store: JobStore, job, predictors: Predictors, max_events_cap: int) -> None:
    job_dir = store.job_dir(job.id)
    log = open(job_dir / "predict.log", "a")
    try:
        hepmc = _prepare_input(job_dir)
        predictor = predictors.get(job.analysis)
        a = predictors.analyses[job.analysis]
        cap = min(max_events_cap, int(a.record.get("max_events", max_events_cap)))
        max_events = min(job.max_events or cap, cap)

        def progress(msg):
            store.update(job.id, progress=msg); print(msg, file=log, flush=True)

        from .predictors import call_run
        summary, per_event, extras = call_run(predictor, hepmc, max_events, progress, options=job.options or {})
        if job.options:
            summary["options"] = job.options
        out = job_dir / "results"; out.mkdir(exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        with h5py.File(out / "prediction.h5", "w") as h:
            h.create_dataset("event_probability", data=np.asarray(per_event, np.float32))
            for k, v in extras.items():
                h.create_dataset(k, data=v)
            h.attrs.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
        hepmc.unlink(missing_ok=True)                    # uploads are not kept once processed
        store.update(job.id, status="done", finished=time.time(), result=summary, progress="done")
    except Exception as e:                               # noqa: BLE001 - every failure is reported to the user
        traceback.print_exc(file=log)
        store.update(job.id, status="failed", finished=time.time(), error=f"{type(e).__name__}: {e}")
    finally:
        log.close()


def main_loop(poll_seconds: float = 3.0, once: bool = False, cleanup_only: bool = False) -> None:
    cfg = settings()
    store = JobStore(cfg["root"])
    if cleanup_only:
        print(f"removed {store.cleanup(cfg['ttl_hours'])} expired jobs"); return
    predictors = Predictors(registry.load(), device="cpu")
    print(f"worker ready: analyses {sorted(predictors.analyses)}, jobs in {cfg['root']}", flush=True)
    last_cleanup = 0.0
    while True:
        job = store.claim_next()
        if job is not None:
            print(f"job {job.id} [{job.analysis}]: {job.label or job.source}", flush=True)
            run_job(store, job, predictors, cfg["max_events"])
            j = store.get(job.id)
            print(f"job {job.id}: {j.status} {j.error or ''}", flush=True)
        elif once:
            return
        else:
            time.sleep(poll_seconds)
        if time.time() - last_cleanup > 3600:
            store.cleanup(cfg["ttl_hours"]); last_cleanup = time.time()
