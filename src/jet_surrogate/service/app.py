"""FastAPI application: the JSON API under /api and the static Carbon front end at /."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from . import registry
from .jobs import JobStore, settings

STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    cfg = settings()
    store = JobStore(cfg["root"])
    analyses = registry.load()
    app = FastAPI(title="PRISM: surrogate reinterpretation", version=__version__, docs_url="/docs", redoc_url=None)

    @app.get("/api/info")
    def info():
        return {"name": "PRISM", "tagline": "Physics Reinterpretation with Intelligent Surrogate Models",
                "version": __version__, "url": "https://prism.web.cern.ch", "repo_url": registry.repo_url(),
                "max_upload_mb": cfg["max_upload_mb"], "n_analyses": len(analyses)}

    @app.get("/api/analyses")
    def list_analyses():
        return [a.public() for a in analyses.values()]

    @app.get("/api/analyses/{analysis_id}")
    def get_analysis(analysis_id: str):
        a = analyses.get(analysis_id)
        if a is None:
            raise HTTPException(404, "unknown analysis")
        return a.public(detail=True)

    @app.get("/api/analyses/{analysis_id}/figures/{name}")
    def figure(analysis_id: str, name: str):
        a = analyses.get(analysis_id)
        p = (a.path / "figures" / Path(name).name) if a else None
        if p is None or not p.exists():
            raise HTTPException(404, "unknown figure")
        return FileResponse(p)

    @app.get("/api/analyses/{analysis_id}/example")
    def example(analysis_id: str):
        a = analyses.get(analysis_id)
        if a is None or a.example_path is None:
            raise HTTPException(404, "no example file for this analysis")
        return FileResponse(a.example_path, filename=a.example_path.name)

    @app.post("/api/jobs")
    async def submit(analysis: str = Form(...), file: UploadFile = File(...), label: str = Form(""),
                     max_events: int | None = Form(None)):
        a = analyses.get(analysis)
        if a is None:
            raise HTTPException(400, f"unknown analysis '{analysis}'")
        cap = min(cfg["max_events"], int(a.record.get("max_events", cfg["max_events"])))
        n = max(1, min(max_events or int(a.record.get("default_max_events", cap)), cap))
        job = store.create(analysis, label[:200], file.filename or "upload", n)
        dst = store.job_dir(job.id) / "input.dat"
        limit = int(cfg["max_upload_mb"] * 1e6)
        size = 0
        with open(dst, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    out.close(); shutil.rmtree(store.job_dir(job.id), ignore_errors=True)
                    store.update(job.id, status="failed", error=f"upload exceeds {cfg['max_upload_mb']:g} MB")
                    raise HTTPException(413, f"upload exceeds {cfg['max_upload_mb']:g} MB")
                out.write(chunk)
        return JSONResponse({"id": job.id, "status": job.status})

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50):
        return [j.to_dict() for j in store.list(min(max(limit, 1), 200))]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return j.to_dict()

    @app.get("/api/jobs/{job_id}/result.h5")
    def result(job_id: str):
        p = store.job_dir(job_id) / "results" / "prediction.h5"
        if not p.exists():
            raise HTTPException(404, "no result (job not done)")
        return FileResponse(p, filename=f"surrogate_{job_id}.h5")

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def log(job_id: str):
        p = store.job_dir(job_id) / "predict.log"
        return p.read_text() if p.exists() else "no log yet"

    @app.get("/health")
    def health():
        return {"status": "ok", "analyses": sorted(analyses)}

    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
