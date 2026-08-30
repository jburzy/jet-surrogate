"""FastAPI application: upload page, job submission and status endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .jobs import JobStore, settings

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>jet-surrogate</title>
<style>body{font-family:system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#222}
input,button{font:inherit;padding:.4rem} table{border-collapse:collapse} td,th{padding:.3rem .6rem;border-bottom:1px solid #ddd}
code{background:#f4f4f4;padding:0 .2rem}</style></head><body>
<h1>Truth-level surrogate: signal-region efficiency</h1>
<p>Upload a HepMC2/3 file (optionally gzipped) of a signal model, generator level only. The surrogate
clusters truth jets, predicts per jet whether the detector-level tagger would select it, and returns the
efficiency of the two-jet signal region (p<sub>T</sub> &gt; 200 GeV, tagger working point at 10<sup>3</sup>
QCD rejection).</p>
<form id="f" enctype="multipart/form-data">
<p><label>HepMC file <input type="file" name="file" required></label></p>
<p><label>Model description <input name="label" size="50" placeholder="m_pid = 8 GeV, ctau = 0.3 mm"></label></p>
<p><label>Max events <input name="max_events" type="number" value="20000" min="1"></label>
<button type="submit">Submit</button></p></form>
<div id="out"></div>
<h2>Recent jobs</h2><div id="jobs"></div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();out.textContent='uploading...';
 const r=await fetch('/jobs',{method:'POST',body:new FormData(f)});const j=await r.json();
 if(!r.ok){out.textContent='error: '+(j.detail||r.status);return;} poll(j.id);};
async function poll(id){const r=await fetch('/jobs/'+id);const j=await r.json();
 if(j.status==='done'){const s=j.result;out.innerHTML=`<p>Job <code>${id}</code> done: <b>SR efficiency ${s.sr_efficiency.toFixed(4)} &plusmn; ${s.sr_efficiency_err.toFixed(4)}</b>
 (${s.n_events} events, ${s.n_truth_jets} truth jets, mean jet probability ${s.mean_jet_probability.toFixed(3)}).
 <a href="/jobs/${id}/result.h5">per-event probabilities (HDF5)</a> &middot; <a href="/jobs/${id}/log">log</a></p>`;}
 else if(j.status==='failed'){out.innerHTML=`<p>Job <code>${id}</code> failed: ${j.error} (<a href="/jobs/${id}/log">log</a>)</p>`;}
 else{out.textContent=`job ${id}: ${j.status} ${j.progress||''}`;setTimeout(()=>poll(id),3000);} list();}
async function list(){const r=await fetch('/jobs');const js=await r.json();
 document.getElementById('jobs').innerHTML='<table><tr><th>id</th><th>label</th><th>status</th><th>efficiency</th></tr>'+
 js.map(j=>`<tr><td><code>${j.id}</code></td><td>${j.label||''}</td><td>${j.status}</td><td>${j.result?j.result.sr_efficiency.toFixed(4)+' &plusmn; '+j.result.sr_efficiency_err.toFixed(4):''}</td></tr>`).join('')+'</table>';}
list();
</script></body></html>"""


def create_app() -> FastAPI:
    cfg = settings()
    store = JobStore(cfg["root"])
    app = FastAPI(title="jet-surrogate", version="0.2.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/health")
    def health():
        return {"status": "ok", "model": cfg["model"]}

    @app.post("/jobs")
    async def submit(file: UploadFile = File(...), label: str = Form(""), max_events: int = Form(20000)):
        job = store.create(label[:200], file.filename or "upload", max(1, min(max_events, cfg["max_events"])))
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

    @app.get("/jobs")
    def jobs():
        return [j.to_dict() for j in store.list()]

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return j.to_dict()

    @app.get("/jobs/{job_id}/result.h5")
    def result(job_id: str):
        p = store.job_dir(job_id) / "results" / "prediction.h5"
        if not p.exists():
            raise HTTPException(404, "no result (job not done)")
        return FileResponse(p, filename=f"jet-surrogate_{job_id}.h5")

    @app.get("/jobs/{job_id}/log")
    def log(job_id: str):
        p = store.job_dir(job_id) / "predict.log"
        return HTMLResponse(f"<pre>{p.read_text() if p.exists() else 'no log yet'}</pre>")

    return app
