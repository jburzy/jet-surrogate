"""SQLite-backed job table shared by the web app and the workers."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("queued", "running", "done", "failed")


def settings() -> dict:
    root = Path(os.environ.get("JS_SERVICE_DIR", "service_data")).resolve()
    return {
        "root": root,
        "max_upload_mb": float(os.environ.get("JS_MAX_UPLOAD_MB", "2000")),
        "max_events": int(os.environ.get("JS_MAX_EVENTS", "20000")),
        "ttl_hours": float(os.environ.get("JS_JOB_TTL_HOURS", "72")),
    }


@dataclass
class Job:
    id: str
    status: str
    created: float
    analysis: str
    label: str
    source: str
    max_events: int
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    progress: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.db"
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT, created REAL, analysis TEXT, label TEXT, source TEXT,
                max_events INTEGER, started REAL, finished REAL, result TEXT, error TEXT, progress TEXT)""")

    def _conn(self):
        c = sqlite3.connect(self.db, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def create(self, analysis: str, label: str, source: str, max_events: int) -> Job:
        job = Job(uuid.uuid4().hex[:12], "queued", time.time(), analysis, label, source, max_events)
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("INSERT INTO jobs (id, status, created, analysis, label, source, max_events) VALUES (?,?,?,?,?,?,?)",
                      (job.id, job.status, job.created, analysis, label, source, max_events))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT id, status, created, analysis, label, source, max_events, started, finished, "
                            "result, error, progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(*row[:9], json.loads(row[9]) if row[9] else None, row[10], row[11])

    def list(self, limit: int = 50) -> list[Job]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]
        return [self.get(i) for i in ids]

    def claim_next(self) -> Job | None:
        """Atomically move the oldest queued job to running."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                c.execute("COMMIT"); return None
            c.execute("UPDATE jobs SET status = 'running', started = ? WHERE id = ?", (time.time(), row[0]))
            c.execute("COMMIT")
        return self.get(row[0])

    def update(self, job_id: str, **fields) -> None:
        if "result" in fields and fields["result"] is not None:
            fields["result"] = json.dumps(fields["result"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def cleanup(self, ttl_hours: float) -> int:
        """Delete finished jobs (and their files) older than the TTL."""
        cutoff = time.time() - ttl_hours * 3600
        with self._conn() as c:
            old = [r[0] for r in c.execute("SELECT id FROM jobs WHERE status IN ('done','failed') AND created < ?", (cutoff,))]
            for i in old:
                shutil.rmtree(self.job_dir(i), ignore_errors=True)
                c.execute("DELETE FROM jobs WHERE id = ?", (i,))
        return len(old)
