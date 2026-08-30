"""Web service: upload a HepMC file, get the surrogate-predicted signal-region
efficiency. ``app`` (FastAPI) accepts and reports jobs, ``worker`` runs them,
``jobs`` is the shared SQLite job table. Configuration by environment:
JS_SERVICE_DIR (job storage), JS_ANALYSES_DIR (the analysis library),
JS_MAX_UPLOAD_MB, JS_MAX_EVENTS, JS_JOB_TTL_HOURS."""
