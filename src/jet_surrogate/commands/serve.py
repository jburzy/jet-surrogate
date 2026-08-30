"""Run the web service (upload HepMC -> signal-region efficiency).

    jet-surrogate serve [--host 0.0.0.0] [--port 8080]

Jobs are queued in JS_SERVICE_DIR (default service_data/) and executed by
``jet-surrogate worker`` processes sharing that directory.
"""

from __future__ import annotations


def add_arguments(ap) -> None:
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)


def run(args) -> None:
    import uvicorn

    from ..service.app import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)
