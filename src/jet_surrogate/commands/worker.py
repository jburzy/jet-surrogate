"""Run a service worker: claims queued jobs and evaluates them with the surrogate.

    jet-surrogate worker                # loop forever
    jet-surrogate worker --once         # process what is queued, then exit
    jet-surrogate worker --cleanup-only # delete expired jobs (CronJob)
"""

from __future__ import annotations


def add_arguments(ap) -> None:
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between queue polls")


def run(args) -> None:
    from ..service.worker import main_loop
    main_loop(poll_seconds=args.poll, once=args.once, cleanup_only=args.cleanup_only)
