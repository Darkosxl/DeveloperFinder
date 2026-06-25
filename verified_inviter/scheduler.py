from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from verified_inviter import config

logger = logging.getLogger(__name__)


class Scheduler:
    """APScheduler wrapper that runs the daily pipeline on a configurable interval."""

    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run_now")

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler()
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "next_run_time": None,
            "last_run_time": None,
            "last_run_status": None,
            "last_run_error": None,
        }
        self._progress: dict[str, Any] = {
            "active": False,
            "stage": "",
            "candidate": None,
            "candidate_index": 0,
            "candidate_total": 0,
            "log": [],
        }
        self._job_id = "run_daily_job"

    def _update_next_run(self) -> None:
        with self._lock:
            job = self._scheduler.get_job(self._job_id)
            self._state["next_run_time"] = (
                job.next_run_time.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
                if job and job.next_run_time
                else None
            )

    def update_progress(self, stage: str = "", candidate: str | None = None,
                        candidate_index: int = 0, candidate_total: int = 0,
                        log_entry: str | None = None) -> None:
        with self._lock:
            if stage:
                self._progress["stage"] = stage
            if candidate is not None:
                self._progress["candidate"] = candidate
            self._progress["candidate_index"] = candidate_index or self._progress["candidate_index"]
            self._progress["candidate_total"] = candidate_total or self._progress["candidate_total"]
            if log_entry:
                self._progress["log"].append(log_entry)
                # Keep last 50 entries
                self._progress["log"] = self._progress["log"][-50:]

    def _job(self) -> None:
        # Lazy import to avoid circular imports at module load time.
        from verified_inviter import main

        with self._lock:
            self._state["last_run_time"] = datetime.now(timezone.utc)
            self._state["last_run_status"] = "running"
            self._state["last_run_error"] = None
            self._progress["active"] = True
            self._progress["stage"] = "starting"
            self._progress["candidate"] = None
            self._progress["candidate_index"] = 0
            self._progress["candidate_total"] = 0
            self._progress["log"] = []

        try:
            logger.info("scheduled pipeline run started")
            main.run_daily(config.DRY_RUN, progress_callback=self.update_progress)
            with self._lock:
                self._state["last_run_status"] = "success"
        except Exception as exc:
            logger.exception("scheduled pipeline run failed")
            with self._lock:
                self._state["last_run_status"] = "failed"
                self._state["last_run_error"] = str(exc)
        finally:
            with self._lock:
                self._progress["active"] = False
            self._update_next_run()

    def is_running(self) -> bool:
        with self._lock:
            return self._state["running"]

    def start(self) -> None:
        with self._lock:
            if self._state["running"]:
                return
            if not self._scheduler.get_job(self._job_id):
                self._scheduler.add_job(
                    self._job,
                    IntervalTrigger(minutes=config.SCHEDULER_INTERVAL_MINUTES),
                    id=self._job_id,
                    replace_existing=True,
                    max_instances=1,
                )
            self._scheduler.start()
            self._state["running"] = True
        self._update_next_run()
        logger.info("scheduler started", extra={"interval_minutes": config.SCHEDULER_INTERVAL_MINUTES})

    def stop(self) -> None:
        with self._lock:
            if not self._state["running"]:
                return
            self._scheduler.pause()
            self._state["running"] = False
            self._state["next_run_time"] = None
        logger.info("scheduler stopped")

    def run_now(self) -> None:
        """Trigger the pipeline in a background thread; returns immediately."""
        self._executor.submit(self._job)
        logger.info("run_now triggered")

    def status(self) -> dict[str, Any]:
        self._update_next_run()
        with self._lock:
            return {**self._state, "progress": self._progress.copy()}

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.copy()


SCHEDULER = Scheduler()
