"""Background job runner.

Long operations (model training, report rendering) run in a thread pool and
report real stage transitions, so the browser stays responsive and polls a job
endpoint instead of holding an HTTP request open.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import SETTINGS, Settings


class JobNotFound(KeyError):
    pass


@dataclass
class Job:
    id: str
    owner: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "Queued"
    progress: int = 0
    message: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    dataset_id: str | None = None

    def elapsed_ms(self) -> int:
        end = self.finished_at or time.time()
        start = self.started_at or self.created_at
        return int((end - start) * 1000)

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "elapsed_ms": self.elapsed_ms(),
            "result": self.result,
            "error": self.error,
            "dataset_id": self.dataset_id,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
        }


class JobManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.settings.job_workers, thread_name_prefix="aiviz-job"
        )

    def submit(self, owner: str, kind: str, func, *, dataset_id: str | None = None) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            owner=owner,
            kind=kind,
            dataset_id=dataset_id,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
            self._pool.submit(self._run, job, func)
        return job

    def _run(self, job: Job, func) -> None:
        job.status = "running"
        job.started_at = time.time()

        def progress(percent: int, stage: str, message: str | None = None) -> None:
            job.progress = max(0, min(100, int(percent)))
            job.stage = stage
            job.message = message

        try:
            result = func(progress)
            job.result = result
            job.progress = 100
            job.stage = "Complete"
            job.status = "done"
        except Exception as error:  # surfaced to the UI, never swallowed
            job.status = "error"
            job.stage = "Failed"
            job.error = str(error) or error.__class__.__name__
        finally:
            job.finished_at = time.time()

    def get(self, owner: str, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.owner != owner:
                raise JobNotFound(job_id)
            return job

    def _prune_locked(self) -> None:
        cutoff = time.time() - self.settings.job_retention_seconds
        stale = [
            key
            for key, job in self._jobs.items()
            if job.finished_at and job.finished_at < cutoff
        ]
        for key in stale:
            self._jobs.pop(key, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
