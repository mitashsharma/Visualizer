"""In-memory dataset store with immutable revisions.

Each upload creates a *revision*. Cleaning creates a new revision derived from
its parent; the original stays untouched, so "Restore original" is instant and
there is no way for a cached analysis to leak between versions.

Cached artefacts (profile, curated charts, insights) live on the revision
object. Because a revision's DataFrame never changes, caches can be keyed by
the object itself and can never go stale.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.charts import default_charts
from src.config import SETTINGS, Settings
from src.data_loader import LoadedDataset
from src.ml_model import dataset_signature
from src.profile import build_profile
from src.summarizer import build_insights


class DatasetNotFound(KeyError):
    """Raised when a dataset does not exist for this session."""


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class Dataset:
    id: str
    owner: str
    name: str
    source_filename: str
    dataframe: pd.DataFrame
    revision: int = 1
    root_id: str = ""
    parent_id: str | None = None
    cleaning: dict | None = None
    truncated: bool = False
    total_rows_in_file: int | None = None
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    settings: Settings = field(default_factory=lambda: SETTINGS)

    _profile: dict | None = field(default=None, repr=False)
    _charts: list | None = field(default=None, repr=False)
    model_results: dict = field(default_factory=dict, repr=False)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not self.root_id:
            self.root_id = self.id
        if not self.name:
            self.name = Path(self.source_filename).stem.replace("_", " ").replace("-", " ").title()
        if not self.parent_id:
            self.parent_id = None

    # -- lazy, cached analyses ------------------------------------------------
    @property
    def fingerprint(self) -> str:
        return dataset_signature(self.dataframe)

    def profile(self) -> dict:
        with self._lock:
            if self._profile is None:
                self._profile = build_profile(
                    self.dataframe,
                    source=self.source_filename,
                    truncated=self.truncated,
                    extra={"revision": self.revision},
                )
            return self._profile

    def charts(self) -> list[dict]:
        with self._lock:
            if self._charts is None:
                self._charts = default_charts(
                    self.dataframe, self.profile(), settings=self.settings
                )
            return self._charts

    def insights(self, ml_result: dict | None = None) -> dict:
        return build_insights(self.profile(), self.dataframe, ml_result)

    def touch(self) -> None:
        self.last_access = time.time()

    # -- serialisation --------------------------------------------------------
    def summary(self) -> dict:
        profile = self.profile()
        overview = profile["overview"]
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source_filename,
            "rows": overview["rows"],
            "columns": overview["columns"],
            "missing_pct": overview["missing_pct"],
            "duplicate_rows": overview["duplicate_rows"],
            "revision": self.revision,
            "root_id": self.root_id,
            "parent_id": self.parent_id,
            "is_cleaned": self.revision > 1,
            "created_at": _iso(self.created_at),
            "truncated": self.truncated,
            "total_rows_in_file": self.total_rows_in_file,
            "cleaning": self.cleaning,
            "memory_mb": overview["memory_mb"],
        }


class DatasetStore:
    """Bounded, session-scoped store of dataset revisions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._items: dict[str, Dataset] = {}
        # The workspace has one explicit active revision per session. Reads of
        # an older revision must not silently switch the active one.
        self._active: dict[str, str] = {}
        self._lock = threading.RLock()

    # -- lifecycle ------------------------------------------------------------
    def create(
        self,
        owner: str,
        loaded: LoadedDataset,
        *,
        name: str | None = None,
    ) -> Dataset:
        now = time.time()
        dataset_id = uuid.uuid4().hex
        dataset = Dataset(
            id=dataset_id,
            owner=owner,
            name=name or Path(loaded.filename).stem.replace("_", " ").replace("-", " ").title(),
            source_filename=loaded.filename,
            dataframe=loaded.dataframe,
            truncated=loaded.truncated,
            total_rows_in_file=loaded.total_rows_in_file,
            created_at=now,
            last_access=now,
            settings=self.settings,
        )
        with self._lock:
            self._items[dataset_id] = dataset
            self._active[owner] = dataset_id
            self._prune_locked(now, owner)
        return dataset

    def derive(
        self,
        owner: str,
        parent: Dataset,
        dataframe: pd.DataFrame,
        cleaning: dict,
    ) -> Dataset:
        now = time.time()
        dataset = Dataset(
            id=uuid.uuid4().hex,
            owner=owner,
            name=parent.name,
            source_filename=parent.source_filename,
            dataframe=dataframe,
            revision=parent.revision + 1,
            root_id=parent.root_id,
            parent_id=parent.id,
            cleaning=cleaning,
            truncated=parent.truncated,
            total_rows_in_file=parent.total_rows_in_file,
            created_at=now,
            last_access=now,
            settings=self.settings,
        )
        with self._lock:
            self._items[dataset.id] = dataset
            self._active[owner] = dataset.id
            self._prune_locked(now, owner)
        return dataset

    def get(self, owner: str, dataset_id: str, *, touch: bool = True) -> Dataset:
        with self._lock:
            dataset = self._items.get(dataset_id)
            if dataset is None or dataset.owner != owner:
                raise DatasetNotFound(dataset_id)
            now = time.time()
            if now - dataset.last_access > self.settings.dataset_ttl_seconds:
                self._items.pop(dataset_id, None)
                raise DatasetNotFound(dataset_id)
            if touch:
                dataset.touch()
            return dataset

    def delete(self, owner: str, dataset_id: str) -> None:
        with self._lock:
            dataset = self._items.get(dataset_id)
            if dataset is None or dataset.owner != owner:
                raise DatasetNotFound(dataset_id)
            self._items.pop(dataset_id, None)
            if self._active.get(owner) == dataset_id:
                self._active.pop(owner, None)

    def activate(self, owner: str, dataset_id: str) -> Dataset:
        """Make ``dataset_id`` the revision the workspace points at."""
        dataset = self.get(owner, dataset_id)
        with self._lock:
            self._active[owner] = dataset_id
        return dataset

    def latest(self, owner: str) -> Dataset | None:
        with self._lock:
            active_id = self._active.get(owner)
            dataset = self._items.get(active_id) if active_id else None
            if dataset is None or dataset.owner != owner:
                owned = [item for item in self._items.values() if item.owner == owner]
                if not owned:
                    return None
                dataset = max(owned, key=lambda item: item.last_access)
                self._active[owner] = dataset.id
            if time.time() - dataset.last_access > self.settings.dataset_ttl_seconds:
                return None
            return dataset

    def close(self) -> None:
        with self._lock:
            self._items.clear()
            self._active.clear()

    # -- housekeeping ---------------------------------------------------------
    def _prune_locked(self, now: float, owner: str) -> None:
        expired = [
            key
            for key, item in self._items.items()
            if now - item.last_access > self.settings.dataset_ttl_seconds
        ]
        for key in expired:
            dataset = self._items.pop(key, None)
            if dataset and self._active.get(dataset.owner) == key:
                self._active.pop(dataset.owner, None)

        owned = sorted(
            (item for item in self._items.values() if item.owner == owner),
            key=lambda item: item.created_at,
        )
        excess = len(owned) - self.settings.max_datasets
        for dataset in owned[: max(excess, 0)]:
            self._items.pop(dataset.id, None)
            if self._active.get(owner) == dataset.id:
                self._active.pop(owner, None)
