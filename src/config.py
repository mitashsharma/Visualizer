"""Central configuration for the AI Data Visualizer.

All tunables live here so behaviour can be changed without editing modules.
Any value can be overridden with an ``AIVIZ_``-prefixed environment variable,
e.g. ``AIVIZ_ML_MAX_TRAIN_ROWS=20000 python app.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields


def _env(name: str, default):
    raw = os.environ.get(f"AIVIZ_{name.upper()}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    """Runtime limits and defaults."""

    # Upload / parsing
    max_upload_mb: int = 100
    max_rows: int = 500_000
    preview_rows: int = 25

    # Charts
    max_default_charts: int = 5
    max_correlation_columns: int = 10
    max_categories: int = 12
    scatter_points: int = 3_000
    max_line_points: int = 160

    # Machine learning
    ml_max_train_rows: int = 50_000
    ml_max_cardinality: int = 40
    ml_test_size: float = 0.2
    ml_random_state: int = 42
    rf_estimators: int = 60
    hgb_max_iter: int = 150

    # Server / lifecycle
    dataset_ttl_seconds: int = 3_600
    max_datasets: int = 4
    job_workers: int = 2
    job_retention_seconds: int = 1_800

    # Reports
    report_chart_limit: int = 4

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        values = {field.name: _env(field.name, field.default) for field in fields(cls)}
        return cls(**values)


SETTINGS = Settings.from_env()


def clamp_rows(rows: int, settings: Settings | None = None) -> int:
    """Return a sane row cap (never below 100 to keep small demos usable)."""
    settings = settings or SETTINGS
    return int(max(100, min(rows, settings.max_rows)))


def spread_score(values) -> float:
    """Scale-free dispersion used to rank numeric columns for charts."""
    try:
        mean = float(values.mean())
        std = float(values.std())
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(std):
        return 0.0
    if abs(mean) > 1e-9:
        return min(std / abs(mean), 10.0)
    return min(std, 10.0)
