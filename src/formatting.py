"""Small number/date formatting helpers shared by charts, insights and reports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def format_number(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"

    magnitude = abs(number)
    if magnitude >= 1e9:
        return f"{number / 1e9:.{digits}f}B"
    if magnitude >= 1e6:
        return f"{number / 1e6:.{digits}f}M"
    if magnitude >= 1_000:
        return f"{number:,.0f}"
    if magnitude >= 10:
        return f"{number:,.1f}".rstrip("0").rstrip(".")
    return f"{number:.2f}".rstrip("0").rstrip(".")


def format_pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def format_metric(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def safe_number(value):
    """Convert NumPy scalars/NaN to JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [safe_number(item) for item in value.tolist()]
    return value
