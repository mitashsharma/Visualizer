"""Dataset profiling.

Every statistic the rest of the application needs is computed here, once per
dataset revision, in a single vectorised pass. ``perform_eda``, the chart
engine, the ML module and the report all read this profile instead of
recomputing summaries.

Design notes
------------
* Missing counts are computed for the whole frame at once (``df.isna().sum()``).
* Per-column statistics use NumPy on a float view, avoiding pandas overhead.
* Identifier detection combines name hints with a uniqueness test, so IDs are
  never charted or fed to a model by accident.
* Everything returned is JSON-serialisable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ID_WORDS = (
    "id|uuid|guid|key|index|code|sku|isbn|email|phone|mobile|account|card|"
    "passport|ssn|aadhaar|iban|no|number|num|ref|reference"
)
# Match the whole name or a trailing word ("customer_id", "referral_code",
# "order number") but not a compound measure such as "account_balance".
IDENTIFIER_NAME = re.compile(
    rf"^(?:{_ID_WORDS})$|(?:^|[_\s-])(?:{_ID_WORDS})$",
    re.IGNORECASE,
)

_CODE_WORDS = _ID_WORDS + "|zip|zipcode|postal|postcode|pin"
CODE_NAME = re.compile(
    rf"^(?:{_CODE_WORDS})$|(?:^|[_\s-])(?:{_CODE_WORDS})$",
    re.IGNORECASE,
)

UNNAMED = re.compile(r"^Unnamed:\s*\d+$", re.IGNORECASE)
NUMERIC_TEXT = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")

KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"
KIND_BOOLEAN = "boolean"
KIND_DATETIME = "datetime"

MAX_TOP_VALUES = 8


def column_kind(series: pd.Series) -> str:
    """Return the analytical kind of a pandas Series."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return KIND_BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return KIND_DATETIME
    if pd.api.types.is_numeric_dtype(dtype):
        return KIND_NUMERIC
    return KIND_CATEGORICAL


def _safe_float(value, digits: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _label(value) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()[:19]
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _numeric_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}

    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    skew = 0.0
    if std > 0:
        centred = finite - mean
        skew = float(np.mean(centred**3) / (std**3))

    iqr = float(q3 - q1)
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(np.count_nonzero((finite < lower) | (finite > upper)))
    else:
        lower = upper = None
        outliers = 0

    return {
        "count": int(finite.size),
        "mean": _safe_float(mean),
        "std": _safe_float(std),
        "min": _safe_float(finite.min()),
        "q1": _safe_float(q1),
        "median": _safe_float(median),
        "q3": _safe_float(q3),
        "max": _safe_float(finite.max()),
        "zeros": int(np.count_nonzero(finite == 0)),
        "negative": int(np.count_nonzero(finite < 0)),
        "skew": _safe_float(skew, 3),
        "outliers": outliers,
        "lower_bound": _safe_float(lower),
        "upper_bound": _safe_float(upper),
    }


def _datetime_stats(series: pd.Series) -> dict:
    valid = series.dropna()
    if valid.empty:
        return {}
    minimum = valid.min()
    maximum = valid.max()
    return {
        "min": minimum.isoformat()[:19],
        "max": maximum.isoformat()[:19],
        "span_days": int((maximum - minimum).days),
    }


def _top_values(series: pd.Series, unique: int, total: int) -> list[dict]:
    if unique == 0 or unique > 200:
        return []
    counts = series.value_counts(dropna=True).head(MAX_TOP_VALUES)
    non_null = max(int(series.notna().sum()), 1)
    return [
        {
            "value": _label(index),
            "count": int(count),
            "pct": round(100 * count / non_null, 2),
        }
        for index, count in counts.items()
    ]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_identifier: bool = False
    is_constant: bool = False
    is_high_cardinality: bool = False
    whitespace_issues: bool = False
    stats: dict = field(default_factory=dict)
    top_values: list = field(default_factory=list)
    sample: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _profile_column(
    series: pd.Series,
    *,
    missing: int,
    rows: int,
    unique: int,
) -> ColumnProfile:
    kind = column_kind(series)
    non_null = rows - missing
    unique_pct = (unique / non_null * 100) if non_null else 0.0
    name_is_identifier = bool(IDENTIFIER_NAME.search(series.name or ""))
    unique_ratio = (unique / non_null) if non_null else 0.0

    # A column is an identifier if its *name* says so, or if it is a text /
    # sequential-integer column where every row is unique. Continuous numeric
    # measures with many distinct values (income, scores, prices) are data,
    # not identifiers.
    unique_rule = False
    if kind == KIND_CATEGORICAL and rows >= 5 and unique == non_null and unique >= 5:
        # Every row has a different label: a key in any realistic dataset.
        unique_rule = True
    elif rows >= 50 and unique >= 20 and unique_ratio >= 0.98 and non_null:
        if kind == KIND_CATEGORICAL:
            unique_rule = True
        elif kind == KIND_NUMERIC and pd.api.types.is_integer_dtype(series):
            sample = series.dropna().head(200).to_numpy(dtype="float64")
            if sample.size >= 20:
                steps = np.diff(sample)
                unique_rule = bool(
                    np.all(steps == steps[0])
                    and abs(steps[0]) <= 2  # row number, not a measure
                )

    is_identifier = kind != KIND_DATETIME and (name_is_identifier or unique_rule)
    is_constant = unique <= 1

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        kind=kind,
        missing=int(missing),
        missing_pct=round(100 * missing / rows, 2) if rows else 0.0,
        unique=int(unique),
        unique_pct=round(unique_pct, 2),
        is_identifier=is_identifier,
        is_constant=is_constant,
        is_high_cardinality=kind == KIND_CATEGORICAL and unique > 50,
    )

    if kind == KIND_NUMERIC:
        values = series.to_numpy(dtype="float64", copy=False, na_value=np.nan)
        profile.stats = _numeric_stats(values)
    elif kind == KIND_DATETIME:
        profile.stats = _datetime_stats(series)
    elif kind == KIND_CATEGORICAL:
        sample = series.dropna().head(400)
        if not sample.empty and any(
            isinstance(value, str) and value != value.strip() for value in sample
        ):
            profile.whitespace_issues = True

    profile.top_values = _top_values(series, unique, rows)

    if not is_identifier:
        samples = series.dropna().head(3).tolist()
        profile.sample = [_label(value) for value in samples]

    return profile


def build_profile(
    dataframe: pd.DataFrame,
    *,
    source: str | None = None,
    truncated: bool = False,
    extra: dict | None = None,
) -> dict:
    """Compute a full, JSON-ready profile for ``dataframe``."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Dataset has no rows.")

    rows, columns = dataframe.shape
    missing_counts = dataframe.isna().sum()
    duplicate_rows = int(dataframe.duplicated().sum())

    # One pass per column for cardinality; value_counts is reused by the
    # profile where needed.
    profiles = []
    for name in dataframe.columns:
        series = dataframe[name]
        unique = int(series.nunique(dropna=True))
        profiles.append(
            _profile_column(
                series,
                missing=int(missing_counts[name]),
                rows=rows,
                unique=unique,
            )
        )

    total_cells = rows * columns
    missing_cells = int(missing_counts.sum())
    numeric = [c for c in profiles if c.kind == KIND_NUMERIC]
    categorical = [c for c in profiles if c.kind == KIND_CATEGORICAL]
    datetimes = [c for c in profiles if c.kind == KIND_DATETIME]
    booleans = [c for c in profiles if c.kind == KIND_BOOLEAN]

    overview = {
        "rows": int(rows),
        "columns": int(columns),
        "cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2)
        if total_cells
        else 0.0,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(100 * duplicate_rows / rows, 2) if rows else 0.0,
        "numeric_columns": len(numeric),
        "categorical_columns": len(categorical),
        "datetime_columns": len(datetimes),
        "boolean_columns": len(booleans),
        "identifier_columns": sum(1 for c in profiles if c.is_identifier),
        "constant_columns": sum(1 for c in profiles if c.is_constant),
        "memory_mb": round(
            float(dataframe.memory_usage(deep=True).sum()) / 1_048_576, 2
        ),
        "source": source,
        "truncated": truncated,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        overview.update(extra)

    quality = _quality_issues(profiles, overview)

    return {
        "overview": overview,
        "columns": [column.to_dict() for column in profiles],
        "quality": quality,
        "kinds": {
            "numeric": [c.name for c in numeric],
            "categorical": [c.name for c in categorical],
            "datetime": [c.name for c in datetimes],
            "boolean": [c.name for c in booleans],
            "identifier": [c.name for c in profiles if c.is_identifier],
        },
    }


def _quality_issues(profiles: list[ColumnProfile], overview: dict) -> list[dict]:
    issues: list[dict] = []

    if overview["duplicate_rows"]:
        share = overview["duplicate_pct"]
        issues.append(
            {
                "severity": "high" if share >= 5 else "medium",
                "code": "duplicate_rows",
                "title": f"{overview['duplicate_rows']:,} duplicate rows",
                "detail": f"{share}% of rows are exact duplicates.",
                "column": None,
                "action": "remove_duplicates",
            }
        )

    if overview["missing_cells"]:
        heavy = [c for c in profiles if c.missing_pct >= 10]
        issues.append(
            {
                "severity": "high" if heavy else "medium",
                "code": "missing_values",
                "title": f"{overview['missing_cells']:,} missing cells "
                f"({overview['missing_pct']}%)",
                "detail": (
                    "Columns most affected: "
                    + ", ".join(f"{c.name} ({c.missing_pct}%)" for c in heavy[:4])
                    if heavy
                    else "Missing values are spread across the dataset."
                ),
                "column": None,
                "action": "handle_missing",
            }
        )

    constant = [c for c in profiles if c.is_constant]
    if constant:
        issues.append(
            {
                "severity": "medium",
                "code": "constant_columns",
                "title": f"{len(constant)} column(s) hold a single value",
                "detail": ", ".join(c.name for c in constant[:6])
                + " carry no information and can be dropped.",
                "column": None,
                "action": "drop_columns",
            }
        )

    whitespace = [c for c in profiles if c.whitespace_issues]
    if whitespace:
        issues.append(
            {
                "severity": "low",
                "code": "whitespace",
                "title": "Inconsistent whitespace in text columns",
                "detail": ", ".join(c.name for c in whitespace[:6])
                + " contain values with leading/trailing spaces.",
                "column": None,
                "action": "trim_whitespace",
            }
        )

    identifiers = [c for c in profiles if c.is_identifier]
    if identifiers:
        issues.append(
            {
                "severity": "low",
                "code": "identifiers",
                "title": f"{len(identifiers)} identifier column(s) detected",
                "detail": ", ".join(c.name for c in identifiers[:6])
                + " uniquely identify rows. They are excluded from charts and "
                "models unless selected explicitly.",
                "column": None,
                "action": None,
            }
        )

    high_card = [c for c in profiles if c.is_high_cardinality]
    if high_card:
        issues.append(
            {
                "severity": "low",
                "code": "high_cardinality",
                "title": f"{len(high_card)} high-cardinality text column(s)",
                "detail": ", ".join(f"{c.name} ({c.unique:,} values)" for c in high_card[:5])
                + ". They are summarised as top values and excluded from models.",
                "column": None,
                "action": None,
            }
        )

    if not overview["numeric_columns"]:
        issues.append(
            {
                "severity": "low",
                "code": "no_numeric",
                "title": "No numeric columns",
                "detail": "Charts and models will use categorical data only.",
                "column": None,
                "action": None,
            }
        )

    return issues


def numeric_columns(profile: dict, *, include_identifiers: bool = False) -> list[str]:
    return [
        column["name"]
        for column in profile["columns"]
        if column["kind"] == KIND_NUMERIC
        and (include_identifiers or not column["is_identifier"])
        and not column["is_constant"]
    ]


def categorical_columns(profile: dict, *, include_identifiers: bool = False) -> list[str]:
    return [
        column["name"]
        for column in profile["columns"]
        if column["kind"] in (KIND_CATEGORICAL, KIND_BOOLEAN)
        and (include_identifiers or not column["is_identifier"])
        and not column["is_constant"]
    ]


def datetime_columns(profile: dict) -> list[str]:
    return [
        column["name"]
        for column in profile["columns"]
        if column["kind"] == KIND_DATETIME
    ]


def columns_by_name(profile: dict) -> dict:
    return {column["name"]: column for column in profile["columns"]}
