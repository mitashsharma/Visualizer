"""Exploratory data analysis.

Thin, cached layer over :mod:`src.profile`. Every function accepts an optional
pre-computed profile so callers can avoid re-scanning the DataFrame; this is
what removes the duplicated overview/outlier passes in the original code.

The public functions keep their original signatures and return types for
backwards compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.profile import (
    KIND_CATEGORICAL,
    build_profile,
    columns_by_name,
    numeric_columns as profile_numeric_columns,
)


def validate_dataframe(dataframe: pd.DataFrame) -> bool:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    return True


def _overview_from_profile(profile: dict) -> dict:
    overview = dict(profile["overview"])
    overview.update(
        {
            "column_names": [column["name"] for column in profile["columns"]],
            "numerical_columns": profile["kinds"]["numeric"],
            "categorical_columns": profile["kinds"]["categorical"]
            + profile["kinds"]["boolean"],
            "datetime_columns": profile["kinds"]["datetime"],
            "missing_values": overview["missing_cells"],
        }
    )
    return overview


def dataset_overview(dataframe: pd.DataFrame, profile: dict | None = None) -> dict:
    validate_dataframe(dataframe)
    profile = profile or build_profile(dataframe)
    return _overview_from_profile(profile)


def numerical_statistics(dataframe: pd.DataFrame, profile: dict | None = None) -> pd.DataFrame:
    validate_dataframe(dataframe)
    profile = profile or build_profile(dataframe)

    rows = []
    for column in profile["columns"]:
        if column["kind"] != "numeric" or not column["stats"]:
            continue
        stats = column["stats"]
        rows.append(
            {
                "column": column["name"],
                "count": stats.get("count"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "q1": stats.get("q1"),
                "median": stats.get("median"),
                "q3": stats.get("q3"),
                "max": stats.get("max"),
                "outliers": stats.get("outliers"),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("column")


def categorical_statistics(dataframe: pd.DataFrame, profile: dict | None = None) -> dict:
    validate_dataframe(dataframe)
    profile = profile or build_profile(dataframe)

    statistics = {}
    for column in profile["columns"]:
        if column["kind"] not in (KIND_CATEGORICAL, "boolean"):
            continue

        if column["top_values"]:
            top = column["top_values"][0]
            statistics[column["name"]] = {
                "unique_values": column["unique"],
                "most_frequent": top["value"],
                "frequency": top["count"],
            }
        else:
            series = dataframe[column["name"]].dropna()
            counts = series.value_counts()
            statistics[column["name"]] = {
                "unique_values": column["unique"],
                "most_frequent": str(counts.index[0]) if not counts.empty else None,
                "frequency": int(counts.iloc[0]) if not counts.empty else 0,
            }

    return statistics


def correlation_matrix(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    max_columns: int | None = None,
) -> pd.DataFrame:
    validate_dataframe(dataframe)

    if columns is None:
        columns = dataframe.select_dtypes(include="number").columns.tolist()

    selected = []
    for name in columns:
        if name not in dataframe.columns:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[name]):
            selected.append(name)

    if max_columns:
        selected = selected[:max_columns]

    if len(selected) < 2:
        return pd.DataFrame()

    return dataframe[selected].corr(numeric_only=True)


def detect_outliers(dataframe: pd.DataFrame, profile: dict | None = None) -> dict:
    if profile is None:
        validate_dataframe(dataframe)
        profile = build_profile(dataframe)

    results = {}
    for column in profile["columns"]:
        if column["kind"] != "numeric":
            continue
        stats = column["stats"]
        results[column["name"]] = {
            "outlier_count": int(stats.get("outliers", 0)),
            "lower_bound": stats.get("lower_bound"),
            "upper_bound": stats.get("upper_bound"),
        }
    return results


def generate_insights(dataframe: pd.DataFrame, profile: dict | None = None) -> list[str]:
    """Return human-readable findings (string list, for compatibility)."""
    validate_dataframe(dataframe)
    from src.summarizer import build_insights

    profile = profile or build_profile(dataframe)
    insights = build_insights(profile, dataframe)
    messages = [insights["summary"]] if insights.get("summary") else []
    messages += [finding["detail"] for finding in insights["findings"]]
    return messages


def perform_eda(dataframe: pd.DataFrame, profile: dict | None = None) -> dict:
    """Run the classic EDA bundle with a single profile pass."""
    validate_dataframe(dataframe)
    profile = profile or build_profile(dataframe)

    return {
        "overview": _overview_from_profile(profile),
        "numerical_statistics": numerical_statistics(dataframe, profile),
        "categorical_statistics": categorical_statistics(dataframe, profile),
        "correlation_matrix": correlation_matrix(
            dataframe, profile_numeric_columns(profile, include_identifiers=True)
        ),
        "outliers": detect_outliers(dataframe, profile),
        "insights": generate_insights(dataframe, profile),
        "profile": profile,
    }


def correlation_payload(dataframe: pd.DataFrame, columns: list[str]) -> dict:
    """Correlation matrix as JSON-friendly lists (None for undefined pairs)."""
    matrix = correlation_matrix(dataframe, columns)
    if matrix.empty:
        return {"columns": [], "values": []}

    values = []
    for row in matrix.to_numpy():
        values.append(
            [None if not np.isfinite(value) else round(float(value), 3) for value in row]
        )

    return {"columns": [str(name) for name in matrix.columns], "values": values}


def outlier_payload(profile: dict) -> list[dict]:
    payload = []
    for column in profile["columns"]:
        if column["kind"] != "numeric":
            continue
        stats = column["stats"]
        count = int(stats.get("outliers", 0))
        total = int(stats.get("count", 0) or 0)
        payload.append(
            {
                "column": column["name"],
                "outlier_count": count,
                "lower_bound": stats.get("lower_bound"),
                "upper_bound": stats.get("upper_bound"),
                "share_pct": round(100 * count / total, 2) if total else 0.0,
            }
        )
    return payload
