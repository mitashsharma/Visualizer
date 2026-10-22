"""Chart engine.

Charts are described as small JSON *specs*: the server does the aggregation
(counts, bins, correlations, samples) and the browser draws SVG. This keeps
payloads tiny, avoids shipping PNGs, and makes charts interactive.

Selection is deliberately conservative. ``default_charts`` returns at most
``settings.max_default_charts`` charts chosen by analytical value:

* 1–2 distributions of the most informative numeric columns,
* a missing-value summary when data is incomplete,
* a correlation heatmap when ≥3 suitable numeric columns exist,
* one categorical comparison *or* a time trend, whichever fits the data.

Identifiers, constants and high-cardinality text are never auto-charted.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from src.config import SETTINGS, Settings
from src.eda import correlation_payload
from src.formatting import format_number as _fmt
from src.formatting import format_pct as _fmt_pct
from src.formatting import safe_number as _safe
from src.profile import (
    KIND_CATEGORICAL,
    KIND_DATETIME,
    KIND_NUMERIC,
    build_profile,
    columns_by_name,
)

MEASURE_HINT = re.compile(
    r"(income|revenue|sales|amount|price|cost|salary|score|rating|age|value|"
    r"spend|profit|balance|duration|tenure|count|quantity|units|margin)",
    re.IGNORECASE,
)

CHART_TYPES = [
    {
        "id": "histogram",
        "label": "Distribution",
        "description": "How one numeric column is spread out.",
        "x_kinds": [KIND_NUMERIC],
        "y_kinds": [],
        "requires_y": False,
    },
    {
        "id": "bar",
        "label": "Category counts",
        "description": "Most common values of a categorical column.",
        "x_kinds": [KIND_CATEGORICAL, "boolean"],
        "y_kinds": [],
        "requires_y": False,
    },
    {
        "id": "line",
        "label": "Trend over time",
        "description": "Row volume or a numeric average over time.",
        "x_kinds": [KIND_DATETIME],
        "y_kinds": [KIND_NUMERIC],
        "requires_y": False,
    },
    {
        "id": "scatter",
        "label": "Relationship",
        "description": "How two numeric columns move together.",
        "x_kinds": [KIND_NUMERIC],
        "y_kinds": [KIND_NUMERIC],
        "requires_y": True,
    },
    {
        "id": "box",
        "label": "Compare groups",
        "description": "Distribution of a numeric column per category.",
        "x_kinds": [KIND_CATEGORICAL, "boolean"],
        "y_kinds": [KIND_NUMERIC],
        "requires_y": True,
    },
    {
        "id": "heatmap",
        "label": "Correlation matrix",
        "description": "Pairwise correlation between numeric columns.",
        "x_kinds": [KIND_NUMERIC],
        "y_kinds": [],
        "requires_y": False,
    },
]


class ChartError(ValueError):
    """Raised for invalid chart requests (reported as HTTP 400)."""


# ---------------------------------------------------------------------------
# Column ranking
# ---------------------------------------------------------------------------


def _numeric_rank(profile: dict, *, min_unique: int = 8) -> list[dict]:
    ranked = []
    for column in profile["columns"]:
        if column["kind"] != KIND_NUMERIC:
            continue
        if column["is_identifier"] or column["is_constant"]:
            continue
        stats = column["stats"]
        if not stats or stats.get("count", 0) < 3:
            continue
        # Binary / low-cardinality numeric columns (targets, flags) are poor
        # distribution examples; they are still available in the builder.
        if column["unique"] < min_unique:
            continue

        mean = stats.get("mean") or 0.0
        std = stats.get("std") or 0.0
        cv = (std / abs(mean)) if abs(mean) > 1e-9 else min(std, 3.0)
        completeness = 1 - column["missing_pct"] / 100
        continuity = min(column["unique"] / 200, 1.0)
        hint = 0.35 if MEASURE_HINT.search(column["name"]) else 0.0

        score = completeness * (0.7 * continuity + 0.3 * min(cv, 2.0)) + hint
        ranked.append({"name": column["name"], "score": score, "column": column})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _categorical_rank(profile: dict, max_categories: int) -> list[dict]:
    ranked = []
    for column in profile["columns"]:
        if column["kind"] not in (KIND_CATEGORICAL, "boolean"):
            continue
        if column["is_identifier"] or column["is_constant"]:
            continue
        if not (2 <= column["unique"] <= max_categories):
            continue

        top_share = column["top_values"][0]["pct"] if column["top_values"] else 0.0
        completeness = 1 - column["missing_pct"] / 100
        balance = 1 - min(top_share, 100) / 100
        score = completeness * (0.5 + balance) + (
            0.2 if re.search(r"(segment|category|type|city|region|plan|group)", column["name"], re.I) else 0
        )
        ranked.append({"name": column["name"], "score": score, "column": column})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------


def histogram_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    column: str,
    *,
    bins: int = 24,
    settings: Settings | None = None,
) -> dict:
    settings = settings or SETTINGS
    info = columns_by_name(profile).get(column)
    if info is None:
        raise ChartError(f"Column '{column}' does not exist.")
    if info["kind"] != KIND_NUMERIC:
        raise ChartError(f"'{column}' is not numeric; distributions need numeric data.")

    series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if series.empty:
        raise ChartError(f"'{column}' has no values to plot.")

    unique = int(series.nunique())
    if unique <= 12:
        counts = series.value_counts().sort_index()
        edges = np.append(counts.index.to_numpy(dtype=float), counts.index.max() + 1)
        bins_data = [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts.iloc[i])}
            for i in range(len(counts))
        ]
    else:
        counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
        bins_data = [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]

    stats = info["stats"]
    skew = stats.get("skew") or 0.0
    if abs(skew) < 0.5:
        shape = "roughly symmetric"
    elif skew > 0:
        shape = "right-skewed"
    else:
        shape = "left-skewed"

    insight = (
        f"Median {_fmt(stats.get('median'))}, middle 50% between "
        f"{_fmt(stats.get('q1'))} and {_fmt(stats.get('q3'))}. "
        f"Distribution is {shape}"
    )
    if stats.get("outliers"):
        insight += f"; {stats['outliers']:,} values sit outside 1.5×IQR."
    else:
        insight += "."

    return {
        "id": f"histogram:{column}",
        "type": "histogram",
        "title": f"Distribution of {column}",
        "insight": insight,
        "x_label": column,
        "y_label": "Rows",
        "data": {
            "bins": bins_data,
            "stats": {
                key: _safe(stats.get(key))
                for key in ("min", "q1", "median", "q3", "max", "mean", "outliers")
            },
        },
        "meta": {
            "sampled": False,
            "rows": int(len(series)),
            "missing": info["missing"],
        },
    }


def bar_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    column: str,
    *,
    top_n: int = 10,
    settings: Settings | None = None,
) -> dict:
    settings = settings or SETTINGS
    top_n = max(3, min(int(top_n), 30))
    info = columns_by_name(profile).get(column)
    if info is None:
        raise ChartError(f"Column '{column}' does not exist.")
    if info["kind"] not in (KIND_CATEGORICAL, "boolean"):
        raise ChartError(f"'{column}' is numeric; use a distribution chart instead.")

    series = dataframe[column].dropna()
    total = int(len(series))
    if total == 0:
        raise ChartError(f"'{column}' has no values to plot.")

    counts = series.astype(str).value_counts()
    noted = False
    if len(counts) > top_n:
        head = counts.head(top_n)
        other = int(counts.iloc[top_n:].sum())
        noted = True
    else:
        head = counts
        other = 0

    non_null = max(total, 1)
    categories = [
        {
            "label": str(label)[:48],
            "count": int(count),
            "pct": round(100 * count / non_null, 2),
        }
        for label, count in head.items()
    ]
    if other:
        categories.append(
            {"label": "Other", "count": other, "pct": round(100 * other / non_null, 2)}
        )

    leading = categories[0]
    insight = (
        f"'{leading['label']}' is the most common value "
        f"({_fmt_pct(leading['pct'])} of {non_null:,} rows)."
    )
    if noted:
        top_sum = int(head.sum())
        insight += (
            f" Top {len(head)} of {len(counts):,} values cover "
            f"{_fmt_pct(100 * top_sum / non_null)} of rows."
        )
    if info["missing"]:
        insight += f" {info['missing']:,} missing values excluded."

    return {
        "id": f"bar:{column}",
        "type": "bar",
        "title": f"{column} by count",
        "insight": insight,
        "x_label": column,
        "y_label": "Rows",
        "data": {"categories": categories, "distinct": int(len(counts))},
        "meta": {
            "sampled": False,
            "rows": total,
            "missing": info["missing"],
            "truncated_categories": noted,
        },
    }


def line_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    x_column: str,
    y_column: str | None = None,
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or SETTINGS
    info = columns_by_name(profile).get(x_column)
    if info is None:
        raise ChartError(f"Column '{x_column}' does not exist.")
    if info["kind"] != KIND_DATETIME:
        raise ChartError(f"'{x_column}' is not a date column.")

    if y_column:
        y_info = columns_by_name(profile).get(y_column)
        if y_info is None:
            raise ChartError(f"Column '{y_column}' does not exist.")
        if y_info["kind"] != KIND_NUMERIC:
            raise ChartError(f"'{y_column}' must be numeric to average over time.")

    columns = [x_column] + ([y_column] if y_column else [])
    frame = dataframe[columns].dropna(subset=[x_column])
    frame = frame[frame[x_column].notna()]
    if frame.empty:
        raise ChartError(f"'{x_column}' has no valid dates.")

    span_days = (frame[x_column].max() - frame[x_column].min()).days
    if span_days <= 120:
        freq = "D"
    elif span_days <= 365 * 6:
        freq = "M"
    else:
        freq = "Y"

    def aggregate(frequency: str):
        keys = frame[x_column].dt.to_period(frequency)
        if y_column:
            grouped = frame.groupby(keys)[y_column].mean().dropna()
        else:
            grouped = keys.value_counts().sort_index()
        return grouped

    grouped = aggregate(freq)
    while len(grouped) > settings.max_line_points and freq != "Y":
        freq = {"D": "M", "M": "Y"}[freq]
        grouped = aggregate(freq)

    if len(grouped) > settings.max_line_points:
        step = math.ceil(len(grouped) / settings.max_line_points)
        grouped = grouped.iloc[::step]

    points = [[str(index), _safe(value)] for index, value in grouped.items()]

    first, last = grouped.iloc[0], grouped.iloc[-1]
    metric = f"mean {y_column}" if y_column else "row count"
    insight = (
        f"{x_column} spans {frame[x_column].min():%d %b %Y} to "
        f"{frame[x_column].max():%d %b %Y} ({span_days:,} days). "
    )
    if abs(float(first)) > 1e-9:
        change = 100 * (float(last) - float(first)) / abs(float(first))
        direction = "up" if change >= 0 else "down"
        insight += f"{metric.capitalize()} is {direction} {abs(change):.0f}% from the first to the last period."
    else:
        insight += f"Latest {metric} is {_fmt(last)}."

    return {
        "id": f"line:{x_column}:{y_column or 'count'}",
        "type": "line",
        "title": f"{y_column + ' over time' if y_column else 'Rows over time'}",
        "insight": insight,
        "x_label": x_column,
        "y_label": y_column or "Rows",
        "data": {"points": points, "metric": metric, "frequency": freq},
        "meta": {
            "sampled": False,
            "rows": int(len(frame)),
            "missing": info["missing"],
        },
    }


def scatter_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    x_column: str,
    y_column: str,
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or SETTINGS
    lookup = columns_by_name(profile)
    for name in (x_column, y_column):
        info = lookup.get(name)
        if info is None:
            raise ChartError(f"Column '{name}' does not exist.")
        if info["kind"] != KIND_NUMERIC:
            raise ChartError(f"'{name}' must be numeric for a scatter plot.")
    if x_column == y_column:
        raise ChartError("Choose two different columns.")

    frame = dataframe[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ChartError("These columns have no rows with both values present.")

    total = len(frame)
    correlation = frame[x_column].corr(frame[y_column])
    sampled = total > settings.scatter_points
    if sampled:
        frame = frame.sample(n=settings.scatter_points, random_state=settings.ml_random_state)

    points = [
        [round(float(x), 6), round(float(y), 6)]
        for x, y in zip(frame[x_column], frame[y_column])
    ]

    strength = "no"
    if correlation is not None and math.isfinite(float(correlation)):
        magnitude = abs(float(correlation))
        strength = (
            "strong" if magnitude >= 0.7 else "moderate" if magnitude >= 0.4 else "weak" if magnitude >= 0.2 else "no"
        )
        direction = "positive" if correlation >= 0 else "negative"
        insight = f"Correlation r = {float(correlation):.2f} ({strength} {direction} relationship)."
    else:
        insight = "Correlation could not be computed."

    if sampled:
        insight += f" {settings.scatter_points:,} of {total:,} rows plotted."

    return {
        "id": f"scatter:{x_column}:{y_column}",
        "type": "scatter",
        "title": f"{y_column} vs {x_column}",
        "insight": insight,
        "x_label": x_column,
        "y_label": y_column,
        "data": {
            "points": points,
            "correlation": _safe(correlation),
            "strength": strength,
        },
        "meta": {"sampled": sampled, "rows": int(total)},
    }


def heatmap_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    columns: list[str] | None = None,
    *,
    settings: Settings | None = None,
) -> dict:
    settings = settings or SETTINGS

    if columns is None:
        ranked = _numeric_rank(profile, min_unique=2)
        columns = [item["name"] for item in ranked[: settings.max_correlation_columns]]

    columns = [name for name in columns if name in dataframe.columns]
    if len(columns) < 2:
        raise ChartError("A correlation heatmap needs at least two numeric columns.")

    payload = correlation_payload(dataframe, columns)
    if not payload["columns"]:
        raise ChartError("Correlation could not be computed for these columns.")

    strongest = None
    for row_index, row in enumerate(payload["values"]):
        for col_index in range(row_index + 1, len(row)):
            value = row[col_index]
            if value is None:
                continue
            if strongest is None or abs(value) > abs(strongest["r"]):
                strongest = {
                    "a": payload["columns"][row_index],
                    "b": payload["columns"][col_index],
                    "r": value,
                }

    if strongest:
        insight = (
            f"Strongest relationship: {strongest['a']} ↔ {strongest['b']} "
            f"(r = {strongest['r']:.2f})."
        )
        strong_pairs = sum(
            1
            for i, row in enumerate(payload["values"])
            for j in range(i + 1, len(row))
            if row[j] is not None and abs(row[j]) >= 0.7
        )
        if strong_pairs:
            insight += f" {strong_pairs} pair(s) correlate above 0.70."
        else:
            insight += " No pair exceeds |r| = 0.70."
    else:
        insight = "Not enough complete numeric columns to compare."

    return {
        "id": "heatmap",
        "type": "heatmap",
        "title": "Correlation matrix",
        "insight": insight,
        "x_label": "",
        "y_label": "",
        "data": payload,
        "meta": {"sampled": False, "rows": int(len(dataframe))},
    }


def missing_spec(dataframe: pd.DataFrame, profile: dict) -> dict:
    columns = [c for c in profile["columns"] if c["missing"] > 0]
    if not columns:
        raise ChartError("This dataset has no missing values.")

    columns = sorted(columns, key=lambda item: item["missing_pct"], reverse=True)[:12]
    total_missing = int(profile["overview"]["missing_cells"])
    rows = max(profile["overview"]["rows"], 1)

    worst = columns[0]
    affected = int(profile["overview"]["columns"] - sum(
        1 for c in profile["columns"] if c["missing"] == 0
    ))

    return {
        "id": "missing",
        "type": "missing",
        "title": "Missing values by column",
        "insight": (
            f"{affected} of {profile['overview']['columns']} columns have gaps; "
            f"'{worst['name']}' is the most affected at {_fmt_pct(worst['missing_pct'])}. "
            f"{total_missing:,} cells missing overall ({_fmt_pct(profile['overview']['missing_pct'])})."
        ),
        "x_label": "Missing %",
        "y_label": "Column",
        "data": {
            "columns": [
                {
                    "column": column["name"],
                    "missing": int(column["missing"]),
                    "pct": float(column["missing_pct"]),
                    "rows": rows,
                }
                for column in columns
            ],
            "total_missing": total_missing,
        },
        "meta": {"sampled": False, "rows": rows},
    }


def box_spec(
    dataframe: pd.DataFrame,
    profile: dict,
    y_column: str,
    x_column: str,
    *,
    max_groups: int = 8,
    settings: Settings | None = None,
) -> dict:
    lookup = columns_by_name(profile)
    y_info = lookup.get(y_column)
    x_info = lookup.get(x_column)
    if y_info is None or x_info is None:
        raise ChartError("Both columns must exist.")
    if y_info["kind"] != KIND_NUMERIC:
        raise ChartError(f"'{y_column}' must be numeric.")
    if x_info["kind"] not in (KIND_CATEGORICAL, "boolean"):
        raise ChartError(f"'{x_column}' must be categorical.")

    frame = dataframe[[x_column, y_column]].copy()
    frame[y_column] = pd.to_numeric(frame[y_column], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise ChartError("No rows with both values present.")

    counts = frame[x_column].astype(str).value_counts()
    groups = counts.head(max_groups).index.tolist()
    frame["__group"] = frame[x_column].astype(str)

    payload = []
    medians = {}
    for group in groups:
        values = frame.loc[frame["__group"] == group, y_column].to_numpy(dtype=float)
        if values.size < 2:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr if iqr > 0 else values.min()
        upper = q3 + 1.5 * iqr if iqr > 0 else values.max()
        inside = values[(values >= lower) & (values <= upper)]
        payload.append(
            {
                "label": str(group)[:32],
                "count": int(values.size),
                "min": _safe(values.min()),
                "q1": _safe(q1),
                "median": _safe(median),
                "q3": _safe(q3),
                "max": _safe(values.max()),
                "whisker_low": _safe(inside.min() if inside.size else values.min()),
                "whisker_high": _safe(inside.max() if inside.size else values.max()),
                "outliers": int(values.size - inside.size),
            }
        )
        medians[str(group)] = median

    if not payload:
        raise ChartError("Not enough rows per group to draw a box plot.")

    top = max(payload, key=lambda item: item["median"])
    low = min(payload, key=lambda item: item["median"])
    insight = (
        f"Median {y_column} is highest for '{top['label']}' ({_fmt(top['median'])}) "
        f"and lowest for '{low['label']}' ({_fmt(low['median'])})."
    )

    return {
        "id": f"box:{y_column}:{x_column}",
        "type": "box",
        "title": f"{y_column} by {x_column}",
        "insight": insight,
        "x_label": x_column,
        "y_label": y_column,
        "data": {"groups": payload},
        "meta": {
            "sampled": False,
            "rows": int(len(frame)),
            "groups_shown": len(payload),
            "groups_total": int(x_info["unique"]),
        },
    }


# ---------------------------------------------------------------------------
# Curated defaults
# ---------------------------------------------------------------------------


def default_charts(
    dataframe: pd.DataFrame,
    profile: dict | None = None,
    *,
    settings: Settings | None = None,
) -> list[dict]:
    """Build a small, high-value chart set for this dataset."""
    settings = settings or SETTINGS
    profile = profile or build_profile(dataframe)
    charts: list[dict] = []

    def add(builder, *args, **kwargs):
        if len(charts) >= settings.max_default_charts:
            return
        try:
            spec = builder(dataframe, profile, *args, **kwargs)
        except (ChartError, ValueError, TypeError):
            return
        if all(existing["id"] != spec["id"] for existing in charts):
            charts.append(spec)

    row_count = profile["overview"]["rows"]
    min_unique = max(3, min(8, row_count // 10))
    ranked_all = _numeric_rank(profile, min_unique=2)
    numeric = [item for item in ranked_all if item["column"]["unique"] >= min_unique]
    for item in numeric[:2]:
        add(histogram_spec, item["name"])

    # Missing data is a first-class finding, not an afterthought.
    if profile["overview"]["missing_cells"]:
        add(missing_spec)

    numeric_candidates = [item["name"] for item in ranked_all]
    if len(ranked_all) >= 3:
        add(heatmap_spec, None)

    categories = _categorical_rank(profile, settings.max_categories)
    datetime_columns = profile["kinds"]["datetime"]

    if categories:
        add(bar_spec, categories[0]["name"])
    if datetime_columns:
        add(line_spec, datetime_columns[0])
    if not categories and not datetime_columns and len(numeric_candidates) >= 2:
        best_pair = _strongest_pair(dataframe, numeric_candidates[:6])
        if best_pair and abs(best_pair[2]) >= 0.3:
            add(scatter_spec, best_pair[0], best_pair[1])

    return charts


def _strongest_pair(dataframe: pd.DataFrame, columns: list[str]):
    if len(columns) < 2:
        return None
    matrix = dataframe[columns].apply(pd.to_numeric, errors="coerce").corr()
    best = None
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            value = matrix.loc[first, second]
            if value is None or not math.isfinite(float(value)):
                continue
            if best is None or abs(float(value)) > abs(best[2]):
                best = (first, second, float(value))
    return best


def chart_catalog(profile: dict) -> dict:
    return {
        "types": CHART_TYPES,
        "columns": [
            {
                "name": column["name"],
                "kind": column["kind"],
                "unique": column["unique"],
                "missing_pct": column["missing_pct"],
                "is_identifier": column["is_identifier"],
                "is_constant": column["is_constant"],
            }
            for column in profile["columns"]
        ],
    }


def build_chart(
    dataframe: pd.DataFrame,
    profile: dict,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    *,
    bins: int = 24,
    top_n: int = 10,
    settings: Settings | None = None,
) -> dict:
    """Validate a manual chart request and return its spec."""
    settings = settings or SETTINGS
    chart_type = (chart_type or "").strip().lower()

    if chart_type == "histogram":
        if not x:
            raise ChartError("Choose a numeric column for the distribution.")
        return histogram_spec(dataframe, profile, x, bins=bins, settings=settings)

    if chart_type == "bar":
        if not x:
            raise ChartError("Choose a categorical column.")
        return bar_spec(dataframe, profile, x, top_n=top_n, settings=settings)

    if chart_type == "line":
        if not x:
            raise ChartError("Choose a date column for the trend.")
        return line_spec(dataframe, profile, x, y, settings=settings)

    if chart_type == "scatter":
        if not x or not y:
            raise ChartError("Choose two numeric columns.")
        return scatter_spec(dataframe, profile, x, y, settings=settings)

    if chart_type == "box":
        if not x or not y:
            raise ChartError("Choose a categorical column and a numeric column.")
        return box_spec(dataframe, profile, y, x, settings=settings)

    if chart_type == "heatmap":
        columns = [name for name in (x or "").split(",") if name] or None
        return heatmap_spec(dataframe, profile, columns, settings=settings)

    if chart_type == "missing":
        return missing_spec(dataframe, profile)

    raise ChartError(f"Unknown chart type '{chart_type}'.")


def all_identifier_free_columns(profile: dict) -> list[str]:
    return [
        column["name"]
        for column in profile["columns"]
        if not column["is_identifier"] and not column["is_constant"]
    ]
