"""Data cleaning.

Cleaning is expressed as an explicit plan so the UI can preview it, apply it
to a *new* immutable revision, and let the user undo it. The legacy helpers
(``remove_duplicates``, ``handle_missing_values``, ``clean_text_columns``,
``clean_data``) are kept because they are a convenient API and are covered by
tests.

Nothing here mutates the caller's DataFrame.
"""

from __future__ import annotations

import time

import pandas as pd

MISSING_STRATEGIES = ("leave", "median", "mean", "mode", "zero", "drop_rows")
DEFAULT_PLAN = {
    "remove_duplicates": False,
    "trim_whitespace": False,
    "drop_columns": [],
    "missing": {},
}


def _require_frame(dataframe) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def _strip_text(series: pd.Series) -> pd.Series:
    """Trim text cells without corrupting non-string values.

    ``Series.str.strip`` returns NaN for non-strings, so those positions are
    restored from the original column.
    """
    stripped = series.str.strip()
    restore = stripped.isna() & series.notna()
    if bool(restore.any()):
        stripped = stripped.where(~restore, series)
    return stripped


def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without duplicate rows."""
    _require_frame(dataframe)
    return dataframe.drop_duplicates().reset_index(drop=True)


def handle_missing_values(dataframe: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """Fill or drop missing values.

    Strategies: ``auto`` (median for numeric, mode for text), ``drop`` (drop
    rows with any missing value), ``none``.
    """
    _require_frame(dataframe)
    if strategy not in {"auto", "drop", "none"}:
        raise ValueError("Invalid strategy. Use 'auto', 'drop', or 'none'.")

    cleaned = dataframe.copy()
    if strategy == "none":
        return cleaned
    if strategy == "drop":
        return cleaned.dropna().reset_index(drop=True)

    missing = cleaned.isna().sum()
    for column in cleaned.columns:
        if missing[column] == 0:
            continue

        series = cleaned[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            fill = series.median()
            if pd.isna(fill):
                continue
        else:
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else "Unknown"

        cleaned[column] = series.fillna(fill)

    return cleaned


def clean_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from text columns (vectorised)."""
    _require_frame(dataframe)
    cleaned = dataframe.copy()
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = _strip_text(cleaned[column])
    return cleaned


def clean_data(dataframe: pd.DataFrame, missing_strategy: str = "auto") -> pd.DataFrame:
    """Apply the classic cleaning pipeline to a copy of ``dataframe``."""
    _require_frame(dataframe)
    cleaned = remove_duplicates(dataframe)
    cleaned = handle_missing_values(cleaned, strategy=missing_strategy)
    return clean_text_columns(cleaned)


def get_cleaning_summary(original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict:
    """Compare two frames and report what changed."""
    _require_frame(original_data)
    _require_frame(cleaned_data)

    return {
        "original_rows": int(len(original_data)),
        "cleaned_rows": int(len(cleaned_data)),
        "rows_removed": int(len(original_data) - len(cleaned_data)),
        "duplicates_removed": int(
            len(original_data) - len(original_data.drop_duplicates())
        ),
        "missing_values_before": int(original_data.isna().sum().sum()),
        "missing_values_after": int(cleaned_data.isna().sum().sum()),
    }


def recommended_plan(dataframe: pd.DataFrame, profile: dict) -> dict:
    """Build the default cleaning plan for the Prepare page."""
    _require_frame(dataframe)

    columns = {column["name"]: column for column in profile["columns"]}
    total_missing = int(profile["overview"]["missing_cells"])
    duplicates = int(profile["overview"]["duplicate_rows"])

    drop_columns = [
        column["name"]
        for column in profile["columns"]
        if column["is_constant"] and not column["is_identifier"]
    ]

    strategies: dict[str, str] = {}
    for name, column in columns.items():
        if name in drop_columns or not column["missing"]:
            continue
        if column["missing_pct"] > 30:
            strategies[name] = "leave"
        elif column["kind"] == "numeric":
            strategies[name] = "median"
        elif column["kind"] in ("categorical", "boolean"):
            strategies[name] = "mode"
        else:
            strategies[name] = "leave"

    return {
        "remove_duplicates": duplicates > 0,
        "trim_whitespace": any(c["whitespace_issues"] for c in profile["columns"]),
        "drop_columns": drop_columns,
        "missing": strategies,
        "summary": {
            "duplicate_rows": duplicates,
            "missing_cells": total_missing,
            "whitespace_columns": [
                c["name"] for c in profile["columns"] if c["whitespace_issues"]
            ],
        },
    }


def normalise_plan(plan: dict | None, columns: list[str]) -> dict:
    """Validate a client-supplied plan and fill defaults."""
    if not isinstance(plan, dict):
        raise ValueError("Cleaning plan must be an object.")
    if not columns:
        raise ValueError("Dataset has no columns.")

    drop_columns = plan.get("drop_columns") or []
    if not isinstance(drop_columns, list):
        raise ValueError("'drop_columns' must be a list of column names.")

    unknown_drops = [name for name in drop_columns if name not in columns]
    if unknown_drops:
        raise ValueError(f"Unknown column(s) in drop list: {', '.join(unknown_drops)}")
    if len(drop_columns) >= len(columns):
        raise ValueError("At least one column must remain after cleaning.")

    missing = plan.get("missing") or {}
    if not isinstance(missing, dict):
        raise ValueError("'missing' must map column names to strategies.")

    cleaned_missing: dict[str, str] = {}
    for name, strategy in missing.items():
        if name not in columns:
            raise ValueError(f"Unknown column in missing-value plan: {name}")
        if name in drop_columns:
            continue
        if strategy not in MISSING_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}' for '{name}'. "
                f"Allowed: {', '.join(MISSING_STRATEGIES)}."
            )
        if strategy != "leave":
            cleaned_missing[name] = strategy

    return {
        "remove_duplicates": bool(plan.get("remove_duplicates")),
        "trim_whitespace": bool(plan.get("trim_whitespace")),
        "drop_columns": list(dict.fromkeys(drop_columns)),
        "missing": cleaned_missing,
    }


def apply_cleaning_plan(
    dataframe: pd.DataFrame,
    plan: dict | None,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply a validated plan and return ``(cleaned_frame, summary)``."""
    _require_frame(dataframe)
    started = time.perf_counter()
    plan = normalise_plan(plan, list(dataframe.columns))

    rows_before, columns_before = dataframe.shape
    missing_before = int(dataframe.isna().sum().sum())

    working = dataframe.copy()

    if plan["trim_whitespace"]:
        text_columns = working.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            working[column] = _strip_text(working[column])

    duplicates_removed = 0
    if plan["remove_duplicates"]:
        before = len(working)
        working = working.drop_duplicates()
        duplicates_removed = before - len(working)

    filled_cells = 0
    dropped_rows_for_missing = 0
    applied: list[str] = []

    if "drop_columns" in plan and plan["drop_columns"]:
        dropped = plan["drop_columns"]
        working = working.drop(columns=dropped)
        applied.append(f"Dropped {len(dropped)} column(s): {', '.join(dropped)}")

    if plan["missing"]:
        drop_rows_columns = []
        for column, strategy in plan["missing"].items():
            if column not in working.columns:
                continue
            series = working[column]
            missing = int(series.isna().sum())
            if missing == 0:
                continue

            if strategy == "drop_rows":
                drop_rows_columns.append(column)
                continue

            if strategy == "median":
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"Median fill requires a numeric column: {column}")
                fill = series.median()
            elif strategy == "mean":
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"Mean fill requires a numeric column: {column}")
                fill = series.mean()
            elif strategy == "zero":
                fill = 0
            else:  # mode
                modes = series.mode(dropna=True)
                fill = modes.iloc[0] if not modes.empty else "Unknown"

            if pd.isna(fill):
                continue
            working[column] = series.fillna(fill)
            filled_cells += missing
            applied.append(f"Filled {missing:,} missing value(s) in '{column}' ({strategy})")

        if drop_rows_columns:
            before = len(working)
            working = working.dropna(subset=drop_rows_columns)
            dropped_rows_for_missing = before - len(working)
            applied.append(
                f"Dropped {dropped_rows_for_missing:,} row(s) missing "
                f"{', '.join(drop_rows_columns)}"
            )

    if plan["remove_duplicates"]:
        applied.insert(0, f"Removed {duplicates_removed:,} duplicate row(s)")
    if plan["trim_whitespace"]:
        applied.insert(0, "Trimmed whitespace in text columns")

    working = working.reset_index(drop=True)

    if max_rows and len(working) > max_rows:
        working = working.iloc[:max_rows].copy()

    missing_after = int(working.isna().sum().sum())

    summary = {
        "rows_before": int(rows_before),
        "rows_after": int(len(working)),
        "rows_removed": int(rows_before - len(working)),
        "columns_before": int(columns_before),
        "columns_after": int(working.shape[1]),
        "duplicates_removed": int(duplicates_removed),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "filled_cells": int(filled_cells),
        "dropped_rows_for_missing": int(dropped_rows_for_missing),
        "actions": applied,
        "plan": plan,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }

    return working, summary
