"""Automated insights.

Findings are derived from computed statistics — correlations, skew, outlier
share, concentration, missingness and model metrics. Nothing here is generated
by a language model and no numbers are invented: every sentence can be traced
back to the profile or the trained model.

This module replaces the duplicated ``eda.generate_insights`` and the old
``generate_summary`` (both of which only restated row/column counts).
"""

from __future__ import annotations

import math

import pandas as pd

from src.formatting import format_number, format_pct
from src.profile import (
    KIND_CATEGORICAL,
    columns_by_name,
    numeric_columns,
)


def _finding(kind: str, title: str, detail: str, severity: str = "info") -> dict:
    return {"kind": kind, "title": title, "detail": detail, "severity": severity}


def _correlation_pairs(dataframe: pd.DataFrame, profile: dict, limit: int = 8) -> list[dict]:
    """Strongest |r| pairs among the most variable numeric columns."""
    candidates = []
    lookup = columns_by_name(profile)
    for name in numeric_columns(profile):
        stats = lookup[name]["stats"]
        if not stats or stats.get("count", 0) < 10:
            continue
        candidates.append((abs(stats.get("std") or 0), name))

    candidates.sort(reverse=True)
    columns = [name for _std, name in candidates[:limit]]
    if len(columns) < 2:
        return []

    matrix = dataframe[columns].apply(pd.to_numeric, errors="coerce").corr()
    pairs = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            value = matrix.loc[first, second]
            if value is None or not math.isfinite(float(value)):
                continue
            pairs.append({"a": first, "b": second, "r": round(float(value), 3)})

    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    return pairs


def build_insights(
    profile: dict,
    dataframe: pd.DataFrame | None = None,
    ml_result: dict | None = None,
) -> dict:
    overview = profile["overview"]
    lookup = columns_by_name(profile)
    findings: list[dict] = []
    recommendations: list[dict] = []

    kinds = profile["kinds"]
    parts = [f"{len(kinds['numeric'])} numeric"]
    if kinds["categorical"]:
        parts.append(f"{len(kinds['categorical'])} text")
    if kinds["boolean"]:
        parts.append(f"{len(kinds['boolean'])} boolean")
    if kinds["datetime"]:
        parts.append(f"{len(kinds['datetime'])} date")
    summary = (
        f"{overview['rows']:,} rows × {overview['columns']} columns "
        f"({', '.join(parts)}). {format_pct(100 - overview['missing_pct'])} of cells are populated."
    )

    # --- Data quality -----------------------------------------------------
    if overview["missing_cells"]:
        worst = max(profile["columns"], key=lambda item: item["missing_pct"])
        findings.append(
            _finding(
                "quality",
                f"{overview['missing_cells']:,} missing cells",
                f"{format_pct(overview['missing_pct'])} of the dataset is empty; "
                f"'{worst['name']}' is the most affected at {format_pct(worst['missing_pct'])}.",
                "warning",
            )
        )
        recommendations.append(
            {
                "action": "Handle missing values",
                "detail": f"Fill or drop gaps for {worst['name']} and related columns in Prepare.",
                "page": "prepare",
            }
        )

    if overview["duplicate_rows"]:
        findings.append(
            _finding(
                "quality",
                f"{overview['duplicate_rows']:,} duplicate rows",
                f"{format_pct(overview['duplicate_pct'])} of rows are exact duplicates, "
                "which can bias counts and models.",
                "warning",
            )
        )
        recommendations.append(
            {
                "action": "Remove duplicates",
                "detail": "One click in Prepare removes exact duplicate rows.",
                "page": "prepare",
            }
        )

    if not overview["missing_cells"] and not overview["duplicate_rows"]:
        findings.append(
            _finding(
                "quality",
                "Clean dataset",
                "No missing cells and no duplicate rows were found.",
                "positive",
            )
        )

    # --- Distribution shape ----------------------------------------------
    skewed = [
        column
        for column in profile["columns"]
        if column["kind"] == "numeric"
        and column["stats"]
        and abs(column["stats"].get("skew") or 0) >= 1
        and column["stats"].get("count", 0) >= 10
    ]
    if skewed:
        worst = max(skewed, key=lambda item: abs(item["stats"]["skew"]))
        stats = worst["stats"]
        direction = "higher" if stats["skew"] > 0 else "lower"
        findings.append(
            _finding(
                "distribution",
                f"{worst['name']} is skewed",
                f"Mean {format_number(stats['mean'])} vs median {format_number(stats['median'])} — "
                f"a long tail of {direction} values (skew {stats['skew']:.2f}).",
            )
        )

    outlier_columns = [
        column
        for column in profile["columns"]
        if column["kind"] == "numeric"
        and column["stats"]
        and column["stats"].get("count", 0) >= 20
        and (column["stats"].get("outliers") or 0) / max(column["stats"]["count"], 1) >= 0.02
    ]
    if outlier_columns:
        worst = max(
            outlier_columns,
            key=lambda item: item["stats"]["outliers"] / item["stats"]["count"],
        )
        stats = worst["stats"]
        share = 100 * stats["outliers"] / stats["count"]
        findings.append(
            _finding(
                "distribution",
                f"Outliers in {worst['name']}",
                f"{stats['outliers']:,} values ({format_pct(share)}) fall outside "
                f"1.5×IQR — between {format_number(stats.get('lower_bound'))} and "
                f"{format_number(stats.get('upper_bound'))}.",
            )
        )

    # --- Concentration ----------------------------------------------------
    concentrated = [
        column
        for column in profile["columns"]
        if column["kind"] in (KIND_CATEGORICAL, "boolean")
        and column["top_values"]
        and 2 <= column["unique"] <= 30
        and column["top_values"][0]["pct"] >= 55
    ]
    if concentrated:
        worst = max(concentrated, key=lambda item: item["top_values"][0]["pct"])
        top = worst["top_values"][0]
        findings.append(
            _finding(
                "pattern",
                f"{worst['name']} is dominated by one value",
                f"'{top['value']}' accounts for {format_pct(top['pct'])} of non-null rows. "
                "Comparisons across this column will be imbalanced.",
            )
        )

    # --- Relationships ----------------------------------------------------
    if dataframe is not None:
        pairs = _correlation_pairs(dataframe, profile)
        if pairs:
            best = pairs[0]
            magnitude = abs(best["r"])
            if magnitude >= 0.4:
                direction = "positive" if best["r"] > 0 else "negative"
                findings.append(
                    _finding(
                        "relationship",
                        f"{best['a']} and {best['b']} move together",
                        f"Correlation r = {best['r']:.2f} ({direction}). "
                        f"{'Strong' if magnitude >= 0.7 else 'Moderate'} relationship across "
                        f"{overview['rows']:,} rows.",
                    )
                )
            redundant = [pair for pair in pairs if abs(pair["r"]) >= 0.85]
            if redundant:
                pair = redundant[0]
                recommendations.append(
                    {
                        "action": "Avoid redundant features",
                        "detail": f"{pair['a']} and {pair['b']} correlate at r = {pair['r']:.2f}; "
                        "one of them is usually enough for modelling.",
                        "page": "model",
                    }
                )

    # --- Time -------------------------------------------------------------
    if kinds["datetime"]:
        column = lookup[kinds["datetime"][0]]
        stats = column["stats"]
        findings.append(
            _finding(
                "pattern",
                f"{column['name']} spans {stats.get('span_days', 0):,} days",
                f"From {stats.get('min')} to {stats.get('max')}; "
                "use Explore to see how volume changes over time.",
            )
        )

    if kinds["categorical"]:
        high_card = [name for name in kinds["categorical"] if lookup[name]["unique"] > 50]
        if high_card:
            recommendations.append(
                {
                    "action": "Reduce text cardinality",
                    "detail": f"{', '.join(high_card[:3])} have many distinct values. "
                    "Keep the top categories or drop them before modelling.",
                    "page": "model",
                }
            )

    model_review = _model_review(ml_result, recommendations) if ml_result else None

    if not recommendations:
        recommendations.append(
            {
                "action": "Ready for analysis",
                "detail": "The dataset is complete and consistent. Explore charts or train a model.",
                "page": "explore",
            }
        )

    return {
        "summary": summary,
        "findings": findings[:8],
        "recommendations": recommendations[:6],
        "model": model_review,
    }


def _model_review(ml_result: dict, recommendations: list[dict]) -> dict:
    task = ml_result.get("task")
    metrics = ml_result.get("metrics", {})
    baseline = ml_result.get("baseline") or {}
    model_label = (
        ml_result.get("model_label")
        or (ml_result.get("model") or {}).get("label")
        or ml_result.get("model_id")
        or "Model"
    )
    rows = ml_result.get("rows") or {}
    test_rows = int(ml_result.get("test_rows") or rows.get("test") or 0)
    bullets: list[str] = []
    headline = ""
    severity = "info"

    if task == "classification" and "accuracy" in metrics:
        accuracy = metrics["accuracy"]
        base_accuracy = baseline.get("accuracy")
        headline = f"{model_label} reached {accuracy:.1%} accuracy on {test_rows:,} held-out rows."
        if base_accuracy is not None:
            delta = (accuracy - base_accuracy) * 100
            headline += f" That is {delta:+.1f} pts versus the majority-class baseline ({base_accuracy:.1%})."
            severity = "positive" if accuracy >= base_accuracy + 0.02 else "warning"
        f1 = metrics.get("f1_macro")
        if f1 is not None:
            bullets.append(f"Macro F1 is {f1:.1%}, which balances performance across classes.")
        balanced = metrics.get("balanced_accuracy")
        if balanced is not None:
            bullets.append(f"Balanced accuracy is {balanced:.1%}.")
        if base_accuracy is not None and accuracy < base_accuracy:
            recommendations.append(
                {
                    "action": "Model underperforms a trivial baseline",
                    "detail": "Check class imbalance, leakage or try a different model/fewer features.",
                    "page": "model",
                }
            )
    elif task == "regression" and "r2" in metrics:
        r2 = metrics["r2"]
        if r2 < 0:
            headline = (
                f"{model_label} performs worse than predicting the average "
                f"{ml_result.get('target')} (R² = {r2:.3f})."
            )
        else:
            headline = (
                f"{model_label} explains {r2:.1%} of the variance in "
                f"{ml_result.get('target')} (R² = {r2:.3f})."
            )
        severity = "positive" if r2 >= 0.5 else "warning" if r2 < 0.2 else "info"
        if metrics.get("mae") is not None:
            bullets.append(f"Mean absolute error is {format_number(metrics['mae'])}.")
        if metrics.get("rmse") is not None:
            bullets.append(f"RMSE is {format_number(metrics['rmse'])}.")
        if r2 < 0.2:
            recommendations.append(
                {
                    "action": "Weak regression fit",
                    "detail": (
                        "The model barely beats the mean baseline. Add better "
                        "features or treat this as an exploratory result."
                    ),
                    "page": "model",
                }
            )

    importance = ml_result.get("importance") or []
    if importance:
        top = ", ".join(item["feature"] for item in importance[:3])
        bullets.append(f"Strongest drivers: {top}.")

    for warning in ml_result.get("warnings") or []:
        bullets.append(warning)

    if rows.get("sampled"):
        bullets.append(
            f"Trained on a sample of {rows.get('used'):,} of {rows.get('total'):,} rows."
        )

    return {"headline": headline, "bullets": bullets, "severity": severity}


def generate_summary(eda_results: dict, ml_results: dict | None = None) -> dict:
    """Backwards-compatible summary builder for the classic EDA bundle.

    New code should call :func:`build_insights`, which also uses the profile,
    correlations and model metrics.
    """
    if not isinstance(eda_results, dict):
        raise TypeError("EDA results must be a dictionary.")

    overview = eda_results.get("overview", {}) or {}
    summary = {
        "dataset_overview": [],
        "key_findings": [],
        "important_variables": [],
        "outliers": [],
        "ml_results": [],
        "recommendations": [],
    }

    if overview:
        summary["dataset_overview"].append(
            f"The dataset contains {overview.get('rows', 0):,} rows and "
            f"{overview.get('columns', 0)} columns."
        )
        numeric = overview.get("numerical_columns", [])
        categorical = overview.get("categorical_columns", [])
        if numeric:
            summary["dataset_overview"].append(
                f"It contains {len(numeric)} numerical columns."
            )
            summary["important_variables"].append("Numerical variables: " + ", ".join(numeric))
        if categorical:
            summary["dataset_overview"].append(
                f"It contains {len(categorical)} categorical columns."
            )
            summary["important_variables"].append(
                "Categorical variables: " + ", ".join(categorical)
            )

    missing = int(overview.get("missing_values", 0) or 0)
    duplicates = int(overview.get("duplicate_rows", 0) or 0)
    summary["key_findings"].append(
        f"The dataset contains {missing:,} missing values." if missing
        else "The dataset contains no missing values."
    )
    summary["key_findings"].append(
        f"The dataset contains {duplicates:,} duplicate rows." if duplicates
        else "The dataset contains no duplicate rows."
    )
    for insight in eda_results.get("insights") or []:
        summary["key_findings"].append(str(insight))

    outliers = eda_results.get("outliers") or {}
    for column, result in outliers.items():
        count = (result or {}).get("outlier_count", 0)
        if count:
            summary["outliers"].append(f"{column} contains {count} potential outlier(s).")
    if not summary["outliers"]:
        summary["outliers"].append("No potential numerical outliers were detected.")

    if missing:
        summary["recommendations"].append(
            "Review and handle missing values before further analysis."
        )
    if duplicates:
        summary["recommendations"].append("Remove duplicate rows before final analysis.")
    if not summary["recommendations"]:
        summary["recommendations"].append("The dataset appears suitable for further analysis.")

    if ml_results:
        if not isinstance(ml_results, dict):
            raise TypeError("ML results must be a dictionary.")

        problem_type = ml_results.get("problem_type") or ml_results.get("task")
        target = ml_results.get("target_column") or ml_results.get("target")
        metrics = ml_results.get("metrics", {})

        summary["ml_results"].append("A machine learning model was trained on the dataset.")
        if problem_type:
            summary["ml_results"].append(f"Problem type: {problem_type}.")
        if target:
            summary["ml_results"].append(f"Target column: {target}.")
        if metrics.get("accuracy") is not None:
            summary["ml_results"].append(f"Accuracy: {metrics['accuracy']:.2%}.")
        for key in ("mae", "rmse", "r2"):
            if metrics.get(key) is not None:
                summary["ml_results"].append(f"{key.upper()}: {metrics[key]:.4f}.")
    else:
        summary["ml_results"].append("No machine learning results are currently available.")

    return summary
