"""Machine learning: feature selection, training and evaluation.

Optimisations over the original implementation
-----------------------------------------------
* Identifiers, constants and high-cardinality text are excluded before
  encoding (the original built a 48 GiB dense one-hot matrix from an ID).
* One-hot output is sparse; numeric scaling only happens for linear models.
* A small model registry replaces the single hard-coded 100-tree forest.
* Rows are capped with *honest* stratified sampling above a configurable limit.
* One split, one fit, one predict; timings are recorded per stage.
* Results are cacheable: ``cache_key`` captures data + configuration and
  ``reusable`` refuses results whose feature signature no longer matches.

The public API (``detect_problem_type``, ``prepare_data``, ``train_model``,
``get_feature_importance``) is backwards compatible.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SETTINGS, Settings
from src.formatting import safe_number
from src.profile import KIND_CATEGORICAL, KIND_DATETIME, KIND_NUMERIC, build_profile, columns_by_name

CLASSIFICATION = "classification"
REGRESSION = "regression"
MIN_ROWS = 30


class TrainingError(ValueError):
    """Raised when a model cannot be trained for a valid reason."""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _model_definitions(settings: Settings) -> dict:
    return {
        CLASSIFICATION: [
            {
                "id": "hist_gradient_boosting",
                "label": "Gradient boosting",
                "description": "Fast and strong on tabular data. Recommended above 20k rows.",
                "speed": "fast",
                "build": lambda: HistGradientBoostingClassifier(
                    max_iter=settings.hgb_max_iter,
                    learning_rate=0.08,
                    random_state=settings.ml_random_state,
                ),
            },
            {
                "id": "random_forest",
                "label": "Random forest",
                "description": "Robust baseline with built-in feature importance.",
                "speed": "medium",
                "build": lambda: RandomForestClassifier(
                    n_estimators=settings.rf_estimators,
                    n_jobs=-1,
                    random_state=settings.ml_random_state,
                ),
            },
            {
                "id": "logistic_regression",
                "label": "Logistic regression",
                "description": "Fast linear model; useful as an interpretable baseline.",
                "speed": "fast",
                "build": lambda: LogisticRegression(max_iter=1000),
            },
        ],
        REGRESSION: [
            {
                "id": "hist_gradient_boosting",
                "label": "Gradient boosting",
                "description": "Fast and strong on tabular data. Recommended above 20k rows.",
                "speed": "fast",
                "build": lambda: HistGradientBoostingRegressor(
                    max_iter=settings.hgb_max_iter,
                    learning_rate=0.08,
                    random_state=settings.ml_random_state,
                ),
            },
            {
                "id": "random_forest",
                "label": "Random forest",
                "description": "Robust baseline with built-in feature importance.",
                "speed": "medium",
                "build": lambda: RandomForestRegressor(
                    n_estimators=settings.rf_estimators,
                    n_jobs=-1,
                    random_state=settings.ml_random_state,
                ),
            },
            {
                "id": "ridge",
                "label": "Ridge regression",
                "description": "Fast linear model; useful as an interpretable baseline.",
                "speed": "fast",
                "build": lambda: Ridge(alpha=1.0),
            },
        ],
    }


def available_models(
    task: str,
    rows: int | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    settings = settings or SETTINGS
    if task not in (CLASSIFICATION, REGRESSION):
        raise ValueError("task must be 'classification' or 'regression'.")

    definitions = _model_definitions(settings)[task]
    default_id = (
        "hist_gradient_boosting"
        if rows is not None and rows >= 20_000
        else "random_forest"
    )

    return [
        {
            "id": item["id"],
            "label": item["label"],
            "description": item["description"],
            "speed": item["speed"],
            "recommended": item["id"] == default_id,
        }
        for item in definitions
    ]


def default_model_id(task: str, rows: int, settings: Settings | None = None) -> str:
    settings = settings or SETTINGS
    return "hist_gradient_boosting" if rows >= 20_000 else "random_forest"


# ---------------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------------


def validate_data(dataframe: pd.DataFrame, target_column: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")
    if dataframe[target_column].isna().all():
        raise ValueError(f"Target column '{target_column}' contains only missing values.")


def detect_problem_type(dataframe: pd.DataFrame, target_column: str) -> str:
    """Return ``'classification'`` or ``'regression'`` for a target column."""
    validate_data(dataframe, target_column)
    return _detect_task(dataframe[target_column])[0]


def _detect_task(target: pd.Series) -> tuple[str, str]:
    target = target.dropna()
    if target.empty:
        raise ValueError("Target column contains no valid values.")

    if pd.api.types.is_bool_dtype(target):
        return CLASSIFICATION, "Boolean targets are classification problems."
    if pd.api.types.is_datetime64_any_dtype(target):
        return REGRESSION, "Dates are modelled as a numeric trend."
    if not pd.api.types.is_numeric_dtype(target):
        return CLASSIFICATION, "Text targets are treated as classes."

    unique = int(target.nunique())
    total = len(target)
    if unique <= 2:
        return CLASSIFICATION, "A numeric target with two values is binary classification."

    unique_ratio = unique / total
    if unique <= 10 and unique_ratio < 0.5:
        return CLASSIFICATION, (
            f"The target has only {unique} distinct values "
            f"({unique_ratio:.0%} of rows) — treated as class labels."
        )
    return REGRESSION, (
        f"The target has {unique:,} distinct numeric values — treated as a continuous outcome."
    )


def analyse_target(
    dataframe: pd.DataFrame,
    target_column: str,
    profile: dict | None = None,
) -> dict:
    """Describe a candidate target for the model configuration UI."""
    validate_data(dataframe, target_column)
    profile = profile or build_profile(dataframe)
    info = columns_by_name(profile)[target_column]
    target = dataframe[target_column].dropna()

    task, reason = _detect_task(target)
    eligible = True
    message = None

    if info["is_constant"]:
        eligible = False
        message = "This column has a single value and cannot be predicted."
    elif info["is_identifier"]:
        eligible = False
        message = "This column looks like an identifier; there is no pattern to learn."
    elif info["kind"] == KIND_DATETIME:
        eligible = False
        message = "Date columns cannot be used as a target."
    elif len(target) < MIN_ROWS:
        eligible = False
        message = f"Only {len(target)} non-missing rows — at least {MIN_ROWS} are needed."

    top_values = info["top_values"][:5]
    imbalance = top_values[0]["pct"] if task == CLASSIFICATION and top_values else None

    warnings = []
    if imbalance is not None and imbalance >= 80:
        warnings.append(
            f"The largest class holds {imbalance:.0f}% of rows; accuracy will be "
            "misleading — check balanced accuracy and macro F1."
        )
    if info["is_identifier"]:
        warnings.append(
            "This column looks like an identifier; models usually cannot learn from it."
        )

    return {
        "column": target_column,
        "kind": info["kind"],
        "suggested_task": task,
        "reason": reason,
        "eligible": eligible,
        "message": message,
        "rows": int(len(target)),
        "missing": int(info["missing"]),
        "unique": int(info["unique"]),
        "imbalance_pct": imbalance,
        "top_values": top_values,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Feature selection / preprocessing
# ---------------------------------------------------------------------------


def select_features(
    profile: dict,
    target_column: str,
    settings: Settings | None = None,
) -> tuple[list[str], list[dict]]:
    settings = settings or SETTINGS
    features: list[str] = []
    excluded: list[dict] = []

    for column in profile["columns"]:
        name = column["name"]
        if name == target_column:
            continue
        if column["is_constant"]:
            excluded.append({"column": name, "reason": "constant value"})
        elif column["is_identifier"]:
            excluded.append({"column": name, "reason": "identifier"})
        elif column["kind"] == KIND_CATEGORICAL and column["unique"] > settings.ml_max_cardinality:
            excluded.append(
                {
                    "column": name,
                    "reason": f"high-cardinality text ({column['unique']:,} values)",
                }
            )
        elif column["missing"] >= profile["overview"]["rows"]:
            excluded.append({"column": name, "reason": "no usable values"})
        else:
            features.append(name)

    return features, excluded


def _coerce_datetimes(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Turn date columns into days since the epoch (NaT stays NaN)."""
    lookup = columns_by_name(profile)
    epoch = pd.Timestamp("1970-01-01")
    for name in frame.columns:
        info = lookup.get(name)
        if info and info["kind"] == KIND_DATETIME:
            frame[name] = (
                (frame[name] - epoch).dt.total_seconds() / 86400.0
            )
    return frame


def build_preprocessor(feature_frame: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_frame.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    transformers = []
    if numeric_columns:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps=steps), numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise TrainingError("No usable feature columns found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    profile: dict | None = None,
    settings: Settings | None = None,
    scale_numeric: bool = False,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Separate target, coerce dates and build the preprocessing pipeline."""
    settings = settings or SETTINGS
    validate_data(dataframe, target_column)
    profile = profile or build_profile(dataframe)

    features, _excluded = select_features(profile, target_column, settings)
    if not features:
        raise TrainingError(
            "No usable feature columns: every other column is an identifier, "
            "constant, or too high-cardinality."
        )

    data = dataframe.dropna(subset=[target_column])
    X = data[features].copy()
    y = data[target_column]

    if X.empty:
        raise TrainingError("No rows remain after removing missing targets.")

    X = _coerce_datetimes(X, profile)
    preprocessor = build_preprocessor(X, scale_numeric=scale_numeric)
    return X, y, preprocessor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _limit_rows(
    dataframe: pd.DataFrame,
    target_column: str,
    task: str,
    max_rows: int,
    settings: Settings,
) -> tuple[pd.DataFrame, dict]:
    total = len(dataframe)
    if max_rows <= 0 or total <= max_rows:
        return dataframe, {
            "total": int(total),
            "used": int(total),
            "sampled": False,
            "limit": int(max_rows),
        }

    if task == CLASSIFICATION:
        counts = dataframe[target_column].value_counts()
        try:
            if counts.min() >= 2:
                sample, _ = train_test_split(
                    dataframe,
                    train_size=max_rows,
                    random_state=settings.ml_random_state,
                    stratify=dataframe[target_column],
                )
            else:
                raise ValueError("sparse class")
        except (ValueError, TypeError):
            sample = dataframe.sample(n=max_rows, random_state=settings.ml_random_state)
    else:
        sample = dataframe.sample(n=max_rows, random_state=settings.ml_random_state)

    return sample, {
        "total": int(total),
        "used": int(len(sample)),
        "sampled": True,
        "limit": int(max_rows),
    }


def _classification_metrics(y_test, predictions, probabilities, classes):
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "classification_report": report,
    }

    if probabilities is not None and len(classes) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        except (ValueError, IndexError):
            pass

    labels = list(classes)
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
    per_class = [
        {
            "label": str(label),
            "precision": float(report.get(str(label), {}).get("precision", 0) or 0),
            "recall": float(report.get(str(label), {}).get("recall", 0) or 0),
            "f1": float(report.get(str(label), {}).get("f1-score", 0) or 0),
            "support": int(report.get(str(label), {}).get("support", 0) or 0),
        }
        for label in labels
    ]

    baseline_counts = pd.Series(y_test).value_counts(normalize=True)
    baseline = {
        "accuracy": float(baseline_counts.iloc[0]) if not baseline_counts.empty else None,
        "label": str(baseline_counts.index[0]) if not baseline_counts.empty else None,
        "strategy": "majority class",
    }

    return metrics, baseline, per_class, matrix


def _regression_metrics(y_test, predictions, y_train) -> tuple[dict, dict]:
    mse = float(mean_squared_error(y_test, predictions))
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
        "median_ae": float(median_absolute_error(y_test, predictions)),
    }

    actual = np.asarray(y_test, dtype=float)
    nonzero = actual != 0
    if nonzero.any():
        metrics["mape"] = float(
            np.mean(np.abs((actual[nonzero] - np.asarray(predictions)[nonzero]) / actual[nonzero]))
            * 100
        )

    baseline_predictions = np.full(len(y_test), float(np.mean(y_train)))
    baseline = {
        "mae": float(mean_absolute_error(y_test, baseline_predictions)),
        "r2": 0.0,
        "strategy": "predict the training mean",
    }

    return metrics, baseline


def _expanded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover - defensive
        return []
    return [str(name) for name in names]


def _group_importance(
    names: list[str], values: np.ndarray, raw_columns: list[str]
) -> list[dict]:
    """Collapse one-hot columns back to their source column."""
    grouped: dict[str, float] = {}
    raw_columns = [str(column) for column in raw_columns]

    for name, value in zip(names, values):
        body = name.split("__", 1)[1] if "__" in name else name
        if body in raw_columns:
            raw = body
        else:
            matches = [column for column in raw_columns if body.startswith(column + "_")]
            raw = max(matches, key=len) if matches else body
        grouped[raw] = grouped.get(raw, 0.0) + float(value)

    return [
        {"feature": feature, "importance": round(score, 5)}
        for feature, score in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]


def _importance_from_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    settings: Settings,
) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = _expanded_feature_names(preprocessor)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        sample_size = min(1200, len(X_test))
        if sample_size < 20:
            return []
        try:
            transformed = preprocessor.transform(X_test.iloc[:sample_size])
            scoring = "balanced_accuracy" if task == CLASSIFICATION else "r2"
            result = permutation_importance(
                model,
                transformed,
                y_test.iloc[:sample_size],
                n_repeats=2,
                random_state=settings.ml_random_state,
                scoring=scoring,
                n_jobs=1,
            )
            values = np.clip(np.asarray(result.importances_mean, dtype=float), 0, None)
        except Exception:  # pragma: no cover - importance is best-effort
            return []

    if len(values) != len(names) or not names:
        return []

    total = float(values.sum()) or 1.0
    return _group_importance(names, values / total, list(X_test.columns))


def _signature(target: str, features: list[str], rows: int, task: str) -> dict:
    return {
        "target": target,
        "task": task,
        "features": list(features),
        "rows": int(rows),
    }


def train_model(
    dataframe: pd.DataFrame,
    target_column: str,
    problem_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    model_id: str | None = None,
    max_rows: int | None = None,
    settings: Settings | None = None,
    profile: dict | None = None,
    progress=None,
) -> dict:
    """Train and evaluate one model.

    ``progress`` is called as ``progress(percent, stage, message)`` and is used
    by the background job runner to report real stage transitions.
    """
    settings = settings or SETTINGS
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def report(percent: int, stage: str, message: str | None = None) -> None:
        if progress:
            progress(percent, stage, message)

    report(4, "Validating target")
    validate_data(dataframe, target_column)
    profile = profile or build_profile(dataframe)

    detected_task, _reason = _detect_task(dataframe[target_column].dropna())
    task = problem_type or detected_task
    if task not in (CLASSIFICATION, REGRESSION):
        raise TrainingError("problem_type must be 'classification' or 'regression'.")
    if task == CLASSIFICATION and dataframe[target_column].dropna().nunique() < 2:
        raise TrainingError("A classification target needs at least two classes.")
    if task == REGRESSION and not pd.api.types.is_numeric_dtype(
        dataframe[target_column].dropna()
    ):
        raise TrainingError(
            "Regression needs a numeric target; this column contains text."
        )
    if not 0.05 <= float(test_size) <= 0.5:
        raise TrainingError("Test size must be between 5% and 50%.")

    features, excluded = select_features(profile, target_column, settings)
    if not features:
        raise TrainingError(
            "No usable feature columns: every other column is an identifier, "
            "constant, or too high-cardinality."
        )

    data = dataframe.dropna(subset=[target_column])
    dropped_target_rows = int(len(dataframe) - len(data))
    if len(data) < MIN_ROWS:
        raise TrainingError(
            f"Only {len(data)} rows with a known target — at least {MIN_ROWS} are needed."
        )
    if task == CLASSIFICATION and data[target_column].nunique() < 2:
        raise TrainingError("The target has a single class; there is nothing to learn.")

    limit = settings.ml_max_train_rows if max_rows is None else int(max_rows)
    report(14, "Sampling rows")
    sample_started = time.perf_counter()
    sample, rows_info = _limit_rows(data, target_column, task, limit, settings)
    timings["sample"] = round(time.perf_counter() - sample_started, 3)

    if task == CLASSIFICATION and sample[target_column].nunique() < 2:
        raise TrainingError("The sampled data has a single class; increase the row limit.")

    model_id = model_id or default_model_id(task, len(sample), settings)
    definitions = {item["id"]: item for item in _model_definitions(settings)[task]}
    if model_id not in definitions:
        raise TrainingError(
            f"Model '{model_id}' is not available for {task}. "
            f"Choose one of: {', '.join(definitions)}."
        )
    definition = definitions[model_id]
    scale_numeric = model_id in {"logistic_regression", "ridge"}

    report(24, "Preparing features")
    prep_started = time.perf_counter()
    X, y, preprocessor = prepare_data(
        sample,
        target_column,
        profile=profile,
        settings=settings,
        scale_numeric=scale_numeric,
    )
    timings["prepare"] = round(time.perf_counter() - prep_started, 3)

    if len(X) < MIN_ROWS:
        raise TrainingError(
            f"Only {len(X)} usable feature rows remain — at least {MIN_ROWS} are needed."
        )

    stratify = None
    warnings: list[str] = []
    if task == CLASSIFICATION:
        counts = y.value_counts()
        if counts.min() >= 2 and int(len(y) * test_size) >= len(counts):
            stratify = y
        else:
            warnings.append(
                "Some classes are too small to stratify; using a random split instead."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if rows_info["sampled"]:
        warnings.append(
            f"Trained on a sample of {rows_info['used']:,} of "
            f"{rows_info['total']:,} rows (limit {rows_info['limit']:,})."
        )
    if excluded:
        summary = ", ".join(f"{item['column']} ({item['reason']})" for item in excluded[:5])
        warnings.append(f"Excluded {len(excluded)} column(s) from features: {summary}.")
    if task == CLASSIFICATION and y.value_counts(normalize=True).iloc[0] >= 0.8:
        warnings.append(
            "The target is imbalanced (largest class "
            f"{y.value_counts(normalize=True).iloc[0]:.0%}); prefer balanced accuracy and macro F1."
        )
    if dropped_target_rows:
        warnings.append(
            f"{dropped_target_rows:,} row(s) had a missing target and were not used."
        )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", definition["build"]())]
    )

    report(38, f"Training {definition['label']}", "Fitting the model on the training split")
    fit_started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    timings["fit"] = round(time.perf_counter() - fit_started, 3)

    report(76, "Evaluating", "Scoring the held-out test split")
    predict_started = time.perf_counter()
    predictions = pipeline.predict(X_test)
    probabilities = None
    if task == CLASSIFICATION and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_test)
        except Exception:  # pragma: no cover
            probabilities = None
    timings["predict"] = round(time.perf_counter() - predict_started, 3)

    classes = list(pipeline.named_steps["model"].classes_) if task == CLASSIFICATION else []
    report(86, "Computing metrics")
    metrics_started = time.perf_counter()
    confusion = None
    per_class: list = []

    if task == CLASSIFICATION:
        metrics, baseline, per_class, confusion = _classification_metrics(
            y_test, predictions, probabilities, classes
        )
        metrics["classes"] = [str(label) for label in classes]
        predictions_preview = None
    else:
        metrics, baseline = _regression_metrics(y_test, predictions, y_train)
        rng = np.random.default_rng(settings.ml_random_state)
        sample_size = min(400, len(y_test))
        indices = rng.choice(len(y_test), size=sample_size, replace=False)
        y_test_array = np.asarray(y_test, dtype=float)
        predictions_preview = [
            [safe_number(y_test_array[index]), safe_number(predictions[index])]
            for index in indices
        ]
    timings["metrics"] = round(time.perf_counter() - metrics_started, 3)

    report(93, "Ranking features")
    importance_started = time.perf_counter()
    importance = _importance_from_model(pipeline, X_test, y_test, task, settings)
    timings["importance"] = round(time.perf_counter() - importance_started, 3)
    timings["total"] = round(time.perf_counter() - started, 3)

    if task == CLASSIFICATION and len(y_test) < 50:
        warnings.append(
            f"Only {len(y_test)} rows in the test split; metrics are indicative only."
        )

    report(100, "Complete")

    return {
        "model": pipeline,
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "problem_type": task,
        "task": task,
        "target_column": target_column,
        "target": target_column,
        "model_id": model_id,
        "model_label": definition["label"],
        "metrics": metrics,
        "baseline": baseline,
        "classes": [str(label) for label in classes],
        "confusion_matrix": confusion,
        "per_class": per_class,
        "importance": importance,
        "predictions": predictions_preview,
        "timings": timings,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rows": rows_info,
        "warnings": warnings,
        "excluded_features": excluded,
        "feature_names": features,
        "feature_signature": _signature(target_column, features, len(data), task),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampled": rows_info["sampled"],
    }


# ---------------------------------------------------------------------------
# Feature importance API
# ---------------------------------------------------------------------------


def get_feature_importance(trained_result: dict, top_n: int = 15) -> pd.DataFrame:
    """Return a ``feature``/``importance`` table (one-hot columns grouped)."""
    if not isinstance(trained_result, dict):
        raise TypeError("trained_result must be a dictionary.")

    importance = trained_result.get("importance")
    if importance is None:
        pipeline = trained_result.get("pipeline") or trained_result.get("model")
        if pipeline is None:
            raise ValueError("Trained model not found.")
        model = pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            raise ValueError("The selected model does not provide feature importance.")
        names = _expanded_feature_names(pipeline.named_steps["preprocessor"])
        values = np.asarray(model.feature_importances_, dtype=float)
        total = float(values.sum()) or 1.0
        raw_columns = trained_result.get("feature_names") or []
        importance = _group_importance(names, values / total, raw_columns)

    frame = pd.DataFrame(importance, columns=["feature", "importance"])
    if frame.empty:
        return frame
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def dataset_signature(dataframe: pd.DataFrame) -> str:
    """Cheap content fingerprint used to prevent stale model reuse."""
    digest = hashlib.sha1()
    digest.update(str(dataframe.shape).encode())
    digest.update("|".join(str(name) for name in dataframe.columns).encode())
    if len(dataframe):
        sample = dataframe.head(50)
        try:
            hashes = pd.util.hash_pandas_object(sample, index=True).to_numpy()
            digest.update(hashes.tobytes())
        except TypeError:
            digest.update(sample.astype(str).to_csv(index=False).encode())
    return digest.hexdigest()[:16]


def cache_key(
    data_signature: str,
    target: str,
    task: str,
    model_id: str,
    max_rows: int,
    test_size: float,
    random_state: int,
) -> str:
    payload = json.dumps(
        {
            "data": data_signature,
            "target": target,
            "task": task,
            "model": model_id,
            "max_rows": max_rows,
            "test_size": round(float(test_size), 4),
            "seed": random_state,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def reusable(cached: dict | None, signature: dict, model_id: str, task: str) -> bool:
    """A cached result may only be reused for the exact same problem."""
    if not cached:
        return False
    if cached.get("feature_signature") != signature:
        return False
    return cached.get("model_id") == model_id and cached.get("task") == task


def public_result(result: dict) -> dict:
    """Strip fitted objects so a result can be JSON-serialised."""
    hidden = {"model", "pipeline", "preprocessor"}
    public = {key: value for key, value in result.items() if key not in hidden}
    public["metrics"] = {
        key: value for key, value in result["metrics"].items()
        if key != "classification_report"
    }
    return public
