"""Model training: task detection, feature selection, metrics and caching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.ml_model import (
    TrainingError,
    analyse_target,
    available_models,
    cache_key,
    dataset_signature,
    default_model_id,
    detect_problem_type,
    get_feature_importance,
    prepare_data,
    public_result,
    reusable,
    select_features,
    train_model,
)
from src.profile import build_profile


# ---------------------------------------------------------------------------
# Detection and feature selection
# ---------------------------------------------------------------------------


def test_classification_detection(numeric_frame):
    assert detect_problem_type(numeric_frame, "target") == "classification"


def test_regression_detection(numeric_frame):
    assert detect_problem_type(numeric_frame, "price") == "regression"


def test_text_target_is_classification(numeric_frame):
    assert detect_problem_type(numeric_frame, "segment") == "classification"


def test_analyse_target(numeric_frame):
    analysis = analyse_target(numeric_frame, "target")

    assert analysis["eligible"] is True
    assert analysis["suggested_task"] == "classification"
    assert analysis["rows"] == len(numeric_frame)
    assert analysis["unique"] == 2
    assert analysis["reason"]
    assert 0 <= analysis["imbalance_pct"] <= 100


def test_analyse_target_rejects_dates_and_constants():
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=60),
            "constant": ["x"] * 60,
            "value": range(60),
        }
    )
    assert analyse_target(frame, "when")["eligible"] is False
    assert analyse_target(frame, "constant")["eligible"] is False


def test_analyse_target_rejects_tiny_data():
    frame = pd.DataFrame({"x": range(10), "y": range(10)})
    analysis = analyse_target(frame, "y")

    assert analysis["eligible"] is False
    assert "at least" in analysis["message"]


def test_select_features_excludes_ids_and_high_cardinality(numeric_frame):
    frame = numeric_frame.copy()
    frame["customer_id"] = [f"C{index:04d}" for index in range(len(frame))]
    frame["tag"] = [f"r{index % 60}" for index in range(len(frame))]

    features, excluded = select_features(build_profile(frame), "target", Settings())
    reasons = {item["column"]: item["reason"] for item in excluded}

    assert "customer_id" not in features
    assert "identifier" in reasons["customer_id"]
    assert "tag" not in features
    assert "high-cardinality" in reasons["tag"]
    assert "x1" in features


def test_prepare_data_excludes_the_target(numeric_frame):
    X, y, preprocessor = prepare_data(numeric_frame, "target")

    assert "target" not in X.columns
    assert len(X) == len(y) == len(numeric_frame)
    assert preprocessor is not None
    assert set(X.columns) & {"segment", "x1"}


def test_prepare_data_handles_dates(numeric_frame):
    frame = numeric_frame.copy()
    frame["joined"] = pd.date_range("2020-01-01", periods=len(frame))
    X, _y, _pre = prepare_data(frame, "target")
    assert pd.api.types.is_numeric_dtype(X["joined"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_classification_metrics(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="random_forest", max_rows=0)

    assert result["task"] == "classification"
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert 0 <= result["metrics"]["f1_macro"] <= 1
    assert "roc_auc" in result["metrics"]
    assert result["baseline"]["accuracy"] >= 0
    assert len(result["confusion_matrix"]) == 2
    assert {row["label"] for row in result["per_class"]} == {"0", "1"}
    assert result["train_rows"] + result["test_rows"] == len(numeric_frame)
    assert result["timings"]["total"] >= result["timings"]["fit"]


def test_regression_metrics(numeric_frame):
    result = train_model(numeric_frame, "price", model_id="random_forest", max_rows=0)

    assert result["task"] == "regression"
    assert np.isfinite(result["metrics"]["r2"])
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["rmse"] >= result["metrics"]["mae"]
    assert result["baseline"]["strategy"] == "predict the training mean"
    assert result["predictions"]
    assert all(len(pair) == 2 for pair in result["predictions"][:10])


def test_sampling_is_reported(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="random_forest", max_rows=60)

    assert result["rows"]["sampled"] is True
    assert result["rows"]["used"] == 60
    assert result["rows"]["total"] == len(numeric_frame)
    assert any("sample" in warning.lower() for warning in result["warnings"])


def test_sampling_is_stratified(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="random_forest", max_rows=80)
    overall_share = float(numeric_frame["target"].mean())
    positive = next(row for row in result["per_class"] if row["label"] == "1")
    test_share = positive["support"] / result["test_rows"]

    assert result["rows"]["sampled"] is True
    assert abs(test_share - overall_share) < 0.15


def test_feature_importance(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="random_forest", max_rows=0)
    importance = result["importance"]

    assert importance
    assert sum(item["importance"] for item in importance) == pytest.approx(1.0, abs=0.01)
    assert importance == sorted(importance, key=lambda item: item["importance"], reverse=True)

    frame = get_feature_importance(result)
    assert list(frame.columns) == ["feature", "importance"]
    assert len(frame) <= 15


def test_linear_model_importance(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="logistic_regression", max_rows=0)
    assert result["importance"]


def test_gradient_boosting_importance(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="hist_gradient_boosting", max_rows=0)
    assert result["metrics"]["accuracy"] > 0.5


def test_warnings_mention_excluded_columns(numeric_frame):
    frame = numeric_frame.copy()
    frame["customer_id"] = [f"C{index:04d}" for index in range(len(frame))]
    result = train_model(frame, "target", model_id="random_forest", max_rows=0)

    assert any("Excluded" in warning for warning in result["warnings"])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unknown_target():
    frame = pd.DataFrame({"x": range(50), "y": range(50)})
    with pytest.raises(ValueError, match="does not exist"):
        train_model(frame, "missing")


def test_too_few_rows():
    frame = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    with pytest.raises(TrainingError, match="at least"):
        train_model(frame, "y", max_rows=0)


def test_single_class():
    frame = pd.DataFrame({"x": range(60), "y": [1] * 60})
    with pytest.raises(TrainingError, match="two classes"):
        train_model(frame, "y", max_rows=0)


def test_regression_on_text_target(numeric_frame):
    with pytest.raises(TrainingError, match="numeric target"):
        train_model(numeric_frame, "segment", problem_type="regression", max_rows=0)


def test_unknown_model_id(numeric_frame):
    with pytest.raises(TrainingError, match="not available"):
        train_model(numeric_frame, "target", model_id="magic", max_rows=0)


def test_no_usable_features():
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 40,
            "constant": ["x"] * 80,
            "id": [f"C{index}" for index in range(80)],
        }
    )
    with pytest.raises(TrainingError, match="No usable feature"):
        train_model(frame, "target", max_rows=0)


# ---------------------------------------------------------------------------
# Registry, caching and serialisation
# ---------------------------------------------------------------------------


def test_available_models():
    classification = available_models("classification", rows=120)
    regression = available_models("regression", rows=120)

    assert {item["id"] for item in classification} == {
        "hist_gradient_boosting",
        "random_forest",
        "logistic_regression",
    }
    assert {item["id"] for item in regression} == {
        "hist_gradient_boosting",
        "random_forest",
        "ridge",
    }
    assert sum(item["recommended"] for item in classification) == 1
    assert default_model_id("classification", rows=120) == "random_forest"
    assert default_model_id("classification", rows=50_000) == "hist_gradient_boosting"


def test_cache_key_changes_with_configuration():
    signature = dataset_signature(pd.DataFrame({"a": [1, 2, 3]}))
    base = cache_key(signature, "y", "classification", "random_forest", 1000, 0.2, 42)

    assert base != cache_key(signature, "y", "classification", "logistic_regression", 1000, 0.2, 42)
    assert base != cache_key(signature, "y", "classification", "random_forest", 500, 0.2, 42)
    assert base != cache_key(signature, "y", "classification", "random_forest", 1000, 0.3, 42)
    assert base == cache_key(signature, "y", "classification", "random_forest", 1000, 0.2, 42)


def test_dataset_signature_changes_with_content():
    first = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    second = first.copy()
    second.loc[0, "a"] = 99

    assert dataset_signature(first) == dataset_signature(first.copy())
    assert dataset_signature(first) != dataset_signature(second)
    assert dataset_signature(first) != dataset_signature(pd.DataFrame({"a": [1, 2, 3]}))


def test_reusable_guards_against_incompatible_models(numeric_frame):
    result = train_model(numeric_frame, "target", model_id="random_forest", max_rows=0)
    signature = result["feature_signature"]

    assert reusable(result, signature, "random_forest", "classification") is True
    assert reusable(result, signature, "logistic_regression", "classification") is False
    assert reusable(result, signature, "random_forest", "regression") is False
    assert reusable(result, {**signature, "rows": 1}, "random_forest", "classification") is False
    assert reusable(None, signature, "random_forest", "classification") is False


def test_public_result_is_serialisable(numeric_frame):
    import json

    result = train_model(numeric_frame, "target", model_id="logistic_regression", max_rows=0)
    public = public_result(result)

    assert "pipeline" not in public and "model" not in public
    assert "classification_report" not in public["metrics"]
    json.dumps(public)
