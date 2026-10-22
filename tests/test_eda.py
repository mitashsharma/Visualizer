"""EDA helpers: classic bundle plus JSON payloads used by the API."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    categorical_statistics,
    correlation_matrix,
    correlation_payload,
    dataset_overview,
    detect_outliers,
    generate_insights,
    numerical_statistics,
    outlier_payload,
    perform_eda,
)
from src.profile import build_profile


def create_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
            "Age": [20, 21, 22, 23, 100],
            "Score": [80, 85, 90, 88, 95],
            "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"],
        }
    )


def test_dataset_overview():
    result = dataset_overview(create_test_data())

    assert result["rows"] == 5
    assert result["columns"] == 4
    assert "Age" in result["numerical_columns"]
    assert "Score" in result["numerical_columns"]
    assert "City" in result["categorical_columns"]
    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0


def test_numerical_statistics():
    result = numerical_statistics(create_test_data())

    assert not result.empty
    assert "Age" in result.index
    assert "mean" in result.columns


def test_categorical_statistics():
    result = categorical_statistics(create_test_data())

    assert "Name" in result
    assert result["City"]["unique_values"] == 3


def test_correlation_matrix():
    result = correlation_matrix(create_test_data())

    assert not result.empty
    assert "Age" in result.columns
    assert "Score" in result.columns


def test_outlier_detection():
    result = detect_outliers(create_test_data())

    assert "Age" in result
    assert result["Age"]["outlier_count"] == 1


def test_generate_insights():
    messages = generate_insights(create_test_data())

    assert isinstance(messages, list)
    assert messages
    assert all(isinstance(message, str) for message in messages)


def test_complete_eda():
    result = perform_eda(create_test_data())

    for key in ("overview", "numerical_statistics", "categorical_statistics", "correlation_matrix", "outliers", "insights", "profile"):
        assert key in result


def test_perform_eda_accepts_precomputed_profile():
    frame = create_test_data()
    profile = build_profile(frame)
    result = perform_eda(frame, profile=profile)

    assert result["profile"] is profile
    assert result["overview"]["rows"] == len(frame)


def test_eda_bundle_is_json_serialisable(customers_frame):
    result = perform_eda(customers_frame)

    # DataFrames are the only non-JSON part; everything else must encode.
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            continue
        json.dumps(value, default=str)


def test_correlation_payload_shape(customers_frame):
    payload = correlation_payload(customers_frame, ["age", "income", "satisfaction_score"])

    assert payload["columns"] == ["age", "income", "satisfaction_score"]
    assert len(payload["values"]) == 3
    assert all(len(row) == 3 for row in payload["values"])
    assert all(value is None or -1 <= value <= 1 for row in payload["values"] for value in row)


def test_outlier_payload(customers_frame):
    profile = build_profile(customers_frame)
    payload = outlier_payload(profile)

    assert payload
    assert all({"column", "outlier_count", "share_pct"} <= set(item) for item in payload)
    assert all(item["outlier_count"] >= 0 for item in payload)


def test_validation_rejects_non_frames():
    with pytest.raises(TypeError):
        dataset_overview([1, 2, 3])


def test_correlation_needs_numeric_columns():
    frame = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    assert correlation_matrix(frame).empty


def test_outliers_ignore_constant_columns():
    frame = pd.DataFrame({"constant": [5.0] * 30, "value": list(np.linspace(0, 10, 30))})
    result = detect_outliers(frame)
    assert result["constant"]["outlier_count"] == 0
