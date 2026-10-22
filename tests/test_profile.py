"""Profiling tests, including regressions for identifier detection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.profile import (
    build_profile,
    categorical_columns,
    column_kind,
    columns_by_name,
    datetime_columns,
    numeric_columns,
)


def test_overview_counts(customers_frame):
    profile = build_profile(customers_frame)
    overview = profile["overview"]

    assert overview["rows"] == len(customers_frame)
    assert overview["columns"] == customers_frame.shape[1]
    assert overview["missing_cells"] == int(customers_frame.isna().sum().sum())
    assert overview["duplicate_rows"] == 1
    assert overview["cells"] == len(customers_frame) * customers_frame.shape[1]
    assert overview["memory_mb"] > 0


def test_column_kinds(customers_frame):
    profile = build_profile(customers_frame)
    lookup = columns_by_name(profile)

    assert lookup["age"]["kind"] == "numeric"
    assert lookup["city"]["kind"] == "categorical"
    assert lookup["signup_date"]["kind"] == "datetime"
    assert lookup["customer_id"]["is_identifier"] is True
    assert lookup["referral_code"]["is_identifier"] is True


def test_continuous_numbers_are_not_identifiers(customers_frame):
    """Regression: high-cardinality floats must remain usable measures."""
    profile = build_profile(customers_frame)
    lookup = columns_by_name(profile)

    assert lookup["income"]["is_identifier"] is False
    assert lookup["income"]["is_high_cardinality"] is False
    assert "income" in numeric_columns(profile)


def test_named_measures_are_not_identifiers():
    frame = pd.DataFrame(
        {
            "account_balance": np.linspace(10, 100, 60),
            "order_value": np.arange(60, dtype=float),
            "rating": np.arange(60) % 7,
        }
    )
    profile = build_profile(frame)
    lookup = columns_by_name(profile)

    assert all(not lookup[name]["is_identifier"] for name in frame.columns)


def test_identifier_detection_on_small_data():
    frame = pd.DataFrame({"customer": [f"A-{index}" for index in range(6)], "value": [1, 2, 3, 4, 5, 6]})
    profile = build_profile(frame)
    assert columns_by_name(profile)["customer"]["is_identifier"] is True


def test_constant_and_missing():
    frame = pd.DataFrame({"const": ["x"] * 10, "gap": [None] * 10, "ok": range(10)})
    profile = build_profile(frame)
    lookup = columns_by_name(profile)

    assert lookup["const"]["is_constant"] is True
    assert lookup["gap"]["is_constant"] is True
    assert lookup["gap"]["missing"] == 10
    assert lookup["gap"]["missing_pct"] == 100.0
    assert 0 <= lookup["gap"]["unique"] <= 1


def test_quality_issue_codes(customers_frame):
    profile = build_profile(customers_frame)
    codes = {issue["code"] for issue in profile["quality"]}

    assert "duplicate_rows" in codes
    assert "missing_values" in codes
    assert "whitespace" in codes
    assert "identifiers" in codes
    for issue in profile["quality"]:
        assert issue["severity"] in {"low", "medium", "high"}
        assert issue["title"]


def test_numeric_statistics(customers_frame):
    stats = columns_by_name(build_profile(customers_frame))["income"]["stats"]

    assert stats["count"] > 200
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["q1"] <= stats["q3"]
    assert stats["outliers"] >= 0


def test_datetime_statistics(customers_frame):
    stats = columns_by_name(build_profile(customers_frame))["signup_date"]["stats"]

    assert stats["span_days"] > 0
    assert stats["min"] < stats["max"]


def test_top_values_are_limited_and_sorted(customers_frame):
    city = columns_by_name(build_profile(customers_frame))["city"]
    counts = [item["count"] for item in city["top_values"]]

    assert 0 < len(city["top_values"]) <= 8
    assert counts == sorted(counts, reverse=True)
    assert all(0 <= item["pct"] <= 100 for item in city["top_values"])


def test_profile_is_json_serialisable(customers_frame):
    profile = build_profile(customers_frame)
    encoded = json.dumps(profile)
    assert '"columns"' in encoded


def test_helper_lists(customers_frame):
    profile = build_profile(customers_frame)

    assert "age" in numeric_columns(profile)
    assert "customer_id" not in numeric_columns(profile)
    assert "city" in categorical_columns(profile)
    assert "customer_id" not in categorical_columns(profile)
    assert datetime_columns(profile) == ["signup_date"]
    assert numeric_columns(profile, include_identifiers=True) == numeric_columns(profile)


def test_column_kind_helper():
    assert column_kind(pd.Series([1, 2, 3])) == "numeric"
    assert column_kind(pd.Series(["a", "b"])) == "categorical"
    assert column_kind(pd.Series([True, False])) == "boolean"
    assert column_kind(pd.Series(pd.to_datetime(["2024-01-01"]))) == "datetime"
