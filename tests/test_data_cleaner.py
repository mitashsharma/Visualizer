"""Cleaning: legacy helpers, plan-based cleaning and summary accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaner import (
    apply_cleaning_plan,
    clean_data,
    clean_text_columns,
    get_cleaning_summary,
    handle_missing_values,
    normalise_plan,
    recommended_plan,
    remove_duplicates,
)
from src.profile import build_profile


def create_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [" Alice ", "Bob", "Bob", "Charlie", None],
            "Age": [20, 21, 21, None, 25],
            "Score": [85.0, 90.0, 90.0, 88.0, None],
            "City": [" Delhi ", "Mumbai", "Mumbai", " Pune ", None],
        }
    )


# ---------------------------------------------------------------------------
# Legacy API (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def test_remove_duplicates():
    cleaned = remove_duplicates(create_test_data())
    assert len(cleaned) == 4


def test_handle_missing_values():
    cleaned = handle_missing_values(create_test_data())
    assert int(cleaned.isna().sum().sum()) == 0


def test_handle_missing_values_strategy():
    data = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "x"]})
    auto = handle_missing_values(data, strategy="auto")
    assert auto.loc[1, "a"] == 2.0
    assert auto.loc[1, "b"] == "x"
    dropped = handle_missing_values(data, strategy="drop")
    assert len(dropped) == 2
    untouched = handle_missing_values(data, strategy="none")
    assert int(untouched.isna().sum().sum()) == 2


def test_clean_text_columns():
    cleaned = clean_text_columns(create_test_data())
    assert cleaned.loc[0, "Name"] == "Alice"
    assert cleaned.loc[0, "City"] == "Delhi"


def test_clean_text_preserves_non_strings():
    data = pd.DataFrame({"mixed": ["  a  ", 5, None, 2.5]})
    cleaned = clean_text_columns(data)
    assert cleaned.loc[0, "mixed"] == "a"
    assert cleaned.loc[1, "mixed"] == 5
    assert pd.isna(cleaned.loc[2, "mixed"])
    assert cleaned.loc[3, "mixed"] == 2.5


def test_clean_text_does_not_mutate_input():
    data = create_test_data()
    clean_text_columns(data)
    assert data.loc[0, "Name"] == " Alice "


def test_complete_cleaning():
    cleaned = clean_data(create_test_data())
    assert len(cleaned) == 4
    assert int(cleaned.isna().sum().sum()) == 0
    assert cleaned.loc[0, "Name"] == "Alice"


def test_cleaning_summary():
    original = create_test_data()
    summary = get_cleaning_summary(original, clean_data(original))

    assert summary["original_rows"] == 5
    assert summary["cleaned_rows"] == 4
    assert summary["duplicates_removed"] == 1
    assert summary["missing_values_after"] == 0


# ---------------------------------------------------------------------------
# Plan-based cleaning (used by the Prepare page)
# ---------------------------------------------------------------------------


def test_recommended_plan_uses_profile(customers_frame):
    profile = build_profile(customers_frame)
    plan = recommended_plan(customers_frame, profile)

    assert plan["remove_duplicates"] is True
    assert plan["trim_whitespace"] is True
    assert plan["missing"]["income"] == "median"
    assert "city" not in plan["missing"]  # whitespace only, no gaps
    # identifiers are never dropped automatically
    assert "customer_id" not in plan["drop_columns"]


def test_apply_plan_fills_and_trims(customers_frame):
    profile = build_profile(customers_frame)
    plan = recommended_plan(customers_frame, profile)

    cleaned, summary = apply_cleaning_plan(customers_frame, plan)

    assert summary["rows_before"] == len(customers_frame)
    assert summary["rows_after"] == len(customers_frame) - 1  # one duplicate
    assert summary["duplicates_removed"] == 1
    assert summary["missing_after"] == 0
    assert cleaned["income"].isna().sum() == 0
    assert all(value == value.strip() for value in cleaned["city"])
    assert summary["duration_ms"] >= 0


def test_apply_plan_drops_columns_and_rows():
    data = pd.DataFrame(
        {
            "keep": [1, 2, 3, 4],
            "drop": ["x", "x", "x", "x"],
            "gap": [1.0, np.nan, np.nan, 4.0],
        }
    )
    cleaned, summary = apply_cleaning_plan(
        data,
        {"drop_columns": ["drop"], "missing": {"gap": "drop_rows"}},
    )

    assert "drop" not in cleaned.columns
    assert len(cleaned) == 2
    assert summary["columns_after"] == 2
    assert summary["dropped_rows_for_missing"] == 2
    assert any("Dropped 1 column" in action for action in summary["actions"])


def test_apply_plan_strategies():
    data = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", "a", None, "b"],
        }
    )
    cleaned, _summary = apply_cleaning_plan(data, {"missing": {"num": "mean", "cat": "mode"}})
    assert cleaned.loc[2, "num"] == pytest.approx(7 / 3)
    assert cleaned.loc[2, "cat"] == "a"

    cleaned, _summary = apply_cleaning_plan(data, {"missing": {"num": "zero"}})
    assert cleaned.loc[2, "num"] == 0


def test_apply_plan_does_not_mutate_input(customers_frame):
    before = customers_frame.copy(deep=True)
    apply_cleaning_plan(customers_frame, {"remove_duplicates": True, "trim_whitespace": True})
    pd.testing.assert_frame_equal(customers_frame, before)


def test_normalise_plan_rejects_unknown_columns():
    with pytest.raises(ValueError, match="Unknown column"):
        normalise_plan({"drop_columns": ["nope"]}, ["a", "b"])


def test_normalise_plan_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        normalise_plan({"missing": {"a": "magic"}}, ["a"])


def test_median_on_text_column_is_rejected():
    data = pd.DataFrame({"text": ["a", None, "b"]})
    with pytest.raises(ValueError, match="numeric"):
        apply_cleaning_plan(data, {"missing": {"text": "median"}})


def test_empty_plan_is_a_no_op(customers_frame):
    cleaned, summary = apply_cleaning_plan(customers_frame, {})
    assert summary["rows_before"] == summary["rows_after"]
    assert summary["duplicates_removed"] == 0
    assert summary["missing_after"] == summary["missing_before"]
    assert len(cleaned) == len(customers_frame)


def test_plan_summary_is_json_safe(customers_frame):
    import json

    _cleaned, summary = apply_cleaning_plan(customers_frame, {"missing": {"income": "median"}})
    json.dumps(summary)  # must not raise
