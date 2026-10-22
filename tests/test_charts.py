"""Chart engine: curated defaults, aggregation caps and request validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.charts import (
    ChartError,
    build_chart,
    chart_catalog,
    default_charts,
    histogram_spec,
    scatter_spec,
)
from src.config import Settings
from src.profile import build_profile


# ---------------------------------------------------------------------------
# Curated defaults
# ---------------------------------------------------------------------------


def test_default_chart_count_and_types(customers_frame):
    profile = build_profile(customers_frame)
    charts = default_charts(customers_frame, profile)

    assert 1 <= len(charts) <= 5
    types = [chart["type"] for chart in charts]
    assert types.count("histogram") == 2
    assert "missing" in types  # the fixture has gaps
    assert "heatmap" in types
    assert "bar" in types
    assert "line" not in types  # the category chart wins the last slot


def test_default_charts_never_use_identifiers(customers_frame):
    profile = build_profile(customers_frame)
    charts = default_charts(customers_frame, profile)
    ids = {chart["id"] for chart in charts}

    assert not any("customer_id" in chart_id for chart_id in ids)
    assert not any("referral_code" in chart_id for chart_id in ids)


def test_default_charts_are_json_serialisable(customers_frame):
    charts = default_charts(customers_frame, build_profile(customers_frame))
    json.dumps(charts)


def test_defaults_skip_missing_chart_when_complete(numeric_frame):
    charts = default_charts(numeric_frame, build_profile(numeric_frame))
    assert all(chart["type"] != "missing" for chart in charts)


def test_defaults_handle_categorical_only_data():
    frame = pd.DataFrame({"city": ["a", "b", "a", "c", "a", "b", "c", "a"] * 5})
    charts = default_charts(frame, build_profile(frame))

    assert [chart["type"] for chart in charts] == ["bar"]
    assert charts[0]["data"]["categories"]


def test_defaults_do_not_crash_on_tiny_data():
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "group": ["a", "b", "a", "b", "a", "b"]})
    charts = default_charts(frame, build_profile(frame))
    assert isinstance(charts, list)


def test_high_cardinality_bar_is_capped():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"code": [f"value-{index}" for index in rng.integers(0, 60, 600)]})
    profile = build_profile(frame)
    spec = build_chart(frame, profile, "bar", "code", top_n=10)

    assert len(spec["data"]["categories"]) <= 11  # 10 + "Other"
    assert spec["meta"]["truncated_categories"] is True


def test_scatter_is_sampled():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x": rng.normal(size=10_000), "y": rng.normal(size=10_000)})
    profile = build_profile(frame)
    spec = scatter_spec(frame, profile, "x", "y", settings=Settings(scatter_points=400))

    assert spec["meta"]["sampled"] is True
    assert len(spec["data"]["points"]) == 400


def test_scatter_reports_correlation():
    frame = pd.DataFrame({"x": range(50), "y": [value * 2 for value in range(50)]})
    spec = scatter_spec(frame, build_profile(frame), "x", "y")

    assert spec["data"]["correlation"] == pytest.approx(1.0)
    assert spec["data"]["strength"] in {"strong", "moderate", "weak", "none"}


# ---------------------------------------------------------------------------
# Spec shapes used by static/js/charts.js
# ---------------------------------------------------------------------------


def test_histogram_spec_contract(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "histogram", "income", bins=12)

    assert spec["type"] == "histogram"
    assert len(spec["data"]["bins"]) == 12
    first = spec["data"]["bins"][0]
    assert {"start", "end", "count"} <= set(first)
    assert spec["data"]["stats"]["median"] is not None
    assert spec["insight"]


def test_bar_spec_contract(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "bar", "city", top_n=6)
    category = spec["data"]["categories"][0]

    assert {"label", "count", "pct"} <= set(category)
    assert spec["x_label"] == "city"


def test_heatmap_spec_contract(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "heatmap")
    columns = spec["data"]["columns"]
    values = spec["data"]["values"]

    assert 2 <= len(columns) <= 10
    assert len(values) == len(columns)
    assert all(len(row) == len(columns) for row in values)


def test_line_spec_contract(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "line", "signup_date", "income")
    point = spec["data"]["points"][0]

    assert len(point) == 2
    assert isinstance(point[0], str)
    assert spec["data"]["metric"]


def test_box_spec_contract(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "box", "city", "income")
    group = spec["data"]["groups"][0]

    assert {"label", "q1", "median", "q3", "whisker_low", "whisker_high"} <= set(group)


def test_missing_spec_lists_only_gaps(customers_frame):
    profile = build_profile(customers_frame)
    spec = build_chart(customers_frame, profile, "missing")
    columns = {item["column"] for item in spec["data"]["columns"]}

    # "city" has whitespace, not gaps, in this fixture.
    assert columns == {"income"}
    assert all(item["pct"] > 0 for item in spec["data"]["columns"])


def test_catalog_contract(customers_frame):
    catalog = chart_catalog(build_profile(customers_frame))

    assert set(catalog) == {"types", "columns"}
    type_ids = {item["id"] for item in catalog["types"]}
    assert {"histogram", "bar", "line", "scatter", "box", "heatmap"} == type_ids
    for item in catalog["types"]:
        assert {"label", "description", "requires_y", "x_kinds", "y_kinds"} <= set(item)
    for column in catalog["columns"]:
        assert {"name", "kind", "is_identifier", "is_constant"} <= set(column)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_scatter_requires_numeric_columns(customers_frame):
    profile = build_profile(customers_frame)
    with pytest.raises(ChartError, match="numeric"):
        build_chart(customers_frame, profile, "scatter", "city", "income")


def test_line_requires_dates(customers_frame):
    profile = build_profile(customers_frame)
    with pytest.raises(ChartError, match="date"):
        build_chart(customers_frame, profile, "line", "age")


def test_unknown_chart_type(customers_frame):
    with pytest.raises(ChartError, match="Unknown chart type"):
        build_chart(customers_frame, build_profile(customers_frame), "pie", "age")


def test_missing_column(customers_frame):
    with pytest.raises(ChartError):
        histogram_spec(customers_frame, build_profile(customers_frame), "does_not_exist")


def test_heatmap_needs_two_numeric_columns():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    with pytest.raises(ChartError, match="at least two numeric"):
        build_chart(frame, build_profile(frame), "heatmap")


def test_constant_column_is_documented_not_charted():
    frame = pd.DataFrame({"const": ["x"] * 20, "value": range(20)})
    profile = build_profile(frame)
    catalog = chart_catalog(profile)
    column = next(item for item in catalog["columns"] if item["name"] == "const")
    assert column["is_constant"] is True
    charts = default_charts(frame, profile)
    assert all("const" not in chart["title"] for chart in charts)
