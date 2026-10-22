"""Report rendering: self-contained HTML, PDF bytes and safe escaping."""

from __future__ import annotations

import json

import pytest

from src.charts import default_charts
from src.ml_model import public_result, train_model
from src.profile import build_profile
from src.report_generator import build_html, build_pdf, render_report, write_html
from src.summarizer import build_insights


@pytest.fixture()
def context(customers_frame):
    cleaned = customers_frame.drop_duplicates().copy()
    cleaned["income"] = cleaned["income"].fillna(cleaned["income"].median())
    cleaned["city"] = cleaned["city"].str.strip()
    profile = build_profile(cleaned, source="customers.csv")
    summary = {
        "id": "abc",
        "name": "Customers",
        "source": "customers.csv",
        "rows": profile["overview"]["rows"],
        "columns": profile["overview"]["columns"],
        "revision": 2,
        "created_at": "2024-01-01T00:00:00+00:00",
        "cleaning": None,
    }
    return {
        "dataset": summary,
        "profile": profile,
        "insights": build_insights(profile, cleaned),
        "ml": None,
        "cleaning": {
            "actions": ["Removed 1 duplicate row(s)"],
            "rows_before": len(customers_frame),
            "rows_after": len(cleaned),
            "missing_before": 10,
            "missing_after": 0,
        },
        "charts": default_charts(cleaned, profile),
        "generated_at": "2024-05-01T10:00:00+00:00",
    }


def test_html_report_has_the_expected_sections(context):
    html = build_html(context, include_charts=True)

    for heading in ["Executive summary", "Data quality", "Key charts", "Columns", "Cleaning applied", "Recommended next steps"]:
        assert heading in html
    assert "<img" in html
    assert "Customers" in html


def test_html_report_can_skip_charts(context):
    html = build_html(context, include_charts=False)
    assert "<img" not in html
    assert "Data quality" in html


def test_html_report_escapes_user_content(context):
    context["dataset"]["name"] = "<script>alert('x')</script>"
    context["profile"]["columns"][0]["name"] = "<img onerror=1>"
    html = build_html(context, include_charts=False)

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<img onerror" not in html


def test_html_report_includes_model_section(customers_frame, context):
    cleaned = customers_frame.drop_duplicates().copy()
    result = train_model(cleaned, "churned", model_id="random_forest", max_rows=0)
    context["ml"] = public_result(result)
    context["insights"] = build_insights(context["profile"], cleaned, context["ml"])

    html = build_html(context, include_charts=False)
    assert "Machine learning" in html
    assert "Random forest" in html
    assert "Accuracy" in html
    assert "Feature importance" in html
    assert str(result["target"]) in html


def test_pdf_report_starts_with_magic_bytes(context):
    pdf = build_pdf(context, include_charts=False)
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1_000


def test_pdf_report_with_charts(context):
    pdf = build_pdf(context, include_charts=True)
    assert pdf.startswith(b"%PDF-")


def test_render_report_returns_download_metadata(context):
    data, mimetype, filename = render_report(context, "html", include_charts=False)

    assert isinstance(data, bytes)
    assert mimetype.startswith("text/html")
    assert filename.endswith(".html")
    assert "customers" in filename


def test_render_report_pdf(context):
    data, mimetype, filename = render_report(context, "pdf", include_charts=False)

    assert data.startswith(b"%PDF-")
    assert mimetype == "application/pdf"
    assert filename.endswith(".pdf")


def test_render_report_rejects_unknown_format(context):
    with pytest.raises(ValueError):
        render_report(context, "docx")


def test_write_html(context, tmp_path):
    path = write_html(context, tmp_path / "report.html", include_charts=False)
    assert (tmp_path / "report.html").exists()
    assert "Data quality" in (tmp_path / "report.html").read_text()
    assert str(path).endswith("report.html")


def test_report_context_is_json_serialisable(context):
    serialisable = {key: value for key, value in context.items() if key != "profile"}
    serialisable["profile_columns"] = len(context["profile"]["columns"])
    json.dumps(serialisable)
