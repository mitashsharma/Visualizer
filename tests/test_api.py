"""HTTP contract tests for the Flask API and the static pages."""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest

from tests.conftest import upload_frame

PAGES = ["/", "/overview", "/prepare", "/explore", "/model", "/insights"]


def wait_for_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.get_json()["job"]
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def submit_training(client, dataset_id: str, **overrides):
    body = {
        "target": "churned",
        "model": "random_forest",
        "task": "classification",
        "max_rows": 0,
        "test_size": 0.2,
    }
    body.update(overrides)
    return client.post(f"/api/datasets/{dataset_id}/train", json=body)


# ---------------------------------------------------------------------------
# Pages and session
# ---------------------------------------------------------------------------


def test_pages_are_served(client):
    for path in PAGES:
        response = client.get(path)
        assert response.status_code == 200, path
        html = response.get_data(as_text=True)
        assert "<main" in html
        assert "data-page=" in html
        assert "/static/js/" in html


def test_favicon(client):
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert "svg" in response.get_data(as_text=True)


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["uptime_s"] >= 0


def test_session_without_dataset(client):
    body = client.get("/api/session").get_json()
    assert body["dataset"] is None
    assert body["limits"]["max_rows"] > 0


def test_session_after_upload(client, dataset):
    body = client.get("/api/session").get_json()
    assert body["dataset"]["id"] == dataset["id"]
    assert body["dataset"]["rows"] == 241


# ---------------------------------------------------------------------------
# Upload and overview
# ---------------------------------------------------------------------------


def test_upload_contract(client, customers_frame):
    response = upload_frame(client, customers_frame)
    assert response.status_code == 201
    body = response.get_json()

    assert {"dataset", "warnings", "profile"} <= set(body)
    summary = body["dataset"]
    for key in ("id", "name", "rows", "columns", "revision", "missing_pct", "is_cleaned"):
        assert key in summary
    assert summary["revision"] == 1
    # The loader reports inferred types as notes (for example a date column).
    assert isinstance(body["warnings"], list)


def test_upload_rejects_unknown_type(client):
    response = client.post(
        "/api/datasets",
        data={"file": (io.BytesIO(b"not a spreadsheet"), "notes.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["error"]


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/datasets",
        data={"file": (io.BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"].lower()


def test_upload_without_file(client):
    response = client.post("/api/datasets")
    assert response.status_code == 400


def test_overview_contract(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/overview").get_json()

    assert {"dataset", "overview", "columns", "kinds", "quality", "insights", "suggestions", "preview"} <= set(body)
    overview = body["overview"]
    for key in ("rows", "columns", "missing_cells", "missing_pct", "duplicate_rows", "memory_mb"):
        assert key in overview
    assert body["kinds"]["identifier"] == ["customer_id", "referral_code"]
    assert body["preview"]["rows"]
    assert body["suggestions"]["missing"]
    assert body["insights"]["summary"]
    assert body["insights"]["findings"]


def test_preview_pagination(client, dataset):
    url = f"/api/datasets/{dataset['id']}/preview"

    first = client.get(f"{url}?limit=10").get_json()
    second = client.get(f"{url}?limit=10&offset=5").get_json()

    assert len(first["rows"]) == 10
    assert first["offset"] == 0
    assert first["has_more"] is True
    assert second["offset"] == 5
    assert first["rows"][5] == second["rows"][0]
    assert first["total"] == 241


def test_preview_caps_limit(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/preview?limit=10000").get_json()
    assert len(body["rows"]) <= 200


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def test_charts_contract(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/charts").get_json()

    assert len(body["charts"]) <= 5
    assert body["charts"]
    for chart in body["charts"]:
        assert {"id", "type", "title", "insight", "data", "meta"} <= set(chart)
    assert {"types", "columns"} == set(body["catalog"])
    assert body["catalog"]["columns"]


def test_chart_builder(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/charts",
        json={"type": "histogram", "x": "income", "bins": 12},
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["chart"]["type"] == "histogram"
    assert len(body["chart"]["data"]["bins"]) == 12


def test_chart_builder_validation(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/charts",
        json={"type": "scatter", "x": "city", "y": "income"},
    )
    assert response.status_code == 400
    assert "numeric" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Cleaning and revisions
# ---------------------------------------------------------------------------


def test_cleaning_preview_does_not_create_a_revision(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/cleaning/preview",
        json={"plan": {"remove_duplicates": True, "trim_whitespace": True, "missing": {"income": "median"}}},
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["summary"]["rows_after"] == 240
    assert body["preview"]["rows"]

    session = client.get("/api/session").get_json()
    assert session["dataset"]["revision"] == 1


def test_clean_creates_a_revision(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={
            "plan": {
                "remove_duplicates": True,
                "trim_whitespace": True,
                "drop_columns": [],
                "missing": {"income": "median", "city": "mode"},
            }
        },
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["dataset"]["revision"] == 2
    assert body["dataset"]["id"] != dataset["id"]
    assert body["dataset"]["parent_id"] == dataset["id"]
    assert body["dataset"]["rows"] == 240
    assert body["summary"]["missing_after"] == 0
    assert body["profile"]["missing_cells"] == 0
    assert body["dataset"]["is_cleaned"] is True


def test_restore_returns_to_the_root(client, dataset):
    cleaned = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={"plan": {"remove_duplicates": True}},
    ).get_json()["dataset"]

    response = client.post(f"/api/datasets/{cleaned['id']}/restore")
    restored = response.get_json()["dataset"]

    assert response.status_code == 200
    assert restored["id"] == dataset["id"]
    assert restored["revision"] == 1


def test_clean_switches_the_active_revision(client, dataset):
    cleaned = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={"plan": {"remove_duplicates": True}},
    ).get_json()["dataset"]

    active = client.get("/api/session").get_json()["dataset"]
    assert active["id"] == cleaned["id"]
    assert active["revision"] == 2


def test_reading_an_older_revision_does_not_steal_the_session(client, dataset):
    cleaned = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={"plan": {"remove_duplicates": True}},
    ).get_json()["dataset"]

    # Inspect the root revision, then ask which dataset is active again.
    assert client.get(f"/api/datasets/{dataset['id']}/overview").status_code == 200
    active = client.get("/api/session").get_json()["dataset"]
    assert active["id"] == cleaned["id"]


def test_restore_switches_back_to_the_root(client, dataset):
    cleaned = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={"plan": {"remove_duplicates": True}},
    ).get_json()["dataset"]
    client.post(f"/api/datasets/{cleaned['id']}/restore")

    active = client.get("/api/session").get_json()["dataset"]
    assert active["id"] == dataset["id"]
    assert active["revision"] == 1


def test_clean_rejects_unknown_columns(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/clean",
        json={"plan": {"drop_columns": ["nope"]}},
    )
    assert response.status_code == 400
    assert "Unknown column" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_model_options(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/model/options").get_json()
    eligible = [item for item in body["targets"] if item["eligible"]]

    assert body["default_target"] == "churned"
    assert {"customer_id", "referral_code"}.isdisjoint({item["name"] for item in eligible})
    assert body["limits"]["rows"] == 241
    assert body["models"]["classification"][0]["id"] == "hist_gradient_boosting"


def test_target_analysis(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/target/churned").get_json()

    assert body["column"] == "churned"
    assert body["suggested_task"] == "classification"
    assert body["eligible"] is True
    assert body["models"]
    assert body["default_model"] == "random_forest"


def test_target_analysis_handles_identifiers(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/target/customer_id").get_json()
    assert body["eligible"] is False


def test_training_job_and_history(client, dataset):
    response = submit_training(client, dataset["id"])
    assert response.status_code == 202

    job = wait_for_job(client, response.get_json()["job"]["id"])
    assert job["status"] == "done", job.get("error")
    result = job["result"]

    assert result["task"] == "classification"
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["importance"]
    assert result["confusion_matrix"]
    assert result["timings"]["total"] > 0
    assert result["target"] == "churned"

    history = client.get(f"/api/datasets/{dataset['id']}/models").get_json()["models"]
    assert len(history) == 1
    assert history[0]["target"] == "churned"


def test_training_result_is_cached(client, dataset):
    first = submit_training(client, dataset["id"])
    wait_for_job(client, first.get_json()["job"]["id"])

    second = submit_training(client, dataset["id"]).get_json()
    assert second["cached"] is True
    assert second["job"]["status"] == "done"
    assert second["job"]["result"]["metrics"]["accuracy"] > 0


def test_training_with_different_settings_is_not_cached(client, dataset):
    first = submit_training(client, dataset["id"])
    wait_for_job(client, first.get_json()["job"]["id"])

    second = submit_training(client, dataset["id"], model="logistic_regression").get_json()
    assert second["cached"] is False


def test_training_rejects_invalid_target(client, dataset):
    response = submit_training(client, dataset["id"], target="customer_id")
    assert response.status_code == 400


def test_training_error_is_reported_by_the_job(client, dataset):
    response = submit_training(client, dataset["id"], target="city", task="regression")
    assert response.status_code == 202
    job = wait_for_job(client, response.get_json()["job"]["id"])

    assert job["status"] == "error"
    assert "numeric target" in job["error"]


def test_job_not_found(client):
    assert client.get("/api/jobs/nope").status_code == 404


# ---------------------------------------------------------------------------
# Insights, reports and export
# ---------------------------------------------------------------------------


def test_insights_contract(client, dataset):
    body = client.get(f"/api/datasets/{dataset['id']}/insights").get_json()

    assert body["dataset"]["id"] == dataset["id"]
    assert body["insights"]["summary"]
    assert body["insights"]["findings"]
    assert body["insights"]["recommendations"]
    assert body["model"] is None


def test_report_html(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/report",
        json={"format": "html", "include_charts": True},
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.content_type == "text/html; charset=utf-8"
    assert "Data quality" in html
    assert "Columns" in html
    assert "<img" in html
    assert response.headers["Content-Disposition"].startswith("attachment")


def test_report_without_charts(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/report",
        json={"format": "html", "include_charts": False},
    )
    assert "<img" not in response.get_data(as_text=True)


def test_report_pdf(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/report",
        json={"format": "pdf", "include_charts": True},
    )
    assert response.status_code == 200
    assert response.data.startswith(b"%PDF-")
    assert response.content_type == "application/pdf"


def test_report_rejects_unknown_format(client, dataset):
    response = client.post(
        f"/api/datasets/{dataset['id']}/report",
        json={"format": "docx"},
    )
    assert response.status_code == 400


def test_csv_export(client, dataset):
    response = client.get(f"/api/datasets/{dataset['id']}/export.csv")

    assert response.status_code == 200
    assert response.content_type.startswith("text/csv")
    text = response.get_data(as_text=True)
    assert text.splitlines()[0].startswith("customer_id")


# ---------------------------------------------------------------------------
# Isolation and errors
# ---------------------------------------------------------------------------


def test_datasets_are_scoped_to_the_session(app, client, dataset):
    other = app.test_client()
    response = other.get(f"/api/datasets/{dataset['id']}/overview")

    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found."


def test_unknown_dataset(client):
    assert client.get("/api/datasets/nope/overview").status_code == 404
    assert client.delete("/api/datasets/nope").status_code == 404


def test_delete_dataset(client, dataset):
    response = client.delete(f"/api/datasets/{dataset['id']}")
    assert response.status_code == 200
    assert client.get(f"/api/datasets/{dataset['id']}/overview").status_code == 404
    assert client.get("/api/session").get_json()["dataset"] is None


def test_api_errors_are_json(client):
    response = client.get("/api/datasets/nope/overview")
    assert response.is_json
    assert "error" in response.get_json()
