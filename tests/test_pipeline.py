"""End-to-end workflows: the modern API pipeline and the classic helper chain."""

from __future__ import annotations

import time

import pandas as pd

from src.data_cleaner import clean_data
from src.data_loader import load_data
from src.eda import perform_eda
from src.ml_model import train_model


def wait(client, job_id: str) -> dict:
    for _ in range(600):
        job = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError("training did not finish")


def test_full_api_workflow(client, customers_frame, tmp_path):
    source = tmp_path / "customers.csv"
    customers_frame.to_csv(source, index=False)

    # 1. upload
    with source.open("rb") as handle:
        response = client.post(
            "/api/datasets",
            data={"file": (handle, "customers.csv")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 201
    dataset = response.get_json()["dataset"]
    assert dataset["rows"] == 241

    # 2. overview
    overview = client.get(f"/api/datasets/{dataset['id']}/overview").get_json()
    assert overview["overview"]["missing_cells"] > 0
    assert overview["kinds"]["datetime"] == ["signup_date"]

    # 3. curated charts
    charts = client.get(f"/api/datasets/{dataset['id']}/charts").get_json()["charts"]
    assert 3 <= len(charts) <= 5

    # 4. cleaning preview then apply
    plan = overview["suggestions"]
    preview = client.post(
        f"/api/datasets/{dataset['id']}/cleaning/preview", json={"plan": plan}
    ).get_json()
    assert preview["summary"]["rows_after"] == 240

    cleaned = client.post(
        f"/api/datasets/{dataset['id']}/clean", json={"plan": plan}
    ).get_json()
    assert cleaned["dataset"]["revision"] == 2
    assert cleaned["summary"]["missing_after"] == 0
    cleaned_id = cleaned["dataset"]["id"]

    # 5. train on the cleaned revision
    started = client.post(
        f"/api/datasets/{cleaned_id}/train",
        json={"target": "churned", "model": "random_forest", "task": "classification", "max_rows": 0},
    )
    assert started.status_code == 202
    job = wait(client, started.get_json()["job"]["id"])
    assert job["status"] == "done", job.get("error")
    result = job["result"]
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["rows"]["sampled"] is False

    # 6. insights include the trained model
    insights = client.get(f"/api/datasets/{cleaned_id}/insights").get_json()
    assert insights["model"]["target"] == "churned"
    assert insights["insights"]["model"]["headline"]

    # 7. report mentions the model and the cleaning step
    report = client.post(
        f"/api/datasets/{cleaned_id}/report",
        json={"format": "html", "include_charts": True},
    )
    html = report.get_data(as_text=True)
    assert "Random forest" in html
    assert "Cleaning applied" in html
    assert "Accuracy" in html


def test_classic_helper_chain(customers_frame, tmp_path):
    """The original load → clean → EDA → train path still works."""
    source = tmp_path / "classic.csv"
    customers_frame.to_csv(source, index=False)

    frame = load_data(source)
    assert "signup_date" in frame.columns

    cleaned = clean_data(frame)
    assert cleaned.isna().sum().sum() == 0

    eda = perform_eda(cleaned)
    assert eda["overview"]["rows"] == len(cleaned)
    assert eda["insights"]

    result = train_model(cleaned, "churned", model_id="logistic_regression", max_rows=0)
    assert result["problem_type"] == "classification"
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0


def test_small_numeric_dataset(customers_frame):
    frame = customers_frame[["age", "income", "churned"]].dropna()
    result = train_model(frame, "churned", model_id="random_forest", max_rows=0)
    assert result["metrics"]["balanced_accuracy"] >= 0


def test_categorical_only_dataset():
    frame = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai", "Pune"] * 20,
            "plan": ["Basic", "Pro", "Basic"] * 20,
            "churn": ["yes", "no", "yes"] * 20,
        }
    )
    result = train_model(frame, "churn", model_id="random_forest", max_rows=0)
    assert result["feature_names"] == ["city", "plan"]
    assert result["metrics"]["accuracy"] >= 0
