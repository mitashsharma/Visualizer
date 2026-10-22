"""Shared fixtures for the test suite."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def customers_frame() -> pd.DataFrame:
    """Mixed dataset: numeric, text, dates, ids, gaps, duplicates, whitespace."""
    rng = np.random.default_rng(7)
    n = 240
    age = rng.integers(18, 70, n)
    income = rng.normal(52_000, 15_000, n).round(2)
    score = np.clip(rng.normal(70, 12, n), 0, 100).round(1)
    city = rng.choice(["Delhi", "Mumbai", "Pune", "Indore"], n, p=[0.4, 0.3, 0.2, 0.1])
    plan = rng.choice(["Basic", "Pro", "Enterprise"], n)
    churned = ((score < 62) | (rng.random(n) < 0.12)).astype(int)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 500, n), unit="D")
    codes = [f"REF-{value:04d}" for value in rng.integers(0, 400, n)]

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:05d}" for index in range(n)],
            "signup_date": dates,
            "age": age,
            "income": income,
            "satisfaction_score": score,
            "plan": plan,
            "city": city,
            "referral_code": codes,
            "churned": churned,
        }
    )
    frame.loc[::25, "income"] = np.nan
    frame.loc[::31, "city"] = "  " + frame.loc[::31, "city"] + "  "
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


@pytest.fixture()
def numeric_frame() -> pd.DataFrame:
    """Small, clean numeric frame used for model tests."""
    rng = np.random.default_rng(11)
    n = 120
    x1 = rng.normal(50, 10, n)
    x2 = rng.integers(1, 10, n)
    noise = rng.normal(0, 3, n)
    target = (x1 + noise > 50).astype(int)
    return pd.DataFrame(
        {
            "x1": x1.round(3),
            "x2": x2,
            "x3": rng.uniform(0, 1, n).round(4),
            "segment": rng.choice(["a", "b", "c"], n),
            "target": target,
            "price": (x1 * 10 + x2).round(2),
        }
    )


@pytest.fixture()
def app():
    from src.api import create_app

    application = create_app()
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def upload_frame(client, frame: pd.DataFrame, filename: str = "customers.csv"):
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    return client.post(
        "/api/datasets",
        data={"file": (buffer, filename)},
        content_type="multipart/form-data",
    )


@pytest.fixture()
def dataset(client, customers_frame):
    response = upload_frame(client, customers_frame)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["dataset"]
