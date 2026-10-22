"""Generate deterministic benchmark datasets (not committed to Git)."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

CATEGORY = ["Basic", "Plus", "Pro", "Enterprise", "Trial"]
CITIES = [
    "Delhi", "Mumbai", "Pune", "Bengaluru", "Chennai", "Kolkata",
    "Hyderabad", "Jaipur", "Ahmedabad", "Lucknow", "Indore", "Surat",
]
PLANS = ["Monthly", "Annual", "Quarterly"]


def make_dataset(n_rows: int, seed: int = 7, label_noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, n_rows)
    income = rng.normal(52_000, 18_000, n_rows).round(2)
    score = rng.normal(68, 12, n_rows).round(2)
    tenure = rng.integers(0, 96, n_rows).astype(float)
    price = rng.choice([199.0, 299.0, 499.0, 899.0], n_rows)
    support_calls = rng.poisson(1.6, n_rows).astype(float)

    churn = (
        (score < 60).astype(int)
        | ((support_calls > 2) & (tenure < 24)).astype(int)
    )
    if label_noise:
        # Real-world churn is not a deterministic rule; flip some labels so a
        # model cannot reach a suspicious 100% accuracy.
        flips = rng.random(n_rows) < label_noise
        churn = np.where(flips, 1 - churn, churn)

    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 600, n_rows), unit="D"
    )

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:07d}" for index in range(n_rows)],
            "signup_date": dates.strftime("%Y-%m-%d"),
            "age": age,
            "income": income,
            "satisfaction_score": score,
            "tenure_months": tenure,
            "plan_price": price,
            "support_calls": support_calls,
            "segment": rng.choice(CATEGORY, n_rows),
            "city": rng.choice(CITIES, n_rows),
            "billing_plan": rng.choice(PLANS, n_rows),
            "referral_code": [
                f"REF-{value:04d}" for value in rng.integers(0, 400, n_rows)
            ],
            "churned": churn,
        }
    )

    # Realistic data-quality problems.
    missing_income = rng.choice(n_rows, size=max(1, n_rows // 40), replace=False)
    frame.loc[missing_income, "income"] = np.nan

    missing_city = rng.choice(n_rows, size=max(1, n_rows // 80), replace=False)
    frame.loc[missing_city, "city"] = None

    missing_score = rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)
    frame.loc[missing_score, "satisfaction_score"] = np.nan

    frame["segment"] = frame["segment"].map(lambda value: f"  {value} ")

    duplicate_rows = frame.sample(frac=0.01, random_state=seed)
    frame = pd.concat([frame, duplicate_rows], ignore_index=True)

    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ensure_datasets(sizes=(20_000, 100_000)) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    for size in sizes:
        path = DATA_DIR / f"customers_{size}.csv"
        if not path.exists():
            make_dataset(size).to_csv(path, index=False)
        paths[size] = path

    return paths


def ensure_sample(rows: int = 1_200, seed: int = 11) -> Path:
    """A small, realistic dataset used by the "Load sample" button."""
    path = Path(__file__).resolve().parent.parent / "data" / "sample_customers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_dataset(rows, seed=seed, label_noise=0.12).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys

    if "--sample" in sys.argv:
        print(f"sample -> {ensure_sample()}")
    else:
        for size, path in ensure_datasets().items():
            print(f"{size:>8,} rows -> {path}")
