import pandas as pd

from src.ml_model import (
    detect_problem_type,
    prepare_data,
    train_model,
    get_feature_importance,
)


def create_classification_data():
    return pd.DataFrame({
        "Age": [
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39
        ],
        "Income": [
            20000, 22000, 24000, 25000, 27000,
            30000, 32000, 34000, 36000, 38000,
            40000, 42000, 44000, 46000, 48000,
            50000, 52000, 54000, 56000, 58000
        ],
        "City": [
            "Delhi", "Mumbai", "Delhi", "Pune", "Mumbai",
            "Delhi", "Pune", "Mumbai", "Delhi", "Pune",
            "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi",
            "Pune", "Mumbai", "Delhi", "Pune", "Mumbai"
        ],
        "Purchased": [
            "No", "No", "No", "No", "No",
            "Yes", "Yes", "No", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "Yes"
        ],
    })


def create_regression_data():
    return pd.DataFrame({
        "Area": [
            500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400,
            1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400
        ],
        "Bedrooms": [
            1, 1, 2, 2, 2,
            2, 3, 3, 3, 3,
            3, 4, 4, 4, 4,
            4, 5, 5, 5, 5
        ],
        "City": [
            "Delhi", "Delhi", "Mumbai", "Mumbai", "Pune",
            "Delhi", "Delhi", "Mumbai", "Pune", "Pune",
            "Delhi", "Mumbai", "Mumbai", "Pune", "Delhi",
            "Pune", "Mumbai", "Delhi", "Pune", "Mumbai"
        ],
        "Price": [
            20, 24, 28, 32, 36,
            40, 44, 48, 52, 56,
            60, 64, 68, 72, 76,
            80, 84, 88, 92, 96
        ],
    })


def test_classification_detection():
    data = create_classification_data()

    result = detect_problem_type(data, "Purchased")

    assert result == "classification"

    print("PASS - Classification detected")


def test_regression_detection():
    data = create_regression_data()

    result = detect_problem_type(data, "Price")

    assert result == "regression"

    print("PASS - Regression detected")


def test_prepare_classification_data():
    data = create_classification_data()

    X, y, preprocessor = prepare_data(
        data,
        "Purchased"
    )

    assert "Purchased" not in X.columns
    assert len(X) == len(y)
    assert preprocessor is not None

    print("PASS - Classification data prepared")


def test_prepare_regression_data():
    data = create_regression_data()

    X, y, preprocessor = prepare_data(
        data,
        "Price"
    )

    assert "Price" not in X.columns
    assert len(X) == len(y)
    assert preprocessor is not None

    print("PASS - Regression data prepared")


def test_classification_model():
    data = create_classification_data()

    result = train_model(
        data,
        "Purchased"
    )

    assert result["problem_type"] == "classification"
    assert "accuracy" in result["metrics"]
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0

    print("PASS - Classification model trained")


def test_regression_model():
    data = create_regression_data()

    result = train_model(
        data,
        "Price"
    )

    assert result["problem_type"] == "regression"
    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]

    assert result["train_rows"] > 0
    assert result["test_rows"] > 0

    print("PASS - Regression model trained")


def test_feature_importance():
    data = create_classification_data()

    result = train_model(
        data,
        "Purchased"
    )

    importance = get_feature_importance(result)

    assert not importance.empty
    assert "feature" in importance.columns
    assert "importance" in importance.columns

    print("PASS - Feature importance generated")


if __name__ == "__main__":
    test_classification_detection()
    test_regression_detection()
    test_prepare_classification_data()
    test_prepare_regression_data()
    test_classification_model()
    test_regression_model()
    test_feature_importance()

    print("\nAll ML model tests passed!")