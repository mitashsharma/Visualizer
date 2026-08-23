import pandas as pd

from src.eda import (
    dataset_overview,
    numerical_statistics,
    categorical_statistics,
    correlation_matrix,
    detect_outliers,
    generate_insights,
    perform_eda,
)


def create_test_data():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Age": [20, 21, 22, 23, 100],
        "Score": [80, 85, 90, 88, 95],
        "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"],
    })


def test_dataset_overview():
    data = create_test_data()

    result = dataset_overview(data)

    assert result["rows"] == 5
    assert result["columns"] == 4
    assert "Age" in result["numerical_columns"]
    assert "Score" in result["numerical_columns"]
    assert "City" in result["categorical_columns"]
    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0

    print("PASS - Dataset overview")


def test_numerical_statistics():
    data = create_test_data()

    result = numerical_statistics(data)

    assert not result.empty
    assert "Age" in result.index
    assert "Score" in result.index
    assert "mean" in result.columns

    print("PASS - Numerical statistics")


def test_categorical_statistics():
    data = create_test_data()

    result = categorical_statistics(data)

    assert "Name" in result
    assert "City" in result
    assert result["City"]["unique_values"] == 3

    print("PASS - Categorical statistics")


def test_correlation_matrix():
    data = create_test_data()

    result = correlation_matrix(data)

    assert not result.empty
    assert "Age" in result.columns
    assert "Score" in result.columns

    print("PASS - Correlation matrix")


def test_outlier_detection():
    data = create_test_data()

    result = detect_outliers(data)

    assert "Age" in result
    assert "Score" in result

    # Age = 100 should be detected as an outlier
    assert result["Age"]["outlier_count"] == 1

    print("PASS - Outlier detection")


def test_generate_insights():
    data = create_test_data()

    result = generate_insights(data)

    assert isinstance(result, list)
    assert len(result) > 0

    print("PASS - Automatic insights")


def test_complete_eda():
    data = create_test_data()

    result = perform_eda(data)

    assert "overview" in result
    assert "numerical_statistics" in result
    assert "categorical_statistics" in result
    assert "correlation_matrix" in result
    assert "outliers" in result
    assert "insights" in result

    print("PASS - Complete EDA")


if __name__ == "__main__":
    test_dataset_overview()
    test_numerical_statistics()
    test_categorical_statistics()
    test_correlation_matrix()
    test_outlier_detection()
    test_generate_insights()
    test_complete_eda()

    print("\nAll EDA tests passed!")