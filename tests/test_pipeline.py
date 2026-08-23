import pandas as pd

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.eda import perform_eda
from src.ml_model import train_model


def create_test_dataset():
    data = pd.DataFrame({
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
            " Delhi ", "Mumbai", "Delhi", "Pune", "Mumbai",
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

    # Add a duplicate row
    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True
    )

    # Add missing values
    data.loc[2, "Income"] = None
    data.loc[5, "City"] = None

    return data


def main():
    print("Creating test dataset...")

    original_data = create_test_dataset()

    # Save dataset
    original_data.to_csv(
        "tests/pipeline_dataset.csv",
        index=False
    )

    print("PASS - Test dataset created")

    # 1. Data loading
    data = load_data("tests/pipeline_dataset.csv")

    assert isinstance(data, pd.DataFrame)

    print("PASS - Data loading")

    # 2. Data cleaning
    cleaned_data = clean_data(data)

    assert len(cleaned_data) < len(data)
    assert cleaned_data.isnull().sum().sum() == 0

    print("PASS - Data cleaning")

    # 3. EDA
    eda_results = perform_eda(cleaned_data)

    assert "overview" in eda_results
    assert "numerical_statistics" in eda_results
    assert "categorical_statistics" in eda_results
    assert "correlation_matrix" in eda_results
    assert "outliers" in eda_results
    assert "insights" in eda_results

    print("PASS - EDA")

    # 4. Machine learning
    ml_results = train_model(
        cleaned_data,
        "Purchased"
    )

    assert ml_results["problem_type"] == "classification"
    assert "accuracy" in ml_results["metrics"]

    print("PASS - Machine learning")

    print("\n================================")
    print("FULL PIPELINE TEST PASSED!")
    print("================================")


if __name__ == "__main__":
    main()