import pandas as pd

from src.data_cleaner import (
    remove_duplicates,
    handle_missing_values,
    clean_text_columns,
    clean_data,
    get_cleaning_summary,
)


def create_test_data():
    return pd.DataFrame({
        "Name": [" Alice ", "Bob", "Bob", "Charlie", None],
        "Age": [20, 21, 21, None, 25],
        "Score": [85.0, 90.0, 90.0, 88.0, None],
        "City": [" Delhi ", "Mumbai", "Mumbai", " Pune ", None],
    })


def test_remove_duplicates():
    data = create_test_data()

    cleaned = remove_duplicates(data)

    assert len(cleaned) == 4

    print("PASS - Duplicate rows removed")


def test_handle_missing_values():
    data = create_test_data()

    cleaned = handle_missing_values(data)

    missing_values = cleaned.isnull().sum().sum()

    assert missing_values == 0

    print("PASS - Missing values handled")


def test_clean_text_columns():
    data = create_test_data()

    cleaned = clean_text_columns(data)

    assert cleaned.loc[0, "Name"] == "Alice"
    assert cleaned.loc[0, "City"] == "Delhi"

    print("PASS - Text whitespace cleaned")


def test_complete_cleaning():
    data = create_test_data()

    cleaned = clean_data(data)

    assert len(cleaned) == 4
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.loc[0, "Name"] == "Alice"

    print("PASS - Complete cleaning process")


def test_cleaning_summary():
    original = create_test_data()

    cleaned = clean_data(original)

    summary = get_cleaning_summary(original, cleaned)

    assert summary["original_rows"] == 5
    assert summary["cleaned_rows"] == 4
    assert summary["duplicates_removed"] == 1
    assert summary["missing_values_after"] == 0

    print("PASS - Cleaning summary generated")


if __name__ == "__main__":
    test_remove_duplicates()
    test_handle_missing_values()
    test_clean_text_columns()
    test_complete_cleaning()
    test_cleaning_summary()

    print("\nAll data cleaner tests passed!")