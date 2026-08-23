"""
data_cleaner.py

Handles basic data cleaning and preprocessing for the AI Data Visualizer.
"""

import pandas as pd


def remove_duplicates(dataframe):
    """
    Remove duplicate rows from a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to clean.

    Returns
    -------
    pandas.DataFrame
        DataFrame with duplicate rows removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return dataframe.drop_duplicates().reset_index(drop=True)


def handle_missing_values(dataframe, strategy="auto"):
    """
    Handle missing values in a DataFrame.

    Strategies:
    - 'auto': numeric columns use median, categorical columns use mode.
    - 'drop': remove rows containing missing values.
    - 'none': leave missing values unchanged.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to clean.
    strategy : str
        Missing-value handling strategy.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values handled.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if strategy not in {"auto", "drop", "none"}:
        raise ValueError(
            "Invalid strategy. Use 'auto', 'drop', or 'none'."
        )

    cleaned_data = dataframe.copy()

    if strategy == "none":
        return cleaned_data

    if strategy == "drop":
        return cleaned_data.dropna().reset_index(drop=True)

    # Automatic handling
    for column in cleaned_data.columns:

        if cleaned_data[column].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_data[column]):
            median_value = cleaned_data[column].median()
            cleaned_data[column] = cleaned_data[column].fillna(median_value)

        else:
            mode_values = cleaned_data[column].mode()

            if not mode_values.empty:
                cleaned_data[column] = cleaned_data[column].fillna(
                    mode_values.iloc[0]
                )
            else:
                cleaned_data[column] = cleaned_data[column].fillna(
                    "Unknown"
                )

    return cleaned_data


def clean_text_columns(dataframe):
    """
    Remove unnecessary whitespace from text columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to clean.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned text columns.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_data = dataframe.copy()

    text_columns = cleaned_data.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned_data[column] = cleaned_data[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return cleaned_data


def clean_data(dataframe, missing_strategy="auto"):
    """
    Perform the complete basic cleaning process.

    Steps:
    1. Remove duplicate rows.
    2. Handle missing values.
    3. Clean whitespace from text columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to clean.
    missing_strategy : str
        Strategy for handling missing values.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_data = dataframe.copy()

    cleaned_data = remove_duplicates(cleaned_data)
    cleaned_data = handle_missing_values(
        cleaned_data,
        strategy=missing_strategy
    )
    cleaned_data = clean_text_columns(cleaned_data)

    return cleaned_data


def get_cleaning_summary(original_data, cleaned_data):
    """
    Compare the original and cleaned datasets.

    Parameters
    ----------
    original_data : pandas.DataFrame
        Original dataset.

    cleaned_data : pandas.DataFrame
        Cleaned dataset.

    Returns
    -------
    dict
        Summary of cleaning changes.
    """
    if not isinstance(original_data, pd.DataFrame):
        raise TypeError("original_data must be a pandas DataFrame.")

    if not isinstance(cleaned_data, pd.DataFrame):
        raise TypeError("cleaned_data must be a pandas DataFrame.")

    return {
        "original_rows": len(original_data),
        "cleaned_rows": len(cleaned_data),
        "duplicates_removed": (
            len(original_data) - len(original_data.drop_duplicates())
        ),
        "missing_values_before": int(
            original_data.isnull().sum().sum()
        ),
        "missing_values_after": int(
            cleaned_data.isnull().sum().sum()
        ),
    }