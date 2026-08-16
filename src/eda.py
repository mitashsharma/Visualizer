"""
eda.py

Performs Exploratory Data Analysis (EDA) for the AI Data Visualizer.
"""

import pandas as pd


def validate_dataframe(dataframe):
    """
    Validate that the input is a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    return True


def dataset_overview(dataframe):
    """
    Generate basic information about the dataset.

    Returns
    -------
    dict
        Dataset overview.
    """
    validate_dataframe(dataframe)

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_values": int(dataframe.isnull().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def numerical_statistics(dataframe):
    """
    Generate descriptive statistics for numerical columns.

    Returns
    -------
    pandas.DataFrame
        Numerical statistics.
    """
    validate_dataframe(dataframe)

    numerical_data = dataframe.select_dtypes(include="number")

    if numerical_data.empty:
        return pd.DataFrame()

    return numerical_data.describe().transpose()


def categorical_statistics(dataframe):
    """
    Generate basic statistics for categorical columns.

    Returns
    -------
    dict
        Unique values and most frequent values for each categorical column.
    """
    validate_dataframe(dataframe)

    categorical_data = dataframe.select_dtypes(
        include=["object", "category", "string"]
    )

    statistics = {}

    for column in categorical_data.columns:
        value_counts = categorical_data[column].value_counts(dropna=False)

        statistics[column] = {
            "unique_values": int(categorical_data[column].nunique()),
            "most_frequent": (
                value_counts.index[0]
                if not value_counts.empty
                else None
            ),
            "frequency": (
                int(value_counts.iloc[0])
                if not value_counts.empty
                else 0
            ),
        }

    return statistics


def correlation_matrix(dataframe):
    """
    Calculate correlations between numerical columns.

    Returns
    -------
    pandas.DataFrame
        Correlation matrix.
    """
    validate_dataframe(dataframe)

    numerical_data = dataframe.select_dtypes(include="number")

    if numerical_data.shape[1] < 2:
        return pd.DataFrame()

    return numerical_data.corr()


def detect_outliers(dataframe):
    """
    Detect outliers in numerical columns using the IQR method.

    Returns
    -------
    dict
        Number of outliers and boundaries for each numerical column.
    """
    validate_dataframe(dataframe)

    numerical_data = dataframe.select_dtypes(include="number")

    results = {}

    for column in numerical_data.columns:
        series = numerical_data[column].dropna()

        if series.empty:
            results[column] = {
                "outlier_count": 0,
                "lower_bound": None,
                "upper_bound": None,
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        results[column] = {
            "outlier_count": int(len(outliers)),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return results


def generate_insights(dataframe):
    """
    Generate basic automatic insights from the dataset.

    Returns
    -------
    list
        List of human-readable insights.
    """
    validate_dataframe(dataframe)

    insights = []

    overview = dataset_overview(dataframe)

    insights.append(
        f"The dataset contains {overview['rows']} rows "
        f"and {overview['columns']} columns."
    )

    if overview["numerical_columns"]:
        insights.append(
            f"There are {len(overview['numerical_columns'])} "
            "numerical columns."
        )

    if overview["categorical_columns"]:
        insights.append(
            f"There are {len(overview['categorical_columns'])} "
            "categorical columns."
        )

    if overview["missing_values"] > 0:
        insights.append(
            f"The dataset contains {overview['missing_values']} "
            "missing values."
        )
    else:
        insights.append("The dataset contains no missing values.")

    if overview["duplicate_rows"] > 0:
        insights.append(
            f"The dataset contains {overview['duplicate_rows']} "
            "duplicate rows."
        )
    else:
        insights.append("The dataset contains no duplicate rows.")

    outlier_results = detect_outliers(dataframe)

    total_outliers = sum(
        result["outlier_count"]
        for result in outlier_results.values()
    )

    if total_outliers > 0:
        insights.append(
            f"A total of {total_outliers} potential numerical "
            "outliers were detected using the IQR method."
        )
    else:
        insights.append("No potential numerical outliers were detected.")

    return insights


def perform_eda(dataframe):
    """
    Run the complete EDA process.

    Returns
    -------
    dict
        Complete EDA results.
    """
    validate_dataframe(dataframe)

    return {
        "overview": dataset_overview(dataframe),
        "numerical_statistics": numerical_statistics(dataframe),
        "categorical_statistics": categorical_statistics(dataframe),
        "correlation_matrix": correlation_matrix(dataframe),
        "outliers": detect_outliers(dataframe),
        "insights": generate_insights(dataframe),
    }