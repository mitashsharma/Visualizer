"""
summarizer.py

Generates human-readable summaries from EDA and ML results.
"""


def generate_summary(eda_results, ml_results=None):
    """
    Generate a human-readable summary of the dataset.

    Parameters
    ----------
    eda_results : dict
        Results returned by the EDA module.
    ml_results : dict, optional
        Results returned by the ML module.

    Returns
    -------
    dict
        Summary containing overview, key findings,
        important variables, outliers, ML results,
        and recommendations.
    """

    if not isinstance(eda_results, dict):
        raise TypeError("EDA results must be a dictionary.")

    summary = {
        "dataset_overview": [],
        "key_findings": [],
        "important_variables": [],
        "outliers": [],
        "ml_results": [],
        "recommendations": [],
    }

    # Dataset Overview
    overview = eda_results.get("overview", {})

    if overview:
        rows = overview.get("rows", 0)
        columns = overview.get("columns", 0)

        summary["dataset_overview"].append(
            f"The dataset contains {rows} rows and {columns} columns."
        )

        numerical_columns = overview.get("numerical_columns", [])
        categorical_columns = overview.get("categorical_columns", [])

        if numerical_columns:
            summary["dataset_overview"].append(
                f"It contains {len(numerical_columns)} numerical columns."
            )

        if categorical_columns:
            summary["dataset_overview"].append(
                f"It contains {len(categorical_columns)} categorical columns."
            )

    # Key Findings
    missing_values = overview.get("missing_values", 0)
    duplicate_rows = overview.get("duplicate_rows", 0)

    if missing_values > 0:
        summary["key_findings"].append(
            f"The dataset contains {missing_values} missing values."
        )
    else:
        summary["key_findings"].append(
            "The dataset contains no missing values."
        )

    if duplicate_rows > 0:
        summary["key_findings"].append(
            f"The dataset contains {duplicate_rows} duplicate rows."
        )
    else:
        summary["key_findings"].append(
            "The dataset contains no duplicate rows."
        )

    # Existing EDA insights
    eda_insights = eda_results.get("insights", [])

    if isinstance(eda_insights, list):
        summary["key_findings"].extend(eda_insights)

    # Important Variables
    numerical_columns = overview.get("numerical_columns", [])

    if numerical_columns:
        summary["important_variables"].append(
            "Numerical variables: " + ", ".join(numerical_columns)
        )

    categorical_columns = overview.get("categorical_columns", [])

    if categorical_columns:
        summary["important_variables"].append(
            "Categorical variables: " + ", ".join(categorical_columns)
        )

    # Outliers
    outliers = eda_results.get("outliers", {})

    if isinstance(outliers, dict):
        for column, result in outliers.items():
            count = result.get("outlier_count", 0)

            if count > 0:
                summary["outliers"].append(
                    f"{column} contains {count} potential outlier(s)."
                )

    if not summary["outliers"]:
        summary["outliers"].append(
            "No potential numerical outliers were detected."
        )

    # ML Results
    if ml_results:
        summary["ml_results"].append(
            "Machine learning analysis was performed on the dataset."
        )

        if isinstance(ml_results, dict):
            problem_type = ml_results.get("problem_type")

            if problem_type:
                summary["ml_results"].append(
                    f"Problem type detected: {problem_type}."
                )

    else:
        summary["ml_results"].append(
            "No machine learning results are currently available."
        )

    # Recommendations
    if missing_values > 0:
        summary["recommendations"].append(
            "Review and handle missing values before further analysis."
        )

    if duplicate_rows > 0:
        summary["recommendations"].append(
            "Remove duplicate rows before using the dataset for final analysis."
        )

    if summary["outliers"] and not (
        len(summary["outliers"]) == 1
        and "No potential" in summary["outliers"][0]
    ):
        summary["recommendations"].append(
            "Review detected outliers and determine whether they are valid observations."
        )

    if not summary["recommendations"]:
        summary["recommendations"].append(
            "The dataset appears suitable for further analysis."
        )

    return summary