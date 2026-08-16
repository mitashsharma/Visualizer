"""
ml_model.py

Provides basic machine learning analysis for the AI Data Visualizer.
Supports classification and regression tasks.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def validate_data(dataframe, target_column):
    """
    Validate the dataset and target column.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if dataframe[target_column].isnull().all():
        raise ValueError(
            f"Target column '{target_column}' contains only missing values."
        )


def detect_problem_type(dataframe, target_column):
    """
    Automatically determine whether the problem is classification
    or regression.

    Classification is selected for categorical targets or numeric
    targets with a small number of unique values.

    Returns
    -------
    str
        'classification' or 'regression'
    """
    validate_data(dataframe, target_column)

    target = dataframe[target_column]

    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique()

        if unique_values <= 10:
            return "classification"

        return "regression"

    return "classification"


def prepare_data(dataframe, target_column):
    """
    Separate features and target and prepare preprocessing pipelines.

    Returns
    -------
    X : pandas.DataFrame
        Feature data.

    y : pandas.Series
        Target data.

    preprocessor : ColumnTransformer
        Preprocessing pipeline.
    """
    validate_data(dataframe, target_column)

    data = dataframe.dropna(subset=[target_column]).copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] == 0:
        raise ValueError("No feature columns available for training.")

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found.")

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return X, y, preprocessor


def train_model(
    dataframe,
    target_column,
    problem_type=None,
    test_size=0.2,
    random_state=42,
):
    """
    Train a Random Forest classification or regression model.

    Returns
    -------
    dict
        Trained model and evaluation results.
    """
    X, y, preprocessor = prepare_data(
        dataframe,
        target_column,
    )

    if problem_type is None:
        problem_type = detect_problem_type(
            dataframe,
            target_column,
        )

    if problem_type not in {"classification", "regression"}:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    stratify = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

    else:
        mse = mean_squared_error(y_test, predictions)

        metrics = {
            "mae": float(
                mean_absolute_error(y_test, predictions)
            ),
            "mse": float(mse),
            "rmse": float(mse ** 0.5),
            "r2": float(
                r2_score(y_test, predictions)
            ),
        }

    return {
        "model": pipeline,
        "problem_type": problem_type,
        "target_column": target_column,
        "metrics": metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def get_feature_importance(trained_result):
    """
    Extract feature importance from a trained Random Forest model.

    Returns
    -------
    pandas.DataFrame
        Feature importance table.
    """
    if not isinstance(trained_result, dict):
        raise TypeError("trained_result must be a dictionary.")

    if "model" not in trained_result:
        raise ValueError("Trained model not found.")

    pipeline = trained_result["model"]

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The selected model does not provide feature importance."
        )

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_data = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return importance_data.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)