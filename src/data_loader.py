"""
data_loader.py

Handles loading CSV and Excel datasets for the AI Data Visualizer.
"""

from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_data(file_path):
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats are: CSV, XLSX, XLS."
        )

    try:
        if extension == ".csv":
            return pd.read_csv(file_path)

        return pd.read_excel(file_path)

    except Exception as error:
        raise ValueError(f"Unable to load file: {error}") from error


def get_basic_info(dataframe):
    """
    Return basic information about a dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to analyze.

    Returns
    -------
    dict
        Basic dataset information.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "missing_values": int(dataframe.isnull().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def display_basic_info(dataframe):
    """
    Display basic information about a dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to display.
    """
    info = get_basic_info(dataframe)

    print(f"Rows: {info['rows']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing values: {info['missing_values']}")
    print(f"Duplicate rows: {info['duplicate_rows']}")
    print(f"Column names: {info['column_names']}")
    