import matplotlib.pyplot as plt
import pandas as pd


def plot_histograms(dataframe):
    """
    Generate histograms for all numerical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Dictionary containing histogram figures.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    numerical_columns = dataframe.select_dtypes(include="number").columns

    figures = {}

    for column in numerical_columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(dataframe[column].dropna(), bins=20)

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        figures[column] = fig

    return figures
