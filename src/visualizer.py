"""
visualizer.py

Creates visualizations for the AI Data Visualizer project.
"""

import matplotlib.pyplot as plt
import pandas as pd


def validate_dataframe(dataframe):
    """Validate the input DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    return True


def plot_histograms(dataframe):
    """Generate histograms for all numerical columns."""

    validate_dataframe(dataframe)

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    figures = {}

    for column in numerical_columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            dataframe[column].dropna(),
            bins=20
        )

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        figures[column] = fig

    return figures


def plot_bar_chart(dataframe, column):
    """Generate a bar chart for a categorical column."""

    validate_dataframe(dataframe)

    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    values = dataframe[column].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    values.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Bar Chart of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    return fig


def plot_line_chart(dataframe, x_column, y_column):
    """Generate a line chart using two columns."""

    validate_dataframe(dataframe)

    if x_column not in dataframe.columns:
        raise ValueError(f"Column '{x_column}' does not exist.")

    if y_column not in dataframe.columns:
        raise ValueError(f"Column '{y_column}' does not exist.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        dataframe[x_column],
        dataframe[y_column],
        marker="o"
    )

    ax.set_title(
        f"{y_column} vs {x_column}"
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return fig


def plot_scatter_plot(dataframe, x_column, y_column):
    """Generate a scatter plot for two numerical columns."""

    validate_dataframe(dataframe)

    if x_column not in dataframe.columns:
        raise ValueError(f"Column '{x_column}' does not exist.")

    if y_column not in dataframe.columns:
        raise ValueError(f"Column '{y_column}' does not exist.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        dataframe[x_column],
        dataframe[y_column]
    )

    ax.set_title(
        f"{y_column} vs {x_column}"
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return fig


def plot_box_plots(dataframe):
    """Generate box plots for numerical columns."""

    validate_dataframe(dataframe)

    numerical_data = dataframe.select_dtypes(
        include="number"
    )

    if numerical_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    numerical_data.boxplot(
        ax=ax
    )

    ax.set_title("Numerical Columns Box Plot")
    ax.set_ylabel("Values")

    plt.xticks(rotation=45)

    return fig


def plot_correlation_heatmap(dataframe):
    """Generate a correlation heatmap for numerical columns."""

    validate_dataframe(dataframe)

    numerical_data = dataframe.select_dtypes(
        include="number"
    )

    if numerical_data.shape[1] < 2:
        return None

    correlation = numerical_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        correlation,
        interpolation="nearest"
    )

    ax.set_title("Correlation Heatmap")

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    fig.colorbar(image, ax=ax)

    return fig


def generate_visualizations(dataframe):
    """
    Automatically generate suitable visualizations.

    Returns
    -------
    dict
        Dictionary containing generated figures.
    """

    validate_dataframe(dataframe)

    visualizations = {}

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Histograms
    if numerical_columns:
        visualizations["histograms"] = plot_histograms(
            dataframe
        )

    # Box plot
    if numerical_columns:
        visualizations["box_plot"] = plot_box_plots(
            dataframe
        )

    # Correlation heatmap
    if len(numerical_columns) >= 2:
        visualizations["correlation_heatmap"] = (
            plot_correlation_heatmap(dataframe)
        )

    # Bar charts
    if categorical_columns:
        visualizations["bar_charts"] = {}

        for column in categorical_columns:
            visualizations["bar_charts"][column] = (
                plot_bar_chart(
                    dataframe,
                    column
                )
            )

    # Scatter plots
    if len(numerical_columns) >= 2:
        visualizations["scatter_plots"] = {}

        for index, x_column in enumerate(
            numerical_columns
        ):
            for y_column in numerical_columns[
                index + 1:
            ]:
                visualizations["scatter_plots"][
                    f"{x_column}_vs_{y_column}"
                ] = plot_scatter_plot(
                    dataframe,
                    x_column,
                    y_column
                )

    return visualizations