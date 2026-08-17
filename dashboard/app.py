import streamlit as st
import pandas as pd

from src.eda import perform_eda
from src.visualizer import generate_visualizations
from src.data_cleaner import clean_data
from src.ml_model import (
    train_model,
    get_feature_importance,
    detect_problem_type,
)


st.set_page_config(
    page_title="AI Data Visualizer",
    page_icon="📊",
    layout="wide",
)


st.title("📊 AI Data Visualizer")

st.write(
    "Upload a dataset to explore, visualize and analyze your data."
)


uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)


if uploaded_file is not None:

    try:
        # ---------------------------------------------------------
        # DATA LOADING
        # ---------------------------------------------------------

        dataframe = pd.read_csv(uploaded_file)

        # ---------------------------------------------------------
        # DATA CLEANING
        # ---------------------------------------------------------

        cleaned_dataframe = clean_data(dataframe)

        st.success(
            "Dataset uploaded successfully!"
        )

        # ---------------------------------------------------------
        # DATASET PREVIEW
        # ---------------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

        # ---------------------------------------------------------
        # DATASET INFORMATION
        # ---------------------------------------------------------

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                dataframe.shape[0],
            )

        with col2:
            st.metric(
                "Columns",
                dataframe.shape[1],
            )

        with col3:
            st.metric(
                "Missing Values",
                int(
                    dataframe.isnull().sum().sum()
                ),
            )

        st.divider()

        # =========================================================
        # EXPLORATORY DATA ANALYSIS
        # =========================================================

        st.header("🔍 Exploratory Data Analysis")

        eda_results = perform_eda(
            cleaned_dataframe
        )

        # ---------------------------------------------------------
        # DATASET OVERVIEW
        # ---------------------------------------------------------

        st.subheader("Dataset Overview")

        overview = eda_results["overview"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                overview["rows"],
            )

        with col2:
            st.metric(
                "Columns",
                overview["columns"],
            )

        with col3:
            st.metric(
                "Missing Values",
                overview["missing_values"],
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                overview["duplicate_rows"],
            )

        # ---------------------------------------------------------
        # NUMERICAL STATISTICS
        # ---------------------------------------------------------

        st.subheader("Numerical Statistics")

        numerical_stats = (
            eda_results["numerical_statistics"]
        )

        if not numerical_stats.empty:

            st.dataframe(
                numerical_stats,
                use_container_width=True,
            )

        else:

            st.info(
                "No numerical columns found."
            )

        # ---------------------------------------------------------
        # CATEGORICAL STATISTICS
        # ---------------------------------------------------------

        st.subheader("Categorical Statistics")

        categorical_stats = (
            eda_results["categorical_statistics"]
        )

        if categorical_stats:

            categorical_rows = []

            for column, stats in categorical_stats.items():

                categorical_rows.append(
                    {
                        "Column": column,
                        "Unique Values": stats[
                            "unique_values"
                        ],
                        "Most Frequent": stats[
                            "most_frequent"
                        ],
                        "Frequency": stats[
                            "frequency"
                        ],
                    }
                )

            st.dataframe(
                pd.DataFrame(categorical_rows),
                use_container_width=True,
            )

        else:

            st.info(
                "No categorical columns found."
            )

        # ---------------------------------------------------------
        # CORRELATION MATRIX
        # ---------------------------------------------------------

        st.subheader("Correlation Matrix")

        correlation = (
            eda_results["correlation_matrix"]
        )

        if not correlation.empty:

            st.dataframe(
                correlation,
                use_container_width=True,
            )

        else:

            st.info(
                "Correlation matrix requires at least "
                "two numerical columns."
            )

        # ---------------------------------------------------------
        # OUTLIER DETECTION
        # ---------------------------------------------------------

        st.subheader("Outlier Detection")

        outliers = eda_results["outliers"]

        outlier_rows = []

        for column, result in outliers.items():

            outlier_rows.append(
                {
                    "Column": column,
                    "Outlier Count": result[
                        "outlier_count"
                    ],
                    "Lower Bound": result[
                        "lower_bound"
                    ],
                    "Upper Bound": result[
                        "upper_bound"
                    ],
                }
            )

        if outlier_rows:

            st.dataframe(
                pd.DataFrame(outlier_rows),
                use_container_width=True,
            )

        else:

            st.info(
                "No numerical columns available "
                "for outlier detection."
            )

        # ---------------------------------------------------------
        # AUTOMATIC INSIGHTS
        # ---------------------------------------------------------

        st.subheader(
            "💡 Automatic Insights"
        )

        for insight in eda_results["insights"]:

            st.write(
                f"• {insight}"
            )

        st.divider()

        # =========================================================
        # VISUALIZATIONS
        # =========================================================

        st.header("📊 Visualizations")

        visualizations = (
            generate_visualizations(
                cleaned_dataframe
            )
        )

        # ---------------------------------------------------------
        # HISTOGRAMS
        # ---------------------------------------------------------

        if "histograms" in visualizations:

            st.subheader("Histograms")

            for column, figure in (
                visualizations[
                    "histograms"
                ].items()
            ):

                st.write(
                    f"**{column}**"
                )

                st.pyplot(
                    figure
                )

        # ---------------------------------------------------------
        # BAR CHARTS
        # ---------------------------------------------------------

        if "bar_charts" in visualizations:

            st.subheader("Bar Charts")

            for column, figure in (
                visualizations[
                    "bar_charts"
                ].items()
            ):

                st.write(
                    f"**{column}**"
                )

                st.pyplot(
                    figure
                )

        # ---------------------------------------------------------
        # BOX PLOT
        # ---------------------------------------------------------

        if (
            visualizations.get(
                "box_plot"
            ) is not None
        ):

            st.subheader("Box Plot")

            st.pyplot(
                visualizations[
                    "box_plot"
                ]
            )

        # ---------------------------------------------------------
        # SCATTER PLOTS
        # ---------------------------------------------------------

        if "scatter_plots" in visualizations:

            st.subheader(
                "Scatter Plots"
            )

            for name, figure in (
                visualizations[
                    "scatter_plots"
                ].items()
            ):

                st.write(
                    f"**{name}**"
                )

                st.pyplot(
                    figure
                )

        # ---------------------------------------------------------
        # CORRELATION HEATMAP
        # ---------------------------------------------------------

        if (
            visualizations.get(
                "correlation_heatmap"
            ) is not None
        ):

            st.subheader(
                "Correlation Heatmap"
            )

            st.pyplot(
                visualizations[
                    "correlation_heatmap"
                ]
            )

        st.divider()

        # =========================================================
        # MACHINE LEARNING
        # =========================================================

        st.header(
            "🤖 Machine Learning Analysis"
        )

        target_column = st.selectbox(
            "Select Target Column",
            options=cleaned_dataframe.columns,
        )

        detected_type = None

        if target_column:

            try:

                detected_type = (
                    detect_problem_type(
                        cleaned_dataframe,
                        target_column,
                    )
                )

                st.info(
                    f"Detected problem type: "
                    f"{detected_type}"
                )

            except Exception as error:

                st.error(
                    f"Unable to detect problem type: "
                    f"{error}"
                )

        if st.button(
            "Train Machine Learning Model"
        ):

            if detected_type is None:

                st.warning(
                    "Unable to determine the "
                    "problem type."
                )

            else:

                try:

                    ml_results = train_model(
                        cleaned_dataframe,
                        target_column,
                        problem_type=detected_type,
                    )

                    # Save results for later
                    # report/summary integration.

                    st.session_state[
                        "ml_results"
                    ] = ml_results

                    st.success(
                        "Machine learning model "
                        "trained successfully!"
                    )

                    # -------------------------------------------------
                    # MODEL INFORMATION
                    # -------------------------------------------------

                    st.subheader(
                        "Model Information"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "Problem Type",
                            ml_results[
                                "problem_type"
                            ],
                        )

                    with col2:

                        st.metric(
                            "Training Rows",
                            ml_results[
                                "train_rows"
                            ],
                        )

                    with col3:

                        st.metric(
                            "Testing Rows",
                            ml_results[
                                "test_rows"
                            ],
                        )

                    # -------------------------------------------------
                    # MODEL PERFORMANCE
                    # -------------------------------------------------

                    st.subheader(
                        "Model Performance"
                    )

                    metrics = ml_results[
                        "metrics"
                    ]

                    if (
                        ml_results[
                            "problem_type"
                        ]
                        == "classification"
                    ):

                        accuracy = metrics[
                            "accuracy"
                        ]

                        st.metric(
                            "Accuracy",
                            f"{accuracy * 100:.2f}%",
                        )

                        st.subheader(
                            "Classification Report"
                        )

                        report = metrics[
                            "classification_report"
                        ]

                        report_rows = []

                        for label, values in (
                            report.items()
                        ):

                            if isinstance(
                                values,
                                dict,
                            ):

                                report_rows.append(
                                    {
                                        "Class": label,
                                        "Precision": values.get(
                                            "precision",
                                            0,
                                        ),
                                        "Recall": values.get(
                                            "recall",
                                            0,
                                        ),
                                        "F1 Score": values.get(
                                            "f1-score",
                                            0,
                                        ),
                                        "Support": values.get(
                                            "support",
                                            0,
                                        ),
                                    }
                                )

                        if report_rows:

                            st.dataframe(
                                pd.DataFrame(
                                    report_rows
                                ),
                                use_container_width=True,
                            )

                    else:

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )

                        with col1:

                            st.metric(
                                "MAE",
                                f"{metrics['mae']:.4f}",
                            )

                        with col2:

                            st.metric(
                                "MSE",
                                f"{metrics['mse']:.4f}",
                            )

                        with col3:

                            st.metric(
                                "RMSE",
                                f"{metrics['rmse']:.4f}",
                            )

                        with col4:

                            st.metric(
                                "R² Score",
                                f"{metrics['r2']:.4f}",
                            )

                    # -------------------------------------------------
                    # FEATURE IMPORTANCE
                    # -------------------------------------------------

                    st.subheader(
                        "Feature Importance"
                    )

                    importance_data = (
                        get_feature_importance(
                            ml_results
                        )
                    )

                    st.dataframe(
                        importance_data,
                        use_container_width=True,
                    )

                except Exception as error:

                    st.error(
                        f"Unable to train the model: "
                        f"{error}"
                    )

    except Exception as error:

        st.error(
            f"Unable to read the dataset: "
            f"{error}"
        )