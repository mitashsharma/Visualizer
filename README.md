# AI Data Visualizer

AI Data Visualizer is a Python-based data analysis and visualization platform that allows users to upload datasets and automatically perform data cleaning, exploratory data analysis, visualization, machine learning analysis, and generate easy-to-understand insights and reports.

## 🚀 Project Overview

The goal of this project is to simplify the data analysis workflow by bringing multiple stages of data processing into a single application.

The planned workflow is:

**Upload Dataset → Data Cleaning → EDA → Visualization → ML Insights → AI Summary → Report**

The application will support common dataset formats such as CSV and Excel files.

## ✨ Planned Features

* 📂 Upload CSV and Excel datasets
* 🧹 Automatic data cleaning

  * Missing value handling
  * Duplicate detection and removal
  * Data type handling
  * Basic data formatting
* 📊 Exploratory Data Analysis (EDA)

  * Descriptive statistics
  * Correlation analysis
  * Outlier detection
  * Numerical and categorical analysis
* 📈 Automatic data visualizations
* 🤖 Machine learning-based insights
* 📝 AI-generated dataset summaries
* 📄 PDF/HTML report generation
* 🖥️ Interactive Streamlit dashboard
* 📥 Downloadable analysis reports

## 🏗️ Project Structure

```text
AI_Data_Visualizer/
│
├── data/
│   └── uploaded_files/
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── eda.py
│   ├── ml_model.py
│   ├── visualizer.py
│   ├── summarizer.py
│   └── report_generator.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* OpenPyXL
* Streamlit
* ReportLab

## 👥 Development

The project is being developed collaboratively using Git and GitHub.

### Data Processing & Machine Learning

The data analysis side includes:

* `data_loader.py` — Dataset loading and validation
* `data_cleaner.py` — Data cleaning and preprocessing
* `eda.py` — Exploratory data analysis
* `ml_model.py` — Machine learning analysis

### Visualization & Application

The application side includes:

* `visualizer.py` — Charts and graphs
* `summarizer.py` — AI-generated summaries
* `report_generator.py` — Report generation
* `dashboard/app.py` — Streamlit dashboard

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mitashsharma/Visualizer.git
cd Visualizer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Dashboard

Once the dashboard has been implemented, run:

```bash
streamlit run dashboard/app.py
```

The application will open in your browser.

## 📌 Project Status

**Currently in development.**

The initial repository structure and project dependencies have been set up. Development is being carried out module-by-module, with each component tested before integration.

## 🔮 Future Improvements

* Support for additional file formats
* Automated feature selection
* More machine learning algorithms
* Advanced AI-powered insights
* Interactive visualizations
* Improved report customization
* Model performance comparison
* Deployment as a web application

## 📄 License

This project is developed for educational and project purposes.
