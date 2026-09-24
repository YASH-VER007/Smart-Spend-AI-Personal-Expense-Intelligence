# SmartSpend AI: Personal Expense Intelligence

SmartSpend AI is a Python-based personal expense analytics application that analyzes spending patterns, detects unusual transactions, and provides a simple trend-based estimate of future spending.

## Features

* 📊 **Overview Dashboard** — Total spending, average transaction, top category, and highest-spend month.
* 📈 **Spending Analysis** — Monthly, category-wise, and transaction-level visualizations.
* 🔍 **Anomaly Detection** — Uses **Isolation Forest** to identify statistically unusual transactions.
* 🔮 **Spending Forecast** — Uses **Linear Regression** with a 9-month training and 3-month testing approach to estimate the next month's spending.
* 🖥️ **Interactive Dashboard** — Built with Streamlit.

## Dataset

**File:** `SmartSpend_AI_Expense_Dataset.csv`

* Raw transactions: **1,208**
* Cleaned transactions: **1,200**
* Period: **September 2025 – August 2026**
* Currency: INR
* Categories: Food, Groceries, Shopping, Transport, Bills, Subscriptions, Education, Entertainment, Travel, Health.

## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Quick Start

### Prerequisites

Before running the project, make sure you have:

* **Python 3.10 or higher**
* **pip** package manager
* **Git** (optional, if cloning the repository)

### Setup

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SmartSpend-AI
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

To reproduce the analysis, open:

```text
SmartSpend_AI.ipynb
```

## Project Structure

```text
SmartSpend-AI/
├── app.py
├── SmartSpend_AI.ipynb
├── SmartSpend_AI_Expense_Dataset.csv
├── requirements.txt
├── README.md
└── Project_Report.docx
```

## Methodology

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA & KPIs
   ↓
Anomaly Detection
   ↓
Forecast Evaluation
   ↓
Next-Month Estimate
   ↓
Streamlit Dashboard
```

### Machine Learning

**Isolation Forest:** Detects statistically unusual transactions.

**Linear Regression:** Uses the first 9 months for training and the last 3 months as unseen test data. The final model is then trained on all 12 months for the next-month estimate.

## Limitations

* Dataset is synthetic.
* Forecast is a simple trend-based estimate, not a precise prediction.
* Anomalies are statistical outliers and are **not necessarily fraud**.

## Objective

SmartSpend AI demonstrates a complete **Data Analytics + AI/ML workflow** using Python, from data cleaning and visualization to anomaly detection and forecasting.
