"""
SmartSpend AI: Personal Expense Intelligence
Streamlit application – uses SmartSpend_AI_Expense_Dataset.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(page_title="SmartSpend AI", layout="wide")

# ── Data loading & cleaning ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("SmartSpend_AI_Expense_Dataset.csv")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fill missing Merchant / Description with a placeholder
    df["Merchant"] = df["Merchant"].fillna("Unknown")
    df["Description"] = df["Description"].fillna("Unknown")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Keep only rows with valid numeric Amount_INR
    df["Amount_INR"] = pd.to_numeric(df["Amount_INR"], errors="coerce")
    df = df.dropna(subset=["Amount_INR", "Date"])

    # Derive a clean YearMonth string for grouping
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    return df


df = load_data()

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("SmartSpend AI")
st.sidebar.markdown("Personal Expense Intelligence")
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Spending Analysis", "Anomaly Detection", "Spending Forecast"],
)

# ── Helper: consistent figure style ─────────────────────────────────────────
def new_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if section == "Overview":
    st.title("💰 SmartSpend AI: Personal Expense Intelligence")
    st.markdown(
        "A simple personal finance analytics tool powered by Python, "
        "Streamlit, and scikit-learn. Explore your spending, spot unusual "
        "transactions, and see a simple trend-based forecast."
    )
    st.divider()
    st.subheader("Overview")

    total_spending = df["Amount_INR"].sum()
    avg_transaction = df["Amount_INR"].mean()
    highest_category = df.groupby("Category")["Amount_INR"].sum().idxmax()
    monthly_totals = df.groupby("YearMonth")["Amount_INR"].sum()
    highest_month = monthly_totals.idxmax()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spending", f"₹{total_spending:,.2f}")
    c2.metric("Avg Transaction", f"₹{avg_transaction:,.2f}")
    c3.metric("Top Category", highest_category)
    c4.metric("Highest-Spend Month", highest_month)

    st.divider()
    st.subheader("Recent Transactions (last 10)")
    recent = (
        df.sort_values("Date", ascending=False)
        .head(10)[["Date", "Category", "Merchant", "Description", "Amount_INR", "Payment_Method"]]
        .reset_index(drop=True)
    )
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(recent, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. SPENDING ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif section == "Spending Analysis":
    st.title("📊 Spending Analysis")
    st.divider()

    # Monthly spending chart
    st.subheader("Monthly Spending")
    monthly = df.groupby("YearMonth")["Amount_INR"].sum().sort_index()
    fig, ax = new_fig((11, 4))
    ax.bar(monthly.index, monthly.values, color="#3b82d4")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spending (₹)")
    ax.set_title("Monthly Spending")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Category-wise spending chart
    st.subheader("Category-wise Spending")
    cat_spending = df.groupby("Category")["Amount_INR"].sum().sort_values(ascending=False)
    fig, ax = new_fig((10, 4))
    ax.barh(cat_spending.index, cat_spending.values, color="#7c5cd8")
    ax.set_xlabel("Total Spending (₹)")
    ax.set_title("Spending by Category")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Transaction amount distribution
    st.subheader("Transaction Amount Distribution")
    fig, ax = new_fig((10, 4))
    ax.hist(df["Amount_INR"], bins=40, color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Amount (₹)")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Distribution of Transaction Amounts")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 3. ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif section == "Anomaly Detection":
    st.title("🔍 Anomaly Detection")
    st.info(
        "**About this section:** Isolation Forest is used to identify "
        "transactions whose amounts deviate significantly from typical "
        "spending patterns. These are labelled as *unusual transactions* "
        "— not necessarily fraud, just spending that stands out."
        
    )
    st.divider()

    # Fit Isolation Forest on Amount_INR
    X = df[["Amount_INR"]].values
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_anomaly = df.copy()
    df_anomaly["anomaly_flag"] = iso.fit_predict(X)  # -1 = unusual, 1 = normal

    unusual = df_anomaly[df_anomaly["anomaly_flag"] == -1]
    num_unusual = len(unusual)

    c1, c2 = st.columns(2)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unusual Transactions Detected", f"{num_unusual:,}")

    st.divider()
    st.subheader("Top 10 Unusual / High-Value Transactions")
    top10 = (
        unusual.sort_values("Amount_INR", ascending=False)
        .head(10)[["Date", "Category", "Merchant", "Description", "Amount_INR", "Payment_Method"]]
        .reset_index(drop=True)
    )
    top10["Date"] = top10["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(top10, use_container_width=True)

    st.divider()

    # Scatter: normal vs unusual
    st.subheader("Transaction Amounts — Normal vs Unusual")
    normal = df_anomaly[df_anomaly["anomaly_flag"] == 1]
    fig, ax = new_fig((11, 4))
    ax.scatter(
        normal["Date"], normal["Amount_INR"],
        s=10, color="#3b82d4", alpha=0.5, label="Normal"
    )
    ax.scatter(
        unusual["Date"], unusual["Amount_INR"],
        s=30, color="#e03e3e", alpha=0.8, label="Unusual"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Transaction Amounts Over Time")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 4. SPENDING FORECAST
# ════════════════════════════════════════════════════════════════════════════
elif section == "Spending Forecast":
    st.title("📈 Simple Spending Forecast")
    st.info(
        "**Note:** This forecast is an illustrative trend-based estimate. "
        "The dataset covers approximately one year of transactions (Sep 2025 – Aug 2026). "
        "The **last 3 months are held out as a test set**; MAE and RMSE are calculated on "
        "those unseen months only. The final model is then **retrained on all 12 months** "
        "to produce the next-month estimate. The estimate should be treated as a "
        "directional indicator, not a precise prediction."
    )
    st.divider()

    # Aggregate monthly spending and create sequential month number
    monthly = df.groupby("YearMonth")["Amount_INR"].sum().sort_index().reset_index()
    monthly.columns = ["YearMonth", "Total_Spending"]
    monthly["Month_Num"] = np.arange(1, len(monthly) + 1)

    # Train / test split: first 9 months train, last 3 months test
    train = monthly.iloc[:9]
    test  = monthly.iloc[9:]

    X_tr = train["Month_Num"].values.reshape(-1, 1)
    y_tr = train["Total_Spending"].values
    X_te = test["Month_Num"].values.reshape(-1, 1)
    y_te = test["Total_Spending"].values

    # Fit on training set, evaluate on test set
    eval_model = LinearRegression()
    eval_model.fit(X_tr, y_tr)
    y_te_pred = eval_model.predict(X_te)

    mae  = mean_absolute_error(y_te, y_te_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))

    # Retrain final model on all 12 months, then forecast next month
    X_all = monthly["Month_Num"].values.reshape(-1, 1)
    y_all = monthly["Total_Spending"].values
    final_model = LinearRegression()
    final_model.fit(X_all, y_all)
    y_pred_all = final_model.predict(X_all)          # trend line over full history

    next_month_num      = len(monthly) + 1
    next_month_forecast = final_model.predict(np.array([[next_month_num]]))[0]
    last_period  = pd.Period(monthly["YearMonth"].iloc[-1], freq="M")
    next_period  = last_period + 1

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (test – last 3 months)", f"₹{mae:,.2f}")
    c2.metric("RMSE (test – last 3 months)", f"₹{rmse:,.2f}")
    c3.metric(f"Forecast for {next_period}", f"₹{next_month_forecast:,.2f}")

    st.divider()
    st.subheader("Historical Spending & Trend Line")

    fig, ax = new_fig((11, 5))
    # Training months
    ax.bar(train["YearMonth"], train["Total_Spending"], color="#3b82d4",
           label="Train (months 1–9)", alpha=0.85)
    # Test months
    ax.bar(test["YearMonth"], test["Total_Spending"], color="#7c5cd8",
           label="Test (months 10–12)", alpha=0.85)
    # Full trend line from final model
    ax.plot(monthly["YearMonth"], y_pred_all, color="#e03e3e", linewidth=2,
            marker="o", markersize=5, label="Trend (final model, all months)")

    # Forecast bar for next month
    ax.bar(str(next_period), next_month_forecast, color="#f59e0b",
           alpha=0.8, label=f"Forecast ({next_period})")

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spending (₹)")
    ax.set_title("Monthly Spending with Train/Test Split, Trend & Forecast")
    ax.legend()
    plt.xticks(
        list(monthly["YearMonth"]) + [str(next_period)],
        rotation=45, ha="right"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Monthly Spending Data")
    display = monthly[["YearMonth", "Total_Spending"]].copy()
    display.columns = ["Month", "Total Spending (₹)"]
    display["Total Spending (₹)"] = display["Total Spending (₹)"].round(2)
    st.dataframe(display, use_container_width=True)
