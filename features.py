"""Shared feature engineering for training and the Streamlit app."""
import pandas as pd

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean TotalCharges and add num_services, avg_monthly_spend, tenure_group."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                labels=["0-12m", "13-24m", "25-48m", "49-72m"]).astype(str)
    return df


def predict_churn_probability(model, customer: dict) -> float:
    """Probability (0-1) that one customer churns. `customer` holds the raw Telco columns."""
    row = add_features(pd.DataFrame([customer]))
    return float(model.predict_proba(row)[0, 1])
