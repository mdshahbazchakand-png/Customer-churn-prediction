"""Customer Churn Prediction (binary classification).

Dataset: Telco Customer Churn (Kaggle, IBM sample data), 21 columns, target = Churn.
Usage:   python churn_prediction.py [path_to_csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SEED = 42

# ---------- 1. Load & clean ----------
df = pd.read_csv(CSV)
n_records = len(df)
print(f"Records loaded: {n_records}")

df = df.drop(columns=["customerID"])
# TotalCharges has blank strings for brand-new customers -> NaN -> median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(f"Missing TotalCharges values imputed (median): {n_missing}")

y = (df.pop("Churn") == "Yes").astype(int)
print(f"Churn rate: {y.mean():.1%}")

# ---------- 2. Feature engineering ----------
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                            labels=["0-12m", "13-24m", "25-48m", "49-72m"]).astype(str)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend"]
cat_cols = [c for c in df.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---------- 3. Split & compare models with cross-validation ----------
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                            class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipes, lines = {}, []
print("\n5-fold stratified CV (F1 on training set):")
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])  # preprocessing inside CV: no leakage
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")
    lines.append(f"CV F1 | {name}: {scores.mean():.3f} +/- {scores.std():.3f}")
    pipes[name] = pipe.fit(X_train, y_train)

# ---------- 4. Evaluate on held-out test set ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
print(f"\nHeld-out test set ({len(y_test)} records):")
for ax, (name, pipe) in zip(axes, pipes.items()):
    pred = pipe.predict(X_test)
    p, r, f = (precision_score(y_test, pred), recall_score(y_test, pred),
               f1_score(y_test, pred))
    print(f"\n{name}\n  precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    print(confusion_matrix(y_test, pred))
    lines.append(f"TEST | {name}: precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stayed", "Churned"], ax=ax, colorbar=False)
    ax.set_title(f"{name}\nF1 = {f:.3f}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# ---------- 5. Top features (Random Forest) ----------
rf = pipes["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
top = imp.sort_values().tail(10)
plt.figure(figsize=(7, 4))
top.plot.barh()
plt.title("Top 10 features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)

with open("results.txt", "w") as fh:
    fh.write(f"Records: {n_records}\nChurn rate: {y.mean():.1%}\n" + "\n".join(lines))
print("\nSaved: confusion_matrix.png, feature_importance.png, results.txt")
