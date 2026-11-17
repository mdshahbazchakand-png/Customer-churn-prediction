"""Train the tuned Random Forest churn model and save it for the app.

Same data handling, split and tuned hyperparameters as churn_prediction_v2.py,
so the test metrics should match that script's tuned Random Forest results.

Usage: python train_model.py [path_to_csv] [output_model_path]
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import NUM_COLS, add_features

CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "churn_model.joblib"
SEED = 42

df = pd.read_csv(CSV).drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
y = (df.pop("Churn") == "Yes").astype(int)
df = add_features(df)

cat_cols = [c for c in df.columns if c not in NUM_COLS]
pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=SEED)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(f"Test precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f} "
      f"F1={f1_score(y_test, pred):.3f} ROC-AUC={roc_auc_score(y_test, proba):.3f}")

joblib.dump(model, OUT, compress=3)
size_mb = os.path.getsize(OUT) / 1e6
print(f"Saved {OUT} ({size_mb:.1f} MB)")
if size_mb > 24:
    print("WARNING: file is over 24 MB; GitHub's browser upload limit is 25 MB.")
