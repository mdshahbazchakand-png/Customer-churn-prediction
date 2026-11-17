"""Customer Churn Prediction v2: hyperparameter tuning + ROC-AUC.

Compares untuned vs tuned Logistic Regression and Random Forest.
Dataset: Telco Customer Churn (Kaggle). Target = Churn.
Usage:   python churn_prediction_v2.py [path_to_csv]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SEED = 42

# ---------- 1. Load, clean, engineer features ----------
df = pd.read_csv(CSV)
n_records = len(df)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
y = (df.pop("Churn") == "Yes").astype(int)

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

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------- 2. Models: untuned baseline vs tuned (GridSearchCV) ----------
base = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
}
grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"clf__n_estimators": [200, 400],
                      "clf__max_depth": [8, 12, None],
                      "clf__min_samples_leaf": [3, 5]},
}


def test_metrics(model):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return dict(precision=precision_score(y_test, pred), recall=recall_score(y_test, pred),
                f1=f1_score(y_test, pred), roc_auc=roc_auc_score(y_test, proba), pred=pred, proba=proba)


results, tuned_models, lines = {}, {}, []
for name, clf in base.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    untuned = clone(pipe).fit(X_train, y_train)
    gs = GridSearchCV(pipe, grids[name], cv=cv, scoring={"f1": "f1", "roc_auc": "roc_auc"},
                      refit="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned_models[name] = gs.best_estimator_
    u, t = test_metrics(untuned), test_metrics(gs.best_estimator_)
    results[name] = (u, t)
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    i = gs.best_index_
    lines += [
        f"{name}",
        f"  best params: {best}",
        f"  CV F1 (tuned): {gs.cv_results_['mean_test_f1'][i]:.3f} | CV ROC-AUC (tuned): {gs.cv_results_['mean_test_roc_auc'][i]:.3f}",
        f"  TEST untuned: precision={u['precision']:.3f} recall={u['recall']:.3f} F1={u['f1']:.3f} ROC-AUC={u['roc_auc']:.3f}",
        f"  TEST tuned  : precision={t['precision']:.3f} recall={t['recall']:.3f} F1={t['f1']:.3f} ROC-AUC={t['roc_auc']:.3f}",
        f"  confusion matrix (tuned, test): {confusion_matrix(y_test, t['pred']).tolist()}",
        "",
    ]

header = f"Records: {n_records} | Churn rate: {y.mean():.1%} | Test set: {len(y_test)} records\n"
report = header + "\n".join(lines)
print(report)
with open("results_v2.txt", "w") as fh:
    fh.write(report)

# ---------- 3. Plots ----------
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, model in tuned_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                   name=f"{name} (AUC = {results[name][1]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess")
ax.set_title("ROC curves (tuned models, test set)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes, tuned_models.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, results[name][1]["pred"],
                                            display_labels=["Stayed", "Churned"], ax=ax, colorbar=False)
    ax.set_title(f"{name} (tuned)\nF1 = {results[name][1]['f1']:.3f}")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)

rf = tuned_models["Random Forest"]
imp = pd.Series(rf.named_steps["clf"].feature_importances_,
                index=rf.named_steps["pre"].get_feature_names_out())
plt.figure(figsize=(7, 4))
imp.sort_values().tail(10).plot.barh()
plt.title("Top 10 features (tuned Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance_v2.png", dpi=150)
print("Saved: roc_curve.png, confusion_matrix_v2.png, feature_importance_v2.png, results_v2.txt")
