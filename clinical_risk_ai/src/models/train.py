from pathlib import Path
import json
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.data.validation import load_and_validate
from src.features.engineering import prepare_xy, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.config import ROOT, MODEL_VERSION
from src.models.registry import register_model

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES)
    ])

def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist()
    }

def main():
    df = load_and_validate(ROOT / "data/raw/clinical_risk_5000.csv")
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=.04, max_depth=3, random_state=42),
    }
    import os

    mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
)
    best = None
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_version", MODEL_VERSION)
                mlflow.log_param("algorithm", name)
                mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float))})
                mlflow.sklearn.log_model(pipe, "model")
        if best is None or metrics["roc_auc"] > best[2]["roc_auc"]:
            best = (name, pipe, metrics)
    name, model, metrics = best
    joblib.dump(model, MODELS / "model.joblib")
    feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    (MODELS / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (MODELS / "metrics.json").write_text(json.dumps({"model_version": MODEL_VERSION, "best_model": name, "metrics": metrics}, indent=2))
    register_model(name, metrics)
    print(json.dumps({"model_version": MODEL_VERSION, "best_model": name, **metrics}, indent=2))

if __name__ == "__main__": main()
