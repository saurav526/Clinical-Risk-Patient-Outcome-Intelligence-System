from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from src.config import ROOT


def explain_patient(patient_features: pd.DataFrame, top_n: int = 8):
    model = joblib.load(ROOT / "models/model.joblib")
    pre = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = pre.transform(patient_features)
    names = pre.get_feature_names_out()
    try:
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            vals = explainer.shap_values(transformed)
            if isinstance(vals, list): vals = vals[1]
            contributions = np.asarray(vals)[0]
        else:
            explainer = shap.LinearExplainer(estimator, transformed)
            contributions = np.asarray(explainer.shap_values(transformed))[0]
    except Exception:
        contributions = transformed[0] * getattr(estimator, "coef_", np.ones(transformed.shape[1]))[0]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [{"feature": str(names[i]), "contribution": round(float(contributions[i]), 5), "direction": "increases risk" if contributions[i] > 0 else "decreases risk"} for i in order]
