# this file is used to provide explanations for model predictions using SHAP values and feature contributions
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import ROOT
from src.data.validation import load_and_validate
from src.features.engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES


MODEL_PATH = ROOT / "models" / "model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "clinical_risk_5000.csv"


def _load_background(model, sample_size=1000):
    df = load_and_validate(DATA_PATH)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    background = df[features].copy()

    if len(background) > sample_size:
        background = background.sample(
            sample_size,
            random_state=42
        )

    return model.named_steps["preprocessor"].transform(background)


def _group_name(transformed_name: str) -> str:
    name = transformed_name

    if name.startswith("num__"):
        return name.replace("num__", "", 1)

    if name.startswith("cat__"):
        name = name.replace("cat__", "", 1)

        for feature in CATEGORICAL_FEATURES:
            prefix = feature + "_"

            if name.startswith(prefix):
                return feature

    return name


def _pretty_name(name: str) -> str:

    labels = {
        "age": "Age",
        "bmi": "BMI",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "heart_rate": "Heart Rate",
        "spo2": "SpO2",
        "temperature_c": "Temperature",
        "respiratory_rate": "Respiratory Rate",
        "glucose_mg_dl": "Glucose",
        "hba1c_pct": "HbA1c",
        "total_cholesterol_mg_dl": "Total Cholesterol",
        "creatinine_mg_dl": "Creatinine",
        "hemoglobin_g_dl": "Hemoglobin",
        "wbc_10e3_ul": "WBC",
        "diabetes": "Diabetes",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "ckd": "CKD",
        "prior_admissions": "Prior Admissions",
        "length_of_stay_days": "Length of Stay",
        "medication_count": "Medication Count",
        "emergency_visits_last_year": "Emergency Visits",
        "lab_abnormal_count": "Abnormal Labs",
        "followup_days": "Follow-up Days",
        "gender": "Gender",
        "smoking_status": "Smoking Status",
        "physical_activity": "Physical Activity",
    }

    return labels.get(
        name,
        name.replace("_", " ").title()
    )


def explain_patient(
    patient_features: pd.DataFrame,
    top_n: int = 8
):

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed = np.asarray(
        preprocessor.transform(patient_features),
        dtype=float
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    # Logistic Regression
    if hasattr(estimator, "coef_"):

        background = _load_background(
            model,
            sample_size=1000
        )

        background_mean = np.asarray(
            background,
            dtype=float
        ).mean(axis=0)

        coefficients = np.asarray(
            estimator.coef_,
            dtype=float
        )[0]

        contributions = (
            transformed[0] - background_mean
        ) * coefficients

    # Tree-based models
    else:

        background = _load_background(
            model,
            sample_size=300
        )

        explainer = shap.TreeExplainer(
            estimator,
            background
        )

        values = explainer.shap_values(
            transformed
        )

        if isinstance(values, list):

            contributions = np.asarray(
                values[1]
            )[0]

        else:

            values = np.asarray(values)

            if values.ndim == 3:
                contributions = values[0, :, 1]
            else:
                contributions = values[0]

    # Aggregate one-hot encoded features
    # back into their original feature names.
    grouped = {}

    for i, transformed_name in enumerate(
        feature_names
    ):

        group = _group_name(
            str(transformed_name)
        )

        grouped.setdefault(
            group,
            0.0
        )

        grouped[group] += float(
            contributions[i]
        )

    # Sort by absolute contribution
    ranked = sorted(
        grouped.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    ranked = ranked[:top_n]

    result = []

    for feature, value in ranked:

        if value > 0:
            direction = "increases risk"
        elif value < 0:
            direction = "decreases risk"
        else:
            direction = "neutral"

        result.append(
            {
                "feature": _pretty_name(feature),
                "contribution": round(
                    float(value),
                    5
                ),
                "direction": direction,
            }
        )

    return result