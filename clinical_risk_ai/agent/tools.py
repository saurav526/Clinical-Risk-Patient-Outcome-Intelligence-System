from pathlib import Path
import pandas as pd
from src.data.repository import get_data, get_patient
from src.models.predict import predict_dataframe
from src.models.explain import explain_patient

FEATURE_DROP = ["patient_id", "readmitted_30d", "admission_date"]

def patient_lookup(patient_id: str): return get_patient(patient_id)

def predict_patient(patient_id: str):
    row = get_patient(patient_id)
    if row is None: return {"error": f"Patient {patient_id} not found"}
    x = pd.DataFrame([row]).drop(columns=FEATURE_DROP, errors="ignore")
    return predict_dataframe(x, patient_id=patient_id, source="agent")

def explain_patient_tool(patient_id: str):
    row = get_patient(patient_id)
    if row is None: return {"error": f"Patient {patient_id} not found"}
    x = pd.DataFrame([row]).drop(columns=FEATURE_DROP, errors="ignore")
    return {"patient_id": patient_id, "drivers": explain_patient(x)}

def high_risk_patients(limit=10):
    df = get_data(); model_df = df.drop(columns=FEATURE_DROP, errors="ignore")
    from src.models.predict import load_model
    probs = load_model().predict_proba(model_df)[:,1]
    out = df[["patient_id","age","diabetes","heart_disease","ckd"]].copy(); out["risk_probability"] = probs
    return out.sort_values("risk_probability", ascending=False).head(limit).round(4).to_dict("records")

def readmission_factors():
    df = get_data()
    return {
      "overall_readmission_rate": round(float(df.readmitted_30d.mean()),4),
      "by_diabetes": df.groupby("diabetes").readmitted_30d.mean().round(4).to_dict(),
      "by_hypertension": df.groupby("hypertension").readmitted_30d.mean().round(4).to_dict(),
      "by_heart_disease": df.groupby("heart_disease").readmitted_30d.mean().round(4).to_dict(),
      "by_ckd": df.groupby("ckd").readmitted_30d.mean().round(4).to_dict()
    }

def dataset_summary():
    df = get_data()
    return {"rows": len(df), "features": len(df.columns), "readmission_rate": round(float(df.readmitted_30d.mean()),4), "missing_cells": int(df.isna().sum().sum())}
