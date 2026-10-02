# this file is used to load the trained model and make predictions on new data, as well as log the predictions to the database for tracking and analysis
from pathlib import Path
import joblib
import pandas as pd
from src.config import ROOT, MODEL_VERSION
from src.db import SessionLocal, PredictionLog

MODEL_PATH = ROOT / "models/model.joblib"

def load_model():
    if not MODEL_PATH.exists(): raise FileNotFoundError("Model not found. Run: python -m src.models.train")
    return joblib.load(MODEL_PATH)

def predict_dataframe(df: pd.DataFrame, patient_id=None, source="api"):
    model = load_model()
    probability = float(model.predict_proba(df)[:,1][0])
    risk = "High" if probability >= .70 else "Moderate" if probability >= .40 else "Low"
    result = {"patient_id": patient_id, "probability": round(probability,4), "prediction": int(probability >= .5), "risk": risk, "model_version": MODEL_VERSION}
    try:
        db = SessionLocal(); db.add(PredictionLog(patient_id=patient_id, probability=probability, risk=risk, model_version=MODEL_VERSION, request_source=source)); db.commit(); db.close()
    except Exception: pass
    return result
