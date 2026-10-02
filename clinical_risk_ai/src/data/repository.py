from pathlib import Path
import pandas as pd
from src.config import ROOT

DATA_PATH = ROOT / "data/raw/clinical_risk_5000.csv"

def get_data():
    return pd.read_csv(DATA_PATH)

def get_patient(patient_id: str):
    df = get_data()
    row = df[df.patient_id.astype(str).str.upper() == patient_id.upper()]
    return None if row.empty else row.iloc[0].to_dict()
