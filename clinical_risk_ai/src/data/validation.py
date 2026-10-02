from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "patient_id", "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
    "heart_rate", "spo2", "temperature_c", "respiratory_rate",
    "glucose_mg_dl", "hba1c_pct", "total_cholesterol_mg_dl",
    "creatinine_mg_dl", "hemoglobin_g_dl", "wbc_10e3_ul",
    "diabetes", "hypertension", "heart_disease", "ckd",
    "smoking_status", "physical_activity", "prior_admissions",
    "length_of_stay_days", "medication_count", "emergency_visits_last_year",
    "lab_abnormal_count", "followup_days", "admission_date", "readmitted_30d"
}

def load_and_validate(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["patient_id"].duplicated().any():
        raise ValueError("patient_id must be unique")
    if not set(df["readmitted_30d"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1")
    return df
