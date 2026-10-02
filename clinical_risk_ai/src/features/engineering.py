import pandas as pd

TARGET = "readmitted_30d"
ID_COL = "patient_id"

NUMERIC_FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "spo2",
    "temperature_c", "respiratory_rate", "glucose_mg_dl", "hba1c_pct",
    "total_cholesterol_mg_dl", "creatinine_mg_dl", "hemoglobin_g_dl",
    "wbc_10e3_ul", "diabetes", "hypertension", "heart_disease", "ckd",
    "prior_admissions", "length_of_stay_days", "medication_count",
    "emergency_visits_last_year", "lab_abnormal_count", "followup_days"
]
CATEGORICAL_FEATURES = ["gender", "smoking_status", "physical_activity"]

def prepare_xy(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y
