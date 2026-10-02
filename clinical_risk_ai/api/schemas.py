from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    age:int=Field(ge=18,le=120); gender:str; bmi:float; systolic_bp:float; diastolic_bp:float
    heart_rate:float; spo2:float; temperature_c:float; respiratory_rate:float; glucose_mg_dl:float
    hba1c_pct:float; total_cholesterol_mg_dl:float; creatinine_mg_dl:float; hemoglobin_g_dl:float
    wbc_10e3_ul:float; diabetes:int; hypertension:int; heart_disease:int; ckd:int; smoking_status:str
    physical_activity:str; prior_admissions:int; length_of_stay_days:float; medication_count:int
    emergency_visits_last_year:int; lab_abnormal_count:int; followup_days:int


class QueryRequest(BaseModel): question:str=Field(min_length=3,max_length=1000)
