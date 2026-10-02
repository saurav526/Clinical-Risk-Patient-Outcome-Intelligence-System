import json, pandas as pd
from fastapi import APIRouter, HTTPException, Header, Depends
from api.schemas import PatientInput, QueryRequest
from agent.agent import agent_query
from agent.tools import patient_lookup,predict_patient,explain_patient_tool,high_risk_patients,readmission_factors,dataset_summary
from src.models.predict import predict_dataframe
from src.config import API_KEY

router=APIRouter()

def require_api_key(x_api_key: str|None=Header(default=None)):
    if API_KEY != "change-me-in-production" and x_api_key != API_KEY: raise HTTPException(status_code=401,detail="Invalid API key")

@router.get("/health")
def health(): return {"status":"ok","service":"clinical-risk-ai","version":"2.0.0"}

@router.get("/model")
def model_info():
    from src.models.registry import get_registry
    return get_registry()

@router.post("/predict",dependencies=[Depends(require_api_key)])
def predict(patient:PatientInput): return predict_dataframe(pd.DataFrame([patient.model_dump()]),source="api")

@router.get("/patients/{patient_id}")
def patient(patient_id:str):
    row=patient_lookup(patient_id)
    if row is None: raise HTTPException(404,"Patient not found")
    return row

@router.get("/patients/{patient_id}/risk",dependencies=[Depends(require_api_key)])
def patient_risk(patient_id:str):
    result=predict_patient(patient_id)
    if "error" in result: raise HTTPException(404,result["error"])
    return result

@router.get("/patients/{patient_id}/explanation",dependencies=[Depends(require_api_key)])
def explanation(patient_id:str):
    result=explain_patient_tool(patient_id)
    if "error" in result: raise HTTPException(404,result["error"])
    return result

@router.get("/analytics/high-risk")
def high_risk(limit:int=10): return high_risk_patients(min(max(limit,1),100))
@router.get("/analytics/factors")
def factors(): return readmission_factors()
@router.get("/analytics/summary")
def summary(): return dataset_summary()


@router.post("/query",dependencies=[Depends(require_api_key)])
def query(request:QueryRequest): return agent_query(request.question)
