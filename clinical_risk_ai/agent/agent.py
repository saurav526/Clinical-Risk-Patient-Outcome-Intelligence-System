import os
import re
from dotenv import load_dotenv

from agent.tools import (
    patient_lookup,
    predict_patient,
    explain_patient_tool,
    high_risk_patients,
    readmission_factors,
    dataset_summary,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def extract_patient_id(question: str):
    match = re.search(r"\bP\d{6}\b", question.upper())

    if match:
        return match.group(0)

    return None


def deterministic_router(question: str):
    """
    Reliable fallback router for common healthcare-agent requests.
    """

    q = question.lower()
    patient_id = extract_patient_id(question)

    # ---------------------------------------------------------
    # 1. SHAP / EXPLANATION REQUEST
    # ---------------------------------------------------------

    explanation_keywords = [
        "why",
        "explain",
        "explanation",
        "risk factor",
        "risk factors",
        "contributing factor",
        "contributing factors",
        "reason",
        "reasons",
        "shap",
        "feature importance",
        "what caused",
        "what is causing",
        "drivers",
    ]

    if patient_id and any(keyword in q for keyword in explanation_keywords):

        result = explain_patient_tool(patient_id)

        return {
            "tool": "explain_patient_tool",
            "arguments": {
                "patient_id": patient_id
            },
            "result": result,
        }

    # ---------------------------------------------------------
    # 2. PREDICTION REQUEST
    # ---------------------------------------------------------

    prediction_keywords = [
        "predict",
        "prediction",
        "probability",
        "risk probability",
        "readmission risk",
        "readmission probability",
        "will be readmitted",
        "risk score",
        "risk level",
    ]

    if patient_id and any(keyword in q for keyword in prediction_keywords):

        result = predict_patient(patient_id)

        return {
            "tool": "predict_patient",
            "arguments": {
                "patient_id": patient_id
            },
            "result": result,
        }

    # ---------------------------------------------------------
    # 3. PATIENT LOOKUP
    # ---------------------------------------------------------

    patient_keywords = [
        "patient information",
        "patient info",
        "patient details",
        "patient data",
        "show patient",
        "get patient",
        "lookup patient",
        "look up patient",
        "about patient",
        "information about patient",
    ]

    if patient_id and any(keyword in q for keyword in patient_keywords):

        result = patient_lookup(patient_id)

        return {
            "tool": "patient_lookup",
            "arguments": {
                "patient_id": patient_id
            },
            "result": result,
        }

    # ---------------------------------------------------------
    # 4. HIGH-RISK PATIENTS
    # ---------------------------------------------------------

    if (
        "high risk patients" in q
        or "high-risk patients" in q
        or "highest risk patients" in q
        or "top risk patients" in q
        or "top high risk" in q
    ):

        result = high_risk_patients()

        return {
            "tool": "high_risk_patients",
            "arguments": {},
            "result": result,
        }

    # ---------------------------------------------------------
    # 5. READMISSION FACTORS
    # ---------------------------------------------------------

    if (
        "factors associated with readmission" in q
        or "readmission factors" in q
        or "factors for readmission" in q
        or "what factors affect readmission" in q
        or "what factors influence readmission" in q
    ):

        result = readmission_factors()

        return {
            "tool": "readmission_factors",
            "arguments": {},
            "result": result,
        }

    # ---------------------------------------------------------
    # 6. DATASET SUMMARY
    # ---------------------------------------------------------

    if (
        "dataset" in q
        or "how many patients" in q
        or "number of patients" in q
        or "dataset summary" in q
        or "data summary" in q
    ):

        result = dataset_summary()

        return {
            "tool": "dataset_summary",
            "arguments": {},
            "result": result,
        }

    # ---------------------------------------------------------
    # 7. DEFAULT
    # ---------------------------------------------------------

    return {
        "tool": "dataset_summary",
        "arguments": {},
        "result": dataset_summary(),
    }


def run_agent(question: str):

    question = question.strip()

    if not question:
        return {
            "tool": "dataset_summary",
            "arguments": {},
            "result": dataset_summary(),
        }

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use deterministic routing first.
    #
    # This guarantees that explicit requests such as:
    # "Why is patient P100001 at risk?"
    # always use explain_patient_tool.
    # ---------------------------------------------------------

    routed = deterministic_router(question)

    return routed
def agent_query(question: str):
    return run_agent(question)