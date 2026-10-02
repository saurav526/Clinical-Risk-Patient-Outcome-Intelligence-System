import json, os, re
from dotenv import load_dotenv
from agent.tools import predict_patient, explain_patient_tool, high_risk_patients, readmission_factors, dataset_summary
from src.db import SessionLocal, AgentAuditLog

load_dotenv()

TOOLS = {
 "predict_patient": predict_patient,
 "explain_patient": explain_patient_tool,
 "high_risk_patients": high_risk_patients,
 "readmission_factors": readmission_factors,
 "dataset_summary": dataset_summary,
}

def _patient_id(text):
    m = re.search(r"\bP\d{6}\b", text.upper()); return m.group(0) if m else None

def route_without_llm(question):
    q = question.lower(); pid = _patient_id(question)
    if pid and any(x in q for x in ["predict","risk","probability"]): return "predict_patient", {"patient_id":pid}
    if pid and any(x in q for x in ["why","explain","factor","driver"]): return "explain_patient", {"patient_id":pid}
    if any(x in q for x in ["top","high-risk","high risk"]): return "high_risk_patients", {"limit":10}
    if any(x in q for x in ["factor","associated","readmission rate"]): return "readmission_factors", {}
    return "dataset_summary", {}

def run_tool(name, args):
    result = TOOLS[name](**args)
    return {"tool":name,"arguments":args,"result":result}

def _audit(question, tool, status):
    try:
        db=SessionLocal(); db.add(AgentAuditLog(question=question, selected_tool=tool, response_status=status)); db.commit(); db.close()
    except Exception: pass

def agent_query(question):
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        tool,args=route_without_llm(question); out=run_tool(tool,args); _audit(question,tool,"success"); return out
    try:
        from groq import Groq
        client=Groq(api_key=api_key)
        tool_specs=[
          {"type":"function","function":{"name":"predict_patient","description":"Predict 30-day readmission risk for a synthetic patient","parameters":{"type":"object","properties":{"patient_id":{"type":"string"}},"required":["patient_id"]}}},
          {"type":"function","function":{"name":"explain_patient","description":"Explain the top ML feature contributions for a synthetic patient","parameters":{"type":"object","properties":{"patient_id":{"type":"string"}},"required":["patient_id"]}}},
          {"type":"function","function":{"name":"high_risk_patients","description":"Return top high-risk synthetic patients","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}},
          {"type":"function","function":{"name":"readmission_factors","description":"Return population readmission rates by major comorbidity","parameters":{"type":"object","properties":{}}}},
          {"type":"function","function":{"name":"dataset_summary","description":"Return dataset and target summary","parameters":{"type":"object","properties":{}}}}
        ]
        messages=[{"role":"system","content":"You are a healthcare analytics assistant. Data is synthetic. Never diagnose, prescribe, or claim clinical certainty. Use tools for every factual claim about the dataset/model."},{"role":"user","content":question}]
        first=client.chat.completions.create(model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),messages=messages,tools=tool_specs,tool_choice="auto",temperature=.1)
        msg=first.choices[0].message
        if not msg.tool_calls:
            _audit(question,"none","success"); return {"tool":"none","answer":msg.content}
        messages.append(msg)
        used=[]
        for call in msg.tool_calls:
            name=call.function.name; args=json.loads(call.function.arguments or "{}")
            if name not in TOOLS: continue
            result=TOOLS[name](**args); used.append(name)
            messages.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(result,default=str)})
        final=client.chat.completions.create(model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),messages=messages,temperature=.1)
        _audit(question,",".join(used),"success")
        return {"tools_used":used,"answer":final.choices[0].message.content}
    except Exception as exc:
        tool,args=route_without_llm(question); out=run_tool(tool,args); out["llm_fallback"]=str(exc); _audit(question,tool,"fallback"); return out
