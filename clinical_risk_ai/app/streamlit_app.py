import os, requests, pandas as pd, streamlit as st, plotly.express as px
from dotenv import load_dotenv
load_dotenv()
API=os.getenv("API_URL","http://127.0.0.1:8000"); KEY=os.getenv("API_KEY","change-me-in-production")
HEAD={"X-API-Key":KEY}
st.set_page_config(page_title="Clinical Risk AI",page_icon="🏥",layout="wide")
st.markdown("# 🏥 Clinical Risk Intelligence")
st.caption("Synthetic healthcare data • ML prediction • SHAP explainability • Agentic AI")
try:
    health=requests.get(f"{API}/api/health",timeout=3).json()
    st.success(f"API online · v{health['version']}")
except Exception: st.error("API offline — start FastAPI first.")

summary=None
try: summary=requests.get(f"{API}/api/analytics/summary",timeout=5).json()
except Exception: pass
if summary:
    a,b,c,d=st.columns(4); a.metric("Patients",f"{summary['rows']:,}"); b.metric("Features",summary['features']); c.metric("Readmission rate",f"{summary['readmission_rate']*100:.1f}%"); d.metric("Missing cells",f"{summary['missing_cells']:,}")

t1,t2,t3=st.tabs(["🤖 AI Analyst","👤 Patient 360","📊 Population"])
with t1:
    q=st.text_area("Ask the agent",placeholder="Why is patient P100123 high risk?",height=100)
    if st.button("Run analysis",type="primary") and q:
        r=requests.post(f"{API}/api/query",json={"question":q},headers=HEAD,timeout=90)
        if r.ok:
            data=r.json();
            if "answer" in data: st.markdown(data["answer"])
            st.json(data)
        else: st.error(r.text)
with t2:
    pid=st.text_input("Patient ID",value="P100000")
    if st.button("Analyze patient"):
        risk=requests.get(f"{API}/api/patients/{pid}/risk",headers=HEAD).json()
        exp=requests.get(f"{API}/api/patients/{pid}/explanation",headers=HEAD).json()
        x,y,z=st.columns(3); x.metric("Risk",risk.get("risk")); y.metric("Probability",f"{risk.get('probability',0)*100:.1f}%"); z.metric("Model",risk.get("model_version"))
        drivers=pd.DataFrame(exp.get("drivers",[]));
        if not drivers.empty: st.plotly_chart(px.bar(drivers.sort_values("contribution"),x="contribution",y="feature",orientation="h",title="SHAP model drivers"),use_container_width=True)
with t3:
    if st.button("Refresh population"):
        high=pd.DataFrame(requests.get(f"{API}/api/analytics/high-risk?limit=20").json())
        factors=requests.get(f"{API}/api/analytics/factors").json()
        st.dataframe(high,use_container_width=True)
        rows=[]
        for key in ["by_diabetes","by_hypertension","by_heart_disease","by_ckd"]:
            for group,rate in factors[key].items(): rows.append({"factor":key[3:],"group":str(group),"rate":rate})
        st.plotly_chart(px.bar(pd.DataFrame(rows),x="factor",y="rate",color="group",barmode="group",title="Synthetic readmission rates by comorbidity"),use_container_width=True)
