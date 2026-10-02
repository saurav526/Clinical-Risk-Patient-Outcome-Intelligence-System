import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Clinical Risk Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Risk Intelligence")
st.caption(
    "Synthetic healthcare data • ML prediction • SHAP explainability • Agentic AI"
)


# ---------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------

def api_get(endpoint):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        return {
            "error": response.text,
            "status_code": response.status_code
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def api_post(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()

        return {
            "error": response.text,
            "status_code": response.status_code
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ---------------------------------------------------------
# API STATUS
# ---------------------------------------------------------

health = api_get("/api/health")

if "error" in health:

    st.error(
        "API is not reachable. Start FastAPI with:\n\n"
        "python -m uvicorn api.main:app --reload"
    )

else:

    st.success("API online • v2.0.0")


# ---------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------

summary = api_get("/api/analytics/summary")

if "error" not in summary:

    rows = summary.get("rows", 0)
    features = summary.get("features", 0)
    readmission_rate = summary.get(
        "readmission_rate",
        0
    )
    missing_cells = summary.get(
        "missing_cells",
        0
    )

else:

    rows = 0
    features = 0
    readmission_rate = 0
    missing_cells = 0


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patients",
        f"{rows:,}"
    )

with col2:
    st.metric(
        "Features",
        features
    )

with col3:
    st.metric(
        "Readmission rate",
        f"{readmission_rate * 100:.1f}%"
    )

with col4:
    st.metric(
        "Missing cells",
        missing_cells
    )


st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab_agent, tab_patient, tab_population = st.tabs(
    [
        "🤖 AI Analyst",
        "👤 Patient 360",
        "📊 Population"
    ]
)


# =========================================================
# AI ANALYST
# =========================================================

with tab_agent:

    st.subheader("Ask the agent")

    question = st.text_area(
        "Ask the agent",
        value="Predict readmission risk for patient P100001",
        height=100
    )

    if st.button(
        "Run analysis",
        type="primary",
        key="agent_button"
    ):

        result = api_post(
            "/api/query",
            {
                "question": question
            }
        )

        if "error" in result:

            st.error(
                result["error"]
            )

        else:

            # Show final answer if available
            answer = result.get(
                "answer"
            )

            if answer:
                st.markdown("### AI Analysis")
                st.info(answer)

            # Show tool execution
            tool = result.get(
                "tool"
            )

            if tool:

                st.markdown(
                    "### Tool used"
                )

                st.code(
                    tool
                )

            # Show complete structured result
            st.markdown(
                "### Detailed result"
            )

            st.json(result)


# =========================================================
# PATIENT 360
# =========================================================

with tab_patient:

    st.subheader("Patient 360")

    patient_id = st.text_input(
        "Patient ID",
        value="P100016",
        key="patient_id"
    )

    analyze = st.button(
        "Analyze patient",
        key="patient_analyze"
    )

    if analyze:

        # -------------------------------------------------
        # GET RISK
        # -------------------------------------------------

        risk = api_get(
            f"/api/patients/{patient_id}/risk"
        )

        # -------------------------------------------------
        # GET SHAP EXPLANATION
        # -------------------------------------------------

        explanation = api_get(
            f"/api/patients/{patient_id}/explanation"
        )

        if "error" in risk:

            st.error(
                risk["error"]
            )

        else:

            # -------------------------------------------------
            # PATIENT SUMMARY
            # -------------------------------------------------

            risk_value = risk.get(
                "risk",
                "Unknown"
            )

            probability = risk.get(
                "probability",
                0
            )

            model_version = risk.get(
                "model_version",
                "Unknown"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Risk",
                    risk_value
                )

            with c2:

                st.metric(
                    "Probability",
                    f"{probability * 100:.1f}%"
                )

            with c3:

                st.metric(
                    "Model",
                    model_version
                )

            st.divider()

            # -------------------------------------------------
            # SHAP
            # -------------------------------------------------

            st.subheader(
                "🔍 SHAP model drivers"
            )

            if "error" in explanation:

                st.error(
                    explanation["error"]
                )

            else:

                drivers = explanation.get(
                    "drivers",
                    []
                )

                if not drivers:

                    st.warning(
                        "No SHAP drivers were returned for this patient."
                    )

                else:

                    drivers_df = pd.DataFrame(
                        drivers
                    )

                    # Make sure contribution is numeric
                    drivers_df["contribution"] = pd.to_numeric(
                        drivers_df["contribution"],
                        errors="coerce"
                    )

                    drivers_df = drivers_df.dropna(
                        subset=["contribution"]
                    )

                    # Sort for horizontal chart
                    drivers_df = drivers_df.sort_values(
                        "contribution"
                    )

                    # -----------------------------------------
                    # SHAP BAR CHART
                    # -----------------------------------------

                    fig = px.bar(
                        drivers_df,
                        x="contribution",
                        y="feature",
                        orientation="h",
                        text="contribution",
                        title="Top factors influencing model prediction"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition="outside"
                    )

                    fig.update_layout(
                        height=max(
                            450,
                            len(drivers_df) * 55
                        ),
                        xaxis_title="Model contribution",
                        yaxis_title="Feature",
                        margin=dict(
                            l=20,
                            r=80,
                            t=70,
                            b=40
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # -----------------------------------------
                    # SHAP TABLE
                    # -----------------------------------------

                    st.subheader(
                        "Feature contribution details"
                    )

                    display_df = drivers_df.copy()

                    display_df[
                        "contribution"
                    ] = display_df[
                        "contribution"
                    ].map(
                        lambda x: f"{x:+.4f}"
                    )

                    display_df = display_df.rename(
                        columns={
                            "feature": "Feature",
                            "contribution": "Contribution",
                            "direction": "Effect"
                        }
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Positive values increase the model's "
                        "predicted risk score; negative values "
                        "decrease it. This demonstration uses "
                        "synthetic data and is not a clinical diagnosis."
                    )

            # -------------------------------------------------
            # PATIENT DATA
            # -------------------------------------------------

            patient = api_get(
                f"/api/patients/{patient_id}"
            )

            if "error" not in patient:

                st.divider()

                st.subheader(
                    "Patient information"
                )

                patient_result = patient.get(
                    "patient",
                    patient
                )

                if isinstance(
                    patient_result,
                    dict
                ):

                    patient_df = pd.DataFrame(
                        [
                            patient_result
                        ]
                    )

                    st.dataframe(
                        patient_df,
                        use_container_width=True,
                        hide_index=True
                    )


# =========================================================
# POPULATION
# =========================================================

with tab_population:

    st.subheader(
        "Population Analytics"
    )

    # -----------------------------------------------------
    # HIGH-RISK PATIENTS
    # -----------------------------------------------------

    high_risk = api_get(
        "/api/analytics/high-risk"
    )

    if "error" not in high_risk:

        patients = high_risk.get(
            "patients",
            high_risk
        )

        if isinstance(
            patients,
            list
        ) and patients:

            st.markdown(
                "### High-risk patients"
            )

            high_risk_df = pd.DataFrame(
                patients
            )

            st.dataframe(
                high_risk_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No high-risk patients returned."
            )

    else:

        st.error(
            high_risk["error"]
        )

    st.divider()

    # -----------------------------------------------------
    # READMISSION FACTORS
    # -----------------------------------------------------

    factors = api_get(
        "/api/analytics/factors"
    )

    if "error" not in factors:

        factor_data = factors.get(
            "factors",
            factors
        )

        if isinstance(
            factor_data,
            list
        ) and factor_data:

            st.markdown(
                "### Readmission factors"
            )

            factor_df = pd.DataFrame(
                factor_data
            )

            st.dataframe(
                factor_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No factor analysis returned."
            )

    else:

        st.error(
            factors["error"]
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Clinical Risk Intelligence v2.0 • "
    "ML + SHAP + Agentic AI • "
    "Synthetic healthcare dataset"
)