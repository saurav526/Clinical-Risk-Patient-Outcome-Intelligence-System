
import os
import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 5%, rgba(99,102,241,.12), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(16,185,129,.08), transparent 25%),
                #090b10;
        }

        [data-testid="stHeader"] {
            background: rgba(9,11,16,0.78);
        }

        [data-testid="stSidebar"] {
            background: #0d1017;
            border-right: 1px solid #222633;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 26px 30px;
            border: 1px solid #242938;
            border-radius: 22px;
            background: linear-gradient(135deg, #111521 0%, #0d1119 55%, #101722 100%);
            box-shadow: 0 15px 45px rgba(0,0,0,.24);
            margin-bottom: 18px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: -1px;
        }

        .hero p {
            color: #9aa3b2;
            margin-top: 8px;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .metric-card {
            background: linear-gradient(145deg, #121621, #0e1118);
            border: 1px solid #252a38;
            border-radius: 17px;
            padding: 18px 20px;
            min-height: 118px;
            box-shadow: 0 10px 30px rgba(0,0,0,.18);
        }

        .metric-label {
            color: #8e98a9;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .7px;
        }

        .metric-value {
            color: #f4f6fa;
            font-size: 2rem;
            font-weight: 750;
            margin-top: 7px;
        }

        .metric-sub {
            color: #727d8f;
            font-size: .78rem;
            margin-top: 3px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 20px 0 10px 0;
        }

        .risk-card {
            border-radius: 18px;
            padding: 22px;
            border: 1px solid #2a3040;
            background: #10141d;
            min-height: 155px;
        }

        .risk-number {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1;
            margin: 8px 0;
        }

        .risk-label {
            color: #9ba4b4;
            font-size: .9rem;
        }

        .pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: .75rem;
            font-weight: 700;
            background: #1b2230;
            color: #cbd5e1;
        }

        .info-box {
            padding: 14px 16px;
            border-radius: 13px;
            border: 1px solid #283044;
            background: #101520;
            color: #aeb7c6;
        }

        .small-muted {
            color: #7e8899;
            font-size: .82rem;
        }

        div[data-testid="stMetric"] {
            background: #11151f;
            border: 1px solid #252a38;
            padding: 14px 16px;
            border-radius: 15px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }

        .stTextArea textarea, .stTextInput input {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(path: str, timeout: int = 15):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:500]}"
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


def api_post(path: str, payload: Dict[str, Any], timeout: int = 60):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:500]}"
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


def first_value(obj: Any, *keys, default=None):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
    return default


def as_records(value: Any) -> List[Dict[str, Any]]:
    """Convert API list/dict responses into a list of dictionaries."""
    if value is None:
        return []

    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]

    if isinstance(value, dict):
        # Common API wrappers
        for key in ("patients", "results", "data", "items", "records", "drivers", "factors"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [x for x in candidate if isinstance(x, dict)]

        # A single object
        return [value]

    return []


def unwrap_result(value: Any):
    """Safely unwrap {'result': ...} without assuming the result type."""
    if isinstance(value, dict) and "result" in value:
        return value["result"]
    return value


def format_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=20, show_spinner=False)
def load_summary():
    data, err = api_get("/api/analytics/summary")
    return data, err


@st.cache_data(ttl=20, show_spinner=False)
def load_high_risk():
    data, err = api_get("/api/analytics/high-risk")
    return data, err


@st.cache_data(ttl=20, show_spinner=False)
def load_factors():
    data, err = api_get("/api/analytics/factors")
    return data, err


@st.cache_data(ttl=20, show_spinner=False)
def load_model():
    data, err = api_get("/api/model")
    return data, err


def get_summary_values(summary):
    summary = unwrap_result(summary)

    if not isinstance(summary, dict):
        return {
            "rows": 0,
            "features": 0,
            "readmission_rate": 0,
            "missing_cells": 0,
        }

    return {
        "rows": first_value(summary, "rows", "patients", "n_rows", default=0),
        "features": first_value(summary, "features", "n_features", default=0),
        "readmission_rate": first_value(
            summary, "readmission_rate", "target_rate", "positive_rate", default=0
        ),
        "missing_cells": first_value(
            summary, "missing_cells", "missing", "missing_values", default=0
        ),
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size:2.7rem;margin-bottom:3px;">🏥</div>
        <h1>Clinical Risk Intelligence</h1>
        <p>
            ML risk prediction &nbsp;•&nbsp; SHAP explainability
            &nbsp;•&nbsp; Agentic AI &nbsp;•&nbsp; Model monitoring
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ System")
    st.caption("Clinical Risk Intelligence v2.0")

    if st.button("🔄 Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### API")
    health, health_err = api_get("/api/health", timeout=5)

    if health_err:
        st.error("API offline")
        st.caption(health_err[:180])
    else:
        st.success("API online")
        if isinstance(health, dict):
            st.caption(
                f"Status: {health.get('status', 'healthy')}  "
                f"• Version: {health.get('version', '2.0.0')}"
            )

    st.markdown("---")
    st.markdown("### Architecture")
    st.caption("Data → ML → Explainability → Agent → API → Dashboard")
    st.caption("Dataset: 5,000 synthetic patient records")
    st.warning("Demo only — synthetic data is not clinically validated.")


# ============================================================
# TOP METRICS
# ============================================================

summary, summary_err = load_summary()
sv = get_summary_values(summary)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Patients</div>
            <div class="metric-value">{int(sv["rows"]):,}</div>
            <div class="metric-sub">Synthetic records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{int(sv["features"]):,}</div>
            <div class="metric-sub">Clinical + utilization features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Readmission rate</div>
            <div class="metric-value">{format_percent(sv["readmission_rate"])}</div>
            <div class="metric-sub">Target prevalence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Missing cells</div>
            <div class="metric-value">{int(sv["missing_cells"]):,}</div>
            <div class="metric-sub">Across dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if summary_err:
    st.info(f"Summary endpoint note: {summary_err}")


# ============================================================
# MAIN NAVIGATION
# ============================================================

tab_ai, tab_patient, tab_population = st.tabs(
    ["🤖 AI Analyst", "👤 Patient 360", "📊 Population Intelligence"]
)


# ============================================================
# AI ANALYST
# ============================================================

with tab_ai:
    st.markdown('<div class="section-title">AI Clinical Data Analyst</div>', unsafe_allow_html=True)
    st.caption(
        "Ask questions in natural language. The agent selects the appropriate "
        "prediction, explanation, patient lookup, or analytics tool."
    )

    examples = [
        "Predict readmission risk for patient P100001",
        "Explain the risk factors for patient P100001 using SHAP",
        "Show the top high-risk patients",
        "What factors are associated with readmission?",
        "Give me a summary of the dataset",
    ]

    selected_example = st.selectbox("Quick question", ["Custom"] + examples)

    default_question = (
        "Predict readmission risk for patient P100001"
        if selected_example == "Custom"
        else selected_example
    )

    question = st.text_area(
        "Ask the agent",
        value=default_question,
        height=90,
        placeholder="Example: Why is patient P100001 at risk?",
    )

    if st.button("🚀 Run analysis", type="primary", use_container_width=False):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Agent is analyzing the request..."):
                result, err = api_post("/api/query", {"question": question.strip()}, timeout=90)

            if err:
                st.error("Agent request failed")
                st.code(err)
            else:
                st.success("Analysis completed")

                # Show a clean answer if the API returned one.
                if isinstance(result, dict):
                    answer = first_value(
                        result,
                        "answer",
                        "response",
                        "message",
                        "output",
                        "final_answer",
                    )

                    if answer:
                        st.markdown("### 🧠 Agent answer")
                        st.markdown(
                            f'<div class="info-box">{str(answer).replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True,
                        )

                    tool = first_value(result, "tool", "selected_tool", "function")
                    if tool:
                        st.caption(f"Tool used: `{tool}`")

                    # The old dashboard showed raw JSON only. Keep it available
                    # under an expander, but do not make it the main result.
                    with st.expander("Technical response / tool trace"):
                        st.json(result)
                else:
                    st.write(result)


# ============================================================
# PATIENT 360
# ============================================================

with tab_patient:
    st.markdown('<div class="section-title">Patient 360</div>', unsafe_allow_html=True)
    st.caption(
        "Patient-level prediction and SHAP explanation from the deployed ML model."
    )

    p1, p2 = st.columns([3, 1])

    with p1:
        patient_id = st.text_input(
            "Patient ID",
            value="P100001",
            placeholder="Example: P100001",
        ).strip()

    with p2:
        analyze = st.button("🔎 Analyze patient", type="primary", use_container_width=True)

    if analyze:
        if not patient_id:
            st.warning("Enter a patient ID.")
        else:
            with st.spinner("Loading patient risk profile..."):
                risk_data, risk_err = api_get(
                    f"/api/patients/{patient_id}/risk", timeout=30
                )
                explain_data, explain_err = api_get(
                    f"/api/patients/{patient_id}/explanation", timeout=60
                )
                patient_data, patient_err = api_get(
                    f"/api/patients/{patient_id}", timeout=30
                )

            if risk_err:
                st.error(f"Risk prediction failed: {risk_err}")
            else:
                risk = unwrap_result(risk_data)
                if not isinstance(risk, dict):
                    risk = {}

                probability = first_value(
                    risk, "probability", "risk_probability", "score", default=0
                )
                risk_label = first_value(
                    risk, "risk", "risk_level", "label", default="Unknown"
                )
                prediction = first_value(risk, "prediction", "predicted_class")
                model_version = first_value(
                    risk, "model_version", "version", default="—"
                )

                try:
                    probability_float = float(probability)
                except Exception:
                    probability_float = 0.0

                # Risk header
                st.markdown("### Risk assessment")

                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    st.metric("Patient", patient_id)

                with r2:
                    st.metric("Risk level", str(risk_label))

                with r3:
                    st.metric("Probability", f"{probability_float * 100:.1f}%")

                with r4:
                    st.metric("Model", str(model_version))

                # Probability gauge
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability_float * 100,
                        number={"suffix": "%", "font": {"size": 30}},
                        title={"text": "Predicted readmission probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"thickness": 0.25},
                            "steps": [
                                {"range": [0, 30], "name": "Lower"},
                                {"range": [30, 70], "name": "Moderate"},
                                {"range": [70, 100], "name": "Higher"},
                            ],
                        },
                    )
                )
                gauge.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=55, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce2ec"),
                )

                g1, g2 = st.columns([1.2, 1])
                with g1:
                    st.plotly_chart(gauge, use_container_width=True)

                with g2:
                    st.markdown("#### Prediction details")
                    st.markdown(
                        f"""
                        <div class="risk-card">
                            <div class="risk-label">Predicted class</div>
                            <div class="risk-number">{prediction if prediction is not None else "—"}</div>
                            <div class="risk-label">Risk category: <b>{risk_label}</b></div>
                            <br>
                            <div class="small-muted">
                                This is a model output on synthetic data and should not
                                be interpreted as a clinical diagnosis.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Patient raw profile
                patient = unwrap_result(patient_data)
                if isinstance(patient, dict):
                    with st.expander("📋 Patient data"):
                        # Remove noisy wrapper fields if present
                        clean_patient = patient.copy()
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        str(k): (
                                            "—"
                                            if pd.isna(v)
                                            else v
                                        )
                                        for k, v in clean_patient.items()
                                    }
                                ]
                            ).T.rename(columns={0: "Value"}),
                            use_container_width=True,
                        )

                # SHAP
                st.markdown("### 🔬 SHAP model drivers")

                if explain_err:
                    st.error(f"Explanation failed: {explain_err}")
                else:
                    explanation = unwrap_result(explain_data)

                    if not isinstance(explanation, dict):
                        explanation = {}

                    drivers = explanation.get("drivers", [])

                    # Handle possible nested formats
                    if isinstance(drivers, dict):
                        drivers = as_records(drivers)

                    if not isinstance(drivers, list):
                        drivers = []

                    normalized = []

                    for item in drivers:
                        if not isinstance(item, dict):
                            continue

                        feature = first_value(
                            item,
                            "feature",
                            "name",
                            "feature_name",
                            default="Unknown",
                        )
                        contribution = first_value(
                            item,
                            "contribution",
                            "shap_value",
                            "value",
                            default=0,
                        )
                        direction = first_value(
                            item,
                            "direction",
                            default=None,
                        )

                        try:
                            contribution = float(contribution)
                        except Exception:
                            contribution = 0.0

                        # Make generated one-hot feature names readable.
                        readable = str(feature)
                        for prefix in (
                            "cat__",
                            "num__",
                            "onehot__",
                            "standardscaler__",
                        ):
                            readable = readable.replace(prefix, "")

                        normalized.append(
                            {
                                "feature": readable,
                                "contribution": contribution,
                                "direction": direction or (
                                    "increases risk"
                                    if contribution > 0
                                    else "decreases risk"
                                    if contribution < 0
                                    else "neutral"
                                ),
                            }
                        )

                    # Aggregate duplicate one-hot/original features.
                    if normalized:
                        df = pd.DataFrame(normalized)

                        def base_feature(name):
                            name = str(name)
                            if "_" in name:
                                # Only strip common encoded category suffixes.
                                for base in [
                                    "physical_activity",
                                    "smoking_status",
                                    "gender",
                                ]:
                                    if name.startswith(base + "_"):
                                        return base
                            return name

                        df["display_feature"] = df["feature"].apply(base_feature)

                        grouped = (
                            df.groupby("display_feature", as_index=False)["contribution"]
                            .sum()
                            .sort_values(
                                "contribution",
                                key=lambda s: s.abs(),
                                ascending=False,
                            )
                            .head(12)
                        )

                        grouped["direction"] = grouped["contribution"].apply(
                            lambda x: "Increases risk"
                            if x > 0
                            else "Decreases risk"
                            if x < 0
                            else "Neutral"
                        )

                        grouped["label"] = grouped["display_feature"].str.replace(
                            "_", " ", regex=False
                        ).str.title()

                        chart = px.bar(
                            grouped.sort_values("contribution"),
                            x="contribution",
                            y="label",
                            orientation="h",
                            text="contribution",
                            labels={
                                "contribution": "SHAP contribution",
                                "label": "Feature",
                            },
                        )

                        chart.update_traces(
                            texttemplate="%{text:.3f}",
                            textposition="outside",
                        )
                        chart.update_layout(
                            height=480,
                            margin=dict(l=20, r=50, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#dce2ec"),
                        )

                        st.plotly_chart(chart, use_container_width=True)

                        table = grouped[
                            ["label", "contribution", "direction"]
                        ].rename(
                            columns={
                                "label": "Feature",
                                "contribution": "SHAP contribution",
                                "direction": "Interpretation",
                            }
                        )
                        table["SHAP contribution"] = table[
                            "SHAP contribution"
                        ].round(4)

                        st.dataframe(
                            table,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.warning(
                            "The API returned an explanation, but no SHAP driver values "
                            "were available for this patient."
                        )

                        with st.expander("Raw explanation response"):
                            st.json(explain_data)


# ============================================================
# POPULATION INTELLIGENCE
# ============================================================

with tab_population:
    st.markdown('<div class="section-title">Population Intelligence</div>', unsafe_allow_html=True)
    st.caption(
        "Population-level view of high-risk synthetic patients and model-associated factors."
    )

    high_risk, high_risk_err = load_high_risk()
    factors, factors_err = load_factors()

    records = as_records(high_risk)

    # Important: /api/analytics/high-risk returns a LIST in the current API.
    # Never call .get() directly on high_risk.
    if high_risk_err:
        st.error(f"High-risk endpoint failed: {high_risk_err}")
    elif not records:
        st.info("No high-risk patient records were returned.")
    else:
        high_df = pd.DataFrame(records)

        # Normalize common column names.
        rename_map = {}
        for col in high_df.columns:
            low = str(col).lower()
            if low in ("patient_id", "patientid"):
                rename_map[col] = "Patient ID"
            elif low in ("probability", "risk_probability", "score"):
                rename_map[col] = "Probability"
            elif low in ("risk", "risk_level"):
                rename_map[col] = "Risk"
            elif low in ("prediction", "predicted_class"):
                rename_map[col] = "Prediction"

        high_df = high_df.rename(columns=rename_map)

        st.markdown("### 🚨 Highest-risk patients")

        display_df = high_df.copy()

        if "Probability" in display_df.columns:
            display_df["Probability"] = pd.to_numeric(
                display_df["Probability"], errors="coerce"
            )
            display_df["Probability"] = (
                display_df["Probability"] * 100
            ).round(1).astype(str) + "%"

        st.dataframe(
            display_df.head(15),
            use_container_width=True,
            hide_index=True,
        )

        if "Probability" in high_df.columns and "Patient ID" in high_df.columns:
            plot_df = high_df.copy()
            plot_df["Probability"] = pd.to_numeric(
                plot_df["Probability"], errors="coerce"
            )
            plot_df = plot_df.dropna(subset=["Probability"]).head(15)
            plot_df["Probability %"] = plot_df["Probability"] * 100

            fig = px.bar(
                plot_df.sort_values("Probability %"),
                x="Probability %",
                y="Patient ID",
                orientation="h",
                text="Probability %",
                labels={"Probability %": "Readmission probability"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                height=520,
                margin=dict(l=20, r=60, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dce2ec"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Readmission-associated factors")

    factor_records = as_records(factors)

    if factors_err:
        st.info(f"Factors endpoint note: {factors_err}")
    elif factor_records:
        factor_df = pd.DataFrame(factor_records)

        # Try to identify a useful numeric metric automatically.
        numeric_candidates = [
            c
            for c in factor_df.columns
            if pd.api.types.is_numeric_dtype(factor_df[c])
        ]

        if numeric_candidates:
            metric_col = numeric_candidates[0]

            label_candidates = [
                c for c in factor_df.columns
                if c not in numeric_candidates
            ]

            label_col = label_candidates[0] if label_candidates else factor_df.columns[0]

            chart_df = factor_df[[label_col, metric_col]].copy()
            chart_df.columns = ["Factor", "Value"]
            chart_df["Value"] = pd.to_numeric(
                chart_df["Value"], errors="coerce"
            )
            chart_df = chart_df.dropna().head(15)

            if not chart_df.empty:
                fig2 = px.bar(
                    chart_df.sort_values("Value"),
                    x="Value",
                    y="Factor",
                    orientation="h",
                    text="Value",
                )
                fig2.update_layout(
                    height=500,
                    margin=dict(l=20, r=50, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce2ec"),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(factor_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
    else:
        st.info("No factor analytics were returned by the API.")


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🧠 Model information"):
    model, model_err = load_model()

    if model_err:
        st.info(model_err)
    elif model:
        st.json(model)
    else:
        st.info("Model metadata is not available.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Clinical Risk Intelligence • ML + SHAP + Agentic AI • "
    "Synthetic dataset for demonstration only"
)
