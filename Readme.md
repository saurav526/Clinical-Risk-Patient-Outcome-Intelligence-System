# Clinical Risk & Patient Outcome Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end MLOps project that predicts **30-day hospital readmission risk**
from synthetic patient data, explains each prediction with SHAP, and exposes the
model through a secured API, an LLM tool-calling agent, and an interactive dashboard.

> ⚠️ The dataset is fully synthetic. This is an engineering/education demo and
> must not be used for diagnosis or clinical decisions.

![Dashboard](docs/dashboard.png)

## Highlights

- 3 candidate models (Logistic Regression, Random Forest, Gradient Boosting) with automatic ROC-AUC selection
- SHAP explanations per patient
- MLflow experiment tracking and a versioned model registry
- FastAPI with API-key protection, Prometheus `/metrics`, and SQLAlchemy audit logs
- Groq function-calling agent with a deterministic fallback
- Docker Compose setup and unit/API/tool tests

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 98.42 |
| Random Forest | 99.89 |
| Gradient Boosting | 99.88 |

## Quick start

```bash
cd clinical_risk_ai
python -m venv .venv
.venv\Scripts\activate        # Windows (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env
python -m src.models.train
uvicorn api.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Core use case

Predict **30-day hospital readmission risk** from synthetic patient demographics, vitals, laboratory measurements, comorbidities and utilization history.

## What was upgraded

- 5,000-row synthetic clinical dataset with realistic correlations and missing values having 30 features
- Candidate model training: Logistic Regression, Random Forest, Gradient Boosting
- Automatic model selection using ROC-AUC
- Model version + registry metadata
- MLflow experiment logging
- Real SHAP explanations for tree/linear estimators
- FastAPI API with API-key protection on inference/agent endpoints
- SQLAlchemy persistence (SQLite by default; PostgreSQL supported through `DATABASE_URL`)
- Prediction audit logs
- Agent audit logs
- Real Groq function/tool calling when `GROQ_API_KEY` is configured
- Deterministic agent fallback when an LLM is unavailable
- Population analytics tools
- Interactive Plotly Streamlit dashboard
- Prometheus metrics endpoint
- Docker support
- Unit/API/tool tests

## Repository

```text
clinical_risk_ai/
├── data/
│   ├── raw/clinical_risk_5000.csv
│   └── DATA_DICTIONARY.md
├── notebooks/01_eda.ipynb
├── src/
│   ├── config.py
│   ├── db.py
│   ├── data/{validation.py,repository.py}
│   ├── features/engineering.py
│   └── models/{train.py,predict.py,explain.py,registry.py}
├── agent/{agent.py,tools.py}
├── api/{main.py,routes.py,schemas.py}
├── app/streamlit_app.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
```

## Train the model

```bash
python -m src.models.train
```

The training process evaluates three algorithms and saves the best model to `models/model.joblib`, metrics, feature names and registry metadata.

## Start the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Metrics are exposed at `/metrics`.

## Start dashboard

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

## Agentic AI

Set:

```env
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

The LLM can select from tools including:

- `predict_patient`
- `explain_patient`
- `high_risk_patients`
- `readmission_factors`
- `dataset_summary`

The agent does not generate the ML prediction itself. It calls the trained model/tool and then uses the LLM to communicate the result.

## Example questions

```text
Predict readmission risk for patient P100123
Why is patient P100123 high risk?
Show the top high-risk patients
What factors are associated with readmission?
Give me a summary of the dataset
```

## PostgreSQL

The project defaults to SQLite so it runs immediately. For PostgreSQL, set:

```env
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/clinical_risk
```

The database stores prediction and agent audit events; the supplied synthetic CSV remains the source dataset.

## Tests

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

## MLOps lifecycle

```text
Data → Validation → Feature Engineering → Training
     → Evaluation → MLflow → Model Registry
     → FastAPI → Agent Tools → LLM → Dashboard
     → Prediction/Audit Monitoring
```

## Healthcare safety

This is not a clinical decision-support system. The data is synthetic and the prediction has no medical validity. A real deployment would require governance, external validation, privacy controls, clinical review, bias analysis, monitoring, and regulatory compliance.

## Feedback and support
Provide suitable feedback for improvement and please star the repo
