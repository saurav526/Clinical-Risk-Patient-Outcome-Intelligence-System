# Clinical Risk & Patient Outcome Intelligence System

Predicts 30-day readmission risk on a fully synthetic 5,000-row dataset.
Educational/engineering demo only. Never present as clinical decision support.

## Commands (run inside clinical_risk_ai/)
- Setup: `pip install -r requirements.txt`, then copy .env.example to .env
- Train: `python -m src.models.train`
- Test: `pytest -q`
- API: `uvicorn api.main:app --reload` (docs at /docs, Prometheus at /metrics)
- Dashboard: `streamlit run app/streamlit_app.py`
- Docker: `docker compose up --build`

## Architecture
- src/data: validation + repository; src/features: engineering
- src/models: train (3 candidates, best by ROC-AUC), predict, explain (SHAP), registry
- agent/: Groq tool-calling agent with deterministic fallback; tools = predict_patient,
  explain_patient, high_risk_patients, readmission_factors, dataset_summary
- api/: FastAPI with API-key auth; src/db.py: SQLAlchemy audit logs (SQLite or PostgreSQL)
- MLflow for experiment tracking

## Conventions
- The LLM never produces predictions itself; it calls tools backed by the trained model.
- Must work without GROQ_API_KEY (fallback agent).
- Never read or print .env or API keys.
- Keep the synthetic-data disclaimer in README, API docs, and dashboard.
- Add tests in tests/ for any change to src/, agent/, or api/.
