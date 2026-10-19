---
description: Start the API and smoke-test it
---
From clinical_risk_ai/ start `uvicorn api.main:app --reload` in the background.
Check /docs and /metrics, then call a prediction endpoint using the API key
from the environment (never print the key). Report status codes, then stop the server.
