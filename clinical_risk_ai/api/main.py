from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None
from src.db import init_db
from api.routes import router

app=FastAPI(title="Clinical Risk Intelligence API",version="2.0.0",description="Production-style ML + SHAP + Agentic AI API using synthetic healthcare data.")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router,prefix="/api")
if Instrumentator:
    Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def startup(): init_db()


@app.get("/")
def root(): return {"service":"clinical-risk-ai","version":"2.0.0","docs":"/docs"}
