# this file is used to store model information and metrics for tracking and registry purposes
from pathlib import Path
import json
from src.config import ROOT


REGISTRY = ROOT / "models/registry.json"

def register_model(name, metrics):
    payload = {"model_name": name, "version": "1.0.0", "metrics": metrics, "status": "production"}
    REGISTRY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

def get_registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
