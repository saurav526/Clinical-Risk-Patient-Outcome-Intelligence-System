from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

APP_ENV = os.getenv("APP_ENV", "development")
API_KEY = os.getenv("API_KEY", "change-me-in-production")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'clinical_risk.db'}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
