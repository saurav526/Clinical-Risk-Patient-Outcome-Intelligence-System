from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(32), nullable=True, index=True)
    probability = Column(Float, nullable=False)
    risk = Column(String(20), nullable=False)
    model_version = Column(String(40), nullable=False)
    request_source = Column(String(40), default="api")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class AgentAuditLog(Base):
    __tablename__ = "agent_audit_logs"
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    selected_tool = Column(String(80), nullable=False)
    response_status = Column(String(30), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)
