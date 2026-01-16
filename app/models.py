from sqlalchemy import Column, Integer, String, JSON, DateTime
from .database import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    inputs = Column(JSON)
    prediction = Column(JSON)
    model_version = Column(String, default="v1")
