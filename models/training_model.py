import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from models.database import Base


class Training(Base):
    __tablename__ = "trainings"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id           = Column(String(100), unique=True, nullable=False, index=True)
    user_email       = Column(String, nullable=False, index=True)
    dataset          = Column(String(50), nullable=False)
    epochs_requested = Column(Integer, default=10)
    samples_requested= Column(Integer, default=1000)
    samples_used     = Column(Integer, default=1000)
    batch_size       = Column(Integer, default=512)
    status           = Column(String(20), default="queued", nullable=False)
    progress         = Column(Integer, default=0)   # 0–100
    current_epoch    = Column(Integer, default=0)
    loss             = Column(Float, nullable=True)
    accuracy         = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    error_message    = Column(Text, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at     = Column(DateTime, nullable=True)
