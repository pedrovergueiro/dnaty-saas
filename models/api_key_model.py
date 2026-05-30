import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from models.database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # FK references users.email (existing PK), not a UUID — avoids migration on users table
    user_email = Column(String, ForeignKey("users.email", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, unique=True)
    key_prefix = Column(String(20), nullable=False)
    plan = Column(String(20), default="free", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    trainings_used = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)
    user_email = Column(String, ForeignKey("users.email"), nullable=True, index=True)
    dataset = Column(String(50), nullable=True)
    samples_used = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    hardware = Column(String(20), default="cpu")
    created_at = Column(DateTime, default=datetime.utcnow)
