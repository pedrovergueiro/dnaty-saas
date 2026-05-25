import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID

from models.database import Base


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_log"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action      = Column(String(100), nullable=False, index=True)
    target_type = Column(String(50), nullable=True)   # 'user', 'ban', 'license', 'auth'
    target_id   = Column(String(100), nullable=True)
    details     = Column(JSON, nullable=True)
    ip_address  = Column(String(45), nullable=True)
    user_agent  = Column(String(512), nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)


class AdminLoginAttempt(Base):
    __tablename__ = "admin_login_attempts"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ip_address  = Column(String(45), nullable=False, index=True)
    email_tried = Column(String(254), nullable=True)
    success     = Column(Boolean, default=False, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)


class CommercialLicense(Base):
    __tablename__ = "commercial_licenses"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company        = Column(String(200), nullable=False)
    contact_email  = Column(String(254), nullable=False)
    tier           = Column(String(50), default="standard")   # standard, enterprise
    value_usd      = Column(Float, nullable=False)
    start_date     = Column(DateTime, nullable=False)
    end_date       = Column(DateTime, nullable=False)
    notes          = Column(String(1000), nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
