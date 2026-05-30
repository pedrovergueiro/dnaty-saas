import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID

from models.database import Base


class DeviceFingerprint(Base):
    __tablename__ = "device_fingerprints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fingerprint_hash = Column(String(64), nullable=False, unique=True, index=True)
    # Nullable FK — SET NULL on user delete; also null before signup completes
    user_email = Column(String, ForeignKey("users.email", ondelete="SET NULL"), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class Ban(Base):
    __tablename__ = "bans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fingerprint_hash = Column(String(64), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    # No FK — bans survive user deletion and may be created before user exists
    user_email = Column(String, nullable=True, index=True)
    reason = Column(String(50), nullable=True)   # 'multi_account', 'ip_abuse', 'behavior'
    banned_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # NULL = permanent; set for ip_abuse (30 days)
