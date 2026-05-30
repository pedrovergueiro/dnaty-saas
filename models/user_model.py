from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String

from models.database import Base


class User(Base):
    __tablename__ = "users"

    email                  = Column(String, primary_key=True, index=True)
    password_hash          = Column(String, nullable=True)
    name                   = Column(String, default="")
    subscription_active    = Column(Boolean, default=False, nullable=False)
    plan                   = Column(String(20), default="free", nullable=True)
    stripe_customer_id     = Column(String, default="")
    stripe_subscription_id = Column(String, default="", nullable=True)
    created_at             = Column(DateTime, default=datetime.utcnow, nullable=True)


class StripeSession(Base):
    __tablename__ = "stripe_sessions"

    session_id  = Column(String, primary_key=True)
    email       = Column(String, nullable=False, index=True)
    customer_id = Column(String, default="")
    used        = Column(Boolean, default=False, nullable=False)
