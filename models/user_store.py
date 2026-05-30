"""
Camada de acesso a dados — PostgreSQL via SQLAlchemy.
"""
import logging

from fastapi import HTTPException
import models.database as _db
from models.user_model import StripeSession, User

logger = logging.getLogger(__name__)


def _require_db():
    if _db.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured (DATABASE_URL missing)")


def _user_dict(u: User) -> dict:
    # Derive plan: DB column if set, else fallback from subscription_active
    plan = u.plan if u.plan else ("pro" if u.subscription_active else "free")
    return {
        "email": u.email,
        "password_hash": u.password_hash,
        "name": u.name or "",
        "subscription_active": u.subscription_active,
        "plan": plan,
        "stripe_customer_id": u.stripe_customer_id or "",
        "stripe_subscription_id": getattr(u, "stripe_subscription_id", "") or "",
    }


def _session_dict(s: StripeSession) -> dict:
    return {
        "email": s.email,
        "customer_id": s.customer_id or "",
        "used": s.used,
    }


# ── Users ──────────────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> dict | None:
    _require_db()
    with _db.SessionLocal() as db:
        u = db.get(User, email.lower())
        return _user_dict(u) if u else None


def create_user(
    email: str,
    password_hash: str,
    name: str = "",
    subscription_active: bool = False,
    stripe_customer_id: str = "",
) -> dict:
    _require_db()
    with _db.SessionLocal() as db:
        u = User(
            email=email.lower(),
            password_hash=password_hash,
            name=name,
            subscription_active=subscription_active,
            stripe_customer_id=stripe_customer_id,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return _user_dict(u)


def update_user(email: str, **fields) -> None:
    _require_db()
    with _db.SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            for k, v in fields.items():
                setattr(u, k, v)
            db.commit()


def activate_subscription(email: str, stripe_customer_id: str = "", plan: str = "pro") -> None:
    _require_db()
    with _db.SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            u.subscription_active = True
            u.plan = plan
            if stripe_customer_id:
                u.stripe_customer_id = stripe_customer_id
        else:
            u = User(
                email=email.lower(),
                password_hash=None,
                name="",
                subscription_active=True,
                plan=plan,
                stripe_customer_id=stripe_customer_id,
            )
            db.add(u)
        db.commit()


def deactivate_subscription(email: str) -> None:
    _require_db()
    with _db.SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            u.subscription_active = False
            u.plan = "free"
            db.commit()


def find_user_by_customer_id(stripe_customer_id: str) -> dict | None:
    _require_db()
    with _db.SessionLocal() as db:
        u = (
            db.query(User)
            .filter(User.stripe_customer_id == stripe_customer_id)
            .first()
        )
        return _user_dict(u) if u else None


# ── Stripe checkout sessions ────────────────────────────────────────────────────

def mark_session_paid(session_id: str, email: str, customer_id: str) -> None:
    _require_db()
    with _db.SessionLocal() as db:
        existing = db.get(StripeSession, session_id)
        if existing:
            existing.email = email.lower()
            existing.customer_id = customer_id
            existing.used = False
        else:
            db.add(StripeSession(
                session_id=session_id,
                email=email.lower(),
                customer_id=customer_id,
                used=False,
            ))
        db.commit()


def get_session(session_id: str) -> dict | None:
    _require_db()
    with _db.SessionLocal() as db:
        s = db.get(StripeSession, session_id)
        return _session_dict(s) if s else None


def consume_session(session_id: str) -> dict | None:
    _require_db()
    with _db.SessionLocal() as db:
        s = db.get(StripeSession, session_id)
        if s and not s.used:
            s.used = True
            db.commit()
            return _session_dict(s)
        return None
