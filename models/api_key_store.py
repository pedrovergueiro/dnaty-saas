"""
Camada de acesso a dados para API keys e usage logs.

Design: trainings_used é incrementado atomicamente em validate_and_reserve_training
(SELECT FOR UPDATE), não em log_usage — elimina race condition TOCTOU.
"""
import hashlib
import logging
import secrets
import uuid
from datetime import datetime

from fastapi import HTTPException

from models.api_key_model import ApiKey, UsageLog
import models.database as _db

logger = logging.getLogger(__name__)

PLAN_LIMITS: dict[str, dict] = {
    "free": {
        "trainings_limit": 1,
        "samples_limit": 1_000,
        "allowed_datasets": ["MNIST"],
        "can_export_model": False,
    },
    "pro": {
        "trainings_limit": None,
        "samples_limit": 100_000,
        "allowed_datasets": None,
        "can_export_model": True,
    },
    "enterprise": {
        "trainings_limit": None,
        "samples_limit": None,
        "allowed_datasets": None,
        "can_export_model": True,
    },
}


def _require_db() -> None:
    if _db.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured (DATABASE_URL missing)")


def _make_key() -> tuple[str, str, str]:
    """Returns (full_key, sha256_hash, display_prefix)."""
    raw = secrets.token_hex(32)
    full_key = f"dnaty_sk_{raw}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:16]  # "dnaty_sk_" + 7 hex chars
    return full_key, key_hash, key_prefix


def _plan_info(plan: str) -> dict:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


def _key_dict(k: ApiKey) -> dict:
    limits = _plan_info(k.plan)
    tl = limits["trainings_limit"]
    return {
        "id": str(k.id),
        "user_email": k.user_email,
        "key_prefix": k.key_prefix,
        "plan": k.plan,
        "is_active": k.is_active,
        "trainings_used": k.trainings_used,
        "trainings_limit": tl,
        "samples_limit": limits["samples_limit"],
        "can_train": tl is None or k.trainings_used < tl,
        "allowed_datasets": limits["allowed_datasets"],
        "can_export_model": limits["can_export_model"],
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
    }


# ── Create ─────────────────────────────────────────────────────────────────────

def create_api_key(user_email: str, plan: str = "free") -> tuple[str, dict]:
    """Create a fresh api key. Returns (full_key, key_info)."""
    _require_db()
    full_key, key_hash, key_prefix = _make_key()
    with _db.SessionLocal() as db:
        k = ApiKey(
            id=uuid.uuid4(),
            user_email=user_email.lower(),
            key_hash=key_hash,
            key_prefix=key_prefix,
            plan=plan,
        )
        db.add(k)
        db.commit()
        db.refresh(k)
        return full_key, _key_dict(k)


# ── Read ───────────────────────────────────────────────────────────────────────

def get_key_by_email(user_email: str) -> dict | None:
    """Get active key info for a user (no plain key — only prefix)."""
    _require_db()
    with _db.SessionLocal() as db:
        k = (
            db.query(ApiKey)
            .filter(ApiKey.user_email == user_email.lower(), ApiKey.is_active == True)
            .first()
        )
        return _key_dict(k) if k else None


def get_key_by_raw(full_key: str) -> dict | None:
    """Lookup active key by raw value. Updates last_used_at."""
    _require_db()
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    with _db.SessionLocal() as db:
        k = (
            db.query(ApiKey)
            .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            .first()
        )
        if not k:
            return None
        k.last_used_at = datetime.utcnow()
        db.commit()
        db.refresh(k)
        return _key_dict(k)


# ── Regenerate ─────────────────────────────────────────────────────────────────

def regenerate_api_key(user_email: str, plan: str = "free") -> tuple[str, str]:
    """Deactivate all existing keys and create a new one. Returns (full_key, prefix)."""
    _require_db()
    full_key, key_hash, key_prefix = _make_key()
    with _db.SessionLocal() as db:
        db.query(ApiKey).filter(ApiKey.user_email == user_email.lower()).update(
            {"is_active": False}
        )
        k = ApiKey(
            id=uuid.uuid4(),
            user_email=user_email.lower(),
            key_hash=key_hash,
            key_prefix=key_prefix,
            plan=plan,
        )
        db.add(k)
        db.commit()
    return full_key, key_prefix


# ── Validate + reserve (atomic) ────────────────────────────────────────────────

def validate_and_reserve(full_key: str, dataset: str, samples: int) -> dict:
    """
    Validate key + check limits + atomically reserve training slot.
    Uses SELECT FOR UPDATE to prevent TOCTOU race conditions.
    Increments trainings_used on success (slot consumed even if training later fails).
    """
    _require_db()
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    with _db.SessionLocal() as db:
        k = (
            db.query(ApiKey)
            .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            .with_for_update()
            .first()
        )
        if not k:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        limits = _plan_info(k.plan)
        tl = limits["trainings_limit"]
        sl = limits["samples_limit"]
        allowed = limits["allowed_datasets"]

        if tl is not None and k.trainings_used >= tl:
            return {
                "allowed": False,
                "reason": "limit_reached",
                "samples_limit": sl,
                "effective_samples": 0,
                "upgrade_url": "https://dnaty.co/pricing",
            }

        if allowed and dataset.upper() not in [d.upper() for d in allowed]:
            return {
                "allowed": False,
                "reason": "dataset_not_allowed",
                "samples_limit": sl,
                "effective_samples": 0,
                "upgrade_url": "https://dnaty.co/pricing",
            }

        effective = min(samples, sl) if sl is not None else samples

        k.trainings_used += 1
        k.last_used_at = datetime.utcnow()
        db.commit()

        return {
            "allowed": True,
            "reason": "ok",
            "samples_limit": sl,
            "effective_samples": effective,
        }


# ── Log usage ──────────────────────────────────────────────────────────────────

def log_usage(
    full_key: str,
    dataset: str,
    samples_used: int,
    duration_seconds: float,
    accuracy: float,
    hardware: str,
) -> dict:
    """Record completed training. Counter already incremented in validate_and_reserve."""
    _require_db()
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    with _db.SessionLocal() as db:
        k = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if not k:
            raise HTTPException(status_code=401, detail="Invalid API key")

        tl = _plan_info(k.plan)["trainings_limit"]
        db.add(UsageLog(
            id=uuid.uuid4(),
            api_key_id=k.id,
            user_email=k.user_email,
            dataset=dataset,
            samples_used=samples_used,
            duration_seconds=duration_seconds,
            accuracy=accuracy,
            hardware=hardware,
        ))
        db.commit()

        remaining = None if tl is None else max(0, tl - k.trainings_used)
        return {
            "logged": True,
            "trainings_used": k.trainings_used,
            "trainings_remaining": remaining,
        }


# ── Plan management ────────────────────────────────────────────────────────────

def upgrade_plan(user_email: str, plan: str = "pro") -> None:
    """Set plan on active key. Creates a key if none exists (pre-paid webhook scenario)."""
    _require_db()
    with _db.SessionLocal() as db:
        k = (
            db.query(ApiKey)
            .filter(ApiKey.user_email == user_email.lower(), ApiKey.is_active == True)
            .first()
        )
        if k:
            k.plan = plan
        else:
            _, key_hash, key_prefix = _make_key()
            db.add(ApiKey(
                id=uuid.uuid4(),
                user_email=user_email.lower(),
                key_hash=key_hash,
                key_prefix=key_prefix,
                plan=plan,
            ))
        db.commit()


def downgrade_plan(user_email: str) -> None:
    """Reset plan to free (subscription cancelled)."""
    _require_db()
    with _db.SessionLocal() as db:
        k = (
            db.query(ApiKey)
            .filter(ApiKey.user_email == user_email.lower(), ApiKey.is_active == True)
            .first()
        )
        if k:
            k.plan = "free"
            db.commit()
