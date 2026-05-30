"""
API Key management + usage endpoints.

Auth pattern:
  - Bearer JWT  → endpoints voltados ao dashboard (my-key, regenerate, summary)
  - X-API-Key   → endpoints chamados pelo .exe (validate, validate-training, usage/log)
"""
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.api_key_store import (
    PLAN_LIMITS,
    create_api_key,
    downgrade_plan,
    get_key_by_email,
    get_key_by_raw,
    log_usage,
    regenerate_api_key,
    upgrade_plan,
    validate_and_reserve,
)
from models.user_store import get_user_by_email
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_raw_key(x_api_key: str | None) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    return x_api_key


# ── Dashboard endpoints (Bearer JWT) ───────────────────────────────────────────

@router.get("/keys/my-key", summary="Get active API key info (no plain key)")
async def my_key(current_user: dict = Depends(get_current_user)):
    email = current_user["sub"]
    info = get_key_by_email(email)
    if not info:
        raise HTTPException(
            status_code=404,
            detail="No API key found. Call POST /api/v1/keys/regenerate to generate one.",
        )
    return {
        "key_prefix": info["key_prefix"],
        "plan": info["plan"],
        "trainings_used": info["trainings_used"],
        "trainings_limit": info["trainings_limit"],
        "samples_limit": info["samples_limit"],
        "can_train": info["can_train"],
        "can_export_model": info["can_export_model"],
        "created_at": info["created_at"],
        "last_used_at": info["last_used_at"],
    }


@router.post("/keys/regenerate", summary="Invalidate old key and issue a new one")
async def regenerate(current_user: dict = Depends(get_current_user)):
    email = current_user["sub"]

    # Carry over existing plan (pro users keep pro after regeneration)
    existing = get_key_by_email(email)
    if existing:
        plan = existing["plan"]
    else:
        # Fallback: check subscription status from user table
        user = get_user_by_email(email)
        plan = "pro" if user and user.get("subscription_active") else "free"

    full_key, prefix = regenerate_api_key(email, plan)
    logger.info("Key regenerated for %s (plan=%s)", email, plan)
    return {
        "new_key": full_key,
        "key_prefix": prefix,
        "message": "Old key invalidated. Save this key — it won't be shown again.",
    }


@router.get("/usage/summary", summary="Usage summary for the authenticated user")
async def usage_summary(current_user: dict = Depends(get_current_user)):
    email = current_user["sub"]
    info = get_key_by_email(email)
    if not info:
        return {
            "used": 0,
            "limit": PLAN_LIMITS["free"]["trainings_limit"],
            "plan": "free",
            "can_train": True,
            "reset_date": None,
        }
    tl = info["trainings_limit"]
    return {
        "used": info["trainings_used"],
        "limit": tl,
        "plan": info["plan"],
        "can_train": info["can_train"],
        "reset_date": None,
    }


# ── .exe endpoints (X-API-Key) ─────────────────────────────────────────────────

_SILENT_BAN_RESPONSE = JSONResponse(
    status_code=200,
    content={"valid": False, "error": "Server error"},
)

@router.get("/keys/validate", summary="Validate API key (called by .exe on startup)")
async def validate_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    x_device_fp: str | None = Header(None, alias="X-Device-FP"),
):
    raw = _resolve_raw_key(x_api_key)
    info = get_key_by_raw(raw)
    if not info:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    # Abuse check: fp ban first, then fall back to email ban when no fp header
    from models.abuse_store import check_validate_abuse
    if not check_validate_abuse(x_device_fp, info["user_email"]):
        return _SILENT_BAN_RESPONSE

    return {
        "valid": True,
        "user_email": info["user_email"],
        "plan": info["plan"],
        "trainings_used": info["trainings_used"],
        "trainings_limit": info["trainings_limit"],
        "samples_limit": info["samples_limit"],
        "can_train": info["can_train"],
        "allowed_datasets": info["allowed_datasets"],
        "can_export_model": info["can_export_model"],
    }


class ValidateTrainingRequest(BaseModel):
    dataset: str
    samples: int


@router.post("/keys/validate-training", summary="Pre-flight check before training (reserves slot)")
async def validate_training(
    body: ValidateTrainingRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    raw = _resolve_raw_key(x_api_key)
    result = validate_and_reserve(raw, body.dataset, body.samples)
    if not result["allowed"]:
        logger.info(
            "Training denied — reason=%s dataset=%s", result["reason"], body.dataset
        )
    else:
        logger.info(
            "Training slot reserved — dataset=%s effective_samples=%d",
            body.dataset,
            result["effective_samples"],
        )
    return result


class UsageLogRequest(BaseModel):
    dataset: str
    samples_used: int
    duration_seconds: float
    accuracy: float
    hardware: str = "cpu"


@router.post("/usage/log", summary="Log completed training (counter already reserved)")
async def log_training(
    body: UsageLogRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    raw = _resolve_raw_key(x_api_key)
    result = log_usage(
        raw,
        body.dataset,
        body.samples_used,
        body.duration_seconds,
        body.accuracy,
        body.hardware,
    )
    logger.info(
        "Usage logged — dataset=%s samples=%d accuracy=%.4f hardware=%s",
        body.dataset,
        body.samples_used,
        body.accuracy,
        body.hardware,
    )
    return result
