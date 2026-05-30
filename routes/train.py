"""
Training routes — POST /train, GET /train/{job_id}, GET /train/history.
Auth: Bearer JWT (dashboard) OR X-API-Key header (Pro/Enterprise direct API).
Plan limits enforced here before any job is created.
"""
import hashlib
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config import settings
from models.dnaty_model import _jobs, create_job, start_training
from models.training_store import (
    count_trainings_today,
    create_training,
    get_training,
    list_user_trainings,
    update_training,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_bearer = HTTPBearer(auto_error=False)

# ── Plan limits ────────────────────────────────────────────────────────────────

PLAN_LIMITS: dict[str, dict] = {
    "free": {
        "samples_per_training": 1_000,
        "trainings_per_day": 1,
        "allowed_datasets": ["MNIST"],
        "can_export": False,
        "has_api_key": False,
    },
    "pro": {
        "samples_per_training": 100_000,
        "trainings_per_day": None,   # unlimited
        "allowed_datasets": ["MNIST", "FashionMNIST", "CIFAR10"],
        "can_export": True,
        "has_api_key": True,
    },
    "enterprise": {
        "samples_per_training": 1_000_000,
        "trainings_per_day": None,
        "allowed_datasets": None,    # all
        "can_export": True,
        "has_api_key": True,
    },
}


def get_plan_limits(plan: str) -> dict:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


# ── Auth: JWT Bearer OR X-API-Key ──────────────────────────────────────────────

def _resolve_user(
    credentials: HTTPAuthorizationCredentials | None,
    x_api_key: str | None,
) -> dict:
    """Returns {email, plan}. Raises 401 if neither auth method works."""

    # 1. Try JWT Bearer
    if credentials:
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
            )
            email = payload.get("sub")
            if email:
                # Re-read plan from DB to get live value
                from models.user_store import get_user_by_email
                u = get_user_by_email(email)
                plan = (u or {}).get("plan") or ("pro" if payload.get("subscription_active") else "free")
                return {"email": email, "plan": plan}
        except JWTError:
            pass

    # 2. Try API key
    if x_api_key:
        import models.database as _db
        from models.api_key_model import ApiKey
        if _db.SessionLocal:
            key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
            with _db.SessionLocal() as db:
                k = db.query(ApiKey).filter(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active == True,
                ).first()
                if k:
                    return {"email": k.user_email, "plan": k.plan}

    raise HTTPException(status_code=401, detail="Authentication required (Bearer token or X-API-Key)")


def get_auth_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    return _resolve_user(credentials, x_api_key)


# ── Request / Response schemas ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset:    str = "MNIST"
    epochs:     int = 10
    samples:    int = 1_000
    batch_size: int = 512


class TrainStartResponse(BaseModel):
    job_id:      str
    status:      str
    dataset:     str
    samples_used:int
    epochs:      int
    message:     str


# ── POST /train ────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainStartResponse, status_code=202)
async def start_train(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_auth_user),
) -> TrainStartResponse:
    plan = user["plan"]
    email = user["email"]
    limits = get_plan_limits(plan)

    # Dataset check
    allowed = limits["allowed_datasets"]
    dataset_norm = req.dataset.upper().replace("-", "").replace("_", "")
    dataset_map = {
        "MNIST": "MNIST",
        "FASHIONMNIST": "FashionMNIST",
        "CIFAR10": "CIFAR10",
    }
    canonical = dataset_map.get(dataset_norm, req.dataset)
    if allowed is not None:
        if canonical not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Dataset '{canonical}' not available on {plan} plan. "
                       f"Allowed: {', '.join(allowed)}. Upgrade at dnaty.co/pricing",
            )

    # Daily limit check (free only)
    tpd = limits["trainings_per_day"]
    if tpd is not None:
        count = count_trainings_today(email)
        if count >= tpd:
            raise HTTPException(
                status_code=429,
                detail=f"Daily training limit reached ({tpd}/day on free plan). "
                       f"Upgrade to Pro for unlimited trainings: dnaty.co/pricing",
            )

    # Cap samples to plan limit
    sl = limits["samples_per_training"]
    effective_samples = min(req.samples, sl) if sl is not None else req.samples
    effective_samples = max(effective_samples, 100)

    # Clamp epochs
    epochs = min(max(req.epochs, 1), 100)
    batch_size = min(max(req.batch_size, 32), 2048)

    # Create in-memory job (for live status)
    internal_params: dict[str, Any] = {
        "dataset": canonical.lower().replace("fashionmnist", "fashion_mnist"),
        "device": "cpu",
        "n_pop": 10,
        "n_generations": epochs,
        "t_local": 2,
        "lr": 1e-3,
        "lambda1": 1e-4,
        "lambda2": 1e-3,
        "init_hidden": [128, 64],
        "batch_size": batch_size,
        # Metadata for DB record
        "_user_email": email,
        "_samples": effective_samples,
        "_canonical_dataset": canonical,
        "_epochs": epochs,
        "_batch_size": batch_size,
    }
    job_id = create_job(internal_params)

    # Persist to DB (for history + daily limit counting)
    create_training(
        job_id=job_id,
        user_email=email,
        dataset=canonical,
        epochs=epochs,
        samples_requested=req.samples,
        samples_used=effective_samples,
        batch_size=batch_size,
    )

    # Launch background training
    background_tasks.add_task(start_training, job_id)
    logger.info("Job %s queued (user=%s plan=%s dataset=%s samples=%d)", job_id, email, plan, canonical, effective_samples)

    msg = f"Training queued. CPU mode (~2–25 minutes depending on dataset)."
    if effective_samples < req.samples:
        msg = f"Samples capped to {effective_samples:,} ({plan} plan limit). {msg}"

    return TrainStartResponse(
        job_id=job_id,
        status="queued",
        dataset=canonical,
        samples_used=effective_samples,
        epochs=epochs,
        message=msg,
    )


# ── GET /train/history ─────────────────────────────────────────────────────────

@router.get("/train/history")
async def training_history(
    page: int = 1,
    limit: int = 20,
    user: dict = Depends(get_auth_user),
):
    result = list_user_trainings(user["email"], page=page, limit=min(limit, 100))
    # Enrich with live status from _jobs if available
    for t in result["trainings"]:
        live = _jobs.get(t["job_id"])
        if live and t["status"] in ("queued", "running"):
            t["progress"] = int((live.get("progress", 0) * 100))
            t["current_epoch"] = live.get("current_generation", 0)
            t["accuracy"] = live.get("best_acc") or t.get("accuracy")
            t["status"] = str(live.get("status", t["status"]))
    return result


# ── GET /train/{job_id} ────────────────────────────────────────────────────────

@router.get("/train/{job_id}")
async def get_train_status(
    job_id: str,
    user: dict = Depends(get_auth_user),
):
    # Try live in-memory first for real-time progress
    live = _jobs.get(job_id)

    # Also get DB record for ownership check + persistent data
    db_record = get_training(job_id)

    if not live and not db_record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Ownership check: users can only see their own jobs
    if db_record and db_record["user_email"] != user["email"]:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if live:
        progress_pct = int(live.get("progress", 0) * 100)
        history = live.get("history", [])
        result = live.get("result")
        return {
            "job_id": job_id,
            "status": str(live.get("status", "queued")),
            "progress": progress_pct,
            "current_epoch": live.get("current_generation", 0),
            "total_epochs": live.get("total_generations", 10),
            "accuracy": round(live.get("best_acc", 0) * 100, 2) if live.get("best_acc") else None,
            "loss": None,  # loss not tracked in current evolver output
            "dataset": db_record["dataset"] if db_record else "",
            "samples_used": db_record["samples_used"] if db_record else 0,
            "duration_seconds": (result or {}).get("duration_seconds") if result else None,
            "history": [
                {"epoch": h.generation, "accuracy": round(h.best_acc * 100, 2)}
                for h in history
            ] if history else [],
            "error": live.get("error"),
        }

    # Fallback: return from DB record only (job completed before restart)
    return {
        "job_id": job_id,
        "status": db_record["status"],
        "progress": db_record["progress"],
        "current_epoch": db_record["current_epoch"],
        "total_epochs": db_record["epochs_requested"],
        "accuracy": round(db_record["accuracy"] * 100, 2) if db_record.get("accuracy") else None,
        "loss": db_record.get("loss"),
        "dataset": db_record["dataset"],
        "samples_used": db_record["samples_used"],
        "duration_seconds": db_record.get("duration_seconds"),
        "history": [],
        "error": db_record.get("error_message"),
    }
