"""
Data access layer for Training records.
Provides DB-backed persistence for jobs (supplements in-memory _jobs dict).
"""
import uuid
import logging
from datetime import datetime, date
from typing import Any

import models.database as _db
from models.training_model import Training

logger = logging.getLogger(__name__)


def _db_ok() -> bool:
    return _db.SessionLocal is not None


def _row_to_dict(t: Training) -> dict:
    return {
        "id":               str(t.id),
        "job_id":           t.job_id,
        "user_email":       t.user_email,
        "dataset":          t.dataset,
        "epochs_requested": t.epochs_requested,
        "samples_requested":t.samples_requested,
        "samples_used":     t.samples_used,
        "batch_size":       t.batch_size,
        "status":           t.status,
        "progress":         t.progress,
        "current_epoch":    t.current_epoch,
        "loss":             t.loss,
        "accuracy":         t.accuracy,
        "duration_seconds": t.duration_seconds,
        "error_message":    t.error_message,
        "created_at":       t.created_at.isoformat() if t.created_at else None,
        "completed_at":     t.completed_at.isoformat() if t.completed_at else None,
    }


def create_training(
    job_id: str,
    user_email: str,
    dataset: str,
    epochs: int,
    samples_requested: int,
    samples_used: int,
    batch_size: int = 512,
) -> dict:
    if not _db_ok():
        return {}
    try:
        with _db.SessionLocal() as db:
            t = Training(
                id=uuid.uuid4(),
                job_id=job_id,
                user_email=user_email.lower(),
                dataset=dataset,
                epochs_requested=epochs,
                samples_requested=samples_requested,
                samples_used=samples_used,
                batch_size=batch_size,
                status="queued",
            )
            db.add(t)
            db.commit()
            db.refresh(t)
            return _row_to_dict(t)
    except Exception as e:
        logger.error("create_training failed: %s", e)
        return {}


def update_training(job_id: str, **fields: Any) -> None:
    if not _db_ok():
        return
    try:
        with _db.SessionLocal() as db:
            t = db.query(Training).filter(Training.job_id == job_id).first()
            if t:
                for k, v in fields.items():
                    setattr(t, k, v)
                db.commit()
    except Exception as e:
        logger.warning("update_training failed: %s", e)


def get_training(job_id: str) -> dict | None:
    if not _db_ok():
        return None
    try:
        with _db.SessionLocal() as db:
            t = db.query(Training).filter(Training.job_id == job_id).first()
            return _row_to_dict(t) if t else None
    except Exception as e:
        logger.error("get_training failed: %s", e)
        return None


def list_user_trainings(user_email: str, page: int = 1, limit: int = 20) -> dict:
    if not _db_ok():
        return {"trainings": [], "total": 0}
    offset = (page - 1) * limit
    try:
        with _db.SessionLocal() as db:
            q = db.query(Training).filter(Training.user_email == user_email.lower())
            total = q.count()
            rows = q.order_by(Training.created_at.desc()).offset(offset).limit(limit).all()
            return {"trainings": [_row_to_dict(r) for r in rows], "total": total}
    except Exception as e:
        logger.error("list_user_trainings failed: %s", e)
        return {"trainings": [], "total": 0}


def count_trainings_today(user_email: str) -> int:
    """Count trainings started today (UTC) — used for free plan daily limit."""
    if not _db_ok():
        return 0
    today_start = datetime.combine(date.today(), datetime.min.time())
    try:
        with _db.SessionLocal() as db:
            return (
                db.query(Training)
                .filter(
                    Training.user_email == user_email.lower(),
                    Training.created_at >= today_start,
                )
                .count()
            )
    except Exception as e:
        logger.error("count_trainings_today failed: %s", e)
        return 0
