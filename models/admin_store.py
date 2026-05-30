"""
Admin data access layer — all DB reads/writes for the admin panel.
"""
import uuid
import logging
from datetime import datetime, timedelta, date
from typing import Any

import models.database as _db
from models.admin_model import AdminAuditLog, AdminLoginAttempt, CommercialLicense
from models.abuse_model import Ban, DeviceFingerprint
from models.api_key_model import ApiKey, UsageLog
from models.user_model import User

logger = logging.getLogger(__name__)


def _db_ok() -> bool:
    return _db.SessionLocal is not None


def _today_start() -> datetime:
    return datetime.combine(date.today(), datetime.min.time())


# ── Audit log ──────────────────────────────────────────────────────────────────

def append_audit_log(
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    details: dict | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    if not _db_ok():
        return
    try:
        with _db.SessionLocal() as db:
            db.add(AdminAuditLog(
                id=uuid.uuid4(),
                action=action,
                target_type=target_type,
                target_id=str(target_id) if target_id else None,
                details=details or {},
                ip_address=ip,
                user_agent=(user_agent or "")[:512],
            ))
            db.commit()
    except Exception as e:
        logger.warning("audit_log write failed: %s", e)


# ── Rate limiting ──────────────────────────────────────────────────────────────

def is_admin_rate_limited(ip: str, email: str) -> bool:
    """Block if ≥5 failed attempts in last 15 min from same IP or same email."""
    if not _db_ok():
        return False
    cutoff = datetime.utcnow() - timedelta(minutes=15)
    try:
        with _db.SessionLocal() as db:
            ip_fails = (
                db.query(AdminLoginAttempt)
                .filter(
                    AdminLoginAttempt.ip_address == ip,
                    AdminLoginAttempt.success == False,
                    AdminLoginAttempt.created_at >= cutoff,
                )
                .count()
            )
            email_fails = (
                db.query(AdminLoginAttempt)
                .filter(
                    AdminLoginAttempt.email_tried == email.lower(),
                    AdminLoginAttempt.success == False,
                    AdminLoginAttempt.created_at >= cutoff,
                )
                .count()
            )
            return ip_fails >= 5 or email_fails >= 5
    except Exception:
        return False


def record_login_attempt(ip: str, email: str, success: bool) -> None:
    if not _db_ok():
        return
    try:
        with _db.SessionLocal() as db:
            db.add(AdminLoginAttempt(
                id=uuid.uuid4(),
                ip_address=ip,
                email_tried=email.lower(),
                success=success,
            ))
            db.commit()
    except Exception as e:
        logger.warning("record_login_attempt failed: %s", e)


# ── Overview ───────────────────────────────────────────────────────────────────

def get_overview_stats(live_jobs: int = 0) -> dict:
    if not _db_ok():
        return {}
    today = _today_start()
    try:
        with _db.SessionLocal() as db:
            total_users  = db.query(User).count()
            pro_users    = db.query(User).filter(User.subscription_active == True).count()
            users_today  = db.query(User).filter(User.created_at >= today).count()
            active_bans  = db.query(Ban).filter(Ban.is_active == True).count()
            bans_today   = db.query(Ban).filter(Ban.banned_at >= today).count()
            total_trains = db.query(UsageLog).count()
            trains_today = db.query(UsageLog).filter(UsageLog.created_at >= today).count()
            failed_logins_today = (
                db.query(AdminLoginAttempt)
                .filter(
                    AdminLoginAttempt.success == False,
                    AdminLoginAttempt.created_at >= today,
                )
                .count()
            )
            mrr = pro_users * 19.0
            return {
                "users": {
                    "total": total_users,
                    "today": users_today,
                    "pro": pro_users,
                    "free": total_users - pro_users,
                    "banned": active_bans,
                },
                "revenue": {
                    "mrr": mrr,
                    "arr": mrr * 12,
                    "today": 0.0,        # Need Stripe events for real-time
                    "this_month": mrr,   # Approximate
                },
                "trainings": {
                    "running_now": live_jobs,
                    "today": trains_today,
                    "failed_today": 0,   # No failure tracking in usage_logs
                    "total": total_trains,
                },
                "infrastructure": {
                    "api_status": "online",
                    "response_time_ms": 0,
                    "error_rate_24h": 0.0,
                },
                "security": {
                    "bans_today": bans_today,
                    "suspicious_today": 0,
                    "failed_logins_today": failed_logins_today,
                },
            }
    except Exception as e:
        logger.error("get_overview_stats failed: %s", e)
        return {}


# ── Users ──────────────────────────────────────────────────────────────────────

def list_users(
    page: int = 1,
    limit: int = 50,
    plan: str = "all",
    search: str = "",
) -> dict:
    if not _db_ok():
        return {"users": [], "total": 0}
    offset = (page - 1) * limit
    try:
        with _db.SessionLocal() as db:
            q = db.query(User)
            if search:
                q = q.filter(User.email.ilike(f"%{search}%"))
            if plan == "pro":
                q = q.filter(User.subscription_active == True)
            elif plan == "free":
                q = q.filter(User.subscription_active == False)

            total = q.count()
            rows  = q.order_by(User.created_at.desc().nullslast()).offset(offset).limit(limit).all()

            # Build result with extra info
            results = []
            for u in rows:
                key = db.query(ApiKey).filter(ApiKey.user_email == u.email, ApiKey.is_active == True).first()
                banned = db.query(Ban).filter(Ban.user_email == u.email, Ban.is_active == True).first() is not None
                results.append({
                    "email":            u.email,
                    "name":             u.name or "",
                    "plan":             "pro" if u.subscription_active else "free",
                    "subscription_active": u.subscription_active,
                    "stripe_customer_id": u.stripe_customer_id or "",
                    "created_at":       u.created_at.isoformat() if u.created_at else None,
                    "trainings_used":   key.trainings_used if key else 0,
                    "last_used_at":     key.last_used_at.isoformat() if key and key.last_used_at else None,
                    "is_banned":        banned,
                })
            return {"users": results, "total": total}
    except Exception as e:
        logger.error("list_users failed: %s", e)
        return {"users": [], "total": 0}


def get_user_detail(email: str) -> dict | None:
    if not _db_ok():
        return None
    try:
        with _db.SessionLocal() as db:
            u = db.get(User, email.lower())
            if not u:
                return None
            keys     = db.query(ApiKey).filter(ApiKey.user_email == u.email).all()
            logs     = db.query(UsageLog).filter(UsageLog.user_email == u.email).order_by(UsageLog.created_at.desc()).limit(20).all()
            fps      = db.query(DeviceFingerprint).filter(DeviceFingerprint.user_email == u.email).all()
            bans     = db.query(Ban).filter(Ban.user_email == u.email).all()
            return {
                "email":            u.email,
                "name":             u.name or "",
                "subscription_active": u.subscription_active,
                "stripe_customer_id": u.stripe_customer_id or "",
                "created_at":       u.created_at.isoformat() if u.created_at else None,
                "api_keys": [
                    {
                        "prefix": k.key_prefix,
                        "plan": k.plan,
                        "is_active": k.is_active,
                        "trainings_used": k.trainings_used,
                        "created_at": k.created_at.isoformat() if k.created_at else None,
                        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                    }
                    for k in keys
                ],
                "recent_trainings": [
                    {
                        "dataset": l.dataset,
                        "samples_used": l.samples_used,
                        "accuracy": l.accuracy,
                        "duration_seconds": l.duration_seconds,
                        "hardware": l.hardware,
                        "created_at": l.created_at.isoformat() if l.created_at else None,
                    }
                    for l in logs
                ],
                "fingerprints": [
                    {
                        "hash": f.fingerprint_hash[:16] + "...",
                        "ip_address": f.ip_address,
                        "first_seen": f.first_seen_at.isoformat() if f.first_seen_at else None,
                    }
                    for f in fps
                ],
                "bans": [
                    {
                        "id": str(b.id),
                        "reason": b.reason,
                        "is_active": b.is_active,
                        "banned_at": b.banned_at.isoformat() if b.banned_at else None,
                    }
                    for b in bans
                ],
            }
    except Exception as e:
        logger.error("get_user_detail failed: %s", e)
        return None


def change_user_plan(email: str, plan: str) -> bool:
    if not _db_ok():
        return False
    try:
        with _db.SessionLocal() as db:
            u = db.get(User, email.lower())
            if not u:
                return False
            u.subscription_active = (plan == "pro")
            key = db.query(ApiKey).filter(ApiKey.user_email == email, ApiKey.is_active == True).first()
            if key:
                key.plan = plan
            db.commit()
            return True
    except Exception as e:
        logger.error("change_user_plan failed: %s", e)
        return False


def admin_ban_user(email: str, reason: str = "admin") -> None:
    if not _db_ok():
        return
    from models.api_key_model import ApiKey as AK
    try:
        with _db.SessionLocal() as db:
            # Ban all fingerprints associated with the user
            fps = db.query(DeviceFingerprint).filter(DeviceFingerprint.user_email == email).all()
            for fp in fps:
                db.add(Ban(
                    id=uuid.uuid4(),
                    fingerprint_hash=fp.fingerprint_hash,
                    ip_address=fp.ip_address,
                    user_email=email,
                    reason=reason,
                    is_active=True,
                ))
            # Also ban by email directly
            db.add(Ban(
                id=uuid.uuid4(),
                fingerprint_hash=None,
                ip_address=None,
                user_email=email,
                reason=reason,
                is_active=True,
            ))
            # Deactivate all API keys
            db.query(AK).filter(AK.user_email == email).update({"is_active": False})
            db.commit()
    except Exception as e:
        logger.error("admin_ban_user failed: %s", e)


def admin_unban_user(email: str) -> None:
    if not _db_ok():
        return
    from models.api_key_model import ApiKey as AK
    try:
        with _db.SessionLocal() as db:
            db.query(Ban).filter(Ban.user_email == email).update({"is_active": False})
            # Re-activate most recent API key
            key = db.query(AK).filter(AK.user_email == email).order_by(AK.created_at.desc()).first()
            if key:
                key.is_active = True
            db.commit()
    except Exception as e:
        logger.error("admin_unban_user failed: %s", e)


def admin_delete_user(email: str) -> bool:
    if not _db_ok():
        return False
    try:
        with _db.SessionLocal() as db:
            u = db.get(User, email.lower())
            if not u:
                return False
            db.delete(u)
            db.commit()
            return True
    except Exception as e:
        logger.error("admin_delete_user failed: %s", e)
        return False


# ── Trainings ──────────────────────────────────────────────────────────────────

def list_trainings(page: int = 1, limit: int = 50, user_email: str = "") -> dict:
    if not _db_ok():
        return {"trainings": [], "total": 0}
    offset = (page - 1) * limit
    try:
        with _db.SessionLocal() as db:
            q = db.query(UsageLog)
            if user_email:
                q = q.filter(UsageLog.user_email.ilike(f"%{user_email}%"))
            total = q.count()
            rows  = q.order_by(UsageLog.created_at.desc()).offset(offset).limit(limit).all()
            return {
                "trainings": [
                    {
                        "id": str(r.id),
                        "user_email": r.user_email,
                        "dataset": r.dataset,
                        "samples_used": r.samples_used,
                        "accuracy": r.accuracy,
                        "duration_seconds": r.duration_seconds,
                        "hardware": r.hardware,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ],
                "total": total,
            }
    except Exception as e:
        logger.error("list_trainings failed: %s", e)
        return {"trainings": [], "total": 0}


# ── Security / Bans ────────────────────────────────────────────────────────────

def list_bans(page: int = 1, limit: int = 50, active_only: bool = True) -> dict:
    if not _db_ok():
        return {"bans": [], "total": 0}
    offset = (page - 1) * limit
    try:
        with _db.SessionLocal() as db:
            q = db.query(Ban)
            if active_only:
                q = q.filter(Ban.is_active == True)
            total = q.count()
            rows  = q.order_by(Ban.banned_at.desc()).offset(offset).limit(limit).all()
            now   = datetime.utcnow()
            return {
                "bans": [
                    {
                        "id": str(r.id),
                        "fingerprint_hash": (r.fingerprint_hash or "")[:20] + "..." if r.fingerprint_hash else None,
                        "ip_address": r.ip_address,
                        "user_email": r.user_email,
                        "reason": r.reason,
                        "is_active": r.is_active,
                        "banned_at": r.banned_at.isoformat() if r.banned_at else None,
                        "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                        "expired": bool(r.expires_at and r.expires_at < now),
                    }
                    for r in rows
                ],
                "total": total,
            }
    except Exception as e:
        logger.error("list_bans failed: %s", e)
        return {"bans": [], "total": 0}


def list_failed_logins(hours: int = 24) -> list:
    if not _db_ok():
        return []
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    try:
        with _db.SessionLocal() as db:
            rows = (
                db.query(AdminLoginAttempt)
                .filter(AdminLoginAttempt.success == False, AdminLoginAttempt.created_at >= cutoff)
                .order_by(AdminLoginAttempt.created_at.desc())
                .limit(200)
                .all()
            )
            return [
                {
                    "ip_address": r.ip_address,
                    "email_tried": r.email_tried,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("list_failed_logins failed: %s", e)
        return []


# ── Revenue chart ──────────────────────────────────────────────────────────────

def get_revenue_chart(days: int = 30) -> list[dict]:
    """Daily cumulative pro-user count × $19. Approximation without payment events."""
    if not _db_ok():
        return []
    try:
        with _db.SessionLocal() as db:
            result = []
            for i in range(days - 1, -1, -1):
                day_end = datetime.utcnow() - timedelta(days=i)
                day_label = day_end.date().isoformat()
                # Pro users who existed at end of that day (approximate)
                pro_count = (
                    db.query(User)
                    .filter(
                        User.subscription_active == True,
                        User.created_at <= day_end,
                    )
                    .count()
                )
                result.append({"date": day_label, "mrr": pro_count * 19.0})
            return result
    except Exception as e:
        logger.error("get_revenue_chart failed: %s", e)
        return []


# ── Licenses ───────────────────────────────────────────────────────────────────

def list_licenses() -> list[dict]:
    if not _db_ok():
        return []
    try:
        with _db.SessionLocal() as db:
            rows = db.query(CommercialLicense).order_by(CommercialLicense.end_date.desc()).all()
            now  = datetime.utcnow()
            return [
                {
                    "id": str(r.id),
                    "company": r.company,
                    "contact_email": r.contact_email,
                    "tier": r.tier,
                    "value_usd": r.value_usd,
                    "start_date": r.start_date.isoformat() if r.start_date else None,
                    "end_date": r.end_date.isoformat() if r.end_date else None,
                    "status": (
                        "expired" if r.end_date < now
                        else "expiring_soon" if r.end_date < now + timedelta(days=30)
                        else "active"
                    ),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("list_licenses failed: %s", e)
        return []


def create_license(data: dict) -> dict:
    if not _db_ok():
        return {}
    try:
        with _db.SessionLocal() as db:
            lic = CommercialLicense(
                id=uuid.uuid4(),
                company=data["company"],
                contact_email=data["contact_email"],
                tier=data.get("tier", "standard"),
                value_usd=float(data["value_usd"]),
                start_date=datetime.fromisoformat(data["start_date"]),
                end_date=datetime.fromisoformat(data["start_date"]) + timedelta(days=30 * int(data.get("duration_months", 12))),
                notes=data.get("notes", ""),
            )
            db.add(lic)
            db.commit()
            db.refresh(lic)
            return {"id": str(lic.id), "company": lic.company}
    except Exception as e:
        logger.error("create_license failed: %s", e)
        return {}


# ── Audit log ──────────────────────────────────────────────────────────────────

def list_audit_log(page: int = 1, limit: int = 100, action_filter: str = "") -> dict:
    if not _db_ok():
        return {"logs": [], "total": 0}
    offset = (page - 1) * limit
    try:
        with _db.SessionLocal() as db:
            q = db.query(AdminAuditLog)
            if action_filter:
                q = q.filter(AdminAuditLog.action.ilike(f"%{action_filter}%"))
            total = q.count()
            rows  = q.order_by(AdminAuditLog.created_at.desc()).offset(offset).limit(limit).all()
            return {
                "logs": [
                    {
                        "id": str(r.id),
                        "action": r.action,
                        "target_type": r.target_type,
                        "target_id": r.target_id,
                        "details": r.details,
                        "ip_address": r.ip_address,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ],
                "total": total,
            }
    except Exception as e:
        logger.error("list_audit_log failed: %s", e)
        return {"logs": [], "total": 0}
