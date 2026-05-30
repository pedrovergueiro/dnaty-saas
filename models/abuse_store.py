"""
Anti-abuse layer: device fingerprint tracking + silent banning.

Design principles:
- All checks are silent — the caller sees only "allowed" or "blocked".
- Bans are stored as active records; deactivation is manual (admin) or via expires_at.
- DB unavailable → fail open (don't block legitimate users).
- Fingerprint header is optional; missing header uses email/IP fallback.
- X-Device-FP format: "<fp_hash>.<hmac_sig_hex32>" — verified if DEVICE_FP_HMAC_KEY is set.
  If the key is empty, signature check is skipped (dev/test mode).
"""
import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta

import models.database as _db
from models.abuse_model import Ban, DeviceFingerprint

logger = logging.getLogger(__name__)

MAX_ACCOUNTS_PER_IP_24H = 3
IP_BAN_TTL_DAYS = 30


def _db_ok() -> bool:
    return _db.SessionLocal is not None


# ── Fingerprint parsing + verification ────────────────────────────────────────

def _parse_fp(raw_fp: str | None) -> str | None:
    """
    Parse and optionally verify X-Device-FP header.

    Format: "<fp_hash>.<hmac_hex>"
      - fp_hash: the device fingerprint (arbitrary hex string from client)
      - hmac_hex: first 32 hex chars of HMAC-SHA256(fp_hash, DEVICE_FP_HMAC_KEY)

    Returns the fp_hash on success, None if the header is missing or the
    signature is invalid (when DEVICE_FP_HMAC_KEY is configured).
    Legacy headers without a dot (no sig) are accepted only when the key is empty.
    """
    if not raw_fp:
        return None

    from config import settings

    if "." not in raw_fp:
        # No signature field — accept only in dev/test (key not set)
        if settings.device_fp_hmac_key:
            logger.debug("X-Device-FP rejected: missing HMAC signature")
            return None
        return raw_fp  # dev/test mode: treat whole value as fp_hash

    fp_hash, sig = raw_fp.rsplit(".", 1)
    if not fp_hash:
        return None

    if settings.device_fp_hmac_key:
        expected = hmac.new(
            settings.device_fp_hmac_key.encode(),
            fp_hash.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        if not hmac.compare_digest(expected, sig[:32].lower()):
            logger.debug("X-Device-FP rejected: HMAC mismatch fp=%s...", fp_hash[:12])
            return None

    return fp_hash


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _ban_active(ban: Ban) -> bool:
    """Return True only if the ban has not expired."""
    if not ban.is_active:
        return False
    if ban.expires_at and ban.expires_at < datetime.utcnow():
        return False
    return True


def _is_banned_fingerprint(db, fingerprint: str) -> bool:
    rows = (
        db.query(Ban)
        .filter(Ban.is_active == True, Ban.fingerprint_hash == fingerprint)
        .all()
    )
    return any(_ban_active(b) for b in rows)


def _is_banned_ip(db, ip: str) -> bool:
    rows = (
        db.query(Ban)
        .filter(Ban.is_active == True, Ban.ip_address == ip)
        .all()
    )
    return any(_ban_active(b) for b in rows)


def _is_banned_email(db, email: str) -> bool:
    rows = (
        db.query(Ban)
        .filter(Ban.is_active == True, Ban.user_email == email)
        .all()
    )
    return any(_ban_active(b) for b in rows)


def _create_ban(
    db,
    fingerprint: str | None,
    ip: str | None,
    reason: str,
    user_email: str | None = None,
    ttl_days: int | None = None,
) -> None:
    expires_at = datetime.utcnow() + timedelta(days=ttl_days) if ttl_days else None
    db.add(Ban(
        id=uuid.uuid4(),
        fingerprint_hash=fingerprint,
        ip_address=ip,
        user_email=user_email,
        reason=reason,
        expires_at=expires_at,
    ))
    db.commit()
    logger.warning(
        "Ban created — reason=%s fp=%s ip=%s email=%s expires=%s",
        reason,
        (fingerprint or "")[:12] + "..." if fingerprint else "none",
        ip or "none",
        user_email or "none",
        expires_at.date() if expires_at else "permanent",
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def check_signup_abuse(raw_fp: str | None, ip: str | None) -> bool:
    """
    Returns True (allowed to sign up) or False (silently blocked).
    Runs BEFORE creating the user account.

    Steps (in order):
      1. Validate + parse fingerprint header (HMAC check if key configured).
      2. Fingerprint already in active bans? → block
      3. IP already in active bans? → block
      4. Fingerprint already registered (= existing account)? → multi-account ban + block
      5. Same IP created 3+ accounts in last 24h? → ip_abuse ban (30d TTL) + block
    """
    fingerprint = _parse_fp(raw_fp)

    if not fingerprint and not ip:
        return True
    if not _db_ok():
        return True  # fail open if DB not ready

    with _db.SessionLocal() as db:
        if fingerprint and _is_banned_fingerprint(db, fingerprint):
            logger.info("Signup blocked: fp banned fp=%s...", fingerprint[:12])
            return False

        if ip and _is_banned_ip(db, ip):
            logger.info("Signup blocked: IP banned ip=%s", ip)
            return False

        # Multi-account: ban fingerprint only (not IP — avoids blocking shared networks)
        if fingerprint:
            existing = (
                db.query(DeviceFingerprint)
                .filter(DeviceFingerprint.fingerprint_hash == fingerprint)
                .first()
            )
            if existing:
                _create_ban(db, fingerprint, None, "multi_account", existing.user_email)
                return False

        # IP velocity abuse: ban both fp and IP with 30-day TTL
        if ip:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            count = (
                db.query(DeviceFingerprint)
                .filter(
                    DeviceFingerprint.ip_address == ip,
                    DeviceFingerprint.first_seen_at > cutoff,
                )
                .count()
            )
            if count >= MAX_ACCOUNTS_PER_IP_24H:
                _create_ban(db, fingerprint, ip, "ip_abuse", ttl_days=IP_BAN_TTL_DAYS)
                logger.warning("IP abuse: ip=%s had %d accounts in 24h", ip, count)
                return False

    return True


def check_validate_abuse(raw_fp: str | None, user_email: str | None = None) -> bool:
    """
    Returns True (ok to proceed) or False (silently blocked).
    Checks fingerprint ban, then falls back to email ban if no valid fp header.
    """
    fingerprint = _parse_fp(raw_fp)

    if not fingerprint and not user_email:
        return True
    if not _db_ok():
        return True

    with _db.SessionLocal() as db:
        if fingerprint and _is_banned_fingerprint(db, fingerprint):
            logger.info("Validate blocked: fp banned fp=%s...", fingerprint[:12])
            return False
        if not fingerprint and user_email and _is_banned_email(db, user_email):
            logger.info("Validate blocked: email banned email=%s", user_email)
            return False

    return True


def check_login_abuse(raw_fp: str | None, ip: str | None) -> bool:
    """
    Returns True (ok to proceed) or False (silently blocked).
    Called AFTER password verification to avoid timing oracle.
    Checks fingerprint ban and IP ban.
    """
    fingerprint = _parse_fp(raw_fp)

    if not fingerprint and not ip:
        return True
    if not _db_ok():
        return True

    with _db.SessionLocal() as db:
        if fingerprint and _is_banned_fingerprint(db, fingerprint):
            logger.info("Login blocked: fp banned fp=%s...", fingerprint[:12])
            return False
        if ip and _is_banned_ip(db, ip):
            logger.info("Login blocked: IP banned ip=%s", ip)
            return False

    return True


def register_fingerprint(raw_fp: str, user_email: str, ip: str | None) -> None:
    """
    Record device fingerprint after a successful signup.
    UPSERT: updates last_seen_at on conflict.
    """
    fingerprint = _parse_fp(raw_fp)
    if not fingerprint or not _db_ok():
        return
    try:
        with _db.SessionLocal() as db:
            existing = (
                db.query(DeviceFingerprint)
                .filter(DeviceFingerprint.fingerprint_hash == fingerprint)
                .first()
            )
            if existing:
                existing.last_seen_at = datetime.utcnow()
                if not existing.user_email:
                    existing.user_email = user_email
            else:
                db.add(DeviceFingerprint(
                    id=uuid.uuid4(),
                    fingerprint_hash=fingerprint,
                    user_email=user_email,
                    ip_address=ip,
                ))
            db.commit()
    except Exception:
        logger.debug("register_fingerprint: conflict on fp=%s (race, ignored)", fingerprint[:12])


# ── Admin helpers ──────────────────────────────────────────────────────────────

def get_bans(
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return ban records (for admin view). Expired bans are included unless active_only=True."""
    if not _db_ok():
        return []
    with _db.SessionLocal() as db:
        q = db.query(Ban)
        if active_only:
            q = q.filter(Ban.is_active == True)
        rows = q.order_by(Ban.banned_at.desc()).offset(offset).limit(limit).all()
        return [
            {
                "id": str(r.id),
                "fingerprint_hash": r.fingerprint_hash,
                "ip_address": r.ip_address,
                "user_email": r.user_email,
                "reason": r.reason,
                "banned_at": r.banned_at.isoformat() if r.banned_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "is_active": r.is_active,
                "expired": (r.expires_at is not None and r.expires_at < datetime.utcnow()),
            }
            for r in rows
        ]


def unban_by_id(ban_id: str) -> bool:
    """Deactivate a ban by its UUID. Returns True if found and updated."""
    if not _db_ok():
        return False
    try:
        with _db.SessionLocal() as db:
            ban = db.query(Ban).filter(Ban.id == uuid.UUID(ban_id)).first()
            if not ban:
                return False
            ban.is_active = False
            db.commit()
            logger.info("Ban deactivated — id=%s", ban_id)
            return True
    except (ValueError, Exception) as e:
        logger.warning("unban_by_id failed: %s", e)
        return False
