"""
Admin authentication endpoints.

Endpoints:
  POST /admin/auth/login           → authenticate with email + password + TOTP
  GET  /admin/setup-2fa?bootstrap_secret=X  → initialize TOTP secret (one-time)
"""
import logging
from datetime import datetime, timedelta

import bcrypt
import pyotp
from fastapi import APIRouter, Depends, HTTPException, Request
from jose import JWTError, jwt

from config import settings
from models.admin_store import is_admin_rate_limited, record_login_attempt

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_client_ip(request: Request) -> str:
    """Extract client IP, checking X-Forwarded-For (Railway proxy) first."""
    if request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def validate_admin_token(token: str) -> bool:
    """Verify JWT with admin_jwt_secret. Return False if invalid or expired."""
    if not settings.admin_jwt_secret:
        return False
    try:
        payload = jwt.decode(
            token,
            settings.admin_jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload.get("role") == "admin"
    except JWTError:
        return False


async def require_admin(request: Request) -> str:
    """
    FastAPI Depends — extract Bearer token from Authorization header.
    Return the token if valid, raise HTTP 404 if invalid (never 401/403).
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=404, detail="Not found")
    
    token = auth_header[7:]  # Remove "Bearer "
    if not validate_admin_token(token):
        raise HTTPException(status_code=404, detail="Not found")
    
    return token


@router.post("/admin/auth/login", summary="Authenticate admin (email + password + TOTP)")
async def login(
    request: Request,
    body: dict,  # {email, password, totp_code}
):
    """
    Authenticate admin user. Rate-limit by IP/email. Perform all three checks
    (email, password, TOTP) without early-exit to prevent timing attacks.
    Return 404 on any failure, never 401/403.
    """
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    totp_code = body.get("totp_code", "").strip()
    
    ip = _get_client_ip(request)
    
    # Check rate limit
    if is_admin_rate_limited(ip, email):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Perform all three checks without early-exit
    email_ok = email == settings.admin_email
    password_ok = (
        settings.admin_password_hash
        and bcrypt.checkpw(password.encode(), settings.admin_password_hash.encode())
    )
    
    totp_ok = False
    if settings.admin_totp_secret:
        try:
            totp = pyotp.TOTP(settings.admin_totp_secret)
            totp_ok = totp.verify(totp_code, valid_window=1)
        except Exception:
            totp_ok = False
    
    # Record attempt
    success = email_ok and password_ok and totp_ok
    record_login_attempt(ip, email, success)

    if not success:
        # TEMP DIAGNOSTIC — shows which check failed (no secrets logged)
        logger.warning(
            "Admin login FAILED from %s: email_ok=%s password_ok=%s totp_ok=%s (pwd_len=%d totp_len=%d)",
            ip, email_ok, bool(password_ok), totp_ok, len(password), len(totp_code),
        )
        raise HTTPException(status_code=404, detail="Not found")
    
    # Issue JWT (8 hours)
    now = datetime.utcnow()
    payload = {
        "sub": "admin",
        "role": "admin",
        "exp": now + timedelta(hours=8),
        "iat": now,
    }
    token = jwt.encode(
        payload,
        settings.admin_jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    
    logger.info("Admin login successful from IP %s", ip)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/admin/setup-2fa", summary="Initialize TOTP secret (one-time bootstrap)")
async def setup_2fa(request: Request, bootstrap_secret: str = ""):
    """
    Generate and store TOTP secret. Only works if:
    - settings.admin_totp_secret is empty (not yet configured)
    - bootstrap_secret matches settings.admin_bootstrap_secret
    
    Return 404 if conditions not met.
    """
    if settings.admin_totp_secret != "":
        # TOTP already configured
        raise HTTPException(status_code=404, detail="Not found")
    
    if bootstrap_secret != settings.admin_bootstrap_secret or not bootstrap_secret:
        # Invalid or missing bootstrap secret
        raise HTTPException(status_code=404, detail="Not found")
    
    # Generate new secret
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    qr_uri = totp.provisioning_uri(
        name=settings.admin_email or "admin@dnaty",
        issuer_name="dNATY Admin",
    )
    
    # Save secret to settings
    # Note: In production, this should update .env file or env vars
    settings.admin_totp_secret = secret
    
    logger.info("Admin TOTP secret initialized")
    return {
        "secret": secret,
        "qr_uri": qr_uri,
    }
