import logging
from datetime import datetime, timedelta

import bcrypt
import stripe
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

from config import settings
from models.user_store import (
    consume_session,
    create_user,
    get_session,
    get_user_by_email,
    update_user,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_bearer = HTTPBearer(auto_error=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _create_token(email: str, subscription_active: bool) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode(
        {"sub": email, "subscription_active": subscription_active, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# ── Schemas ────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str = ""
    email: EmailStr
    password: str
    session_id: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/auth/signup", status_code=201)
async def signup(req: SignupRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = get_user_by_email(req.email)

    # Determine subscription status
    subscription_active = False
    stripe_customer_id = ""

    if existing:
        if existing.get("password_hash"):
            raise HTTPException(status_code=409, detail="Email already registered. Sign in instead.")
        # Pre-created by webhook → already paid
        subscription_active = existing.get("subscription_active", False)
        stripe_customer_id = existing.get("stripe_customer_id", "")

    # Validate Stripe session if provided (and subscription not yet confirmed)
    if req.session_id and not subscription_active:
        session = consume_session(req.session_id)
        if session is None:
            raise HTTPException(status_code=400, detail="Invalid or already used payment session")
        if session["email"] != req.email.lower():
            raise HTTPException(status_code=400, detail="Email does not match the payment session")
        subscription_active = True
        stripe_customer_id = session.get("customer_id", "")

    pw_hash = _hash_password(req.password)

    if existing:
        update_user(req.email, password_hash=pw_hash, name=req.name or existing.get("name", ""),
                    subscription_active=subscription_active, stripe_customer_id=stripe_customer_id)
        user = {**existing, "password_hash": pw_hash, "name": req.name or existing.get("name", ""),
                "subscription_active": subscription_active}
    else:
        user = create_user(req.email, pw_hash, req.name, subscription_active, stripe_customer_id)

    token = _create_token(req.email, user["subscription_active"])
    logger.info("Signup: %s (subscribed=%s)", req.email, user["subscription_active"])
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user.get("name", ""),
            "subscription_active": user["subscription_active"],
        },
    }


@router.post("/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_token(req.email, user["subscription_active"])
    logger.info("Login: %s", req.email)
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user.get("name", ""),
            "subscription_active": user["subscription_active"],
        },
    }


@router.get("/auth/verify-session")
async def verify_stripe_session(session_id: str):
    """Check if a Stripe checkout session was paid — called by frontend after redirect."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"email": session["email"], "paid": True, "used": session.get("used", False)}


@router.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's info from the JWT."""
    user = get_user_by_email(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "email": user["email"],
        "name": user.get("name", ""),
        "subscription_active": user["subscription_active"],
    }


# ── Pre-signup: cria conta + sessão Stripe em uma única chamada ────────────────

class PresignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


@router.post("/auth/presignup", status_code=200)
async def presignup(req: PresignupRequest):
    """
    Fluxo da landing: coleta dados de cadastro, cria conta inativa e
    retorna URL do Stripe Checkout. Após pagamento, webhook ativa a conta
    e Stripe redireciona para /login?email=...&paid=true.
    """
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if not settings.stripe_secret_key or not settings.stripe_price_id_pro:
        raise HTTPException(status_code=503, detail="Stripe not configured (STRIPE_SECRET_KEY or STRIPE_PRICE_ID_PRO missing)")

    existing = get_user_by_email(req.email)
    if existing and existing.get("subscription_active"):
        raise HTTPException(status_code=409, detail="This email already has an active subscription. Sign in instead.")

    pw_hash = _hash_password(req.password)
    if existing:
        update_user(req.email, password_hash=pw_hash, name=req.name or existing.get("name", ""))
    else:
        create_user(req.email, pw_hash, req.name, subscription_active=False)

    stripe.api_key = settings.stripe_secret_key
    try:
        session = stripe.checkout.Session.create(
            customer_email=req.email,
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": settings.stripe_price_id_pro, "quantity": 1}],
            success_url=(
                f"{settings.frontend_url}/login"
                f"?email={req.email}&paid=true"
            ),
            cancel_url=f"{settings.landing_url}/#pricing",
            allow_promotion_codes=True,
        )
    except stripe.StripeError as e:
        logger.error("Stripe error on presignup: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    logger.info("Presignup: %s → Stripe session %s", req.email, session.id)
    return {"url": session.url}
