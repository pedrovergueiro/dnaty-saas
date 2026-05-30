import json
import logging

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config import settings
from models.user_store import (
    activate_subscription,
    deactivate_subscription,
    find_user_by_customer_id,
    get_user_by_email,
    mark_session_paid,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_bearer = HTTPBearer(auto_error=False)


def _stripe_client() -> None:
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    stripe.api_key = settings.stripe_secret_key


def _get_current_email(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload["sub"]
    except (JWTError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── Schemas ────────────────────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str = "pro"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/billing/create-checkout")
async def create_checkout(req: CheckoutRequest, email: str = Depends(_get_current_email)):
    _stripe_client()

    if req.plan not in ("pro",):
        raise HTTPException(status_code=400, detail="Only 'pro' plan available via self-service checkout. Enterprise: contact legal@vergueiro.co")

    price_id = settings.stripe_price_id_pro
    if not price_id:
        raise HTTPException(status_code=503, detail="Stripe price ID not configured (STRIPE_PRICE_ID_PRO)")

    user = get_user_by_email(email)
    customer_id = (user or {}).get("stripe_customer_id") or None

    try:
        create_kwargs: dict = dict(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{settings.frontend_url}/dashboard?upgraded=true",
            cancel_url=f"{settings.frontend_url}/pricing",
            allow_promotion_codes=True,
        )
        if customer_id:
            create_kwargs["customer"] = customer_id
        else:
            create_kwargs["customer_email"] = email

        session = stripe.checkout.Session.create(**create_kwargs)
        logger.info("Created Stripe checkout session %s", session.id)
        return {"url": session.url, "session_id": session.id}
    except stripe.StripeError as e:
        logger.error("Stripe error creating session: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/billing/checkout-session")  # legacy alias
async def create_checkout_session_legacy(req: CheckoutRequest, email: str = Depends(_get_current_email)):
    return await create_checkout(req, email)


@router.get("/billing/portal")
async def billing_portal(email: str = Depends(_get_current_email)):
    """Redirect to Stripe Customer Portal for subscription management."""
    _stripe_client()
    user = get_user_by_email(email)
    customer_id = (user or {}).get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer ID found. Complete a checkout first.")
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{settings.frontend_url}/account",
        )
        return {"url": session.url}
    except stripe.StripeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/stripe/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(None, alias="stripe-signature"),
):
    payload = await request.body()

    if settings.stripe_webhook_secret and stripe_signature:
        try:
            stripe.api_key = settings.stripe_secret_key
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, settings.stripe_webhook_secret
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

        event_type = event.type
        obj = dict(event.data.object)
    else:
        logger.warning("Webhook signature check skipped — STRIPE_WEBHOOK_SECRET not set")
        raw = json.loads(payload)
        event_type = raw.get("type", "")
        obj = raw.get("data", {}).get("object", {})

    logger.info("Stripe webhook: %s", event_type)

    from models.api_key_store import downgrade_plan, upgrade_plan

    if event_type == "checkout.session.completed":
        customer_details = obj.get("customer_details") or {}
        email = obj.get("customer_email") or customer_details.get("email", "")
        customer_id = obj.get("customer", "")
        session_id = obj.get("id", "")

        if email:
            mark_session_paid(session_id, email, customer_id)
            activate_subscription(email, customer_id)
            upgrade_plan(email, "pro")
            logger.info("Activated subscription + upgraded to pro for %s (session=%s)", email, session_id)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        customer_id = obj.get("customer", "")
        if customer_id:
            user = find_user_by_customer_id(customer_id)
            if user:
                deactivate_subscription(user["email"])
                downgrade_plan(user["email"])
                logger.info("Deactivated subscription + downgraded to free for %s", user["email"])

    elif event_type == "invoice.payment_failed":
        customer_id = obj.get("customer", "")
        if customer_id:
            user = find_user_by_customer_id(customer_id)
            if user:
                deactivate_subscription(user["email"])
                downgrade_plan(user["email"])
                logger.info("Payment failed — deactivated + downgraded %s", user["email"])

    return {"received": True}
