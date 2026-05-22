import json
import logging

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from config import settings
from models.user_store import (
    activate_subscription,
    deactivate_subscription,
    find_user_by_customer_id,
    mark_session_paid,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _stripe_client() -> None:
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    stripe.api_key = settings.stripe_secret_key


# ── Schemas ────────────────────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str = "pro"  # "pro" is the only paid plan for now


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/billing/checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    _stripe_client()

    if req.plan != "pro":
        raise HTTPException(status_code=400, detail="Only 'pro' plan available via checkout")

    price_id = settings.stripe_price_id_pro
    if not price_id:
        raise HTTPException(status_code=503, detail="Stripe price ID not configured (STRIPE_PRICE_ID_PRO)")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=(
                f"{settings.frontend_url}/signup?session_id={{CHECKOUT_SESSION_ID}}"
            ),
            cancel_url=f"{settings.frontend_url}/#pricing",
            allow_promotion_codes=True,
        )
        logger.info("Created Stripe checkout session %s", session.id)
        return {"url": session.url, "session_id": session.id}
    except stripe.StripeError as e:
        logger.error("Stripe error creating session: %s", e)
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

    if event_type == "checkout.session.completed":
        customer_details = obj.get("customer_details") or {}
        email = obj.get("customer_email") or customer_details.get("email", "")
        customer_id = obj.get("customer", "")
        session_id = obj.get("id", "")

        if email:
            mark_session_paid(session_id, email, customer_id)
            activate_subscription(email, customer_id)
            logger.info("Activated subscription for %s (session=%s)", email, session_id)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        customer_id = obj.get("customer", "")
        if customer_id:
            user = find_user_by_customer_id(customer_id)
            if user:
                deactivate_subscription(user["email"])
                logger.info("Deactivated subscription for %s", user["email"])

    elif event_type == "invoice.payment_failed":
        customer_id = obj.get("customer", "")
        if customer_id:
            user = find_user_by_customer_id(customer_id)
            if user:
                deactivate_subscription(user["email"])
                logger.info("Payment failed — deactivated %s", user["email"])

    return {"received": True}
