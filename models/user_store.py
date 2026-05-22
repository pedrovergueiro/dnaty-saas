"""
Camada de acesso a dados — PostgreSQL via SQLAlchemy.
A interface pública é idêntica à versão anterior em JSON,
então nenhuma rota precisa mudar.
"""
from models.database import SessionLocal
from models.user_model import StripeSession, User


# ── Helpers ────────────────────────────────────────────────────────────────────

def _user_dict(u: User) -> dict:
    return {
        "email": u.email,
        "password_hash": u.password_hash,
        "name": u.name or "",
        "subscription_active": u.subscription_active,
        "stripe_customer_id": u.stripe_customer_id or "",
    }


def _session_dict(s: StripeSession) -> dict:
    return {
        "email": s.email,
        "customer_id": s.customer_id or "",
        "used": s.used,
    }


# ── Users ──────────────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> dict | None:
    with SessionLocal() as db:
        u = db.get(User, email.lower())
        return _user_dict(u) if u else None


def create_user(
    email: str,
    password_hash: str,
    name: str = "",
    subscription_active: bool = False,
    stripe_customer_id: str = "",
) -> dict:
    with SessionLocal() as db:
        u = User(
            email=email.lower(),
            password_hash=password_hash,
            name=name,
            subscription_active=subscription_active,
            stripe_customer_id=stripe_customer_id,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return _user_dict(u)


def update_user(email: str, **fields) -> None:
    with SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            for k, v in fields.items():
                setattr(u, k, v)
            db.commit()


def activate_subscription(email: str, stripe_customer_id: str = "") -> None:
    with SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            u.subscription_active = True
            if stripe_customer_id:
                u.stripe_customer_id = stripe_customer_id
        else:
            # Pagou antes de criar conta — pré-cadastro
            u = User(
                email=email.lower(),
                password_hash=None,
                name="",
                subscription_active=True,
                stripe_customer_id=stripe_customer_id,
            )
            db.add(u)
        db.commit()


def deactivate_subscription(email: str) -> None:
    with SessionLocal() as db:
        u = db.get(User, email.lower())
        if u:
            u.subscription_active = False
            db.commit()


def find_user_by_customer_id(stripe_customer_id: str) -> dict | None:
    with SessionLocal() as db:
        u = (
            db.query(User)
            .filter(User.stripe_customer_id == stripe_customer_id)
            .first()
        )
        return _user_dict(u) if u else None


# ── Stripe checkout sessions ────────────────────────────────────────────────────

def mark_session_paid(session_id: str, email: str, customer_id: str) -> None:
    with SessionLocal() as db:
        existing = db.get(StripeSession, session_id)
        if existing:
            existing.email = email.lower()
            existing.customer_id = customer_id
            existing.used = False
        else:
            db.add(StripeSession(
                session_id=session_id,
                email=email.lower(),
                customer_id=customer_id,
                used=False,
            ))
        db.commit()


def get_session(session_id: str) -> dict | None:
    with SessionLocal() as db:
        s = db.get(StripeSession, session_id)
        return _session_dict(s) if s else None


def consume_session(session_id: str) -> dict | None:
    """Retorna os dados da sessão e marca como usada (uso único)."""
    with SessionLocal() as db:
        s = db.get(StripeSession, session_id)
        if s and not s.used:
            s.used = True
            db.commit()
            return _session_dict(s)
        return None
