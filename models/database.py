import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

logger = logging.getLogger(__name__)

engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


def init_db() -> bool:
    """Inicializa engine e cria tabelas. Retorna False se DATABASE_URL não estiver configurada."""
    global engine, SessionLocal

    if not settings.database_url:
        logger.error("DATABASE_URL não configurada — banco de dados indisponível")
        return False

    try:
        engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        return True
    except Exception as e:
        logger.error("Falha ao conectar ao banco: %s", e)
        return False


def _run_migrations() -> None:
    """Apply additive schema changes that Base.metadata.create_all cannot handle."""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE bans ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP NULL",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS plan VARCHAR(20) NULL",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR NULL",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id VARCHAR NULL",
        "UPDATE users SET plan = CASE WHEN subscription_active = true THEN 'pro' ELSE 'free' END WHERE plan IS NULL",
    ]
    try:
        with engine.connect() as conn:
            for sql in migrations:
                try:
                    conn.execute(text(sql))
                except Exception as e:
                    logger.debug("Migration skipped (non-fatal): %s | %s", sql[:60], e)
            conn.commit()
    except Exception as e:
        logger.warning("Migration batch failed (non-fatal): %s", e)


def create_tables() -> None:
    if engine is None:
        logger.warning("create_tables ignorado — engine não inicializado")
        return
    from models.user_model import User, StripeSession          # noqa: F401
    from models.api_key_model import ApiKey, UsageLog          # noqa: F401
    from models.abuse_model import DeviceFingerprint, Ban      # noqa: F401
    from models.admin_model import (                           # noqa: F401
        AdminAuditLog, AdminLoginAttempt, CommercialLicense,
    )
    from models.training_model import Training                  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _run_migrations()
