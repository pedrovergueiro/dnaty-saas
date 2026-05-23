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


def create_tables() -> None:
    if engine is None:
        logger.warning("create_tables ignorado — engine não inicializado")
        return
    from models.user_model import User, StripeSession  # noqa: F401
    Base.metadata.create_all(bind=engine)
