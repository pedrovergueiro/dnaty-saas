from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    api_key: str = ""
    debug: bool = False
    log_level: str = "INFO"
    allowed_origins: list[str] = ["*"]

    # PostgreSQL
    database_url: str = ""

    # Stripe
    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id_pro: str = ""

    # Auth / JWT
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 43200  # 30 days

    # URLs para redirects do Stripe
    frontend_url: str = "http://localhost:5173"
    landing_url: str = "http://localhost:5500"

    # Claude / Anthropic
    anthropic_api_key: str = ""

    # Worker (Colab GPU)
    worker_api_key: str = ""

    # Anti-abuse
    device_fp_hmac_key: str = ""   # HMAC-SHA256 key for X-Device-FP signature verification
    admin_api_key: str = ""         # Legacy — kept for backwards compat, superseded by admin JWT

    # Admin panel (separate from user auth)
    admin_email: str = ""                    # Your login email
    admin_password_hash: str = ""            # bcrypt hash of your password
    admin_totp_secret: str = ""              # base32 TOTP secret (Google Authenticator)
    admin_jwt_secret: str = ""               # Must be ≥32 chars — DIFFERENT from jwt_secret
    admin_bootstrap_secret: str = ""         # One-time secret for /setup-2fa endpoint

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8")


settings = Settings()
