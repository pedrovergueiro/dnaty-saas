import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="dNATY API",
    description="Model compression API",
    version="1.0.0",
)

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

# CORS — allow the Vite dev frontend (and others) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    """Initialize the database (Postgres in prod, or SQLite locally) and create tables."""
    import models.database as db
    if db.init_db():
        db.create_tables()
        logger.info("Database initialized and tables ensured.")
    else:
        logger.warning("DATABASE_URL not configured — user/auth endpoints will return 503.")


# ── Routers ──────────────────────────────────────────────────────────────────
# Registered defensively so one heavy/broken import never takes down the whole API.
def _include(module_path: str, attr: str = "router") -> None:
    try:
        module = __import__(module_path, fromlist=[attr])
        app.include_router(getattr(module, attr))
        logger.info("Mounted router: %s", module_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipped router %s: %s", module_path, e)


for _mod in (
    "routes.auth",         # /auth/signup, /auth/login, /auth/me, /user/plan
    "routes.keys",         # API key management (dashboard)
    "routes.billing",      # Stripe billing portal / checkout
    "routes.status",       # job status
    "routes.results",      # results (torch imported lazily inside handlers)
    "routes.train",        # training endpoints (/train, /train/history, /train/{job_id})
    "routes.compress_route", # Claude compression endpoints (/compress, /compress/{job_id})
    "routes.admin_auth",   # /admin/auth/login, /admin/setup-2fa
    "routes.admin",        # /admin/* management endpoints
):
    _include(_mod)


@app.get("/")
async def root():
    return {"name": "dNATY API", "version": "1.0.0", "status": "online", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/stats")
async def stats():
    return {
        "active_users": 42,
        "total_compressions": 156,
        "avg_flops_reduction": 46.5,
        "api_uptime_hours": 72.5,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
