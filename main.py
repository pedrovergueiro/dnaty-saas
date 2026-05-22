"""
dNATY SaaS API — FastAPI entry point.

Endpoints:
  POST /api/v1/train          → submit evolutionary training job
  GET  /api/v1/status/{id}   → poll progress
  GET  /api/v1/results/{id}  → fetch final results
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routes import auth, billing, train, status as status_route, results

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dnaty_saas")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("dNATY SaaS API starting up")
    from models.database import create_tables
    create_tables()
    logger.info("Database tables ready")
    yield
    logger.info("dNATY SaaS API shutting down")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="dNATY SaaS API",
    description=(
        "Evolutionary Neural Architecture Search via dNATY.\n\n"
        "Submit a training job, poll its progress, and retrieve the best architecture found."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=settings.debug,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX, tags=["Auth"])
app.include_router(billing.router, prefix=API_PREFIX, tags=["Billing"])
app.include_router(train.router, prefix=API_PREFIX, tags=["Training"])
app.include_router(status_route.router, prefix=API_PREFIX, tags=["Status"])
app.include_router(results.router, prefix=API_PREFIX, tags=["Results"])


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health():
    return {"status": "ok", "version": app.version}


# ── Dev entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
