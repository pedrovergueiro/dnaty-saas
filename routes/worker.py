"""
Rotas internas usadas pelo worker Colab — nunca expostas ao usuário final.
Autenticadas por WORKER_API_KEY (variável de ambiente no Railway).
"""
import time
import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from models.dnaty_model import _jobs
from models.schemas import ArchitectureInfo, GenerationInfo, JobStatus
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _auth(key: str | None) -> None:
    if not settings.worker_api_key or key != settings.worker_api_key:
        raise HTTPException(status_code=401, detail="Invalid worker key")


# ── Pega próximo job da fila ───────────────────────────────────────────────────

@router.get("/worker/next-job", include_in_schema=False)
async def next_job(x_worker_key: str | None = Header(None)):
    _auth(x_worker_key)
    for job_id, job in _jobs.items():
        if job["status"] == JobStatus.queued:
            job["status"] = JobStatus.running
            job["started_at"] = time.time()
            logger.info("Worker claimed job %s", job_id)
            return {"job_id": job_id, "params": job["params"]}
    return {"job_id": None}


# ── Progresso por geração ──────────────────────────────────────────────────────

class ProgressUpdate(BaseModel):
    generation: int
    best_acc: float
    delta_grad: float
    delta_mem: float
    n_params: int


@router.post("/worker/jobs/{job_id}/progress", include_in_schema=False)
async def update_progress(
    job_id: str,
    update: ProgressUpdate,
    x_worker_key: str | None = Header(None),
):
    _auth(x_worker_key)
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    info = GenerationInfo(
        generation=update.generation,
        best_acc=update.best_acc,
        delta_grad=update.delta_grad,
        delta_mem=update.delta_mem,
        n_params=update.n_params,
    )
    job["history"].append(info)
    job["current_generation"] = update.generation
    job["best_acc"] = update.best_acc
    job["progress"] = update.generation / max(job["total_generations"], 1)
    return {"ok": True}


# ── Job concluído ──────────────────────────────────────────────────────────────

class CompletePayload(BaseModel):
    best_accuracy: float
    layer_sizes: list[int]
    activations: list[str]
    n_params: int
    duration_seconds: float


@router.post("/worker/jobs/{job_id}/complete", include_in_schema=False)
async def complete_job(
    job_id: str,
    payload: CompletePayload,
    x_worker_key: str | None = Header(None),
):
    _auth(x_worker_key)
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    arch = ArchitectureInfo(
        layer_sizes=payload.layer_sizes,
        activations=payload.activations,
        n_params=payload.n_params,
    )
    job["result"] = {
        "best_accuracy": payload.best_accuracy,
        "final_architecture": arch,
        "history": job["history"],
        "duration_seconds": payload.duration_seconds,
        "dataset": job["params"]["dataset"],
        "metadata": {
            "n_pop": job["params"]["n_pop"],
            "n_generations": job["params"]["n_generations"],
            "device": "cuda",
        },
    }
    job["best_acc"] = payload.best_accuracy
    job["progress"] = 1.0
    job["status"] = JobStatus.completed
    job["finished_at"] = time.time()
    logger.info("Job %s completed — acc=%.4f", job_id, payload.best_accuracy)
    return {"ok": True}


# ── Job falhou ─────────────────────────────────────────────────────────────────

class FailPayload(BaseModel):
    error: str


@router.post("/worker/jobs/{job_id}/fail", include_in_schema=False)
async def fail_job(
    job_id: str,
    payload: FailPayload,
    x_worker_key: str | None = Header(None),
):
    _auth(x_worker_key)
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job["status"] = JobStatus.failed
    job["error"] = payload.error
    job["finished_at"] = time.time()
    logger.error("Job %s failed: %s", job_id, payload.error)
    return {"ok": True}
