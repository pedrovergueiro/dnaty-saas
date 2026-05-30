"""
Compress route — POST /compress, GET /compress/{job_id}

Runs dNATY evolutionary NAS in background, then asks Claude to explain
results and generate ready-to-use deployment code.

Auth: same Bearer JWT / X-API-Key as the train route.
"""
from __future__ import annotations
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from routes.train import get_auth_user

logger  = logging.getLogger(__name__)
router  = APIRouter()

# In-memory store (same pattern as train route)
_jobs: dict[str, dict[str, Any]] = {}

SUPPORTED_DATASETS = {"MNIST", "FashionMNIST"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class CompressRequest(BaseModel):
    description:   str   = Field(default="", max_length=500,
                                  description="Describe your problem (used by Claude for context)")
    dataset:       str   = Field(default="MNIST")
    target_flops:  float = Field(default=0.5, ge=0.1, le=0.9,
                                  description="Target FLOPs as fraction of original (0.5 = 50% less)")
    n_generations: int   = Field(default=30, ge=5, le=100)
    n_pop:         int   = Field(default=15, ge=5, le=30)


class CompressStartResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


# ── Background worker ─────────────────────────────────────────────────────────

def _run_compress(job_id: str, req: dict, user_email: str) -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    job = _jobs[job_id]
    job["status"] = "running"

    try:
        import torch
        from dnaty.experiments.fast_dataset import FastDataset
        from dnaty.evolution.evolver import DnatyEvolver

        dataset  = req.get("dataset", "MNIST")
        n_gen    = req.get("n_generations", 30)
        n_pop    = req.get("n_pop", 15)
        t_flops  = req.get("target_flops", 0.5)
        device   = "cuda" if torch.cuda.is_available() else "cpu"

        ds = FastDataset(dataset, device=device, train_subset=10_000)

        lambda2     = 1e-6 if t_flops <= 0.5 else 5e-7
        init_hidden = [256, 128]

        evolver = DnatyEvolver(
            n_pop=n_pop,
            n_generations=n_gen,
            t_local=3,
            input_size=784,
            n_classes=10,
            init_hidden=init_hidden,
            device=device,
            verbose=False,
            lambda2=lambda2,
        )

        # Baseline before search
        init_ind    = evolver._make_individual()
        orig_flops  = init_ind.count_flops()
        orig_params = init_ind.count_params()

        def _cb(log):
            job["progress"]    = log.gen / n_gen
            job["current_gen"] = log.gen
            job["best_acc"]    = log.best_acc

        best, _ = evolver.run(ds, ds, progress_callback=_cb)

        compressed_flops  = best.count_flops()
        compressed_params = best.count_params()
        arch = list(getattr(best.model, "layer_sizes", init_hidden))

        result_data: dict[str, Any] = {
            "original_flops":    orig_flops,
            "compressed_flops":  compressed_flops,
            "original_params":   orig_params,
            "compressed_params": compressed_params,
            "accuracy":          round(best.acc, 4),
            "flops_reduction":   round(max(0.0, 1.0 - compressed_flops / max(orig_flops, 1)), 4),
            "params_reduction":  round(max(0.0, 1.0 - compressed_params / max(orig_params, 1)), 4),
            "arch":              arch,
            "input_size":        784,
            "n_classes":         10,
            "dataset":           dataset,
        }

        # Claude explanation (graceful fallback if key missing)
        from services.claude_service import explain_compression
        explanation, deploy_code = explain_compression(result_data)
        result_data["explanation"]     = explanation
        result_data["deployment_code"] = deploy_code

        job["status"]   = "completed"
        job["progress"] = 1.0
        job["result"]   = result_data

        logger.info(
            "Compress %s done | user=%s acc=%.4f flops_red=%.1f%%",
            job_id, user_email, best.acc, result_data["flops_reduction"] * 100,
        )

    except Exception:
        logger.exception("Compress job %s failed", job_id)
        job["status"] = "failed"
        import traceback
        job["error"] = traceback.format_exc(limit=5)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/compress", response_model=CompressStartResponse, status_code=202)
async def start_compress(
    req: CompressRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_auth_user),
) -> CompressStartResponse:
    dataset_norm = req.dataset.strip()
    if dataset_norm not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_norm}' not supported. Use: {sorted(SUPPORTED_DATASETS)}",
        )

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":      "queued",
        "progress":    0.0,
        "current_gen": 0,
        "best_acc":    None,
        "result":      None,
        "error":       None,
        "user_email":  user["email"],
    }

    background_tasks.add_task(_run_compress, job_id, req.model_dump(), user["email"])
    logger.info("Compress job %s queued | user=%s dataset=%s gens=%d",
                job_id, user["email"], req.dataset, req.n_generations)

    return CompressStartResponse(
        job_id=job_id,
        status="queued",
        message=(
            f"Compressao iniciada (dNATY {req.n_generations} geracoes, "
            f"dataset={req.dataset}). "
            f"Acompanhe em GET /api/v1/compress/{job_id}"
        ),
    )


@router.get("/compress/{job_id}")
async def get_compress_status(
    job_id: str,
    user: dict = Depends(get_auth_user),
):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job["user_email"] != user["email"]:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return {
        "job_id":      job_id,
        "status":      job["status"],
        "progress":    round(job["progress"] * 100, 1),
        "current_gen": job["current_gen"],
        "best_acc":    round(job["best_acc"] * 100, 2) if job["best_acc"] else None,
        "error":       job.get("error"),
        "result":      job.get("result"),
    }
