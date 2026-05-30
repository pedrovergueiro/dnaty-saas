import io
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from models.dnaty_model import get_job
from models.schemas import JobStatus, ResultsResponse
from models.training_store import get_training
from routes.train import get_auth_user

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_job_or_404(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get(
    "/results/{job_id}",
    response_model=ResultsResponse,
    summary="Fetch final results of a completed training job",
)
async def get_results(job_id: str) -> ResultsResponse:
    job = _get_job_or_404(job_id)

    if job["status"] == JobStatus.failed:
        raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")

    if job["status"] in (JobStatus.queued, JobStatus.running):
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job['status']}. Progress: {job['progress']:.0%}",
        )

    result = job["result"]
    return ResultsResponse(
        job_id=job_id,
        status=job["status"],
        dataset=result["dataset"],
        best_accuracy=result["best_accuracy"],
        final_architecture=result["final_architecture"],
        history=result["history"],
        duration_seconds=result["duration_seconds"],
        metadata=result["metadata"],
    )


@router.get("/results/{job_id}/model", summary="Download trained model (.pt) — Pro/Enterprise only")
async def download_model(
    job_id: str,
    user: dict = Depends(get_auth_user),
):
    from routes.train import get_plan_limits
    limits = get_plan_limits(user["plan"])
    if not limits["can_export"]:
        raise HTTPException(
            status_code=403,
            detail=f"Model export requires Pro or Enterprise plan. Upgrade at dnaty.co/pricing",
        )

    # Ownership check
    db_record = get_training(job_id)
    if db_record and db_record["user_email"] != user["email"]:
        raise HTTPException(status_code=404, detail="Job not found")

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired (server restarted)")

    if job["status"] != JobStatus.completed:
        raise HTTPException(status_code=400, detail="Training not complete yet")

    result = job.get("result", {})
    arch = result.get("final_architecture")
    if not arch:
        raise HTTPException(status_code=404, detail="Model not available for this job")

    # Serialize model weights to bytes
    try:
        import torch
        from dnaty.core.arch import DynamicMLP

        best_ind = None
        # Find best individual from in-memory job
        from models.dnaty_model import _jobs
        live = _jobs.get(job_id)
        if live:
            pop = getattr(live.get("_evolver"), "population", None)
            if pop:
                best_ind = max(pop, key=lambda ind: ind.acc)

        if best_ind is None:
            # Reconstruct minimal model from arch info
            model = DynamicMLP(
                layer_sizes=arch.layer_sizes,
                activations=arch.activations,
                n_classes=10,
            )
        else:
            model = best_ind.model

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        dataset = (result.get("dataset") or "model").lower().replace("_", "")
        filename = f"dnaty_{dataset}_{job_id[:8]}.pt"

        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="PyTorch not available on this server")
    except Exception as e:
        logger.error("Model download failed for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to serialize model")
