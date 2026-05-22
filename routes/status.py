import logging
from fastapi import APIRouter, HTTPException

from models.schemas import StatusResponse, JobStatus
from models.dnaty_model import get_job

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Poll training job progress",
    responses={
        200: {"description": "Job status and progress"},
        404: {"description": "Job not found"},
    },
)
async def get_status(job_id: str) -> StatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_generation=job["current_generation"],
        total_generations=job["total_generations"],
        best_acc=job["best_acc"],
        history=job["history"],
        error=job["error"],
    )
