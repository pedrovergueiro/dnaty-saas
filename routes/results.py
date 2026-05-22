import logging
from fastapi import APIRouter, HTTPException

from models.schemas import ResultsResponse, JobStatus
from models.dnaty_model import get_job

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/results/{job_id}",
    response_model=ResultsResponse,
    summary="Fetch final results of a completed training job",
    responses={
        200: {"description": "Final results"},
        202: {"description": "Job still running"},
        404: {"description": "Job not found"},
    },
)
async def get_results(job_id: str) -> ResultsResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == JobStatus.failed:
        raise HTTPException(
            status_code=500,
            detail=f"Job '{job_id}' failed: {job['error']}",
        )

    if job["status"] in (JobStatus.queued, JobStatus.running):
        raise HTTPException(
            status_code=202,
            detail=f"Job '{job_id}' is still {job['status']}. Progress: {job['progress']:.0%}",
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
