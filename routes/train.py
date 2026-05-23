import logging
from fastapi import APIRouter, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from models.schemas import TrainRequest, TrainResponse, JobStatus
from models.dnaty_model import create_job, get_job
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str | None) -> None:
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=202,
    summary="Start a dNATY evolutionary training job",
    responses={
        202: {"description": "Job accepted and queued"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Unauthorized"},
    },
)
async def start_train(
    request: TrainRequest,
    api_key: str | None = Security(_api_key_header),
) -> TrainResponse:
    _verify_api_key(api_key)

    job_id = create_job(request.model_dump())
    logger.info("Job %s criado (dataset=%s) — aguardando worker Colab", job_id, request.dataset)

    return TrainResponse(
        job_id=job_id,
        status=JobStatus.queued,
        message=f"Job na fila. Acompanhe em /api/v1/status/{job_id}",
    )
