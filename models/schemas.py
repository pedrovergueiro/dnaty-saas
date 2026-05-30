from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class DatasetName(str, Enum):
    mnist = "mnist"
    fashion_mnist = "fashion_mnist"
    cifar10 = "cifar10"
    custom = "custom"


class JobStatus(str, Enum):
    queued    = "queued"
    running   = "running"
    completed = "complete"   # DB and frontend both use "complete"
    failed    = "failed"


# ── Train ──────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset: DatasetName = DatasetName.mnist
    n_pop: int = Field(default=20, ge=4, le=200, description="Population size")
    n_generations: int = Field(default=50, ge=1, le=500)
    t_local: int = Field(default=3, ge=1, le=20, description="Local training epochs per generation")
    lr: float = Field(default=1e-3, gt=0, le=1.0)
    lambda1: float = Field(default=1e-4, ge=0, description="Gradient diversity weight")
    lambda2: float = Field(default=1e-3, ge=0, description="Memory weight")
    batch_size: int = Field(default=512, ge=16, le=4096)
    init_hidden: list[int] = Field(default=[256, 128], description="Initial hidden layer sizes")
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(:\d+)?)$")

    @field_validator("init_hidden")
    @classmethod
    def hidden_must_be_positive(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("init_hidden must be a non-empty list of positive integers")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "dataset": "mnist",
            "n_pop": 20,
            "n_generations": 50,
            "t_local": 3,
            "lr": 0.001,
            "batch_size": 512,
            "init_hidden": [256, 128],
            "device": "cpu",
        }
    }}


class TrainResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


# ── Status ─────────────────────────────────────────────────────────────────────

class GenerationInfo(BaseModel):
    generation: int
    best_acc: float
    delta_grad: float
    delta_mem: float
    n_params: int


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0, description="Fraction completed [0, 1]")
    current_generation: int
    total_generations: int
    best_acc: float
    history: list[GenerationInfo] = []
    error: str | None = None


# ── Results ────────────────────────────────────────────────────────────────────

class ArchitectureInfo(BaseModel):
    layer_sizes: list[int]
    activations: list[str]
    n_params: int


class ResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    dataset: str
    best_accuracy: float
    final_architecture: ArchitectureInfo | None = None
    history: list[GenerationInfo] = []
    duration_seconds: float
    metadata: dict[str, Any] = {}


# ── Error ──────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    job_id: str | None = None
