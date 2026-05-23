"""
Thin async wrapper around dNatyEvolver for use inside FastAPI background tasks.
Manages job state and forwards progress to the in-memory job store.
"""
from __future__ import annotations
import sys
import time
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Any

# Allow importing the local dnaty package from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.schemas import (
    JobStatus,
    GenerationInfo,
    ArchitectureInfo,
)

logger = logging.getLogger(__name__)

# ── In-memory job store (replace with Redis/DB in production) ──────────────────

_jobs: dict[str, dict[str, Any]] = {}


def create_job(params: dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": JobStatus.queued,
        "params": params,
        "progress": 0.0,
        "current_generation": 0,
        "total_generations": params["n_generations"],
        "best_acc": 0.0,
        "history": [],
        "result": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


# ── Dataset loader ─────────────────────────────────────────────────────────────

def _load_dataset(name: str, device: str):
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset_map = {
        "mnist": (datasets.MNIST, (0.5,), (0.5,)),
        "fashion_mnist": (datasets.FashionMNIST, (0.5,), (0.5,)),
        "cifar10": (datasets.CIFAR10, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    }

    if name not in dataset_map:
        raise ValueError(f"Dataset '{name}' not supported via auto-download. Use 'custom'.")

    cls, mean, std = dataset_map[name]
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = cls(root="/tmp/data", train=True, download=True, transform=t)
    test_ds = cls(root="/tmp/data", train=False, download=True, transform=t)

    loader_kwargs = {"batch_size": 512, "num_workers": 0, "pin_memory": device == "cuda"}
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs)

    input_size = 784 if name in ("mnist", "fashion_mnist") else 3072
    n_classes = 10
    return train_loader, test_loader, input_size, n_classes


# ── Training runner (runs in a thread, not the event loop) ────────────────────

def _run_training(job_id: str) -> None:
    job = _jobs[job_id]
    params = job["params"]
    job["status"] = JobStatus.running
    job["started_at"] = time.time()

    try:
        try:
            import torch  # noqa: F401
        except ImportError:
            job["status"] = JobStatus.error
            job["error"] = "PyTorch not installed in this environment. Training unavailable."
            return

        from dnaty.evolution.evolver import DnatyEvolver

        train_loader, test_loader, input_size, n_classes = _load_dataset(
            params["dataset"], params["device"]
        )

        evolver = DnatyEvolver(
            n_pop=params["n_pop"],
            n_generations=params["n_generations"],
            t_local=params["t_local"],
            lr=params["lr"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            device=params["device"],
            input_size=input_size,
            n_classes=n_classes,
            init_hidden=params["init_hidden"],
            batch_size=params["batch_size"],
            verbose=False,
        )

        # run() returns (best_individual, list[GenerationLog])
        best, raw_history = evolver.run(train_loader, test_loader)

        progress_history: list[GenerationInfo] = []
        for log in raw_history:
            info = GenerationInfo(
                generation=log.gen,
                best_acc=float(log.best_acc),
                delta_grad=float(log.delta_grad),
                delta_mem=float(log.delta_mem),
                n_params=int(log.n_params),
            )
            progress_history.append(info)
            logger.info("job=%s gen=%d acc=%.4f", job_id, log.gen, log.best_acc)

        job["history"] = progress_history
        job["current_generation"] = len(raw_history)
        job["best_acc"] = float(best.acc)
        job["progress"] = 1.0

        # Build architecture info from best individual
        arch_info = None
        model = best.model if hasattr(best, "model") else None
        if model is not None:
            arch_info = ArchitectureInfo(
                layer_sizes=list(model.layer_sizes),
                activations=list(model.activations),
                n_params=sum(p.numel() for p in model.parameters()),
            )

        job["result"] = {
            "best_accuracy": float(best.acc),
            "final_architecture": arch_info,
            "history": progress_history,
            "duration_seconds": time.time() - job["started_at"],
            "dataset": params["dataset"],
            "metadata": {
                "n_pop": params["n_pop"],
                "n_generations": params["n_generations"],
                "device": params["device"],
            },
        }
        job["status"] = JobStatus.completed
        job["progress"] = 1.0

    except Exception as exc:
        logger.exception("Training failed for job %s", job_id)
        job["status"] = JobStatus.failed
        job["error"] = str(exc)
    finally:
        job["finished_at"] = time.time()


async def start_training(job_id: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_training, job_id)
