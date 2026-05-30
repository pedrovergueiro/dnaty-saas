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

# Add both parent levels so import works locally (parents[2]=dNATY/) and in
# Docker (parents[1]=/app/ where dnaty_saas/dnaty/ is copied).
for _offset in (1, 2):
    _p = str(Path(__file__).resolve().parents[_offset])
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

def _load_dataset(name: str, device: str, train_subset: int | None = None):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import Subset

    dataset_map = {
        "mnist":         (datasets.MNIST, (0.1307,), (0.3081,)),
        "fashion_mnist": (datasets.FashionMNIST, (0.2860,), (0.3530,)),
        "cifar10":       (datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    }

    if name not in dataset_map:
        raise ValueError(f"Dataset '{name}' not supported. Allowed: mnist, fashion_mnist, cifar10")

    cls, mean, std = dataset_map[name]
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = cls(root="/tmp/data", train=True, download=True, transform=t)
    test_ds  = cls(root="/tmp/data", train=False, download=True, transform=t)

    if train_subset and train_subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(train_subset)))

    pin = (device == "cuda")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=0, pin_memory=pin)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0, pin_memory=pin)

    input_size = 784 if name in ("mnist", "fashion_mnist") else 3072
    return train_loader, test_loader, input_size, 10


# ── Training runner (runs in a thread, not the event loop) ────────────────────

def _run_training(job_id: str) -> None:
    from models.training_store import update_training

    job = _jobs[job_id]
    params = job["params"]
    job["status"] = JobStatus.running
    job["started_at"] = time.time()
    update_training(job_id, status="running")

    try:
        try:
            import torch  # noqa: F401
        except ImportError:
            job["status"] = JobStatus.failed
            job["error"] = "PyTorch not installed in this environment. Training unavailable."
            update_training(job_id, status="failed", error_message=job["error"])
            return

        from dnaty.evolution.evolver import DnatyEvolver

        # Use train_subset to cap samples per plan
        train_subset = params.get("_samples")
        train_loader, test_loader, input_size, n_classes = _load_dataset(
            params["dataset"], params["device"], train_subset=train_subset
        )

        total_gens = params["n_generations"]

        def _on_progress(log) -> None:
            """Called per generation — updates in-memory job AND DB record."""
            pct = int((log.gen / total_gens) * 100)
            job["current_generation"] = log.gen
            job["best_acc"] = float(log.best_acc)
            job["progress"] = log.gen / total_gens
            job["history"].append(GenerationInfo(
                generation=log.gen,
                best_acc=float(log.best_acc),
                delta_grad=float(log.delta_grad),
                delta_mem=float(log.delta_mem),
                n_params=int(log.n_params),
            ))
            update_training(
                job_id,
                status="running",
                progress=pct,
                current_epoch=log.gen,
                accuracy=float(log.best_acc),
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
        # history is built incrementally by _on_progress callback
        best, raw_history = evolver.run(train_loader, test_loader, progress_callback=_on_progress)

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

        duration = time.time() - job["started_at"]
        job["result"] = {
            "best_accuracy": float(best.acc),
            "final_architecture": arch_info,
            "history": job["history"],
            "duration_seconds": duration,
            "dataset": params["dataset"],
            "metadata": {
                "n_pop": params["n_pop"],
                "n_generations": params["n_generations"],
                "device": params["device"],
            },
        }
        job["status"] = JobStatus.completed
        job["progress"] = 1.0

        update_training(
            job_id,
            status="complete",
            progress=100,
            accuracy=float(best.acc),
            duration_seconds=duration,
            completed_at=__import__("datetime").datetime.utcnow(),
        )

    except Exception as exc:
        logger.exception("Training failed for job %s", job_id)
        job["status"] = JobStatus.failed
        job["error"] = str(exc)
        update_training(job_id, status="failed", error_message=str(exc))
    finally:
        job["finished_at"] = time.time()


async def start_training(job_id: str) -> None:
    await asyncio.to_thread(_run_training, job_id)
