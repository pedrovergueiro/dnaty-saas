"""Experiment tracking utilities for reproducible dNaty runs."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from dnaty import __version__


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root(),
            text=True,
            capture_output=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def create_run_dir(experiment_id: str, base_dir: str | Path = "results/runs") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in experiment_id)
    run_dir = repo_root() / base_dir / f"{stamp}_{safe_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_existing(paths: list[str | Path], destination: str | Path) -> list[str]:
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for raw in paths:
        src = repo_root() / raw
        if not src.exists():
            continue
        target = dest / src.name
        shutil.copy2(src, target)
        copied.append(str(target.relative_to(repo_root())))
    return copied


def build_manifest(
    *,
    experiment_id: str,
    config: dict[str, Any],
    command: list[str],
    outputs: list[str],
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "dnaty_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "notes": notes,
        "command": command,
        "config": config,
        "outputs": outputs,
        "git": {
            "commit": git_commit(),
            "dirty": git_dirty(),
        },
        "environment": environment_info(),
        "cwd": os.getcwd(),
    }
