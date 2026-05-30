from celery import Celery, Task
from dnaty.compress import Compressor
import os

broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

app = Celery("dnaty_saas", broker=broker_url, backend=result_backend)

class CallbackTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"Task {task_id} failed: {exc}")

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@app.task(bind=True, base=CallbackTask)
def compress_model_async(
    self,
    user_id: str,
    model_name: str,
    dataset: str,
    target_flops: float,
    epochs: int = 10,
    population_size: int = 20,
):
    """Asynchronous model compression task"""
    try:
        compressor = Compressor(device="cuda")

        self.update_state(state="PROGRESS", meta={"progress": 10})

        result = compressor.compress(
            model_name=model_name,
            dataset=dataset,
            target_flops=target_flops,
            epochs=epochs,
            population_size=population_size,
        )

        self.update_state(state="PROGRESS", meta={"progress": 90})

        return {
            "user_id": user_id,
            "model_name": model_name,
            "flops_reduction": result.get("flops_reduction"),
            "latency_ms": result.get("latency_ms"),
            "model_path": result.get("model_path"),
            "status": "completed",
        }

    except Exception as exc:
        return {
            "user_id": user_id,
            "status": "failed",
            "error": str(exc),
        }

@app.task
def cleanup_old_models(days: int = 30):
    """Cleanup models older than X days"""
    import os
    from pathlib import Path
    import time

    results_dir = Path("/app/results")
    if not results_dir.exists():
        return {"cleaned": 0}

    current_time = time.time()
    cutoff_time = current_time - (days * 86400)

    cleaned = 0
    for model_file in results_dir.glob("**/*.pt"):
        if os.path.getmtime(model_file) < cutoff_time:
            os.remove(model_file)
            cleaned += 1

    return {"cleaned": cleaned}
