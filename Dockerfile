FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Deps da web API (rápido, sem torch)
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 2. Torch CPU-only (~200MB, separado para cache do Docker)
RUN pip install --no-cache-dir \
    "torch==2.2.0+cpu" "torchvision==0.17.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
