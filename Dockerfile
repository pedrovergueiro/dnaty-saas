FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU-only — layer separado para cache (não baixa de novo se só o código mudar)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Deps da web API
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
