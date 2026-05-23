FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch CPU-only (~200MB vs ~2GB da versão GPU)
RUN pip install --no-cache-dir \
    "torch==2.2.0+cpu" "torchvision==0.17.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# Demais deps — exclui torch/torchvision pois já estão instalados acima
RUN grep -Ev "^torch|^torchvision" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
