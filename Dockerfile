FROM python:3.11-slim

WORKDIR /app

# Deps de sistema mínimas para psycopg2-binary e bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala apenas as deps da web API (sem torch — build rápido)
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
