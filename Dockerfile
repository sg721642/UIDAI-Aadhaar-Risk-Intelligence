# Production-grade multi-stage Dockerfile for FastAPI Backend
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir --user -r requirements-backend.txt

# Final stage
FROM python:3.10-slim AS runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose FastAPI backend port
EXPOSE 8000

# Run uvicorn on the port provided by Render (defaulting to 8000)
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
