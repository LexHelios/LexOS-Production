# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Install system dependencies (builder and final both need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libmagic1 \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Builder stage: install dependencies into a venv
FROM base AS builder
WORKDIR /app

# Copy requirements.txt only (for better cache usage)
COPY --link requirements.txt ./

# Create venv and install dependencies using pip cache
RUN python -m venv .venv \
    && .venv/bin/pip install --upgrade setuptools wheel \
    && .venv/bin/pip install -r requirements.txt \
    --mount=type=cache,target=/root/.cache/pip

# Copy application code (excluding files that should not be in the image)
COPY --link . .

# Final stage: runtime image
FROM base AS final

# Create non-root user and group
RUN groupadd -r lexos && useradd -r -g lexos lexos

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code from builder (preserves permissions, avoids double copy)
COPY --from=builder /app /app

# Set environment variables for Python and venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/lexos_memory /app/uploads /app/static \
    && chown -R lexos:lexos /app

USER lexos

EXPOSE 8080 8081

# Healthcheck (curl is installed in base)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint: use the startup wrapper to launch the app
CMD ["python", "startup_wrapper.py"]
