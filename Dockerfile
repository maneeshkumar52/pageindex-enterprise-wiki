# ============================================================
# PageIndex Enterprise Wiki — Multi-stage Docker Build
# ============================================================
# Stage 1: base image with system deps
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: install Python dependencies
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: production image
FROM deps AS production

COPY . .

# Create data directories
RUN mkdir -p data/uploads data/indexes data/workspace data/chromadb outputs

# Expose Streamlit default port
EXPOSE 8501

# Health check — Streamlit serves on /healthz by default
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
