# syntax=docker/dockerfile:1
#
# FastAPI service image — separate from the Streamlit Dockerfile at the repo
# root. Multi-stage; non-root runtime; pip-installs both the root deps (for
# the LangGraph workflow + tools) and api/requirements.txt (for FastAPI).
#
# Build:    docker build -f api/Dockerfile -t healthcare-api .
# Run:      docker run -p 8000:8000 --env-file .env healthcare-api
# Fly.io:   flyctl deploy   (uses fly.toml at the repo root)

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the two requirements files into their respective paths.
# NB: `COPY a b /dest/` flattens — both files would land at /dest/ with the
# second clobbering the first. Use separate COPY instructions to preserve
# the api/ subdirectory.
COPY requirements.txt /build/requirements.txt
COPY api/requirements.txt /build/api/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt -r api/requirements.txt && \
    pip install --no-deps fastapi "uvicorn[standard]"


FROM python:3.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

RUN useradd --create-home --uid 10001 --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . /app

RUN mkdir -p /app/data /app/logs && chown -R app:app /app/data /app/logs

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
