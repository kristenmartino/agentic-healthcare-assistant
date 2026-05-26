# syntax=docker/dockerfile:1
#
# Healthcare Assistant — production-style image.
#
# Multi-stage:
#   1. `builder` — installs Python deps into a venv (slow; cacheable).
#   2. final     — copies the venv + source; runs as non-root.
#
# Build:    docker build -t healthcare-assistant .
# Run:      docker run -p 8501:8501 --env-file .env healthcare-assistant
# Compose:  docker compose up

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

# Install Python deps in their own layer so source changes don't rebuild them.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


FROM python:3.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Non-root user for the runtime — minor defense-in-depth.
RUN useradd --create-home --uid 10001 --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . /app

# Pre-create the data dir so the SqliteSaver / audit log / fixture overlay
# can write without permission grief on the first run.
RUN mkdir -p /app/data /app/data/fhir_fixtures && chown -R app:app /app/data

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
