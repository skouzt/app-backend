# =========================
# Builder stage
# =========================
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt


# =========================
# Runtime stage (THIS WAS MISSING)
# =========================
FROM python:3.11-slim-bullseye

WORKDIR /app

# Runtime libs only (no compilers)
# ---- Runtime system deps ONLY (no dev headers) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt .

# IMPORTANT: force wheels only, no source builds
ENV PIP_ONLY_BINARY=:all:
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ---- App code ----
COPY bots /app/bots
COPY config /app/config
COPY core /app/core
COPY db /app/db
COPY services /app/services
COPY schemas /app/schemas
COPY prompts /app/prompts
COPY task /app/task
COPY api /app/api
COPY .env /app/.env
COPY runner.py /app/runner.py
COPY main.py /app/main.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "main.py"]
