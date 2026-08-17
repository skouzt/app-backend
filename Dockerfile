# Lily API — pure Python web service. No audio pipeline, so no ffmpeg/portaudio/torch.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api /app/api
COPY config /app/config
COPY core /app/core
COPY db /app/db
COPY services /app/services
COPY schemas /app/schemas
COPY prompts /app/prompts
COPY main.py /app/main.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "main.py"]
