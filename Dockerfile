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

# One COPY per package the app actually imports. `schemas/` used to be listed
# here; it held the voice session models and was deleted with the rest of that
# stack, but the COPY stayed and failed the build with "/schemas: not found".
# A missing source is a hard error in Docker, so anything removed from the repo
# has to come out of this list too.
COPY api /app/api
COPY config /app/config
COPY core /app/core
COPY db /app/db
COPY services /app/services
COPY prompts /app/prompts
COPY main.py /app/main.py

# Operator tooling rather than something the server imports — send_push.py is
# run by hand over `railway ssh`. Kept in a separate COPY so it is obvious this
# one is not an application dependency.
COPY scripts /app/scripts

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "main.py"]
