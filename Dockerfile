# Dockerfile for the FastAPI backend on Hugging Face Spaces (free CPU tier)
#
# HF Spaces with SDK=docker exposes port 7860 by default. The pipeline runs
# once at build time so the container starts fast; the BERT model downloads
# at runtime into the HF cache layer.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /app

# System deps for some Python wheels (e.g. matplotlib, pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt torch --extra-index-url https://download.pytorch.org/whl/cpu

# App code
COPY backend ./backend
COPY src ./src
COPY data ./data
COPY models ./models
COPY cover_art ./cover_art

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
