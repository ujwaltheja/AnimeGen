# CUDA 12.1 + Python 3.11 (diffusers main requires >=3.11)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    ANIMEGEN_MODEL=hunyuan15_i2v \
    ANIMEGEN_LOAD_MODE=offload \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY runpod/requirements.txt /tmp/runpod-requirements.txt

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r /tmp/runpod-requirements.txt

COPY config ./config
COPY src ./src
COPY .runpod/handler.py ./handler.py

CMD ["python", "-u", "handler.py"]
