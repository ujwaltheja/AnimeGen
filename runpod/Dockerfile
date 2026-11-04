# Build from repo root: docker build -f runpod/Dockerfile .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    ANIMEGEN_MODEL=hunyuan15_i2v \
    ANIMEGEN_LOAD_MODE=offload

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY runpod/requirements.txt /tmp/runpod-requirements.txt

RUN python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && python3.11 -m pip install -r /tmp/runpod-requirements.txt

COPY config ./config
COPY src ./src
COPY handler.py ./handler.py

CMD ["python3.11", "-u", "handler.py"]
