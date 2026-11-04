# Runpod Hub build (linux/amd64). Build context: repository root.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    ANIMEGEN_MODEL=hunyuan15_i2v \
    ANIMEGEN_LOAD_MODE=offload

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod/requirements.txt /tmp/runpod-requirements.txt

RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install -r /tmp/runpod-requirements.txt

COPY config ./config
COPY src ./src
COPY .runpod/handler.py ./handler.py

CMD ["python3", "-u", "handler.py"]
