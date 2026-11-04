# CUDA 12.1 + PyTorch for Runpod GPU workers (linux/amd64)
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

COPY requirements.txt /tmp/requirements-base.txt
COPY runpod/requirements.txt /tmp/runpod-requirements.txt

RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install -r /tmp/runpod-requirements.txt

COPY config ./config
COPY src ./src
COPY runpod/handler.py ./handler.py

# Optional: bake weights into the image (large). Uncomment and set HF_TOKEN at build time.
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# RUN python3 -c "from animegen.inference import WanI2VEngine; WanI2VEngine('ti2v_5b')._load()"

CMD ["python3", "-u", "handler.py"]
